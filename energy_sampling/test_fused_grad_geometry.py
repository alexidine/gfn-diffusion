"""
Correctness tests for the fused-branch gradient-geometry diagnostic
(train.Modeller._log_fused_gradient_geometry).

The point of the diagnostic is to tell ALIGNED from ORTHOGONAL from CONFLICTING,
so a test that only asserts "it ran and produced finite numbers" would be exactly
the swallowed diagnostic the thing is meant to replace. Every case below pins a
gradient geometry whose cosines are known in closed form by construction, and
asserts the reported value against the hand-computed one.

Geometry used throughout:
  shared  -- a parameter BOTH branches differentiate (the policy trunk analogue)
  priv_a  -- only branch a's loss touches it (the flow-head analogue)
  priv_b  -- only branch b's loss touches it
so the shared/private split has a known right answer independent of the cosines.
"""
import math

import pytest
import torch
from torch import nn

import train


class _FakeModeller:
    """Just enough of Modeller for the two methods under test."""
    _log_fused_gradient_geometry = train.Modeller._log_fused_gradient_geometry
    _fused_grad_diag_armed = train.Modeller._fused_grad_diag_armed

    def __init__(self, gfn_model):
        self.gfn_model = gfn_model
        self._fused_grad_geom_report = None


class _TwoBranchModel(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.shared = nn.Parameter(torch.zeros(n))
        self.priv_a = nn.Parameter(torch.zeros(n))
        self.priv_b = nn.Parameter(torch.zeros(n))


def _linear_loss(param_terms):
    """sum_i (w_i . p_i) -- gradient wrt p_i is exactly w_i, for every i."""
    return sum((w * p).sum() for p, w in param_terms)


def _run(model, losses, weights):
    """Drive the diagnostic over {branch: loss} and return its report."""
    fake = _FakeModeller(model)
    sub_losses = {k: (v, {}, True) for k, v in losses.items()}
    fake._log_fused_gradient_geometry(sub_losses, weights, sum(weights.values()))
    return fake._fused_grad_geom_report


@pytest.fixture
def model():
    return _TwoBranchModel()


# --------------------------------------------------------------------------
# The three regimes the diagnostic exists to separate.
# --------------------------------------------------------------------------

@pytest.mark.parametrize('wb, expected_shared_cos, label', [
    (torch.tensor([1.0, 0.0, 0.0, 0.0]), 1.0, 'aligned'),
    (torch.tensor([-1.0, 0.0, 0.0, 0.0]), -1.0, 'conflicting'),
    (torch.tensor([0.0, 1.0, 0.0, 0.0]), 0.0, 'orthogonal'),
])
def test_shared_block_cosine_matches_constructed_geometry(model, wb, expected_shared_cos, label):
    wa = torch.tensor([1.0, 0.0, 0.0, 0.0])
    ones = torch.ones(4)
    rep = _run(model,
               {'a': _linear_loss([(model.shared, wa), (model.priv_a, ones)]),
                'b': _linear_loss([(model.shared, wb), (model.priv_b, ones)])},
               {'a': 0.5, 'b': 0.5})

    assert rep['fused_grad/cos_a_b_shared'] == pytest.approx(expected_shared_cos, abs=1e-6), label


def test_no_private_cosine_is_reported(model):
    """There must be NO branch-private cosine. 'Private' means at most one of
    the pair touched the parameter, so every term of that dot product has a
    zero factor and the cosine is identically 0 by construction. Logging it
    would put a structural constant on the dashboard dressed as a measurement
    of orthogonality."""
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    ones = torch.ones(4)
    rep = _run(model,
               {'a': _linear_loss([(model.shared, w), (model.priv_a, ones)]),
                'b': _linear_loss([(model.shared, w), (model.priv_b, ones)])},
               {'a': 0.5, 'b': 0.5})

    assert not any(k.endswith('_private') for k in rep)


def test_shared_block_is_undiluted_by_uncontested_mass(model):
    """The shared-block cosine restricts to what the pair actually contends
    over, so it can differ from the whole-model cosine, which is dragged toward
    0 by each branch's uncontested mass. Both are reported for that reason."""
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    big = torch.tensor([0.0, 10.0, 0.0, 0.0])  # large, uncontested
    rep = _run(model,
               {'a': _linear_loss([(model.shared, w), (model.priv_a, big)]),
                'b': _linear_loss([(model.shared, w), (model.priv_b, big)])},
               {'a': 0.5, 'b': 0.5})

    # perfectly aligned on the contested trunk...
    assert rep['fused_grad/cos_a_b_shared'] == pytest.approx(1.0, abs=1e-6)
    # ...but the whole-model cosine is diluted to 1/101 by the private mass
    assert rep['fused_grad/cos_a_b'] == pytest.approx(1 / 101, abs=1e-5)


# --------------------------------------------------------------------------
# The parameter-disjoint trap: mk_dev's fwd (freeze_policy -> Z only) against
# bwd/replay (freeze_z -> policy only). See bench/fused_stage.py:48.
# --------------------------------------------------------------------------

def test_parameter_disjoint_pair_is_flagged_not_reported_as_orthogonal(model):
    """A pair that shares no parameter has cosine 0 for reasons that have
    nothing to do with cooperation. overlap must read exactly 0 to say so, and
    the shared-block cosine must be ABSENT rather than a NaN or a spurious
    number. Without this the fwd-vs-bwd panel under mk_dev would sit at 0.0
    forever and read as a measured 'orthogonal regime'."""
    w = torch.tensor([1.0, 2.0, 0.0, 0.0])
    rep = _run(model,
               {'a': _linear_loss([(model.priv_a, w)]),    # 'Z head'
                'b': _linear_loss([(model.priv_b, w)])},   # 'policy trunk'
               {'a': 0.5, 'b': 0.5})

    assert rep['fused_grad/overlap_a_b'] == pytest.approx(0.0, abs=1e-9)
    assert 'fused_grad/cos_a_b_shared' not in rep
    # the whole-model cosine IS still emitted, and is the structural 0 -- which
    # is only safe to publish because overlap_a_b sits beside it saying why
    assert rep['fused_grad/cos_a_b'] == pytest.approx(0.0, abs=1e-9)
    # and each branch is fully uncontested
    assert rep['fused_grad/a_uncontested_frac'] == pytest.approx(1.0, abs=1e-6)
    assert rep['fused_grad/b_uncontested_frac'] == pytest.approx(1.0, abs=1e-6)


def test_fully_contested_pair_reports_overlap_one(model):
    w = torch.tensor([1.0, 2.0, 0.0, 0.0])
    rep = _run(model,
               {'a': _linear_loss([(model.shared, w)]),
                'b': _linear_loss([(model.shared, -w)])},
               {'a': 0.5, 'b': 0.5})

    assert rep['fused_grad/overlap_a_b'] == pytest.approx(1.0, abs=1e-6)
    assert rep['fused_grad/cos_a_b_shared'] == pytest.approx(-1.0, abs=1e-6)
    assert rep['fused_grad/a_uncontested_frac'] == pytest.approx(0.0, abs=1e-6)


def test_overlap_is_an_energy_fraction(model):
    """Half of each branch's gradient energy on a jointly-touched parameter and
    half on its own -> overlap 0.5. Pins the metric as an ENERGY (squared-norm)
    share, so a reader can add it up rather than guessing the convention."""
    contested = torch.tensor([3.0, 0.0, 0.0, 0.0])
    private = torch.tensor([3.0, 0.0, 0.0, 0.0])
    rep = _run(model,
               {'a': _linear_loss([(model.shared, contested), (model.priv_a, private)]),
                'b': _linear_loss([(model.shared, contested), (model.priv_b, private)])},
               {'a': 0.5, 'b': 0.5})

    assert rep['fused_grad/overlap_a_b'] == pytest.approx(0.5, abs=1e-6)
    assert rep['fused_grad/a_uncontested_frac'] == pytest.approx(0.5, abs=1e-6)


def test_uncontested_frac_uses_all_active_branches_not_just_a_pair(model):
    """A parameter touched by b and c is contested for BOTH of them even though
    a never sees it. Computing this per-pair instead of over the active set
    would call it uncontested."""
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    rep = _run(model,
               {'a': _linear_loss([(model.priv_a, w)]),
                'b': _linear_loss([(model.shared, w)]),
                'c': _linear_loss([(model.shared, w)])},
               {'a': 1.0, 'b': 1.0, 'c': 1.0})

    assert rep['fused_grad/a_uncontested_frac'] == pytest.approx(1.0, abs=1e-6)
    assert rep['fused_grad/b_uncontested_frac'] == pytest.approx(0.0, abs=1e-6)
    assert rep['fused_grad/c_uncontested_frac'] == pytest.approx(0.0, abs=1e-6)
    # the shared mask must be PER PAIR, not 'touched by >=2 of the active set':
    # `shared` is touched by b and c, so a global mask would call it shared for
    # the (a,b) pair too and emit a NaN cosine over a block a never touched
    assert 'fused_grad/cos_a_b_shared' not in rep
    assert 'fused_grad/cos_a_c_shared' not in rep
    assert rep['fused_grad/cos_b_c_shared'] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# Norms, and the fused-vs-components ratio.
# --------------------------------------------------------------------------

def test_reported_norms_are_pre_weighting(model):
    """*_norm_raw must be the branch's OWN gradient norm. Weighting it by the
    frac would make the metric move when the balance controller moves, which is
    the confound the 'before weighting' in the spec is guarding against."""
    wa = torch.tensor([3.0, 4.0, 0.0, 0.0])  # norm 5
    wb = torch.tensor([0.0, 0.0, 6.0, 8.0])  # norm 10
    rep = _run(model,
               {'a': _linear_loss([(model.shared, wa)]),
                'b': _linear_loss([(model.shared, wb)])},
               {'a': 0.9, 'b': 0.1})  # lopsided on purpose

    assert rep['fused_grad/a_norm_raw'] == pytest.approx(5.0, abs=1e-5)
    assert rep['fused_grad/b_norm_raw'] == pytest.approx(10.0, abs=1e-5)


def test_fused_norm_ratio_is_one_when_aligned(model):
    w = torch.tensor([1.0, 2.0, 0.0, 0.0])
    rep = _run(model,
               {'a': _linear_loss([(model.shared, w)]),
                'b': _linear_loss([(model.shared, w)])},
               {'a': 0.5, 'b': 0.5})

    assert rep['fused_grad/fused_norm_ratio'] == pytest.approx(1.0, abs=1e-6)


def test_fused_norm_ratio_is_zero_under_total_cancellation(model):
    """Equal weights, exactly opposite gradients: the fused gradient is the zero
    vector while both components are large. A ratio that did not collapse here
    would be unable to show conflict at all."""
    w = torch.tensor([1.0, 2.0, 0.0, 0.0])
    rep = _run(model,
               {'a': _linear_loss([(model.shared, w)]),
                'b': _linear_loss([(model.shared, -w)])},
               {'a': 0.5, 'b': 0.5})

    assert rep['fused_grad/fused_norm'] == pytest.approx(0.0, abs=1e-6)
    assert rep['fused_grad/weighted_component_norm_sum'] > 1.0  # components are NOT small
    assert rep['fused_grad/fused_norm_ratio'] == pytest.approx(0.0, abs=1e-6)


def test_fused_norm_ratio_sits_at_one_over_sqrt_n_when_orthogonal(model):
    """The ORTHOGONAL regime's landmark: n equal-weighted, equal-size mutually
    orthogonal components give |sum| / sum|.| = 1/sqrt(n). Without this the
    middle regime has no reference value to be read against."""
    wa = torch.tensor([1.0, 0.0, 0.0, 0.0])
    wb = torch.tensor([0.0, 1.0, 0.0, 0.0])
    wc = torch.tensor([0.0, 0.0, 1.0, 0.0])
    rep = _run(model,
               {'a': _linear_loss([(model.shared, wa)]),
                'b': _linear_loss([(model.shared, wb)]),
                'c': _linear_loss([(model.shared, wc)])},
               {'a': 1.0, 'b': 1.0, 'c': 1.0})

    assert rep['fused_grad/fused_norm_ratio'] == pytest.approx(1 / math.sqrt(3), abs=1e-6)


# --------------------------------------------------------------------------
# Gating: what must NOT be measured.
# --------------------------------------------------------------------------

def test_zero_weight_branch_is_excluded(model):
    """A force-refresh-only branch is detached and carries no graph, so it must
    never reach autograd.grad -- which would raise. Its weight is 0, and that is
    the gate. Here the third branch is detached exactly as fused_train_step
    leaves it; the call must succeed and must not report it."""
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    losses = {'a': _linear_loss([(model.shared, w)]),
              'b': _linear_loss([(model.shared, w)]),
              'refresh': _linear_loss([(model.shared, w)]).detach()}
    rep = _run(model, losses, {'a': 0.5, 'b': 0.5, 'refresh': 0.0})

    assert 'fused_grad/refresh_norm_raw' not in rep
    assert not any('refresh' in k for k in rep)
    assert 'fused_grad/cos_a_b' in rep


def test_single_active_branch_reports_nothing(model):
    """One branch has no pair to be compared against; emitting a lone cosine of
    itself (1.0, always) would read as permanent perfect cooperation."""
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    rep = _run(model,
               {'a': _linear_loss([(model.shared, w)]),
                'b': _linear_loss([(model.shared, w)]).detach()},
               {'a': 1.0, 'b': 0.0})

    assert rep is None


def test_graph_survives_for_the_real_backward(model):
    """retain_graph=True is load-bearing: the diagnostic runs BEFORE
    step_loss's fused_loss.backward(). If it freed the graph, training would die
    with 'backward through the graph a second time'. Re-introduce that by
    dropping retain_graph and this test fails."""
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    loss_a = _linear_loss([(model.shared, w), (model.priv_a, w)])
    loss_b = _linear_loss([(model.shared, w), (model.priv_b, w)])
    _run(model, {'a': loss_a, 'b': loss_b}, {'a': 0.5, 'b': 0.5})

    fused = 0.5 * loss_a + 0.5 * loss_b
    fused.backward()  # must not raise

    assert model.shared.grad is not None
    # and the diagnostic must not have polluted .grad on the way through:
    # autograd.grad() returns gradients, it does not accumulate them, so this is
    # the fused gradient alone (w from each branch, halved and summed = w).
    assert torch.allclose(model.shared.grad, w)


def test_diagnostic_does_not_perturb_the_fused_gradient(model):
    """The strong form of the above: the parameter gradients after a normal
    backward must be bitwise what they would have been with the diagnostic off."""
    w = torch.tensor([1.0, -2.0, 0.5, 0.0])

    def fused_backward(run_diagnostic):
        m = _TwoBranchModel()
        la = _linear_loss([(m.shared, w), (m.priv_a, w)])
        lb = _linear_loss([(m.shared, 2 * w), (m.priv_b, w)])
        if run_diagnostic:
            _run(m, {'a': la, 'b': lb}, {'a': 0.25, 'b': 0.75})
        (0.25 * la + 0.75 * lb).backward()
        return {n: p.grad.clone() for n, p in m.named_parameters()}

    off, on = fused_backward(False), fused_backward(True)
    for name in off:
        assert torch.equal(off[name], on[name]), name


# --------------------------------------------------------------------------
# The cadence gate.
# --------------------------------------------------------------------------

class _Args:
    def __init__(self, grad_geometry=None):
        if grad_geometry is not None:
            self.grad_geometry = grad_geometry


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.mark.parametrize('cfg, step, expected', [
    (None, 50, False),                                    # block absent -> off
    (_Cfg(enabled=False, every=10), 50, False),           # explicitly off
    (_Cfg(enabled=True, every=0), 50, False),             # every=0 -> off, not ZeroDivisionError
    (_Cfg(enabled=True, every=10), 50, True),
    (_Cfg(enabled=True, every=10), 55, False),
    (_Cfg(enabled=True, every=1), 7, True),
])
def test_arming_gate(cfg, step, expected):
    fake = _FakeModeller(_TwoBranchModel())
    fake.args = _Args(cfg)
    fake.fused_step_count = step

    assert fake._fused_grad_diag_armed() is expected


def test_absent_config_block_is_off_not_on():
    """Omitting the block must not silently start paying for extra backward
    passes on every existing config in the repo."""
    fake = _FakeModeller(_TwoBranchModel())
    fake.args = _Args(None)
    fake.fused_step_count = 0  # 0 % every == 0, the step most likely to fire

    assert fake._fused_grad_diag_armed() is False
