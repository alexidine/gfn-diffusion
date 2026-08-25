"""
The larder and the composite ray reading -- section 6A of
`docs/design/lr_handoff_2026-08-21.md`.

WHAT CHANGED, and therefore what these pin. `ray` used to draw its sub-batches
from the replay buffer and score `replay_loss_coeffs`, which made it coherent
only in a fused stage training replay TB -- so phase 1 was never measured. It now
deals from a per-branch ring of batches harvested off the live steps and scores
the FUSED COMPOSITE the stage's own optimizer step descends.

Three properties are load-bearing and each is a separate way the change could
have gone wrong silently:

  * the deal is DETERMINISTIC. The replay draw consumed NumPy RNG that nothing
    restored, so a discarded reading still moved every subsequent training step
    (findings.md F-039). If sampling ever creeps back in,
    `test_taking_from_the_larder_consumes_no_rng` fails.
  * the deal is HELD OUT. Scoring the ray on the batches the step's own gradient
    came from is biased high, in the direction that licenses too-large steps.
  * a branch that cannot be replay-scored is REFUSED, not dropped. A composite
    missing an active branch is the optimum for a direction nobody took.

Marked fast: torch is imported for RNG state and tensors, but nothing here
builds a model, runs a rollout or reads the data drive.
"""

import copy

import numpy as np
import pytest
import torch

from bench.fake_modeller import make_args
from energy_sampling.lr_larder import (BranchRefused, Harvested, Larder,
                                       LarderScorer, to_device, to_host)
from energy_sampling.ray_calibration import COMPOSITE, RayCalibration

pytestmark = pytest.mark.fast


def _rec(branch, step, payload=None):
    return Harvested(branch=branch, step=step, condition=None, condition_id=None,
                     log_r=payload, mol_batch=None, traj=None, repeats=1,
                     scramble_tiles=0, sample_weights=None)


def _fill(larder, branches, steps):
    for s in steps:
        for b in branches:
            larder.record(_rec(b, s, payload=f'{b}@{s}'))
    return larder


# ------------------------------------------------------------------ the ring --

def test_the_ring_is_per_branch_and_bounded():
    larder = _fill(Larder(depth=3), ('fwd', 'bwd'), range(10))
    assert larder.count('fwd') == 3 and larder.count('bwd') == 3
    assert larder.count('replay') == 0
    assert set(larder.branches()) == {'fwd', 'bwd'}
    assert larder.n_seen == 20, 'n_seen counts what went past, not what is held'
    assert [r.step for r in larder.rings['fwd']] == [7, 8, 9]


def test_clear_drops_everything():
    larder = _fill(Larder(depth=8), ('fwd',), range(8))
    larder.clear()
    assert larder.branches() == () and larder.count('fwd') == 0


# --------------------------------------------------------------- held out ----

def test_the_steps_own_batches_are_not_eligible():
    """The pending optimizer step's own training data is excluded, and the
    exclusion is what does it -- not an accident of ordering.

    Re-introducing the bug (asking for `before_step = step + 1`) must make the
    same record visible again, or this test would pass on a larder that simply
    never held it.
    """
    larder = _fill(Larder(depth=8), ('bwd',), range(5))     # steps 0..4
    assert [r.step for r in larder.eligible('bwd', 4)] == [0, 1, 2, 3]
    assert [r.step for r in larder.eligible('bwd', 5)] == [0, 1, 2, 3, 4]


def test_an_accumulation_window_is_excluded_whole():
    """Under gradient accumulation one optimizer step descends SEVERAL host
    iterations' batches, so `before_step` is the start of the window, not the
    last iteration. Excluding only the last one would score the ray on data its
    own gradient came from."""
    larder = _fill(Larder(depth=16), ('fused_branch',), range(10))
    window_start = 7                       # steps 7, 8, 9 all fed the pending step
    got = [r.step for r in larder.eligible('fused_branch', window_start)]
    assert got == [0, 1, 2, 3, 4, 5, 6]


# ------------------------------------------------------------------ the deal --

def test_take_is_deterministic_disjoint_and_branch_aligned():
    larder = _fill(Larder(depth=16), ('fwd', 'replay'), range(12))
    deal = larder.take(('fwd', 'replay'), 4, before_step=12)
    assert deal is not None and len(deal) == 4
    # newest last, one record per branch per sub-batch, same step across branches
    assert [s['fwd'].step for s in deal] == [8, 9, 10, 11]
    assert all(s['fwd'].step == s['replay'].step for s in deal)
    # disjoint: no record appears in two sub-batches
    ids = [id(s['fwd']) for s in deal]
    assert len(set(ids)) == len(ids)
    # deterministic: the same call twice is the same deal
    again = larder.take(('fwd', 'replay'), 4, before_step=12)
    assert [s['fwd'].step for s in again] == [s['fwd'].step for s in deal]


def test_take_refuses_rather_than_short_dealing():
    larder = _fill(Larder(depth=16), ('fwd', 'bwd'), range(6))
    assert larder.take(('fwd', 'bwd'), 8, before_step=6) is None
    assert larder.have(('fwd', 'bwd'), 6, before_step=6) is True
    # one branch short is the whole deal short: a composite missing a branch is
    # not a smaller composite, it is a different objective
    larder.rings['bwd'].pop()
    assert larder.have(('fwd', 'bwd'), 6, before_step=6) is False
    assert larder.take(('fwd', 'bwd'), 6, before_step=6) is None


def test_take_on_no_branches_is_not_ready():
    assert Larder(depth=4).have((), 1, before_step=10) is False


def test_taking_from_the_larder_consumes_no_rng():
    """F-039, structurally prevented rather than accounted for.

    The replay draw this replaced consumed NumPy RNG nothing restored, so a
    calibration whose reading was discarded still shifted every subsequent
    training step and a probed run was not comparable with an unprobed one.
    """
    larder = _fill(Larder(depth=32), ('fwd', 'bwd', 'replay'), range(20))
    torch_before = torch.get_rng_state().clone()
    np_before = copy.deepcopy(np.random.get_state())

    for _ in range(5):
        assert larder.take(('fwd', 'bwd', 'replay'), 8, before_step=20) is not None

    assert torch.equal(torch.get_rng_state(), torch_before)
    after = np.random.get_state()
    assert after[0] == np_before[0] and np.array_equal(after[1], np_before[1])
    assert after[2] == np_before[2]


# ------------------------------------------------------ the branch refusal ----

class _Bare:
    """Just enough modeller for LarderScorer's bank/refusal path."""

    def __init__(self, **overrides):
        self.args = make_args(**overrides)


def test_no_branch_is_refused_any_more():
    """The refusal path is empty by owner decision (2026-08-22): the Z sidecar is
    excluded from ALL LR control, and it was the only structural blocker."""
    s = LarderScorer(_Bare(), verbose=False)
    for branch in ('fwd', 'bwd', 'replay'):
        assert s.refusal(branch) is None


@pytest.mark.parametrize('term', ['emp_z', 'emp_z_persistent', 'z_level'])
def test_a_z_sidecar_term_is_zeroed_not_refused(term):
    """THE REVERSAL, and the reason for it.

    These terms exist to train the FLOW HEAD -- the one thing the ray
    deliberately does not measure. `ray` rays policy parameters only (decision
    D26b) and holds the flow head at its post-step value throughout, so scoring
    its loss would put a quantity the ray holds FIXED into the objective the ray
    is differencing.

    It also unblocks a real stage: `var_conditioning` ships `emp_z: 1.0` on its
    forward bank, and the backward evaluator ASSERTS against emp_z under
    vg_by_condition -- so the whole stage used to be unmeasurable. Zeroed, its
    VarGrad terms replay normally.
    """
    scorer = LarderScorer(_Bare(**{f'fwd_loss_coeffs.{term}': 1.0}), verbose=False)
    assert scorer.refusal('fwd') is None
    bank = scorer.bank('fwd')
    assert float(getattr(bank, term)) == 0.0
    # ...and the LIVE bank is untouched -- training still trains the term
    assert float(getattr(scorer._raw_bank('fwd'), term)) == 1.0


@pytest.mark.parametrize('term', ['reward_grads', 'traj_grads'])
def test_a_gradient_path_flag_is_left_alone(term):
    """NOT zeroed, deliberately. These only decide which paths carry gradient,
    and the ray runs under no_grad -- so they cannot change a scored value, and
    rewriting them would be a change with no effect pretending to be a
    correction. The canonical bwd bank runs traj_grads 1.0."""
    scorer = LarderScorer(_Bare(**{f'fwd_loss_coeffs.{term}': 1.0}), verbose=False)
    assert float(getattr(scorer.bank('fwd'), term)) == 1.0


def test_missing_required_coeffs_are_padded_not_raised():
    """The mirror case: the replay evaluator reads `mle`/`pf_boost` without a
    getattr guard, and the fwd bank legitimately has neither."""
    s = LarderScorer(_Bare(), verbose=False)
    bank = s.bank('fwd')
    for k in LarderScorer.REQUIRED_COEFFS:
        assert hasattr(bank, k)
    assert float(bank.mle) == 0.0
    # the LIVE bank is not mutated -- padding is a copy
    assert not hasattr(s._raw_bank('fwd'), 'mle')


# ------------------------------------------------- the composite ray reading --

def _ray_over(components, weights, n_sub=8,
              alphas=(0.0, 1.0, 2.0, 4.0, 8.0, 16.0)):
    """One arm/step/measure cycle whose loss is an exact quadratic in alpha.

    theta after the step is 1.0 and the step is 1.0, so theta(alpha) == alpha and
    a component with minimum at `t` has alpha* == t by construction. The
    per-sub-batch scale gives the paired differences a real variance; it is
    alpha-independent, so it cannot move any sign.
    """
    p = torch.nn.Parameter(torch.zeros(1))
    cal = RayCalibration([p], alphas=alphas, n_sub=n_sub, period=10, enabled=True)
    cal._last_done = -1
    assert cal.arm(10)
    with torch.no_grad():
        p.add_(1.0)

    k = {'i': 0}

    def draw():
        k['i'] += 1
        return k['i']

    def loss(sub):
        a = float(p.detach().item())
        scale = 1.0 + 0.05 * sub
        out = {name: scale * (a - t) ** 2 + 100.0 * sub
               for name, t in components.items()}
        if weights is not None:
            out[COMPOSITE] = sum(weights[n] * out[n] for n in components)
        return out

    return cal, cal.measure(draw, loss)


def test_the_composite_is_what_alpha_star_describes():
    """Two branches wanting very different rates, and the reading reports the
    optimum of the SUM the optimizer step actually descends -- not either
    branch's own, and not their average."""
    cal, r = _ray_over({'bwd': 6.0, 'replay': 1.2},
                       weights={'bwd': 0.5, 'replay': 0.5})
    assert r is not None and r['status'] == 'bracketed'
    # composite minimum is at (6.0 + 1.2)/2 = 3.6, which brackets [2, 4] --
    # neither branch's own bracket ([4, 8] and [1, 2] below)
    assert (r['lo'], r['hi']) == (2.0, 4.0)
    assert r['alpha_star'] == pytest.approx(8.0 ** 0.5)


def test_per_branch_brackets_come_back_free():
    """Each branch is evaluated at each alpha to form the sum, so its own
    bracket costs nothing. Diagnostic only -- nothing actuates on it -- but
    branch disagreement is what says the fused step is a compromise."""
    cal, r = _ray_over({'bwd': 6.0, 'replay': 1.2},
                       weights={'bwd': 0.5, 'replay': 0.5})
    comp = r['components']
    assert (comp['bwd']['lo'], comp['bwd']['hi']) == (4.0, 8.0)
    assert (comp['replay']['lo'], comp['replay']['hi']) == (1.0, 2.0)

    rep = cal.report()
    assert rep['raycal/branch/alpha_star_bwd'] == pytest.approx(32.0 ** 0.5)
    assert rep['raycal/branch/alpha_star_replay'] == pytest.approx(2.0 ** 0.5)
    assert rep['raycal/alpha_star'] == pytest.approx(8.0 ** 0.5)
    assert 'raycal/branch/alpha_star_composite' not in rep, \
        'the composite is reported as raycal/alpha_star, not twice'


def test_a_scalar_loss_fn_still_reads_as_the_composite():
    """Back-compatible by construction: every caller predating component
    scoring returns one number, and that number IS the composite."""
    p = torch.nn.Parameter(torch.zeros(1))
    cal = RayCalibration([p], n_sub=6, period=10, enabled=True)
    cal._last_done = -1
    assert cal.arm(10)
    with torch.no_grad():
        p.add_(1.0)
    k = {'i': 0}

    def draw():
        k['i'] += 1
        return k['i']

    r = cal.measure(draw, lambda sub: (1.0 + 0.05 * sub)
                    * (float(p.detach().item()) - 3.0) ** 2)
    assert r is not None and r['status'] == 'bracketed'
    assert (r['lo'], r['hi']) == (2.0, 4.0)
    assert set(r['components']) == {COMPOSITE}


def test_a_reading_without_a_composite_is_void():
    """A component table with no `composite` is not a reading with a missing
    field, it is a measurement of nothing the controller can act on."""
    cal, r = _ray_over({'bwd': 6.0}, weights=None)
    assert r is None


def test_a_component_absent_from_one_sub_batch_is_dropped_not_imputed():
    """An absent branch is not a branch at loss zero. It is dropped from the
    component list; the composite -- present throughout -- still reads."""
    p = torch.nn.Parameter(torch.zeros(1))
    cal = RayCalibration([p], n_sub=6, period=10, enabled=True)
    cal._last_done = -1
    assert cal.arm(10)
    with torch.no_grad():
        p.add_(1.0)
    k = {'i': 0}

    def draw():
        k['i'] += 1
        return k['i']

    def loss(sub):
        a = float(p.detach().item())
        out = {COMPOSITE: (1.0 + 0.05 * sub) * (a - 3.0) ** 2}
        if sub > 1:                       # a branch that woke up mid-calibration
            out['replay'] = (a - 0.75) ** 2
        return out

    r = cal.measure(draw, loss)
    assert r is not None
    assert set(r['components']) == {COMPOSITE}
    assert 'raycal/branch/alpha_star_replay' not in cal.report()


# ------------------------------------------- the per-stage probe cadence -----

def test_a_ray_stage_may_override_period_and_n_sub():
    """MEASURED, not preferred. The probe's absolute cost is
    n_sub x len(alphas) forward passes over one batch; its OVERHEAD is that
    divided by the stage's step cost. On elj/mipcas a calibration costs ~28
    training steps on `train_prior`, whose bwd/dataset step runs no rollout and
    no energy call (median 0.158 s) -- 5.6% at period 500, against the 1.2%
    recorded for the same probe on the fused stage. One global period cannot
    serve both."""
    from energy_sampling.protocol import Stage
    parse = Stage._parse_lr_sensor
    me = type('S', (), {'name': 'train_prior'})()
    assert parse(me, {'kind': 'ray'}) == {'kind': 'ray'}
    assert parse(me, {'kind': 'ray', 'period': 1500}) == {'kind': 'ray', 'period': 1500}
    assert parse(me, {'kind': 'ray', 'n_sub': 4})['n_sub'] == 4


@pytest.mark.parametrize('node,match', [
    ({'kind': 'ray', 'period': 155}, 'multiple of 10'),
    ({'kind': 'ray', 'n_sub': 1}, 'n_sub'),
    ({'kind': 'ray', 'beta': 0.1}, 'takes only'),
])
def test_a_ray_stage_refuses_an_incoherent_override(node, match):
    """The 10-step rule is the one RayCalibration enforces -- metrics drain on
    that clock, so an aliased period means some calibrations never reach the log
    and `raycal/*` silently describes a subset."""
    from energy_sampling.protocol import Stage
    me = type('S', (), {'name': 'train_prior'})()
    with pytest.raises(ValueError, match=match):
        Stage._parse_lr_sensor(me, node)


# ------------------------------------------- the condition-grouped VarGrad ---

def test_condition_ids_survive_the_larder_round_trip_unchanged():
    """The groups are formed FROM this tensor, so if it comes back intact the
    replayed grouping is identical to the live step's by construction. The
    larder stores the whole batch, so it does -- this pins that rather than
    leaving it to be re-derived."""
    cid = torch.tensor([3, 3, 7, 7, 7, 1, 1, 3], dtype=torch.long)
    rec = _rec('bwd', 10)._replace(condition_id=to_host(cid))
    back = to_device(rec.condition_id, 'cpu')
    assert torch.equal(back, cid)
    live = {int(c): int((cid == c).sum()) for c in cid.unique()}
    replay = {int(c): int((back == c).sum()) for c in back.unique()}
    assert live == replay == {1: 2, 3: 3, 7: 3}


def test_a_condition_grouped_bank_without_ids_is_refused_not_silently_regrouped():
    """THE HAZARD, and it is a silent one. `get_gfn_backward_loss` gates the
    condition-grouped VarGrad on `condition_id is not None`; drop the ids and
    control falls through to the LEGACY repeats-grouped branch, which for
    same-terminal tiles is TBC in disguise. A different objective, no error --
    so the ray would rate a loss the stage does not train."""
    scorer = LarderScorer(_Bare(**{'bwd_loss_coeffs.vg_by_condition': 1.0,
                                   'bwd_loss_coeffs.vg_lb': 1.0}), verbose=False)
    bank = scorer.bank('bwd')
    with pytest.raises(BranchRefused, match='condition_id'):
        scorer._check_condition_grouping(bank, _rec('bwd', 10))

    # ...and it passes once the ids are there
    withids = _rec('bwd', 10)._replace(
        condition_id=to_host(torch.tensor([1, 1, 2, 2])))
    scorer._check_condition_grouping(bank, withids)


def test_the_guard_abstains_when_the_bank_is_not_condition_grouped():
    """Mutation guard: an ungrouped bank has nothing to fall through TO, so a
    missing condition_id is not a fault there. A rule that fired on every
    record would be no rule."""
    scorer = LarderScorer(_Bare(), verbose=False)
    for branch in ('fwd', 'bwd', 'replay'):
        scorer._check_condition_grouping(scorer.bank(branch), _rec(branch, 10))


def test_the_larder_reports_its_own_host_footprint():
    """MEASURED RATHER THAN ASSUMED, and that distinction cost a run.

    `depth` was `4 * n_sub` on the reasoning that headroom above n_sub "only
    buys older data" -- true about the BENEFIT, silent about the PRICE, which is
    per ACTIVE BRANCH. Phase 1 has one branch; a fused crystal stage has three,
    each holding full trajectories and PyG graphs for a batch of 1000. Published
    via `nbytes()` so nobody has to reason about it again (no longer logged)."""
    lard = Larder(depth=4)
    assert lard.nbytes() == 0
    rec = _rec('bwd', 1)._replace(traj=to_host(torch.zeros(64, 10, 12)),
                                  log_r=to_host(torch.zeros(64)))
    lard.record(rec)
    one = lard.nbytes()
    assert one >= 64 * 10 * 12 * 4, 'the trajectory must be counted'

    for i in range(3):
        lard.record(rec._replace(step=2 + i))
    assert lard.nbytes() == pytest.approx(4 * one, rel=1e-6)

    # ...and it is PER BRANCH, which is the whole point
    for i in range(4):
        lard.record(rec._replace(branch='replay', step=10 + i))
    assert lard.nbytes() == pytest.approx(8 * one, rel=1e-6)


def test_the_ring_cap_bounds_the_footprint():
    """A ring at its cap must stop growing -- otherwise `depth` bounds nothing
    and the metric would climb forever."""
    lard = Larder(depth=3)
    rec = _rec('bwd', 1)._replace(traj=to_host(torch.zeros(32, 10, 12)))
    for i in range(3):
        lard.record(rec._replace(step=i))
    capped = lard.nbytes()
    for i in range(10):
        lard.record(rec._replace(step=100 + i))
    assert lard.nbytes() == pytest.approx(capped, rel=1e-6)
    assert lard.count('bwd') == 3
