"""
DUAL SCORING: the ray reads its own objective against the one training descends.

THE DEFECT THIS EXISTS TO MEASURE. `ray` rates a step by REPLAYING the stored
trajectory at each alpha. For a branch that trains on stored trajectories
(`replay`) that is the trained objective exactly. For a branch whose live draw
sets `traj=None` -- `bwd` with `bwd_sampling_mode: dataset` or `prior` -- it is
not: training re-samples a backward path from the current P_B every step. The
stored path was drawn from P_B at theta_before, so it goes off-distribution as
the ray moves theta, and the replayed loss can RISE where the trained objective
falls. The predicted consequence is one-directional: alpha* pushed DOWN, toward
and then past the grid floor, at every rate. Measured on elj/mipcas phase 1,
alpha* sat at `below_range` across 1e-5 to 8.9e-5 while the historical converged
run of that route trained at ~8e-4.

`ray_calibration.dual_score: true` scores every alpha BOTH ways on the SAME
sub-batches at the SAME parameters and brackets both, so the gap is measured
rather than argued. It is a diagnostic: the controller is still handed the
replayed reading, and these tests pin that.

WHAT IS PINNED HERE:

  * the primary reading -- the one the controller actuates on -- is identical to
    what it would have been with the diagnostic off;
  * the fresh pass is bracketed by the SAME code, so the two are comparable by
    construction rather than by a parallel implementation that can drift;
  * `rayfresh/gap_octaves` has the sign and magnitude the two minima imply, and
    is ZERO when they agree (the control case -- a `replay` branch);
  * the alpha grid is scored under COMMON RANDOM NUMBERS, without which the
    paired differences would be sampling noise reported at high confidence;
  * the pass consumes no NET randomness, so turning the diagnostic on does not
    change the run it is measuring;
  * a dual calibration that fails to bracket CLEARS its reading rather than
    leaving the last one to be re-logged as if it were this period's.
"""

import numpy as np
import pytest
import torch

from bench.fake_modeller import make_args
from energy_sampling.lr_larder import Harvested, LarderScorer
from energy_sampling.ray_calibration import COMPOSITE, RayCalibration

pytestmark = pytest.mark.fast

ALPHAS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)


def _armed(n_sub=8, alphas=ALPHAS, dual=True):
    """A cal with a step of exactly 1.0 taken from 0.0, so theta(alpha) == alpha.

    A component whose loss is quadratic with minimum at `t` therefore has
    alpha* == t by construction, and the bracket is whichever pair of grid
    points straddles it.
    """
    p = torch.nn.Parameter(torch.zeros(1))
    cal = RayCalibration([p], alphas=alphas, n_sub=n_sub, period=10,
                         enabled=True, dual_score=dual)
    cal._last_done = -1
    assert cal.arm(10)
    with torch.no_grad():
        p.add_(1.0)
    return p, cal


def _quadratic(p, t, noisy=False):
    """loss(sub) with its minimum at alpha == t.

    The per-sub-batch scale gives the paired differences a real variance and is
    alpha-independent, so it cannot move a sign. `noisy` adds a draw from the
    global RNG, which is what makes the coupling and neutrality tests bite.
    """
    def loss(sub):
        a = float(p.detach().item())
        value = (1.0 + 0.05 * sub) * (a - t) ** 2 + 100.0 * sub
        if noisy:
            value = value + float(torch.randn(()))
        return {COMPOSITE: value}
    return loss


def _subs():
    """A fresh sub-batch index each call -- `draw_fn` for `measure`."""
    it = iter(range(1, 10_000))
    return lambda: next(it)


def _dual(replayed_t, fresh_t, n_sub=8, alphas=ALPHAS):
    p, cal = _armed(n_sub=n_sub, alphas=alphas)
    r = cal.measure(_subs(), _quadratic(p, replayed_t), _quadratic(p, fresh_t))
    return cal, r


# ------------------------------------------------- the actuating path is safe --

def test_the_controller_still_reads_the_replayed_reading():
    """THE WHOLE POINT OF CALLING IT A DIAGNOSTIC. `measure` returns the
    replayed reading and nothing else; the fresh one is reachable only via
    `last_fresh` and is never what `on_calibration` is handed."""
    cal, r = _dual(replayed_t=1.2, fresh_t=6.0)
    assert r is cal.last
    assert (r['lo'], r['hi']) == (1.0, 2.0)
    assert r['alpha_star'] == pytest.approx(2.0 ** 0.5)


def test_the_primary_reading_is_unchanged_by_the_diagnostic():
    """Run the SAME replayed loss with the diagnostic off and demand an
    identical reading. If the fresh pass leaked into the primary table -- a
    shared mutable row, a `self.last` clobbered by the second `_reading` --
    this is what catches it."""
    p_on, cal_on = _armed(dual=True)
    on = cal_on.measure(_subs(), _quadratic(p_on, 1.2), _quadratic(p_on, 6.0))

    p_off, cal_off = _armed(dual=False)
    off = cal_off.measure(_subs(), _quadratic(p_off, 1.2))

    assert (on['lo'], on['hi'], on['status']) == (off['lo'], off['hi'], off['status'])
    assert on['alpha_star'] == off['alpha_star']
    assert on['aggregate'] == off['aggregate']


# ------------------------------------------------------------- the second read --

def test_the_fresh_pass_is_bracketed_by_the_same_code():
    """Both readings come out of `_reading`, so a change to the bracket rule
    moves both together. A second implementation would drift, and the gap would
    then be measuring the drift instead of the objectives."""
    cal, _ = _dual(replayed_t=1.2, fresh_t=6.0)
    f = cal.last_fresh
    assert f['status'] == 'bracketed'
    assert (f['lo'], f['hi']) == (4.0, 8.0)
    assert f['alpha_star'] == pytest.approx(32.0 ** 0.5)


def test_the_gap_is_reported_in_octaves_with_the_defects_sign():
    """The number to read first. Positive means the trained objective wants a
    BIGGER step than the replayed one is willing to license -- the direction
    the frozen-trajectory account predicts, and the direction that pins alpha*
    at the grid floor while the route trains happily far above it."""
    cal, r = _dual(replayed_t=1.2, fresh_t=6.0)
    rep = cal.report()
    assert rep['rayfresh/alpha_star'] == pytest.approx(32.0 ** 0.5)
    assert rep['rayfresh/gap_octaves'] == pytest.approx(
        float(np.log2(32.0 ** 0.5 / 2.0 ** 0.5)))
    assert rep['rayfresh/gap_octaves'] > 0
    # ...and the actuated reading is still published unchanged alongside it
    assert rep['raycal/alpha_star'] == pytest.approx(2.0 ** 0.5)


def test_agreement_reads_as_zero_gap():
    """THE CONTROL ARM. A `replay` branch trains on stored trajectories, so its
    two readings SHOULD coincide -- that is what says a gap on the other
    branches is about the sampling rule rather than about the diagnostic."""
    cal, _ = _dual(replayed_t=3.6, fresh_t=3.6)
    rep = cal.report()
    assert rep['rayfresh/gap_octaves'] == pytest.approx(0.0)
    assert rep['rayfresh/alpha_star'] == rep['raycal/alpha_star']


def test_per_branch_gaps_come_back_too():
    """Each branch is scored at each alpha under both rules to form the two
    composites, so its own pair costs nothing -- and the per-branch gap is what
    localises the defect to the branches that actually re-sample."""
    p, cal = _armed()

    def two(t_bwd, t_replay):
        def loss(sub):
            a = float(p.detach().item())
            scale = 1.0 + 0.05 * sub
            out = {'bwd': scale * (a - t_bwd) ** 2 + 100.0 * sub,
                   'replay': scale * (a - t_replay) ** 2 + 100.0 * sub}
            out[COMPOSITE] = 0.5 * out['bwd'] + 0.5 * out['replay']
            return out
        return loss

    # replay agrees under both rules; bwd does not
    cal.measure(_subs(), two(1.2, 3.0), two(6.0, 3.0))
    rep = cal.report()
    assert rep['rayfresh/branch/alpha_star_replay'] == \
        rep['raycal/branch/alpha_star_replay']
    assert rep['rayfresh/branch/alpha_star_bwd'] > \
        rep['raycal/branch/alpha_star_bwd']


# ---------------------------------------------------------------------- the RNG --

def test_alphas_are_scored_under_common_random_numbers():
    """WITHOUT THIS THE FRESH READING IS NOISE WITH A t-STAT ATTACHED.

    The bracket differences L(2a) - L(0) are taken WITHIN a sub-batch. If each
    evaluation drew its own path, that difference would be dominated by path
    noise rather than by the parameter change -- which is the exact reason the
    sensor was restricted to stored trajectories before this. `_rng_pinned`
    rewinds before every alpha, so the sampled path is a deterministic function
    of theta and the contrast isolates the step.

    Pinned by construction, not by tolerance: the draw is RECORDED per call, and
    every alpha within a sub-batch must have seen the identical one.
    """
    p, cal = _armed()
    seen = []

    def fresh(sub):
        draw = float(torch.randn(()))
        seen.append(draw)
        a = float(p.detach().item())
        return {COMPOSITE: (1.0 + 0.05 * sub) * (a - 6.0) ** 2 + draw}

    cal.measure(_subs(), _quadratic(p, 1.2), fresh)

    rows = [seen[i:i + len(ALPHAS)] for i in range(0, len(seen), len(ALPHAS))]
    assert len(rows) == cal.n_sub
    for row in rows:
        assert len(set(row)) == 1, 'alphas within a sub-batch must share the draw'
    assert len({row[0] for row in rows}) > 1, \
        'sub-batches must NOT share it -- that would be one replicate, not n_sub'


def test_the_fresh_pass_consumes_no_net_randomness():
    """The replayed pass is RNG-neutral by construction, and that is what makes
    probed and unprobed runs comparable (`Trainer._probe_dealer`, F-039). A
    fresh pass consumes plenty. If it did not restore, the diagnostic would
    change the trajectory of the very run it was measuring.

    Checked as an OBSERVABLE -- the next draw is the one an unprobed run would
    have taken -- not merely as equal state blobs.
    """
    torch.manual_seed(7)
    np.random.seed(7)
    expected = float(torch.randn(()))

    torch.manual_seed(7)
    np.random.seed(7)
    np_before = np.random.get_state()
    p, cal = _armed()
    r = cal.measure(_subs(), _quadratic(p, 1.2), _quadratic(p, 6.0, noisy=True))
    assert r is not None, 'the diagnostic must not cost the primary reading'

    assert float(torch.randn(())) == expected
    np_after = np.random.get_state()
    assert np_before[1].tolist() == np_after[1].tolist()
    assert np_before[2] == np_after[2]


# ------------------------------------------------------------------- staleness --

def test_a_dual_reading_that_cannot_bracket_clears_rather_than_restates():
    """A SENSOR THAT STOPPED AND ONE THAT AGREES MUST NOT LOOK ALIKE. If a fresh
    pass fails to resolve, its previous reading must not be re-published as
    though it were this period's -- that is how a dead channel reads as
    reassurance."""
    p, cal = _armed()
    cal.measure(_subs(), _quadratic(p, 1.2), _quadratic(p, 6.0))
    assert cal.last_fresh

    with torch.no_grad():
        p.zero_()                    # back to theta_before...
    cal._last_done = -1
    assert cal.arm(20)
    with torch.no_grad():
        p.add_(1.0)                  # ...so there is a step of 1.0 to rate
    # a fresh loss with NO composite cannot be bracketed
    cal.measure(_subs(), _quadratic(p, 1.2), lambda sub: {'bwd': float(sub)})
    assert cal.last_fresh == {}
    assert 'rayfresh/alpha_star' not in cal.report()


def test_the_diagnostic_is_off_by_default():
    """It costs a second scoring pass over the whole grid. Off unless asked."""
    p, cal = _armed(dual=False)
    assert cal.dual_score is False
    r = cal.measure(_subs(), _quadratic(p, 1.2))
    assert r is not None
    rep = cal.report()
    assert not [k for k in rep if k.startswith('rayfresh/')]


# ------------------------------------------------------- what resample changes --

class _Bare:
    """Just enough modeller for `LarderScorer.score`."""

    device = 'cpu'
    gfn_model = None
    condition_log_z = None
    step_ind = 0

    def __init__(self):
        self.args = make_args()

    def tb_z_source(self, branch):
        return None


@pytest.mark.parametrize('resample, expected', [(False, 'replayed'), (True, None)])
def test_resample_is_exactly_the_trajectories_argument(monkeypatch, resample,
                                                       expected):
    """THE ONE LINE THE WHOLE DIAGNOSTIC RESTS ON. `get_gfn_backward_loss`
    routes to `get_traj_replay` when handed a trajectory and to `get_traj_bwd`
    -- a fresh backward sample from the current P_B -- when handed None. If this
    argument stopped varying, both passes would score the same objective and the
    gap would read a reassuring zero on every stage."""
    seen = {}

    def fake(bank, terminal, model, log_r, disc, mol, **kw):
        seen.update(kw)
        return torch.tensor(0.0), {}

    monkeypatch.setattr('energy_sampling.lr_larder.get_gfn_backward_loss', fake)
    scorer = LarderScorer(_Bare(), verbose=False)
    stored = torch.zeros(2, 3)
    rec = Harvested(branch='bwd', step=1, condition=None, condition_id=None,
                    log_r=torch.zeros(2), mol_batch=None, traj=stored,
                    repeats=1, scramble_tiles=0, sample_weights=None)
    scorer.score(rec, discretizer=None, resample=resample)
    got = seen['trajectories']
    if expected is None:
        assert got is None, 'resample must hand the evaluator NO path to replay'
    else:
        assert torch.equal(got, stored)
