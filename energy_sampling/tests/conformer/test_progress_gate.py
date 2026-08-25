"""The rate-based exit gate must not fire on a curve that is still descending.

That is the entire point of replacing the slope gate: on tetraglycine the slope criterion
fired at rate 0.04 nats/100 steps while bwd/mle fell in a straight line, because "slow" and
"converged" are different claims about a slowly-decaying curve. So the load-bearing tests
here are the NEGATIVE ones -- a still-improving run must be refused an exit, and the veto
must override an otherwise-passing verdict.
"""
import numpy as np
import pytest

from conformer_modeller import progress_gate

SPEC = {
    'window': 20000, 'horizon': 10000, 'min_history': 20000,
    'rate_bar': 0.02, 'veto_rate': 0.05,
    'metrics': [
        {'key': 'w1r/median', 'target_key': 'w1r/perfect_median', 'bar': 1.5},
        {'key': 'meanE', 'target_key': 'E/ref_median', 'bar': 2.0},
    ],
    'veto_metrics': [{'key': 'w1r/worst', 'target_key': 'w1r/perfect_worst'}],
}


def hist(start, end, target, n=60, last=60000, span=60000, const=None):
    """A geometric approach from `start` to `target`, ending at `end`."""
    t = np.linspace(last - span, last, n)
    if const is not None:
        v = np.full(n, float(const))
    else:
        f = np.linspace(0.0, 1.0, n)
        v = target + (start - target) * ((end - target) / (start - target)) ** f
    return list(zip(t.tolist(), v.tolist()))


def base(w1_end, e_end, mle_end=-300.0):
    return {
        'w1r/median': hist(20.0, w1_end, 0.9),
        'w1r/perfect_median': hist(0.9, 0.9, 0.9, const=0.9),
        'meanE': hist(250.0, e_end, 30.6),
        'E/ref_median': hist(30.6, 30.6, 30.6, const=30.6),
        'bwd/mle': hist(-100.0, mle_end, -1e9),
    }


def test_it_refuses_while_the_fit_is_still_improving_fast():
    """THE FAILURE THE SLOPE GATE HAD. Still descending briskly -- must not exit."""
    h = base(w1_end=5.0, e_end=60.0)
    out = progress_gate(h, SPEC, step=60000)
    assert out['gates/progress_done'] == 0.0
    assert out['progress/w1r/median/rate'] > SPEC['rate_bar'], out


def test_it_exits_as_CONVERGED_when_everything_is_inside_its_bar():
    h = base(w1_end=1.2, e_end=31.5, mle_end=-300.0)
    # flatten the tails so nothing is still moving
    for k in ('w1r/median', 'meanE', 'bwd/mle'):
        v = h[k][-1][1]
        h[k] = h[k][:-20] + [(t, v) for t, _ in h[k][-20:]]
    out = progress_gate(h, SPEC, step=60000)
    assert out['gates/progress_done'] == 1.0, out
    assert out['progress/reason'] == 1.0, 'should report CONVERGED'


def test_it_exits_as_SATURATED_when_stalled_far_from_target():
    """Stopped improving 29 kcal short is a real outcome and must be reported as its own
    reason, not silently as success."""
    h = base(w1_end=7.0, e_end=59.5)
    for k in ('w1r/median', 'meanE', 'bwd/mle'):
        v = h[k][-1][1]
        h[k] = h[k][:-20] + [(t, v) for t, _ in h[k][-20:]]
    out = progress_gate(h, SPEC, step=60000)
    assert out['gates/progress_done'] == 1.0, out
    assert out['progress/reason'] == 2.0, 'should report SATURATED, not converged'


def test_the_veto_blocks_an_otherwise_passing_verdict():
    """A veto metric still moving must block SATURATED even when the triggers have stalled.

    The veto carries a TARGET like every other metric. It previously did not, and was
    normalised by total-progress-so-far instead -- which equals horizon/t for any steadily
    moving series, so it released at t = horizon/veto_rate = 200,000 on every problem
    regardless of timescale. Simulated, that pinned a molecule stalling at step 3,000 to
    200,000 steps.
    """
    h = base(w1_end=7.0, e_end=59.5)
    for k in ('w1r/median', 'meanE'):
        v = h[k][-1][1]
        h[k] = h[k][:-20] + [(t, v) for t, _ in h[k][-20:]]
    t = np.array([a for a, _ in h['w1r/median']])
    # the veto metric is still descending briskly toward its own target
    h['w1r/worst'] = list(zip(t.tolist(), (2.7 + 30.0 * np.exp(-t / 20000.0)).tolist()))
    h['w1r/perfect_worst'] = [(x, 2.7) for x in t.tolist()]
    out = progress_gate(h, SPEC, step=60000)
    assert out['progress/vetoed'] == 1.0, out
    assert out['gates/progress_done'] == 0.0, out


def test_the_veto_cannot_block_the_converged_verdict():
    """A metric inside its bar is done whether or not more steps would still help, so the
    veto must scope to SATURATED alone."""
    h = base(w1_end=1.2, e_end=31.5)
    t = np.array([a for a, _ in h['w1r/median']])
    h['w1r/worst'] = list(zip(t.tolist(), (2.7 + 30.0 * np.exp(-t / 20000.0)).tolist()))
    h['w1r/perfect_worst'] = [(x, 2.7) for x in t.tolist()]
    out = progress_gate(h, SPEC, step=60000)
    assert out['gates/progress_done'] == 1.0, out
    assert out['progress/reason'] == 1.0, 'converged should not be vetoable' 


def test_it_abstains_before_enough_history_exists():
    """A verdict from two points is a guess; abstaining is the safe direction."""
    h = base(w1_end=1.0, e_end=30.7)
    assert progress_gate(h, SPEC, step=5000)['gates/progress_done'] == 0.0


def test_a_missing_target_key_does_not_fabricate_a_verdict():
    h = base(w1_end=1.0, e_end=30.7)
    del h['w1r/perfect_median']
    del h['E/ref_median']
    out = progress_gate(h, SPEC, step=60000)
    assert out['gates/progress_done'] == 0.0, 'no targets -> no verdict'


def test_a_brief_pause_does_not_trigger_an_exit():
    """A pause SHORTER than the fit window must not read as convergence.

    The protection is the window length itself: the slope is fit over the trailing 20k, so a
    10k pause still has 10k of descent inside the fit and the projected rate stays high. This
    replaces an earlier confirm_windows=2 rule, which was measured to be the worse way to
    spend the same lookback -- ANDing two noisy 10k tests detects true saturation ~42% of the
    time against ~81% for one 20k fit.
    """
    t = np.linspace(0.0, 60000.0, 121)
    v = np.where(t <= 50000.0, 20.0 * np.exp(-t / 22000.0), 20.0 * np.exp(-50000.0 / 22000.0))
    h = {
        'w1r/median': list(zip(t.tolist(), (0.9 + v).tolist())),
        'w1r/perfect_median': [(x, 0.9) for x in t],
        'meanE': [(x, 30.6 + 0.5) for x in t],
        'E/ref_median': [(x, 30.6) for x in t],
        'bwd/mle': [(x, -300.0) for x in t],
    }
    out = progress_gate(h, SPEC, step=60000)
    assert out['gates/progress_done'] == 0.0, (
        'exited on a 10k pause while the 20k fit window still contained descent')


def test_a_pause_longer_than_the_window_does_trigger():
    """The complement: once the plateau fills the whole window, the gate must fire. A rule
    that never fires is as useless as one that always does."""
    t = np.linspace(0.0, 60000.0, 121)
    v = np.where(t <= 33000.0, 20.0 * np.exp(-t / 22000.0), 20.0 * np.exp(-33000.0 / 22000.0))
    h = {
        'w1r/median': list(zip(t.tolist(), (0.9 + v).tolist())),
        'w1r/perfect_median': [(x, 0.9) for x in t],
        'meanE': [(x, 30.6 + 0.5) for x in t],
        'E/ref_median': [(x, 30.6) for x in t],
        'bwd/mle': [(x, -300.0) for x in t],
    }
    out = progress_gate(h, SPEC, step=60000)
    assert out['gates/progress_done'] == 1.0, out
    assert out['progress/reason'] == 2.0, 'stalled far from target -> SATURATED'


def test_the_horizon_is_free_and_does_not_touch_the_estimator():
    """The point of projecting rather than window-averaging: N sets the question, not the
    measurement. One fitted decay must answer any horizon, as 1 - exp(slope*N)."""
    t = np.linspace(0.0, 60000.0, 121)
    v = 20.0 * np.exp(-t / 25000.0)
    h = {
        'w1r/median': list(zip(t.tolist(), (0.9 + v).tolist())),
        'w1r/perfect_median': [(x, 0.9) for x in t],
        'meanE': [(x, 30.6 + 0.5) for x in t],
        'E/ref_median': [(x, 30.6) for x in t],
        'bwd/mle': [(x, -300.0) for x in t],
    }
    got = {}
    for N in (1000, 10000, 100000):
        spec = dict(SPEC); spec['horizon'] = N
        got[N] = progress_gate(h, spec, step=60000)['progress/w1r/median/rate']
    for N, expect in ((1000, 0.039), (10000, 0.330), (100000, 0.982)):
        assert abs(got[N] - expect) < 0.03, (N, got[N], expect)


def test_an_easy_molecule_exits_as_soon_as_it_is_actually_good():
    """A problem that converges at ~3k must not wait out the slope window.

    The CONVERGED branch is a level test and needs no rate, so holding it behind a 20k slope
    window would burn 17k steps proving something already true. Only SATURATED -- a claim
    about a derivative -- needs the window.
    """
    t = np.arange(0.0, 6000.0, 250.0)
    v = 20.0 * np.exp(-t / 600.0)
    h = {
        'w1r/median': list(zip(t.tolist(), (0.9 + v).tolist())),
        'w1r/perfect_median': [(x, 0.9) for x in t],
        'meanE': list(zip(t.tolist(), (30.6 + 0.3 + v * 0.05).tolist())),
        'E/ref_median': [(x, 30.6) for x in t],
        'bwd/mle': list(zip(t.tolist(), (-300.0 - t / 50.0).tolist())),
    }
    spec = dict(SPEC); spec['min_history'] = 2000; spec['level_window'] = 2500
    cut = {k: [(a, b) for a, b in vv if a <= 3000.0] for k, vv in h.items()}
    out = progress_gate(cut, spec, step=3000)
    assert out['gates/progress_done'] == 1.0, out
    assert out['progress/reason'] == 1.0, 'level test should report CONVERGED'


def test_a_still_improving_run_is_not_let_out_by_the_level_path():
    """The complement, and the one that matters: decoupling the level test must not become a
    back door. Far from target and still descending -> no exit by either route."""
    t = np.arange(0.0, 30000.0, 250.0)
    v = 20.0 * np.exp(-t / 40000.0)
    h = {
        'w1r/median': list(zip(t.tolist(), (0.9 + v).tolist())),
        'w1r/perfect_median': [(x, 0.9) for x in t],
        'meanE': list(zip(t.tolist(), (30.6 + 25.0).tolist() if False else [30.6 + 25.0] * len(t))),
        'E/ref_median': [(x, 30.6) for x in t],
        'bwd/mle': list(zip(t.tolist(), (-300.0 - t / 50.0).tolist())),
    }
    spec = dict(SPEC); spec['min_history'] = 2000; spec['level_window'] = 2500
    out = progress_gate(h, spec, step=30000)
    assert out['gates/progress_done'] == 0.0, out


def test_a_metric_moving_backwards_is_reported_as_DEGRADING_not_saturated():
    """Under the bar for two different reasons: stopped, or going backwards. Both are exits,
    but 'saturated' reads as finished and would hide a model that is getting worse."""
    t = np.linspace(0.0, 60000.0, 121)
    v = 3.0 + 0.6 * (t / 60000.0)          # steadily WORSENING, far from target
    h = {
        'w1r/median': list(zip(t.tolist(), (0.9 + v).tolist())),
        'w1r/perfect_median': [(x, 0.9) for x in t],
        'meanE': [(x, 30.6 + 25.0) for x in t],
        'E/ref_median': [(x, 30.6) for x in t],
        'w1r/worst': [(x, 2.7) for x in t],
        'w1r/perfect_worst': [(x, 2.7) for x in t],
    }
    out = progress_gate(h, SPEC, step=60000)
    assert out['progress/degrading'] == 1.0, out
    if out['gates/progress_done'] == 1.0:
        assert out['progress/reason'] == 3.0, 'must not be labelled SATURATED'
