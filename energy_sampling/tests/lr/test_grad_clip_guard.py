"""
Tests for grad_clip_guard.py.

WHAT HAS TO BE PROVEN, in the order it matters:

  1. The bar converges to the p-th quantile of a distribution the WARMUP MODEL
     GETS WRONG. Seeding from a lognormal fit and then testing on lognormal data
     would pass with the tracker deleted -- so every convergence test here runs
     on a heavy-tailed mixture the lognormal seed necessarily misestimates, and
     what is asserted is the realized firing rate, which is the defining
     property of a quantile.
  2. BOUNDED INFLUENCE. `test_a_587x_spike_and_a_1p001x_exceedance_move_the_bar_
     identically` is the mutation guard for this whole module: rewrite the update
     to read the magnitude (`tau *= (norm/tau)**eta`, an EMA, a running mean --
     any of the estimators docs/to_do_rebuild.md:753 records failing) and it
     fails. Nothing else here would catch that.
  3. The cap holds in BOTH directions. Upward is the diverging-run case; the
     downward one is the quieter hazard, because a bar that sinks under the body
     of the distribution silently rebuilds the always-binding preconditioner the
     module exists to replace, and every other metric would look fine.
  4. Per-branch isolation, since a shared bar was the original defect.
  5. Inertness when disabled -- every config in the tree today.

    python test_grad_clip_guard.py        (or: pytest test_grad_clip_guard.py)
"""
import argparse
import math
import os
import random
import sys

import pytest

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from grad_clip_guard import CHANNELS, GradClipGuard

STATIC = 37.88  # the resolver's T=10/W=512 value, i.e. what this replaces


def _guard(**kw):
    kw.setdefault('static_clip', STATIC)
    kw.setdefault('enabled', True)
    kw.setdefault('warmup_steps', 50)
    return GradClipGuard(**kw)


def _mixture(n, seed=12345):
    """95% ordinary steps, 5% an order of magnitude larger.

    Deliberately NOT lognormal: the warmup seeds from a lognormal fit, so a
    lognormal test would be satisfied by the seed alone and would still pass
    with the quantile update removed entirely.
    """
    rng = random.Random(seed)
    return [math.exp(rng.gauss(3.0, 0.5)) if rng.random() < 0.05
            else math.exp(rng.gauss(0.0, 0.5))
            for _ in range(n)]


def _drive(guard, samples, channel='fwd'):
    """Feed samples, returning (fired flags, bars used) in lockstep."""
    fired, bars = [], []
    for x in samples:
        bar = guard.threshold(channel)
        bars.append(bar)
        fired.append(x > bar)
        guard.observe(channel, x)
    return fired, bars


# --------------------------------------------------------------- convergence

@pytest.mark.parametrize('p', [0.9, 0.95])
def test_realized_firing_rate_converges_to_one_minus_p(p):
    """The defining property. Measured on the TAIL of the run, after the
    tracker has had time to correct the seed."""
    n = 40000
    samples = _mixture(n)
    g = _guard(p=p, eta=0.02)
    fired, _ = _drive(g, samples)
    tail = fired[-15000:]
    rate = sum(tail) / len(tail)
    assert abs(rate - (1.0 - p)) < 0.25 * (1.0 - p), (
        f'p={p}: realized firing rate {rate:.4f}, wanted ~{1 - p:.4f}')


def test_the_bar_lands_on_the_empirical_quantile():
    """Same run, scored against the sample's own order statistic rather than
    against the firing rate, so a tracker that oscillated around the wrong
    level with the right rate could not pass both."""
    samples = _mixture(40000)
    g = _guard(p=0.9, eta=0.02)
    _drive(g, samples)
    truth = sorted(samples)[int(0.9 * len(samples))]
    tau = g._branches['fwd'].tau
    assert 0.7 * truth < tau < 1.4 * truth, f'bar {tau:.4g} vs empirical q0.9 {truth:.4g}'


def test_the_warmup_seed_alone_is_not_already_correct():
    """Guards the two tests above from being vacuous. If the lognormal seed
    happened to land on the right answer for this mixture, they would pass with
    the quantile update deleted, and this asserts it does not."""
    samples = _mixture(40000)
    g = _guard(p=0.9, eta=0.02)
    _drive(g, samples[:50])           # exactly the warmup, no tracker steps yet
    seeded = g._branches['fwd'].tau
    truth = sorted(samples)[int(0.9 * len(samples))]
    assert seeded is not None
    assert not (0.7 * truth < seeded < 1.4 * truth), (
        f'lognormal seed {seeded:.4g} already inside the tolerance band around '
        f'{truth:.4g} -- the convergence tests are not testing the tracker')


# ---------------------------------------------------------- bounded influence

def test_a_587x_spike_and_a_1p001x_exceedance_move_the_bar_identically():
    """THE MUTATION GUARD FOR THIS MODULE.

    587x is the pre-clip norm the aug02 arm died at (docs/to_do_rebuild.md:231).
    The update reads the INDICATOR, never the magnitude, so the two must produce
    a bit-identical bar. Any magnitude-driven estimator -- an EMA of the norm, a
    running mean, tau *= (norm/tau)**eta -- separates them and fails here.
    """
    eta, p = 0.01, 0.99
    big, small = _guard(p=p, eta=eta), _guard(p=p, eta=eta)
    warm = _mixture(50)
    _drive(big, warm)
    _drive(small, warm)
    tau0 = big._branches['fwd'].tau
    assert tau0 == small._branches['fwd'].tau

    big.observe('fwd', tau0 * 587.0)
    small.observe('fwd', tau0 * 1.001)

    assert big._branches['fwd'].tau == small._branches['fwd'].tau
    assert big._branches['fwd'].tau == pytest.approx(tau0 * math.exp(eta * p), rel=1e-12)


def test_one_spike_costs_exactly_eta_in_log_space():
    """The quantitative claim the design rests on: at eta=0.01 a spike moves the
    bar by ~1%, so no single observation can meaningfully recalibrate it."""
    g = _guard(p=0.99, eta=0.01)
    _drive(g, _mixture(50))
    tau0 = g._branches['fwd'].tau
    g.observe('fwd', tau0 * 1e6)
    assert abs(math.log(g._branches['fwd'].tau / tau0)) <= 0.01 + 1e-12


def test_a_sustained_excursion_cannot_walk_the_bar_past_the_cap():
    """Per-step boundedness is not cumulative boundedness: 5000 consecutive
    exceedances at eta=0.01 would be e^49 without the cap."""
    g = _guard(p=0.99, eta=0.01, max_ratio=100.0)
    _drive(g, _mixture(50))
    base = g._branches['fwd'].baseline
    for _ in range(5000):
        g.observe('fwd', 1e12)
    st = g._branches['fwd']
    assert st.tau == pytest.approx(base * 100.0)
    assert st.n_saturated > 0, 'saturation must be COUNTED, not just clamped'


def test_the_bar_cannot_sink_into_always_binding():
    """The quieter half of the cap. Without a floor the bar drifts under the
    body of the distribution and the guard becomes the preconditioner again --
    with a firing rate that still reads near 1-p, so nothing else catches it."""
    # p=0.9 rather than 0.99 deliberately: downward drift is eta*(1-p) per quiet
    # step against eta*p upward, so at p=0.99 reaching the floor takes
    # ln(100)/1e-4 ~ 46k steps. The asymmetry is real and documented in the
    # module; this test is about the floor existing, not about that timescale.
    g = _guard(p=0.9, eta=0.02, max_ratio=100.0)
    _drive(g, _mixture(50))
    base = g._branches['fwd'].baseline
    for _ in range(5000):
        g.observe('fwd', base * 1e-9)   # never fires: pure downward drift
    assert g._branches['fwd'].tau == pytest.approx(base / 100.0)


def test_a_non_finite_norm_does_not_move_the_bar():
    """train.py skips the optimizer step on a non-finite gradient, so it is
    neither a quantile observation nor a training event. Folding it in would let
    a NaN streak ratchet the bar upward while no learning happens at all."""
    g = _guard(p=0.99, eta=0.01)
    _drive(g, _mixture(50))
    tau0 = g._branches['fwd'].tau
    for bad in (float('inf'), float('nan'), float('-inf')):
        g.observe('fwd', bad)
    assert g._branches['fwd'].tau == tau0
    assert g.report()['gradclip/nonfinite'] == 3.0


# ------------------------------------------------------------- per-branch

def test_branches_track_independently():
    """The original defect: one bar across four gradient distributions is set by
    whichever branch dominates the step mixture."""
    g = _guard(p=0.9, eta=0.02)
    _drive(g, [x * 1000.0 for x in _mixture(4000)], channel='bwd')
    _drive(g, _mixture(4000, seed=7), channel='fwd')
    fwd, bwd = g._branches['fwd'].tau, g._branches['bwd'].tau
    assert bwd / fwd > 100.0, f'bwd bar {bwd:.4g} vs fwd bar {fwd:.4g} -- not separated'


def test_feeding_one_branch_never_moves_another():
    g = _guard(p=0.9, eta=0.02)
    _drive(g, _mixture(2000), channel='fwd')
    before = {c: g._branches[c].tau for c in CHANNELS}
    _drive(g, [1e9] * 500, channel='replay')
    for c in CHANNELS:
        if c != 'replay':
            assert g._branches[c].tau == before[c]


def test_unknown_step_type_raises_rather_than_falling_back():
    """A silent fallback to the static bar would leave the run looking guarded
    and not being -- the inert-flag failure mode."""
    g = _guard()
    with pytest.raises(KeyError):
        g.threshold('z_cal')
    with pytest.raises(KeyError):
        g.observe('z_cal', 1.0)


# ---------------------------------------------------------------- refresh

def test_refresh_holds_the_outgoing_bar_live_while_recalibrating():
    """A hard reset would leave the run unguarded across exactly the turbulence
    a stage boundary creates."""
    g = _guard(p=0.9, eta=0.02, warmup_steps=50)
    _drive(g, _mixture(2000))
    old = g._branches['fwd'].tau
    assert g.refresh(reason='test') is True
    assert g.threshold('fwd') == old, 'bar dropped at the boundary'
    for x in _mixture(49, seed=3):
        g.observe('fwd', x)
        assert g.threshold('fwd') == old, 'bar moved before recalibration completed'
    g.observe('fwd', 1.0)                        # the 50th: recalibration lands
    assert g.threshold('fwd') != old
    assert g._branches['fwd'].warming is False


def test_refresh_rebases_the_saturation_cap():
    """A stage whose gradients genuinely live 50x higher must not spend its
    whole life pinned at the previous stage's ceiling."""
    g = _guard(p=0.9, eta=0.02, max_ratio=10.0)
    _drive(g, _mixture(2000))
    g.refresh()
    _drive(g, [x * 1000.0 for x in _mixture(50, seed=5)])
    assert g._branches['fwd'].baseline == g._branches['fwd'].tau
    assert g._branches['fwd'].tau > 100.0


def test_refresh_is_a_no_op_when_disabled_or_opted_out():
    assert _guard(refresh_on_stage=False).refresh() is False
    assert _guard(enabled=False).refresh() is False


# ------------------------------------------------------------------ inertness

def test_disabled_guard_is_exactly_the_static_clip():
    """Every config in the tree today. The bar must be the constant that
    train.py's clip_grad_norm_ call used to read, on every channel, forever."""
    g = _guard(enabled=False)
    for c in CHANNELS:
        assert g.threshold(c) == STATIC
        g.observe(c, 1e9)
        assert g.threshold(c) == STATIC
    assert g.report() == {}


def test_warmup_clips_at_the_static_bar_by_default():
    g = _guard(p=0.9, warmup_clip='static')
    assert g.threshold('fwd') == STATIC
    _drive(g, _mixture(50))
    assert g.threshold('fwd') != STATIC


def test_warmup_clip_off_leaves_the_branch_unclipped():
    g = _guard(p=0.9, warmup_clip='off')
    assert g.threshold('fwd') == float('inf')


# -------------------------------------------------------------------- config

@pytest.mark.parametrize('kw', [
    {'p': 0.0}, {'p': 1.0}, {'p': -0.5},
    {'eta': 0.0}, {'eta': -1e-3},
    {'warmup_steps': 29},
    {'warmup_clip': 'auto'},
    {'max_ratio': 1.0},
])
def test_invalid_config_raises_at_construction(kw):
    with pytest.raises(ValueError):
        _guard(**kw)


def test_unknown_config_key_is_a_hard_error():
    """`percentile: 0.9` where `p: 0.9` was meant reads as one bar and behaves
    as another, with nothing in the log to say so."""
    # argparse.Namespace, because that is exactly what dict2namespace hands the
    # modeller -- a class with CLASS attributes would leave vars() empty and the
    # test would pass against a from_config that checked nothing.
    cfg = argparse.Namespace(enabled=True, percentile=0.9)
    with pytest.raises(ValueError, match='percentile'):
        GradClipGuard.from_config(STATIC, cfg)


def test_every_real_config_key_is_accepted():
    """The other half: the strict check must not reject the documented schema."""
    cfg = argparse.Namespace(enabled=True, p=0.95, eta=0.02, warmup_steps=200,
                             warmup_clip='off', max_ratio=50.0, refresh_on_stage=False)
    g = GradClipGuard.from_config(STATIC, cfg)
    assert (g.enabled, g.p, g.eta, g.warmup_steps) == (True, 0.95, 0.02, 200)
    assert (g.warmup_clip, g.max_ratio, g.refresh_on_stage) == ('off', 50.0, False)


def test_absent_config_block_is_disabled():
    g = GradClipGuard.from_config(STATIC, None)
    assert g.enabled is False
    assert g.threshold('fused') == STATIC


def test_enabled_guard_requires_a_usable_warmup_fallback():
    with pytest.raises(ValueError):
        GradClipGuard(static_clip=float('inf'), enabled=True)


# ------------------------------------------------------------------- report

def test_report_drains_and_omits_branches_that_never_ran():
    g = _guard(p=0.9, eta=0.02)
    _drive(g, _mixture(500))
    r = g.report()
    assert r['gradclip/fwd_n'] == 500.0
    assert 'gradclip/fwd_tau' in r and 'gradclip/fwd_fire_rate' in r
    for c in ('bwd', 'replay', 'fused'):
        assert f'gradclip/{c}_tau' not in r, f'{c} never ran but published a bar'
    assert g.report()['gradclip/fwd_n'] == 0.0, 'window counters must drain'


def test_fire_rate_distinguishes_guard_from_preconditioner():
    """The one number that says which algorithm is running."""
    g = _guard(p=0.9, eta=0.02)
    _drive(g, _mixture(20000))
    g.report()                                  # drain the transient
    _drive(g, _mixture(5000, seed=99))
    assert 0.05 < g.report()['gradclip/fwd_fire_rate'] < 0.20

    # The preconditioner signature, read over a SHORT window: pin the bar under
    # the body of the distribution and every step binds. It is deliberately short
    # because the saturation floor plus the upward ratchet actively climb back
    # out of this state -- which is the mechanism working, and is covered by
    # test_the_bar_cannot_sink_into_always_binding.
    always = _guard(p=0.9, eta=0.02)
    _drive(always, _mixture(50))
    always._branches['fwd'].tau = always._branches['fwd'].baseline / 1e3
    always.report()
    _drive(always, _mixture(50, seed=4))
    assert always.report()['gradclip/fwd_fire_rate'] == 1.0


# -------------------------------------------------------------------- state

def test_state_dict_round_trip_restores_the_bar():
    """A divergence rewind must not drop the tracker back into warmup with the
    run already unstable."""
    g = _guard(p=0.9, eta=0.02)
    _drive(g, _mixture(2000))
    saved = g.state_dict()
    fresh = _guard(p=0.9, eta=0.02)
    assert fresh.load_state_dict(saved) is True
    for c in CHANNELS:
        assert fresh._branches[c].tau == g._branches[c].tau
        assert fresh._branches[c].warming == g._branches[c].warming


@pytest.mark.parametrize('bad', [None, {}, {'ver': 999}, 'nonsense'])
def test_stale_state_is_discarded_not_reinterpreted(bad):
    g = _guard(p=0.9)
    assert g.load_state_dict(bad) is False
    assert g._branches['fwd'].tau is None


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
