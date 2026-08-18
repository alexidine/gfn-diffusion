"""
The three changes that make the hypergradient loop worth trusting with authority:
a validity gate on cos, an asymmetric gain, and a leak.

THE PROBLEM EACH ONE SOLVES.

1. cos STOPS BEING A LEARNING-RATE STATISTIC IN A BLOWUP. Once the gradient clip
   binds on essentially every step, the update magnitude is set by the LR alone
   and decoupled from curvature, so the cosine measures clip geometry. Measured
   2026-08-17 (hyperslope_aug17, QM9 conditional): `gradclip/fused_fire_rate` is
   0.000 through every healthy window and 1.000 through every window where cos
   misreads, with lr2e4's pre-clip norm at 3.7e4 against a healthy 37. In the arm
   that died at 5e-4, cos sat at -0.067 (100% negative) for 970 steps and then
   RELAXED toward -0.006 as the run detonated -- i.e. it eased off exactly when
   it should have braked. The gate cuts on the clip evidence instead, because
   sustained saturation is unambiguous about the rate even when cos is not.

2. THE GAIN WAS SYMMETRIC IN PRACTICE. `beta_down` was plumbed and set by nothing,
   so a speculative raise and a corrective cut moved the rate equally. `ray`
   ships eta_up 0.25 against eta_down 0.5 for the opposite reason.

3. IT WAS A PURE INTEGRATOR. Pole on the unit circle, so any constant bias in the
   error drifts the rate exponentially and zero-mean noise random-walks it; the
   `bounds` clip is saturation, not a restoring force.

EVERY DEFAULT-OFF SWITCH IS TESTED AS OFF. A safety mechanism that quietly
changed unrelated runs would be a worse bug than the one it fixes.
"""

import math

import pytest

from controller import LRController

pytestmark = pytest.mark.fast

SEED_LR = 1.25e-4
BETA = 0.05


class _Bag:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _controller(**adaptive_overrides):
    adaptive = dict(warmup_steps=0, seed_lr=SEED_LR, bounds=(0.01, 2000.0),
                    divergence_loss_abs=1.0e9, divergence_grad_abs=1.0e9,
                    divergence_cut=0.5, envelope_freeze_drop=None,
                    restart_after=None, control_flow_lr=False)
    adaptive.update(adaptive_overrides)
    args = _Bag(adaptive_lr=_Bag(**adaptive), lr_warmup_ratio=1, min_lr=1.0e-9,
                max_lr=None, lr_policy=SEED_LR, lr_back=SEED_LR,
                lr_replay=SEED_LR, lr_fused=SEED_LR, lr_flow=0.1,
                lr_servo_managed=['lr_fused'])
    modeller = _Bag(
        args=args, step_ind=0, phase=0, lr_ctrl=None,
        optimizers={'fused': _Bag(param_groups=[{'lr': 0.0}, {'lr': 0.0}]),
                    'fwd': _Bag(param_groups=[{'lr': 0.0}])},
        protocol=_Bag(stage=_Bag(lr_sensor={'kind': 'hyper', 'beta': BETA})))
    return modeller, LRController(modeller)


def _peak(ctrl):
    return float(ctrl._state()['peak_scale'])


def _fire(modeller, ctrl, cos, n=1, clip_ratio=None, **kw):
    for _ in range(n):
        modeller.step_ind += 2
        ctrl.on_hypergradient(cos, BETA, kw.get('beta_down'),
                              kw.get('cos_target', 0.0), clip_ratio=clip_ratio)


# ------------------------------------------------------- 1. the regime gate

def test_sustained_clip_saturation_cuts_the_rate():
    """cos says "too cold" (+0.5) throughout; the clip says the rate is far too
    high. The clip must win -- this is the case where trusting cos kills the run."""
    m, c = _controller()
    _fire(m, c, cos=+0.5, n=200, clip_ratio=5.0)
    assert _peak(c) < 1.0, (
        f'peak_scale is {_peak(c):.4g}: the loop followed cos upward while the '
        f'clip was saturated, which is the failure this gate exists to stop')


def test_an_occasional_clip_firing_does_not_cut():
    """The guard TARGETS a 1-p fire rate (0.01 at the shipped p=0.99), so
    single firings are the design, not evidence."""
    m, c = _controller()
    for i in range(200):
        _fire(m, c, cos=+0.5, clip_ratio=(5.0 if i % 100 == 0 else 0.2))
    assert _peak(c) > 1.0, 'a 1% fire rate must not be read as saturation'


def test_the_gate_needs_a_full_window_before_it_can_fire():
    """One reading cannot end anything -- the same discipline the warmup freeze
    uses. Below the window the loop must still be following cos."""
    m, c = _controller(hyper_clip_window=50)
    _fire(m, c, cos=+0.5, n=10, clip_ratio=5.0)
    assert _peak(c) > 1.0, 'the gate fired before a full window of evidence'


def test_the_gate_is_rate_limited_not_compounding():
    """Cutting on every firing while saturation persists would compound to
    0.5**n and floor the rate inside one window. Each cut re-arms the detector,
    so sustained saturation costs a halving per window, not per step."""
    m, c = _controller(hyper_clip_window=50, hyper_clip_cut=0.5)
    _fire(m, c, cos=+0.5, n=400, clip_ratio=5.0)
    # 400 firings / 50-step window -> at most ~8 cuts, so 0.5**8 ~ 3.9e-3
    assert _peak(c) > 0.5 ** 12, \
        f'peak_scale {_peak(c):.4g} fell far faster than one cut per window'


def test_the_gate_can_be_disabled():
    m, c = _controller(hyper_clip_fire_rate_max=None)
    _fire(m, c, cos=+0.5, n=200, clip_ratio=5.0)
    assert _peak(c) > 1.0, 'a null bar must disable the gate entirely'


def test_a_gate_cut_is_reported():
    m, c = _controller()
    _fire(m, c, cos=+0.5, n=200, clip_ratio=5.0)
    c.step()
    report = c.report()
    assert report.get('lr_ctrl/hyper_clip_cuts', 0) > 0, \
        'the gate braked but nothing said so'


def test_absent_clip_ratio_is_inert():
    """Callers that do not supply the evidence must behave exactly as before."""
    m, c = _controller()
    _fire(m, c, cos=+0.5, n=200, clip_ratio=None)
    assert _peak(c) > 1.0


# --------------------------------------------------- 2. the asymmetric gain

def test_cuts_move_further_than_raises_by_default():
    m_up, c_up = _controller()
    _fire(m_up, c_up, cos=+0.2, n=1)
    up = abs(math.log(_peak(c_up)))

    m_dn, c_dn = _controller()
    _fire(m_dn, c_dn, cos=-0.2, n=1)
    down = abs(math.log(_peak(c_dn)))

    assert down == pytest.approx(2.0 * up, rel=1e-6), \
        f'down/up gain ratio is {down / up:.3f}, expected the 2.0 default'


def test_symmetry_is_restorable():
    m, c = _controller(hyper_down_gain=1.0)
    _fire(m, c, cos=-0.2, n=1)
    assert math.log(_peak(c)) == pytest.approx(-BETA * 0.2, rel=1e-6)


def test_an_explicit_beta_down_still_wins():
    m, c = _controller()
    _fire(m, c, cos=-0.2, n=1, beta_down=0.01)
    assert math.log(_peak(c)) == pytest.approx(-0.01 * 0.2, rel=1e-6)


# ------------------------------------------------------------- 3. the leak

def test_without_a_leak_a_constant_bias_runs_away():
    """The behaviour being fixed, asserted so the fix has something to beat."""
    m, c = _controller()
    _fire(m, c, cos=+0.04, n=3000)
    assert _peak(c) > 50.0, \
        f'peak_scale only reached {_peak(c):.4g}; the integrator should ramp'


def test_a_leak_bounds_a_constant_bias():
    """Stationary point is b*err/lam. At beta 0.05, err 0.04, lam 2e-3 that is
    1.0 nat ~ 2.7x -- finite, where the un-leaked loop is not."""
    lam, err = 2.0e-3, 0.04
    m, c = _controller(peak_leak=lam)
    _fire(m, c, cos=err, n=6000)
    expected = math.exp(BETA * err / lam)
    assert _peak(c) == pytest.approx(expected, rel=0.05), \
        f'settled at {_peak(c):.4g}, expected ~{expected:.4g}'


def test_the_leak_is_off_by_default():
    """lam encodes a timescale and has been measured on one route only, so the
    default must reproduce today's behaviour bit for bit."""
    m_a, c_a = _controller()
    m_b, c_b = _controller(peak_leak=0.0)
    _fire(m_a, c_a, cos=+0.03, n=500)
    _fire(m_b, c_b, cos=+0.03, n=500)
    assert _peak(c_a) == _peak(c_b) == pytest.approx(math.exp(BETA * 0.03 * 500))


# ------------------------- the gate must not read an uncalibrated guard

def test_gate_is_inert_while_the_guard_is_warming():
    """THE FALSE POSITIVE THAT SPOILED newlogic_qm9cond_newlogic.

    `grad_clip_guard` clips against the STATIC fallback bar for a branch's first
    `warmup_steps` observations, and re-warms at every stage transition when
    refresh_on_stage is set. A 100% fire rate there is about the BAR, not the
    rate. Measured: var_conditioning opened at step 150 with fused grad norms
    155.7 / 111 / 67 / 53 / 44 against a static bar of 37.88, so fused_fire_rate
    sat at 1.000 through the warmup and collapsed to 0.000 the moment the fitted
    bar took over -- and the gate cut on the stale EMA at step 250, AFTER the
    condition had already cleared.

    train.py withholds clip_ratio (passes None) while the guard is warming. This
    pins the controller half of that contract: no evidence, no cut, however long
    the run goes.
    """
    m, c = _controller()
    _fire(m, c, cos=+0.5, n=400, clip_ratio=None)
    assert _peak(c) > 1.0, 'the gate acted with no clip evidence supplied'
    c.step()
    assert 'lr_ctrl/hyper_clip_cuts' not in c.report()


def test_a_transient_saturation_does_not_latch():
    """The cut is self-clearing: once the clip stops firing, the EMA decays and
    the gate stops braking. What must NOT happen is a brief saturation leaving a
    permanent mark on the loop -- that is how a transient latched the envelope
    freeze in newlogic_qm9cond_newlogic."""
    m, c = _controller(hyper_clip_window=50)
    _fire(m, c, cos=+0.05, n=120, clip_ratio=5.0)      # saturated
    cut_to = _peak(c)
    _fire(m, c, cos=+0.05, n=400, clip_ratio=0.2)      # cleared
    assert _peak(c) > cut_to, \
        'the loop never resumed after saturation cleared -- the gate latched'


# ------------------- the warmup hold must end when the ramp does

def test_hyper_actuates_once_the_envelope_is_frozen():
    """THE 800 DEAD STEPS. hyper is freeze-only while the ramp is MOVING, because
    a cosine read through a deliberately-suppressed rate says "too cold" whatever
    the operating point is. Once the envelope is frozen that argument expires --
    the envelope is a constant and the reading is ordinary evidence.

    Measured, newlogic_qm9cond_newlogic: the stage opened at 150, froze at 350 on
    a negative smoothed cos, and hyper stayed mute until 1150 with hypergrads at
    0 and cos at -0.02..-0.04 the whole way.
    """
    m, c = _controller(warmup_steps=100000, warmup_freeze_cos_window=5)
    st = c._state()

    _fire(m, c, cos=-0.2, n=40)                 # drives the smoothed error < 0
    assert st.get('envelope_frozen_at') is not None, 'fixture never froze the ramp'
    frozen_at_peak = _peak(c)

    _fire(m, c, cos=-0.2, n=40)                 # still far inside warmup_steps
    assert _peak(c) < frozen_at_peak, (
        'hyper is still muted after the ramp froze -- the hold is keyed on the '
        'step budget rather than on whether the envelope is actually moving')


def test_hyper_stays_muted_while_the_ramp_is_actually_moving():
    """The other half: before any freeze, actuation must still be withheld."""
    m, c = _controller(warmup_steps=100000, warmup_freeze_cos_window=100000)
    _fire(m, c, cos=+0.5, n=60)
    assert _peak(c) == pytest.approx(1.0), \
        'hyper actuated during a live ramp, which is what freeze-only prevents'
