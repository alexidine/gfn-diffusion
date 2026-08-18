"""
`max_lr`: an absolute rail on every rate the controller writes.

WHY A RAIL AND NOT NARROWER BOUNDS. `adaptive_lr.bounds` is wide on purpose --
the servo is meant to discover its own operating range, and narrowing bounds to
express a safety limit would delete the exploration along with the hazard. This
is the other thing: a hard number in absolute learning-rate units that no group
may exceed however the servo arrived there.

THE MEASUREMENT BEHIND IT (hyperslope_aug17, 2026-08-17, QM9 conditional, one
pinned rate per arm): 5e-6, 1.25e-5, 3e-5 and 8e-5 each run 2000 steps with zero
non-finite gradients; 2e-4 goes non-finite at step 1560; 5e-4 at step 560. So the
survivable band is roughly 16x wide, while the default `bounds` span 200,000x --
four orders of magnitude more room than the run tolerates.

WHAT THESE TESTS PIN:

  * absent max_lr changes nothing -- the default must reproduce the behaviour of
    every config written before the key existed;
  * the cap binds on the PRODUCT (base x peak_scale x envelope), not on any one
    factor, so a servo that has climbed cannot walk through it;
  * it binds on the FLOW group, which is the one rate with no other guard: the
    envelope never reaches it, peak_scale never reaches it while control_flow_lr
    is false, and a divergence cut cannot move it;
  * it binds on EXPLICIT-FLOAT groups too, since the rail is about what the
    optimizer receives rather than about who chose the number;
  * min_lr still wins where it should, and an incoherent pair is REFUSED at
    construction rather than resolved by clamp order -- whichever clamp ran
    second would otherwise silently defeat the other, and the run would train at
    a rate neither bound describes;
  * a binding rail is VISIBLE. A clamped controller and a satisfied one are
    indistinguishable from the rate alone.
"""

import math

import pytest

from controller import LRController

pytestmark = pytest.mark.fast

SEED_LR = 1.25e-4


class _Bag:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _controller(max_lr=None, min_lr=1.0e-6, lr_flow=0.1, managed=('lr_fused',),
                lr_fused=SEED_LR):
    adaptive = _Bag(warmup_steps=0, seed_lr=SEED_LR, bounds=(0.01, 2000.0),
                    divergence_loss_abs=1.0e9, divergence_grad_abs=1.0e9,
                    divergence_cut=0.5, envelope_freeze_drop=None,
                    restart_after=None, control_flow_lr=False)
    args = _Bag(adaptive_lr=adaptive, lr_warmup_ratio=1, min_lr=min_lr,
                max_lr=max_lr, lr_policy=SEED_LR, lr_back=SEED_LR,
                lr_replay=SEED_LR, lr_fused=lr_fused, lr_flow=lr_flow,
                lr_servo_managed=list(managed))
    modeller = _Bag(
        args=args, step_ind=0, phase=0, lr_ctrl=None,
        optimizers={'fused': _Bag(param_groups=[{'lr': 0.0}, {'lr': 0.0}]),
                    'fwd': _Bag(param_groups=[{'lr': 0.0}])},
        protocol=_Bag(stage=_Bag(lr_sensor={'kind': 'hyper', 'beta': 0.05})))
    return modeller, LRController(modeller)


def _rates(modeller):
    """(policy-ish groups, flow group) as written onto the optimizers."""
    fused = modeller.optimizers['fused'].param_groups
    return [g['lr'] for g in fused[:-1]], fused[-1]['lr']


def _drive_peak(ctrl, factor):
    """Move peak_scale to `factor` the way the servo would, then re-apply."""
    st = ctrl._state()
    st['peak_scale'] = factor
    ctrl._apply_lrs(st)


# ------------------------------------------------------------- the default

def test_absent_cap_changes_nothing():
    """A config written before this key existed must behave identically."""
    m, c = _controller(max_lr=None)
    _drive_peak(c, 100.0)
    policy, _ = _rates(m)
    assert policy[0] == pytest.approx(SEED_LR * 100.0), \
        'an absent max_lr must not clamp anything'


# --------------------------------------------------------- the cap binding

def test_cap_binds_on_the_product_not_a_factor():
    """peak_scale 100x on a 1.25e-4 base asks for 1.25e-2; the rail refuses."""
    m, c = _controller(max_lr=5.0e-4)
    _drive_peak(c, 100.0)
    policy, _ = _rates(m)
    assert policy[0] == pytest.approx(5.0e-4)


def test_cap_binds_on_the_flow_group():
    """The flow rate has no other guard: no envelope, no peak_scale, and a
    divergence cut cannot move it. Before this cap nothing could lower it."""
    m, c = _controller(max_lr=5.0e-4, lr_flow=0.1)
    c.step()
    _, flow = _rates(m)
    assert flow == pytest.approx(5.0e-4), \
        f'flow group escaped the rail at {flow:g}'


def test_cap_binds_on_an_explicit_float_group():
    """Not servo-managed, so peak_scale never applies -- but the optimizer still
    receives the number, which is what the rail is about."""
    m, c = _controller(max_lr=5.0e-4, managed=(), lr_fused=3.0e-3)
    c.step()
    policy, _ = _rates(m)
    assert policy[0] == pytest.approx(5.0e-4)


def test_cap_does_not_raise_a_rate_below_it():
    """A rail is one-sided. It must never push a low rate up."""
    m, c = _controller(max_lr=5.0e-4)
    _drive_peak(c, 0.1)
    policy, _ = _rates(m)
    assert policy[0] == pytest.approx(SEED_LR * 0.1)


def test_min_lr_still_wins_underneath():
    m, c = _controller(max_lr=5.0e-4, min_lr=1.0e-5)
    _drive_peak(c, 1.0e-4)
    policy, _ = _rates(m)
    assert policy[0] == pytest.approx(1.0e-5)


# ------------------------------------------------- the incoherent pair

def test_cap_below_floor_is_refused_at_construction():
    """Clamp order cannot resolve this: whichever runs second wins and the other
    bound is silently defeated. Refuse instead of picking one."""
    with pytest.raises(ValueError, match='below min_lr'):
        _controller(max_lr=1.0e-7, min_lr=1.0e-6)


def test_nonpositive_cap_is_refused():
    with pytest.raises(ValueError, match='must be positive'):
        _controller(max_lr=0.0)


# ------------------------------------------------------------ visibility

def test_a_binding_rail_is_reported():
    m, c = _controller(max_lr=5.0e-4)
    _drive_peak(c, 100.0)
    c.step()
    report = c.report()
    assert report.get('lr_ctrl/lr_capped_groups', 0) > 0, \
        'the rail bound but nothing said so'


def test_no_cap_configured_publishes_no_metric():
    """Absent, not zero: 'no rail' and 'rail never bound' are different states
    and a constant 0.0 could not tell them apart."""
    m, c = _controller(max_lr=None)
    c.step()
    assert 'lr_ctrl/lr_capped_groups' not in c.report()


# ------------------------------------------------- the floor, symmetrically

def test_a_binding_floor_is_reported():
    """min_lr binding is a CLAMP, not a default, and on the conditional VarGrad
    route it is the more likely of the two rails: measured quality optimum
    ~2e-6 to 2e-5 against a shipped min_lr of 1e-6, so a controller asking to go
    lower is refused with nothing said."""
    m, c = _controller(max_lr=None, min_lr=1.0e-5)
    _drive_peak(c, 1.0e-4)                      # asks for 1.25e-8
    c.step()
    assert c.report().get('lr_ctrl/lr_floored_groups', 0) > 0


def test_no_floor_binding_reports_zero_not_absent():
    """min_lr has no 'off', so unlike the cap there is no absent-means-no-rail
    state to preserve -- it must publish 0.0 rather than vanish."""
    m, c = _controller(max_lr=None, min_lr=1.0e-12)
    _drive_peak(c, 1.0)
    c.step()
    assert c.report().get('lr_ctrl/lr_floored_groups', None) == 0.0
