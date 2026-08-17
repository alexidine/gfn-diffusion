"""
Does the fake modeller still stand in for the real one?

This is the test that keeps the whole bench honest. Everything else here asserts
things about controllers driven by `FakeModeller`; those assertions are only
worth anything while `FakeModeller` still presents the surface the real
`Modeller` presents. That claim dies silently -- someone adds an attribute a
controller reads, the fake does not grow it, and the bench keeps reporting green
about a shape that no longer exists.

So: build the REAL `train.Modeller` from the REAL `configs/mk_dev.yaml`, and
compare.

These tests need `train.py` (~11 s) and a config on disk. They do NOT need a
GPU -- which is the point of the guard at `train.py:130`.
"""

import pytest
import torch

from bench.fake_modeller import (MK_DEV_ADAPTIVE, MK_DEV_BATCH, MK_DEV_CALIBRATION,
                                 MK_DEV_LR, MK_DEV_RAYCAL, FakeModeller, FakeStage,
                                 make_args)
from bench.real_modeller import build_real_modeller
from energy_sampling.controller import LRController


@pytest.fixture(scope='module')
def real():
    return build_real_modeller()


#: distinguishes "the key is absent" from "the key is None", which a config
#: legitimately is (adaptive_lr.restart_after).
_MISSING = object()


def _dig(node, dotted):
    """Walk a dotted path, `_MISSING` if any hop is absent."""
    for part in dotted.split('.'):
        node = getattr(node, part, _MISSING)
        if node is _MISSING:
            return _MISSING
    return node


# ------------------------------------------------------- the coupling surface

#: Every attribute the controllers read off the modeller, that exists after a
#: bare __init__. Transcribed from controller.py, ray_calibration.py and
#: train.py's increment_batch_size / handle_train_epoch_error. If a controller
#: starts reading something new, add it here -- that is the maintenance contract.
COUPLING_SURFACE = [
    'args', 'step_ind', 'phase', 'lr_ctrl',
    'batch_size', 'batch_size_last_grow', 'batch_size_ever_oomed',
    'batch_size_cooldown_until', 'batch_size_oom_ceiling', 'protocol',
]

#: Read by the controllers but NOT created by __init__ -- init_gfn builds them,
#: and that needs the model, the energy function and the datasets. So these
#: cannot be checked against a bare Modeller, and the bench supplies its own.
#: `optimizers` is the important one: train.py:1619-1664 builds all five keys
#: unconditionally, which is why LRController.step()'s bare optimizers['fwd']
#: lookup is safe in production.
DEFERRED_SURFACE = ['optimizers', 'fused_accum_count']

#: Config values that are collections without meaningful order.
UNORDERED_KEYS = {'lr_servo_managed'}

#: args keys the controllers read. DOTTED WHERE THE REAL CONFIG NESTS THEM: the
#: ray block moved under `adaptive_lr` (utils._RETIRED_KEYS, "moved ->
#: adaptive_lr.ray_calibration"), and a top-level-only check cannot see that move
#: -- `adaptive_lr` is present either way, so the block could go missing in
#: silence.
ARGS_SURFACE = [
    'lr_policy', 'lr_back', 'lr_replay', 'lr_fused', 'lr_flow', 'min_lr',
    'lr_warmup_ratio', 'lr_servo_managed', 'adaptive_lr',
    'adaptive_lr.ray_calibration',
    'batch_size', 'max_batch_size', 'grow_batch_size', 'batch_growth_factor',
    'batch_growth_interval', 'batch_growth_slow_interval',
    'auto_batch_throughput_opt', 'batch_growth_min_throughput_gain',
    'max_step_seconds', 'batch_knee_recheck_steps', 'oom_batch_shrink_factor',
    'oom_cooldown_steps', 'fused_grad_accum_min_samples',
    # occupancy metric windows. NOT a control input -- gpu_util_floor is retired;
    # the controller reads no utilization at all.
    'gpu_util_window_s', 'gpu_util_policy_window_s', 'gpu_util_sample_period_s',
]


def test_real_modeller_builds_on_cpu(real):
    """The guard at train.py:130 is what makes every test in this file possible."""
    assert real.step_ind == 0
    assert real.protocol.stage is not None
    assert isinstance(real.lr_controller, LRController)


def test_fake_carries_the_whole_coupling_surface(real):
    missing = [a for a in COUPLING_SURFACE if not hasattr(real, a)]
    assert not missing, (
        f'{missing} is read by a controller but absent from the REAL Modeller -- '
        f'COUPLING_SURFACE is stale')

    fake = FakeModeller(make_args(), {}, )
    missing = [a for a in COUPLING_SURFACE if not hasattr(fake, a)]
    assert not missing, (
        f'{missing} exists on the real Modeller but not on FakeModeller. The fake '
        f'has stopped standing in; add it to bench/fake_modeller.py.')


def test_fake_args_carry_every_key_the_controllers_read(real):
    missing = [k for k in ARGS_SURFACE if _dig(real.args, k) is _MISSING]
    assert not missing, f'{missing} missing from the REAL resolved config'

    fake_args = make_args()
    missing = [k for k in ARGS_SURFACE if _dig(fake_args, k) is _MISSING]
    assert not missing, (
        f'{missing} present in the real config but not in make_args(). '
        f'Add it to the MK_DEV_* dicts in bench/fake_modeller.py.')


def test_deferred_surface_is_absent_after_init(real):
    """
    Pins the split. These are read by controllers but built by init_gfn, so a
    bare Modeller does not have them -- and if that ever changes, the fake should
    stop inventing them and take the real ones instead.
    """
    for attr in DEFERRED_SURFACE:
        assert not hasattr(real, attr), (
            f'{attr!r} now exists after __init__; move it into COUPLING_SURFACE '
            f'and check the fake against the real value')


def test_fake_supplies_the_deferred_surface(real):
    """The bench has to provide what init_gfn would have, with the same shape."""
    fake = FakeModeller(make_args(), {})
    for attr in DEFERRED_SURFACE:
        assert hasattr(fake, attr), attr


# ------------------------------------------------------------- value fidelity

@pytest.mark.parametrize('block,source,path', [
    ('lr', MK_DEV_LR, None),
    ('batch', MK_DEV_BATCH, None),
    ('adaptive_lr', MK_DEV_ADAPTIVE, 'adaptive_lr'),
    ('calibration', MK_DEV_CALIBRATION, 'adaptive_lr.calibration'),
    ('ray_calibration', MK_DEV_RAYCAL, 'adaptive_lr.ray_calibration'),
])
def test_transcribed_defaults_match_the_shipping_config(real, block, source, path):
    """
    The bench claims to start from mk_dev's actual values. When mk_dev is retuned
    that claim goes stale, and a stale claim is worse than no claim -- every
    result would silently describe a configuration nobody runs.

    If this fires: update the MK_DEV_* dict in bench/fake_modeller.py to the new
    value, then re-read any finding whose conclusion depended on it.
    """
    node = real.args
    for part in (path.split('.') if path else []):
        node = getattr(node, part)

    drift = {}
    for key, mine in source.items():
        if key == 'calibration' or not hasattr(node, key):
            continue
        theirs = getattr(node, key)
        if key in UNORDERED_KEYS:
            # `_managed_keys()` wraps this in set(), so ordering carries no
            # meaning and comparing it as a sequence reports phantom drift.
            same = set(theirs or ()) == set(mine or ())
        elif isinstance(theirs, (list, tuple)) or isinstance(mine, (list, tuple)):
            # `alphas` IS order-bearing -- it must be closed under doubling, and
            # RayCalibration indexes it positionally for the report keys.
            same = tuple(theirs or ()) == tuple(mine or ())
        elif isinstance(theirs, float) or isinstance(mine, float):
            same = theirs == pytest.approx(mine, rel=1e-9)
        else:
            same = theirs == mine
        if not same:
            drift[key] = (mine, theirs)

    assert not drift, (
        f'{block}: bench default vs configs/mk_dev.yaml -- '
        + '; '.join(f'{k}: bench {m!r} vs config {t!r}' for k, (m, t) in drift.items()))


def test_the_retired_ray_keys_are_gone_from_the_bench_too(real):
    """
    THE BLIND SPOT OF THE TEST ABOVE. Value fidelity skips any bench key the real
    config lacks (`not hasattr(node, key)`), so a RETIRED key living on in a
    MK_DEV_* dict is invisible to it -- which is how `ray_calibration.enabled`
    stayed in this bench after the trainer deleted it (utils._RETIRED_KEYS
    retires both the top-level block and `enabled` in either spelling).

    Both directions, because both were wrong: the block MOVED under adaptive_lr,
    and its flag was DELETED in favour of deriving the switch from the stages
    that declare `lr_sensor: {kind: ray}`.
    """
    assert _dig(real.args, 'ray_calibration') is _MISSING, (
        'a top-level ray_calibration block survived the load -- it is retired '
        '(moved under adaptive_lr), so the preflight should have refused it')
    assert _dig(real.args, 'adaptive_lr.ray_calibration.enabled') is _MISSING, (
        '`enabled` is deleted: the switch is which stages declare '
        'lr_sensor: {kind: ray}, and a second flag could disagree with them')
    assert 'enabled' not in MK_DEV_RAYCAL, (
        'the bench is still transcribing a key the trainer refuses at load')

    # ...and the retired spellings must FAIL here rather than being carried as
    # private bench keys, which is what the unchecked nested branch allowed.
    for retired in ('ray_calibration.enabled', 'ray_calibration.period',
                    'adaptive_lr.ray_calibration.enabled'):
        with pytest.raises(KeyError):
            make_args(**{retired: False})


def test_the_fake_derives_the_probe_switch_as_the_trainer_does():
    """
    `enabled` is DERIVED, so the derivation is now load-bearing code -- and the
    fake owns a copy of it (`FakeModeller._ray_askers`) to stay off train.py's
    11 s import. Run the REAL function against the fake's protocol and require
    the same answer, so the copy cannot drift.

    Getting this wrong is silent in the dangerous direction: a ray arm whose
    switch reads False runs a probe that never arms, scores bit-identical to
    `null`, and still posts a row. That has fired twice here.
    """
    import train  # noqa: E402 -- deferred, ~11 s; this module pays it anyway

    cases = [(None, False),
             ({'kind': 'ray'}, True),
             ({'kind': 'hyper', 'beta': 0.1}, False),
             ({'kind': 'plateau'}, False),
             ({'kind': 'none'}, False)]
    for sensor, asks in cases:
        fake = FakeModeller(make_args(), {},
                            stage=FakeStage(name='s', lr_sensor=sensor))
        mine, theirs = fake._ray_askers(), train.Modeller._ray_askers(fake)
        assert mine == theirs, (
            f'lr_sensor={sensor!r}: the bench derives {mine} where the trainer '
            f'derives {theirs} -- FakeModeller._ray_askers has drifted')
        assert bool(mine) is asks, sensor


def test_controller_produces_the_same_lr_on_both(real):
    """
    End-to-end equivalence: given the same args and the same step, the real and
    fake modellers must come out at the same learning rate. This is what makes a
    bench measurement a statement about the shipping controller.
    """
    def optimizer_set(a):
        p = torch.nn.Parameter(torch.zeros(4))
        q = torch.nn.Parameter(torch.zeros(4))
        return {
            'fwd': torch.optim.SGD([p], lr=a.lr_policy),
            'bwd': torch.optim.SGD([p], lr=a.lr_back),
            'replay': torch.optim.SGD([p], lr=a.lr_replay),
            'fused': torch.optim.SGD([{'params': [p]}, {'params': [q]}], lr=a.lr_fused),
            'flow': torch.optim.SGD([q], lr=a.lr_flow),
        }

    fake = FakeModeller(real.args, optimizer_set(real.args))
    fake.ray_cal = None
    fake.lr_controller = LRController(fake)

    # SEPARATE optimizer objects, or the two controllers would be writing into
    # the same param_groups and the comparison would be vacuous. init_gfn is
    # what normally builds these; the shapes match (train.py:1619-1664).
    real.optimizers = optimizer_set(real.args)

    for step in (0, 1, 250, 999, 1000, 5000):
        real.step_ind = fake.step_ind = step
        real.lr_ctrl = {'phase_seen': None, 'scale': None}
        fake.lr_ctrl = {'phase_seen': None, 'scale': None}
        real.lr_controller.step()
        fake.lr_controller.step()
        for key in ('fwd', 'bwd', 'replay', 'flow'):
            assert fake.lr_of(key) == pytest.approx(
                real.optimizers[key].param_groups[0]['lr'], rel=1e-12), (step, key)
        assert fake.optimizers['fused'].param_groups[-1]['lr'] == pytest.approx(
            real.optimizers['fused'].param_groups[-1]['lr'], rel=1e-12), step
