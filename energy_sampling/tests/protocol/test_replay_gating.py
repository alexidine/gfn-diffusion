"""
CPU tests for the two opt-in gates: replay-buffer management (replay_in_play,
manage_replay_buffer) and the ray probe (_ray_probe_armed, _check_ray_wiring).

WHAT THE CLAIM IS. A VarGrad-only protocol -- var_conditioning all the way down,
no stage training replay TB -- has no consumer for the replay buffer, so it
should never build or churn one, and the ray probe should never arm. Both used to
run anyway: the probe armed by OMISSION (any stage with no lr_sensor block, under
the global ray_calibration.enabled) and churn ran on every fwd step regardless of
what the stage did with it.

WHY THE PIN IS THE WHOLE TEST. The engine's own "does this stage use replay"
predicate is mode_boostable, and on the real var_conditioning stage it answers
TRUE -- Stage.active_modes counts a pinned mode by the PRESENCE of its key, so
`pinned: {replay: 0.0}` reads as boostable. replay_in_play reads the pin by
value. test_pin_is_load_bearing asserts both halves on the same stage, so if that
correction is ever dropped the suite goes red rather than quietly re-enabling
churn on every VarGrad run.

The stage specs are REAL: transcribed from configs/qm9anchor_aug14/base.yaml and
parsed by the real protocol.Stage, not hand-built objects. A fake stage would
test the predicate against my own idea of the config's shape.

Mutation checks (each re-introduces the bug and requires a FAILURE):
  - read the pin by presence instead of value    -> var_conditioning churns again
  - drop the early return from manage_replay_buffer -> the poisoned fwd_stats it
    is handed here get touched, and the call raises
  - arm the probe on a stage with no lr_sensor  -> the old omission default

    python test_replay_gating.py
"""
import os
import sys
from types import MethodType, SimpleNamespace

import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from protocol import Stage, StageProtocol  # noqa: E402
from ray_calibration import RayCalibration  # noqa: E402
from train import Modeller  # noqa: E402

# ---------------------------------------------------------------- real stages
# configs/qm9anchor_aug14/base.yaml, trimmed to the keys these gates read.

VAR_CONDITIONING = {
    'name': 'var_conditioning',
    'train_mode': 'fused',
    'bwd_sampling_mode': 'prior',
    'deactivate_threshold': 0.01,
    'fracs': {'fwd': 0.5, 'bwd': 0.5, 'replay': 0.0},
    'lr_sensor': {'kind': 'hyper', 'beta': 0.05},
    'balance': {
        'kind': 'proportional',
        'alpha': 0.01,
        'drive': 'relative',
        'floor': 0.1,
        'default_boost': {'bwd': 0.5, 'fwd': 0.5},
        'metrics': {'bwd': 'bwd/logw_std_within', 'fwd': 'fwd/logw_std_within'},
        'pinned': {'replay': 0.0},
        'targets': {'bwd': 1.0, 'fwd': 1.0},
    },
}

NAIVE = {
    'name': 'naive',
    'train_mode': 'fused',
    'bwd_sampling_mode': 'prior',
    'deactivate_threshold': 0.01,
    'fracs': {'fwd': 0.4, 'bwd': 0.55, 'replay': 0.05},
    'min_fracs': {'fwd': 0.02, 'bwd': 0.05, 'replay': 0.02},
    'balance': {
        'kind': 'ratio',
        'gain': 0.05,
        'max_step': 0.05,
        'setpoint': 5.0,
        'converge_floor': 1.0,
        'numerator': 'replay',
        'bounds': {'replay': [0.05, 0.6]},
        'metrics': {'bwd': 'bwd/relative_under_wcen', 'replay': 'fwd/over_coverage'},
        'pinned': {'fwd': 0.4},
    },
}

TRAIN_PRIOR = {
    'name': 'train_prior',
    'train_mode': 'bwd',
    'bwd_sampling_mode': 'dataset',
}


def stage(spec, index=0, **patch):
    """Parse a stage spec with the REAL parser, optionally patched."""
    return Stage({**spec, **patch}, index)


def modeller(stg, z_calibration=None, ray_enabled=True, step_ind=0):
    """A stub carrying only what these gates read, with the REAL methods bound.

    protocol.mode_boostable is the real StageProtocol method too -- it is the
    predicate replay_in_play corrects, so faking it would test the correction
    against a copy of the thing being corrected.
    """
    proto = SimpleNamespace(stage=stg, stages=[stg])
    proto.mode_boostable = MethodType(StageProtocol.mode_boostable, proto)
    # ...and the REAL flag reader, for the same reason. Which stages run the Z
    # sidecar is a stage flag: `z_calibration.enabled` was relocated INTO
    # `flags: {z_calibration: true}`, so a stub that answered from the block
    # would be modelling the retired contract, which is the defect that made
    # replay_in_play's z_calibration clause dead in the first place.
    proto.flag = MethodType(StageProtocol.flag, proto)
    m = SimpleNamespace(
        protocol=proto,
        step_ind=step_ind,
        _replay_managed=None,
        # _ray_probe_armed consults the pre-draw refusal predicate (F-039);
        # the shipping predicate currently refuses nothing (freeze-only warmup
        # reversal), so None is the faithful stub
        lr_controller=SimpleNamespace(calibration_refusal=lambda: None),
        args=SimpleNamespace(
            z_calibration=z_calibration,
            ray_calibration=SimpleNamespace(enabled=ray_enabled, period=500,
                                            n_sub=8, alphas=(0.0, 1.0, 2.0)),
        ),
        ray_cal=RayCalibration([torch.zeros(4, requires_grad=True)],
                               alphas=(0.0, 1.0, 2.0), n_sub=2, period=10,
                               enabled=ray_enabled),
    )
    # _probe_refusal is bound too, and its state stubbed: _ray_probe_armed calls
    # it on any stage that DOES declare a ray sensor, so without it this suite
    # could only ever exercise the early-return cases (no sensor / none / hyper)
    # and would raise on the one case the test is named for.
    #
    # `larder=None` is the faithful stub here: these tests are about whether the
    # probe ARMS, and a run with no larder refuses structurally -- which is a
    # verdict, not an error.
    m.larder = None
    m._probe_weights = {}
    m._probe_refusals_seen = set()
    m._probe_exclude_from = None
    for name in ('replay_in_play', 'manage_replay_buffer', '_probe_refusal',
                 '_ray_probe_armed', '_check_ray_wiring', '_ray_askers'):
        setattr(m, name, MethodType(getattr(Modeller, name), m))
    return m


class Poisoned(dict):
    """fwd_stats that cannot be read without exploding. Handed to
    manage_replay_buffer so 'the gate returned early' is proved by the call
    surviving, not by inspecting a buffer that a no-op would also leave empty."""

    def __getitem__(self, key):
        raise AssertionError(f"manage_replay_buffer read fwd_stats[{key!r}] in a stage "
                             f"with no replay consumer -- the early return is gone or "
                             f"has moved below the flow_states transfer")

    def __contains__(self, key):
        return self.__getitem__(key)


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")
    return ok


# ------------------------------------------------------------------- the gates

def test_replay_in_play():
    """The predicate, over the real protocol's stages and the three consumers."""
    print('replay_in_play')
    # The block says HOW the sidecar runs; the stage flag says WHETHER. Both
    # axes are exercised, because the clause under test needs both.
    zc_replay = SimpleNamespace(mode='replay')
    zc_rollout = SimpleNamespace(mode='rollout')
    cases = [
        # (name, modeller, expected)
        ('var_conditioning (replay pinned at 0)',
         modeller(stage(VAR_CONDITIONING), zc_rollout), False),
        ('train_prior (train_mode bwd)',
         modeller(stage(TRAIN_PRIOR), zc_rollout), False),
        ('naive (replay frac 0.05, ratio on replay)',
         modeller(stage(NAIVE), zc_rollout), True),
        ('var_conditioning + lr_sensor ray',
         modeller(stage(VAR_CONDITIONING, lr_sensor={'kind': 'ray'}), zc_rollout), True),
        ('var_conditioning + z_calibration mode replay, stage FLAGGED',
         modeller(stage(VAR_CONDITIONING, flags={'z_calibration': True}),
                  zc_replay), True),
        ('...same block, stage does NOT flag it: off by omission',
         modeller(stage(VAR_CONDITIONING), zc_replay), False),
        ('var_conditioning, no z_calibration block at all',
         modeller(stage(VAR_CONDITIONING), None), False),
        # the parser requires a pin to agree with the stage's entry frac, so the
        # nonzero-pin case moves both (protocol.py::_parse_pinned)
        ('pin read by VALUE: replay pinned at 0.2',
         modeller(stage(VAR_CONDITIONING,
                        fracs={'fwd': 0.4, 'bwd': 0.4, 'replay': 0.2},
                        balance={**VAR_CONDITIONING['balance'],
                                 'pinned': {'replay': 0.2}}), zc_rollout), True),
    ]
    return all(check(name, m.replay_in_play(), want) for name, m, want in cases)


def test_pin_is_load_bearing():
    """MUTATION. Reading the pin by presence -- what Stage.active_modes does --
    must give the WRONG answer on var_conditioning, or this suite is blind."""
    print('pin read by value, not presence (mutation)')
    m = modeller(stage(VAR_CONDITIONING), SimpleNamespace(mode='rollout'))
    ok = check('engine says replay is boostable (the trap)',
               m.protocol.mode_boostable('replay'), True)
    ok &= check('replay_in_play corrects it', m.replay_in_play(), False)
    ok &= check('and the correction is the only difference',
                'replay' in (m.protocol.stage.balance.get('pinned') or {}), True)
    return ok


def test_manage_replay_buffer_returns_first():
    """The gate must fire AHEAD of every read of fwd_stats -- the flow_states
    transfer is the cost, not the bookkeeping."""
    print('manage_replay_buffer early return')
    off = modeller(stage(VAR_CONDITIONING), SimpleNamespace(mode='rollout'))
    ok = True
    try:
        off.manage_replay_buffer(Poisoned(), sample_batch=None)
        ok &= check('no-op stage: fwd_stats never touched', True, True)
    except AssertionError as e:
        print(f'  FAIL  {e}')
        ok = False
    ok &= check('state latched for the transition print', off._replay_managed, False)

    # MUTATION: the same call on a stage that DOES use replay must reach the
    # body. If it does not, the test above proves nothing.
    on = modeller(stage(NAIVE), SimpleNamespace(mode='rollout'))
    try:
        on.manage_replay_buffer(Poisoned(), sample_batch=None)
        print('  FAIL  naive stage: manage_replay_buffer returned early too')
        ok = False
    except AssertionError:
        ok &= check('naive stage: the body runs (mutation check)', True, True)
    return ok


def test_ray_probe_opt_in():
    """The probe arms on an explicit 'ray' and on nothing else."""
    print('_ray_probe_armed')
    ok = True
    for name, spec, patch, want in [
        ('lr_sensor omitted (the retired default)', VAR_CONDITIONING, {'lr_sensor': None}, False),
        ('kind: none', VAR_CONDITIONING, {'lr_sensor': {'kind': 'none'}}, False),
        ('kind: hyper', VAR_CONDITIONING, {}, False),
        ('kind: ray', NAIVE, {'lr_sensor': {'kind': 'ray'}}, True),
    ]:
        # RayCalibration.due latches the period bucket on first sight and fires
        # on the NEXT one, so every case is primed once and read in the bucket
        # after -- otherwise every answer is False and the suite proves nothing
        m = modeller(stage(spec, **patch), ray_enabled=True, step_ind=3)
        m._ray_probe_armed()
        m.step_ind = 15
        ok &= check(name, m._ray_probe_armed(), want)
    # ...and an armed stage must NOT arm inside the bucket it already saw, or
    # 'True' above would just mean 'always'
    m = modeller(stage(NAIVE, lr_sensor={'kind': 'ray'}), ray_enabled=True, step_ind=3)
    m._ray_probe_armed()
    m.step_ind = 5
    ok &= check('kind: ray, same period bucket', m._ray_probe_armed(), False)
    return ok


def test_check_ray_wiring():
    """The two ways a config can disagree with an opt-in probe."""
    print('_check_ray_wiring')
    ok = True
    m = modeller(stage(NAIVE, lr_sensor={'kind': 'ray'}), ray_enabled=False)
    try:
        m._check_ray_wiring()
        print('  FAIL  ray stage + ray_calibration.enabled false did not raise')
        ok = False
    except ValueError as e:
        ok &= check('ray stage with the block disabled raises', 'never arm' in str(e), True)

    m = modeller(stage(VAR_CONDITIONING), ray_enabled=True)
    m._check_ray_wiring()  # warns, must not raise
    ok &= check('enabled with no asker is a warning, not an error', True, True)

    m = modeller(stage(NAIVE, lr_sensor={'kind': 'ray'}), ray_enabled=True)
    m._check_ray_wiring()
    ok &= check('coherent pair is silent', True, True)
    return ok


if __name__ == '__main__':
    results = [t() for t in (test_replay_in_play,
                             test_pin_is_load_bearing,
                             test_manage_replay_buffer_returns_first,
                             test_ray_probe_opt_in,
                             test_check_ray_wiring)]
    print(f"\n{sum(results)}/{len(results)} groups passed")
    sys.exit(0 if all(results) else 1)
