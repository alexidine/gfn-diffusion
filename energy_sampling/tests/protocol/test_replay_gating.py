"""
CPU tests for the two opt-in gates: replay-buffer management (replay_in_play,
manage_replay_buffer) and the ray probe's arming gate (_ray_probe_armed).

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

EVERY VERDICT REACHES PYTEST BY `assert`. It did not used to: each test ended
`return ok`, pytest discarded the value, and the file passed green for months
while three of its checks reported FAIL. `Checks` below prints every sub-check
(so a partial failure stays readable) and then asserts ONCE naming all of them,
which is the property a bare `assert` per check would give up.

WHAT THIS FILE NO LONGER COVERS, AND WHY. The ray sensor reaches no learning
rate: LRs come from the run-level brute-force bracket (lr_control), and
`configs/mk_dev.yaml` keeps `ray` and `hyper` only as opt-in DIAGNOSTICS, priced
at ~4.8% of step time. So the probe ARMING is not worth a fixture that has to
stock a larder to reach it, and `_check_ray_wiring` was DELETED on 2026-09-10:
it only printed a NOTE once `enabled` became derived from the askers, and
nothing acted on it. `_ray_askers` carries why no check stands in its place.
What survives is the direction that still costs something on a run nobody
intends to probe: a stage that did not ask must build no ray apparatus and must
not arm. The old omission default is exactly that bug, so those cases stay.

Mutation checks (each re-introduces the bug and requires a FAILURE):
  - read the pin by presence instead of value    -> var_conditioning churns again
  - drop the early return from manage_replay_buffer -> the poisoned fwd_stats it
    is handed here get touched, and the call raises
  - arm the probe on a stage with no lr_sensor  -> the old omission default
  - build a larder / enable ray_cal for a stage that never asked -> the probe's
    apparatus is back on every run, whether or not it ever fires

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

from lr_larder import Larder  # noqa: E402
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

#: The probe's clock, small enough to read twice inside one test. Production
#: takes these from the stage's own lr_sensor block or the shared one.
N_SUB, PERIOD = 2, 10


def stage(spec, index=0, **patch):
    """Parse a stage spec with the REAL parser, optionally patched."""
    return Stage({**spec, **patch}, index)


def modeller(stg, z_calibration=None, step_ind=0):
    """A stub carrying only what these gates read, with the REAL methods bound.

    protocol.mode_boostable is the real StageProtocol method too -- it is the
    predicate replay_in_play corrects, so faking it would test the correction
    against a copy of the thing being corrected.

    THE RAY APPARATUS IS DERIVED, NOT PASSED. This used to take `ray_enabled=`,
    from when a `ray_calibration.enabled` config flag governed the probe
    independently of what stages asked for. That flag is gone:
    init_model_and_optimizers builds `RayCalibration(enabled=bool(
    self._ray_askers()))` and keeps a larder on exactly the same condition, so
    `enabled` IS the askers and the two cannot disagree. Deriving it here the
    same way holds the stub to the shipping contract -- passing it separately is
    how this fixture came to model a run that can no longer exist, and to answer
    False for its own reasons on every ray case while the file asked for True.

    THE LARDER IS BUILT BUT NOT STOCKED, deliberately. Whether the probe ARMS is
    no longer tested (see the module docstring: the sensor reaches no learning
    rate), and reaching `arm()` costs a fixture that has to fabricate a harvest.
    What IS tested is that a stage which never asked gets no larder at all --
    for which an empty one on the asking side is the right contrast, and an
    honest one: nothing here scores a record.
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
        args=SimpleNamespace(z_calibration=z_calibration),
    )
    for name in ('replay_in_play', 'manage_replay_buffer', '_probe_refusal',
                 '_ray_probe_armed', '_ray_askers'):
        setattr(m, name, MethodType(getattr(Modeller, name), m))

    asks_ray = bool(m._ray_askers())
    m.ray_cal = RayCalibration([torch.zeros(4, requires_grad=True)],
                               alphas=(0.0, 1.0, 2.0), n_sub=N_SUB,
                               period=PERIOD, enabled=asks_ray)
    m.larder = Larder(depth=8) if asks_ray else None
    m.larder_scorer = None
    # The composite the last optimizer step actually descended, stashed by the
    # step that formed it (fused_train_step, or the single-branch dispatch) and
    # never recomputed. Empty means no step has formed one yet.
    m._probe_weights = {}
    # Lowest step_ind whose batches fed the pending optimizer step; records at
    # or after it are the step's own data and are held out of the ray.
    m._probe_exclude_from = int(step_ind)
    m._probe_refusals_seen = set()
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


#: Read by the conftest hook (pytest_pyfunc_call) under the `_R` spelling it
#: already honours. Belt and braces only: every test below asserts its own
#: verdict, so this matters if a `verdict()` call is ever dropped -- which is
#: precisely the hole this file sat in.
_R = []


class Checks:
    """Run and PRINT every sub-check, then fail once naming all the failures.

    Asserting per check would stop at the first bad case, and a partial failure
    that hides the ones after it is most of what makes a report unreadable."""

    def __init__(self, title):
        print(title)
        self.title = title
        self.n = 0
        self.failed = []

    def __call__(self, name, got, want):
        ok = got == want
        detail = f'got {got!r}, want {want!r}'
        self.n += 1
        _R.append((name, ok, detail))
        if not ok:
            self.failed.append(f'{name}: {detail}')
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
        return ok

    def verdict(self):
        assert not self.failed, (
            f'{self.title}: {len(self.failed)} of {self.n} checks failed:\n  '
            + '\n  '.join(self.failed))


def armed_next_bucket(m):
    """`_ray_probe_armed` in the period bucket AFTER the one it first sees.

    RayCalibration.due latches the bucket on first sight and fires on the next
    one, so reading a stage once would answer False for the clock's reasons
    rather than the gate's. Belt and braces on a non-asker, whose ray_cal is
    disabled and so never comes due at all -- which is the point, and is
    asserted separately below rather than left as the reason this says False."""
    m._ray_probe_armed()
    m.step_ind = PERIOD + 5
    return m._ray_probe_armed()


# ------------------------------------------------------------------- the gates

def test_replay_in_play():
    """The predicate, over the real protocol's stages and its two consumers."""
    c = Checks('replay_in_play')
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
        # RETIRED CONSUMER, kept as a pin in the opposite direction. This case
        # wanted True: the ray probe drew its sub-batches from the replay
        # buffer, so declaring the sensor forced a buffer the stage might never
        # train. It draws from the larder now (_larder_harvest tees whatever
        # branches the stage runs), and replay_in_play's docstring names only
        # two consumers. A stage's SENSOR must not put the buffer back in play,
        # or a diagnostic nobody reads costs per-step churn and a flow_states
        # D2H on every run that declares it.
        ('var_conditioning + lr_sensor ray: NOT a consumer any more',
         modeller(stage(VAR_CONDITIONING, lr_sensor={'kind': 'ray'}), zc_rollout), False),
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
    for name, m, want in cases:
        c(name, m.replay_in_play(), want)
    c.verdict()


def test_pin_is_load_bearing():
    """MUTATION. Reading the pin by presence -- what Stage.active_modes does --
    must give the WRONG answer on var_conditioning, or this suite is blind."""
    c = Checks('pin read by value, not presence (mutation)')
    m = modeller(stage(VAR_CONDITIONING), SimpleNamespace(mode='rollout'))
    c('engine says replay is boostable (the trap)',
      m.protocol.mode_boostable('replay'), True)
    c('replay_in_play corrects it', m.replay_in_play(), False)
    c('and the correction is the only difference',
      'replay' in (m.protocol.stage.balance.get('pinned') or {}), True)
    c.verdict()


def test_manage_replay_buffer_returns_first():
    """The gate must fire AHEAD of every read of fwd_stats -- the flow_states
    transfer is the cost, not the bookkeeping."""
    c = Checks('manage_replay_buffer early return')
    off = modeller(stage(VAR_CONDITIONING), SimpleNamespace(mode='rollout'))
    try:
        off.manage_replay_buffer(Poisoned(), sample_batch=None)
        touched = False
    except AssertionError as e:
        print(f'  (Poisoned fired: {e})')
        touched = True
    c('no-op stage: fwd_stats never touched', touched, False)
    c('state latched for the transition print', off._replay_managed, False)

    # MUTATION: the same call on a stage that DOES use replay must reach the
    # body. If it does not, the check above proves nothing.
    on = modeller(stage(NAIVE), SimpleNamespace(mode='rollout'))
    try:
        on.manage_replay_buffer(Poisoned(), sample_batch=None)
        reached_body = False
    except AssertionError:
        reached_body = True
    c('naive stage: the body runs (mutation check)', reached_body, True)
    c.verdict()


def test_ray_probe_stays_off_unless_asked():
    """A stage that did not declare `lr_sensor: {kind: ray}` must neither arm the
    probe nor carry its apparatus.

    THE DIRECTION THAT STILL COSTS SOMETHING. Arming is not tested (the sensor
    reaches no learning rate; see the module docstring), but the old omission
    default is a live regression risk in exactly this direction: any stage with
    no lr_sensor block used to arm under a global flag, buying a parameter clone
    and a set of forward passes on runs that never asked to be probed.

    THE ASKER CHECKS ARE THE CONTROL, and without them this test would be a
    column of Falses with nothing proving the harness can produce anything else
    -- which is how a suite comes to read as reassurance. They stop deliberately
    short of arming: that a ray stage IS routed to the apparatus is the switch
    working, and it is all this file still claims about the probe."""
    c = Checks('_ray_probe_armed: off unless asked')
    for name, spec, patch in [
        ('lr_sensor omitted (the retired default)', VAR_CONDITIONING, {'lr_sensor': None}),
        ('kind: none', VAR_CONDITIONING, {'lr_sensor': {'kind': 'none'}}),
        ('kind: hyper', VAR_CONDITIONING, {}),
    ]:
        m = modeller(stage(spec, **patch), step_ind=3)
        c(name, armed_next_bucket(m), False)
        # ...and it is off because nothing asked, not because some later gate
        # happened to refuse. A non-asker builds no apparatus at all.
        c(f'  ...{name}: no askers', m._ray_askers(), [])
        c(f'  ...{name}: ray_cal disabled', m.ray_cal.enabled, False)
        c(f'  ...{name}: no larder kept', m.larder, None)

    # THE CONTROL. A stage that DOES ask is routed to the apparatus -- so the
    # Falses above are the gate answering, not the fixture being incapable.
    asks = modeller(stage(NAIVE, lr_sensor={'kind': 'ray'}), step_ind=3)
    c('kind: ray IS an asker', asks._ray_askers(), ['naive'])
    c('...so ray_cal.enabled is derived true, never configured against it',
      asks.ray_cal.enabled, True)
    c('...and a larder is kept for it', isinstance(asks.larder, Larder), True)
    c.verdict()


if __name__ == '__main__':
    tests = (test_replay_in_play,
             test_pin_is_load_bearing,
             test_manage_replay_buffer_returns_first,
             test_ray_probe_stays_off_unless_asked)
    # Each test asserts its own verdict now, so script mode catches the
    # AssertionError per group rather than letting the first failure abort the
    # run: printing every group is the reason the file reports as it goes.
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed.append(t.__name__)
            print(f'  -> {e}')
    print(f"\n{len(tests) - len(failed)}/{len(tests)} groups passed")
    sys.exit(1 if failed else 0)
