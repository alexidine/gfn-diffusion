"""
A duck-typed stand-in for Modeller, carrying only what the controllers read.

THE DISCIPLINE, and the reason this file exists rather than a reimplementation:
the bench fakes the MODELLER, never the controller. LRController, RayCalibration
and increment_batch_size are imported and run unmodified. If any of them changes
shape, this file raises AttributeError loudly instead of quietly testing a copy
that has drifted.

That failure mode is not hypothetical here. energies/twenty_five_gmm.py still
defines `energy(self, x)` while BaseSet has required
`energy(x, mol_batch, log_temperature, return_exp)` for months -- the toy cannot
run and nothing noticed, because nothing imported it. A bench that re-implements
the control law would rot exactly that way and be worse than no bench, since it
would report green.

WHAT THE CONTROLLERS ACTUALLY READ (the whole coupling surface):

  LRController          args, optimizers, step_ind, phase, lr_ctrl, ray_cal
  RayCalibration        nothing -- it takes params + two callables
  the ray probe SWITCH   protocol.stages[*].lr_sensor -- `enabled` is derived,
                        not configured (train.py:1871, _ray_askers)
  increment_batch_size  args, protocol.stage.{name,train_mode}, step_ind,
                        batch_size, _recent_step_times, _recent_step_work,
                        _rung_throughput, batch_size_* state
  handle_train_...      args, optimizers, step_ind, batch_size, fused_accum_count

No energy function anywhere. That is the fact the whole bench rests on.

Defaults below are lifted from configs/mk_dev.yaml so the bench starts from the
shipping configuration rather than an invented one.
"""

from collections import deque
from types import SimpleNamespace

# Defaults transcribed from configs/mk_dev.yaml. Deliberately a flat literal:
# importing the real YAML would drag in the config loader and its derived-key
# resolution, and the point here is to control every input explicitly.
MK_DEV_LR = dict(
    lr_policy=1.25e-4, lr_back=1.25e-4, lr_replay=1.25e-4, lr_fused=1.25e-4,
    lr_flow=0.1, min_lr=1.0e-6, lr_warmup_ratio=10,
    # what resolve_derived_config records for every lr_* key written `auto`;
    # an empty set is the controller's own control arm (reads and logs, actuates nothing)
    lr_servo_managed=('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'),
)

MK_DEV_ADAPTIVE = dict(
    warmup_steps=1000, seed_lr=1.25e-4, bounds=(0.01, 2000.0),
    divergence_loss_abs=1.0e9, divergence_grad_abs=1.0e9, divergence_cut=0.5,
    control_flow_lr=False, restart_after=None,
)

MK_DEV_CALIBRATION = dict(alpha_target=4.0, eta_up=0.25, eta_down=0.5)

# The block lives UNDER adaptive_lr (utils._RETIRED_KEYS: "moved ->
# adaptive_lr.ray_calibration"), because it parameterises one of the LR sensors.
#
# `enabled` IS NOT HERE, and its absence is the mechanism rather than an
# omission: the flag is deleted in both spellings, and a stage declaring
# `lr_sensor: {kind: ray}` IS the switch (train.py:1871 passes
# `enabled=bool(self._ray_askers())`). The bench derives it the same way, from
# FakeStage.lr_sensor via FakeModeller._ray_askers.
MK_DEV_RAYCAL = dict(period=500, n_sub=8, alphas=(0, 1, 2, 4, 8, 16, 32, 64))

MK_DEV_BATCH = dict(
    batch_size=1000, max_batch_size=1000, grow_batch_size=False,
    batch_growth_factor=1.65, batch_growth_interval=50,
    batch_growth_slow_interval=300, auto_batch_throughput_opt=True,
    batch_growth_min_throughput_gain=0.05, max_step_seconds=60,
    batch_knee_recheck_steps=2000, oom_batch_shrink_factor=0.5,
    oom_cooldown_steps=200, fused_grad_accum_min_samples=1000,
    # how long an OOM ceiling stands before the walk re-probes past it. NOT the same
    # clock as batch_knee_recheck_steps: a knee retest is free, an OOM retest costs a
    # wasted step plus a cooldown, and turning the knee recheck off must not also
    # disable OOM recovery.
    batch_oom_ceiling_retest_steps=1000,
    # occupancy: METRIC WINDOWS ONLY. The controller no longer reads utilization --
    # `gpu_util_floor` is retired (utils._RETIRED_KEYS) because growing the batch
    # does not raise occupancy on the MLIP route. Kept here so the fake's args
    # surface still matches the real one.
    gpu_util_window_s=900, gpu_util_policy_window_s=7200,
    gpu_util_sample_period_s=60,
)


def make_args(**overrides):
    """
    Build an args namespace shaped like the real one, with mk_dev defaults.

    Nested blocks are addressed with dots: make_args(**{'adaptive_lr.warmup_steps': 0}).
    The controller reads them via getattr chains, so SimpleNamespace is faithful.

    THE DOTTED PATH IS THE SHIPPING SPELLING, and unknown keys raise in EVERY
    block, not just the flat one. The nested branches used to write through
    unchecked, which is how `ray_calibration.enabled` -- a key the trainer now
    refuses at load -- went on being set here for free: the override landed in a
    dict nobody compared against the real config, so the bench kept steering a
    flag production had deleted.
    """
    adaptive = dict(MK_DEV_ADAPTIVE)
    calibration = dict(MK_DEV_CALIBRATION)
    raycal = dict(MK_DEV_RAYCAL)
    flat = {**MK_DEV_LR, **MK_DEV_BATCH}

    #: prefix -> the block it addresses. LONGEST FIRST: 'adaptive_lr.' is a
    #: prefix of the other two, so an unordered scan would file
    #: 'adaptive_lr.ray_calibration.period' into the adaptive block under a key
    #: literally named 'ray_calibration.period'. The empty prefix is top level.
    blocks = (('adaptive_lr.calibration.', calibration),
              ('adaptive_lr.ray_calibration.', raycal),
              ('adaptive_lr.', adaptive),
              ('', flat))

    for key, value in overrides.items():
        prefix, block = next((p, b) for p, b in blocks if key.startswith(p))
        name = key[len(prefix):]
        if name not in block:
            raise KeyError(
                f'{key!r} is not a known arg. Add it to the MK_DEV_* dicts if the '
                f'real config has it -- silently accepting unknown keys is how a '
                f'bench ends up testing a config the trainer would reject. Two '
                f'live cases: the ray block MOVED to adaptive_lr.ray_calibration, '
                f'and its `enabled` was DELETED (utils._RETIRED_KEYS) -- the '
                f'switch is now a stage declaring lr_sensor: {{kind: ray}}.')
        block[name] = value

    adaptive['calibration'] = SimpleNamespace(**calibration)
    adaptive['ray_calibration'] = SimpleNamespace(**raycal)
    return SimpleNamespace(adaptive_lr=SimpleNamespace(**adaptive), **flat)


class FakeStage:
    """protocol.stage, as increment_batch_size and the probe gate see it."""

    def __init__(self, name='naive', train_mode='fused', lr_sensor=None, balance=None):
        self.name = name
        self.train_mode = train_mode
        # None = NO LR sensor. The ray probe is opt-in per stage (train.py
        # _ray_probe_armed): omitting the block used to arm it anyway under the
        # global ray_calibration.enabled, and that default is retired.
        #
        # So this field is now LOAD-BEARING rather than decorative: it is the
        # only thing that turns the probe on, here as in production
        # (FakeModeller._ray_askers / train.py::_ray_askers). A ray arm on a
        # stage that does not declare it runs a probe that never arms and posts
        # a row bit-identical to `null` -- the exact silent no-op bench/README
        # records, and what test_arms.py's reading counters exist to catch.
        self.lr_sensor = lr_sensor
        self.balance = balance          # parsed balance dict, as protocol.Stage builds it


class FakeModeller:
    """
    Minimal Modeller surface. The batch-sizer methods are bound from the REAL
    class in attach_real_batch_sizer() rather than defined here.
    """

    def __init__(self, args, optimizers, stage=None, batch_size=None):
        self.args = args
        self.optimizers = optimizers
        # `stages` as well as `stage`, because the probe's switch is derived by
        # scanning the WHOLE protocol for askers (train.py::_ray_askers), not by
        # asking the current stage. One stage, so the bench's list holds exactly
        # the stage it is running.
        stage = stage or FakeStage()
        self.protocol = SimpleNamespace(stage=stage, stages=[stage])
        self.device = 'cpu'
        self.step_ind = 0

        # MODELLER_STATE_DEFAULTS subset the batch sizer owns
        self.batch_size = int(batch_size if batch_size is not None else args.batch_size)
        self.batch_size_last_grow = 0
        self.batch_size_ever_oomed = False
        self.batch_size_cooldown_until = -1
        self.batch_size_oom_ceiling = None
        self.batch_size_oom_ceiling_at = None   # None = unstamped, NOT step 0
        self.batch_size_oom_min = None
        self.batch_size_saturated_stage = None
        self.batch_size_pinned_at = 0
        self._rung_throughput = None
        self._recent_step_times = deque(maxlen=64)
        self._recent_step_work = deque(maxlen=64)
        self.fused_accum_count = 0
        self._z_cal_rollouts = 0

        # GPU-occupancy sensor state. Both the SAMPLER and the READER are the real
        # ones (bound in attach_real_batch_sizer); only the raw NVML read is faked,
        # via _read_gpu_util below. The harness used to append readings itself on a
        # per-10-step cadence, i.e. it reimplemented the sampling policy -- which is
        # exactly how it went on passing while the shipped sampler could not fill its
        # own 900 s window at 200 s/step. Fake the sensor, never the logic above it.
        self._gpu_util = deque(maxlen=4096)
        self._gpu_util_off = False
        self._gpu_util_last_sample = None
        # set by the harness each step: the synthetic GPU and the batch it just ran
        self._bench_gpu = None
        self._bench_attempted = None
        # VIRTUAL CLOCK. A bench run does thousands of steps in about a second, so
        # real timestamps would land every sample inside every window and the
        # time-based averaging would never actually be exercised. The harness
        # advances this by each step's SIMULATED duration.
        self.sim_time = 0.0

        # LRController state
        self.lr_ctrl = {'phase_seen': None, 'scale': None}
        self.ray_cal = None
        self.lr_controller = None

        # bench bookkeeping (never read by the real code)
        self.history = []

    def _ray_askers(self):
        """Stages declaring `lr_sensor: {kind: ray}`. This IS the probe's switch.

        A THREE-LINE COPY of train.Modeller._ray_askers, and copies are the thing
        this file exists to avoid. Binding the real one would drag in the 11 s
        train.py import that every LR-controller and ray test currently avoids,
        so the copy is pinned against the original instead, in
        test_fidelity.py::test_the_fake_derives_the_probe_switch_as_the_trainer_does
        -- which pays that import anyway.
        """
        return [s.name for s in self.protocol.stages
                if s.lr_sensor is not None and s.lr_sensor['kind'] == 'ray']

    def _now(self):
        """Overrides Modeller._now so the windowed sensors run on simulated time."""
        return self.sim_time

    def _read_gpu_util(self):
        """The one faked leaf of the occupancy sensor: NVML does not exist here, so
        the reading comes from the synthetic GPU at the batch that just ran. The
        cadence, the window and the <5-sample guard above it are all real."""
        if self._bench_gpu is None:
            return None
        return float(self._bench_gpu.utilization(self._bench_attempted or self.batch_size))

    @property
    def phase(self):
        return 1

    def lr_of(self, key='fwd', group=0):
        return self.optimizers[key].param_groups[group]['lr']


def attach_real_batch_sizer(cls=FakeModeller):
    """
    Bind train.Modeller's REAL increment_batch_size / handle_train_epoch_error /
    _batch_floor onto the fake class. Python methods are plain functions, so a
    fake `self` works as long as it carries the attributes they read.

    Import cost is ~11 s (train.py pulls wandb, mxtaltools and PyG), so this is
    called lazily -- the LR-controller and ray-probe tests need neither and stay
    at sub-second.
    """
    import train  # noqa: E402  -- deliberately deferred, see docstring

    cls.increment_batch_size = train.Modeller.increment_batch_size
    cls.handle_train_epoch_error = train.Modeller.handle_train_epoch_error
    cls._batch_floor = train.Modeller._batch_floor
    # the occupancy sensor, both halves: the sampling cadence AND the windowed read.
    # Only _read_gpu_util (the NVML leaf) stays faked -- see FakeModeller.
    cls._sample_gpu_util = train.Modeller._sample_gpu_util
    cls._gpu_util_mean = train.Modeller._gpu_util_mean
    return cls
