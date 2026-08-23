"""
A duck-typed stand-in for Modeller, carrying only what the controllers read.

THE DISCIPLINE, and the reason this file exists rather than a reimplementation:
the bench fakes the MODELLER, never the controller. LRController, RayCalibration
and select_batch_size are imported and run unmodified. If any of them changes
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
                        not configured (train.py, _ray_askers)
  the ray probe GATE    larder, larder_scorer, _probe_weights,
                        _probe_exclude_from, _probe_refusals_seen. The sensor
                        scores the fused composite on harvested batches now, so
                        the gate asks whether every ACTIVE branch can be
                        replay-scored and whether the larder holds n_sub records
                        of each that this step did not train on
  select_batch_size     args, protocol.stage.{name,train_mode}, step_ind,
                        batch_size, _recent_step_times, _recent_step_work,
                        batch_sizer, batch_size_* state, _gpu_util, _now
  handle_train_...      args, optimizers, step_ind, batch_size, fused_accum_count

No energy function anywhere. That is the fact the whole bench rests on.

Defaults below are lifted from configs/mk_dev.yaml so the bench starts from the
shipping configuration rather than an invented one.
"""

from collections import deque
from types import SimpleNamespace

import torch

from energy_sampling.lr_larder import Larder, LarderScorer
from energy_sampling.utils import MetricTracker

# Defaults transcribed from configs/mk_dev.yaml. Deliberately a flat literal:
# importing the real YAML would drag in the config loader and its derived-key
# resolution, and the point here is to control every input explicitly.
MK_DEV_LR = dict(
    lr_policy=1.25e-4, lr_back=1.25e-4, lr_replay=1.25e-4, lr_fused=1.25e-4,
    lr_flow=0.1, min_lr=1.0e-8, lr_warmup_ratio=10,
    # what resolve_derived_config records for every lr_* key written `auto`;
    # an empty set is the controller's own control arm (reads and logs, actuates nothing)
    lr_servo_managed=('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'),
)

MK_DEV_ADAPTIVE = dict(
    warmup_steps=1000, seed_lr=1.25e-4, bounds=(0.01, 2000.0),
    divergence_loss_abs=1.0e9, divergence_grad_abs=1.0e9, divergence_cut=0.5,
    # relative bar, as a multiple of the QUIETEST loss the current stage has
    # produced on that branch -- the absolute bars above are a numerical-death
    # backstop and cannot see a 100x excursion on an O(1) loss. Transcribed from
    # the shipping value rather than disabled here: a bench that starts from a
    # different configuration than production is the drift test_fidelity exists
    # to catch.
    divergence_loss_rel=100.0,
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

# The three coefficient banks, transcribed from configs/mk_dev.yaml. Only the
# keys the probe gate reads are here: LarderScorer.refusal asks whether the FWD
# bank carries a term the backward (replay) evaluator has no counterpart for,
# and all four are 0 on the canonical route. `var_conditioning` runs emp_z 1.0,
# which is the case the refusal exists for -- flip it in a test, do not ship it
# as a default here.
MK_DEV_BANKS = dict(
    fwd_loss_coeffs=dict(tb=1.0, z_level=0.0, emp_z=0.0, emp_z_persistent=0.0,
                         traj_grads=0.0, reward_grads=0.0),
    bwd_loss_coeffs=dict(tb=1.0, traj_grads=1.0),
    replay_loss_coeffs=dict(tb=1.0, traj_grads=0.0),
)

MK_DEV_BATCH = dict(
    batch_size=1000, max_batch_size=20000, grow_batch_size=True,
    batch_growth_factor=1.6, batch_growth_interval=50,
    # rung-step cap: geometric growth until the increment reaches this, linear
    # after -- bounds the selection's absolute overshoot at one accum quantum.
    # 0 = uncapped geometric.
    batch_growth_cap=1000,
    max_step_seconds=60, oom_batch_shrink_factor=0.625,
    oom_cooldown_steps=200, fused_grad_accum_min_samples=1000,
    # the occupancy target, a FRACTION of the card since state 9 (out-of-process
    # bracket: cancelled <=0.40, survived >=0.494). The canonical config ships
    # the ladder ARMED at 0.6 as of 2026-08-19; 0 still means off -- hold the
    # base batch (S3). train.select_batch_size does the one percent conversion.
    batch_util_target=0.6,
    # how long an OOM ceiling stands before the ladder re-probes past it. An OOM
    # retest is not free -- a failed re-probe costs a wasted step plus a cooldown.
    batch_oom_ceiling_retest_steps=1000,
    # occupancy windows. The sizer reads RAW samples per calibration rung and the
    # policy-window mean once for its S2 audit; the `gpu_util_floor` actuator
    # (grow on a low windowed mean) stays retired (utils._RETIRED_KEYS).
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
    banks = {k: dict(v) for k, v in MK_DEV_BANKS.items()}
    flat = {**MK_DEV_LR, **MK_DEV_BATCH}

    #: prefix -> the block it addresses. LONGEST FIRST: 'adaptive_lr.' is a
    #: prefix of the other two, so an unordered scan would file
    #: 'adaptive_lr.ray_calibration.period' into the adaptive block under a key
    #: literally named 'ray_calibration.period'. The empty prefix is top level.
    blocks = (('adaptive_lr.calibration.', calibration),
              ('adaptive_lr.ray_calibration.', raycal),
              ('adaptive_lr.', adaptive),
              ('fwd_loss_coeffs.', banks['fwd_loss_coeffs']),
              ('bwd_loss_coeffs.', banks['bwd_loss_coeffs']),
              ('replay_loss_coeffs.', banks['replay_loss_coeffs']),
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
    flat.update({k: SimpleNamespace(**v) for k, v in banks.items()})
    return SimpleNamespace(adaptive_lr=SimpleNamespace(**adaptive), **flat)


class FakeStage:
    """protocol.stage, as select_batch_size and the probe gate see it."""

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
        self.batch_size_cooldown_until = -1
        self.batch_size_oom_ceiling = None
        self.batch_size_oom_ceiling_at = None   # None = unstamped, NOT step 0
        self.batch_size_oom_min = None
        self.batch_sizer = None                 # the sizer's per-stage conclusion
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

        # Ray-probe gate state. REAL objects, not stubs: Larder is pure Python
        # and LarderScorer's refusal path reads only the coefficient banks, so
        # both run unmodified here and a change to either shape fails loudly.
        # `_probe_weights` is the composite the last step descended -- one
        # branch at weight 1 unless a test says otherwise.
        self.larder = Larder(depth=32)
        self.larder_scorer = LarderScorer(self, verbose=False)
        # NB 'fused' is a TRAIN MODE, never a branch: a fused step's composite
        # is the three real branches at the weights it formed (mk_dev's
        # equilibration fracs here), and a bwd stage is that one branch at 1.
        self._probe_weights = ({'fwd': 0.05, 'bwd': 0.45, 'replay': 0.5}
                               if stage.train_mode == 'fused'
                               else {stage.train_mode: 1.0})
        self._probe_exclude_from = 0
        self._probe_refusals_seen = set()

        # Ramp-driver surface (lr_ramp_probe.RampDriver): a real model to
        # snapshot and a real MetricTracker to read coherence from. Tiny, but
        # REAL -- the driver's contract is that a rollback is bitwise, and a
        # stub with no parameters could not test that at all.
        #
        # BUILT INSIDE fork_rng. `nn.Linear` initialises its weights from the
        # GLOBAL torch stream, so constructing a fake would shift the RNG for
        # everything built after it -- and this class is instantiated by test
        # modules that have nothing to do with the ramp. A harness that perturbs
        # the runs it measures is the exact defect this bench exists to catch,
        # one level up (findings F-039).
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.gfn_model = torch.nn.Linear(4, 3)
        self.metric_tracker = MetricTracker(period=100)
        self._hyper_prev_step = None

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
    Bind train.Modeller's REAL select_batch_size / handle_train_epoch_error /
    _batch_floor onto the fake class. Python methods are plain functions, so a
    fake `self` works as long as it carries the attributes they read.

    Import cost is ~11 s (train.py pulls wandb, mxtaltools and PyG), so this is
    called lazily -- the LR-controller and ray-probe tests need neither and stay
    at sub-second.
    """
    import train  # noqa: E402  -- deliberately deferred, see docstring

    cls.select_batch_size = train.Modeller.select_batch_size
    cls._conclude_batch_calibration = train.Modeller._conclude_batch_calibration
    cls.handle_train_epoch_error = train.Modeller.handle_train_epoch_error
    cls._batch_floor = train.Modeller._batch_floor
    # handle_train_epoch_error's eval branch sizes the EVAL draw, which shrinks
    # independently of the train batch. Bound REAL rather than stubbed: a stub
    # here would let the handler's eval path diverge from the loops it feeds,
    # which is precisely the failure these tests exist to catch.
    cls.eval_draw_size = train.Modeller.eval_draw_size
    cls.reset_eval_draw_size = train.Modeller.reset_eval_draw_size
    # the occupancy sensor, both halves: the sampling cadence AND the windowed read.
    # Only _read_gpu_util (the NVML leaf) stays faked -- see FakeModeller.
    cls._sample_gpu_util = train.Modeller._sample_gpu_util
    cls._gpu_util_mean = train.Modeller._gpu_util_mean
    return cls


def attach_real_probe_gate(cls=FakeModeller):
    """Bind train.Modeller's REAL ray-probe gate onto the fake class.

    Same discipline and the same ~11 s import cost as attach_real_batch_sizer:
    the gate decides refuse-vs-defer-vs-arm, and a bench copy of that decision
    would be the one thing capable of drifting from the trainer while reporting
    green.
    """
    import train  # noqa: E402  -- deliberately deferred, see docstring

    cls._ray_probe_armed = train.Modeller._ray_probe_armed
    cls._probe_refusal = train.Modeller._probe_refusal
    return cls
