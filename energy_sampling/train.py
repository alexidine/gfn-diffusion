import copy
import gc
import math
import os
from collections import defaultdict, deque
from copy import deepcopy
from typing import Optional

# ---------------------------------------------------------------------------
# CUDA ALLOCATOR CONFIG. Two things about these lines are load-bearing.
#
# ORDER: this must precede every import that pulls in torch, and the eval
# imports below do (eval/evaluations.py imports torch). It used to sit AFTER
# them, leaving the setting's effect dependent on when torch happens to
# initialise its allocator -- a silent, version-dependent coin flip on the one
# knob that governs fragmentation.
#
# setdefault, NOT assignment: it was `os.environ[...] = ...`, which CLOBBERS
# whatever the operator exported before launch, so a sbatch setting a different
# allocator policy was silently overridden by the process it was configuring.
# As a default, the environment wins and the code supplies a floor.
#
# WHY THE FLOOR IS expandable_segments: reserved memory otherwise runs far ahead
# of live -- measured on a MACE arm, 890 MiB live against 57.7 GB reserved, 98.5%
# of it cached-but-held -- and cuda_memory_fraction is a HARD cap that counts
# those unusable blocks, so an allocation can fail with the card nearly empty.
# `garbage_collection_threshold` and `max_split_size_mb` target the same problem
# and are deliberately NOT defaulted: they change allocation for every route and
# have not been measured. Set them per run through the environment, which works
# now that this is a setdefault.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

from energy_sampling.eval.evaluations import to_loggable, sliced_wasserstein, adjust_fig_filesize, eval_figs, \
    log_ess_frac, condition_tracker_figs, fig_guard
from energy_sampling.eval.traj_reporting import traj_overlap_report, to_scalars


from time import time

import numpy as np
import torch

import profiling
import wandb
from tqdm import trange

from energies.molecular_crystal import MolecularCrystal
from energy_sampling.buffer import CrystalBuffer, AnchorBuffer, ConditionLogZTracker, _per_condition_min, \
    _per_condition_max, strip_lazy_sg_caches, DEFAULT_HALF_LIFE_VISITS
from energy_sampling.checkpointing import Checkpointer, MODELLER_STATE_DEFAULTS
from energy_sampling.controller import LRController
from energy_sampling.grad_clip_guard import GradClipGuard
from energy_sampling.protocol import StageProtocol, TRAIN_MODES
from energy_sampling.ray_calibration import RayCalibration
from energy_sampling.eval.utils import sample_eval_fwd_trajs
from energy_sampling.utils import is_cuda_oom, \
    dict2namespace, \
    get_discretizer, drain_elapsed_times, MetricTracker, quick_tb_stats, uniform_discretizer, logmeanexp, \
    cal_subtb_coef_matrix, per_condition_fraction
from gflownet_losses import get_gfn_forward_loss, get_gfn_backward_loss, log_pf_estimate
from models import GFN
from energy_sampling.models.aunit_periodicity import sg_periodic_centroid_axes, describe
from energy_sampling.models.dead_latent_rows import (
    resolve_dead_rows, verify_dead_rows, describe as describe_dead_rows)
from mxtaltools.common.training_utils import flatten_wandb_params
from mxtaltools.dataset_utils.utils import collate_data_list
from utils import get_train_args, get_gfn_init_state, set_seed, \
    update_ema, get_problem_definition, problem_hash, problem_slug


# bulky per-sample analysis artifacts (fingerprints, RDFs -- huge tensors) that
# can ride in on loaded datasets or analyzed batches; never read off any buffer
# draw, so stripped from EVERY buffer's storage just in case they are present
BULKY_ATTR_EXCLUDE_KEYS = ('fingerprint', 'rdf')

# --- batch sizer (select_batch_size) -----------------------------------------
#: Fewer raw occupancy samples than this is a coin flip, not a rung reading. The
#: same domain boundary _gpu_util_mean draws at 5 for its windowed mean; smaller
#: here because a calibration rung reads a dedicated dwell rather than a trailing
#: window that may straddle rungs.
_BS_MIN_UTIL_SAMPLES = 3
#: Dwell multiplier after which a rung still short of _BS_MIN_UTIL_SAMPLES is
#: declared starved and the walk concludes -- a sensor that answers nothing
#: removes nothing and grows nothing (S3), and waiting forever on it would leave
#: the batch parked mid-ladder with no conclusion recorded.
_BS_RUNG_TIMEOUT_INTERVALS = 20
#: Series encodings for the sizer's state, so wandb carries the account of what
#: it concluded (prints do not survive a hard-killed run).
_BS_PHASE_CODES = {None: 0, 'calibrating': 1, 'hold': 2}
_BS_REASON_CODES = {None: 0, 'no_target': 1, 'target_met': 2, 'infeasible': 3,
                    'sensor_off': 4, 'no_headroom': 5, 'stood_down': 6,
                    'wallclock_cut': 7}


# stripped from churned-buffer STORAGE at admission (draws already drop them):
# string/list attrs are never read off a buffer draw, and python-list keys make
# every subsample pay a per-element copy plus -- on GPU-resident buffers -- an
# idx.tolist() device sync. mol_dataset/prior_dataset keep them (init_identifiers
# reads .identifier); the anchor buffer keeps them (eval-cadence only).
CHURNED_BUFFER_EXCLUDE_KEYS = ('symmetry_operators', 'smiles', 'identifier') + BULKY_ATTR_EXCLUDE_KEYS


class FrozenTrainingState(RuntimeError):
    """Raised when a run is unrecoverable and should release the GPU. Emitting
    identical numbers, so its remaining wall clock is waste. Named distinctly
    so a log sweep can tell it apart from a KeyboardInterrupt (a manual kill,
    not a crash -- the replay_july26 postmortem lesson) and from an ordinary
    exception."""


def safe_histogram(data, num_bins=32):
    """
    wandb.Histogram that tolerates degenerate input. numpy's histogram raises
    "Too many bins for data range" not only on constant data but on any tiny
    float32 range -- the reported crash was a log-temperature array with amax
    3.1659272 / amin 3.165927 (range ~2e-7), which overflows float32 bin-edge
    math even though it is nominally non-constant. Casting to float64 gives
    enough headroom to bin such data; empty/all-nonfinite and exactly-constant
    inputs (e.g. a Jensen gap that is all-zeros before conditions diverge) get an
    explicit single-bin histogram; and a try/except catches any residual
    degenerate case so logging never crashes.
    """
    a = np.asarray(data, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return wandb.Histogram(np_histogram=([0], [0.0, 1.0]))
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        try:
            return wandb.Histogram(a, num_bins=num_bins)
        except ValueError:
            pass
    return wandb.Histogram(np_histogram=([int(a.size)], [lo - 0.5, lo + 0.5]))


def loggable_array(a, num_bins=32):
    """
    Prep an array-valued metric for wandb.log. wandb auto-compresses any raw
    array (>32 elements) into a 32-bin histogram via the same fragile
    numpy.histogram path safe_histogram guards -- so build the histogram
    ourselves instead of handing wandb a raw array. Size<=1 (incl. 0-d) reduces
    to a plain scalar.
    """
    a = np.nan_to_num(np.asarray(a, dtype=np.float64)).ravel()
    if a.size <= 1:
        return float(a[0]) if a.size else 0.0
    return safe_histogram(a, num_bins=num_bins)


def _uniform_draw(n: int, k: int) -> torch.Tensor:
    """
    Draw up to k positions from range(n), uniformly, without replacement.
    NOT the caller's original indices -- callers index their own index
    tensor by the result (e.g. `elig[_uniform_draw(elig.numel(), k)]`).

    Used for replay-buffer admission/purge (see manage_replay_buffer):
    selection lives at the draw (prioritise/kappa), not at intake, so both
    sides of buffer churn are residual-independent by construction.
    """
    k = min(k, n)
    if k <= 0:
        return torch.zeros(0, dtype=torch.long)
    return torch.randperm(n)[:k]


class Modeller:
    def __init__(self, args=None):
        self.step_ind = None
        # Guaranteed to exist before any path can reach the training loop, so
        # the per-step call needs no None check and no getattr. Replaced by the
        # configured window in init_energy_function; a disabled one here means a
        # trainer built without that step still runs rather than raising.
        self._trace_window = profiling.TraceWindow(enabled=False)
        self.args = get_train_args() if args is None else args
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(self.args.cuda_memory_fraction, device=0)
            torch.cuda.init()  # create context with the cap already in place
        else:
            # CPU-only construction has to be possible: without this guard both
            # calls raise, so Modeller could not be built at all on a machine
            # with no visible GPU -- a CI box, a login node, or any local probe
            # run under CUDA_VISIBLE_DEVICES="". The memory cap is meaningless
            # there; nothing else in __init__ needs a device.
            print("cuda unavailable -- skipping memory-fraction cap and context init")

        set_seed(self.args.seed)
        if 'SLURM_PROCID' in os.environ:
            self.args.seed += int(os.environ["SLURM_PROCID"])

        config = self.args.__dict__
        config["Experiment"] = "{args.energy}"
        self.run_name = str(self.args.tag) + '_' + str(self.args.run_name)

        # fingerprint of the energy function + prior this run is training against,
        # as opposed to training hyperparameters - see checkpointer.save / checkpointer.path_for
        self.problem_def = get_problem_definition(self.args)
        self.problem_hash = problem_hash(self.problem_def)
        self.problem_slug = problem_slug(self.args, self.problem_def)

        self.times = {}
        # replay churn tallied across every manage_replay_buffer call (train
        # steps and eval alike), drained and logged once per eval in log_metrics
        self.replay_churn = {'admitted': 0, 'evicted': 0, 'reward_rejected': 0}
        # eviction-cause tallies (see manage_replay_buffer's eviction split):
        # drained and logged once per eval alongside replay_churn. 'expired*'
        # keys carry the HAZARD (random-eviction) cohort -- the unbiased
        # readout of the live population -- while 'backstop' counts the one
        # remaining age-targeted cause.
        self.replay_cohort = {'backstop': 0,
                              'expired': 0, 'expired_undrawn': 0,
                              'expired_drawn': 0, 'expired_draws_sum': 0,
                              'expired_delta_sum': 0.0, 'expired_delta_n': 0}
        # last answer from replay_in_play(), for the transition print only (None
        # = never asked). Deliberately NOT checkpointed: it drives nothing.
        self._replay_managed = None
        # prior_buffer churn decomposed by admission SOURCE, tallied across every
        # manage_prior_buffer/top_up_prior_from_anchors/grow_prior_buffer call and
        # drained once per eval in log_buffer_stats. The point is the source mix:
        # from_anchors dominating from_prior_model means the prior has stopped
        # discovering admissible samples on its own and the buffer is living off
        # replayed archive material (see manage_prior_buffer's reach trigger)
        # 'budget' is the churn quota the prior-model draw was asked for, so the
        # admitted counts can be read as an admission RATE, not just a raw count
        self.prior_churn = {'from_prior_model': 0, 'from_anchors': 0, 'from_seed': 0,
                            'evicted': 0, 'budget': 0, 'expired': 0, 'anchor_floor': 0}
        self.device = self.args.device
        self.checkpointer = Checkpointer(self)
        self.lr_controller = LRController(self)  # fixed-peak ramp/hold/decay; tripwires always on
        # Adaptive clip bar. Built HERE and not in init_schedulers_optimizers
        # (where ray_cal is) precisely because that runs again at every stage
        # transition: the guard's tracker is state we want to survive a
        # boundary, and what the boundary should do to it is the softer
        # refresh() -- recalibrate while the old bar stays live. Absent config
        # block => disabled => threshold() is args.gradient_norm_clip, so this is
        # inert for every existing config.
        self.grad_guard = GradClipGuard.from_config(
            self.args.gradient_norm_clip, getattr(self.args, 'grad_clip_guard', None))
        self.grad_guard.announce()
        self.protocol = StageProtocol(self)  # the declarative stage engine: coeffs, balance, exits, transitions
        self._check_ray_wiring()
        self.init_train_constants()

    def _ray_probe_armed(self) -> bool:
        """Arm the ray probe iff THIS stage asked for it (lr_sensor kind 'ray').

        The probe is coherent only in a fused stage that trains replay TB -- it
        draws from replay (needing STORED trajectories) and scores with
        replay_loss_coeffs, so anywhere else it rates a loss nobody is
        optimising.

        A stage with NO lr_sensor block used to arm it anyway, governed by the
        global ray_calibration.enabled. That default is retired: arming by
        omission put a replay-dependent sensor into stages that never train
        replay (where _draw_probe_batch can only return None and tally
        raycal/skipped), and it was the last thing keeping the replay buffer
        load-bearing in a VarGrad-only protocol. Omitting the block now means NO
        LR sensor, which is what it reads like. _check_ray_wiring reports the two
        ways a config can disagree with that, at startup rather than here.

        arm() is called ONLY on the armed path: it clones every policy parameter,
        which is not something to spend on a stage that will not measure.

        THE SECOND GATE IS NOT AN OPTIMISATION. `measure` draws n_sub sub-batches
        from the replay buffer, and those draws consume RNG that nothing
        restores -- so a calibration whose reading the controller then discards
        still changes every subsequent training step. Measured: a 600-step
        tier-C pair was bit-identical to step 500 and diverged from step 501,
        the first probe, while every learning rate stayed bit-identical and
        `cal_applied` was 0.0 (findings.md F-039). The probe must therefore ask
        whether its result will be USED before it draws, not after.
        """
        sensor = self.protocol.stage.lr_sensor
        if sensor is None or sensor['kind'] != 'ray':
            return False
        refusal = self.lr_controller.calibration_refusal()
        if refusal is not None:
            # Consumes the period exactly as a completed calibration would, so
            # the first APPLIED calibration still lands on the step it always
            # did; see RayCalibration.refuse.
            self.ray_cal.refuse(refusal, self.step_ind)
            return False
        return self.ray_cal.arm(self.step_ind)

    def _ray_askers(self):
        """Stages declaring `lr_sensor: {kind: ray}`. This IS the probe's switch."""
        return [s.name for s in self.protocol.stages
                if s.lr_sensor is not None and s.lr_sensor['kind'] == 'ray']

    def _check_ray_wiring(self):
        """The ray probe is OPT-IN per stage: it runs where and only where a
        stage declares `lr_sensor: {kind: ray}` (see the gate in train_step).

        It used to arm by OMISSION -- any stage with no lr_sensor block ran it
        whenever a separate `ray_calibration.enabled` flag was true. That is
        backwards for a probe with a hard coherence requirement (it draws from
        replay and scores replay_loss_coeffs, so outside a fused stage training
        replay TB it rates a loss nobody is optimising).

        That flag is now GONE and `enabled` is derived from these askers, which
        removes the disagreement it made possible -- a stage asking for `ray`
        while the flag said false used to train at its seed LR with the config
        claiming a sensor. That case is unrepresentable now, so the check that
        caught it is gone with it.

        What remains is the other direction, which derivation cannot rule out:
        the block's parameters are present and nothing asks. Not an error -- the
        parameters are shared storage, and a run with no replay-TB stage is a
        legitimate configuration -- but worth saying, so the block does not read
        as an active sensor."""
        if self.protocol.stages and not self._ray_askers():
            print("NOTE: no stage declares lr_sensor kind 'ray', so the "
                  "ray_calibration block is inert (parameters only). It still "
                  "supplies alphas/n_sub/period to any stage that opts in.")

    def init_train_constants(self):
        for k, v in MODELLER_STATE_DEFAULTS.items():
            if k in self.args.__dict__:
                setattr(self, k, self.args.__dict__[k])
            else:
                setattr(self, k, deepcopy(v))

        self.metric_tracker = MetricTracker(period=100)
        # latest RAW per-branch loss stats (pre-EMA), refreshed by _update_rolling.
        # Deliberately not checkpointed: it's a one-step cache (the MLE slope gate
        # samples it every 10 steps into its own checkpointed window).
        self._last_stats = {}
        # last-logged value of every energy_func/ and loss_coeffs/ setting, so
        # dump_numeric can emit only the ones a stage transition actually moved.
        # Not checkpointed: an empty cache on resume just re-logs the full set
        # once, which is the right baseline for a fresh wandb run anyway.
        self._settings_log_cache = {}
        # per-submodel grad norms from the most recent reporting step, and the
        # count of non-finite-gradient steps since the last report
        self._last_grad_norms = {}
        self._grad_nonfinite = 0
        # samples and seconds accumulated since the last report, so throughput
        # is a true window ratio rather than batch_size / one sampled step time
        # (batch_size moves by 3-4x over a run, so the two are not the same)
        self._throughput = {'samples': 0, 'seconds': 0.0, 'energy_seconds': 0.0}

    # position in the protocol, derived -- checkpoints store the stage NAME; the
    # int only feeds wandb continuity and the LR controller's stage-change marker
    @property
    def phase(self):
        return self.protocol.stage.index + 1

    #: THE BUFFER CLASS EVERY CHURNED STORE IS BUILT FROM. A class attribute rather than
    #: a hardcoded name so a non-crystal Modeller can supply its own without this file
    #: growing an energy-function branch at six construction sites. Crystal behaviour is
    #: unchanged: same class, same kwargs.
    buffer_cls = CrystalBuffer

    #: the anchor store. A SEPARATE hook because AnchorBuffer is not buffer_cls: it carries
    #: its own reward/energy/surprise state and is constructed with a different signature,
    #: so it cannot be swapped by the same name. Same rationale otherwise.
    anchor_buffer_cls = AnchorBuffer

    def _buffer_kwargs(self):
        """Construction kwargs specific to the buffer's DATA MODEL, not to its policy.

        Split out for the same reason as buffer_cls: ``max_z_prime`` is meaningless on a
        conformer graph and does not merely go unused there -- MXtalBase defers unknown
        attributes to the PyG store, which raises. Sizes, devices and exclude-key sets stay
        at the call sites because they are the caller's business.
        """
        return dict(max_z_prime=max(self.args.z_primes))

    def _eval_extra_stats(self, mol_batch):
        """Per-sample columns the eval accumulator should carry beyond the TB family.

        Crystal adds the packing coefficient. A conformer has no cell and therefore no
        packing coefficient at all -- this is a genuinely absent quantity, not one that
        defaults to zero, so the conformer override returns nothing rather than inventing
        a column of NaNs that would then be averaged into a metric.
        """
        return {'packing_coeff': mol_batch.packing_coeff}

    def log_physical_properties(self, metrics, sample_batch, val, arr):
        """The domain's own physical-plausibility metrics.

        Split out for the same reason as _eval_extra_stats: packing coefficient and
        reduction energy are properties of a periodic cell. Overriding this is how a
        non-crystal route publishes its OWN physical metrics rather than suppressing the
        block -- an empty override would be a silent loss of the eval's physical reading.
        """
        metrics['Mean Packing Coeff'] = val(sample_batch.packing_coeff.mean())
        metrics['Packing Coeff'] = arr(sample_batch.packing_coeff.clip(max=2))
        metrics['Reduction Energy'] = arr((1e-3 + sample_batch.reduction_en).log10())
        metrics['Reduced Valid Fraction'] = np.mean(arr(sample_batch.reduction_en) < 1e-1)

    def _has_prior_sampler(self):
        """Is there anything to draw prior samples FROM?

        On the crystal route the prior is a frozen GFN (``prior_model``), produced by
        train_prior's snapshot_prior action or loaded by name. A route whose prior is not a
        GFN at all -- a fitted analytic prior, say -- answers this differently, and
        without the hook its phase-2 churn silently degrades to an anchor-only buffer:
        _prior_churn_cycle's guard is a report, not a raise, because an anchor-only
        composition is legal.
        """
        return hasattr(self, 'prior_model')

    def _noise_and_condition(self, batch, noise_log_range):
        """Jitter a stored batch's state, then condition it. IN PLACE on `batch`.

        The two anchor paths (refresh_anchor_buffer_surprise, top_up_prior_from_anchors)
        ran the identical three-line crystal preamble -- log_noise_latent_parameters,
        condition_samples with sg_ind/z_prime, orient_molecule -- so it is one seam, hooked
        once. Noising happens BEFORE conditioning so the noised state is what gets
        conditioned and scored.
        """
        batch.log_noise_latent_parameters(*noise_log_range)
        batch, log_T_tensor, condition, condition_id = self.energy_function.condition_samples(
            batch, sg_inds=batch.sg_ind, z_primes=batch.z_prime)
        batch.orient_molecule(mode='std')
        return batch, log_T_tensor, condition, condition_id

    _domain_figs = None      #: None = eval_figs uses its own crystal block, unchanged

    def _batch_latents(self, batch):
        """The GFN state carried by a STORED graph batch.

        The crystal state is the cell/pose latent, read off the graph by
        ``latent_params()``; the conformer state is the stored ``torsion_state``. Same
        rationale as buffer_cls -- ``MXtalBase.__getattr__`` defers to the PyG store, so
        ``latent_params`` does not go unused on a conformer graph, it raises. Crystal
        behaviour is unchanged: the same call, at the same sites.
        """
        return batch.latent_params()

    def _buffer_y_fn(self):
        """The batch KEY a churned buffer reads its scalar `y` from.

        ``energy_function`` doubles as that key on the crystal route -- the analysis
        attaches the term under its own name, so 'elj' is both the backend and the
        attribute. That coincidence does not survive a different energy: the conformer
        bakes its scalar as ``conformer_energy`` while its energy_function is
        ``conformer_torsions``, and passing the latter raises KeyError at buffer
        construction. Naming the key separately is what lets the two differ.
        """
        return self.args.energy_function

    @property
    def buffer_device(self):
        """
        Where the sample stores live ('cpu' default, 'cuda' opt-in): the
        churned prior/replay buffers, the static mol/prior datasets, AND the
        anchor buffer. GPU residency turns every per-step buffer op -- draws,
        admits, and purge_by_index's full-store rebuild -- into async device
        gathers instead of serial CPU work + host<->device round trips; for
        the anchor buffer specifically it keeps the every-5th-eval full-sweep
        maintenance (refresh/thin) on-device, which is what the tall
        eval_step_time blocks are made of. Measured footprint is ~1 KB/graph
        (250k graphs ~ 250 MB), so VRAM cost is a rounding error next to
        activations.
        """
        return getattr(self.args, 'buffer_device', 'cpu')

    @property
    def bwd_sampling_mode(self):
        return self.protocol.stage.bwd_sampling_mode

    def train_logic(self, it):
        return self.protocol.stage.train_mode

    def _now(self):
        """
        Wall clock for the time-windowed sensors, as a method so bench/ can drive
        them on a VIRTUAL clock. The bench runs thousands of steps in about a second,
        so real timestamps would put every sample inside every window and the
        averaging logic -- which is the part worth testing -- would never be
        exercised. One seam here beats reimplementing the windowing in the fake.
        """
        return time()

    def _sample_gpu_util(self):
        """
        One NVML utilization reading, appended with its timestamp. Cheap and
        time-gated, so the caller may invoke it every step.

        THIS IS THE JOB-SURVIVAL NUMBER: the cluster cancels a job whose GPU
        utilization averages under a threshold for a couple of hours -- prod0810's
        uma arm (4r351oqm) died that way at 5.2 h with hourly means
        75/62/54/49/48/48%. In-process it feeds select_batch_size's per-rung
        calibration readings and its S2 audit; the deleted gpu_util_floor rule --
        which grew on the windowed mean directly -- stays deleted
        (utils._RETIRED_KEYS holds the record). NB the trailing-window means built
        on these samples disagree with the out-of-process samplers by a
        batch-dependent, sign-flipping error (handoff §2), so the number the
        scheduler judges is the out-of-process one.

        SAMPLED ON A TIME CADENCE, NOT A STEP CADENCE. It used to be sampled once per
        ten_step_reporting, which is fine at 2 s/step and useless at 200 s/step: two
        readings 2000 s apart cannot populate a 900 s window, so `_gpu_util_mean`
        returned None and the metric was simply absent -- on exactly the slow MLIP
        arms whose utilization we most needed to watch. A wall-clock period decouples
        the sensor from the step time it is trying to characterise.

        TWO SOURCES, because the obvious one is not always there.
        `torch.cuda.utilization()` needs the pynvml bindings, which are NOT installed
        in this project's venv -- the first local run printed "gpu util sensor
        unavailable (ModuleNotFoundError)" and logged no occupancy at all for the
        whole run. So it falls back to nvidia-smi via gpu_guard, which is already a
        train.py dependency and demonstrably works on this box. Only after BOTH fail
        does it go inert, and then loudly: a missing sensor must never stop training,
        but a run with no occupancy trace is a run we cannot defend to the scheduler.
        """
        if getattr(self, '_gpu_util_off', False):
            return
        now = self._now()
        period = float(getattr(self.args, 'gpu_util_sample_period_s', 60) or 0)
        last = getattr(self, '_gpu_util_last_sample', None)
        if period > 0 and last is not None and now - last < period:
            return
        self._gpu_util_last_sample = now
        reading = self._read_gpu_util()
        if reading is None:
            self._gpu_util_off = True
            print("gpu util sensor unavailable (no pynvml AND no nvidia-smi) -- "
                  "no gpu/util_* metrics for this run. Nothing depends on it to train, "
                  "but on a usage-policed cluster this is the number the scheduler "
                  "cancels on and we will have no record of it.")
            return
        if not hasattr(self, '_gpu_util'):
            self._gpu_util = deque(maxlen=4096)
        self._gpu_util.append((now, reading))

    def _read_gpu_util(self):
        """
        One raw utilization percent, or None if no sensor answers. Split out from
        the sampling cadence above purely as a SEAM: bench/ substitutes a synthetic
        GPU here and then runs the real `_sample_gpu_util` on top of it, so the
        cadence logic under test is the shipping one rather than a copy of it. The
        harness used to reimplement the cadence, which is how it kept passing while
        the real sampler could not populate its own window on slow steps.
        """
        try:
            return float(torch.cuda.utilization())
        except Exception:
            pass
        try:
            import gpu_guard
            mem = gpu_guard.gpu_memory()              # (used, free, total, util%)
            if mem is not None:
                return float(mem[3])
        except Exception:
            pass
        return None

    def _gpu_util_mean(self, window_s: float):
        """Mean utilization over the trailing `window_s` seconds, or None if the
        sensor is off or the window is not yet populated. None means 'no reading',
        never 'fine'."""
        samples = getattr(self, '_gpu_util', None)
        if not samples:
            return None
        cutoff = self._now() - window_s
        recent = [u for ts, u in samples if ts >= cutoff]
        # a couple of readings inside a 15-minute window is not a mean, it is a coin
        # flip. With time-cadence sampling this is reachable at any step time; on the
        # old step-cadence it was what made the metric vanish on slow MLIP arms.
        if len(recent) < 5:
            return None
        return sum(recent) / len(recent)

    def _batch_floor(self) -> int:
        """
        The domain's lower bound: the configured `batch_size`, which is the base the
        run was designed around (and what protocol.advance re-enters each stage at).

        Under select_batch_size nothing ever walks downward -- the base rung IS the
        selection unless occupancy evidence lifts it -- so this floor is no longer
        what stops a descent; it bounds the domain. It is kept (rather than folded
        into the selection) because two cut paths can still land BELOW it, on
        purpose: an OOM cut, and max_step_seconds -- a hard wall-clock ceiling
        outranks a batch-size preference.
        """
        return max(1, min(int(self.args.batch_size), int(self.args.max_batch_size)))

    def select_batch_size(self):
        """
        Batch selection under the phase-6 replacement
        (docs/design/phase6_batch_sizer.md). Two objectives in strict priority:
        satisfy the cluster occupancy requirement, then maximize optimizer-step
        throughput subject to it.

        THE THROUGHPUT HALF IS A CONSTANT, NOT A SEARCH. With the effective
        update size held at A = fused_grad_accum_min_samples, updates/sec =
        samples_per_sec / A for any B <= A and samples/sec rises in B, so
        priority 2's answer is B = A exactly -- the configured batch_size, which
        the canonical config already sets equal to A. There is no throughput
        walk. Growth above the base exists ONLY to buy occupancy, must be
        minimal, and stops the moment the constraint is met.

        THREE STRUCTURAL RULES:
          S1  occupancy evidence may only VETO candidate sizes; the selection
              rule itself is fixed (the smallest measured rung that clears the
              target). No occupancy reading ever orders the batch on its own --
              the deleted gpu_util_floor was an actuator, and it drove four
              growths a throughput gate had refused (utils._RETIRED_KEYS holds
              the record).
          S2  a growth justified by "occupancy rises with batch" must produce
              that rise: after a full policy window at the selected rung, the
              lived occupancy is compared against the base rung's calibration
              reading, and if the growth did not deliver, the batch STANDS DOWN
              to the base rung. The max_step_seconds guard below carries the
              same rule for cuts (its unresponsiveness stand-down).
          S3  UNKNOWN never removes and never grows: no target configured, no
              sensor, or no reading -> hold the base. That is the shipping
              default (batch_util_target unset), under which this method holds
              B = batch_size and only the safety bounds ever move it.

        CALIBRATION IS FEED-FORWARD, OVER REAL TRAIN STEPS, ONCE PER STAGE.
        With a target set, the walk climbs the ladder from the base rung,
        dwelling at each rung until it has a step-time median and a few RAW
        occupancy samples taken at that rung -- never the trailing windowed
        mean, whose window straddles rungs and is what makes closed-loop
        control on this sensor unworkable. It then holds the smallest rung
        whose measured occupancy clears the target. If no rung clears, it holds
        the argmax-occupancy rung and says INFEASIBLE loudly, naming the
        binding bound: batch is not the lever then -- work per kernel launch,
        host stalls, or an energy-pinned memory ceiling are
        (docs/design/phase6_handoff.md §4.4).

        EVERY SAFETY MECHANISM IS A BOUND ON THE DOMAIN, NEVER A SELECTOR: the
        OOM ceiling (which expires), max_batch_size, the max_step_seconds
        runaway guard, and the post-cut cooldown all shrink what may be
        selected; none of them picks a size. protocol.advance clears the
        conclusion, the ceiling and the timing window at every stage
        transition, because stages have different step-cost profiles.
        """
        stage_name = getattr(getattr(self.protocol, 'stage', None), 'name', None)
        times = getattr(self, '_recent_step_times', None)
        med = float(np.median(list(times)[-20:])) if times and len(times) >= 10 else None

        # OCCUPANCY IS READ HERE AGAIN, but not the way gpu_util_floor read it. That
        # rule was an ACTUATOR -- "utilization low, grow" -- and on the route it was
        # written for the premise was measured false; the retirement record
        # (`utils._RETIRED_KEYS['gpu_util_floor']`) holds the numbers once. The
        # calibration below differs in every load-bearing respect: it MEASURES
        # occupancy per rung from raw samples taken at that rung (not a trailing
        # window straddling rungs), it can only conclude "this rung clears / does not
        # clear the target" (S1: veto plus a fixed selection rule, never an ordering),
        # it terminates on a finite ladder, and a growth it keeps must later survive
        # the S2 audit or stand down. Where occupancy genuinely does not rise with
        # batch, the walk finds nothing, says INFEASIBLE, and returns to the best
        # measured rung -- which is the behaviour the old rule lacked.

        # RUNAWAY GUARD. An absolute wall-clock ceiling on one train step, checked
        # before the growth walk because a step already over the ceiling must never be
        # grown. Set it far above the operating point -- it exists for the 181-262 s
        # pathology, not for tuning.
        max_step_s = float(getattr(self.args, 'max_step_seconds', 0) or 0)
        if max_step_s > 0 and med is not None and med > max_step_s:
            if self.step_ind < getattr(self, 'batch_size_cooldown_until', -1):
                return
            # ...unless cutting has already been shown not to help. The guard's whole
            # model is "step time scales with batch"; when the overrun is FIXED cost
            # (an MLIP hiccup, a z_cal transition at ~12x, a slow host) that model is
            # wrong and the proportional cut below ratchets forever -- 1000 -> 538 ->
            # 290 -> ... -> 1, training at batch 1 without ever raising an error, one
            # torch.compile recompile per rung. So MEASURE the previous cut: if the
            # batch fell materially and the median step time did not, stop cutting.
            if getattr(self, '_runaway_unresponsive_stage', None) == stage_name:
                return
            last = getattr(self, '_runaway_last_cut', None)
            if last is not None:
                prev_b, prev_med = last
                if self.batch_size <= prev_b / 1.4 and med >= 0.9 * prev_med:
                    self._runaway_unresponsive_stage = stage_name
                    print(f"batch: step time {med:.1f}s still over max_step_seconds "
                          f"{max_step_s:.1f}s after cutting {prev_b} -> {self.batch_size} "
                          f"({prev_med:.1f}s -> {med:.1f}s). The overrun is NOT batch-driven, "
                          f"so cutting further only shrinks the gradient -- standing down for "
                          f"stage '{stage_name}'. Look at fixed per-step cost (z_calibration "
                          f"rollouts at a transition, MLIP call overhead, host stalls).")
                    return
            f = float(getattr(self.args, 'batch_growth_factor', 2.0))
            # PROPORTIONAL, not one rung: step cost is close enough to linear in batch
            # that the overshoot ratio estimates the target directly, and a fixed /f
            # ladder converges far too slowly in exactly the case that matters -- a
            # 181 s step needs 4 cuts at oom_cooldown_steps apart, i.e. ~40 h of
            # 181 s steps to reach a size it should have taken in one. Never cut less
            # than one rung, and keep 10% margin so the next measurement lands under
            # the ceiling rather than on it.
            shrunk = max(1, min(int(round(self.batch_size / f)),
                                int(self.batch_size * (max_step_s / med) * 0.9)))
            # ...but not into the grad-accumulation regime. Below
            # fused_grad_accum_min_samples a fused step is a MICRO-step: cutting the
            # batch buys proportionally more micro-steps for the same samples, so
            # time per OPTIMIZER UPDATE does not fall at all -- only the per-iteration
            # number this ceiling happens to measure. Cutting 1000->400 against a
            # 1000-sample target turns one 25 s update into three 10 s micro-steps:
            # ceiling satisfied, update rate slightly worse. Memory pressure is the
            # OOM path's job, not this one's.
            accum = int(getattr(self.args, 'fused_grad_accum_min_samples', 0) or 0)
            if accum > 0 and getattr(self.protocol.stage, 'train_mode', None) == 'fused':
                shrunk = max(shrunk, min(accum, self.batch_size))
            if shrunk >= self.batch_size:
                # ONCE PER STAGE. select_batch_size runs every step, so an
                # unconditional print here is ~10k copies of the same paragraph on a
                # 7-day run -- and it holds whenever batch_size <= accum_target, which
                # is mk_dev's shipped 1000/1000 pair at the base batch of any fused
                # stage. Freezing the controller is right (growing a step that is
                # already over the ceiling would make it worse); shouting is not.
                if getattr(self, '_accum_floor_warned_stage', None) != stage_name:
                    self._accum_floor_warned_stage = stage_name
                    print(f"batch: step time {med:.1f}s over max_step_seconds {max_step_s:.1f}s "
                          f"at batch {self.batch_size}, which is already the smallest size that "
                          f"still makes one optimizer update ({accum} samples). Cutting further "
                          f"cannot speed the update up -- lower fused_grad_accum_min_samples, "
                          f"raise max_step_seconds, or accept the step cost. Growth is frozen "
                          f"for stage '{stage_name}' while this holds (said once).")
            if shrunk < self.batch_size:
                print(f"batch: step time {med:.1f}s over max_step_seconds {max_step_s:.1f}s "
                      f"-- cutting {self.batch_size} -> {shrunk}")
                # remembered so the NEXT firing can ask whether that cut did anything;
                # see the unresponsiveness check above
                self._runaway_last_cut = (self.batch_size, med)
                self.batch_size = shrunk
                # the wallclock bound outranks the occupancy ladder: whatever the walk
                # had measured or concluded is void at this size, and regrowing for
                # occupancy would re-trip the guard -- hold here.
                self.batch_sizer = dict(phase='hold', reason='wallclock_cut',
                                        selected=int(shrunk), table=[])
                self.batch_size_last_grow = self.step_ind
                self.batch_size_cooldown_until = self.step_ind + int(
                    getattr(self.args, 'oom_cooldown_steps', 200) or 0)
                times.clear()
                self._recent_step_work.clear()
            return
        # THE OOM CEILING EXPIRES. It is a permanent conclusion about VRAM drawn from
        # ONE failed allocation -- and the recovery for that failure (gc +
        # empty_cache, in handle_train_epoch_error) is itself what can make the
        # conclusion stale. An OOM caused by fragmentation left over from init, where
        # the mace/uma prior re-analysis reserves supercell-shaped blocks that a T-step
        # MLP rollout can never reuse, is CLEARED by the very cut that records it.
        #
        # Latched forever, that was catastrophic rather than merely conservative.
        # prod0810's acridine/mace arms: one OOM at the BASE batch of 1000 latched a
        # ceiling of 1000, under which the previous controller ran the whole of
        # train_prior at 0.825x its configured batch, on a stage that makes no energy
        # calls at all and is judged by the scheduler on GPU occupancy. All three died.
        #
        # So the ceiling decays like any other AIMD limit: after a quiet spell with no
        # further OOM, drop it and let the ladder re-probe. Re-probing costs one step
        # and a cooldown when the ceiling was right; NOT re-probing costs the whole
        # stage when it was wrong.
        ceiling = getattr(self, 'batch_size_oom_ceiling', None)
        # default matches the shipping value in configs/mk_dev.yaml and prod0810;
        # a code default that disagrees with every config is a trap for the one
        # config that forgets the key
        retest = int(getattr(self.args, 'batch_oom_ceiling_retest_steps', 1000) or 0)
        if ceiling is not None:
            stamped = getattr(self, 'batch_size_oom_ceiling_at', None)
            if stamped is None:
                # A RESTORED CEILING STARTS ITS CLOCK HERE. A resume brings back the
                # ceiling but not necessarily the stamp, and step_ind is already large
                # -- measured against an absent (0) clock the ceiling would expire on
                # the first post-resume step, which is precisely the OOM the resumed
                # run was checkpointed to avoid re-discovering.
                self.batch_size_oom_ceiling_at = stamped = self.step_ind
            if retest > 0 and self.step_ind - int(stamped) >= retest:
                print(f"batch: OOM ceiling {ceiling} has stood {retest}+ steps without "
                      f"another OOM -- clearing it and re-probing upward from "
                      f"{self.batch_size}")
                self.batch_size_oom_ceiling = None
                self.batch_size_oom_ceiling_at = None
                self.batch_ceiling_expiries = getattr(self, 'batch_ceiling_expiries', 0) + 1
                sizer = getattr(self, 'batch_sizer', None)
                if sizer is not None and sizer.get('reason') not in ('wallclock_cut',
                                                                     'stood_down'):
                    # the ceiling may have been what bounded the conclusion -- an
                    # INFEASIBLE verdict, or a base batch held below the configured
                    # one -- so with it gone, re-derive: restore the base and re-run
                    # any ladder. The two exceptions are conclusions the ceiling did
                    # not produce: a wallclock cut (max_step_seconds outranks this)
                    # and an S2 stand-down (growth already failed its audit once;
                    # re-climbing on a timer would oscillate).
                    self.batch_sizer = None

        # ----------------------------------------------------------- the ladder law
        # THE CONFIG UNIT IS A FRACTION, THE SENSOR'S IS PERCENT, and this is the
        # one place they meet. `batch_util_target: 0.6` means 60% of the card;
        # every comparison and message below is in the sensor's percent, so the
        # conversion happens once, here, rather than at each use. Written as a
        # fraction in the config because that is how a target on a [0,1] quantity
        # reads, and because a bare 0.6 meant as "60%" would otherwise be a legal
        # 0.6% target that any rung clears -- an inert constraint reporting
        # itself as served.
        target = 100.0 * float(getattr(self.args, 'batch_util_target', 0) or 0)
        hi = int(self.args.max_batch_size)
        ceiling = getattr(self, 'batch_size_oom_ceiling', None)
        ceiling_binds = ceiling is not None and int(ceiling) <= hi
        if ceiling is not None:
            # a size that OOM'd in this stage is a measurement, not noise: the domain
            # stops strictly below it. A BOUND, never a pin -- it shrinks what may be
            # selected and selects nothing itself.
            hi = min(hi, int(ceiling) - 1)

        s = getattr(self, 'batch_sizer', None)
        if s is None:
            # RESTORE THE BASE before concluding anything. An OOM or wallclock cut
            # can leave the batch BELOW the configured base, and with no growth walk
            # nothing else would ever bring it back -- the prod0810 failure (a stage
            # judged on occupancy running under-sized for its life) rebuilt by
            # omission. Restore only when the base is not itself the size that
            # OOM'd (base < ceiling, or no ceiling stands) and the cooldown is over;
            # otherwise hold the cut size and let the ceiling's expiry re-open this.
            base = self._batch_floor()
            if self.batch_size < base and (ceiling is None or base < int(ceiling)):
                if self.step_ind < self.batch_size_cooldown_until:
                    return               # the cut is settling; restore on a later call
                print(f"batch: restoring {self.batch_size} -> {base} (the configured "
                      f"base"
                      + ("" if ceiling is None
                         else f"; the OOM ceiling {ceiling} sits above it") + ")")
                self.batch_size = int(base)
                self._recent_step_times.clear()
                self._recent_step_work.clear()
            # first call since a stage transition / OOM / ceiling expiry cleared the
            # conclusion: decide whether there is anything to measure at all
            if target <= 0:
                # S3, and the shipping default: no constraint configured, so the
                # answer is the base batch, held. No walk, no probe, no print.
                self.batch_sizer = dict(phase='hold', reason='no_target',
                                        selected=int(self.batch_size), table=[])
            elif getattr(self, '_gpu_util_off', False):
                self.batch_sizer = dict(phase='hold', reason='sensor_off',
                                        selected=int(self.batch_size), table=[])
                print("batch: batch_util_target is set but the occupancy sensor is "
                      "off -- holding the base batch. UNKNOWN never grows (S3).")
            elif hi <= self.batch_size:
                self.batch_sizer = dict(phase='hold', reason='no_headroom',
                                        selected=int(self.batch_size), table=[])
            else:
                self.batch_sizer = dict(
                    phase='calibrating', reason=None, selected=int(self.batch_size),
                    table=[], rung_start_step=int(self.step_ind),
                    rung_start_time=float(self._now()), audit_at=None)
            return

        if s.get('phase') == 'hold':
            # S2 AUDIT, one shot per selection, armed only when the hold is a growth
            # above the base rung. The calibration predicted this rung's occupancy
            # from a short dwell; a full policy window of lived occupancy is the
            # falsification test. If growing did not deliver more occupancy than the
            # base rung measured, the constraint's own model is wrong here and the
            # removal of the small rungs is withdrawn.
            audit_at = s.get('audit_at')
            if audit_at is not None and self._now() >= float(audit_at):
                s['audit_at'] = None
                table = s.get('table') or []
                base = table[0] if table else None
                lived = self._gpu_util_mean(float(getattr(
                    self.args, 'gpu_util_policy_window_s', 7200) or 7200))
                if (base is not None and base.get('util') is not None
                        and lived is not None and lived <= float(base['util'])):
                    print(f"batch: occupancy audit FAILED -- a full policy window at "
                          f"{self.batch_size} reads {lived:.1f}%, not above the base "
                          f"rung {base['batch']}'s calibration {base['util']:.1f}%. "
                          f"The growth did not deliver the effect that justified it, "
                          f"so it stands down (S2): {self.batch_size} -> {base['batch']}.")
                    self.batch_size = int(base['batch'])
                    s.update(phase='hold', reason='stood_down',
                             selected=int(base['batch']))
                    self._recent_step_times.clear()
                    self._recent_step_work.clear()
            return

        # phase == 'calibrating': dwell at the current rung until it is MEASURED --
        # a step-time median from this rung's own timings plus a few raw occupancy
        # samples taken while it ran -- then act on the reading. The walk visits
        # each rung once, ascending (geometric below the cap, linear at it), so it
        # terminates by construction.
        if self.step_ind < self.batch_size_cooldown_until:
            return                       # an OOM cut is settling; measure after it
        if int(self.batch_size) != int(s.get('selected', -1)):
            # something outside the walk moved the batch (an OOM cut): the rungs it
            # was climbing are now bounded away, so conclude from what is measured
            self._conclude_batch_calibration(
                target, hi, ceiling_binds, stage_name,
                note='the walk was interrupted by a cut')
            return
        interval = max(1, int(getattr(self.args, 'batch_growth_interval', 0) or 50))
        rung_steps = self.step_ind - int(s.get('rung_start_step', 0))
        samples = [u for ts, u in getattr(self, '_gpu_util', ())
                   if ts >= float(s.get('rung_start_time', 0.0))]
        if med is None or rung_steps < interval or len(samples) < _BS_MIN_UTIL_SAMPLES:
            if rung_steps >= _BS_RUNG_TIMEOUT_INTERVALS * interval and \
                    len(samples) < _BS_MIN_UTIL_SAMPLES:
                # the sensor is not producing on this cadence. S3: a sensor that
                # answers nothing removes nothing, and it grows nothing either.
                self._conclude_batch_calibration(
                    target, hi, ceiling_binds, stage_name,
                    note='the occupancy sensor starved a rung')
            return
        util = float(sum(samples) / len(samples))
        s['table'].append(dict(batch=int(self.batch_size), med_s=float(med),
                               util=util, n_util=len(samples)))
        if util >= target:
            # the smallest measured rung clearing the target, because the walk
            # ascends: growth was minimal and stops the moment the constraint is met
            grew = len(s['table']) > 1
            policy_s = float(getattr(self.args, 'gpu_util_policy_window_s', 7200) or 7200)
            s.update(phase='hold', reason='target_met', selected=int(self.batch_size),
                     audit_at=(self._now() + policy_s) if grew else None)
            print(f"batch: occupancy calibration -- {util:.1f}% at {self.batch_size} "
                  f"clears the {target:.0f}% target"
                  + (f" (base rung {s['table'][0]['batch']} read "
                     f"{s['table'][0]['util']:.1f}%); holding, audit in one policy "
                     f"window" if grew else "; holding the base batch"))
            return
        # CAPPED-GEOMETRIC RUNG STEP: multiply by batch_growth_factor while the
        # increment stays under batch_growth_cap, then climb linearly at the cap.
        # The cap bounds the selection's ABSOLUTE overshoot -- the held rung can
        # exceed the true crossing by at most one cap's worth of samples, and the
        # relative update-rate cost of that shrinks as B grows, exactly where the
        # pure-geometric worst case (a factor-f overshoot) hurts most. The price
        # is more rungs on a high climb, and the currency there is COMPILE
        # SHAPES, not dwell time: every distinct size is a recompile, and dynamo
        # falls back to eager past its cache limit. 0 = uncapped geometric, the
        # escape hatch if the shape budget ever becomes the binding problem.
        f = float(getattr(self.args, 'batch_growth_factor', 2.0))
        cap = int(getattr(self.args, 'batch_growth_cap', 1000) or 0)
        step = int(round(self.batch_size * (f - 1.0)))
        if cap > 0:
            step = min(step, cap)
        nxt = min(hi, max(self.batch_size + 1, self.batch_size + step))
        if nxt <= self.batch_size:
            self._conclude_batch_calibration(target, hi, ceiling_binds, stage_name)
            return
        self.batch_size = int(nxt)
        s['selected'] = int(nxt)
        s['rung_start_step'] = int(self.step_ind)
        s['rung_start_time'] = float(self._now())
        # timings from the previous rung would contaminate this rung's median
        self._recent_step_times.clear()
        self._recent_step_work.clear()

    def _conclude_batch_calibration(self, target, hi, ceiling_binds, stage_name,
                                    note=None):
        """
        End the ladder walk with no rung having cleared the occupancy target.

        The selection rule stays fixed (S1): hold the argmax-occupancy rung among
        those measured and still inside the domain, and say INFEASIBLE loudly --
        naming the binding bound, because "which bound bit" is the diagnosis. An
        OOM ceiling binding here usually means the ENERGY call's memory is pinned
        to the rollout batch (docs/design/phase6_handoff.md §4.4): the fix is
        decoupling the energy batch, not any batch this controller can pick.
        """
        s = self.batch_sizer
        eligible = [r for r in (s.get('table') or [])
                    if r.get('util') is not None and r['batch'] <= hi]
        if not eligible:
            # nothing measured inside the domain -> hold whatever is running; it is
            # the only size with evidence it runs at all (S3)
            s.update(phase='hold', reason='sensor_off' if note else 'infeasible',
                     selected=int(self.batch_size))
            print(f"batch: occupancy calibration ended with no usable rung"
                  + (f" ({note})" if note else "")
                  + f" -- holding {self.batch_size}")
            return
        best = max(eligible, key=lambda r: (r['util'], -r['batch']))
        base = eligible[0]
        if ceiling_binds:
            bound = ("an OOM ceiling -- on an MLIP route that is usually the ENERGY "
                     "call's memory, pinned to the rollout batch "
                     "(docs/design/phase6_handoff.md §4.4); the lever is decoupling "
                     "the energy batch, not growing this one")
        else:
            bound = "max_batch_size"
        print(f"batch: occupancy INFEASIBLE for stage '{stage_name}' -- no batch in "
              f"[{base['batch']}..{eligible[-1]['batch']}] reached the {target:.0f}% "
              f"target (best {best['util']:.1f}% at {best['batch']}). Batch is not "
              f"the lever here; the binding bound is {bound}. Holding {best['batch']}"
              + (f" ({note})" if note else "") + ".")
        grew = int(best['batch']) != int(base['batch'])
        policy_s = float(getattr(self.args, 'gpu_util_policy_window_s', 7200) or 7200)
        if int(best['batch']) != int(self.batch_size):
            self.batch_size = int(best['batch'])
            self._recent_step_times.clear()
            self._recent_step_work.clear()
        s.update(phase='hold', reason='infeasible', selected=int(self.batch_size),
                 audit_at=(self._now() + policy_s) if grew else None)


    def step_lr_schedule(self):
        # the LRController owns the LRs unconditionally (v7): warmup envelope x
        # a peak the alpha* servo owns on any group whose config key was written
        # `auto`. There is no separate legacy scheduler path to fall back to.
        return self.lr_controller.step()

    def ten_step_reporting(self):
        metrics = {}
        metrics.update(self.metric_tracker.snapshot(changed_only=True))

        for opt_type in ['fwd', 'bwd', 'replay', 'fused', 'flow']:
            if opt_type in self.optimizers:
                metrics.update({f'lr_{opt_type}': self.optimizers[opt_type].param_groups[0]['lr']})
        if 'fused' in self.optimizers:
            # the Z head's REAL LR under fused mode: the fused optimizer's trailing
            # flow group. The standalone-optimizer 'lr_flow' above is unused (and
            # anneals) on fused steps, so it reads misleadingly low there
            metrics['lr_fused_flow'] = self.optimizers['fused'].param_groups[-1]['lr']

        metrics['phase'] = self.phase
        # live batch size at step-time resolution: paired with train_step_time
        # this makes every run a samples/sec-vs-batch knee scan for free
        metrics['Batch Size'] = self.batch_size
        # ray-calibration readings (empty dict unless ray_calibration.enabled). Read
        # probe/alpha_median against probe/alpha_iqr and probe/fit_*_rate: a
        # wandering median, or a rising downward/flat rate, voids the sensor
        # independently of what the alpha* values say.
        metrics.update(self.ray_cal.report())
        metrics.update(getattr(self, '_replay_is_stats', {}) or {})
        # Memorisation sensor. replay/resid_vs_intake is the servo's input:
        # 1.0 = delay line, 1/e = 0.368 = the lambda*tau = 1 boundary, below
        # that the buffer is being fitted at its own trajectories.
        if getattr(self, 'replay_buffer', None) is not None and len(self.replay_buffer) > 0:
            metrics.update(self.replay_buffer.absorption_stats())
        if hasattr(self, 'last_grad_norm_pre_clip'):
            metrics['grad_norm_pre_clip'] = self.last_grad_norm_pre_clip
        # Adaptive clip bar, per branch. gradclip/*_fire_rate is the one that
        # says WHICH ALGORITHM is running: near 1-p is a guard, 0 is a guard that
        # has gone absent, 1 is normalized gradient descent wearing a clip's name.
        # grad_norm_pre_clip above cannot answer that -- it is a single scalar
        # holding whichever branch stepped last.
        metrics.update(self.grad_guard.report())
        # per-submodel grad norms (see _submodel_grad_norms) + how many steps
        # since the last report had a non-finite gradient and were skipped
        metrics.update(self._last_grad_norms)
        metrics['gradnorm/nonfinite_steps'] = self._grad_nonfinite
        self._grad_nonfinite = 0
        # fused-branch gradient-geometry diagnostic (grad_geometry.enabled) --
        # consume-on-read: it's computed far less often than every 10 steps,
        # so once logged it must not repeat as a stale value on later reports
        if getattr(self, '_fused_grad_geom_report', None):
            metrics.update(self._fused_grad_geom_report)
            self._fused_grad_geom_report = None
        # interspersed z-calibration telemetry, drained each report
        # (z_cal/steps is a count SINCE the last report, not a rate)
        if getattr(self, '_z_cal_report', None):
            metrics.update(self._z_cal_report)
            self._z_cal_report = {}
        # true throughput over the window: total samples / total seconds. Step
        # time alone can't answer 'did that change make it faster' while
        # batch_size is a moving denominator (982-3000 on dev, 1650-7410 on
        # cluster runs), which is exactly the regime the grow/OOM cycle creates
        if self._throughput['seconds'] > 0:
            metrics['samples_per_sec'] = (
                    self._throughput['samples'] / self._throughput['seconds'])
            # OPTIMIZER-STEP throughput, which is a DIFFERENT objective from
            # samples/sec and has a different argmax. Below
            # fused_grad_accum_min_samples a fused step is a micro-step, so N
            # batches accumulate into one update and updates/sec =
            # samples_per_sec / accum_target; at or above it every step is one
            # update. The identity "maximising samples/sec maximises opt-steps/sec"
            # is asserted in several places and holds only in the second regime --
            # so on any arm running below the crossover (every MLIP arm at batch
            # 50-500) samples/sec is NOT the quantity to maximise, and without
            # this line nothing logs the one that is.
            _accum = int(getattr(self.args, 'fused_grad_accum_min_samples', 0) or 0)
            _target = max(_accum, int(self.batch_size)) if _accum else int(self.batch_size)
            if _target > 0:
                metrics['updates_per_sec'] = metrics['samples_per_sec'] / _target
                metrics['batch/accum_target'] = _target
        # THE STATISTIC THE BATCH CONTROLLER ACTUALLY ACTS ON, which until now was
        # computed and thrown away (select_batch_size medians the last 20 step
        # times and decides on that). Every other cost metric here is a mean over
        # the 10-step report window, so a ladder built from them describes a curve
        # no in-process controller can be shown to reproduce -- and nothing
        # anywhere measures whether 20 samples is enough or what that median's own
        # dispersion is. Logging it makes every measured operating point a
        # measurement of the controller's own estimator.
        _times = getattr(self, '_recent_step_times', None)
        if _times and len(_times) >= 10:
            _med = float(np.median(list(_times)[-20:]))
            metrics['batch/med_step_s'] = _med
            if _med > 0:
                metrics['batch/sps_rung'] = float(self.batch_size) / _med
        # WHERE THE STEP'S SECONDS GO. energy/frac_of_step is the load-bearing one:
        # paired with GPU utilization it separates 'the MLIP call is expensive' from
        # 'the MLIP call is idle waiting on the host'. Denominator is the same window
        # the throughput numbers use, so the fraction is directly comparable to them.
        # Empty dict on stages that never evaluate the energy (bwd/dataset MLE), so
        # nothing misleading gets logged there.
        energy_timing = self.energy_function.drain_energy_timing()
        if energy_timing:
            # energy/seconds is EVERY call in the window -- training steps, eval
            # sampling, anchor screening, prior churn. Only the in-step share can be
            # divided by the step window; using the raw total gave 1.48 on the first
            # real run. Both are logged: in_step for "is the MLIP the thing to
            # optimise", total for "how much MLIP is this run doing at all".
            in_step = self._throughput.get('energy_seconds', 0.0)
            energy_timing['energy/seconds_in_step'] = in_step
            if self._throughput['seconds'] > 0:
                energy_timing['energy/frac_of_step'] = in_step / self._throughput['seconds']
                energy_timing['energy/frac_outside_step'] = max(
                    0.0, energy_timing['energy/seconds'] - in_step) / self._throughput['seconds']
        metrics.update(energy_timing)
        # GPU occupancy. Two consumers now: these metrics, and select_batch_size --
        # which reads RAW per-rung samples during calibration and the policy-window
        # mean once for its S2 audit, never these windows as a control input. NB the
        # in-process gpu/util_policy is known to disagree with the out-of-process
        # samplers by a batch-dependent, sign-flipping error (handoff §2): the number
        # the scheduler judges is the OUT-OF-PROCESS one (wandb system stream /
        # nvidia-smi sidecar); this series survives as the in-process view of it.
        # Sampling happens in the train loop on a wall-clock cadence; this only reads.
        util_recent = self._gpu_util_mean(
            float(getattr(self.args, 'gpu_util_window_s', 900) or 900))
        if util_recent is not None:
            metrics['gpu/util_recent'] = util_recent
        policy_s = float(getattr(self.args, 'gpu_util_policy_window_s', 7200) or 7200)
        policy = self._gpu_util_mean(policy_s)
        if policy is not None:
            metrics['gpu/util_policy'] = policy
        self._throughput = {'samples': 0, 'seconds': 0.0, 'energy_seconds': 0.0}
        # VRAM, on the same cadence as occupancy. Read vram/cached_mb against
        # 'Batch Size': a batch that falls while cached_mb stays high is the caching
        # allocator holding ground the run cannot use, which is the difference between
        # "this job needs a smaller batch" and "this job needs its memory back".
        metrics.update(self.vram_metrics())
        # BATCH CONTROLLER STATE AS SERIES. Everything this controller concludes it
        # otherwise says only in prints, and prints do not survive the runs that most
        # need a postmortem: wandb uploads no console log at all for a run left in
        # state 'crashed' (a hard kill / node loss, where the exit handler never runs),
        # so a scancelled job loses its whole account of itself. History streams
        # continuously and survives that. Counters are CUMULATIVE rather than per-step
        # flags so an event landing between two ten-step reports still shows up as a
        # step change instead of being missed entirely.
        #
        # Read `batch/oom_events` against 'Batch Size' to place every cut, and
        # `batch/oom_ceiling` against `vram/cached_mb` to tell a real memory limit from
        # a stale one -- a ceiling standing while cached_mb stays high is the caching
        # allocator holding ground the run cannot use, not a batch that genuinely does
        # not fit. 0 = no ceiling currently standing (the state is None).
        metrics['batch/oom_events'] = float(getattr(self, 'batch_oom_events', 0))
        metrics['batch/ceiling_expiries'] = float(getattr(self, 'batch_ceiling_expiries', 0))
        metrics['batch/oom_ceiling'] = float(getattr(self, 'batch_size_oom_ceiling', None) or 0)
        metrics['batch/oom_min'] = float(getattr(self, 'batch_size_oom_min', None) or 0)
        # the sizer's conclusion as series, encoded by _BS_PHASE_CODES /
        # _BS_REASON_CODES (train.py module constants). reason 3 = INFEASIBLE is the
        # one to alarm on: it means no batch reaches the occupancy target and the
        # binding bound is named in the console print.
        _sizer = getattr(self, 'batch_sizer', None) or {}
        metrics['batch/sizer_phase'] = float(_BS_PHASE_CODES.get(_sizer.get('phase'), 0))
        metrics['batch/sizer_reason'] = float(_BS_REASON_CODES.get(_sizer.get('reason'), 0))
        # rungs measured this stage: the compile-shape budget's own account. A long
        # climb under the capped-linear tail is what would threaten dynamo's
        # recompile cache, and it should announce itself here rather than as a
        # silent fallback to eager.
        metrics['batch/sizer_rungs'] = float(len(_sizer.get('table') or ()))
        metrics['Fwd Frac'] = self.fwd_frac
        metrics['Bwd Frac'] = self.bwd_frac
        metrics['Replay Frac'] = self.replay_frac
        # boost state, per-rule live (annealed) thresholds/elevations, exit streaks
        metrics.update(self.protocol.report())
        metrics.update(drain_elapsed_times(self.times))
        # PERSISTENT (cross-visit) views of the same quantities the rolling
        # fwd|bwd|replay channels carry per batch. Namespaced 'tracker/' on
        # purpose: only the rolling channels reach metric_tracker, so only THEY
        # are resolvable by protocol rules/exit terms ('dir/name'). Anything
        # published here is a dashboard reading, never a control variable --
        # which also matters because these reductions fail OPEN (0.0) whenever
        # no condition has enough evidence yet.
        cwq = self.args.conditional_worst_quantile
        metrics['tracker/tb_err_rms'] = self.condition_log_z.rms_tb_err()      # quality of fit, nats
        metrics['tracker/tb_err_worst'] = self.condition_log_z.worst_tb_err(quantile=cwq)
        metrics['tracker/z_grad_rms'] = self.condition_log_z.rms_z_grad()      # dL/dZ ruler, level-only
        metrics['tracker/z_grad_worst'] = self.condition_log_z.worst_z_grad(quantile=cwq)
        metrics['tracker/z_bias_rms'] = self.condition_log_z.rms_z_bias()      # unclipped level (diagnostic)
        # UNCLIPPED tail + dispersion. z_grad_worst above reads the CLIPPED
        # stream, so it saturates at clip_beta and goes blind exactly where the
        # damage is: a condition mis-levelled by 30 nats reads the same as one
        # off by 10, while the policy's only way to comply with a too-high Z is
        # to spread mass off-support (P_F is normalized), which inflates
        # variance, worsens samples, and grows the residual that caused it.
        # 'z_bias narrow and light-tailed' is the health condition -- rms gives
        # the width, worst gives the tail, var gives what the z_var loss term
        # actually penalizes.
        metrics['tracker/z_bias_worst'] = self.condition_log_z.worst_z_bias(quantile=cwq)
        metrics['tracker/z_bias_var'] = self.condition_log_z.var_z_bias()
        metrics['tracker/logw_std_rms'] = self.condition_log_z.rms_logw_std()  # spread; +inf until warmed
        # z_match level-matching gap: worst-condition |bwd_level - fwd_level|
        # over the two single-mode streams (+inf until enough conditions are
        # characterized, so the gates/delta_worst exit term can never pass on
        # ignorance -- protocol's _resolve drops non-finite values). 'worst' is
        # the conditional_worst_quantile upper-tail quantile across conditions,
        # the same worst-case-tolerance knob tb_err_worst uses -- so a handful of
        # stale/outlier conditions can't hold the gate hostage the way the strict
        # max did on a large condition library (cw02).
        delta = self.condition_log_z.delta_stats(quantile=cwq)
        metrics['zmatch/delta_worst'] = delta['worst']
        metrics['zmatch/delta_mean'] = delta['mean']
        metrics['zmatch/delta_n_trusted'] = delta['n']
        self.protocol.publish_gate('delta_worst', delta['worst'])
        # the two sides of the gap on the gate's own fast clock -- the
        # bwd/fwd jensen_z panels are longer-horizon metric_tracker EMAs and
        # lag the true levels by nats during a fast z_match walkdown
        levels = self.condition_log_z.pooled_levels()
        metrics['zmatch/fwd_level'] = levels['fwd']
        metrics['zmatch/bwd_level'] = levels['bwd']

        # always logged, including on arms with no servo-managed LR: a flat
        # peak_scale is itself the reading, and lr_ctrl/servo_hold says WHY the
        # loop is not moving. A silently absent block and a satisfied
        # controller must not look the same.
        metrics.update(self.lr_controller.report())

        if hasattr(self, 'condition_log_z'):
            if self.condition_log_z.library_size == 2:
                for cid, val in enumerate(self.condition_log_z.ema_logw.tolist()):
                    metrics[f'condition_log_z_ema_logw/{cid}'] = val
                for cid, val in enumerate(self.condition_log_z.ema_log_z_emp.tolist()):
                    metrics[f'condition_log_z_ema_log_z_emp/{cid}'] = val
                for cid, val in enumerate(self.condition_log_z.fwd_level_ema.tolist()):
                    metrics[f'condition_fwd_level/{cid}'] = val
                for cid, val in enumerate(self.condition_log_z.bwd_level_ema.tolist()):
                    metrics[f'condition_bwd_level/{cid}'] = val

        # --- step-time tail probes: WINDOW-MAX over the 10 steps since the last
        # report, so a rare slow step can't slip between the 1-in-10 samples the
        # way plain train_step_time does. Convicts the spike source by
        # correlation: a step-time spike + a churn-cohort spike = avalanche; a
        # step-time spike + a device_alloc burst = CUDA allocator expansion; a
        # step-time spike with neither = look elsewhere.
        probe = getattr(self, 'probe_window', None)
        if probe is not None:
            metrics.update({f'probe/{k}': v for k, v in probe.items()})
        self.probe_window = {}  # reset the window
        if torch.cuda.is_available():
            stats = torch.cuda.memory_stats()
            prev = getattr(self, '_prev_alloc_stats', None)
            cur = {k: stats.get(k, 0) for k in ('num_device_alloc', 'num_device_free')}
            if prev is not None:
                metrics['probe/device_alloc_delta'] = cur['num_device_alloc'] - prev['num_device_alloc']
                metrics['probe/device_free_delta'] = cur['num_device_free'] - prev['num_device_free']
            self._prev_alloc_stats = cur

        return metrics

    def _probe_max(self, key, value):
        """Roll value into the 10-step probe window as a running max."""
        w = getattr(self, 'probe_window', None)
        if w is None:
            w = self.probe_window = {}
        if value > w.get(key, float('-inf')):
            w[key] = value

    def set_loss_coeffs(self):
        """Live loss coefficients are a pure function of (base config defaults,
        current stage): the protocol overlays the stage's non-default overrides
        on the base fwd/bwd/replay_loss_coeffs blocks. No schedules, no
        mutation across steps -- the namespaces are rebuilt each call, so a
        stage transition takes effect the moment this runs."""
        for mode in ('fwd', 'bwd', 'replay'):
            setattr(self.args, f'{mode}_loss_coeffs', dict2namespace(self.protocol.coeffs(mode)))

        # SubTB coefficient matrix: a pure function of (subtb_lambda, T), both
        # static for the run, so build each distinct one once and hand out the
        # same tensor. Only materialised if some branch actually runs subtb.
        if any(getattr(self.args, f'{m}_loss_coeffs').subtb > 0 for m in ('fwd', 'bwd', 'replay')):
            cache = getattr(self, '_subtb_coeff_cache', None)
            if cache is None:
                cache = self._subtb_coeff_cache = {}
            for mode in ('fwd', 'bwd', 'replay'):
                coeffs = getattr(self.args, f'{mode}_loss_coeffs')
                key = (coeffs.subtb_lambda, self.args.integrator.T)
                if key not in cache:
                    cache[key] = cal_subtb_coef_matrix(*key).to(self.gfn_model.device)
                coeffs.coeff_matrix = cache[key]

        self._warn_if_z_untrained()

    def _warn_if_z_untrained(self):
        """Warn when the live coefficients leave the flow (Z) head with no
        trainer at all. The silent combination is tb_z_source: 'persistent' with
        no sidecar: get_tb_loss substitutes a DETACHED per-condition target
        wherever the tracker is warmed, so those rows give the flow model no TB
        gradient, and emp_z_persistent is the term that is supposed to take over
        (see gflownet_losses.get_tb_loss). Config-level mistake rather than a
        code path, so this reports rather than raises."""
        trainers = []
        for mode in ('fwd', 'bwd', 'replay'):
            c = getattr(self.args, f'{mode}_loss_coeffs', None)
            if c is None or getattr(c, 'freeze_z', 0) > 0.5:
                continue
            tb_trains_z = (getattr(c, 'tb', 0) > 0
                           and self.tb_z_source(mode) != 'persistent')
            if tb_trains_z or any(getattr(c, k, 0) > 0 for k in
                                  ('emp_z', 'emp_z_persistent', 'z_level', 'db', 'subtb')):
                trainers.append(mode)
        stage_name = self.protocol.stage.name
        if not trainers:
            if getattr(self, '_z_untrained_warned_stage', None) != stage_name:
                self._z_untrained_warned_stage = stage_name
                print(f"WARNING [stage '{stage_name}']: no mode trains the flow "
                      f"(Z) head -- every branch is freeze_z, or its TB reads a detached "
                      f"persistent target with no emp_z/emp_z_persistent/z_level sidecar. "
                      f"log_Z will not move for this stage.")
        else:
            self._z_untrained_warned_stage = None

    def set_energy_coeffs(self):
        """Live energy-function coefficients (e.g. bounding_coeff,
        reduction_coeff): mirrors set_loss_coeffs, but unlike loss_coeffs
        these CAN mutate step-to-step within a stage. The base energy_config
        value is the SOFT one -- self.energy_function is constructed with
        it, so it's already in effect from run start; a stage's
        balance.anneal_coeffs only ramps its named coefficients UP toward
        their own `target` once that stage's balance rules run clean
        (protocol.py StageProtocol.energy_coeffs)."""
        for name, value in self.protocol.energy_coeffs().items():
            if hasattr(self.energy_function, name):
                setattr(self.energy_function, name, value)

    def get_conditioning_dim(self):
        conditions_dim = 0
        if self.args.temperature_conditioning:
            conditions_dim += 1
        if self.args.sg_conditioning:
            conditions_dim += 237
        if self.args.zp_conditioning:
            conditions_dim += 1
        if getattr(self.args, 'vector_conditioning', False):
            conditions_dim += self.args.vector_conditioning_dim
        if getattr(self.args, 'embedding_conditioning', False):
            # pre-embedded molecule conditions: the FLATTENED encoder latent baked onto
            # each molecules_path entry (3 * bottleneck). conditions_type stays 'vector',
            # so this rides the same scalarMLP conditioner as `c` does
            conditions_dim += self.args.embedding_conditioning_dim
        return conditions_dim

    def init_energy_function(self):
        energy_config = {
            'device': self.device,
            'energy_function': self.args.energy_function,
            'mlip_path': self.args.mlip_path,
            'space_groups': self.args.space_groups,
            'z_primes': self.args.z_primes,
            'sg_conditioning': self.args.sg_conditioning,
            'temperature_conditioning': self.args.temperature_conditioning,
            'zp_conditioning': self.args.zp_conditioning,
            'vector_conditioning': getattr(self.args, 'vector_conditioning', False),
            'vector_conditioning_dim': getattr(self.args, 'vector_conditioning_dim', None),
            'embedding_conditioning': getattr(self.args, 'embedding_conditioning', False),
            'embedding_conditioning_dim': getattr(self.args, 'embedding_conditioning_dim', None),
        }
        energy_config.update(self.args.energy_config.__dict__)
        self.energy_function = MolecularCrystal(**energy_config)
        # ONE ATTACHMENT POINT, and nothing else in the trainer changes. The
        # region timer lives on the energy function because log_reward is the
        # single funnel every energy evaluation goes through, so instrumenting
        # it needs no edit to train_step -- the hottest code here, and the one
        # place a profiling feature has no business touching. Disabled by
        # default: `from_config` reads args.profiling, and an absent block is
        # off, so an unprofiled run allocates nothing per call.
        import profiling
        prof = profiling.from_config(self.args, cuda=str(self.device).startswith('cuda'))
        if prof.enabled:
            self.energy_function._region_profiler = prof
            print(f'profiling: region timer ON (cuda={prof.cuda}) -- '
                  f'energy/seconds_gpu will accompany energy/seconds')
        # Built unconditionally so the hot-loop call never needs a None check;
        # disabled is the default and its `step()` returns on the first line.
        self._trace_window = profiling.trace_from_config(
            self.args, cuda=str(self.device).startswith('cuda'),
            tag=str(getattr(self.args, 'run_name', 'run')))

    def tb_z_source(self, group: str) -> str:
        """
        Per-branch Z source, read from {group}_loss_coeffs.tb_z_source.

        It lives with the loss coefficients, not in condition_log_z, because it
        is a property of what a branch's loss DOES -- get_tb_loss substitutes a
        detached per-condition target under 'persistent' -- and because that is
        the one place a protocol stage can override per branch. The conditional
        route needs 'persistent' where the unconditional route needs 'learned',
        and as a global it was a hand-edit every mode switch silently required.

        Falls back to 'learned', which is the unconditional behaviour.
        """
        c = getattr(self.args, f'{group}_loss_coeffs', None)
        if c is None:
            return 'learned'
        return getattr(c, 'tb_z_source', 'learned')

    def mode_repeats(self, mode: str) -> int:
        """
        Per-mode K-tiling factor, read from {mode}_loss_coeffs.repeats so it
        can be phase-scheduled like any other coefficient (e.g. bwd repeats
        > 1 only during the MLE/TBC warm-start stage, where K same-terminal
        rollouts define the exact-MLE and consistency objectives, and 1
        everywhere else so no mode pays the K-times batch tiling for losses
        that don't need it). Falls back to the legacy global top-level
        `repeats` (exactly the old behavior) for configs that predate the
        per-mode key, then to 1. Schedules interpolate as floats, so round
        and floor at 1 here.
        """
        coeffs = getattr(self.args, f'{mode}_loss_coeffs', None)
        r = getattr(coeffs, 'repeats', None) if coeffs is not None else None
        if r is None:
            r = getattr(self.args, 'repeats', 1)
        return max(1, int(round(float(r))))

    def init_condition_log_z(self):
        """
        Persistent per-condition empirical log Z tracker (buffer.py's
        ConditionLogZTracker) -- decoupled from prior/replay/anchor buffers
        entirely, keyed by the immutable condition_id computed in
        energy_function.condition_samples(). init_gfn() (called before this,
        in train()) may already have reloaded one from a checkpoint via
        checkpointer.load_full() -- don't clobber it with a fresh table sized
        off this run's (possibly different) mol_dataset. condition_log_z
        config section is optional -- missing fields (or the whole section)
        fall back to ConditionLogZTracker's own defaults.
        """
        if hasattr(self, 'condition_log_z'):
            return

        cfg = getattr(self.args, 'condition_log_z', None)
        min_visits = getattr(cfg, 'min_visits', 20) if cfg is not None else 20
        # ONE definition of the default, in buffer.py -- a second literal here is
        # how a fresh run and a resumed run come to disagree about the decay.
        half_life_visits = (getattr(cfg, 'half_life_visits', DEFAULT_HALF_LIFE_VISITS)
                            if cfg is not None else DEFAULT_HALF_LIFE_VISITS)
        trim_frac = getattr(cfg, 'trim_frac', 0.1) if cfg is not None else 0.1
        max_batch_weight = getattr(cfg, 'max_batch_weight', 200.0) if cfg is not None else 200.0
        discovery_half_life_steps = getattr(cfg, 'discovery_half_life_steps', 200.0) if cfg is not None else 200.0
        self.condition_log_z = ConditionLogZTracker(
            library_size=self.energy_function.condition_library_size,
            min_visits=min_visits,
            half_life_visits=half_life_visits,
            trim_frac=trim_frac,
            max_batch_weight=max_batch_weight,
            discovery_half_life_steps=discovery_half_life_steps,
            # FIXED reference beta for the tracker's z_grad ruler: the BASE fwd
            # coefficient, never a stage override, so the ruler reads the same in
            # every stage (see ConditionLogZTracker.clip_beta)
            clip_beta=getattr(self.args.fwd_loss_coeffs, 'beta', 10.0),
        )

    def bootstrap_log_z(self, max_steps: int = 1000, lr_ramp_steps: int = 200,
                        holdout_frac: float = 0.1, min_conditions_for_holdout: int = 50,
                        lr: Optional[float] = None, target_rms: float = 0.5,
                        max_attempts: int = 10, lr_restart_factor: float = 0.3,
                        coverage_quantile: float = 0.99, coverage_tol: float = 2.0,
                        n_eval_batches: Optional[int] = None, seed: int = 0,
                        train_conditioner: bool = False):
        """
        Supervised fit of the flow head's Z(c) onto condition_log_z's ema_logw:

            min_theta  sum_c  trust_c * MSE(Z(c; theta), ema_logw(c))

        No rollouts and no reward calls -- draws conditions like fwd_train_step
        but only needs get_condition_embedding() + flow_model(), so it converges
        in seconds. Trains flow_model through a FRESH local Adam, not
        self.optimizers['flow'], so it starts from clean moments.

        Runs once at the warm-start stage's exit, to put the flow head somewhere
        reasonable before rollout-driven training resumes.

        trust_c = eff_c / (eff_c + min_visits), from the tracker's time-decayed,
        ESS-capped effective_count. Normalised to mean 1 over the trustworthy
        set, so target_rms and the coverage gate keep an unweighted calibration.

        ⚠ Regresses onto ema_logw, NEVER ema_log_z_emp: this runs in exactly the
        cold-start regime where ema_log_z_emp can reach billions of nats.
        ema_logw is rank-trimmed and evidence-capped.

        Structure -- one simple run, wrapped in a restart loop gated on a
        per-condition test rather than a batch mean:

          run         LR decays linearly lr -> lr/100 over lr_ramp_steps, flat
                      after. Early stop at target_rms (0.5 nats). best_state
                      snapshots the trough and is restored at the end; a
                      non-finite loss breaks straight to that restore.
          holdout     with >= min_conditions_for_holdout (50) trustworthy
                      conditions, holdout_frac (0.1) are drawn but never
                      backpropped and the early stop watches THEIR error.
          acceptance  coverage_quantile (0.99) of per-condition abs error must
                      be within coverage_tol (2.0 nats). This is the real gate.
          restart     on failure, re-seed and re-init at lr * lr_restart_factor
                      ** attempt, up to max_attempts (10). Best-by-coverage
                      attempt is kept either way.

        train_conditioner also fits conditions_embedding_model. ⚠ Only valid
        when the preceding stage ran scramble_conditions -- the trunk was then
        trained to ignore the embedding, so reshaping it cannot move the policy.
        After a stage that DID train the conditioner, use plain 'bootstrap_z'.
        """
        if not hasattr(self, 'condition_log_z'):
            return

        conditioner = self.gfn_model.conditions_embedding_model if train_conditioner else None
        cond_entry_state = deepcopy(conditioner.state_dict()) if conditioner is not None else None

        def _snapshot():
            state = {'flow': deepcopy(self.gfn_model.flow_model.state_dict())}
            if conditioner is not None:
                state['cond'] = deepcopy(conditioner.state_dict())
            return state

        def _restore(state):
            self.gfn_model.flow_model.load_state_dict(state['flow'])
            if conditioner is not None and 'cond' in state:
                conditioner.load_state_dict(state['cond'])

        lr = 0.05 if lr is None else lr
        tracker = self.condition_log_z
        valid_mask = (tracker.count >= tracker.min_visits) & (~torch.isnan(tracker.ema_logw))
        valid_ids = torch.nonzero(valid_mask, as_tuple=True)[0]
        target_mse = target_rms ** 2

        # evidence-confidence weights (see docstring). Fallback to uniform when the
        # tracker's evidence is entirely stale (all-decayed effective_count) rather
        # than silently zeroing the fit's gradient.
        eff = tracker.effective_count.float().clamp(min=0.0)
        trust = eff / (eff + float(tracker.min_visits))
        mean_trust = trust[valid_mask].mean() if valid_ids.numel() > 0 else torch.tensor(0.0)
        if mean_trust.item() > 1e-6:
            trust_norm = trust / mean_trust
        else:
            trust = torch.ones_like(trust)
            trust_norm = torch.ones_like(trust)

        # holdout split drawn ONCE, up front, so every attempt trains and is judged
        # on the same partition -- keeps best-by-coverage an apples-to-apples choice
        # across attempts rather than each getting a different lucky/unlucky split
        is_val = None
        if valid_ids.numel() >= min_conditions_for_holdout:
            n_val = max(1, int(round(valid_ids.numel() * holdout_frac)))
            perm = valid_ids[torch.randperm(valid_ids.numel())]
            val_ids = perm[:n_val]
            is_val = torch.zeros(tracker.library_size, dtype=torch.bool)
            is_val[val_ids] = True

        # cover the trustworthy set a few times over on average; discrete-condition
        # draws hit each id stochastically, so a handful of passes keeps the tail of
        # the coverage quantile honest without a full deterministic enumeration
        if n_eval_batches is None:
            n_eval_batches = max(20, math.ceil(3 * max(valid_ids.numel(), 1) / int(self.batch_size)))

        def _draw_batch():
            mol_batch = next(self.mol_dataset.loader(int(self.batch_size), mode='graphs'))
            mol_batch = mol_batch.to(self.device)
            mol_batch.orient_molecule(mode='std')
            mol_batch, _, _, _, condition, condition_id = \
                self.energy_function.condition_samples(mol_batch)
            return mol_batch, condition.to(self.device), condition_id

        def _reinit_flow():
            for m in self.gfn_model.flow_model.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()

        def _run_attempt(attempt_lr):
            """One simple fit; returns (best_state, best_monitored_mse, steps)."""
            fit_params = list(self.gfn_model.flow_model.parameters())
            if conditioner is not None:
                fit_params += list(conditioner.parameters())
            optimizer = torch.optim.Adam(fit_params, lr=attempt_lr)
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.01, total_iters=lr_ramp_steps)
            best_loss, best_state, steps_run = float('inf'), None, 0
            for steps_run in range(1, max_steps + 1):
                mol_batch, condition, condition_id = _draw_batch()
                target, mask = tracker.lookup(condition_id)
                target, mask = target.to(self.device), mask.to(self.device)
                if mask.sum() == 0:
                    continue
                w_rows = trust_norm[condition_id.detach().cpu().flatten().long()].to(self.device)

                if is_val is not None:
                    batch_is_val = is_val[condition_id.cpu()].to(self.device)
                    train_mask = mask & ~batch_is_val
                    val_mask = mask & batch_is_val
                else:
                    train_mask = mask
                    val_mask = torch.zeros_like(mask)

                if conditioner is None:
                    # no_grad on the embedding: only flow_model trains here, so backprop
                    # through the (frozen-for-this-purpose) conditioner is wasted compute
                    with torch.no_grad():
                        condition_embedding = self.gfn_model.get_condition_embedding(condition, mol_batch)
                else:  # train_conditioner: the embedding IS part of the fit
                    condition_embedding = self.gfn_model.get_condition_embedding(condition, mol_batch)
                z_pred = self.gfn_model.flow_model(condition_embedding).flatten()

                monitored_loss = None
                if train_mask.sum() > 0:
                    train_loss = (w_rows[train_mask]
                                  * (z_pred[train_mask] - target[train_mask]).pow(2)).mean()
                    optimizer.zero_grad(set_to_none=True)
                    train_loss.backward()
                    optimizer.step()
                    scheduler.step()  # linear decay, fixed schedule
                    if is_val is None:  # no holdout -- fall back to training loss
                        monitored_loss = train_loss.item()

                if val_mask.sum() > 0:
                    with torch.no_grad():
                        # same weighting as the train objective, so early stop /
                        # best_state selection optimize the quantity being trained
                        monitored_loss = (w_rows[val_mask]
                                          * (z_pred[val_mask] - target[val_mask]).pow(2)).mean().item()

                if monitored_loss is None:
                    continue

                # explosion tripwire: only a non-finite loss (not a mere increase)
                # bails, straight to the best_state restore
                if not math.isfinite(monitored_loss):
                    break
                if monitored_loss < best_loss:
                    best_loss = monitored_loss
                    best_state = _snapshot()
                if monitored_loss <= target_mse:
                    break

            if best_state is not None:
                _restore(best_state)
            return best_state, best_loss, steps_run

        @torch.no_grad()
        def _coverage_eval():
            """
            Per-condition acceptance test on the current flow weights: sweep the
            deployment draw, accumulate mean abs error per trustworthy condition_id
            seen, return (quantile error, #conditions covered). Runs in eval mode
            so the number is deterministic (no dropout) -- this is a quality gate,
            not a training signal.
            """
            was_training = self.gfn_model.flow_model.training
            self.gfn_model.flow_model.eval()
            cond_was_training = conditioner is not None and conditioner.training
            if cond_was_training:  # the conditioner is part of the fit -- gate it deterministically too
                conditioner.eval()
            lib = tracker.library_size
            err_sum = torch.zeros(lib)
            err_cnt = torch.zeros(lib)
            for _ in range(n_eval_batches):
                mol_batch, condition, condition_id = _draw_batch()
                target, mask = tracker.lookup(condition_id)
                target = target.to(self.device)
                mask = mask.to(self.device)
                if mask.sum() == 0:
                    continue
                condition_embedding = self.gfn_model.get_condition_embedding(condition, mol_batch)
                z_pred = self.gfn_model.flow_model(condition_embedding).flatten()
                abs_err = (z_pred - target).abs().cpu()
                cid = condition_id.cpu().flatten().long()
                keep = mask.cpu()
                err_sum.index_add_(0, cid[keep], abs_err[keep])
                err_cnt.index_add_(0, cid[keep], torch.ones(int(keep.sum())))
            if was_training:
                self.gfn_model.flow_model.train()
            if cond_was_training:
                conditioner.train()
            seen = err_cnt > 0
            if seen.sum() == 0:
                return float('inf'), 0
            per_id = err_sum[seen] / err_cnt[seen]
            # trust-weighted quantile, consistent with the weighted fit: a condition
            # the regression was told to down-weight must not fail the acceptance
            # gate for missing that same target. Weight-scale invariant, so the
            # unnormalized trust is used; plain quantile fallback if every seen
            # condition carries ~zero trust (nothing meaningful to weight by).
            w_id = trust[seen]
            if w_id.sum() > 1e-6:
                order = torch.argsort(per_id)
                cum = torch.cumsum(w_id[order], dim=0)
                q_idx = int(torch.searchsorted(cum, coverage_quantile * cum[-1]).clamp(max=order.numel() - 1))
                q = per_id[order[q_idx]].item()
            else:
                q = torch.quantile(per_id, coverage_quantile).item()
            return q, int(seen.sum())

        # snapshot global RNG state: the per-attempt seeding below makes restarts
        # reproducible, but must not leak into the outer run's stochastic stream --
        # restore whatever the main training loop was about to draw next
        torch_rng_state = torch.get_rng_state()
        numpy_rng_state = np.random.get_state()

        if train_conditioner:
            print("Z(c) bootstrap: conditioner unfrozen (direct 1->3 route under unconditional prior)")

        best_cov, best_overall_state, best_meta = float('inf'), None, None
        for attempt in range(max_attempts):
            torch.manual_seed(seed + attempt)
            np.random.seed(seed + attempt)
            _reinit_flow()  # genuine fresh start, not the same basin re-drawn
            if conditioner is not None:  # conditioner analog: back to its at-entry weights
                conditioner.load_state_dict(cond_entry_state)
            attempt_lr = lr * (lr_restart_factor ** attempt)

            best_state, best_loss, steps_run = _run_attempt(attempt_lr)
            cov_q, n_cov = _coverage_eval()

            print(f"Z(c) bootstrap attempt {attempt + 1}/{max_attempts}: "
                  f"{steps_run} steps, lr {attempt_lr:.3g}, best monitored MSE {best_loss:.4g}, "
                  f"P{coverage_quantile * 100:.0f} abs err {cov_q:.3g} over {n_cov} conditions "
                  f"(accept <= {coverage_tol:.3g})")

            if cov_q < best_cov:
                best_cov, best_overall_state = cov_q, best_state
                best_meta = (attempt + 1, steps_run, n_cov)
            if cov_q <= coverage_tol:
                break

        torch.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)

        # keep the best-by-coverage attempt regardless of whether any run passed --
        # a marginal fit is still better to hand the next stage than a fresh random head
        if best_overall_state is not None:
            _restore(best_overall_state)
        # keep ema_model's flow head in sync (a no-op when ema_decay is null and
        # ema_model literally is gfn_model, but correct if EMA is ever re-enabled)
        self.ema_model.flow_model.load_state_dict(self.gfn_model.flow_model.state_dict())
        if conditioner is not None:
            self.ema_model.conditions_embedding_model.load_state_dict(conditioner.state_dict())

        passed = best_cov <= coverage_tol
        holdout_msg = f", {int(is_val.sum())} held out" if is_val is not None \
            else f", no holdout ({valid_ids.numel()} conditions)"
        attempt_msg = f" on attempt {best_meta[0]}" if best_meta is not None else ""
        print(f"Z(c) bootstrap: {'ACCEPTED' if passed else 'FAILED'}{attempt_msg} -- "
              f"P{coverage_quantile * 100:.0f} abs err {best_cov:.3g} nats "
              f"(accept <= {coverage_tol:.3g}){holdout_msg}")
        if not passed:
            print("  WARNING: Z(c) bootstrap did not reach the coverage bar; the next stage "
                  "training may be unreliable. Consider more attempts or a lower lr.")

    def weighted_condition_sampling(self, temperature: float = 1.0,
                                    clip_quantile: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Per-row sampling distribution over self.mol_dataset, weighting each
        molecule toward its tracked per-condition QUALITY OF FIT instead of
        sampling uniformly, so badly-fit conditions get drawn more often.
        Mirrors CrystalBuffer._loss_weights (same softmax-over-temperature
        shape, same moderate-fill treatment for not-yet-visited entries), just
        sourced from condition_log_z instead of ema_loss.

        This is the FORWARD half of the sampling-weight pair: forward draws
        pick CONDITIONS, so they weight on a per-condition statistic (each
        condition is judged by how well it is fit); backward draws pick stored
        terminal STATES, so they weight on each row's own ema_loss (each state
        is its own keeper). The signals are deliberately different because the
        unit of choice is different.

        A molecule's tracked statistics are spread over its whole SG/Z' BLOCK
        (condition_id is mixed-radix: mol_id * n_sg*n_zp + sg_local*n_zp +
        zp_local), and training draws sample SG/Z' per batch -- so the block is
        averaged over its VISITED members here. Reading only local index 0, as
        this used to, addressed a condition that need never have been visited
        on a multi-SG/Z' run, leaving the weighting to fall back to the
        unvisited-fill for every molecule. Single-SG/Z' problems (all toys)
        have a one-member block, so their behavior is unchanged.

        Returns None (falls back to uniform sampling) if condition_log_z
        isn't initialized yet, mol_dataset's resident batch has no mol_id
        (predates init_identifiers()), or nothing has been visited yet.
        """
        tracker = getattr(self, 'condition_log_z', None)
        mol_id = getattr(self.mol_dataset.batch, 'mol_id', None)
        if tracker is None or mol_id is None:
            return None

        n_combos = self.energy_function.n_sg * self.energy_function.n_zp
        block = (mol_id.detach().cpu().long().flatten()[:, None] * n_combos
                 + torch.arange(n_combos, dtype=torch.long)[None, :])
        # signal = z_bias_ema^2 + Var(log w), the per-condition mean SQUARED TB
        # residual (level + spread) -- left unrooted because a sampling weight
        # only needs a monotone priority.
        err, mask = tracker.lookup_fit_error(block.flatten())
        err = err.reshape(block.shape)
        mask = mask.reshape(block.shape)
        # per-molecule mean over VISITED block members; molecules with no visited
        # member anywhere in their block fall through to the unvisited-fill below
        n_seen = mask.sum(dim=1)
        err = torch.where(mask, err, torch.zeros_like(err)).sum(dim=1) / n_seen.clamp(min=1)
        mask = n_seen > 0

        if not mask.any():
            return None

        # unvisited conditions take the visited MEAN, never zero: a zero priority
        # is self-reinforcing (never sampled -> never warms -> never sampled)
        fill = err[mask].mean()
        err = torch.where(mask, err, fill)

        # Robust, SCALE-FREE priority. The absolute range is problem- and
        # time-dependent (hundreds-to-thousands of nats here, and it shrinks as
        # training converges + differs per energy function), so a raw softmax
        # over err/temperature would need the temperature retuned constantly,
        # and a single blowup condition -- z_bias_ema can be sent enormous by one
        # degenerate off-policy log_pf (see ConditionLogZTracker.lookup) -- would
        # take all the mass. Instead: take a high quantile as the scale (also
        # excludes blowups from setting it), clip to it, and NORMALIZE by it ->
        # priority in [0, 1]. temperature then acts on [0, 1], so ~0.3-1.0 is
        # sensible on ANY problem and never needs rescaling. If that scale is ~0
        # (no meaningful tail -- everything converged), there's nothing to steer
        # toward, so fall back to uniform. clip_quantile=None keeps the legacy
        # raw-err behavior (temperature then in nats, scale-dependent).
        if clip_quantile is not None:
            cap = torch.quantile(err, clip_quantile)
            if cap <= 1e-6:
                return None
            err = err.clamp(min=0.0, max=cap) / cap

        logits = err / max(temperature, 1e-8)
        logits = logits - logits.max()
        p = torch.softmax(logits, dim=0).double().numpy()
        p = np.clip(p, 1e-8, None)
        p /= p.sum()
        return p

    def _resolve_periodic_centroid_axes(self):
        """
        Which aunit centroid axes may be wrapped. Returns None when the feature is off.

        Which axes are periodic is a property of the space group, so this makes the model
        space-group specific: we require exactly one. Intersecting over several would
        "work", but it would quietly hand back a weaker (possibly empty) wrap instead of
        surfacing that the config asked for something this feature doesn't cover.
        """
        if not getattr(self.args.model, 'periodic_centroids', False):
            return None
        if not self.energy_function.is_crystal:
            raise ValueError("model.periodic_centroids is a molecular-crystal feature, but "
                             f"energy_function={self.args.energy_function!r} is not a crystal")
        if len(self.args.space_groups) != 1:
            raise ValueError(
                "model.periodic_centroids makes the model space-group specific, so it needs "
                f"exactly one entry in space_groups; got {list(self.args.space_groups)}")
        sg = int(self.args.space_groups[0])
        axes = sg_periodic_centroid_axes(sg)
        print(describe(sg))
        if not axes:
            print(f"WARNING: model.periodic_centroids is on but SG{sg} has no full-width "
                  "(auv == 1) aunit axis -- no centroid dim will be wrapped")
        return axes

    def _resolve_dead_latent_rows(self, quiet: bool = False):
        """
        Which latent rows are structurally dead and so held out of the SDE. Returns
        None when the feature is off or does not apply. See decisions.md D33.

        Crystal-only. Toy energies carry `space_groups: [1]` as a PLACEHOLDER, and P1
        has all three aunit centroid axes free, so keying off the space group without
        the is_crystal gate would eventually freeze three dims a toy genuinely uses --
        silently, as lost coverage rather than an error. `resolve_dead_rows` enforces
        the gate; this method also skips the probe, which is crystal machinery.

        Like periodic_centroids this makes the model space-group specific, so the space
        groups present must agree on their dead rows, and the check below compares the
        RESOLVED SETS rather than the crystal systems.

        That distinction is load-bearing, and an earlier version of this comment claimed
        the opposite ("they always agree within a crystal system"), which is FALSE.
        Monoclinic alone carries three different sets, because free aunit axes do not
        follow the crystal system:
            sg 3, 4, 5      -> (3, 5, 7)      2-fold along b: free translation along b
            sg 6, 7, 8, 9   -> (3, 5, 6, 8)   mirror perp b: free translation in a-c
            sg 10 - 15      -> (3, 5)         centrosymmetric: no free axis
        So `space_groups: [4, 14]` is same-system and still ambiguous. Do NOT "simplify"
        this into a crystal-system comparison -- that would silently serve one index set
        to space groups that need different ones.
        """
        if not getattr(self.args.model, 'hold_dead_latent_rows', True):
            return None
        if not self.energy_function.is_crystal:
            if not quiet:
                print(describe_dead_rows(1, max(self.args.z_primes), is_crystal=False))
            return None
        if not len(self.args.space_groups):
            raise ValueError(
                "hold_dead_latent_rows needs at least one entry in space_groups to know "
                "which latent rows are dead; got an empty list")

        zp = max(self.args.z_primes)
        per_sg = {int(sg): resolve_dead_rows(int(sg), is_crystal=True, max_z_prime=zp)
                  for sg in self.args.space_groups}
        distinct = set(per_sg.values())
        if len(distinct) > 1:
            raise ValueError(
                "hold_dead_latent_rows makes the model space-group specific, but the "
                f"configured space_groups disagree about which latent rows are dead: "
                f"{per_sg}. One index set cannot serve them; split the run by crystal "
                f"system, or set model.hold_dead_latent_rows: false to opt out (which "
                f"restores the D33 defect -- prior rows pinned at the canonical value "
                f"while the energy is flat there).")
        rows = distinct.pop() if distinct else ()
        if not quiet:
            print(describe_dead_rows(int(self.args.space_groups[0]), max(self.args.z_primes)))
        return rows

    def _verify_dead_latent_rows(self, crystal_batch, n_probe: int = 8):
        """
        Assert the tabulated dead rows still match what the crystal build actually
        ignores. This is the guard against the exact failure that motivated D33: a
        table drifting away from enforce_crystal_system, unnoticed because nothing
        crashes. Cheap -- one small clone, a dozen forward transforms.

        Probes a CLONE: probe_dead_rows drives latent_to_cell_params repeatedly and
        must not perturb the live prior batch.
        """
        if not self.energy_function.is_crystal:
            return
        if not getattr(self.args.model, 'hold_dead_latent_rows', True):
            return
        sg = int(self.args.space_groups[0])
        try:
            n = min(n_probe, crystal_batch.num_graphs)
            probe_batch = crystal_batch.subsample_new_batch(
                torch.arange(n, device=crystal_batch.device)).clone()
            # NO pose_aunit()/build_unit_cell() here. The probe only drives
            # latent_to_cell_params -> latent_params, which needs cell parameters and
            # nothing built from them. Those two calls were dead weight AND they broke
            # the probe at Z'>1: aunit2ucell assumes a 3-wide centroid, but at Z'=2
            # aunit_centroid is stored FLATTENED as (n, 6), so appending the affine 1
            # gives a 7-vector against a 4x4 operator --
            #   "einsum(): subscript j has size 7 ... previously seen size 4".
            # That raised, was swallowed by the except below, and every Z'>1 run printed
            # "the tabulated rows are UNVERIFIED this run" -- so the ONE runtime guard on
            # the dead-row table was silently absent for exactly the layout with the
            # least other coverage. Found by the sg 9 Z'=2 smoke run, 2026-08-12.
            found = verify_dead_rows(probe_batch, sg, max(self.args.z_primes))
            self._dead_rows_verified = True
            print(f"dead-row probe: latent_to_cell_params confirms rows {found} "
                  f"are ignored for SG{sg} (n={n})")
            # The value the SDE pins these rows to must equal the value latent_params()
            # writes into them, or bwd terminals drawn from the buffer would disagree
            # with fwd terminals. Zero holds for every space group reachable today
            # (pi/2 -> latent 0.0); hexagonal gamma = 2pi/3 does not, and must surface
            # here rather than be silently pinned to the wrong constant.
            if found:
                lat = probe_batch.latent_params()[:, list(found)]
                worst = float(lat.abs().max().item())
                if worst > 1e-6:
                    raise AssertionError(
                        f"SG{sg} dead rows {found} round-trip to a NONZERO canonical "
                        f"latent value (max |v| = {worst:.6g}), but the SDE pins them to "
                        f"0. Resolve dead_latent_values alongside dead_latent_rows and "
                        f"pass it through _build_gfn_config before running this space "
                        f"group -- see decisions.md D33.")
        except AssertionError:
            raise
        except Exception as e:
            # a probe that cannot run is not evidence the table is wrong; say so loudly
            # rather than either crashing the run or implying the check passed
            # Record it as well as printing it. A startup WARNING is exactly what went
            # unread when the probe was broken at Z'>1 for every run of that layout, so
            # the unverified state is now a logged series and can be queried after the
            # fact instead of relying on someone having scrolled the log.
            self._dead_rows_verified = False
            print(f"WARNING: dead-row probe could not run ({type(e).__name__}: {e}); "
                  f"the tabulated rows for SG{sg} are UNVERIFIED this run")

    def _build_gfn_config(self):
        return dict(
            dim=self.energy_function.data_ndim,
            conditions_dim=self.get_conditioning_dim(),
            conditions_type='molecule' if self.args.molecule_conditioning else 'vector',
            periodic_centroid_axes=self._resolve_periodic_centroid_axes(),
            dead_latent_rows=self._resolve_dead_latent_rows(),
            conditional=any([
                self.args.temperature_conditioning,
                self.args.molecule_conditioning,
                self.args.sg_conditioning,
                self.args.zp_conditioning,
                getattr(self.args, 'vector_conditioning', False),
                getattr(self.args, 'embedding_conditioning', False),
            ]),
            device=self.device,
            max_z_prime=max(self.args.z_primes),
            do_periodic_angles=self.energy_function.is_crystal,
            **vars(self.args.model),
        )

    def init_gfn(self):
        reload = False

        if self.args.checkpoint_name is not None:
            reload = True
            reload_path = f'{self.args.checkpoints_dir}/{self.args.checkpoint_name}'
            if getattr(self.args, 'load_weights_only', False):
                print(f"Loading model weights only from checkpoint {reload_path} - "
                      f"optimizers, buffers, and training state start fresh")
                self.checkpointer.load_weights_only(reload_path)
                self.init_schedulers_optimizers()
            else:
                print(f"Loading model from checkpoint {reload_path}")
                self.checkpointer.load_full(reload_path)

        elif self.args.continue_from_checkpoint:
            reload_path = self.checkpointer.find_matching('running')
            if reload_path is not None:
                print(f"Reloading automatically from prior checkpoint {reload_path}")
                reload = True
                self.checkpointer.load_full(reload_path)

        if not reload:
            self.gfn_config = self._build_gfn_config()
            self.gfn_model = GFN(**self.gfn_config).to(self.device)
            self.ema_model = deepcopy(self.gfn_model)
            self.init_schedulers_optimizers()

        # runtime flag like compile_policy, set on both fresh-build and reload
        # paths; deliberately not part of gfn_config (checkpoints/problem
        # hashing unaffected)
        for model in (self.gfn_model, self.ema_model):
            model.traj_checkpoint = bool(getattr(self.args, 'traj_checkpoint', False))

        self.maybe_compile_policy()

    def maybe_compile_policy(self):
        """
        torch.compile the dense per-step trunk of the policy. The trajectory
        loop launches many tiny kernels (T steps x several small MLPs x 3
        fused branches), so launch overhead caps GPU utilization even at large
        batch; inductor's kernel fusion (MLP + layernorm + gelu collapse to a
        few kernels) recovers most of the launch-count reduction.

        Deliberately DEFAULT mode, not 'reduce-overhead': CUDA-graph capture
        gives each compiled module STATIC output buffers that every
        re-invocation overwrites, and the trajectory loop re-invokes the same
        modules T times per rollout while earlier outputs are still referenced
        (and autograd retains all T iterations' activations until backward) --
        crashes at best ("accessing tensor output of CUDAGraphs that has been
        overwritten"), silent corruption at worst. Revisit cudagraphs only
        with cudagraph_mark_step_begin/cloning discipline around the loop.

        compile_policy: false (default) | true | 'auto'. 'auto' enables only on
        Linux+CUDA -- inductor does not support CUDA on native Windows, so dev
        boxes stay eager while cluster runs pick it up. The conditioner is
        deliberately NOT compiled (the molecule-GNN variant has dynamic node
        counts). Compilation happens lazily at first forward, so backend
        failures surface there -- suppress_errors degrades those to eager with
        a warning instead of killing the run. NB every distinct batch size is a
        recompile: run this with a static batch OR grow_batch_size + jump-mode
        growth (batch_growth_interval > 0 -- ~8 shapes total, within the raised
        recompile limit below); legacy continuous 1.01x growth blows the limit
        immediately and silently reverts to eager. Uses in-place
        nn.Module.compile(), so state_dict keys (and therefore checkpoints) are
        unaffected.
        """
        setting = getattr(self.args, 'compile_policy', False)
        if setting == 'auto':
            import platform
            enable = platform.system() == 'Linux' and torch.cuda.is_available()
        else:
            enable = bool(setting)
        if not enable:
            return

        trunk = ('t_model', 's_model', 'forward_policy', 'backward_policy', 'flow_model')
        try:
            # NB "as _dynamo": a bare `import torch._dynamo` would bind `torch`
            # as a LOCAL for this whole function, making the module-level torch
            # unreachable at the is_available() call above (UnboundLocalError
            # -- and only on Linux, since the `and` short-circuits elsewhere)
            import torch._dynamo as _dynamo
            _dynamo.config.suppress_errors = True
            # jump-mode batch growth is ~8 shapes end to end, exactly at
            # dynamo's default recompile limit of 8 -- give it headroom so the
            # top rungs don't silently fall back to eager
            _dynamo.config.cache_size_limit = 24
            # DONATED BUFFERS MUST BE OFF WHILE ANYTHING TAKES A SECOND BACKWARD.
            # AOTAutograd donates (frees-and-reuses) intermediate buffers on the
            # assumption each compiled backward runs exactly once, and then
            # HARD-RAISES on any backward with retain_graph=True/create_graph=True.
            # _log_fused_gradient_geometry does exactly that -- one
            # torch.autograd.grad(retain_graph=True) per active branch -- so with
            # the trunk compiled, the first armed fused step kills the run:
            #   "This backward function was compiled with non-empty donated
            #    buffers which requires create_graph=False and retain_graph=False"
            # Measured on the cluster 2026-08-16 (a100_stab_aug16 f3, step 320,
            # ~50 steps after the first stage transition, i.e. the first armed
            # fused step). It cannot reproduce on the dev box: compile_policy
            # 'auto' resolves OFF on native Windows, so this whole failure mode
            # is invisible locally and every local shakeout passed.
            # Set here rather than at the diagnostic, because the choice is baked
            # in when AOTAutograd traces -- flipping it after first forward is too
            # late. Costs some activation-memory reuse; that is the price of
            # keeping a diagnostic that reads gradients.
            import torch._functorch.config as _functorch_config
            _functorch_config.donated_buffer = False
            for model in (self.gfn_model, self.ema_model):
                for name in trunk:
                    mod = getattr(model, name, None)
                    if isinstance(mod, torch.nn.Module):
                        mod.compile()  # default mode -- see docstring for why not reduce-overhead
        except Exception as e:
            print(f"compile_policy: torch.compile unavailable here ({e}); continuing eager")
            return
        print(f"compile_policy: trunk {trunk} compiled (default mode, lazy on first forward)")

    def init_schedulers_optimizers(self):
        """
        (Re)build every optimizer from scratch. Called at startup and again
        at each stage transition (protocol.StageProtocol.advance): each stage
        optimizes a different loss surface, so Adam moments must not carry
        across the boundary.

        No LR scheduler objects here any more -- the AdaptiveLRController
        (controller.py) sets every param group's LR directly, every tick
        (step_lr_schedule -> lr_controller.step -> _apply_lrs), so there is
        nothing left to build or step. init_policy_lrs/init_flow_lr below
        still matter for exactly the first train_step of a (re)build: it runs
        before step_lr_schedule's first call fires, so the optimizer needs a
        safe (warmup-start) initial value to construct with.
        """
        init_flow_lr = self.args.lr_flow
        init_policy_lrs = {'fwd': self.args.lr_policy / self.args.lr_warmup_ratio,
                           'bwd': self.args.lr_back / self.args.lr_warmup_ratio,
                           'replay': self.args.lr_replay / self.args.lr_warmup_ratio,
                           'fused': self.args.lr_fused / self.args.lr_warmup_ratio}

        """
        Initialize Optimizers
        """

        def get_policy_params(gfn_model):
            plist = [{'params': gfn_model.t_model.parameters()},
                     {'params': gfn_model.s_model.parameters()},
                     {'params': gfn_model.forward_policy.parameters()},
                     {'params': gfn_model.backward_policy.parameters()},
                     ]
            if gfn_model.conditional:
                plist += [{'params': gfn_model.conditions_embedding_model.parameters()}]
                # flow_model gets its own optimizer (below) unconditionally now, so it's
                # deliberately NOT folded in here -- Z-head training needs to run at its
                # own LR/momentum, decoupled from whatever the policy optimizers are doing.

            return plist

        self.optimizers = {}
        weight_decay = self.args.weight_decay if self.args.use_weight_decay else 0
        # FUSED ADAM ON CUDA. The default (foreach) path computes its bias
        # corrections in Python -- `1 - beta1 ** _get_value(step)` per parameter
        # tensor, and `_get_value` is `.item()` -- so it performs TWO
        # device->host synchronisations per parameter tensor per optimizer step
        # (~154 here), each one blocking the host until the device drains.
        # Measured 2026-08-19 by patching Tensor.item around a real run:
        # adam.py:755/758 was 70% of the Python-visible syncs. Keep the scale
        # honest, though -- that is ~1.3% of the ~11.6k aten::item calls the
        # torch.profiler window counted per step, so this removes a real
        # mechanism, not most of the cost.
        # NOT bit-identical to the foreach path, and adopted on that
        # understanding (user, 2026-08-19). CUDA only: `fused=True` raises on
        # CPU params, so a CPU run silently keeps the default path.
        # MXT_FUSED_ADAM=0 forces the old foreach path. Present so the change
        # is A/B-able end to end: an isolated timing of optimizer.step() cannot
        # show what removing a host sync buys, because it synchronises anyway --
        # the cost of a sync is the NEXT step's work not being queued yet.
        fused_ok = (str(getattr(self, 'device', 'cpu')).startswith('cuda')
                    and os.environ.get('MXT_FUSED_ADAM', '1') != '0')
        adam_kw = {'fused': True} if fused_ok else {}
        print(f'optimizers: Adam fused={fused_ok}')
        # the turn-taking policy optimizers are identical up to their LR
        for mode in ('fwd', 'bwd', 'replay'):
            self.optimizers[mode] = torch.optim.Adam(get_policy_params(self.gfn_model), init_policy_lrs[mode],
                                                     weight_decay=weight_decay, **adam_kw)
        # a fused stage fires fwd/bwd/replay in one backward(), so -- unlike the
        # turn-taking fwd/bwd/replay optimizers, which piggyback on optimizers['flow']
        # via step_loss's non-fused branch -- the flow (Z head) params must ride the
        # fused optimizer directly or they'd never get stepped on a fused step. They go
        # in as their OWN param group at init_flow_lr (not the policy's fused init LR),
        # and the fused schedulers below leave that group's LR flat, so Z keeps its
        # dedicated, decoupled learning rate rather than inheriting the policy warmup/
        # anneal. On a fused step where the fwd sub-loss (the only Z-training branch) is
        # skipped, these params simply get grad=None and Adam skips them -- no spurious
        # update.
        self.optimizers['fused'] = torch.optim.Adam(
            get_policy_params(self.gfn_model) + [{'params': self.gfn_model.flow_model.parameters(),
                                                  'lr': init_flow_lr}],
            init_policy_lrs['fused'], weight_decay=weight_decay, **adam_kw)
        flow_params = self.gfn_model.flow_model.parameters()
        self.optimizers['flow'] = torch.optim.Adam(flow_params, init_flow_lr,
                                                  weight_decay=weight_decay, **adam_kw)

        # Two-point step probe (docs/to_do_rebuild.md A3/A4b) -- SENSOR ONLY, no
        # actuation. Built over the same param list the fused optimizer's policy
        # groups cover and NOT the flow head, per decision D26 option (b): the Z
        # head is LR-pinned separately, so folding it in would make alpha* rate a
        # composite step the servo would not control. Absent config block =>
        # disabled, so this is inert for every existing config.
        # The block lives under `adaptive_lr` -- it parameterises one of the LR
        # sensors, so it belongs with the rest of the LR machinery rather than at
        # top level. The old top-level spelling is retired (utils._RETIRED_KEYS),
        # so a config still carrying it fails at load rather than silently
        # falling through to the defaults below.
        sp_cfg = getattr(getattr(self.args, 'adaptive_lr', None),
                         'ray_calibration', None)
        # ONE list, built once, shared by both sensors. `get_policy_params` is a
        # local of this method, so the hypergradient sensor cannot call it later
        # -- and it must snapshot exactly what the ray probe does (policy only,
        # decision D26b) or the two sensors would disagree about what they are
        # measuring.
        self._hyper_param_cache = [
            p for g in get_policy_params(self.gfn_model) for p in g['params']]
        self._hyper_prev_step = None
        # ENABLED IS DERIVED, not configured. A stage declaring
        # `lr_sensor: {kind: ray}` IS the switch; a separate `ray_calibration.
        # enabled` was a second mechanism for the same decision, and the two could
        # disagree. Note the asymmetry that gave it away: `hyper` has no block at
        # all and declares itself inline at the stage.
        #
        # Defaulting this to False on a missing key would silently kill the probe,
        # so it is computed from the protocol rather than read.
        self.ray_cal = RayCalibration(
            self._hyper_param_cache,
            alphas=tuple(getattr(sp_cfg, 'alphas', (0.0, 1.0, 2.0, 4.0, 8.0))),
            n_sub=int(getattr(sp_cfg, 'n_sub', 8)),
            period=int(getattr(sp_cfg, 'period', 500)),
            enabled=bool(self._ray_askers()),
        )
        # Announced HERE and not in LRController.__init__: the controller is built
        # before the calibrator exists, so it cannot describe its own sensor there.
        self.lr_controller.announce()

    def init_prior_dataset(self):

        prior_data = torch.load(self.args.prior_path, weights_only=False)
        prior = prior_data['equalized_prior']
        prior['smiles'] = None
        # identifier is NOT wiped (unlike smiles) -- init_identifiers() needs it
        # to build the mol_id registry, and since backward training draws
        # directly from this prior_dataset, an identifier-less prior can never
        # be matched back to the right condition_id (see init_identifiers()'s
        # docstring)
        # A pickled .pt file may carry stale/differently-shaped versions of the
        # lazily-built space-group caches (e.g. from an older codebase or a
        # dataset covering fewer space groups). Strip them so they rebuild fresh
        # and consistent with every other batch in this run. See
        # buffer.strip_lazy_sg_caches for details.
        strip_lazy_sg_caches(prior)
        if 'thermal_scaling_factor' in prior_data:
            # SILENT UNIT CHANGE, be loud about it: the dataset's ELJ->UMA thermal
            # calibration REPLACES the config's lj_coeff for the whole run, so every
            # energy is scaled by this factor and the effective sampling temperature
            # is energy_config.temperature / factor (mipcas elj: 0.3636 -> kT 2.5
            # reads as ~6.9 in raw elj units). Toy priors don't carry the key, so
            # toy runs keep the config value -- a systematic toy-vs-physical unit
            # difference. Applied BEFORE the re-analysis pass below so every
            # generator_energy call this run, init included, uses one coefficient
            # (it used to apply after, leaving the re-analyzed gfn_energy stamps in
            # config units).
            self.energy_function.lj_coeff = prior_data['thermal_scaling_factor']
        if True:  # not hasattr(prior, self.args.energy_function):
            print("Re-analyzing prior energies")
            prior = prior.to(self.device)
            # NO_GRAD, explicitly. keep_grads defaults False but that only DETACHES
            # the output -- it does not stop the graph being built, and fairchem's
            # _run_inference uses nullcontext (not no_grad) whenever direct_forces is
            # False, which is what this predictor sets. So without this the whole
            # ~176k-row scoring pass built an autograd graph nothing ever
            # differentiates, holding activations for the entire scan. That is the
            # pass whose MLIP chunk size collapsed to 144 on the uma arm.
            with torch.no_grad():
                energy, prior = self.energy_function.batched_analyze_crystal_batch(
                    prior.latent_params(),
                    prior,
                    self.args.energy_config.temperature * torch.ones((prior.num_graphs), dtype=torch.float32,
                                                                     device=self.device),
                    return_batch=True,
                    internal_oom_recovery=True,
                    # one-off pass over the whole prior dataset at init -- prefer the adaptive, self-healing chunked path over a hard crash, regardless of the training-time flag
                )

            # HAND THE CARD BACK BEFORE TRAINING STARTS. This pass is the largest
            # allocation the process ever makes -- the whole prior scored through the
            # MLIP at supercell_size 10 -- and its blocks are shaped like supercell
            # neighbour lists, which a T-step MLP rollout can never reuse. They are
            # not leaked (the allocator holds them as free), but cuda_memory_fraction
            # is a HARD cap (set_per_process_memory_fraction, init above), so cached
            # blocks the run cannot reuse still count against it and the first large
            # training allocation can OOM behind them.
            #
            # ONE SHOT, HERE, deliberately: the same cleanup inside
            # batched_analyze_crystal_batch is commented out because that function runs
            # every training step on the uma/mace route, where empty_cache() costs a
            # device sync per call. At init it costs milliseconds, once.
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

        # D33: confirm the tabulated dead latent rows still match what the crystal build
        # actually discards, now that a real batch with physical cell parameters exists.
        # Must run on real structures: a degenerate batch (a == b) cannot reveal the
        # tetragonal length constraint and would misreport a live row as dead.
        self._verify_dead_latent_rows(prior)

        self.prior_dataset = CrystalBuffer(prior,
                                           device=self.buffer_device,
                                           **self._buffer_kwargs(),
                                           x_fn=None,  # 'latent_params',
                                           y_fn=self._buffer_y_fn(),
                                           exclude_keys=BULKY_ATTR_EXCLUDE_KEYS,
                                           )

        # The frozen prior model is named EXPLICITLY or it is not loaded. The
        # old `reuse_prior` auto-discovery (this run identity's own *_prior.pt,
        # then find_shared_prior over any matching problem_def) is deleted: it
        # made "which prior did this arm train against" a function of what
        # happened to be on disk, so a battery silently inherited a prior
        # trained under a different scoring rule, and every generator script in
        # the tree had to set `reuse_prior: false` defensively to stop it. A
        # stage that needs a prior now either gets a path or trains one.
        prior_path = None
        if self.args.prior_model_name is not None:
            prior_path = f'{self.args.checkpoints_dir}/{self.args.prior_model_name}'
        if prior_path is not None:
            checkpoint = torch.load(prior_path, map_location=self.device, weights_only=False)
            # a prior model from a different problem wouldn't crash, but it would
            # silently grow the prior buffer with samples from the wrong target.
            # The conditioning flags are exempt: the prior is a frozen
            # sampling-only object, rebuilt below from its OWN stored
            # gfn_config, so it may be any architecture/conditioning (an
            # unconditional model just ignores the condition tensors it's
            # handed at sampling time) -- only the TARGET has to match.
            self.checkpointer.assert_problem_match(checkpoint, prior_path, 'prior_model_name',
                                                   ignore_keys=('mol_cond', 'temp_cond', 'vec_cond'))
            # the stored config stamps the device it was trained on; GFN uses
            # its device arg internally at sampling time, so override it or a
            # cross-device load breaks past .to()
            gfn_config = {**checkpoint['gfn_config'], 'device': self.device}
            self.prior_model = GFN(**gfn_config).to(self.device)
            self.prior_model.load_state_dict(checkpoint['model_eval'])
            self.prior_model.eval()
            # sample this reused prior at ITS OWN training T, not the consumer's
            # eval_T (None on pre-2026-07-23 priors -> sample_from_prior falls
            # back to the run's integrator.T)
            self.prior_train_T = checkpoint.get('train_T')
            # NB: grow_prior_buffer() is deliberately NOT called here. It samples
            # from mol_dataset, whose batch only gains mol_id in init_identifiers()
            # (which runs after this). Growing here would append a mol_id-less
            # sample to a checkpoint-restored prior_buffer that already carries
            # mol_id, and append_batch would reject the key mismatch. train()
            # calls grow_prior_buffer() once init_identifiers() has run.

    @staticmethod
    def _load_condition_file(path):
        """Condition sets are saved as {'prior': batch} (see generate_toy_prior);
        unwrap to the batch. Shared by molecules_path and test_molecules_path -- the
        test branch used to hand the raw dict to CrystalBuffer, which fails on
        data.max_z_prime, so test_molecules_path could never load."""
        data = torch.load(path, weights_only=False)
        if isinstance(data, dict):
            for key, value in data.items():
                if key == 'prior':
                    return value
        return data

    def init_mol_dataset(self):
        self.mol_dataset = CrystalBuffer(self._load_condition_file(self.args.molecules_path),
                                         device=self.buffer_device,
                                         **self._buffer_kwargs(),
                                         exclude_keys=BULKY_ATTR_EXCLUDE_KEYS)

        if self.args.test_molecules_path is not None:
            self.test_mol_dataset = CrystalBuffer(self._load_condition_file(self.args.test_molecules_path),
                                                  device=self.buffer_device,
                                                  **self._buffer_kwargs(),
                                                  exclude_keys=BULKY_ATTR_EXCLUDE_KEYS)
        else:
            self.test_mol_dataset = None

    def init_identifiers(self):
        """
        Assigns every distinct crystal_batch.identifier (a molecule for
        physical runs, or a standalone condition for toy runs) a stable
        integer `mol_id`, and attaches it to mol_dataset/prior_dataset/
        test_mol_dataset's resident batches -- condition_samples() reads it
        straight off the batch, the same way sg_ind/z_prime already ride
        along, to build condition_id.

        Backward training draws directly from prior_dataset (a separately
        loaded file, not sourced from mol_dataset), so the registry spans
        both -- otherwise samples drawn from prior_dataset would have no
        way to match the condition_id of the molecule they actually are.
        This assumes molecules_path/prior_path are prepared with matching
        .identifier fields for the same underlying molecule/condition
        (prep-side responsibility, not resolved here via fingerprint or
        geometric matching).

        Must run after mol_dataset/prior_dataset/test_mol_dataset are all
        loaded (registry needs every identifier upfront) and before
        init_condition_log_z() (which sizes the tracker table off
        energy_function.condition_library_size == n_molecules * n_sg * n_zp).
        """
        datasets = [self.mol_dataset, self.prior_dataset]
        if self.test_mol_dataset is not None:
            datasets.append(self.test_mol_dataset)

        identifiers = set()
        for dataset in datasets:
            if hasattr(dataset.batch, 'identifier'):
                identifiers.update(dataset.batch.identifier)

        self.identifier_registry = {ident: i for i, ident in enumerate(sorted(identifiers))}

        for dataset in datasets:
            if hasattr(dataset.batch, 'identifier'):
                mol_id = torch.tensor(
                    [self.identifier_registry[ident] for ident in dataset.batch.identifier],
                    dtype=torch.long, device=dataset.batch.device)  # match the (possibly GPU-resident) store
                dataset.batch.add_graph_attr(mol_id, 'mol_id')

        self.energy_function.set_n_molecules(max(len(self.identifier_registry), 1))

    def train(self):
        with (wandb.init(project="GFN Energy",
                         config=flatten_wandb_params(self.args),
                         name=self.run_name,
                         tags=[self.args.tag])):
            self.times['initialization_start'] = time()

            self.vram_ledger('baseline')

            # Reward init
            self.init_energy_function()
            self.vram_ledger('energy fn (MLIP loaded)')

            # Model Init
            self.init_gfn()
            self.vram_ledger('gfn model')

            # data init -- init_identifiers() needs mol_dataset/prior_dataset/test_mol_dataset
            # all loaded first (it builds one registry spanning all of them), and must itself
            # run before init_condition_log_z() (which preallocates the tracker table off
            # energy_function.condition_library_size, set by init_identifiers())
            self.init_mol_dataset()
            self.vram_ledger('mol_dataset')
            self.init_prior_dataset()
            # THE ONE TO WATCH: this is the whole-prior MLIP re-analysis. If `cached`
            # jumps here and never comes back down, the startup energy evaluations are
            # holding the card and every later OOM is downstream of this line.
            self.vram_ledger('prior_dataset (MLIP scan)')
            self.init_identifiers()
            # grow_prior_buffer() must run after init_identifiers() so the freshly
            # sampled batch inherits mol_id (via mol_dataset's registry) and matches
            # a checkpoint-restored prior_buffer. Kept before init_condition_log_z()
            # so the prior-sampling pass still sees no condition_log_z (its updates
            # are hasattr-gated), preserving the previous init-time behaviour.
            # prior_buffer seed (like grow_prior_buffer, needs init_identifiers()'s
            # mol_id on prior_dataset.batch; no sampling/energy involved). Runs
            # first so grow_prior_buffer's top-up sees the seeded fill level.
            self.init_prior_buffer_seed()
            if self._has_prior_sampler():
                self.grow_prior_buffer()
            self.init_condition_log_z()
            self.init_anchor_buffer_seed()
            # buffer_device: cuda puts prior/replay/anchor on the card. A big jump in
            # LIVE (not cached) here is the buffers, not a leak -- a different fix.
            self.vram_ledger('buffers seeded')

            # pin the starting stage on a fresh run and walk any skip_if chain
            # (e.g. a prior loaded by path skips the MLE warm-start stage);
            # resumed runs stay wherever their checkpoint says
            self.protocol.begin()

            self.times['initialization_end'] = time()
            # the number every training allocation has to fit under, printed BEFORE
            # the first step so a launch that is already doomed says so at step 0
            self.vram_ledger('READY TO TRAIN')
            if torch.cuda.is_available():
                _total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                _cap = float(getattr(self.args, 'cuda_memory_fraction', 1.0) or 1.0) * _total
                _res = torch.cuda.memory_reserved() / (1024 ** 2)
                print(f"vram: cuda_memory_fraction {getattr(self.args, 'cuda_memory_fraction', None)} "
                      f"x {_total:.0f} MiB = {_cap:.0f} MiB hard cap; "
                      f"{_cap - _res:.0f} MiB of it left for training allocations")

            # no wandb.watch(log='gradients'): 110 per-tensor histograms answered
            # 'is gradient reaching every submodel / is one cooling off' badly and
            # only every few thousand steps. gradnorm/* in ten_step_reporting is
            # the same question as six plottable, gateable scalars at train cadence
            # (see _submodel_grad_norms).

            self.gfn_model.train()
            self.set_detect_anomaly(do_anomaly_detection=self.args.anomaly_detection)
            init_step = self.step_ind * 1
            for self.step_ind in trange(init_step, self.args.epochs + 1):
                # THE ONLY PROFILING CALL IN THE HOT LOOP, and it is one boolean
                # compare unless a trace window is configured AND still open.
                # Once the window has written, `done` latches and this never
                # touches the profiler again for the rest of the run.
                self._trace_window.step(self.step_ind)
                current_loss = None
                metrics = {}
                if self.step_ind % 10 == 0:
                    self.set_loss_coeffs()
                    self.set_energy_coeffs()

                step_type = self.train_logic(self.step_ind)
                # captured BEFORE the step: an OOM slashes self.batch_size in
                # handle_train_epoch_error, and the throughput denominator wants
                # the size that was actually attempted at that cost
                attempted_batch = self.batch_size
                self.times['train_step_start'] = time()
                self._z_cal_rollouts = 0
                # energy seconds spent INSIDE this step, isolated by a before/after
                # snapshot. The raw counter on the energy function accumulates EVERY
                # call -- eval sampling, anchor screening, prior churn -- none of
                # which is inside the step timing, so dividing the raw total by the
                # step window gave energy/frac_of_step = 1.48 on the first real run.
                # A "fraction" above 1 is the metric confessing it is measuring two
                # different denominators.
                energy_s_before = getattr(self.energy_function, 'energy_seconds', 0.0)
                try:
                    current_loss = self.train_step(step_type)
                except FrozenTrainingState:
                    # FrozenTrainingState subclasses RuntimeError, so without this it
                    # would be caught below and treated as an OOM -- the batch would be
                    # slashed and the run would carry on in the very state the exception
                    # exists to end. It means UNRECOVERABLE; let it out.
                    raise
                except (RuntimeError, ValueError) as e:  # if we do hit OOM, slash the batch size
                    self.handle_train_epoch_error(e, step_type)
                # interspersed Z-only calibration steps (z_calibration_tick);
                # inside the timing window so their cost shows in step_dt
                if current_loss is not None:
                    self.z_calibration_tick(step_type)
                self.times['train_step_end'] = time()
                step_dt = self.times['train_step_end'] - self.times['train_step_start']
                self._probe_max('step_time_max10', step_dt)
                if not hasattr(self, '_recent_step_times'):
                    self._recent_step_times = deque(maxlen=64)
                    self._recent_step_work = deque(maxlen=64)
                self._recent_step_times.append(step_dt)  # feeds the sizer's rung median
                # WORK, not just the training batch: z_calibration_tick runs
                # self._z_cal_rollouts extra full-batch rollouts inside the timing
                # window above, and its rate is frequency-modulated by a sensor that
                # decays on its own clock. Charging the rung only for attempted_batch
                # while timing all of it made step_dt fall at CONSTANT batch as the
                # sensor converged, and the knee credited that decay to its own
                # growth -- which is what walked prod0810 to batch 12226.
                self._recent_step_work.append(attempted_batch * (1 + self._z_cal_rollouts))
                self._throughput['samples'] += attempted_batch
                self._throughput['seconds'] += step_dt
                self._throughput['energy_seconds'] += max(
                    0.0, getattr(self.energy_function, 'energy_seconds', 0.0) - energy_s_before)
                # occupancy sampled on a WALL-CLOCK cadence (the call is a cheap
                # no-op between periods), so the trailing windows populate at 200 s
                # a step as well as at 2 s. Doing this from ten_step_reporting tied
                # the sample rate to the step rate and left the metric absent on
                # exactly the slow MLIP arms it was added to watch.
                self._sample_gpu_util()
                # the controller scores the step it just timed, at the batch that
                # actually ran it: moving the batch before the append paired a new
                # batch with the old rung's timings (and made 'Batch Size' log one
                # rung ahead of train_step_time)
                if self.args.grow_batch_size:
                    self.select_batch_size()

                # train monitoring
                if self.step_ind % 10 == 0:
                    lr = self.step_lr_schedule()
                    metrics.update(self.ten_step_reporting())
                    self.monitor_losses(current_loss, step_type)
                    # gate publishers feed the exit triggers (gates/*); the
                    # protocol tick then runs the stage's balance nudge and
                    # arms the exit trigger (which pulls the next eval forward)
                    if self.protocol.stage.mle_gate is not None:
                        metrics.update(self.update_mle_gate())
                    # the stage's declared LR sensor. Returns {} (and touches
                    # nothing) unless the stage declares lr_sensor kind:
                    # plateau, so this is inert for every config without one.
                    metrics.update(self.update_lr_plateau())
                    self.protocol.tick()

                # evaluation work -- stage_ctrl 'request_eval' pulls the eval
                # forward to the step an exit trigger armed (or a reloaded
                # pre-transition snapshot stamped it), instead of waiting out
                # the rest of the eval period while the exit metrics degrade
                if ((self.step_ind % self.args.eval_period == 0 and self.step_ind > 0)
                        or self.step_ind == 50
                        or self.stage_ctrl.get('request_eval', False)):
                    # buffers are ~90% of a full save's bytes, so they ride the
                    # eval cadence rather than every 'running' save. Written
                    # BEFORE evaluation(), which is where stage transitions fire:
                    # this is the pre-transition buffer state, and a transition
                    # freezes its own tagged copy on top (protocol._snapshot)
                    self.checkpointer.save_buffers()
                    metrics.update(self.evaluation(override_do_figs=self.stage_ctrl.get('request_eval', False)))

                if len(metrics) > 0:
                    # array-valued metrics ride the eval_period grid only, so their
                    # wandb histogram-over-time panels get a uniform x-spacing --
                    # the off-grid evals (step 50, and the stage-transition
                    # request_eval pull-forwards) rendered the panels illegible.
                    # Those steps still log all their scalars.
                    if self.step_ind % self.args.eval_period != 0:
                        metrics = {k: v for k, v in metrics.items()
                                   if not (isinstance(v, wandb.Histogram)
                                           or (isinstance(v, np.ndarray) and v.size > 1))}
                    wandb.log(metrics, step=self.step_ind, commit=True)

                    # Only AFTER the first eval. record_peak used to fire from step 10,
                    # so a run killed early wrote a partial high-water mark that later
                    # launches then trusted as a measurement -- and the eval rollout is
                    # the single largest allocation of the step, so a pre-eval peak is a
                    # systematic UNDER-estimate. configs/gauss_aug12/make.py already
                    # documents this exact trap ("taken at step 150, before stage-2 fused
                    # training ... those arms would have died mid-run"); the lesson was
                    # never applied here.
                    # Record this config's VRAM high-water mark so the NEXT launch's
                    # pre-flight can project from a measurement instead of falling back
                    # to cuda_memory_fraction x total (a bound, not a prediction).
                    # Written here rather than at exit on purpose: an OOM kill or a
                    # BSOD never reaches an atexit hook, and those are exactly the runs
                    # whose footprint most needs to be on record. See gpu_guard.py.
                    if self.step_ind >= max(50, int(self.args.eval_period)):
                        self.record_vram_peak()

                if self.step_ind % 50 == 0:  # save running model
                    self.checkpointer.save('running')
                    # 'best' used to re-serialize the whole checkpoint every 10
                    # steps it improved, which alone was ~a quarter of train
                    # time. It's the same bytes 'running' just wrote, so link
                    # rather than rewrite them. The min is now only sampled
                    # every 50 steps, so 'best' means best of those samples.
                    if (self.combo_loss_record
                            and self.combo_loss_record[-1] <= np.amin(self.combo_loss_record)):
                        self.checkpointer.link('running', 'best')
                    # PERIODIC ARCHIVE. The single-stage protocol fires no
                    # on_exit snapshots, so 'running'/'best' are the only saves
                    # and both are rewritten in place -- a killed run leaves no
                    # static reload point behind (this is how flwl4jxz's
                    # oscillating state was lost to the next launch within
                    # minutes). Linked, not re-serialized: atomic_save swaps the
                    # directory entry, so the archive keeps the old inode and
                    # these bytes are frozen at zero write cost.
                    self.checkpointer.archive(self.step_ind)

            self.checkpointer.save('final', with_buffers=True)
            print("Finished Training!")

    def monitor_losses(self, current_loss, step_type):
        # NON-FINITE GRADIENT -> the divergence response, not silence. step_loss
        # returns early on a non-finite pre-clip norm, so current_loss is None and
        # the check_spike branch below never runs -- which is how a run could sit
        # in that state making zero progress. check_spike could not have caught it
        # anyway: its non-finite trigger reads last_grad_norm_pre_clip, which that
        # same early return leaves at its last FINITE value.
        # Fired on streak 1, 11, 21... rather than every step: a rewind needs a few
        # steps to show whether it took, and fire_loss_spike's own rewind budget is
        # the escalation path -- it raises FrozenTrainingState once exhausted.
        if getattr(self, '_nonfinite_pending', False):
            self._nonfinite_pending = False
            if (getattr(self, '_grad_nonfinite_streak', 0) % 10) == 1:
                print(f"non-finite gradient at {self.step_ind} "
                      f"(streak {self._grad_nonfinite_streak}) -> rewind + peak cut")
                self.fire_loss_spike()
        if current_loss is not None:
            # check_spike (LRController) is the one remaining tripwire: an
            # absolute ~1e9 bar on branch loss / pre-clip grad norm, or a
            # non-finite reading. fire_loss_spike does the rewind + peak cut.
            trig = self.lr_controller.check_spike(
                step_type, current_loss, getattr(self, 'last_grad_norm_pre_clip', None))
            if trig == 'diverged':
                self.fire_loss_spike()

            current_fwd = self.metric_tracker.get('fwd', 'r2')
            current_bwd = self.metric_tracker.get('bwd', 'r2')
            current_replay = self.metric_tracker.get('replay', 'r2')

            if current_fwd is None and current_bwd is None and current_replay is None:
                self.combo_loss_record.append(float('inf'))
            else:
                total = (current_fwd or 0) + (current_bwd or 0) + (current_replay or 0)
                self.combo_loss_record.append(3 - total)  # (1-x) + (1-y) + (1-z) = 3-x-y-z

    def _rewind_checkpoint_path(self):
        """Pick fire_loss_spike's rewind target, NEVER reverting to an earlier
        stage than the current one. Reloading a stale 'best' from a prior stage
        restores that stage's name + stage_ctrl but not its on_enter buffer
        surgery, so the run resumes the old stage against the current stage's
        live buffers (stab_july21c 512x6_T60: rewound buildout -> z_match and
        hung 11k steps fitting a 174k broad buffer a z_match Z was never
        calibrated for).

        Prefer the same-stage 'best' (freshest). Else this stage's 'stage_start'
        turnover point, saved post-on_enter in protocol.advance and healthy by
        construction (it cleared the previous stage's exit gate). Else fall back
        to 'best' so behavior is never worse than before the fix.
        """
        idx = {s.name: s.index for s in self.protocol.stages}
        current = idx.get(self.protocol.stage.name, -1)

        def stage_index(tag):
            path = self.checkpointer.path_for(tag)
            if not os.path.exists(path):
                return None, path
            try:
                ck = torch.load(path, map_location='cpu', weights_only=False)
                return idx.get(ck.get('modeller_state', {}).get('stage')), path
            except Exception:
                return None, path

        best_idx, best_path = stage_index('best')
        if best_idx is not None and best_idx >= current:
            return best_path
        start_idx, start_path = stage_index('stage_start')
        if start_idx == current:
            print(f"rewind: 'best' stage {best_idx} precedes current stage {current}; "
                  f"reverting to this stage's start rather than reversing the phase")
            return start_path
        return best_path if os.path.exists(best_path) else None

    def fire_loss_spike(self):
        """
        Divergence response: rewind to the best checkpoint AND cut the servo
        peak, recording the ceiling.

        The pairing is the point (docs/to_do_rebuild.md A5). A reload without
        an LR cut re-enters the same state at the same LR and explodes again;
        an LR cut without a reload keeps the damaged weights. The deleted
        middle layer's failure was never that it did both -- it was that it did
        them on a graduated trigger with a latch and a recovery ramp.

        Repeated divergences compound: each one halves the peak from wherever
        it stood, so the ceiling descends across policy deaths instead of
        sawtoothing (the djr13t0j failure -- a rewind restores checkpointed LR
        state, so the ceiling is held on the controller INSTANCE where the
        rewind cannot reach it).
        """
        self.total_reloads = getattr(self, 'total_reloads', 0) + 1
        # RATE, not a count: a fixed cap makes a long run likelier to abort for
        # the same per-step behaviour. Budget scales with steps elapsed, with a
        # floor so a detonation in the first few hundred steps still aborts.
        per_k = float(getattr(self.args, 'max_reloads_per_1k_steps', 0.2) or 0)
        cap = max(3.0, per_k * self.step_ind / 1000.0) if per_k > 0 else 0
        if cap > 0 and self.total_reloads > cap:
            # A rewind restores healthy WEIGHTS but not necessarily a
            # survivable LR: the peak cut applies to servo-managed groups, and
            # a config pinning every LR to an explicit float can rewind,
            # re-detonate at the same LR, and repeat forever -- never dying,
            # never recovering, holding a GPU indefinitely. N rewinds without
            # recovery IS the unrecoverable signal; the frozen detector cannot
            # see it because the grad norm keeps CHANGING (2471 -> 1.3e5 ->
            # 4.2e6 -> 9.2e7 on aug02 arm a2_T25_lr16_tight / d7z705wc).
            msg = (f"UNRECOVERABLE at step {self.step_ind}: {self.total_reloads} rewinds "
                   f"(budget {cap:.1f} at {per_k}/1k steps) and the run keeps "
                   f"re-detonating -- rewinding restores "
                   f"weights but not a survivable LR. Aborting so the GPU is released.")
            print(msg)
            try:
                wandb.run.summary['unrecoverable_abort'] = msg
            except Exception:
                pass
            raise FrozenTrainingState(msg)
        print(f"Divergence response: rewind #{self.total_reloads} + peak cut")
        running_checkpoint_path = self._rewind_checkpoint_path()
        if not (running_checkpoint_path and os.path.exists(running_checkpoint_path)):
            # NO REWIND TARGET. Previously this fell straight through in
            # silence, which is how the abort test (jgyk2lzl) detonated and
            # then ran on with no brake whatsoever: the run had a fresh
            # run_name so no '<prefix>_best.pt' existed yet.
            #
            # This is reachable in normal use: 'best' is only written once an
            # eval has improved, so any detonation inside the first eval_period
            # of a from-scratch run lands here. Say so LOUDLY and still take
            # the half of the response that is available.
            print(f"lr_ctrl WARNING: divergence at step {self.step_ind} but NO rewind "
                  f"target exists (looked for {running_checkpoint_path}). Cannot restore "
                  f"healthy weights; cutting the peak alone.")
            self.lr_controller.on_divergence()
            return
        if running_checkpoint_path and os.path.exists(running_checkpoint_path):
            self.checkpointer.load_model_only(running_checkpoint_path,
                                              load_optimizers=True)
            # fix also rolling metrics with appropriate rebase
            checkpoint = torch.load(running_checkpoint_path, map_location=self.device, weights_only=False)
            step = deepcopy(self.step_ind)
            self.checkpointer.set_state_dict(checkpoint['modeller_state'])
            self.metric_tracker.load_state_dict(checkpoint.get('metrics', {}))
            self.step_ind = step
            # only pre-sidecar checkpoints carry buffers inline. Otherwise the
            # LIVE buffers are kept rather than pulled from the sidecar: the
            # spike damaged the weights, not the buffers, and the live ones are
            # strictly fresher than a sidecar written up to eval_period ago.
            if any(checkpoint.get(k) is not None
                   for k in ('prior_buffer', 'replay_buffer', 'anchor_buffer')):
                self.checkpointer.restore_buffers(checkpoint, running_checkpoint_path)
            if checkpoint.get('condition_log_z') is not None:
                self.condition_log_z = ConditionLogZTracker.from_state_dict(
                    checkpoint['condition_log_z'], current_step=self.step_ind)
                # config owns behavior: mixing-rate/trust hyperparams re-assert
                # from the live config on every resume (from_state_dict restores
                # the stored ones, which silently deadens the config knobs).
                # clip_beta deliberately excluded: z_grad_ema's accumulated
                # history is denominated in it (fixed-ruler reasoning).
                cz = getattr(self.args, 'condition_log_z', None)
                if cz is not None:
                    for key in ('min_visits', 'half_life_visits', 'trim_frac',
                                'max_batch_weight'):
                        val = getattr(cz, key, None)
                        if val is not None:
                            setattr(self.condition_log_z, key, val)

        # set_state_dict above restored lr_ctrl (warmup clock + peak_scale)
        # from the healthy best checkpoint; the CEILING lives on the controller
        # instance and survives the rewind, so the cut below is applied to a
        # healthy peak and then clamped by evidence the rewind cannot erase.
        self.lr_controller.on_divergence()

    def update_ema_model(self):
        if self.args.ema_decay is not None:
            update_ema(self.gfn_model, self.ema_model, decay=self.args.ema_decay)
        else:
            self.ema_model = self.gfn_model

    def set_detect_anomaly(self, do_anomaly_detection: bool):
        if do_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)  # for debugging

            def grad_check_hook(grad, name):
                if not torch.isfinite(grad).all():
                    raise RuntimeError(f"NaN/Inf gradient in {name}")
                return grad

            for p_name, p in self.gfn_model.named_parameters():
                if p.requires_grad:
                    p.register_hook(lambda g, n=p_name: grad_check_hook(g, n))

    def train_step(self,
                   step_type,  # 'fwd' | 'bwd' | 'replay' | 'fused'
                   ):
        discretizer = get_discretizer(self.args.integrator)

        accum_target = self.args.fused_grad_accum_min_samples if step_type == 'fused' else 0
        # batch >= target degenerates to a plain unscaled step: accumulation
        # only engages BELOW the target (e.g. an OOM-cut batch under the
        # gradient-stability floor), so a large batch is never loss-scaled by
        # batch/target > 1
        accumulating = accum_target > self.batch_size
        if not accumulating:
            self.fused_accum_count = 0  # drop any partial cycle from before a batch jump
        self.fused_accum_count = getattr(self, 'fused_accum_count', 0)
        starting_new_cycle = (not accumulating) or (self.fused_accum_count == 0)

        if starting_new_cycle:
            # flow (Z head) params live in optimizers['flow'] for non-fused steps and
            # as a dedicated param group of optimizers['fused'] for fused steps, so the
            # right owner zeroes them either way: skip the standalone flow optimizer on
            # fused steps (optimizers[step_type] == 'fused' already covers its group).
            if 'flow' in self.optimizers and step_type != 'fused':
                self.optimizers['flow'].zero_grad(set_to_none=True)
            self.optimizers[step_type].zero_grad(set_to_none=True)

        if step_type == 'fwd':
            loss, crystal_batch, loss_dict = self.fwd_train_step(
                discretizer,
                return_exp=True,
                repeats=self.mode_repeats('fwd'),
                report_losses=True
            )
            self.fwd_step_count += 1
            current_step_count = self.fwd_step_count

        elif step_type == 'bwd':
            loss, loss_dict = self.bwd_train_step(
                discretizer,
                repeats=self.mode_repeats('bwd'),
                report_losses=True)
            self.bwd_step_count += 1
            current_step_count = self.bwd_step_count

        elif step_type == 'replay':
            loss, loss_dict = self.replay_train_step(
                discretizer,
                repeats=self.mode_repeats('replay'),
                report_losses=True)
            self.replay_step_count += 1
            current_step_count = self.replay_step_count

        elif step_type == 'fused':
            loss, sub_losses = self.fused_train_step(
                discretizer,
                report_losses=True)

        else:
            assert False

        if step_type == 'fwd':
            # churn on the fly
            self.manage_replay_buffer(loss_dict,
                                      crystal_batch)
            del crystal_batch

        reported_loss = loss.cpu().detach().item()

        # Step probe: snapshot policy params, let the optimizer step land, then
        # re-score a frozen held-out batch at alpha in {0, 1/2, 1} along the step
        # actually taken. Sensor only -- measure() restores every parameter it
        # touches bitwise, and a probe that finds no step (non-finite gradient,
        # or mid-accumulation) tallies 'nostep' and returns without reading.
        probe_armed = self._ray_probe_armed()

        if accumulating:
            self.fused_accum_count += self.batch_size
            do_step = self.fused_accum_count >= accum_target
            self.step_loss(step_type, loss * (self.batch_size / accum_target), do_step=do_step)
            if do_step:
                self.fused_accum_count = 0
        else:
            self.step_loss(step_type, loss)

        if probe_armed:
            # Sub-batches are drawn lazily, one at a time, inside measure(): the
            # loop is sub-batch outer / alpha inner, so peak memory is one draw
            # regardless of n_sub while every contrast stays within one batch.
            def _loss(drawn):
                batch, coeffs, source, repeats = drawn
                return self._probe_loss(batch, coeffs, source, discretizer, repeats)

            reading = self.ray_cal.measure(self._draw_probe_batch, _loss)
            if reading is not None:
                self.lr_controller.on_calibration(reading)

        if step_type == 'fused':
            self.record_fused_substep_losses(sub_losses)
        elif current_step_count % 10 == 0:
            self._update_rolling(loss_dict, loss, step_type)

        # torch.cuda.synchronize()
        self.update_ema_model()
        return reported_loss

    def fused_train_step(self,
                         discretizer,
                         report_losses: bool = True):
        """
        Fires fwd, bwd, and replay steps together and fuses their losses into a
        single weighted-sum loss (weighted by fwd_frac/bwd_frac/replay_frac),
        backed by its own optimizer -- the stage's balance rules move the fracs,
        which act here as loss weights rather than throughput shares.

        A branch whose frac has fallen below controller.deactivate_threshold is skipped
        entirely (not just down-weighted) to save its compute; the remaining active
        branches' weights are renormalized to sum to 1. Since the three fracs always sum
        to 1, at least one is guaranteed to survive as long as the threshold is < 1/3.

        Every controller.refresh_every steps, each NON-DORMANT branch (a branch some
        rule or exit term actually reads -- protocol.mode_dormant) is force-evaluated
        regardless of its frac, so its rolling metric_tracker stats don't go stale. A
        force-evaluated branch that's still below threshold contributes zero weight to
        the fused loss -- it's run only to refresh its stats, not for gradient.
        """
        # replay joins the fused loss only in stages whose balance can boost it
        # (mode_boostable, derived from the rule list -- the old 'fused stages only'
        # check): a stage that never boosts replay pins its frac at zero, and
        # the gate additionally keeps force_refresh from burning a replay pass
        # (and polluting its rolling stats) on a branch that isn't part of the stage
        replay_available = (self.protocol.mode_boostable('replay')
                            and hasattr(self, 'replay_buffer') and len(self.replay_buffer) > 0)
        # per-stage override first (stage.deactivate_threshold), global default
        # otherwise -- pairs with the stage's min_fracs so each phase states
        # explicitly which modes may switch off (a min_frac at or above the
        # deactivate threshold = that mode is never skipped)
        stage_deact = self.protocol.stage.deactivate_threshold
        deactivate_threshold = (self.args.controller.deactivate_threshold
                                if stage_deact is None else stage_deact)

        self.fused_step_count = getattr(self, 'fused_step_count', 0) + 1
        force_refresh = self.fused_step_count % self.args.controller.refresh_every == 0

        sub_losses = {}
        weights = {}

        fwd_active = self.fwd_frac >= deactivate_threshold
        fwd_ran = fwd_active or (force_refresh and not self.protocol.mode_dormant('fwd'))
        if fwd_ran:
            fwd_loss, crystal_batch, fwd_loss_dict = self.fwd_train_step(
                discretizer,
                return_exp=True,
                repeats=self.mode_repeats('fwd'),
                report_losses=report_losses)
            if not fwd_active:  # force-refresh only -- keep its graph out of the fused loss
                fwd_loss = fwd_loss.detach()
            sub_losses['fwd'] = (fwd_loss, fwd_loss_dict, fwd_active)
            weights['fwd'] = self.fwd_frac if fwd_active else 0.0

        # a DORMANT mode (protocol.mode_dormant: nothing in this stage's rules
        # or exit trigger reads its rolling stats) skips its force-refresh
        # entirely -- a full rollout every refresh_every steps purely to keep
        # unread stats fresh is the dominant waste in a stage that doesn't use
        # them (the old forward-first stage-A bwd_dormant, generalized). Stats
        # just start stale in the next stage and populate once it reads them.
        bwd_refresh = force_refresh and not self.protocol.mode_dormant('bwd')
        bwd_active = self.bwd_frac >= deactivate_threshold
        if bwd_active or bwd_refresh:
            bwd_loss, bwd_loss_dict = self.bwd_train_step(
                discretizer,
                repeats=self.mode_repeats('bwd'),
                report_losses=report_losses)
            if not bwd_active:
                bwd_loss = bwd_loss.detach()
            sub_losses['bwd'] = (bwd_loss, bwd_loss_dict, bwd_active)
            weights['bwd'] = self.bwd_frac if bwd_active else 0.0

        if replay_available:
            replay_active = self.replay_frac >= deactivate_threshold
            if replay_active or (force_refresh and not self.protocol.mode_dormant('replay')):
                replay_loss, replay_loss_dict = self.replay_train_step(
                    discretizer,
                    repeats=self.mode_repeats('replay'),
                    report_losses=report_losses)
                if not replay_active:
                    replay_loss = replay_loss.detach()
                sub_losses['replay'] = (replay_loss, replay_loss_dict, replay_active)
                weights['replay'] = self.replay_frac if replay_active else 0.0
        elif bwd_active:  # buffer not populated yet - fold its share into backward, as the alternating controller does
            weights['bwd'] += self.replay_frac

        assert sub_losses, "fused_train_step deactivated all three branches -- controller.deactivate_threshold must be < 1/3"

        total_weight = sum(weights.values())
        fused_loss = sum((weights[k] / total_weight) * sub_losses[k][0]
                         for k in sub_losses if weights[k] > 0)

        if self._fused_grad_diag_armed():
            self._log_fused_gradient_geometry(sub_losses, weights, total_weight)

        if fwd_ran:
            # churn on the fly
            self.manage_replay_buffer(fwd_loss_dict,
                                      crystal_batch)
            del crystal_batch

        return fused_loss, sub_losses

    def record_fused_substep_losses(self, sub_losses):
        # *_step_count means "times trained on" -- it paces buffer churn via
        # bwd_step_delta/replay_step_delta -- so a force-refresh-only run
        # (trained=False, zero weight in the fused loss) must not advance it.
        # Its rolling stats update unconditionally instead: fresh stats are the
        # run's whole purpose, and its frozen counter would otherwise pin the
        # %10 gate open or shut forever.
        for sub_type, (sub_loss, loss_dict, trained) in sub_losses.items():
            # unconditional, ahead of the %10 gate: the probe's whole purpose is
            # to sample faster than that gate allows (no-op unless armed)
            self._per_step_probe(loss_dict, sub_type)
            if trained:
                count = getattr(self, f'{sub_type}_step_count') + 1
                setattr(self, f'{sub_type}_step_count', count)
                if count % 10 == 0:
                    self._update_rolling(loss_dict, sub_loss, sub_type)
            else:
                self._update_rolling(loss_dict, sub_loss, sub_type)

    def _ramp_params(self):
        """
        (ramp_floor, ramp_width) from buffers.prior_buffer, validated so a
        degenerate ramp fails loudly at read time instead of silently zeroing
        or inverting under_coverage. Both live in ENERGY units of depth
        d = E - E*_c above each sample's own condition's best known energy:

            w_raw(d) = clamp((ramp_floor - d) / ramp_width, 0, 1)

        i.e. weight 0 for d >= ramp_floor (only samples within ramp_floor of
        their conditional minimum contribute at all), ramping linearly to 1 at
        d = ramp_floor - ramp_width, flat at 1 from there down to the minimum.
        ramp_width is the width of the transition band ANCHORED AT THE FLOOR
        -- not a second independent depth -- so validity is just
        0 < ramp_width <= ramp_floor and no subtraction of two config values
        can flip the ramp's sign (the legacy floor_range/knee_range pair
        silently inverted whenever knee > floor, zeroing under_coverage on
        every batch of good samples). Legacy configs still carrying
        ramp_floor_range/ramp_knee_range are converted (floor = floor_range,
        width = floor_range - knee_range) and pass through the same
        validation, so an inverted legacy pair now crashes instead.
        Returns (None, None) when the ramp is unconfigured.
        """
        cfg = getattr(getattr(self.args, 'buffers', None), 'prior_buffer', None)
        if cfg is None:
            return None, None
        floor = getattr(cfg, 'ramp_floor', None)
        width = getattr(cfg, 'ramp_width', None)
        if floor is None and width is None:  # legacy key names
            floor = getattr(cfg, 'ramp_floor_range', None)
            knee = getattr(cfg, 'ramp_knee_range', None)
            if floor is None and knee is None:
                return None, None
            width = None if (floor is None or knee is None) else floor - knee
        if floor is None or width is None:
            raise ValueError(
                f"prior_buffer ramp half-configured: need both ramp_floor and ramp_width "
                f"(got ramp_floor={floor}, ramp_width={width})")
        if not (0 < width <= floor):
            raise ValueError(
                f"prior_buffer ramp misconfigured: need 0 < ramp_width <= ramp_floor, "
                f"got ramp_floor={floor}, ramp_width={width} "
                f"(if using legacy keys, ramp_knee_range must be < ramp_floor_range)")
        return floor, width

    def _reward_ramp_kwargs(self, condition_id=None):
        """
        reward_floor/ramp_width for quick_tb_stats' weighted under_coverage:
        the _ramp_params depth-space ramp translated into per-sample reward
        space, expressed relative to the current best anchor reward rather
        than as absolute numbers -- so the ramp automatically tracks the
        achievable reward scale as it rises over training. Given
        anchor_buffer's PER-CONDITION current max M_c (each sample in
        `condition_id` looked up against its own condition's anchors, via
        _per_condition_max), depth d = M_c - log_r at temperature 1, so:

            reward_floor = M_c - ramp_floor   (weight 0 at or below)
            -> weight saturates to 1 at reward_floor + ramp_width,
               and stays at 1 all the way up through M_c.

        Per-condition rather than a single buffer-wide max because conditions
        can sit on very different achievable reward scales (e.g. under
        temperature_conditioning=True) -- a global max would let the
        best-off condition's scale gate every other condition's ramp.
        Conditions with no anchors yet fall back to the buffer-wide max.
        Falls back to (None, None) -- quick_tb_stats' old uniform-RMS behavior
        -- until there's an anchor buffer (and condition_id) to anchor the scale to.
        """
        anchor_buffer = getattr(self, 'anchor_buffer', None)
        if anchor_buffer is None or len(anchor_buffer) == 0 or condition_id is None:
            return dict(reward_floor=None, ramp_width=None)

        floor, width = self._ramp_params()
        if floor is None:
            return dict(reward_floor=None, ramp_width=None)

        condition_id = torch.as_tensor(condition_id)
        query_ids = condition_id.detach().cpu().long().flatten()
        anchor_max = _per_condition_max(anchor_buffer.condition_id, anchor_buffer.reward, query_ids)
        missing = ~torch.isfinite(anchor_max)
        if missing.any():
            anchor_max = anchor_max.clone()
            anchor_max[missing] = anchor_buffer.reward.max()

        return dict(reward_floor=(anchor_max - floor).to(condition_id.device),
                    ramp_width=width)

    def _condition_energy_floor(self, condition_id):
        """
        Per-sample best-known energy for that sample's own condition_id --
        the E_min(condition) baseline manage_prior_buffer's admission gate
        and reach trigger measure relative excess against, instead of an
        absolute reward threshold (mirrors AnchorBuffer.admit's per-condition
        energy gate; see _per_condition_min). Prefers condition_log_z's
        persistent best_energy, since it's updated from every fwd-eval batch
        and so sees a strict superset of what the anchor buffer's own
        (thinned) per-condition min sees; falls back to the anchor buffer's
        per-condition min when the tracker isn't configured. A condition with
        no observations from either source gets +inf, the same convention
        _per_condition_min/AnchorBuffer.admit use, so its first sample is
        always admitted regardless of margin. Returns None only when neither
        source exists yet at all (pre-bootstrap), signaling callers to fall
        back to the absolute reward_min floor. Result is moved back onto
        condition_id's original device, so it can be compared directly
        against energy computed upstream on either CPU (fwd-eval-sourced
        batches) or GPU (top_up_prior_from_anchors' freshly-rescored batch)
        without callers having to think about it.
        """
        condition_id = torch.as_tensor(condition_id)
        device = condition_id.device
        condition_id = condition_id.detach().cpu().long().flatten()

        tracker = getattr(self, 'condition_log_z', None)
        if tracker is not None:
            best_energy, mask = tracker.lookup_best_energy(condition_id)
            return torch.where(mask, best_energy, torch.full_like(best_energy, float('inf'))).to(device)

        anchor_buffer = getattr(self, 'anchor_buffer', None)
        if anchor_buffer is not None and len(anchor_buffer) > 0:
            return _per_condition_min(anchor_buffer.condition_id, anchor_buffer.energy, condition_id).to(device)

        return None

    def _per_step_probe(self, loss_dict, sub_type):
        """EVERY-STEP capture of the flat-direction coordinates, for the one
        test that the ordinary 1-in-10 logging cannot do.

        [[flat-direction-limit-cycle-phase2]] leaves one degeneracy open: every
        scalar is logged 1-in-10, and a true ~2.004-step mode (edge-of-stability
        period 2) sampled at that cadence aliases to a clean ~550-step
        sinusoid, which is exactly what the ripple looks like. A lag problem and
        a curvature problem want opposite fixes, so this has to be settled
        before any mechanism work. The discriminator is the lag-1
        autocorrelation of the DETRENDED per-step series:

            true ~550-step mode  -> rho_1 close to +1
            aliased 2-step mode  -> rho_1 close to -1

        Off unless per_step_probe_steps > 0, and it captures a bounded window
        (start at the first step taken, run for N steps, dump, stop) so it can
        never grow without limit on a long run. Cost while armed is two
        reductions over flow_states -- the same ones _update_rolling already
        does -- which is negligible against a rollout.

        Deliberately raw per-step values, no EMA and no tracker: the tracker is
        an EMA whose smoothing is precisely what would destroy a 2-step mode.
        """
        n = int(getattr(self.args, 'per_step_probe_steps', 0) or 0)
        if n <= 0 or sub_type != 'fwd':
            return
        buf = getattr(self, '_probe_buf', None)
        if buf is None:
            buf = self._probe_buf = []
            self._probe_start = self.step_ind
        if self.step_ind - self._probe_start >= n:
            return
        states = loss_dict.get('flow_states')
        if states is None or states.ndim != 3:
            return
        # CALIBRATION PANEL, not just the variance coordinates. The first probe
        # run (vlqklgmy) captured step_var/terminal_var only and came back
        # near-white at T=10 -- but [[flat-direction-limit-cycle-phase2]] already
        # says that at T=10 the ~120 step-dims are ~6x stiffer, the noise budget
        # cannot move, and the mode surfaces in slope_err/intercept_err instead.
        # So that null was measured on an observable predicted to be quiet.
        # These come from the same quick_tb_stats _update_rolling uses; one extra
        # call per step while armed is cheap against a rollout.
        cal = {}
        try:
            cal = quick_tb_stats(
                loss_dict['log_pf'], loss_dict['log_pb'],
                loss_dict['log_Z'], loss_dict['log_r'],
                clip_beta=getattr(getattr(self.args, f'{sub_type}_loss_coeffs'),
                                  'beta', None),
                condition_id=loss_dict.get('condition_id'),
                worst_quantile=self.args.conditional_worst_quantile,
                **self._reward_ramp_kwargs(loss_dict.get('condition_id')))
        except Exception:
            pass  # a diagnostic probe must never be able to kill a run
        row = [float(self.step_ind),
               float(states[:, -1].var(dim=0).mean()),
               float((states[:, 1:] - states[:, :-1]).pow(2).mean())
               if states.shape[1] > 1 else float('nan'),
               float(getattr(self, 'last_grad_norm_pre_clip', float('nan')) or float('nan')),
               float(cal.get('slope_err', float('nan'))),
               float(cal.get('intercept_err', float('nan'))),
               float(cal.get('scatter_err', float('nan'))),
               float(cal.get('tb_err', float('nan')))]
        buf.append(row)
        if self.step_ind - self._probe_start == n - 1:
            import numpy as _np
            path = os.path.join(self.args.checkpoints_dir,
                                f'per_step_probe_{self.args.run_name}.npz')
            arr = _np.asarray(buf, dtype=float)
            _np.savez(path, probe=arr,
                      columns=_np.array(['step', 'terminal_var', 'step_var',
                                         'grad_norm_pre_clip', 'slope_err',
                                         'intercept_err', 'scatter_err', 'tb_err']))
            print(f"per-step probe: wrote {arr.shape[0]} rows to {path}")

    def _update_rolling(self, loss_dict, sub_loss, sub_type):
        stats = quick_tb_stats(loss_dict['log_pf'], loss_dict['log_pb'],
                               loss_dict['log_Z'], loss_dict['log_r'],
                               clip_beta=getattr(getattr(self.args, f'{sub_type}_loss_coeffs'), 'beta', None),
                               condition_id=loss_dict.get('condition_id'),
                               worst_quantile=self.args.conditional_worst_quantile,
                               **self._reward_ramp_kwargs(loss_dict.get('condition_id')))
        stats.update({k: v.item() for k, v in loss_dict.items() if k not in
                      ['log_pf', 'log_pb', 'log_Z', 'log_r', 'losses', 'flow_states', 'resid', 'condition_id']})
        stats.update({'loss': sub_loss.cpu().detach().item()})
        stats.update({'log_Z_learned': loss_dict['log_Z'].cpu().mean().detach().item()})
        # NB the condition-aware metrics -- 'logw_std_within' (the clean
        # per-direction convergence signal, vs the batch-wide 'logw_std' that
        # between-condition log Z(c) spread dominates at scale) and the
        # calibration family (cond_tb_err / tb_err_worst / z_grad_worst) -- all
        # come straight out of quick_tb_stats above off the condition_id passed
        # in, so there is no branch here. See its docstring for which of them
        # degrade to their pooled counterparts on an unconditional batch and
        # which (only logw_std_within) are omitted outright.
        # box containment on the TRAIN-step cadence. 'Mean bounding_energy' exists
        # already but is EVAL-cadence (log_thermo_properties loops the eval batch),
        # far too coarse to gate a controller on: in 1219ddv9 boundary drift led the
        # r2 collapse by ~100 steps and unrecoverability by ~1300, so the warning
        # window is shorter than the eval interval. Mirrors generator_energy's
        # bounding_energy -- relu(|x|-1)^2 summed over dims, pre-temperature and
        # pre-bounding_coeff -- but read straight off the terminal latents, so it
        # costs no extra energy evaluation. contact_frac is the bounded companion:
        # box_violation spans 4500x over a run and says nothing on its own about how
        # much of the batch is involved.
        states = loss_dict.get('flow_states')
        if states is not None and states.ndim == 3:
            viol = (states[:, -1].abs() - 1.0).clamp(min=0.0)
            stats['box_violation'] = (viol ** 2).sum(dim=-1).mean().item()
            stats['box_contact_frac'] = (viol > 0).any(dim=-1).float().mean().item()
            # REALIZED spread, on the train-step clock. 'Mean F/B Var' are the
            # model's logvar PARAMETERS and are only available at eval cadence
            # (the train sampler runs return_gauss_params=False, so retaining
            # them would cost per-step tensors) -- but the quantity that
            # actually matters, how wide the sampled cloud is and how big its
            # steps are, is already sitting in flow_states for free. Both are
            # per-dim means so they're comparable across problems:
            #   terminal_var -- variance of the terminal latents across the
            #     batch: the width of what the policy is actually producing
            #   step_var -- mean squared increment along the trajectory: the
            #     realized per-step noise+drift budget, i.e. the train-cadence
            #     proxy for the joint P_F/P_B inflation that runs along TB's
            #     flat direction. Logged so the variance breathing can be
            #     phase-resolved against logw_std/box_contact instead of being
            #     sampled ~3x per cycle at eval cadence.
            # Averaged over LIVE dims only. The claim just above -- "both are per-dim
            # means so they're comparable across problems" -- fails if the denominator
            # counts dims that structurally cannot move: a monoclinic run holds 2 of 12
            # dead (D33), so a full-width mean would report terminal_var and step_var
            # ~10/12 of their true per-dof value and would drop DISCONTINUOUSLY at the
            # change, which reads as a coverage regression that never happened. Exactly
            # equal to the full-width mean for triclinic and toys.
            live_states = states
            live_idx = getattr(getattr(self, 'gfn_model', None), 'live_idx', None)
            if live_idx is not None and live_idx.numel() != states.shape[-1]:
                live_states = states.index_select(-1, live_idx.to(states.device))
            stats['terminal_var'] = live_states[:, -1].var(dim=0).mean().item()
            if states.shape[1] > 1:
                stats['step_var'] = (live_states[:, 1:] - live_states[:, :-1]).pow(2).mean().item()
        # RAW (pre-EMA) stats cache, one step deep: the MLE slope gate samples
        # bwd/mle from here every 10 steps -- raw batch losses are ~independent
        # across steps, so its OLS slope needs no autocorrelation correction
        self._last_stats[sub_type] = stats
        self.metric_tracker.update(sub_type, stats, self.step_ind)

    def _submodel_grad_norms(self):
        """
        Per-submodel gradient L2 norm, one row per named child of the GFN
        (forward_policy, backward_policy, flow_model, conditions_embedding_model,
        s_model, t_model -- whatever the arch actually has; nothing is hardcoded,
        so an arch change is picked up for free).

        This replaces wandb.watch(log='gradients'), which spent 110 histogram
        channels (~7000 bins per event) on two questions these scalars answer
        directly and at train cadence: is gradient reaching every part of the
        model (a line pinned at 0 = it is not), and is any sub-model cooling off
        (a line trending down). A submodel whose params all have grad=None
        reports 0.0 rather than being omitted -- a missing key vanishes silently
        on a dashboard, a zero is visible.

        MUST be called before clip_grad_norm_, which rescales grads in place.
        One stacked .cpu() so the whole set costs a single device sync, and only
        on reporting steps.
        """
        names, norms = [], []
        for name, mod in self.gfn_model.named_children():
            grads = [p.grad.detach() for p in mod.parameters(recurse=True)
                     if p.grad is not None]
            names.append(name)
            norms.append(torch.linalg.vector_norm(torch.stack(
                [torch.linalg.vector_norm(g) for g in grads]))
                if grads else torch.zeros((), device=self.device))
        if not names:
            return {}
        return {f'gradnorm/{n}': v for n, v in
                zip(names, torch.stack(norms).cpu().tolist())}

    def _fused_grad_diag_armed(self) -> bool:
        """Opt-in cadence gate for _log_fused_gradient_geometry.

        grad_geometry.enabled defaults absent -> off (config generation is not
        loading -- an omitted block must not silently start paying for extra
        backward passes). every <= 0 also disables it. Ticks on fused_step_count,
        which only advances inside fused_train_step, so this is a no-op outside
        a fused stage.
        """
        cfg = getattr(self.args, 'grad_geometry', None)
        if cfg is None or not getattr(cfg, 'enabled', False):
            return False
        if getattr(self, '_fused_grad_geom_dead', False):
            return False        # self-disabled after a failure; see the handler
        every = int(getattr(cfg, 'every', 0) or 0)
        return every > 0 and self.fused_step_count % every == 0

    def _log_fused_gradient_geometry(self, sub_losses, weights, total_weight):
        """
        Periodic, cheap diagnostic on a fused step's ACTIVE branches (weight >
        0, i.e. carrying a graph -- a force-refresh-only branch is already
        detached and has none): are their gradients COOPERATING (aligned --
        higher batch/LR likely accelerates everyone), ORTHOGONAL (independent
        -- optimization capacity/preconditioning may be the limiting factor),
        or CONFLICTING (fighting -- the current slow thermalization may be the
        unavoidable Pareto path, or benefit from conflict-aware geometry like
        PCGrad)? Without this measurement any of those three stories is
        speculation.

        One torch.autograd.grad per active branch, retain_graph=True so the
        real fused_loss.backward() downstream (in step_loss) is unaffected --
        this only reads gradients, it never writes .grad or frees the graph.
        The fused gradient itself is DERIVED from the branch grads (linearity
        of d/dtheta) rather than measured with a second backward, so the cost
        is exactly num_active_branches extra passes, on armed steps only
        (_fused_grad_diag_armed).

        WHICH PARAMETERS A PAIR ACTUALLY CONTENDS OVER IS THE FIRST QUESTION,
        not a detail. Under mk_dev's equilibration stage `fwd` carries
        freeze_policy and `bwd`/`replay` carry freeze_z, so fwd is Z-only and
        the other two are policy-only: they are PARAMETER-DISJOINT (bench/
        fused_stage.py:48 measures the Jacobian off-diagonals at exactly 0).
        A whole-model cosine over such a pair is 0 no matter what either branch
        is doing -- read as 'orthogonal regime' it would be a pure artifact of
        the freeze flags. So every pair also reports `overlap_*`, the share of
        the pair's gradient energy living in jointly-touched parameters:
        overlap 0 means the cosine is structural and carries NO information
        about cooperation, and `cos_*_shared` is omitted rather than emitted as
        a NaN or a spurious 0. Under mk_dev the one pair that genuinely shares
        the trunk is bwd-vs-replay.

        Membership is by OBSERVED autograd touch (a non-None grad), not by
        module name, so it follows freeze flags and arch changes for free.
        There is deliberately NO branch-private COSINE: 'private' means at most
        one of the pair touched the parameter, so at least one factor of every
        term in that dot product is exactly zero and the cosine is identically
        0 by construction -- a metric that cannot ever say anything. The
        informative form of the same question is per-branch:
        `{k}_uncontested_frac`, the share of branch k's own gradient energy in
        parameters no other active branch touches at all (1.0 for fwd under
        mk_dev, i.e. it contends with nobody).
        """
        active = [k for k in sub_losses if weights.get(k, 0.0) > 0]
        if len(active) < 2:
            return  # nothing to compare a single branch's gradient against

        params = [p for p in self.gfn_model.parameters() if p.requires_grad]
        if not params:
            return

        flat, touched = {}, {}
        for k in active:
            try:
                raw = torch.autograd.grad(sub_losses[k][0], params,
                                          retain_graph=True, allow_unused=True)
            except RuntimeError as e:
                # A DIAGNOSTIC MUST NEVER KILL A TRAINING RUN. This one reads a
                # second backward through a possibly-compiled graph, which is a
                # standing hazard: donated buffers are handled at the compile
                # site above, but the next backend optimisation that assumes
                # one-backward-per-graph would land here the same way -- 50
                # steps into a stage, hours in, on the cluster only.
                #
                # DISABLED LOUDLY AND VISIBLY, not swallowed: the print says
                # what died and why, and `fused_grad/disabled` is logged from
                # here on, because a metric that merely STOPS APPEARING reads as
                # "the diagnostic says nothing is wrong". Nothing else about the
                # step changes -- the real fused backward downstream is
                # untouched, and the run keeps its actual work.
                self._fused_grad_geom_dead = True
                self._fused_grad_geom_report = {'fused_grad/disabled': 1.0}
                print(f"grad_geometry: DISABLED for the rest of this run -- the "
                      f"branch-gradient probe raised on branch '{k}': {e}\n"
                      f"  Training is unaffected (this reads gradients, it does not "
                      f"write them). If the message mentions donated buffers, the "
                      f"compiled trunk was built with donated_buffer on -- see "
                      f"maybe_compile_policy.")
                return
            touched[k] = torch.cat([
                torch.full((p.numel(),), g is not None, dtype=torch.bool, device=p.device)
                for g, p in zip(raw, params)])
            flat[k] = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                                 for g, p in zip(raw, params)])

        def cos(u, v):
            nu, nv = float(u.norm()), float(v.norm())
            return float(torch.dot(u, v) / (nu * nv)) if nu > 0 and nv > 0 else float('nan')

        coef = {k: weights[k] / total_weight for k in active}
        norms = {k: float(flat[k].norm()) for k in active}
        report = {f'fused_grad/{k}_norm_raw': norms[k] for k in active}  # BEFORE weighting

        for k in active:
            others = torch.zeros_like(touched[k])
            for j in active:
                if j != k:
                    others |= touched[j]
            mine = float((flat[k] * ~others).norm())
            report[f'fused_grad/{k}_uncontested_frac'] = (
                (mine / norms[k]) ** 2 if norms[k] > 0 else float('nan'))

        for i, a in enumerate(active):
            for b in active[i + 1:]:
                report[f'fused_grad/cos_{a}_{b}'] = cos(flat[a], flat[b])
                both = touched[a] & touched[b]
                pair_energy = norms[a] ** 2 + norms[b] ** 2
                shared_energy = float((flat[a] * both).norm()) ** 2 + float((flat[b] * both).norm()) ** 2
                report[f'fused_grad/overlap_{a}_{b}'] = (
                    shared_energy / pair_energy if pair_energy > 0 else float('nan'))
                # parameter-disjoint pair: the cosine above is 0 by construction
                # and a 'shared block' cosine would be a cosine of two empty
                # vectors. Emit neither rather than a number that reads as a
                # measurement of orthogonality.
                if bool(both.any()):
                    report[f'fused_grad/cos_{a}_{b}_shared'] = cos(flat[a][both], flat[b][both])

        g_fused = sum(coef[k] * flat[k] for k in active)
        fused_norm = float(g_fused.norm())
        weighted_norm_sum = sum(coef[k] * norms[k] for k in active)
        report['fused_grad/fused_norm'] = fused_norm
        report['fused_grad/weighted_component_norm_sum'] = weighted_norm_sum
        # 1.0 = fully aligned (triangle inequality tight); falls toward 0 as the
        # weighted components cancel each other out (conflicting), and lands
        # around 1/sqrt(n_active) for mutually orthogonal components of equal
        # weighted size -- the ORTHOGONAL regime lives between the two poles.
        report['fused_grad/fused_norm_ratio'] = (
            fused_norm / weighted_norm_sum if weighted_norm_sum > 0 else float('nan'))

        self._fused_grad_geom_report = report

    def _draw_probe_batch(self):
        """
        Draw the batch the step probe scores at all three alpha. FRESH EVERY
        PROBE -- the invariant that matters is 'identical data across the three
        alpha WITHIN one probe', not 'identical across probes'. Only the former
        keeps the second difference free of trajectory/condition noise; the
        latter just goes stale, and re-drawing per probe averages the particular
        draw out for free (a buffer draw costs no energy calls).

        SOURCE: replay, i.e. ON-POLICY ROLLOUTS. Replay rows are fed by the fwd
        branch (to_do_rebuild B2 -- 'it inherits Q's blindness, by intake'), so a
        replay draw IS the on-policy distribution, with stored energies. That is
        the right distribution to probe on because it carries the highest loss
        variance in the system, and a step-size sensor should be read at its
        worst case for stability rather than its most forgiving one. Falls back
        to the backward draw only when replay is unavailable (a bwd-only stage, or an
        empty buffer).

        HELD OUT: a fresh draw is disjoint from this step's training batch in
        expectation, which is the property that matters -- same-batch probing is
        biased high, because a step reduces loss on its own batch more than on
        the population and would systematically license too-large steps.
        """
        # repeats MUST come from the branch whose coeffs we score with: a coeff
        # bank is only valid at its own branch's K. bwd_loss_coeffs carries tbc,
        # whose residual is defined over K same-terminal rollouts and which
        # asserts K > 1 -- so scoring a bwd draw at replay's K crashes on a bwd-only stage,
        # where replay does not exist yet and the fallback is the only path.
        # REPLAY ONLY, and the fallback to bwd is deliberately GONE. The whole
        # calibration rests on evaluating one fixed batch at several alpha, and
        # that requires the batch to carry its STORED trajectories: a replay draw
        # does (return_traj=True), a bwd/dataset draw sets traj = None, so
        # get_gfn_backward_loss would re-sample a fresh backward trajectory at
        # every alpha. The differences would then be dominated by trajectory
        # noise rather than by the step -- pairing silently broken, and the CI
        # would report high confidence in nothing. Returning None instead makes
        # the calibration visibly skip (raycal/skipped) rather than lie.
        if getattr(self, 'replay_buffer', None) is None or len(self.replay_buffer) == 0:
            return None
        k = self.mode_repeats('replay')
        return self.draw_replay_sample(k), self.args.replay_loss_coeffs, 'replay', k

    @torch.no_grad()
    def _probe_loss(self, batch, coeffs, source, discretizer, repeats):
        """
        Re-score the probe batch under the CURRENT parameters. Stored
        trajectories and stored energies, so no resampling and no energy calls.

        FORWARD ONLY -- the whole probe runs under no_grad and builds no graph,
        so it costs three forward passes and zero backward passes.

Two things deliberately NOT done here, both recorded in
        docs/to_do_rebuild.md A4c so they are not re-proposed:

        - Running one evaluation WITH grad to harvest a 'free' gradient. A
          backward is ~2x a forward, so the already-paid forward is the cheap
          third of a forward+backward; the saving is under 1% of training
          compute and it would put probe gradients into the training path.
        - Moving to alpha* instead of restoring to alpha=1. That is a line
          search, a live stage-2 option, but stage 1 restores so the probe
          cannot influence the trajectory it is measuring.

        Must not mutate training state: update_log_z=False is the single gate on
        the condition-log-Z tracker (gflownet_losses.py:432), report_losses=False
        keeps it cheap, and unlike replay_train_step this deliberately does NOT
        call buffer.update_losses -- a probe draw is not a training visit and
        must not count as one for churn, priority, or residence.
        """
        condition, condition_id, _inds, latents, log_reward, mol_batch, traj = batch
        loss, _ = get_gfn_backward_loss(coeffs,
                                        latents.to(self.device),
                                        self.gfn_model,
                                        log_reward.to(self.device),
                                        discretizer,
                                        mol_batch,
                                        condition=condition,
                                        repeats=repeats,
                                        report_losses=False,
                                        trajectories=traj,
                                        condition_log_z=self.condition_log_z,
                                        condition_id=condition_id,
                                        tb_z_source=self.tb_z_source(source),
                                        update_log_z=False,
                                        step=self.step_ind)
        return loss.detach().item()

    def _hyper_sensor_cfg(self, step_type):
        """The stage's hypergradient config, or None if it is not this sensor.

        Gated on the TRAINED step type: the sensor differences the policy's own
        displacement, so it is only meaningful on a step that moved the policy."""
        sensor = self.protocol.stage.lr_sensor
        if sensor is None or sensor.get('kind') != 'hyper':
            return None
        if step_type not in ('fused', 'fwd', 'bwd', 'replay'):
            return None
        if self.step_ind % int(sensor.get('every', 1)):
            return None
        return sensor

    def _hyper_params(self):
        # THE SAME LIST THE RAY PROBE SNAPSHOTS -- policy only, built once where
        # the probe is built. The flow head is LR-pinned separately and excluded
        # there for the same reason (decision D26b); including it would mix a
        # parameter on a different, unservoed rate into the displacement.
        return getattr(self, '_hyper_param_cache', None) or []

    @torch.no_grad()
    def _hyper_flat(self):
        return torch.cat([p.detach().reshape(-1) for p in self._hyper_params()])

    @torch.no_grad()
    def _hyper_apply(self, cfg, clip_ratio=None):
        """cos(current gradient, previous displacement) -> the controller.

        `clip_ratio` is pre-clip grad norm / the guard's bar for this branch,
        passed in rather than read off grad_clip_guard because that object's
        counters are DRAINED at every report and reading them here would race
        the reporter. The controller uses it as a validity gate on cos: once the
        clip binds on essentially every step the update magnitude is set by the
        LR alone and cos stops being a curvature statistic -- see
        LRController._clip_saturated."""
        gs = [p.grad.reshape(-1) for p in self._hyper_params() if p.grad is not None]
        if not gs:
            return
        g = torch.cat(gs)
        d = -self._hyper_prev_step
        if g.numel() != d.numel():
            # parameter set changed under us (a stage rebuilt the optimizers);
            # drop the stale operand rather than difference across it
            self._hyper_prev_step = None
            return
        ng, nd = float(g.norm()), float(d.norm())
        if not (ng > 0 and nd > 0):
            return
        cos = float(torch.dot(g, d) / (ng * nd))
        self.lr_controller.on_hypergradient(cos, cfg['beta'], cfg.get('beta_down'),
                                            cfg.get('cos_target', 0.0),
                                            clip_ratio=clip_ratio)

    def step_loss(self, step_type, loss, do_step: bool = True):
        loss.backward()
        if not do_step:
            return  # mid-accumulation: keep piling up gradients, don't clip/step yet

        # sampled on the same 1-in-10 clock as the rest of ten_step_reporting.
        # When a step runs more than one branch (an unfused stage), this holds
        # the LAST branch to reach an optimizer step, not their sum.
        if self.step_ind % 10 == 0:
            self._last_grad_norms = self._submodel_grad_norms()

        # THE BAR IS PER-BRANCH when grad_clip_guard is enabled: fwd/bwd/replay/
        # fused reach their own optimizer step here and an MLE gradient and a TB
        # gradient are different distributions, so one shared number is set by
        # whichever branch dominates the mixture (grad_clip_guard.py). Disabled =>
        # threshold() returns args.gradient_norm_clip, i.e. exactly the constant
        # this line used to read.
        bar = self.grad_guard.threshold(step_type)
        pre_clip = torch.nn.utils.clip_grad_norm_(
            self.gfn_model.parameters(), bar).item()
        # BEFORE the finiteness gate below, so a non-finite reading is counted by
        # the guard rather than vanishing down that early return. observe()
        # deliberately does not fold it into the bar: it is not a quantile
        # observation, and it is not a training event either since the step is
        # about to be skipped.
        self.grad_guard.observe(step_type, pre_clip)
        if not math.isfinite(pre_clip):
            print(f"Non-finite gradient at {self.step_ind}")
            self._grad_nonfinite += 1  # drained into gradnorm/nonfinite_steps
            # CONSECUTIVE streak, deliberately NOT drained by the reporter:
            # every non-finite step returns here without stepping the optimizer
            # AND without updating last_grad_norm_pre_clip, so a run in this
            # state makes literally zero progress while presenting a STALE
            # finite grad norm to every downstream check. The streak counter is
            # kept as telemetry (gradnorm/nonfinite_steps).
            self._grad_nonfinite_streak = getattr(self, '_grad_nonfinite_streak', 0) + 1
            # A non-finite GRADIENT is the same class of event as a non-finite
            # loss or a 1e9 excursion, so it gets the same response: rewind to the
            # last good checkpoint and cut peak_scale. It cannot be raised from
            # here -- fire_loss_spike reloads model/optimizer state, and this is
            # mid-step, after backward() and before the optimizer step -- so flag
            # it and let monitor_losses fire it at the point the divergence path
            # already runs. Rate-limited there, because the rewind needs a few
            # steps to show whether it took.
            self._nonfinite_pending = True
            # ...and the streak is no longer telemetry ONLY. Measured 2026-08-17 on
            # the QM9 conditional route: 1,579 consecutive non-finite steps from 902
            # to 4,058 -- ~3,150 steps of zero progress with tqdm advancing, the GPU
            # busy and the loss curves smooth, and nothing would have stopped it
            # inside its 13-hour wall clock. check_spike's non-finite trigger cannot
            # catch this: it reads last_grad_norm_pre_clip, which the return below
            # deliberately leaves at its last FINITE value, so the one guard meant to
            # fire is the one this path blinds. Abort instead, same exception and the
            # same "release the GPU" rationale as the rewind-budget path.
            bar = int(getattr(self.args, 'nonfinite_abort_streak', 50))
            if bar > 0 and self._grad_nonfinite_streak >= bar:
                first = self.step_ind - self._grad_nonfinite_streak + 1
                stale = getattr(self, 'last_grad_norm_pre_clip', float('nan'))
                raise FrozenTrainingState(
                    f"UNRECOVERABLE at step {self.step_ind}: {self._grad_nonfinite_streak} "
                    f"consecutive non-finite gradients (since step {first}). The optimizer "
                    f"has not stepped in that window, so the run is making no progress; "
                    f"last_grad_norm_pre_clip is stale at {stale:.4g} and every guard "
                    f"reading it is blind. Aborting so the GPU is released.")
            return  # skip non-finite
        self._grad_nonfinite_streak = 0  # a finite gradient landed: streak broken
        # raw (pre-clip) global grad norm, for reading how hard the clip binds:
        # persistently >> the bar means every step is rescaled and Adam is
        # effectively running on normalized gradients. With grad_clip_guard
        # enabled, gradclip/*_fire_rate answers that directly and per branch;
        # this scalar holds only the LAST branch to step, so it cannot.
        self.last_grad_norm_pre_clip = pre_clip

        # ---- hypergradient sensor: read BEFORE the step, difference AFTER.
        # `cos(g_t, d_{t-1})` -- the current gradient against the direction the
        # PREVIOUS step actually moved the policy. Both operands exist whatever
        # the stage trains, which is the whole reason this sensor is available
        # where `ray` is not (protocol.py::_parse_lr_sensor).
        _hyp = self._hyper_sensor_cfg(step_type)
        if _hyp is not None and not self._hyper_params():
            _hyp = None
        _theta_before = self._hyper_flat() if _hyp else None
        if _hyp is not None and getattr(self, '_hyper_prev_step', None) is not None:
            # pre-clip norm against the bar that was actually applied above: >= 1
            # means the clip fired on this step. The controller cares about the
            # sustained rate, not this one reading.
            #
            # WITHHELD WHILE THE GUARD IS WARMING. During a branch's first
            # `grad_clip_guard.warmup_steps` observations the bar is the STATIC
            # fallback, fitted to nothing in particular, so a high fire rate is
            # about the bar rather than the rate -- and with refresh_on_stage the
            # guard re-warms at every stage transition. None makes the
            # controller's gate inert rather than feeding it evidence that means
            # something else (see GradClipGuard.is_calibrated).
            _ratio = (float(pre_clip) / float(bar)
                      if bar and self.grad_guard.is_calibrated(step_type) else None)
            self._hyper_apply(_hyp, clip_ratio=_ratio)

        self.optimizers[step_type].step()

        if _hyp is not None:
            # the REALISED displacement, not `-lr*g`: read this way it is
            # optimizer-agnostic and cannot drift out of sync with what Adam
            # actually did, which matters because Adam's step direction is
            # mhat/(sqrt(vhat)+eps) and not the gradient.
            self._hyper_prev_step = self._hyper_flat() - _theta_before
        # Non-fused steps: step the standalone flow optimizer here (fwd/bwd/replay run
        # separately, so whichever one had freeze_z=False unambiguously trained Z).
        # Fused steps: skip it -- flow is a param group of optimizers['fused'], so the
        # .step() above already updated Z, using only the grad from whichever sub-loss
        # trained it (the fwd branch); if none did, those params had grad=None and Adam
        # skipped them, so there's no spurious update to guard against.
        if 'flow' in self.optimizers and step_type != 'fused':
            self.optimizers['flow'].step()

    def z_calibration_tick(self, step_type):
        """
        Interspersed Z-only optimizer steps, on top of the ordinary training
        step. Frequency-modulated, never size-modulated: each step taken is a
        plain Adam step at the live flow LR, so there is no discontinuous
        re-level. (Frequency is the axis Adam cannot normalise away; an in-batch
        coefficient like emp_z_persistent is largely cancelled.)

        mode -- the step body:
          rollout      fresh fwd rollout under freeze_policy with the auxiliary
                       Z terms zeroed, so the head gets the fused step's own
                       Huber TB Z gradient. Sensor, actuator and fused loss then
                       share one fixed point. Costs a rollout + energy call.
          replay       same gradient over stored trajectories: no energy call,
                       but Z is calibrated to the buffer's measure, which lags
                       the policy. Raises unless intake and purge are both
                       residual-independent. Not recommended.
          regression   least squares onto the tracker's ema_logw over cached
                       condition embeddings. Nearly free; a mean-family target,
                       so its optimum differs from TB's winsorized one.

        sensor -- the trigger reading, all compared against `threshold`:
          grad_rms     RMS per-condition CLIPPED signed residual = the loss's
                       own first-order condition. Zeroes at the rollout
                       actuator's fixed point, so it cannot latch.
          rms          UNCLIPPED level dispersion. Floors at the winsorized-vs-
                       mean skew gap, i.e. reads a standing offset at the fixed
                       point -- a latch, not convergence.
          worst        upper-tail quantile over conditions (sensor_quantile).
          pooled       |EMA fwd/tb_resid_clipped|. Blind to per-condition
                       disagreement that cancels in the pool.

        Steps taken per train step = min(gain * (sensor/threshold - 1),
        max_steps_per_step), Bernoulli on the fraction, cut short once a
        rollout's own fresh reading falls under threshold * grace (rollout
        mode only).
        """
        cfg = getattr(self.args, 'z_calibration', None)
        # WHICH stages run Z-calibration is a stage property, so it is a stage
        # flag; the z_calibration block holds only HOW. As a global `enabled` it
        # was one of the keys a conditional run had to remember to switch off by
        # hand, and forgetting it drives Z-only steps into a per-condition flow
        # network that no stage has trained.
        if cfg is None or not self.protocol.flag('z_calibration'):
            return
        rep = self._z_cal_report = getattr(self, '_z_cal_report', {})
        rep['z_cal/p'] = 0.0
        if getattr(self, 'fused_accum_count', 0):
            return  # mid-accumulation: a zero_grad here would clobber piled-up grads
        if self.protocol.flag('scramble_conditions'):
            return
        mode = getattr(cfg, 'mode', 'rollout')
        if mode == 'regression' and getattr(self, '_z_cal_cache', None) is None:
            return
        owner = (self.optimizers.get('fused') if step_type == 'fused'
                 else self.optimizers.get('flow'))
        if owner is None:
            return
        sensor_name = getattr(cfg, 'sensor', 'grad_rms')
        if sensor_name == 'grad_rms':
            sensor = self.condition_log_z.rms_z_grad()  # 0.0 when cold
        elif sensor_name == 'rms':
            sensor = self.condition_log_z.rms_z_bias()  # 0.0 when cold
        elif sensor_name == 'worst':
            sensor = self.condition_log_z.worst_z_bias(
                quantile=getattr(cfg, 'sensor_quantile', 0.5))
        else:
            pooled = self.metric_tracker.get('fwd', 'tb_resid_clipped')
            sensor = abs(pooled) if pooled is not None else 0.0
        if not math.isfinite(sensor):
            return
        rep['z_cal/sensor'] = sensor
        threshold = getattr(cfg, 'threshold', 2.0)
        excess = sensor / max(threshold, 1e-9) - 1.0
        if excess <= 0:
            return
        p = min(getattr(cfg, 'gain', 1.0) * excess,
                getattr(cfg, 'max_steps_per_step', 2.0))
        rep['z_cal/p'] = p
        n = int(p)
        if torch.rand(()).item() < p - n:
            n += 1
        # ARM off the EMA'd sensor above, DISARM off each rollout's OWN fresh
        # first-order reading: the sensor is a smoothed decision to start, the
        # early-out is an unsmoothed measurement of whether there is anything
        # left to do. The two cannot be the same reading -- calibration steps
        # deliberately don't feed the rolling metrics (see _z_rollout_step), so
        # a loop keyed on the EMA never observes its own effect and always runs
        # to n. grace < 1 keeps arm and disarm from chattering across the bar.
        #
        # The fresh reading is single-batch and therefore noisy (SE ~ beta /
        # sqrt(batch)), but it does not need to be precise: the tick re-fires
        # every train step, so breaking one step early costs a re-entry and
        # breaking one step late costs one Adam step at lr_flow. Averaging
        # across the tick's steps would be WORSE -- Z moves between them, so
        # the mean is biased toward the pre-catch-up level and would overshoot.
        grace = float(getattr(cfg, 'grace', 0.8))
        bar = threshold * grace
        for _ in range(n):
            if mode == 'rollout':
                ok, fresh = self._z_rollout_step(owner, cfg)
            elif mode == 'replay':
                ok, fresh = self._z_replay_step(owner, cfg)
            else:
                ok, fresh = self._z_calibration_step(owner, cfg), None
            if not ok:
                break
            rep['z_cal/steps'] = rep.get('z_cal/steps', 0) + 1
            # per-STEP tally (the rep counter above is cumulative-since-report).
            # Each rollout/replay step processes a full self.batch_size, so the
            # batch controller needs this to charge the rung for the calibration
            # work it caused -- see select_batch_size.
            self._z_cal_rollouts += 1
            if fresh is not None:
                rep['z_cal/fresh'] = fresh
                if fresh <= bar:
                    # Z is already inside the bar at the level this step STARTED
                    # from, so the step just taken was the last one worth paying
                    # for. Stale-EMA overshoot costs 1 rollout/train step here
                    # instead of max_steps_per_step of them.
                    rep['z_cal/early_out'] = rep.get('z_cal/early_out', 0) + 1
                    break

    def _z_rollout_step(self, owner, cfg):
        """
        One Z-only step on a FRESH forward rollout, under freeze_policy with the
        auxiliary Z terms (z_level, z_var, emp_z, emp_z_persistent) zeroed, so
        the flow head receives exactly the fused step's own Huber TB Z gradient
        on new on-policy data. Sensor, actuator and fused loss share one fixed
        point by construction.

        DOES feed replay-buffer intake -- one manage_replay_buffer call per
        step. A reward call is the expensive thing, so a rollout that has paid
        for one is never discarded unconsidered. Cost: churn_rate is a
        per-CALL budget, so a saturated tick admits up to
        (1 + max_steps_per_step) x churn_rate in one train step. Watch
        replay_buffer_admitted if the buffer starts reading as one instant.

        Does NOT update the rolling metrics: fwd/* drives the balance
        controller and must stay a reading of the TRAIN batch stream, not of an
        instrument firing at its own rate.

        The first call asserts the gradient really is confined to flow_model
        and raises otherwise -- the freeze_policy contract, checked once rather
        than trusted.

        Returns (ok, fresh). `fresh` = |mean clip(resid, +/-beta)| on this
        rollout's own batch, i.e. the loss's first-order condition in Z at the
        level this step STARTED from, clipped at the SAME beta the step's loss
        uses. Consumed by z_calibration_tick's early-out. None when ok is False.
        """
        coeffs = copy.deepcopy(self.args.fwd_loss_coeffs)
        coeffs.freeze_policy = 1.0
        for k in ('z_level', 'z_var', 'emp_z', 'emp_z_persistent'):
            if hasattr(coeffs, k):
                setattr(coeffs, k, 0.0)
        saved = self.args.fwd_loss_coeffs
        self.args.fwd_loss_coeffs = coeffs
        try:
            # report_losses=True for the early-out reading below and for the
            # churn call's fwd_stats; return_exp=True for the crystal batch the
            # buffer stores (a D2H copy -- see the REVISIT note).
            loss, crystal_batch, loss_dict = self.fwd_train_step(
                get_discretizer(self.args.integrator),
                return_exp=True,
                repeats=self.mode_repeats('fwd'),
                report_losses=True)
        except (RuntimeError, ValueError):
            self._z_cal_report['z_cal/rollout_errors'] = (
                self._z_cal_report.get('z_cal/rollout_errors', 0) + 1)
            return False, None
        finally:
            self.args.fwd_loss_coeffs = saved

        # churn BEFORE the requires_grad bail: the reward call is paid either
        # way, and these samples are on-policy regardless of whether this step
        # can train Z. Everything the buffer reads is already detached.
        if loss_dict is not None and self.protocol.flag('buffers_active'):
            self.manage_replay_buffer(loss_dict, crystal_batch)
        del crystal_batch

        if not loss.requires_grad:
            return False, None  # e.g. a stage running fwd freeze_z: nothing trains Z

        fresh = None
        if loss_dict is not None:
            with torch.no_grad():
                resid = ((loss_dict['log_pf'] + loss_dict['log_Z'])
                         - (loss_dict['log_pb'] + loss_dict['log_r']))
                fresh = float(resid.clamp(-coeffs.beta, coeffs.beta).mean().abs())
        owner.zero_grad(set_to_none=True)
        loss.backward()
        if not getattr(self, '_z_rollout_grads_verified', False):
            flow_ids = {id(p) for p in self.gfn_model.flow_model.parameters()}
            stray = [name for name, p in self.gfn_model.named_parameters()
                     if p.grad is not None and id(p) not in flow_ids]
            if stray:
                raise RuntimeError(
                    "z_calibration rollout step leaked gradient outside "
                    f"flow_model: {stray[:5]}")
            self._z_rollout_grads_verified = True
        torch.nn.utils.clip_grad_norm_(self.gfn_model.flow_model.parameters(),
                                       self.args.gradient_norm_clip)
        owner.step()
        owner.zero_grad(set_to_none=True)
        self._z_cal_report['z_cal/rollout_loss'] = float(loss.detach().cpu())
        return True, fresh

    def _z_replay_step(self, owner, cfg):
        """
        One Z-only step over a REPLAY draw: the same Huber TB Z gradient as
        rollout mode but over stored trajectories, so no policy rollout and no
        energy call.

        ⚠ Calibrates Z to the BUFFER's measure. TB's Z fixed point belongs to
        the measure the batch was drawn from, and the buffer differs from the
        on-policy stream in two ways:
          - scored admission skims the residual tail (raises rather than run)
          - even under uniform intake it LAGS the policy by ~tau steps
        The lag alone mis-centres the fused fwd loss, which reads the same
        log_Z. Not recommended; rollout mode is the reference.

        Draws read-only (side_effects=False): this tick fires at its own rate,
        and its corrections would otherwise reach the memorisation sensor as
        absorption the training branch never performed.

        Returns (ok, fresh) as _z_rollout_step does.
        """
        if not len(getattr(self, 'replay_buffer', []) or []):
            return False, None
        if self.replay_priority_config() is None:
            raise ValueError(
                "z_calibration.mode: replay requires buffers.replay_buffer.prioritise."
                "enabled -- without it the buffer runs SCORED admission (softmax over "
                "clipped |resid|), which skims the residual tail, so Z would be "
                "calibrated to that tail rather than to the on-policy mean. Use mode: "
                "rollout, or turn uniform intake on.")
        coeffs = copy.deepcopy(self.args.replay_loss_coeffs)
        coeffs.freeze_policy = 1.0
        coeffs.freeze_z = 0.0        # base config freezes Z on replay; this step IS the Z step
        for k in ('z_level', 'z_var', 'emp_z', 'emp_z_persistent', 'mle', 'vg_lb', 'vg_lme'):
            if hasattr(coeffs, k):
                setattr(coeffs, k, 0.0)
        saved = self.args.replay_loss_coeffs
        self.args.replay_loss_coeffs = coeffs
        try:
            loss, loss_dict = self.replay_train_step(
                get_discretizer(self.args.integrator),
                repeats=self.mode_repeats('replay'),
                report_losses=True,
                # read-only on the buffer: see replay_train_step's docstring.
                # This tick fires at its own rate, and its corrections would
                # otherwise reach the memorisation sensor as absorption the
                # training branch never performed.
                side_effects=False)
        except (RuntimeError, ValueError):
            self._z_cal_report['z_cal/replay_errors'] = (
                self._z_cal_report.get('z_cal/replay_errors', 0) + 1)
            return False, None
        finally:
            self.args.replay_loss_coeffs = saved

        if not loss.requires_grad:
            return False, None
        fresh = None
        if loss_dict is not None:
            with torch.no_grad():
                resid = ((loss_dict['log_pf'] + loss_dict['log_Z'])
                         - (loss_dict['log_pb'] + loss_dict['log_r']))
                fresh = float(resid.clamp(-coeffs.beta, coeffs.beta).mean().abs())
        owner.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gfn_model.flow_model.parameters(),
                                       self.args.gradient_norm_clip)
        owner.step()
        owner.zero_grad(set_to_none=True)
        self._z_cal_report['z_cal/replay_loss'] = float(loss.detach().cpu())
        return True, fresh

    def _z_calibration_step(self, owner, cfg):
        """
        One Z-only calibration step: weighted least squares of flow_model's
        log_Z(c) onto condition_log_z.ema_logw over the cached fwd-batch
        conditions (targets/weights: ConditionLogZTracker.calibration_targets).

        Only flow_model params ever receive gradients -- the cached embedding
        is detached -- so owner.step() moves nothing else (grad=None params
        are skipped by Adam entirely: no moment decay, no movement) and the
        flow group's Adam state stays the single warm one the training loop
        itself uses. Conditions with id % holdout_modulus == 0 are excluded
        from the loss and reported as an out-of-sample residual
        (z_cal/holdout_rms vs z_cal/train_rms: a growing gap means the head is
        fitting tracker noise rather than the level field).

        Returns False when nothing trustworthy was available to regress to.
        """
        emb, ids = self._z_cal_cache
        tgt, w = self.condition_log_z.calibration_targets(
            ids, step=self.step_ind,
            min_visits=getattr(cfg, 'min_visits', None),
            freshness_half_life_steps=getattr(cfg, 'freshness_half_life_steps', 300.0),
            se2_floor=getattr(cfg, 'se2_floor', 0.25))
        holdout_modulus = getattr(cfg, 'holdout_modulus', 8)
        holdout = ((ids % holdout_modulus == 0) if holdout_modulus > 0
                   else torch.zeros_like(ids, dtype=torch.bool))
        train_w = torch.where(holdout, torch.zeros_like(w), w)
        wsum = train_w.sum()
        if wsum <= 0:
            return False
        owner.zero_grad(set_to_none=True)
        pred = self.gfn_model.flow_model(emb).flatten()
        resid = pred - tgt.to(self.device)
        loss = ((train_w / wsum).to(self.device) * resid.pow(2)).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gfn_model.flow_model.parameters(),
                                       self.args.gradient_norm_clip)
        owner.step()
        owner.zero_grad(set_to_none=True)
        rep = self._z_cal_report
        with torch.no_grad():
            r2 = resid.detach().pow(2).cpu()
            tr = (~holdout) & (w > 0)
            ho = holdout & (w > 0)
            if tr.any():
                rep['z_cal/train_rms'] = float(torch.sqrt(
                    (r2[tr] * w[tr]).sum() / w[tr].sum()))
                rep['z_cal/n_conditions'] = int(tr.sum())
            if ho.any():
                rep['z_cal/holdout_rms'] = float(torch.sqrt(
                    (r2[ho] * w[ho]).sum() / w[ho].sum()))
        return True

    def fwd_train_step(self,
                       discretizer,
                       return_exp=False,
                       repeats: int = 1,
                       report_losses: bool = False,
                       ):
        cfg = getattr(self.args, 'condition_log_z', None)
        p = None
        # weighted condition sampling is a stage flag (weighted_condition_sampling):
        # it belongs to stages where forward trains the POLICY, so steering at
        # badly-fit conditions reduces their Var(log w) -- it fixes the root
        # cause. In a Z-only forward stage (policy frozen), high fit-error is
        # exactly where per-trajectory TB gives Z the WORST gradient, and
        # forward can't lower that variance anyway -- weighting there aims the
        # weak lever at its worst conditions while starving the clean-gradient
        # bulk. Declare the flag per-stage to A/B it.
        if (self.protocol.flag('weighted_condition_sampling')
                and getattr(cfg, 'weighted_condition_sampling', False)):
            p = self.weighted_condition_sampling(
                temperature=getattr(cfg, 'weighted_condition_sampling_temperature', 0.5),
                clip_quantile=getattr(cfg, 'weighted_condition_sampling_clip_quantile', 0.99))
        mol_batch = next(self.mol_dataset.loader(
            self.batch_size, mode='graphs', repeats=repeats, p=p,
            beta=getattr(cfg, 'weighted_condition_sampling_uniform_beta', 0.0) if p is not None else None))
        mol_batch = mol_batch.to(self.device)
        mol_batch.orient_molecule(mode='std')
        init_state = get_gfn_init_state(mol_batch.num_graphs, self.energy_function.data_ndim, self.device)
        mol_batch, log_T_tensor, condition, condition_id = self.energy_function.condition_samples(
            mol_batch, repeats=repeats)

        out = get_gfn_forward_loss(self.args.fwd_loss_coeffs,
                                   init_state,
                                   self.gfn_model,
                                   self.energy_function.log_reward,
                                   discretizer,
                                   mol_batch,
                                   log_T_tensor,
                                   exploration_std=None,
                                   return_exp=return_exp,
                                   condition=condition,
                                   repeats=repeats,
                                   report_losses=report_losses,
                                   condition_log_z=self.condition_log_z,
                                   condition_id=condition_id,
                                   tb_z_source=self.tb_z_source('fwd'),
                                   step=self.step_ind,
                                   )
        self._stash_z_cal_cache(condition_id)
        return out

    def _stash_z_cal_cache(self, condition_id):
        """
        Pair the detached condition embedding the model stashed during the fwd
        rollout that just ran (gfn._z_cal_embedding, written in get_traj_fwd)
        with that rollout's condition_ids, reduced to one row per unique
        condition (repeats/tiling broadcast identical embedding rows per id).
        Consumed by z_calibration_tick, which re-feeds the cached embeddings to
        flow_model between rollouts -- so calibration steps never need their
        own rollout or conditioner pass. The cache is at most one train step
        stale in embedding terms (conditioner params move under it), which is
        the same one-step staleness every other use of the batch tolerates.
        """
        cfg = getattr(self.args, 'z_calibration', None)
        if cfg is None or not getattr(cfg, 'enabled', False):
            return
        emb = getattr(self.gfn_model, '_z_cal_embedding', None)
        if emb is None or condition_id is None:
            return
        ids = condition_id.detach().to(emb.device).flatten()
        if ids.shape[0] != emb.shape[0]:
            return  # unexpected pairing; drop rather than mis-pair
        uniq, inverse = torch.unique(ids, return_inverse=True)
        first = torch.full((uniq.shape[0],), ids.shape[0],
                           dtype=torch.long, device=emb.device)
        first.scatter_reduce_(0, inverse,
                              torch.arange(ids.shape[0], device=emb.device),
                              reduce='amin', include_self=True)
        self._z_cal_cache = (emb[first], uniq.cpu())

    def scramble_applicable(self):
        """
        Structural applicability guard for the condition-embedding scramble
        (the stage flag scramble_conditions decides intent; this decides
        whether the mechanism can apply at all): the model must be conditional,
        and on vector conditions only -- under conditions_type='molecule'
        erasing molecule identity isn't the goal. The scramble itself lives
        inside the model (GFN._maybe_scramble_condition_embedding: conditioner
        runs on the TRUE pairing so the trunk sees real-scale embeddings, then
        its detached output rows are tile-permuted at the conditioner->trunk
        seam), so the trunk is actively trained to ignore the embedding and the
        stage's product is the unconditional mixture by construction rather
        than by MLE's empirical insensitivity to conditioning. No scrambled
        tensor ever exists outside the model, so condition/condition_id pairing
        in buffers, trackers, and per-sample losses is structurally safe.
        """
        return self.gfn_model.conditional and self.gfn_model.conditions_type == 'vector'

    def bwd_train_step(self,
                       discretizer,
                       repeats: int,
                       report_losses: bool = False):

        condition, condition_id, inds, latents, log_reward, mol_batch, traj = self.draw_bwd_sample(repeats)

        # unconditional-prior training: the scramble lives INSIDE the model, at the
        # conditioner->trunk seam (see GFN._maybe_scramble_condition_embedding) --
        # train.py hands down only the K-tile size. `condition`/`condition_id` here
        # stay correctly paired everywhere they flow: buffers, the tracker update
        # inside the loss, and the per-sample resids written back below.
        scramble_tiles = repeats if (self.protocol.flag('scramble_conditions')
                                     and self.scramble_applicable()) else 0

        loss, loss_dict = get_gfn_backward_loss(self.args.bwd_loss_coeffs,
                                                latents.to(self.device),
                                                self.gfn_model,
                                                log_reward.to(self.device),
                                                discretizer,
                                                mol_batch,
                                                condition=condition,
                                                scramble_condition_tiles=scramble_tiles,
                                                repeats=repeats,
                                                report_losses=report_losses,
                                                trajectories=traj,
                                                condition_log_z=self.condition_log_z,
                                                condition_id=condition_id,
                                                tb_z_source=self.tb_z_source('bwd'),
                                                update_log_z=self.protocol.flag('update_log_z'),
                                                step=self.step_ind,
                                                # J_B side of the z_match delta gate; a scrambled
                                                # embedding's log_pf is not the conditional
                                                # policy's level, so skip the feed there. Replay
                                                # deliberately never feeds either stream.
                                                mode_level_stream=None if scramble_tiles else 'bwd')

        priority = self._bwd_retention_priority(loss_dict)
        if self.bwd_sampling_mode == 'dataset':
            self.prior_dataset.update_losses(priority, inds)
        elif self.bwd_sampling_mode == 'prior':
            self.prior_buffer.update_losses(priority, inds)

        return loss, loss_dict

    def _bwd_retention_priority(self, loss_dict):
        """
        Per-sample priority for prior/dataset buffer retention (purge_lowest
        keeps the HIGH-priority samples), centered on the tracker's
        per-condition mean (ema_logw) -- the buffer's OWN normalizer -- in
        every stage, not just the terminal one. Rationale (same as the re-centered
        bwd under_coverage metric, see _bwd_under_center): whenever the
        learned Z lags the buffer-implied level -- the standing condition of
        any fused stage, not just the untrained-Z init the terminal stage starts
        from -- |log_Z - log_w| is
        dominated by that collective offset, so ranking by it degenerates to
        ranking by log_w alone (one-sided) and skews retention to
        off-policy/blowup tails instead of the samples that define each
        condition's Var(log w) spread. The centered two-sided ranking keeps
        the spread-defining samples in every regime, and coincides with the
        raw ranking exactly when Z has genuinely caught up. Falls back to raw
        resid where condition_id is absent or the tracker isn't warmed for
        that condition.
        """
        resid = loss_dict['resid']
        cid = loss_dict.get('condition_id')
        if cid is not None and hasattr(self, 'condition_log_z'):
            center, mask = self.condition_log_z.lookup(cid)
            log_w = loss_dict['log_r'] + loss_dict['log_pb'] - loss_dict['log_pf']
            centered = log_w - center.to(log_w.device)
            resid = torch.where(mask.to(log_w.device), centered, resid)
        return resid.abs()

    def replay_train_step(self,
                          discretizer,
                          repeats: int,
                          report_losses: bool = False,
                          side_effects: bool = True):
        """side_effects=False draws and scores WITHOUT writing back to the
        buffer or the metric tracker. Used by z_calibration's replay mode, which
        fires at its own rate and must not be mistaken for training.

        The three writes below are not bookkeeping, they are the inputs to two
        other controllers: `ema_loss` and `ema_logw` set the prioritised draw
        and the displacement purge, and `ema_loss` against the frozen
        `birth_loss` IS the memorisation sensor (module_buffers.md B8). A
        calibration tick correcting rows at its own cadence would show up there
        as absorption the training branch did not do, so the servo would read a
        memorisation signal manufactured by the instrument watching for it."""

        condition, condition_id, inds, latents, log_reward, mol_batch, traj = self.draw_replay_sample(repeats)

        loss, loss_dict = get_gfn_backward_loss(self.args.replay_loss_coeffs,
                                                latents.to(self.device),
                                                self.gfn_model,
                                                log_reward.to(self.device),
                                                discretizer,
                                                mol_batch,
                                                condition=condition,
                                                repeats=repeats,
                                                report_losses=report_losses,
                                                trajectories=traj,
                                                condition_log_z=self.condition_log_z,
                                                condition_id=condition_id,
                                                tb_z_source=self.tb_z_source('replay'),
                                                update_log_z=self.protocol.flag('update_log_z'),
                                                step=self.step_ind,
                                                sample_weights=self._replay_is_w)

        if not side_effects:
            return loss, loss_dict

        self.replay_buffer.update_losses(loss_dict['resid'].abs(), inds)
        # Refresh the per-row log w EMA. This is what makes the prioritised draw
        # possible at all -- prioritised_weights reconstructs the SIGNED residual
        # as log_Z - ema_logw, and ema_loss (|resid|) cannot supply the sign.
        # The field and its checkpoint round-trip already existed; nothing
        # called it until now.
        logw = (loss_dict['log_r'] + loss_dict['log_pb'] - loss_dict['log_pf']).detach()
        self.replay_buffer.update_logw_stats(logw, inds)

        # Memorisation sensor -> the METRIC TRACKER, not just the wandb metrics
        # dict: buffer_servo resolves its sensor through metric_tracker.get, so
        # a stat that only reaches the report path is invisible to it.
        # On the 10-step metric cadence, like every other rolling stat -- it is
        # a mean over all resident rows, and the only consumer (the servo) ticks
        # every 10 steps.
        if self.step_ind % 10 == 0 and (st := self.replay_buffer.absorption_stats()):
            self.metric_tracker.update(
                'replay',
                {'ema_loss_mean': st['replay/ema_loss_mean'],
                 'birth_loss_mean': st['replay/birth_loss_mean'],
                 'resid_vs_intake': st['replay/resid_vs_intake'],
                 'lambda_tau': st['replay/lambda_tau']},
                self.step_ind)

        return loss, loss_dict

    @torch.no_grad()
    def draw_bwd_sample(self, repeats):
        traj = None
        if self.bwd_sampling_mode == 'dataset':
            mol_batch, inds = next(
                self.prior_dataset.loader(
                    batch_size=self.batch_size, mode='graphs',
                    repeats=repeats, return_inds=True,
                    weighted=False,
                    temperature=0.1, beta=1.0))

            latents = self._batch_latents(mol_batch)
            latents = latents.to(self.device)

        elif self.bwd_sampling_mode == 'prior':
            # condition-blocked draws (C conditions x up to M distinct terminals
            # each) only while condition-grouped bwd VarGrad is active (var_conditioning:
            # vg_lb = phase2_bwd_vg_lb; _activate_phase3_losses turns it off) --
            # its cross-terminal signal otherwise only arrives via birthday
            # collisions. Phase 3's per-sample TB prefers the broad-coverage
            # independent draws, which block_m = 0 restores automatically.
            blc = self.args.bwd_loss_coeffs
            block_m = int(getattr(blc, 'condition_block_m', 0) or 0) \
                if getattr(blc, 'vg_lb', 0) > 0 else 0
            # gentle loss-weighted draw when the stage sets weighted_bwd_sampling:
            # tilt a small slice of the batch toward high-residual conditions via
            # the buffer's own ema_loss (the _bwd_retention_priority signal), so a
            # lagging condition draws extra bwd gradient without starving the rest.
            # beta is the UNIFORM fraction (floor): 0.9 => 10% of the batch
            # loss-weighted, 90% uniform. block_m draws bypass weighting entirely.
            weighted_bwd = self.protocol.flag('weighted_bwd_sampling')
            bwd_beta = getattr(self.args.buffers.prior_buffer, 'weighted_bwd_beta', 0.9) \
                if weighted_bwd else 1.0
            mol_batch, inds = next(
                self.prior_buffer.loader(
                    batch_size=self.batch_size, mode='graphs',
                    repeats=repeats, return_inds=True,
                    weighted=weighted_bwd,
                    temperature=0.5, beta=bwd_beta,
                    condition_block_m=block_m))

            latents = self._batch_latents(mol_batch)
            latents = latents.to(self.device)
        else:
            assert False, f"sampling method {self.args.sampling} not implemented"
        mol_batch = mol_batch.to(self.device)
        mol_batch, log_T_tensor, condition, condition_id = self.energy_function.condition_samples(
            mol_batch, repeats=repeats)
        temperature = 10 ** log_T_tensor
        log_reward = self.energy_function.prebuilt_sample_to_reward(mol_batch,
                                                                    temperature)  # relies on the energy terms being attached to the graphs!

        return condition, condition_id, inds, latents, log_reward, mol_batch, traj

    @torch.no_grad()
    @torch.no_grad()
    def current_log_z(self):
        """
        Live log Z_learned as a plain float, without a rollout -- the
        prioritised draw needs it BEFORE any trajectory is scored, to turn the
        buffer's stored ema_logw into a signed residual (delta = log_Z - log_w).

        Unconditional route: flow_model is a LearnableScalar, so calling it with
        no arguments returns the scalar. Anything else (a conditional FlowModel
        needing an embedding) returns None, and the caller degrades to a uniform
        draw rather than guessing a level.
        """
        try:
            v = self.gfn_model.flow_model()
        except Exception:
            return None
        if v is None:
            return None
        v = torch.as_tensor(v).detach().flatten()
        if v.numel() != 1 or not torch.isfinite(v).all():
            return None
        return float(v[0])

    def replay_priority_config(self):
        """kappa for the prioritised replay draw; None/<=0 disables it and the
        draw stays uniform. Absent config block => off, so this is inert for
        every existing config."""
        cfg = getattr(self.args.buffers.replay_buffer, 'prioritise', None)
        if cfg is None or not bool(getattr(cfg, 'enabled', False)):
            return None
        return float(getattr(cfg, 'kappa', 1.0))

    def replay_priority_symmetric(self):
        """`prioritise.symmetric` (default False): draw on |delta|^kappa instead
        of delta_plus^kappa, admitting UNDER-weighted rows (delta < 0) into the
        eligible set instead of zeroing them. See F-003 (docs/findings.md) --
        the default one-sided draw measurably leaves the forward tail
        uncorrected; this is the untested alternative it names. Absent =>
        False, so inert for every existing config."""
        cfg = getattr(self.args.buffers.replay_buffer, 'prioritise', None)
        return bool(getattr(cfg, 'symmetric', False))

    def draw_replay_sample(self, repeats):
        # Condition-blocked draw (C conditions x up to M distinct terminals
        # each), same mechanism as draw_bwd_sample and active only while
        # condition-grouped replay VarGrad is. Without it that branch is a SILENT
        # NO-OP: vg_loss is identically zero on singleton groups, and a uniform
        # draw groups rows only by birthday collision at a rate ~ batch_size /
        # library -- per-condition occupancy and buffer size cancel, so a bigger
        # buffer does NOT raise it. Zero loss, zero gradient, no error.
        # The gate also tests vg_by_condition, unlike draw_bwd_sample's narrower
        # vg_lb-only test: bwd's base block pins vg_by_condition at 1.0 while
        # replay's defaults to 0, and the legacy repeats-grouped estimator does
        # not read this grouping (replay repeats is 1 by design -- stored
        # trajectories, so K copies would be identical -- which makes that path
        # singleton-tiled and equally dead).
        rlc = self.args.replay_loss_coeffs
        block_m = int(getattr(rlc, 'condition_block_m', 0) or 0) \
            if (getattr(rlc, 'vg_by_condition', 0) > 0.5
                and (getattr(rlc, 'vg_lb', 0) > 0 or getattr(rlc, 'vg_lme', 0) > 0)) else 0
        if block_m >= 2 and not hasattr(self.replay_buffer.batch, 'condition_id'):
            raise ValueError(
                "replay_loss_coeffs.condition_block_m needs condition_id on the "
                "stored batch. "
                "manage_replay_buffer admits the post-condition_samples forward "
                "batch, which carries it, so a buffer without it was built by "
                "another route.")

        # Prioritised-IS draw (docs/to_do_rebuild.md B5): p ~ delta_plus^kappa
        # (or |delta|^kappa under prioritise.symmetric) with the row weights
        # that undo it. delta is reconstructed from the buffer's ema_logw
        # against the live log Z -- signed, which |resid| is not.
        # self._replay_is_w carries the drawn rows' weights to
        # replay_train_step.
        kappa = self.replay_priority_config()
        p = None
        self._replay_is_w = None
        # Cleared rather than left to persist: log_metrics merges
        # _replay_is_stats by getattr on every call, so a blocked draw inheriting
        # the previous prioritised draw's ESS would keep reporting a healthy
        # prioritised estimator that is no longer running.
        self._replay_is_stats = {}
        if kappa is not None and block_m >= 2:
            # Not a silent precedence rule: _sample_indices returns from the
            # blocked branch BEFORE p is consulted, so the draw would come from
            # the blocked measure while the IS weights below still divided by p
            # -- the same inverse-measure error as the beta=1.0 bug documented
            # further down, and just as invisible in the loss.
            raise ValueError(
                "replay_loss_coeffs.condition_block_m >= 2 is mutually "
                "exclusive with buffers.replay_buffer.prioritise: blocked draws "
                "bypass `p` entirely in CrystalBuffer._sample_indices. Set "
                "prioritise.enabled false or condition_block_m 0.")
        if kappa is not None:
            log_z = self.current_log_z()
            if log_z is not None:
                p, w_row = self.replay_buffer.prioritised_weights(
                    log_z, kappa=kappa, symmetric=self.replay_priority_symmetric())

        # beta is the fraction of the batch drawn UNIFORMLY, not a temperature:
        # _sample_indices splits the batch into n_uniform = batch*beta and
        # n_weighted = batch - n_uniform. The legacy call passed beta=1.0, which
        # is 100% uniform -- so a supplied `p` was silently ignored.
        #
        # That is fatal for the prioritised estimator rather than merely
        # suboptimal: the IS weights w ~ 1/delta_plus^kappa were still applied to
        # the loss, so a UNIFORM draw carrying 1/p weights targets a measure
        # ~ 1/delta^kappa -- the INVERSE of the intended one, up-weighting the
        # lowest-residual rows. It also drew ineligible (w = 0) rows, which is
        # why is_ess_frac tracked is_elig_frac exactly (0.40 vs 0.39 at kappa=0).
        #
        # With a computed p the draw must come entirely from it.
        draw_beta = 0.0 if p is not None else 1.0
        mol_batch, traj, inds = next(
            self.replay_buffer.loader(
                batch_size=self.batch_size, mode='graphs',
                repeats=repeats, return_inds=True,
                weighted=False,
                temperature=0.1, beta=draw_beta,
                return_traj=True, p=p,
                condition_block_m=block_m))

        if p is not None:
            wb = np.asarray(w_row)[np.asarray(inds).ravel()]
            self._replay_is_w = torch.as_tensor(wb, dtype=torch.float32)
            # ESS of the BATCH weights is the load-bearing diagnostic: the
            # estimator is unbiased at every kappa by construction, so the only
            # thing that can go wrong is variance -- and it goes wrong through
            # a heavy weight tail, not through the mean. ess_frac near 1 is
            # near-uniform; near 0 means a handful of rows own the batch.
            s1, s2 = float(wb.sum()), float((wb ** 2).sum())
            self._replay_is_stats = {
                'replay/is_ess_frac': (s1 * s1 / s2 / len(wb)) if s2 > 0 else 1.0,
                'replay/is_w_max_ratio': float(wb.max() / max(wb.mean(), 1e-12)),
                'replay/is_elig_frac': float((np.asarray(p) > 0).mean()),
            }

        latents = self._batch_latents(mol_batch)
        latents = latents.to(self.device)
        traj = traj.to(self.device)

        mol_batch = mol_batch.to(self.device)
        mol_batch, log_T_tensor, condition, condition_id = self.energy_function.condition_samples(
            mol_batch, repeats=repeats)
        temperature = 10 ** log_T_tensor
        log_reward = self.energy_function.prebuilt_sample_to_reward(mol_batch,
                                                                    temperature)  # relies on the energy terms being attached to the graphs!
        return condition, condition_id, inds, latents, log_reward, mol_batch, traj

    def handle_train_epoch_error(self, e, step_type):
        """
        Single shared OOM recovery path for every VRAM-bound loop (train fwd/bwd/replay/
        fused steps AND eval sampling all call this) -- there's one batch_size and one
        recovery policy, rather than several independently-tuned loops that can OOM at
        different, decorrelated moments. On OOM: zero all grads, free what we can, cut
        batch_size multiplicatively, and start a cooldown (see select_batch_size).
        """
        print(f"Caught error during '{step_type}' step: {str(e)}")
        if not is_cuda_oom(e):
            raise e  # will simply raise error if other or if training on CPU

        print("OOMED!")
        # counted BEFORE the step_ind == 0 bail below, and for eval OOMs as well as
        # train ones: this is the record that an allocation failed, not the record of
        # what the controller did about it. Which of the two it was is recoverable from
        # whether batch/oom_ceiling moved on the same step.
        self.batch_oom_events = getattr(self, 'batch_oom_events', 0) + 1
        if self.step_ind == 0:
            return

        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)
        self.fused_accum_count = 0  # wiped along with the gradients above

        # break reference cycles
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

        # the size that just failed is a hard ceiling for this stage. Clearing the
        # baseline below re-arms the growth walk, and without a remembered ceiling
        # that walk climbs straight back into this same OOM -- prod0810 mipcas_elj
        # ran that sawtooth (6113->10086->OOM->5043->8321->OOM->4160->...) for the
        # rest of the run, burning a step plus a gc/empty_cache cycle each lap.
        #
        # TRAIN STEPS ONLY. This function is the shared recovery path for eval too
        # (eval_bwd, eval_fwd, anchor_refresh), and eval has a different memory
        # profile entirely: eval_num_samples per pass, the EMA model, no gradients.
        # Recording the TRAIN batch as a ceiling because an EVAL pass overflowed
        # installs a stage-lifetime cap that only a stage transition clears, on
        # evidence about a different allocation. Eval OOMs are self-limiting -- their
        # loops cut and retry -- so the cut below still applies to them; only the
        # ceiling is withheld. Gated on protocol.TRAIN_MODES rather than a list of
        # eval names so a new eval call site cannot opt itself in by accident.
        if step_type in TRAIN_MODES:
            oomed_at = self.batch_size
            # the stage's all-time smallest OOM, which the expiry does NOT clear. The
            # ceiling is re-derived from it so that a re-probe -- which climbs from
            # below and therefore re-OOMs a little HIGHER than the original -- cannot
            # ratchet the ceiling upward and forget the smallest size known not to fit.
            prior_min = getattr(self, 'batch_size_oom_min', None)
            self.batch_size_oom_min = (oomed_at if prior_min is None
                                       else min(prior_min, oomed_at))
            prior_ceiling = getattr(self, 'batch_size_oom_ceiling', None)
            self.batch_size_oom_ceiling = (self.batch_size_oom_min if prior_ceiling is None
                                           else min(prior_ceiling, oomed_at))
            # restart the expiry clock on every OOM, whether or not the ceiling
            # moved: a ceiling that keeps being re-confirmed must keep standing.
            self.batch_size_oom_ceiling_at = self.step_ind
        self.batch_size = max(1, int(self.batch_size * self.args.oom_batch_shrink_factor))
        self.batch_size_cooldown_until = self.step_ind + self.args.oom_cooldown_steps
        # whatever the sizer measured or concluded was at sizes the new ceiling now
        # bounds -- re-run the ladder under it after the cooldown. Never grow blind:
        # every rung is re-measured before it is held.
        self.batch_sizer = None
        # timings from the pre-cut rung would be scored against the post-cut batch
        if getattr(self, '_recent_step_times', None) is not None:
            self._recent_step_times.clear()
            self._recent_step_work.clear()
        if self.batch_size <= 1:
            raise RuntimeError("Cascading OOM Failure")
        print(f"Reducing batch size to {self.batch_size}")

    @torch.no_grad()
    def bwd_eval_sampling(
            self,
            discretizer, ):
        acc = defaultdict(list)
        samples = 0
        while samples < self.args.eval_num_samples:
            try:
                if self.bwd_sampling_mode == 'dataset':  # todo consider mixing, or adding anchor states
                    mol_batch = next(self.prior_dataset.loader(batch_size=self.batch_size, mode='graphs'))
                elif self.bwd_sampling_mode == 'prior':
                    mol_batch = next(self.prior_buffer.loader(batch_size=self.batch_size, mode='graphs'))
                else:
                    assert False

                mol_batch = mol_batch.to(self.ema_model.device)
                terminal_state = self._batch_latents(mol_batch)

                mol_batch, log_T_tensor, condition, condition_id = self.energy_function.condition_samples(
                    mol_batch,
                    temperature=torch.ones(mol_batch.num_graphs, dtype=torch.float32,
                                           device=mol_batch.device) * self.args.energy_config.temperature)

                log_r = self.energy_function.prebuilt_sample_to_reward(mol_batch,
                                                                       temperature=10 ** log_T_tensor)

                terminal_state = terminal_state.to(self.ema_model.device)
                condition = condition.to(self.ema_model.device)

                (backward_flow_states, b_log_pfs, b_log_pbs, log_flow,
                 b_gauss_params) = self.ema_model.get_traj_bwd(
                    terminal_state, discretizer, condition, mol_batch, return_gauss_params=True)
                log_z = log_flow[:, 0]

                samples += mol_batch.num_graphs

            except (RuntimeError, ValueError) as e:
                self.handle_train_epoch_error(e, 'eval_bwd')
                continue

            cpu = lambda t: t.cpu().detach()
            acc['flow_states'].append(cpu(backward_flow_states))
            acc['log_pfs'].append(cpu(b_log_pfs))
            acc['log_pbs'].append(cpu(b_log_pbs))
            for k, v in b_gauss_params.items():
                acc[k].append(cpu(v))
            acc['log_r'].append(cpu(log_r))
            acc['log_Z_learned'].append(cpu(log_z))
            for _k, _v in self._eval_extra_stats(mol_batch).items():
                acc[_k].append(cpu(_v))
            acc['condition_id'].append(cpu(condition_id))

        pooled = {k: torch.cat(v, dim=0) for k, v in acc.items()}
        if not self.gfn_model.conditional:
            pooled['log_Z_learned'] = torch.mean(pooled['log_Z_learned'])

        # deliberately NO tracker update() here (there used to be a phase-gated
        # one): same single-protocol principle as fwd_eval_sampling -- eval-time
        # backward sampling (ema_model, fixed eval temperature) is a different
        # measurement protocol from the train-step bwd stream that feeds the
        # tracker, and mixing protocols inflates the tracker's second moment by
        # the between-stream mean shift, spiking the terminal stage's logw_std gate (and
        # the ema_logw sawblade) at every eval. The train-time bwd/replay
        # update() calls (phase-gated, in get_gfn_backward_loss) are untouched.

        return pooled

    def _eval_conditional_stats(self, stats, coeffs):
        """
        THE ONLY eval-time quick_tb_stats call. All three eval streams --
        'eval_fwd/' (train conditions), 'eval_bwd/', 'eval_test/' (held-out) --
        go through here, differing only in the accumulated tensors and in
        `coeffs` (the direction's *_loss_coeffs, for the Huber clip_beta behind
        tb_resid_clipped / z_grad_worst).

        ONE CALL SITE IS THE POINT, not a tidiness preference. The metric names
        are shared across streams and the dashboard reads them as a set, so any
        argument that changes what a name MEANS -- worst_quantile above all --
        has to be impossible to set on one stream and not another. It was not:
        the two calls this replaced omitted worst_quantile entirely and took
        quick_tb_stats' 0.5 default, while log_test_metrics recomputed the
        train-condition stats at the protocol quantile and overwrote four
        'eval_fwd/' keys on the way out. Which definition reached wandb was
        settled by dict-update order in evaluation(), and (because the overwrite
        only ran when a held-out set was configured) 'eval_fwd/tb_err_worst'
        silently meant the MEDIAN condition on a run with no
        `test_molecules_path` and the upper-tail one on a run with it. A new
        keyword added to one of two parallel calls would have inherited exactly
        the same defect, which is why there is now only one.

        worst_quantile is `conditional_worst_quantile` -- the same value every
        train-step site already passes (_update_rolling, the per-step probe) and
        the same one log_condition_fraction and condition_tracker_figs read, so
        'the worst-case condition' is one convention across the whole run
        rather than a per-call-site accident. Never a bare default here.
        """
        log_pf = stats['log_pfs'].sum(-1)
        log_pb = stats['log_pbs'].sum(-1)
        log_z = stats['log_Z_learned']
        log_r = stats['log_r']
        cid = stats.get('condition_id')
        return quick_tb_stats(log_pf, log_pb, log_z, log_r,
                              clip_beta=getattr(coeffs, 'beta', None),
                              condition_id=cid,
                              worst_quantile=self.args.conditional_worst_quantile,
                              **self._reward_ramp_kwargs(cid))

    @torch.no_grad()
    def log_test_metrics(self, eval_discretizer, fwd_stats):
        """
        Conditional generalization check: the same on-policy eval protocol run
        against HELD-OUT conditions (test_molecules_path), logged under
        'eval_test/', against the matching train-condition readings under
        'eval_fwd/'. The generalization gap is the difference between the two
        and is left to the dashboard rather than logged as its own channels.

        Pure measurement. fwd_eval_sampling runs with side_effects=False, so the
        held-out conditions never reach condition_log_z, the anchor buffer, or
        prior-buffer churn, and nothing here feeds a gate, controller or loss.
        """
        n = getattr(self.args, 'test_eval_num_samples', None) or self.args.eval_num_samples
        test_stats, test_batch = self.fwd_eval_sampling(self.ema_model, eval_discretizer,
                                                        override_num_samples=int(n),
                                                        dataset=self.test_mol_dataset,
                                                        side_effects=False)
        test_m = self._eval_conditional_stats(test_stats, self.args.fwd_loss_coeffs)

        metrics = {f'eval_test/{k}': v for k, v in test_m.items()}
        # THIS METHOD WRITES NOTHING UNDER 'eval_fwd/'. It used to re-run the
        # train-condition stats and republish four of them, because log_metrics
        # computed those keys at quick_tb_stats' default quantile rather than
        # the protocol's and they were not like-for-like with 'eval_test/'.
        # log_metrics now takes the quantile from the config (see
        # _eval_conditional_stats), so the train-condition series is already
        # correct at its own site, the duplicate pass over the full train eval
        # batch is gone, and the two streams cannot collide on one key.
        #
        # no 'eval_gap/' block either: it was exactly eval_fwd - eval_test on six
        # keys that are both already logged, so the generalization gap is a wandb
        # panel expression rather than six more channels. Read it on the QUANTILE
        # family, not the RMS one -- see module_metrics.md: the two streams pool
        # very different sample counts over very different condition counts, so
        # scatter_err / over_coverage / logw_std_within reach the tail at
        # different rates and their difference is sampling geometry.

        # SAMPLE QUALITY on held-out conditions, pooled and per-condition, the
        # same pair log_thermo_properties publishes at top level for the train
        # conditions. Only the 'reasonable' window transfers: it is an absolute
        # physical bar needing no per-condition reference, whereas the
        # non-thermal family is scored against Emin(c), which side_effects=False
        # never writes for a test condition (its condition_ids are disjoint --
        # init_identifiers mints one registry over distinct identifier strings),
        # so it would have nothing to reference and read a constant 0.
        if test_batch is not None:
            arr = lambda t: t.cpu().detach().numpy()
            is_good = self._reasonable_sample_mask(test_batch)
            metrics['eval_test/Reasonable Sample Fraction'] = is_good.float().mean().item()
            self.log_condition_fraction(metrics, arr, 'Reasonable', is_good,
                                        test_stats.get('condition_id'),
                                        getattr(self.args, 'reasonable_cond_bar', 0.5),
                                        higher_is_worse=False, prefix='eval_test/')
        return metrics

    def vram_ledger(self, tag: str):
        """
        One line of the init-time VRAM ledger: what is LIVE, what the allocator is
        HOLDING, and the difference.

        WHY THE DIFFERENCE IS THE WHOLE POINT. `allocated` is live tensors; `reserved`
        is what the caching allocator took from the driver and -- expandable_segments
        being unsupported here -- does not give back. `reserved - allocated` is free
        cache. That is normally harmless, except that `cuda_memory_fraction` is applied
        as a HARD per-process cap (set_per_process_memory_fraction, __init__), so cache
        the run cannot reuse still counts against the ceiling a training allocation has
        to fit under. Blocks shaped like MLIP supercell neighbour lists are exactly the
        kind a T-step MLP rollout will never ask for.

        Called at each init milestone so the question "do the startup mace evaluations
        give their VRAM back" is ANSWERED BY THE LOG rather than inferred from a batch
        size hours later. Read the ledger down the page: a large `cached` that appears
        at the prior re-analysis and never falls is the leak signature; a large
        `allocated` that appears when the buffers are built is buffer_device: cuda
        doing what it was told, and needs a different fix entirely.
        """
        if not torch.cuda.is_available():
            return
        try:
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            res = torch.cuda.memory_reserved() / (1024 ** 2)
            peak = torch.cuda.max_memory_reserved() / (1024 ** 2)
            print(f"vram [{tag:<28}] live {alloc:8.0f} MiB | reserved {res:8.0f} MiB "
                  f"| cached {res - alloc:8.0f} MiB | peak reserved {peak:8.0f} MiB")
            if not hasattr(self, '_vram_ledger'):
                self._vram_ledger = {}
            self._vram_ledger[tag] = (round(alloc), round(res))
        except Exception:
            pass        # a diagnostic must never be able to kill a run

    def vram_metrics(self):
        """The same three numbers as wandb series, so the ledger has a time axis past
        init. `vram/cached_mb` is the actionable one -- see vram_ledger."""
        if not torch.cuda.is_available():
            return {}
        try:
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            res = torch.cuda.memory_reserved() / (1024 ** 2)
            return {'vram/live_mb': alloc, 'vram/reserved_mb': res,
                    'vram/cached_mb': res - alloc,
                    'vram/peak_reserved_mb': torch.cuda.max_memory_reserved() / (1024 ** 2),
                    # SEGMENTED peak: reset at the top of every evaluation, so this
                    # is the peak of the TRAIN phase since the last eval rather than
                    # a run-lifetime maximum. A batch sizer's memory constraint is a
                    # train-phase quantity, and eval shares `batch_size` -- so on a
                    # run with eval on, the lifetime peak is frequently the EVAL peak
                    # and using it to bound the training batch bounds the wrong
                    # thing. peak_reserved_mb is kept as-is; nothing that reads it
                    # today changes meaning.
                    'vram/peak_train_mb': self._phase_peak_mb()}
        except Exception:
            return {}

    def _phase_peak_mb(self):
        """Peak reserved MB since the last `reset_peak_memory_stats`, in MB.

        Separate from `vram/peak_reserved_mb` deliberately: that one is a
        run-lifetime high-water mark and several call sites already depend on it
        meaning exactly that (the pre-flight registry, the OOM ceiling report).
        """
        try:
            return torch.cuda.max_memory_reserved() / (1024 ** 2)
        except Exception:
            return 0.0

    def _reset_phase_peak(self):
        """Start a new VRAM phase window. Called at the top of `evaluation`, so a
        train-phase peak is never contaminated by the previous eval's allocation
        burst -- eval draws `eval_num_samples` through the same batch machinery
        and is often the larger of the two."""
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def record_vram_peak(self):
        """
        Publish this run's peak RESERVED VRAM to the pre-flight registry.

        Reserved, not allocated: reserved is what the caching allocator holds from the
        driver and (with expandable_segments unsupported on this platform) does not
        give back, so it is what a co-tenant would actually find missing.

        Best-effort by construction -- a diagnostic must never be able to kill a
        training run, so every failure path here is swallowed.
        """
        try:
            if not torch.cuda.is_available():
                return
            peak_mb = int(torch.cuda.max_memory_reserved() / (1024 ** 2))
            live_mb = int(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if peak_mb <= 0:
                return
            from gpu_guard import record_peak
            # both: their RATIO says whether co-tenancy needs a genuinely smaller job
            # or merely a lower cuda_memory_fraction. See record_peak's docstring.
            record_peak(self.args, peak_mb, live_mb)
        except Exception:
            pass

    def log_metrics(self, fwd_stats, bwd_stats, sample_batch):

        metrics = {}
        arr = lambda t: t.cpu().detach().numpy()
        val = lambda t: t.cpu().detach().item()

        """Forward TB Stats"""
        log_r = fwd_stats['log_r']
        log_Z_learned = fwd_stats['log_Z_learned']
        log_T_tensor = fwd_stats['log_T_tensor']
        metrics.update({f'eval_fwd/{k}': v for k, v in
                        self._eval_conditional_stats(fwd_stats, self.args.fwd_loss_coeffs).items()})

        self.log_thermo_properties(arr, fwd_stats, log_T_tensor, log_Z_learned, log_r, metrics, sample_batch, val)

        """Backward TB Stats"""
        # parity / Z diagnostics (shared with fwd) -- same single call site, so
        # 'eval_bwd/tb_err_worst' and 'eval_fwd/tb_err_worst' name the same
        # quantile and the fwd/bwd parity read is a like-for-like one
        metrics.update({f'eval_bwd/{k}': v for k, v in
                        self._eval_conditional_stats(bwd_stats, self.args.bwd_loss_coeffs).items()})
        bwd_log_pf = bwd_stats['log_pfs'].sum(-1)   # log_dist_stats reads the BWD stream

        def dump_numeric(metrics, prefix, obj):
            """Log the numeric settings behind this eval, but ONLY the ones that
            moved since the last eval (plus everything once, on the first eval,
            to establish the baseline).

            These are settings, not measurements: the energy function's
            constants never change at all within a run, and the loss
            coefficients change only when the protocol swaps stages. Re-logging
            all 68 of them every eval put a wall of flat lines in the run --
            the whole point of having them here is to see the STEP where a
            stage transition retuned something, and a channel that only emits
            on change shows exactly that (wandb holds the last value forward
            between points, so the panels still read correctly)."""
            cache = self._settings_log_cache
            d = obj if isinstance(obj, dict) else vars(obj)
            for k, v in d.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    key = f'{prefix}/{k}'
                    if key not in cache or cache[key] != v:
                        metrics[key] = v
                        cache[key] = v

        dump_numeric(metrics, 'energy_func/', self.energy_function)
        dump_numeric(metrics, 'loss_coeffs/fwd_', self.args.fwd_loss_coeffs)
        dump_numeric(metrics, 'loss_coeffs/bwd_', self.args.bwd_loss_coeffs)
        dump_numeric(metrics, 'loss_coeffs/replay_', self.args.replay_loss_coeffs)

        self.log_dist_stats(bwd_log_pf, metrics, sample_batch)

        "Trajectory Stats"
        for prefix in ['fwd', 'bwd']:
            if prefix == 'fwd':
                stats = fwd_stats
            elif prefix == 'bwd':
                stats = bwd_stats
            metrics[f'{prefix} Mean F Drift'] = stats['means_f'].abs().mean()
            metrics[f'{prefix} Mean B Drift'] = stats['means_b'].abs().mean()
            metrics[f'{prefix} Mean F Var'] = stats['logvars_f'].mean()  # total per-dim variance budget (s^2)
            metrics[f'{prefix} Mean B Var'] = stats['logvars_b'].mean()
            metrics[f'{prefix} Mean F Diag Var'] = stats['diag_logvars_f'].mean()  # private (non-DPLR) diagonal
            metrics[f'{prefix} Mean F Rho'] = stats['rho_f'].mean()  # DPLR correlated variance fraction; 0 when off
            metrics = {k: to_loggable(v) for k, v in metrics.items()}

        res = traj_overlap_report(fwd_stats, bwd_stats)  # torch tensors are fine; auto-converted
        metrics.update({**to_scalars(res)})

        return metrics

    def log_dist_stats(self, log_pf, metrics, sample_batch):
        std_params = self._batch_latents(sample_batch)
        metrics['Total Var'] = std_params.var(dim=0).mean().cpu().detach().numpy()
        metrics['Total Mean'] = std_params.mean(dim=0).mean().cpu().detach().numpy()
        U, S, Vh = torch.linalg.svd(std_params - std_params.mean(0), full_matrices=False)
        eigvals = S ** 2
        explained_var_ratio = eigvals / eigvals.sum()
        loadings = Vh.T  # shape: (num_features, num_components)
        contrib_per_feature = (loadings ** 2) @ explained_var_ratio  # shape: (num_features,)
        d_eff = (explained_var_ratio ** 2).sum() ** -1
        metrics['Effective Dimension'] = d_eff.item()
        if getattr(self.args, 'repeats',
                   1) > 1:  # legacy global-repeats configs only; repeats now lives per-mode in *_loss_coeffs
            metrics['ess'] = log_ess_frac(log_pf, log_pf, repeats=1)  # only useful with repeats > 1
        sampled = std_params  # forward-sampled latents, already computed above
        n = sampled.shape[0]
        # draw the reference -- and an independent SECOND reference for the null
        # floor -- at the SAME sample count as the sampler, so the raw and null
        # sliced-W estimates share the same finite-sample floor (see below).
        x, _ = next(self.prior_dataset.loader(batch_size=n, mode='tensors'))
        x2, _ = next(self.prior_dataset.loader(batch_size=n, mode='tensors'))
        x = x.to(sampled.device, sampled.dtype)
        x2 = x2.to(sampled.device, sampled.dtype)
        # Debiased sliced-Wasserstein. SW has a positive, distribution-dependent
        # finite-sample floor: two independent N-sample empirical clouds of the
        # SAME distribution sit ~C/sqrt(N) apart, not at 0. We estimate that floor
        # directly on the reference (null = SW(ref, ref')) and subtract it. This is
        # valid exactly where it matters: near convergence sampler ~= ref, so the
        # raw and null floors coincide and cancel; far from convergence the true
        # signal swamps the floor, so the imperfect cancellation is negligible.
        # Both terms use the SAME seeded projections, so projection-level MC noise
        # is correlated between them and partly cancels in the difference too.
        proj_seed = 0
        raw = sliced_wasserstein(
            sampled, x, n_proj=500,
            generator=torch.Generator(device=sampled.device).manual_seed(proj_seed))
        null = sliced_wasserstein(
            x, x2, n_proj=500,
            generator=torch.Generator(device=sampled.device).manual_seed(proj_seed))
        metrics['wass'] = raw  # raw SW, logged for reference; phase1to2 gate uses wass_debiased
        metrics['wass_null'] = null
        metrics['wass_debiased'] = raw - null

        # Same debiased sliced-W, but against the anchor buffer's latent cloud:
        # how far the on-policy distribution sits from the confirmed-good
        # archive. The buffer is split into two disjoint draws (and the sampler
        # subsampled to match) so raw and null share the same finite-sample
        # floor, mirroring the prior-reference construction above.
        anchor_buffer = getattr(self, 'anchor_buffer', None)
        if anchor_buffer is not None and len(anchor_buffer) >= 4:
            ax = anchor_buffer.x.to(sampled.device, sampled.dtype)
            m = min(n, len(ax) // 2)
            perm = torch.randperm(len(ax), device=ax.device)
            a1, a2 = ax[perm[:m]], ax[perm[m:2 * m]]
            sub = sampled if m == n else sampled[torch.randperm(n, device=sampled.device)[:m]]
            raw = sliced_wasserstein(
                sub, a1, n_proj=500,
                generator=torch.Generator(device=sampled.device).manual_seed(proj_seed))
            null = sliced_wasserstein(
                a1, a2, n_proj=500,
                generator=torch.Generator(device=sampled.device).manual_seed(proj_seed))
            metrics['wass_anchor'] = raw
            metrics['wass_anchor_null'] = null
            metrics['wass_anchor_debiased'] = raw - null

    def log_thermo_properties(self, arr, fwd_stats, log_T_tensor, log_Z_learned, log_r, metrics, sample_batch, val):
        # energies. gfn_energy is skipped: it is logged just below as 'Mean
        # Sample Energy' (identical to float precision), and that name says
        # which of the several energies on the batch is the one the loss sees.
        for key in sample_batch.keys():
            if ('energy' in key or 'pot' in key) and key != 'gfn_energy':
                metrics['Mean ' + key] = val(sample_batch[key].mean())

        # the rotational Haar jacobian terms diverge (log) at r -> 0 and theta -> 0/pi
        # (clamped at ~37 nats), so a rare singularity graze moves the batch mean by
        # ~nothing -- the max is the diagnostic. jacob_july24: healthy-window peaks run
        # 0.8-3.5 nats and only hit the cap during an excursion already underway, so a
        # peak in the tens is a symptom to read, not a cause to chase
        for key in ['rot_r_jacobian_energy', 'rot_theta_jacobian_energy']:
            if key in sample_batch.keys():
                metrics['Max ' + key] = val(sample_batch[key].max())

        # physical properties
        self.log_physical_properties(metrics, sample_batch, val, arr)
        # conditions
        metrics['Crystal Mean Log Temperature'] = val(log_T_tensor.mean())
        metrics['Crystal Log Temperature'] = arr(log_T_tensor)
        # training metrics
        metrics['Mean Sample Energy'] = val(sample_batch.gfn_energy.mean())
        metrics['Sample Energy'] = arr(sample_batch.gfn_energy.clip(max=50))
        metrics['Mean Sample Reward'] = val(log_r.mean())
        # no 'Sample Reward' histogram: log_r is -energy/T, so it was the
        # 'Sample Energy' histogram mirrored and rescaled, nothing more.
        # 'Empirical log Z' / 'Empirical log Z LB' are likewise gone -- they
        # were bit-identical to eval_fwd/emp_z and eval_fwd/jensen_z, which
        # sit next to the rest of the forward parity block where they belong.
        metrics['log Z learned'] = val(log_Z_learned.mean())
        # 1 = the dead-row table was confirmed against the real crystal build at startup,
        # 0 = the probe could not run and the table is UNVERIFIED for this run,
        # absent = not a crystal problem, or hold_dead_latent_rows is off. See D33.
        _drv = getattr(self, '_dead_rows_verified', None)
        if _drv is not None:
            metrics['dead_rows/probe_verified'] = float(_drv)

        sample_is_good = self._reasonable_sample_mask(sample_batch)
        metrics["Reasonable Sample Fraction"] = sample_is_good.float().mean().item()

        # ...and the same indicator read PER CONDITION rather than pooled: a
        # batch fraction of 0.5 is a different (much worse) model when half the
        # library is at 0 than when every condition is at 0.5, and only the
        # per-condition view can tell them apart. See per_condition_fraction.
        self.log_condition_fraction(metrics, arr, 'Reasonable', sample_is_good,
                                    fwd_stats.get('condition_id'),
                                    getattr(self.args, 'reasonable_cond_bar', 0.5),
                                    higher_is_worse=False)

        # 'Reasonable Sample Fraction' just above is an ABSOLUTE, hand-set
        # physical window. This is the distribution-relative counterpart: how
        # much of the batch is so far above its own condition's best known
        # energy that no realisable density of states could explain it.
        self.log_nonthermal_tail(arr, fwd_stats, log_T_tensor, log_r, metrics)

    def _reasonable_sample_mask(self, sample_batch):
        """
        Per-sample 'is this a physically reasonable crystal', the ABSOLUTE
        hand-set window behind 'Reasonable Sample Fraction': bound energy, and a
        packing coefficient in 0.55-0.95.

        Prefers the rescaled `mol_energy` (matches the actual loss scale) over
        the bare energy_function attribute, which is only correct for toy (non
        lj-rescaled) energy functions that never set mol_energy.

        Shared by the train-condition and held-out readings so the two are
        computed by the same code, not by two copies that can drift -- the same
        like-for-like rule _eval_conditional_stats follows for the TB family.
        """
        en_func = self.energy_function.energy_function
        scaled_mol_energy = getattr(sample_batch, 'mol_energy', None)
        if scaled_mol_energy is None:
            scaled_mol_energy = sample_batch[en_func]
        return ((scaled_mol_energy < 0) * (sample_batch.packing_coeff > 0.55)
                * (sample_batch.packing_coeff < 0.95))

    def log_condition_fraction(self, metrics, arr, name, indicator, condition_id,
                               bar, higher_is_worse, prefix=''):
        """
        Publish the per-condition breakdown of a per-sample 0/1 indicator under
        '{prefix}Cond {name} *':

          'Cond {name} Failing Frac'  share of CONDITIONS that fail `bar`
          'Cond {name} Worst'         the conditional_worst_quantile bad tail
                                      across conditions, same convention as
                                      tb_err_worst
          'Cond {name} Spread'        binomial-debiased sd of the per-condition
                                      fraction -- the ONLY key here comparable
                                      across streams (see below)
          'Cond {name} Frac'          histogram of the per-condition fractions
          'Cond {name} N'             conditions scored -- divide the batch by
                                      it before trusting any of the above
                                      (per_condition_fraction: at 1 sample per
                                      condition the family degenerates to the
                                      pooled fraction)
          'Cond {name} Bar'           the bar itself, on change only

        `prefix` namespaces the SERIES ('eval_test/' for the held-out stream).
        The bar is deliberately NOT prefixed: it is a property of the metric
        family, not of the stream reading against it, so every prefix scores
        against one series and _log_setting's cache makes the second writer a
        no-op rather than a duplicate channel.

        CROSS-STREAM COMPARISONS GO ON 'Spread', not 'Failing Frac'. The two
        streams do not draw the same samples per condition (cond_aug11: 10000
        over ~900 train conditions vs 2000 over ~100 held-out ones, so n_c ~ 11
        against ~20), and 'Failing Frac' is biased by n_c -- binomial smearing
        pushes conditions across the bar, always toward that stream's own pooled
        fraction, so two identically-good streams report different numbers.
        'Spread' subtracts that noise and is unbiased at any n_c; the LEVEL
        comparison is already carried by the pooled fractions, which are batch
        means and so have no n_c dependence at all. 'Failing Frac' / 'Worst' /
        'Frac' stay WITHIN-stream readings: n_c is fixed for a given stream, so
        their trends over a run are clean.

        The whole family is ABSENT -- never nan, never 0 -- on unconditional
        runs, on a batch with fewer than 2 conditions, and when the bar is set
        to null. `bar` is a threshold on the per-condition fraction, not on the
        indicator; `higher_is_worse` says which side of it fails.
        """
        stats = per_condition_fraction(indicator, condition_id, bar,
                                       worst_quantile=self.args.conditional_worst_quantile,
                                       higher_is_worse=higher_is_worse)
        if stats is None:
            return
        metrics[f'{prefix}Cond {name} Failing Frac'] = stats['failing_frac']
        metrics[f'{prefix}Cond {name} Worst'] = stats['worst']
        metrics[f'{prefix}Cond {name} Frac'] = arr(stats['per_condition'])
        metrics[f'{prefix}Cond {name} N'] = stats['n_conditions']
        if stats['spread'] is not None:  # absent, not 0, when every group is a singleton
            metrics[f'{prefix}Cond {name} Spread'] = stats['spread']
        self._log_setting(metrics, f'Cond {name} Bar', float(bar))

    def _log_setting(self, metrics, key, value):
        """
        Emit a SETTING, not a series: logged on the first eval and thereafter
        only when it changes. A reading scored against a bar is uninterpretable
        later without the bar (module_metrics.md S3), but re-logging a constant
        every eval is a flat line that looks like a measurement. wandb holds the
        last value forward, so the panels still read correctly. Same rule
        dump_numeric applies to the energy-function / loss-coefficient blocks.
        """
        if self._settings_log_cache.get(key) != value:
            metrics[key] = value
            self._settings_log_cache[key] = value

    def log_nonthermal_tail(self, arr, fwd_stats, log_T_tensor, log_r, metrics):
        """
        The high-energy tail stated DIRECTLY, rather than inferred from mean
        energy, reasonable-sample fraction or an effective temperature.

        Per sample, the reduced excess energy -- its log-Boltzmann deficit
        against the best state known for its own condition:

            u = (E - Emin(c)) / T   ==   log R*(c) - log R

        u is in nats and is already both condition- and temperature-reduced,
        so it pools legitimately across a mixed-condition, mixed-T eval batch
        (unlike a raw energy, and unlike r2 -- module_metrics.md T0).

        HOW BIG IS OBVIOUSLY NON-THERMAL? Under any Boltzmann target, drawing a
        state u nats up costs e^-u, and the only thing that can pay for it is
        the number of states available up there:

            P(u > u*) = int_{u>u*} g(E) e^-E/T dE / Z  <=  (V_acc / V_ref) e^-u*

        so log P <= S - u*, where S = log(V_acc / V_ref) is the ENTROPY BUDGET
        -- the log-ratio of accessible configuration volume to that of the
        reference low-energy region. The latents live in a bounded box, so S is
        finite and extensive in the latent dimension:

            u* = data_ndim * nonthermal_entropy_per_dim

        with s = nonthermal_entropy_per_dim the per-axis budget in nats. Two
        equivalent readings of the default s = 4 (u* = 48 at data_ndim 12): as a
        pure entropy budget it grants the reference region as little as e^-4 ~
        1.8% of each axis; or, taking anchor_buffer's dup_cutoff (5% of an axis)
        as the reference size instead, S = 12*log(1/0.05) = 36 and the remaining
        12 nats are rarity margin, e^-12 ~ 1 draw in 1.6e5. Either way it is a
        conservative bar, which is what 'obviously' requires.

        `Nonthermal Fraction` is then the fraction of the batch that NO density
        of states this problem could have is able to explain -- a strictly
        stronger claim than 'high energy'. At elj/T=2.5 with data_ndim 12 and
        s=4, u* = 48 nats = 120 kJ/mol above Emin(c), well past anchor_buffer's
        thin_energy_window (50 kJ/mol, 'hopeless under any reweighting').

        The threshold is one bar on a distribution, so the tail's SHAPE is
        logged next to it: a fraction that is 0 while P99 climbs is a tail
        growing under the bar, which is the same event seen earlier.

        Emin(c) is the running per-condition record from condition_log_z, which
        fwd_eval_sampling has already updated with THIS batch, so u >= 0 by
        construction on that path and the metric cannot be flattered by a batch
        that is uniformly bad. The mirror-image weakness: early in a run Emin(c)
        is only as deep as what has been seen, so the tail is UNDER-reported
        until the records mature. Same direction as the other soft spot --
        non-finite energies are patched to 0 upstream (molecular_crystal.py's
        generator_energy), so a numerically blown-up sample reads as mildly
        excited rather than as tail. Both make this a lower bound on the tail;
        neither can manufacture one.

        Conditions with no record yet are excluded and counted in
        `Excess Energy Referenced Fraction`. Off when
        nonthermal_entropy_per_dim is 0 or null.
        """
        s_per_dim = getattr(self.args, 'nonthermal_entropy_per_dim', 4.0)
        s_per_dim = 0.0 if s_per_dim is None else float(s_per_dim)
        condition_id = fwd_stats.get('condition_id')
        if s_per_dim <= 0 or condition_id is None:
            return
        floor = self._condition_energy_floor(condition_id)
        if floor is None:  # pre-bootstrap: no tracker and no anchors to measure against
            return

        temperature = (10 ** log_T_tensor).detach().cpu().flatten()
        energy = -log_r.detach().cpu().flatten() * temperature
        floor = floor.detach().cpu().flatten().to(energy.dtype)
        seen = torch.isfinite(floor)
        metrics['Excess Energy Referenced Fraction'] = seen.float().mean().item()
        if not bool(seen.any()):
            return

        # clamped at 0: a negative excess means this sample IS the new record
        # for its condition, reachable only on the anchor-buffer fallback path
        # (which this batch has not been folded into). That is not a tail event.
        u = ((energy[seen] - floor[seen]) / temperature[seen]).clamp_min(0.0)
        # Per DEGREE OF FREEDOM, not per state slot: with dead latent rows held out of
        # the SDE (D33) the reachable entropy lives on live_dim dims, so scaling by the
        # full data_ndim would set the threshold too lenient by exactly the dead
        # fraction (12 -> 10 for monoclinic, i.e. 20% too high). Exactly identical for
        # triclinic and for toys, where live_dim == data_ndim, so no existing run or
        # logged threshold moves.
        n_dof = getattr(getattr(self, 'gfn_model', None), 'live_dim', None)
        if n_dof is None:
            n_dof = self.energy_function.data_ndim
        u_star = s_per_dim * float(n_dof)

        metrics['Nonthermal Fraction'] = (u > u_star).float().mean().item()
        # per-CONDITION view of the same bar, over the SAME scored rows: 'which
        # conditions are broken', not 'how much of the batch is'. cid is
        # subsetted by `seen` to match u, whose unreferenced rows were dropped.
        cid_seen = torch.as_tensor(condition_id).detach().cpu().flatten()[seen]
        self.log_condition_fraction(metrics, arr, 'Nonthermal', u > u_star, cid_seen,
                                    getattr(self.args, 'nonthermal_cond_bar', 0.1),
                                    higher_is_worse=True)
        q = torch.quantile(u, torch.tensor([0.5, 0.9, 0.99], dtype=u.dtype))
        metrics['Excess Energy Nats Mean'] = u.mean().item()
        metrics['Excess Energy Nats P50'] = q[0].item()
        metrics['Excess Energy Nats P90'] = q[1].item()
        metrics['Excess Energy Nats P99'] = q[2].item()
        metrics['Excess Energy Nats Max'] = u.max().item()
        metrics['Excess Energy Nats'] = arr(torch.log10(1.0 + u))  # log10 hist: u is non-negative and heavy-tailed

        # the bar itself, emitted only when it moves: a reading in nats is
        # uninterpretable later without the threshold it was scored against
        # (module_metrics.md S3), but it is a setting, not a series
        self._log_setting(metrics, 'Nonthermal Threshold', u_star)

    def update_mle_gate(self):
        """
        MLE flatness gate, publishing gates/mle_flat for the warm-start
        stage's exit trigger. Runs at train cadence (every 10 steps) on the
        stages that declare the mle_gate flag.

        Samples the RAW per-step bwd MLE batch loss (self._last_stats, the
        pre-EMA value _update_rolling just computed) every 10 steps into a
        window of mle_gate.window train steps, then least-squares fits the
        slope. Raw batch losses are ~independent across steps -- unlike the
        old 100-step-time-constant EMA input, whose ~0.9 autocorrelation
        needed an AR(1) effective-sample-size correction and forced the
        window out to 1000 steps just to hold ~5 independent samples. On raw
        input, plain OLS standard errors apply and the same confidence fits
        in a ~3x shorter window.

        'Flat' = an EQUIVALENCE test on the descent rate: the upper
        mle_gate.slope_t-sigma bound on the rate (nats per 100 train steps -- the
        RATE needs no per-system normalization; nats are nats) lies below
        mle_gate.min_rate. Deliberately NOT a significance test: 'descent not
        significantly nonzero' also holds when the data is uninformative, so
        a significance gate fires hardest where it knows least (lcmft1z4:
        rate +0.96 nats/100 across a reload transient read as 'flat').
        Bounding the rate makes noise argue AGAINST exiting: an uncertain
        window cannot clear the bound. A RISING MLE (overfitting onset) still
        counts as flat -- the whole interval sits below min_rate -- since
        that's an exit signal, not progress.

        No internal latch or patience any more: the verdict is published
        per-check as gates/mle_flat in {0, 1} and the exit trigger's own
        `patience` does the latching (a resumed descent publishes 0 and
        resets the trigger streak -- the old un-latch-on-descent behavior,
        for free). The window rides in stage_ctrl['gate_state'], so it is
        checkpointed with the trigger streaks it feeds, and a short restored
        window publishes 0 until re-proven -- the old stale-latch guard,
        also for free. Returns loggable metrics.
        """
        metrics = {}
        raw = self._last_stats.get('bwd', {}).get('mle')
        if raw is None or not math.isfinite(raw):
            return metrics
        gs = self.protocol.gate_state('mle')
        # Parameters come from the STAGE that declares the gate, not from args:
        # they shape this stage's exit and nothing else consults them.
        gate = self.protocol.stage.mle_gate
        checks = max(int(gate['window']) // 10, 4)
        window = gs.setdefault('window', [])
        window.append(float(raw))
        del window[:-checks]
        if len(window) < checks:
            self.protocol.publish_gate('mle_flat', 0.0)
            return metrics
        y = np.asarray(window, dtype=float)
        n = len(y)
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)  # slope per check = per 10 steps
        resid = y - (slope * x + intercept)
        sxx = float(((x - x.mean()) ** 2).sum())
        s2 = float((resid ** 2).sum()) / max(n - 2, 1)
        se = np.sqrt(s2 / sxx) if sxx > 0 and s2 > 0 else 0.0
        rate = -slope * 10.0  # nats per 100 train steps; > 0 = descending = improving
        se_rate = se * 10.0
        rate_hi = rate + float(gate['slope_t']) * se_rate
        flat = rate_hi < float(gate['min_rate'])
        self.protocol.publish_gate('mle_flat', float(flat))
        metrics['mle_gate_rate'] = rate  # nats/100 train steps; > 0 = improving
        metrics['mle_gate_rate_se'] = se_rate
        metrics['mle_gate_rate_hi'] = rate_hi  # the quantity actually tested
        metrics['mle_gate_flat'] = float(flat)
        return metrics

    def update_lr_plateau(self):
        """
        ReduceLROnPlateau over the stage's declared loss channels.

        Track each channel's best value; a check counts as progress if any
        channel improved on its best by more than `threshold` (relative). If no
        channel has improved for `patience` checks, cut the LR by `factor` and
        wait `cooldown` checks before counting again.

        One criterion covers every failure mode: a rising loss, a stalled one,
        and one that has blown up and plateaued at a ruinous level are all
        simply "no improvement over best". The earlier slope-fitting version
        tested only whether the loss was RISING, and so sat clean for 3000 steps
        through lrs_normal, which was merely stalled at 8x the healthy LR.

        WHY DECLARED AND NOT DERIVED from the active coefficients: bwd level_gap
        carries a coefficient while being sign-indefinite and not a convergence
        signal, and vg_by_condition carries one while being a boolean switch
        with no series behind it.
        """
        sensor = self.protocol.stage.lr_sensor
        if sensor is None or sensor['kind'] != 'plateau':
            return {}
        # The LR is deliberately non-stationary while the envelope ramps, so a
        # lack of progress there says nothing about the operating point.
        if self.lr_controller.in_warmup():
            return {}

        gs = self.protocol.gate_state('lr_plateau')
        best = gs.setdefault('best', {})
        improved = False
        for name in sensor['metrics']:
            mode, _, channel = name.partition('/')
            # The SMOOTHED value (metric_tracker EMA), not the raw per-step one.
            # Tracking a best-ever over raw values ratchets the best down to the
            # luckiest noise sample, after which nothing beats it and the sensor
            # cuts while the run is still genuinely improving -- observed on
            # lrs_blowup, where stale climbed to patience over steps 770-850
            # while the logged (EMA) curve was setting new lows. A slope test
            # wants raw input, because EMA autocorrelation wrecks its standard
            # errors; a best-tracking test wants the opposite.
            raw = self.metric_tracker.get(mode, channel)
            if raw is None or not math.isfinite(float(raw)):
                continue
            val = float(raw)
            prev = best.get(name)
            # ABSOLUTE threshold. A relative one has nothing to be relative to
            # here: bwd/mle is unbounded below, so abs(best) * frac grows without
            # limit as the run improves, and the multiplicative form flips sign
            # outright once best goes negative.
            if prev is None or val < prev - sensor['threshold']:
                improved = True
            if prev is None or val < prev:
                best[name] = val

        if gs.get('cooldown', 0) > 0:
            gs['cooldown'] -= 1
            gs['stale'] = 0
        elif improved:
            gs['stale'] = 0
        else:
            gs['stale'] = int(gs.get('stale', 0)) + 1

        fired = gs['stale'] >= sensor['patience']
        if fired:
            gs['stale'] = 0
            gs['cooldown'] = sensor['cooldown']
        # called unconditionally so a satisfied sensor and an absent one are
        # distinguishable in lr_ctrl/plateau_status
        self.lr_controller.on_plateau(fired, sensor['factor'])
        return {'lr_plateau/stale': float(gs['stale']),
                'lr_plateau/fired': float(fired)}

    def _merge_metrics(self, metrics, new, source):
        """
        dict.update() that REFUSES to overwrite. Two writers reaching the same
        metric key is not a merge, it is a disagreement about what that channel
        means, and plain update() resolves it by call order -- silently, and in
        favour of whichever function happens to run last.

        This is not hypothetical. log_metrics and log_test_metrics both wrote
        eval_fwd/{logw_std_within, cond_tb_err, tb_err_worst, z_grad_worst} with
        different worst_quantile values; the held-out call ran second and won,
        so a knob nobody had set on the eval path decided the meaning of four
        published series (see _eval_conditional_stats). The values were close
        enough to look like the same series and it went unnoticed until a
        cross-stream audit. This assertion would have fired on the first eval.

        HARD FAILURE, not a warning. Evals run every eval_period from the start,
        so a genuine collision kills the run in minutes rather than corrupting
        a week of logging, and a warning on a channel nobody is reading yet is
        the same class of defect as the bug it is meant to catch. The message
        carries both values because 'which one won' is the actual question.

        Not a licence to share a key deliberately: settings emitted from more
        than one stream go through _log_setting, whose cache makes the second
        writer a no-op, so they never reach here as duplicates.
        """
        dup = sorted(set(new) & set(metrics))
        assert not dup, (
            f"metric key collision merging {source}: {len(dup)} key(s) already "
            f"written by an earlier writer this eval. Give one of the two "
            f"streams its own namespace -- do NOT let update() order decide. "
            + "; ".join(f"{k!r} kept={metrics[k]!r} incoming={new[k]!r}"
                        for k in dup[:8])
            + (f" (+{len(dup) - 8} more)" if len(dup) > 8 else ""))
        metrics.update(new)
        return metrics

    def evaluation(self, override_do_figs: bool = False):
        metrics = {}
        # close the TRAIN-phase VRAM window before eval allocates anything, so
        # vram/peak_train_mb reported on the next 10-step report describes
        # training and not this eval's sampling burst
        metrics['vram/peak_train_phase_mb'] = self._phase_peak_mb()
        self._reset_phase_peak()
        # NB any pending pulled-forward request (stage_ctrl['request_eval'] --
        # set by an exit trigger arming, or stamped into a reloaded
        # pre-transition snapshot) is cleared by protocol.maybe_advance below,
        # which every evaluation reaches regardless of stage.
        # wall clock spent in the train loop since the last eval finished --
        # skipped on the first eval, which has no predecessor to measure from
        if 'eval_step_end' in self.times:
            self.times['inter_eval_start'] = self.times['eval_step_end']
            self.times['inter_eval_end'] = time()
        self.times['eval_step_start'] = time()
        eval_discretizer = lambda bsz: uniform_discretizer(bsz, self.args.eval_T)

        do_figs = self.step_ind % self.args.figs_period == 0

        '''sampling and metrics analysis'''
        fwd_stats, sample_batch = self.fwd_eval_sampling(self.ema_model, eval_discretizer)
        bwd_stats = self.bwd_eval_sampling(eval_discretizer)
        self._merge_metrics(metrics, self.log_metrics(fwd_stats, bwd_stats, sample_batch),
                            'log_metrics')
        if getattr(self, 'test_mol_dataset', None) is not None:
            self._merge_metrics(metrics, self.log_test_metrics(eval_discretizer, fwd_stats),
                                'log_test_metrics')

        self.times['eval_figs_start'] = time()
        fig_dict = {}
        if do_figs or override_do_figs:
            # backstop on top of eval_figs' per-block guards: nothing in
            # figure land may interrupt a live run
            with fig_guard('eval figure generation'):
                x, y = next(self.prior_dataset.loader(batch_size=10000, mode='tensors'))
                anchor_buffer = getattr(self, 'anchor_buffer', None)
                anchor_latents = (anchor_buffer.x.detach().cpu().numpy()
                                  if anchor_buffer is not None and len(anchor_buffer) > 0 else None)
                # always sample from forward policy
                fig_dict, metrics = eval_figs(fwd_stats,
                                              bwd_stats,
                                              sample_batch.cpu(),
                                              x,
                                              self.args.energy_function,
                                              metrics,
                                              temperature_conditioning=self.args.temperature_conditioning,
                                              anchor_latents=anchor_latents,
                                              domain_figs=self._domain_figs)
            if hasattr(self, 'condition_log_z'):
                # cross-sections of the per-condition tracker state -- built
                # purely from its running stats, no sampling / energy calls
                with fig_guard('condition tracker figs'):
                    fig_dict.update(condition_tracker_figs(
                        self.condition_log_z, self.energy_function, self.step_ind,
                        worst_quantile=self.args.conditional_worst_quantile))
        self.times['eval_figs_end'] = time()

        '''logging and wrap up'''
        self.times['eval_wrapup_start'] = time()
        if do_figs:
            with fig_guard('fig filesize adjustment'):
                adjust_fig_filesize(fig_dict)
            metrics.update(fig_dict)

        metrics.update({
            'Batch Size': self.batch_size})  # single shared batch size -- train and eval sampling now use the same value
        self.times['eval_wrapup_end'] = time()
        self.times['eval_step_end'] = time()
        # drained AFTER the two end stamps above, so every pair this eval opened
        # is closed by the time it is read. It used to run before them, which
        # paired THIS eval's eval_step_start with the PREVIOUS eval's
        # eval_step_end -- one garbage point per eval, at minus the inter-eval
        # gap (eval_step_time bottomed at -358 s against a true 9.2 s median,
        # eval_wrapup_time at -363 s), and the real value only arrived on the
        # next train-cadence report. Both now land once, correct, on the eval step.
        metrics.update(drain_elapsed_times(self.times))

        for key in metrics.keys():  # cleanup before logging
            if isinstance(metrics[key], np.ndarray):
                metrics[key] = loggable_array(metrics[key])
            elif torch.is_tensor(metrics[key]):
                metrics[key] = loggable_array(metrics[key].detach().cpu().numpy())

        # stage-exit check + transition: the trigger's tick-resolvable terms
        # were latched at train cadence; eval/* terms (e.g. eval/wass_debiased)
        # are checked here against the fresh metrics, and the transition --
        # snapshots, coeff switch, optimizer rebuild, LR re-warm, on_enter
        # actions -- executes inside this call. Also clears any pending
        # pulled-forward eval request, whoever set it.
        self.protocol.maybe_advance(metrics)

        if self.protocol.flag('buffers_active'):  # add samples to off-policy buffer
            self.manage_prior_buffer(sample_batch)
            self.manage_replay_buffer(fwd_stats, sample_batch)

        if hasattr(self, 'anchor_buffer'):
            self.anchor_eval_cycle_count = getattr(self, 'anchor_eval_cycle_count', 0) + 1
            cfg = self.args.buffers.anchor_buffer
            if self.anchor_eval_cycle_count % cfg.thin_every_n_evals == 0:
                self.anchor_buffer.thin(
                    self.condition_log_z.best_energy,
                    energy_window=cfg.thin_energy_window,
                    max_size=cfg.max_size,
                )
            if self.anchor_eval_cycle_count % cfg.refresh_every_n_evals == 0:
                self.refresh_anchor_buffer_surprise()

        metrics.update(self.log_buffer_stats())
        metrics.update(self.log_condition_log_z_stats())

        return metrics

    def _buffer_core_stats(self, buff, prefix, loss_clip_min=0):
        """Length/steps/loss/energy readout shared by prior_buffer and
        replay_buffer -- both are CrystalBuffers scored by select_counts +
        ema_loss. loss_clip_min differs per caller (prior allows slightly
        negative log-loss; replay clips at 0)."""
        metrics = {
            f'{prefix}_length': len(buff),
            f'{prefix}_mean_steps': torch.nanmean(buff.select_counts.float()).item(),
            f'{prefix}_median_steps': torch.nanmedian(buff.select_counts.float()).item(),
            f'{prefix}_mean_loss': torch.nanmean(buff.ema_loss).item(),
            f'{prefix}_median_loss': torch.nanmedian(buff.ema_loss).item(),
            f'{prefix}_step_hist': safe_histogram(buff.select_counts.cpu().numpy()),
        }
        metrics.update(self.energy_stats(prefix, energy=buff.y))
        valid_losses = buff.ema_loss[~torch.isnan(buff.ema_loss)].cpu().numpy()
        if len(valid_losses) > 0:
            metrics[f'{prefix}_loss_hist'] = safe_histogram(
                np.clip(np.log10(valid_losses), min=loss_clip_min, max=3))
        return metrics

    def log_buffer_stats(self):
        # report on whatever backward is actually drawing from this stage:
        # the prior_buffer whenever bwd samples 'prior' (churned or -- in the
        # localized stages, buffers_active off -- frozen), the static
        # prior_dataset during the warm-start. Keyed on bwd_sampling_mode,
        # not buffers_active: the flag gates buffer MANAGEMENT, and the
        # localized anchor_seed/z_match variant deliberately draws from a
        # frozen prior_buffer with management paused
        if self.bwd_sampling_mode == 'prior':
            buff = getattr(self, 'prior_buffer', None)
        else:
            buff = getattr(self, 'prior_dataset', None)

        metrics = {}
        if buff is not None:
            metrics.update(self._buffer_core_stats(buff, 'prior_buffer', loss_clip_min=-1))

        # Prior churn decomposed by source, accumulated since the previous eval's
        # drain (manage_prior_buffer runs inside eval, immediately before this) and
        # drained here, so these are per-eval-window rates rather than run totals.
        # Emission needs a prior_buffer to exist, but the drain below is
        # unconditional: a window that produced churn without a live buffer to
        # report it against must not carry stale counts into the next window.
        churn = self.prior_churn
        if hasattr(self, 'prior_buffer'):
            added = churn['from_prior_model'] + churn['from_anchors'] + churn['from_seed']
            # counts emit even when zero: a window with no intake at all is the
            # stall we most want to see, and gating it away would render that as a
            # gap in the series indistinguishable from missing data
            metrics.update({
                'prior_buffer_added': added,
                'prior_buffer_evicted': churn['evicted'],
                # subset of 'evicted': rows that stopped clearing their OWN
                # condition's admission gate because Emin(c) ratcheted down under
                # them, as opposed to loss-quantile evictions made to free
                # headroom. Persistently 0 with expire_max_frac > 0 means Emin(c)
                # has stopped moving, not that the channel is broken
                'prior_buffer_expired': churn['expired'],
                # rows the anchor top-up requested PURELY to satisfy
                # anchor_floor_frac, i.e. beyond what the per-condition shortfall
                # already asked for. 0 means the floor never bound this window
                'prior_buffer_anchor_floor_rows': churn['anchor_floor'],
                'prior_buffer_from_prior_model': churn['from_prior_model'],
                'prior_buffer_from_anchors': churn['from_anchors'],
                'prior_buffer_from_seed': churn['from_seed'],
                'prior_buffer_turnover': added / max(len(self.prior_buffer), 1),
                # retained key: previously emitted from the standalone
                # last_anchor_topup counter, now sourced from prior_churn so the
                # two can't drift apart
                'anchor_topup_last_n': churn['from_anchors'],
            })
            # the headline diagnostic: share of this window's intake that came from
            # replayed anchors rather than fresh search. -> 1 means the buffer is
            # feeding on its own archive. nan (not 0) when nothing was admitted at
            # all -- the fraction is genuinely undefined there, and 0 would read as
            # "healthy, all fresh"; same nan-means-no-qualifying-samples convention
            # as the under_coverage ramp
            metrics['prior_buffer_anchor_fraction'] = (
                churn['from_anchors'] / added if added > 0 else float('nan'))
            # how much of what the prior model was ASKED to supply passed the
            # admission gate; the shortfall is what anchors backfilled. nan when no
            # draw was requested this window (budget consumed by headroom/eligible
            # -drop caps, or no prior_model yet under the forward-first protocol)
            metrics['prior_buffer_prior_admit_rate'] = (
                churn['from_prior_model'] / churn['budget'] if churn['budget'] > 0 else float('nan'))
        for key in churn:
            churn[key] = 0

        if hasattr(self, 'replay_buffer'):
            metrics.update(self._buffer_core_stats(self.replay_buffer, 'replay_buffer'))
            replay_age = (self.step_ind - self.replay_buffer.birth_step).float()
            metrics.update({
                'replay_buffer_mean_age': replay_age.mean().item(),
                'replay_buffer_max_age': replay_age.max().item() if replay_age.numel() > 0 else 0.0,
                # WIDTH of the residence distribution, not its centre: this is
                # the quantity that governs how much the policy->buffer->
                # gradient path lowpasses itself, and mean/max say nothing
                # about it. A hard age cap gives a ~uniform age profile, CV
                # ~0.58; the memoryless eviction this buffer now uses gives
                # exponential residence, CV ~1. So ~1 is the healthy reading
                # here, and a drift toward ~0.58 means something has started
                # capping age after all.
                'replay_buffer_age_cv': (
                    (replay_age.std(unbiased=False) / replay_age.mean().clamp(min=1e-6)).item()
                    if replay_age.numel() > 1 else 0.0),
            })
            # LIVE improvement-since-admission, over rows drawn at least once.
            # The expired_delta below only sees rows on their way out; this is
            # the standing population, and it is what says whether the buffer
            # is full of rows still being incorporated (negative) or of
            # unincorporable ones (>= 0). stalled_frac is the acted-on slice of
            # this same distribution.
            live_delta = (self.replay_buffer.ema_loss - self.replay_buffer.birth_loss)
            live_delta = live_delta[(self.replay_buffer.select_counts > 0)
                                    & torch.isfinite(live_delta)]
            if live_delta.numel() > 0:
                metrics['replay_buffer_live_delta_mean'] = live_delta.mean().item()
                metrics['replay_buffer_live_delta_stalled_frac'] = (
                    (live_delta >= 0).float().mean().item())

            # churn accumulated since the previous eval's drain, i.e. one full
            # eval period of train steps; drained here so the counts are a rate
            # per window rather than a run-total
            admitted = self.replay_churn['admitted']
            metrics.update({
                'replay_buffer_admitted': admitted,
                'replay_buffer_evicted': self.replay_churn['evicted'],
                'replay_buffer_turnover': admitted / max(len(self.replay_buffer), 1),
                # candidates hard-excluded by admit_reward_min this window --
                # counted upstream of eligibility over the whole batch, so
                # this is a rate against sampled candidates, not churn_rate
                'replay_buffer_reward_rejected': self.replay_churn['reward_rejected'],
            })
            for key in self.replay_churn:
                self.replay_churn[key] = 0

            # Eviction-cause readouts (see manage_replay_buffer's tally
            # comments). The *_frac keys are shares of everything evicted this
            # window and are the primary diagnostic: under the old hard TTL,
            # age silently did 100% of eviction and that was invisible without
            # this split. backstop = hit the absolute age ceiling, which
            # should stay NEAR ZERO -- a non-trivial share means tau is set
            # too long for the intake rate. hazard = ordinary memoryless
            # turnover and should dominate.
            # expired_* keys describe the hazard cohort: expired_undrawn_frac =
            # evicted without a single draw (wasted slots), expired_delta =
            # mean death-minus-birth |resid| (negative = the live population is
            # being incorporated), expired_draws = draws received.
            coh = self.replay_cohort
            resolved = coh['backstop'] + coh['expired']
            if resolved > 0:
                metrics['replay_buffer_backstop_frac'] = coh['backstop'] / resolved
                metrics['replay_buffer_hazard_frac'] = coh['expired'] / resolved
            if coh['expired'] > 0:
                metrics['replay_buffer_expired_undrawn_frac'] = coh['expired_undrawn'] / coh['expired']
            if coh['expired_delta_n'] > 0:
                metrics['replay_buffer_expired_delta'] = coh['expired_delta_sum'] / coh['expired_delta_n']
            if coh['expired_drawn'] > 0:
                metrics['replay_buffer_expired_draws'] = coh['expired_draws_sum'] / coh['expired_drawn']
            self.replay_cohort = {'backstop': 0,
                                  'expired': 0, 'expired_undrawn': 0,
                                  'expired_drawn': 0, 'expired_draws_sum': 0,
                                  'expired_delta_sum': 0.0, 'expired_delta_n': 0}

        if hasattr(self, 'anchor_buffer'):
            metrics['anchor_buffer_length'] = len(self.anchor_buffer)
            metrics.update(self.energy_stats('anchor_buffer', energy=self.anchor_buffer.energy))
            # "the average anchor sample got XX (energy units) better this
            # window": per-condition mean energy drop since last eval,
            # averaged over conditions present in both snapshots -- see
            # AnchorBuffer.pop_mean_energy_improvement for why it's
            # condition-decomposed rather than a raw whole-buffer mean delta
            anchor_improvement, n_shared = self.anchor_buffer.pop_mean_energy_improvement()
            if anchor_improvement is not None:
                metrics['anchor_buffer_mean_energy_improvement_last_window'] = anchor_improvement
                metrics['anchor_buffer_mean_energy_improvement_n_conditions'] = n_shared
            if len(self.anchor_buffer) > 0:
                # energy above each anchor's own condition's minimum -- absolute
                # energy conflates per-condition offsets (conditions sit on very
                # different energy scales), so this is the spread that actually
                # reflects anchor quality within a condition
                cond_ids = self.anchor_buffer.condition_id
                cond_min = _per_condition_min(cond_ids, self.anchor_buffer.energy, cond_ids)
                rel_energy = (self.anchor_buffer.energy - cond_min).cpu().numpy()
                metrics.update({
                    'anchor_buffer_mean_energy_above_cond_min': float(np.mean(rel_energy)),
                    'anchor_buffer_median_energy_above_cond_min': float(np.median(rel_energy)),
                    'anchor_buffer_max_energy_above_cond_min': float(np.max(rel_energy)),
                    'anchor_buffer_energy_above_cond_min_hist': safe_histogram(rel_energy, num_bins=128),
                })
            # the *_last_n counters accumulate across every admission call in
            # the window (eval batch + prior batch + up to two topups per
            # cycle), so pop-and-reset here: read once, then zero for the next
            # window -- a plain last-call value gets clobbered by whichever
            # call runs last (see screen_and_admit_anchors)
            if hasattr(self, 'last_anchor_confirm_spread'):
                metrics['anchor_confirm_spread_last'] = self.last_anchor_confirm_spread
            if hasattr(self, 'last_anchor_admitted'):
                metrics['anchor_admitted_last_n'] = self.last_anchor_admitted
                self.last_anchor_admitted = 0
            if hasattr(self, 'last_anchor_topup_admitted'):
                metrics['anchor_topup_admitted_last_n'] = self.last_anchor_topup_admitted
                self.last_anchor_topup_admitted = 0
            valid_surprise = self.anchor_buffer.original_surprise[
                ~torch.isnan(self.anchor_buffer.original_surprise)].cpu().numpy()
            if len(valid_surprise) > 0:
                metrics['anchor_original_surprise_hist'] = safe_histogram(valid_surprise, num_bins=128)
            valid_priority = self.anchor_buffer.ema_loss[~torch.isnan(self.anchor_buffer.ema_loss)].cpu().numpy()
            if len(valid_priority) > 0:
                metrics['anchor_replay_priority_hist'] = safe_histogram(valid_priority, num_bins=128)

        return metrics

    def log_condition_log_z_stats(self):
        """
        Histograms/summary stats over all visited conditions in
        self.condition_log_z: ema_logw (the Jensen lower-bound estimate
        actually fed back into training, see ConditionLogZTracker.lookup),
        ema_log_z_emp (the logmeanexp/empirical estimate), and their gap
        (ema_log_z_emp - ema_logw >= 0 by Jensen's inequality -- large gap
        means high-variance log importance weights for that condition).

        Also drains the tracker's minima-discovery telemetry (see
        ConditionLogZTracker.pop_discovery_stats) -- this is its single
        drain site, on eval cadence: per-window counts/depth of strict
        per-condition best_energy improvements, plus per-training-step
        rate EMAs that sit near 0 when minima have gone static and rise
        with discovery churn. Depth metrics are intensive -- energy units
        per visited condition -- so they don't grow with library size.
        """
        if not hasattr(self, 'condition_log_z'):
            return {}

        tracker = self.condition_log_z
        disc = tracker.pop_discovery_stats(self.step_ind)
        metrics = {
            'condition_minima_improved_last_window': disc['improved'],
            'condition_minima_first_visits_last_window': disc['first_visits'],
            'condition_minima_depth_last_window': disc['depth'],
            'condition_minima_rate_ema': disc['rate_ema'],
            'condition_minima_depth_rate_ema': disc['depth_rate_ema'],
            'condition_minima_improved_total': disc['improved_total'],
            'condition_minima_depth_total': disc['depth_total'],
        }

        valid = ~torch.isnan(tracker.ema_logw)
        if valid.sum() == 0:
            return metrics

        ema_logw = tracker.ema_logw[valid].cpu().numpy()
        ema_log_z_emp = tracker.ema_log_z_emp[valid].cpu().numpy()
        gap = ema_log_z_emp - ema_logw

        # other per-condition tracker fields, already maintained per condition
        # (same cross-sections the Condition Tracker Histograms plotly builds) --
        # surfaced here as native wandb histograms too. The tb_err distribution is
        # what the tb_err_worst gate takes its quantile OF, so this histogram is
        # the gate's own tail made visible: a single blown condition sits out here
        # while the pooled fit reads fine. tb_err/z_grad/z_bias are nan on
        # conditions not yet residual-warm; safe_histogram drops non-finite
        # entries, so no extra masking is needed.
        logw_std = np.sqrt(
            (tracker.ema_logw_sq - tracker.ema_logw ** 2).clamp(min=0.0)[valid].cpu().numpy())
        tb_err = tracker.cond_tb_err[valid].cpu().numpy()
        z_grad = tracker.z_grad_ema[valid].cpu().numpy()
        z_bias = tracker.z_bias_ema[valid].cpu().numpy()
        best_energy = tracker.best_energy[valid].cpu().numpy()
        best_energy = best_energy[np.isfinite(best_energy)]  # inf == visited but no energy yet

        metrics.update({
            'condition_log_z_num_visited': int(valid.sum().item()),
            'condition_log_z_mean_ema_logw': float(np.mean(ema_logw)),
            'condition_log_z_median_ema_logw': float(np.median(ema_logw)),
            'condition_log_z_ema_logw_hist': safe_histogram(ema_logw, num_bins=128),
            'condition_log_z_mean_ema_log_z_emp': float(np.mean(ema_log_z_emp)),
            'condition_log_z_median_ema_log_z_emp': float(np.median(ema_log_z_emp)),
            'condition_log_z_ema_log_z_emp_hist': safe_histogram(ema_log_z_emp, num_bins=128),
            'condition_log_z_mean_gap': float(np.mean(gap)),
            'condition_log_z_median_gap': float(np.median(gap)),
            'condition_log_z_gap_hist': safe_histogram(gap, num_bins=128),
            'condition_logw_std_hist': safe_histogram(logw_std, num_bins=128),
            'condition_tb_err_hist': safe_histogram(tb_err, num_bins=128),
            'condition_z_grad_hist': safe_histogram(z_grad, num_bins=128),
            'condition_z_bias_hist': safe_histogram(z_bias, num_bins=128),
            'condition_best_energy_hist': safe_histogram(best_energy, num_bins=128),
        })
        if np.isfinite(tb_err).any():
            metrics['condition_tb_err_median'] = float(np.nanmedian(tb_err))
        if best_energy.size:
            metrics['condition_best_energy_median'] = float(np.median(best_energy))
        return metrics

    def energy_stats(self, prefix, energy=None, reward=None):
        """
        Mean, median, min/max and histogram of a buffer's energies.

        Reward is NOT reported alongside: it is -energy / temperature (the same
        fixed-T convention as prebuilt_sample_to_reward and
        top_up_prior_from_anchors), so every reward panel was the energy panel
        mirrored and rescaled by a constant -- verified at pearson -1.0000
        across prior/replay/anchor. Callers that only hold reward can still
        pass reward= and get it converted.
        """
        assert energy is not None or reward is not None, "must pass energy and/or reward"
        if energy is None:
            energy = -reward * self.energy_function.temperature

        energy_np = energy.detach().cpu().numpy()

        # an empty buffer is a legitimate state (a fresh/flushed replay buffer
        # before its first admission, or one the TTL has fully drained), and
        # min/max have no identity on an empty array -- log nothing rather than
        # crash the eval; the corresponding *_length metric carries the fact
        if energy_np.size == 0:
            return {}

        return {
            f'{prefix}_mean_energy': float(np.mean(energy_np)),
            f'{prefix}_median_energy': float(np.median(energy_np)),
            f'{prefix}_min_energy': float(np.min(energy_np)),
            f'{prefix}_max_energy': float(np.max(energy_np)),
            f'{prefix}_energy_hist': safe_histogram(energy_np, num_bins=128),
        }

    def manage_prior_buffer(self, sample_batch):
        if not hasattr(self, 'prior_buffer'):
            self.prior_buffer = self.buffer_cls(
                sample_batch,
                device=self.buffer_device,
                **self._buffer_kwargs(),
                x_fn=None,  # 'latent_params',
                y_fn=self._buffer_y_fn(),
                exclude_keys=CHURNED_BUFFER_EXCLUDE_KEYS,
            )

        num_bwd_steps = self.bwd_step_delta()

        # always churn at least a little bit
        churn_batch_ref = getattr(self.args.buffers.prior_buffer, 'churn_batch_ref', self.batch_size)
        n_churn = max(1000,
                      int((num_bwd_steps / self.args.buffers.prior_buffer.mean_lifetime)
                          * churn_batch_ref))
        n_to_add = min(self.args.eval_num_samples,
                       n_churn)  # cap unrelated to GPU batch size -- eval_batch_size is retired, this is just a churn-rate limiter

        # membership is admission -- drop rows that would no longer clear their
        # own condition's gate. Runs BEFORE headroom is measured so its evictions
        # become intake room for this same call's churn cycle, which demotes the
        # loss-quantile branch below from the primary eviction channel to an
        # overflow handler.
        self._expire_stale_prior_rows()

        headroom = max(0, self.args.buffers.prior_buffer.max_size - len(self.prior_buffer))

        # Eligibility for eviction is RELATIVE (the bottom `quantile` of
        # visited rows), never an absolute loss bar. get_elig_drop_count cuts
        # at min(loss_floor, quantile), so a finite loss_floor in NATS turns
        # into a rate gate the moment the buffer is full: with headroom 0 the
        # eligible set is the ONLY intake path, and on any problem whose
        # per-sample backward residuals sit above that bar (elj plateaus at
        # 24-27) nothing is ever eligible, n_to_add collapses to 0, and churn
        # stops for good -- silently, since the buffer just looks full and
        # quiet. Passing +inf keeps the cut at the quantile alone, so ~25% of
        # visited rows are always evictable: the RATE stays owned by the churn
        # policy (n_churn) and loss only decides WHICH rows go, which was the
        # intent. This is what makes prior_buffer.init_fraction: 1.0 safe.
        _EVICT_QUANTILE, _EVICT_MIN_VISITS = 0.25, 5

        if n_to_add > headroom:
            elig_idx, _, _ = self.prior_buffer.get_elig_drop_count(
                quantile=_EVICT_QUANTILE,
                loss_floor=float('inf'),
                min_visits=_EVICT_MIN_VISITS,
            )
            elig_to_drop = elig_idx.numel()
            # only the overflow portion needs to be backed by eligible drops
            overflow = n_to_add - headroom
            n_to_add = headroom + min(elig_to_drop, overflow)

        space_needed = max(0, len(self.prior_buffer) + n_to_add - self.args.buffers.prior_buffer.max_size)
        if space_needed > 0:
            # purge_lowest returns nothing and may purge fewer than asked (forced
            # + eligible stochastic drops only), so count the realized eviction
            # from the length delta rather than trusting space_needed
            len_before = len(self.prior_buffer)
            self.prior_buffer.purge_lowest(
                space_needed,
                quantile=_EVICT_QUANTILE,
                loss_floor=float('inf'),
                min_visits=_EVICT_MIN_VISITS,
            )
            self.prior_churn['evicted'] += len_before - len(self.prior_buffer)

        self._prior_churn_cycle(n_to_add)

        # reach trigger: even prior_buffer's own upper tail is still hugging its
        # own condition's floor instead of reaching up toward that condition's
        # best anchor -- it isn't discovering/retaining good samples on its own,
        # so actively replace some of its worst material with anchor-sourced
        # top-up rather than waiting for a shortfall to expose the problem.
        # excess-above-condition-best (not raw energy/reward) is pooled across
        # the whole buffer for the quantile, so one condition's easier scale
        # can't make the buffer look healthier than it actually is elsewhere.
        if getattr(self, 'anchor_buffer', None) is not None and len(self.anchor_buffer) > 0 and len(
                self.prior_buffer) > 0:
            cfg = self.args.buffers.anchor_buffer
            margin = self._ramp_params()[0]
            # host-side cid: _condition_energy_floor returns on the input's
            # device, and everything below (y, quantile) is CPU bookkeeping --
            # with buffer_device: cuda the raw batch attr is a CUDA tensor
            prior_condition_id = self.prior_buffer.batch.condition_id.detach().cpu()
            energy_floor = self._condition_energy_floor(prior_condition_id)
            if energy_floor is not None:
                valid = torch.isfinite(energy_floor)
                if valid.any():
                    excess = self.prior_buffer.y.cpu()[valid] - energy_floor[valid]
                    reach = 1.0 - torch.quantile(excess, cfg.reach_quantile).item() / margin
                    if reach < cfg.reach_threshold:
                        self.top_up_prior_from_anchors(cfg.reach_topup_size, purge_worst=True)

    def _expire_stale_prior_rows(self):
        """
        Gate-staleness eviction: MEMBERSHIP IS ADMISSION.

        A prior_buffer row belongs iff it would still be admitted today --
        `energy < Emin(c) + ramp_floor`, the identical test _prior_churn_cycle
        and top_up_prior_from_anchors gate on. Emin(c) ratchets down as the
        tracker sees better structures, so rows admitted under an older, looser
        gate stop clearing the current one.

        That is the ONLY sense in which a prior row is stale. prior_model is a
        frozen snapshot (snapshot_prior), so an old draw is still a perfectly
        valid draw from it -- unlike a replay trajectory, which goes stale
        because the policy that generated it moved. Nothing here needs a
        rollout: the energy is already stored and Emin(c) is already tracked.

        Energy-intrinsic and policy-independent, so this channel does not
        interact with the loss-quantile eviction in manage_prior_buffer, which
        it runs ahead of and thereby demotes to an overflow handler.

        Capped at `expire_max_frac` of the buffer per call, worst-excess first:
        one record-breaking anchor can drop Emin(c) far enough to stale a large
        slice of a condition at once, and an uncapped purge driven by an
        absolute bar is exactly the failure mode manage_prior_buffer's
        loss_floor=+inf comment is about (and that purge_lowest's `loss_min`
        still has). Truncation prints, never silent.

        Returns the number of rows dropped. expire_max_frac <= 0 -- the default
        for configs predating the key -- disables the channel outright.
        """
        cfg = self.args.buffers.prior_buffer
        max_frac = float(getattr(cfg, 'expire_max_frac', 0.0))
        n_rows = self._prior_buffer_len()
        if max_frac <= 0 or n_rows == 0:
            return 0

        # host-side throughout: _condition_energy_floor returns on its input's
        # device, and y/argsort/purge_by_index are all CPU bookkeeping (with
        # buffer_device: cuda the raw batch attr is a CUDA tensor)
        condition_id = self.prior_buffer.batch.condition_id.detach().cpu()
        energy_floor = self._condition_energy_floor(condition_id)
        if energy_floor is None:
            return 0  # pre-bootstrap: no tracker and no anchors, so no gate to apply

        energy_floor = energy_floor.cpu().flatten()
        # a condition with no observations carries Emin(c) = +inf, so its excess
        # is -inf and its rows are always kept -- the same convention that makes
        # a condition's FIRST sample admissible regardless of margin
        excess = self.prior_buffer.y.cpu().flatten() - energy_floor
        stale = excess >= self._ramp_params()[0]
        n_stale = int(stale.sum())
        if n_stale == 0:
            return 0

        stale_idx = torch.nonzero(stale, as_tuple=False).flatten()
        cap = max(1, int(max_frac * n_rows))
        if n_stale > cap:
            order = torch.argsort(excess[stale_idx], descending=True)
            stale_idx = stale_idx[order[:cap]]
            print(f"prior_buffer gate-staleness purge capped at {cap} of {n_stale} stale rows "
                  f"({n_rows} resident): Emin(c) is moving faster than expire_max_frac "
                  f"{max_frac} allows -- the remainder expires over subsequent calls")

        self.prior_buffer.purge_by_index(stale_idx.numpy())
        n_dropped = int(stale_idx.numel())
        self.prior_churn['evicted'] += n_dropped
        self.prior_churn['expired'] += n_dropped
        return n_dropped

    def _prior_buffer_len(self):
        """len(prior_buffer) that tolerates the buffer not existing --
        rebuild_prior_by_churn deletes it and refills from nothing, so its loop
        can't assume the attribute is there yet."""
        buff = getattr(self, 'prior_buffer', None)
        return 0 if buff is None else len(buff)

    def _admit_to_prior_buffer(self, batch):
        """Add an admitted batch, constructing prior_buffer around the first one
        when it doesn't exist yet. manage_prior_buffer builds the buffer up
        front from its own sample_batch, but the from-zero rebuild has no such
        batch in hand, and either source (prior model or anchor top-up) may be
        the one that lands first."""
        if getattr(self, 'prior_buffer', None) is None:
            self.prior_buffer = self._fresh_prior_buffer(batch)
        else:
            self.prior_buffer.add(batch)

    def _prior_churn_cycle(self, budget: int):
        """
        One churn admission cycle: draw `budget` from the prior model, admit
        what sits within ramp_floor of its own condition's best known energy,
        and backfill the shortfall from the permanent anchor archive.

        Shared by manage_prior_buffer's eval-cadence churn and
        rebuild_prior_by_churn's from-zero fill, so a rebuilt buffer holds the
        same source mix the running churn would have produced rather than a
        composition that has to be ground out afterwards.

        Before the warm-start stage's snapshot_prior action has run (or on a
        run whose protocol never takes one) no prior model exists, so the draw
        is skipped entirely and the caller's buffer is left to the anchor
        top-up paths.
        """
        if budget <= 0:
            return
        if not self._has_prior_sampler():
            # Silent here changes the buffer's SOURCE MIX to 100% anchor with no
            # error and no log line -- the only tell is prior_buffer_prior_admit_rate
            # going nan (0/0) instead of 0.0, because budget below is never
            # incremented. Reached whenever a run resumes PAST train_prior's
            # snapshot_prior on_exit action without prior_model_name set. Reports
            # once per stage rather than raising: an anchor-only buffer is a legal
            # composition, it just is not the one the config describes.
            stage_name = self.protocol.stage.name
            if getattr(self, '_no_prior_model_warned_stage', None) != stage_name:
                self._no_prior_model_warned_stage = stage_name
                print(f"WARNING [stage '{stage_name}']: prior churn requested but no "
                      f"prior_model exists -- the draw is SKIPPED and the prior buffer "
                      f"fills 100% from anchors. Set prior_model_name to a *_prior.pt, "
                      f"or run through train_prior's snapshot_prior, to restore it.")
            return

        metrics, sample_batch = self.sample_from_prior(budget)
        reward = metrics['log_r']
        energy = -reward * (10 ** metrics['log_T_tensor'])
        energy_floor = self._condition_energy_floor(metrics['condition_id'])
        if energy_floor is not None:
            good_inds = torch.argwhere(
                energy < energy_floor + self._ramp_params()[0]).flatten()
        else:
            good_inds = torch.argwhere(reward > self.args.buffers.prior_buffer.reward_min).flatten()
        if good_inds.numel() > 0:
            self._admit_to_prior_buffer(sample_batch.subsample_new_batch(good_inds))
        # denominator for the source mix: how much of the churn budget the
        # prior model earned before any anchor fallback fires below
        self.prior_churn['from_prior_model'] += int(good_inds.numel())
        self.prior_churn['budget'] += int(budget)

        if getattr(self, 'anchor_buffer', None) is None or len(self.anchor_buffer) == 0:
            return

        floor_frac = float(getattr(self.args.buffers.prior_buffer, 'anchor_floor_frac', 0.0))
        if floor_frac <= 0:
            # this cycle's prior-model draw came up short of admissible samples --
            # top up the gap from the permanent anchor archive instead of just
            # accepting a smaller churn this round
            shortfall = budget - int(good_inds.numel())
            if shortfall > 0:
                self.top_up_prior_from_anchors(shortfall)
            return

        # Per-condition intake. The pooled branch above computes ONE shortfall
        # over the whole draw and hands it to an archive-wide priority draw, so a
        # condition whose prior yield is zero carries no guarantee of any anchor
        # coverage -- the backfill lands wherever archive-wide priority sends it.
        # Here every condition the draw touched gets its own shortfall AND a
        # guaranteed quota, so the prior/anchor mix is specified rather than
        # emergent. Conditions absent from this draw are not shorted: the draw's
        # own condition sampling covers them across cycles. N_c == 1 makes the
        # two branches identical by construction.
        drawn_cid = torch.as_tensor(metrics['condition_id']).detach().cpu().long().flatten()
        if drawn_cid.numel() == 0:
            return
        admitted_cid = drawn_cid[good_inds.detach().cpu().flatten()]
        n_cid = int(drawn_cid.max().item()) + 1
        drawn_per_c = torch.bincount(drawn_cid, minlength=n_cid)
        admitted_per_c = torch.bincount(admitted_cid, minlength=n_cid)
        shortfall_per_c = (drawn_per_c - admitted_per_c).clamp(min=0)
        quota_per_c = torch.ceil(floor_frac * drawn_per_c.float()).long()
        want_per_c = torch.maximum(shortfall_per_c, quota_per_c)

        n_want = int(want_per_c.sum().item())
        if n_want > 0:
            # what the floor asked for BEYOND the shortfall the pooled branch
            # would have requested; 0 means the floor never bound this cycle
            self.prior_churn['anchor_floor'] += max(0, n_want - int(shortfall_per_c.sum().item()))
            self.top_up_prior_from_anchors(n_want, per_condition=want_per_c)

    @torch.no_grad()
    def rebuild_prior_by_churn(self, target_size: Optional[int] = None):
        """
        Stage action ('rebuild_prior_by_churn[:N]'): discard prior_buffer and
        refill it from zero by repeating the ordinary churn admission cycle
        (_prior_churn_cycle) until it holds target_size rows.

        Replaces reseed_prior_from_dataset:flush at the post-handoff boundary,
        which refilled to max_size straight from the prior dataset with no
        admission gate. That left two problems this avoids:

          composition -- the buffer arrived holding material the online gate
          would never have admitted, so backward TB trained against it while
          churn slowly ground it back out;

          rate -- a buffer at max_size has no headroom, so manage_prior_buffer
          falls through to the eligible-drop branch, and get_elig_drop_count
          needs select_counts >= min_visits, which only draws supply. Across a
          stage whose backward branch sits below deactivate_threshold nothing
          is ever drawn, so the eligible set stays empty, n_to_add collapses to
          zero and the prior-model draw is skipped for the whole stage -- the
          reach trigger's fixed top-up becomes the only intake.

        target_size defaults to buffers.prior_buffer.init_fraction of max_size.
        Stopping short of max_size is the point: the leftover headroom is what
        keeps the draw path open afterwards.

        Re-drawing each cycle (rather than one large draw split by source)
        keeps the result a genuine sample of what churn produces -- a low-yield
        prior lands anchor-dominated through the same shortfall backfill the
        eval-cadence path uses, instead of a fixed ratio imposed up front. A
        cycle that admits nothing at all ends the loop: with no prior model and
        no anchors to backfill from, further rounds cannot make progress.

        Admissions tally into prior_churn as usual, so the eval window
        containing this transition reports the rebuild's own source mix rather
        than a steady-state rate.
        """
        cfg = self.args.buffers.prior_buffer
        if target_size is None:
            target_size = int(cfg.max_size * getattr(cfg, 'init_fraction', 0.25))
        target_size = max(0, min(int(target_size), cfg.max_size))

        n_before = self._prior_buffer_len()
        # the outgoing buffer is held, not dropped, until the rebuild has
        # something to replace it with: bwd_train_step's draw doesn't guard on
        # the attribute existing, so a rebuild that admitted nothing must leave
        # the incumbent in place rather than a hole
        incumbent = getattr(self, 'prior_buffer', None)
        if hasattr(self, 'prior_buffer'):
            del self.prior_buffer
        if target_size == 0:
            print("rebuild_prior_by_churn: target_size 0, prior_buffer left empty")
            return

        per_cycle = max(1, min(cfg.min_size, target_size))
        # ceil division without importing math; 4x slack over the ideal cycle
        # count absorbs partial-admission rounds without spinning forever
        max_cycles = 4 * (-(-target_size // per_cycle)) + 4

        cycles = 0
        while self._prior_buffer_len() < target_size and cycles < max_cycles:
            before = self._prior_buffer_len()
            self._prior_churn_cycle(min(per_cycle, target_size - before))
            cycles += 1
            if self._prior_buffer_len() == before:
                print(f"rebuild_prior_by_churn: cycle {cycles} admitted nothing, stopping early")
                break
        else:
            if self._prior_buffer_len() < target_size:
                print(f"rebuild_prior_by_churn: hit the {max_cycles}-cycle cap "
                      f"{self._prior_buffer_len()}/{target_size} rows short of target -- "
                      f"admission yield is low, not a stall")

        if getattr(self, 'prior_buffer', None) is None and incumbent is not None:
            self.prior_buffer = incumbent
            print(f"rebuild_prior_by_churn: nothing admitted, restored the incumbent "
                  f"{n_before}-row buffer (no prior model and no anchors to draw from)")
            return

        print(f"rebuild_prior_by_churn: prior_buffer {n_before} -> {self._prior_buffer_len()} rows "
              f"(target {target_size}, {cycles} churn cycles, {cfg.max_size - self._prior_buffer_len()} "
              f"rows of headroom left for churn)")

    @torch.no_grad()
    def _stratified_anchor_draw(self, per_condition):
        """
        Draw anchors with EXACT per-condition counts, so a guaranteed floor is
        guaranteed rather than merely expected.

        sample_graphs' `p` argument would fix the condition mix only in
        expectation, which is not a floor -- so the index set is built per
        condition here and handed to a single subsample. Within each condition
        the draw keeps sample_graphs' own semantics: priority-weighted on
        ema_loss (the AnchorBuffer replay priority) with cfg.replay_beta of the
        slice drawn uniformly as a random floor, weighted portion WITH
        replacement (_sample_indices does the same, and ignores replace= on that
        path). select_counts is bumped exactly as sample_graphs would.

        A condition holding no anchors leaves its quota unfilled and prints:
        borrowing that shortfall from another condition would silently undo the
        stratification this exists to provide.

        Returns (graphs, inds), or (None, None) if nothing could be drawn.
        """
        cfg = self.args.buffers.anchor_buffer
        anchor_cid = self.anchor_buffer.batch.condition_id.detach().cpu().long().flatten()
        weights = np.asarray(self.anchor_buffer._loss_weights(temperature=1.0),
                             dtype=np.float64).flatten()

        chosen, starved = [], []
        for cid in torch.nonzero(per_condition > 0, as_tuple=False).flatten().tolist():
            k = int(per_condition[cid].item())
            pool = torch.nonzero(anchor_cid == cid, as_tuple=False).flatten().numpy()
            if pool.size == 0:
                starved.append(cid)
                continue

            n_uniform = max(1, int(k * cfg.replay_beta))
            n_weighted = max(0, k - n_uniform)
            picks = [np.random.choice(pool, size=n_uniform, replace=n_uniform > pool.size)]
            if n_weighted > 0:
                p = weights[pool]
                total = p.sum()
                # a condition whose anchors are all NaN/zero-weight falls back to
                # uniform rather than raising inside np.random.choice
                p = (p / total) if (np.isfinite(total) and total > 0) else None
                picks.append(np.random.choice(pool, size=n_weighted, replace=True, p=p))
            chosen.append(np.concatenate(picks))

        if starved:
            print(f"anchor floor unfilled for {len(starved)} condition(s) holding no anchors "
                  f"(e.g. {starved[:5]}) -- quota skipped, never reassigned")
        if not chosen:
            return None, None

        inds = np.concatenate(chosen)
        self.anchor_buffer._bump_counts(inds)
        graphs = self.anchor_buffer.batch.subsample_new_batch(inds)
        graphs = self.anchor_buffer._drop_keys(
            graphs, ("symmetry_operators", "smiles", "identifier"))
        return graphs, inds

    def top_up_prior_from_anchors(self, n, purge_worst: bool = False, per_condition=None):
        """
        Top up prior_buffer from the anchor buffer: isotropically noise a
        batch of anchors in latent space and rescore them -- the same
        noise-then-rescore pattern already used by
        substitute_prior/calibrate_prior_noise in utils.py, just sourced from
        the permanent archive instead of the live buffer being calibrated.

        Anchors are drawn priority-weighted (AnchorBuffer.ema_loss, repurposed
        here as a replay-priority EMA -- see AnchorBuffer's docstring) with a
        random floor (cfg.replay_beta), rather than uniformly. The priority
        EMA itself is refreshed elsewhere, by refresh_anchor_buffer_surprise's
        periodic full sweep over the whole anchor buffer -- not here. An
        earlier version of this method measured each noised child's own
        surprise and routed it back to its parent per-draw, but that's a
        biased partial refresh (only touches priority-weighted-plus-floor
        draws) and was dropped once the full sweep existed to do the same job
        uniformly and without that bias.

        Children that strictly improve their condition's Emin(c) additionally
        stand for anchor admission (cfg.topup_admit_record_breakers) -- see
        the inline comment at that block for why no surprise measurement is
        spent on them. Previously these children could set records (and
        thereby trigger thin()'s energy-window purges against other anchors)
        while being structurally exempt from admission themselves.

        purge_worst: if True, first purge up to n of prior_buffer's worst
        entries by excess above their OWN condition's minimum, so the
        anchor-sourced batch actively replaces stale/pinned material instead of
        just padding on top of it (used by the reach trigger; the shortfall
        trigger leaves this False since headroom for that case is already
        handled upstream).

        per_condition: optional per-condition-id count vector. None keeps the
        archive-wide priority draw. Supplied, the draw is stratified to those
        exact counts (_stratified_anchor_draw) so an anchor floor is guaranteed
        per condition rather than merely expected -- `n` must equal its sum.
        """
        cfg = self.args.buffers.anchor_buffer

        if purge_worst and self._prior_buffer_len() > 0:
            n_purge = min(n, len(self.prior_buffer))
            # Rank on EXCESS above each row's own condition minimum, not raw
            # energy. Absolute elj scales with molecule size, so a pooled raw-y
            # sort strips the small-molecule conditions wholesale on a
            # multi-condition run however good their structures are for their
            # own condition. The reach trigger that fires this already measures
            # excess (manage_prior_buffer) -- this is the same statistic applied
            # to the purge it drives. An unobserved condition's +inf floor sends
            # its rows to -inf, i.e. last, so they are never purged: the same
            # convention that makes a condition's first sample admissible.
            y = self.prior_buffer.y.cpu().flatten()
            energy_floor = self._condition_energy_floor(
                self.prior_buffer.batch.condition_id.detach().cpu())
            score = y if energy_floor is None else y - energy_floor.cpu().flatten()
            worst_first = torch.argsort(score, descending=True)
            self.prior_buffer.purge_by_index(worst_first[:n_purge].numpy())
            self.prior_churn['evicted'] += int(n_purge)

        if per_condition is None:
            n_draw = min(n, len(self.anchor_buffer))
            anchor_batch, anchor_inds, _ = self.anchor_buffer.sample_graphs(
                n_draw, replace=False, weighted=True, temperature=1.0, beta=cfg.replay_beta)
        else:
            anchor_batch, anchor_inds = self._stratified_anchor_draw(per_condition)
            if anchor_batch is None:
                return
        anchor_batch = anchor_batch.clone().to(self.device)
        anchor_batch, log_T_tensor, condition, condition_id = self._noise_and_condition(
            anchor_batch, cfg.noise_log_range)

        terminal_latents = self._batch_latents(anchor_batch)
        # Bulk anchor scan, not the per-step hot path, and it runs inside a stage
        # transition's on_enter hook -- OUTSIDE the try/except around train_step that
        # slashes the batch on OOM. So an OOM here is fatal with nothing to catch it.
        # Force the chunked self-healing path, same as the two init-time whole-dataset
        # scans do. Per-call only: it does not touch self.internal_oom_recovery.
        reward, anchor_batch = self.energy_function.log_reward(
            terminal_latents, anchor_batch, log_T_tensor, return_exp=True,
            internal_oom_recovery=True)

        temperature = 10 ** log_T_tensor
        energy = -reward.detach() * temperature

        old_best = None
        if hasattr(self, 'condition_log_z'):
            # the one buffer-fill path that doesn't route through fwd_eval_sampling
            # (it noises+rescores stored anchors directly), so it needs its own hook.
            # Emin(c) is snapshotted first so the record-breaker admission block
            # below can identify which children strictly improved it.
            old_best = self.condition_log_z.best_energy[condition_id.detach().cpu().flatten()].clone()
            self.condition_log_z.update_best_energy(condition_id, energy)

        energy_floor = self._condition_energy_floor(condition_id)
        if energy_floor is not None:
            good_inds = torch.argwhere(
                energy < energy_floor + self._ramp_params()[0]).flatten()
        else:
            good_inds = torch.argwhere(reward > self.args.buffers.prior_buffer.reward_min).flatten()
        if good_inds.numel() > 0:
            self._admit_to_prior_buffer(anchor_batch.subsample_new_batch(good_inds))
        # accumulated across calls (shortfall + reach trigger can both fire in
        # one cycle); log_buffer_stats zeroes it on read
        self.prior_churn['from_anchors'] += int(good_inds.numel())

        # Record-breaker admission: each drawn anchor's noised child stands for
        # admission iff it STRICTLY lowered its condition's Emin(c). A parent can
        # appear more than once -- _sample_indices ignores replace= entirely on
        # the beta>0 path and draws the weighted portion WITH replacement, as
        # does the stratified draw -- so AnchorBuffer.admit's same-condition
        # dup_cutoff pass is what resolves repeats, not the draw. No
        # surprise measurement is spent here -- the child is a strictly deeper
        # version of an already-vetted anchor, so it inherits its parent's
        # frozen original_surprise, and AnchorBuffer.admit's same-condition
        # dup_cutoff pass does the near-duplicate resolution: with small topup
        # noise the child usually lands within dup_cutoff of its parent and
        # replaces it in place (strictly-lower-energy rule), deepening the
        # archive rather than growing it. The strict-record-breaker
        # requirement is the anti-drift guard: per condition, admitted
        # children can only descend monotonically toward the physical
        # envelope, never wander laterally or revisit known depths -- so this
        # can't behave like an unguided MCMC chain concentrating into wells.
        # false/absent disables, restoring the old behavior (children update
        # Emin(c) but never stand for admission).
        self.last_anchor_topup_admitted = getattr(self, 'last_anchor_topup_admitted', 0)
        if getattr(cfg, 'topup_admit_record_breakers', False) and old_best is not None:
            energy_cpu = energy.detach().cpu().flatten()
            improved = torch.nonzero(energy_cpu < old_best, as_tuple=False).flatten()
            if improved.numel() > 0:
                parent_inds = torch.as_tensor(anchor_inds, dtype=torch.long).flatten().cpu()
                parent_surprise = self.anchor_buffer.original_surprise[parent_inds[improved]].clone()
                admit_batch = anchor_batch.subsample_new_batch(
                    improved.to(anchor_batch.device)).cpu()
                self.last_anchor_topup_admitted += self.anchor_buffer.admit(
                    admit_batch,
                    reward.detach().cpu().flatten()[improved],
                    energy_cpu[improved],
                    dup_cutoff=cfg.dup_cutoff, admit_range=None,
                    original_surprise=parent_surprise,
                )
                if len(self.anchor_buffer) > cfg.max_size:
                    self.anchor_buffer.thin(
                        self.condition_log_z.best_energy,
                        energy_window=cfg.thin_energy_window,
                        max_size=cfg.max_size,
                    )

    @torch.no_grad()
    def seed_prior_from_condition_minima(self, n_per_condition: int = 1, flush: bool = False):
        """
        Deterministic prior_buffer seed: one row per condition present in
        anchor_buffer (its lowest-energy entry, AnchorBuffer.
        best_per_condition_indices -- not a priority-weighted random draw, so
        coverage doesn't degrade as the condition library grows), tiled
        n_per_condition times and isotropically noised
        (log_noise_latent_parameters, same cfg.noise_log_range as
        top_up_prior_from_anchors). Total set size is exactly
        n_conditions_present * n_per_condition. Rescored and added to
        prior_buffer unconditionally (no energy-floor gate: these are each
        condition's own best known state, tiled and noised).

        flush=True ('seed_prior_from_anchors:N:flush') REPLACES the buffer
        with the seed set instead of adding to it: the localized-anchor_seed
        variant, where backward TB must fit the tight noise ball around each
        condition's best anchor rather than the (possibly broad, dataset-
        seeded) incumbent content. Pair with buffers_active: false on the
        localized stages -- otherwise eval-cadence churn (admissions within
        ramp_floor of best + anchor top-ups) re-broadens the buffer -- and
        with reseed_prior_from_dataset at the following stage boundary to
        restore coverage wholesale.

        Each tiled copy gets ONE log-uniform isotropic latent kick
        (noise_log_range). The local Metropolis reheat that used to replace
        that kick -- relaxing every copy at its own target temperature so the
        seed approximated each basin's thermal shape -- is deleted along with
        its `mcmc` config block; it is in git history if the staged
        anchor_seed -> z_match route it was built for ever returns.
        """
        cfg = self.args.buffers.anchor_buffer
        uniq_ids, row_idx = self.anchor_buffer.best_per_condition_indices()
        if row_idx.numel() == 0:
            return

        tiled_idx = row_idx.repeat_interleave(n_per_condition)
        seed_batch = self.anchor_buffer.batch.subsample_new_batch(tiled_idx).clone().to(self.device)

        # noised BEFORE conditioning, so the noised state is what gets
        # conditioned, oriented and scored
        seed_batch, log_T_tensor, condition, condition_id = self._noise_and_condition(
            seed_batch, cfg.noise_log_range)

        terminal_latents = self._batch_latents(seed_batch)
        reward, seed_batch = self.energy_function.log_reward(
            terminal_latents, seed_batch, log_T_tensor, return_exp=True)

        temperature = 10 ** log_T_tensor
        energy = -reward.detach() * temperature

        if hasattr(self, 'condition_log_z'):
            self.condition_log_z.update_best_energy(condition_id, energy)

        good_inds = torch.argwhere(torch.isfinite(energy)).flatten()
        if good_inds.numel() > 0:
            if flush:
                n_before = len(self.prior_buffer) if hasattr(self, 'prior_buffer') else 0
                self.prior_buffer = self._fresh_prior_buffer(seed_batch.subsample_new_batch(good_inds))
                print(f"seed_prior_from_anchors (flush): prior_buffer {n_before} -> "
                      f"{len(self.prior_buffer)} rows ({n_per_condition} noised best-anchor "
                      f"copies x {uniq_ids.numel()} conditions, minus non-finite rescores)")
            else:
                self.prior_buffer.add(seed_batch.subsample_new_batch(good_inds))
        self.prior_churn['from_anchors'] += int(good_inds.numel())

    @torch.no_grad()
    def refresh_anchor_buffer_surprise(self):
        """
        Full periodic re-measurement of every anchor's current surprise,
        refreshing AnchorBuffer.ema_loss (the replay-priority EMA -- see
        AnchorBuffer's docstring) directly. Almost identical to
        bwd_eval_sampling's backward-rollout call, except:
          - one rollout (K=1) per anchor, not the confirm-grade K=5-10 used
            at admission -- this is meant to be cheap and just needs to
            cover every anchor, not produce a low-variance estimate
          - iterated as a deterministic sweep over the whole anchor buffer
            (index chunks 0:bsz, bsz:2*bsz, ...), not a random draw-with-
            replacement over a held-out eval set -- every anchor gets
            touched exactly once per call, not a sample of them
          - no loss/metric aggregation or condition_log_z update; the only
            effect is the routed update_losses call at the end

        This is the sole source of replay-priority signal: routing from
        top_up_prior_from_anchors's own noised-child rollouts was dropped
        as a biased partial refresh (only touches priority-weighted-plus-
        floor draws), and continuous routing from real backward-training
        draws was ruled out as too invasive (anchor_buffer positions aren't
        stable across thin()'s reindexing, so it would need its own stable-
        id scheme). A periodic full sweep sidesteps both: every anchor is
        touched uniformly, and since indices are this call's own fresh
        torch.arange(n) rather than anything held across calls, there's no
        staleness to worry about.

        Called from evaluation()'s anchor_buffer block on its own cadence
        (buffers.anchor_buffer.refresh_every_n_evals), independent of
        thin()'s cadence -- a full sweep needs to guarantee coverage, so
        it's paced by a period rather than the rate-based churn prior_buffer/
        replay_buffer use elsewhere (see discussion at the call site).
        """
        if not hasattr(self, 'anchor_buffer') or len(self.anchor_buffer) == 0:
            return

        eval_discretizer = lambda bsz: uniform_discretizer(bsz, self.args.eval_T)
        n = len(self.anchor_buffer)
        condition = self.anchor_buffer.batch.conditions.detach().to(self.device)
        latents = self.anchor_buffer.x.to(self.device)

        # centering terms: the refresh measures the SAME TB-residual axis as
        # admission/thin (screen_and_admit_anchors: log_r - log_Z(c) - log_p_hat,
        # larger = more surprising = more under-credited relative to merit),
        # NOT raw unreachability -(log_pf - log_pb), which it used to track.
        # The uncentered form is not comparable across conditions: anchors in
        # intrinsically diffuse/low-Z conditions score high forever -- even
        # once the policy weights them correctly -- so the priority-weighted
        # top-up draws (this EMA's sole consumer) skew toward hard-to-reach
        # conditions wholesale and never rotate off absorbed modes. Centering
        # costs nothing: reward is stored per anchor and log_Z(c) is a tracker
        # lookup. Where a condition's tracker slot isn't warmed (z_mask False,
        # e.g. seeded anchors), fall back to the uncentered value.
        reward_cpu = self.anchor_buffer.reward.cpu().flatten()
        if hasattr(self, 'condition_log_z'):
            anchor_cid = self.anchor_buffer.batch.condition_id.detach().cpu().flatten()
            log_Z_c, z_mask = self.condition_log_z.lookup(anchor_cid)
            log_Z_c, z_mask = log_Z_c.cpu(), z_mask.cpu()
        else:
            log_Z_c = torch.zeros(n)
            z_mask = torch.zeros(n, dtype=torch.bool)

        surprise = torch.empty(n, dtype=torch.float32)
        start = 0
        while start < n:
            end = min(start + self.batch_size, n)
            idx = torch.arange(start, end, device=self.device)
            try:
                chunk_batch = self.anchor_buffer.batch.subsample_new_batch(idx).to(self.device)
                _, log_pfs, log_pbs, _ = self.ema_model.get_traj_bwd(
                    latents[idx], eval_discretizer, condition[idx], chunk_batch)
            except (RuntimeError, ValueError) as e:
                self.handle_train_epoch_error(e, 'anchor_refresh')
                continue
            log_p_est = (log_pfs.sum(-1) - log_pbs.sum(-1)).cpu()  # k=1 estimate of log_pf - log_pb
            idx_cpu = torch.arange(start, end)
            centered = reward_cpu[idx_cpu] - log_Z_c[idx_cpu] - log_p_est
            surprise[start:end] = torch.where(z_mask[idx_cpu], centered, -log_p_est)
            start = end

        self.anchor_buffer.update_losses(surprise, torch.arange(n))

    def bwd_step_delta(self):
        if not hasattr(self, 'prev_bwd_step_count'):
            num_bwd_steps = self.args.eval_period
        else:
            num_bwd_steps = self.bwd_step_count - self.prev_bwd_step_count
        self.prev_bwd_step_count = self.bwd_step_count
        return num_bwd_steps

    def replay_in_play(self) -> bool:
        """Has the CURRENT stage any use for the replay buffer? False means
        manage_replay_buffer is skipped outright -- no admission, no eviction,
        and (the part that actually costs) no per-step D2H of flow_states.

        DERIVED FROM THE STAGE, not from a config switch, so a protocol that
        never trains replay simply never builds the buffer, and one that does
        cannot be starved by a stale key. The three consumers, each named:

          - the fused replay branch. mode_boostable('replay') is the engine's
            own answer to "can this stage's balance raise replay off the floor"
            (fused_train_step gates the branch on exactly it), with one
            correction: active_modes counts a PINNED mode by the presence of
            its key, so `pinned: {replay: 0.0}` -- var_conditioning's way of
            saying "replay off" -- reads as boostable. A pin at zero is a pin at
            zero, so it is read by VALUE here.
          - the ray probe, which draws from replay (see _draw_probe_batch).
          - z_calibration mode 'replay', which draws from it too.

        Cost of being wrong in the OFF direction is bounded and self-healing:
        the buffer is supply-paced, so a stage that does want replay fills it
        from its own first fwd steps (churn_rate rows per step, equilibrium
        occupancy churn_rate x mean_residence_steps) instead of inheriting a
        warm one from the stage before. Rows that survive from an earlier
        managed stage are aged out by the backstop on the first managed call.
        That is the opposite of the tsched_july24 failure this must not
        recreate: there the buffer emptied DURING a stage that was drawing from
        it, because intake was paced demand-side. Nothing here changes intake
        while a stage draws.
        """
        stage = self.protocol.stage
        sensor = stage.lr_sensor
        if sensor is not None and sensor['kind'] == 'ray':
            return True
        zc = getattr(self.args, 'z_calibration', None)
        if getattr(zc, 'enabled', False) and getattr(zc, 'mode', None) == 'replay':
            return True
        if stage.train_mode != 'fused':
            return False  # TRAIN_MODES is ('bwd', 'fused'); only fused has branches
        if not self.protocol.mode_boostable('replay'):
            return False
        bal = stage.balance
        if bal is not None and bal['kind'] in ('proportional', 'constraint', 'ratio'):
            pinned = bal.get('pinned') or {}
            if 'replay' in pinned:
                return float(pinned['replay']) > 0.0
        return True

    def manage_replay_buffer(self, fwd_stats, sample_batch):
        """
        Store the full forward trajectory of on-policy samples with strongly
        over- or under-weighted terminals, so they can be replayed exactly
        (get_traj_replay) instead of re-sampled backward.

        NO-OP in a stage with no use for the buffer (replay_in_play), which is
        checked FIRST -- ahead of the residual arithmetic and the flow_states
        transfer, so an unused buffer costs nothing rather than costing the
        expensive part of a managed call.

        ADMISSION is unconditionally uniform over the sane pool (decisions.md
        D5, to_do_rebuild.md Phase 3 step 3). Selection lives entirely at the
        DRAW (buffers.replay_buffer.prioritise, draw_replay_sample): p ~
        delta_plus^kappa with self-normalised IS weights that undo it. A
        residual-scored admission rule would shape the buffer's density on
        top of that correction, re-entering the force spectrum uncorrected
        and counting the residual twice.

        PURGE is hazard (memoryless) + backstop (age) only -- both are
        residual-independent, so p_survive drops out of the IS weight cleanly.
        The old residual-dependent causes (floor: ema_loss corrected below a
        threshold; stalled: a draw count paired with a delta threshold) are
        retired: once the draw itself is prioritised, a corrected row already
        draws at p ~ 0, so they were buying memory, not gradient budget, and
        hazard reclaims that memory without reintroducing a residual-dependent
        survival probability. Their config keys are rejected at LOAD
        (utils._RETIRED_KEYS), not here -- a guard in this function first runs
        at a stage transition, which is hours into a run.

        Not just a variance-reduction gap -- it fabricated evidence. A row
        driven to delta ~ 0 by repeated replay IS the memorisation sensor's
        positive case (birth_loss vs ema_loss); floor/stalled purging it
        early would have deleted the evidence the sensor reads.

        DISPLACEMENT purge (freeing headroom beyond what toxic eviction frees)
        is uniform-random over the live pool, same reasoning: no residual-
        scored cap survives now that admission itself carries none.

        Negative-residual rows are admitted deliberately: they are dormant,
        not dead (delta moves as the policy moves), they draw with p ~ 0 under
        delta_plus so they cost memory rather than gradient budget, and
        filtering them CENSORS rather than reweights -- a row admitted at
        delta < 0 that would have gone positive is unrecoverable.

        Hard exclusions, upstream of admission and applied unconditionally:
        non-finite log_r/resid, and log_r < admit_reward_min. Eligibility is a
        BADNESS criterion, so without a reward floor the buffer's energy
        distribution is unbounded above.
        """
        in_play = self.replay_in_play()
        if in_play is not self._replay_managed:
            # announced on every flip (both directions) so the transition print
            # is followed by an explicit statement of what the buffer is doing,
            # rather than leaving "is replay churning?" to be inferred from
            # whether replay_buffer_* metrics moved
            self._replay_managed = in_play
            print(f"replay buffer: management {'ON' if in_play else 'OFF'} under stage "
                  f"'{self.protocol.stage.name}'"
                  + ('' if in_play else " -- nothing in it trains replay, draws from it, "
                                        "or probes it (replay_in_play)"))
        if not in_play:
            return

        log_r = fwd_stats['log_r']
        log_pf = fwd_stats['log_pf']
        log_pb = fwd_stats['log_pb']
        log_Z_learned = fwd_stats['log_Z_learned'] if 'log_Z_learned' in fwd_stats else fwd_stats['log_Z']

        # resid stays on CPU: all the eviction logic below runs against the
        # buffer's CPU-resident ema_loss bookkeeping
        log_r_cpu = log_r.cpu()
        resid = ((log_pf - log_pb) - (log_r - log_Z_learned)).cpu()
        # domain-sanity gate: hard-exclude, never throttled (see docstring)
        sane = torch.isfinite(log_r_cpu) & torch.isfinite(resid)

        rb_cfg = self.args.buffers.replay_buffer
        # Absolute reward floor, on the same hard-exclude footing as the
        # finiteness check above: without it a physically garbage state is
        # admitted at full weight the moment the policy is badly calibrated
        # on it. None disables.
        reward_min = getattr(rb_cfg, 'admit_reward_min', None)
        if reward_min is not None:
            below = sane & (log_r_cpu < float(reward_min))
            self.replay_churn['reward_rejected'] += int(below.sum())
            sane &= ~below

        elig = torch.argwhere(sane).flatten()
        # trajectories go wherever the buffer lives -- no forced D2H when GPU-resident
        flow_states = fwd_stats['flow_states'].detach().to(self.buffer_device)

        # --- bootstrap ---
        if not hasattr(self, 'replay_buffer'):
            if elig.numel() == 0:
                return
            add_inds = elig[_uniform_draw(elig.numel(), rb_cfg.max_size)]
            self.replay_buffer = self.buffer_cls(
                sample_batch.subsample_new_batch(add_inds),
                device=self.buffer_device,
                **self._buffer_kwargs(),
                x_fn=None,
                y_fn=self._buffer_y_fn(),
                traj=flow_states[add_inds.to(flow_states.device)],
                init_loss=resid[add_inds].abs(),
                exclude_keys=CHURNED_BUFFER_EXCLUDE_KEYS,
                birth_step=self.step_ind,
            )
            self.replay_churn['admitted'] += int(add_inds.numel())
            return

        # Single-pass churn: every eviction source (toxic/TTL, softmax purge)
        # is collected against the CURRENT indexing, then ONE purge_by_index
        # and ONE add run at the end -- purge_by_index rebuilds the whole
        # resident store, so doing it once instead of per-source matters.
        ema = self.replay_buffer.ema_loss
        n = len(self.replay_buffer)

        # EVICTION, two causes -- neither shapes the bulk of the age
        # distribution, which is the point:
        #   hazard    memoryless: evict n/tau rows per call, uniformly at
        #             random -> exponential residence, mean tau, CV ~1. A WIDE
        #             lag distribution, so a strong lowpass on the
        #             policy->buffer->gradient path.
        #   backstop  hard ceiling at backstop_mult * tau, binding on
        #             ~exp(-backstop_mult) of rows. Bounds worst-case staleness
        #             without reshaping the bulk.
        # Both are residual-independent, so p_survive drops out of the IS
        # weight cleanly -- no residual-scored purge cause survives now that
        # the draw itself carries the prioritisation. The keys that drove the
        # old floor/stalled causes are rejected at LOAD (utils._RETIRED_KEYS);
        # tau = 0 here means neither of these two arms, which is why
        # mean_residence_steps is load-bearing rather than optional.
        # Occupancy is emergent (Little's law: n = admit_rate * tau); max_size
        # is a memory guard, not a target. Eviction is proportional to n, so
        # occupancy decays toward the intake equilibrium rather than falling off
        # a cliff at a fixed age. The hazard budget is per manage CALL, matching
        # churn_rate.
        age = self.step_ind - self.replay_buffer.birth_step
        tau = float(getattr(rb_cfg, 'mean_residence_steps', 0) or 0)
        expired_mask = torch.zeros(n, dtype=torch.bool)
        hazard_mask = torch.zeros(n, dtype=torch.bool)
        if tau > 0:
            backstop = int(tau * float(getattr(rb_cfg, 'backstop_mult', 5.0)))
            if backstop > 0:
                expired_mask = age > backstop
            # hazard draws only from rows backstop didn't already claim, so
            # the per-call budget is not silently spent on rows that were
            # leaving anyway
            surv = torch.argwhere(~expired_mask).flatten()
            n_hazard = int(round(surv.numel() / tau))
            if n_hazard > 0:
                hazard_mask[surv[torch.randperm(surv.numel())[:n_hazard]]] = True

        toxic_mask = expired_mask | hazard_mask
        toxic = torch.argwhere(toxic_mask).flatten()

        # --- cohort telemetry. backstop is tallied as a plain count. Death-vs
        # -birth deltas are tallied on the HAZARD cohort specifically: random
        # eviction is independent of both the loss value and age, so it's the
        # one exit carrying no selection bias in either direction -- it reads
        # the live population. (An age-based TTL cohort would over-sample the
        # long-lived high-|resid| rows the draw protects, which is what made
        # an earlier version of this delta ambiguous between "learning" and
        # "survivor composition".) NB an undrawn row's ema never updates after
        # admission (update_losses only touches drawn rows), so deltas are
        # only defined on the drawn subset; undrawn expiries are counted
        # separately as wasted slots.
        self.replay_cohort['backstop'] += int(expired_mask.sum())
        hazard_cohort = hazard_mask
        n_expired = int(hazard_cohort.sum())
        if n_expired > 0:
            counts = self.replay_buffer.select_counts[hazard_cohort]
            drawn = counts > 0
            delta = (ema[hazard_cohort] - self.replay_buffer.birth_loss[hazard_cohort])[drawn]
            delta = delta[torch.isfinite(delta)]
            self.replay_cohort['expired'] += n_expired
            self.replay_cohort['expired_undrawn'] += int((~drawn).sum())
            self.replay_cohort['expired_drawn'] += int(drawn.sum())
            self.replay_cohort['expired_draws_sum'] += int(counts[drawn].sum())
            self.replay_cohort['expired_delta_sum'] += float(delta.sum())
            self.replay_cohort['expired_delta_n'] += int(delta.numel())

        # --- admission: uniform draw without replacement over this batch's
        # eligible pool, budgeted churn_rate rows PER MANAGE CALL (one call
        # per fwd/fused train step, plus one per eval). SUPPLY-side pacing,
        # deliberately: v1 paced this by elapsed REPLAY steps (demand-side),
        # which deadlocked -- replay only runs on a non-empty buffer, so the
        # admission budget required replay steps while replay steps required
        # admissions. Any TTL mass-expiry during a replay-dormant stage
        # (z_match, replay frac 0.001) zeroed the buffer PERMANENTLY, and
        # buildout then rode the variance blowup into terminal-rewind
        # sawtooths (survival was a race between the z_match stage's duration
        # and the since-retired hard age cap). Supply-side intake
        # runs whether or not replay is training, so the buffer is warm
        # whenever a stage wants to draw from it ---
        n_admit = min(elig.numel(), int(rb_cfg.churn_rate))
        add_inds = elig[_uniform_draw(elig.numel(), n_admit)]

        # --- purge: TTL/toxic eviction frees headroom first; a uniform-random
        # draw over the LIVE incumbents covers whatever admission needs beyond
        # that -- residual-independent, same reasoning as hazard/backstop ---
        headroom = max(0, rb_cfg.max_size - (n - toxic.numel()))
        n_extra_purge = max(0, add_inds.numel() - headroom)
        extra_purge = torch.zeros(0, dtype=torch.long)
        if n_extra_purge > 0:
            live = torch.argwhere(~toxic_mask).flatten()
            extra_purge = live[_uniform_draw(live.numel(), n_extra_purge)]

        purge_idx = torch.cat([toxic, extra_purge])

        t_purge = time()
        if purge_idx.numel() > 0:
            self.replay_buffer.purge_by_index(purge_idx)
            self.replay_churn['evicted'] += int(purge_idx.numel())
        t_add = time()

        if add_inds.numel() > 0:
            self.replay_buffer.add(
                sample_batch.subsample_new_batch(add_inds),
                traj=flow_states[add_inds.to(flow_states.device)],
                init_loss=resid[add_inds].abs(),
                birth_step=self.step_ind,
            )
            self.replay_churn['admitted'] += int(add_inds.numel())

        # tail probes (see ten_step_reporting): wall time is what the train step
        # actually pays, syncs included -- deliberately no cuda.synchronize here
        self._probe_max('churn_purge_ms_max', (t_add - t_purge) * 1e3)
        self._probe_max('churn_add_ms_max', (time() - t_add) * 1e3)
        self._probe_max('churn_purged_max', int(purge_idx.numel()))
        self._probe_max('churn_added_max', int(add_inds.numel()))

    def init_prior_buffer_seed(self):
        """
        Optionally pre-populate self.prior_buffer at init time from the
        prebuilt prior dataset, instead of letting it start empty and fill at
        churn rate (manage_prior_buffer) once a buffers_active stage starts drawing from
        it. buffers.prior_buffer.seed_source:
          'generated'      -- default, unchanged: lazy creation from the first
                              fwd-eval batch, then churn-rate filling
          'prior_dataset'  -- seed from self.prior_dataset.batch (already
                              re-analyzed at init and carrying mol_id after
                              init_identifiers(), so it matches generated
                              candidates' append_batch key set), randomly
                              subsampled to max_size if larger
        Seeding constructs a fresh CrystalBuffer, so every per-sample record
        (ema_loss, select_counts, ema_logw) starts clean: churn/purge
        priorities re-form under the current policy instead of inheriting the
        dataset's own warm-start bookkeeping. Must run after init_identifiers()
        and is skipped when a checkpoint-restored prior_buffer already exists.
        """
        if hasattr(self, 'prior_buffer'):
            return  # already populated from a reloaded checkpoint -- don't clobber it
        if getattr(self.args.buffers.prior_buffer, 'seed_source', 'generated') != 'prior_dataset':
            return

        seed_batch = self._prior_dataset_seed_batch(self.args.buffers.prior_buffer.max_size)
        self.prior_buffer = self._fresh_prior_buffer(seed_batch)
        print(f"Seeded prior_buffer with {len(self.prior_buffer)} prior-dataset samples (fresh loss records)")

    def _prior_dataset_seed_batch(self, limit: int):
        """
        The prior-dataset draw both seeding paths share (init_prior_buffer_seed
        and the reseed_prior_from_dataset stage action): clone, random-subsample
        to `limit` if larger, then give it the exact treatment manage_prior_buffer's
        generated candidates get -- append_batch demands exact key parity, so
        condition_samples attaches `conditions`/`condition_id` (riding along
        into the buffer generically), with the batch's own sg_ind/z_prime
        passed through so prebuilt values are honored rather than resampled;
        then drop the string keys candidates never carry (sample_graphs drops
        them at draw time; identifier was already consumed into mol_id by
        init_identifiers()).
        """
        seed_batch = self.prior_dataset.batch.clone()
        if seed_batch.num_graphs > limit:
            keep = torch.randperm(seed_batch.num_graphs)[:limit]
            seed_batch = seed_batch.subsample_new_batch(keep)
        seed_batch = seed_batch.to(self.device)
        seed_batch, _, _, _ = self.energy_function.condition_samples(
            seed_batch,
            sg_inds=getattr(seed_batch, 'sg_ind', None),
            z_primes=getattr(seed_batch, 'z_prime', None))
        seed_batch = AnchorBuffer._drop_keys(seed_batch, ("smiles", "identifier"))
        return seed_batch

    def _fresh_prior_buffer(self, seed_batch):
        """Construct a new CrystalBuffer around seed_batch with clean
        per-sample records -- the single construction recipe shared by init
        seeding and seed_prior_from_condition_minima's flush path."""
        return self.buffer_cls(
            seed_batch,
            device=self.buffer_device,
            **self._buffer_kwargs(),
            x_fn=None,
            y_fn=self._buffer_y_fn(),
            exclude_keys=CHURNED_BUFFER_EXCLUDE_KEYS,
        )

    def reseed_prior_from_dataset(self, flush: bool = False):
        """
        Stage action (protocol: 'reseed_prior_from_dataset[:flush]'): re-add the
        prior dataset into prior_buffer with fresh per-sample records -- the
        coverage counterpart to seed_prior_from_anchors:N:flush. A localized
        anchor_seed/z_match runs on the flushed-down buffer (with
        buffers_active off so churn can't re-broaden it); this restores broad
        coverage wholesale at the next stage boundary instead of waiting on
        churn top-ups (~1k/eval window never re-broadens a 150k dataset).

        flush=True REPLACES the buffer with a fresh full-size dataset draw,
        discarding the localized anchor_seed content first. That content is the
        tight noised ball around each condition's single lowest-energy anchor --
        a heavily biased sub-population, all mass at one point -- so adding the
        dataset over it (flush=False) leaves a stripe of it in the reseeded
        buffer (a distinct diagonal cluster in the backward-TB parity plot).
        flush=False (default) is additive: the current (local) content is
        untouched and the dataset draw is subsampled to the remaining headroom.
        """
        if not hasattr(self, 'prior_dataset'):
            return
        current = len(self.prior_buffer) if hasattr(self, 'prior_buffer') else 0
        max_size = self.args.buffers.prior_buffer.max_size
        limit = max_size if flush else max_size - current
        if limit <= 0:
            return
        seed_batch = self._prior_dataset_seed_batch(limit)
        if flush or not hasattr(self, 'prior_buffer'):
            self.prior_buffer = self._fresh_prior_buffer(seed_batch)
        else:
            self.prior_buffer.add(seed_batch)
        self.prior_churn['from_seed'] += int(seed_batch.num_graphs)
        mode = '(flush, replaced)' if flush else '(additive)'
        print(f"reseed_prior_from_dataset {mode}: prior_buffer {current} -> {len(self.prior_buffer)} rows "
              f"({'replaced with ' if flush else '+'}{seed_batch.num_graphs} prior-dataset samples, "
              f"fresh loss records)")

    def init_anchor_buffer_seed(self):
        """
        Optionally pre-populate self.anchor_buffer at init time instead of
        letting screen_and_admit_anchors's lazy bootstrap build it from the
        first fwd-eval-sampled batch (its `if not hasattr(self,
        'anchor_buffer')` branch). buffers.anchor_buffer.seed_source:
          'generated'      -- default, unchanged: lazy bootstrap from generated samples
          'prior_dataset'  -- seed from self.prior_dataset, the real high-quality
                              dataset already loaded for backward training
          <a path>         -- anything else is treated as a path and torch.load'd
                              the same way molecules_path/prior_path are, for
                              side-loading a curated set of good seeds
        Since screen_and_admit_anchors only bootstraps when self.anchor_buffer
        doesn't exist yet, pre-populating it here fully replaces that path --
        no changes needed there. Seed entries carry no rollout-based surprise
        measurement (original_surprise is left NaN, per AnchorBuffer's legacy
        fallback), so they're immune to thin()'s original-surprise-ranked
        hard-cap backstop until they're confirmed/re-admitted through normal
        training.
        """
        if hasattr(self, 'anchor_buffer'):
            return  # already populated from a reloaded checkpoint -- don't clobber it

        cfg = self.args.buffers.anchor_buffer
        seed_source = getattr(cfg, 'seed_source', 'generated')
        if seed_source == 'generated':
            return

        if seed_source == 'prior_dataset':
            seed_batch = self.prior_dataset.batch.clone().to(self.device)
        else:
            seed_data = torch.load(seed_source, weights_only=False)
            if isinstance(seed_data, dict):
                seed_data = seed_data.get('equalized_prior', seed_data.get('prior', seed_data))
            seed_batch = collate_data_list(seed_data, max_z_prime=max(self.args.z_primes)) \
                if isinstance(seed_data, list) else seed_data
            seed_batch = seed_batch.to(self.device)
            # Strip lazily-built caches before anything computes off a
            # side-loaded batch. See buffer.strip_lazy_sg_caches for why.
            strip_lazy_sg_caches(seed_batch)
            _, seed_batch = self.energy_function.batched_analyze_crystal_batch(
                self._batch_latents(seed_batch), seed_batch,
                self.args.energy_config.temperature * torch.ones(
                    seed_batch.num_graphs, dtype=torch.float32, device=self.device),
                return_batch=True, internal_oom_recovery=True)
            seed_batch = seed_batch.to(self.device)

        # condition_samples resolves condition_id through mol_id (the dense
        # integer init_identifiers() minted per identifier string), not the
        # identifier itself. A side-loaded seed batch never went through
        # init_identifiers(), so without this mapping every entry would
        # collapse onto molecule index 0 (condition_samples' legacy fallback)
        # and be tracked under the wrong condition -- and mol_id-less entries
        # would break append_batch key parity against generated candidates,
        # which do carry mol_id. Registry misses are fatal: they mean
        # seed_source wasn't prepared alongside molecules_path/prior_path.
        if not hasattr(seed_batch, 'mol_id'):
            if not hasattr(seed_batch, 'identifier'):
                raise KeyError(
                    "anchor seed batch has neither mol_id nor identifier -- can't resolve "
                    "condition_id; regenerate seed_source with generate_toy_prior.py")
            missing = sorted({ident for ident in seed_batch.identifier
                              if ident not in self.identifier_registry})
            if missing:
                raise KeyError(
                    f"anchor seed batch has {len(missing)} identifiers absent from the identifier "
                    f"registry (e.g. {missing[0]!r}) -- seed_source must be prepared with the same "
                    f"identifiers as molecules_path/prior_path")
            seed_batch.add_graph_attr(
                torch.tensor([self.identifier_registry[ident] for ident in seed_batch.identifier],
                             dtype=torch.long, device=seed_batch.device),
                'mol_id')

        seed_batch, log_T_tensor, condition, condition_id = self.energy_function.condition_samples(
            seed_batch, sg_inds=getattr(seed_batch, 'sg_ind', None), z_primes=getattr(seed_batch, 'z_prime', None))
        temperature = 10 ** log_T_tensor
        reward = self.energy_function.prebuilt_sample_to_reward(seed_batch, temperature)
        energy = -reward.detach() * temperature

        # Warm each seeded condition's Emin(c) from these on-target seed
        # energies. The anchor buffer and condition_log_z are distinct objects:
        # seeding the former does NOT inform the latter, and BOTH the admission
        # plausibility gate (screen_and_admit_anchors) and thin()'s purge gate
        # calibrate against best_energy(c), not against the anchor buffer's own
        # energies. Without this, best_energy(c) stays inf until the terminal stage's prior
        # churn warms it from broad, high-energy prior-model samples -- which
        # then admits (and can't purge) those bad samples as each condition's
        # "per-condition best", exactly the behaviour these good seeds are meant
        # to pre-empt. best_energy is a protocol-independent running min, so
        # folding in real scored seed samples is always valid.
        if hasattr(self, 'condition_log_z'):
            self.condition_log_z.update_best_energy(condition_id, energy)

        # Generated candidates arrive without the string keys (sample_graphs
        # drops them at draw time), and append_batch demands key parity, so the
        # resident batch must not carry them either. identifier has already
        # been consumed into mol_id above. symmetry_operators is NOT dropped:
        # unlike the strings, condition_samples' reset_sg_info re-attaches it
        # per graph, so candidates always carry it and parity needs it kept.
        seed_batch = AnchorBuffer._drop_keys(seed_batch, ("smiles", "identifier"))

        self.anchor_buffer = self.anchor_buffer_cls(
            seed_batch,  # function-owned transient; the buffer moves it to buffer_device itself
            device=self.buffer_device,
            reward=reward.cpu(),
            energy=energy.cpu(),
            **self._buffer_kwargs(),
            exclude_keys=BULKY_ATTR_EXCLUDE_KEYS,
        )

    @torch.no_grad()
    def screen_and_admit_anchors(self, sample_batch, log_r, energy, log_pf_est):
        """
        Promote a state to the anchor buffer iff it has good Boltzmann weight
        (energy near Emin(c)) AND the forward policy under-samples it relative
        to that weight.

        "Surprise" is the TB residual, 0 at the fixed point:

            surprise = log_Z(c) + log_pf - log_pb - log_r

        surprise << 0 = an under-weighted high-reward mode. log_r and log_Z(c)
        are deterministic given (x, condition), so centring adds no rollout
        variance to log_pf - log_pb.

        Two stages:
          1. screen   energy pre-filter AND the surprise gate, both pure tensor
                      comparisons. Uses the free k=1 log_pf_est from
                      fwd_eval_sampling, so no backward rollout is spent.
                      Candidates whose condition has no warmed-up log_Z(c) are
                      held back -- the axis is not trustworthy yet.
          2. confirm  K backward rollouts (IWAE/logsumexp) on survivors only,
                      through the SAME residual form. Dominated by the best
                      rollout, so one unlucky trajectory cannot fake surprise.

        Called on both prior-model batches and on-policy eval batches, via
        fwd_eval_sampling's single call site, so one criterion gates both.
        top_up_prior_from_anchors' record-breaker children bypass it: a strictly
        deeper version of an admitted anchor needs no novelty judgement, and
        energies are real so a damaged policy cannot fake one.

        Behind a policy-health gate (health_gate_floor / health_gate_ceiling):
        admissions pause while the ruler is miscalibrated, which also means they
        pause through early buildout and briefly after stage transitions.

        original_surprise is stored frozen at admission and used by thin() to
        rank the buffer; AnchorBuffer.admit's dup_cutoff is a literal-duplicate
        catch on the confirmed set, not a novelty judgement.
        """
        if not hasattr(self, 'condition_log_z'):
            # init-time grow_prior_buffer() runs before init_condition_log_z()
            # by contract (every condition_log_z use on that path is hasattr-
            # gated, see train()); no tracker means no warmed-up log_Z(c) axis,
            # which is exactly the all-False z_mask case -> nothing to screen
            return
        cfg = self.args.buffers.anchor_buffer
        # admission count accumulated since the last log_buffer_stats read
        # (which zeroes it): this runs more than once per eval cycle (the eval
        # batch at evaluation(), then manage_prior_buffer's prior batch), so a
        # plain per-call value would be clobbered by whichever call runs last
        # -- systematically the prior batch, whose unsurprising samples the
        # gate correctly rejects, making the logged count read 0 while the
        # buffer grows
        self.last_anchor_admitted = getattr(self, 'last_anchor_admitted', 0)
        # Policy-health gate: refuse to adjudicate novelty while the ruler is
        # broken. Surprise is measured THROUGH the live policy's log_pf, so a
        # damaged policy reads its own log_pf collapse as fake surprise on
        # everything, and the resulting flood outruns condition_log_z's
        # absorption (which cancels a uniform shift only after its half-life).
        #
        # Two channels, both naming a fwd metric in config so the ruler can be
        # swapped without a code change: FLOOR blocks when its metric falls
        # below the bar, CEILING when |its metric| rises above. Bars are named
        # for their ROLE (health_gate_floor / _ceiling), not for a metric, so a
        # swap cannot leave a bar named after the metric it no longer uses.
        # A cold channel abstains rather than blocks, so warm-start seeding is
        # unaffected. floor_metric: null disables that channel.
        #
        # ⚠ A BAR DOES NOT SURVIVE A RULER SWAP. tb_resid_clipped is signed and
        # beta-bounded and its 0.5 is the Z-currency bar z_calibration holds;
        # tb_err_worst is an unbounded RMS reading 18-21 when healthy. Change
        # metric and bar together or not at all.
        floor_name = getattr(cfg, 'health_gate_floor_metric', 'r2')
        ceil_name = getattr(cfg, 'health_gate_ceiling_metric', 'tb_resid_clipped')
        floor_bar = getattr(cfg, 'health_gate_floor', 0.9)
        ceil_bar = getattr(cfg, 'health_gate_ceiling', 0.5)
        floor_val = self.metric_tracker.get('fwd', floor_name) if floor_bar is not None else None
        ceil_val = self.metric_tracker.get('fwd', ceil_name) if ceil_bar is not None else None
        if ((floor_val is not None and floor_val < float(floor_bar))
                or (ceil_val is not None and abs(ceil_val) > float(ceil_bar))):
            return
        log_r = torch.as_tensor(log_r).detach().to(self.device).flatten()
        energy = torch.as_tensor(energy).detach().to(self.device).flatten()
        log_pf_est = torch.as_tensor(log_pf_est).detach().to(self.device).flatten()
        if energy.numel() == 0:
            return

        sample_batch = sample_batch.clone().to(self.device)
        condition = sample_batch.conditions.detach().to(self.device)
        condition_id = sample_batch.condition_id.detach().to(self.device).flatten()

        best_energy = self.condition_log_z.best_energy.to(self.device)[condition_id]
        visited = torch.isfinite(best_energy)
        plausible = visited & (energy < best_energy + cfg.screen_energy_window)

        # per-condition log Z (stable ema_logw target) anchoring the TB-residual
        # axis; z_mask is False for conditions not yet warmed up, where the
        # axis -- and therefore the gate -- can't be trusted.
        log_Z_c, z_mask = self.condition_log_z.lookup(condition_id)
        log_Z_c = log_Z_c.to(self.device)
        z_mask = z_mask.to(self.device)

        # TB residual from the free k=1 forward-path estimate of log_pf - log_pb.
        # No backward rollout spent screening; confirm below re-checks
        # survivors with the proper K-sample IWAE under the same residual
        # form, so this cheap gate only narrows the confirm set.
        screen_surprise = log_Z_c + log_pf_est - log_r
        screen_idx = torch.nonzero(
            plausible & z_mask & (screen_surprise < cfg.surprise_cutoff),
            as_tuple=False).flatten()
        if screen_idx.numel() == 0:
            return

        eval_discretizer = lambda bsz: uniform_discretizer(bsz, self.args.eval_T)
        latents = self._batch_latents(sample_batch)

        K = cfg.confirm_k
        n_cand = screen_idx.numel()
        confirm_batch = sample_batch.subsample_new_batch(screen_idx)
        tile = torch.arange(n_cand, device=self.device).repeat_interleave(K)
        tiled_batch = confirm_batch.subsample_new_batch(tile)

        _, c_log_pfs, c_log_pbs, _ = self.ema_model.get_traj_bwd(
            latents[screen_idx][tile], eval_discretizer,
            condition[screen_idx][tile], tiled_batch)
        log_w_k, log_p_hat = log_pf_estimate(c_log_pfs.sum(-1), c_log_pbs.sum(-1), K)

        # same TB-residual axis as the screen gate, now on the accurate
        # K-sample log_p_hat (IWAE estimate of log_pf - log_pb) instead of the
        # k=1 screen estimate.
        confirm_surprise = log_Z_c[screen_idx] + log_p_hat - log_r[screen_idx]
        confirmed_local = torch.nonzero(confirm_surprise < cfg.confirm_cutoff, as_tuple=False).flatten()
        if confirmed_local.numel() == 0:
            return
        confirmed_idx = screen_idx[confirmed_local]

        spread = (log_w_k.max(dim=1).values - log_w_k.min(dim=1).values)[confirmed_local]
        self.last_anchor_confirm_spread = spread.mean().item()

        # store the TB residual itself, not -log_p_hat: sign flipped so larger
        # = more surprising, matching thin()'s "drop lowest first".
        original_surprise = (-confirm_surprise[confirmed_local]).cpu()
        admit_batch = sample_batch.subsample_new_batch(confirmed_idx).cpu()
        admit_reward = log_r[confirmed_idx].cpu()
        admit_energy = energy[confirmed_idx].cpu()

        if not hasattr(self, 'anchor_buffer'):
            self.anchor_buffer = self.anchor_buffer_cls(
                admit_batch, device=self.buffer_device,
                reward=admit_reward, energy=admit_energy,
                original_surprise=original_surprise,
                **self._buffer_kwargs(),
                exclude_keys=BULKY_ATTR_EXCLUDE_KEYS,
            )
            self.last_anchor_admitted += len(self.anchor_buffer)
            return

        self.last_anchor_admitted += self.anchor_buffer.admit(
            admit_batch, admit_reward, admit_energy,
            dup_cutoff=cfg.dup_cutoff, admit_range=None,
            original_surprise=original_surprise,
        )

        if len(self.anchor_buffer) > cfg.max_size:
            self.anchor_buffer.thin(
                self.condition_log_z.best_energy,
                energy_window=cfg.thin_energy_window,
                max_size=cfg.max_size,
            )

    def sample_from_prior(self, num_samples):
        "sample from prior"
        # a reused prior carries its own training T (checkpoint 'train_T'); a
        # prior trained live in this run's warm-start stage does not, so fall back to the
        # run's own rollout length. Either way, sample at the T the prior was
        # TRAINED at -- not eval_T, which need not match it (a T=10 shared prior
        # sampled at T=100 integrates 10x finer than it was fit to).
        prior_T = getattr(self, 'prior_train_T', None) or self.args.integrator.T
        eval_discretizer = lambda bsz: uniform_discretizer(bsz, prior_T)
        metrics, sample_batch = self.fwd_eval_sampling(self.prior_model,
                                                       eval_discretizer,
                                                       override_num_samples=num_samples)
        return metrics, sample_batch

    def grow_prior_buffer(self):
        if not hasattr(self, 'prior_buffer'):
            buffer_length = 0
        else:
            buffer_length = len(self.prior_buffer)

        missing = self.args.buffers.prior_buffer.max_size - buffer_length
        num_samples = min(self.args.buffers.prior_buffer.min_size, missing)
        if num_samples > 0:
            metrics, sample_batch = self.sample_from_prior(num_samples)

            if not hasattr(self, 'prior_buffer'):
                self.prior_buffer = self.buffer_cls(
                    sample_batch,
                    device=self.buffer_device,
                    **self._buffer_kwargs(),
                    x_fn=None,  # 'latent_params',
                    y_fn=self._buffer_y_fn(),
                    exclude_keys=CHURNED_BUFFER_EXCLUDE_KEYS,
                )
            else:
                self.prior_buffer.add(sample_batch)
            # ungated fill (no admission gate on this path) -- tracked separately
            # so it can't be mistaken for the prior model earning its churn budget
            self.prior_churn['from_seed'] += int(num_samples)

    @torch.no_grad()
    def fwd_eval_sampling(self, model, eval_discretizer, override_num_samples: Optional[int] = None,
                          dataset=None, side_effects: bool = True):
        """
        On-policy evaluation sampling.

        dataset: condition source, defaulting to mol_dataset. Pass test_mol_dataset
        to run the same protocol against held-out conditions.

        side_effects: when False this pass updates NOTHING -- no condition_log_z
        best-energy, no anchor screen/admit, no eval-timing writes. Required for the
        held-out pass: both update paths below would otherwise fold test conditions
        into training state (tracker Emin(c), anchor buffer, and transitively
        prior-buffer churn, since manage_prior_buffer consumes the same pooled
        batch), which is exactly what a held-out set must never touch.
        """
        dataset = self.mol_dataset if dataset is None else dataset
        if side_effects:
            self.times['eval_sampling_start'] = time()

        acc = defaultdict(list)
        sample_batch = None
        n_collected = 0

        if override_num_samples is not None:
            num_samples = override_num_samples
        else:
            num_samples = self.args.eval_num_samples

        while n_collected < num_samples:
            bsz = min(num_samples - n_collected, self.batch_size)
            try:
                mol_batch = next(dataset.loader(bsz, mode='graphs'))
                mol_batch = mol_batch.to(self.device)
                mol_batch.orient_molecule(mode='standard')
                init_state = get_gfn_init_state(bsz,
                                                self.energy_function.data_ndim, self.device)

                if self.args.temperature_conditioning:
                    u = torch.rand(mol_batch.num_graphs, dtype=torch.float32, device=self.device)
                    # Transform to [log_low, log_high]
                    random_log_T_tensor = (self.energy_function.log_temperature_range[0] + u *
                                           (
                                                   self.energy_function.log_temperature_range[1] -
                                                   self.energy_function.log_temperature_range[
                                                       0]))

                    random_temperatures = 10 ** random_log_T_tensor
                    temperatures = random_temperatures

                else:
                    temperatures = self.energy_function.temperature * torch.ones(mol_batch.num_graphs,
                                                                                 dtype=torch.float32,
                                                                                 device=self.device)

                out = sample_eval_fwd_trajs(init_state, model, eval_discretizer,
                                            self.energy_function, mol_batch,
                                            no_conditioning=False,
                                            temperatures=temperatures)
            except (RuntimeError, ValueError) as e:
                self.handle_train_epoch_error(e, 'eval_fwd')
                continue

            sample_batch_i = out.pop('sample_batch')
            sample_batch_i = sample_batch_i.detach().cpu()
            if sample_batch is None:
                sample_batch = sample_batch_i
            else:
                sample_batch = sample_batch.append_batch(sample_batch_i)
            n_collected += sample_batch_i.num_graphs

            for k, v in out.pop('gauss_params').items():
                acc[k].append(v)
            for k, v in out.items():
                acc[k].append(v)
            acc['temperature'].append(temperatures.cpu().detach())

        pooled = {k: torch.cat(v) for k, v in acc.items()}

        # Z estimates computed ONCE over the pooled trajectories
        log_weight = pooled['log_r'] + pooled['log_pbs'].sum(-1) - pooled['log_pfs'].sum(-1)
        pooled['log_Z'] = logmeanexp(log_weight)
        pooled['log_Z_lb'] = log_weight.mean()
        pooled['log_Z_learned'] = pooled['log_flow'][:, 0]

        # energy computed unconditionally (not just under condition_log_z) since
        # screen_and_admit_anchors below needs it regardless of whether the persistent
        # per-condition tracker is present
        temperature = 10 ** pooled['log_T_tensor']
        energy = -pooled['log_r'] * temperature

        # deliberately NO tracker update() or update_z_residual() here (both used
        # to be): this eval batch is a different measurement protocol on several
        # axes at once -- ema_model weights, a fresh log-uniform temperature
        # sweep, uniform mol draws, ~eval_num_samples pooled rows -- while the
        # ConditionLogZTracker is a rolling estimate over the *train-step*
        # stream, and every training-facing consumer of it (the terminal logw_std
        # gate, the bootstrap_log_z handoff target, z_grad/controller, persistent
        # tb_z_source) needs a single, homogeneous protocol. Mixing the eval
        # stream in spiked those statistics at every eval: the z-residual and
        # (worse) the second moment, since folding in a stream with a shifted
        # mean inflates ema_logw_sq - ema_logw^2 by the between-stream
        # (delta mean)^2 term even when both streams are individually tight --
        # and the phase gates read the tracker moments immediately after this
        # function runs, i.e. at peak contamination. This was also the root of
        # the old ema_logw "sawblade at eval cycles" that max_batch_weight only
        # bandaged (it caps trust, but can't fix a shifted value). The train
        # stream updates every step and dominates on evidence anyway; eval-batch
        # statistics remain fully available in the eval metrics themselves.
        if side_effects and hasattr(self, 'condition_log_z'):
            # best-seen energy IS protocol-independent (a running min), so it
            # still takes eval evidence -- covers eval sampling
            # itself, and (since sample_from_prior/manage_prior_buffer/
            # screen_and_admit_anchors all consume this same pooled batch) transitively
            # covers prior-buffer churn and anchor-buffer admission too, with no
            # separate hook needed at either of those call sites. Runs before
            # screen_and_admit_anchors below, so this batch's own energies are
            # already folded into Emin(c) by the time it screens against it.
            self.condition_log_z.update_best_energy(pooled['condition_id'], energy)

        # covers both evaluation()'s on-policy eval sampling and sample_from_prior's
        # prior-model sampling (itself called from manage_prior_buffer's churn and
        # grow_prior_buffer's init-time bootstrap) -- both are freshly generated,
        # freshly scored batches, so both are valid anchor candidates.
        # log_pf_est = k=1 forward-path estimate of log_pf - log_pb; reused by
        # the screen stage (TB-residual axis built inside) so it spends no
        # backward rollouts pre-filtering.
        if side_effects:
            log_pf_est = pooled['log_pf'] - pooled['log_pb']
            self.screen_and_admit_anchors(sample_batch, pooled['log_r'], energy, log_pf_est)

            self.times['eval_sampling_end'] = time()

        return pooled, sample_batch


if __name__ == '__main__':
    # Parse and fully preflight the config before asking for a GPU.  A malformed
    # launch command or invalid canonical config is a CPU-side contract failure,
    # not a reason to inspect or reserve accelerator state.
    _args = get_train_args()

    # GPU pre-flight, BEFORE Modeller() touches CUDA. Two training runs on one card
    # took this machine down with a BSOD three times on 2026-08-11/12 -- the driver
    # does not politely OOM, so there is nothing to catch afterwards and the check has
    # to happen here, first. Judges occupancy on other train.py/train_conformer.py
    # processes and on free VRAM, NOT on the ~30 desktop apps nvidia-smi reports as
    # compute processes. Override with GFN_ALLOW_GPU_SHARING=1; see gpu_guard.py.
    from gpu_guard import require_free_gpu, GPUBusy

    try:
        require_free_gpu()
    except GPUBusy as _e:
        # SystemExit, not sys.exit: `sys` is not imported in this module, and a
        # NameError here would defeat the check it is guarding.
        raise SystemExit(str(_e))

    modeller = Modeller(args=_args)
    modeller.train()
