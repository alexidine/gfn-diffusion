import gc
import math
import os
from collections import defaultdict, deque
from copy import deepcopy
from typing import Optional

from energy_sampling.eval.evaluations import to_loggable, sliced_wasserstein, adjust_fig_filesize, eval_figs, \
    log_ess_frac, condition_tracker_figs, fig_guard
from energy_sampling.eval.traj_reporting import traj_overlap_report, to_scalars

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
#     "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from time import time

import numpy as np
import torch
import wandb
from tqdm import trange

from energies.molecular_crystal import MolecularCrystal
from energy_sampling.buffer import CrystalBuffer, AnchorBuffer, ConditionLogZTracker, _per_condition_min, \
    _per_condition_max, strip_lazy_sg_caches
from energy_sampling.checkpointing import Checkpointer, MODELLER_STATE_DEFAULTS
from energy_sampling.controller import LRController
from energy_sampling.protocol import StageProtocol
from energy_sampling.eval.utils import sample_eval_fwd_trajs
from energy_sampling.utils import is_cuda_oom, \
    dict2namespace, \
    get_discretizer, log_elapsed_times, MetricTracker, quick_tb_stats, uniform_discretizer, logmeanexp, \
    cal_subtb_coef_matrix, within_condition_logw_std
from gflownet_losses import get_gfn_forward_loss, get_gfn_backward_loss, log_pf_estimate
from models import GFN
from energy_sampling.models.aunit_periodicity import sg_periodic_centroid_axes, describe
from mxtaltools.common.training_utils import flatten_wandb_params
from mxtaltools.dataset_utils.utils import collate_data_list
from utils import get_train_args, get_gfn_init_state, set_seed, \
    update_ema, get_problem_definition, problem_hash, problem_slug


# bulky per-sample analysis artifacts (fingerprints, RDFs -- huge tensors) that
# can ride in on loaded datasets or analyzed batches; never read off any buffer
# draw, so stripped from EVERY buffer's storage just in case they are present
BULKY_ATTR_EXCLUDE_KEYS = ('fingerprint', 'rdf')

# stripped from churned-buffer STORAGE at admission (draws already drop them):
# string/list attrs are never read off a buffer draw, and python-list keys make
# every subsample pay a per-element copy plus -- on GPU-resident buffers -- an
# idx.tolist() device sync. mol_dataset/prior_dataset keep them (init_identifiers
# reads .identifier); the anchor buffer keeps them (eval-cadence only).
CHURNED_BUFFER_EXCLUDE_KEYS = ('symmetry_operators', 'smiles', 'identifier') + BULKY_ATTR_EXCLUDE_KEYS


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


def _softmax_draw(scores: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    """
    Draw up to k positions into `scores`, without replacement, weighted by
    softmax(scores / temperature). Returns positions (0..scores.numel()-1),
    NOT the caller's original indices -- callers index their own index
    tensor by the result (e.g. `elig[_softmax_draw(...)]`).

    Used for replay-buffer admission/purge (see manage_replay_buffer):
    `scores` is expected to already be clipped to an absolute cap by the
    caller BEFORE this divides by temperature, so a single extreme value
    can't dominate the softmax (see the buffer-redesign discussion this
    implements -- clip-then-divide keeps the cap's meaning independent of T).
    """
    n = scores.numel()
    k = min(k, n)
    if k <= 0:
        return torch.zeros(0, dtype=torch.long)
    logits = scores.double() / max(temperature, 1e-8)
    logits = logits - logits.max()
    p = torch.softmax(logits, dim=0).cpu().numpy()
    p = np.clip(p, 1e-12, None)
    p /= p.sum()
    choice = np.random.choice(n, size=k, replace=False, p=p)
    return torch.as_tensor(choice, dtype=torch.long)


class Modeller:
    def __init__(self):
        self.step_ind = None
        self.args = get_train_args()
        torch.cuda.set_per_process_memory_fraction(self.args.cuda_memory_fraction, device=0)
        torch.cuda.init()  # create context with the cap already in place

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
        self.replay_churn = {'admitted': 0, 'evicted': 0}
        # TTL-cohort tallies (see manage_replay_buffer's eviction split):
        # drained and logged once per eval alongside replay_churn
        self.replay_cohort = {'absorbed': 0, 'expired': 0, 'expired_undrawn': 0,
                              'expired_drawn': 0, 'expired_draws_sum': 0,
                              'expired_delta_sum': 0.0, 'expired_delta_n': 0}
        # prior_buffer churn decomposed by admission SOURCE, tallied across every
        # manage_prior_buffer/top_up_prior_from_anchors/grow_prior_buffer call and
        # drained once per eval in log_buffer_stats. The point is the source mix:
        # from_anchors dominating from_prior_model means the prior has stopped
        # discovering admissible samples on its own and the buffer is living off
        # replayed archive material (see manage_prior_buffer's reach trigger)
        # 'budget' is the churn quota the prior-model draw was asked for, so the
        # admitted counts can be read as an admission RATE, not just a raw count
        self.prior_churn = {'from_prior_model': 0, 'from_anchors': 0, 'from_seed': 0,
                            'evicted': 0, 'budget': 0}
        self.device = self.args.device
        self.checkpointer = Checkpointer(self)
        self.lr_controller = LRController(self)  # fixed-peak ramp/hold/decay; tripwires always on
        self.protocol = StageProtocol(self)  # the declarative stage engine: coeffs, balance, exits, transitions
        self.init_train_constants()

        # counts TERMINAL reloads only (not ordinary loss spikes), and deliberately
        # does NOT live in MODELLER_STATE_DEFAULTS: fire_loss_spike restores modeller
        # state from the healthy checkpoint, so anything tracked there is wiped by the
        # very event this needs to remember. See _terminal_policy_state.
        self.terminal_reloads = 0

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

    # position in the protocol, derived -- checkpoints store the stage NAME; the
    # int only feeds wandb continuity and the LR controller's stage-change marker
    @property
    def phase(self):
        return self.protocol.stage.index + 1

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

    def increment_batch_size(self):
        """
        Batch growth, still AIMD-shaped (grow until OOM, cut + cooldown, regrow
        slower): multiply by batch_growth_factor once every
        batch_growth_interval steps (batch_growth_slow_interval after the first
        OOM), capped at max_batch_size. The run then visits only
        ~log_f(max/base) distinct batch sizes -- torch.compile treats every
        distinct size as a recompile + its own CUDA graph, so rare large jumps
        are what make compile viable alongside growth.

        THROUGHPUT KNEE (auto_batch_throughput_opt: true): before each jump,
        check whether the PREVIOUS jump actually paid in samples/sec (median
        step time over the trailing window x batch). Past GPU saturation a
        factor-f batch jump returns ~x1.0 throughput while step time grows xf
        -- pure steps/hour loss (rpvez6ep: batch 50k ran the same 13.3k
        samples/s as batch 1.6k at 31x the step time). A jump that gains less
        than batch_growth_min_gain reverts one rung and PINS the batch for the
        current protocol stage (stages have different step-cost profiles, so
        protocol.advance clears the pin, the rung baseline, and the step-time
        window at every stage transition -- a baseline measured under the
        outgoing stage's step cost would poison the incoming stage's first
        knee comparison -- and the walk re-measures upward from the current
        rung on in-stage timings). With the flag on, max_batch_size is a
        safety ceiling rather than a target. NB nvidia-smi-style utilization
        can't drive this: it saturates at 100% below the knee; marginal
        throughput is the discriminating signal on both sides.
        """
        knee_on = bool(getattr(self.args, 'auto_batch_throughput_opt', False))
        stage_name = getattr(getattr(self.protocol, 'stage', None), 'name', None)
        if (knee_on and stage_name is not None
                and getattr(self, 'batch_size_saturated_stage', None) == stage_name):
            # periodic re-estimation: the knee moves WITHIN a stage as the fused
            # composition drifts (branch fracs shift, fwd activates/deactivates,
            # buffers grow), so a pin decays -- drop one rung and re-climb,
            # which adapts in BOTH directions: a healthy re-climb re-pins at or
            # above the old knee, a failed one pins lower
            recheck = int(getattr(self.args, 'batch_knee_recheck_steps', 0) or 0)
            if recheck > 0 and self.step_ind - getattr(self, 'batch_size_pinned_at', 0) >= recheck:
                f = float(getattr(self.args, 'batch_growth_factor', 2.0))
                self.batch_size = max(1, int(round(self.batch_size / f)))
                self.batch_size_saturated_stage = None
                self._rung_throughput = None
                self.batch_size_last_grow = self.step_ind
                print(f"batch growth: knee recheck -- dropping to {self.batch_size} and re-measuring")
            return  # throughput knee already found for this stage's step profile
        if self.batch_size >= self.args.max_batch_size:
            return
        if self.step_ind < self.batch_size_cooldown_until:
            return  # recently cut -- hold flat until the new level proves stable

        interval = int(getattr(self.args, 'batch_growth_interval', 0) or 0)
        slow = int(getattr(self.args, 'batch_growth_slow_interval', 0) or 0) or interval
        wait = slow if self.batch_size_ever_oomed else interval
        if self.step_ind - getattr(self, 'batch_size_last_grow', 0) < wait:
            return

        min_gain = float(getattr(self.args, 'batch_growth_min_gain', 0.15) or 0)
        times = getattr(self, '_recent_step_times', None)
        if knee_on and min_gain > 0 and times is not None and len(times) >= 10:
            # median over the trailing window: robust to the one-off compile
            # stall at rung entry and any churn/OS spikes inside the dwell
            med = float(np.median(list(times)[-20:]))
            sps = self.batch_size / max(med, 1e-9)
            base = getattr(self, '_rung_throughput', None)
            if base is not None and base[0] < self.batch_size:
                prev_batch, prev_sps = base
                if sps < prev_sps * (1.0 + min_gain):
                    # pin at the TRUE knee; gradient stability below
                    # fused_grad_accum_min_samples is provided by fused-step
                    # accumulation (see train_step), not batch inflation
                    accum = int(getattr(self.args, 'fused_grad_accum_min_samples', 0) or 0)
                    print(f"batch growth: throughput knee -- {prev_batch}->{self.batch_size} bought "
                          f"{prev_sps:.0f}->{sps:.0f} samples/s (< +{min_gain * 100:.0f}%); pinning "
                          f"{prev_batch} for stage '{stage_name}'"
                          + (f" (fused steps will grad-accumulate to {accum})" if prev_batch < accum else ""))
                    self.batch_size = prev_batch
                    self.batch_size_saturated_stage = stage_name
                    self.batch_size_pinned_at = self.step_ind
                    self._rung_throughput = None
                    self.batch_size_last_grow = self.step_ind
                    return
            self._rung_throughput = (self.batch_size, sps)

        f = float(getattr(self.args, 'batch_growth_factor', 2.0))
        self.batch_size = min(self.args.max_batch_size,
                              max(self.batch_size + 1, int(round(self.batch_size * f))))
        self.batch_size_last_grow = self.step_ind


    def step_lr_schedule(self):
        # the LRController owns the LRs unconditionally (v6): fixed-peak
        # ramp/hold/decay -- adaptive_lr.enabled toggles the schedule (flat
        # scale=1.0 when off; see LRController.step). There is no separate
        # legacy scheduler path left to fall back to.
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
        if hasattr(self, 'last_grad_norm_pre_clip'):
            metrics['grad_norm_pre_clip'] = self.last_grad_norm_pre_clip
        metrics['Fwd Frac'] = self.fwd_frac
        metrics['Bwd Frac'] = self.bwd_frac
        metrics['Replay Frac'] = self.replay_frac
        # boost state, per-rule live (annealed) thresholds/elevations, exit streaks
        metrics.update(self.protocol.report())
        metrics.update(log_elapsed_times(self.times))
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

        # always logged now -- report() is meaningful whether or not
        # adaptive_lr.enabled (scale sits flat at 1.0 when disabled, which is
        # itself worth seeing on the dashboard rather than silently absent)
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

        if any([self.args.fwd_loss_coeffs.subtb > 0, self.args.bwd_loss_coeffs.subtb > 0,
                self.args.replay_loss_coeffs.subtb > 0]):
            self.args.fwd_loss_coeffs.coeff_matrix = cal_subtb_coef_matrix(  # todo delete this re-instantiation
                self.args.fwd_loss_coeffs.subtb_lambda, self.args.integrator.T).to(self.gfn_model.device)
            self.args.bwd_loss_coeffs.coeff_matrix = cal_subtb_coef_matrix(
                self.args.bwd_loss_coeffs.subtb_lambda, self.args.integrator.T).to(self.gfn_model.device)
            self.args.replay_loss_coeffs.coeff_matrix = cal_subtb_coef_matrix(
                self.args.replay_loss_coeffs.subtb_lambda, self.args.integrator.T).to(self.gfn_model.device)

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
        # if we do pre-embedded molecule conditions, we'll add the dimension here
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
        }
        energy_config.update(self.args.energy_config.__dict__)
        self.energy_function = MolecularCrystal(**energy_config)

    def tb_z_source(self, group: str) -> str:
        """
        Safe accessor for condition_log_z.{fwd,bwd,replay}_tb_z_source --
        falls back to 'learned' (today's exact behavior) if the
        condition_log_z config section, or this field within it, is absent
        (e.g. an older/other config that predates this feature).
        """
        cfg = getattr(self.args, 'condition_log_z', None)
        if cfg is None:
            return 'learned'
        return getattr(cfg, f'{group}_tb_z_source', 'learned')

    def mode_repeats(self, mode: str) -> int:
        """
        Per-mode K-tiling factor, read from {mode}_loss_coeffs.repeats so it
        can be phase-scheduled like any other coefficient (e.g. bwd repeats
        > 1 only during phase-1 MLE/TBC pre-training, where K same-terminal
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
        half_life_visits = getattr(cfg, 'half_life_visits', 7.0) if cfg is not None else 7.0
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
        Fast, rollout-free supervised fit of the flow head's Z(c) prediction
        onto condition_log_z's ema_logw:

            min_theta  sum_c trust_c * MSE(Z(c; theta), ema_logw(c))
                                                        over conditions c with
                                                        a trustworthy estimate

        trust_c = eff_c / (eff_c + min_visits) is an evidence-confidence
        weight built from the tracker's effective_count -- time-decayed AND
        ESS-capped, so a condition whose lifetime count passed the
        min_visits gate but whose evidence is stale or came from degenerate
        (ESS ~ 1) batches still reads as low-confidence. Low-trust targets
        pull weakly and the network resolves them by interpolating from
        trusted neighbors (the right prior for a badly-sampled condition);
        a genuine discontinuity backed by strong evidence still carries the
        weight to be learned. Weights are normalized to mean 1 over the
        trustworthy set so target_rms/best-loss bookkeeping keeps its
        unweighted calibration, and the coverage acceptance gate uses a
        trust-weighted quantile for consistency -- a condition the fit was
        told to down-weight must not fail the gate for missing that same
        target.

        Intended to run once, at the phase 1 -> 2 transition (see
        phase1to2), on whatever ema_logw the tracker accumulated during
        phase 1's MLE warm-start -- purely to get the flow head starting
        from a reasonable baseline instead of its random init, before
        ordinary (rollout-driven) training resumes. Deliberately regresses
        onto ema_logw, never ema_log_z_emp: this runs in exactly the
        cold-start regime (right after phase 1, before the policy has had
        time to build real overlap with the reward-tilted target) that we
        directly observed can send ema_log_z_emp to billions of nats -- see
        ConditionLogZTracker.lookup's docstring. ema_logw is rank-trimmed
        and evidence-capped, so it can't hand this a corrupted target the
        same way.

        No trajectory sampling, no reward evaluation -- each step draws a
        batch of conditions the same way fwd_train_step does, but only
        needs gfn_model.get_condition_embedding() + flow_model(), skipping
        get_traj_fwd entirely, so this converges in seconds rather than
        needing real rollouts. Trains only flow_model, via a fresh, local
        Adam instance -- deliberately NOT self.optimizers['flow'] (that one's
        for ordinary training's per-step Z updates and persists across the
        whole run; reusing it would mean this "from-scratch" fit starts with
        whatever momentum/variance state it happened to have accumulated).

        This Z(c) head is load-bearing: phase-2 training can't proceed
        usefully until it predicts every trustworthy condition well, not just
        on average. So the fit is structured as an outer restart loop around
        a deliberately simple single run, gated on a real per-condition
        acceptance test rather than a batch-mean:

          single run  -- LR is one linear decay from lr to lr/100 over
            lr_ramp_steps, held at lr/100 after (large early steps to close
            the bulk of the gap, small fixed steps to settle without
            overshoot; no reactive cuts / plateau detection that can fight
            itself). target_rms is the in-run early stop: stop once the
            monitored RMS(MSE) drops to it (default 0.5 ~ half a nat). Two
            deterministic safety nets, neither reactive: best_state snapshots
            the trough weights on every improvement and is restored at the
            end (so a final step that rang up off the trough -- or one that
            corrupted the weights, since a good snapshot can't be retroed --
            is discarded), and a non-finite loss (a genuine NaN/inf
            explosion, not a mere increase) breaks straight to that restore.

          holdout     -- with >= min_conditions_for_holdout trustworthy
            conditions, holdout_frac are excluded from the gradient (still
            drawn, never backpropped) and the in-run monitor tracks that
            held-out MSE, so early stopping reflects generalization, not
            memorization. Below that, the split isn't meaningful and this
            falls back to training loss.

          acceptance  -- after each run, coverage_eval sweeps n_eval_batches
            under the deployment draw and records per-condition abs error,
            then checks the coverage_quantile (default 0.99) of those errors
            against coverage_tol (default 2.0 nats): "~every trustworthy
            condition is within 2 nats". This is the gate that actually
            matters; the batch-mean target_rms above only decides when a
            single run has stopped improving.

          restart     -- if a run's coverage fails, re-seed, re-initialize
            the flow head, and retry at a lower starting LR (lr *
            lr_restart_factor ** attempt), up to max_attempts. Cheaper and
            more robust than making one run bulletproof: a bad basin or an
            unlucky overshoot just gets a fresh draw. The best-by-coverage
            attempt is kept regardless of whether any run passed.

        train_conditioner: also fit conditions_embedding_model, not just the
        flow head. Declared per-protocol via the 'bootstrap_z:train_conditioner'
        action -- only correct when the preceding prior stage scrambled its
        conditions (scramble_conditions), so nothing ever trained the
        conditioner (bwd MLE detached it; fwd Z-only TB freezes the policy) and
        its random-init features may not separate conditions well enough for a
        frozen-embedding regression -- this fit is then the first thing that
        ever trains it. Safe there precisely BECAUSE of that scrambled prior:
        the trunk was trained to ignore the embedding, so reshaping the
        conditioner can't move the policy. After a stage that DID train the
        conditioner (e.g. condition-grouped VarGrad), leave the action plain
        ('bootstrap_z') -- that structure is policy-relevant and a Z regression
        has no business rewriting it. Each restart attempt resets the
        conditioner to its at-entry weights (the analog of _reinit_flow), and
        snapshots/ema-sync cover both modules.
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
        # a marginal fit is still better to hand phase 2 than a fresh random head
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
            print("  WARNING: Z(c) bootstrap did not reach the coverage bar; phase-2 "
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

    def _build_gfn_config(self):
        return dict(
            dim=self.energy_function.data_ndim,
            conditions_dim=self.get_conditioning_dim(),
            conditions_type='molecule' if self.args.molecule_conditioning else 'vector',
            periodic_centroid_axes=self._resolve_periodic_centroid_axes(),
            conditional=any([
                self.args.temperature_conditioning,
                self.args.molecule_conditioning,
                self.args.sg_conditioning,
                self.args.zp_conditioning,
                getattr(self.args, 'vector_conditioning', False),
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
        # the turn-taking policy optimizers are identical up to their LR
        for mode in ('fwd', 'bwd', 'replay'):
            self.optimizers[mode] = torch.optim.Adam(get_policy_params(self.gfn_model), init_policy_lrs[mode],
                                                     weight_decay=weight_decay)
        # fused fires fwd/bwd/replay in one backward() (phase 3), so -- unlike the
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
            init_policy_lrs['fused'], weight_decay=weight_decay)
        flow_params = self.gfn_model.flow_model.parameters()
        self.optimizers['flow'] = torch.optim.Adam(flow_params, init_flow_lr, weight_decay=weight_decay)

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
            energy, prior = self.energy_function.batched_analyze_crystal_batch(
                prior.latent_params(),
                prior,
                self.args.energy_config.temperature * torch.ones((prior.num_graphs), dtype=torch.float32,
                                                                 device=self.device),
                return_batch=True,
                internal_oom_recovery=True,
                # one-off pass over the whole prior dataset at init -- prefer the adaptive, self-healing chunked path over a hard crash, regardless of the training-time flag
            )

        self.prior_dataset = CrystalBuffer(prior,
                                           device=self.buffer_device,
                                           max_z_prime=max(self.args.z_primes),
                                           x_fn=None,  # 'latent_params',
                                           y_fn=self.args.energy_function,
                                           exclude_keys=BULKY_ATTR_EXCLUDE_KEYS,
                                           )

        prior_path = None
        if self.args.prior_model_name is not None:
            prior_path = f'{self.args.checkpoints_dir}/{self.args.prior_model_name}'
        elif getattr(self.args, 'reuse_prior', False):
            # this run identity's own earlier phase-1 product, if already on
            # disk. find_matching validates the stored problem_def against the
            # current config (full match, stricter than the exempted check
            # below), so missing/mismatched resolves to None and the warm-start
            # stage simply runs -- and re-saves the prior for the next rerun.
            # On a RESUMED run past phase 1 this also restores prior_model,
            # which snapshot_prior only ever set live at the transition.
            prior_path = self.checkpointer.find_matching('prior')
            if prior_path is None:
                # no prior under this run's own identity -- fall back to any
                # other run's *_prior.pt with an exactly matching problem_def,
                # so a battery of runs shares one pretrained prior without
                # naming it (problem_def has no architecture/T/lr, only the
                # target identity)
                prior_path = self.checkpointer.find_shared_prior()
            if prior_path is not None:
                print(f"reuse_prior: reloading existing prior checkpoint {prior_path} "
                      f"as the frozen prior model")
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
                                         max_z_prime=max(self.args.z_primes),
                                         exclude_keys=BULKY_ATTR_EXCLUDE_KEYS)

        if self.args.test_molecules_path is not None:
            self.test_mol_dataset = CrystalBuffer(self._load_condition_file(self.args.test_molecules_path),
                                                  device=self.buffer_device,
                                                  max_z_prime=max(self.args.z_primes),
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

            # Reward init
            self.init_energy_function()

            # Model Init
            self.init_gfn()

            # data init -- init_identifiers() needs mol_dataset/prior_dataset/test_mol_dataset
            # all loaded first (it builds one registry spanning all of them), and must itself
            # run before init_condition_log_z() (which preallocates the tracker table off
            # energy_function.condition_library_size, set by init_identifiers())
            self.init_mol_dataset()
            self.init_prior_dataset()
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
            if hasattr(self, 'prior_model'):
                self.grow_prior_buffer()
            self.init_condition_log_z()
            self.init_anchor_buffer_seed()

            # pin the starting stage on a fresh run and walk any skip_if chain
            # (e.g. a prior loaded by path skips the MLE warm-start stage);
            # resumed runs stay wherever their checkpoint says
            self.protocol.begin()

            self.times['initialization_end'] = time()

            wandb.watch(self.gfn_model,
                        log_graph=False,
                        log_freq=1000,
                        log='gradients')

            self.gfn_model.train()
            self.set_detect_anomaly(do_anomaly_detection=self.args.anomaly_detection)
            init_step = self.step_ind * 1
            for self.step_ind in trange(init_step, self.args.epochs + 1):
                current_loss = None
                metrics = {}
                if self.step_ind % 10 == 0:
                    self.set_loss_coeffs()
                    self.set_energy_coeffs()

                step_type = self.train_logic(self.step_ind)
                self.times['train_step_start'] = time()
                try:
                    current_loss = self.train_step(step_type)

                    if self.args.grow_batch_size:
                        self.increment_batch_size()

                except (RuntimeError, ValueError) as e:  # if we do hit OOM, slash the batch size
                    self.handle_train_epoch_error(e, step_type)
                self.times['train_step_end'] = time()
                step_dt = self.times['train_step_end'] - self.times['train_step_start']
                self._probe_max('step_time_max10', step_dt)
                if not hasattr(self, '_recent_step_times'):
                    self._recent_step_times = deque(maxlen=64)
                self._recent_step_times.append(step_dt)  # feeds the throughput-knee check

                # train monitoring
                if self.step_ind % 10 == 0:
                    lr = self.step_lr_schedule()
                    metrics.update(self.ten_step_reporting())
                    self.monitor_losses(current_loss, step_type)
                    # gate publishers feed the exit triggers (gates/*); the
                    # protocol tick then runs the stage's balance nudge and
                    # arms the exit trigger (which pulls the next eval forward)
                    if self.protocol.flag('mle_gate'):
                        metrics.update(self.update_mle_gate())
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
                    wandb.log(metrics, step=self.step_ind, commit=True)

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

            self.checkpointer.save('final', with_buffers=True)
            print("Finished Training!")

    def _terminal_policy_state(self):
        """
        Standalone terminal-failure detector: the policy is emitting numerically
        absurd samples and is not coming back on its own. Returns a reason string
        (truthy) or None.

        Distinct from the LRController tripwires, and deliberately so: those
        read the per-step training signals (branch loss, pre-clip grad norm),
        which a policy can keep numerically tame while emitting garbage
        states. These are ABSOLUTE bounds on the sampled-state statistics
        themselves. Not "worse than lately" but "no longer physics".

        Two channels OR'd, because the two observed deaths do not look alike:
          djr13t0j  detonation -- logw_std 8.6 -> 1.8e5 inside 100 steps; box
                    violation 0.0012 -> ~1100 inside ONE eval interval.
          1219ddv9  slow creep -- logw_std only ever reached ~994 (under the bound),
                    but box_violation climbed to 19.8 over ~3000 steps.
        Neither channel catches both; together they catch both. Observed healthy
        ranges are logw_std 8-46 and box_violation 0.001-0.004, so each bound sits
        >20x above anything legitimate and >100x below the observed death, which is
        as wide a margin as this failure offers. NOT an early warning -- the policy
        variance channels move only 0.85 nats at the kill vs 1.2 nats on a benign
        excursion, so nothing here leads; it is a terminal-state detector, and the
        actuator is a rewind, not a nudge.

        Reads 'fwd' only: bwd/replay share the same policy network, so a genuine
        policy blowup shows up here regardless of which branch is stepping.
        """
        t = self.metric_tracker
        bounds = (('logw_std', getattr(self.args, 'terminal_logw_std', 1000.0)),
                  ('box_violation', getattr(self.args, 'terminal_box_violation', 1.0)))
        for name, bound in bounds:
            v = t.get('fwd', name)
            if v is None or bound is None:
                continue
            if not math.isfinite(v) or v > bound:
                return f"fwd/{name}={v:.4g} (bound {bound:g})"
        return None

    def monitor_losses(self, current_loss, step_type):
        if current_loss is not None:
            # check_spike (LRController) runs two ABSOLUTE tripwire tiers per
            # branch loss and pre-clip grad norm (no medians, no relative
            # bars): 'cut' = parameter thrash, the LR is cut in place and
            # training continues on the live weights; 'reset' = true
            # explosion (or non-finite), rewind to best + cut. Always checked
            # regardless of adaptive_lr.enabled -- this is the FIRE half, and
            # it was never gated on that flag even in the old design.
            trig = self.lr_controller.check_spike(
                step_type, current_loss, getattr(self, 'last_grad_norm_pre_clip', None))

            terminal = self._terminal_policy_state()
            if terminal is not None:
                print(f"TERMINAL policy state at step {self.step_ind}: {terminal} "
                      f"-- rewinding to best and ratcheting LR down")
                self.fire_loss_spike(terminal=True)
            elif trig == 'reset':
                self.fire_loss_spike()
            elif trig == 'cut':
                print("Firing LR cut (thrash tier -- no rewind)")
                self.lr_controller.on_explosion()

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

    def fire_loss_spike(self, terminal: bool = False):
        """
        Rewind to the best checkpoint and cut LR.

        terminal=True marks a _terminal_policy_state rewind rather than a
        reset-tier tripwire fire, and ratchets the LR cut by the number of
        terminal rewinds so far, so repeated policy deaths cut to
        cut_ratio**n and the ceiling actually descends (the djr13t0j
        sawtooth: rewind restores checkpointed LR state, and without a
        rewind-proof memory the run walks back into the same detonation --
        the cut factor itself is instance-held on the controller for the
        same reason).

        A one-way ratchet is the right shape HERE, unlike the threshold anneal it
        superficially resembles: it is driven by an unambiguous catastrophic event,
        not by a marginal metric, so it cannot creep its way into a permanent
        breach -- it only moves when the policy has already died.
        """
        if terminal:
            self.terminal_reloads += 1
        print("Firing LR spike & recovery"
              + (f" (TERMINAL rewind #{self.terminal_reloads})" if terminal else ""))
        running_checkpoint_path = self._rewind_checkpoint_path()
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

        # set_state_dict above restored lr_ctrl (schedule clock) from the
        # healthy best checkpoint; the cut factor lives on the controller
        # INSTANCE and survives the rewind. terminal_reloads compounds the
        # cut across policy deaths; a reset-tier tripwire fire keeps its
        # single flat cut -- a recoverable event must not inherit a
        # punishment meant for a death.
        self.lr_controller.on_explosion(count=self.terminal_reloads if terminal else 1)

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
        # only engages BELOW the target (e.g. a knee-pinned batch under the
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

        if accumulating:
            self.fused_accum_count += self.batch_size
            do_step = self.fused_accum_count >= accum_target
            self.step_loss(step_type, loss * (self.batch_size / accum_target), do_step=do_step)
            if do_step:
                self.fused_accum_count = 0
        else:
            self.step_loss(step_type, loss)

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
        # (mode_boostable, derived from the rule list -- the old 'phase 3 only'
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
        # clean per-direction convergence signal: quick_tb_stats' 'logw_std' is
        # BATCH-WIDE, dominated by between-condition log Z(c) spread at scale and
        # a misleading VarGrad convergence signal. logw_std_within removes that,
        # measuring only the within-condition spread VarGrad reduces. Omitted
        # (not logged) on batches with no multi-member condition group -- nan --
        # so metric_tracker never sees a spurious value.
        cid = loss_dict.get('condition_id')
        if cid is not None:
            within = within_condition_logw_std(
                loss_dict['log_pf'], loss_dict['log_pb'], loss_dict['log_r'], cid)
            if math.isfinite(within):
                stats['logw_std_within'] = within
            # NB the conditional CALIBRATION metrics (cond_tb_err / tb_err_worst /
            # z_grad_worst) come straight out of quick_tb_stats above, which does
            # its own condition grouping -- they need no branch here because they
            # are defined on unconditional batches too (one group). Only
            # logw_std_within is conditional-only: a within-group spread is
            # undefined without groups.
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
        # RAW (pre-EMA) stats cache, one step deep: the MLE slope gate samples
        # bwd/mle from here every 10 steps -- raw batch losses are ~independent
        # across steps, so its OLS slope needs no autocorrelation correction
        self._last_stats[sub_type] = stats
        self.metric_tracker.update(sub_type, stats, self.step_ind)

    def step_loss(self, step_type, loss, do_step: bool = True):
        loss.backward()
        if not do_step:
            return  # mid-accumulation: keep piling up gradients, don't clip/step yet

        pre_clip = torch.nn.utils.clip_grad_norm_(
            self.gfn_model.parameters(), self.args.gradient_norm_clip).item()
        if not math.isfinite(pre_clip):
            print(f"Non-finite gradient at {self.step_ind}")
            return  # skip non-finite
        # raw (pre-clip) global grad norm, for reading how hard the clip binds:
        # persistently >> gradient_norm_clip means every step is rescaled and
        # Adam is effectively running on normalized gradients
        self.last_grad_norm_pre_clip = pre_clip

        self.optimizers[step_type].step()
        # Non-fused steps: step the standalone flow optimizer here (fwd/bwd/replay run
        # separately, so whichever one had freeze_z=False unambiguously trained Z).
        # Fused steps: skip it -- flow is a param group of optimizers['fused'], so the
        # .step() above already updated Z, using only the grad from whichever sub-loss
        # trained it (the fwd branch); if none did, those params had grad=None and Adam
        # skipped them, so there's no spurious update to guard against.
        if 'flow' in self.optimizers and step_type != 'fused':
            self.optimizers['flow'].step()

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
        mol_batch, log_T_tensor, sg_inds, zps, condition, condition_id = self.energy_function.condition_samples(
            mol_batch, repeats=repeats)

        return get_gfn_forward_loss(self.args.fwd_loss_coeffs,
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
        every phase, not just phase 2. Rationale (same as the re-centered
        bwd under_coverage metric, see _bwd_under_center): whenever the
        learned Z lags the buffer-implied level -- the standing condition of
        phase 3, not just phase 2's untrained-Z init -- |log_Z - log_w| is
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
                          report_losses: bool = False):

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
                                                step=self.step_ind)

        self.replay_buffer.update_losses(loss_dict['resid'].abs(), inds)

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

            latents = mol_batch.latent_params()
            latents = latents.to(self.device)

        elif self.bwd_sampling_mode == 'prior':
            # condition-blocked draws (C conditions x up to M distinct terminals
            # each) only while condition-grouped bwd VarGrad is active (phase 2:
            # vg_lb = phase2_bwd_vg_lb; _activate_phase3_losses turns it off) --
            # its cross-terminal signal otherwise only arrives via birthday
            # collisions. Phase 3's per-sample TB prefers the broad-coverage
            # independent draws, which block_m = 0 restores automatically.
            block_m = getattr(self.args.buffers.prior_buffer, 'condition_block_m', 0) \
                if getattr(self.args.bwd_loss_coeffs, 'vg_lb', 0) > 0 else 0
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

            latents = mol_batch.latent_params()
            latents = latents.to(self.device)
        else:
            assert False, f"sampling method {self.args.sampling} not implemented"
        mol_batch = mol_batch.to(self.device)
        mol_batch, log_T_tensor, sg_inds, zps, condition, condition_id = self.energy_function.condition_samples(
            mol_batch, repeats=repeats)
        temperature = 10 ** log_T_tensor
        log_reward = self.energy_function.prebuilt_sample_to_reward(mol_batch,
                                                                    temperature)  # relies on the energy terms being attached to the graphs!

        return condition, condition_id, inds, latents, log_reward, mol_batch, traj

    @torch.no_grad()
    def draw_replay_sample(self, repeats):
        mol_batch, traj, inds = next(
            self.replay_buffer.loader(
                batch_size=self.batch_size, mode='graphs',
                repeats=repeats, return_inds=True,
                weighted=False,
                temperature=0.1, beta=1.0,
                return_traj=True))

        latents = mol_batch.latent_params()
        latents = latents.to(self.device)
        traj = traj.to(self.device)

        mol_batch = mol_batch.to(self.device)
        mol_batch, log_T_tensor, sg_inds, zps, condition, condition_id = self.energy_function.condition_samples(
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
        batch_size multiplicatively, and start a cooldown (see increment_batch_size).
        """
        print(f"Caught error during '{step_type}' step: {str(e)}")
        if not is_cuda_oom(e):
            raise e  # will simply raise error if other or if training on CPU

        print("OOMED!")
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

        self.batch_size = max(1, int(self.batch_size * self.args.oom_batch_shrink_factor))
        self.batch_size_ever_oomed = True
        self.batch_size_cooldown_until = self.step_ind + self.args.oom_cooldown_steps
        # stale throughput baseline/latch would compare across the cut -- re-measure
        self._rung_throughput = None
        self.batch_size_saturated_stage = None
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
                terminal_state = mol_batch.latent_params()

                mol_batch, log_T_tensor, sg_inds, zps, condition, condition_id = self.energy_function.condition_samples(
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
            acc['packing_coeff'].append(cpu(mol_batch.packing_coeff))
            acc['condition_id'].append(cpu(condition_id))

        pooled = {k: torch.cat(v, dim=0) for k, v in acc.items()}
        if not self.gfn_model.conditional:
            pooled['log_Z_learned'] = torch.mean(pooled['log_Z_learned'])

        # deliberately NO tracker update() here (there used to be a phase-gated
        # one): same single-protocol principle as fwd_eval_sampling -- eval-time
        # backward sampling (ema_model, fixed eval temperature) is a different
        # measurement protocol from the train-step bwd stream that feeds the
        # tracker, and mixing protocols inflates the tracker's second moment by
        # the between-stream mean shift, spiking the phase-2 logw_std gate (and
        # the ema_logw sawblade) at every eval. The train-time bwd/replay
        # update() calls (phase-gated, in get_gfn_backward_loss) are untouched.

        return pooled

    def _eval_conditional_stats(self, stats):
        """quick_tb_stats plus the condition-aware metrics (_update_rolling's set)
        for a pooled EVAL batch. log_metrics runs quick_tb_stats on the eval streams
        without a condition axis, which omits logw_std_within -- exactly the kind of
        metric a conditional generalization check turns on -- so it is computed here
        for train and held-out alike, off the same code path, to keep the comparison
        like-for-like."""
        log_pf = stats['log_pfs'].sum(-1)
        log_pb = stats['log_pbs'].sum(-1)
        log_z = stats['log_Z_learned']
        log_r = stats['log_r']
        cid = stats.get('condition_id')
        out = quick_tb_stats(log_pf, log_pb, log_z, log_r,
                             clip_beta=getattr(self.args.fwd_loss_coeffs, 'beta', None),
                             condition_id=cid,
                             worst_quantile=self.args.conditional_worst_quantile,
                             **self._reward_ramp_kwargs(cid))
        if cid is not None:
            within = within_condition_logw_std(log_pf, log_pb, log_r, cid)
            if math.isfinite(within):
                out['logw_std_within'] = within
        return out

    @torch.no_grad()
    def log_test_metrics(self, eval_discretizer, fwd_stats):
        """
        Conditional generalization check: the same on-policy eval protocol run
        against HELD-OUT conditions (test_molecules_path), logged under
        'eval_test/', with 'eval_gap/' = train - test on the headline metrics.

        Pure measurement. fwd_eval_sampling runs with side_effects=False, so the
        held-out conditions never reach condition_log_z, the anchor buffer, or
        prior-buffer churn, and nothing here feeds a gate, controller or loss.
        """
        n = getattr(self.args, 'test_eval_num_samples', None) or self.args.eval_num_samples
        test_stats, _ = self.fwd_eval_sampling(self.ema_model, eval_discretizer,
                                               override_num_samples=int(n),
                                               dataset=self.test_mol_dataset,
                                               side_effects=False)
        train_m = self._eval_conditional_stats(fwd_stats)
        test_m = self._eval_conditional_stats(test_stats)

        metrics = {f'eval_test/{k}': v for k, v in test_m.items()}
        for k in ('logw_std_within', 'cond_tb_err', 'tb_err_worst', 'z_grad_worst'):
            if k in train_m:
                metrics[f'eval_fwd/{k}'] = train_m[k]
        for k in ('cond_tb_err', 'tb_err_worst', 'z_grad_worst', 'scatter_err',
                  'logw_std_within', 'relative_under'):
            if k in train_m and k in test_m:
                metrics[f'eval_gap/{k}'] = train_m[k] - test_m[k]
        return metrics

    def log_metrics(self, fwd_stats, bwd_stats, sample_batch):

        metrics = {}
        arr = lambda t: t.cpu().detach().numpy()
        val = lambda t: t.cpu().detach().item()

        """Forward TB Stats"""
        log_r = fwd_stats['log_r']
        log_pf = fwd_stats['log_pfs'].sum(-1)
        log_pb = fwd_stats['log_pbs'].sum(-1)
        log_Z_learned = fwd_stats['log_Z_learned']
        log_T_tensor = fwd_stats['log_T_tensor']
        metrics.update({f'eval_fwd/{k}': v for k, v in
                        quick_tb_stats(log_pf, log_pb, log_Z_learned, log_r,
                                       clip_beta=getattr(self.args.fwd_loss_coeffs, 'beta', None),
                                       condition_id=fwd_stats.get('condition_id'),
                                       **self._reward_ramp_kwargs(fwd_stats.get('condition_id'))).items()})

        self.log_thermo_properties(arr, fwd_stats, log_T_tensor, log_Z_learned, log_r, metrics, sample_batch, val)

        """Backward TB Stats"""
        log_pf = bwd_stats['log_pfs'].sum(-1)
        log_pb = bwd_stats['log_pbs'].sum(-1)
        log_z = bwd_stats['log_Z_learned']
        log_r = bwd_stats['log_r']
        # parity / Z diagnostics (shared with fwd)
        metrics.update({f'eval_bwd/{k}': v for k, v in
                        quick_tb_stats(log_pf, log_pb, log_z, log_r,
                                       clip_beta=getattr(self.args.bwd_loss_coeffs, 'beta', None),
                                       condition_id=bwd_stats.get('condition_id'),
                                       **self._reward_ramp_kwargs(bwd_stats.get('condition_id'))).items()})

        def dump_numeric(metrics, prefix, obj):
            d = obj if isinstance(obj, dict) else vars(obj)
            for k, v in d.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metrics[f'{prefix}/{k}'] = v

        dump_numeric(metrics, 'energy_func/', self.energy_function)
        dump_numeric(metrics, 'loss_coeffs/fwd_', self.args.fwd_loss_coeffs)
        dump_numeric(metrics, 'loss_coeffs/bwd_', self.args.bwd_loss_coeffs)
        dump_numeric(metrics, 'loss_coeffs/replay_', self.args.replay_loss_coeffs)

        self.log_dist_stats(log_pf, metrics, sample_batch)

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
        std_params = sample_batch.latent_params()
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
        # energies
        for key in sample_batch.keys():
            if 'energy' in key or 'pot' in key:
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
        metrics['Mean Packing Coeff'] = val(sample_batch.packing_coeff.mean())
        metrics['Packing Coeff'] = arr(sample_batch.packing_coeff.clip(max=2))
        metrics['Reduction Energy'] = arr((1e-3 + sample_batch.reduction_en).log10())
        metrics['Reduced Valid Fraction'] = np.mean(arr(sample_batch.reduction_en) < 1e-1)
        # conditions
        metrics['Crystal Mean Log Temperature'] = val(log_T_tensor.mean())
        metrics['Crystal Log Temperature'] = arr(log_T_tensor)
        # training metrics
        metrics['Mean Sample Energy'] = val(sample_batch.gfn_energy.mean())
        metrics['Sample Energy'] = arr(sample_batch.gfn_energy.clip(max=50))
        metrics['Mean Sample Reward'] = val(log_r.mean())
        metrics['Sample Reward'] = arr(log_r.clip(min=-50))
        metrics['Empirical log Z'] = val(fwd_stats['log_Z'])
        metrics['Empirical log Z LB'] = val(fwd_stats['log_Z_lb'])
        metrics['log Z learned'] = val(log_Z_learned.mean())

        # get fraction of samples which are 'reasonable' at this energy,
        # prefer the rescaled mol_energy (matches the actual loss scale) over
        # the bare energy_function attribute, which is only correct for toy
        # (non lj-rescaled) energy functions that never set mol_energy
        en_func = self.energy_function.energy_function
        scaled_mol_energy = getattr(sample_batch, 'mol_energy', None)
        if scaled_mol_energy is None:
            scaled_mol_energy = sample_batch[en_func]
        sample_is_good = (scaled_mol_energy < 0) * (sample_batch.packing_coeff > 0.55) * (
                sample_batch.packing_coeff < 0.95)
        metrics["Reasonable Sample Fraction"] = sample_is_good.float().mean().item()

    def update_mle_gate(self):
        """
        MLE flatness gate, publishing gates/mle_flat for the warm-start
        stage's exit trigger. Runs at train cadence (every 10 steps) on the
        stages that declare the mle_gate flag.

        Samples the RAW per-step bwd MLE batch loss (self._last_stats, the
        pre-EMA value _update_rolling just computed) every 10 steps into a
        window of mle_slope_window train steps, then least-squares fits the
        slope. Raw batch losses are ~independent across steps -- unlike the
        old 100-step-time-constant EMA input, whose ~0.9 autocorrelation
        needed an AR(1) effective-sample-size correction and forced the
        window out to 1000 steps just to hold ~5 independent samples. On raw
        input, plain OLS standard errors apply and the same confidence fits
        in a ~3x shorter window.

        'Flat' = an EQUIVALENCE test on the descent rate: the upper
        mle_slope_t-sigma bound on the rate (nats per 100 train steps -- the
        RATE needs no per-system normalization; nats are nats) lies below
        mle_min_rate. Deliberately NOT a significance test: 'descent not
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
        checks = max(int(getattr(self.args, 'mle_slope_window', 300)) // 10, 4)
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
        rate_hi = rate + float(getattr(self.args, 'mle_slope_t', 2.0)) * se_rate
        flat = rate_hi < float(getattr(self.args, 'mle_min_rate', 0.05))
        self.protocol.publish_gate('mle_flat', float(flat))
        metrics['mle_gate_rate'] = rate  # nats/100 train steps; > 0 = improving
        metrics['mle_gate_rate_se'] = se_rate
        metrics['mle_gate_rate_hi'] = rate_hi  # the quantity actually tested
        metrics['mle_gate_flat'] = float(flat)
        return metrics

    def evaluation(self, override_do_figs: bool = False):
        metrics = {}
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
        metrics.update(self.log_metrics(fwd_stats, bwd_stats, sample_batch))
        if getattr(self, 'test_mol_dataset', None) is not None:
            metrics.update(self.log_test_metrics(eval_discretizer, fwd_stats))

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
                                              anchor_latents=anchor_latents)
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
        metrics.update(log_elapsed_times(self.times))
        self.times['eval_wrapup_end'] = time()
        self.times['eval_step_end'] = time()

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
            valid_losses = buff.ema_loss[~torch.isnan(buff.ema_loss)].cpu().numpy()
            metrics.update({
                'prior_buffer_length': len(buff),
                'prior_buffer_mean_steps': torch.nanmean(buff.select_counts.float()).item(),
                'prior_buffer_median_steps': torch.nanmedian(buff.select_counts.float()).item(),
                'prior_buffer_mean_loss': torch.nanmean(buff.ema_loss).item(),
                'prior_buffer_median_loss': torch.nanmedian(buff.ema_loss).item(),
                'prior_buffer_step_hist': safe_histogram(buff.select_counts.cpu().numpy()),
            })
            metrics.update(self.energy_reward_stats('prior_buffer', energy=buff.y))
            if len(valid_losses) > 0:
                metrics['prior_buffer_loss_hist'] = safe_histogram(np.clip(np.log10(valid_losses), min=-1, max=3))

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
            valid_replay_losses = self.replay_buffer.ema_loss[~torch.isnan(self.replay_buffer.ema_loss)].cpu().numpy()
            replay_age = (self.step_ind - self.replay_buffer.birth_step).float()
            metrics.update({
                'replay_buffer_length': len(self.replay_buffer),
                'replay_buffer_mean_steps': torch.nanmean(self.replay_buffer.select_counts.float()).item(),
                'replay_buffer_median_steps': torch.nanmedian(self.replay_buffer.select_counts.float()).item(),
                'replay_buffer_mean_loss': torch.nanmean(self.replay_buffer.ema_loss).item(),
                'replay_buffer_median_loss': torch.nanmedian(self.replay_buffer.ema_loss).item(),
                'replay_buffer_step_hist': safe_histogram(self.replay_buffer.select_counts.cpu().numpy()),
                'replay_buffer_mean_age': replay_age.mean().item(),
                'replay_buffer_max_age': replay_age.max().item() if replay_age.numel() > 0 else 0.0,
            })
            if hasattr(self, '_replay_admit_cap'):
                metrics['replay_buffer_admit_cap'] = self._replay_admit_cap
                metrics['replay_buffer_admit_health'] = self._replay_admit_health
            metrics.update(self.energy_reward_stats('replay_buffer', energy=self.replay_buffer.y))
            if len(valid_replay_losses) > 0:
                metrics['replay_buffer_loss_hist'] = safe_histogram(
                    np.clip(np.log10(valid_replay_losses), min=0, max=3))

            # churn accumulated since the previous eval's drain, i.e. one full
            # eval period of train steps; drained here so the counts are a rate
            # per window rather than a run-total
            admitted = self.replay_churn['admitted']
            metrics.update({
                'replay_buffer_admitted': admitted,
                'replay_buffer_evicted': self.replay_churn['evicted'],
                'replay_buffer_turnover': admitted / max(len(self.replay_buffer), 1),
            })
            for key in self.replay_churn:
                self.replay_churn[key] = 0

            # TTL-cohort readouts (see manage_replay_buffer's tally comments):
            # absorbed_frac = of rows resolved this window (corrected below the
            # floor OR expired), the fraction replay finished before the clock
            # -- the direct "is the TTL long enough for the supersampling rate"
            # signal. expired_undrawn_frac = expiries that never got a single
            # draw (wasted slots: buffer oversized or replay share too low).
            # expired_delta = mean death-minus-birth |resid| over drawn
            # expiries (negative = partial progress on the rows replay didn't
            # finish). expired_draws = mean draws those rows received.
            coh = self.replay_cohort
            resolved = coh['absorbed'] + coh['expired']
            if resolved > 0:
                metrics['replay_buffer_absorbed_frac'] = coh['absorbed'] / resolved
            if coh['expired'] > 0:
                metrics['replay_buffer_expired_undrawn_frac'] = coh['expired_undrawn'] / coh['expired']
            if coh['expired_delta_n'] > 0:
                metrics['replay_buffer_expired_delta'] = coh['expired_delta_sum'] / coh['expired_delta_n']
            if coh['expired_drawn'] > 0:
                metrics['replay_buffer_expired_draws'] = coh['expired_draws_sum'] / coh['expired_drawn']
            self.replay_cohort = {'absorbed': 0, 'expired': 0, 'expired_undrawn': 0,
                                  'expired_drawn': 0, 'expired_draws_sum': 0,
                                  'expired_delta_sum': 0.0, 'expired_delta_n': 0}

        if hasattr(self, 'anchor_buffer'):
            metrics['anchor_buffer_length'] = len(self.anchor_buffer)
            metrics.update(self.energy_reward_stats('anchor_buffer', energy=self.anchor_buffer.energy))
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

    def energy_reward_stats(self, prefix, energy=None, reward=None):
        """
        Mean, median, and histogram for both energy and reward, deriving
        whichever one wasn't passed in via the fixed sampling temperature
        (same reward = -energy / T convention as prebuilt_sample_to_reward
        and top_up_prior_from_anchors).
        """
        assert energy is not None or reward is not None, "must pass energy and/or reward"
        temperature = self.energy_function.temperature
        if energy is None:
            energy = -reward * temperature
        elif reward is None:
            reward = -energy / temperature

        energy_np = energy.detach().cpu().numpy()
        reward_np = reward.detach().cpu().numpy()

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
            f'{prefix}_mean_reward': float(np.mean(reward_np)),
            f'{prefix}_median_reward': float(np.median(reward_np)),
            f'{prefix}_min_reward': float(np.min(reward_np)),
            f'{prefix}_max_reward': float(np.max(reward_np)),
            f'{prefix}_reward_hist': safe_histogram(reward_np, num_bins=128),
        }

    def manage_prior_buffer(self, sample_batch):
        if not hasattr(self, 'prior_buffer'):
            self.prior_buffer = CrystalBuffer(
                sample_batch,
                device=self.buffer_device,
                max_z_prime=max(self.args.z_primes),
                x_fn=None,  # 'latent_params',
                y_fn=self.args.energy_function,
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
        headroom = max(0, self.args.buffers.prior_buffer.max_size - len(self.prior_buffer))

        if n_to_add > headroom:
            elig_idx, _, _ = self.prior_buffer.get_elig_drop_count(
                quantile=0.25,
                loss_floor=10.0,
                min_visits=5,
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
                quantile=0.25,
                loss_floor=10.0,
                min_visits=5,
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
        if budget <= 0 or not hasattr(self, 'prior_model'):
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

        # this cycle's prior-model draw came up short of admissible samples --
        # top up the gap from the permanent anchor archive instead of just
        # accepting a smaller churn this round
        shortfall = budget - int(good_inds.numel())
        if shortfall > 0 and getattr(self, 'anchor_buffer', None) is not None and len(self.anchor_buffer) > 0:
            self.top_up_prior_from_anchors(shortfall)

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
    def top_up_prior_from_anchors(self, n, purge_worst: bool = False):
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

        purge_worst: if True, first purge up to n of prior_buffer's lowest-
        reward (highest-energy) entries, so the anchor-sourced batch actively
        replaces stale/pinned material instead of just padding on top of it
        (used by the reach trigger; the shortfall trigger leaves this False
        since headroom for that case is already handled upstream).
        """
        cfg = self.args.buffers.anchor_buffer

        if purge_worst and self._prior_buffer_len() > 0:
            n_purge = min(n, len(self.prior_buffer))
            worst_first = torch.argsort(self.prior_buffer.y.cpu(), descending=True)  # highest energy = lowest reward
            self.prior_buffer.purge_by_index(worst_first[:n_purge].numpy())
            self.prior_churn['evicted'] += int(n_purge)

        n_draw = min(n, len(self.anchor_buffer))

        anchor_batch, anchor_inds, _ = self.anchor_buffer.sample_graphs(
            n_draw, replace=False, weighted=True, temperature=1.0, beta=cfg.replay_beta)
        anchor_batch = anchor_batch.clone().to(self.device)
        anchor_batch.log_noise_latent_parameters(*cfg.noise_log_range)

        anchor_batch, log_T_tensor, sg_inds, zps, condition, condition_id = self.energy_function.condition_samples(
            anchor_batch, sg_inds=anchor_batch.sg_ind, z_primes=anchor_batch.z_prime)
        anchor_batch.orient_molecule(mode='std')

        terminal_latents = anchor_batch.latent_params()
        reward, anchor_batch = self.energy_function.log_reward(
            terminal_latents, anchor_batch, log_T_tensor, return_exp=True)

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

        # Record-breaker admission: each drawn anchor's noised child (exactly
        # one per parent per topup, since draws are replace=False) stands for
        # admission iff it STRICTLY lowered its condition's Emin(c). No
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

        If buffers.anchor_buffer.mcmc is set (sweeps > 0), each tiled copy is
        relaxed by a short local Metropolis walk at its own target temperature
        (_metropolis_reheat) instead of a single isotropic kick, so the seed
        approximates each basin's local thermal shape -- a near-calibrated
        z_match log-Z handoff and a realistic mode handed to buildout, rather
        than a fixed-width noise ball. noise_log_range still drives the isotropic
        fallback (and top_up_prior_from_anchors, which is unchanged).
        """
        cfg = self.args.buffers.anchor_buffer
        uniq_ids, row_idx = self.anchor_buffer.best_per_condition_indices()
        if row_idx.numel() == 0:
            return

        tiled_idx = row_idx.repeat_interleave(n_per_condition)
        seed_batch = self.anchor_buffer.batch.subsample_new_batch(tiled_idx).clone().to(self.device)

        mcmc_cfg = getattr(cfg, 'mcmc', None)
        use_mcmc = mcmc_cfg is not None and getattr(mcmc_cfg, 'sweeps', 0) > 0
        if not use_mcmc:
            # isotropic fallback: one log-uniform latent kick per copy, applied
            # (as before) before conditioning so the noised state is what gets
            # conditioned, oriented and scored
            seed_batch.log_noise_latent_parameters(*cfg.noise_log_range)

        seed_batch, log_T_tensor, sg_inds, zps, condition, condition_id = self.energy_function.condition_samples(
            seed_batch, sg_inds=seed_batch.sg_ind, z_primes=seed_batch.z_prime)
        seed_batch.orient_molecule(mode='std')

        if use_mcmc:
            # replace the single kick with a local MH walk at target T; conditions
            # and log_T are now fixed per row (each chain lives at one condition)
            terminal_latents = self._metropolis_reheat(seed_batch, log_T_tensor, mcmc_cfg)
            if getattr(mcmc_cfg, 'log_geometry', True):
                self._reheat_geometry(terminal_latents, n_per_condition, uniq_ids)
        else:
            terminal_latents = seed_batch.latent_params()
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
    def _metropolis_reheat(self, batch, log_T_tensor, mcfg):
        """
        Local Metropolis-Hastings reheat of the seed anchors, replacing the
        single isotropic-noise kick in seed_prior_from_condition_minima. Every
        row of `batch` is an INDEPENDENT chain started at its own (argmin-anchor)
        latent; the ensemble of terminal latents is returned [B, D] and becomes
        the flushed prior seed.

        Rationale: a log-uniform noise shell has an entropy unrelated to the
        target, so the level backward-TB converges to on it (bounded by the
        buffer's own entropy) carries an E_q[log q/p] bias that z_match then has
        to walk off. MH at target T instead relaxes each anchor toward the LOCAL
        thermal shape of its basin, so the seeded buffer's level sits near the
        local free energy (near-calibrated handoff) and the mode shape handed to
        buildout's forward policy is realistic rather than a fixed-radius ball.

        Correctness notes (see energies/molecular_crystal.py):
          - energy() builds the crystal straight from the raw latent x, so we
            score proposals directly -- no latent_to_cell_params round trip.
          - the +-1 latent box is a SOFT temperature-scaled quadratic wall in
            generator_energy (the raw_latents penalty), and the change of
            variables is already in jacobian_energy, so plain MH on log_reward
            samples the true target with NO boundary clip/reject or Jacobian
            bookkeeping. Non-finite scores are rejected defensively (they should
            not arise: inf/nan crystal terms are patched to 0, which scores worse
            than a cohesive minimum and so is rejected on its own).

        Locality: single-basin is held by starting at the one lowest-energy
        anchor and taking SMALL isotropic steps (barriers >> kT at target T), not
        by a geometric radius (which would re-impose the shape bias we remove).
        Optional energy_ceiling_kt rejects proposals more than that many kT above
        the chain's running-best reward -- a barrier guard that preserves the
        thermal shape; it works in log_reward space, already per-row-kT-scaled,
        so one dimensionless value is correct across conditions. Default off.

        sigma is per-row Robbins-Monro toward target_accept during burn_in, then
        frozen so the collected terminal is a draw from a fixed-kernel chain.
        """
        sweeps = int(getattr(mcfg, 'sweeps', 200))
        burn_in = min(int(getattr(mcfg, 'burn_in', sweeps // 2)), sweeps)
        target_accept = float(getattr(mcfg, 'target_accept', 0.3))
        adapt_rate = float(getattr(mcfg, 'sigma_adapt_rate', 0.1))
        sigma_min = float(getattr(mcfg, 'sigma_min', 1.0e-4))
        sigma_max = float(getattr(mcfg, 'sigma_max', 0.5))
        ceiling = getattr(mcfg, 'energy_ceiling_kt', None)
        ceiling = None if ceiling is None else float(ceiling)

        x = batch.latent_params().clone()                # [B, D], std-oriented start
        B, device = x.shape[0], x.device

        def score(latents):
            # log_reward = -energy/T, per row; the one-time big-batch pass is made
            # self-healing (chunk-on-OOM) rather than a hard crash at the boundary
            return -self.energy_function.energy(
                latents, batch, log_T_tensor,
                return_exp=False, internal_oom_recovery=True).to(device)

        lr = score(x)                                    # [B]
        best_lr = lr.clone()                             # running basin floor (max reward)
        sigma = torch.full((B, 1), float(getattr(mcfg, 'init_sigma', 0.02)),
                           device=device, dtype=x.dtype)

        x0 = x.clone()
        acc_ema = torch.full((B,), target_accept, device=device)  # log-only
        max_drift = torch.zeros(B, device=device)

        for t in range(sweeps):
            x_prop = x + sigma * torch.randn_like(x)
            lr_prop = score(x_prop)
            accept = torch.isfinite(lr_prop) & (
                torch.log(torch.rand(B, device=device)) < (lr_prop - lr))
            if ceiling is not None:
                accept = accept & (lr_prop >= best_lr - ceiling)

            x = torch.where(accept[:, None], x_prop, x)
            lr = torch.where(accept, lr_prop, lr)

            if t < burn_in:
                best_lr = torch.maximum(best_lr, lr)
                sigma = (sigma * torch.exp(
                    adapt_rate * (accept.float()[:, None] - target_accept))
                         ).clamp_(sigma_min, sigma_max)

            acc_ema = 0.98 * acc_ema + 0.02 * accept.float()
            max_drift = torch.maximum(max_drift, (x - x0).norm(dim=-1))

        # eyeball diagnostics: excursion = best_lr - lr is the terminal energy
        # above the basin floor in units of that row's kT; equipartition puts its
        # mean near D/2 for a ~D-dim basin, so a mean near 0 means the walk never
        # heated (sigma too small / burn_in too short) and a very large mean or
        # drift means chains ran out of the basin
        excursion = best_lr - lr
        print(
            f"metropolis_reheat: {sweeps} sweeps (burn {burn_in}), {B} chains | "
            f"accept {acc_ema.mean().item():.2f} (min {acc_ema.min().item():.2f}) | "
            f"sigma mean {sigma.mean().item():.3f} [{sigma.min().item():.3f}, {sigma.max().item():.3f}] | "
            f"terminal excursion/kT mean {excursion.mean().item():.2f} max {excursion.max().item():.2f} | "
            f"latent drift max {max_drift.max().item():.3f}"
            + ("" if ceiling is None else f" | ceiling {ceiling:.1f} kT"))
        return x

    @torch.no_grad()
    def _reheat_geometry(self, x, n_per_condition, uniq_ids=None):
        """
        Landscape survey riding along free on _metropolis_reheat's terminal
        ensemble: per condition, the shape of its local thermal mode IN LATENT
        SPACE -- the space the policy actually has to fit, so this is the
        anisotropy that matters for fitting, not the physical-space one.

        One DxD covariance + eigendecomposition per condition, once at the stage
        boundary. Per condition:

          log10_kappa   log10 of the covariance condition number lam_max/lam_min
                        -- the anisotropy. Large = the mode is a thin sliver:
                        stiff cooperative contact-compression directions against
                        soft collective slide/libration directions.
          n_soft        participation ratio (sum lam)^2 / sum(lam^2): the
                        effective number of directions carrying the thermal
                        variance, i.e. the floppy-mode count. The jamming
                        reading is that denser / higher-coordination targets have
                        FEWER of these and should be correspondingly harder.
          bend_deg      angle between the leading eigenvector of the low half and
                        of the high half of the cloud (split at the median
                        projection onto the global leading eigenvector). ~0 for a
                        straight anisotropic valley; large means the principal
                        axis ROTATES as you traverse the mode -- a curved valley,
                        which no whitening / linear reparameterization can
                        straighten, and the reason plain ill-conditioning
                        understates the difficulty.
          lam_gap       log10(lam_max / lam_2nd). ONLY read bend_deg where this
                        is appreciable: with no well-separated leading axis the
                        two half-clouds pick near-orthogonal noise directions and
                        bend saturates near 90 for a perfectly isotropic mode.
                        The printed bend summary is restricted to lam_gap > 0.1.
                        That bar has to stay LOW: curvature itself inflates
                        lam_2nd (the bent coordinate carries real variance), so a
                        strict bar throws away exactly the curved modes it is
                        meant to qualify. Calibrated on synthetics -- isotropic
                        0.02, curved valley 0.38, straight sliver 3.8.
          max_abs_skew  largest |skewness| over the principal axes: the signature
                        of asymmetric truncation -- a direction that is soft one
                        way and runs into an exponential contact wall the other.

        Read bend and skew together: straight-but-wall-clipped is (low bend, high
        skew); a curved valley is (high bend, high skew) since bending itself
        skews the bent coordinate. Curvature is what bend alone identifies.

        Needs n_per_condition > D + 1 for the covariance (and > 2D + 2 for the
        bend's half-cloud split); returns None otherwise. Full per-condition
        tensors are stashed on self._last_reheat_geometry for inspection.
        """
        n = int(n_per_condition)
        if n < 2 or x.shape[0] % n != 0:
            return None
        M, D = x.shape[0] // n, x.shape[1]
        if n <= D + 1:
            print(f"reheat geometry: skipped -- n_per_condition {n} <= latent dim {D} + 1, "
                  f"the per-condition covariance would be rank-deficient")
            return None

        xc = x.detach().to(torch.float64).reshape(M, n, D)
        dx = xc - xc.mean(dim=1, keepdim=True)
        cov = dx.transpose(1, 2) @ dx / (n - 1)                   # [M, D, D]
        lam, vec = torch.linalg.eigh(cov)                         # ascending
        lam = lam.clamp_min(1e-30)

        log10_kappa = torch.log10(lam[:, -1] / lam[:, 0])
        lam_gap = torch.log10(lam[:, -1] / lam[:, -2])   # leading-axis separation
        n_soft = lam.sum(dim=1) ** 2 / (lam ** 2).sum(dim=1)

        proj = dx @ vec                                           # [M, n, D] principal basis
        skew = ((proj / proj.std(dim=1, keepdim=True).clamp_min(1e-30)) ** 3).mean(dim=1)
        max_abs_skew = skew.abs().max(dim=1).values

        half = n // 2
        if half > D + 1:
            order = proj[:, :, -1].argsort(dim=1)

            def half_cloud(idx):
                return torch.gather(dx, 1, idx[:, :, None].expand(-1, -1, D))

            def lead_vec(block):
                b = block - block.mean(dim=1, keepdim=True)
                _, v = torch.linalg.eigh(b.transpose(1, 2) @ b / (b.shape[1] - 1))
                return v[:, :, -1]

            cos = (lead_vec(half_cloud(order[:, :half])) *
                   lead_vec(half_cloud(order[:, -half:]))
                   ).sum(dim=1).abs().clamp(max=1.0)             # sign-folded
            bend_deg = torch.rad2deg(torch.arccos(cos))
        else:
            bend_deg = torch.full((M,), float('nan'), dtype=torch.float64, device=x.device)

        stats = {'log10_kappa': log10_kappa.cpu(), 'n_soft': n_soft.cpu(),
                 'bend_deg': bend_deg.cpu(), 'max_abs_skew': max_abs_skew.cpu(),
                 'lam_gap': lam_gap.cpu()}
        if uniq_ids is not None:
            stats['condition_id'] = uniq_ids.detach().cpu().flatten()
        self._last_reheat_geometry = stats

        def med(t):
            finite = t[torch.isfinite(t)]
            return float(finite.median()) if finite.numel() else float('nan')

        # bend is only meaningful where a leading axis is actually resolved
        bend_ok = stats['bend_deg'][torch.isfinite(stats['bend_deg']) & (stats['lam_gap'] > 0.1)]
        bend_worst = float(bend_ok.max()) if bend_ok.numel() else float('nan')
        bend_med = float(bend_ok.median()) if bend_ok.numel() else float('nan')
        worst = int(torch.argmax(stats['log10_kappa']))
        worst_id = ('' if uniq_ids is None
                    else f" @condition_id {int(stats['condition_id'][worst])}")
        print(
            f"reheat geometry: {M} conditions, D={D}, n={n} per condition | "
            f"log10 kappa med {med(stats['log10_kappa']):.2f} "
            f"worst {float(stats['log10_kappa'].max()):.2f}{worst_id} | "
            f"n_soft med {med(stats['n_soft']):.1f} min {float(stats['n_soft'].min()):.1f} | "
            f"bend deg [{bend_ok.numel()}/{M} resolved] med {bend_med:.0f} worst {bend_worst:.0f} | "
            f"max|skew| med {med(stats['max_abs_skew']):.2f} "
            f"worst {float(stats['max_abs_skew'].max()):.2f}")
        return stats

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

    def replay_step_delta(self):
        if not hasattr(self, 'prev_replay_step_count'):
            num_replay_steps = self.args.eval_period
        else:
            num_replay_steps = self.replay_step_count - self.prev_replay_step_count
        self.prev_replay_step_count = self.replay_step_count
        return num_replay_steps

    def manage_replay_buffer(self, fwd_stats, sample_batch):
        """
        Stash the full forward trajectory of on-policy samples with sufficiently
        overweighted (high positive or negative residual) terminal states, so
        they can later be exactly replayed (get_traj_replay) rather than
        re-sampled backward.

        Two-parameter softmax admission/purge (admit_temperature T, a health-
        modulated cap) plus a residence TTL, replacing rank-based argsort /
        beat-the-min admission:

        - Score each candidate/incumbent by |resid| (admission) or ema_loss
          (purge), CLIPPED to `cap` before dividing by T, then draw without
          replacement from softmax(clipped_score / T) -- rather than always
          taking the top-k. Clip-then-divide keeps `cap`'s meaning (an
          absolute nats bound) independent of whatever T is set to. A single
          T can't both preserve fine discrimination among ordinary candidates
          AND stop one extreme value from dominating (the two live at
          different score scales); clipping is what makes the softmax
          indifferent among everything past `cap`, so no single outlier can
          monopolize admission -- it merely competes on equal footing with
          whatever else is also pinned at the ceiling. Purge mirrors this:
          LOW ema_loss (boring, already-corrected incumbents) gets high
          eviction weight.
        - `cap` is health-modulated: cap(h) = cap_min + (cap_max-cap_min) /
          (1 + h/h0), h = the tracker's EMA of fwd/scatter_err -- a signal
          computed off fresh on-policy rollouts, external to this buffer's
          own contents, so it can't ratchet off its own contamination the
          way a self-referential (buffer-derived) ceiling would. Healthy
          policy -> cap ~ cap_max, softmax sharply prefers the worst-
          surviving candidates (supersample the shoulder -- stiff-wall
          margins etc., under-visited by on-policy forward sampling alone).
          Unhealthy policy -> cap ~ cap_min, most candidates clip to the
          same logit and admission goes near-uniform across a bounded
          population: a live, representative, non-poisoning snapshot of the
          CURRENT (bad) distribution rather than a targeted chase of
          whatever's currently worst -- and the buffer stays populated
          through the excursion instead of draining, so it's ready the
          moment health recovers. T stays FIXED throughout -- it only
          governs shoulder sharpness and must not be recoupled to health,
          for the same reason a single T can't serve both cap's jobs above:
          making T health-dependent would re-couple shoulder sharpness to
          health even though cap already isolates the tail response.
        - Domain-sanity gate: candidates with non-finite log_r/resid are
          hard-excluded upstream of the softmax entirely (never merely
          throttled) -- a NaN/inf-energy state isn't "a real sample that's
          extreme," there's no slow-not-stop tradeoff to make there.
        - Residence TTL (max_residence_steps, counted in TRAIN STEPS, not
          select_counts): unconditional age eviction regardless of ema_loss,
          so the buffer is a decaying reservoir of the CURRENT policy's tail
          rather than an accumulating one -- any one row's contents are
          meaningless on their own, only the live population matters.
        """
        log_r = fwd_stats['log_r']
        log_pf = fwd_stats['log_pf']
        log_pb = fwd_stats['log_pb']
        log_Z_learned = fwd_stats['log_Z_learned'] if 'log_Z_learned' in fwd_stats else fwd_stats['log_Z']

        # resid stays on CPU: all the eviction logic below runs against the
        # buffer's CPU-resident ema_loss bookkeeping
        resid = ((log_pf - log_pb) - (log_r - log_Z_learned)).cpu()
        # domain-sanity gate: hard-exclude, never throttled (see docstring)
        sane = torch.isfinite(log_r).cpu() & torch.isfinite(resid)

        rb_cfg = self.args.buffers.replay_buffer
        floor = 0.1  # pool-definition cutoff only -- all real discrimination
        # happens in the softmax + cap below, not in this threshold

        h = self.metric_tracker.get('fwd', 'scatter_err')
        h = max(float(h), 0.0) if h is not None else 0.0  # cold start: treat as healthy
        cap_max = float(rb_cfg.admit_cap_max)
        cap_min = float(rb_cfg.admit_cap_min)
        h0 = float(rb_cfg.admit_cap_health_h0)
        cap = cap_min + (cap_max - cap_min) / (1.0 + h / h0)
        T = float(rb_cfg.admit_temperature)
        self._replay_admit_cap = cap
        self._replay_admit_health = h

        elig = torch.argwhere(sane & (resid.abs() > floor)).flatten()
        clipped_score = resid[elig].abs().clamp(max=cap)
        # trajectories go wherever the buffer lives -- no forced D2H when GPU-resident
        flow_states = fwd_stats['flow_states'].detach().to(self.buffer_device)

        # --- bootstrap ---
        if not hasattr(self, 'replay_buffer'):
            if elig.numel() == 0:
                return
            add_inds = elig[_softmax_draw(clipped_score, rb_cfg.max_size, T)]
            self.replay_buffer = CrystalBuffer(
                sample_batch.subsample_new_batch(add_inds),
                device=self.buffer_device,
                max_z_prime=max(self.args.z_primes),
                x_fn=None,
                y_fn=self.args.energy_function,
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

        # --- unconditional eviction: strictly overfit incumbents (ema below
        # floor -- resid corrected toward zero) plus rows past the residence
        # ceiling. This is the buffer's primary turnover mechanism: "any one
        # row should be meaningless" is enforced by age, not by a separate
        # random-churn pass -- see docstring ---
        floor_mask = ema < floor
        expired_mask = torch.zeros_like(floor_mask)
        max_residence = int(getattr(rb_cfg, 'max_residence_steps', 0) or 0)
        if max_residence > 0:
            expired_mask = (self.step_ind - self.replay_buffer.birth_step) > max_residence
        toxic_mask = floor_mask | expired_mask
        toxic = torch.argwhere(toxic_mask).flatten()

        # --- TTL-cohort telemetry, tallied by eviction CAUSE. Floor eviction =
        # absorbed (a row admitted above the floor only gets under it by being
        # drawn and corrected). TTL expiry = death-by-clock, exogenous to the
        # loss value, so death-vs-birth deltas on that cohort carry no
        # selection-on-outcome bias -- unlike floor/displacement evictions,
        # which are excluded from the delta for exactly that reason. NB an
        # undrawn row's ema never updates after admission (update_losses only
        # touches drawn rows), so deltas are only defined on the drawn subset;
        # undrawn expiries are counted separately as wasted slots.
        self.replay_cohort['absorbed'] += int(floor_mask.sum())
        expired_only = expired_mask & ~floor_mask
        n_expired = int(expired_only.sum())
        if n_expired > 0:
            counts = self.replay_buffer.select_counts[expired_only]
            drawn = counts > 0
            delta = (ema[expired_only] - self.replay_buffer.birth_loss[expired_only])[drawn]
            delta = delta[torch.isfinite(delta)]
            self.replay_cohort['expired'] += n_expired
            self.replay_cohort['expired_undrawn'] += int((~drawn).sum())
            self.replay_cohort['expired_drawn'] += int(drawn.sum())
            self.replay_cohort['expired_draws_sum'] += int(counts[drawn].sum())
            self.replay_cohort['expired_delta_sum'] += float(delta.sum())
            self.replay_cohort['expired_delta_n'] += int(delta.numel())

        # --- admission, paced by elapsed replay steps (mirrors how the prior
        # buffer paces on num_bwd_steps): draw without replacement from
        # softmax(clipped |resid| / T) over this batch's eligible pool ---
        num_replay_steps = self.replay_step_delta()
        n_admit = min(elig.numel(), int(num_replay_steps * rb_cfg.churn_rate))
        add_inds = elig[_softmax_draw(clipped_score, n_admit, T)]

        # --- purge: TTL/toxic eviction frees headroom first; softmax-drawn
        # displacement of the weakest LIVE incumbents (low ema_loss = high
        # eviction weight, same cap/T basis as admission) covers whatever
        # admission needs beyond that ---
        headroom = max(0, rb_cfg.max_size - (n - toxic.numel()))
        n_extra_purge = max(0, add_inds.numel() - headroom)
        extra_purge = torch.zeros(0, dtype=torch.long)
        if n_extra_purge > 0:
            live = torch.argwhere(~toxic_mask).flatten()
            purge_score = ema[live].clamp(max=cap)
            extra_purge = live[_softmax_draw(-purge_score, n_extra_purge, T)]

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
        churn rate (manage_prior_buffer) once phase 2/3 starts drawing from
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
        dataset's own phase-1 bookkeeping. Must run after init_identifiers()
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
        seed_batch, _, _, _, _, _ = self.energy_function.condition_samples(
            seed_batch,
            sg_inds=getattr(seed_batch, 'sg_ind', None),
            z_primes=getattr(seed_batch, 'z_prime', None))
        seed_batch = AnchorBuffer._drop_keys(seed_batch, ("smiles", "identifier"))
        return seed_batch

    def _fresh_prior_buffer(self, seed_batch):
        """Construct a new CrystalBuffer around seed_batch with clean
        per-sample records -- the single construction recipe shared by init
        seeding and seed_prior_from_condition_minima's flush path."""
        return CrystalBuffer(
            seed_batch,
            device=self.buffer_device,
            max_z_prime=max(self.args.z_primes),
            x_fn=None,
            y_fn=self.args.energy_function,
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
                seed_batch.latent_params(), seed_batch,
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

        seed_batch, log_T_tensor, sg_inds, zps, condition, condition_id = self.energy_function.condition_samples(
            seed_batch, sg_inds=getattr(seed_batch, 'sg_ind', None), z_primes=getattr(seed_batch, 'z_prime', None))
        temperature = 10 ** log_T_tensor
        reward = self.energy_function.prebuilt_sample_to_reward(seed_batch, temperature)
        energy = -reward.detach() * temperature

        # Warm each seeded condition's Emin(c) from these on-target seed
        # energies. The anchor buffer and condition_log_z are distinct objects:
        # seeding the former does NOT inform the latter, and BOTH the admission
        # plausibility gate (screen_and_admit_anchors) and thin()'s purge gate
        # calibrate against best_energy(c), not against the anchor buffer's own
        # energies. Without this, best_energy(c) stays inf until phase-2 prior
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

        self.anchor_buffer = AnchorBuffer(
            seed_batch,  # function-owned transient; the buffer moves it to buffer_device itself
            device=self.buffer_device,
            reward=reward.cpu(),
            energy=energy.cpu(),
            max_z_prime=max(self.args.z_primes),
            exclude_keys=BULKY_ATTR_EXCLUDE_KEYS,
        )

    @torch.no_grad()
    def screen_and_admit_anchors(self, sample_batch, log_r, energy, log_pf_est):
        """
        Surprise-gated anchor promotion: a state is anchor-worthy iff it has
        good Boltzmann weight (energy near Emin(c), the per-condition running
        best tracked by condition_log_z) *and* the current forward policy
        under-samples it *relative to its Boltzmann weight* -- not in an
        absolute sense.

        "Surprise" is the trajectory-balance residual itself: the forward/
        backward log-ratio measured against the reward/Z log-ratio,

            surprise = (log_pf - log_pb) - (log_r - log_Z(c))
                     = log_Z(c) + log_pf - log_pb - log_r

        which is 0 at the TB fixed point. surprise << 0 means the policy's
        forward-vs-backward ratio falls far short of what the state's reward
        (relative to Z(c)) warrants -- an under-weighted high-reward mode.
        log_Z(c) = condition_log_z.lookup(c) (its stable ema_logw target).
        Since log_r and log_Z(c) are both deterministic given x/condition,
        centering is a per-candidate shift that adds no rollout variance to
        the log_pf - log_pb estimate. Two-stage:

          1. Screen: cheap energy pre-filter AND'd with the surprise gate. The
             log_pf - log_pb term here is the free k=1 forward-path value
             computed in fwd_eval_sampling and passed in as log_pf_est, so no
             backward rollout is spent screening. Both gates are pure tensor
             comparisons; nothing is rolled out until confirm. Candidates
             whose condition lacks a warmed-up log_Z(c) (lookup mask False)
             are held back -- the axis isn't yet trustworthy.
          2. Confirm: K backward rollouts (gflownet_losses.log_pf_estimate,
             IWAE/logsumexp) on screen-survivors only give an accurate
             log p_hat in place of log_pf - log_pb, fed through the *same*
             residual form -- rare, so K=5-10 is cheap in aggregate. Dominated
             by the best rollout, so a single unlucky trajectory can't fake
             surprise; still poor -> admitted.

        Runs on both prior-model-sampled batches (via sample_from_prior ->
        fwd_eval_sampling) and on-policy eval batches (evaluation() ->
        fwd_eval_sampling) -- both are freshly generated, freshly scored
        terminal batches routed through fwd_eval_sampling's single call site,
        so one criterion gates both instead of two divergent admission rules.
        (top_up_prior_from_anchors' record-breaker children deliberately
        bypass this gate: a strictly deeper version of an already-admitted
        anchor needs no fresh novelty judgment -- and a damaged policy cannot
        fake a record-breaker, since energies are real; see that method.)

        The whole screen is additionally behind a POLICY-HEALTH gate
        (health_gate_r2 / health_gate_zerr, see the body): admissions pause
        while forward calibration or Z's gradient signal is unhealthy --
        which also means they pause through early buildout and briefly after
        stage transitions, by design (novelty judged against a miscalibrated
        ruler isn't novelty).

        original_surprise (the TB residual at admission) is stored frozen on
        each admitted anchor and used by thin() to rank the buffer -- centered
        on log_Z(c), so it's comparable across conditions. AnchorBuffer.admit's
        dup_cutoff is a cheap literal-duplicate catch on this (already rare)
        confirmed set, not a novelty judgment -- see AnchorBuffer's docstring.
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
        # policy-health gate: refuse to adjudicate novelty while the ruler is
        # broken. Surprise is measured THROUGH the live policy's log_pf, so a
        # damaged policy reads its own log_pf collapse as nats of fake
        # surprise on everything (c8utdn8q: 382 admissions in the single eval
        # inside the 20-22k LR excursion, with fwd/r2 at 0.84-0.89 the whole
        # window -- and the flood beats condition_log_z's ema_logw absorption,
        # which cancels a uniform shift only after its half-life lag). Gate on
        # POOLED fwd/r2 plus Z's own gradient signal (|EMA'd
        # fwd/tb_resid_clipped|). Pooled r2 deliberately, even though the
        # conditional gates elsewhere no longer use it: this is a DAMAGE
        # detector ('is the policy still physics', bar 0.9 vs the 0.84-0.89
        # collapse), not a calibration gate, and pooled r2's between-condition
        # inflation doesn't hide a collapse that severe. A cold channel (phase-1
        # seeding: no fwd stats yet) abstains rather than blocks, preserving
        # seeding behavior exactly.
        r2 = self.metric_tracker.get('fwd', 'r2')
        z_grad = self.metric_tracker.get('fwd', 'tb_resid_clipped')
        if ((r2 is not None and r2 < getattr(cfg, 'health_gate_r2', 0.9))
                or (z_grad is not None and abs(z_grad) > getattr(cfg, 'health_gate_zerr', 0.5))):
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
        latents = sample_batch.latent_params()

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
            self.anchor_buffer = AnchorBuffer(
                admit_batch, device=self.buffer_device,
                reward=admit_reward, energy=admit_energy,
                original_surprise=original_surprise,
                max_z_prime=max(self.args.z_primes),
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
        # prior trained live in this run's phase 1 does not, so fall back to the
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
                self.prior_buffer = CrystalBuffer(
                    sample_batch,
                    device=self.buffer_device,
                    max_z_prime=max(self.args.z_primes),
                    x_fn=None,  # 'latent_params',
                    y_fn=self.args.energy_function,
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
        # stream, and every training-facing consumer of it (phase-2 logw_std
        # gate, phase-3 bootstrap_log_z target, z_grad/controller, persistent
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
    modeller = Modeller()
    modeller.train()
