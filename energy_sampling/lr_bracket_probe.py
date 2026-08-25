"""
The bracket's trainer side: takes a complete root snapshot, runs each candidate
as a fixed-LR continuation from it, and promotes the winner's end state.

WHAT MAKES A TRIAL A MEASUREMENT RATHER THAN A STORY, in order of how easily each
one fails silently:

  1. EVERY CANDIDATE STARTS FROM THE SAME STATE. Not "the same weights" -- the
     same weights, optimizer moments, replay/prior/anchor buffers, metric-tracker
     EMAs, clip-guard quantiles, protocol counters, logical step and RNG. Any one
     of those left live carries the previous candidate's luck into the next, and
     the carry is invisible: the trial still runs, still reports, still gets
     selected. The test that catches it is not an assertion in here -- it is
     running two candidates at the SAME scale and requiring bitwise-identical
     losses (tests/lr/test_lr_bracket_driver.py).

  2. NOTHING ALIASES. `state_dict()` hands back LIVE tensors, and in this
     codebase PyG's `.to()` / `.cpu()` MUTATE IN PLACE. A snapshot holding
     references would therefore be rewritten by the first restore that used it,
     and the second candidate would start from the first one's end state while
     every seam reported success. Everything captured here is copied to host
     memory at capture time and copied AGAIN on the way out, and the buffer
     restore deep-copies before handing a state dict to `from_state_dict`.

  3. THE ADAM STEP COUNTER ROUND-TRIPS, AND ITS FAILURE IS LOUD. Adam's update
     carries sqrt(1 - beta2^t) / (1 - beta1^t) relative to steady state: 0.153 at
     t=10, 0.309 at t=100, 0.795 at t=1000, 0.975 at t=3000. A trial that
     silently restarted the counter would run at 15-30% of its nominal rate for
     its first hundred steps -- so a too-hot rung survives, and the bracket
     reports a boundary it never found. `Checkpointer.load_optimizer_state` has
     two fallbacks that reset the optimizer and only PRINT; those are legitimate
     recovery outside a bracket and a corrupted measurement inside one, so this
     module calls it with `strict=True` and then reads the counter BACK and
     raises if it moved.

  4. THE HARD-FAILURE BAR CAN ACTUALLY FIRE. See `HardFailureBars`. A bracket
     whose bars never fire finds no boundary, reports `unbracketed_high` every
     time, and returns the same answer forever while appearing to work.

WHERE THE SNAPSHOTS LIVE, and why. Host RAM, not disk. A snapshot is the model,
the optimizer moments and the buffers -- on the crystal route about 350 MB, so a
six-rung bracket holds around 2.4 GB. That is affordable and it is reported
(`lr_bracket/held_mb`) rather than assumed. The consequence is deliberate and
stated: they die with the process, so a resume that lands mid-bracket does NOT
pretend to hold checkpoints it no longer has -- it re-arms the cycle from the
resumed state, which is already mature, and says so.

WHAT IS DELIBERATELY NOT SNAPSHOTTED, with the reason, so a future reader does
not read the omission as an oversight:

  * the conformer `ConformerTorsions._batch` cache -- a memoisation of the
    collated tree and force field keyed on batch size. It is a pure function of
    the batch size and the molecule, so restoring it changes nothing a
    continuation can observe; it costs recompute and nothing else.
  * the ray larder, when one exists. It is a rolling harvest feeding a
    diagnostic-only sensor and never reaches an optimizer step. It is CLEARED at
    every trial start rather than restored, which is identical for every
    candidate and so cannot contaminate one with another.
  * wall-clock telemetry (`_throughput`, `_gpu_util`). Trial compute is not
    promoted training, and charging its seconds to the run's throughput meters
    would make the batch sizer read a rung it never ran.
"""

from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch

from energy_sampling.buffer import ConditionLogZTracker
from energy_sampling.lr_bracket import LRBracket
from energy_sampling.utils import is_cuda_oom


# --------------------------------------------------------------------- helpers

def _to_host(obj):
    """Deep copy onto the host, tensors included.

    `copy.deepcopy` alone keeps CUDA tensors on the CARD, so deep-copying an
    optimizer state dict would double VRAM per snapshot and OOM the bracket long
    before it ran out of host memory. Everything here is walked explicitly.
    """
    if torch.is_tensor(obj):
        return obj.detach().to('cpu', copy=True)
    if isinstance(obj, dict):
        return {k: _to_host(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_to_host(v) for v in obj]
        return type(obj)(out) if isinstance(obj, tuple) else out
    return copy.deepcopy(obj)


def _adam_t(opt) -> int | None:
    """The optimizer's step counter, or None if it has never stepped.

    MINIMUM over parameters that carry state, not maximum: bias correction is a
    property of the update each parameter receives, and the least-stepped
    parameter is the one whose rate is furthest from nominal. Under gradient
    accumulation or a branch that only sometimes fires, the spread is real.
    """
    ts = []
    for group in opt.param_groups:
        for p in group['params']:
            st = opt.state.get(p)
            if not st or 'step' not in st:
                continue
            t = st['step']
            ts.append(int(t.item()) if torch.is_tensor(t) else int(t))
    return min(ts) if ts else None


def bias_correction(t: int, beta1: float, beta2: float) -> float:
    """Adam's update magnitude relative to its steady state.

        sqrt(1 - beta2^t) / (1 - beta1^t)

    At beta1 0.9 / beta2 0.999 this is 0.153 at t=10 and 0.975 at t=3000, i.e. a
    fresh optimizer applies a SIXTH of the rate its config names. That is why the
    root has to be old before anything is bracketed from it.
    """
    if t is None or t <= 0:
        return 0.0
    num = math.sqrt(max(0.0, 1.0 - beta2 ** t))
    den = 1.0 - beta1 ** t
    return num / den if den > 0 else 0.0


# ------------------------------------------------------------------- the bars

class HardFailureBars:
    """Direct, hard safety conditions, DERIVED FROM THE ROOT'S OWN LOSS SCALE.

    WHY DERIVED RATHER THAN CONFIGURED. The shipped absolute bars were 1e9 and
    caught nothing short of numerical overflow: a rate one rung too hot on this
    route took the loss from about -25 to +318, eight orders of magnitude under
    the bar. Every rung then completes, no boundary is found, and the bracket
    returns the same answer forever while every seam fires correctly.

    WHY A SPAN AND NOT A RATIO. The obvious relative bar is `loss > k x min_seen`,
    and it is inert on exactly the route this mechanism is being validated on:
    the MLE channel passes through zero and goes NEGATIVE, so the running minimum
    has no positive scale to take a ratio against and the rule correctly declines
    rather than manufacturing a fixed absolute bar out of a floor. The span bar
    has no such blind spot -- `hi + k x (hi - lo)` is well defined whatever the
    sign, and on the -25 -> +318 excursion with a burn-in span of a few nats it
    fires within tens of steps.

    The window is the TAIL of burn-in, not all of it: the opening of burn-in is
    the transient the burn-in exists to pass through, and folding it into the
    scale would inflate the span by the very movement that is not supposed to
    count as normal.

    Absolute bars survive as what they always were -- a backstop against
    numerical death -- and `LRBracket._check_bars` refuses a configuration in
    which they are the only thing standing.

    THE ROOT BARS ARE FITTED AT `burn_in_scale`, and that is the right scale for
    judging a TRIAL -- every trial restores the same root and is comparable to
    it. It is the WRONG scale for the live tripwire that runs for the rest of the
    stage at the promoted rate. Refitting the live bars to that rate is the
    CONTROLLER's job (`LRController._arm_cruise_bar` / `_tick_cruise_bar`); this
    class only knows how to draw a bar from a window it is handed.
    """

    #: Channels that reach `check_spike`. Mirrors train_step's `step_type`.
    CHANNELS = ('fwd', 'bwd', 'replay', 'fused')

    def __init__(self, loss_excursion_k=10.0, grad_excursion_x=100.0,
                 loss_abs=1.0e6, grad_abs=1.0e6, root_window=200,
                 min_observations=20):
        self.loss_excursion_k = float(loss_excursion_k)
        self.grad_excursion_x = float(grad_excursion_x)
        self.loss_abs = None if loss_abs is None else float(loss_abs)
        self.grad_abs = None if grad_abs is None else float(grad_abs)
        self.root_window = int(root_window)
        self.min_observations = int(min_observations)
        self.loss_bar = {}          # channel -> absolute value that counts as failure
        self.loss_hi = {}           # channel -> root window hi (bar - hi = k*span)
        self.grad_bar = None
        self.grad_hi = None
        self.last_fire = None       # structured facts of the most recent verdict
        self.scale_note = ''

    def derive(self, loss_history, grad_history):
        """Fit the bars to the root. `loss_history` is {channel: [values]} and
        `grad_history` a list of pre-clip gradient norms, both from burn-in.

        Returns None on success, or a string saying why a bracket taken from this
        root could not fail a candidate -- which is a refusal, not a warning.
        """
        self.loss_bar, self.loss_hi, notes = {}, {}, []
        for channel, values in (loss_history or {}).items():
            tail = [v for v in list(values)[-self.root_window:] if math.isfinite(v)]
            if len(tail) < self.min_observations:
                continue
            hi, lo = max(tail), min(tail)
            span = hi - lo
            if not (span > 0):
                # A perfectly flat channel gives the span nothing to work with.
                # Fall back to the magnitude, which is the only scale left.
                span = abs(hi) if abs(hi) > 0 else 1.0
            self.loss_bar[channel] = hi + self.loss_excursion_k * span
            self.loss_hi[channel] = hi
            notes.append(f'{channel}: [{lo:.4g}, {hi:.4g}] -> bar '
                         f'{self.loss_bar[channel]:.4g}')
        gtail = [g for g in list(grad_history or ())[-self.root_window:]
                 if math.isfinite(g) and g > 0]
        if len(gtail) >= self.min_observations:
            self.grad_bar = max(gtail) * self.grad_excursion_x
            self.grad_hi = max(gtail)
            notes.append(f'grad: max {max(gtail):.4g} -> bar {self.grad_bar:.4g}')
        self.scale_note = '; '.join(notes)
        if not self.loss_bar:
            return (f'no channel accumulated {self.min_observations} finite loss '
                    f'observations during burn-in, so no bar can be derived from the '
                    f'root scale and no candidate could be failed on anything but '
                    f'numerical overflow. A bracket that cannot fail a candidate is '
                    f'not a bracket.')
        return None

    #: An excursion this far past the bar, measured in bar-excesses over the
    #: root hi (bar - hi = k*span), is DECISIVE: it skips the confirmation
    #: re-run. 3x means the value overshot the root by triple what the bar
    #: already demanded. Non-finite and absolute-backstop verdicts are decisive
    #: by kind. (Evidence-scaled confirmation, owner 2026-08-25.)
    DECISIVE_X = 3.0

    def judge(self, channel, loss, grad_norm):
        """'why this candidate failed', or None. Direct conditions only: no
        fitted trend, no loss ratio, no cosine, no composite health score.

        Every non-None verdict also stamps `self.last_fire` with the structured
        facts (kind, value, bar, hi, decisive) so the DRIVER can classify the
        failure without parsing its own strings. Read it immediately after a
        hit; it is overwritten by the next one."""
        def fire(kind, decisive, why, value=None, bar=None, hi=None):
            self.last_fire = {'kind': kind, 'decisive': bool(decisive),
                              'value': value, 'bar': bar, 'hi': hi}
            return why

        if loss is not None:
            if not math.isfinite(loss):
                return fire('nonfinite', True, f'nonfinite_loss_{channel}')
            if self.loss_abs is not None and abs(loss) >= self.loss_abs:
                return fire('abs', True, f'loss_abs_{channel}_{loss:.4g}', loss)
            bar = self.loss_bar.get(channel)
            if bar is not None and loss >= bar:
                hi = self.loss_hi.get(channel, bar)
                decisive = loss >= hi + self.DECISIVE_X * (bar - hi)
                return fire('loss_excursion', decisive,
                            f'loss_excursion_{channel}_{loss:.4g}_over_{bar:.4g}',
                            loss, bar, hi)
        if grad_norm is not None:
            if not math.isfinite(grad_norm):
                return fire('nonfinite', True, 'nonfinite_grad')
            if self.grad_abs is not None and grad_norm >= self.grad_abs:
                return fire('abs', True, f'grad_abs_{grad_norm:.4g}', grad_norm)
            if self.grad_bar is not None and grad_norm >= self.grad_bar:
                hi = getattr(self, 'grad_hi', None) or (self.grad_bar / self.grad_excursion_x)
                decisive = grad_norm >= hi + self.DECISIVE_X * (self.grad_bar - hi)
                return fire('grad_excursion', decisive,
                            f'grad_excursion_{grad_norm:.4g}_over_{self.grad_bar:.4g}',
                            grad_norm, self.grad_bar, hi)
        return None

    def report(self) -> dict:
        out = {}
        for c, bar in self.loss_bar.items():
            out[f'lr_bracket/bar_loss_{c}'] = float(bar)
        if self.grad_bar is not None:
            out['lr_bracket/bar_grad'] = float(self.grad_bar)
        return out


# -------------------------------------------------------------- the snapshot

class TrainerSnapshot:
    """A complete, host-resident, non-aliasing copy of everything a continuation
    depends on.

    THE FIELD LIST IS THE CONTRACT. Anything mutable that a training step can
    move and a later step can read has to be here; a field left out does not
    raise, it just leaks one candidate's state into the next. The heavy stores
    (model, optimizers, buffers, condition_log_z) go through their own
    state_dicts because those are already tested; the pure-Python trackers are
    copied WHOLE rather than through `state_dict()`, because several of them
    drain counters on read -- `grad_clip_guard`'s fire counters and
    `MetricTracker.written_at` / `changed_keys` are all absent from their
    state_dicts and all of them are read by something that decides.
    """

    #: Modeller attributes copied verbatim. The first block is exactly
    #: MODELLER_STATE_DEFAULTS -- what a disk checkpoint already carries -- and
    #: the second is the mutable trainer state a disk checkpoint does NOT carry
    #: but a within-process continuation still depends on.
    FIELDS = (
        'step_ind', 'stage', 'stage_ctrl', 'batch_size', 'batch_size_last_grow',
        'batch_size_cooldown_until', 'batch_size_oom_ceiling',
        'batch_size_oom_ceiling_at', 'batch_size_oom_min', 'batch_sizer',
        'grow_buffer', 'fwd_step_count', 'bwd_step_count', 'replay_step_count',
        'fwd_frac', 'bwd_frac', 'replay_frac', 'combo_loss_record', 'lr_ctrl',
        # ---- not in MODELLER_STATE_DEFAULTS, and each one decides something
        'fused_accum_count',        # a partial accumulation cycle
        '_probe_exclude_from',      # held-out boundary for the diagnostic probe
        '_grad_nonfinite_streak',   # the abort counter; a stale one aborts early
        '_nonfinite_pending',       # a pending divergence response
        'last_grad_norm_pre_clip',  # read by every guard
        '_hyper_prev_step',         # the diagnostic sensor's operand
        '_z_cal_rollouts',
        # DRAINED-AT-REPORT COUNTERS. These accumulate during a trial and are
        # published at the next 10-step report as if they described promoted
        # training, so without them the run's own gradnorm and z_cal channels
        # carry the discarded candidates' events. `_grad_nonfinite` is the one
        # that matters: a hot rung's non-finite steps would be reported against
        # the rate the run actually kept.
        '_grad_nonfinite', '_z_cal_report',
        '_recent_step_times', '_recent_step_work',   # the batch sizer's rung median
        'batch_oom_events',
    )

    def __init__(self, modeller, label: str):
        m = self.m = modeller
        self.label = label
        self.model = _to_host(m.gfn_model.state_dict())
        # `update_ema_model` ALIASES the two models when ema_decay is None, so a
        # blind second state_dict would store the same tensors twice and, worse,
        # a blind restore would write the EMA weights over the train weights.
        self.ema_is_model = m.ema_model is m.gfn_model
        self.ema = None if self.ema_is_model else _to_host(m.ema_model.state_dict())
        self.optimizers = {k: _to_host(o.state_dict()) for k, o in m.optimizers.items()}
        self.adam_t = {k: _adam_t(o) for k, o in m.optimizers.items()}
        self.rng = {
            'torch': torch.get_rng_state(),
            'cuda': (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
        self.fields = {k: copy.deepcopy(getattr(m, k))
                       for k in self.FIELDS if hasattr(m, k)}
        # `_to_host` ON THE WAY IN. A buffer's `state_dict()` moves its tensors
        # with `.cpu()` -- which on a CPU-RESIDENT buffer returns the tensor
        # ITSELF, not a copy -- so on `buffer_device: cpu` the state dict hands
        # back the live store and a snapshot built straight from it would alias
        # the trainer.
        #
        # THIS IS DEFENCE IN DEPTH, stated plainly rather than dressed up as the
        # live fix: `restore` deep-copies before handing anything to
        # `from_state_dict`, so with that in place nothing currently mutates the
        # captured tensors and no behavioural test convicts this line -- measured
        # by mutation. The structural assertion in
        # tests/lr/test_lr_bracket_driver.py is what pins it. Both copies guard
        # the same hazard from opposite ends, and only one of them sits on the
        # path a future refactor is likely to touch.
        self.buffers = {name: _to_host(getattr(m, name).state_dict())
                        for name in ('prior_buffer', 'replay_buffer', 'anchor_buffer')
                        if getattr(m, name, None) is not None}
        self.condition_log_z = (m.condition_log_z.state_dict()
                                if getattr(m, 'condition_log_z', None) is not None
                                else None)
        self.metric_tracker = copy.deepcopy(m.metric_tracker.__dict__)
        self.grad_guard = copy.deepcopy(m.grad_guard.__dict__)

    # ------------------------------------------------------------------ size

    def held_bytes(self) -> int:
        total = 0

        def walk(o):
            nonlocal total
            if torch.is_tensor(o):
                total += o.element_size() * o.nelement()
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    walk(v)
            elif hasattr(o, '__dict__'):
                walk(vars(o))

        for part in (self.model, self.ema, self.optimizers, self.buffers,
                     self.condition_log_z):
            walk(part)
        return total

    # --------------------------------------------------------------- restore

    @torch.no_grad()
    def restore(self, seed=None):
        """Put the trainer back exactly where this snapshot was taken.

        `seed` is the ONE thing a caller may vary, and only the boundary
        confirmation does. Every screen trial restores the captured RNG state so
        the candidates are comparable; a confirmation re-run under the SAME seed
        would be a deterministic replay of the trial it is meant to confirm, so
        it would reproduce the failure by construction and confirm nothing.
        """
        m = self.m
        # COPY ON THE WAY OUT AS WELL AS IN. `load_state_dict` copies tensor
        # contents so the model is safe, but the buffer path below hands its
        # state dict to `from_state_dict`, which calls `.to(device)` on a PyG
        # Data -- and that MUTATES IN PLACE in this codebase. Without the copy
        # the first restore would promote this snapshot's stored CPU batch to the
        # GPU and the second candidate would then share storage with the first.
        m.gfn_model.load_state_dict(_to_host(self.model))
        if not self.ema_is_model:
            m.ema_model.load_state_dict(_to_host(self.ema))
        elif m.ema_model is not m.gfn_model:
            m.ema_model = m.gfn_model

        # STRICT. Outside a bracket the two fallbacks in `load_optimizer_state`
        # are legitimate recovery; inside a trial they are a corrupted
        # measurement that looks normal, because a reset counter runs the trial
        # at 15-30% of its nominal rate for its first hundred steps.
        m.checkpointer.load_optimizer_state(
            {'optimizers': {k: _to_host(v) for k, v in self.optimizers.items()}},
            strict=True)
        for key, opt in m.optimizers.items():
            want, got = self.adam_t.get(key), _adam_t(opt)
            if want != got:
                raise RuntimeError(
                    f"lr_bracket: optimizer '{key}' restored with step counter {got}, "
                    f"but the root was saved at {want}. Adam's update carries "
                    f"sqrt(1-beta2^t)/(1-beta1^t), so a trial from a reset counter runs "
                    f"at a fraction of its nominal rate and a too-hot rung survives "
                    f"because the rate under test is not the rate applied. Refusing to "
                    f"measure from a corrupted root.")

        for name, state in self.buffers.items():
            cls = type(getattr(m, name))
            setattr(m, name, cls.from_state_dict(copy.deepcopy(state),
                                                 device=m.buffer_device))
        if self.condition_log_z is not None:
            m.condition_log_z = ConditionLogZTracker.from_state_dict(
                copy.deepcopy(self.condition_log_z),
                current_step=int(self.fields.get('step_ind', 0)))

        m.metric_tracker.__dict__.update(copy.deepcopy(self.metric_tracker))
        m.grad_guard.__dict__.update(copy.deepcopy(self.grad_guard))
        for k, v in self.fields.items():
            setattr(m, k, copy.deepcopy(v))

        # The harvest feeding the diagnostic ray sensor is dropped rather than
        # restored -- see the module docstring. Identical for every candidate,
        # so it cannot carry one into another.
        larder = getattr(m, 'larder', None)
        if larder is not None:
            larder.clear()

        # THE HOT-LR SENSOR'S TRAILING WINDOWS DESCRIBE STEPS THIS TRIAL DID NOT
        # TAKE. Left alone they would carry the PREVIOUS candidate's rows into
        # this one -- and the grid is ascending, so the previous rung is COOLER
        # and its tail dilutes this rung's rise. That masks a failure and pushes
        # the boundary up, which is the unsafe direction. Cleared, not restored:
        # the sensor is report-only, so there is nothing to preserve.
        hot = getattr(m, '_hot_lr', None)
        if hot is not None:
            hot.reset()

        if seed is None:
            r = self.rng
            torch.set_rng_state(r['torch'])
            if r['cuda'] is not None:
                torch.cuda.set_rng_state_all(r['cuda'])
            np.random.set_state(r['numpy'])
            random.setstate(r['python'])
        else:
            seed = int(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2 ** 32))
            random.seed(seed)


# ----------------------------------------------------------------- the driver

class BracketDriver:
    """Runs one `LRBracket` against a live trainer.

    SERIAL, ON ONE GPU, ON PURPOSE. Candidates could be run concurrently in
    separate processes; that would need a second orchestration layer, a second
    failure mode, and a second place for the root state to drift. The bracket is
    cheap enough that it does not need one -- six rungs at 150 steps is 900
    discarded steps, once per stage.
    """

    def __init__(self, modeller, bracket: LRBracket, bars: HardFailureBars,
                 verbose: bool = True):
        self.m = modeller
        self.bracket = bracket
        self.bars = bars
        self.verbose = bool(verbose)
        self.root = None
        self.trial_states = {}       # label -> TrainerSnapshot for a surviving trial
        self.trial_branch_peaks = {}  # label -> {branch: peak loss over the trial}
        self.root_log_z = None       # the guiding-star baseline; None = guard off
        self.root_log_z_slope = 0.0  # root Z drift (<=0), extrapolated under trials
        self.last_summary = ''
        self._held_bytes = 0
        self.race_ooms = 0           # CUDA OOMs this race; the budget is ONE retry

    # ------------------------------------------------------------------ root

    def take_root(self, step: int):
        """Snapshot the mature state and check it is fit to bracket from.

        Returns None on success, or a refusal string. The bias-correction check
        reads the optimizer's ACTUAL step counter rather than trusting that
        burn-in was long enough -- if the stage did not rebuild its optimizers,
        t is already large and the check passes trivially, which is correct
        rather than a loophole.
        """
        self.root = TrainerSnapshot(self.m, 'root')
        self._held_bytes = self.root.held_bytes()
        # The guiding-star baseline: candidates are judged against the ROOT's
        # log Z (fail on a detour of logz_detour_nats below it). None wherever Z
        # is not being trained/logged, which disables the guard for the stage.
        #
        # DRIFT-COMPENSATED (2026-08-25, qm9c_wk_aug25/fj119r1o): at STAGE ENTRY
        # Z is still walking toward its empirical level, so the root's snapshot
        # is not a stationarity reference -- the first var_conditioning ladder
        # failed EVERY rung on a uniform ~0.1 nat/step walk-down that had
        # nothing to do with the rates (c0, the burn-in rate itself, "detoured"
        # 2.2 nats; the confirm reproduced it exactly). The baseline therefore
        # extrapolates the root's own measured drift: an EMA with period tau
        # lags a linear walk by drift*tau, so (raw - ema)/tau estimates the
        # slope from state that already exists. Only DOWNWARD drift is
        # compensated -- the star is monotone-up, and a rung that merely stalls
        # a rising Z is not a detour. The rate-driven signal survives cleanly:
        # on the failed ladder, compensation acquits c0/c1 (pure drift) and
        # still convicts 0.2x and above (3.6+ nats over the drifting baseline).
        self.root_log_z = self._live_log_z()
        self.root_log_z_slope = 0.0
        if self.root_log_z is not None:
            stats = getattr(self.m, '_last_stats', None) or {}
            tracker = getattr(self.m, 'metric_tracker', None)
            for d in ('fused', 'fwd', 'bwd', 'replay'):
                raw = stats.get(d, {}).get('log_Z_learned')
                ema = tracker.get(d, 'log_Z_learned') if tracker is not None else None
                if raw is not None and ema is not None and math.isfinite(float(ema)):
                    tau = float(getattr(tracker, 'period', 100.0) or 100.0)
                    self.root_log_z_slope = min(0.0, (float(raw) - float(ema)) / tau)
                    break
        factor, t, key = self.root_bias_correction()
        if self.verbose:
            print(f'lr_bracket: root at step {step}; Adam step counter t={t} on '
                  f"'{key}', bias correction {factor:.3f} "
                  f'(min required {self.bracket.min_root_bias_correction:.3f}); '
                  f'root state {self._held_bytes / 1e6:.0f} MB')
        if factor < self.bracket.min_root_bias_correction:
            return (
                f'root bias correction {factor:.3f} is below '
                f'min_root_bias_correction {self.bracket.min_root_bias_correction:.3f} '
                f"(optimizer '{key}' has taken t={t} steps). Adam applies "
                f'sqrt(1-beta2^t)/(1-beta1^t) of its nominal rate, so every trial '
                f'descended from this root would run at {factor:.2f}x the rate under '
                f'test and a too-hot rung would survive for a reason that has nothing '
                f'to do with the rate. RAISE lr_control.burn_in_steps.')
        return None

    def root_bias_correction(self):
        """(factor, t, optimizer key) for the WORST managed optimizer.

        Worst rather than mean: one under-stepped optimizer is enough to make the
        composite step smaller than nominal, and the bracket is asking whether
        the rate under test is the rate applied.
        """
        managed = self._managed_optimizer_keys()
        worst = None
        for key, opt in self.m.optimizers.items():
            if key not in managed:
                continue
            t = _adam_t(opt)
            betas = opt.param_groups[0].get('betas', (0.9, 0.999))
            # t is None means the optimizer has never stepped, which is a
            # bias correction of ZERO, not of one. A root whose managed
            # optimizer has taken no steps is the extreme case of the failure
            # this check exists for, so it must read as the worst value rather
            # than being skipped.
            f = 0.0 if t is None else bias_correction(t, float(betas[0]), float(betas[1]))
            if worst is None or f < worst[0]:
                worst = (f, t, key)
        # NOTHING MANAGED AND STEPPING means the bracket's multiplier reaches no
        # learning rate this stage takes a step with -- the documented control
        # arm. There is no rate under test, so there is nothing to refuse.
        return worst if worst is not None else (1.0, None, 'none-stepping')

    def _managed_optimizer_keys(self):
        """The optimizers the bracket's multiplier actually reaches AND this stage
        actually steps.

        BOTH HALVES ARE LOAD-BEARING. Asking about bias correction on an
        optimizer the bracket does not steer would refuse a bracket over an
        irrelevant fact; asking about one the STAGE does not step refuses every
        bracket on every canonical config, because a stage runs one train_mode
        while all four policy rates are managed, so three of the four optimizers
        have never taken a step and report a counter of None. See
        `LRController.stepping_optimizer_keys`."""
        return self.m.lr_controller.stepping_optimizer_keys()

    # ----------------------------------------------------------------- trials

    def run_trial(self, trial):
        """One candidate, with a one-OOM retry budget for the whole race.

        AN OOM IS NOT A MEASUREMENT (owner, 2026-08-25). It says nothing about
        the rate -- trials run at whatever batch the preceding cruise grew to,
        plus the race's own overhead (root restores fragment the pool; qm9c's
        c5 OOMed after five clean 150-step rungs at the same batch). So the
        first OOM in a race discards the attempt, clears the allocator cache
        and REPEATS THE RUNG from the root -- at the SAME batch, because a
        rung measured at a smaller batch is not comparable to the rungs
        already on the board, and because the shared recovery's batch cut +
        OOM ceiling describe cruise memory, not race memory. A SECOND OOM
        anywhere in the race is a symptom (VRAM squeeze or leak, not rate
        evidence): it propagates to the controller's abort seat, which hands
        it to the shared recovery and holds the kept rate.
        """
        while True:
            try:
                return self._run_trial_once(trial)
            except Exception as e:               # noqa: BLE001 -- OOM-classified
                if not is_cuda_oom(e):
                    raise
                self.race_ooms += 1
                if self.race_ooms > 1:
                    raise
                print(f'lr_bracket: {trial.label} OOMed mid-trial -- not a '
                      f'measurement, and not a batch cut either (a rung at a '
                      f'smaller batch is not comparable). Cache cleared, '
                      f'repeating the rung once; a second OOM in this race '
                      f'aborts it.')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _run_trial_once(self, trial):
        """One attempt: restore the root, set the rate once, hold it for the
        whole horizon, record what happened.

        THE CANDIDATE MAY NOT RESCUE ITSELF. Divergence handling inside a trial
        reports failure and ENDS the trial -- it never lowers the candidate's
        rate and continues, because a candidate that cut its way to survival has
        measured a rate the bracket never tested and would then be selected as
        though it had held the one it was given.
        """
        m = self.m
        self.root.restore(seed=trial.seed)
        ctrl = m.lr_controller
        ctrl.set_scale(trial.scale, why=f'trial {trial.label}')
        applied = ctrl.live_rates()
        if self.verbose:
            print(f'lr_bracket: {trial.label} -- scale {trial.scale:.6g}, '
                  f'{trial.kind}'
                  + (f', seed {trial.seed}' if trial.seed is not None else '')
                  + f'; effective LRs ' + ', '.join(
                      f'{k}={v:.3g}' for k, v in sorted(applied.items()))
                  + f'; {self.bracket.trial_steps} steps')

        start_step = int(m.step_ind)
        done, reason, fail_at = 0, None, None
        settle = int(getattr(self.bracket, 'trial_settle_steps', 0) or 0)
        peaks, logz_decisive = {}, False
        for i in range(self.bracket.trial_steps):
            m.step_ind = start_step + i + 1
            reason = self._one_step()
            done = i + 1
            # RACE-TABLE VISIBILITY: per-branch peak losses over the trial, so
            # a minority branch destabilizing under a dominant frac is at least
            # SEEN even though it holds no veto (owner decision 2026-08-24).
            stats = getattr(m, '_last_stats', None) or {}
            for d in ('fwd', 'bwd', 'replay', 'fused'):
                v = stats.get(d, {}).get('loss')
                if v is not None and math.isfinite(v):
                    peaks[d] = max(peaks.get(d, float('-inf')), float(v))
            if reason is None and self.bracket.logz_detour_nats is not None \
                    and done > settle and self.root_log_z is not None:
                lz = self._live_log_z()
                # baseline extrapolates the root's own (downward-only) drift --
                # see take_root: a walking Z at stage entry is the stage, not
                # the rung, and judging against a frozen snapshot convicted a
                # whole ladder uniformly (fj119r1o).
                baseline = (self.root_log_z
                            + getattr(self, 'root_log_z_slope', 0.0) * done)
                if lz is not None and lz < baseline - self.bracket.logz_detour_nats:
                    detour = baseline - lz
                    logz_decisive = detour >= (HardFailureBars.DECISIVE_X
                                               * self.bracket.logz_detour_nats)
                    reason = (f'logz_detour_{detour:.4g}_nats_below_'
                              f'drift-adjusted_root'
                              f'_(bar_{self.bracket.logz_detour_nats:g})')
            if reason is not None and done <= settle and not (
                    reason.startswith('nonfinite') or reason.startswith('exception')):
                # THE SWITCH-SPLASH WINDOW: bar verdicts inside the first
                # trial_settle_steps are not judged -- the instant rate jump
                # produces a stochastic few-step excursion that measures the
                # JUMP, not the rate (toy workout D9: the same rate failed a
                # trial at step 3 and cruised 1500 steps clean in fixed mode).
                # Non-finite values and exceptions still fail immediately; a
                # genuinely fatal rate is convicted the step the window closes,
                # and its wrecked state is discarded on restore either way.
                reason = None
            if reason is not None:
                fail_at = done
                break
        self.trial_branch_peaks[trial.label] = peaks
        # EVIDENCE CLASS of the failure (owner, 2026-08-25): decisive failures
        # skip their confirmation re-run. Non-finite and exceptions are decisive
        # by kind; bar verdicts carry the classification the bars stamped at
        # fire time; a log-Z detour is decisive at DECISIVE_X x its own bar.
        decisive = False
        if reason is not None:
            if reason.startswith('nonfinite') or reason.startswith('exception'):
                decisive = True
            elif reason.startswith('logz_detour'):
                decisive = logz_decisive
            else:
                lf = getattr(self.bars, 'last_fire', None) or {}
                decisive = bool(lf.get('decisive'))

        ok = reason is None
        if ok:
            # EVERY SURVIVOR'S END STATE IS KEPT, on a uniform path. Branching on
            # "will this one be selected" is how a restore ends up on the wrong
            # snapshot; the extra host memory is trivial against the trial
            # compute that produced it.
            self.trial_states[trial.label] = TrainerSnapshot(m, trial.label)
            self._held_bytes += self.trial_states[trial.label].held_bytes()
        self.bracket.record(trial, ok, reason, done, fail_at,
                            decisive=decisive)
        if self.verbose:
            print(f'lr_bracket: {trial.label} '
                  + ('SURVIVED all ' + str(done) + ' steps'
                     if ok else f'FAILED after {done} steps -- {reason}'
                          + (' [DECISIVE]' if decisive else '')))
        return ok

    def _live_log_z(self):
        """The last training step's batch-mean log_Z_learned, read from the raw
        per-direction stats -- present on both routes (conditional = the batch
        mean over conditions), absent (None) wherever Z does not train or
        nothing has logged yet, which switches the detour guard off."""
        stats = getattr(self.m, '_last_stats', None) or {}
        for d in ('fused', 'fwd', 'bwd', 'replay'):
            v = stats.get(d, {}).get('log_Z_learned')
            if v is not None and math.isfinite(float(v)):
                return float(v)
        return None

    def _one_step(self):
        """One training step at the candidate's fixed rate. Returns a hard-failure
        reason or None.

        The body mirrors the host loop's step -- train_logic, train_step, the
        interspersed Z work -- and deliberately omits everything that is
        reporting rather than training: no eval, no checkpoint write, no batch
        sizing, no protocol tick. A trial that ran the protocol tick could fire a
        stage transition from inside a measurement.
        """
        m = self.m
        m._z_cal_rollouts = 0
        step_type = m.train_logic(m.step_ind)
        try:
            loss = m.train_step(step_type)
        except Exception as e:                       # noqa: BLE001 -- classified below
            if is_cuda_oom(e):
                # NOT a candidate failure. An OOM says nothing about the rate, and
                # convicting the lowest rung on one would bound the bracket at a
                # rate the run never tested. Let it out: run_trial repeats the
                # rung once at the same batch; a second OOM aborts the race.
                raise
            return f'exception_{type(e).__name__}'
        if loss is not None:
            m.z_level_fill()
            m.z_calibration_tick(step_type)
        if getattr(m, '_nonfinite_pending', False):
            m._nonfinite_pending = False
            return 'nonfinite_gradient'
        return self.bars.judge(step_type, loss,
                               getattr(m, 'last_grad_norm_pre_clip', None))

    # -------------------------------------------------------------- the cycle

    def _publish_race(self):
        """Stamp the finished race into wandb: the human-readable table into the
        RUN SUMMARY (keyed per cycle, so repeats never overwrite each other) and
        a per-rung wandb.Table for API queries. Until this, the per-rung results
        lived only in the run log -- comparing ladders across battery arms meant
        grepping N SLURM .out files.

        FIRE-AND-FORGET: reporting may never kill (or stall) a run, and the CPU
        test harnesses drive this path with no wandb.init at all -- so any
        failure is one print, never a raise.
        """
        try:
            import wandb
            if wandb.run is None:
                return
            cycle = self.bracket.cycle_index
            wandb.run.summary[f'lr_bracket/summary_cycle_{cycle}'] = self.last_summary
            rows = self.bracket.race_rows()
            if rows:
                cols = ['label', 'kind', 'scale', 'seed', 'survived',
                        'steps_to_failure', 'reason', 'decisive',
                        'peak_fwd', 'peak_bwd', 'peak_replay', 'peak_fused']
                for r in rows:
                    peaks = self.trial_branch_peaks.get(r['label'], {})
                    for d in ('fwd', 'bwd', 'replay', 'fused'):
                        r[f'peak_{d}'] = peaks.get(d)
                wandb.log({f'lr_bracket/race_cycle_{cycle}':
                           wandb.Table(columns=cols,
                                       data=[[r[c] for c in cols] for r in rows])})
        except Exception as e:                       # noqa: BLE001 -- reporting only
            print(f'lr_bracket: race report not published '
                  f'({type(e).__name__}: {e})')

    def run(self, step: int):
        """The whole bracket, start to promotion. Returns the number of PROMOTED
        steps the host loop's clock must skip -- the winner's horizon, never the
        sum of all trial compute.
        """
        while True:
            trial = self.bracket.next_trial()
            if trial is None:
                break
            self.run_trial(trial)

        verdict = self.bracket.select()
        self.last_summary = self.bracket.summary()
        if self.verbose:
            print(self.last_summary)
        self._publish_race()

        label = verdict['restore']
        if label is not None and label in self.trial_states:
            self.trial_states[label].restore()
            promoted = self.bracket.trial_steps
        else:
            # ALL_FAILED, NO_ELIGIBLE, or a survivor whose state went missing --
            # the last cannot happen on the uniform save path above, and is
            # handled rather than asserted because falling back to the root is
            # always safe and silently restoring the WRONG state is not.
            self.root.restore()
            promoted = 0
            if label is not None:
                print(f'lr_bracket: selected {label} but its end state is not held; '
                      f'falling back to the root at the burn-in scale.')
        self.m.lr_controller.set_scale(verdict['scale'], why='promoted')
        self.bracket.promote(verdict['scale'], int(self.m.step_ind))
        self.release()
        return promoted

    def release(self):
        """Drop every held state. The bracket has promoted one of them into the
        live trainer, so what is left is a few gigabytes describing rates the run
        is not going to use."""
        self.root = None
        self.trial_states.clear()
        self._held_bytes = 0

    def report(self) -> dict:
        out = dict(self.bars.report())
        if self._held_bytes:
            out['lr_bracket/held_mb'] = self._held_bytes / 1e6
        if self.race_ooms:
            out['lr_bracket/race_ooms'] = self.race_ooms
        return out
