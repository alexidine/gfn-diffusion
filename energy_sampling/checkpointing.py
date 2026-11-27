import inspect
import os
import re
import shutil
from copy import deepcopy
from typing import Optional

import torch

from energy_sampling.buffer import CrystalBuffer, AnchorBuffer, ConditionLogZTracker, _per_condition_min, \
    BufferCurrencyError
from energy_sampling import noised_prior
from energy_sampling.config_invariants import row_store_settings
from energy_sampling.protocol import fresh_stage_ctrl
from energy_sampling.utils import atomic_save, keeps_ema_copy, normalize_problem_def, trajectory_T
from models import GFN
from models.nf_policy import FRAME_ONLY_KEYS, LOSS_ONLY_KEYS, PRECISION_ONLY_KEYS, resolve_policy_entry

MODELLER_STATE_DEFAULTS = {
    'step_ind': 0,
    # THE SOURCE'S STEP, for labels only. A weights-only warm start trains from step_ind 0 (stages, schedules and
    # counters start fresh), but its curves and archives should read on from where the weights came from: wandb logs
    # global_step = step_ind + step_offset (the x-axis of every chart) and step archives are named by it.
    # load_weights_only sets it to the source checkpoint's global step, so chains of warm starts accumulate.
    'step_offset': 0,
    # rows the dataset-mode MLE draw has taken from prior_dataset (data/mle_passes = this / its rows); rides in the
    # checkpoint so a resumed MLE stage keeps counting its passes over the fixed set
    '_mle_rows_drawn': 0,
    # z_level_fill's absorber (z_calibration.fill_mode: absorb): the running
    # uncertainty of log Z and the step of the last applied measurement. They
    # ride in the checkpoint so a resume or a rewind does not re-snap to the
    # first batch it sees as if nothing were known.
    '_z_fill_P': None,
    '_z_fill_last_applied': None,
    # THE PROGRESS GATE'S EVIDENCE. progress_gate needs SIX evals per metric
    # before _series will return anything, and the production phase-1 stage exits
    # on `gates/progress_done` -- whose on_exit is the ONLY thing that writes
    # phase1_exit.pt and the prior snapshot. Unpersisted, a resumed leg started
    # with an empty history and could not conclude for ~6 evals (~3000 steps at
    # eval_period 500, ~5.8 h on the MLIP route at 7 s/step). min_history does not
    # save it: that guard is keyed on the ABSOLUTE step, which a resume inherits
    # at ~19000, so it passes instantly and the blackout is invisible.
    # The failure that matters is not the wasted steps: a leg that ENDS inside the
    # blackout never fires on_exit, so it writes NO phase1_exit and NO prior, and
    # a _best.pt without a phase1_exit is the known-fatal case. On a leg shorter
    # than the rebuild, phase 1 could never exit at all.
    # A plain dict of key -> list of (step, value), already capped at 400 entries
    # per key by the writer, so it costs a few hundred floats.
    '_progress_history': None,
    # the run's position in the config's protocol.stages list, BY NAME --
    # checkpoints carry position only; behavior (coeffs, balance rules, exit
    # thresholds) is always re-derived from the current config, so editing the
    # config rewrites a resumed run's future without any override flag. None =
    # fresh run; StageProtocol.begin pins it to the first stage.
    'stage': None,
    # all mutable stage-engine state (see protocol.fresh_stage_ctrl): exit-term
    # pass streaks, balance-rule running bests / floors / live annealed
    # thresholds, gate windows + published values (gates/mle_flat), the chosen
    # boost, and the pulled-forward-eval request. request_eval is stamped True
    # into pre-transition snapshots ('phase1_exit' etc.) so a resumed run pulls
    # its eval to the first post-resume step and the exit trigger -- whose
    # streaks ride in this same dict -- re-fires the transition through the
    # normal eval -> maybe_advance path. Reset at every stage transition.
    'stage_ctrl': fresh_stage_ctrl(),
    # restored, but never above the current config's max_batch_size, and not at
    # all when that config pins the batch (grow_batch_size: false) -- see
    # Checkpointer.reconcile_batch_size
    'batch_size': 1,
    # jump-mode growth (batch_growth_interval > 0): step of the last size jump
    # (or OOM cut), so the growth clock survives resume
    'batch_size_last_grow': 0,
    'batch_size_cooldown_until': -1,  # step_ind until which batch size growth is frozen after a cut
    # smallest batch known to OOM in the CURRENT stage (None = none seen). The
    # sizer's domain stops strictly below it; protocol.advance clears it because
    # the incoming stage has its own memory profile. Worth checkpointing so a
    # mid-stage resume doesn't have to rediscover the ceiling the hard way.
    'batch_size_oom_ceiling': None,
    # step the ceiling above was last recorded. It EXPIRES (see select_batch_size),
    # so the clock has to travel with it. None means UNSTAMPED, not step 0: a resume
    # at step 20000 that restored the ceiling against a 0 clock would expire it on the
    # first post-resume step -- the opposite failure from the one expiry exists to fix.
    # select_batch_size stamps an unstamped ceiling at the current step, so a
    # restored ceiling serves its full quiet window from the resume.
    'batch_size_oom_ceiling_at': None,
    # smallest batch EVER seen to OOM in this stage. Unlike the ceiling above it does
    # not expire, and it exists only to keep the ceiling MONOTONE: when an expired
    # ceiling is re-probed and OOMs again, the walk approaches from below and so
    # re-discovers a slightly HIGHER size than the original. Without this the ceiling
    # would ratchet upward one probe at a time, forgetting the smallest size it has
    # direct evidence does not fit.
    'batch_size_oom_min': None,
    # the batch sizer's conclusion for the current stage (None = not yet decided;
    # see train.select_batch_size): phase, reason, selection, and the measured rung
    # table. It MUST travel with batch_size_oom_ceiling above -- a resume that
    # restored the ceiling but not the conclusion would re-run a calibration the
    # run already paid for. Its predecessor state (the knee pin) taught the same
    # lesson the hard way: a resume restored the ceiling and left the pin
    # undefined, and the job died of AttributeError outside the train loop's
    # try/except.
    'batch_sizer': None,
    'grow_buffer': False,
    'fwd_step_count': 0,
    'bwd_step_count': 0,
    'replay_step_count': 0,
    'fwd_frac': 0.0,
    'bwd_frac': 1.0,
    'replay_frac': 0.0,
    'combo_loss_record': [],
    # which metric family produced combo_loss_record. A record compared under a
    # different metric makes 'best' degenerate to 'running' (or freeze) -- the
    # default None mismatches every real signature, so old checkpoints clear
    # the record on their first post-resume tick.
    'combo_loss_metric': None,
    # LRController state (see controller.py): the live scale and the stage's
    # cumulative fire cut. The schedule's entry step rides stage_ctrl
    # ('lr_sched_entry'). A missing/mismatched 'ver' means "not yet attached, or
    # written by a controller this build does not speak" -- discarded and
    # rebuilt, never reinterpreted.
    'lr_ctrl': {'ver': None, 'scale': None, 'cut': 1.0},
    # the conformer route's prior-draw stream (ConformerModeller._prior_rng, a numpy
    # Generator created from cfg seed on first use). Unsaved, every leg recreated it from
    # the seed, so a requeued leg REPLAYED the first leg's InternalPrior draws: the prior
    # dataset and every later churn draw repeated, with nothing saying so. It pickles
    # under torch.save and set_state_dict restores it before init_prior_dataset reads it.
    # The crystal route never creates one, so it round-trips as None there.
    '_prior_rng_state': None,
}

# Buffers live in their own sidecar file rather than inside each checkpoint:
# they are ~90% of the bytes (prior 138MB + anchor 139MB vs 74MB for model +
# optimizers), and re-serializing them at every 'best'/'running' save made
# checkpointing 48% of total train time. Sidecars are written only at eval
# cadence, so a resume restores buffers up to eval_period steps stale --
# acceptable because the anchor buffer's ema_loss priorities are already
# refreshed on that same cadence (buffers.anchor_buffer.refresh_every_n_evals).
#
# Each buffer dict inside the sidecar carries buffer.BUFFER_FORMAT_VERSION and the
# energy currency its rows are stored in (the per-row `lj_coeff` stamp). A sidecar
# written before that existed is REFUSED at restore, never silently re-stamped --
# migrate_buffer_sidecar.py is the one sanctioned path. See
# CrystalBuffer._refuse_unknown_currency for the failure this closes.
BUFFER_SUFFIX = '_buffers.pt'

# Every tag save() writes, used to strip a checkpoint's tag back to the
# run-level prefix when resolving its rolling sidecar. 'prior' is no longer
# written; it stays so an existing *_prior.pt still resolves its sidecar.
CHECKPOINT_TAGS = ('best', 'running', 'prior', 'thermalized', 'final',
                   'phase1_exit', 'phase2_exit', 'ff_calibrated')


def own_store_refusal(run, step, why) -> str:
    """The refusal of a launch that cannot take the crystal store entry of its own run's sidecar."""
    return (f"buffer sidecar of this run ({run!r}, step {step}) cannot be restored: {why}. Going on without it would "
            f"discard its grown rows. To start clean, move the sidecar away, or load the checkpoint under a new "
            f"run_name with buffers.fresh_on_switch")


def fresh_buffers_on_switch(modeller, checkpoint: dict, path: str) -> bool:
    """cfg:buffers.fresh_on_switch -- when THIS run loads ANOTHER run's checkpoint (the checkpoint's stored
    run_name differs from the modeller's), leave every resident buffer unrestored: the init path then seeds the
    prior and anchor buffers from the prior dataset the live energy function has just re-analysed
    (Modeller.init_prior_buffer_seed / init_anchor_buffer_seed) and the replay buffer refills from live rollouts,
    so every stored row is scored by this run's energy function. Built for a model switch under an unchanged
    problem identity (mlip_path is not part of it). A resume of the run's OWN checkpoint restores the sidecar as
    usual, so a requeue never resets the buffers. Off by default; prints when it fires."""
    buffers = getattr(modeller.args, 'buffers', None)
    if not bool(getattr(buffers, 'fresh_on_switch', False)):
        return False
    src = checkpoint.get('run_name')
    if src == modeller.run_name:
        return False
    print(f"buffers.fresh_on_switch: {path} belongs to run {src!r}, this run is {modeller.run_name!r} -- the buffer "
          f"sidecar is NOT restored; prior/anchor buffers seed from the re-analysed prior dataset and the replay "
          f"buffer refills from live rollouts, so every stored row is scored by this run's energy function", flush=True)
    return True


def best_energy_from_anchors_on_switch(modeller, checkpoint: dict, path: str) -> bool:
    """cfg:condition_log_z.best_energy_from_anchors_on_switch -- when THIS run loads ANOTHER run's checkpoint,
    replace the restored tracker's per-condition best energy with the restored anchor buffer's per-condition minimum
    (+inf, "never visited", for a condition that holds no anchor). A resume of the run's OWN checkpoint is untouched.
    Off by default; prints when it fires.

    Why it exists. The record is a running minimum over every energy the run has scored, forward-eval draws
    included, and nothing stores the row that set it. One absurd evaluation therefore stays for good, and everything
    measured as an excess above it breaks at once: the prior buffer's admission gate (energy < floor + ramp_floor)
    admits no noised anchor, its staleness rule (excess >= ramp_floor) expires every row it holds, and the
    excess-energy and nonthermal metrics read the offset. ppl_acr_n5 (2026-10-12) inherited -1.33e6 from its MLE
    seed against an anchor minimum of -45.3: zero top-ups from step 86,660 on, and the prior buffer fell from 2.9M
    rows to 43k in 37k steps.

    The anchor buffer's energies are the anchor currency (best_energy_phys). On a run that never supplied a separate
    physical energy the tracker's two minima are one stream and both are set; a run that has (lambda mixing) is
    refused, since its mixture minimum cannot be read off the anchors. An absent or empty anchor buffer is refused
    too (buffers.fresh_on_switch leaves none restored at this point): the key would otherwise do nothing, silently."""
    cfg = getattr(modeller.args, 'condition_log_z', None)
    if not bool(getattr(cfg, 'best_energy_from_anchors_on_switch', False)):
        return False
    src = checkpoint.get('run_name')
    if src == modeller.run_name:
        return False
    tracker = modeller.condition_log_z
    anchors = getattr(modeller, 'anchor_buffer', None)
    if anchors is None or len(anchors) == 0:
        raise ValueError(f"condition_log_z.best_energy_from_anchors_on_switch: {path} restored no anchor buffer to "
                         f"read the minima from (is buffers.fresh_on_switch also set?)")
    if not tracker.phys_is_alias:
        raise ValueError("condition_log_z.best_energy_from_anchors_on_switch: the restored tracker carries a physical "
                         "minimum separate from its mixture minimum; the anchors give only the physical one")
    ids = torch.arange(tracker.best_energy.numel(), dtype=torch.long)
    mins = _per_condition_min(anchors.condition_id.detach().cpu().long().flatten(),
                              anchors.energy.detach().cpu().float().flatten(), ids)
    old = tracker.best_energy.clone()
    tracker.best_energy = mins.clone()
    tracker.best_energy_phys = mins.clone()
    seen = torch.isfinite(old) & torch.isfinite(mins)
    moved = (old[seen] - mins[seen]).abs()
    print(f"condition_log_z.best_energy_from_anchors_on_switch: {path} belongs to run {src!r}, this run is "
          f"{modeller.run_name!r} -- best energy of {int(torch.isfinite(mins).sum())} condition(s) set to the anchor "
          f"buffer's minimum ({len(anchors)} anchors); "
          + (f"restored record {float(old[seen].min()):.6g} .. {float(old[seen].max()):.6g}, now "
             f"{float(mins[seen].min()):.6g} .. {float(mins[seen].max()):.6g}, largest move {float(moved.max()):.6g}"
             if bool(seen.any()) else "no condition had both a restored record and an anchor"), flush=True)
    return True


class Checkpointer:
    """
    Saves/restores a Modeller's full training state (model, the conformer
    route's EMA copy, optimizers, buffers, condition_log_z, and the MODELLER_STATE_DEFAULTS
    fields) to/from disk. Holds a reference to its owning Modeller rather
    than duplicating that state, since nearly every field on Modeller is
    part of what gets checkpointed.
    """

    def __init__(self, modeller):
        self.modeller = modeller
        self._read_only_announced = False
        # the step_ind a full load (load_full) restored in THIS process, else None. A
        # resumed leg re-runs that step, and `archive` must not rewrite its archive.
        self.resume_step: Optional[int] = None

    @property
    def read_only(self) -> bool:
        """Suppress every checkpoint WRITE; loading is unaffected.

        For smoke tests and refactor validation, which need to resume real state and run
        a few steps but must not touch the run's checkpoints. Without this, any launch
        reusing a run_name rewrites 'running' every 50 steps and 'final' unconditionally
        at exit -- including a launch that trains zero steps, since 'final' is outside
        the loop. Defaults False, so configs that don't set it behave exactly as before.
        """
        if not getattr(self.modeller.args, 'checkpoint_read_only', False):
            return False
        if not self._read_only_announced:
            print("checkpoint_read_only: checkpoint WRITES ARE SUPPRESSED for this run "
                  "(loading still active)")
            self._read_only_announced = True
        return True

    def get_state_dict(self):
        m = self.modeller
        state = {k: getattr(m, k) for k in MODELLER_STATE_DEFAULTS}
        # The adaptive clip tracker (grad_clip_guard.py) is state on an OBJECT
        # rather than a Modeller attribute, so it cannot ride the comprehension
        # above. Worth persisting because a rewind is exactly when the bar
        # matters most: a divergence-triggered reload that dropped it would
        # re-enter the warmup window with the run already unstable, i.e. the
        # guard would be absent precisely during the excursion.
        state['grad_guard'] = m.grad_guard.state_dict()
        return state

    def set_state_dict(self, state):
        m = self.modeller
        for k, default in MODELLER_STATE_DEFAULTS.items():
            setattr(m, k, state[k] if k in state else deepcopy(default))
        # Absent key, or a state version this build does not speak: the tracker
        # warms from scratch. load_state_dict DISCARDS rather than reinterprets,
        # for the same reason lr_ctrl's 'ver' does.
        m.grad_guard.load_state_dict(state.get('grad_guard'))
        self.reconcile_batch_size()

    def reconcile_batch_size(self):
        """
        Give the current config's batch settings authority over the restored ones.

        batch_size and its sizer bookkeeping are run state, so a resume used to
        inherit whatever size the checkpoint had grown to and ignore both
        `batch_size:` and `max_batch_size:` outright: the previous controller
        only ever moved the size UP and returned early once the restored size
        was already at or above the ceiling, so nothing ever pulled an
        oversized batch back down. Run p307hzip resumed with `batch_size: 1000`
        + `max_batch_size: 1000` and trained at 2831, silently.

        With growth ON the checkpoint's size is legitimate run state, so it is
        kept - just clamped to this config's ceiling. With growth OFF the
        config's batch_size is an explicit pin, so it is restored verbatim and
        the sizer's conclusion goes with it: it describes a selection this
        config no longer allows to happen, and carrying a stale cooldown or
        conclusion into a pinned run just leaves stale state lying around.
        The OOM path is untouched either way - handle_oom still cuts and re-arms
        them if the configured size turns out not to fit.
        """
        m = self.modeller
        args = m.args
        restored = m.batch_size
        grow = bool(getattr(args, 'grow_batch_size', True))
        if not grow:
            m.batch_size = int(args.batch_size)
            m.batch_size_last_grow = 0
            m.batch_sizer = None
            m.batch_size_cooldown_until = -1
        m.batch_size = min(m.batch_size, int(args.max_batch_size))
        if m.batch_size != restored:
            print(f"batch_size: checkpoint restored {restored} -> using {m.batch_size} "
                  f"(config batch_size={args.batch_size}, max_batch_size={args.max_batch_size}, "
                  f"grow_batch_size={grow})")
        # A PENDING S2 AUDIT DOES NOT SURVIVE A RESTART AS STORED. Its deadline
        # (`audit_at`, train.select_batch_size) is a WALL-CLOCK time one policy
        # window after the rung was selected, and what it judges is the occupancy
        # LIVED over that window. A requeued leg starts hours past the deadline with
        # an empty occupancy record, so the audit fired on its first step against
        # start-up idle, failed, and stood the batch down to the base rung -- and
        # `stood_down` is never re-probed, so the leg stayed there. mlepl_neh_lr2
        # (2026-10-06) resumed holding 1600, read "a full policy window at 1600
        # reads 0.7%", dropped to 1000 and ran 2.4 h at 51% occupancy until the
        # cluster cancelled it for low usage. Give the audit a window this process
        # has actually lived.
        sizer = getattr(m, 'batch_sizer', None)
        if isinstance(sizer, dict) and sizer.get('audit_at') is not None:
            policy_s = float(getattr(args, 'gpu_util_policy_window_s', 7200) or 7200)
            sizer['audit_at'] = float(m._now()) + policy_s
            print(f"batch: resumed holding {m.batch_size} with an occupancy audit pending -- "
                  f"re-armed for {policy_s:g} s from now (its stored deadline was a wall-clock "
                  f"time in the previous process)")

    def path_for(self, tag: str) -> str:
        m = self.modeller
        return f'{m.args.checkpoints_dir}/{m.run_name}_{m.problem_slug}_{tag}.pt'

    def find_matching(self, tag: str) -> Optional[str]:
        """
        Look for a checkpoint saved under this run_name/tag whose *stored*
        problem_def dict matches the current config's - not just its filename
        hash, since the slug format (and even the hash length) may change
        later. Refuses to reload (rather than raising) on any mismatch or on
        older checkpoints saved before problem_def existed, so a stale/renamed
        checkpoint never gets silently treated as a valid resume point.
        """
        m = self.modeller
        path = self.path_for(tag)
        if not os.path.exists(path):
            return None

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        stored_def = checkpoint.get('problem_def')
        if normalize_problem_def(stored_def) != normalize_problem_def(m.problem_def):
            print(f"Checkpoint {path} exists but its stored problem definition "
                  f"doesn't match the current config - starting fresh instead.\n"
                  f"{self.problem_mismatch_report(stored_def)}")
            return None

        return path

    @staticmethod
    def _drop_problem_keys(defn, keys):
        """A copy of the normalized problem_def `defn` without `keys`: a plain key drops that field, a dotted key
        'block.field' drops one field of a dict-valued block (energy_config.temperature). A non-dict passes through."""
        if not isinstance(defn, dict):
            return defn
        out = dict(defn)
        for key in keys:
            head, _, tail = key.partition('.')
            if not tail:
                out.pop(key, None)
            elif isinstance(out.get(head), dict):
                out[head] = {k: v for k, v in out[head].items() if k != tail}
        return out

    def problem_mismatch_report(self, stored_def: Optional[dict], ignore_keys: tuple = ()) -> str:
        """Readable stored-vs-current problem_def comparison, one line per differing field."""
        current_def = self._drop_problem_keys(normalize_problem_def(self.modeller.problem_def), ignore_keys)
        if stored_def is None:
            return ("  stored:  <none - checkpoint predates problem_def>\n"
                    f"  current: {current_def}")
        stored_def = self._drop_problem_keys(normalize_problem_def(stored_def), ignore_keys)
        lines = []
        for key in sorted(set(stored_def) | set(current_def)):
            stored_val = stored_def.get(key, '<missing>')
            current_val = current_def.get(key, '<missing>')
            if stored_val != current_val:
                lines.append(f"  {key}: stored={stored_val!r}  current={current_val!r}")
        return '\n'.join(lines) if lines else f"  stored:  {stored_def}\n  current: {current_def}"

    def assert_problem_match(self, checkpoint: dict, path: str, config_key: str,
                             ignore_keys: tuple = ()):
        """
        Hard-fail when an explicitly requested checkpoint (checkpoint_name)
        was trained on a different problem than the current
        config. Restoring across problems corrupts state that is only
        meaningful for the problem it was saved under - most destructively the
        buffers, whose resident batches carry the old energy function's keys
        (e.g. 'latent_harmonic') and later crash append_batch when fresh
        samples arrive stamped with the new energy's keys instead. Failing
        here turns that deep, inscrutable KeyError into an immediate,
        self-explanatory config error.

        ignore_keys: problem_def keys exempt from the comparison. The prior-
        model load passes the conditioning flags (mol_cond/temp_cond/vec_cond) here:
        they describe a model's INTERFACE rather than the target it samples,
        and the prior model is a sampling-only object rebuilt from its own
        stored gfn_config, so it need not share the live model's interface.
        """
        stored_def_cmp = self._drop_problem_keys(normalize_problem_def(checkpoint.get('problem_def')), ignore_keys)
        current_def_cmp = self._drop_problem_keys(normalize_problem_def(self.modeller.problem_def), ignore_keys)
        if stored_def_cmp != current_def_cmp:
            raise ValueError(
                f"{config_key} checkpoint {path} was trained on a different problem "
                f"than the current config solves - either point {config_key} at a "
                f"checkpoint for this problem, or change the config to match it.\n"
                f"{self.problem_mismatch_report(checkpoint.get('problem_def'), ignore_keys)}")

    def buffers_path(self, tag: Optional[str] = None) -> str:
        """
        Sidecar path holding the replay buffers. tag=None is the ROLLING
        sidecar (rewritten every eval, serves 'running'/'best' resume); a tag
        gives that snapshot its own frozen copy, so durable reload points keep
        the buffer state they were taken with even after the run moves on.
        """
        m = self.modeller
        stem = f'{m.args.checkpoints_dir}/{m.run_name}_{m.problem_slug}'
        return f'{stem}_{tag}{BUFFER_SUFFIX}' if tag else f'{stem}{BUFFER_SUFFIX}'

    @staticmethod
    def sidecar_candidates(checkpoint_path: str) -> list:
        """
        Buffer sidecars to try for a checkpoint, best pairing first: the
        snapshot's own frozen sidecar, then its run's rolling one. Derived
        purely from the filename, so pointing checkpoint_name at ANOTHER run's
        checkpoint automatically finds that run's buffers if they're on disk.
        """
        stem = checkpoint_path[:-3] if checkpoint_path.endswith('.pt') else checkpoint_path
        candidates = [stem + BUFFER_SUFFIX]
        for tag in CHECKPOINT_TAGS:
            if stem.endswith(f'_{tag}'):
                candidates.append(stem[:-(len(tag) + 1)] + BUFFER_SUFFIX)
                break
        else:
            # periodic archives (Checkpointer.archive) are tagged '_step<N>',
            # an open-ended family that can't live in CHECKPOINT_TAGS. They
            # only get their own frozen sidecar when archive_buffers is on, so
            # falling back to the run's rolling one is the normal path.
            trimmed = re.sub(r'_step\d+$', '', stem)
            if trimmed != stem:
                candidates.append(trimmed + BUFFER_SUFFIX)
        return candidates

    def buffer_state(self) -> dict:
        m = self.modeller
        return {
            'problem_def': m.problem_def,  # guards against pairing buffers with another problem's checkpoint
            'problem_hash': m.problem_hash,  # legacy guard - hash changes whenever the exclusion list grows, the def compare doesn't
            'run_name': getattr(m, 'run_name', None),  # the writer: restore_row_store refuses a mismatch on its own run
            'step_ind': m.step_ind,  # buffers lag their checkpoint by up to eval_period; this reports by how much
            'prior_buffer': self._store_state('prior_buffer'),
            'replay_buffer': m.replay_buffer.state_dict() if hasattr(m, 'replay_buffer') else None,
            'anchor_buffer': self._store_state('anchor_buffer'),
            # buffers.store, stamped with what its rows depend on (Modeller._store_stamp)
            'row_store': self._row_store_entry(),
            # the Boltzmann-slope readout's panel (Modeller._boltzmann_slope_panel), so the run's own resume reads the
            # same rows; None until it is built
            'boltzmann_panel': (getattr(m, '_boltzmann_panel', None)
                                if isinstance(getattr(m, '_boltzmann_panel', None), dict) else None),
        }

    def _row_store_entry(self):
        """The row store's sidecar entry, compact when its state_dict takes `compact`; None without a store."""
        m = self.modeller
        st = getattr(m, 'row_store', None)
        if st is None:
            return None
        entry = st.state_dict(m._store_stamp(), compact=True)
        entry['seed_keep'] = getattr(m, '_store_seed_mask', None)
        return entry

    def restore_row_store(self, store) -> bool:
        """Apply the sidecar entry restore_buffers parked onto `store`, which this launch has just seeded. Returns True
        when applied, False when none is parked or another run's entry does not fit (the store stays as seeded). An
        entry the run wrote itself that does not fit raises. After applying: log q written after the resumed step is
        undone (store.invalidate_q_after), and the generator is rekeyed on the resumed step unless the sidecar was
        written at that step."""
        m = self.modeller
        state = getattr(m, '_row_store_state', None)
        m._row_store_state = None
        if state is None:
            return False
        src_run, src_step = getattr(m, '_row_store_source', (None, None))
        why = store.state_mismatch(state, m._store_stamp())
        if why is not None:
            if src_run is not None and src_run == getattr(m, 'run_name', None):
                raise ValueError(own_store_refusal(src_run, src_step, why))
            print(f"row store in the buffer sidecar of run {src_run!r} ignored: {why}; keeping this launch's seed",
                  flush=True)
            return False
        store.apply_state(state)
        step = int(m.step_ind)
        if src_step is None or int(src_step) != step:
            store.reseed(store.base_seed, step)
        undone = store.invalidate_q_after(step)
        print(f"row store restored from the buffer sidecar (step {src_step}): {store.n_rows} rows, generation "
              f"{store.generation}, log q undone on {undone} rows written after step {step}", flush=True)
        return True

    #: the stores whose rows a launch's own seed rebuilds (train.py::Modeller.settle_seeded_store)
    SEEDED_STORES = ('prior_buffer', 'anchor_buffer')

    def _store_state(self, name):
        """A resident store's sidecar entry. COMPACT -- every column but the rows' graphs, keyed by the seed's row
        digest (`compact_seed`) -- while the store's rows are still exactly the rows this launch's seed built
        (Modeller._seed_digests, set by settle_seeded_store), so a resume rebuilds them from the prior file instead of
        reading GBs of graphs; FULL once anything changed its rows, for a store the seed never built, and for a
        store class whose state_dict cannot leave its rows out."""
        m = self.modeller
        b = getattr(m, name, None)
        if b is None:
            return None
        seed = (getattr(m, '_seed_digests', None) or {}).get(name)
        x = getattr(b, 'x', None)
        if (seed is not None and torch.is_tensor(x) and 'rows' in inspect.signature(b.state_dict).parameters
                and noised_prior.digest(x) == seed):
            state = b.state_dict(rows=False)
            state['x'] = None   # the seed rebuilds the rows and their latents; the digest checks them
            state['compact_seed'] = seed
            return state
        return b.state_dict()

    def apply_compact_store(self, name, state):
        """Restore a compact sidecar entry (_store_state) onto the store the seed just rebuilt: refused unless the
        rebuilt rows' digest is the one the records were written against (the prior file, its window or the seed
        changed); the store is then rebuilt through its class's own from_state_dict with the records and its rows."""
        m = self.modeller
        b = getattr(m, name, None)
        if b is None:
            raise ValueError(f"buffer sidecar: {name} was written compactly (its rows rebuilt from the prior at "
                             f"resume), but this launch built no {name} to rebuild them into")
        got = noised_prior.digest(b.x)
        if got != state['compact_seed']:
            raise ValueError(f"buffer sidecar: {name}'s records belong to rows with digest {state['compact_seed']}, "
                             f"but this launch's seed built rows with digest {got}: the prior file, its load "
                             f"window or the seed changed since the sidecar was written. Resume from a full "
                             f"sidecar or start the buffers fresh")
        full = dict(state, batch=b.batch, x=b.x)
        setattr(m, name, type(b).from_state_dict(full, device=m.buffer_device))
        if name == 'anchor_buffer' and hasattr(m, 'apply_anchor_buffer_policy'):
            m.apply_anchor_buffer_policy('sidecar restore')   # config owns the anchor policy, as on a full restore
        print(f"buffer sidecar: {name} rebuilt from this launch's seed ({int(b.x.shape[0])} rows) with its saved "
              f"per-row records", flush=True)

    #: the per-row columns a buffer's fingerprint sums (_buffer_fingerprint)
    FINGERPRINT_COLUMNS = ('x', 'y', 'ema_loss', 'select_counts', 'birth_step', 'birth_loss', 'ema_logw', 'traj')

    def _buffer_fingerprint(self):
        """A cheap identity of the resident stores: per store its row count and the sum of each per-row column
        (FINGERPRINT_COLUMNS), plus a live row store's `generation`; None (always write) for a row store without one.
        Any admission, eviction, draw count or loss update moves it."""
        m = self.modeller
        st = getattr(m, 'row_store', None)
        if st is not None and getattr(st, 'generation', None) is None:
            return None
        out = []
        for name in self.CURRENCY_CHECKED_BUFFERS:
            b = getattr(m, name, None)
            if b is None:
                out.append(None)
                continue
            x = getattr(b, 'x', None)
            if not torch.is_tensor(x):
                return None   # a store this cannot read: always write
            sums = []
            for col in self.FINGERPRINT_COLUMNS:
                t = getattr(b, col, None)
                if not torch.is_tensor(t):
                    sums.append(None)
                elif t.is_floating_point():
                    # NaN is a legitimate column value (birth_log_pf of a seeded or eval-born row) and NaN != NaN:
                    # the sum around the NaNs and their count
                    sums.append((float(t.double().nansum()), int(torch.isnan(t).sum())))
                else:
                    sums.append(float(t.double().sum()))
            out.append((name, int(x.shape[0]), tuple(sums)))
        return tuple(out) if st is None else (int(st.generation), tuple(out))

    def save_buffers(self, tag: Optional[str] = None):
        """Write the buffer sidecar. Called only at eval cadence -- see BUFFER_SUFFIX. The untagged eval-cadence
        write is skipped when no store changed since the last one (_buffer_fingerprint) and its file is on disk: an
        MLE stage never touches the static buffer it seeded, which is GBs per write. Tagged writes always happen."""
        if self.read_only:
            return
        path = self.buffers_path(tag)
        fp = self._buffer_fingerprint() if tag is None else None
        if fp is not None and fp == getattr(self, '_last_buffer_fp', None) and os.path.exists(path):
            if not getattr(self, '_announced_buffer_skip', False):
                self._announced_buffer_skip = True
                print(f"buffers unchanged since the last write: {path} kept (skipped until a store changes)",
                      flush=True)
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        before = self._file_id(path)
        atomic_save(self.buffer_state(), path)
        if tag is None:
            # atomic_save prints a failed write and returns: with a row store the next eval then writes again
            failed = getattr(self.modeller, 'row_store', None) is not None and self._file_id(path) == before
            self._last_buffer_fp = None if failed else fp

    @staticmethod
    def _file_id(path):
        """(inode, mtime in ns, size) of a file, or None when it is absent."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        return st.st_ino, st.st_mtime_ns, st.st_size

    def restore_buffers(self, state: dict, source: str):
        m = self.modeller
        # prefer the stored problem_def (compared normalized, so it survives
        # non-identity keys leaving the definition); older sidecars only carry
        # the hash, which is exact-def-sensitive but better than no guard
        stored_def = state.get('problem_def')
        if stored_def is not None:
            mismatched = normalize_problem_def(stored_def) != normalize_problem_def(m.problem_def)
        else:
            mismatched = state.get('problem_hash') is not None and state['problem_hash'] != m.problem_hash
        if mismatched:
            writer = state.get('run_name')
            if state.get('row_store') is not None and writer is not None and writer == getattr(m, 'run_name', None):
                raise ValueError(f"buffer sidecar {source} of this run was saved under a different problem: seeding "
                                 f"afresh would discard its grown rows. To start clean, move the sidecar away, or load "
                                 f"the checkpoint under a new run_name with buffers.fresh_on_switch")
            print(f"Buffer sidecar {source} was saved under a different problem "
                  f"- ignoring it, buffers start fresh")
            return
        if row_store_settings(getattr(m, 'args', None)) is not None and state.get('row_store') is None:
            held = [k for k in self.CURRENCY_CHECKED_BUFFERS if state.get(k) is not None]
            if held:
                raise ValueError(f"buffer sidecar {source} holds {held} and no crystal store, and this launch has "
                                 f"buffers.store on: those buffers would stay resident beside the store, outside its "
                                 f"RAM limit. Load it under a new run_name with buffers.fresh_on_switch, or with "
                                 f"load_weights_only")
        # all resident stores restore onto the configured buffer_device, and through the
        # modeller's OWN buffer classes. Hardcoding CrystalBuffer/AnchorBuffer here restored
        # a crystal store on a non-crystal route: the rows are fine, but the class carries
        # the graph hooks, so the next `add` went through CrystalBuffer._as_batch and died
        # on max_z_prime. Fresh runs never saw it because the seeded buffer is already full
        # and grow_prior_buffer returns early -- it only fires on RESUME.
        buf_cls = getattr(m, 'buffer_cls', CrystalBuffer)
        anchor_cls = getattr(m, 'anchor_buffer_cls', AnchorBuffer)
        # COMPACT entries (_store_state) hold records without rows: applied now onto a store that already exists (a
        # mid-run reload), else parked for train.py::Modeller.settle_seeded_store to apply once the seed rebuilds it
        for name in self.SEEDED_STORES:
            entry = state.get(name)
            if entry is not None and entry.get('compact_seed') is not None:
                state = dict(state, **{name: None})
                if getattr(m, name, None) is not None:
                    self.apply_compact_store(name, entry)
                else:
                    m.__dict__.setdefault('_compact_records', {})[name] = entry
        if state.get('prior_buffer') is not None:
            m.prior_buffer = buf_cls.from_state_dict(state['prior_buffer'], device=m.buffer_device)
        if state.get('replay_buffer') is not None:
            m.replay_buffer = buf_cls.from_state_dict(state['replay_buffer'], device=m.buffer_device)
        # parked for Modeller.init_row_store, which runs after the identifier registry and the energy reference exist
        m._row_store_state = state.get('row_store')
        m._row_store_source = (state.get('run_name'), state.get('step_ind'))
        # the readout keeps reading the panel its own run built; another run builds its own at its first evaluation
        if state.get('boltzmann_panel') is not None and state.get('run_name') == getattr(m, 'run_name', None):
            m._boltzmann_panel = state['boltzmann_panel']
        if state.get('anchor_buffer') is not None:
            m.anchor_buffer = anchor_cls.from_state_dict(state['anchor_buffer'], device=m.buffer_device)
            # CONFIG OWNS THE ANCHOR POLICY, the sidecar does not. from_state_dict
            # restores ema_loss verbatim, so without this a chained run silently
            # comes back with priorities no enabled writer can ever refresh.
            if hasattr(m, 'apply_anchor_buffer_policy'):
                m.apply_anchor_buffer_policy('sidecar restore')

    #: the resident stores whose rows are scored by reading a stored energy off
    #: the row (prebuilt_sample_to_reward), i.e. the ones whose currency matters
    CURRENCY_CHECKED_BUFFERS = ('prior_buffer', 'replay_buffer', 'anchor_buffer')

    def assert_buffer_currency(self, context: str = 'startup'):
        """Every resident crystal store must be stamped with THIS run's lj_coeff.

        The structural half (rows carry a stamp at all) is enforced by
        CrystalBuffer.from_state_dict the moment a sidecar is read. This is the
        VALUE half, and it cannot live there: the run's coefficient is only
        known after init_prior_dataset reads the prior's thermal_scaling_factor,
        which happens AFTER init_gfn has restored the sidecar. So train.py calls
        this once the buffers are seeded/restored and the coefficient is final.

        Delegates to MolecularCrystal.assert_lj_coeff_stamped, the same check
        every draw runs, so a sidecar from a differently calibrated run (an
        anchor set built at 1.0 restored into a 0.3636 run) is refused at
        startup rather than at the first backward draw. Skipped on routes that
        never read `.elj` (latent toys, conformers), mirroring generator_energy.
        """
        m = self.modeller
        ef = getattr(m, 'energy_function', None)
        check = getattr(ef, 'assert_lj_coeff_stamped', None)
        if check is None or not getattr(ef, 'is_crystal', False) \
                or getattr(ef, 'latent_energy', False):
            return
        for name in self.CURRENCY_CHECKED_BUFFERS:
            buf = getattr(m, name, None)
            if buf is None or len(buf) == 0:
                continue
            try:
                check(buf.batch)
            except (AttributeError, ValueError) as e:
                raise BufferCurrencyError(
                    f'{name} ({len(buf)} rows) is not in this run\'s energy currency '
                    f'(lj_coeff={float(ef.lj_coeff)!r}) at {context}: {e}') from e

    def load_buffers_for(self, checkpoint_path: str) -> bool:
        """
        Restore buffers for a checkpoint that carries none inline. Returns
        False when no sidecar is on disk -- not an error: the caller's normal
        init path then seeds/grows the buffers from scratch, exactly as it
        does for a fresh run.
        """
        for candidate in self.sidecar_candidates(checkpoint_path):
            if not os.path.exists(candidate):
                continue
            state = torch.load(candidate, map_location='cpu', weights_only=False)
            self.restore_buffers(state, candidate)
            print(f"Restored buffers from sidecar {candidate} (saved at step {state.get('step_ind')})")
            return True
        print(f"No buffer sidecar found for {checkpoint_path} - buffers will initialize fresh")
        return False

    def save(self, tag: str, with_buffers: bool = False):
        """
        tag: 'best' | 'running' | 'final', or a pre-transition
             snapshot tag from a stage's on_exit actions ('phase1_exit',
             'ff_calibrated', ... -- see protocol.StageProtocol._snapshot)

        Buffers are NOT written here - they live in a sidecar (see
        BUFFER_SUFFIX). with_buffers=True additionally freezes a tagged copy
        alongside this snapshot, for durable reload points that must keep the
        buffer state they were taken with; only pass it from call sites that
        run during evaluation.
        """
        if self.read_only:
            return
        m = self.modeller
        checkpoint = {
            'tag': tag,
            'run_name': m.run_name,
            'gfn_config': m.gfn_config,  # store once, reload from here
            'problem_def': m.problem_def,  # human-readable dict: energy function + prior this checkpoint solves
            'problem_hash': m.problem_hash,  # fast fingerprint of problem_def, also embedded in the filename
            'model_train': m.gfn_model.state_dict(),
            # P_B's frozen snapshot (Modeller.set_pb_freeze 'full'), or None. Lives
            # outside the model state_dict on purpose (not a submodule) and is
            # restored by every load path below, so a resumed leg scores P_B on
            # the SAME function rather than re-snapshotting a drifted trunk.
            'pb_frozen': m.gfn_model.pb_snapshot_state(),
            # the rollout length this checkpoint was trained at, read by offline
            # consumers that sample a checkpoint at its own T (build_prior_flow,
            # eval/paper1_results, eval/cond_panel). None on pre-2026-07-23 checkpoints and
            # on a policy without a trajectory length (model.kind nf without model.sde.T).
            'train_T': trajectory_T(m.args),
            'modeller_state': self.get_state_dict(),
            'metrics': m.metric_tracker.state_dict(),
            'optimizers': {k: opt.state_dict() for k, opt in m.optimizers.items()},
            'condition_log_z': m.condition_log_z.state_dict() if hasattr(m, 'condition_log_z') else None,
            # the per-condition energy reference the rewards were scored against (mode +
            # table); a full resume must match it (Modeller.init_energy_reference)
            'energy_reference': (m.energy_function.energy_reference_state()
                                 if hasattr(getattr(m, 'energy_function', None), 'energy_reference_state')
                                 else None),
        }
        if keeps_ema_copy(m):
            # the conformer route's EMA copy (ema_model, aliased to gfn_model under a null ema_decay)
            checkpoint['model_eval'] = m.ema_model.state_dict()
        path = self.path_for(tag)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        atomic_save(checkpoint, path)
        if with_buffers:
            self.save_buffers(tag)

    def archive(self, step: int):
        """
        Freeze a step-tagged copy of the CURRENT 'running' checkpoint every
        `archive_period` steps (config: archive_period, 0/null = off).

        Exists because the single-stage naive protocol fires no on_exit
        snapshots: 'running' and 'best' are the only saves, both are rewritten
        in place, and every run shares one run_name -- so relaunching destroys
        the previous run's state within minutes and any interesting
        intermediate (a mid-oscillation branch point) is unrecoverable.

        Model state rides `link`, so it is a hardlink off the bytes 'running'
        just wrote: no extra serialization, and atomic_save's swap-the-
        directory-entry semantics leave this archive on the old inode when
        'running' is next written. ~30MB per archive.

        Buffers are ~880MB (30x the model state) and are only written at eval
        cadence, so they are OPT-IN via archive_buffers -- and they are a real
        copy, not a link, because the sidecar this would link to may be many
        steps stale. Without them an archive still restores the model and
        optimizers; the buffer state (which is part of the dynamics -- a
        corrupted replay buffer acts as hidden state) starts fresh.

        THE STEP A LEG RESUMED AT IS NOT RE-ARCHIVED. A full load restores
        step_ind and the loop re-runs that step, so a leg resumed at an archive
        step reaches this call again at that step. Re-linking then replaced the
        archive with the rerun's weights and the rolling sidecar -- which the
        resume had filled from a LATER eval -- and a CK_STEP restart loaded that
        replaced pair (G0 gate, 2026-09-27: step50_buffers.pt held leg 1's
        step-75 rows). So at `resume_step`, whatever is on disk (model, frozen
        sidecar, or both) is kept and nothing is written: writing only the
        missing half would pair files from two legs. Kept files get a fresh
        mtime: the sbatch seeds that pick the newest step archive with `ls -t`
        ranked the old re-link as newest, and still do.
        Every other step overwrites as before: a CK_STEP fork has to replace
        the abandoned branch's later archives, or a later CK_STEP or
        newest-archive seed would load that branch.
        """
        if self.read_only:
            return
        m = self.modeller
        period = int(getattr(m.args, 'archive_period', 0) or 0)
        if period <= 0 or step <= 0 or step % period != 0:
            return None
        tag = f'step{step + int(getattr(m, "step_offset", 0) or 0)}'   # global step: a warm start's archives read on
        if step == self.resume_step:
            kept = [p for p in (self.path_for(tag), self.buffers_path(tag)) if os.path.exists(p)]
            if kept:
                for p in kept:
                    try:
                        os.utime(p)
                    except OSError as e:
                        print(f'archive: could not refresh the mtime of kept {p} ({e}); '
                              f'`ls -t` newest-archive seeds may not rank it newest')
                print(f'archive: {tag} is the step this leg resumed at and is already on disk '
                      f'({", ".join(os.path.basename(p) for p in kept)}) -- kept, not overwritten')
                return tag
        self.link('running', tag)
        if getattr(m.args, 'archive_buffers', True):
            # HARDLINK the rolling sidecar too, for the same reason: it is
            # written by atomic_save, so a later save swaps the directory entry
            # and leaves this archive on the old inode. Same disk footprint as a
            # copy (the old inode stops being freed) but no ~880MB of I/O inside
            # the train loop. Falls back to a real copy if the filesystem
            # refuses the link.
            src = self.buffers_path()          # the rolling sidecar
            dst = self.buffers_path(tag)
            if os.path.exists(src):
                try:
                    tmp = dst + '.tmp'
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    os.link(src, tmp)
                    os.replace(tmp, dst)
                except OSError:
                    try:
                        shutil.copy2(src, dst)
                    except OSError as e:
                        print(f'archive: buffer freeze failed for {tag}: {e}')
        print(f'archived checkpoint at step {step} -> {self.path_for(tag)}')
        return tag

    def link(self, src_tag: str, dst_tag: str):
        """
        Point dst_tag at src_tag's bytes via a hardlink instead of
        re-serializing them (saves are write-bandwidth-bound, so a copy would
        cost the same as a save; a link costs nothing). Safe because every
        writer goes through atomic_save's write-tmp-then-os.replace, which
        swaps the directory entry rather than mutating the file: a later
        src_tag save leaves dst_tag on the old inode, which is exactly the
        'best' snapshot semantics. Falls back to a real save if the
        filesystem refuses the link.
        """
        if self.read_only:
            return
        src, dst = self.path_for(src_tag), self.path_for(dst_tag)
        try:
            tmp = dst + '.tmp'
            if os.path.exists(tmp):
                os.remove(tmp)
            os.link(src, tmp)
            os.replace(tmp, dst)
        except OSError as e:
            print(f"Could not hardlink {dst_tag} -> {src_tag} ({e}) - saving a full copy instead")
            self.save(dst_tag)

    # gfn_config keys carrying NO parameters and no state_dict entries, so a
    # resumed run may take them from its own config. Everything else fixes the
    # weight layout and has to come from the file.
    RECONFIGURABLE_GFN_KEYS = ('t_scale_ratio', 't_scale_power', 't_scale_preserve_budget')
    # The force terms in the kernel means (GFN.init_force_drift) are this run's to decide as well:
    # an arm that adds one to a shared checkpoint would otherwise run as the control. Unlike the
    # keys above they can ADD parameters, the gates, which a checkpoint written without them does
    # not carry; _load_state and _grown_optimizer_state are what let such a checkpoint load.
    FORCE_DRIFT_GFN_KEYS = ('force_drift_fwd', 'force_drift_bwd', 'force_drift_learned',
                            'force_drift_max_sigma', 'force_drift_t_min', 'force_drift_differentiable')
    FORCE_GATE_KEY = '.force_gate.'

    def _build_gfn(self, config):
        """The model a checkpoint's gfn_config describes, BEFORE any weights load.

        A modeller that assembles its model after GFN(**gfn_config) -- the conformer
        route swaps in a set head and re-classes the GFN (ConformerModeller.
        _install_set_policy) -- supplies `gfn_from_config`, and gets to rebuild that
        architecture from what it stamped into gfn_config. Without the seam the strict
        load_state_dict below meets a flat GFN and fails on every set-head tensor, and a
        swap AFTER the load would discard the restored weights, EMA and Adam state.
        The crystal Modeller has no such method and takes the unchanged branch.
        """
        builder = getattr(self.modeller, 'gfn_from_config', None)
        return builder(config) if builder is not None else GFN(**config)

    def _gfn_config_from(self, checkpoint, frames_follow_run: bool = False):
        """
        The checkpoint's gfn_config, with RECONFIGURABLE_GFN_KEYS re-derived
        from THIS run's config. The in-rollout variance schedule adds no
        parameters and registers no persistent buffer, so it is behaviour
        rather than architecture -- a battery that varies it while resuming a
        shared snapshot would otherwise inherit the parent's value and run
        every arm as the control. Changes print, so a silent miss can't happen.

        `frames_follow_run` (the weights-only warm start): the NF frame flags
        (models/nf_policy.py::FRAME_ONLY_KEYS) are not compared either and
        follow this run's config. They change what a training step presents and
        no parameter's shape; the caller re-lays the condition scaler for them
        (_reframed_state). A full resume still compares them.
        """
        config = dict(checkpoint['gfn_config'])
        # the policy is architecture: the stored entry decides what loads, so a run asking for another kind or other flow
        # settings would silently train the checkpoint's (an SDE checkpoint carries no entry). Both sides are compared
        # resolved on the checkpoint's route and rows (a checkpoint written before gfn_config stored them resolved
        # loads when its resolution matches); the seed and PRECISION_ONLY_KEYS are not compared, and the precision
        # keys follow this run's config. This run's side carries the FD chart's group fields a fresh build adds
        # (train.py::Modeller._fd_policy_fields: fd_space_group, fd_space_groups), so they are compared like the rest
        entry = getattr(self.modeller, '_policy_entry', None)
        mine = (entry() if entry is not None else {}).get('policy')
        fd_fields = getattr(self.modeller, '_fd_policy_fields', None)
        if mine and fd_fields is not None:
            mine = dict(mine, **fd_fields(mine))
        asked = resolve_policy_entry(mine, config)
        held = resolve_policy_entry(config.get('policy'), config)

        follow = PRECISION_ONLY_KEYS + LOSS_ONLY_KEYS + (FRAME_ONLY_KEYS if frames_follow_run else ())

        def strip(e):
            if e is None:
                return None
            e = {k: v for k, v in e.items() if k != 'seed'}
            if 'nf' in e:
                e['nf'] = {k: v for k, v in e['nf'].items() if k not in follow}
            return e

        if strip(asked) != strip(held):
            raise ValueError(f"policy in this run's config {strip(asked) or {'kind': 'sde'}} differs from the "
                             f"checkpoint's {strip(held) or {'kind': 'sde'}}: the checkpoint's policy is what would "
                             f"load and train")
        if held is not None and 'nf' in held:
            nf = dict(held['nf'])
            for key in follow:
                if nf.get(key) != asked['nf'].get(key):
                    print(f"gfn_config['policy']['nf']['{key}']: checkpoint {nf.get(key)!r} -> "
                          f"config {asked['nf'].get(key)!r}")
                nf[key] = asked['nf'].get(key)
            config['policy'] = {**held, 'nf': nf}
        current = vars(self.modeller.args.model)
        # the state encoder is architecture: the checkpoint's is the one that loads, so a run asking for
        # another would silently train as the checkpoint's arm
        asked, held = int(current.get('state_atoms', 0) or 0), int(config.get('state_atoms', 0) or 0)
        if asked != held:
            raise ValueError(f"model.state_atoms is {asked} in this run's config and {held} in the checkpoint: "
                             f"the state encoder that reads atoms cannot be added to or taken from saved weights")
        asked, held = bool(current.get('state_force_input', False)), bool(config.get('state_force_input', False))
        if asked != held:
            raise ValueError(f"model.state_force_input is {asked} in this run's config and {held} in the checkpoint: "
                             f"the force input widens the state encoder's first layer, so it cannot be added to "
                             f"or taken from saved weights")
        # no parameters, but they decide what the state encoder is shown before those times: weights trained
        # on one window read another's inputs as a different arm, so a change is refused rather than taken
        for key in ('state_crystal_t_min', 'state_atoms_t_min'):
            asked, held = (float(c.get(key, 0.0) or 0.0) for c in (current, config))
            if asked != held:
                raise ValueError(f"model.{key} is {asked:g} in this run's config and {held:g} in the "
                                 f"checkpoint: the saved state encoder was trained on the checkpoint's window")
        if frames_follow_run and self._adds_temperature_column(config):
            want = int(self.modeller.get_conditioning_dim())
            print(f"gfn_config['conditions_dim']: checkpoint {config.get('conditions_dim')!r} -> config {want} "
                  f"(the temperature column, added with zero weight)")
            config['conditions_dim'] = want
        for key in self.RECONFIGURABLE_GFN_KEYS + self.FORCE_DRIFT_GFN_KEYS:
            if key not in current:
                continue
            if current[key] != config.get(key):
                print(f"gfn_config['{key}']: checkpoint {config.get(key)!r} -> config {current[key]!r}")
            config[key] = current[key]
        self._assert_dead_rows_match(config)
        return config

    @staticmethod
    def _reframed_state(model, state, what):
        """`state` with each NF conditional flow's condition scaler (c_mu, c_sd) re-laid for the model's frame
        columns, when the checkpoint was written with other frame flags (FRAME_ONLY_KEYS; a weights-only warm start).

        The scaler spans the condition vector [row id, frame columns, condition columns], and only the condition
        columns' entries are read (models/nf_policy.py::NFPolicy._context, c_mu[lead:]). So the id's and the
        condition columns' entries carry over unchanged and the frame columns get the identity (0, 1), entries
        nothing reads. Appending instead would shift every condition column's mean and scale. Every parameter has the
        same shape with or without frames, so nothing else moves. Prints what it re-laid."""
        from models.spline_flow import _ContextFlow
        out = None
        for name, mod in model.named_modules():
            if not isinstance(mod, _ContextFlow):
                continue
            for buf, fill in (('c_mu', 0.0), ('c_sd', 1.0)):
                key = f'{name}.{buf}' if name else buf
                if key not in state or state[key].shape == getattr(mod, buf).shape:
                    continue
                old, k_new = state[key], int(mod.frame_cols)
                k_old = k_new - (getattr(mod, buf).numel() - old.numel())
                if old.dim() != 1 or k_old not in (0, 4, 5) or old.numel() < 1 + k_old:
                    raise RuntimeError(f"{what}: {key} has {old.numel()} entries and this model's {getattr(mod, buf).numel()} "
                                       f"({k_new} frame columns); no frame setting accounts for the difference")
                pad = torch.full((k_new,), fill, dtype=old.dtype, device=old.device)
                out = dict(state) if out is None else out
                out[key] = torch.cat([old[:1], pad, old[1 + k_old:]])
                print(f"{what}: {key} re-laid for {k_new} frame columns (checkpoint {k_old}); the condition "
                      f"columns' scaling carries over")
        return state if out is None else out

    def _adds_temperature_column(self, config) -> bool:
        """Whether this run asks for exactly one condition column more than the checkpoint's gfn_config, the temperature
        column: temperature_conditioning on here, and the run's condition width the checkpoint's plus one."""
        m = self.modeller
        if not bool(getattr(m.args, 'temperature_conditioning', False)) or not hasattr(m, 'get_conditioning_dim'):
            return False
        return int(m.get_conditioning_dim()) == int(config.get('conditions_dim') or 0) + 1

    def _retempered_state(self, model, state, what):
        """`state` with the temperature condition column added to each NF conditional flow (models/spline_flow.py::
        _ContextFlow), when the model has one condition column more than the checkpoint (_adds_temperature_column).

        Temperature is the first condition column (MolecularCrystal.condition_samples), and the condition columns are
        the tail of the context and level heads' inputs (_ContextFlow.context_input), so the column goes in at the start
        of that tail: a zero column in `context.inp.weight` and `level.inp.weight`, so the loaded policy is the
        checkpoint's at every temperature. The condition scaler (c_mu, c_sd: [row id, frame columns, condition
        columns]) gets the column's mean and spread over energy_config.log_temperature_range, as a fresh fit over a
        uniform draw would. Shared modules are walked under every name, as the state dict lists them. Prints what it
        added."""
        from models.spline_flow import _ContextFlow
        rng = getattr(getattr(self.modeller.args, 'energy_config', None), 'log_temperature_range', None)
        out = None
        names, done = {}, set()
        for name, mod in model.named_modules(remove_duplicate=False):
            names.setdefault(id(mod), []).append(name)
        for name, mod in model.named_modules(remove_duplicate=False):
            if not isinstance(mod, _ContextFlow):
                continue
            pre = f'{name}.' if name else ''
            mu_key, sd_key = f'{pre}c_mu', f'{pre}c_sd'
            if mu_key not in state or state[mu_key].numel() + 1 != mod.c_mu.numel():
                continue
            if rng is None:
                raise RuntimeError(f"{what}: adding the temperature column needs energy_config.log_temperature_range")
            if not getattr(mod, 'atoms', False):
                raise NotImplementedError(f"{what}: adding the temperature column is implemented for the molecule "
                                          f"(atom-token) flow only, whose condition columns follow the row id")
            lo, hi = (float(v) for v in rng)
            at = 1 + int(mod.frame_cols)                         # the first condition column in the scaler
            n_cond_old = state[mu_key].numel() - at
            out = dict(state) if out is None else out
            for key, val in ((mu_key, 0.5 * (lo + hi)), (sd_key, max((hi - lo) / 12 ** 0.5, 1e-4))):
                old = state[key]
                out[key] = torch.cat([old[:at], old.new_tensor([val]), old[at:]])
            for head in ('context', 'level'):
                sub_mod = getattr(mod, head)
                # the head is registered under other names too (e.g. flow_model.head, conditions_embedding_model.context):
                # every name the state dict lists for it gets the column
                for key in [f'{n}.inp.weight' if n else 'inp.weight' for n in names[id(sub_mod)]]:
                    if key not in state or key in done:
                        continue
                    w = state[key]
                    want = sub_mod.inp.weight.shape[1]
                    if w.shape[1] + 1 != want:
                        raise RuntimeError(f"{what}: {key} has {w.shape[1]} inputs and this model's {want}; adding "
                                           f"the temperature column accounts for one")
                    col = w.shape[1] - n_cond_old                     # the start of the condition columns' tail
                    out[key] = torch.cat([w[:, :col], w.new_zeros(w.shape[0], 1), w[:, col:]], 1)
                    done.add(key)
            print(f"{what}: {pre or 'flow '}temperature column added (zero weight; scaler mean {0.5 * (lo + hi):.3g}, "
                  f"sd {(hi - lo) / 12 ** 0.5:.3g} in log10 kT)")
        return state if out is None else out

    def _load_state(self, model, state, what, in_place: bool = False):
        """load_state_dict, strict but for the force gates (FORCE_GATE_KEY) a checkpoint predates.

        A model built with a force term has gate parameters; a checkpoint written without one
        does not. They are left as constructed, at their starting values. `in_place` says the
        model was not just built (a rewind): its gates may have trained, so they are put back
        at their starting values, which is what the checkpoint's model had in effect. Any other
        missing key, and any unexpected key, raises as a strict load would.
        """
        result = model.load_state_dict(state, strict=False)
        gates = [k for k in result.missing_keys if self.FORCE_GATE_KEY in k]
        missing = [k for k in result.missing_keys if self.FORCE_GATE_KEY not in k]
        if missing or result.unexpected_keys:
            raise RuntimeError(f"{what}: the state dict does not fit the model. Missing keys: {missing}; "
                               f"unexpected keys: {list(result.unexpected_keys)}")
        if gates:
            if in_place:
                model.restart_force_gates()
            print(f"{what}: written without this model's force gates ({len(gates)} tensors); they "
                  f"{'go back to' if in_place else 'start at'} their configured starting values")

    @staticmethod
    def _grown_optimizer_state(opt, saved):
        """`saved` re-indexed for `opt` when opt's parameter groups are the saved ones with
        parameters ADDED AT THEIR ENDS: (state dict, number of added parameters), else None.

        The case is a model that gained parameters inside existing groups, the force gates,
        which sit after their policy's own parameters. `opt.load_state_dict(saved)` refuses a
        group of another size, and the fallback starts the whole optimizer fresh: every
        parameter loses its moments and its step count, and an arm that differs from its
        control by one small module would also differ by a reset optimizer. Here each saved
        parameter keeps its state under its new index and the added ones start without any,
        which Adam initialises at their first step. Refused (None) unless the group count is
        unchanged, no group shrank, and every carried parameter has the saved shape.
        """
        live, old = opt.param_groups, saved['param_groups']
        if len(live) != len(old):
            return None
        sizes = [(len(g['params']), len(o['params'])) for g, o in zip(live, old)]
        if all(n == o for n, o in sizes) or any(n < o for n, o in sizes):
            return None
        fresh = opt.state_dict()['param_groups']
        state = {}
        for g_live, g_new, g_old in zip(live, fresh, old):
            for p, new_id, old_id in zip(g_live['params'], g_new['params'], g_old['params']):
                entry = saved['state'].get(old_id)
                if entry is None:
                    continue
                shaped = next((v for v in entry.values() if torch.is_tensor(v) and v.dim() > 0), None)
                if shaped is not None and shaped.shape != p.shape:
                    return None
                state[new_id] = entry
        groups = [dict(g_old, params=g_new['params']) for g_new, g_old in zip(fresh, old)]
        return {'state': state, 'param_groups': groups}, sum(n - o for n, o in sizes)

    def _assert_dead_rows_match(self, config):
        """
        Dead latent rows fix the weight layout (they shrink `expanded_dim`), so they can
        NOT be reconfigured on resume -- the checkpoint's value has to win, like every
        other architectural key. That makes a stale value dangerous rather than merely
        wrong: a pre-change monoclinic checkpoint carries no `dead_latent_rows` at all, so
        the model would silently rebuild with the rows LIVE while this run's startup probe
        prints reassurance that they are held. log Z would quietly revert to the old
        scale with nothing in the log to say so. See decisions.md D33.

        So compare and refuse. Orphaning affected checkpoints is the accepted cost.
        Triclinic and toys resolve to (), which matches a pre-change checkpoint's absent
        key, so existing sg-1/sg-2 resumes are unaffected.

        On the FD chart the policy is built with no dead rows (Modeller.init_gfn passes
        dead_latent_rows None there: the chart drops a group's fixed rows itself), so an FD
        run wants () whatever its space groups resolve to on the latent chart.
        """
        m = self.modeller
        if not hasattr(m, '_resolve_dead_latent_rows'):
            return
        try:
            fd = getattr(getattr(m, 'energy_function', None), 'crystal_coordinates', 'latent') == 'fd'
            wanted = () if fd else (m._resolve_dead_latent_rows(quiet=True) or ())
        except Exception:
            return  # resolution itself is broken; let the normal path report it
        stored = tuple(config.get('dead_latent_rows') or ())
        if tuple(wanted) == stored:
            return
        raise ValueError(
            f"dead_latent_rows mismatch: this run resolves {tuple(wanted)} for "
            f"space_groups={list(m.args.space_groups)}, but the checkpoint was built with "
            f"{stored}. These fix the policy input width (expanded_dim), so they cannot be "
            f"swapped on resume. The checkpoint predates the dead-row work (decisions.md "
            f"D33) or was trained on a different crystal system -- retrain from scratch.")

    def load_full(self, path, load_opt_state: bool = True):
        m = self.modeller
        checkpoint = torch.load(path, map_location=m.device, weights_only=False)
        self.assert_problem_match(checkpoint, path, 'checkpoint_name')
        m.gfn_config = self._gfn_config_from(checkpoint)
        m.gfn_model = self._build_gfn(m.gfn_config).to(m.device)
        self._load_state(m.gfn_model, checkpoint['model_train'], f'{path} (model_train)')
        self._install_ema(checkpoint, path)
        self._restore_pb_snapshot(checkpoint)

        m.gfn_model.train()

        if 'stage' not in checkpoint['modeller_state']:
            raise ValueError(
                f"checkpoint {path} predates the stage protocol (it stores a numeric "
                f"'phase' instead of a stage name) and cannot be resumed -- no backward "
                f"compatibility is kept. Start fresh, or load just its weights with "
                f"load_weights_only.")
        self.set_state_dict(checkpoint['modeller_state'])
        self.resume_step = int(m.step_ind)
        m.metric_tracker.load_state_dict(checkpoint.get('metrics', {}))

        # checkpoints written before buffers moved to a sidecar carry them
        # inline; prefer those, since they pair exactly with these weights
        if fresh_buffers_on_switch(m, checkpoint, path):
            pass    # said so above; the init path seeds every store afresh
        elif any(checkpoint.get(k) is not None
                 for k in ('prior_buffer', 'replay_buffer', 'anchor_buffer')):
            self.restore_buffers(checkpoint, path)
        else:
            self.load_buffers_for(path)

        if checkpoint.get('condition_log_z') is not None:
            m.condition_log_z = ConditionLogZTracker.from_state_dict(
                checkpoint['condition_log_z'], current_step=m.step_ind)
            best_energy_from_anchors_on_switch(m, checkpoint, path)
        # judged against the live config by Modeller.init_energy_reference once the
        # identifier registry exists; absent (older checkpoints) = trained unreferenced
        m._resumed_energy_reference = checkpoint.get('energy_reference') or {'mode': None, 'table': None}

        # NB no override_loss_coeffs any more: checkpoints carry only the stage
        # NAME, and live coefficients are always re-derived from the current
        # config's protocol block -- the config owns behavior unconditionally.

        if load_opt_state:
            m.init_schedulers_optimizers()
            self.load_optimizer_state(checkpoint)

            if getattr(m.args, 'override_learning_rates', False):
                # overwrite just the numeric LR the checkpoint's optimizer state
                # restored with this config's target rate - LRController's
                # own state (lr_ctrl: stage_start_step/scale) is untouched, so its
                # schedule re-stamps every LR on its very next tick
                # (which will itself overwrite whatever is set here -- this only
                # matters for the steps between restore and that next tick).
                # Adam's momentum buffers (exp_avg/exp_avg_sq) are also left as-is.
                target_lrs = {'fwd': m.args.lr_policy, 'bwd': m.args.lr_back,
                              'replay': m.args.lr_replay, 'fused': m.args.lr_fused,
                              'flow': m.args.lr_flow}
                for key, opt in m.optimizers.items():
                    n_groups = len(opt.param_groups)
                    for gi, group in enumerate(opt.param_groups):
                        if key == 'fused' and gi == n_groups - 1:
                            # the fused optimizer's trailing group is the flow (Z
                            # head), parked there at its own decoupled lr_flow (see
                            # init_schedulers_optimizers) -- stamping it with
                            # lr_fused silently cut Z training ~1000x on resume
                            # (run bcwvhdaq), the same policy/Z decoupling break
                            # as _apply_lrs's v1 bug (ylmtpqjy)
                            group['lr'] = m.args.lr_flow
                        else:
                            group['lr'] = target_lrs[key]

    #: The ONLY problem_def keys `warm_start_ignore_problem_keys` may exempt, and
    #: only on the weights-only path. Deliberately few: a warm start across a
    #: different ENERGY FUNCTION or space group would load weights that mean
    #: nothing for the new target, and no config should be able to ask for that.
    #: prior_path is exemptible because it names the MLE sample, which
    #: `load_weights_only` does not restore any part of -- see below.
    #: energy_config.temperature (one field of the energy_config block, nothing else
    #: in it) is exemptible because the MLE stage is reward-free, so its weights do
    #: not depend on T, and the weights-only path restores no stored reward, buffer or
    #: Z level: a TB stage entered on them re-levels log Z (bootstrap_z) at its own T.
    #: temp_cond and energy_config.log_temperature_range are exemptible so a run with
    #: temperature conditioning can start from weights trained without it: the load adds
    #: the temperature column with zero weight (_retempered_state), so the loaded policy
    #: is the checkpoint's at every temperature until it trains.
    WARM_START_EXEMPTIBLE = ('prior_path', 'energy_config.temperature', 'temp_cond',
                             'energy_config.log_temperature_range')

    def _warm_start_ignore_keys(self):
        """Problem-def keys this run is permitted to warm-start ACROSS.

        Empty unless the config asks, and every entry must sit in
        WARM_START_EXEMPTIBLE. The motivating case: a prior rebuilt to hold more
        noised rows around the SAME anchors at the SAME calibrated noise range is
        the same problem, but `prior_path` hashes the FILENAME, so the identity
        moves when the distribution has not.

        Whoever sets this is asserting the two priors describe one target. Nothing
        here can check that, so it prints what it exempted -- a silent exemption
        is how a genuinely different prior would slip through wearing an old
        model's weights.
        """
        wanted = tuple(getattr(self.modeller.args, 'warm_start_ignore_problem_keys', None) or ())
        bad = [k for k in wanted if k not in self.WARM_START_EXEMPTIBLE]
        if bad:
            raise ValueError(
                f"warm_start_ignore_problem_keys={list(wanted)} names {bad}, which "
                f"may not be exempted. Only {list(self.WARM_START_EXEMPTIBLE)} can be: "
                f"the rest change what the target IS, so weights trained under the old "
                f"value do not transfer and the guard is the only thing that says so.")
        return wanted

    def load_weights_only(self, path):
        """
        Warm-start from a checkpoint's model weights alone: rebuilds the GFN
        from the checkpoint's stored gfn_config (bar RECONFIGURABLE_GFN_KEYS,
        which follow this run's config) and loads the policy weights (and the
        conformer route's EMA copy, _install_ema), but
        restores nothing else - optimizers, schedulers, buffers, metrics,
        condition_log_z, and every MODELLER_STATE_DEFAULTS field are left for
        the caller to initialize fresh (phase 1, step 0, LR warmup from
        scratch).

        WHY THIS PATH MAY EXEMPT A PROBLEM KEY AND `load_full` MAY NOT.
        assert_problem_match exists because restoring across problems corrupts
        state meaningful only under the old one -- most destructively the buffers,
        whose rows carry the old energy's keys. This path restores NO buffers, no
        optimizer and no step count; it carries weights and the P_B snapshot and
        nothing else. So `warm_start_ignore_problem_keys` is honoured here and
        deliberately ignored by load_full, which does restore buffers.
        """
        m = self.modeller
        checkpoint = torch.load(path, map_location=m.device, weights_only=False)
        ignore = self._warm_start_ignore_keys()
        if ignore:
            print(f"checkpoint: warm start EXEMPTING {list(ignore)} from the problem "
                  f"match for weights-only load of {path} -- asserting these describe "
                  f"the same target. Buffers, optimizers and step count start fresh.")
        self.assert_problem_match(checkpoint, path, 'checkpoint_name', ignore_keys=ignore)
        m.gfn_config = self._gfn_config_from(checkpoint, frames_follow_run=True)
        m.gfn_model = self._build_gfn(m.gfn_config).to(m.device)
        self._load_state(m.gfn_model, self._reframed_state(
            m.gfn_model, self._retempered_state(m.gfn_model, checkpoint['model_train'], path), path),
                         f'{path} (model_train)')
        self._install_ema(checkpoint, path)
        self._restore_pb_snapshot(checkpoint)

        m.gfn_model.train()
        src = checkpoint.get('modeller_state') or {}
        m.step_offset = int(src.get('step_ind', 0) or 0) + int(src.get('step_offset', 0) or 0)
        print(f"checkpoint: weights from global step {m.step_offset}; this run's curves and step archives continue "
              f"from there (global_step = step + {m.step_offset}), its training state starts at step 0")
        # read by the protocol's `skip_if: weights_loaded` (StageProtocol.begin): the
        # phase-2 entry from a weights-only warm start. Set only here, after the load
        # succeeded, so a failed load cannot skip a stage.
        m.weights_only_loaded = True

    def load_optimizer_state(self, checkpoint, strict: bool = False):
        """Restore every optimizer's state, including Adam's step counter.

        `strict` TURNS THE TWO FALLBACKS BELOW INTO ERRORS. Without it, "no
        saved state for this optimizer" and "the param groups no longer line up"
        print and continue with that optimizer fresh, which resets Adam's step
        counter (`opt.load_state_dict` is what carries `state[p]['step']`).
        """
        m = self.modeller
        saved_optimizers = checkpoint['optimizers']
        for key, opt in m.optimizers.items():
            if key not in saved_optimizers:
                if strict:
                    raise KeyError(
                        f"no saved optimizer state for '{key}'. Starting it fresh would "
                        f"reset Adam's step counter.")
                print(f"No saved optimizer state for '{key}' - starting it fresh")
                continue
            saved = saved_optimizers[key]
            grown = self._grown_optimizer_state(opt, saved)
            if grown is not None:
                saved, added = grown
                print(f"optimizer '{key}': {added} parameters are new in this model and start without "
                      f"optimizer state; every other parameter keeps its moments and step count")
            try:
                opt.load_state_dict(saved)
            except (ValueError, RuntimeError) as e:
                # e.g. checkpoint predates flow params folding into the policy
                # optimizers, so param group counts no longer line up
                if strict:
                    raise RuntimeError(
                        f"could not restore optimizer state for '{key}': {e}. Starting "
                        f"it fresh would reset Adam's step counter.") from e
                print(f"Could not restore optimizer state for '{key}' ({e}) - starting it fresh")

    def _install_ema(self, checkpoint, path):
        """ema_model, after gfn_model has loaded. On the conformer route (utils.keeps_ema_copy)
        a separate copy holding the checkpoint's `model_eval`, in eval mode; on every other
        route gfn_model itself, and a stored `model_eval` is not read."""
        m = self.modeller
        if not keeps_ema_copy(m):
            m.ema_model = m.gfn_model
            return
        m.ema_model = deepcopy(m.gfn_model)
        self._load_state(m.ema_model, checkpoint['model_eval'], f'{path} (model_eval)')
        m.ema_model.eval()

    def _restore_pb_snapshot(self, checkpoint):
        """A checkpoint written under a full P_B freeze carries the snapshot;
        one written trainable carries None, which also LIFTS a freeze on a
        model that had one (a rewind to a pre-freeze checkpoint must not keep
        the later snapshot). Pre-snapshot checkpoints have no key: no-op."""
        if 'pb_frozen' not in checkpoint:
            return
        state = checkpoint['pb_frozen']
        m = self.modeller
        if state is None:
            if m.gfn_model.pb_frozen:
                m.set_pb_freeze(None)
            return
        m.set_pb_freeze('full', source_state=state)

    def load_model_only(self, path, load_optimizers: bool = False):
        m = self.modeller
        checkpoint = torch.load(path, map_location=m.device, weights_only=False)
        self._load_state(m.gfn_model, checkpoint['model_train'], f'{path} (model_train)', in_place=True)
        if m.ema_model is not m.gfn_model:
            # the conformer route's separate EMA copy (utils.keeps_ema_copy)
            self._load_state(m.ema_model, checkpoint['model_eval'], f'{path} (model_eval)', in_place=True)
            m.ema_model.eval()
        self._restore_pb_snapshot(checkpoint)
        m.gfn_model.train()
        if load_optimizers:
            self.load_optimizer_state(checkpoint)
