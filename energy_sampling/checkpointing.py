import glob
import os
import re
import shutil
from copy import deepcopy
from typing import Optional

import torch

from energy_sampling.buffer import CrystalBuffer, AnchorBuffer, ConditionLogZTracker
from energy_sampling.protocol import fresh_stage_ctrl
from energy_sampling.utils import atomic_save, normalize_problem_def
from models import GFN

MODELLER_STATE_DEFAULTS = {
    'step_ind': 0,
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
    # LRController state (see controller.py): a missing/mismatched 'ver' means
    # "not yet attached" -- the controller builds a fresh state on its first
    # tick, so disabled runs carry this dict around inertly
    'lr_ctrl': {'phase_seen': None, 'scale': None},
}

# Buffers live in their own sidecar file rather than inside each checkpoint:
# they are ~90% of the bytes (prior 138MB + anchor 139MB vs 74MB for model +
# optimizers), and re-serializing them at every 'best'/'running' save made
# checkpointing 48% of total train time. Sidecars are written only at eval
# cadence, so a resume restores buffers up to eval_period steps stale --
# acceptable because the anchor buffer's ema_loss priorities are already
# refreshed on that same cadence (buffers.anchor_buffer.refresh_every_n_evals).
BUFFER_SUFFIX = '_buffers.pt'

# Every tag save() writes, used to strip a checkpoint's tag back to the
# run-level prefix when resolving its rolling sidecar.
CHECKPOINT_TAGS = ('best', 'running', 'prior', 'thermalized', 'final',
                   'phase1_exit', 'phase2_exit', 'ff_calibrated')


class Checkpointer:
    """
    Saves/restores a Modeller's full training state (model, EMA model,
    optimizers, buffers, condition_log_z, and the MODELLER_STATE_DEFAULTS
    fields) to/from disk. Holds a reference to its owning Modeller rather
    than duplicating that state, since nearly every field on Modeller is
    part of what gets checkpointed.
    """

    def __init__(self, modeller):
        self.modeller = modeller
        self._read_only_announced = False

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

    def find_shared_prior(self) -> Optional[str]:
        """
        Any *_prior.pt in checkpoints_dir - regardless of which run saved it -
        whose stored problem_def matches the current config's exactly (the
        same strict comparison as find_matching: no conditioning exemption).
        problem_def carries only the target identity (energy function/config,
        prior_path, space groups, z_primes, conditioning flags), never
        architecture/T/lr, so one pretrained prior matches every run on the
        same problem. Candidates are tried newest-mtime-first; ties in
        content are the caller's responsibility to avoid (one shared prior
        per problem per checkpoints_dir). The run's own path_for('prior') is
        skipped here since find_matching already covers it.
        """
        m = self.modeller
        own = os.path.abspath(self.path_for('prior'))
        pattern = os.path.join(m.args.checkpoints_dir, '*_prior.pt')
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for path in candidates:
            if os.path.abspath(path) == own:
                continue
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"find_shared_prior: skipping unreadable candidate {path}: {e}")
                continue
            stored_def = checkpoint.get('problem_def')
            if normalize_problem_def(stored_def) == normalize_problem_def(m.problem_def):
                return path
        return None

    def problem_mismatch_report(self, stored_def: Optional[dict], ignore_keys: tuple = ()) -> str:
        """Readable stored-vs-current problem_def comparison, one line per differing field."""
        current_def = normalize_problem_def(self.modeller.problem_def)
        if stored_def is None:
            return ("  stored:  <none - checkpoint predates problem_def>\n"
                    f"  current: {current_def}")
        stored_def = normalize_problem_def(stored_def)
        lines = []
        for key in sorted((set(stored_def) | set(current_def)) - set(ignore_keys)):
            stored_val = stored_def.get(key, '<missing>')
            current_val = current_def.get(key, '<missing>')
            if stored_val != current_val:
                lines.append(f"  {key}: stored={stored_val!r}  current={current_val!r}")
        return '\n'.join(lines) if lines else f"  stored:  {stored_def}\n  current: {current_def}"

    def assert_problem_match(self, checkpoint: dict, path: str, config_key: str,
                             ignore_keys: tuple = ()):
        """
        Hard-fail when an explicitly requested checkpoint (checkpoint_name /
        prior_model_name) was trained on a different problem than the current
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
        stored_def = normalize_problem_def(checkpoint.get('problem_def'))
        current_def = normalize_problem_def(self.modeller.problem_def)
        if isinstance(stored_def, dict):
            stored_def_cmp = {k: v for k, v in stored_def.items() if k not in ignore_keys}
        else:
            stored_def_cmp = stored_def
        current_def_cmp = {k: v for k, v in current_def.items() if k not in ignore_keys}
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
            'step_ind': m.step_ind,  # buffers lag their checkpoint by up to eval_period; this reports by how much
            'prior_buffer': m.prior_buffer.state_dict() if hasattr(m, 'prior_buffer') else None,
            'replay_buffer': m.replay_buffer.state_dict() if hasattr(m, 'replay_buffer') else None,
            'anchor_buffer': m.anchor_buffer.state_dict() if hasattr(m, 'anchor_buffer') else None,
        }

    def save_buffers(self, tag: Optional[str] = None):
        """Write the buffer sidecar. Called only at eval cadence -- see BUFFER_SUFFIX."""
        if self.read_only:
            return
        atomic_save(self.buffer_state(), self.buffers_path(tag))

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
            print(f"Buffer sidecar {source} was saved under a different problem "
                  f"- ignoring it, buffers start fresh")
            return
        # all resident stores restore onto the configured buffer_device, and through the
        # modeller's OWN buffer classes. Hardcoding CrystalBuffer/AnchorBuffer here restored
        # a crystal store on a non-crystal route: the rows are fine, but the class carries
        # the graph hooks, so the next `add` went through CrystalBuffer._as_batch and died
        # on max_z_prime. Fresh runs never saw it because the seeded buffer is already full
        # and grow_prior_buffer returns early -- it only fires on RESUME.
        buf_cls = getattr(m, 'buffer_cls', CrystalBuffer)
        anchor_cls = getattr(m, 'anchor_buffer_cls', AnchorBuffer)
        if state.get('prior_buffer') is not None:
            m.prior_buffer = buf_cls.from_state_dict(state['prior_buffer'], device=m.buffer_device)
        if state.get('replay_buffer') is not None:
            m.replay_buffer = buf_cls.from_state_dict(state['replay_buffer'], device=m.buffer_device)
        if state.get('anchor_buffer') is not None:
            m.anchor_buffer = anchor_cls.from_state_dict(state['anchor_buffer'], device=m.buffer_device)

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
        tag: 'best' | 'running' | 'prior' | 'final', or a pre-transition
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
            'model_eval': m.ema_model.state_dict(),
            # the rollout length this checkpoint was trained at. A prior reused
            # by another run (prior_model_name) must be SAMPLED at
            # its own training T, not the consumer's eval_T -- a T=10 prior fed
            # into a T=100 run is a 10x discretization mismatch (see
            # sample_from_prior). None on pre-2026-07-23 checkpoints.
            'train_T': getattr(getattr(m.args, 'integrator', None), 'T', None),
            'modeller_state': self.get_state_dict(),
            'metrics': m.metric_tracker.state_dict(),
            'optimizers': {k: opt.state_dict() for k, opt in m.optimizers.items()},
            'condition_log_z': m.condition_log_z.state_dict() if hasattr(m, 'condition_log_z') else None,
        }
        path = self.path_for(tag)
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
        """
        if self.read_only:
            return
        m = self.modeller
        period = int(getattr(m.args, 'archive_period', 0) or 0)
        if period <= 0 or step <= 0 or step % period != 0:
            return None
        tag = f'step{step}'
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

    def _gfn_config_from(self, checkpoint):
        """
        The checkpoint's gfn_config, with RECONFIGURABLE_GFN_KEYS re-derived
        from THIS run's config. The in-rollout variance schedule adds no
        parameters and registers no persistent buffer, so it is behaviour
        rather than architecture -- a battery that varies it while resuming a
        shared snapshot would otherwise inherit the parent's value and run
        every arm as the control. Changes print, so a silent miss can't happen.
        """
        config = dict(checkpoint['gfn_config'])
        current = vars(self.modeller.args.model)
        for key in self.RECONFIGURABLE_GFN_KEYS:
            if key not in current:
                continue
            if current[key] != config.get(key):
                print(f"gfn_config['{key}']: checkpoint {config.get(key)!r} -> config {current[key]!r}")
            config[key] = current[key]
        self._assert_dead_rows_match(config)
        return config

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
        """
        m = self.modeller
        if not hasattr(m, '_resolve_dead_latent_rows'):
            return
        try:
            wanted = m._resolve_dead_latent_rows(quiet=True) or ()
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
            f"D33) or was trained on a different crystal system -- retrain from scratch, or "
            f"set model.hold_dead_latent_rows: false to reproduce the old architecture "
            f"exactly (which also restores the D33 defect).")

    def load_full(self, path, load_opt_state: bool = True):
        m = self.modeller
        checkpoint = torch.load(path, map_location=m.device, weights_only=False)
        self.assert_problem_match(checkpoint, path, 'checkpoint_name')
        m.gfn_config = self._gfn_config_from(checkpoint)
        m.gfn_model = GFN(**m.gfn_config).to(m.device)
        m.gfn_model.load_state_dict(checkpoint['model_train'])
        m.ema_model = deepcopy(m.gfn_model)
        m.ema_model.load_state_dict(checkpoint['model_eval'])

        m.gfn_model.train()
        m.ema_model.eval()

        if 'stage' not in checkpoint['modeller_state']:
            raise ValueError(
                f"checkpoint {path} predates the stage protocol (it stores a numeric "
                f"'phase' instead of a stage name) and cannot be resumed -- no backward "
                f"compatibility is kept. Start fresh, or load just its weights with "
                f"load_weights_only.")
        self.set_state_dict(checkpoint['modeller_state'])
        m.metric_tracker.load_state_dict(checkpoint.get('metrics', {}))

        # checkpoints written before buffers moved to a sidecar carry them
        # inline; prefer those, since they pair exactly with these weights
        if any(checkpoint.get(k) is not None
               for k in ('prior_buffer', 'replay_buffer', 'anchor_buffer')):
            self.restore_buffers(checkpoint, path)
        else:
            self.load_buffers_for(path)

        if checkpoint.get('condition_log_z') is not None:
            m.condition_log_z = ConditionLogZTracker.from_state_dict(
                checkpoint['condition_log_z'], current_step=m.step_ind)

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

    def load_weights_only(self, path):
        """
        Warm-start from a checkpoint's model weights alone: rebuilds the GFN
        from the checkpoint's stored gfn_config (bar RECONFIGURABLE_GFN_KEYS,
        which follow this run's config) and loads the train/EMA weights, but
        restores nothing else - optimizers, schedulers, buffers, metrics,
        condition_log_z, and every MODELLER_STATE_DEFAULTS field are left for
        the caller to initialize fresh (phase 1, step 0, LR warmup from
        scratch).
        """
        m = self.modeller
        checkpoint = torch.load(path, map_location=m.device, weights_only=False)
        self.assert_problem_match(checkpoint, path, 'checkpoint_name')
        m.gfn_config = self._gfn_config_from(checkpoint)
        m.gfn_model = GFN(**m.gfn_config).to(m.device)
        m.gfn_model.load_state_dict(checkpoint['model_train'])
        m.ema_model = deepcopy(m.gfn_model)
        m.ema_model.load_state_dict(checkpoint['model_eval'])

        m.gfn_model.train()
        m.ema_model.eval()

    def load_optimizer_state(self, checkpoint):
        m = self.modeller
        saved_optimizers = checkpoint['optimizers']
        for key, opt in m.optimizers.items():
            if key not in saved_optimizers:
                print(f"No saved optimizer state for '{key}' - starting it fresh")
                continue
            try:
                opt.load_state_dict(saved_optimizers[key])
            except (ValueError, RuntimeError) as e:
                # e.g. checkpoint predates flow params folding into the policy
                # optimizers, so param group counts no longer line up
                print(f"Could not restore optimizer state for '{key}' ({e}) - starting it fresh")

    def load_model_only(self, path, load_optimizers: bool = False):
        m = self.modeller
        checkpoint = torch.load(path, map_location=m.device, weights_only=False)
        m.gfn_model.load_state_dict(checkpoint['model_train'])
        m.ema_model.load_state_dict(checkpoint['model_eval'])
        m.gfn_model.train()
        m.ema_model.eval()
        if load_optimizers:
            self.load_optimizer_state(checkpoint)
