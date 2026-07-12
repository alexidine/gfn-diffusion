import os
from copy import deepcopy
from typing import Optional

import torch

from energy_sampling.buffer import CrystalBuffer, AnchorBuffer, ConditionLogZTracker
from energy_sampling.utils import atomic_save
from models import GFN

MODELLER_STATE_DEFAULTS = {
    'step_ind': 0,
    'phase': 1,
    'batch_size': 1,
    'batch_size_ever_oomed': False,
    # flips permanently once we've OOM'd at least once - switches growth from fast slow-start to slow congestion-avoidance
    'batch_size_cooldown_until': -1,  # step_ind until which batch size growth is frozen after a cut
    'lr_warmup_finished': False,
    'hit_init_kld': False,
    'grow_buffer': False,
    'fwd_loss_schedule': {},
    'bwd_loss_schedule': {},
    'replay_loss_schedule': {},
    'bwd_sampling_mode': 'dataset',
    'fwd_step_count': 0,
    'bwd_step_count': 0,
    'replay_step_count': 0,
    'fwd_frac': 0.0,
    'bwd_frac': 1.0,
    'replay_frac': 0.0,
    'combo_loss_record': [],
    'controller_anneal_streak': 0,
    'controller_lookahead': {
        'under': {'level': None, 'trend': 0.0},
        'over': {'level': None, 'trend': 0.0},
        'zerr': {'level': None, 'trend': 0.0},
    },
}


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

    def get_state_dict(self):
        m = self.modeller
        return {k: getattr(m, k) for k in MODELLER_STATE_DEFAULTS}

    def set_state_dict(self, state):
        m = self.modeller
        for k, default in MODELLER_STATE_DEFAULTS.items():
            setattr(m, k, state[k] if k in state else deepcopy(default))

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
        if stored_def != m.problem_def:
            print(f"Checkpoint {path} exists but its stored problem definition "
                  f"doesn't match the current config - starting fresh instead.\n"
                  f"{self.problem_mismatch_report(stored_def)}")
            return None

        return path

    def problem_mismatch_report(self, stored_def: Optional[dict]) -> str:
        """Readable stored-vs-current problem_def comparison, one line per differing field."""
        current_def = self.modeller.problem_def
        if stored_def is None:
            return ("  stored:  <none - checkpoint predates problem_def>\n"
                    f"  current: {current_def}")
        lines = []
        for key in sorted(set(stored_def) | set(current_def)):
            stored_val = stored_def.get(key, '<missing>')
            current_val = current_def.get(key, '<missing>')
            if stored_val != current_val:
                lines.append(f"  {key}: stored={stored_val!r}  current={current_val!r}")
        return '\n'.join(lines) if lines else f"  stored:  {stored_def}\n  current: {current_def}"

    def assert_problem_match(self, checkpoint: dict, path: str, config_key: str):
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
        """
        stored_def = checkpoint.get('problem_def')
        if stored_def != self.modeller.problem_def:
            raise ValueError(
                f"{config_key} checkpoint {path} was trained on a different problem "
                f"than the current config solves - either point {config_key} at a "
                f"checkpoint for this problem, or change the config to match it.\n"
                f"{self.problem_mismatch_report(stored_def)}")

    def save(self, tag: str):
        """
        tag: 'best' | 'hit_prior' | 'thermalized' | 'final'
        """
        m = self.modeller
        checkpoint = {
            'tag': tag,
            'run_name': m.run_name,
            'gfn_config': m.gfn_config,  # store once, reload from here
            'problem_def': m.problem_def,  # human-readable dict: energy function + prior this checkpoint solves
            'problem_hash': m.problem_hash,  # fast fingerprint of problem_def, also embedded in the filename
            'model_train': m.gfn_model.state_dict(),
            'model_eval': m.ema_model.state_dict(),
            'modeller_state': self.get_state_dict(),
            'metrics': m.metric_tracker.state_dict(),
            'optimizers': {k: opt.state_dict() for k, opt in m.optimizers.items()},
            'prior_buffer': m.prior_buffer.state_dict() if hasattr(m, 'prior_buffer') else None,
            'replay_buffer': m.replay_buffer.state_dict() if hasattr(m, 'replay_buffer') else None,
            'anchor_buffer': m.anchor_buffer.state_dict() if hasattr(m, 'anchor_buffer') else None,
            'condition_log_z': m.condition_log_z.state_dict() if hasattr(m, 'condition_log_z') else None,
        }
        path = self.path_for(tag)
        atomic_save(checkpoint, path)

    def load_full(self, path, load_opt_state: bool = True):
        m = self.modeller
        checkpoint = torch.load(path, map_location=m.device, weights_only=False)
        self.assert_problem_match(checkpoint, path, 'checkpoint_name')
        m.gfn_config = checkpoint['gfn_config']
        m.gfn_model = GFN(**m.gfn_config).to(m.device)
        m.gfn_model.load_state_dict(checkpoint['model_train'])
        m.ema_model = deepcopy(m.gfn_model)
        m.ema_model.load_state_dict(checkpoint['model_eval'])

        m.gfn_model.train()
        m.ema_model.eval()

        self.set_state_dict(checkpoint['modeller_state'])
        m.metric_tracker.load_state_dict(checkpoint.get('metrics', {}))

        if checkpoint.get('prior_buffer') is not None:
            m.prior_buffer = CrystalBuffer.from_state_dict(checkpoint['prior_buffer'], device='cpu')
        if checkpoint.get('replay_buffer') is not None:
            m.replay_buffer = CrystalBuffer.from_state_dict(checkpoint['replay_buffer'], device='cpu')
        if checkpoint.get('anchor_buffer') is not None:
            m.anchor_buffer = AnchorBuffer.from_state_dict(checkpoint['anchor_buffer'], device='cpu')
        if checkpoint.get('condition_log_z') is not None:
            m.condition_log_z = ConditionLogZTracker.from_state_dict(
                checkpoint['condition_log_z'], current_step=m.step_ind)

        if getattr(m.args, 'override_loss_coeffs', False):
            # discard the schedule baked into the checkpoint so set_loss_coeffs()
            # re-parses fwd/bwd/replay_loss_coeffs from the current config instead
            m.fwd_loss_schedule = {}
            m.bwd_loss_schedule = {}
            m.replay_loss_schedule = {}

        if load_opt_state:
            m.init_schedulers_optimizers()
            self.load_optimizer_state(checkpoint)

            if getattr(m.args, 'override_learning_rates', False):
                # overwrite just the numeric LR the checkpoint's optimizer state
                # restored with this config's target rate - warmup/anneal status
                # (lr_warmup_finished) and the schedulers themselves are untouched,
                # so they carry on stepping from this new value. Adam's momentum
                # buffers (exp_avg/exp_avg_sq) are also left as-is.
                target_lrs = {'fwd': m.args.lr_policy, 'bwd': m.args.lr_back,
                              'replay': m.args.lr_replay, 'fused': m.args.lr_fused,
                              'flow': m.args.lr_flow}
                for key, opt in m.optimizers.items():
                    for group in opt.param_groups:
                        group['lr'] = target_lrs[key]

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
