"""The policy interface: every attribute and method of a policy model (models/gfn.py::GFN, models/nf_policy.py::
NFPolicy) that the trainer reads, beyond torch.nn.Module's own. tests/models/test_nf_policy.py re-derives this set
from the trainer's source by AST, so a new read in train.py fails the test until both policies answer it: a missing
attribute on one policy is otherwise a flag that goes silently inert.

RECEIVERS are the names the trainer holds a policy under; TRAINER_FILES the modules scanned."""
TRAINER_FILES = ('train.py', 'protocol.py', 'checkpointing.py', 'gflownet_losses.py', 'utils.py', 'eval/utils.py',
                 'eval/xcond.py', 'eval/evaluations.py', 'eval/traj_reporting.py', 'grad_clip_guard.py',
                 'controller.py')
RECEIVERS = ('self.gfn_model', 'self.ema_model', 'm.gfn_model', 'm.ema_model', 'gfn_model',
             'ema_model', 'gfn', 'model', 'policy')

POLICY_ATTRIBUTES = (
    # the trajectory calls and the condition
    'get_traj_fwd', 'get_traj_bwd', 'get_traj_replay', 'get_condition_embedding',
    # geometry and switches
    'device', 'conditional', 'conditions_type', 'ang_mask', 'dead_rows', 'full_flow', 'force_on', 'features_on',
    '_wrap_ang', '_mean_over_live',
    # parameter groups
    't_model', 's_model', 'forward_policy', 'backward_policy', 'conditions_embedding_model', 'flow_model',
    # P_B freeze and snapshots
    'pb_frozen', 'freeze_backward_policy', 'unfreeze_backward_policy', 'install_pb_snapshot', 'pb_snapshot_state',
    # runtime flags the trainer sets on every build
    'traj_checkpoint', 'traj_checkpoint_modes',
    # SDE-only machinery (force terms, compiled step kernels, stored-force replay)
    'install_drift_force', 'compile_step_kernels', 'accum_var', 'force_gate_values', 'force_nonfinite_rows',
    'restart_force_gates', 'force_drift_t_min', 'last_step_mean_shift',
)

# state the trainer, the losses or the trajectory calls write on the policy before anything reads it: not required at
# construction (GFN does not have them then either), but the trajectory calls must write the ones they own
WRITTEN_STATE = ('_pb_frozen', '_stash_live_branches', '_live_fwd', '_live_bwd', '_live_replay',
                 'path_grad_scale_live', '_reward_grad_stats', '_path_grad_stats', '_replay_force_stats',
                 '_force_stats', '_z_cal_embedding')

# read only through getattr with a default (absent = the z-calibration step silently does nothing)
GETATTR_ONLY = ('_z_cal_embedding',)

# optional hooks: the trainer checks for them first (train.py::Modeller.fit_policy_data_scalers, ::density_readouts,
# ::log_xcond_metrics), and a policy without them skips that step or takes its estimated path (the SDE policy has none)
OPTIONAL_HOOKS = ('fit_data_scalers', 'log_density')


def missing_policy_attributes(model):
    """The interface names `model` does not have at construction (an empty list for a conforming policy)."""
    return [a for a in POLICY_ATTRIBUTES if not hasattr(model, a)]
