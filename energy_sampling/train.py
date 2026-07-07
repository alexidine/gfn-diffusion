import gc
import math
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional

from energy_sampling.eval.evaluations import to_loggable, sliced_wasserstein, adjust_fig_filesize, eval_figs, \
    log_ess_frac
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
from torch.optim import lr_scheduler
from tqdm import trange

from energies.molecular_crystal import MolecularCrystal
from energy_sampling.buffer import CrystalBuffer
from energy_sampling.eval.utils import sample_eval_fwd_trajs, LossSpikeMonitor
from energy_sampling.utils import is_cuda_oom, get_annealing_factor, \
    parse_loss_schedules, dict2namespace, update_loss_schedule, \
    atomic_save, get_discretizer, log_elapsed_times, MetricTracker, quick_tb_stats, uniform_discretizer, logmeanexp, \
    cal_subtb_coef_matrix
from gflownet_losses import get_gfn_forward_loss, get_gfn_backward_loss
from models import GFN
from mxtaltools.common.training_utils import flatten_wandb_params
from utils import get_train_args, get_gfn_init_state, set_seed, \
    update_ema, get_problem_definition, problem_hash, problem_slug

MODELLER_STATE_DEFAULTS = {
    'step_ind': 0,
    'phase': 1,
    'batch_size': 1,
    'batch_size_ever_oomed': False,  # flips permanently once we've OOM'd at least once - switches growth from fast slow-start to slow congestion-avoidance
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


class Modeller:
    def __init__(self):
        self.step_ind = None
        self.args = get_train_args()
        torch.cuda.set_per_process_memory_fraction(self.args.cuda_memory_fraction, device=0)
        torch.cuda.init()  # create context with the cap already in place

        self.fwd_loss_monitor = LossSpikeMonitor(window=200, warmup=250, cooldown=100, ceiling_factor=100.0)
        self.bwd_loss_monitor = LossSpikeMonitor(window=200, warmup=250, cooldown=100, ceiling_factor=100.0)
        self.replay_loss_monitor = LossSpikeMonitor(window=200, warmup=250, cooldown=100, ceiling_factor=100.0)
        self.fused_loss_monitor = LossSpikeMonitor(window=200, warmup=250, cooldown=100, ceiling_factor=100.0)
        self.loss_monitors = {'fwd': self.fwd_loss_monitor,
                              'bwd': self.bwd_loss_monitor,
                              'replay': self.replay_loss_monitor,
                              'fused': self.fused_loss_monitor}

        set_seed(self.args.seed)
        if 'SLURM_PROCID' in os.environ:
            self.args.seed += int(os.environ["SLURM_PROCID"])

        if self.args.both_ways and self.args.bwd:
            self.args.bwd = False

        config = self.args.__dict__
        config["Experiment"] = "{args.energy}"
        self.run_name = str(self.args.tag) + '_' + str(self.args.run_name)

        # fingerprint of the energy function + prior this run is training against,
        # as opposed to training hyperparameters - see save_checkpoint / _checkpoint_path
        self.problem_def = get_problem_definition(self.args)
        self.problem_hash = problem_hash(self.problem_def)
        self.problem_slug = problem_slug(self.args, self.problem_def)

        self.times = {}
        self.device = self.args.device
        self.init_train_constants()

    def init_train_constants(self):
        for k, v in MODELLER_STATE_DEFAULTS.items():
            if k in self.args.__dict__:
                setattr(self, k, self.args.__dict__[k])
            else:
                setattr(self, k, deepcopy(v))

        self.metric_tracker = MetricTracker(period=100)

    def _get_modeller_state_dict(self):
        return {k: getattr(self, k) for k in MODELLER_STATE_DEFAULTS}

    def _set_modeller_state_dict(self, state):
        for k, default in MODELLER_STATE_DEFAULTS.items():
            setattr(self, k, state[k] if k in state else deepcopy(default))

    def save_checkpoint(self, tag: str):
        """
        tag: 'best' | 'hit_prior' | 'thermalized' | 'final'
        """
        checkpoint = {
            'tag': tag,
            'run_name': self.run_name,
            'gfn_config': self.gfn_config,  # store once, reload from here
            'problem_def': self.problem_def,  # human-readable dict: energy function + prior this checkpoint solves
            'problem_hash': self.problem_hash,  # fast fingerprint of problem_def, also embedded in the filename
            'model_train': self.gfn_model.state_dict(),
            'model_eval': self.ema_model.state_dict(),
            'modeller_state': self._get_modeller_state_dict(),
            'metrics': self.metric_tracker.state_dict(),
            'optimizers': {k: opt.state_dict() for k, opt in self.optimizers.items()},
            'prior_buffer': self.prior_buffer.state_dict() if hasattr(self, 'prior_buffer') else None,
            'replay_buffer': self.replay_buffer.state_dict() if hasattr(self, 'replay_buffer') else None,
        }
        path = self._checkpoint_path(tag)
        atomic_save(checkpoint, path)

    def load_model_and_state(self, path, load_opt_state: bool = True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.gfn_config = checkpoint['gfn_config']
        self.gfn_model = GFN(**self.gfn_config).to(self.device)
        self.gfn_model.load_state_dict(checkpoint['model_train'])
        self.ema_model = deepcopy(self.gfn_model)
        self.ema_model.load_state_dict(checkpoint['model_eval'])

        self.gfn_model.train()
        self.ema_model.eval()

        self._set_modeller_state_dict(checkpoint['modeller_state'])
        self.metric_tracker.load_state_dict(checkpoint.get('metrics', {}))

        if checkpoint.get('prior_buffer') is not None:
            self.prior_buffer = CrystalBuffer.from_state_dict(checkpoint['prior_buffer'], device='cpu')
        if checkpoint.get('replay_buffer') is not None:
            self.replay_buffer = CrystalBuffer.from_state_dict(checkpoint['replay_buffer'], device='cpu')

        if getattr(self.args, 'override_loss_coeffs', False):
            # discard the schedule baked into the checkpoint so set_loss_coeffs()
            # re-parses fwd/bwd/replay_loss_coeffs from the current config instead
            self.fwd_loss_schedule = {}
            self.bwd_loss_schedule = {}
            self.replay_loss_schedule = {}

        if load_opt_state:
            self.init_schedulers_optimizers()
            self.load_optimizer_state(checkpoint)

            if getattr(self.args, 'override_learning_rates', False):
                # overwrite just the numeric LR the checkpoint's optimizer state
                # restored with this config's target rate - warmup/anneal status
                # (lr_warmup_finished) and the schedulers themselves are untouched,
                # so they carry on stepping from this new value. Adam's momentum
                # buffers (exp_avg/exp_avg_sq) are also left as-is.
                target_lrs = {'fwd': self.args.lr_policy, 'bwd': self.args.lr_back,
                              'replay': self.args.lr_replay, 'fused': self.args.lr_fused,
                              'flow': self.args.lr_flow}
                for key, opt in self.optimizers.items():
                    for group in opt.param_groups:
                        group['lr'] = target_lrs[key]

    def load_optimizer_state(self, checkpoint):
        saved_optimizers = checkpoint['optimizers']
        for key, opt in self.optimizers.items():
            if key not in saved_optimizers:
                print(f"No saved optimizer state for '{key}' - starting it fresh")
                continue
            try:
                opt.load_state_dict(saved_optimizers[key])
            except (ValueError, RuntimeError) as e:
                # e.g. checkpoint predates flow params folding into the policy
                # optimizers, so param group counts no longer line up
                print(f"Could not restore optimizer state for '{key}' ({e}) - starting it fresh")

    def load_model_state(self, path, load_optimizers: bool = False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.gfn_model.load_state_dict(checkpoint['model_train'])
        self.ema_model.load_state_dict(checkpoint['model_eval'])
        self.gfn_model.train()
        self.ema_model.eval()
        if load_optimizers:
            self.load_optimizer_state(checkpoint)

    def _checkpoint_path(self, tag: str) -> str:
        return f'{self.args.checkpoints_dir}/{self.run_name}_{self.problem_slug}_{tag}.pt'

    def _find_matching_checkpoint(self, tag: str) -> Optional[str]:
        """
        Look for a checkpoint saved under this run_name/tag whose *stored*
        problem_def dict matches the current config's - not just its filename
        hash, since the slug format (and even the hash length) may change
        later. Refuses to reload (rather than raising) on any mismatch or on
        older checkpoints saved before problem_def existed, so a stale/renamed
        checkpoint never gets silently treated as a valid resume point.
        """
        path = self._checkpoint_path(tag)
        if not os.path.exists(path):
            return None

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        stored_def = checkpoint.get('problem_def')
        if stored_def != self.problem_def:
            print(f"Checkpoint {path} exists but its stored problem definition "
                  f"doesn't match the current config - starting fresh instead.\n"
                  f"  stored:  {stored_def}\n"
                  f"  current: {self.problem_def}")
            return None

        return path

    def train_logic(self, it):
        replay_available = hasattr(self, 'replay_buffer') and len(self.replay_buffer) > 0

        if self.args.anchor_fwd_bwd:
            if self.phase < 3:
                return 'bwd'
            elif self.phase == 3:
                if self.args.fused:  # fwd/bwd/replay fire together every step, fused by loss weight instead of turn-taking
                    return 'fused'
                probs = np.array([self.fwd_frac, self.bwd_frac, self.replay_frac])
                if not replay_available:  # buffer not populated yet - fold its share into backward
                    probs = np.array([probs[0], probs[1] + probs[2], 0.0])
                return np.random.choice(['fwd', 'bwd', 'replay'], p=probs)

        elif self.args.both_ways:
            return 'fwd' if it % 2 == 0 else 'bwd'  # alternate, always fwd first

        elif self.args.bwd:  # backward ONLY
            return 'bwd'

        else:  # forward ONLY
            return 'fwd'

    def increment_batch_size(self):
        """
        AIMD-style growth: fast multiplicative growth ("slow start") until the first
        OOM this run; after that, growth freezes for oom_cooldown_steps following any
        cut (letting the reduced size prove stable), then resumes at a much slower
        multiplicative rate ("congestion avoidance") so we don't immediately re-trigger
        the same OOM. A later OOM (e.g. moving into a more VRAM-hungry training phase)
        cuts and re-cools exactly the same way.
        """
        if self.batch_size >= self.args.max_batch_size:
            return
        if self.step_ind < self.batch_size_cooldown_until:
            return  # recently cut -- hold flat until the new level proves stable

        growth_factor = (self.args.batch_growth_increment if not self.batch_size_ever_oomed
                         else self.args.batch_growth_slow_increment)
        self.batch_size = min(self.args.max_batch_size,
                              max(self.batch_size + 1, int(self.batch_size * growth_factor)))

    def step_lr_schedule(self):
        lr = self.optimizers['fwd'].param_groups[0]['lr']
        if not self.lr_warmup_finished:
            self.schedulers['policy_1'].step()
            self.schedulers['policy_1b'].step()
            self.schedulers['policy_1r'].step()
            self.schedulers['policy_1u'].step()

            if lr >= self.args.lr_policy:
                self.lr_warmup_finished = True

        elif lr > self.args.min_lr:
            self.schedulers['policy_2'].step()
            self.schedulers['policy_2b'].step()
            self.schedulers['policy_2r'].step()
            self.schedulers['policy_2u'].step()

        if 'flow' in self.schedulers:
            self.schedulers['flow'].step()

        return lr

    def ten_step_reporting(self):
        metrics = {}
        metrics.update(self.metric_tracker.snapshot(changed_only=True))

        for opt_type in ['fwd', 'bwd', 'replay', 'fused', 'flow']:
            if opt_type in self.optimizers:
                metrics.update({f'lr_{opt_type}': self.optimizers[opt_type].param_groups[0]['lr']})

        metrics['phase'] = self.phase
        metrics['Fwd Frac'] = self.fwd_frac
        metrics['Bwd Frac'] = self.bwd_frac
        metrics['Replay Frac'] = self.replay_frac
        metrics['under_coverage_threshold'] = self.args.controller.under_threshold
        metrics['over_coverage_threshold'] = self.args.controller.over_threshold
        metrics['zerr_threshold'] = self.args.controller.zerr_threshold
        metrics.update(log_elapsed_times(self.times))
        return metrics

    def set_loss_coeffs(self):
        if not self.fwd_loss_schedule:
            self.fwd_loss_schedule = parse_loss_schedules(self.args.fwd_loss_coeffs)
            self.bwd_loss_schedule = parse_loss_schedules(self.args.bwd_loss_coeffs)
            self.replay_loss_schedule = parse_loss_schedules(self.args.replay_loss_coeffs)

            self.args.fwd_loss_coeffs = dict2namespace({k: 0.0 for k in self.fwd_loss_schedule})
            self.args.bwd_loss_coeffs = dict2namespace({k: 0.0 for k in self.bwd_loss_schedule})
            self.args.replay_loss_coeffs = dict2namespace({k: 0.0 for k in self.replay_loss_schedule})

        update_loss_schedule(self.step_ind, self.fwd_loss_schedule, self.args.fwd_loss_coeffs.__dict__)
        update_loss_schedule(self.step_ind, self.bwd_loss_schedule, self.args.bwd_loss_coeffs.__dict__)
        update_loss_schedule(self.step_ind, self.replay_loss_schedule, self.args.replay_loss_coeffs.__dict__)

        if any([self.args.fwd_loss_coeffs.subtb > 0, self.args.bwd_loss_coeffs.subtb > 0,
                self.args.replay_loss_coeffs.subtb > 0]):
            self.args.fwd_loss_coeffs.coeff_matrix = cal_subtb_coef_matrix(  # todo delete this re-instantiation
                self.args.fwd_loss_coeffs.subtb_lambda, self.args.integrator.T).to(self.gfn_model.device)
            self.args.bwd_loss_coeffs.coeff_matrix = cal_subtb_coef_matrix(
                self.args.bwd_loss_coeffs.subtb_lambda, self.args.integrator.T).to(self.gfn_model.device)
            self.args.replay_loss_coeffs.coeff_matrix = cal_subtb_coef_matrix(
                self.args.replay_loss_coeffs.subtb_lambda, self.args.integrator.T).to(self.gfn_model.device)

    def get_conditioning_dim(self):
        conditions_dim = 0
        if self.args.temperature_conditioning:
            conditions_dim += 1
        if self.args.sg_conditioning:
            conditions_dim += 237
        if self.args.zp_conditioning:
            conditions_dim += 1
        return conditions_dim

    def init_energy_function(self):
        energy_config = {
            'device': self.device,
            'energy_function': self.args.energy_function,
            'mlip_path': self.args.mlip_path,
            'space_groups': self.args.space_groups,
            'z_primes': self.args.z_primes,
            'sg_conditioning': self.args.sg_conditioning,
            'molecule_conditioning': self.args.molecule_conditioning,
            'temperature_conditioning': self.args.temperature_conditioning,
            'zp_conditioning': self.args.zp_conditioning,
        }
        energy_config.update(self.args.energy_config.__dict__)
        self.energy_function = MolecularCrystal(**energy_config)

    def _build_gfn_config(self):
        return dict(
            dim=self.energy_function.data_ndim,
            conditions_dim=self.get_conditioning_dim(),
            conditions_type='molecule' if self.args.molecule_conditioning else 'vector',
            conditional=any([
                self.args.temperature_conditioning,
                self.args.molecule_conditioning,
                self.args.sg_conditioning,
                self.args.zp_conditioning,
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
            print(f"Loading model from checkpoint {reload_path}")
            self.load_model_and_state(reload_path)

        elif self.args.continue_from_checkpoint:
            reload_path = self._find_matching_checkpoint('running')
            if reload_path is not None:
                print(f"Reloading automatically from prior checkpoint {reload_path}")
                reload = True
                self.load_model_and_state(reload_path)

        if not reload:
            self.gfn_config = self._build_gfn_config()
            self.gfn_model = GFN(**self.gfn_config).to(self.device)
            self.ema_model = deepcopy(self.gfn_model)
            # opt init
            self.init_schedulers_optimizers()

    def init_schedulers_optimizers(self):
        init_fwd_lr = self.args.lr_policy / self.args.lr_warmup_ratio
        init_bwd_lr = self.args.lr_back / self.args.lr_warmup_ratio
        init_replay_lr = self.args.lr_replay / self.args.lr_warmup_ratio
        init_fused_lr = self.args.lr_fused / self.args.lr_warmup_ratio
        init_flow_lr = self.args.lr_flow

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
                # flow model folds into the policy optimizers rather than getting its own
                plist += [{'params': gfn_model.flow_model.parameters()}]

            return plist

        self.optimizers = {}
        weight_decay = self.args.weight_decay if self.args.use_weight_decay else 0
        self.optimizers['fwd'] = torch.optim.Adam(get_policy_params(self.gfn_model), init_fwd_lr,
                                                  weight_decay=weight_decay)
        self.optimizers['bwd'] = torch.optim.Adam(get_policy_params(self.gfn_model), init_bwd_lr,
                                                  weight_decay=weight_decay)
        self.optimizers['replay'] = torch.optim.Adam(get_policy_params(self.gfn_model), init_replay_lr,
                                                     weight_decay=weight_decay)
        self.optimizers['fused'] = torch.optim.Adam(get_policy_params(self.gfn_model), init_fused_lr,
                                                    weight_decay=weight_decay)
        if not self.gfn_model.conditional:
            flow_params = self.gfn_model.flow_model.parameters()
            self.optimizers['flow'] = torch.optim.Adam(flow_params, init_flow_lr, weight_decay=weight_decay)

        self.schedulers = {}
        lr_warmup_lambda = get_annealing_factor(1,
                                                self.args.lr_warmup_ratio,
                                                self.args.lr_warmup_time,
                                                10)
        lr_annealing_lambda = get_annealing_factor(self.args.lr_policy,
                                                   self.args.min_lr,
                                                   self.args.lr_anneal_time,
                                                   10)

        self.schedulers['policy_1'] = lr_scheduler.MultiplicativeLR(
            self.optimizers['fwd'], lr_lambda=lambda epoch: lr_warmup_lambda)
        self.schedulers['policy_2'] = lr_scheduler.MultiplicativeLR(
            self.optimizers['fwd'], lr_lambda=lambda epoch: lr_annealing_lambda)

        self.schedulers['policy_1b'] = lr_scheduler.MultiplicativeLR(
            self.optimizers['bwd'], lr_lambda=lambda epoch: lr_warmup_lambda)
        self.schedulers['policy_2b'] = lr_scheduler.MultiplicativeLR(
            self.optimizers['bwd'], lr_lambda=lambda epoch: lr_annealing_lambda)

        self.schedulers['policy_1r'] = lr_scheduler.MultiplicativeLR(
            self.optimizers['replay'], lr_lambda=lambda epoch: lr_warmup_lambda)
        self.schedulers['policy_2r'] = lr_scheduler.MultiplicativeLR(
            self.optimizers['replay'], lr_lambda=lambda epoch: lr_annealing_lambda)

        self.schedulers['policy_1u'] = lr_scheduler.MultiplicativeLR(
            self.optimizers['fused'], lr_lambda=lambda epoch: lr_warmup_lambda)
        self.schedulers['policy_2u'] = lr_scheduler.MultiplicativeLR(
            self.optimizers['fused'], lr_lambda=lambda epoch: lr_annealing_lambda)

        if not self.gfn_model.conditional:
            flow_annealing_lambda = get_annealing_factor(1,
                                                         0.1,
                                                         self.args.lr_anneal_time,
                                                         10)
            self.schedulers['flow'] = lr_scheduler.MultiplicativeLR(self.optimizers['flow'],
                                                                    lr_lambda=lambda epoch: flow_annealing_lambda)

    def init_prior_dataset(self):

        prior_data = torch.load(self.args.prior_path, weights_only=False)
        prior = prior_data['equalized_prior']
        prior['smiles'] = None
        prior['identifier'] = None
        if True:  # not hasattr(prior, self.args.energy_function):
            print("Re-analyzing prior energies")
            prior = prior.to(self.device)
            energy, prior = self.energy_function.batched_analyze_crystal_batch(
                prior.latent_params(),
                prior,
                self.args.energy_config.temperature * torch.ones((prior.num_graphs), dtype=torch.float32,
                                                                 device=self.device),
                return_batch=True,
                internal_oom_recovery=True,  # one-off pass over the whole prior dataset at init -- prefer the adaptive, self-healing chunked path over a hard crash, regardless of the training-time flag
            )
        if hasattr(prior_data, 'thermal_scaling_factor'):
            self.energy_function.lj_coeff = prior_data['thermal_scaling_factor']

        self.prior_dataset = CrystalBuffer(prior,
                                           device='cpu',
                                           max_z_prime=max(self.args.z_primes),
                                           x_fn=None,  # 'latent_params',
                                           y_fn=self.args.energy_function
                                           )

        if self.args.prior_model_name is not None:
            prior_path = f'{self.args.checkpoints_dir}/{self.args.prior_model_name}'
            checkpoint = torch.load(prior_path, map_location=self.device, weights_only=False)
            gfn_config = checkpoint['gfn_config']
            self.prior_model = GFN(**gfn_config).to(self.device)
            self.prior_model.load_state_dict(checkpoint['model_eval'])
            self.prior_model.eval()
            self.grow_prior_buffer()

    def init_mol_dataset(self):
        data_list = torch.load(self.args.molecules_path, weights_only=False)
        if isinstance(data_list, dict):
            for key, value in data_list.items():
                if key == 'prior':
                    data_list = value

        self.mol_dataset = CrystalBuffer(data_list,
                                         device='cpu',
                                         max_z_prime=max(self.args.z_primes))

        if self.args.test_molecules_path is not None:
            data_list = torch.load(self.args.test_molecules_path, weights_only=False)
            self.test_mol_dataset = CrystalBuffer(data_list,
                                                  device='cpu',
                                                  max_z_prime=max(self.args.z_primes))
        else:
            self.test_mol_dataset = None

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

            # data init
            self.init_mol_dataset()
            self.init_prior_dataset()

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

                step_type = self.train_logic(self.step_ind)
                self.times['train_step_start'] = time()
                try:
                    current_loss = self.train_step(step_type)

                    if self.args.grow_batch_size:
                        self.increment_batch_size()

                except (RuntimeError, ValueError) as e:  # if we do hit OOM, slash the batch size
                    self.handle_train_epoch_error(e, step_type)
                self.times['train_step_end'] = time()

                # train monitoring
                if self.step_ind % 10 == 0:
                    lr = self.step_lr_schedule()
                    metrics.update(self.ten_step_reporting())
                    self.monitor_losses(current_loss, step_type)

                    if self.combo_loss_record[-1] <= np.amin(self.combo_loss_record):
                        self.save_checkpoint('best')

                    if self.phase == 3:
                        self.three_phase_controller()

                # evaluation work
                if (self.step_ind % self.args.eval_period == 0 and self.step_ind > 0) or self.step_ind == 50:
                    metrics.update(self.evaluation())

                if len(metrics) > 0:
                    wandb.log(metrics, step=self.step_ind, commit=True)

                if self.step_ind % 50 == 0:  # save running model
                    self.save_checkpoint('running')

            self.save_checkpoint('final')
            print("Finished Training!")

    def monitor_losses(self, current_loss, step_type):
        if current_loss is not None:
            trig = self.loss_monitors[step_type].record(current_loss, self.step_ind)

            if trig:
                self.fire_loss_spike()

            current_fwd = self.metric_tracker.get('fwd', 'r2')
            current_bwd = self.metric_tracker.get('bwd', 'r2')
            current_replay = self.metric_tracker.get('replay', 'r2')

            if current_fwd is None and current_bwd is None and current_replay is None:
                self.combo_loss_record.append(float('inf'))
            else:
                total = (current_fwd or 0) + (current_bwd or 0) + (current_replay or 0)
                self.combo_loss_record.append(3 - total)  # (1-x) + (1-y) + (1-z) = 3-x-y-z

    def fire_loss_spike(self):
        print("Firing LR spike & recovery")
        running_checkpoint_path = self._checkpoint_path('best')
        if os.path.exists(running_checkpoint_path):
            self.load_model_state(running_checkpoint_path,
                                  load_optimizers=True)
            # fix also rolling metrics with appropriate rebase
            checkpoint = torch.load(running_checkpoint_path, map_location=self.device, weights_only=False)
            step = deepcopy(self.step_ind)
            self._set_modeller_state_dict(checkpoint['modeller_state'])
            self.metric_tracker.load_state_dict(checkpoint.get('metrics', {}))
            self.step_ind = step
            if checkpoint.get('prior_buffer') is not None:
                self.prior_buffer = CrystalBuffer.from_state_dict(checkpoint['prior_buffer'], device='cpu')
            if checkpoint.get('replay_buffer') is not None:
                self.replay_buffer = CrystalBuffer.from_state_dict(checkpoint['replay_buffer'], device='cpu')

        lr_cut_val = 0.75

        for key, opt in self.optimizers.items():
            # opt.state = defaultdict(dict)  # wipe momentum buffers too
            for g in opt.param_groups:
                if g['lr'] > self.args.min_lr:
                    g['lr'] = max(g['lr'] * lr_cut_val, self.args.min_lr)

        self.lr_warmup_finished = True

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
        accumulating = accum_target > 0
        self.fused_accum_count = getattr(self, 'fused_accum_count', 0)
        starting_new_cycle = (not accumulating) or (self.fused_accum_count == 0)

        if starting_new_cycle:
            if 'flow' in self.optimizers:
                self.optimizers['flow'].zero_grad(set_to_none=True)
            self.optimizers[step_type].zero_grad(set_to_none=True)

        if step_type == 'fwd':
            loss, crystal_batch, loss_dict = self.fwd_train_step(
                discretizer,
                return_exp=True,
                repeats=self.args.repeats,
                report_losses=True
            )
            self.fwd_step_count += 1
            current_step_count = self.fwd_step_count

        elif step_type == 'bwd':
            loss, loss_dict = self.bwd_train_step(
                discretizer,
                repeats=self.args.repeats,
                report_losses=True)
            self.bwd_step_count += 1
            current_step_count = self.bwd_step_count

        elif step_type == 'replay':
            loss, loss_dict = self.replay_train_step(
                discretizer,
                repeats=self.args.repeats,
                report_losses=True)
            self.replay_step_count += 1
            current_step_count = self.replay_step_count

        elif step_type == 'fused':
            loss, sub_losses = self.fused_train_step(
                discretizer,
                repeats=self.args.repeats,
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
                         repeats: int,
                         report_losses: bool = True):
        """
        Fires fwd, bwd, and replay steps together and fuses their losses into a single
        weighted-sum loss (weighted by fwd_frac/bwd_frac/replay_frac), backed by its own
        optimizer, rather than randomly picking one of the three per step as the phase-3
        controller does.

        A branch whose frac has fallen below controller.deactivate_threshold is skipped
        entirely (not just down-weighted) to save its compute; the remaining active
        branches' weights are renormalized to sum to 1. Since the three fracs always sum
        to 1, at least one is guaranteed to survive as long as the threshold is < 1/3.

        Every controller.refresh_every steps, every branch is force-evaluated regardless
        of its frac, so a long-deactivated branch's rolling metric_tracker stats don't go
        stale. A force-evaluated branch that's still below threshold contributes zero
        weight to the fused loss -- it's run only to refresh its stats, not for gradient.
        """
        replay_available = hasattr(self, 'replay_buffer') and len(self.replay_buffer) > 0
        deactivate_threshold = self.args.controller.deactivate_threshold

        self.fused_step_count = getattr(self, 'fused_step_count', 0) + 1
        force_refresh = self.fused_step_count % self.args.controller.refresh_every == 0

        sub_losses = {}
        weights = {}

        fwd_active = self.fwd_frac >= deactivate_threshold
        if fwd_active or force_refresh:
            fwd_loss, crystal_batch, fwd_loss_dict = self.fwd_train_step(
                discretizer,
                return_exp=True,
                repeats=repeats,
                report_losses=report_losses)
            if not fwd_active:  # force-refresh only -- keep its graph out of the fused loss
                fwd_loss = fwd_loss.detach()
            sub_losses['fwd'] = (fwd_loss, fwd_loss_dict)
            weights['fwd'] = self.fwd_frac if fwd_active else 0.0

        bwd_active = self.bwd_frac >= deactivate_threshold
        if bwd_active or force_refresh:
            bwd_loss, bwd_loss_dict = self.bwd_train_step(
                discretizer,
                repeats=repeats,
                report_losses=report_losses)
            if not bwd_active:
                bwd_loss = bwd_loss.detach()
            sub_losses['bwd'] = (bwd_loss, bwd_loss_dict)
            weights['bwd'] = self.bwd_frac if bwd_active else 0.0

        if replay_available:
            replay_active = self.replay_frac >= deactivate_threshold
            if replay_active or force_refresh:
                replay_loss, replay_loss_dict = self.replay_train_step(
                    discretizer,
                    repeats=repeats,
                    report_losses=report_losses)
                if not replay_active:
                    replay_loss = replay_loss.detach()
                sub_losses['replay'] = (replay_loss, replay_loss_dict)
                weights['replay'] = self.replay_frac if replay_active else 0.0
        elif bwd_active:  # buffer not populated yet - fold its share into backward, as the alternating controller does
            weights['bwd'] += self.replay_frac

        assert sub_losses, "fused_train_step deactivated all three branches -- controller.deactivate_threshold must be < 1/3"

        total_weight = sum(weights.values())
        fused_loss = sum((weights[k] / total_weight) * sub_losses[k][0]
                         for k in sub_losses if weights[k] > 0)

        if fwd_active or force_refresh:
            # churn on the fly
            self.manage_replay_buffer(fwd_loss_dict,
                                      crystal_batch)
            del crystal_batch

        return fused_loss, sub_losses

    def record_fused_substep_losses(self, sub_losses):
        for sub_type in ('fwd', 'bwd', 'replay'):
            if sub_type in sub_losses:
                setattr(self, f'{sub_type}_step_count', getattr(self, f'{sub_type}_step_count') + 1)

        step_counts = {'fwd': self.fwd_step_count,
                       'bwd': self.bwd_step_count,
                       'replay': self.replay_step_count}

        for sub_type, (sub_loss, loss_dict) in sub_losses.items():
            if step_counts[sub_type] % 10 == 0:
                self._update_rolling(loss_dict, sub_loss, sub_type)

    def _update_rolling(self, loss_dict, sub_loss, sub_type):
        stats = quick_tb_stats(loss_dict['log_pf'], loss_dict['log_pb'],
                               loss_dict['log_Z'], loss_dict['log_r'])
        stats.update({k: v.item() for k, v in loss_dict.items() if k not in
                      ['log_pf', 'log_pb', 'log_Z', 'log_r', 'losses', 'flow_states', 'resid']})
        stats.update({'loss': sub_loss.cpu().detach().item()})
        stats.update({'log_Z_learned': loss_dict['log_Z'].cpu().mean().detach().item()})
        self.metric_tracker.update(sub_type, stats, self.step_ind)

    def three_phase_controller(self):
        under, over, zerr = self._get_controller_metrics()
        under, over, zerr = self._lookahead_controller_metrics(under, over, zerr)
        state = self._select_controller_state(under, over, zerr)
        self._nudge_mode_fracs(state)

    def _get_controller_metrics(self):
        under = self.metric_tracker.get('bwd', 'under_coverage')
        over = self.metric_tracker.get('fwd', 'over_coverage')
        # in very, very good terminal convergence, empirical Z becomes the target
        jensen_zerr = self.metric_tracker.get('fwd', 'jensen_z_err')
        zerr = jensen_zerr if jensen_zerr is not None else 0.0

        if under is None:
            under = float("inf")  # bootstrap toward bwd
        if over is None:
            over = 0.0  # do not demand replay before fwd stats exist

        return under, over, zerr

    def _lookahead_controller_metrics(self, under, over, zerr):
        """
        Extrapolate each controller metric a few controller ticks into the
        future by EMA-smoothing its trend and projecting forward linearly, so
        _select_controller_state reacts to where a metric is heading rather
        than where it currently sits. Without this, mode-frac mass keeps
        moving at full speed right up to a threshold crossing and overshoots
        the optimal balance before the (lagging) raw metric ever reflects it.
        """
        ctrl = self.args.controller
        lookahead = self.controller_lookahead
        return (
            self._linear_ema_lookahead(lookahead['under'], under, ctrl),
            self._linear_ema_lookahead(lookahead['over'], over, ctrl),
            self._linear_ema_lookahead(lookahead['zerr'], zerr, ctrl),
        )

    @staticmethod
    def _linear_ema_lookahead(state, value, ctrl):
        """
        under/over/zerr are already EMAs coming out of metric_tracker, so
        `value` is already a smoothed level - only its trend (the per-tick
        delta) needs its own EMA here. state: {'level', 'trend'} dict, mutated
        in place (persisted via MODELLER_STATE_DEFAULTS so it survives
        checkpoint reloads); 'level' just holds the previous value.
        Returns value + horizon * trend, i.e. the predicted value
        `ctrl.lookahead_horizon` controller ticks ahead.
        """
        if not math.isfinite(value):
            return value
        if state['level'] is None:
            state['level'] = value
            return value

        trend_alpha = getattr(ctrl, 'lookahead_trend_alpha', 0.1)
        horizon = getattr(ctrl, 'lookahead_horizon', 5)
        state['trend'] = trend_alpha * (value - state['level']) + (1 - trend_alpha) * state['trend']
        state['level'] = value

        return value + horizon * state['trend']

    def _select_controller_state(self, under, over, zerr):
        """
        Priority order: undercoverage repair > Z convergence > replay repair > global
        tightening. This is the state/step-size selection logic most likely to change
        later, so it's kept isolated from the metric plumbing and frac math around it.
        """
        ctrl = self.args.controller
        if under > ctrl.under_threshold:
            self.controller_anneal_streak = 0
            return "bwd"
        elif zerr > ctrl.zerr_threshold:
            self.controller_anneal_streak = 0
            return "fwd"
        elif over > ctrl.over_threshold:
            self.controller_anneal_streak = 0
            return "replay"
        else:
            # joint condition (all three metrics within threshold) satisfied this tick;
            # require it to hold for `anneal_patience` consecutive ticks before tightening
            # the margins, since a single tick is too susceptible to metric noise
            self.controller_anneal_streak += 1
            if self.controller_anneal_streak >= getattr(ctrl, 'anneal_patience', 1):
                self._anneal_controller_thresholds()
                self.controller_anneal_streak = 0
            return "fwd"

    def _anneal_controller_thresholds(self):
        ctrl = self.args.controller
        if ctrl.under_threshold > ctrl.min_threshold:
            ctrl.under_threshold *= ctrl.decay_rate
        if ctrl.over_threshold > ctrl.min_threshold:
            ctrl.over_threshold *= ctrl.decay_rate
        if ctrl.zerr_threshold > ctrl.zerr_min_threshold:
            ctrl.zerr_threshold *= ctrl.decay_rate

    def _nudge_mode_fracs(self, state):
        ctrl = self.args.controller
        probs = np.array([self.fwd_frac, self.bwd_frac, self.replay_frac], dtype=float)
        probs /= probs.sum()

        idx = {"fwd": 0, "bwd": 1, "replay": 2}[state]

        m = ctrl.min_mode_frac  # requires m < 1/3
        free = 1.0 - 3.0 * m  # total mass available above the floors

        # excess space: x_i = p_i - m, with x_i >= 0 and sum(x) = free
        excess = np.clip(probs - m, 0.0, None)
        s = excess.sum()
        excess = excess * (free / s) if s > 0.0 else np.full(3, free / 3.0)

        # EMA toward the one-hot on the boosted mode
        excess *= 1.0 - ctrl.beta
        excess[idx] += ctrl.beta * free

        self.fwd_frac, self.bwd_frac, self.replay_frac = m + excess

    def step_loss(self, step_type, loss, do_step: bool = True):
        loss.backward()
        if not do_step:
            return  # mid-accumulation: keep piling up gradients, don't clip/step yet

        pre_clip = torch.nn.utils.clip_grad_norm_(
            self.gfn_model.parameters(), self.args.gradient_norm_clip).item()
        if not math.isfinite(pre_clip):
            print(f"Non-finite gradient at {self.step_ind}")
            return  # skip non-finite

        self.optimizers[step_type].step()
        if 'flow' in self.optimizers:
            self.optimizers['flow'].step()

    def fwd_train_step(self,
                       discretizer,
                       return_exp=False,
                       repeats: int = 1,
                       report_losses: bool = False,
                       ):
        mol_batch = next(self.mol_dataset.loader(self.batch_size, mode='graphs', repeats=repeats))
        mol_batch = mol_batch.to(self.device)
        mol_batch.orient_molecule(mode='std')
        init_state = get_gfn_init_state(mol_batch.num_graphs, self.energy_function.data_ndim, self.device)
        mol_batch, log_T_tensor, sg_inds, zps, condition = self.energy_function.condition_samples(
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
                                    )

    def bwd_train_step(self,
                       discretizer,
                       repeats: int,
                       report_losses: bool = False):

        condition, inds, latents, log_reward, mol_batch, traj = self.draw_bwd_sample(repeats)

        loss, loss_dict = get_gfn_backward_loss(self.args.bwd_loss_coeffs,
                                                latents.to(self.device),
                                                self.gfn_model,
                                                log_reward.to(self.device),
                                                discretizer,
                                                mol_batch,
                                                condition=condition,
                                                repeats=repeats,
                                                report_losses=report_losses,
                                                trajectories=traj)

        if self.bwd_sampling_mode == 'dataset':
            self.prior_dataset.update_losses(loss_dict['resid'].abs(),
                                             inds)
        elif self.bwd_sampling_mode == 'prior':
            self.prior_buffer.update_losses(loss_dict['resid'].abs(),
                                            inds)

        return loss, loss_dict

    def replay_train_step(self,
                          discretizer,
                          repeats: int,
                          report_losses: bool = False):

        condition, inds, latents, log_reward, mol_batch, traj = self.draw_replay_sample(repeats)

        loss, loss_dict = get_gfn_backward_loss(self.args.replay_loss_coeffs,
                                                latents.to(self.device),
                                                self.gfn_model,
                                                log_reward.to(self.device),
                                                discretizer,
                                                mol_batch,
                                                condition=condition,
                                                repeats=repeats,
                                                report_losses=report_losses,
                                                trajectories=traj)

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
            energy = mol_batch[self.args.energy_function]
            latents, energy = latents.to(self.device), energy.to(self.device)

        elif self.bwd_sampling_mode == 'prior':
            mol_batch, inds = next(
                self.prior_buffer.loader(
                    batch_size=self.batch_size, mode='graphs',
                    repeats=repeats, return_inds=True,
                    weighted=False,
                    temperature=0.1, beta=1.0))

            latents = mol_batch.latent_params()
            energy = mol_batch[self.args.energy_function]
            latents, energy = latents.to(self.device), energy.to(self.device)
        else:
            assert False, f"sampling method {self.args.sampling} not implemented"
        mol_batch = mol_batch.to(self.device)
        mol_batch, log_T_tensor, sg_inds, zps, condition = self.energy_function.condition_samples(
            mol_batch, repeats=repeats)
        temperature = 10 ** log_T_tensor
        log_reward = self.energy_function.prebuilt_sample_to_reward(mol_batch,
                                                                    temperature)  # relies on the energy terms being attached to the graphs!
        # if self.phase == 1 and self.gfn_model.conditional:
        #     condition = False  # ignore conditioning in data-driven prior training
        return condition, inds, latents, log_reward, mol_batch, traj

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
        energy = mol_batch[self.args.energy_function]
        latents, energy = latents.to(self.device), energy.to(self.device)
        traj = traj.to(self.device)

        mol_batch = mol_batch.to(self.device)
        mol_batch, log_T_tensor, sg_inds, zps, condition = self.energy_function.condition_samples(
            mol_batch, repeats=repeats)
        temperature = 10 ** log_T_tensor
        log_reward = self.energy_function.prebuilt_sample_to_reward(mol_batch,
                                                                    temperature)  # relies on the energy terms being attached to the graphs!
        return condition, inds, latents, log_reward, mol_batch, traj

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
                if self.bwd_sampling_mode == 'dataset':
                    mol_batch = next(self.prior_dataset.loader(batch_size=self.batch_size, mode='graphs'))
                elif self.bwd_sampling_mode == 'prior':
                    mol_batch = next(self.prior_buffer.loader(batch_size=self.batch_size, mode='graphs'))
                else:
                    assert False

                mol_batch = mol_batch.to(self.ema_model.device)
                terminal_state = mol_batch.latent_params()

                mol_batch, log_T_tensor, sg_inds, zps, condition = self.energy_function.condition_samples(
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

        pooled = {k: torch.cat(v, dim=0) for k, v in acc.items()}
        if not self.gfn_model.conditional:
            pooled['log_Z_learned'] = torch.mean(pooled['log_Z_learned'])
        return pooled

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
        metrics.update({f'eval_fwd/{k}': v for k, v in quick_tb_stats(log_pf, log_pb, log_Z_learned, log_r).items()})

        self.log_thermo_properties(arr, fwd_stats, log_T_tensor, log_Z_learned, log_r, metrics, sample_batch, val)

        """Backward TB Stats"""
        log_pf = bwd_stats['log_pfs'].sum(-1)
        log_pb = bwd_stats['log_pbs'].sum(-1)
        log_z = bwd_stats['log_Z_learned']
        log_r = bwd_stats['log_r']
        # parity / Z diagnostics (shared with fwd)
        metrics.update({f'eval_bwd/{k}': v for k, v in quick_tb_stats(log_pf, log_pb, log_z, log_r).items()})

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
        if self.args.repeats > 1:
            metrics['ess'] = log_ess_frac(log_pf, log_pf, repeats=1)  # only useful with repeats > 1
        x, y = next(self.prior_dataset.loader(batch_size=10000, mode='tensors'))
        metrics['wass'] = sliced_wasserstein(sample_batch.latent_params(), x,
                                             n_proj=200)

    def log_thermo_properties(self, arr, fwd_stats, log_T_tensor, log_Z_learned, log_r, metrics, sample_batch, val):
        # energies
        for key in sample_batch.keys():
            if 'energy' in key or 'pot' in key:
                metrics['Mean ' + key] = val(sample_batch[key].mean())

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
        en_func = self.energy_function.energy_function
        sample_is_good = (sample_batch[en_func] < 0) * (sample_batch.packing_coeff > 0.55) * (
                sample_batch.packing_coeff < 0.95)
        metrics["Reasonable Sample Fraction"] = sample_is_good.float().mean().item()

    def evaluation(self):
        metrics = {}
        self.times['eval_step_start'] = time()
        eval_discretizer = lambda bsz: uniform_discretizer(bsz, self.args.eval_T)

        do_figs = self.step_ind % self.args.figs_period == 0

        '''sampling and metrics analysis'''
        fwd_stats, sample_batch = self.fwd_eval_sampling(self.ema_model, eval_discretizer)
        bwd_stats = self.bwd_eval_sampling(eval_discretizer)
        metrics.update(self.log_metrics(fwd_stats, bwd_stats, sample_batch))

        self.times['eval_figs_start'] = time()
        if do_figs:
            x, y = next(self.prior_dataset.loader(batch_size=10000, mode='tensors'))
            # always sample from forward policy
            fig_dict, metrics = eval_figs(fwd_stats,
                                          bwd_stats,
                                          sample_batch,
                                          x,
                                          self.args.energy_function,
                                          metrics,
                                          temperature_conditioning=self.args.temperature_conditioning)
        else:
            fig_dict = {}
        self.times['eval_figs_end'] = time()

        '''logging and wrap up'''
        self.times['eval_wrapup_start'] = time()
        if do_figs:
            adjust_fig_filesize(fig_dict)
            metrics.update(fig_dict)

        metrics.update({'Batch Size': self.batch_size})  # single shared batch size -- train and eval sampling now use the same value
        metrics.update(log_elapsed_times(self.times))
        self.times['eval_wrapup_end'] = time()
        self.times['eval_step_end'] = time()

        for key in metrics.keys():  # cleanup before logging
            if isinstance(metrics[key], np.ndarray):
                metrics[key] = np.nan_to_num(metrics[key])
            elif torch.is_tensor(metrics[key]):
                metrics[key] = torch.nan_to_num(metrics[key])

        if self.phase == 1:
            if metrics['wass'] < self.args.wass_threshold:
                self.phase1to2(metrics)

        elif self.phase == 2:
            if self.metric_tracker.get('bwd', 'under_coverage') < self.args.controller.under_threshold:
                self.phase2to3()

        if self.phase in [2, 3]:  # add samples to off-policy buffer
            self.manage_prior_buffer(sample_batch)
            self.manage_replay_buffer(fwd_stats, sample_batch)

        metrics.update(self.log_buffer_stats())

        return metrics

    def log_buffer_stats(self):
        if self.phase == 1:
            if hasattr(self, 'prior_dataset'):
                buff = self.prior_dataset
            else:
                buff = None
        elif self.phase in [2, 3]:
            if hasattr(self, 'prior_buffer'):
                buff = self.prior_buffer
            else:
                buff = None
        else:
            buff = None

        if buff is not None:
            valid_losses = buff.ema_loss[~torch.isnan(buff.ema_loss)].cpu().numpy()
            metrics = {'prior_buffer_length': len(buff),
                       'prior_buffer_mean_steps': torch.nanmean(buff.select_counts.float()).item(),
                       'prior_buffer_median_steps': torch.nanmedian(buff.select_counts.float()).item(),
                       'prior_buffer_mean_loss': torch.nanmean(buff.ema_loss).item(),
                       'prior_buffer_median_loss': torch.nanmedian(buff.ema_loss).item(),
                       'prior_buffer_step_hist': wandb.Histogram(buff.select_counts.cpu().numpy()),
                       'prior_buffer_energy_hist': wandb.Histogram(buff.batch[self.args.energy_function].cpu().numpy())
                       }
            if len(valid_losses) > 0:
                metrics['prior_buffer_loss_hist'] = wandb.Histogram(np.clip(np.log10(valid_losses), min=-1, max=3))
        else:
            metrics = {}

        if hasattr(self, 'replay_buffer'):
            valid_replay_losses = self.replay_buffer.ema_loss[~torch.isnan(self.replay_buffer.ema_loss)].cpu().numpy()
            metrics.update({
                'replay_buffer_length': len(self.replay_buffer),
                'replay_buffer_mean_steps': torch.nanmean(self.replay_buffer.select_counts.float()).item(),
                'replay_buffer_median_steps': torch.nanmedian(self.replay_buffer.select_counts.float()).item(),
                'replay_buffer_mean_loss': torch.nanmean(self.replay_buffer.ema_loss).item(),
                'replay_buffer_median_loss': torch.nanmedian(self.replay_buffer.ema_loss).item(),
                'replay_buffer_step_hist': wandb.Histogram(self.replay_buffer.select_counts.cpu().numpy()),
                'replay_buffer_energy_hist': wandb.Histogram(
                    self.replay_buffer.batch[self.args.energy_function].cpu().numpy())
            })
            if len(valid_replay_losses) > 0:
                metrics['replay_buffer_loss_hist'] = wandb.Histogram(
                    np.clip(np.log10(valid_replay_losses), min=0, max=3))
        return metrics

    def manage_prior_buffer(self, sample_batch):
        if not hasattr(self, 'prior_buffer'):
            self.prior_buffer = CrystalBuffer(
                sample_batch,
                device='cpu',
                max_z_prime=max(self.args.z_primes),
                x_fn=None,  # 'latent_params',
                y_fn=self.args.energy_function
            )

        num_bwd_steps = self.bwd_step_delta()

        # always churn at least a little bit
        n_churn = max(1000,
                      int((num_bwd_steps / self.args.prior_buffer.mean_lifetime) * self.batch_size))
        n_to_add = min(self.args.eval_num_samples, n_churn)  # cap unrelated to GPU batch size -- eval_batch_size is retired, this is just a churn-rate limiter
        headroom = max(0, self.args.prior_buffer.max_size - len(self.prior_buffer))

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

        space_needed = max(0, len(self.prior_buffer) + n_to_add - self.args.prior_buffer.max_size)
        if space_needed > 0:
            self.prior_buffer.purge_lowest(
                space_needed,
                quantile=0.25,
                loss_floor=10.0,
                min_visits=5,
            )

        remaining_budget = n_to_add  # - len(bad_inds)

        # sample from prior
        if remaining_budget > 0: # todo track conditioning through this step so rewards are consistent
            metrics, sample_batch = self.sample_from_prior(remaining_budget)
            reward = metrics['log_r']
            good_inds = torch.argwhere(reward > self.args.prior_buffer.reward_min).flatten()
            if good_inds.numel() > 0:
                batch_to_add = sample_batch.subsample_new_batch(good_inds)
                self.prior_buffer.add(batch_to_add)

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
        overweighted (high positive residual) terminal states, so they can later
        be exactly replayed (get_traj_replay) rather than re-sampled backward.

        Unlike the prior buffer, every entry here comes from the same residual
        criterion, so there's no source mix to budget between -- but we still
        pace admission by mean_lifetime turnover so a sudden glut of overweighted
        trajectories can't flood the buffer in a single eval cycle.

        On top of the residual-driven admission/eviction above, a small amount of
        pure random churn is swapped in every call (paced by replay_buffer.random_churn_rate
        per replay train step, mirroring how the prior buffer paces on num_bwd_steps) so the
        buffer can't become dominated by a handful of rare, mutually-correlated high-residual
        events. Random adds are still restricted to the top quartile of this batch's residuals,
        so churn doesn't dilute the buffer with well-covered "good" samples.
        """
        log_r = fwd_stats['log_r']
        log_pf = fwd_stats['log_pf']
        log_pb = fwd_stats['log_pb']
        log_Z_learned = fwd_stats['log_Z_learned'] if 'log_Z_learned' in fwd_stats else fwd_stats['log_Z']

        resid = ((log_pf - log_pb) - (log_r - log_Z_learned)).cpu().abs()

        floor = 0.1  # low; doubles as below-capacity admission gate

        # two-sided admission: though in practice it's almost always positive residuals
        elig = torch.argwhere(resid > floor).flatten()
        elig = elig[torch.argsort(resid[elig], descending=True)]
        cand_resid = resid[elig]
        flow_states = fwd_stats['flow_states'].cpu()

        # --- bootstrap ---
        if not hasattr(self, 'replay_buffer'):
            if elig.numel() == 0:
                return
            add_inds = elig[:self.args.replay_buffer.max_size]
            self.replay_buffer = CrystalBuffer(
                sample_batch.subsample_new_batch(add_inds),
                device='cpu',
                max_z_prime=max(self.args.z_primes),
                x_fn=None,
                y_fn=self.args.energy_function,
                traj=flow_states[add_inds],
                init_loss=resid[add_inds],
            )
            return

        # --- unconditional toxic eviction: strictly overfit incumbents ---
        toxic = torch.argwhere(self.replay_buffer.ema_loss < floor).flatten()
        if toxic.numel() > 0:
            self.replay_buffer.purge_by_index(toxic)  # buffer reindexes; read ema_loss AFTER this

        # --- fill freed + spare slots, then beat-the-min for the rest ---
        headroom = max(0, self.args.replay_buffer.max_size - len(self.replay_buffer))
        free = elig[:headroom]
        over_idx = elig[headroom:]
        over_resid = cand_resid[headroom:]

        drop_pos = torch.empty(0, dtype=torch.long)
        if over_idx.numel() > 0:
            order = torch.argsort(self.replay_buffer.ema_loss)  # ascending, worst-first
            inc_loss = self.replay_buffer.ema_loss[order]
            k = min(over_resid.numel(), order.numel())
            beats = (over_resid[:k] > inc_loss[:k]).long()  # strict: no churn on ties
            n_swap = int(torch.cummin(beats, dim=0).values.sum())  # leading True run
            over_idx = over_idx[:n_swap]
            drop_pos = order[:n_swap]

        if drop_pos.numel() > 0:
            self.replay_buffer.purge_by_index(drop_pos)

        add_inds = torch.cat([free, over_idx])
        if add_inds.numel() > 0:
            self.replay_buffer.add(
                sample_batch.subsample_new_batch(add_inds),
                traj=flow_states[add_inds],
                init_loss=resid[add_inds],
            )

        # --- random churn: guard against domination by rare correlated events ---
        num_replay_steps = self.replay_step_delta()
        n_churn = int(num_replay_steps * self.args.replay_buffer.random_churn_rate)
        n_churn = min(n_churn, len(self.replay_buffer))
        if n_churn > 0:
            purge_idx = torch.randperm(len(self.replay_buffer))[:n_churn]
            self.replay_buffer.purge_by_index(purge_idx)

            # only draw random adds from this batch's top quartile of residuals,
            # so churn doesn't dilute the buffer with well-covered "good" samples
            top_quartile = torch.quantile(resid, 0.75)
            churn_cand = torch.argwhere(resid >= top_quartile).flatten()
            n_add = min(n_churn, churn_cand.numel())
            if n_add > 0:
                add_choice = churn_cand[torch.randperm(churn_cand.numel())[:n_add]]
                self.replay_buffer.add(
                    sample_batch.subsample_new_batch(add_choice),
                    traj=flow_states[add_choice],
                    init_loss=resid[add_choice],
                )

    def sample_from_prior(self, num_samples):
        "sample from prior"
        eval_discretizer = lambda bsz: uniform_discretizer(bsz, self.args.eval_T)
        metrics, sample_batch = self.fwd_eval_sampling(self.prior_model,
                                                       eval_discretizer,
                                                       override_num_samples=num_samples)
        return metrics, sample_batch

    def grow_prior_buffer(self):
        if not hasattr(self, 'prior_buffer'):
            buffer_length = 0
        else:
            buffer_length = len(self.prior_buffer)

        missing = self.args.prior_buffer.max_size - buffer_length
        num_samples = min(self.args.prior_buffer.min_size, missing)
        if num_samples > 0:
            metrics, sample_batch = self.sample_from_prior(num_samples)

            if not hasattr(self, 'prior_buffer'):
                self.prior_buffer = CrystalBuffer(
                    sample_batch,
                    device='cpu',
                    max_z_prime=max(self.args.z_primes),
                    x_fn=None,  # 'latent_params',
                    y_fn=self.args.energy_function
                )
            else:
                self.prior_buffer.add(sample_batch)

    @torch.no_grad()
    def fwd_eval_sampling(self, model, eval_discretizer, override_num_samples: Optional[int] = None):
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
                mol_batch = next(self.mol_dataset.loader(bsz, mode='graphs'))
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

        self.times['eval_sampling_end'] = time()

        return pooled, sample_batch

    def phase1to2(self, metrics):
        print("Hit initial KLD threshold. Starting prior thermalization.")
        self.hit_prior = True
        self.phase = 2
        self.bwd_sampling_mode = 'prior'

        "adjust loss coefficients"
        self.bwd_loss_schedule['mle'] = [(0, 1.0),
                                         (self.step_ind, 0.0)]  # turn off mle
        self.bwd_loss_schedule['bwd_tb_z'] = [(0, 2.0),
                                              (self.step_ind, 1.0)]  # start log Z learning
        self.bwd_loss_schedule['tb'] = [(0, 0.0),
                                        (self.step_ind, 1.0)]  # use backward TB for retention

        if (not self.gfn_model.full_flow) and (not self.gfn_model.conditional):
            empirical_Z = metrics['eval_fwd/jensen_z']
            self.gfn_model.flow_model.scalar.data.fill_(empirical_Z)  # warm start at the target value
            self.ema_model.flow_model.scalar.data.fill_(empirical_Z)
        else:
            pass
            # assert False, "put in a quick script to train flow[:, 0] to log Z if we're going to use this"

        # self.bwd_loss_schedule['traj_grads'] = [(0, 1.0),
        #                                         (self.step_ind, 1),
        #                                         (self.step_ind, 0.0)]  # deactivate reward grads at TB stage

        "refresh optimization machinery"
        self.bwd_loss_monitor.reset()  # new losses
        self.fwd_loss_monitor.fire_cooldown(self.step_ind)
        self.bwd_loss_monitor.fire_cooldown(self.step_ind)
        self.replay_loss_monitor.fire_cooldown(self.step_ind)
        self.fused_loss_monitor.fire_cooldown(self.step_ind)

        self.lr_warmup_finished = False
        self.init_schedulers_optimizers()  # re-initialize optimizers for new losses
        self.set_loss_coeffs()  # take effect immediately

        "save checkpoint"
        self.save_checkpoint('prior')
        self.prior_model = deepcopy(self.ema_model)
        self.prior_model.eval()
        self.set_loss_coeffs()

        # delete phase 1 checkpoint - it converged!
        os.remove(self._checkpoint_path('best'))

    def phase2to3(self):
        print("Thermalization complete. Starting on-policy equilibration.")
        "adjust loss and balancing coefficients"
        self.bwd_loss_schedule['bwd_tb_z'] = [(0, 1.0),
                                              (self.step_ind, 0.0)]  # no off-policy log Z
        self.fwd_loss_schedule['tb'] = [(0, 0.0),
                                        (self.step_ind, 1.0)]  # on-policy TB ACTIVATE
        self.replay_loss_schedule['tb'] = [(0, 0.0),
                                           (self.step_ind, 1.0)]  # replay TB ACTIVATE
        self.replay_loss_schedule['bwd_tb_z'] = [(0, 1.0),
                                                 (self.step_ind, 0.0)]  # no off-policy log Z
        # self.fwd_loss_schedule['fwd_tb_z'] = [(0, 1.0),
        #                                          (self.step_ind, 2.0)]  # ONLY log Z training on-policy

        "set cooldown"
        self.fwd_loss_monitor.fire_cooldown(self.step_ind)
        self.bwd_loss_monitor.fire_cooldown(self.step_ind)
        self.replay_loss_monitor.fire_cooldown(self.step_ind)
        self.fused_loss_monitor.fire_cooldown(self.step_ind)

        self.fwd_frac, self.bwd_frac, self.replay_frac = (
            self.args.controller.min_mode_frac, 1 - 2 * self.args.controller.min_mode_frac,
            self.args.controller.min_mode_frac)
        self.set_loss_coeffs()

        self.bwd_frac = 1.0
        self.fwd_frac = 0.0
        self.replay_frac = 0.0

        self.phase = 3
        self.save_checkpoint('thermalized')


if __name__ == '__main__':
    modeller = Modeller()
    modeller.train()
