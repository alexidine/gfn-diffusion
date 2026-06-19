import gc
import os
from collections import defaultdict
from copy import deepcopy

from energy_sampling.eval.evaluations import to_loggable, sliced_wasserstein, adjust_fig_filesize, eval_figs, \
    log_ess_frac
from mxtaltools.common.utils import log_rescale_positive

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
from energy_sampling.buffer import SimpleDataset
from energy_sampling.eval.utils import sample_eval_fwd_trajs, LossSpikeMonitor
from energy_sampling.utils import is_cuda_oom, get_annealing_factor, \
    parse_loss_schedules, dict2namespace, update_loss_schedule, \
    atomic_save, get_discretizer, log_elapsed_times, MetricTracker, quick_tb_stats, uniform_discretizer, logmeanexp
from gflownet_losses import get_gfn_forward_loss, get_gfn_backward_loss
from models import GFN
from mxtaltools.common.training_utils import flatten_wandb_params
from mxtaltools.dataset_utils.utils import collate_data_list
from utils import get_train_args, get_gfn_init_state, set_seed, \
    update_ema

MODELLER_STATE_DEFAULTS = {
    'step_ind': 0,
    'phase': 1,
    'batch_size': 1,
    'fwd_to_bwd_ratio': 1.0,
    'increasing_loss_cooldown': {},
    'loss_records': {},
    'global_action_cooldown': 0,
    'lr_cut_count': 0,
    'lr_warmup_finished': False,
    'hit_init_kld': False,
    'grow_buffer': False,
    'fwd_loss_schedule': {},
    'bwd_loss_schedule': {},
}


class Modeller:
    def __init__(self):
        self.step_ind = None
        self.args = get_train_args()
        torch.cuda.set_per_process_memory_fraction(self.args.cuda_memory_fraction, device=0)
        torch.cuda.init()  # create context with the cap already in place

        self.fwd_loss_monitor = LossSpikeMonitor(window=25, warmup=50, cooldown=100, ceiling_factor=50.0)
        self.bwd_loss_monitor = LossSpikeMonitor(window=25, warmup=50, cooldown=100, ceiling_factor=50.0)

        set_seed(self.args.seed)
        if 'SLURM_PROCID' in os.environ:
            self.args.seed += int(os.environ["SLURM_PROCID"])

        if self.args.both_ways and self.args.bwd:
            self.args.bwd = False

        config = self.args.__dict__
        config["Experiment"] = "{args.energy}"
        self.run_name = str(self.args.tag) + '_' + str(self.args.run_name)

        self.times = {}
        self.device = self.args.device
        self.init_train_constants()

    def init_train_constants(self):
        for k, v in MODELLER_STATE_DEFAULTS.items():
            if k in self.args.__dict__:
                setattr(self, k, self.args.__dict__[k])
            else:
                setattr(self, k, v)

        self.rolling_tracker = MetricTracker(period=100)

        if self.args.anchor_fwd_bwd:
            self.args.fwd_to_bwd_ratio = 1.0E-6

    def _get_modeller_state_dict(self):
        return {k: getattr(self, k) for k in MODELLER_STATE_DEFAULTS}

    def _set_modeller_state_dict(self, state):
        for k in MODELLER_STATE_DEFAULTS:
            setattr(self, k, state[k])

    def save_checkpoint(self, tag: str):
        """
        tag: 'best' | 'hit_prior' | 'thermalized' | 'final'
        """
        checkpoint = {
            'tag': tag,
            'run_name': self.run_name,
            'gfn_config': self.gfn_config,  # store once, reload from here
            'model_train': self.gfn_model.state_dict(),
            'model_eval': self.ema_model.state_dict(),
            'modeller_state': self._get_modeller_state_dict(),
            'metrics': self.rolling_tracker.state_dict(),
            'optimizers': {k: opt.state_dict() for k, opt in self.optimizers.items()},
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
        self.rolling_tracker.load_state_dict(checkpoint.get('metrics', {}))
        if load_opt_state:
            self.load_optimizer_state(checkpoint)

    def load_optimizer_state(self, checkpoint):
        for key, opt in self.optimizers.items():
            opt.load_state_dict(checkpoint['optimizers'][key])

    def load_model_state(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.gfn_model.load_state_dict(checkpoint['model_train'])
        self.ema_model.load_state_dict(checkpoint['model_eval'])
        self.gfn_model.train()
        self.ema_model.eval()

    def _checkpoint_path(self, tag: str) -> str:
        return f'checkpoints/{self.run_name}_{tag}.pt'

    def train_logic(self, it):
        do_forward = False
        do_backward = False
        if self.args.both_ways:
            p_forward = self.args.fwd_to_bwd_ratio / (self.args.fwd_to_bwd_ratio + 1)
            if it == 0:
                do_fwd = True
            elif self.args.fwd_to_bwd_ratio == 1:
                do_fwd = it % 2 == 0  # always do fwd first
            else:
                do_fwd = np.random.choice([0, 1], 1, p=[1 - p_forward, p_forward])

            if do_fwd:
                do_forward = True
            else:
                do_backward = True

        elif self.args.bwd:  # backward ONLY
            do_backward = True

        else:  # forward ONLY
            do_forward = True

        return do_forward, do_backward

    def increment_batch_size(self):
        if self.batch_size < self.args.max_batch_size:
            new_batch_size = min(self.args.max_batch_size,
                                 max(self.batch_size + 1, int(self.batch_size * self.args.batch_growth_increment,
                                                              )))
            self.batch_size = new_batch_size  # gradually increment batch size

    def step_lr_schedule(self):
        lr = self.optimizers['fwd'].param_groups[0]['lr']
        if not self.lr_warmup_finished:
            self.schedulers['policy_1'].step()
            self.schedulers['policy_1b'].step()

            if lr >= self.args.lr_policy:
                self.lr_warmup_finished = True

        elif lr > self.args.min_lr:
            self.schedulers['policy_2'].step()
            self.schedulers['policy_2b'].step()

        self.schedulers['flow'].step()

        return lr

    def ten_step_reporting(self):
        metrics = {}
        metrics.update(self.rolling_tracker.snapshot())

        for opt_type in ['fwd', 'bwd', 'flow']:
            metrics.update({f'lr_{opt_type}': self.optimizers[opt_type].param_groups[0]['lr']})

        metrics['Fwd to Bwd Ratio'] = self.args.fwd_to_bwd_ratio
        metrics.update(log_elapsed_times(self.times))
        return metrics

    def set_loss_coeffs(self):
        if self.step_ind == 0:
            self.fwd_loss_schedule = parse_loss_schedules(self.args.fwd_loss_coeffs)
            self.bwd_loss_schedule = parse_loss_schedules(self.args.bwd_loss_coeffs)

            self.args.fwd_loss_coeffs = dict2namespace({k: 0.0 for k in self.fwd_loss_schedule})
            self.args.bwd_loss_coeffs = dict2namespace({k: 0.0 for k in self.bwd_loss_schedule})

        update_loss_schedule(self.step_ind, self.fwd_loss_schedule, self.args.fwd_loss_coeffs.__dict__)
        update_loss_schedule(self.step_ind, self.bwd_loss_schedule, self.args.bwd_loss_coeffs.__dict__)

    def get_conditioning_dim(self):  # TODO rewrite wholesale
        conditions_dim = 0
        if self.args.temperature_conditioning:
            conditions_dim += 1
        if self.args.sg_conditioning:
            conditions_dim += 237
        if self.args.zp_conditioning:
            conditions_dim += 1
        return conditions_dim

    def init_energy_function(self):  # todo update / simplify
        energy_config = {
            'device': self.device,
            'energy_function': self.args.energy_function,
            'temperature_conditioning': self.args.temperature_conditioning,
            'temperature': self.args.temperature,
            'density_coeff': self.args.energy_density_coeff,
            'lj_coeff': self.args.energy_lj_coeff,
            'molecule_conditioning': self.args.molecule_conditioning,
            'sg_conditioning': self.args.sg_conditioning,
            'space_groups': self.args.space_groups,
            'bounding_coeff': self.args.bounding_coeff,
            'reduction_coeff': self.args.reduction_coeff,
            'z_primes': self.args.z_primes,
            'max_z_prime': max(self.args.z_primes),
            'zp_conditioning': self.args.zp_conditioning,
            'mlip_path': self.args.mlip_path,
            'reward_range': self.args.reward_range,
            'lj_rescale': 1,
        }
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
            **vars(self.args.model),
        )

    def init_gfn(self):
        reload = False  # TODO cleanup / unify model state
        if self.args.checkpoint_path is not None:
            reload = True
            print(f"Loading model from checkpoint {self.args.checkpoint_path}")
            self.load_model_and_state(self.args.checkpoint_path)

        # todo rewrite hash logic
        elif os.path.exists(f'checkpoints/{self.run_name}_model_train.pt') and self.args.continue_from_checkpoint:
            print("Reloading automatically from this prior checkpoint with same run name")
            reload = True
            reload_path = f'checkpoints/{self.run_name}_model_train.pt'
        # todo rewrite hash logic
        elif os.path.exists(
                f'checkpoints/{self.run_name}_model_train_hit_prior.pt') and self.args.continue_from_hit_prior:
            print("Reloading from checkpoint converged on prior")
            reload = True
            reload_path = f'checkpoints/{self.run_name}_model_train_hit_prior.pt'

        if not reload:
            self.gfn_config = self._build_gfn_config()
            self.gfn_model = GFN(**self.gfn_config).to(self.device)
            self.ema_model = deepcopy(self.gfn_model)

    def init_schedulers_optimizers(self):
        init_fwd_lr = self.args.lr_policy / self.args.lr_warmup_ratio
        init_bwd_lr = self.args.lr_back / self.args.lr_warmup_ratio
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

            return plist

        flow_params = self.gfn_model.flow_model.parameters()

        self.optimizers = {}
        weight_decay = self.args.weight_decay if self.args.use_weight_decay else 0
        self.optimizers['fwd'] = torch.optim.Adam(get_policy_params(self.gfn_model), init_fwd_lr,
                                                  weight_decay=weight_decay)

        self.optimizers['bwd'] = torch.optim.Adam(get_policy_params(self.gfn_model), init_bwd_lr,
                                                  weight_decay=weight_decay)
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

        flow_annealing_lambda = get_annealing_factor(1,
                                                     0.1,
                                                     self.args.lr_anneal_time,
                                                     10)
        self.schedulers['flow'] = lr_scheduler.MultiplicativeLR(self.optimizers['flow'],
                                                                lr_lambda=lambda epoch: flow_annealing_lambda)

    def init_prior_dataset(self):
        prior_data = torch.load(self.args.prior_path, weights_only=False)
        self.prior_dataset = SimpleDataset(prior_data['equalized_prior'],
                                           device='cpu',
                                           max_z_prime=max(self.args.z_primes),
                                           x_fn=None,  # 'latent_params',
                                           y_fn=self.args.energy_function
                                           )

    def init_mol_dataset(self):
        data_list = torch.load(self.args.molecules_path, weights_only=False)
        self.mol_dataset = SimpleDataset(data_list,
                                         device='cpu',
                                         max_z_prime=max(self.args.z_primes))

        if self.args.test_molecules_path is not None:
            data_list = torch.load(self.args.test_molecules_path, weights_only=False)
            self.test_mol_dataset = SimpleDataset(data_list,
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

            # opt init
            self.init_schedulers_optimizers()

            # data init
            self.init_prior_dataset()
            self.init_mol_dataset()

            oomed_out = False
            combo_loss_record = []
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

                    if (not oomed_out or (self.step_ind % 500 == 0)) and self.args.grow_batch_size:
                        self.increment_batch_size()

                except (RuntimeError, ValueError) as e:  # if we do hit OOM, slash the batch size
                    oomed_out = self.handle_train_epoch_error(
                        e, oomed_out, step_type)
                self.times['train_step_end'] = time()

                # evaluation work
                if (self.step_ind % self.args.eval_period == 0 and self.step_ind > 0) or self.step_ind == 50:
                    metrics.update(self.training_eval())

                # train monitoring
                if self.step_ind % 10 == 0:
                    lr = self.step_lr_schedule()
                    metrics.update(self.ten_step_reporting())
                    self.monitor_losses(combo_loss_record, current_loss, step_type)
                    if combo_loss_record[-1] <= np.amin(combo_loss_record):
                         self.save_checkpoint('best')

                if len(metrics) > 0:
                    wandb.log(metrics, step=self.step_ind, commit=True)

                if self.step_ind % 50 == 0:  # save running model
                    self.save_checkpoint('running')

            self.save_checkpoint('final')

    def monitor_losses(self, combo_loss_record, current_loss, step_type):
        if current_loss is not None:
            if step_type[0]:
                trig = self.fwd_loss_monitor.record(current_loss, self.step_ind)
            else:
                trig = self.bwd_loss_monitor.record(current_loss, self.step_ind)

            combo_loss_record.append(self.fwd_loss_monitor.best_loss + self.bwd_loss_monitor.best_loss)

            if trig:
                self.fire_loss_spike()

    def fire_loss_spike(self):
        if os.path.exists(self._checkpoint_path('running')):
            self.load_model_state(self._checkpoint_path('running'))

        if not hasattr(self, 'lr_cut_count'):
            self.lr_cut_count = 1
        else:
            self.lr_cut_count += 1

        lr_cut_val = 0.75

        for key, opt in self.optimizers.items():
            opt.state = defaultdict(dict)  # wipe momentum buffers too
            for g in opt.param_groups:
                if g['lr'] > self.args.min_lr:
                    g['lr'] = max(g['lr'] * lr_cut_val, self.args.min_lr)

        if self.lr_cut_count > 0:
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
                   step_type,
                   ):
        do_forward, do_backward = step_type

        discretizer = get_discretizer(self.args.integrator)

        self.optimizers['flow'].zero_grad(set_to_none=True)

        if do_forward:
            self.optimizers['fwd'].zero_grad(set_to_none=True)
            loss, crystal_batch, loss_dict = self.fwd_train_step(
                discretizer,
                return_exp=True,
                repeats=self.args.repeats,
                report_losses=True
            )
            del crystal_batch

        elif do_backward:
            self.optimizers['bwd'].zero_grad(set_to_none=True)
            loss, loss_dict = self.bwd_train_step(
                discretizer,
                repeats=self.args.repeats,
                report_losses=True)
        else:
            assert False

        self.step_loss(do_backward, do_forward, loss)

        if self.step_ind % 5 == 0:
            st = 'fwd' if step_type[0] else 'bwd'
            stats = quick_tb_stats(loss_dict['log_pf'], loss_dict['log_pb'],
                                   loss_dict['log_Z'], loss_dict['log_r'])
            stats.update({k: v.item() for k, v in loss_dict.items() if k not in
                          ['log_pf', 'log_pb', 'log_Z', 'log_r']})
            stats.update({'loss': loss.cpu().detach().item()})
            self.rolling_tracker.update(st, stats, self.step_ind)

        # skip_step = False
        # if self.phase == 2:
        #     if self.bwd_tb_norm <= self.args.thermalization_conv_eps:  # hit stage 2 convergence criteria
        #         self.phase2to3(self.args.min_fwd_bwd_ratio)
        #
        # if self.phase == 3:
        #     skip_step = self.update_controller(self.step_ind, do_backward, skip_step)

        # loss_dict_cpu = {step_type + "_loss/" + key: (value.cpu().detach().numpy() if torch.is_tensor(value) else value)
        #                  for key, value in
        #                  loss_dict.items()}

        torch.cuda.synchronize()
        self.update_ema_model()
        return loss.cpu().detach().item()

    def update_controller(self, it, do_backward, skip_step, eps=1e-3):
        update_this_step = it % 20 == 0
        slope_ceil = self.args.thermalization_slope_err
        intercept_ceil = self.args.thermalization_intercept_err

        slope_err = self.bwd_slope_err
        intercept_err = self.bwd_intercept_err

        # forward errors (should be smoothed upstream)
        fwd_slope_err = self.fwd_slope_err
        fwd_intercept_err = self.fwd_intercept_err

        # backward must be at least as good as forward
        slope_ceil = min(slope_ceil, fwd_slope_err)
        intercept_ceil = min(intercept_ceil, fwd_intercept_err)

        if update_this_step:
            metric = max(
                slope_err / max(slope_ceil, eps),
                intercept_err / max(intercept_ceil, eps)
            )

            if metric > 1.2:
                # buffer unsafe → backward dominance
                self.args.fwd_to_bwd_ratio *= 0.5
                if not do_backward:
                    skip_step = True  # don't do backprop on forward if we're out of range
            elif metric < 0.8:
                # buffer safe → allow forward normalization
                self.args.fwd_to_bwd_ratio *= 1.05
            # else - safe region

        self.args.fwd_to_bwd_ratio = np.clip(
            self.args.fwd_to_bwd_ratio, self.args.min_fwd_bwd_ratio,
            self.args.max_fwd_bwd_ratio)  # need even enough ratios to get reasonable updates to the metrics

        return skip_step

    def step_loss(self, do_backward, do_forward, loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gfn_model.parameters(),
                                       self.args.gradient_norm_clip)  # gradient clipping
        if do_forward:
            self.optimizers['fwd'].step()
            self.optimizers['flow'].step()
        elif do_backward:
            self.optimizers['bwd'].step()
            self.optimizers['flow'].step()

    def fwd_train_step(self,
                       discretizer,
                       return_exp=False,
                       repeats: int = 1,
                       report_losses: bool = False,
                       ):
        mol_batch = next(self.mol_dataset.loader(self.batch_size, mode='graphs'))
        mol_batch = mol_batch.to(self.device)
        mol_batch.orient_molecule(mode='std')
        init_state = get_gfn_init_state(self.batch_size, self.energy_function.data_ndim, self.device)
        mol_batch, log_T_tensor, sg_inds, zps, condition = self.energy_function.condition_samples(
            mol_batch)

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
        with torch.no_grad():
            if self.args.bwd_sampling_mode == 'dataset':
                latents, energy = next(self.prior_dataset.loader(batch_size=self.batch_size, mode='tensors', repeats=repeats))
            elif self.args.bwd_sampling_mode == 'model':
                latents, energy = self.prior_model.sample_tensors(self.batch_size, repeats=repeats)
            else:
                assert False, f"sampling method {self.args.sampling} not implemented"
            latents, energy = latents.to(self.device), energy.to(self.device)
            mol_batch = next(self.mol_dataset.loader(batch_size=self.batch_size, mode='graphs', repeats=repeats))
            mol_batch = mol_batch.to(self.device)
            mol_batch, log_T_tensor, sg_inds, zps, condition = self.energy_function.condition_samples(
                mol_batch)
            log_reward = -energy / log_T_tensor.exp()
            if self.phase == 1:
                condition = False  # ignore conditioning in data-driven prior training
        loss, loss_dict = get_gfn_backward_loss(self.args.bwd_loss_coeffs,
                                                latents.to(self.device),
                                                self.gfn_model,
                                                log_reward.to(self.device),
                                                discretizer,
                                                mol_batch,
                                                condition=condition,
                                                repeats=repeats,
                                                report_losses=report_losses)

        return loss, loss_dict

    def handle_train_epoch_error(self, e, oomed_out, step_type):
        print(f"Caught error: {str(e)}")
        if is_cuda_oom(e):
            print("OOMED!")
            if self.step_ind == 0:
                return oomed_out

            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)

            # break reference cycles
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

            self.batch_size = max(1, int(self.batch_size * 0.95))
            if self.batch_size <= 1:
                raise RuntimeError("Cascading OOM Failure")

            if self.batch_size <= 1:
                raise RuntimeError("Cascading OOM Failure")
            print(f"Reducing batch size to {self.batch_size}")

            gc.collect()
            torch.cuda.empty_cache()

            oomed_out = True
        else:
            raise e  # will simply raise error if other or if training on CPU
        return oomed_out

    @torch.no_grad()
    def bwd_eval_sampling(
            self,
            discretizer, ):
        acc = defaultdict(list)
        samples = 0
        while samples < self.args.eval_num_samples:
            try:
                mol_batch = next(self.prior_dataset.loader(batch_size=self.args.eval_batch_size, mode='graphs'))
                mol_batch = mol_batch.to(self.ema_model.device)

                mol_batch.box_analysis()

                terminal_state = mol_batch.latent_params()

                mol_batch, log_T_tensor, sg_inds, zps, condition = self.energy_function.condition_samples(
                    mol_batch)

                log_r = self.energy_function.prebuilt_sample_to_reward(mol_batch,
                                                                       temperature=log_T_tensor.exp())

                terminal_state = terminal_state.to(self.ema_model.device)
                condition = condition.to(self.ema_model.device)
                if self.phase == 1:
                    condition = False

                (backward_flow_states, b_log_pfs, b_log_pbs, log_z,
                 b_means_f, b_vars_f, b_means_b, b_vars_b) = self.ema_model.get_traj_bwd(
                    terminal_state, discretizer, condition, mol_batch, return_gauss_params=True)

                samples += mol_batch.num_graphs

            except (RuntimeError, ValueError) as e:
                self._shrink_eval_batch_on_oom(e)
                continue

            cpu = lambda t: t.cpu().detach()
            acc['flow_states'].append(cpu(backward_flow_states))
            acc['log_pfs'].append(cpu(b_log_pfs))
            acc['log_pbs'].append(cpu(b_log_pbs))
            acc['means_f'].append(cpu(b_means_f))
            acc['logvars_f'].append(cpu(b_vars_f))
            acc['means_b'].append(cpu(b_means_b))
            acc['logvars_b'].append(cpu(b_vars_b))
            acc['log_r'].append(cpu(log_r))
            acc['log_Z_learned'].append(cpu(log_z))
            acc['packing_coeff'].append(cpu(mol_batch.packing_coeff))

        pooled = {k: torch.cat(v, dim=0) for k, v in acc.items()}
        return pooled

    def log_metrics(self, fwd_stats, bwd_stats, sample_batch):

        metrics = {}
        arr = lambda t: t.cpu().detach().numpy()
        val = lambda t: t.cpu().detach().item()

        log_r = fwd_stats['log_r']
        log_pf = fwd_stats['log_pfs'].sum(-1)
        log_pb = fwd_stats['log_pbs'].sum(-1)
        log_Z_learned = fwd_stats['log_Z']
        log_T_tensor = fwd_stats['log_T_tensor']
        metrics.update({f'fwd_{k}': v for k, v in quick_tb_stats(log_pf, log_pb, log_Z_learned, log_r).items()})

        # energies
        for key in sample_batch.keys():
            if 'energy' in key or 'pot' in key:
                metrics['Mean ' + key] = val(sample_batch[key].mean())

        # physical properties
        metrics['Mean Packing Coeff'] = val(sample_batch.packing_coeff.mean())
        metrics['Packing Coeff'] = arr(sample_batch.packing_coeff.clip(max=2))
        metrics['Reduction Energy'] = arr(sample_batch.reduction_en)
        metrics['Reduced Valid Fraction'] = np.mean(arr(sample_batch.reduction_en) < 1e-1)

        # conditions
        metrics['Crystal Mean Log Temperature'] = val(log_T_tensor.mean())
        metrics['Crystal Log Temperature'] = arr(log_T_tensor)

        # training metrics
        metrics['Mean Sample Energy'] = val(sample_batch.gfn_energy.mean())
        metrics['Sample Energy'] = arr(sample_batch.gfn_energy.clip(max=50))
        metrics['Scaled LJ'] = arr(log_rescale_positive(sample_batch.lj))

        metrics['Mean Sample Reward'] = val(log_r.mean())
        metrics['Sample Reward'] = arr(log_r.clip(min=-50))

        metrics['Empirical log Z'] = val(fwd_stats['log_Z'])
        metrics['Empirical log Z LB'] = val(fwd_stats['log_Z_lb'])
        metrics['log Z learned'] = val(log_Z_learned)

        def dump_numeric(metrics, prefix, obj):
            d = obj if isinstance(obj, dict) else vars(obj)
            for k, v in d.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metrics[f'{prefix}{k}'] = v

        dump_numeric(metrics, 'energy_func/', self.energy_function)
        dump_numeric(metrics, 'loss_coeffs/fwd_', self.args.fwd_loss_coeffs)
        dump_numeric(metrics, 'loss_coeffs/bwd_', self.args.bwd_loss_coeffs)

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

        #
        # prior_sample = sample_backward_prior(self.args, buffer, sample_batch, len(std_params))
        #
        # # this isn't SG conditioned, but that's OK because we're not really using it anymore anyway
        # prior_coverage = get_dimwise_coverage(std_params, prior_sample.to('cpu'),
        #                                       n_bins=24, cmin=1, tau=1 / 0.05)
        # lattice_features = ['cell_a', 'cell_b', 'cell_c',
        #                     'cell_alpha', 'cell_beta', 'cell_gamma',
        #                     'aunit_x', 'aunit_y', 'aunit_z',
        #                     'orientation_1', 'orientation_2', 'orientation_3']
        # for ind, thing in enumerate(lattice_features):
        #     metrics[f'{thing} coverage'] = prior_coverage[ind].item()
        # metrics['Minimum 1d coverage'] = torch.amin(prior_coverage).item()

        # get fraction of samples which are 'reasonable' at this energy,
        en_func = self.energy_function.energy_function
        sample_is_good = (sample_batch[en_func] < 0) * (sample_batch.packing_coeff > 0.55) * (
                sample_batch.packing_coeff < 0.95)
        metrics["Reasonable Sample Fraction"] = sample_is_good.float().mean().item()

        """Backward stats"""

        log_pf = bwd_stats['log_pfs'].sum(-1)
        log_pb = bwd_stats['log_pbs'].sum(-1)
        log_z = bwd_stats['log_Z_learned']
        log_r = bwd_stats['log_r']
        # parity / Z diagnostics (shared with fwd)
        metrics.update({f'bwd {k}': v for k, v in quick_tb_stats(log_pf, log_pb, log_z, log_r).items()})

        "Gauss stats"
        for prefix in ['fwd', 'bwd']:
            if prefix == 'fwd':
                stats = fwd_stats
            elif prefix == 'bwd':
                stats = bwd_stats
            metrics[f'{prefix} Mean F Drift'] = stats['means_f'].abs().mean()
            metrics[f'{prefix} Mean B Drift'] = stats['means_b'].abs().mean()
            metrics[f'{prefix} Mean F Var'] = stats['logvars_f'].mean()
            metrics[f'{prefix} Mean B Var'] = stats['logvars_b'].mean()
            metrics = {k: to_loggable(v) for k, v in metrics.items()}

        return metrics

    def training_eval(self):
        metrics = {}
        self.times['eval_step_start'] = time()
        eval_discretizer = lambda bsz: uniform_discretizer(bsz, self.args.eval_T)

        do_figs = self.step_ind % self.args.figs_period == 0

        '''sampling and metrics analysis'''
        fwd_stats, sample_batch = self.fwd_eval_sampling(eval_discretizer)
        bwd_stats = self.bwd_eval_sampling(eval_discretizer)
        metrics.update(self.log_metrics(fwd_stats, bwd_stats, sample_batch))

        self.times['eval_figs_start'] = time()
        if do_figs:
            x, y = self.prior_dataset.sample_tensors(10000, replace=True)
            # always sample from forward policy
            fig_dict, metrics = eval_figs(fwd_stats,
                                          bwd_stats,
                                          sample_batch,
                                          x,
                                          self.args.energy_function,
                                          metrics
                                          )
        else:
            fig_dict = {}
        self.times['eval_figs_end'] = time()

        '''logging and wrap up'''
        self.times['eval_wrapup_start'] = time()
        if do_figs:
            adjust_fig_filesize(fig_dict)
            metrics.update(fig_dict)

        metrics.update({'Batch Size': self.batch_size})
        metrics.update({'Eval Batch Size': self.args.eval_batch_size})
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
                self.phase1to2()

        return metrics

    def _shrink_eval_batch_on_oom(self, e):
        print(f"Caught error: {str(e)}")
        if not is_cuda_oom(e):
            raise e
        self.args.eval_batch_size = max(1, int(self.args.eval_batch_size * 0.75))
        if self.args.eval_batch_size <= 1:
            raise RuntimeError("Cascading OOM Failure")
        print(f"Reducing eval batch size to {self.args.eval_batch_size}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    @torch.no_grad()
    def fwd_eval_sampling(self, eval_discretizer):
        self.times['eval_sampling_start'] = time()

        acc, eval_samples = defaultdict(list), []

        while len(eval_samples) < self.args.eval_num_samples:
            try:
                mol_batch = next(self.mol_dataset.loader(self.args.eval_batch_size, mode='graphs'))
                mol_batch = mol_batch.to(self.device)
                mol_batch.orient_molecule(mode='standard')
                init_state = get_gfn_init_state(self.args.eval_batch_size,
                                                self.energy_function.data_ndim, self.device)
                out = sample_eval_fwd_trajs(init_state, self.ema_model, eval_discretizer,
                                            self.energy_function, mol_batch, no_conditioning=self.phase == 1)
            except (RuntimeError, ValueError) as e:
                self._shrink_eval_batch_on_oom(e)
                continue

            eval_samples.extend(out.pop('sample_batch').batch_to_list())
            for k, v in out.pop('gauss_params').items():
                acc[k].append(v)
            for k, v in out.items():
                acc[k].append(v)

        pooled = {k: torch.cat(v) for k, v in acc.items()}
        sample_batch = collate_data_list(eval_samples, skip_default_exclusion=True)

        # Z estimates computed ONCE over the pooled trajectories
        log_weight = pooled['log_r'] + pooled['log_pbs'].sum(-1) - pooled['log_pfs'].sum(-1)
        pooled['log_Z'] = logmeanexp(log_weight)
        pooled['log_Z_lb'] = log_weight.mean()
        pooled['log_Z_learned'] = pooled['log_flow'].mean()

        self.times['eval_sampling_end'] = time()

        return pooled, sample_batch

    def phase1to2(self):  # todo rewrite
        print("Hit initial KLD threshold. Starting prior thermalization.")
        self.hit_init_kld = True
        "adjust loss coefficients"
        self.args.bwd_loss_coeffs.bwd_tb_z = 1.0
        self.bwd_loss_schedule['tb'] = [(0, 1.0), (self.step_ind, 0.1),
                                        (self.step_ind + self.args.phase_change_time, 1.0)]
        self.bwd_loss_schedule['mle'] = [(0, 1.0), (self.step_ind, 1),
                                         (self.step_ind + self.args.phase_change_time, 0.0)]
        self.bwd_loss_schedule['bwd_tb_z'] = [(0, 2.0), (self.step_ind, 1.0)]
        self.bwd_loss_schedule['noised_fraction'] = [(0, 0.0), (self.step_ind, self.args.anchor_noise_fraction)]
        self.bwd_loss_schedule['noise_level'] = [(0, 0.0), (self.step_ind, self.args.anchor_noise_level)]
        "set cooldowns"
        # for d in self.increasing_loss_cooldown: # todo update this
        #     self.increasing_loss_cooldown[d] = self.args.phase_change_time
        "align log Z to buffer (it will converge to this value)"
        # z = metrics['Bwd Empirical log Z LB']# todo come back to thinking about this
        # with torch.no_grad():
        #     ema_model.flow_model.weight.data = z
        #     gfn_model.flow_model.weight.data = z
        "save checkpoint"
        self.save_checkpoint('prior')
        self.phase = 2
        self.grow_buffer = True

    def phase2to3(self):  # todo rewrite
        print("Thermalization complete. Starting on-policy equilibration.")
        self.phase = 3
        "save checkpoint"
        self.save_checkpoint('prior_equil')
        "adjust loss and balancing coefficients"
        self.args.fwd_to_bwd_ratio = self.args.min_bwd_ratio
        self.bwd_loss_schedule['bwd_tb_z'] = [(0, 1.0), (self.step_ind, 0)]
        self.fwd_loss_schedule['tb'] = [(0, 1.0), (self.step_ind, 0.0),
                                        (self.step_ind + self.args.phase_change_time // 2, 1.0)]
        "set cooldown"
        # for d in self.increasing_loss_cooldown:  # todo update this
        #     self.increasing_loss_cooldown[d] = self.args.phase_change_time
        self.grow_buffer = True
        self.std_boost_prob = self.args.p3_widevar_prob
        self.std_boost_var = self.args.p3_widevar_var


if __name__ == '__main__':
    modeller = Modeller()
    modeller.train()
