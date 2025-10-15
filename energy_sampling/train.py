import gc
import os
from copy import deepcopy
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
#     "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True")
from time import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from scipy.spatial.transform import Rotation
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from tqdm import trange

from energies.molecular_crystal import MolecularCrystal
from energy_sampling.buffer import CrystalReplayBuffer
from energy_sampling.utils import iter_forever, \
    is_cuda_oom, get_annealing_factor, \
    substitute_prior, parse_loss_schedules, dict2namespace, update_loss_schedule, \
    random_discretizer, low_discrepancy_discretizer, low_discrepancy_discretizer2, shifted_equidistant
from eval.evaluations import eval_step, conditional_eval_step
from gflownet_losses import get_gfn_forward_loss, get_gfn_backward_loss
from models import GFN
from mxtaltools.common.training_utils import flatten_wandb_params
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list
from utils import get_train_args, get_gfn_init_state, set_seed, \
    get_exploration_std, uniform_discretizer, \
    featurize_dataset, embed_dataset, \
    update_ema


class Modeller:
    def __init__(self):
        self.times = {}
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)
        torch.cuda.init()  # create context with the cap already in place

        args = get_train_args()

        set_seed(args.seed)
        if 'SLURM_PROCID' in os.environ:
            args.seed += int(os.environ["SLURM_PROCID"])

        if args.both_ways and args.bwd:
            args.bwd = False

        config = args.__dict__
        config["Experiment"] = "{args.energy}"
        self.args = args
        self.device = self.args.device
        self.increasing_loss_cooldown = 0
        self.lr_warmup_finished = False

    def train_logic(self, buffer, it):
        do_forward = False
        do_backward = False
        add_to_buffer = False
        if self.args.both_ways:
            p_forward = self.args.fwd_to_bwd_ratio / (self.args.fwd_to_bwd_ratio + 1)
            if it == 0:
                do_fwd = True
            elif self.args.fwd_to_bwd_ratio == 1:
                do_fwd = it % 2 == 0  # always do fwd first
            else:
                do_fwd = np.random.choice([0, 1], 1, p=[1 - p_forward, p_forward])

            if do_fwd:
                if self.args.sampling == 'buffer':
                    add_to_buffer = True
                do_forward = True
            else:
                do_backward = True

        elif self.args.bwd:  # backward ONLY
            do_backward = True
            p_forward = 0

        else:  # forward ONLY
            do_forward = True
            p_forward = 1

        if len(buffer) == 0:
            do_forward = True
            do_backward = False

        if it % 21 == 0:
            report_losses = True
        else:
            report_losses = False

        if not any([
            self.args.bwd_loss_coeffs.tb > 0,
            self.args.bwd_loss_coeffs.vg_lb > 0,
            self.args.bwd_loss_coeffs.vg_lme > 0,
            self.args.bwd_loss_coeffs.emp_z > 0,
            self.args.bwd_loss_coeffs.mle > 0,
        ]):
            do_backward = False
            do_forward = True

        return add_to_buffer, do_backward, do_forward, p_forward, report_losses

    def log_elapsed_times(self):
        elapsed_times = {}
        for key in self.times.keys():
            if 'start' in key:
                start_key = key
                end_key = start_key.split('_start')[0] + '_end'
                if end_key in self.times.keys():
                    elapsed_times[start_key.split('_start')[0] + '_time'] = self.times[end_key] - self.times[start_key]

        return elapsed_times

    def get_discretizer(self):
        # discretizer = lambda bsz: uniform_discretizer(bsz, self.args.T)
        # discretizer = lambda bsz: uniform_discretizer(bsz, np.random.randint(10,self.args.T+1))
        # discretizer = lambda bsz: random_discretizer(bsz, self.args.T, 10)
        if self.args.traj_length_strategy == 'static':
            traj_length = self.args.T
        elif self.args.traj_length_strategy == 'sampled':
            traj_length = np.random.randint(low=self.args.min_traj_length, high=self.args.max_traj_length + 1)
        else:
            assert False

        if self.args.discretizer == 'random':
            discretizer = lambda bsz: random_discretizer(bsz, traj_length, max_ratio=self.args.discretizer_max_ratio)
        elif self.args.discretizer == 'low_discrepancy':
            self.args.discretizer = lambda bsz: low_discrepancy_discretizer(bsz, traj_length)
        elif self.args.discretizer == 'low_discrepancy2':
            discretizer = lambda bsz: low_discrepancy_discretizer2(bsz, traj_length)
        elif self.args.discretizer == 'equidistant':
            discretizer = lambda bsz: shifted_equidistant(bsz, traj_length)
        elif self.args.discretizer == 'uniform':
            discretizer = lambda bsz: uniform_discretizer(bsz, traj_length)
        else:
            assert False
        return discretizer

    def increment_batch_size(self, buffer, train_mol_loader, test_mol_loader, batch_growth_increment):
        new_batch_size = max(self.args.batch_size + 1,
                             int(self.args.batch_size * batch_growth_increment))
        self.args.batch_size = new_batch_size  # gradually increment batch size

        if len(buffer) > 0:
            buffer.batch_size = new_batch_size

        train_mol_loader = DataLoader(
            train_mol_loader.dataset,
            batch_size=new_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        test_mol_loader = DataLoader(
            test_mol_loader.dataset,
            batch_size=new_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        train_iterator = iter_forever(train_mol_loader)
        test_iterator = iter_forever(test_mol_loader)

        return buffer, train_mol_loader, test_mol_loader, train_iterator, test_iterator

    def step_lr_schedule(self, schedulers, optimizers):
        if self.args.scheduler:
            lr = optimizers['fwd'].param_groups[0]['lr']
            if not self.lr_warmup_finished:
                schedulers['policy_1'].step()
                schedulers['flow'].step()

                if lr >= self.args.lr_policy:
                    self.lr_warmup_finished = True

            elif lr > self.args.min_lr:
                schedulers['policy_2'].step()
            return lr
        else:
            return False, None

    def ten_step_reporting(self, bwd_loss, bwd_loss_dict, fwd_loss, fwd_loss_dict, metrics, optimizers):
        metrics.update({'lr_fwd': optimizers['fwd'].param_groups[0]['lr']})
        metrics.update({'lr_bwd': optimizers['bwd'].param_groups[0]['lr']})
        metrics.update({'lr_flow': optimizers['flow'].param_groups[0]['lr']})
        metrics.update(self.log_elapsed_times())
        metrics['Forward Loss'] = fwd_loss
        metrics['Backward Loss'] = bwd_loss
        if fwd_loss_dict is not None:
            metrics.update(fwd_loss_dict)
            fwd_loss_dict = None
        if bwd_loss_dict is not None:
            metrics.update(bwd_loss_dict)
            bwd_loss_dict = None

    def anneal_reward(self, it, temp_annealing_lambda, energy_function):
        """anneal reward function"""
        if self.args.anneal_temperature:
            if self.args.temperature_conditioning:
                if energy_function.temperature_scaling_factor < 1:
                    energy_function.temperature_scaling_factor *= temp_annealing_lambda
            else:
                if energy_function.temperature > self.args.energy_min_temperature:
                    energy_function.temperature *= temp_annealing_lambda

        if self.args.core_start_time > 0:
            energy_function.core_coeff = round(
                self.args.energy_core_coeff * F.sigmoid(torch.tensor((it - self.args.core_start_time) / 50)).item(), 2)
        if self.args.lj_start_time > 0:
            energy_function.lj_coeff = round(
                self.args.energy_lj_coeff * F.sigmoid(torch.tensor((it - self.args.lj_start_time) / 50)).item(), 2)

    def set_loss_coeffs(self, it):
        """anneal reward function"""
        if it == 0:
            self.args.fwd_loss_schedule = parse_loss_schedules(self.args.fwd_loss_coeffs)
            self.args.bwd_loss_schedule = parse_loss_schedules(self.args.bwd_loss_coeffs)

            self.args.fwd_loss_coeffs = dict2namespace({k: 0.0 for k in self.args.fwd_loss_schedule})
            self.args.bwd_loss_coeffs = dict2namespace({k: 0.0 for k in self.args.bwd_loss_schedule})

        update_loss_schedule(it, self.args.fwd_loss_schedule, self.args.fwd_loss_coeffs.__dict__)
        update_loss_schedule(it, self.args.bwd_loss_schedule, self.args.bwd_loss_coeffs.__dict__)

    def get_conditioning_dim(self):
        conditioning_dim = 0
        if self.args.temperature_conditioning:
            conditioning_dim += 1
        if self.args.molecule_conditioning:
            if self.args.mol_embedding_type == 'autoencoder':
                conditioning_dim += 64 * 3
            elif self.args.mol_embedding_type == 'principal_axes':
                conditioning_dim += 9
            else:
                assert False
        if self.args.sg_conditioning:
            conditioning_dim += 237
        return conditioning_dim

    def init_energy_function(self):
        energy_config = {
            'device': self.device,
            'energy_function': self.args.energy_function,
            'min_temperature': self.args.energy_min_temperature,
            'max_temperature': self.args.energy_max_temperature,
            'temperature_scaling_factor': self.args.temperature_scaling_factor,
            'temperature_conditioning': self.args.temperature_conditioning,
            'temperature': self.args.energy_static_temperature,
            'density_coeff': self.args.energy_density_coeff,
            'energy_clip': self.args.energy_clip,
            'ellipsoid_scale': self.args.ellipsoid_scale,
            'core_coeff': self.args.energy_core_coeff,
            'lj_coeff': self.args.energy_lj_coeff,
            'lj_turnover_pot': self.args.lj_turnover_pot,
            'lj_repulsion': self.args.lj_repulsion,
            'molecule_conditioning': self.args.molecule_conditioning,
            'sg_conditioning': self.args.sg_conditioning,
            'space_groups': self.args.space_groups,
            'bounding_coeff': self.args.bounding_coeff,
            'niggli_coeff': self.args.niggli_coeff,
            'z_primes': self.args.z_primes,
            'max_z_prime': max(self.args.z_primes),
        }
        energy_function = MolecularCrystal(**energy_config)
        return energy_function

    def init_gfn_model(self, energy_function):
        gfn_config = dict(
            dim=energy_function.data_ndim,
            s_emb_dim=self.args.s_emb_dim,
            hidden_dim=self.args.hidden_dim,
            conditions_dim=self.get_conditioning_dim(),
            harmonics_dim=self.args.harmonics_dim,
            t_dim=self.args.t_emb_dim,
            condition_embedding_dim=self.args.condition_emb_dim,
            trajectory_length=self.args.T,
            clipping=self.args.clipping,
            gfn_clip=self.args.gfn_clip,
            learned_variance=self.args.learned_variance,
            log_var_range=self.args.log_var_range,
            pb_drift_range=self.args.pb_drift_range,
            pb_var_range=self.args.pb_var_range,
            t_scale=self.args.t_scale,
            conditional_flow_model=any([
                self.args.temperature_conditioning,
                self.args.molecule_conditioning,
                self.args.sg_conditioning]
            ),
            learn_pb=self.args.learn_pb,
            joint_layers=self.args.joint_layers,
            dropout=self.args.dropout,
            norm=self.args.norm,
            zero_init=self.args.zero_init,
            device=self.device
        )
        gfn_model = GFN(**gfn_config).to(self.device)

        return gfn_model, gfn_config

    def init_schedulers_optimizers(self, gfn_model):
        if self.args.scheduler:
            init_fwd_lr = self.args.lr_policy / self.args.lr_warmup_ratio
            init_flow_lr = self.args.lr_flow / self.args.lr_warmup_ratio
            init_bwd_lr = self.args.lr_back / self.args.lr_warmup_ratio
        else:
            init_fwd_lr = self.args.lr_policy
            init_bwd_lr = self.args.lr_back
            init_flow_lr = self.args.lr_flow

        """
        Initialize Optimizers
        """
        policy_params = [{'params': gfn_model.t_model.parameters()},
                         {'params': gfn_model.s_model.parameters()},
                         {'params': gfn_model.forward_policy.parameters()},
                         {'params': gfn_model.backward_policy.parameters()},
                         ]
        if self.args.temperature_conditioning:
            policy_params += [{'params': gfn_model.conditions_embedding_model.parameters()}]

        flow_params = gfn_model.flow_model.parameters()

        optimizers = {}
        if self.args.use_weight_decay:
            optimizers['fwd'] = torch.optim.Adam(policy_params, init_fwd_lr, weight_decay=self.args.weight_decay)
            optimizers['bwd'] = torch.optim.Adam(policy_params, init_bwd_lr, weight_decay=self.args.weight_decay)
            optimizers['flow'] = torch.optim.Adam(flow_params, init_flow_lr, weight_decay=self.args.weight_decay)
        else:
            optimizers['fwd'] = torch.optim.Adam(policy_params, init_fwd_lr)
            optimizers['bwd'] = torch.optim.Adam(policy_params, init_bwd_lr)
            optimizers['flow'] = torch.optim.Adam(flow_params, init_flow_lr)

        schedulers = {}
        if self.args.scheduler:
            lr_warmup_lambda = get_annealing_factor(1, self.args.lr_warmup_ratio, self.args.lr_warmup_time, 10)

            lr_annealing_lambda = get_annealing_factor(self.args.lr_policy, self.args.min_lr, self.args.lr_anneal_time,
                                                       10)
            schedulers['policy_1'] = lr_scheduler.MultiplicativeLR(
                optimizers['fwd'], lr_lambda=lambda epoch: lr_warmup_lambda)
            schedulers['policy_2'] = lr_scheduler.MultiplicativeLR(
                optimizers['fwd'], lr_lambda=lambda epoch: lr_annealing_lambda)

            flow_annealing_lambda = get_annealing_factor(1, self.args.lr_warmup_ratio, self.args.lr_anneal_time, 10)
            schedulers['flow'] = lr_scheduler.MultiplicativeLR(
                optimizers['flow'], lr_lambda=lambda epoch: flow_annealing_lambda)

        return optimizers, schedulers

    def init_buffers_datasets(self, energy_function):
        # load dataset of prebuilt and scored molecular crystals into the buffer
        buffer = CrystalReplayBuffer(
            self.args.buffer_size,
            'cpu',
            energy_function,
            self.args.batch_size,
            beta=self.args.beta,
            rank_weight=self.args.rank_weight,
            prioritized=self.args.prioritized,
            keep_initial_samples=False,  # self.args.buffer_path is not None,
            diversity_coeff=self.args.buffer_diversity_coeff,
        )
        if (
                self.args.both_ways or self.args.bwd) and self.args.buffer_path is not None:  # preload samples into the buffer
            buffer = self.add_dataset_to_buffer(self.args.buffer_path, buffer, self.args.space_groups)

        if self.args.molecule == 'urea':
            good_mol = self.init_urea(buffer)
            train_mols_list = [good_mol.clone() for _ in range(int(self.args.max_batch_size * 1.5))]
            test_mols_list = [good_mol.clone() for _ in range(int(self.args.max_batch_size * 1.5))]

        elif self.args.molecule == 'nicotinamide':
            good_mol = self.init_nicotinamide(buffer)
            train_mols_list = [good_mol.clone() for _ in range(int(self.args.max_batch_size * 1.5))]
            test_mols_list = [good_mol.clone() for _ in range(int(self.args.max_batch_size * 1.5))]

        elif self.args.molecule == 'qm9':
            qm9_mols = torch.load(self.args.molecules_path, weights_only=False)
            rng = np.random.RandomState(0)
            rands = rng.choice(len(qm9_mols), len(qm9_mols), replace=False)
            bp = int(len(rands) * 0.8)
            # for mol in qm9_mols:
            #     mol.deprotonate()  # since we'll be comparing against CSD later, deprotonate here
            train_mols_list = [qm9_mols[ind] for ind in rands[:bp]]
            test_mols_list = [qm9_mols[ind] for ind in rands[bp:]]

        else:
            assert False

        if self.args.molecule_conditioning:
            train_mols_list = embed_dataset(train_mols_list, self.args.autoencoder_path, self.device, encoder=None,
                                            embedding_type=self.args.mol_embedding_type)
            test_mols_list = embed_dataset(test_mols_list, self.args.autoencoder_path, self.device, encoder=None,
                                           embedding_type=self.args.mol_embedding_type)

        train_mol_loader = DataLoader(
            train_mols_list,
            batch_size=self.args.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
        test_mol_loader = DataLoader(
            test_mols_list,
            batch_size=self.args.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
        train_iterator = iter_forever(train_mol_loader)
        test_iterator = iter_forever(test_mol_loader)

        return buffer, train_mol_loader, test_mol_loader, train_iterator, test_iterator

    def init_nicotinamide(self, buffer):
        if len(buffer) > 0:  # ensure same conformer as dataset, where possible
            atom_coords = buffer.dataset[0].pos
            atom_types = buffer.dataset[0].z
        else:
            atom_coords = torch.tensor([
                [-2.3940, 1.1116, -0.0088],
                [1.7614, -1.2284, -0.0034],
                [-2.4052, -1.1814, 0.0027],
                [-0.2969, 0.0397, 0.0024],
                [0.4261, 1.2273, 0.0039],
                [0.4117, -1.1510, -0.0013],
                [1.8161, 1.1886, 0.0018],
                [-1.7494, 0.0472, 0.0045],
                [2.4302, -0.0535, -0.0018]
            ], dtype=torch.float32, device='cpu')
            atom_types = torch.tensor([8, 7, 7, 6, 6, 6, 6, 6, 6], dtype=torch.long, device='cpu')
        atom_coords -= atom_coords.mean(dim=0)
        good_mol = MolData(
            z=atom_types,
            pos=atom_coords,
            x=atom_types,
            do_mol_analysis=True,
        )
        return good_mol

    def init_urea(self, buffer):
        if len(buffer.dataset) > 0:  # ensure same conformer as dataset, where possible
            atom_coords = buffer.dataset[0].pos
            atom_types = buffer.dataset[0].z
        else:
            atom_coords = torch.tensor([  # stick with urea for just now
                [-1.3042, - 0.0008, 0.0001],
                [0.6903, - 1.1479, 0.0001],
                [0.6888, 1.1489, 0.0001],
                [- 0.0749, - 0.0001, - 0.0003],
            ], dtype=torch.float32, device='cpu')
            atom_types = torch.tensor([8, 7, 7, 6], dtype=torch.long, device='cpu')
        atom_coords -= atom_coords.mean(0)
        good_mol = MolData(
            z=atom_types,
            pos=atom_coords,
            x=atom_types,
            do_mol_analysis=True,
        )
        return good_mol

    def train(self):
        self.times['initialization_start'] = time()

        # Reward init
        energy_function = self.init_energy_function()

        # Model Init
        gfn_model, gfn_config = self.init_gfn_model(energy_function)
        ema_model = deepcopy(gfn_model)
        name = self.args.tag + '_' + self.args.run_name
        np.save(f'checkpoints/{name}_model_config', gfn_config)  # todo add path to saving directories

        # opt init
        optimizers, schedulers = self.init_schedulers_optimizers(gfn_model)

        # buffer & loaders init
        (buffer, train_mol_loader, test_mol_loader,
         train_iterator, test_iterator) = self.init_buffers_datasets(energy_function)

        # initialize some annealing factors
        if self.args.temperature_conditioning:
            temp_annealing_lambda = get_annealing_factor(self.args.temperature_scaling_factor, 1,
                                                         self.args.temp_annealing_max_steps, 10)

        else:
            temp_annealing_lambda = get_annealing_factor(self.args.energy_max_temperature,
                                                         self.args.energy_min_temperature,
                                                         self.args.temp_annealing_max_steps, 10)

        fwd_loss_dict = None
        bwd_loss_dict = None
        oomed_out = False
        fwd_loss, bwd_loss = 0, 0
        loss_record = []

        self.times['initialization_end'] = time()

        with wandb.init(project="GFN Energy",
                        config=flatten_wandb_params(self.args),
                        name=name,
                        tags=[self.args.tag]):

            wandb.watch(gfn_model,
                        log_graph=False,
                        log_freq=1000,
                        log='gradients')  # for gradient logging

            gfn_model.train()
            self.set_detect_anomaly(gfn_model, do_anomaly_detection=False)

            for step_ind in trange(self.args.epochs + 1):
                metrics = dict()
                if step_ind % 10 == 0:
                    self.set_loss_coeffs(step_ind)

                exploration_std = get_exploration_std(step_ind,
                                                      self.args.exploratory,
                                                      self.args.wd_max_steps,
                                                      self.args.exploration_factor,
                                                      self.args.exploration_wd)

                self.times['train_step_start'] = time()
                try:
                    train_loss, step_type, loss_dict = self.train_step(energy_function,
                                                                       gfn_model,
                                                                       optimizers,
                                                                       step_ind,
                                                                       exploration_std,
                                                                       buffer,
                                                                       train_iterator,
                                                                       repeats=self.args.repeats
                                                                       )
                    if self.args.ema_decay is not None:
                        update_ema(gfn_model, ema_model, decay=self.args.ema_decay)
                    else:
                        ema_model = gfn_model

                    if step_type == 'Forward':
                        fwd_loss = train_loss
                        if loss_dict is not None:
                            fwd_loss_dict = loss_dict
                    elif step_type == 'Backward':
                        bwd_loss = train_loss
                        if loss_dict is not None:
                            bwd_loss_dict = loss_dict

                    if not oomed_out and (
                            self.args.batch_size < self.args.max_batch_size and self.args.grow_batch_size):
                        buffer, train_mol_loader, test_mol_loader, train_iterator, test_iterator = self.increment_batch_size(
                            buffer, train_mol_loader,
                            test_mol_loader,
                            self.args.batch_growth_increment)

                except (RuntimeError, ValueError) as e:  # if we do hit OOM, slash the batch size
                    (oomed_out, buffer, train_mol_loader,
                     test_mol_loader, train_iterator, test_iterator) = self.handle_train_epoch_error(
                        e, oomed_out, buffer,
                        train_mol_loader,
                        test_mol_loader,
                        optimizers)
                self.times['train_step_end'] = time()

                # evaluation work
                if (step_ind % self.args.eval_period == 0 and step_ind > 0) or step_ind == 50:
                    self.eval_work(ema_model, step_ind,
                                   buffer, train_mol_loader, test_mol_loader,
                                   energy_function, metrics)



                # train monitoring
                if step_ind % 10 == 0:
                    loss_record.append(fwd_loss + bwd_loss)
                    if loss_record[-1] == torch.amin(torch.tensor(loss_record)):  # if this is the best model yet
                        torch.save(gfn_model.state_dict(), f'checkpoints/{name}_model_train.pt')
                        torch.save(ema_model.state_dict(), f'checkpoints/{name}_model_eval.pt')

                    metrics['train/expl'] = exploration_std(0) if exploration_std is not None else 0
                    lr = self.step_lr_schedule(schedulers, optimizers)
                    self.anneal_reward(step_ind, temp_annealing_lambda, energy_function)
                    self.ten_step_reporting(bwd_loss, bwd_loss_dict, fwd_loss, fwd_loss_dict, metrics, optimizers)
                    loss_record = self.check_loss_explosion(name, loss_record, gfn_model, ema_model, optimizers)
                    wandb.log(metrics, step=step_ind)

            torch.save(ema_model, f'checkpoints/{name}_model_final.pt')

    def check_loss_explosion(self,
                             name: str,
                             loss_record: list,
                             gfn_model,
                             ema_model,
                             optimizers,
                             explosion_buffer: float = 10,
                             grace_time: int = 10):
        """
        If losses are exploding, reload best prior checkpoint and slash the learning rate

        """
        if len(loss_record) >= (grace_time * 2):
            self.increasing_loss_cooldown -= 1

            losses = torch.tensor(loss_record)
            current_loss = losses[-1]
            best_loss = torch.amin(losses)

            scale = torch.quantile(torch.abs(losses[:-grace_time] - best_loss), 0.95) + 1e-4
            threshold = best_loss + scale * explosion_buffer
            diffs = torch.diff(losses, dim=0)

            hit_threshold = current_loss > threshold
            increasing_loss = torch.all(diffs[-grace_time:] > 0) and self.increasing_loss_cooldown <= 0

            if hit_threshold or increasing_loss:
                print("Losses increasing! Reloading best checkpoint and slashing LR.")
                if hit_threshold:
                    print("Hit loss threshold!")
                if increasing_loss:
                    print(f"Losses increasing over prior {grace_time} steps!")

                gfn_model.load_state_dict(torch.load(f'checkpoints/{name}_model_train.pt'))
                ema_model.load_state_dict(torch.load(f'checkpoints/{name}_model_eval.pt'))
                gfn_model.train()
                ema_model.eval()

                for opt in optimizers.values():
                    for g in opt.param_groups:
                        if g['lr'] > self.args.min_lr:
                            g['lr'] *= 0.75

                self.lr_warmup_finished = True

                if increasing_loss:
                    self.increasing_loss_cooldown = grace_time

                if hit_threshold:
                    to_keep = torch.argwhere(losses <= threshold).flatten().tolist()
                    loss_record = [rec for ind, rec in enumerate(loss_record) if ind in to_keep]

        return loss_record

    def set_detect_anomaly(self, gfn_model, do_anomaly_detection: bool):
        if do_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)  # for debugging

            def grad_check_hook(grad, name):
                if not torch.isfinite(grad).all():
                    raise RuntimeError(f"NaN/Inf gradient in {name}")
                return grad

            for p_name, p in gfn_model.named_parameters():
                if p.requires_grad:
                    p.register_hook(lambda g, n=p_name: grad_check_hook(g, n))

    def train_step(self,
                   energy_function,
                   gfn_model,
                   optimizers,
                   it,
                   exploration_std,
                   buffer,
                   mol_iterator,
                   repeats):
        add_to_buffer, do_backward, do_forward, p_forward, report_losses = self.train_logic(buffer, it)

        discretizer = self.get_discretizer()

        optimizers['flow'].zero_grad(set_to_none=True)
        if do_forward:
            optimizers['fwd'].zero_grad(set_to_none=True)
            mol_batch = next(mol_iterator)
            if self.args.molecule_conditioning:
                # if doing molecule conditioning, augment over mol orientations, adjusting properly the embedding
                random_rotations = torch.tensor(
                    Rotation.random(num=mol_batch.num_graphs).as_matrix(),
                    device=mol_batch.device, dtype=torch.float32)
                mol_batch.orient_molecule(mode='std')
                mol_batch.orient_molecule(mode='random',
                                          # important that the rotation is applied *from* the standard
                                          include_inversion=False,
                                          correct_orientation=True,
                                          override_random_rotations=random_rotations)
                mol_batch.embedding = mol_batch.rotate_embedding(random_rotations)
            else:
                mol_batch.orient_molecule(mode='std')

            loss, crystal_batch, loss_dict = self.fwd_train_step(
                energy_function,
                gfn_model,
                discretizer,
                exploration_std,
                mol_batch,
                buffer,
                return_exp=True,
                repeats=repeats,
                report_losses=report_losses
            )

            # forward_iter = int(it * p_forward)
            # if add_to_buffer and forward_iter % self.args.add_to_buffer_each == 0:
            #     # standard to_data_list won't work with our custom batching in the energy function
            #     data_list = manual_batch_to_data_list(crystal_batch.detach().cpu())
            #     buffer.add(data_list)

            del crystal_batch

        elif do_backward:
            optimizers['bwd'].zero_grad(set_to_none=True)
            loss, loss_dict = self.bwd_train_step(
                gfn_model,
                discretizer,
                buffer,
                energy_function,
                repeats=repeats,
                report_losses=report_losses)
        else:
            assert False

        loss.backward()
        clean_loss = loss.item()
        torch.nn.utils.clip_grad_norm_(gfn_model.parameters(),
                                       self.args.gradient_norm_clip)  # gradient clipping
        if do_forward:
            optimizers['fwd'].step()
            optimizers['flow'].step()
        elif do_backward:
            optimizers['bwd'].step()
            optimizers['flow'].step()

        step_type = "Forward" if do_forward else "Backward"

        if report_losses:
            loss_dict_cpu = {step_type + "_loss/" + key: value.cpu().detach().numpy() for key, value in
                             loss_dict.items()}
        else:
            loss_dict_cpu = None

        del loss, loss_dict  # or whatever is large

        return clean_loss, step_type, loss_dict_cpu

    def fwd_train_step(self, energy_function, gfn_model, discretizer,
                       exploration_std, mol_batch, buffer, return_exp=False,
                       repeats: int = 10,
                       report_losses: bool = False):
        init_state = get_gfn_init_state(self.args.batch_size, energy_function.data_ndim, self.device)
        log_T_tensor, sg_inds, condition, z_primes = energy_function.get_conditioning_tensor(mol_batch)
        mol_batch.sg_ind = sg_inds
        mol_batch.z_prime = z_primes
        return get_gfn_forward_loss(self.args.fwd_loss_coeffs,
                                    init_state,
                                    gfn_model,
                                    energy_function.log_reward,
                                    discretizer,
                                    mol_batch,
                                    buffer,
                                    log_T_tensor,
                                    exploration_std=exploration_std,
                                    return_exp=return_exp,
                                    condition=condition,
                                    repeats=repeats,
                                    report_losses=report_losses)

    def bwd_train_step(self, gfn_model, discretizer,
                       buffer, energy_function, repeats: int = 10,
                       report_losses: bool = False):
        if self.args.sampling == 'buffer':
            samples, rewards, crystal_batch, condition = buffer.sample(
                override_batch=int(self.args.batch_size * self.args.bwd_batch_multiplier),
                randomize_orientations=True if self.args.molecule_conditioning else False)
        else:
            assert False, f"sampling method {self.args.sampling} not implemented"

        if self.args.bwd_loss_coeffs.mle_prior_fraction > 0:
            condition, rewards, samples = substitute_prior(
                self.args.bwd_loss_coeffs, condition, crystal_batch, energy_function, rewards, samples, buffer)

        return get_gfn_backward_loss(self.args.bwd_loss_coeffs,
                                     samples.to(self.device),
                                     gfn_model,
                                     rewards.to(self.device),
                                     discretizer,
                                     condition=condition.to(self.device),
                                     repeats=repeats,
                                     report_losses=report_losses)

    def handle_train_epoch_error(self, e, oomed_out, buffer, train_mol_loader, test_mol_loader, optimizers):
        print(f"Caught error: {str(e)}")
        if is_cuda_oom(e):
            print("OOMED!")

            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)

            # break reference cycles
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

            self.args.batch_size = max(1, int(self.args.batch_size * 0.95))
            if self.args.batch_size <= 1:
                raise RuntimeError("Cascading OOM Failure")

            gc.collect()
            torch.cuda.empty_cache()

            train_mol_loader = DataLoader(
                train_mol_loader.dataset,
                batch_size=self.args.batch_size,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )
            test_mol_loader = DataLoader(
                test_mol_loader.dataset,
                batch_size=self.args.batch_size,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )
            train_iterator = iter_forever(train_mol_loader)
            test_iterator = iter_forever(test_mol_loader)

            oomed_out = True
            print(f"Reducing batch size to {self.args.batch_size}")
        else:
            raise e  # will simply raise error if other or if training on CPU
        return oomed_out, buffer, train_mol_loader, test_mol_loader, train_iterator, test_iterator

    def do_evaluation(self, energy_function, buffer, gfn_model, i, mol_loader,
                      override_do_figures: Optional[bool] = None):
        self.times['eval_step_start'] = time()

        eval_discretizer = lambda bsz: uniform_discretizer(bsz, self.args.eval_T)

        if override_do_figures is not None:
            do_figures = override_do_figures
        else:
            do_figures = i % self.args.figs_period == 0
        eval_batch_size = max(self.args.batch_size, self.args.eval_batch_size)

        eval_rands = np.random.randint(len(mol_loader.dataset), size=eval_batch_size)
        mol_batch = collate_data_list([mol_loader.dataset[ind] for ind in eval_rands]).to(self.device)
        if not self.args.molecule_conditioning:  # always std orientation if we're not conditioning
            mol_batch.orient_molecule(mode='standard')
        else:  # if we are conditioning, randomly rotate and make sure we catch the embedding
            # todo functionalize this
            random_rotations = torch.tensor(
                Rotation.random(num=mol_batch.num_graphs).as_matrix(),
                device=mol_batch.device, dtype=torch.float32)
            mol_batch.orient_molecule(mode='standard')
            mol_batch.orient_molecule(mode='random',
                                      include_inversion=False,
                                      correct_orientation=True,
                                      override_random_rotations=random_rotations
                                      )

            mol_batch.embedding = mol_batch.rotate_embedding(random_rotations)

        # if we are conditioning, we take it as it comes
        init_state = get_gfn_init_state(eval_batch_size, energy_function.data_ndim, self.device)

        eval_metrics = {}
        eval_metrics.update(
            eval_step(energy_function,
                      gfn_model,
                      eval_discretizer,
                      init_state,
                      buffer,
                      self.args,
                      do_figures,
                      mol_batch,
                      bwd_training=len(buffer) > 0,
                      ))

        eval_metrics.update({'Batch Size': self.args.batch_size})
        eval_metrics.update(self.log_elapsed_times())

        self.times['eval_step_end'] = time()

        return eval_metrics

    def do_conditional_evaluation(self, energy_function, gfn_model, mol_loader,
                                  ):  # todo these functions could be cleaned up / consolidated
        self.times['eval_step_start'] = time()
        eval_discretizer = lambda bsz: uniform_discretizer(bsz, self.args.eval_T)

        eval_batch_size = self.args.eval_batch_size

        eval_rands = np.random.randint(len(mol_loader.dataset), size=eval_batch_size)
        mol_batch = collate_data_list([mol_loader.dataset[ind] for ind in eval_rands]).to(self.device)

        init_state = get_gfn_init_state(eval_batch_size, energy_function.data_ndim, self.device)

        eval_metrics = {}
        eval_metrics.update(
            conditional_eval_step(energy_function,
                                  gfn_model,
                                  eval_discretizer,
                                  init_state,
                                  mol_batch,
                                  mols_to_sample=5
                                  ))

        return eval_metrics

    def add_dataset_to_buffer(self, dataset_path, buffer, space_groups=None):
        print("Loading prebuilt buffer")
        dataset = torch.load(dataset_path, weights_only=False)

        if space_groups is not None:
            # filter unwanted space groups
            dataset = [elem for elem in dataset if elem.sg_ind in space_groups]

        if self.args.energy_function in ['ellipsoid_overlap',
                                         'silu_energy',
                                         'combo']:  # reparameterize incoming samples
            print("Re-featurizing preloaded buffer samples")
            dataset = featurize_dataset(dataset, self.device,
                                        self.args.ellipsoid_scale, self.args.lj_repulsion)

        if self.args.molecule_conditioning:  # embed dataset
            print("Getting preloaded dataset molecule embeddings")
            dataset = embed_dataset(dataset, self.args.autoencoder_path, self.device, encoder=None)

        buffer.add(dataset)
        print(f"Buffer loaded with {len(dataset)} samples")

        return buffer

    def eval_work(self,
                  gfn_model,
                  step_ind,
                  buffer,
                  train_mol_loader,
                  test_mol_loader,
                  energy_function,
                  metrics):
        if self.args.molecule_conditioning or self.args.sg_conditioning:

            if step_ind % self.args.conditional_eval_period == 0:  # make conditional sampling figures
                # # so far not useful
                train_metrics = self.do_evaluation(energy_function, buffer, gfn_model,
                                                   step_ind, train_mol_loader,
                                                   override_do_figures=False)
                kk = list(train_metrics.keys())
                for key in kk:
                    metrics['train_eval/' + key] = train_metrics[key]
                conditional_metrics = self.do_conditional_evaluation(energy_function, gfn_model,
                                                                     test_mol_loader,
                                                                     )
                metrics.update(conditional_metrics)

        metrics.update(self.do_evaluation(energy_function,
                                          buffer,
                                          gfn_model,
                                          step_ind,
                                          test_mol_loader))

        self.dynamic_quality_management(metrics)

        wandb.log(metrics, step=step_ind)

    def dynamic_quality_management(self, metrics):
        # minimum_1d_coverage = metrics['Minimum 1d coverage']
        # adjust by a factor of 'multiple' for each 'delta_factor' of miss
        multiple = 2
        delta_factor = 2
        min_rat = 1 / 5
        max_rat = 100
        eps = 1e-6
        fwd_res = np.log(1 - metrics['Forward TB R Value'] + eps)  # metrics['TB Residual']
        bwd_res = np.log(1 - metrics['Backward TB R Value'] + eps)  # metrics['Bwd TB Residual']
        # check the TB losses on forward and backward training
        miss = fwd_res - bwd_res  # want to push this ratio towards zero
        # for positive miss, do more forward training
        # for negative miss, do more backward training
        if self.args.both_ways:
            # must keep a significant amount of forward training at all times, as this is actually the relevant balancing mechanism
            if max_rat >= self.args.fwd_to_bwd_ratio >= min_rat:
                # if miss > self.args.prior_mean_loss_cutoff:
                gated_miss = np.clip(miss, a_max=delta_factor, a_min=-delta_factor)
                # when fwd res is large, increase fwd_to_bwd_ratio, and vice-versa
                adjustment_factor = multiple ** (gated_miss / delta_factor)
                self.args.fwd_to_bwd_ratio *= adjustment_factor
                self.args.fwd_to_bwd_ratio = np.clip(self.args.fwd_to_bwd_ratio, a_min=min_rat, a_max=max_rat)

        metrics['Fwd-Bwd R Value Ratio'] = miss
        if self.args.anneal_repulsion:
            print("We're not doing repulsive annealing anymore!")

            # reasonable_frac = metrics['Reasonable Sample Fraction']
            # if (reasonable_frac >= self.args.anneal_repulsion_cutoff) and diversity_check:
            #     if self.args.lj_repulsion < 1:
            #         self.args.lj_repulsion = min(1, self.args.lj_repulsion * 1.05)
            #         energy_function.lj_repulsion = self.args.lj_repulsion
            #         if buffer is not None:
            #             if len(buffer) > 0:
            #                 buffer.recompute_silu_pot(
            #                     batch_size=min(500, self.args.batch_size),
            #                     lj_repulsion=self.args.lj_repulsion,
            #                     device=self.device
            #                 )
        metrics['LJ Repulsion'] = self.args.lj_repulsion
        metrics['Fwd to Bwd Ratio'] = self.args.fwd_to_bwd_ratio


if __name__ == '__main__':
    modeller = Modeller()
    modeller.train()
