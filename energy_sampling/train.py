import gc
import os
from collections import defaultdict
from copy import deepcopy
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
#     "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True")
from time import time
from typing import Optional

import numpy as np
import torch
import wandb
from scipy.spatial.transform import Rotation
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from tqdm import trange

from energies.molecular_crystal import MolecularCrystal
from energy_sampling.buffer import CrystalReplayBuffer
from energy_sampling.eval.utils import sample_eval_fwd_trajs
from energy_sampling.utils import iter_forever, \
    is_cuda_oom, get_annealing_factor, \
    parse_loss_schedules, dict2namespace, update_loss_schedule, \
    random_discretizer, low_discrepancy_discretizer, low_discrepancy_discretizer2, shifted_equidistant, substitute_prior
from eval.evaluations import conditional_eval_step, adjust_fig_filesize, log_metrics, fwd_figs, bwd_figs
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
        self.hit_init_kld = False
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
        self.phase = None
        self.grow_buffer = self.args.grow_buffer
        if self.args.anchor_fwd_bwd:
            self.args.fwd_to_bwd_ratio = 1.0E-6
        self.forward_batch_size = self.args.batch_size
        self.backward_batch_size = self.args.batch_size
        self.phase = 1
        self.fwd_tb_norm = 10000
        self.bwd_tb_norm = 10000
        self.best_tb_norm = 10000
        self.last_fwd_it = 0
        self.last_bwd_it = 0
        self.step_ind = 0
        self.run_name = str(self.args.tag) + '_' + str(self.args.run_name)

    def train_logic(self, buffer, it):
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
            p_forward = 0

        else:  # forward ONLY
            do_forward = True
            p_forward = 1

        if len(buffer) == 0:
            do_forward = True
            do_backward = False

        if not any([
            self.args.bwd_loss_coeffs.tb > 0,
            self.args.bwd_loss_coeffs.vg_lb > 0,
            self.args.bwd_loss_coeffs.vg_lme > 0,
            self.args.bwd_loss_coeffs.emp_z > 0,
            self.args.bwd_loss_coeffs.mle > 0,
        ]):
            do_backward = False
            do_forward = True

        if do_forward:
            step_type = "Forward"
        elif do_backward:
            step_type = "Backward"
        else:
            assert False
        return step_type

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

    def increment_batch_size(self, buffer, train_mol_loader, test_mol_loader, batch_growth_increment, step_type,
                             train_iterator, test_iterator):
        if step_type == "Forward":
            #print("7")
            if self.forward_batch_size < self.args.max_fwd_batch_size:
                new_batch_size = min(self.args.max_fwd_batch_size, max(self.forward_batch_size + 1,
                                                                       int(self.forward_batch_size * batch_growth_increment)))
                self.forward_batch_size = new_batch_size  # gradually increment batch size

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
        elif step_type == "Backward":
            if self.backward_batch_size < self.args.max_bwd_batch_size:
                new_batch_size = min(self.args.max_bwd_batch_size, max(self.backward_batch_size + 1,
                                                                       int(self.backward_batch_size * batch_growth_increment)))
                self.backward_batch_size = new_batch_size  # gradually increment batch size

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
        metrics['Fwd to Bwd Ratio'] = self.args.fwd_to_bwd_ratio
        metrics['Rolling fTB Norm'] = self.fwd_tb_norm
        metrics['Rolling bTB Norm'] = self.bwd_tb_norm
        metrics['Best TB Norm'] = self.best_tb_norm
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

    def set_loss_coeffs(self, it):
        """anneal reward function"""
        if it == 0:
            self.fwd_loss_schedule = parse_loss_schedules(self.args.fwd_loss_coeffs)
            self.bwd_loss_schedule = parse_loss_schedules(self.args.bwd_loss_coeffs)

            self.args.fwd_loss_coeffs = dict2namespace({k: 0.0 for k in self.fwd_loss_schedule})
            self.args.bwd_loss_coeffs = dict2namespace({k: 0.0 for k in self.bwd_loss_schedule})

        update_loss_schedule(it, self.fwd_loss_schedule, self.args.fwd_loss_coeffs.__dict__)
        update_loss_schedule(it, self.bwd_loss_schedule, self.args.bwd_loss_coeffs.__dict__)

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
        if self.args.zp_conditioning:
            conditioning_dim += 1
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
            'lj_coeff': self.args.energy_lj_coeff,
            'molecule_conditioning': self.args.molecule_conditioning,
            'sg_conditioning': self.args.sg_conditioning,
            'space_groups': self.args.space_groups,
            'bounding_coeff': self.args.bounding_coeff,
            'reduction_coeff': self.args.reduction_coeff,
            'z_primes': self.args.z_primes,
            'max_z_prime': max(self.args.z_primes),
            'zp_conditioning': self.args.zp_conditioning,
            'uma_path': self.args.uma_path,
            'reward_range': self.args.reward_range,
            'lj_rescale': self.args.lj_rescale,
        }
        energy_function = MolecularCrystal(**energy_config)
        return energy_function

    def init_gfn_model(self, energy_function):
        reload = False
        if self.args.checkpoint_path is not None:
            reload = True
            reload_path = self.args.checkpoint_path
            print(f"Loading model from checkpoint {reload_path}")

        elif os.path.exists(f'checkpoints/{self.run_name}_model_train.pt'):
            reload_path = f'checkpoints/{self.run_name}_model_train.pt'
            if 'dev' not in reload_path:
                print("Reloading automatically from this prior checkpoint with same run name")
                reload = True
                reload_path = f'checkpoints/{self.run_name}_model_train.pt'

        if reload:
            eval_path = reload_path.replace('train', 'eval')
            config_path = reload_path.replace('train', 'config').replace('.pt', '.npy').replace(
                '_hit_prior', '').replace('_thermalized', '')
            state_path = reload_path.replace('_hit_prior', '').replace('_thermalized', '').replace('model_train','modeller_state')
            self.load_modeller_state(state_path)

            gfn_config = np.load(config_path, allow_pickle=True).item()
            gfn_model = GFN(**gfn_config).to(self.device)
            gfn_model.load_state_dict(torch.load(reload_path))
            ema_model = deepcopy(gfn_model)
            ema_model.load_state_dict(torch.load(eval_path))
        else:
            gfn_config = dict(
                dim=energy_function.data_ndim,
                s_emb_dim=self.args.s_emb_dim,
                hidden_dim=self.args.hidden_dim,
                conditions_dim=self.get_conditioning_dim(),
                harmonics_dim=self.args.harmonics_dim,
                t_dim=self.args.t_emb_dim,
                condition_embedding_dim=self.args.condition_emb_dim,
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
                    self.args.sg_conditioning,
                    self.args.zp_conditioning,
                ]
                ),
                learn_pb=self.args.learn_pb,
                joint_layers=self.args.joint_layers,
                dropout=self.args.dropout,
                norm=self.args.norm,
                zero_init=self.args.zero_init,
                device=self.device,
                max_z_prime=max(self.args.z_primes),
            )
            gfn_model = GFN(**gfn_config).to(self.device)
            ema_model = deepcopy(gfn_model)

        return gfn_model, gfn_config, ema_model

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
            self.backward_batch_size,
            beta=self.args.beta,
            rank_weight=self.args.rank_weight,
            prioritized=self.args.prioritized,
            keep_initial_samples=self.args.buffer_path is not None,
            max_z_prime=energy_function.max_z_prime,
            buffer_dist_cutoff=self.args.buffer_dist_cutoff,
            noised_buffer_length=self.args.noised_buffer_length,
        )
        if ((self.args.both_ways or self.args.bwd) and
                self.args.buffer_path is not None):  # preload samples into the buffer
            buffer = self.add_dataset_to_buffer(self.args.buffer_path, buffer,
                                                filter_unbound=True)

        if len(buffer) > 0 and (self.args.molecule != 'qm9'):
            mols_list = self.init_mol_from_buffer(buffer, self.args.z_primes)
            train_mols_list = []
            test_mols_list = []
            while len(train_mols_list) < int(self.args.max_fwd_batch_size * 1.5):
                for mol in mols_list:
                    train_mols_list.append(mol.clone())
            while len(test_mols_list) < int(self.args.max_fwd_batch_size * 1.5):
                for mol in mols_list:
                    test_mols_list.append(mol.clone())

        elif self.args.molecule == 'qm9':
            if energy_function.max_z_prime > 1:
                assert False, "Z'>1 mol loading not yet implemented for qm9"
            qm9_mols = torch.load(self.args.molecules_path, weights_only=False)
            rng = np.random.RandomState(0)
            rands = rng.choice(len(qm9_mols), len(qm9_mols), replace=False)
            bp = int(len(rands) * 0.8)
            # for mol in qm9_mols:
            #     mol.deprotonate()  # since we'll be comparing against CSD later, deprotonate here
            train_mols_list = [qm9_mols[ind] for ind in rands[:bp]]
            test_mols_list = [qm9_mols[ind] for ind in rands[bp:]]
        else:
            assert False, "Due to changes in Z'>1 modelling, must now initialize from a buffer"

        if self.args.molecule_conditioning:
            train_mols_list = embed_dataset(train_mols_list, self.args.autoencoder_path, self.device, encoder=None,
                                            embedding_type=self.args.mol_embedding_type)
            test_mols_list = embed_dataset(test_mols_list, self.args.autoencoder_path, self.device, encoder=None,
                                           embedding_type=self.args.mol_embedding_type)

        if hasattr(buffer.dataset[0], 'uma_gas_pot'):  # TODO REWRITE ALL THIS
            pot = buffer.dataset[0].uma_gas_pot
            for elem in train_mols_list:
                elem.uma_gas_pot = pot
            for elem in test_mols_list:
                elem.uma_gas_pot = pot

        train_mol_loader = DataLoader(
            train_mols_list,
            batch_size=self.forward_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
        test_mol_loader = DataLoader(
            test_mols_list,
            batch_size=self.forward_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
        train_iterator = iter_forever(train_mol_loader)
        test_iterator = iter_forever(test_mol_loader)

        return buffer, train_mol_loader, test_mol_loader, train_iterator, test_iterator

    def init_mol_from_buffer(self, buffer, z_primes):
        # this structure assumes the MXT format, where for Z'>1 samples, the mols are stacked in the same spot, in the same order
        sample = buffer.dataset[0]
        atoms_per_mol = sample.num_atoms // sample.z_prime
        atom_types = sample.z[:atoms_per_mol]
        atom_coords = sample.pos[:atoms_per_mol]
        atom_coords -= atom_coords.mean(dim=0)
        mols = []
        for zp in z_primes:
            mols.append(MolData(
                z=atom_types.repeat(zp),
                pos=atom_coords.repeat(zp, 1),
                x=atom_types.repeat(zp),
                do_mol_analysis=False,
                radius=sample.radius,
                mass=sample.mass,
                mol_volume=sample.mol_volume,
                z_prime=zp,
            ))
        return mols

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

    def init_acridine(self, buffer):
        if len(buffer) > 0:  # ensure same conformer as dataset, where possible
            if buffer.dataset[0].z_prime > 1:
                assert False, "not implemented!"

            atom_coords = buffer.dataset[0].pos
            atom_types = buffer.dataset[0].z
        else:
            assert False, "Must load acridine from buffer!"
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
        gfn_model, gfn_config, ema_model = self.init_gfn_model(energy_function)
        np.save(f'checkpoints/{self.run_name}_model_config', gfn_config)  # todo add path to saving directories

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
        #print("-1")

        with (wandb.init(project="GFN Energy",
                         config=flatten_wandb_params(self.args),
                         name=self.run_name,
                         tags=[self.args.tag])):
            #print("-0.5")
            wandb.watch(gfn_model,
                        log_graph=False,
                        log_freq=1000,
                        log='gradients')  # for gradient logging
            #print("-0.25")

            gfn_model.train()
            self.set_detect_anomaly(gfn_model, do_anomaly_detection=False)
            #print("0")
            for step_ind in trange(self.step_ind, self.args.epochs + 1):
                #print("1")
                metrics = dict()
                self.step_ind = step_ind
                if step_ind % 10 == 0:
                    self.set_loss_coeffs(step_ind)
                #print("2")
                exploration_std = get_exploration_std(step_ind,
                                                      self.args.exploratory,
                                                      self.args.wd_max_steps,
                                                      self.args.exploration_factor,
                                                      self.args.exploration_wd)
                #print("3")
                self.times['train_step_start'] = time()
                try:
                    step_type = self.train_logic(buffer, step_ind)
                    #print("4")
                    train_loss, loss_dict = self.train_step(
                        step_type,
                        energy_function,
                        gfn_model,
                        optimizers,
                        step_ind,
                        exploration_std,
                        buffer,
                        train_iterator,
                        repeats=self.args.repeats,
                        ema_model=ema_model,
                    )
                    #print("5")
                    if self.args.ema_decay is not None:
                        update_ema(gfn_model, ema_model, decay=self.args.ema_decay)
                    else:
                        ema_model = gfn_model
                    #print("6")
                    if step_type == 'Forward':
                        fwd_loss = train_loss
                        if loss_dict is not None:
                            fwd_loss_dict = loss_dict
                    elif step_type == 'Backward':
                        bwd_loss = train_loss
                        if loss_dict is not None:
                            bwd_loss_dict = loss_dict

                    if not oomed_out and self.args.grow_batch_size:
                        (buffer, train_mol_loader, test_mol_loader,
                         train_iterator, test_iterator) = self.increment_batch_size(
                            buffer, train_mol_loader,
                            test_mol_loader,
                            self.args.batch_growth_increment,
                            step_type,
                            train_iterator,
                            test_iterator)

                except (RuntimeError, ValueError) as e:  # if we do hit OOM, slash the batch size
                    (oomed_out, buffer, train_mol_loader,
                     test_mol_loader, train_iterator, test_iterator) = self.handle_train_epoch_error(
                        e, oomed_out, buffer,
                        train_mol_loader,
                        test_mol_loader,
                        optimizers, step_type,
                        train_iterator, test_iterator
                    )
                self.times['train_step_end'] = time()

                # evaluation work
                if (step_ind % self.args.eval_period == 0 and step_ind > 0) or step_ind == 50:
                    self.evaluation(ema_model, step_ind,
                                    buffer, train_mol_loader, test_mol_loader,
                                    energy_function, metrics)

                    if self.args.anchor_fwd_bwd and self.args.both_ways:
                        self.manage_prior_anchor(step_ind, metrics, gfn_model, ema_model)

                # train monitoring
                if step_ind % 10 == 0:
                    loss_record.append(fwd_loss + bwd_loss)
                    metrics['train/expl'] = exploration_std(0) if exploration_std is not None else 0
                    lr = self.step_lr_schedule(schedulers, optimizers)
                    self.anneal_reward(step_ind, temp_annealing_lambda, energy_function)
                    self.ten_step_reporting(bwd_loss, bwd_loss_dict, fwd_loss, fwd_loss_dict, metrics, optimizers)
                    loss_record = self.check_loss_explosion(loss_record, gfn_model, ema_model, optimizers)
                    wandb.log(metrics, step=step_ind)

                    if (self.fwd_tb_norm + self.bwd_tb_norm) < self.best_tb_norm:
                        self.best_tb_norm = self.fwd_tb_norm + self.bwd_tb_norm
                        torch.save(gfn_model.state_dict(), f'checkpoints/best_{self.run_name}_model_train.pt')
                        torch.save(ema_model.state_dict(), f'checkpoints/best_{self.run_name}_model_eval.pt')

                if step_ind % 50 == 0:  # save running model
                    torch.save(gfn_model.state_dict(), f'checkpoints/{self.run_name}_model_train.pt')
                    torch.save(ema_model.state_dict(), f'checkpoints/{self.run_name}_model_eval.pt')
                    self.save_modeller_state()

            torch.save(ema_model, f'checkpoints/{self.run_name}_model_final.pt')

    def reload_running_model(self, ema_model, gfn_model):
        gfn_model.load_state_dict(torch.load(f'checkpoints/{self.run_name}_model_train.pt'))
        ema_model.load_state_dict(torch.load(f'checkpoints/{self.run_name}_model_eval.pt'))
        gfn_model.train()
        ema_model.eval()

    def check_loss_explosion(self,
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

            hit_threshold = current_loss > threshold  # loss exploding
            increasing_loss = (torch.mean((diffs[
                                               -grace_time:] > 0).float()) > 0.8) and self.increasing_loss_cooldown <= 0  # loss slowly increasing

            if hit_threshold or increasing_loss:
                print("Losses increasing! Reloading best checkpoint and slashing LR.")
                if hit_threshold:
                    print("Hit loss threshold!")
                if increasing_loss:
                    print(f"Losses increasing over prior {grace_time} steps!")

                self.reload_running_model(ema_model, gfn_model)

                for opt in optimizers.values():
                    opt.state = defaultdict(dict)  # wipe also the momentum buffers
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
                   step_type,
                   energy_function,
                   gfn_model,
                   optimizers,
                   step_ind,
                   exploration_std,
                   buffer,
                   mol_iterator,
                   repeats,
                   ema_model,
                   ):
        if step_type == "Forward":
            do_forward = True
            do_backward = False
        elif step_type == "Backward":
            do_forward = False
            do_backward = True
        else:
            assert False

        discretizer = self.get_discretizer()

        optimizers['flow'].zero_grad(set_to_none=True)
        if do_forward:
            optimizers['fwd'].zero_grad(set_to_none=True)
            mol_batch = next(mol_iterator)
            if self.args.molecule_conditioning:
                # if doing molecule conditioning, augment over mol orientations, adjusting properly the embedding
                self.scramble_mol_and_embedding(mol_batch)
            else:
                mol_batch.orient_molecule(mode='std')

            loss, crystal_batch, loss_dict = self.fwd_train_step(
                energy_function,
                gfn_model,
                discretizer,
                exploration_std,
                mol_batch,
                return_exp=True,
                repeats=repeats,
                report_losses=True
            )
            if self.grow_buffer: #energy_function.energy_function == 'uma':  # save expensive stuff
                del crystal_batch.symmetry_operators, crystal_batch.gfn_energy
                buffer.add_to_staging(data_batch=crystal_batch)
            del crystal_batch

        elif do_backward:
            optimizers['bwd'].zero_grad(set_to_none=True)
            loss, loss_dict = self.bwd_train_step(
                step_ind,
                gfn_model,
                discretizer,
                buffer,
                energy_function,
                repeats=repeats,
                report_losses=True)
        else:
            assert False

        clean_loss = loss.item()
        self.update_rolling_tb(do_backward, do_forward, loss_dict, step_ind)

        skip_step = False
        if self.phase == 2:
            if self.bwd_tb_norm <= self.args.thermalization_conv_eps:  # hit stage 2 convergence criteria
                self.phase2to3(ema_model, gfn_model, 0.1, step_ind)

        if self.phase == 3:
            skip_step = self.update_controller(step_ind, do_backward, skip_step)

        if not skip_step:
            self.step_loss(do_backward, do_forward, gfn_model, loss, optimizers)

        loss_dict_cpu = {step_type + "_loss/" + key: value.cpu().detach().numpy() for key, value in
                         loss_dict.items()}

        del loss, loss_dict  # or whatever is large

        return clean_loss, loss_dict_cpu

    def update_rolling_tb(self, do_backward, do_forward, loss_dict, step_ind):
        T = 25  # effective target update time
        if do_forward and loss_dict is not None:
            if self.fwd_tb_norm == 10000:
                self.fwd_tb_norm = float(loss_dict['normed_tb'])
            else:
                dt = step_ind - self.last_fwd_it
                beta_fwd = np.exp(-dt / T)
                self.fwd_tb_norm = float(
                    self.fwd_tb_norm * beta_fwd + (1 - beta_fwd) * loss_dict['normed_tb'])
            self.last_fwd_it = step_ind
        if do_backward and loss_dict is not None:
            if self.bwd_tb_norm == 10000:
                self.bwd_tb_norm = float(loss_dict['normed_tb'])
            else:
                dt = step_ind - self.last_bwd_it
                beta_bwd = np.exp(-dt / T)
                self.bwd_tb_norm = float(
                    self.bwd_tb_norm * beta_bwd + (1 - beta_bwd) * np.nan_to_num(loss_dict['normed_tb'].cpu().detach(),
                                                                                 posinf=self.bwd_tb_norm))
            self.last_bwd_it = step_ind

    def update_controller(self, it, do_backward, skip_step):
        update_this_step = it % 20 == 0
        static_ceil = self.args.thermalization_conv_eps * self.args.btb_threshold
        static_floor = self.args.thermalization_conv_eps
        static_target = (static_ceil + static_floor) / 2

        if self.fwd_tb_norm > static_target:
            bwd_target = static_target
            bwd_ceil = static_ceil
            bwd_floor = static_floor
        else:
            bwd_target = self.fwd_tb_norm
            bwd_ceil = bwd_target * self.args.btb_threshold
            bwd_floor = bwd_target * 0.75

        metric = self.bwd_tb_norm
        coeff = 0.1
        err = (metric - bwd_target) / bwd_target  # if metric is large

        if metric > bwd_ceil:
            if update_this_step:
                self.args.fwd_to_bwd_ratio /= max(1.05, np.exp(coeff * err))
                #print(f"Firing ceil {metric:.2f}:{bwd_target:.2f}")
        elif metric < bwd_floor:
            if update_this_step:
                # this won't ameliorate below a good threshold - so just set it
                self.args.fwd_to_bwd_ratio = max(0.5, self.args.fwd_to_bwd_ratio)
                self.args.fwd_to_bwd_ratio *= 1.1
                #print(f"Firing floor {metric:.2f}:{bwd_target:.2f}")
            if do_backward:
                skip_step = True  # we're already doing too good
        elif update_this_step:
            self.args.fwd_to_bwd_ratio *= np.exp(-coeff * err)  # this is very negative -> pushes towards bwd, which is good

        self.args.fwd_to_bwd_ratio = np.clip(
            self.args.fwd_to_bwd_ratio, 0.1, 10)  # need even enough ratios to get reasonable updates to the metrics


        return skip_step

    def step_loss(self, do_backward, do_forward, gfn_model, loss, optimizers):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gfn_model.parameters(),
                                       self.args.gradient_norm_clip)  # gradient clipping
        if do_forward:
            optimizers['fwd'].step()
            optimizers['flow'].step()
        elif do_backward:
            optimizers['bwd'].step()
            optimizers['flow'].step()

    def fwd_train_step(self, energy_function, gfn_model, discretizer,
                       exploration_std, mol_batch, return_exp=False,
                       repeats: int = 10,
                       report_losses: bool = False):
        init_state = get_gfn_init_state(self.forward_batch_size, energy_function.data_ndim, self.device)
        log_T_tensor, sg_inds, condition = energy_function.get_conditioning_tensor(mol_batch,
                                                                                   z_primes=mol_batch.z_prime)
        mol_batch.sg_ind = sg_inds
        return get_gfn_forward_loss(self.args.fwd_loss_coeffs,
                                    init_state,
                                    gfn_model,
                                    energy_function.log_reward,
                                    discretizer,
                                    mol_batch,
                                    log_T_tensor,
                                    exploration_std=exploration_std,
                                    return_exp=return_exp,
                                    condition=condition,
                                    repeats=repeats,
                                    report_losses=report_losses)

    def bwd_train_step(self, step_ind, gfn_model, discretizer,
                       buffer, energy_function, repeats: int = 10,
                       report_losses: bool = False):
        if self.args.sampling == 'buffer':
            samples, rewards, crystal_batch, condition = buffer.sample(
                override_batch=int(self.backward_batch_size),
                randomize_orientations=True if self.args.molecule_conditioning else False,
            )
        else:
            assert False, f"sampling method {self.args.sampling} not implemented"

        if self.args.bwd_loss_coeffs.noised_fraction > 0:
            if buffer.noised_size <= 10*self.backward_batch_size or (step_ind % 10 == 0):  # draw noised samples
                condition, rewards, samples, crystal_batch = substitute_prior(
                    self.args.bwd_loss_coeffs, condition, crystal_batch,
                    energy_function, rewards, samples, buffer)
                if self.grow_buffer:
                    buffer.add_to_staging(data_batch=crystal_batch.cpu().detach())
                    buffer.add_to_noised(rewards, samples)
            else:
                rewards, samples = buffer.sample_from_noised(int(self.backward_batch_size))

        return get_gfn_backward_loss(self.args.bwd_loss_coeffs,
                                     samples.to(self.device),
                                     gfn_model,
                                     rewards.to(self.device),
                                     discretizer,
                                     condition=condition.to(self.device),
                                     repeats=repeats,
                                     report_losses=report_losses)

    def handle_train_epoch_error(self, e, oomed_out, buffer, train_mol_loader, test_mol_loader, optimizers, step_type,
                                 train_iterator, test_iterator):
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

            if step_type == 'Forward':
                self.forward_batch_size = max(1, int(self.forward_batch_size * 0.95))
                if self.forward_batch_size <= 1:
                    raise RuntimeError("Cascading OOM Failure")
                train_mol_loader = DataLoader(
                    train_mol_loader.dataset,
                    batch_size=self.forward_batch_size,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True,
                )
                test_mol_loader = DataLoader(
                    test_mol_loader.dataset,
                    batch_size=self.forward_batch_size,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True,
                )
                train_iterator = iter_forever(train_mol_loader)
                test_iterator = iter_forever(test_mol_loader)
                print(f"Reducing forward batch size to {self.forward_batch_size}")

            elif step_type == 'Backward':
                self.backward_batch_size = max(1, int(self.backward_batch_size * 0.95))
                if self.backward_batch_size <= 1:
                    raise RuntimeError("Cascading OOM Failure")
                print(f"Reducing backward batch size to {self.backward_batch_size}")

            gc.collect()
            torch.cuda.empty_cache()

            oomed_out = True
        else:
            raise e  # will simply raise error if other or if training on CPU
        return oomed_out, buffer, train_mol_loader, test_mol_loader, train_iterator, test_iterator


    def do_conditional_evaluation(self, energy_function, gfn_model, mol_loader,
                                  ):  # todo these functions could be cleaned up / consolidated
        self.times['eval_step_start'] = time()
        eval_discretizer = lambda bsz: uniform_discretizer(bsz, self.args.eval_T)

        eval_batch_size = self.args.eval_batch_size

        eval_rands = np.random.randint(len(mol_loader.dataset), size=eval_batch_size)
        mol_batch = collate_data_list([mol_loader.dataset[ind] for ind in eval_rands]).to(self.device)

        if self.args.molecule_conditioning:
            self.scramble_mol_and_embedding(mol_batch)

        init_state = get_gfn_init_state(eval_batch_size, energy_function.data_ndim, self.device)

        eval_metrics = {}
        eval_metrics.update(
            conditional_eval_step(energy_function,
                                  gfn_model,
                                  eval_discretizer,
                                  init_state,
                                  mol_batch,
                                  mols_to_sample=5,
                                  sample_sgs=self.args.space_groups if self.args.sg_conditioning else None,
                                  ))

        return eval_metrics

    def scramble_mol_and_embedding(self, mol_batch):
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

    def add_dataset_to_buffer(self,
                              dataset_path, buffer,
                              filter_unbound=True,
                              ):
        print("Loading prebuilt buffer")
        dataset = torch.load(dataset_path, weights_only=False)

        # todo exclude outside latents and warn if there are a shitload of them
        max_z_prime = max([int(elem.z_prime) for elem in dataset])
        assert max_z_prime == max(self.args.z_primes), "Preloaded data max z prime must match model"

        # filter unwanted SG
        dataset = [elem for elem in dataset if elem.sg_ind in self.args.space_groups]

        # filter unwanted Z'
        dataset = [elem for elem in dataset if elem.z_prime in self.args.z_primes]

        # canonicalize rotvecs (upper half-plane)
        batch = collate_data_list(dataset, max_z_prime=max_z_prime)
        batch.canonicalize_orientation()
        orientations = batch.aunit_orientation
        for ind, elem in enumerate(dataset):
            elem.aunit_orientation = orientations[ind][None, ...]

        # canonicalize aunit parameterizations
        batch = collate_data_list(dataset, max_z_prime=max_z_prime)
        batch.canonicalize_zp_aunits()
        aunits = batch.aunit_centroid
        for ind, elem in enumerate(dataset):
            elem.aunit_centroid = aunits[ind][None, ...]

        # filter invalid latents
        batch = collate_data_list(dataset, max_z_prime=max_z_prime)
        latents = batch.latent_params()
        good_inds = torch.argwhere(torch.all(latents.abs() <= 1, dim=1))  # valid latent space
        dataset = [dataset[ind] for ind in good_inds]

        # filter near-identical samples
        d_cut = 0.01
        latents = collate_data_list(dataset).latent_params()
        dmat = torch.cdist(latents, latents)
        keep = torch.zeros(len(latents), dtype=bool, device=latents.device)

        for i in range(len(latents)):
            # check if this point is far from all previously kept points
            if not (dmat[i, keep] < d_cut).any():
                keep[i] = True

        keep_inds = torch.arange(len(latents), device=latents.device)[keep]
        dataset = [dataset[ind] for ind in keep_inds]

        # todo remove this eventually
        if 'D:' in self.args.buffer_path and self.args.energy_function == 'uma':  # if we're on local, this takes forever
            dataset = dataset[:500]

        print("Re-featurizing preloaded buffer samples")
        dataset = featurize_dataset(dataset,
                                    self.device,
                                    self.args.energy_function,
                                    uma_path=self.args.uma_path)

        # always filter awful crystals
        dataset = [elem for elem in dataset if elem.reduction_en <= 1e-3]
        dataset = [elem for elem in dataset if elem.packing_coeff >= 0.55]
        dataset = [elem for elem in dataset if elem.packing_coeff <= 0.95]

        if filter_unbound:  # filter unbound states under this potential
            en_func = self.args.energy_function
            dataset = [elem for elem in dataset if elem[en_func] < 0]

        if self.args.molecule_conditioning:  # embed dataset
            assert max(self.args.z_primes) == 1, "Molecule conditioning not yet supported for Z'>1"
            print("Getting preloaded dataset molecule embeddings")
            dataset = embed_dataset(dataset, self.args.autoencoder_path, self.device, encoder=None)

        buffer.add(dataset)
        print(f"Buffer loaded with {len(dataset)} samples")

        return buffer

    def evaluation(self,
                   gfn_model,
                   step_ind,
                   buffer,
                   train_mol_loader,
                   test_mol_loader,
                   energy_function,
                   metrics):

        self.times['eval_step_start'] = time()
        '''setup'''
        eval_discretizer = lambda bsz: uniform_discretizer(bsz, self.args.eval_T)

        do_figs = step_ind % self.args.figs_period == 0

        '''fwd sampling'''
        self.times['eval_sampling_start'] = time()
        (flow_states, gauss_params_f, log_T_tensor, log_Z,
         log_Z_lb, log_Z_learned, log_pbs, log_pfs, log_r, sample_batch) = self.eval_sampling(
            buffer, energy_function, eval_discretizer, gfn_model, test_mol_loader)
        self.times['eval_sampling_end'] = time()

        self.times['eval_buffering_start'] = time()
        if len(buffer.staging_buffer) > 0:
            buffer.incorporate_staging_buffer()

        self.times['eval_buffering_end'] = time()

        self.times['eval_log_metrics_start'] = time()
        '''fwd analysis'''
        metrics.update(log_metrics(energy_function, log_Z, log_Z_lb, log_Z_learned, log_r,
                                   sample_batch, log_T_tensor, log_pfs, log_pbs, self.args, buffer))

        self.times['eval_log_metrics_end'] = time()
        self.times['eval_figs_start'] = time()
        if do_figs:
            # always sample from forward policy
            fig_dict = fwd_figs(buffer,
                                flow_states,
                                log_Z_learned,
                                log_pbs,
                                log_pfs,
                                log_r,
                                gauss_params_f,
                                sample_batch.detach().cpu(),
                                )
        else:
            fig_dict = {}
        self.times['eval_figs_end'] = time()

        '''bwd sampling and analysis are combined'''
        if self.args.both_ways or self.args.bwd:
            init_state = get_gfn_init_state(self.args.eval_num_samples,
                                            energy_function.data_ndim,
                                            self.device)

            self.times['eval_bwd_figs_start'] = time()
            bwd_metrics, bwd_fig_dict = bwd_figs(
                buffer, gfn_model,
                init_state, eval_discretizer,
                do_figs=do_figs)
            self.times['eval_bwd_figs_end'] = time()
            metrics.update(bwd_metrics)
            fig_dict.update(bwd_fig_dict)

        '''logging and wrap up'''
        self.times['eval_wrapup_start'] = time()

        if do_figs:
            adjust_fig_filesize(fig_dict)
            metrics.update(fig_dict)

        gfn_model.train()

        metrics.update({'Forward Batch Size': self.forward_batch_size})
        metrics.update({'Backward Batch Size': self.backward_batch_size})
        metrics.update({'Eval Batch Size': self.args.eval_batch_size})
        metrics.update(self.log_elapsed_times())
        self.times['eval_wrapup_end'] = time()

        self.times['eval_step_end'] = time()

        for key in metrics.keys():  # cleanup before logging
            if isinstance(metrics[key], np.ndarray):
                metrics[key] = np.nan_to_num(metrics[key])
            elif torch.is_tensor(metrics[key]):
                metrics[key] = torch.nan_to_num(metrics[key])

        wandb.log(metrics, step=step_ind)

        '''conditional sampling should be rewritten anyway '''  # TODO
        # if self.args.molecule_conditioning or self.args.sg_conditioning or self.args.zp_conditioning:
        #
        #     if step_ind % self.args.conditional_eval_period == 0:  # make conditional sampling figures
        #         conditional_metrics = self.do_conditional_evaluation(energy_function, gfn_model,
        #                                                              test_mol_loader,
        #                                                              )
        #         metrics.update(conditional_metrics)

    def eval_sampling(self, buffer, energy_function, eval_discretizer, gfn_model, test_mol_loader):
        flow_states_list = []
        eval_samples = []
        log_Z_list = []
        log_Z_lb_list = []
        log_Z_learned_list = []
        log_r_list = []
        log_flow_list = []
        log_T_list = []
        log_pfs_list = []
        log_pbs_list = []
        gauss_params_f_list = []
        while len(eval_samples) < self.args.eval_num_samples:
            try:
                eval_rands = np.random.randint(len(test_mol_loader.dataset), size=self.args.eval_batch_size)
                mol_batch = collate_data_list([test_mol_loader.dataset[ind] for ind in eval_rands]).to(self.device)

                if not self.args.molecule_conditioning:  # always std orientation if we're not conditioning
                    mol_batch.orient_molecule(mode='standard')
                else:  # if we are conditioning, randomly rotate and make sure we catch the embedding
                    self.scramble_mol_and_embedding(mol_batch)

                init_state = get_gfn_init_state(self.args.eval_batch_size,
                                                energy_function.data_ndim,
                                                self.device)

                gfn_model.eval()
                (flow_states, samples, log_r, log_Z, log_Z_lb,
                 log_Z_learned, sample_batch, condition, log_pfs, log_pbs, log_flow,
                 gauss_params_f,
                 log_T_tensor) = sample_eval_fwd_trajs(
                    init_state, gfn_model, eval_discretizer, energy_function, mol_batch)
                flow_states_list.append(flow_states)
                log_Z_list.append(log_Z)
                log_Z_lb_list.append(log_Z_lb)
                log_Z_learned_list.append(log_Z_learned)
                log_r_list.append(log_r)
                log_flow_list.append(log_flow)
                log_T_list.append(log_T_tensor)
                log_pfs_list.append(log_pfs)
                log_pbs_list.append(log_pbs)
                gauss_params_f_list.append(gauss_params_f)
                eval_samples.extend(sample_batch.batch_to_list())

                if self.grow_buffer:
                    buffer.add_to_staging(data_batch=sample_batch.cpu().detach())

            except (RuntimeError, ValueError) as e:
                print(f"Caught error: {str(e)}")
                if is_cuda_oom(e):
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
                else:
                    raise e
        flow_states = torch.cat(flow_states_list)
        log_Z = torch.stack(log_Z_list).mean()
        log_Z_lb = torch.stack(log_Z_lb_list).mean()
        log_Z_learned = torch.stack(log_Z_learned_list).mean()
        log_r = torch.cat(log_r_list)
        log_T_tensor = torch.cat(log_T_list)
        log_pfs = torch.cat(log_pfs_list)
        log_pbs = torch.cat(log_pbs_list)
        gauss_params_f = {}
        for key in gauss_params_f_list[0].keys():
            gauss_params_f[key] = torch.cat([d[key] for d in gauss_params_f_list])
        sample_batch = collate_data_list(eval_samples, skip_default_exclusion=True)
        return flow_states, gauss_params_f, log_T_tensor, log_Z, log_Z_lb, log_Z_learned, log_pbs, log_pfs, log_r, sample_batch

    def manage_prior_anchor(self, step_ind, metrics, gfn_model, ema_model):

        if self.phase == 1:
            metric = metrics['Max Latent KLD']
            "check threshold"
            if metric <= self.args.init_kld_threshold:
                self.phase1to2(ema_model, gfn_model, step_ind)

        metrics['Training Phase'] = self.phase
        metrics['Fwd to Bwd Ratio'] = self.args.fwd_to_bwd_ratio

    def phase1to2(self, ema_model, gfn_model, step_ind):
        #print("Hit initial KLD threshold. Moving to backward thermalization.")
        self.hit_init_kld = True
        "adjust loss coefficients"
        self.args.bwd_loss_coeffs.bwd_tb_z = 1.0
        self.bwd_loss_schedule['tb'] = [(0, 1.0), (step_ind, 0.1),
                                             (step_ind + self.args.phase_change_time, 1.0)]
        self.bwd_loss_schedule['mle'] = [(0, 1.0), (step_ind, 1),
                                              (step_ind + self.args.phase_change_time, 0.0)]
        self.bwd_loss_schedule['bwd_tb_z'] = [(0, 2.0), (step_ind, 1.0)]
        self.bwd_loss_schedule['noised_fraction'] = [(0, 0.0), (step_ind, self.args.anchor_noise_fraction)]
        self.bwd_loss_schedule['noise_level'] = [(0, 0.0), (step_ind, self.args.anchor_noise_level)]
        "set cooldowns"
        self.increasing_loss_cooldown = self.args.phase_change_time  # give it time to adjust to new loss landscape
        "align log Z to buffer (it will converge to this value)"
        # z = metrics['Bwd Empirical log Z LB']# todo come back to thinking about this
        # with torch.no_grad():
        #     ema_model.flow_model.weight.data = z
        #     gfn_model.flow_model.weight.data = z
        "save checkpoint"
        torch.save(gfn_model.state_dict(), f'checkpoints/{self.run_name}_model_train_hit_prior.pt')
        torch.save(ema_model.state_dict(), f'checkpoints/{self.run_name}_model_eval_hit_prior.pt')
        self.phase = 2
        self.grow_buffer = True

    def phase2to3(self, ema_model, gfn_model, init_rat, step_ind):
        #print("Thermalization complete. Moving to forward training & refinement.")
        self.phase = 3
        "save checkpoint"
        torch.save(gfn_model.state_dict(), f'checkpoints/{self.run_name}_model_train_thermalized.pt')
        torch.save(ema_model.state_dict(), f'checkpoints/{self.run_name}_model_eval_thermalized.pt')
        "adjust loss and balancing coefficients"
        self.args.fwd_to_bwd_ratio = init_rat
        self.bwd_loss_schedule['bwd_tb_z'] = [(0, 1.0), (step_ind, 0)]
        self.fwd_loss_schedule['tb'] = [(0, 1.0), (step_ind, 0.0),
                                             (step_ind + self.args.phase_change_time // 2, 1.0)]
        "set cooldown"
        self.increasing_loss_cooldown = self.args.phase_change_time
        self.grow_buffer = True

    def save_modeller_state(self):
        state = dict(
            phase=self.phase,
            fwd_tb_norm=self.fwd_tb_norm,
            bwd_tb_norm=self.bwd_tb_norm,
            step_ind=self.step_ind,
            last_fwd_it=self.last_fwd_it,
            last_bwd_it=self.last_bwd_it,
            fwd_loss_schedule=self.fwd_loss_schedule,
            bwd_loss_schedule=self.bwd_loss_schedule,
            fwd_to_bwd_ratio=self.args.fwd_to_bwd_ratio,
            increasing_loss_cooldown=self.increasing_loss_cooldown,
            lr_warmup_finished=self.lr_warmup_finished,
            hit_init_kld=self.hit_init_kld,
            forward_batch_size=self.forward_batch_size,
            backward_batch_size=self.backward_batch_size,
            grow_buffer = self.grow_buffer,
            best_tb_norm = self.best_tb_norm,
        )
        torch.save(state, f'checkpoints/{self.run_name}_modeller_state.pt')

    def load_modeller_state(self, path):
        if not os.path.exists(path):
            print("No modeller state found; starting fresh.")
            return

        state = torch.load(path, weights_only=False)

        self.phase = state['phase']
        self.step_ind = state['step_ind']
        self.fwd_tb_norm = state['fwd_tb_norm']
        self.bwd_tb_norm = state['bwd_tb_norm']
        self.last_fwd_it = state['last_fwd_it']
        self.last_bwd_it = state['last_bwd_it']
        self.fwd_loss_schedule = state['fwd_loss_schedule']
        self.bwd_loss_schedule = state['bwd_loss_schedule']
        self.args.fwd_to_bwd_ratio = state['fwd_to_bwd_ratio']
        self.increasing_loss_cooldown = state['increasing_loss_cooldown']
        self.lr_warmup_finished = state['lr_warmup_finished']
        self.hit_init_kld = state['hit_init_kld']
        self.forward_batch_size = state['forward_batch_size']
        self.backward_batch_size = state['backward_batch_size']
        self.grow_buffer = state['grow_buffer']
        self.best_tb_norm = state['best_tb_norm']


if __name__ == '__main__':
    modeller = Modeller()
    modeller.train()
