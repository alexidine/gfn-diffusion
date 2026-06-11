import gc
import os
from collections import defaultdict
from copy import deepcopy

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
from torch_geometric.loader import DataLoader
from tqdm import trange

from energies.molecular_crystal import MolecularCrystal
from energy_sampling.buffer import CrystalReplayBuffer, SimpleDataset
from energy_sampling.eval.utils import sample_eval_fwd_trajs
from energy_sampling.utils import iter_forever, \
    is_cuda_oom, get_annealing_factor, \
    parse_loss_schedules, dict2namespace, update_loss_schedule, \
    noise_buffer, atomic_save, get_discretizer, log_elapsed_times
from eval.evaluations import adjust_fig_filesize, log_metrics, fwd_figs, bwd_evaluation, analyze_buffer
from gflownet_losses import get_gfn_forward_loss, get_gfn_backward_loss
from models import GFN
from mxtaltools.common.training_utils import flatten_wandb_params
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list
from utils import get_train_args, get_gfn_init_state, set_seed, \
    uniform_discretizer, \
    featurize_dataset, embed_dataset, \
    update_ema

MODELLER_STATE_DEFAULTS = {
    'phase': 1,
    'fwd_tb_norm': 10000,
    'bwd_tb_norm': 10000,
    'best_tb_norm': 10000,
    'fwd_Z_lb': 10000,
    'bwd_Z_lb': 10000,
    'fwd_slope_err': 10000,
    'bwd_slope_err': 10000,
    'fwd_intercept_err': 10000,
    'bwd_intercept_err': 10000,
    'fwd_scatter_err': 10000,
    'bwd_scatter_err': 10000,
    'last_fwd_it': 0,
    'last_bwd_it': 0,
    'step_ind': 0,
    'increasing_loss_cooldown': 0,
    'lr_warmup_finished': False,
    'hit_init_kld': False,
    'fwd_to_bwd_ratio': 1.0,
    'fwd_loss_schedule': {},
    'bwd_loss_schedule': {},
    'batch_size': 1,
    'grow_buffer': False,
}


class Modeller:
    def __init__(self):
        self.args = get_train_args()
        torch.cuda.set_per_process_memory_fraction(self.args.cuda_memory_fraction, device=0)
        torch.cuda.init()  # create context with the cap already in place

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
            setattr(self, k, v)
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
            'step_ind': self.step_ind,
            'gfn_config': self.gfn_config,  # store once, reload from here
            'model_train': self.gfn_model.state_dict(),
            'model_eval': self.ema_model.state_dict(),
            'modeller_state': self._get_modeller_state_dict(),
        }
        path = self._checkpoint_path(tag)
        atomic_save(checkpoint, path)

    def load_model_state(self, checkpoint):
        self.gfn_config = checkpoint['gfn_config']
        self.gfn_model = GFN(**self.gfn_config).to(self.device)
        self.gfn_model.load_state_dict(checkpoint['model_train'])
        self.ema_model = deepcopy(self.gfn_model)
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

        if do_forward:
            step_type = "Forward"
        elif do_backward:
            step_type = "Backward"
        else:
            assert False
        return step_type

    def increment_batch_size(self, buffer,
                             train_mol_loader, test_mol_loader,
                             batch_growth_increment,
                             step_type,
                             train_iterator, test_iterator):
        # TODO rewrite
        if self.batch_size < self.args.max_batch_size:
            new_batch_size = min(self.args.max_batch_size,
                                 max(self.batch_size + 1, int(self.batch_size * batch_growth_increment)))
            self.batch_size = new_batch_size  # gradually increment batch size

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
        lr = optimizers['fwd'].param_groups[0]['lr']
        if not self.lr_warmup_finished:
            schedulers['policy_1'].step()
            schedulers['policy_1b'].step()

            if lr >= self.args.lr_policy:
                self.lr_warmup_finished = True

        elif lr > self.args.min_lr:
            schedulers['policy_2'].step()
            schedulers['policy_2b'].step()

        schedulers['flow'].step()

        return lr

    def ten_step_reporting(self, bwd_loss, bwd_loss_dict, fwd_loss, fwd_loss_dict, metrics, optimizers):
        # TODO update
        for opt_type in ['fwd', 'bwd', 'flow']:
            metrics.update({f'lr_{opt_type}': optimizers[opt_type].param_groups[0]['lr']})
        for direction in ('fwd', 'bwd'):
            for stat in ('tb_norm', 'slope_err', 'intercept_err', 'scatter_err'):
                key = f'{direction}_{stat}'
                metrics[key] = getattr(self, key)

        metrics['Fwd to Bwd Ratio'] = self.args.fwd_to_bwd_ratio
        metrics['Best TB Norm'] = self.best_tb_norm
        metrics.update(log_elapsed_times(self.times))
        metrics['Forward Loss'] = fwd_loss
        metrics['Backward Loss'] = bwd_loss
        if fwd_loss_dict is not None:
            metrics.update(fwd_loss_dict)
            fwd_loss_dict = None
        if bwd_loss_dict is not None:
            metrics.update(bwd_loss_dict)
            bwd_loss_dict = None

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
        energy_function = MolecularCrystal(**energy_config)
        return energy_function

    def init_gfn(self, energy_function):
        reload = False  # TODO cleanup / unify model state
        if self.args.checkpoint_path is not None:
            reload = True
            print(f"Loading model from checkpoint {self.args.checkpoint_path}")
            self.load_model_state(self.args.checkpoint_path)

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
            self.gfn_config = dict(
                dim=energy_function.data_ndim,
                s_emb_dim=self.args.s_emb_dim,
                hidden_dim=self.args.hidden_dim,
                conditions_dim=self.get_conditioning_dim(),
                conditions_type='molecule' if self.args.molecule_conditioning else 'vector',
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
                conditional=any([
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
                plist += [{'params': self.gfn_model.conditions_embedding_model.parameters()}]

            return plist

        flow_params = self.gfn_model.flow_model.parameters()

        optimizers = {}
        weight_decay = self.args.weight_decay if self.args.use_weight_decay else 0
        optimizers['fwd'] = torch.optim.AdamW(get_policy_params(self.gfn_model), init_fwd_lr,
                                              weight_decay=weight_decay)
        optimizers['bwd'] = torch.optim.AdamW(get_policy_params(self.gfn_model), init_bwd_lr,
                                              weight_decay=weight_decay)
        optimizers['flow'] = torch.optim.AdamW(flow_params, init_flow_lr, weight_decay=weight_decay)

        schedulers = {}
        lr_warmup_lambda = get_annealing_factor(1,
                                                self.args.lr_warmup_ratio,
                                                self.args.lr_warmup_time,
                                                10)
        lr_annealing_lambda = get_annealing_factor(self.args.lr_policy,
                                                   self.args.min_lr,
                                                   self.args.lr_anneal_time,
                                                   10)

        schedulers['policy_1'] = lr_scheduler.MultiplicativeLR(
            optimizers['fwd'], lr_lambda=lambda epoch: lr_warmup_lambda)
        schedulers['policy_2'] = lr_scheduler.MultiplicativeLR(
            optimizers['fwd'], lr_lambda=lambda epoch: lr_annealing_lambda)

        schedulers['policy_1b'] = lr_scheduler.MultiplicativeLR(
            optimizers['bwd'], lr_lambda=lambda epoch: lr_warmup_lambda)
        schedulers['policy_2b'] = lr_scheduler.MultiplicativeLR(
            optimizers['bwd'], lr_lambda=lambda epoch: lr_annealing_lambda)

        flow_annealing_lambda = get_annealing_factor(1,
                                                     0.1,
                                                     self.args.lr_anneal_time,
                                                     10)
        schedulers['flow'] = lr_scheduler.MultiplicativeLR(optimizers['flow'],
                                                           lr_lambda=lambda epoch: flow_annealing_lambda)

        return optimizers, schedulers

    def init_buffers_datasets(self, energy_function):  # TODO update/rewrite
        # load dataset of prebuilt and scored molecular crystals into the buffer
        buffer = CrystalReplayBuffer(
            self.args.buffer_size,
            'cpu',
            energy_function,
            self.batch_size,
            beta=self.args.beta,
            rank_weight=self.args.rank_weight,
            prioritized=self.args.prioritized,
            keep_initial_samples=self.args.buffer_path is not None,
            max_z_prime=energy_function.max_z_prime,
            buffer_dist_cutoff=self.args.buffer_dist_cutoff,
            noised_buffer_length=self.args.noised_buffer_length,
            noised_max_steps=self.args.noised_max_steps,
            kT_range=self.args.kT_range
        )
        if ((self.args.both_ways or self.args.bwd) and
                self.args.buffer_path is not None):  # preload samples into the buffer
            prior_file = torch.load(self.args.buffer_path, weights_only=False)
            energy_function.lj_rescale = prior_file['thermal_scaling_factor']
            self.log_noise_range = prior_file['log_noise_range']
            buffer = self.add_dataset_to_buffer(prior_file,
                                                buffer,
                                                filter_unbound=True)

        if len(buffer) > 0 and (self.args.molecule != 'qm9'):
            mols_list = self.init_mol_from_buffer(buffer, self.args.z_primes)
            train_mols_list = []
            test_mols_list = []
            while len(train_mols_list) < int(self.args.max_batch_size * 1.5):
                for mol in mols_list:
                    train_mols_list.append(mol.clone())
            while len(test_mols_list) < int(self.args.max_batch_size * 1.5):
                for mol in mols_list:
                    test_mols_list.append(mol.clone())

        elif self.args.molecule == 'qm9':
            assert False, "QM9 conditional modelling currently deprecated"
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
            assert False, "conditioning currently deprecated"
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

        if hasattr(buffer.dataset[0], 'mace_gas_pot'):  # TODO REWRITE ALL THIS
            pot = buffer.dataset[0].mace_gas_pot
            for elem in train_mols_list:
                elem.mace_gas_pot = pot
            for elem in test_mols_list:
                elem.mace_gas_pot = pot
        train_mol_loader = DataLoader(
            train_mols_list,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
        test_mol_loader = DataLoader(
            test_mols_list,
            batch_size=self.batch_size,
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

    def init_prior_dataset(self):
        prior_data = torch.load(self.args.prior_path, weights_only=False)
        self.prior = SimpleDataset(prior_data,
                                   device='cpu',
                                   max_z_prime=max(self.args.z_primes),
                                   x_fn='latent_params',
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
            energy_function = self.init_energy_function()

            # Model Init
            self.init_gfn(energy_function)

            # opt init
            optimizers, schedulers = self.init_schedulers_optimizers()

            # buffer & loaders init
            self.init_prior_dataset()
            self.init_mol_dataset()
            # (buffer, train_mol_loader, test_mol_loader,
            #  train_iterator, test_iterator) = self.init_buffers_datasets(energy_function)

            # todo update this tracking
            fwd_loss_dict = None
            bwd_loss_dict = None
            oomed_out = False
            fwd_loss, bwd_loss = 0, 0
            loss_record = []

            self.times['initialization_end'] = time()

            wandb.watch(self.gfn_model,
                        log_graph=False,
                        log_freq=1000,
                        log='gradients')

            self.gfn_model.train()
            self.set_detect_anomaly(do_anomaly_detection=self.args.anomaly_detection)
            init_step = self.step_ind * 1
            for self.step_ind in trange(init_step, self.args.epochs + 1):
                metrics = dict()
                if self.step_ind % 10 == 0:
                    self.set_loss_coeffs()

                self.times['train_step_start'] = time()
                try:
                    step_type = self.train_logic()
                    # print("4")
                    train_loss, loss_dict = self.train_step(
                        step_type,
                        energy_function,
                        optimizers,
                        buffer,
                        train_iterator,
                        repeats=self.args.repeats,
                    )
                    # print("5")
                    if self.args.ema_decay is not None:
                        update_ema(self.gfn_model, self.ema_model, decay=self.args.ema_decay)
                    else:
                        self.ema_model = self.gfn_model
                    # print("6")
                    if step_type == 'Forward':
                        fwd_loss = train_loss
                        if loss_dict is not None:
                            fwd_loss_dict = loss_dict
                    elif step_type == 'Backward':
                        bwd_loss = train_loss
                        if loss_dict is not None:
                            bwd_loss_dict = loss_dict

                    if (not oomed_out or (self.step_ind % 500 == 0)) and self.args.grow_batch_size:
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
                if (self.step_ind % self.args.eval_period == 0 and self.step_ind > 0) or self.step_ind == 50:
                    self.evaluation(buffer, train_mol_loader, test_mol_loader,
                                    energy_function, metrics)

                    if self.args.anchor_fwd_bwd and self.args.both_ways:
                        self.manage_prior_anchor(metrics)

                # train monitoring
                if self.step_ind % 10 == 0:
                    loss_record.append(fwd_loss + bwd_loss)
                    lr = self.step_lr_schedule(schedulers, optimizers)
                    self.ten_step_reporting(bwd_loss, bwd_loss_dict, fwd_loss, fwd_loss_dict, metrics, optimizers)
                    loss_record = self.check_loss_explosion(loss_record, optimizers)
                    wandb.log(metrics, step=self.step_ind)

                    if (self.fwd_tb_norm + self.bwd_tb_norm) < self.best_tb_norm:
                        self.best_tb_norm = self.fwd_tb_norm + self.bwd_tb_norm
                        self.save_checkpoint('best')

                if self.step_ind % 50 == 0:  # save running model
                    self.save_checkpoint('running')

            self.save_checkpoint('final')

    def reload_running_model(self):
        self.gfn_model, self.ema_model = self.load_model_state(self._checkpoint_path('running'))
        self.gfn_model.load_state_dict(torch.load(
            f'checkpoints/{self.run_name}_model_eval.pt'))  # eval model is a more stable basline to retry from
        self.ema_model.load_state_dict(torch.load(f'checkpoints/{self.run_name}_model_eval.pt'))
        self.gfn_model.train()
        self.ema_model.eval()

    def check_loss_explosion(self,
                             loss_record: list,
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

                self.reload_running_model()

                for key, opt in optimizers.items():
                    if hit_threshold:
                        opt.state = defaultdict(dict)  # wipe also the momentum buffers
                    for g in opt.param_groups:
                        if g['lr'] > self.args.min_lr:
                            g['lr'] = max(g['lr'] * 0.9, self.args.min_lr)

                if not hasattr(self, 'lr_cut_count'):
                    self.lr_cut_count = 1
                else:
                    self.lr_cut_count += 1

                if self.lr_cut_count > 5:
                    self.lr_warmup_finished = True

                if increasing_loss:
                    self.increasing_loss_cooldown = grace_time

                if hit_threshold:
                    to_keep = torch.argwhere(losses <= threshold).flatten().tolist()
                    loss_record = [rec for ind, rec in enumerate(loss_record) if ind in to_keep]

        return loss_record

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
                   energy_function,
                   optimizers,
                   buffer,
                   mol_iterator,
                   repeats,
                   ):
        if step_type == "Forward":
            do_forward = True
            do_backward = False
        elif step_type == "Backward":
            do_forward = False
            do_backward = True
        else:
            assert False

        discretizer = get_discretizer(self.args.integrator)

        optimizers['flow'].zero_grad(set_to_none=True)
        if do_forward:
            optimizers['fwd'].zero_grad(set_to_none=True)
            mol_batch = next(mol_iterator)
            mol_batch.orient_molecule(mode='std')

            loss, crystal_batch, loss_dict, rewards, log_importance_weight = self.fwd_train_step(
                energy_function,
                discretizer,
                mol_batch,
                return_exp=True,
                repeats=repeats,
                report_losses=True
            )
            # p_add = max(0.1, min(1.0, (1 / 10) / self.args.fwd_to_bwd_ratio))
            # if self.grow_buffer and np.random.rand() < p_add:
            #     del crystal_batch.symmetry_operators, crystal_batch.gfn_energy
            #     buffer.add_to_staging(data_batch=crystal_batch.cpu().detach(),
            #                           importance_weight=log_importance_weight.cpu().detach())
            del crystal_batch

        elif do_backward:
            optimizers['bwd'].zero_grad(set_to_none=True)
            loss, loss_dict = self.bwd_train_step(
                discretizer,
                buffer,
                repeats=repeats,
                report_losses=True)
        else:
            assert False

        clean_loss = loss.item()
        self.update_rolling_tb(do_backward, do_forward, loss_dict, self.step_ind)

        skip_step = False
        if self.phase == 2:
            if self.bwd_tb_norm <= self.args.thermalization_conv_eps:  # hit stage 2 convergence criteria
                self.phase2to3(self.args.min_fwd_bwd_ratio, self.step_ind)

        if self.phase == 3:
            skip_step = self.update_controller(self.step_ind, do_backward, skip_step)

        if not skip_step:
            self.step_loss(do_backward, do_forward, loss, optimizers)

        loss_dict_cpu = {step_type + "_loss/" + key: (value.cpu().detach().numpy() if torch.is_tensor(value) else value)
                         for key, value in
                         loss_dict.items()}

        # loss = None
        # loss_dict = None
        torch.cuda.synchronize()

        return clean_loss, loss_dict_cpu

    def update_rolling_tb(self, do_backward, do_forward, loss_dict):
        T = 25  # effective target update time
        if do_forward:
            self.fwd_tb_norm = self._update_rolling(
                self.fwd_tb_norm,
                loss_dict['normed_tb'],
                self.last_fwd_it,
                self.step_ind,
                T,
                sanitize=True,
            )
            self.fwd_Z_lb = self._update_rolling(
                self.fwd_Z_lb,
                loss_dict['log_Z_lb'],
                self.last_fwd_it,
                self.step_ind,
                T,
                sanitize=True,
            )
            self.fwd_intercept_err = self._update_rolling(
                self.fwd_intercept_err,
                loss_dict['intercept_err'],
                self.last_fwd_it,
                self.step_ind,
                T,
                sanitize=True,
            )
            self.fwd_slope_err = self._update_rolling(
                self.fwd_slope_err,
                loss_dict['slope_err'],
                self.last_fwd_it,
                self.step_ind,
                T,
                sanitize=True,
            )
            self.fwd_scatter_err = self._update_rolling(
                self.fwd_scatter_err,
                loss_dict['scatter_err'],
                self.last_fwd_it,
                self.step_ind,
                T,
                sanitize=True,
            )
            self.last_fwd_it = self.step_ind

        if do_backward:
            self.bwd_tb_norm = self._update_rolling(
                self.bwd_tb_norm,
                loss_dict['normed_tb'].cpu().detach(),
                self.last_bwd_it,
                self.step_ind,
                T,
                sanitize=True,
            )
            self.bwd_Z_lb = self._update_rolling(
                self.bwd_Z_lb,
                loss_dict['log_Z_lb'],
                self.last_bwd_it,
                self.step_ind,
                T,
                sanitize=True,
            )
            self.bwd_intercept_err = self._update_rolling(
                self.bwd_intercept_err,
                loss_dict['intercept_err'],
                self.last_bwd_it,
                self.step_ind,
                T,
                sanitize=True,
            )
            self.bwd_slope_err = self._update_rolling(
                self.bwd_slope_err,
                loss_dict['slope_err'],
                self.last_bwd_it,
                self.step_ind,
                T,
                sanitize=True,
            )
            self.bwd_scatter_err = self._update_rolling(
                self.bwd_scatter_err,
                loss_dict['scatter_err'],
                self.last_bwd_it,
                self.step_ind,
                T,
                sanitize=True,
            )
            self.last_bwd_it = self.step_ind

    def _update_rolling(
            self,
            value,
            new_value,
            last_it,
            T,
            *,
            sanitize=False,
    ):
        if value == 10000:
            value = float(new_value.item() if torch.is_tensor(new_value) else new_value)
        else:
            dt = self.step_ind - last_it
            beta = np.exp(-dt / T)
            if sanitize:
                new_value = np.nan_to_num(
                    float(new_value),
                    posinf=value,
                    neginf=value,
                )
            value = value * beta + (1 - beta) * float(new_value)
        return value

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

    def step_loss(self, do_backward, do_forward, loss, optimizers):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gfn_model.parameters(),
                                       self.args.gradient_norm_clip)  # gradient clipping
        if do_forward:
            optimizers['fwd'].step()
            optimizers['flow'].step()
        elif do_backward:
            optimizers['bwd'].step()
            optimizers['flow'].step()

    def fwd_train_step(self, energy_function, discretizer,
                       mol_batch, return_exp=False,
                       repeats: int = 10,
                       report_losses: bool = False,
                       ):
        init_state = get_gfn_init_state(self.batch_size, energy_function.data_ndim, self.device)
        log_T_tensor, sg_inds, condition = energy_function.get_conditioning_tensor(mol_batch,
                                                                                   z_primes=mol_batch.z_prime)
        mol_batch.sg_ind = sg_inds
        return get_gfn_forward_loss(self.args.fwd_loss_coeffs,
                                    init_state,
                                    self.gfn_model,
                                    energy_function.log_reward,
                                    discretizer,
                                    mol_batch,
                                    log_T_tensor,
                                    exploration_std=None,
                                    return_exp=return_exp,
                                    condition=condition,
                                    repeats=repeats,
                                    report_losses=report_losses,
                                    )

    def bwd_train_step(self, discretizer,
                       buffer, repeats: int = 10,
                       report_losses: bool = False):
        with torch.no_grad():
            if self.args.sampling == 'buffer':
                samples, rewards, crystal_batch, condition = buffer.sample(
                    override_batch=int(self.batch_size),
                    randomize_orientations=True if self.args.molecule_conditioning else False,
                    override_sampler=None,
                    return_sample_inds=False,
                )
            else:
                assert False, f"sampling method {self.args.sampling} not implemented"

            if self.args.bwd_loss_coeffs.noised_fraction > 0:
                if self.args.bwd_loss_coeffs.noised_fraction == 1:
                    rewards, samples, b_inds = buffer.sample_from_noised(int(self.batch_size))
                else:  # todo test this
                    noisy_rewards, noisy_samples, b_inds = buffer.sample_from_noised(int(self.batch_size))
                    replace_inds = np.random.choice(len(samples), max(1,
                                                                      int(self.args.bwd_loss_coeffs.noised_fraction * len(
                                                                          samples))), replace=False)
                    mask = np.zeros(len(samples), dtype=bool)
                    mask[replace_inds] = True
                    for ind in range(len(samples)):
                        if mask[ind]:
                            samples[ind] = noisy_samples[ind]
                            rewards[ind] = noisy_rewards[ind]

                    b_inds = b_inds[mask]

        loss, loss_dict, btb_losses = get_gfn_backward_loss(self.args.bwd_loss_coeffs,
                                                            samples.to(self.device),
                                                            self.gfn_model,
                                                            rewards.to(self.device),
                                                            discretizer,
                                                            condition=condition.to(self.device),
                                                            repeats=repeats,
                                                            report_losses=report_losses)

        if self.args.bwd_loss_coeffs.noised_fraction > 0:
            buffer.update_noised_losses(btb_losses, b_inds)

        return loss, loss_dict

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

            self.batch_size = max(1, int(self.batch_size * 0.95))
            if self.batch_size <= 1:
                raise RuntimeError("Cascading OOM Failure")
            train_mol_loader = DataLoader(
                train_mol_loader.dataset,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )
            test_mol_loader = DataLoader(
                test_mol_loader.dataset,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )
            train_iterator = iter_forever(train_mol_loader)
            test_iterator = iter_forever(test_mol_loader)

            if self.batch_size <= 1:
                raise RuntimeError("Cascading OOM Failure")
            print(f"Reducing batch size to {self.batch_size}")

            gc.collect()
            torch.cuda.empty_cache()

            oomed_out = True
        else:
            raise e  # will simply raise error if other or if training on CPU
        return oomed_out, buffer, train_mol_loader, test_mol_loader, train_iterator, test_iterator

    def add_dataset_to_buffer(self,  # TODO rewrite
                              prior_file, buffer,
                              filter_unbound=True,
                              ):
        print("Loading prebuilt buffer")
        prior_batch = prior_file['prior_batch']
        dataset = prior_batch.batch_to_list()

        self.process_anchors(buffer, dataset, filter_unbound)

        noised_batch = prior_file['noised_batch']
        noised_latents = noised_batch.latent_params()
        noised_rewards = buffer.energy_function.prebuilt_sample_to_reward(noised_batch,
                                                                          self.args.temperature)

        good_inds = torch.argwhere(noised_rewards >= buffer.reward_clip).flatten()
        buffer.add_to_noised(noised_rewards[good_inds],
                             noised_latents[good_inds],
                             losses=torch.zeros_like(noised_rewards[good_inds]),
                             override_size=True,
                             )

        return buffer

    def process_anchors(self, buffer, dataset, filter_unbound):  # TODO rewrite
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

        # filter reasonable densities
        dataset = [elem for elem in dataset if elem.packing_coeff >= 0.55]
        dataset = [elem for elem in dataset if elem.packing_coeff <= 0.95]

        # # filter near-identical samples
        # d_cut = 0.05  # should be relatively sparse or the local density bias becomes large
        # latents = collate_data_list(dataset).latent_params()
        # keep = thin_large_dmat_block(latents.to(self.args.device),
        #                              torch.tensor([elem.lj for elem in dataset], device=self.args.device),
        #                              d_cut).cpu()
        # keep_inds = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        # dataset = [dataset[i] for i in keep_inds]

        # todo remove this eventually, hopefully
        if 'D:' in self.args.buffer_path and self.args.energy_function in ['uma',
                                                                           'mace']:  # if we're on local, this takes forever
            dataset = dataset[:250]

        print("Re-featurizing preloaded buffer samples")
        dataset = featurize_dataset(dataset,
                                    self.device,
                                    self.args.energy_function,
                                    mlip_path=self.args.mlip_path,
                                    batch_size=500)
        # always filter awful crystals
        # re-filter this, as sometimes reparameterization happens inside the feat function
        dataset = [elem for elem in dataset if elem.packing_coeff >= 0.55]
        dataset = [elem for elem in dataset if elem.packing_coeff <= 0.95]
        dataset = [elem for elem in dataset if elem.reduction_en <= 1e-3]
        if filter_unbound:  # filter unbound states under this potential
            dataset = [elem for elem in dataset if elem[self.args.energy_function] < 0]
        #  # todo rewrite
        # if self.args.molecule_conditioning:  # embed dataset
        #     assert max(self.args.z_primes) == 1, "Molecule conditioning not yet supported for Z'>1"
        #     print("Getting preloaded dataset molecule embeddings")
        #     dataset = embed_dataset(dataset, self.args.autoencoder_path, self.device, encoder=None)
        buffer.add_init(dataset)
        print(f"Buffer loaded with {len(dataset)} anchor states")

    def evaluation(self,  # TODO major revision
                   buffer,
                   train_mol_loader,
                   test_mol_loader,
                   energy_function,
                   metrics):

        self.times['eval_step_start'] = time()
        '''setup'''
        eval_discretizer = lambda bsz: uniform_discretizer(bsz, self.args.eval_T)

        do_figs = self.step_ind % self.args.figs_period == 0

        '''fwd sampling'''
        self.times['eval_sampling_start'] = time()
        (flow_states, gauss_params_f, log_T_tensor, log_Z,
         log_Z_lb, log_Z_learned, log_pbs, log_pfs, log_r, sample_batch) = self.eval_sampling(
            buffer, energy_function, eval_discretizer, test_mol_loader)
        self.times['eval_sampling_end'] = time()

        # if len(buffer.staging_buffer) > 0:
        #     buffer.incorporate_staging_buffer()

        log_z, b_log_pbs, b_log_pfs, b_means_b, b_means_f, b_vars_b, b_vars_f, backward_flow_states, b_log_r, b_packing_coeff = analyze_buffer(
            buffer, eval_discretizer, self.ema_model, self.batch_size)
        log_importance_weight = ((b_log_r - log_z) - (b_log_pfs.sum(dim=-1) - b_log_pbs.sum(dim=-1))).cpu()

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
            self.times['eval_bwd_figs_start'] = time()
            bwd_metrics, bwd_fig_dict, rewards, btb_residual, normed_btb_residual = bwd_evaluation(
                log_z, b_log_pbs, b_log_pfs, b_means_b, b_means_f, b_vars_b, b_vars_f, backward_flow_states, b_log_r,
                b_packing_coeff, log_importance_weight,
                do_figs=do_figs)
            self.times['eval_bwd_figs_end'] = time()

            metrics.update(bwd_metrics)
            fig_dict.update(bwd_fig_dict)

            if len(buffer.noised_rewards) > 0:
                self.grow_noised_buffer(buffer, energy_function, log_importance_weight, rewards)

        # if len(buffer) > buffer.buffer_size:
        #     buffer.truncate_buffer(log_importance_weight)

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

        wandb.log(metrics, step=self.step_ind)

    def grow_noised_buffer(self, buffer, energy_function, log_importance_weight, rewards, alpha: float = 1.0,
                           eps: float = 1.0e-2, rand_frac: float = 0.5):  # TODO rewrite
        self.times['eval_bwd_noising_start'] = time()

        noised_losses = np.array(buffer.noised_losses)
        noised_train_steps = np.array(buffer.noised_select_counts)

        loss_cut = min(1, np.quantile(noised_losses, 0.25))
        good_loss = (noised_losses <= loss_cut) & (noised_losses > 0)
        old_enough = (noised_train_steps >= self.args.noised_max_steps)
        too_old = (noised_train_steps >= self.args.noised_max_steps * 2)
        samples_to_be_replaced = (good_loss & old_enough) | too_old

        num_to_replace = sum(samples_to_be_replaced)
        # if it's already full, let it shrink gracefully, while keeping some fresh samples
        if buffer.noised_buffer_length < (len(noised_losses) - num_to_replace):
            num_to_replace = max(1, num_to_replace // 2)

        if num_to_replace >= 4:
            n_rands = max(1, int(num_to_replace * rand_frac))
            rand_inds = np.random.choice(len(buffer), n_rands, replace=True)
            log_importance_weight_i = torch.nan_to_num(log_importance_weight,
                                                       nan=-20, posinf=-20, neginf=-20)
            log_importance_weight_i = (alpha * log_importance_weight_i).clip(
                min=-20,
                max=min(80, log_importance_weight_i.quantile(0.99)))

            importance_weight = (log_importance_weight_i).exp()
            good_reward = rewards > (rewards.quantile(0.05))  # limit importance sampling to high reward samples
            importance_weight[~good_reward] = 1e-20
            importance_weight = importance_weight.numpy().astype(np.float64)
            importance_weight /= importance_weight.sum()

            try:
                sample_inds = np.random.choice(len(buffer), num_to_replace - n_rands, p=importance_weight, replace=True)
            except ValueError:
                print("Value error in importance weights")
                print(importance_weight)
                print(log_importance_weight_i)
                sample_inds = np.random.choice(len(buffer), num_to_replace - n_rands, replace=True)

            sample_inds = np.concatenate([sample_inds, rand_inds])
            new_rewards, samples = noise_buffer(self.log_noise_range,
                                                buffer,
                                                energy_function,
                                                sample_inds=sample_inds, )

            # good_inds = torch.argwhere(rewards >= buffer.reward_clip).flatten()
            buffer.purge_noised_by_index(np.argwhere(samples_to_be_replaced).flatten())
            buffer.add_to_noised(new_rewards,  # [good_inds],
                                 samples,  # [good_inds],
                                 losses=torch.zeros_like(new_rewards),
                                 override_size=True)  # [good_inds]))

        self.times['eval_bwd_noising_end'] = time()

    def eval_sampling(self, buffer, energy_function, eval_discretizer, test_mol_loader):
        flow_states_list = []  # todo rewrite / consolidate
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

                mol_batch.orient_molecule(mode='standard')

                init_state = get_gfn_init_state(self.args.eval_batch_size,
                                                energy_function.data_ndim,
                                                self.device)

                (flow_states, samples, log_r, log_Z, log_Z_lb,
                 log_Z_learned, sample_batch, condition, log_pfs, log_pbs, log_flow,
                 gauss_params_f,
                 log_T_tensor) = sample_eval_fwd_trajs(
                    init_state, self.ema_model, eval_discretizer, energy_function, mol_batch)
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
                #
                # if self.grow_buffer:
                #     log_importance_weight = (
                #             (log_r - log_Z_learned) - (log_pfs.sum(-1) - log_pbs.sum(-1))).cpu().detach()
                #     buffer.add_to_staging(importance_weight=log_importance_weight,
                #                           data_batch=sample_batch.cpu().detach())

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

    def manage_prior_anchor(self, metrics):  # todo rewrite

        if self.phase == 1:
            metric = metrics['Max Latent KLD']
            "check threshold"
            if metric <= self.args.init_kld_threshold:
                self.phase1to2(self.step_ind)

        metrics['Training Phase'] = self.phase
        metrics['Fwd to Bwd Ratio'] = self.args.fwd_to_bwd_ratio

    def phase1to2(self):
        # print("Hit initial KLD threshold. Moving to backward thermalization.")
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
        self.increasing_loss_cooldown = self.args.phase_change_time  # give it time to adjust to new loss landscape
        "align log Z to buffer (it will converge to this value)"
        # z = metrics['Bwd Empirical log Z LB']# todo come back to thinking about this
        # with torch.no_grad():
        #     ema_model.flow_model.weight.data = z
        #     gfn_model.flow_model.weight.data = z
        "save checkpoint"
        self.save_checkpoint('prior')
        self.phase = 2
        self.grow_buffer = True

    def phase2to3(self, init_rat):
        # print("Thermalization complete. Moving to forward training & refinement.")
        self.phase = 3
        "save checkpoint"
        self.save_checkpoint('prior_equil')
        "adjust loss and balancing coefficients"
        self.args.fwd_to_bwd_ratio = init_rat
        self.bwd_loss_schedule['bwd_tb_z'] = [(0, 1.0), (self.step_ind, 0)]
        self.fwd_loss_schedule['tb'] = [(0, 1.0), (self.step_ind, 0.0),
                                        (self.step_ind + self.args.phase_change_time // 2, 1.0)]
        "set cooldown"
        self.increasing_loss_cooldown = self.args.phase_change_time
        self.grow_buffer = True
        self.std_boost_prob = self.args.p3_widevar_prob
        self.std_boost_var = self.args.p3_widevar_var


if __name__ == '__main__':
    modeller = Modeller()
    modeller.train()
