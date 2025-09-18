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
import wandb
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from tqdm import trange

from buffer import CrystalReplayBuffer
from energies.molecular_crystal import MolecularCrystal
from energy_sampling.utils import iter_forever
from eval.evaluations import eval_step, conditional_eval_step
from gflownet_losses import get_gfn_forward_loss, get_gfn_backward_loss
from models import GFN
from mxtaltools.common.training_utils import flatten_wandb_params
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list
from utils import get_train_args, get_gfn_init_state, set_seed, \
    get_exploration_std, random_discretizer, low_discrepancy_discretizer, \
    low_discrepancy_discretizer2, shifted_equidistant, uniform_discretizer, \
    featurize_dataset, embed_dataset, get_conditioning_dim, set_loss_coeffs, anneal_reward, \
    update_ema

#torch.cuda.set_per_process_memory_fraction(0.9, device=0)
torch.cuda.init()  # create context with the cap already in place

args = get_train_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

device = args.device

if args.both_ways and args.bwd:
    args.bwd = False

times = {}


def train_step(energy_function,
               gfn_model,
               optimizers,
               it,
               exploration_std,
               buffer,
               mol_iterator,
               repeats):
    add_to_buffer, do_backward, do_forward, p_forward, report_losses = train_logic(buffer, it)

    discretizer = get_discretizer(args.traj_length_strategy, args.discretizer)

    optimizers['flow'].zero_grad(set_to_none=True)
    if do_forward:
        optimizers['fwd'].zero_grad(set_to_none=True)
        mol_batch = next(mol_iterator)
        loss, crystal_batch, loss_dict = fwd_train_step(
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
        # if add_to_buffer and forward_iter % args.add_to_buffer_each == 0:
        #     # standard to_data_list won't work with our custom batching in the energy function
        #     data_list = manual_batch_to_data_list(crystal_batch.detach().cpu())
        #     buffer.add(data_list)

        del crystal_batch

    elif do_backward:
        optimizers['bwd'].zero_grad(set_to_none=True)
        loss, loss_dict = bwd_train_step(
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
                                   args.gradient_norm_clip)  # gradient clipping
    if do_forward:
        optimizers['fwd'].step()
        optimizers['flow'].step()
    elif do_backward:
        optimizers['bwd'].step()
        optimizers['flow'].step()

    step_type = "Forward" if do_forward else "Backward"

    if report_losses:
        loss_dict_cpu = {step_type + "_loss/" + key: value.cpu().detach().numpy() for key, value in loss_dict.items()}
    else:
        loss_dict_cpu = None

    del loss, loss_dict  # or whatever is large

    return clean_loss, step_type, loss_dict_cpu


def train_logic(buffer, it):
    do_forward = False
    do_backward = False
    add_to_buffer = False
    if args.both_ways:
        p_forward = args.fwd_to_bwd_ratio / (args.fwd_to_bwd_ratio + 1)
        if it == 0:
            do_fwd = True
        elif args.fwd_to_bwd_ratio == 1:
            do_fwd = it % 2 == 0  # always do fwd first
        else:
            do_fwd = np.random.choice([0, 1], 1, p=[1 - p_forward, p_forward])

        if do_fwd:
            if args.sampling == 'buffer':
                add_to_buffer = True
            do_forward = True
        else:
            do_backward = True

    elif args.bwd:  # backward ONLY
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
        args.bwd_loss_coeffs.tb > 0,
        args.bwd_loss_coeffs.vg_lb > 0,
        args.bwd_loss_coeffs.vg_lme > 0,
        args.bwd_loss_coeffs.emp_z > 0,
        args.bwd_loss_coeffs.mle > 0,
    ]):
        do_backward = False
        do_forward = True

    return add_to_buffer, do_backward, do_forward, p_forward, report_losses


def get_discretizer(traj_length_strategy, discretization_type):
    # discretizer = lambda bsz: uniform_discretizer(bsz, args.T)
    # discretizer = lambda bsz: uniform_discretizer(bsz, np.random.randint(10,args.T+1))
    # discretizer = lambda bsz: random_discretizer(bsz, args.T, 10)
    if traj_length_strategy == 'static':
        traj_length = args.T
    elif traj_length_strategy == 'sampled':
        traj_length = np.random.randint(low=args.min_traj_length, high=args.max_traj_length + 1)
    else:
        assert False

    if discretization_type == 'random':
        discretizer = lambda bsz: random_discretizer(bsz, traj_length, max_ratio=args.discretizer_max_ratio)
    elif discretization_type == 'low_discrepancy':
        discretizer = lambda bsz: low_discrepancy_discretizer(bsz, traj_length)
    elif discretization_type == 'low_discrepancy2':
        discretizer = lambda bsz: low_discrepancy_discretizer2(bsz, traj_length)
    elif discretization_type == 'equidistant':
        discretizer = lambda bsz: shifted_equidistant(bsz, traj_length)
    elif discretization_type == 'uniform':
        discretizer = lambda bsz: uniform_discretizer(bsz, traj_length)
    else:
        assert False
    return discretizer


def fwd_train_step(energy_function, gfn_model, discretizer,
                   exploration_std, mol_batch, buffer, return_exp=False,
                   repeats: int = 10,
                   report_losses: bool = False):
    init_state = get_gfn_init_state(args.batch_size, energy_function.data_ndim, device)
    log_T_tensor, sg_inds, condition = energy_function.get_conditioning_tensor(mol_batch)
    mol_batch.sg_ind = sg_inds
    return get_gfn_forward_loss(args.fwd_loss_coeffs,
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


def bwd_train_step(gfn_model, discretizer,
                   buffer, energy_function, repeats: int = 10,
                   report_losses: bool = False):
    if args.sampling == 'buffer':
        samples, rewards, crystal_batch, condition = buffer.sample(
            override_batch=int(args.batch_size * args.bwd_batch_multiplier))
    else:
        assert False, f"sampling method {args.sampling} not implemented"

    condition, rewards, samples = substitute_prior(condition, crystal_batch, energy_function, rewards, samples, buffer)

    return get_gfn_backward_loss(args.bwd_loss_coeffs,
                                 samples.to(device),
                                 gfn_model,
                                 rewards.to(device),
                                 discretizer,
                                 condition=condition.to(device),
                                 repeats=repeats,
                                 report_losses=report_losses)


def substitute_prior(condition, crystal_batch, energy_function, rewards, samples, buffer):
    loss_coeffs = args.bwd_loss_coeffs
    if loss_coeffs.mle_prior_fraction > 0:
        # replace buffer samples with a random prior
        #prior_samples = sample_crystal_prior(crystal_batch, args.bwd_loss_coeffs.pmle_std)
        prior_samples = buffer.sample_mol_unconditional_prior(crystal_batch.sg_ind, loss_coeffs.pmle_std)
        if loss_coeffs.mle_prior_fraction < 1:
            num_to_replace = max(1, int(len(samples) * loss_coeffs.mle_prior_fraction))
            inds_to_replace = np.random.choice(len(samples), num_to_replace, replace=False)
            samples[inds_to_replace] = prior_samples[inds_to_replace]
        else:
            samples = prior_samples

        # have to update the rewards if we are using any loss functions that require them
        # otherwise, if we're not using the reward, just pass the raw sample
        if any([
            loss_coeffs.tb > 0,
            loss_coeffs.vg_lb > 0,
            loss_coeffs.vg_lme > 0,
        ]):
            log_T_tensor, sg_inds, condition = energy_function.get_conditioning_tensor(crystal_batch,
                                                                                       sg_inds=crystal_batch.sg_ind)
            if log_T_tensor is not None:
                log_temperature = log_T_tensor
            else:
                log_temperature = None
            with torch.no_grad():
                rewards = energy_function.log_reward(samples, crystal_batch, log_temperature, False)
    return condition, rewards, samples


def train():
    times['initialization_start'] = time()
    name = args.tag +  '_' + args.run_name
    config = args.__dict__
    config["Experiment"] = "{args.energy}"

    # Reward init
    energy_function = init_energy_function(args)

    # Model Init
    gfn_model, gfn_config = init_gfn_model(args, energy_function)
    ema_model = deepcopy(gfn_model)
    np.save(f'checkpoints/{name}_model_config', gfn_config)  # todo add path to saving directories

    # opt init
    optimizers, schedulers = init_schedulers_optimizers(args, gfn_model)

    # buffer & loaders init
    buffer, train_mol_loader, test_mol_loader, train_iterator, test_iterator = init_buffers_datasets(args, energy_function)

    # initialize some annealing factors
    if args.temperature_conditioning:
        temp_annealing_lambda = get_annealing_factor(args.temperature_scaling_factor, 1,
                                                     args.temp_annealing_max_steps, 10)

    else:
        temp_annealing_lambda = get_annealing_factor(args.energy_max_temperature, args.energy_min_temperature,
                                                     args.temp_annealing_max_steps, 10)
    fwd_loss_dict = None
    bwd_loss_dict = None
    oomed_out = False
    lr_warmup_finished = False
    fwd_loss, bwd_loss = 0, 0

    times['initialization_end'] = time()

    with wandb.init(project="GFN Energy",
               config=flatten_wandb_params(args),
               name=name,
               tags=[args.tag]):

        wandb.watch(gfn_model,
                    log_graph=False,
                    log_freq=1000,
                    log='gradients')  # for gradient logging

        gfn_model.train()

        # torch.autograd.set_detect_anomaly(True)  # for debugging
        #
        # def grad_check_hook(grad, name):
        #     if not torch.isfinite(grad).all():
        #         raise RuntimeError(f"NaN/Inf gradient in {name}")
        #     return grad
        #
        # for name, p in gfn_model.named_parameters():
        #     if p.requires_grad:
        #         p.register_hook(lambda g, n=name: grad_check_hook(g, n))
        #

        for step_ind in trange(args.epochs + 1):
            metrics = dict()
            if step_ind % 10 == 0:
                set_loss_coeffs(step_ind, args)

            exploration_std = get_exploration_std(step_ind,
                                                  args.exploratory,
                                                  args.wd_max_steps,
                                                  args.exploration_factor,
                                                  args.exploration_wd)


            times['train_step_start'] = time()
            try:
                train_loss, step_type, loss_dict = train_step(energy_function,
                                                              gfn_model,
                                                              optimizers,
                                                              step_ind,
                                                              exploration_std,
                                                              buffer,
                                                              train_iterator,
                                                              repeats=args.repeats
                                                              )
                if args.ema_decay is not None:
                    update_ema(gfn_model, ema_model, decay=args.ema_decay)
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

                if not oomed_out and (args.batch_size < args.max_batch_size and args.grow_batch_size):
                    buffer, train_mol_loader, test_mol_loader, train_iterator, test_iterator = increment_batch_size(buffer, train_mol_loader,
                                                                                test_mol_loader,
                                                                                args.batch_growth_increment)

            except (RuntimeError, ValueError) as e:  # if we do hit OOM, slash the batch size
                oomed_out, buffer, train_mol_loader, test_mol_loader, train_iterator, test_iterator = handle_train_epoch_error(e, oomed_out, buffer,
                                                                                                train_mol_loader,
                                                                                                test_mol_loader,
                                                                                                optimizers)
            times['train_step_end'] = time()

            # evaluation work
            if (step_ind % args.eval_period == 0 and step_ind > 0) or step_ind == 50:
                eval_work(args, ema_model, step_ind, name,
                          buffer, train_mol_loader,test_mol_loader,
                          energy_function, metrics)

            # train monitoring
            if step_ind % 10 == 0:
                metrics['train/expl'] = exploration_std(0) if exploration_std is not None else 0
                lr_warmup_finished, lr = step_lr_schedule(schedulers, optimizers, lr_warmup_finished)
                anneal_reward(step_ind, temp_annealing_lambda, energy_function, args)
                ten_step_reporting(bwd_loss, bwd_loss_dict, fwd_loss, fwd_loss_dict, metrics, optimizers)
                wandb.log(metrics, step=step_ind)


        torch.save(ema_model, f'checkpoints/{name}_model_final.pt')




def ten_step_reporting(bwd_loss, bwd_loss_dict, fwd_loss, fwd_loss_dict, metrics, optimizers):
    metrics.update({'lr_fwd': optimizers['fwd'].param_groups[0]['lr']})
    metrics.update({'lr_bwd': optimizers['bwd'].param_groups[0]['lr']})
    metrics.update({'lr_flow': optimizers['flow'].param_groups[0]['lr']})
    metrics.update(log_elapsed_times())
    metrics['Forward Loss'] = fwd_loss
    metrics['Backward Loss'] = bwd_loss
    if fwd_loss_dict is not None:
        metrics.update(fwd_loss_dict)
        fwd_loss_dict = None
    if bwd_loss_dict is not None:
        metrics.update(bwd_loss_dict)
        bwd_loss_dict = None


def step_lr_schedule(schedulers, optimizers,
                     lr_warmup_finished):
    if args.scheduler:
        lr = optimizers['fwd'].param_groups[0]['lr']
        if not lr_warmup_finished:
            schedulers['policy_1'].step()
            schedulers['flow'].step()

            if lr >= args.lr_policy:
                lr_warmup_finished = True

        elif lr > args.min_lr:
            schedulers['policy_2'].step()
        return lr_warmup_finished, lr
    else:
        return False, None


def init_schedulers_optimizers(args, gfn_model):
    if args.scheduler:
        init_fwd_lr = args.lr_policy / args.lr_warmup_ratio
        init_flow_lr = args.lr_flow / args.lr_warmup_ratio
        init_bwd_lr = args.lr_back / args.lr_warmup_ratio
    else:
        init_fwd_lr = args.lr_policy
        init_bwd_lr = args.lr_back
        init_flow_lr = args.lr_flow

    """
    Initialize Optimizers
    """
    policy_params = [{'params': gfn_model.t_model.parameters()},
                     {'params': gfn_model.s_model.parameters()},
                     {'params': gfn_model.forward_policy.parameters()},
                     {'params': gfn_model.backward_policy.parameters()},
                     ]
    if args.temperature_conditioning:
        policy_params += [{'params': gfn_model.conditions_embedding_model.parameters()}]

    flow_params = gfn_model.flow_model.parameters()

    optimizers = {}
    if args.use_weight_decay:
        optimizers['fwd'] = torch.optim.Adam(policy_params, init_fwd_lr, weight_decay=args.weight_decay)
        optimizers['bwd'] = torch.optim.Adam(policy_params, init_bwd_lr, weight_decay=args.weight_decay)
        optimizers['flow'] = torch.optim.Adam(flow_params, init_flow_lr, weight_decay=args.weight_decay)
    else:
        optimizers['fwd'] = torch.optim.Adam(policy_params, init_fwd_lr)
        optimizers['bwd'] = torch.optim.Adam(policy_params, init_bwd_lr)
        optimizers['flow'] = torch.optim.Adam(flow_params, init_flow_lr)

    schedulers = {}
    if args.scheduler:
        lr_warmup_lambda = get_annealing_factor(1, args.lr_warmup_ratio, args.lr_warmup_time, 10)

        lr_annealing_lambda = get_annealing_factor(args.lr_policy, args.min_lr, args.lr_anneal_time, 10)
        schedulers['policy_1'] = lr_scheduler.MultiplicativeLR(
            optimizers['fwd'], lr_lambda=lambda epoch: lr_warmup_lambda)
        schedulers['policy_2'] = lr_scheduler.MultiplicativeLR(
            optimizers['fwd'], lr_lambda=lambda epoch: lr_annealing_lambda)

        flow_annealing_lambda = get_annealing_factor(1, args.lr_warmup_ratio, args.lr_anneal_time, 10)
        schedulers['flow'] = lr_scheduler.MultiplicativeLR(
            optimizers['flow'], lr_lambda=lambda epoch: flow_annealing_lambda)

    return optimizers, schedulers


def init_buffers_datasets(args, energy_function):
    # load dataset of prebuilt and scored molecular crystals into the buffer
    buffer = CrystalReplayBuffer(
        args.buffer_size,
        'cpu',
        energy_function,
        args.batch_size,
        beta=args.beta,
        rank_weight=args.rank_weight,
        prioritized=args.prioritized,
        keep_initial_samples=False,  # args.buffer_path is not None,
        diversity_coeff=args.buffer_diversity_coeff,
    )
    if (args.both_ways or args.bwd) and args.buffer_path is not None:  # preload samples into the buffer
        buffer = add_dataset_to_buffer(args.buffer_path, buffer)
    # load dataset of just molecules
    # mols_list = torch.load(args.molecules_path)
    # good_mol = mols_list[17]  # a nice molecule
    if args.molecule == 'urea':
        atom_coords = torch.tensor([  # stick with urea for just now
            [-1.3042, - 0.0008, 0.0001],
            [0.6903, - 1.1479, 0.0001],
            [0.6888, 1.1489, 0.0001],
            [- 0.0749, - 0.0001, - 0.0003],
        ], dtype=torch.float32, device='cpu')
        atom_coords -= atom_coords.mean(0)
        atom_types = torch.tensor([8, 7, 7, 6], dtype=torch.long, device='cpu')
        good_mol = MolData(
            z=atom_types,
            pos=atom_coords,
            x=atom_types,
            skip_mol_analysis=False,
        )
        train_mols_list = [good_mol for _ in range(int(args.max_batch_size * 1.5))]
        test_mols_list = [good_mol for _ in range(int(args.max_batch_size * 1.5))]

    elif args.molecule == 'nicotinamide':
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
        atom_coords -= atom_coords.mean(dim=0)
        atom_types = torch.tensor([8, 7, 7, 6, 6, 6, 6, 6, 6], dtype=torch.long, device='cpu')
        good_mol = MolData(
            z=atom_types,
            pos=atom_coords,
            x=atom_types,
            skip_mol_analysis=False,
        )
        train_mols_list = [good_mol for _ in range(int(args.max_batch_size * 1.5))]
        test_mols_list = [good_mol for _ in range(int(args.max_batch_size * 1.5))]

    elif args.molecule == 'qm9':
        qm9_mols = torch.load(args.molecules_path, weights_only=False)
        rng = np.random.RandomState(0)
        rands = rng.choice(len(qm9_mols), len(qm9_mols), replace=False)
        bp = int(len(rands) * 0.8)
        # for mol in qm9_mols:
        #     mol.deprotonate()  # since we'll be comparing against CSD later, deprotonate here
        train_mols_list = [qm9_mols[ind] for ind in rands[:bp]]
        test_mols_list = [qm9_mols[ind] for ind in rands[bp:]]

    else:
        assert False

    if args.molecule_conditioning:
        train_mols_list = embed_dataset(train_mols_list, args.autoencoder_path, args.device, encoder=None,
                                        embedding_type=args.mol_embedding_type)
        test_mols_list = embed_dataset(test_mols_list, args.autoencoder_path, args.device, encoder=None,
                                       embedding_type=args.mol_embedding_type)

    train_mol_loader = DataLoader(
        train_mols_list,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    test_mol_loader = DataLoader(
        test_mols_list,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    train_iterator = iter_forever(train_mol_loader)
    test_iterator = iter_forever(test_mol_loader)

    return buffer, train_mol_loader, test_mol_loader, train_iterator, test_iterator


def increment_batch_size(buffer, train_mol_loader, test_mol_loader, batch_growth_increment):

    new_batch_size = max(args.batch_size + 1,
                         int(args.batch_size * batch_growth_increment))
    args.batch_size = new_batch_size  # gradually increment batch size

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


def handle_train_epoch_error(e, oomed_out, buffer, train_mol_loader, test_mol_loader, optimizers):
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

        args.batch_size = max(1, int(args.batch_size * 0.95))
        if args.batch_size <= 1:
            raise RuntimeError("Cascading OOM Failure")

        gc.collect()
        torch.cuda.empty_cache()

        train_mol_loader = DataLoader(
            train_mol_loader.dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        test_mol_loader = DataLoader(
            test_mol_loader.dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        train_iterator = iter_forever(train_mol_loader)
        test_iterator = iter_forever(test_mol_loader)

        oomed_out = True
        print(f"Reducing batch size to {args.batch_size}")
    else:
        raise e  # will simply raise error if other or if training on CPU
    return oomed_out, buffer, train_mol_loader, test_mol_loader, train_iterator, test_iterator



def is_cuda_oom(e: Exception) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    s = str(e).lower()
    return (
        ("cuda" in s and "memory" in s)
        or ("cublas" in s and "alloc" in s)
        or ("cusolver" in s and "alloc" in s)
        or ("out of memory" in s)
        or ("nonzero is not supported for tensors with more than int_max elements" in s)
    )


def do_evaluation(energy_function, buffer, gfn_model, i, mol_loader,
                  override_do_figures: Optional[bool] = None):
    times['eval_step_start'] = time()

    eval_discretizer = lambda bsz: uniform_discretizer(bsz, args.eval_T)

    if override_do_figures is not None:
        do_figures = override_do_figures
    else:
        do_figures = i % args.figs_period == 0
    eval_batch_size = max(args.batch_size, args.eval_batch_size)

    eval_rands = np.random.randint(len(mol_loader.dataset), size=eval_batch_size)
    mol_batch = collate_data_list([mol_loader.dataset[ind] for ind in eval_rands]).to(device)

    init_state = get_gfn_init_state(eval_batch_size, energy_function.data_ndim, args.device)

    eval_metrics = {}
    eval_metrics.update(
        eval_step(energy_function,
                  gfn_model,
                  eval_discretizer,
                  init_state,
                  buffer,
                  args,
                  do_figures,
                  mol_batch,
                  bwd_training=len(buffer) > 0,
                  add_to_buffer=args.both_ways))

    eval_metrics.update({'Batch Size': args.batch_size})
    eval_metrics.update(log_elapsed_times())

    times['eval_step_end'] = time()

    return eval_metrics


def do_conditional_evaluation(energy_function, gfn_model, mol_loader,
                              ):  # todo these functions could be cleaned up / consolidated
    times['eval_step_start'] = time()
    eval_discretizer = lambda bsz: uniform_discretizer(bsz, args.eval_T)

    eval_batch_size = args.eval_batch_size

    eval_rands = np.random.randint(len(mol_loader.dataset), size=eval_batch_size)
    mol_batch = collate_data_list([mol_loader.dataset[ind] for ind in eval_rands]).to(device)

    init_state = get_gfn_init_state(eval_batch_size, energy_function.data_ndim, args.device)

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


def log_elapsed_times():
    elapsed_times = {}
    for key in times.keys():
        if 'start' in key:
            start_key = key
            end_key = start_key.split('_start')[0] + '_end'
            if end_key in times.keys():
                elapsed_times[start_key.split('_start')[0] + '_time'] = times[end_key] - times[start_key]

    return elapsed_times


# def handle_oom(batch_size, e, optimizers_dict):
#     traceback.print_exc()
#     sys.exc_info()  # Break circular references from traceback
#     for opt in optimizers_dict.values():  # if you track them, else do per-optimizer
#         for group in opt.param_groups:
#             for p in group['params']:
#                 if p.grad is not None:
#                     p.grad = None
#     del e
#
#     torch.cuda.synchronize()
#
#     # Garbage collection
#     gc.collect()
#     torch.cuda.empty_cache()
#     batch_size = int(batch_size * 0.5)
#     if batch_size < 1:
#         assert False, "Cascading OOM Failure"
#
#     return batch_size



class SkipBatch(RuntimeError):
    def __init__(self, msg, new_bs):
        super().__init__(msg)
        self.new_bs = new_bs

def robust_zero_grad(optimizers_dict):
    # Use set_to_none=True for instant free
    for opt in optimizers_dict.values():
        for group in opt.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = None


def add_dataset_to_buffer(dataset_path, buffer):
    print("Loading prebuilt buffer")
    dataset = torch.load(dataset_path, weights_only=False)

    if args.energy_function in ['ellipsoid_overlap',
                                'silu_energy',
                                'combo']:  # reparameterize incoming samples
        print("Re-featurizing preloaded buffer samples")
        dataset = featurize_dataset(dataset, args.device,
                                    args.ellipsoid_scale, args.lj_repulsion)

    if args.molecule_conditioning:  # embed dataset
        print("Getting preloaded dataset molecule embeddings")
        dataset = embed_dataset(dataset, args.autoencoder_path, args.device, encoder=None)

    buffer.add(dataset)
    print(f"Buffer loaded with {len(dataset)} samples")

    return buffer


def get_annealing_factor(start_value, stop_value, total_time, step_iters):
    assert stop_value > 0, "Setting final value as zero breaks this module"
    return (stop_value / start_value) ** (1 / (total_time / step_iters))


def init_gfn_model(args, energy_function):
    gfn_config = dict(
        dim=energy_function.data_ndim,
        s_emb_dim=args.s_emb_dim,
        hidden_dim=args.hidden_dim,
        conditions_dim=get_conditioning_dim(args),
        harmonics_dim=args.harmonics_dim,
        t_dim=args.t_emb_dim,
        condition_embedding_dim=args.condition_emb_dim,
        trajectory_length=args.T,
        clipping=args.clipping,
        gfn_clip=args.gfn_clip,
        learned_variance=args.learned_variance,
        log_var_range=args.log_var_range,
        pb_drift_range=args.pb_drift_range,
        pb_var_range=args.pb_var_range,
        t_scale=args.t_scale,
        conditional_flow_model=any([
            args.temperature_conditioning,
            args.molecule_conditioning,
            args.sg_conditioning]
        ),
        learn_pb=args.learn_pb,
        joint_layers=args.joint_layers,
        dropout=args.dropout,
        norm=args.norm,
        zero_init=args.zero_init,
        device=device
    )
    gfn_model = GFN(**gfn_config).to(device)

    return gfn_model, gfn_config

def init_energy_function(args):
    energy_config = {
        'device': device,
        'energy_function': args.energy_function,
        'min_temperature': args.energy_min_temperature,
        'max_temperature': args.energy_max_temperature,
        'temperature_scaling_factor': args.temperature_scaling_factor,
        'temperature_conditioning': args.temperature_conditioning,
        'temperature': args.energy_static_temperature,
        'density_coeff': args.energy_density_coeff,
        'energy_clip': args.energy_clip,
        'ellipsoid_scale': args.ellipsoid_scale,
        'core_coeff': args.energy_core_coeff,
        'lj_coeff': args.energy_lj_coeff,
        'lj_turnover_pot': args.lj_turnover_pot,
        'lj_repulsion': args.lj_repulsion,
        'molecule_conditioning': args.molecule_conditioning,
        'sg_conditioning': args.sg_conditioning,
        'space_groups': args.space_groups,
        'bounding_coeff': args.bounding_coeff,
        'niggli_coeff': args.niggli_coeff,
    }
    energy_function = MolecularCrystal(**energy_config
                                       )
    return energy_function

def eval_work(args,
              gfn_model,
              step_ind,
              name,
              buffer,
              train_mol_loader,
              test_mol_loader,
              energy_function,
              metrics):
    torch.save(gfn_model.state_dict(), f'checkpoints/{name}_model.pt')
    if args.molecule_conditioning:

        if step_ind % args.conditional_eval_period == 0:  # make conditional sampling figures
            # # so far not useful
            train_metrics = do_evaluation(energy_function, buffer, gfn_model,
                                          step_ind, train_mol_loader,
                                          override_do_figures=False)
            kk = list(train_metrics.keys())
            for key in kk:
                metrics['train_eval/' + key] = train_metrics[key]
            conditional_metrics = do_conditional_evaluation(energy_function, gfn_model,
                                                            test_mol_loader,
                                                            )
            metrics.update(conditional_metrics)

    metrics.update(do_evaluation(energy_function, buffer,
                                 gfn_model,
                                 step_ind, test_mol_loader))

    dynamic_quality_management(metrics, buffer, energy_function)

    wandb.log(metrics, step=step_ind)


def dynamic_quality_management(metrics, buffer, energy_function):
    minimum_1d_coverage = metrics['Minimum 1d coverage']
    # adjust by a factor of 'multiple' for each 'delta_factor' of miss
    multiple = 2
    delta_factor = 0.05
    diversity_check = True
    if args.prior_coverage_cutoff is not None and args.both_ways:
        if args.fwd_to_bwd_ratio < 1000 and args.fwd_to_bwd_ratio > 0.001: # don't let it get too crazy
            miss = minimum_1d_coverage - args.prior_coverage_cutoff
            adjustment_factor = multiple ** (miss / delta_factor)
            args.fwd_to_bwd_ratio *= adjustment_factor
            if minimum_1d_coverage < args.prior_coverage_cutoff:
                diversity_check = False

    reasonable_frac = metrics['Reasonable Sample Fraction']

    if args.anneal_repulsion:
        if (reasonable_frac >= args.anneal_repulsion_cutoff) and diversity_check:
            if args.lj_repulsion < 1:
                args.lj_repulsion = min(1, args.lj_repulsion * 1.05)
                energy_function.lj_repulsion = args.lj_repulsion
                if buffer is not None:
                    if len(buffer) > 0:
                        buffer.recompute_silu_pot(
                            batch_size=min(500, args.batch_size),
                            lj_repulsion=args.lj_repulsion,
                            device=args.device
                        )
    metrics['LJ Repulsion'] = args.lj_repulsion
    metrics['Fwd to Bwd Ratio'] = args.fwd_to_bwd_ratio

if __name__ == '__main__':
    train()
