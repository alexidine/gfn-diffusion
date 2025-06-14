import gc
import os
from time import time

import plotly.graph_objects as go
import numpy as np
import torch
import wandb
from mxtaltools.reporting.online import simple_cell_hist, simple_cell_scatter_fig, log_crystal_samples, \
    simple_embedding_fig
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from tqdm import trange

from buffer import CrystalReplayBuffer
from energies.molecular_crystal import MolecularCrystal
from evaluations import log_partition_function
from models import GFN
from plot_utils import get_plotly_fig_size_mb
from utils import get_train_args, get_gfn_init_state
from utils import set_seed, cal_subtb_coef_matrix, get_gfn_optimizer, get_gfn_forward_loss, \
    get_gfn_backward_loss, get_exploration_std

args = get_train_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

device = args.device
coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)

if args.both_ways and args.bwd:
    args.bwd = False

if args.local_search:
    args.both_ways = True

times = {}


def eval_step(energy, gfn_model, batch_size, do_figures: bool = True, mol_batch=None):
    gfn_model.eval()

    metrics = {}
    fig_dict = {}
    init_state = get_gfn_init_state(batch_size, energy.data_ndim, device)
    samples, log_Z, log_Z_lb, log_Z_learned, sample_batch, condition = log_partition_function(
        init_state, gfn_model, energy, mol_batch)

    "Scalar metrics"
    metrics['eval/log_Z'] = log_Z.cpu().detach().numpy()
    metrics['eval/log_Z_lb'] = log_Z_lb.cpu().detach().numpy()
    metrics['eval/log_Z_learned'] = log_Z_learned.cpu().detach().numpy()
    metrics['eval/packing_coeff'] = sample_batch.packing_coeff.mean().cpu().detach().numpy()
    metrics['eval/silu_potential'] = sample_batch.silu_pot.mean().cpu().detach().numpy()
    metrics['eval/energy'] = sample_batch.gfn_energy.mean().cpu().detach().numpy()
    metrics['Crystal Log Temperature'] = condition[:, 0]
    metrics['Crystal Mean Log Temperature'] = condition[:, 0].mean()
    metrics['Crystal Min Temperature'] = energy.min_temperature
    metrics['Crystal Max Temperature'] = energy.max_temperature
    metrics['Ellipsoid Scale'] = energy.ellipsoid_scale
    metrics['Temperature Scaling Factor'] = energy.temperature_scaling_factor
    metrics['Density Loss Coefficient'] = energy.density_coeff
    #metrics['eval/ellipsoid_overlap'] = sample_batch.ellipsoid_overlap.mean().cpu().detach().numpy()

    "Custom Figures"
    if do_figures:
        # todo figs to add
        # pairwise dists to eval sample and dataset
        # RDF dists of same
        # clustering / mode counting / basin counting/mapping
        # known mode coverage
        # diversity vs T / E
        if args.conditional_flow_model:  # todo update this with molecule conditioning when the time comes
            log_temps = torch.linspace(-2, 2, 100).to(args.device)[:, None].flatten()
            Z_at_T = gfn_model.flow_model(
                gfn_model.conditions_embedding_model(log_temps[:, None])).cpu().detach().flatten()
            fig = go.Figure(go.Scatter(x=log_temps.cpu().detach(), y=Z_at_T.cpu().detach(), mode='lines+markers'))
            fig.update_layout(xaxis_title='Log Temperature', yaxis_title='Log Partition Function')
            fig_dict['Learned Z vs T'] = fig

            fig = go.Figure()
            fig.add_histogram2d(x=condition[:, 0].cpu().detach().numpy(),
                                y=sample_batch.gfn_energy.cpu().detach().numpy(),
                                showscale=False,
                                nbinsx=25, nbinsy=50)
            fig.update_layout(xaxis_title='Log Temperature', yaxis_title='Sample Energy')
            fig_dict['T vs Energy'] = fig

        fig_dict['Lattice Features Distribution'] = simple_cell_hist(sample_batch)
        fig_dict['Sample Scatter'] = simple_cell_scatter_fig(sample_batch,
                                                             (condition[:,
                                                              0].cpu().detach().numpy()) if condition is not None else None,
                                                             aux_scalar_name='log_temperature' if condition is not None else None)
        fig_dict['Sample Embedding'] = simple_embedding_fig(sample_batch,
                                                            sample_batch.silu_pot.cpu().detach().numpy())  #condition[:, 0].cpu().detach().numpy() if condition is not None else None)
        for key in fig_dict.keys():
            fig = fig_dict[key]
            if get_plotly_fig_size_mb(fig) > 0.1:  # bigger than .1 MB
                fig.write_image(key + 'fig.png', width=1024,
                                height=512)  # save the image rather than the fig, for size reasons
                fig_dict[key] = wandb.Image(key + 'fig.png')
        metrics.update(fig_dict)

    "Crystal samples"
    samples_to_log = log_crystal_samples(sample_batch=sample_batch)
    [wandb.log({f'crystal_sample_{ind}': samples_to_log[ind]}, commit=False) for ind in range(len(samples_to_log))]

    gfn_model.train()
    return metrics


def train_step(energy_function, gfn_model, gfn_optimizer, it, exploratory, buffer, mol_loader, exploration_factor,
               exploration_wd, repeats: int = 10):
    gfn_model.zero_grad()
    wd_max_steps = 20000
    exploration_std = get_exploration_std(it, exploratory, wd_max_steps, exploration_factor, exploration_wd)

    do_forward = False
    do_backward = False
    add_to_buffer = False
    if args.both_ways:
        if it % args.fwd_train_each == 0:
            if args.sampling == 'buffer':
                add_to_buffer = True
            do_forward = True
        else:
            do_backward = True
    elif args.bwd:  # backward ONLY
        do_backward = True
    else:  # forward ONLY
        do_forward = True

    if do_forward:
        mol_batch = next(iter(mol_loader)).to(device)  # todo add here optional mol adjustments
        loss, states, log_pfs, log_pbs, log_r, log_fs, crystal_batch = fwd_train_step(energy_function,
                                                                                      gfn_model,
                                                                                      exploration_std,
                                                                                      mol_batch,
                                                                                      return_exp=True,
                                                                                      repeats=repeats,
                                                                                      )
        if add_to_buffer:
            buffer.add(crystal_batch.cpu().detach().to_data_list())
    elif do_backward:
        loss, states, log_pfs, log_pbs, log_r, log_fs = bwd_train_step(gfn_model,
                                                                       buffer,
                                                                       exploration_std,
                                                                       repeats=repeats,
                                                                       return_exp=True)
    else:
        assert False

    loss.backward()
    torch.nn.utils.clip_grad_norm_(gfn_model.parameters(),
                                   args.gradient_norm_clip)  # gradient clipping
    gfn_optimizer.step()
    return loss.item(), exploration_std(0)


def fwd_train_step(energy_function, gfn_model, exploration_std, mol_batch, return_exp=False, repeats: int = 10):
    init_state = get_gfn_init_state(args.batch_size, energy_function.data_ndim, device)
    condition = energy_function.get_conditioning_tensor(mol_batch)
    return get_gfn_forward_loss(args.mode_fwd,
                                init_state,
                                gfn_model,
                                energy_function.log_reward,
                                coeff_matrix,
                                mol_batch,
                                exploration_std=exploration_std,
                                return_exp=return_exp,
                                condition=condition,
                                repeats=repeats)


def bwd_train_step(gfn_model, buffer, exploration_std=None, repeats: int = 10, return_exp=False):
    if args.sampling == 'buffer':
        samples, rewards, crystal_batch, condition = buffer.sample(
            return_conditioning=True,
            override_batch=int(buffer.batch_size * args.bwd_batch_multiplier))
    else:
        assert False, f"sampling method {args.sampling} not implemented"

    return get_gfn_backward_loss(args.mode_bwd,
                                 samples.to(device),
                                 gfn_model,
                                 rewards.to(device),
                                 exploration_std=exploration_std,
                                 condition=condition.to(device),
                                 repeats=repeats,
                                 return_exp=return_exp)


def train():
    times['initialization_start'] = time()
    name = args.run_name
    if not os.path.exists(name):
        os.makedirs(name)

    energy_function = MolecularCrystal(device=device,
                                       min_temperature=args.energy_min_temperature,
                                       max_temperature=args.energy_max_temperature,
                                       temperature_scaling_factor=args.temperature_scaling_factor,
                                       temperature_conditioning=args.temperature_conditioning,
                                       density_coeff=args.energy_density_coeff)

    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    wandb.init(project="GFN Energy", config=config, name=name)
    conditioning_dim = 1 if args.temperature_conditioning else 0
    gfn_model = GFN(energy_function.data_ndim, args.s_emb_dim, args.hidden_dim,
                    conditioning_dim, args.harmonics_dim,
                    args.t_emb_dim, condition_embedding_dim=args.condition_emb_dim,
                    trajectory_length=args.T, clipping=args.clipping,
                    gfn_clip=args.gfn_clip,
                    learned_variance=args.learned_variance,
                    partial_energy=args.partial_energy, log_var_range=args.log_var_range,
                    pb_scale_range=args.pb_scale_range,
                    t_scale=args.t_scale,
                    conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                    lgv_layers=args.lgv_layers,
                    joint_layers=args.joint_layers, dropout=args.dropout, norm=args.norm,
                    zero_init=args.zero_init, device=device).to(device)

    wandb.watch(gfn_model, log_graph=True, log_freq=500)  # for gradient logging

    gfn_optimizer = get_gfn_optimizer(gfn_model, args.lr_policy, args.lr_flow, args.lr_back, args.learn_pb,
                                      args.conditional_flow_model, args.use_weight_decay, args.weight_decay)

    if args.scheduler:
        scheduler = lr_scheduler.MultiplicativeLR(gfn_optimizer, lr_lambda=lambda epoch: args.lr_shrink_lambda)

    buffer, mol_loader = init_buffers_datasets(energy_function)

    gfn_model.train()

    times['initialization_end'] = time()
    loss_record, energy_record, learned_Z_record = [], [], []
    oomed_out = False
    old_params = [p.clone().detach() for p in gfn_model.parameters()]
    for i in trange(args.epochs + 1):
        metrics = dict()
        times['train_step_start'] = time()
        try:
            metrics['train/loss'], metrics['train/expl'] = train_step(energy_function,
                                                                      gfn_model,
                                                                      gfn_optimizer,
                                                                      i,
                                                                      args.exploratory,
                                                                      buffer,
                                                                      mol_loader,
                                                                      args.exploration_factor,
                                                                      args.exploration_wd,
                                                                      )

            if args.scheduler:
                scheduler.step()

            loss_record.append(metrics['train/loss'])
            if not oomed_out:
                buffer, mol_loader = grow_batch_size(buffer, mol_loader)

        except (RuntimeError, ValueError) as e:  # if we do hit OOM, slash the batch size
            oomed_out, buffer, mol_loader = handle_train_epoch_error(e, oomed_out, buffer, mol_loader)
        times['train_step_end'] = time()

        if (i % args.eval_period == 0 and i > 0) or i == 50:
            metrics = do_evaluation(energy_function, energy_record, gfn_model, i, learned_Z_record,
                                    metrics, mol_loader)
            wandb.log(metrics, step=i)

        if i % 100 == 0 and i > 0:
            torch.save(gfn_model.state_dict(), f'{name}model.pt')
            metrics.update({'lr': gfn_optimizer.param_groups[0]['lr']})
            if args.energy == 'molecular_crystal' and args.anneal_energy:
                old_params = check_energy_annealing(energy_function, energy_record, gfn_model, learned_Z_record,
                                                    loss_record,
                                                    metrics, old_params)

        elif i % 10 == 0:
            metrics.update(log_elapsed_times())
            metrics['train/loss'] = np.mean(loss_record[-10:])
            wandb.log(metrics, step=i)

    torch.save(gfn_model.state_dict(), f'{name}_model_final.pt')


def init_buffers_datasets(energy_function):
    # load dataset of prebuilt and scored molecular crystals into the buffer
    buffer = CrystalReplayBuffer(
        args.buffer_size,
        device,
        energy_function,
        args.batch_size,
        beta=args.beta,
        rank_weight=args.rank_weight,
        prioritized=args.prioritized)
    if args.learn_pb and args.buffer_path is not None:  # preload samples into the buffer
        buffer = add_dataset_to_buffer(args.buffer_path, buffer)
    # load dataset of just molecules
    # mols_list = torch.load(args.molecules_path)
    # good_mol = mols_list[17]  # a nice molecule
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
    mols_list = [good_mol for _ in range(int(args.max_batch_size * 1.5))]
    mol_loader = DataLoader(
        mols_list,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    return buffer, mol_loader


def grow_batch_size(buffer, mol_loader):
    if args.batch_size < args.max_batch_size and args.grow_batch_size:
        new_batch_size = max(args.batch_size + 1,
                             int(args.batch_size * 1.01))
        args.batch_size = new_batch_size  # gradually increment batch size

        if len(buffer) > 0:
            buffer.loader = DataLoader(
                buffer.dataset,
                batch_size=new_batch_size,
                sampler=buffer.sampler,
                num_workers=0,
                pin_memory=True,
                drop_last=True)
            buffer.batch_size = new_batch_size

        mol_loader = DataLoader(
            mol_loader.dataset,
            batch_size=new_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    return buffer, mol_loader


def handle_train_epoch_error(e, oomed_out, buffer, mol_loader):
    if "CUDA out of memory" in str(
            e) or "nonzero is not supported for tensors with more than INT_MAX elements" in str(e):
        args.batch_size = handle_oom(args.batch_size)

        if len(buffer) > 0:
            buffer.loader = DataLoader(
                buffer.dataset,
                batch_size=args.batch_size,
                sampler=buffer.sampler,
                num_workers=0,
                pin_memory=True,
                drop_last=True)

        mol_loader = DataLoader(
            mol_loader.dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        oomed_out = True
        print(f"Reducing batch size to {args.batch_size}")
    else:
        raise e  # will simply raise error if other or if training on CPU
    return oomed_out, buffer, mol_loader


def do_evaluation(energy_function, energy_record, gfn_model, i, learned_Z_record, metrics, mol_loader):
    times['eval_step_start'] = time()

    do_figures = i % args.figs_period == 0
    eval_batch_size = args.eval_batch_size

    eval_rands = np.random.randint(len(mol_loader.dataset), size=eval_batch_size)
    mol_batch = collate_data_list([mol_loader.dataset[ind] for ind in eval_rands]).to(device)

    metrics.update(eval_step(energy_function, gfn_model, eval_batch_size, do_figures, mol_batch))

    energy_record.append(metrics['eval/energy'])
    learned_Z_record.append(metrics['eval/log_Z_learned'])

    metrics.update({'Batch Size': args.batch_size})
    metrics.update(log_elapsed_times())

    times['eval_step_end'] = time()

    return metrics


def check_energy_annealing(energy_function, energy_record, gfn_model, learned_Z_record, loss_record, metrics,
                           old_params):
    with torch.no_grad():
        total_change = 0.0
        total_norm = 0.0
        for p, old_p in zip(gfn_model.parameters(), old_params):
            delta = (p - old_p).norm()
            total_change += delta.item()
            total_norm += p.norm().item()
        relative_change = total_change / (total_norm + 1e-8)
        metrics['relative_gradient_change'] = relative_change
        old_params = [p.clone().detach() for p in gfn_model.parameters()]
    convergence_history = args.convergence_history
    loss_array = np.array(loss_record)[-convergence_history:]
    energy_array = np.array(energy_record)[-min(10, int(convergence_history / args.eval_period)):]
    log_Z_array = np.array(learned_Z_record)[-min(10, int(convergence_history / args.eval_period)):]
    loss_slope = relative_slope(loss_array)
    energy_slope = relative_slope(energy_array)
    log_Z_slope = relative_slope(log_Z_array)
    annealing_trigger = (np.abs(loss_slope) <= args.energy_annealing_threshold) and \
                        (np.abs(energy_slope) <= args.energy_annealing_threshold) and \
                        (np.abs(log_Z_slope) <= args.energy_annealing_threshold)
    anneal_energy(energy_function, annealing_trigger)
    return old_params


def relative_slope(y):
    if len(y) < 2 or np.ptp(y) == 0:
        return 0.0  # avoid division by zero or meaningless fit
    slope, _ = np.polyfit(np.arange(len(y)), y, 1)
    return abs(slope) / np.std(y)


def log_elapsed_times():
    elapsed_times = {}
    for key in times.keys():
        if 'start' in key:
            start_key = key
            end_key = start_key.split('_start')[0] + '_end'
            if end_key in times.keys():
                elapsed_times[start_key.split('_start')[0] + '_time'] = times[end_key] - times[start_key]

    return elapsed_times


def anneal_energy(energy_function, trigger: bool, min_temperature: float = 0.01):
    # values = np.array(values)[-100:]
    # # Avoid log(0) or log of very small numbers
    # values = np.clip(values, 1e-12, None)
    # log_values = np.log(values)
    #
    # slope, _ = np.polyfit(np.arange(len(log_values)), log_values, 1)
    # curvature, _, _ = np.polyfit(np.arange(len(log_values)), values, 2)

    if trigger:  #slope > -1e-3 and curvature > -1e-1:  # if the loss isn't decaying fast enough / the loss is saturated
        if energy_function.temperature_scaling_factor < 2:
            energy_function.temperature_scaling_factor *= 1.05
            print("Annealing energy function")


def handle_oom(batch_size):
    gc.collect()
    torch.cuda.empty_cache()
    batch_size = int(batch_size * 0.9)
    return batch_size


def add_dataset_to_buffer(dataset_path, buffer):
    print("Loading prebuilt buffer")
    dataset = torch.load(dataset_path)
    # if True:  # add ellipsoid overlaps to each sample here, as they weren't in the original optimization
    #     from tqdm import tqdm
    #     batch_size = 500
    #     loader = DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         drop_last=False
    #     )
    #     overlaps = []
    #
    #     for crystal_batch in tqdm(loader):
    #         crystal_batch = crystal_batch.to('cuda')
    #         crystal_batch.box_analysis()
    #         cluster_batch = crystal_batch.mol2cluster(cutoff=6,
    #                                                   supercell_size=10,
    #                                                   align_to_standardized_orientation=True)
    #
    #         cluster_batch.construct_radial_graph(cutoff=6)
    #         # simplified ellipsoid energy testing
    #         _, _, _, _, _, _, normed_ellipsoid_overlap \
    #             = cluster_batch.compute_ellipsoidal_overlap(
    #             semi_axis_scale=1,
    #             return_details=True)
    #
    #         overlaps.extend(normed_ellipsoid_overlap.cpu().detach().numpy())
    #
    #     overlaps = torch.tensor(overlaps)
    #     for ind, elem in enumerate(dataset):
    #         elem.ellipsoid_overlap = torch.ones(1) * overlaps[ind]

    buffer.add(dataset)
    print(f"Buffer loaded with {len(dataset)} samples")

    return buffer


if __name__ == '__main__':
    train()
