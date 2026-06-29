from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Deque, List, Optional

import numpy as np
import plotly
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from tqdm import tqdm

from energy_sampling.utils import uniform_discretizer, get_gfn_init_state, embed_dataset
from mxtaltools.dataset_utils.utils import collate_data_list


def sample_crystals(
        generator: str,  # 'generator' or 'random'
        gfn_model,
        batch_size,
        mol_list,
        space_group,
        n_steps,
        samples_per_mol,
        device,
        energy_function,
        encoder=None,
        optim_kwargs=None,
        do_opt: bool = False
):
    """
    :param gfn_model:
    :param batch_size:
    :param mol_list:
    :param space_group:
    :param n_steps:
    :param samples_per_mol:
    :param device:
    :param energy_function:
    :return:
    """

    """
    initialize useful things
    """
    if optim_kwargs is None:
        optim_kwargs = dict(
            optim_target='silu',
            show_tqdm=True,
            lr=1e-4,
            convergence_eps=1e-3,
            compression_factor=0.1,
            max_num_steps=300,
            do_box_restriction=True,
            enforce_niggli=True,
            cutoff=6,
            optimizer_func=torch.optim.Rprop,
        )

    with torch.no_grad():
        discretizer = lambda bsz: uniform_discretizer(bsz, n_steps)

        num_batches = len(mol_list) // batch_size
        if len(mol_list) % batch_size != 0:
            num_batches += 1

        energy_function.space_groups = [space_group]
        init_state = get_gfn_init_state(batch_size, energy_function.data_ndim, device)

        params_record = np.zeros((samples_per_mol, len(mol_list), 12))
        energy_record = np.zeros((samples_per_mol, len(mol_list)))
        density_record = np.zeros_like(energy_record)
        sample_record = []
        if do_opt:
            opt_params_record = np.zeros((samples_per_mol, len(mol_list), 12))
            opt_energy_record = np.zeros((samples_per_mol, len(mol_list)))
            opt_density_record = np.zeros_like(energy_record)
            opt_sample_record = []

        if generator == 'generator':
            """embed the dataset"""
            if hasattr(mol_list[0], 'embedding'):
                if mol_list[0].embedding is not None:
                    pass
                else:
                    mol_list = embed_dataset(mol_list, encoder=encoder)
            else:
                mol_list = embed_dataset(mol_list, encoder=encoder)

        """sample"""

        for s_ind in tqdm(range(samples_per_mol)):
            ssample_record = []
            if do_opt:
                opt_ssample_record = []

            for b_ind in range(num_batches):
                batch_inds = np.arange(b_ind * batch_size, (b_ind + 1) * batch_size)
                mol_batch = collate_data_list([mol_list[ind] for ind in batch_inds]).to(device)

                if generator == 'generator':
                    (_, samples, log_r, _, _,
                     _, sample_batch, _, _, _, _,
                     _, _, _, _,
                     log_T_tensor) = sample_eval_fwd_trajs(
                        init_state, gfn_model, discretizer, energy_function, mol_batch)
                # # DEPRECATED
                # elif generator == 'random':
                #     crystal_batch = collate_data_list(
                #         mol_to_blank_crystal_list(mol_batch, [space_group for _ in range(mol_batch.num_graphs)], ))
                #     samples = sample_crystal_prior(crystal_batch, 1)
                #     log_T_tensor = torch.ones(crystal_batch.num_graphs, device=device) * energy_function.temperature
                #     log_r, sample_batch = energy_function.log_reward(
                #         samples, mol_batch=mol_batch,
                #         log_temperature=log_T_tensor,
                #         return_exp=True)

                params_record[s_ind, batch_inds] = samples.cpu().detach().numpy()
                energy_record[s_ind, batch_inds] = sample_batch.lj.cpu().detach().numpy()
                density_record[s_ind, batch_inds] = sample_batch.packing_coeff.cpu().detach().numpy()
                ssample_record.append(sample_batch.cpu().detach().to_data_list())

                if do_opt:
                    opt_batch = sample_batch.clone()
                    opt_batch = opt_batch.to(device)
                    opt_traj = opt_batch.optimize_crystal_parameters(**optim_kwargs)
                    opt_batch = opt_batch.cpu()

                    finished_batch = collate_data_list(opt_traj[-1])

                    opt_params_record[s_ind, batch_inds] = finished_batch.latent_params().cpu().detach().numpy()
                    opt_energy_record[s_ind, batch_inds] = finished_batch.lj.cpu().detach().numpy()
                    opt_density_record[s_ind, batch_inds] = finished_batch.packing_coeff.cpu().detach().numpy()
                    opt_ssample_record.append(opt_traj[-1])

            sample_record.append(ssample_record)
            if do_opt:
                opt_sample_record.append(opt_ssample_record)

    if do_opt:
        return params_record, energy_record, density_record, sample_record, \
            opt_params_record, opt_energy_record, opt_density_record, opt_sample_record


    else:
        return params_record, energy_record, density_record, sample_record


@torch.no_grad()
def sample_eval_fwd_trajs(init_state,
                          gfn_model,
                          discretizer,
                          energy_function,
                          mol_batch, no_conditioning: bool = False,
                          sg_inds=None, z_primes=None):
    mol_batch, log_T_tensor, sg_inds, zps, condition = energy_function.condition_samples(
        mol_batch, sg_inds=sg_inds, z_primes=z_primes)
    condition = condition.to(gfn_model.device)
    if no_conditioning:
        condition = False
    (states, log_pfs, log_pbs, log_flow,
     means_f, logvars_f, means_b, logvars_b) = gfn_model.get_traj_fwd(
        init_state, discretizer, None, condition, mol_batch, return_gauss_params=True)

    log_r, sample_batch = energy_function.log_reward(
        states[:, -1], mol_batch=mol_batch, log_temperature=log_T_tensor, return_exp=True)

    cpu = lambda t: t.cpu().detach()
    return {
        'flow_states': cpu(states),
        'log_r': cpu(log_r),
        'log_pfs': cpu(log_pfs),
        'log_pbs': cpu(log_pbs),
        'log_flow': cpu(log_flow),
        'log_T_tensor': cpu(log_T_tensor),
        'sample_batch': sample_batch.cpu().detach(),
        'gauss_params': {'means_f': cpu(means_f), 'logvars_f': cpu(logvars_f),
                         'means_b': cpu(means_b), 'logvars_b': cpu(logvars_b)},
    }


# @torch.no_grad()
# def mean_log_likelihood(terminal_state, gfn, log_reward_fn, num_evals=10):
#     bsz = terminal_state.shape[0]
#     terminal_state = terminal_state.unsqueeze(1).repeat(1, num_evals, 1).view(bsz * num_evals, -1)
#     states, log_pfs, log_pbs, log_fs = gfn.get_traj_bwd(terminal_state, None, log_reward_fn)
#     log_weight = (log_pfs.sum(-1) - log_pbs.sum(-1)).view(bsz, num_evals, -1)
#     return logmeanexp(log_weight, dim=1).mean()

#
# def crystal_list_rdf(samples, batch_size, device):
#     num_batches = len(samples) // batch_size
#     if len(samples) % batch_size != 0:
#         num_batches += 1
#
#     rdfs = []
#     for b_ind in range(num_batches):
#         batch_inds = np.arange(b_ind * batch_size, min(len(samples), (b_ind + 1) * batch_size))
#         mol_batch = collate_data_list([samples[ind] for ind in batch_inds]).to(device)
#         rdf, rr = get_rdfs(mol_batch)
#         rdfs.append(rdf)
#
#     return torch.cat(rdfs), rr
#
#
# def get_rdfs(crystal_batch):
#     with torch.no_grad():
#         cluster_batch = crystal_batch.mol2cluster(cutoff=6)
#         cluster_batch.construct_radial_graph(cutoff=6)
#         rdf, rr, _ = crystal_rdf(cluster_batch,
#                                  cluster_batch.edges_dict,
#                                  rrange=[0, 6], bins=2000,
#                                  mode='intermolecular',
#                                  elementwise=True,
#                                  raw_density=True,
#                                  cpu_detach=False)
#
#     return rdf.cpu().detach(), rr

#
# @torch.no_grad()
# def sample_csd_rdf_dists(csd_mols, csd_sampling_dict, eval_batch_size, device):
#     sample_rdfs = []
#     for ind in tqdm(range(len(csd_mols))):
#         identifier = csd_mols[ind].identifier
#         for ind2 in range(len(csd_sampling_dict[identifier]['samples'])):
#             samples = csd_sampling_dict[identifier]['samples'][ind2]
#             samples = [item for sublist in samples for item in sublist]
#
#             rdf, rr = crystal_list_rdf(samples, eval_batch_size, device)
#             sample_rdfs.append(rdf)
#
#     per_csd_rdfs = []
#     ii = 0
#     for ind in range(len(csd_mols)):
#         ss_rdf = []
#         for ind2 in range(len(csd_sampling_dict[identifier]['samples'])):
#             ss_rdf.append(sample_rdfs[ii])
#             ii += 1
#         per_csd_rdfs.append(torch.cat(ss_rdf))
#
#     sample_rdfs = torch.stack(per_csd_rdfs)
#     csd_rdfs, rr = crystal_list_rdf(csd_mols,
#                                     eval_batch_size,
#                                     device)
#
#     rdf_dists = torch.zeros_like(sample_rdfs[:, :, 0, 0])
#     for ind in range(len(csd_mols)):
#         rdf_dists[ind] = compute_rdf_distance(csd_rdfs[ind].to(device), sample_rdfs[ind].to(device), rr)
#     return rdf_dists, rr

#
# def sample_csd_lattice_divs(csd_mols, csd_sampling_dict):
#     identifiers = [elem.identifier for elem in csd_mols]
#     js_divs = []
#     for ind, ident in enumerate(identifiers):
#         box_matrix = csd_mols[ind].T_fc[0].T.cpu().detach().numpy()
#         csd_dists = lattice_distance_spectrum(box_matrix,
#                                               max_radius=50,
#                                               resolution=0.01)
#         samples = []
#         for elem in csd_sampling_dict[identifiers[ind]]['samples']:
#             samples.extend(elem)
#         samples = [item for sublist in samples for item in sublist]
#         hist1, hr = np.histogram(csd_dists, bins=100, range=[0, 50])
#         divs = []
#
#         for j in range(len(samples)):
#             box_matrix = samples[j].T_fc[0].T.cpu().detach().numpy()
#             sample_dists = lattice_distance_spectrum(box_matrix,
#                                                      max_radius=50,
#                                                      resolution=0.01)
#             hist2, hr = np.histogram(sample_dists, bins=100, range=[0, 50])
#             divs.append(jensenshannon(hist1, hist2))
#
#         js_divs.append(divs)
#
#     return js_divs


# def lattice_distance_spectrum(cell_matrix, max_radius=0.0, resolution=0.01):
#     """Compute sorted inter-point distances for lattice defined by 3x3 cell_matrix"""
#     max_index = int(np.ceil(max_radius / np.min(np.linalg.norm(cell_matrix, axis=1))))
#     shifts = np.mgrid[-max_index:max_index + 1, -max_index:max_index + 1, -max_index:max_index + 1].reshape(3, -1).T
#     distances = np.linalg.norm(shifts @ cell_matrix, axis=1)
#     distances = distances[(distances > 1e-8) & (distances < max_radius)]
#     distances = np.sort(np.round(distances / resolution) * resolution)  # bin by resolution
#     return distances


def get_plotly_fig_size_mb(fig) -> float:
    # Convert Plotly figure to JSON string
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return sys.getsizeof(fig_json) / (1024 * 1024)


def big_staircse_comparison(dbatch, ebatch):
    gen_samples = dbatch.latent_params()
    elats = ebatch.latent_params()
    if torch.is_tensor(gen_samples):
        gen_samples = gen_samples.detach().cpu().numpy()
    N, D = gen_samples.shape

    # Create D×D subplots (upper triangle empty)
    fig = make_subplots(
        rows=D, cols=D,
        horizontal_spacing=0.01, vertical_spacing=0.01,
        shared_xaxes=True, shared_yaxes=True,
    )

    # Loop over lower triangle
    for i in range(D):
        for j in range(D):
            if j >= i:
                continue  # keep lower triangle only

            x = gen_samples[:, j]
            y = gen_samples[:, i]

            trace = go.Histogram2dContour(
                x=x, y=y,
                ncontours=100,
                colorscale='icefire',
                showscale=False,
                contours=dict(coloring='fill', showlines=False, start=0, end=None, size=None),
                line=dict(smoothing=0.85, width=0),
                nbinsx=100,
                nbinsy=100,
            )
            fig.add_trace(trace, row=i + 1, col=j + 1)

            trace = go.Scatter(x=elats[:, j], y=elats[:, i], mode='markers',
                               marker_color='yellow', marker_line_width=4, opacity=0.5,
                               marker_line_color='black', marker_size=14, showlegend=False)
            fig.add_trace(trace, row=i + 1, col=j + 1)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        # height=1000,
        # width=1000,
        showlegend=False,
    )
    fig.update_layout(
        font=dict(family="Helvetica", size=12),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=30, r=30, t=20, b=30),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, ticks='outside', tickwidth=1)  # , range=[-1,1])
    fig.update_yaxes(showgrid=False, zeroline=False, ticks='outside', tickwidth=1)  # , range=[-1,1])
    fig.update_layout(height=2400, width=3000)
    return fig


@dataclass(frozen=True)
class Trigger:
    """Result of a single record() call.

    Truthy when it fires, so you can write `if monitor.record(loss): ...`.
    `reason` is one of: "non-finite", "ceiling", "spike-factor", "spike-z",
    "trend", or "" when it did not fire.
    """

    fire: bool
    reason: str = ""
    value: float = float("nan")
    baseline: float = float("nan")
    long_baseline: Optional[float] = None
    z: Optional[float] = None
    slope_rel: Optional[float] = None
    call: int = 0
    step: Optional[int] = None

    def __bool__(self) -> bool:
        return self.fire

    def __str__(self) -> str:
        if not self.fire:
            return f"Trigger(fire=False, call={self.call})"
        return (
            f"Trigger(fire=True, reason={self.reason!r}, value={self.value:.4g}, "
            f"baseline={self.baseline:.4g}, call={self.call}, step={self.step})"
        )


def _to_float(x) -> float:
    """Coerce a scalar loss to a python float, accepting torch tensors."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    return float(x)


def _ols_slope(y: List[float]) -> float:
    """Closed-form least-squares slope of y against x = 0, 1, ..., n-1."""
    n = len(y)
    if n < 2:
        return 0.0
    sx = (n - 1) * n / 2.0
    sxx = (n - 1) * n * (2 * n - 1) / 6.0
    sy = math.fsum(y)
    sxy = math.fsum(i * yi for i, yi in enumerate(y))
    denom = n * sxx - sx * sx
    if denom == 0.0:
        return 0.0
    return (n * sxy - sx * sy) / denom


class LossSpikeMonitor:
    """Detects loss explosions / sustained increases / ceiling crossings.

    Typical use (called every N training steps):

        mon = LossSpikeMonitor(warmup=2000, cooldown=3000, ceiling_factor=8.0)
        ...
        if step % N == 0:
            trig = mon.record(loss, step=step)   # pass your global step
            if trig:
                cut_lr()  # e.g. multiply every param_group["lr"] by 0.5

    Detectors (disable any relative one by passing None):

        ceiling_factor  blowout cap relative to the long-tail median: fires if
                        value >= ceiling_factor * median(long_window), and only
                        when that long median is positive. Tracks your loss
                        scale instead of needing a hand-tuned constant.
        spike_factor    fires if value >= spike_factor * baseline_median
                        (only when the baseline is positive). Intuitive
                        "the loss doubled" style check.
        spike_z         robust outlier check: a modified z-score using the
                        median absolute deviation of the window. Catches spikes
                        even when the baseline is non-positive or noisy.
        trend_rel       sustained-increase check: fits a line over the window and
                        fires if the implied rise across the window is at least
                        this fraction of the baseline (e.g. 0.5 -> ~50% rise).

    Timescales: `window` and `long_window` are buffer sizes in *samples* (how
    much history to keep for spike/trend and for the ceiling reference).
    `warmup` and `cooldown` are measured against the `step` you pass to
    record(), i.e. in *training steps* -- warmup=2000 means relative checks stay
    dormant for the first 2000 steps, cooldown=3000 means ~3000 steps of quiet
    after a fire. If you don't pass `step`, both fall back to counting record()
    calls. `min_samples` is a separate floor (in samples) so the statistics stay
    valid even if your step interval makes warmup map to very few samples. Only
    non-finite ever fires during warmup.
    """

    def __init__(
            self,
            window: int = 20,
            warmup: int = 200,
            cooldown: int = 300,
            ceiling_factor: Optional[float] = 8.0,
            long_window: int = 200,
            spike_factor: Optional[float] = 2.0,
            spike_z: Optional[float] = 4.0,
            trend_rel: Optional[float] = 0.5,
            min_samples: int = 5,
            reset_on_fire: bool = False,
            name: str = "loss",
    ):
        if window < 2:
            raise ValueError("window must be >= 2")
        if long_window < window:
            raise ValueError("long_window must be >= window")
        self.window = int(window)
        self.long_window = int(long_window)
        self.warmup = max(0, int(warmup))  # in clock units (steps)
        self.cooldown = max(0, int(cooldown))  # in clock units (steps)
        self.min_samples = max(2, int(min_samples))
        self.ceiling_factor = ceiling_factor
        self.spike_factor = spike_factor
        self.spike_z = spike_z
        self.trend_rel = trend_rel
        self.reset_on_fire = bool(reset_on_fire)
        self.name = name
        self.best_loss = torch.inf

        self._hist: Deque[float] = deque(maxlen=self.window)
        self._long_hist: Deque[float] = deque(maxlen=self.long_window)
        self._calls = 0
        self._clock: float = float("-inf")  # last seen step (or call no.)
        self._start_clock: Optional[float] = None  # clock at first record
        self._cooldown_until: float = float("-inf")
        self._fires = 0

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    @property
    def baseline(self) -> float:
        """Median of the current (short) window (NaN if empty)."""
        return median(self._hist) if self._hist else float("nan")

    @property
    def long_baseline(self) -> float:
        """Median of the long-tail buffer (NaN if empty)."""
        return median(self._long_hist) if self._long_hist else float("nan")

    @property
    def in_cooldown(self) -> bool:
        return self._clock < self._cooldown_until

    @property
    def num_fires(self) -> int:
        return self._fires

    def __len__(self) -> int:
        return len(self._hist)

    def record(self, value, step: Optional[int] = None) -> Trigger:
        """Record one loss value and return a Trigger.

        Fires only on catastrophic explosions: non-finite values, or a value
        that blows past `ceiling_factor` times the slow long-tail median.
        """
        self._calls += 1
        call = self._calls
        v = _to_float(value)
        if v < self.best_loss:
            self.best_loss = v

        clock = float(step) if step is not None else float(self._calls)
        if self._start_clock is None:
            self._start_clock = clock
        self._clock = clock

        long_base = median(self._long_hist) if self._long_hist else float("nan")
        warm = (self._clock - self._start_clock) >= self.warmup

        reason: Optional[str] = None
        if not math.isfinite(v):
            reason = "non-finite"
        elif (
                self.ceiling_factor is not None
                and warm
                and len(self._long_hist) >= self.min_samples
                and long_base > 0.0
                and v >= self.ceiling_factor * long_base
        ):
            reason = "ceiling"

        fire = reason is not None and not self.in_cooldown

        self._long_hist.append(v)

        if fire:
            self._fires += 1
            self.fire_cooldown(clock)

        return Trigger(
            fire=fire,
            reason=reason or "",
            value=v,
            long_baseline=long_base,
            call=call,
            step=step,
        )

    def fire_cooldown(self, clock):
        self._cooldown_until = clock + self.cooldown

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        """Serializable runtime state (config is not stored -- it comes from
        the constructor when you rebuild the object)."""
        return {
            "hist": list(self._hist),
            "long_hist": list(self._long_hist),
            "calls": self._calls,
            "clock": self._clock,
            "start_clock": self._start_clock,
            "cooldown_until": self._cooldown_until,
            "fires": self._fires,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._hist = deque(sd.get("hist", []), maxlen=self.window)
        self._long_hist = deque(sd.get("long_hist", []), maxlen=self.long_window)
        self._calls = int(sd.get("calls", 0))
        self._clock = float(sd.get("clock", float("-inf")))
        sc = sd.get("start_clock", None)
        self._start_clock = None if sc is None else float(sc)
        self._cooldown_until = float(sd.get("cooldown_until", float("-inf")))
        self._fires = int(sd.get("fires", 0))

    def reset(self) -> None:
        self._hist.clear()
        self._long_hist.clear()
        self._calls = 0
        self._clock = float("-inf")
        self._start_clock = None
        self._cooldown_until = float("-inf")
        self._fires = 0


def cal_subtb_coef_matrix(lamda, N):
    """
    diff_matrix: (N+1, N+1)
    0, 1, 2, ...
    -1, 0, 1, ...
    -2, -1, 0, ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    range_vals = torch.arange(N + 1)
    diff_matrix = range_vals - range_vals.view(-1, 1)
    B = np.log(lamda) * diff_matrix
    B[diff_matrix <= 0] = -np.inf
    log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
    coef = torch.exp(B - log_total_lambda)
    return coef