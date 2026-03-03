"""sensitivity analysis"""
from copy import deepcopy

import numpy as np
import torch
from torch_scatter import scatter
from tqdm import tqdm

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.common.geometry_utils import compute_latent_distance, simple_latent_distance
from mxtaltools.dataset_utils.utils import collate_data_list
import plotly.graph_objects as go


def param_dist(ref, sample, scale):
    """
    :param ref: [n, k]
    :param sample: [k]
    :param scale: [k]
    :return: [k]
    """
    return (ref - sample[None, :]).abs() / scale
def wrap_to_pi(x):
    # (-pi, pi]
    return (x + torch.pi) % (2 * torch.pi) - torch.pi




path = r"D:\crystal_datasets\test_new_csd.pt"
dataset = torch.load(path, weights_only=False)
max_z_prime  = 1
dataset = [elem for elem in dataset if elem.z_prime <= max_z_prime]
batch = collate_data_list(dataset[:100], max_z_prime=max_z_prime)

angs = [False] * 6
for zp in range(max_z_prime):
    angs.extend([False, False, False])
for zp in range(max_z_prime):
    angs.extend([False, True, True])
    # phi and r dimensions arein rotational basis
ang_mask = torch.tensor(angs)

batch.pos = batch.pos + torch.randn_like(batch.pos) * 0.01
batch.latent_to_cell_params(batch.latent_params().clip(min=-1, max=1))
batch.clean_cell_parameters(
    mode='hard',
    canonicalize_orientations=False,
)  # box analysis included in here

cell_params = batch.zp1_cell_parameters()
latents = batch.latent_params()

batch.pose_aunit(std_orientation=True)
batch.build_unit_cell()
batch.analyze(['elj','rdf'],cutoff=10, rdf_cutoff=10, assign_outputs=True)
E0 = batch.elj

iters = 100
bins = torch.linspace(0, 10, 500)
log_noise_range = [-3, -1]
rdf_dists = torch.zeros((iters, batch.num_graphs), dtype=torch.float32)
noises = torch.zeros((iters, batch.num_graphs, latents.shape[-1]))
lat_dists = torch.zeros_like(rdf_dists)
rmsds = torch.zeros_like(rdf_dists)
ens = torch.zeros_like(rdf_dists)
with torch.no_grad():
    for iter in tqdm(range(iters)):
        nbatch = deepcopy(batch)

        rand_dir = torch.randn_like(latents)
        rand_dir = rand_dir / rand_dir.norm(dim=-1, keepdim=True)
        # rand_magnitude = torch.randn(len(samples), device=samples.device).abs() * noise_level
        u = torch.rand(len(latents), device=latents.device)
        rand_magnitude = 10 ** (log_noise_range[0] + (log_noise_range[1] - log_noise_range[0]) * u)
        noised_samples = (latents.clone().detach() + rand_dir * rand_magnitude[:, None])
        # wrap orientation angular dimensions
        noised_samples[:, ang_mask] = wrap_to_pi(noised_samples[:, ang_mask] * torch.pi) / torch.pi
        noised_samples = noised_samples.clip(min=-1, max=1)

        nbatch.latent_to_cell_params(noised_samples,
                                     skip_box_analysis=True,
                                     skip_enforce_crystal_system=True)
        nbatch.clean_cell_parameters(
            mode='hard',
            canonicalize_orientations=False,
        )  # box analysis included in here
        nbatch.pose_aunit(std_orientation=True)
        nbatch.build_unit_cell()
        lat_dists[iter] = compute_latent_distance(nbatch.latent_params().cpu(), latents)

        rmsds[iter] = scatter((nbatch.pos - batch.pos).norm(dim=-1), batch.batch, reduce='mean', dim=0, dim_size=batch.num_graphs)
        noise = nbatch.latent_params() - latents
        noises[iter] = noise
        nbatch = nbatch.cuda()
        nbatch.analyze(['elj', 'rdf'], cutoff=10, rdf_cutoff=10, assign_outputs=True)
        rdf_dists[iter] = compute_rdf_distance(nbatch.rdf.cpu(), batch.rdf, bins).cpu()
        ens[iter] = nbatch.elj.cpu()

print(np.corrcoef(lat_dists.flatten().log10(), rdf_dists.flatten().log10()))
go.Figure(go.Scatter(x=lat_dists.flatten().log10(), y=rdf_dists.flatten().log10(), mode='markers')).show()
go.Figure(go.Scatter(x=lat_dists.flatten().log10(), y=(E0[None, ...] - ens).abs().flatten().log10(), mode='markers')).show()
go.Figure(go.Scatter(x=rmsds.flatten().log10(), y=rdf_dists.flatten().log10(), mode='markers')).show()

# filter bug
eps = noises.reshape(-1, 12)           # [M, 12]
#y   = (rdf_dists.reshape(-1) ** 2)         # [M]
y = (ens - E0[None, :]).reshape(-1)
good_inds = (noises.reshape(-1, 12).norm(dim=-1)) < 1
eps = eps[good_inds]
y = y[good_inds]

num = torch.einsum('m,mi,mj->ij', y, eps, eps)   # Σ y ε εᵀ
den = torch.sum((eps**2).sum(dim=1)**2)          # normalization
G_global = num / den

eigvals, eigvecs = torch.linalg.eigh(G_global)

iters, B, d = noises.shape
g_per_sample = torch.zeros(B, d, device=noises.device)

for b in range(B):
    eps_b = noises[:, b, :]           # [iters, 12]
    y_b   = rdf_dists[:, b] ** 2            # [iters]
    g_per_sample[b] = torch.linalg.lstsq(eps_b**2, y_b).solution

g_mean = g_per_sample.mean(dim=0)
g_std  = g_per_sample.std(dim=0)
g_cv   = g_std / g_mean
# per sample G_b=G[b]
# eigvals_b = torch.linalg.eigvalsh(G_b)
# d_eff = (eigvals_b.sum()**2) / (eigvals_b**2).sum()

y_pred = torch.einsum('mi,ij,mj->m', eps, G_global, eps)
print(torch.corrcoef(torch.stack([y, y_pred]))[0,1])
print(g_mean, g_std, g_cv, G_global)

go.Figure(go.Heatmap(z=G_global)).show()
from plotly.subplots import make_subplots
fig = make_subplots(rows=4, cols=3)
for ind in range(12):
    row = ind // 3 + 1
    col = ind % 3 + 1
    fig.add_histogram2d(x=eps[:, ind].abs(), y=rdf_dists.reshape(-1).log(), nbinsx=50, nbinsy=50, row=row, col=col)
fig.show()
aa = 1