import numpy as np
import plotly.graph_objects as go
import torch
torch.cuda.set_per_process_memory_fraction(0.9)

from mxtaltools.dataset_utils.utils import collate_data_list
dataset_path = r"D:\crystal_datasets\test_new_new_csd.pt"  # todo re-do with full CSD
space_group = 14
z_prime = 1
energy_function = 'elj'


data = torch.load(dataset_path,weights_only=False)
data = [elem for elem in data if (elem.sg_ind == space_group) & (elem.z_prime == z_prime)]
batch = collate_data_list(data)

batch.analyze([energy_function], assign_outputs=True)

latents = batch.latent_params()

sample_dmat = torch.cdist(latents[:1000], latents[:1000])

d_cut = sample_dmat.flatten().quantile(0.15)

# sample_dmat.fill_diagonal_(torch.inf)
# density = torch.exp(-sample_dmat**2/(2*d_cut**2)).sum(dim=-1)

@torch.no_grad()
def knn_density(x, k=50, d_cut=1.0, chunk=2048):
    """Gaussian-kernel density from k nearest neighbours. O(N*chunk) memory, GPU-native."""
    N = x.shape[0]
    density = torch.empty(N, device=x.device)
    inv2s2 = 1.0 / (2 * d_cut ** 2)
    for i in range(0, N, chunk):
        d = torch.cdist(x[i:i + chunk], x)                       # (b, N)
        # k+1 smallest includes self at distance 0 -> drop column 0.
        # Robust at chunk boundaries (no fill_diagonal_ bookkeeping needed).
        knn = d.topk(k + 1, dim=-1, largest=False).values[:, 1:]  # (b, k)
        density[i:i + chunk] = torch.exp(-(knn ** 2) * inv2s2).sum(-1)
    return density
@torch.no_grad()
def equalize_density(latents, d_cut, k=500, target_quantile=0.95,
                     spawn_frac=0.5, growth_per_pass=0.1,
                     n_passes=1000, tol=1e-3, max_factor=20.0, chunk=2048,
                     verbose=True):
    x = latents.clone().float()
    if torch.cuda.is_available():
        x = x.cuda()
    d_cut = float(d_cut); D = x.shape[1]
    sigma_spawn = spawn_frac * d_cut / (D ** 0.5)
    N0 = x.shape[0]
    anchors = torch.arange(N0, device=x.device)        # parents are ONLY the originals
    max_total = int(max_factor * N0)

    density = knn_density(x, k=k, d_cut=d_cut, chunk=chunk)
    target = density.quantile(target_quantile)
    history = [density[:N0].cpu()]                       # track originals only
    if verbose:
        d0 = density[:N0]
        print(f"pass  0  N={x.shape[0]:>7d}  CV0={ (d0.std()/d0.mean()):.4f}  "
              f"mean0={d0.mean():.1f}  min0={d0.min():.1f}  target={target:.1f}")

    for p in range(1, n_passes):
        d0 = density[:N0]                                # deficit on anchors only
        deficit = (target - d0).clamp(min=0.0)
        if deficit.sum() == 0:
            print("anchors all filled"); break
        budget = min(int(growth_per_pass * x.shape[0]), max_total - x.shape[0])
        if budget <= 0:
            print("hit growth cap"); break

        parent_idx = anchors[torch.multinomial(deficit, budget, replacement=True)]
        children   = x[parent_idx] + sigma_spawn * torch.randn(budget, D, device=x.device)
        children = children.clip(-1, 1)
        x = torch.cat([x, children], dim=0)

        density = knn_density(x, k=k, d_cut=d_cut, chunk=chunk)
        d0 = density[:N0]
        cv0 = (d0.std() / d0.mean()).item()
        history.append(d0.cpu())
        if verbose:
            print(f"pass {p:2d}  N={x.shape[0]:>7d}  CV0={cv0:.4f}  "
                  f"mean0={d0.mean():.1f}  min0={d0.min():.1f}")
        if cv0 < tol:
            break

    return x.cpu(), history

def plot_density_hist(history, nbins=100):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=history[0].numpy(), nbinsx=nbins,
                               name="initial", opacity=0.6))
    fig.add_trace(go.Histogram(x=history[-1][:len(history[0])].numpy(), nbinsx=nbins,
                               name=f"flattened ({len(history) - 1} passes)",
                               opacity=0.6))
    fig.update_layout(barmode="overlay",
                      xaxis_title="local density",
                      yaxis_title="count",
                      title="Density distribution: before vs after flattening")
    return fig

x_eq, hist = equalize_density(latents, d_cut, k=3000, target_quantile=0.99,
                              spawn_frac=0.5, growth_per_pass = 0.01, tol=0.0001,
                              n_passes = 1000, max_factor=20.0)
plot_density_hist(hist).show()

samps = np.random.choice(len(data), len(x_eq), replace=True)
full_data = [data[s] for s in samps]
full_batch = collate_data_list(full_data)
full_batch.latent_to_cell_params(x_eq, skip_box_analysis=False, skip_enforce_crystal_system=False)
reduction = full_batch.compute_cell_reduction_penalty()
good_inds = torch.argwhere(reduction < 0.01).flatten()
full_batch = collate_data_list([full_data[ind] for ind in good_inds])
full_batch.plot_batch_cell_params(space='latent', ref_dist=batch.latent_params())
full_batch.plot_batch_staircase(space='latent')
batch.plot_batch_staircase(space='latent')

fin_density = knn_density(full_batch.latent_params(), k=3000, d_cut=d_cut, chunk=2048)
fig = go.Figure(go.Histogram(x=fin_density.cpu().detach().numpy(), nbinsx=100, histnorm='probability density'))
fig.add_histogram(x=fin_density[:len(data)].cpu().detach().numpy(), nbinsx=100, histnorm='probability density')
fig.add_histogram(x=hist[0].cpu().detach().numpy(), nbinsx=100, histnorm='probability density')
fig.add_histogram(x=hist[-1].cpu().detach().numpy(), nbinsx=100, histnorm='probability density')
fig.show()
aa = 1

