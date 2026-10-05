"""
Per-molecule draws against a reference, on two bases only: the 12 latent marginals and
lattice energy against packing coefficient. No RDF, no distance matrix, no density.

For each identifier: N draws from one checkpoint through eval/cond_panel/sampler.py, set
against that molecule's reference rows. The default reference is the THIN one already on
disk: a training molecule's rows in the prior file (its anchors, Niggli-migrated, 6-10 per
molecule, from 10 random starts), a held-out molecule's rows in the anchors file (7-10,
cells in the August legacy-wall convention). Energies and packing coefficients are the
values stored on those rows (raw ELJ, cutoff 10 -- the scale the model's mol_energy is on
at lj_coeff 1). Latents come from the same latent_params() call on both sides; a reference
row enters the latent panels only if its cell carries zero reduction penalty under the
current walls, since a cell outside the Niggli domain is a different description of the
same crystal and would differ in latents for gauge reasons alone.

    cd energy_sampling
    python -m eval.cond_panel.panel_dists --checkpoint D:/.../<stem>_step28000.pt \
        --config configs/cond_tb_sep25/ctb25_extreme_l1_cont3.yaml --out <dir> --identifiers SMILES ...
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval.cond_panel.sampler import draw, load_run, reasonable_mask

ANCHORS = r'D:\crystal_datasets\conditional\anchors\qm9c100k_valid.pt'
LATENT_NAMES = ('a', 'b', 'c', 'alpha', 'beta', 'gamma', 'u', 'v', 'w', 'theta', 'phi', 'r')
MODEL, REF, INK, INK2, GRID, SURFACE = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def reference_rows(path, identifiers):
    """Rows of `path` (a prior dict, a batch or a list of crystals) for each identifier."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    data = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        data = data.get('equalized_prior', data.get('prior'))
    out = {}
    if isinstance(data, list):
        for ident in identifiers:
            rows = [c for c in data if c.identifier == ident]
            out[ident] = collate_data_list(rows) if rows else None
    else:
        ids = list(data.identifier)
        for ident in identifiers:
            idx = [i for i, x in enumerate(ids) if x == ident]
            out[ident] = data.subsample_new_batch(torch.tensor(idx)) if idx else None
    return out


def niggli_penalty(batch):
    from mxtaltools.common.sym_utils import cell_reduction_penalty
    return cell_reduction_penalty(batch.cell_angles.float(), batch.cell_lengths.float(),
                                  batch.sg_ind, 0.0).flatten()


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


def page(ident, split, step, d, ref, ref_lat, n_ref_lat, out_png):
    lat = d['latent'].numpy()
    e, cp = d['mol_energy'].numpy(), d['packing_coeff'].numpy()
    fig = plt.figure(figsize=(17, 6.2), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 6, width_ratios=[2.2, 0.15, 1, 1, 1, 1], wspace=0.35, hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    style(ax)
    keep = np.random.default_rng(0).permutation(len(e))[:4000]
    ax.scatter(cp[keep], e[keep], s=4, c=MODEL, alpha=0.25, lw=0, label=f'model, {len(e)} draws (4,000 shown)')
    if ref is not None:
        ax.scatter(ref['cp'], ref['e'], s=70, c=REF, edgecolors=SURFACE, linewidths=1.5, zorder=3,
                   label=f'reference, {len(ref["e"])} anchors')
    for x in (0.55, 0.95):
        ax.axvline(x, color=INK2, lw=0.8, ls='--')
    ax.axhline(0, color=INK2, lw=0.8, ls='--')
    ax.set_xlabel('packing coefficient', color=INK)
    ax.set_ylabel('lattice energy (raw ELJ)', color=INK)
    lo = np.quantile(e, 0.001) if ref is None else min(np.quantile(e, 0.001), ref['e'].min())
    ax.set_ylim(lo - 10, max(np.quantile(e, 0.99), 0) + 10)
    ax.set_xlim(0.35, 1.0)
    ax.legend(loc='upper left', fontsize=8, frameon=False, labelcolor=INK)
    ax.set_title('energy vs packing (dashed: reasonable window)', fontsize=10, color=INK)
    bins = np.linspace(-1, 1, 41)
    for k, name in enumerate(LATENT_NAMES):
        a = fig.add_subplot(gs[k // 4, 2 + k % 4])
        style(a)
        a.hist(lat[:, k], bins=bins, density=True, color=MODEL, alpha=0.35, lw=0)
        a.hist(lat[:, k], bins=bins, density=True, histtype='step', color=MODEL, lw=1.2)
        if ref_lat is not None and len(ref_lat):
            ymax = a.get_ylim()[1]
            a.vlines(ref_lat[:, k], 0, 0.22 * ymax, color=REF, lw=2)
        a.set_xlim(-1, 1)
        a.set_yticks([])
        a.set_title(name, fontsize=9, color=INK)
    fig.suptitle(f'{ident}  ({split}, step {step})', fontsize=12, color=INK, x=0.07, ha='left')
    fig.text(0.07, 0.005,
             f'Latent panels: model draws (blue, density over the [-1, 1] latent box) and reference rows whose cell '
             f'is Niggli-reduced under the current walls (orange ticks, {n_ref_lat} rows). Energies raw ELJ at '
             f'lj_coeff 1; T = 6.9 on this scale.', fontsize=8, color=INK2, ha='left')
    fig.savefig(out_png, dpi=110, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--identifiers', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--anchors', default=ANCHORS)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.device == 'cpu':
        assert not torch.cuda.is_available(), 'CPU run with a visible GPU: set CUDA_VISIBLE_DEVICES=-1'
    os.makedirs(args.out, exist_ok=True)

    run = load_run(args.checkpoint, args.config, device=args.device)
    prior_ref = reference_rows(run.config['prior_path'], args.identifiers)
    anchor_ref = reference_rows(args.anchors, args.identifiers)
    T = run.temperature
    rows_out = []
    for ident in args.identifiers:
        split = 'held-out' if run.is_held_out(ident) else 'train'
        d = draw(run, run.rows(ident, args.n), batch_size=1000, seed=0)
        d['latent'] = d['sample_batch'].latent_params().detach().cpu()
        ok = torch.isfinite(d['mol_energy'])
        red0 = d['reduction_en'] <= 1e-6
        ref_b = prior_ref.get(ident) if split == 'train' else anchor_ref.get(ident)
        ref = ref_lat = None
        n_ref_lat = 0
        if ref_b is not None:
            ref = {'e': ref_b.elj.double().flatten().numpy() / ref_b.z_prime.double().flatten().numpy(),
                   'cp': ref_b.packing_coeff.double().flatten().numpy()}
            in_gauge = niggli_penalty(ref_b) <= 1e-6
            n_ref_lat = int(in_gauge.sum())
            ref_lat = ref_b.latent_params()[in_gauge].detach().cpu().numpy()
        page(ident, split, run.step, d, ref, ref_lat, n_ref_lat,
             os.path.join(args.out, f'panel_step{run.step}_{len(rows_out)}.png'))

        e = d['mol_energy'][ok].double()
        e_min_ref = float(ref['e'].min()) if ref is not None else float('nan')
        rows_out.append({
            'identifier': ident, 'split': split, 'draws': int(ok.sum()),
            'reasonable_frac': float(reasonable_mask(run, d['sample_batch']).float().mean()),
            'reduction_pen_frac': float((~red0).float().mean()),
            'E_model_p05': float(torch.quantile(e, 0.05)), 'E_model_median': float(e.median()),
            'E_ref_min': e_min_ref,
            'E_ref_median': float(np.median(ref['e'])) if ref is not None else float('nan'),
            'n_ref': 0 if ref is None else len(ref['e']), 'n_ref_latent': n_ref_lat,
            'excess_median_nats': (float(e.median()) - e_min_ref) / T,
            'frac_below_ref_min': float((d['mol_energy'][ok & red0] < e_min_ref).float().mean()),
            'cp_model_median': float(d['packing_coeff'][ok].median()),
            'cp_ref_median': float(np.median(ref['cp'])) if ref is not None else float('nan'),
            'J_F_nats': float(d['log_w'][torch.isfinite(d['log_w'])].mean()),
            'head_log_z': float(d['head_log_z'][0]),
        })

    cols = list(rows_out[0])
    with open(os.path.join(args.out, f'panel_step{run.step}.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)
    for r in rows_out:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == '__main__':
    main()
