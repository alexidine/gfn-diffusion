"""Parity of MACE-OFF24 (medium) against UMA-s 1.1 lattice energies on random CSD Z'=1 organic crystals, at the
experimental geometry (single points, no relaxation).

Crystals: D:/crystal_datasets/test_new_prot_csd.pt (10,000 CSD rows with H), filtered to Z'=1, no cocrystal, elements
within MACE-OFF24's (H C N O F P S Cl Br I), at most MAX_ATOMS atoms per molecule; N drawn at random (SEED).

Per crystal, kJ/mol per molecule:
  uma_raw   UMA-s, periodic energy as is, gas leg = mean of the molecule and its mirror image (run_search's
            mirror_mean reference);
  uma_avg   the same with the periodic energy averaged over the crystal and its mirror image (only in Sohncke groups;
            elsewhere = uma_raw);
  mace      MACE-OFF24 (exactly mirror-invariant).
Both MLIPs' gas legs are isolated molecules; UMA-s's uses its 'omc' task, trained on crystals only (see
docs/design/exp6_recall_test.md), so a per-molecule offset is expected in UMA-s's absolute values.

usage: python -m eval.exp6_recall.csd_parity OUT_DIR [--n 2000] [--seed 0] [--max_atoms 60]
"""
import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch

from eval.exp6_recall.make_configs import SITES, SOHNCKE
from eval.exp6_recall.mirror_test import cap_gpu

DATA = 'D:/crystal_datasets/test_new_prot_csd.pt'
MACE_ELEMENTS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}


def pick(n, seed, max_atoms):
    rows = torch.load(DATA, weights_only=False)
    ok = [r for r in rows if int(r.z_prime.reshape(-1)[0]) == 1 and not bool(r.cocrystal)
          and set(r.z.tolist()) <= MACE_ELEMENTS and 6 in set(r.z.tolist()) and int(r.num_atoms) <= max_atoms]
    random.Random(seed).shuffle(ok)
    return ok[:n], len(ok)


def strip(r):
    r = r.clone()
    for k in ('uma_pot', 'uma_gas_pot', 'uma_mirror_average', 'mace_pot', 'mace_gas_pot', 'unit_cell_pos'):
        if hasattr(r, k):
            delattr(r, k)
    return r


def energies(rows, key, pred, chunk):
    """[n] kJ/mol; NaN where a chunk fails even one row at a time (out of memory or a builder error)."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    out = torch.full((len(rows),), float('nan'), dtype=torch.float64)
    kw = dict(cutoff=10, supercell_size=10, std_orientation=True, assign_outputs=False, predictor=pred)

    def go(lo, hi):
        b = collate_data_list(rows[lo:hi]).to('cuda')
        with torch.no_grad():
            out[lo:hi] = b.analyze(computes=[key], **kw)[key].reshape(-1).double().cpu()

    for lo in range(0, len(rows), chunk):
        hi = min(lo + chunk, len(rows))
        try:
            go(lo, hi)
        except (RuntimeError, ValueError) as e:
            torch.cuda.empty_cache()
            for i in range(lo, hi):
                try:
                    go(i, i + 1)
                except (RuntimeError, ValueError) as e1:
                    torch.cuda.empty_cache()
                    print(f'{key} {rows[i].identifier}: {type(e1).__name__}: {str(e1)[:120]}', flush=True)
        if lo % (chunk * 50) == 0:
            print(f'{key}: {hi}/{len(rows)}', flush=True)
    return out


def stats(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    d = y - x
    rank = lambda v: np.argsort(np.argsort(v))
    return dict(n=int(ok.sum()), pearson=float(np.corrcoef(x, y)[0, 1]),
                spearman=float(np.corrcoef(rank(x), rank(y))[0, 1]), mean_diff=float(d.mean()),
                sd_diff=float(d.std()), mae=float(np.abs(d).mean()))


def plot(res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    u, a, m = (np.array(res[k], dtype=float) for k in ('uma_raw', 'uma_avg', 'mace'))
    chiral = np.array(res['sohncke'], dtype=bool)
    keep = np.isfinite(a) & np.isfinite(m) & (a < 0) & (m < 0)  # a positive lattice energy = a broken structure
    n_out = int((~keep).sum())
    u, a, m, chiral = u[keep], a[keep], m[keep], chiral[keep]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    lo = np.nanmin([np.nanmin(a), np.nanmin(m)]) - 10
    hi = np.nanmax([np.nanmax(a), np.nanmax(m)]) + 10
    for sel, c, lab in ((~chiral, '#185FA5', 'groups with an improper element'), (chiral, '#D85A30', 'Sohncke (chiral) groups')):
        ax[0].scatter(a[sel], m[sel], s=5, alpha=0.5, c=c, label=f'{lab} (n={int(sel.sum())})')
    ax[0].plot([lo, hi], [lo, hi], 'k-', lw=0.8)
    ax[0].set_xlabel('UMA-s 1.1 lattice energy, mirror-averaged in chiral groups (kJ/mol per molecule)')
    ax[0].set_ylabel('MACE-OFF24 lattice energy (kJ/mol per molecule)')
    ax[0].set_title(f'MACE-OFF24 vs UMA-s, CSD experimental geometries ({n_out} with E > 0 not shown)', fontsize=10)
    ax[0].legend(fontsize=8, loc='upper left')
    d = m - a
    ax[1].hist(d[np.isfinite(d)], bins=80, color='#534AB7')
    ax[1].set_xlabel('MACE-OFF24 - UMA-s lattice energy (kJ/mol per molecule)')
    ax[1].set_ylabel('crystals')
    ax[1].set_title('Difference (UMA-s carries a per-molecule gas-leg offset)')
    dm = (u - a)[chiral] * 2  # uma_raw - uma_avg = (R(x) - R(sigma x)) / 2
    ax[2].hist(dm[np.isfinite(dm)], bins=60, color='#D85A30')
    ax[2].set_xlabel('UMA-s R(x) - R(mirror x), raw, chiral groups only (kJ/mol per molecule)')
    ax[2].set_ylabel('crystals')
    ax[2].set_title('UMA-s mirror error at experimental structures')
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_atoms', type=int, default=60)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, 'csd_parity.json')):
        raise SystemExit(f'{a.out_dir} already holds a result; use a new directory')
    os.makedirs(a.out_dir, exist_ok=True)
    cap_gpu()
    rows, n_ok = pick(a.n, a.seed, a.max_atoms)
    print(f'{len(rows)} crystals drawn from {n_ok} eligible', flush=True)
    sohncke = [int(r.sg_ind) in SOHNCKE for r in rows]
    from mxtaltools.crystal_search.run_search import attach_mlip_parity
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
    # UMA-s: mirror-mean gas reference on every row; raw and (chiral rows only) mirror-averaged periodic energy
    raw = [strip(r) for r in rows]
    cfg = lambda avg: SimpleNamespace(mlip_gas_reference='mirror_mean', mlip_mirror_average=avg,
                                      opt=[{'optim_target': 'uma'}], uma_predictor_path=SITES['local']['uma'])
    attach_mlip_parity(cfg(False), raw, 'cuda')
    avg = []
    for r, s in zip(raw, sohncke):
        q = strip(r)
        q.uma_gas_pot = r.uma_gas_pot
        if s:
            q.uma_mirror_average = torch.tensor([True])
        avg.append(q)
    upred = init_uma_crystal_predictor(SITES['local']['uma'], device='cuda', activation_checkpointing=False, tf32=False)
    E_raw = energies(raw, 'uma', upred, chunk=4)
    idx = [i for i, s in enumerate(sohncke) if s]
    E_avg = E_raw.clone()
    if idx:
        E_avg[idx] = energies([avg[i] for i in idx], 'uma', upred, chunk=2)
    del upred
    torch.cuda.empty_cache()
    mpred = load_mace_model(SITES['local']['mace'], device='cuda', dtype=torch.float32)
    E_mace = energies([strip(r) for r in rows], 'mace', mpred, chunk=8)
    res = dict(identifier=[r.identifier for r in rows], sg=[int(r.sg_ind) for r in rows], sohncke=sohncke,
               num_atoms=[int(r.num_atoms) for r in rows], uma_raw=E_raw.tolist(), uma_avg=E_avg.tolist(),
               mace=E_mace.tolist(), n_eligible=n_ok)
    u, av, m = E_raw.numpy(), E_avg.numpy(), E_mace.numpy()
    ch = np.array(sohncke)
    res['stats'] = dict(mace_vs_uma_avg=stats(av, m), mace_vs_uma_avg_achiral=stats(av[~ch], m[~ch]),
                        mace_vs_uma_avg_chiral=stats(av[ch], m[ch]),
                        uma_mirror_error_chiral=dict(zip(('mean', 'sd'), (float(np.nanmean(2 * (u - av)[ch])),
                                                                          float(np.nanstd(2 * (u - av)[ch]))))))
    with open(os.path.join(a.out_dir, 'csd_parity.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    plot(res, os.path.join(a.out_dir, 'csd_parity.png'))
    print('PARITY', json.dumps(res['stats']), flush=True)


if __name__ == '__main__':
    main()
