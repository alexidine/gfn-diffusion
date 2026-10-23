"""Image-aware thinning of an old-style prior file (anchors in 'prior', noised rows in 'equalized_prior'), the pre-pass
before data_processing.pool_assemble_mace.

    python image_thin.py PRIOR.pt OUT_DIR --radius R [--ceiling_kT 10] [--mol MOL.pt]

pool_assemble_mace thins candidates on their canonical latents only. Its anchors in pool_acr_oct03 were registry basins,
one description per crystal, so that was enough. An old prior's anchors are not: the same crystal sits there under
several of its normaliser images (acridine_sg14_zp1_mace_prior_dataset_w3.pt: 74% of anchors have another anchor within
0.0242 of one of their images), and the fold would then write each such crystal once per description.

Candidates in pool_assemble_mace's order (anchors in ascending energy, then noised rows in ascending energy), after its
ceiling; a candidate is dropped when one of its 8 normaliser images (Chart.images, the rows the fold writes) lies within
R of one of the images of an already kept candidate (Euclidean in the trainer latent, periodic coordinates wrapped).
Images against images, not the stored row against images: about 1% of an old prior's rows (face states, rotation vectors
longer than pi) are not among their own images, which re-describe them canonically. Writes OUT_DIR/anchors.pt
({'prior': kept anchors}) and OUT_DIR/walk/shard_0.pt (kept noised rows: cell_params [n, 12], E [n]), the inputs
pool_assemble_mace takes; its own thinning at the same R, on the stored rows, still runs after this. The density and
latent-box filters are left to pool_assemble_mace.
"""
import argparse
import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from data_processing.pool_anchors import Chart
from data_processing.pool_assemble import periodic_dims, rows_on, wrapped

KEY = 'mace'
KT = 2.494


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('prior')
    ap.add_argument('out_dir')
    ap.add_argument('--radius', type=float, required=True)
    ap.add_argument('--ceiling_kT', type=float, default=10.0)
    ap.add_argument('--sg', type=int, default=14)
    ap.add_argument('--mol', default='D:/crystal_datasets/acridine/acridine_conformer.pt')
    ap.add_argument('--chunk', type=int, default=2000)
    a = ap.parse_args(argv)
    t0 = time.time()
    d = torch.load(a.prior, weights_only=False)
    A, N = d['prior'], d['equalized_prior']
    eA, eN = A[KEY].double().flatten(), N[KEY].double().flatten()
    top = float(eA.min()) + a.ceiling_kT * KT
    okA, okN = torch.nonzero(eA <= top).flatten(), torch.nonzero(eN <= top).flatten()
    print(f'ceiling {a.ceiling_kT:g} kT ({top:.3f}): {len(okA)} of {len(eA)} anchors, {len(okN)} of {len(eN)} noised rows',
          flush=True)
    cp = torch.cat([A.full_cell_parameters().float()[okA], N.full_cell_parameters().float()[okN]])
    E = torch.cat([eA[okA], eN[okN]])
    nA, n = len(okA), len(cp)
    order = np.concatenate([np.argsort(E[:nA].numpy(), kind='stable'),
                            nA + np.argsort(E[nA:].numpy(), kind='stable')])

    ch = Chart(a.mol, a.sg, KEY, 'cpu')
    ch.setup(rows_on(A, cp[:1]))
    lati, src = [], []
    for lo in range(0, n, a.chunk):
        b, s, _ = ch.images(rows_on(A, cp[lo:lo + a.chunk]))
        lati.append(b.latent_params().double())
        src.append(s + lo)
    lati, src = torch.cat(lati), torch.cat(src).numpy()
    print(f'{n} candidates, {len(src)} image rows ({time.time() - t0:.0f} s)', flush=True)

    per = periodic_dims(a.sg)
    xi, box = wrapped(lati, per)
    tree = cKDTree(xi, boxsize=box)
    pairs = tree.query_pairs(a.radius, output_type='ndarray')
    d = np.abs(xi[pairs[:, 0]] - xi[pairs[:, 1]])
    d = np.minimum(d, box - d)
    pairs = pairs[np.sqrt((d * d).sum(1)) <= a.radius]
    sp = src[pairs]
    sp = np.unique(np.sort(sp[sp[:, 0] != sp[:, 1]], axis=1), axis=0)  # candidate pairs with images within R
    near = [[] for _ in range(n)]
    for i, j in sp:
        near[i].append(j)
        near[j].append(i)
    print(f'{len(sp)} candidate pairs with images within {a.radius:g}', flush=True)
    state = np.zeros(n, dtype=np.int8)  # 0 undecided, 1 kept, -1 dropped
    for i in order:
        if state[i]:
            continue
        state[i] = 1
        for j in near[i]:
            if state[j] == 0:
                state[j] = -1
    kept = np.nonzero(state == 1)[0]
    kA, kN = kept[kept < nA], kept[kept >= nA] - nA
    print(f'image-aware thinning at radius {a.radius:g} (periodic dims {per}): kept {len(kA)} of {nA} anchors and '
          f'{len(kN)} of {n - nA} noised rows ({time.time() - t0:.0f} s)', flush=True)

    os.makedirs(os.path.join(a.out_dir, 'walk'), exist_ok=True)
    anchors = A.subsample_new_batch(okA[torch.from_numpy(np.sort(kA))])
    torch.save({'prior': anchors, 'source': os.path.basename(a.prior), 'radius': a.radius,
                'ceiling_kT': a.ceiling_kT}, os.path.join(a.out_dir, 'anchors.pt'))
    kN = np.sort(kN)
    torch.save({'cell_params': cp[nA:][kN].numpy(), 'E': E[nA:][kN].numpy()},
               os.path.join(a.out_dir, 'walk', 'shard_0.pt'))
    print(f'wrote {a.out_dir}/anchors.pt ({anchors.num_graphs} rows) and walk/shard_0.pt ({len(kN)} rows)')


if __name__ == '__main__':
    main()
