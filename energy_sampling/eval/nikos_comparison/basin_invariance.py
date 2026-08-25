"""
Does the CONFORMER move the basin, or only the energy at the bottom of it?

The workflow assumes a rigid reference conformer is enough to map packing modes:
acridine is rigid, packing is set by large-scale rigid-body arrangement, so the
landscape over CRYSTAL PARAMETERS should be nearly conformer-invariant even though
per-atom detail and precise ranking are not. This tests that assumption instead of
asserting it.

DESIGN. `std_acridine_polymorphs.pt` and `std_opt_acridine_polymorphs.pt` hold the
same crystals with two different conformers and BIT-IDENTICAL cells and poses (the
script asserts this). Relax both with the same rigid-body optimiser and ask where
they land in crystal-parameter space:

    d_conf  = |p_final(conformer A) - p_final(conformer B)|
    d_move  = |p_final - p_start|,  averaged over the two runs

  d_conf << d_move  ->  both conformers fall into the SAME basin; the conformer
                        sets the depth, not the location. Assumption holds.
  d_conf ~  d_move  ->  the conformer relocates the minimum. Basin identity is not
                        transferable across conformers.

Measured in crystal parameters ON PURPOSE. RDF cannot answer this: it is built from
atom positions, so it registers a conformer swap as movement even when the cell and
poses are untouched (0.037-0.045 for exactly the pair used here, against a
clustering cut of 0.05). That sensitivity is real for cross-conformer COMPARISON
but says nothing about where the basin sits.

The old->opt conformer difference is the STRESS CASE: aromatic C-C 1.3668 vs
1.4027 A, larger than the experimental->opt difference of 1.388-1.394 vs 1.4027.

    python -m energy_sampling.eval.nikos_comparison.basin_invariance
"""
import argparse
import os

import torch

from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model

from energy_sampling.eval.nikos_comparison.summarize_search import ROOT

MACE_MODEL = r"D:\crystal_datasets\acr_112025_mh1_stagetwo.model"
#: stage-2 of the production search (acr_rerun_aug21/*.yaml), rigid-body on MACE
OPT = dict(optimizer_func='rprop', init_lr=0.01, max_num_steps=150,
           convergence_eps=1e-5, grad_norm_clip=0.1, optim_target='mace',
           cutoff=10, enforce_reduced=True, anneal_lr=True,
           compression_factor=0.0, target_packing_coeff=None, show_tqdm=False)
#: CPU MACE is the binding cost here -- 5 forms x 2 conformers x 150
#: steps did not finish in 15 min. Two forms answers the question; widen
#: with --names/--steps on a GPU.
NAMES = ['ACRDIN07', 'ACRDIN06']


def load_pair(names):
    out = []
    for f in ('std_acridine_polymorphs.pt', 'std_opt_acridine_polymorphs.pt'):
        b = torch.load(os.path.join(ROOT, f), weights_only=False,
                       map_location='cpu').cpu()
        lst, ids = b.batch_to_list(), list(b.identifier)
        out.append([lst[ids.index(n)] for n in names])
    return out


def relax(lst, pred, device):
    b = collate_data_list([c.clone() for c in lst],
                          exclude_keys=['rdf', 'fingerprint', 'rdf_bins']).to(device)
    b.aunit_handedness = b.aunit_handedness.abs()
    start = b.full_cell_parameters().clone().cpu()
    out = b.optimize_crystal_parameters(predictor=pred, **OPT)
    fb = collate_data_list(out).cpu()
    return start, fb.full_cell_parameters().cpu(), fb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--names', nargs='+', default=NAMES)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--target', default='mace',
                    help="energy the basin is defined by; 'lj' is "
                         'far cheaper but a DIFFERENT landscape')
    cli = ap.parse_args()

    names = cli.names
    OPT['max_num_steps'] = cli.steps
    OPT['optim_target'] = cli.target
    OPT['show_tqdm'] = True
    print(f'relaxing {len(names)} forms x 2 conformers, '
          f'{cli.steps} steps, target {cli.target}')
    old, new = load_pair(names)
    for i, n in enumerate(names):
        dl = float((old[i].cell_lengths - new[i].cell_lengths).abs().max())
        da = float((old[i].cell_angles - new[i].cell_angles).abs().max())
        dc = float((old[i].aunit_centroid - new[i].aunit_centroid).abs().max())
        if max(dl, da, dc) != 0.0:
            raise SystemExit(f"{n}: the two files differ by more than the "
                             f"conformer (cell {dl}, angle {da}, centroid {dc}); "
                             f"this test would not isolate the conformer")
    print(f"verified: cells and poses bit-identical across conformers "
          f"for {len(names)} forms\n")

    pred = load_mace_model(MACE_MODEL, cli.device, torch.float32)
    s_o, f_o, b_o = relax(old, pred, cli.device)
    s_n, f_n, b_n = relax(new, pred, cli.device)

    print(f"{'form':12s} {'d_conf':>8} {'d_move':>8} {'ratio':>7}   "
          f"{'dA(len)':>8} {'dA(ang deg)':>12}")
    for i, n in enumerate(names):
        d_conf = float((f_o[i] - f_n[i]).norm())
        d_move = float(((f_o[i] - s_o[i]).norm() + (f_n[i] - s_n[i]).norm()) / 2)
        dlen = float((f_o[i, :3] - f_n[i, :3]).abs().max())
        dang = float((f_o[i, 3:6] - f_n[i, 3:6]).abs().max()) * 57.2958
        print(f"{n:12s} {d_conf:8.4f} {d_move:8.4f} "
              f"{d_conf / max(d_move, 1e-9):7.2f}   {dlen:8.4f} {dang:12.3f}")

    #: THE RATIO ABOVE IS A WEAK TEST -- d_move is how far THIS relaxation
    #: happened to travel, which depends on how close the start already was to
    #: a minimum, not on the size of the basin. These structures start at their
    #: experimental geometry, so d_move is small and the ratio inflates.
    #:
    #: The operational question is whether the relaxed structure is STILL THE
    #: SAME FORM, and COMPACK is the project's arbiter for that. If both
    #: conformers relax to something still matching the original at 20/20, the
    #: conformer set the DEPTH and not the LOCATION.
    from energy_sampling.eval.nikos_comparison.compare import compack_confirm
    ref = []
    ol, nl2 = b_o.batch_to_list(), b_n.batch_to_list()
    for i2 in range(len(names)):
        ref += [ol[i2], nl2[i2]]
    ref_b = collate_data_list([c.clone() for c in ref])
    q = collate_data_list([c.clone() for c in new])
    q.aunit_handedness = q.aunit_handedness.abs()
    nb = torch.tensor([[2 * i2, 2 * i2 + 1] for i2 in range(len(names))])
    work = os.path.join(ROOT, 'nikos_comparison', '_basin_compack')
    rmsds, matched = compack_confirm(q, ref_b, nb, work)
    print()
    print('relaxed structure vs the ORIGINAL form it started from:')
    print(f"{'form':12s} {'conformer':>12} {'matched':>9} {'rmsd':>8}")
    for i2, n in enumerate(names):
        for j2, tag in enumerate(('old', 'opt')):
            m, r = int(matched[i2, j2]), float(rmsds[i2, j2])
            note = '  <- comparison FAILED' if (m == 0 and r == 0.0) else ''
            print(f'{n:12s} {tag:>12} {m:6d}/20 {r:8.3f}{note}')
    print()
    print('both rows 20/20 -> relaxation under either conformer stays in the '
          'same basin; the packing mode is conformer-transferable.')
    print(f"\nratio << 1 -> the two conformers relax into the SAME basin and the "
          f"conformer sets DEPTH, not LOCATION.")
    print(f"ratio ~ 1  -> the conformer relocates the minimum.")


if __name__ == '__main__':
    main()
