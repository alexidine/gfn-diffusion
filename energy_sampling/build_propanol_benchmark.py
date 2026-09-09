"""Build the paired propanol benchmark: does the conditional machinery cost anything?

THE QUESTION. The conditional route adds an encoder embedding, a per-DoF correlator, a
condition-dependent log Z head and a set policy. Any of that could cost accuracy or sample
efficiency even when there is nothing to condition ON. So: hold the target fixed and vary
only the architecture.

    UNCONDITIONAL   one propanol target.
    CONDITIONAL     N conditions, every one the SAME propanol -- same graph, same
                    stereochemistry, same parameterisation -- with its real embedding.

Repeating one molecule removes molecular diversity as an explanation for any difference. What
remains is the machinery.

THE CHART IS BUILT ONCE AND COPIED, not rebuilt per condition. Rebuilding would re-run an
ETKDG embedding per copy, and any drift in the reference conformer would make the two arms
differ in their TARGET as well as their architecture -- so the benchmark would be measuring
chart variation and calling it conditioning. `verify` below asserts every copy is bitwise
identical on every chart field, rather than trusting determinism.

THE PRIOR STATES ARE DRAWN ONCE and written into both files. The two arms therefore see the
same data, not merely data from the same distribution; the files differ only in which
identifier each row carries, which is what `init_identifiers` needs to mint `mol_id`.

FORCE FIELD IS PASSED EXPLICITLY. `build_conformer_conditions.py` does not forward it, so its
charts are built under the constructor default ('reference') while runs use 'mmff'. Those two
route linearity differently, so a molecule with a linear centre can get one chart in the file
and another in the run. Propanol has no linear centre and is unaffected either way, but the
benchmark states its force field rather than inheriting a default that disagrees with the run.

    python build_propanol_benchmark.py --out-dir <dir> \
        --encoder-ckpt models/results/encoder_ckpt/mp+attn+spd_n20000_s0.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from energies.conformer_data import (CTREE_ATOM_FIELDS, CTREE_GRAPH_FIELDS, attach_states,
                                     bake_energies, check_state_convention, collate_conditions,
                                     condition_from_energy, save_condition_file,
                                     save_prior_file)
from energies.conformer_torsions import ConformerTorsions

SMILES = 'CCCO'                     # propanol: 12 atoms, 3N-6 = 30, no linear centre


def verify_identical(mols, ident):
    """Every condition must carry the SAME chart, bitwise. Asserted, not assumed."""
    ref = mols[0]
    for i, m in enumerate(mols[1:], start=1):
        for f in CTREE_ATOM_FIELDS + CTREE_GRAPH_FIELDS:
            a, b = getattr(ref, f, None), getattr(m, f, None)
            if a is None and b is None:
                continue
            assert a is not None and b is not None, f'{f} present on one copy only'
            assert torch.equal(torch.as_tensor(a), torch.as_tensor(b)), (
                f'condition {ident[i]} differs from {ident[0]} on {f}: the copies do not '
                f'share a chart, so the benchmark would be measuring chart variation')
    print(f'   VERIFIED: {len(mols)} conditions are bitwise identical on all '
          f'{len(CTREE_ATOM_FIELDS + CTREE_GRAPH_FIELDS)} chart fields')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--n-conditions', type=int, default=32,
                    help='how many duplicate propanol conditions the conditional arm sees')
    ap.add_argument('--n-prior', type=int, default=4000,
                    help='prior rows TOTAL, shared bitwise by both arms')
    ap.add_argument('--internal-prior', type=Path, default=Path('conformer_prior_v2.pt'))
    ap.add_argument('--fatten', type=float, default=0.15)
    ap.add_argument('--encoder-ckpt', type=Path, default=None)
    ap.add_argument('--level', default='full')
    ap.add_argument('--force-field', default='mmff')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- one chart
    energy = ConformerTorsions(smiles=SMILES, device='cpu', level=args.level,
                               force_field=args.force_field, seed=args.seed)
    print(energy.describe())
    n_dof = 3 * int(energy.spec.n_atoms) - 6
    assert energy.data_ndim == n_dof and not getattr(energy, 'constrained_rows', 0), (
        f'{SMILES} at {args.level!r} is not a complete chart: d={energy.data_ndim} '
        f'against 3N-6={n_dof}')
    print(f'\n   complete chart: d = {energy.data_ndim} = 3N-6, '
          f'{energy.constrained_rows} held')

    base = condition_from_energy(energy, identifier=SMILES)
    err = check_state_convention(base, energy)
    print(f'   state convention: graph and energy agree to {err:.2e} A')
    assert err < 1e-9

    # ---------------------------------------------------------------- embeddings
    if args.encoder_ckpt is not None:
        from models import encoder_cache
        bundle = encoder_cache.load_encoder(str(args.encoder_ckpt), device='cpu')
        h, g, _ = encoder_cache.embed(bundle, SMILES, perm=energy.spec.perm,
                                      z_tree=energy.spec.z)
        base.embedding = g[None, :].to(torch.get_default_dtype())
        base.atom_embedding = h.to(torch.get_default_dtype())
        print(f'   encoder {bundle["arm"]} @ {bundle["sha256"][:12]}  '
              f'embedding_conditioning_dim: {2 * bundle["hidden"]}')

        # dof_atoms / dof_mask: WHICH ATOMS each state column moves, for the policy's
        # per-DoF correlator. Not optional on the conditional route -- ConformerGFN refuses a
        # batch without them -- and flattened to one row per graph because MolData classifies
        # a tensor by matching dim 0 to the node or graph count, and a [k, R, frame] field
        # matches neither, so it would be taken for shared metadata.
        from energies.dof_features import free_dof_atom_index
        a, msk = free_dof_atom_index(energy)
        base.dof_atoms = torch.as_tensor(a).reshape(1, -1)
        base.dof_mask = torch.as_tensor(msk).reshape(1, -1)
        print(f'   DoF atom frames: k = {a.shape[0]}, R = {a.shape[1]}, '
              f'frame = {a.shape[2]}')

    # ---------------------------------------------------------------- the copies
    idents = [f'propanol_{i:03d}' for i in range(args.n_conditions)]
    mols = []
    for ident in idents:
        m = base.clone()
        m.identifier = ident
        mols.append(m)
    verify_identical(mols, idents)
    cond = collate_conditions(mols)
    save_condition_file(cond, args.out_dir / 'propanol_cond_conditions.pt')
    print(f'   wrote conditions -> propanol_cond_conditions.pt '
          f'({cond.num_graphs} graphs, k = {energy.data_ndim})')

    # ---------------------------------------------------------------- one prior draw
    # NOT `build_conformer_conditions.draw_prior_states`: that routes a fitted prior through
    # `build_prior_states.draw_states`, which is the TORSION-tier path -- it returns one
    # column per rotatable bond (1 for propanol) rather than the 30 this chart drives, and
    # the shape mismatch only surfaces two layers later inside the bounding term, as an
    # index-out-of-bounds with no mention of the prior. At `flex` and above the fitted draw
    # is the energy's own `sample_prior_states`, which adds the joint sibling torsions,
    # thermal r/theta and ring blocks the wider tiers need.
    print(f'\nprior: {args.n_prior} states, drawn ONCE and shared bitwise by both arms')
    rng = np.random.default_rng(args.seed)
    if args.internal_prior is not None and Path(args.internal_prior).exists():
        from build_prior_states import fit_or_load
        fitted = fit_or_load(Path(args.internal_prior), [], args.fatten)
        states, stats = energy.sample_prior_states(fitted, args.n_prior, rng, report=False)
        states = states.to(energy.dtype)
        clip = stats.get('clip_frac', {})
        print('   fitted InternalPrior; clip_frac '
              + ' '.join(f'{k}={v:.4f}' for k, v in clip.items()))
        # a clipped row sits exactly ON the wall, which is a finite-mass point no continuous
        # density can express -- tolerable in small numbers, not as a silent majority
        assert all(v < 0.05 for v in clip.values()), f'clip_frac {clip} -- box too narrow'
    else:
        print(f'   no fitted InternalPrior at {args.internal_prior}; UNIFORM on the box')
        states = torch.as_tensor(rng.uniform(-1.0, 1.0, (args.n_prior, energy.data_ndim)),
                                 dtype=energy.dtype)
    assert states.shape == (args.n_prior, energy.data_ndim), states.shape
    e = bake_energies(energy, states)
    print(f'   E median {e.median():+8.3f}  p10 {torch.quantile(e, 0.1):+8.3f}'
          f'  p90 {torch.quantile(e, 0.9):+8.3f}  kcal/mol')

    # UNCONDITIONAL: one identifier over every row.
    uncond = attach_states(base, states, e, identifier=SMILES,
                           periodic=energy.periodic_dims)
    save_prior_file(uncond, args.out_dir / 'propanol_uncond_prior.pt',
                    source=('InternalPrior' if args.internal_prior else 'uniform'),
                    n_per_molecule=args.n_prior)

    # CONDITIONAL: the SAME states, dealt round-robin across the duplicate identifiers so
    # every condition owns an equal share. Round-robin rather than contiguous blocks so a
    # truncated or partially-consumed buffer still covers every condition evenly.
    parts = []
    for i, ident in enumerate(idents):
        sel = torch.arange(i, states.shape[0], len(idents))
        if not sel.numel():
            continue
        parts.append(attach_states(mols[i], states[sel], e[sel], identifier=ident,
                                   periodic=energy.periodic_dims))
    cprior = parts[0]
    for part in parts[1:]:
        cprior = cprior.append_batch(part)
    save_prior_file(cprior, args.out_dir / 'propanol_cond_prior.pt',
                    source=('InternalPrior' if args.internal_prior else 'uniform'),
                    n_per_molecule=args.n_prior // len(idents))

    # ---------------------------------------------------------------- same data, proven
    a = uncond.torsion_state.reshape(-1, energy.data_ndim)
    b = cprior.torsion_state.reshape(-1, energy.data_ndim)
    assert a.shape == b.shape, (a.shape, b.shape)
    order = torch.cat([torch.arange(i, states.shape[0], len(idents))
                       for i in range(len(idents))])
    assert torch.equal(a[order], b), 'the two arms were given different prior states'
    print(f'\n   VERIFIED: both prior files hold the SAME {a.shape[0]} states, bitwise '
          f'(the conditional file is a permutation, one block per condition)')
    print(f'   wrote prior  -> propanol_uncond_prior.pt ({uncond.num_graphs} rows, '
          f'1 identifier)')
    print(f'   wrote prior  -> propanol_cond_prior.pt   ({cprior.num_graphs} rows, '
          f'{len(idents)} identifiers)')


if __name__ == '__main__':
    main()
