"""
Stage B: put Nikos' structures onto our energy surface, at three levels.

    L0  as-given     his atoms, his cell, exactly as ingested.
    L1  reprojected  OUR reference conformer, rigid, at his pose and cell.
    L2  relaxed      L1 relaxed on our MACE surface (cell + pose; molecule rigid).

L1 is the bridge, not an extra: our landscapes are rigid-body searches over one
fixed acridine conformer, so a structure has to be expressed in those terms
before it can be relaxed in them or compared against them. The L0->L1 RDF
distance is therefore a GUARD, not a result -- it measures how much of his
structure survived the swap to our molecule. A structure whose L0->L1 distance is
large did not come through, and its L1/L2 numbers describe something else.

L2 is the basin-correspondence test: if his structure relaxes into one of our
minima, that is the landmark correspondence, whether or not the unrelaxed
structures looked alike.

An all-atom relaxation (L3) is expected from a colleague; `--write-cifs` exports
the L1 structures for it.

Run (after ingest.py):
    python -m energy_sampling.eval.nikos_comparison.levels --config config.yaml
"""
import argparse
import os

import torch

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.common.adaptive_batching import adaptive_batched_analysis
from mxtaltools.dataset_utils.data_classes import MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list

#: One definition of the RDF, shared by every structure this sub-project
#: compares. These MUST be identical on both sides of any distance -- an RDF
#: computed under different settings still subtracts cleanly and still returns a
#: small-looking number, so a mismatch here would not announce itself.
RDF_KWARGS = dict(rdf_mode='envwise', cutoff=10, rdf_cutoff=10)
RDF_BINS = 100

#: Rigid-body relaxation on the MACE surface. rprop per the settings already used
#: for acridine in eval/paper1_results/analysis.py.
OPT_CONFIG = dict(
    enforce_reduced=False,
    compression_factor=0.0,
    cutoff=10,
    init_lr=1e-4,
    convergence_eps=1e-8,
    optimizer_func='rprop',
    anneal_lr=False,
    grad_norm_clip=0.1,
    show_tqdm=True,
    max_num_steps=500,
    rdf_warmup=None,
    target_packing_coeff=None,
    umbrella=False,
)


def searched_combos(cfg):
    """(sg_ind, z_prime) -> prior name, from config. See config.yaml `searched`."""
    return {(int(r['sg']), int(r['z_prime'])): r['prior'] for r in cfg['searched']}


def restrict(crystals, cfg):
    """
    Keep only structures in an SG/Z' combination our searches actually cover.

    Outside those we never looked, so nothing there can be evidence either way --
    and relaxing them at L2 would spend the expensive step on structures whose
    result could not be interpreted. What is dropped is named, not silently
    filtered.
    """
    combos = searched_combos(cfg)
    kept = [c for c in crystals if (int(c.sg_ind), int(c.z_prime)) in combos]
    dropped = [c for c in crystals if (int(c.sg_ind), int(c.z_prime)) not in combos]
    from collections import Counter
    print(f"restricted to searched SG/Z': keeping {len(kept)} of {len(crystals)}")
    for combo, n in sorted(Counter((int(c.sg_ind), int(c.z_prime))
                                   for c in kept).items()):
        print(f"   kept    sg={combo[0]:3d} Z'={combo[1]}  {n:3d}  "
              f"-> {combos[combo]}")
    for combo, n in sorted(Counter((int(c.sg_ind), int(c.z_prime))
                                   for c in dropped).items()):
        print(f"   dropped sg={combo[0]:3d} Z'={combo[1]}  {n:3d}  (never searched)")
    if not kept:
        raise ValueError("no structures fall in a searched SG/Z' combination")
    return kept


def rdf_bins(like, device=None):
    return torch.linspace(0, 10, like.rdf.shape[-1], device=device or like.rdf.device)


def analyze(batch, energy_function, predictor, device, chunk=None):
    """
    Energy + RDF, under the one shared RDF definition, in bounded chunks.

    The envwise RDF over all 80 of his structures at once exhausts host RAM
    ("RuntimeError: bad allocation" inside get_atomwise_dists), so the work is
    chunked. `adaptive_batched_analysis` only recovers from CUDA OOM -- its
    handler calls torch.cuda.empty_cache() and re-raises anything else -- so on
    CPU the chunk size must be CAPPED rather than left to grow into the same
    wall. `max_batch_size` is what caps it.
    """
    chunk = chunk or (4 if str(device) == 'cpu' else 100)
    with torch.no_grad():
        out = adaptive_batched_analysis(
            batch, analyses=[energy_function, 'rdf', 'vdw_max'], state={},
            initial_batch_size=chunk, max_batch_size=chunk,
            predictor=predictor, device=device, show_tqdm=True, **RDF_KWARGS)
    return out.to('cpu')


def build_l0(crystals, energy_function, predictor, device, chunk=None):
    """His structures as ingested."""
    batch = collate_data_list(crystals)
    return analyze(batch, energy_function, predictor, device, chunk)


def standardize(batch):
    """
    spglib standard setting, standard symmetry operators.

    Required before L1: reprojection rebuilds the crystal from `sg_ind` using
    STANDARD symmetry operators, so a structure sitting in a nonstandard setting
    (his P21/n files) would otherwise be rebuilt as a different crystal. Returns
    the per-structure success flag from `confirm_transform`, which checks that
    the RDF survived the transform and that the result is cell-reduced.
    """
    std, ok = batch.clone().compute_standard_cell(confirm_transform=True)
    return std, ok


def build_l1(std_batch, conformer, max_z_prime):
    """
    OUR reference conformer, rigid, at his pose.

    `aunit_handedness.abs()` follows the convention in process_target.py and
    collate_prior.py: acridine is planar, so the sign of its inertial frame is
    ambiguous and the same physical structure can carry either. Forcing it makes
    the latent parameters single-valued. Whether that stayed faithful is not
    assumed -- the caller measures L0->L1 RDF distance.
    """
    std_batch = std_batch.clone()
    std_batch.aunit_handedness = std_batch.aunit_handedness.abs()

    ones3, ones1 = torch.ones(3), torch.ones(1)
    samples = []
    for zp, sg in zip(std_batch.z_prime.tolist(), std_batch.sg_ind.tolist()):
        samples.append(MolCrystalData(
            molecule=[conformer.clone() for _ in range(zp)] if zp > 1 else conformer.clone(),
            sg_ind=sg,
            aunit_handedness=ones1, cell_lengths=ones3, cell_angles=ones3,
            aunit_centroid=ones3, aunit_orientation=ones3,
            skip_box_analysis=True, max_z_prime=max_z_prime, z_prime=zp,
            do_box_analysis=True,
        ))
    batch = collate_data_list(samples)
    batch.latent_to_cell_params(std_batch.latent_params())
    batch.identifier = list(std_batch.identifier)
    return batch


def build_l2(l1_batch, energy_function, predictor, device, opt_config=None):
    """Rigid-body relaxation of L1 on our energy surface."""
    cfg = dict(OPT_CONFIG)
    cfg.update(opt_config or {})
    cfg['optim_target'] = energy_function
    cfg['predictor'] = predictor
    batch = l1_batch.clone().to(device)
    opt_out, record = batch.optimize_crystal_parameters(return_record=True, **cfg)
    out = collate_data_list(opt_out)
    out.identifier = list(l1_batch.identifier)
    return out, record


def rdf_gap(a, b):
    """Per-structure RDF distance between two aligned batches."""
    bins = torch.linspace(0, 10, a.rdf.shape[-1])
    return torch.stack([
        compute_rdf_distance(a.rdf[i], b.rdf[i][None, ...], bins).flatten()[0]
        for i in range(a.num_graphs)])


def main():
    from energy_sampling.utils import load_yaml
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=os.path.join(os.path.dirname(__file__),
                                                     'config.yaml'))
    ap.add_argument('--write-cifs', action='store_true',
                    help='export L1 unit cells for an all-atom relaxation elsewhere')
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--device', default=None,
                    help="override config device, e.g. 'cpu'")
    ap.add_argument('--skip-l2', action='store_true',
                    help='stop after L1; L2 is the expensive stage')
    ap.add_argument('--limit', type=int, default=None,
                    help='use only the first N structures (smoke run)')
    ap.add_argument('--all-space-groups', action='store_true',
                    help="override config restrict_to_searched and keep every "
                         "structure, including SG/Z' we never searched")
    ap.add_argument('--chunk', type=int, default=None,
                    help='structures per analysis chunk (default 4 on cpu)')
    ap.add_argument('--out', default=None,
                    help='output filename (default nikos_levels.pt). Use a '
                         'distinct name for an --all-space-groups run so it does '
                         'not overwrite the restricted analysis artifact')
    cli = ap.parse_args()

    cfg = load_yaml(cli.config)
    device = cli.device or cfg['device']
    ef = cfg['energy_function']

    if ef == 'mace':
        from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
        predictor = load_mace_model(cfg['mace_model'], device, torch.float32)
    else:
        raise NotImplementedError(f"energy_function {ef!r}: only 'mace' is wired up, "
                                  f"and it is the function the acridine landscapes "
                                  f"were built with")

    blob = torch.load(os.path.join(cfg['out_dir'], 'nikos_structures.pt'),
                      weights_only=False)
    crystals, manifest = blob['crystals'], blob['manifest']
    if cfg.get('restrict_to_searched') and not cli.all_space_groups:
        crystals = restrict(crystals, cfg)
    if cli.limit:
        crystals = crystals[:cli.limit]
        print(f"LIMIT: {len(crystals)} structures only -- this is a smoke run, "
              f"not the result")
    conformer = torch.load(cfg['conformer'], weights_only=False)
    if isinstance(conformer, list):
        conformer = conformer[0]
    max_zp = max(int(c.z_prime) for c in crystals)

    print(f"L0: scoring {len(crystals)} structures as given")
    l0 = build_l0(crystals, ef, predictor, device, cli.chunk)

    print("standardizing cells")
    std, std_ok = standardize(l0)
    n_bad = int((~std_ok).sum())
    if n_bad:
        bad = [l0.identifier[i] for i in (~std_ok).argwhere().flatten().tolist()]
        print(f"  {n_bad} failed standardization (RDF not preserved or cell not "
              f"reduced): {bad}")

    print("L1: reprojecting onto our reference conformer")
    l1 = build_l1(std, conformer, max_zp)
    l1 = analyze(l1, ef, predictor, device, cli.chunk)

    gap = rdf_gap(l0, l1)
    print(f"  L0->L1 RDF distance: median {gap.median():.4f}, "
          f"90th pct {gap.quantile(0.9):.4f}, max {gap.max():.4f}")
    print(f"  L0->L1 energy shift ({ef}, kJ/mol/molecule): "
          f"median {(l1[ef] - l0[ef]).median():.2f}, "
          f"max |{(l1[ef] - l0[ef]).abs().max():.2f}|")

    l2 = moved = None
    if cli.skip_l2:
        print("L2: SKIPPED (--skip-l2)")
    else:
        print("L2: rigid-body relaxation on our surface")
        opt_over = {'max_num_steps': cli.max_steps} if cli.max_steps else {}
        l2, record = build_l2(l1, ef, predictor, device, opt_over)
        l2 = analyze(l2, ef, predictor, device, cli.chunk)
        drop = l1[ef] - l2[ef]
        print(f"  relaxation lowered {ef} by median {drop.median():.2f} kJ/mol, "
              f"max {drop.max():.2f}, min {drop.min():.2f}")
        moved = rdf_gap(l1, l2)
        print(f"  L1->L2 RDF distance: median {moved.median():.4f}, "
              f"max {moved.max():.4f}")

    out = os.path.join(cfg['out_dir'],
                       cli.out or ('nikos_levels.pt' if not cli.limit
                                   else f'nikos_levels_limit{cli.limit}.pt'))
    torch.save({'l0': l0, 'l1': l1, 'l2': l2, 'std_ok': std_ok,
                'l0_l1_rdf_gap': gap, 'l1_l2_rdf_gap': moved,
                'manifest': manifest, 'energy_function': ef,
                'restricted_to_searched': bool(cfg.get('restrict_to_searched')
                                               and not cli.all_space_groups),
                'rdf_kwargs': RDF_KWARGS, 'opt_config': OPT_CONFIG}, out)
    print(f"\nwrote {out}")

    if cli.write_cifs:
        cif_dir = os.path.join(cfg['out_dir'], 'l1_cifs')
        os.makedirs(cif_dir, exist_ok=True)
        cwd = os.getcwd()
        try:
            os.chdir(cif_dir)
            b = l1.clone()
            b.mol2ucell()
            b.write_cif(torch.arange(b.num_graphs), 'nikos_l1', mode='unit cell')
        finally:
            os.chdir(cwd)
        print(f"wrote L1 unit cells to {cif_dir}")


if __name__ == '__main__':
    main()
