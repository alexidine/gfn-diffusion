"""One environment-wise RDF distance matrix over a molecule's samples at every stage and its experimental forms.

Every structure, whatever its description, becomes a P1 crystal of Z rigid copies of the reference conformer
(``p1_relax.p1_crystal``), and its RDF is the repo's shared definition (``eval/nikos_comparison/levels.py``
``RDF_KWARGS``: ``envwise`` channels, 10 Angstrom cutoff, 100 bins). Environment labels come from the bonded
graph alone, so a P1 cell with Z independent molecules and a space-group description of the same crystal get
the same channels. Distances are ``crystal_rdf.compute_rdf_distance`` (mean per-channel earth mover's distance).

Structures, by kind:
- ``raw``: the generator's samples (``cifs/``), the reference fitted onto each molecule;
- ``brief``: the same after ``p1_brief`` (``brief<STEPS>_<energy>/``);
- ``relaxed``: the same after ``p1_relax`` (``relaxed_<energy>/``);
- ``exp_relaxed``: each experimental form relaxed in its space group (``relaxed_<energy>/exp_<refcode>.cif``);
- ``exp_csd``: each experimental form as deposited in the CSD (reference fitted);
- ``control``: the P1 control's perturbed start and relaxed end (``p1_control/``), when present.

usage (CPU, licensed CSD Python API):
    python -m eval.day_bench.rdf_embed SAMPLE_DIR REFCODE OTHER_FORM[,OTHER_FORM] [energy] [STEPS]

Writes SAMPLE_DIR/rdf_embed_<energy>.npz: ``dist`` [n, n], ``kind`` [n], ``name`` [n], ``rdf`` [n, channels, bins].
"""
import glob
import os
import sys
import warnings

import numpy as np
import torch
from ase.io import read

from eval.day_bench.p1_ingest import make_reference, split_molecules
from eval.day_bench.p1_relax import p1_crystal
from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat
from mxtaltools.dataset_utils.utils import collate_data_list

RDF_KWARGS = dict(rdf_mode='envwise', cutoff=10, rdf_cutoff=10, bins=100)  # levels.py RDF_KWARGS + RDF_BINS
CHUNK = 4  # crystals per RDF call on CPU (levels.py: envwise over 80 at once exhausts host RAM)


def load(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return read(path)


def rdfs(crystals, device='cpu', chunk=CHUNK):
    """[n, channels, bins] RDFs (on the CPU) and the bin positions, in chunks of one Z' each, computed on ``device``;
    raises if channel labels differ between chunks (they are batch-assigned integers)."""
    out, labels, bins = [None] * len(crystals), None, None
    by_zp = {}
    for i, c in enumerate(crystals):
        by_zp.setdefault(int(c.z_prime), []).append(i)
    for zp, idx in by_zp.items():
        for s in range(0, len(idx), chunk):
            part = idx[s:s + chunk]
            batch = collate_data_list([crystals[i] for i in part], max_z_prime=zp).to(device)
            with torch.no_grad():
                res = batch.analyze(['rdf'], **RDF_KWARGS)
            rdf, b, pairs = res['rdf']
            rdf, b = rdf.detach().cpu(), b.detach().cpu()
            key = tuple(sorted((str(k), str(v)) for k, v in pairs.items()))
            if labels is None:
                labels, bins = key, b.detach()
            assert key == labels, 'envwise channel labels differ between chunks'
            for j, i in enumerate(part):
                out[i] = rdf[j].detach()
    return torch.stack(out), bins


def main(sample_dir, refcode, other_forms=(), energy='exp6', steps=25):
    from ccdc import io as ccdc_io
    from eval.day_bench.csd_roundtrip import p1_cell
    torch.set_num_threads(2)
    reader = ccdc_io.EntryReader('CSD')
    atoms_ref, _, _ = p1_cell(refcode, reader)
    ref = make_reference(split_molecules(atoms_ref)[0], smiles=reader.entry(refcode).molecule.heaviest_component.smiles)

    entries = []  # (kind, name, ase.Atoms)
    for p in sorted(glob.glob(os.path.join(sample_dir, 'cifs', '*.cif'))):
        entries.append(('raw', os.path.basename(p), load(p)))
    for p in sorted(glob.glob(os.path.join(sample_dir, f'brief{steps}_{energy}', '*.cif'))):
        entries.append(('brief', os.path.basename(p), load(p)))
    for p in sorted(glob.glob(os.path.join(sample_dir, f'relaxed_{energy}', 'sample_*.cif'))):
        entries.append(('relaxed', os.path.basename(p), load(p)))
    for rc in (refcode, *other_forms):
        p = os.path.join(sample_dir, f'relaxed_{energy}', f'exp_{rc}.cif')
        if os.path.exists(p):
            entries.append(('exp_relaxed', rc, load(p)))
        entries.append(('exp_csd', rc, p1_cell(rc, reader)[0]))
    for name in ('start_perturbed', 'end_relaxed'):
        p = os.path.join(sample_dir, 'p1_control', f'{name}.cif')
        if os.path.exists(p):
            entries.append(('control', name, load(p)))

    kinds, names, crystals, failed = [], [], [], []
    for kind, name, atoms in entries:
        c, residual = p1_crystal(atoms, ref, identifier=f'{kind}:{name}')
        if c is None:
            failed.append(f'{kind}:{name} ({residual})')
            continue
        kinds.append(kind)
        names.append(name)
        crystals.append(c)
    print(f'{len(crystals)} structures built ({", ".join(f"{k} {kinds.count(k)}" for k in dict.fromkeys(kinds))}); '
          f'not built: {len(failed)}')
    for f in failed:
        print(f'  {f}')

    # description check: the deposited form as a P1 cell of Z rigid copies against its space-group description
    # (Z' = 1, from p1_ingest.ingest) must be one point in RDF space
    from eval.day_bench.p1_ingest import ingest
    atoms_csd = p1_cell(refcode, reader)[0]
    sg_form = ingest(atoms_csd, ref, identifier=f'sg:{refcode}')
    if sg_form.ok:
        pair, b = rdfs([p1_crystal(atoms_csd, ref)[0], sg_form.crystal])
        d = float(compute_rdf_distmat(pair, b, show_tqdm=False)[0, 1])
        print(f'description check, {refcode} as P1 (Z\' = {int(atoms_csd.get_global_number_of_atoms() / len(ref.mol.z))}) '
              f'vs its space-group description (sg {sg_form.sg}, Z\' = 1): RDF distance {d:.2e}')
    else:
        print(f'description check skipped: {refcode} did not convert ({sg_form.reason})')

    rdf, bins = rdfs(crystals)
    bad = ~torch.isfinite(rdf).reshape(len(rdf), -1).all(1)
    assert not bad.any(), f'non-finite RDF for {[f"{kinds[i]}:{names[i]}" for i in np.nonzero(bad.numpy())[0]]}'
    dist = compute_rdf_distmat(rdf, bins, show_tqdm=False).numpy()
    out = os.path.join(sample_dir, f'rdf_embed_{energy}.npz')
    np.savez_compressed(out, dist=dist, kind=np.array(kinds), name=np.array(names), rdf=rdf.numpy(),
                        bins=bins.numpy())
    print(f'RDF {tuple(rdf.shape)} ({RDF_KWARGS["rdf_mode"]}, cutoff {RDF_KWARGS["rdf_cutoff"]} A); '
          f'distance matrix {dist.shape} written to {out}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], tuple(sys.argv[3].split(',')) if len(sys.argv) > 3 and sys.argv[3] else (),
         sys.argv[4] if len(sys.argv) > 4 else 'exp6', int(sys.argv[5]) if len(sys.argv) > 5 else 25)
