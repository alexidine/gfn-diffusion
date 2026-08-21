"""
Stage A: read Nikos' acridine CIFs into our MolCrystalData format.

Uses the same ccdc-backed helpers that built our own crystal datasets, so the
parameterization and the Z'>1 conformer alignment are identical to the ones our
brute-force samples went through. It does NOT call `process_chunk`, for one
reason, measured on his 80 stage-3 structures:

    `crystal_rebuild_checks` rejects a rebuild whose re-derived aunit_handedness
    or aunit_orientation differs from the one it was given, and tests that BEFORE
    it tests any geometry. Acridine is planar, so reflection through the
    molecular plane maps the molecule onto itself and the inertial frame's sign
    is ambiguous; the round trip flips it for 57 of his 80 files. All 80 rebuild
    the unit cell to <= 2e-4 A and pass `validate_cell_params`. The label flip is
    a relabelling of an exact rebuild, not a failed one.

So acceptance here is GEOMETRIC: every structure must rebuild its own unit cell,
and the residual is recorded per structure rather than assumed. The label flip is
recorded as `mirror_flip`.

The flip is benign for everything downstream (RDF, MACE energy, COMPACK, and
rigid-body relaxation all start from the stored parameters, which rebuild
correctly). It is NOT benign for comparison in cell-parameter/latent space, where
one physical structure can carry either sign -- hence `aunit_handedness.abs()`
in `process_target.py` and `collate_prior.py`.

Two further properties of his files, asserted here:

  * `_symmetry_Int_Tables_number` is 1 in every file regardless of the actual
    group. ccdc reads the H-M symbol instead and gets the group right, but that
    is a property of the reader, not of the file -- so the space group implied by
    the containing folder is cross-checked against what came back.
  * `P21n` is setting 2 of space group 14, and ingests as sg_ind 14 with
    `nonstandard_symmetry` set.

Run:
    python -m energy_sampling.eval.nikos_comparison.ingest --config config.yaml
"""
import argparse
import glob
import os
import re
from collections import Counter

import numpy as np
import torch
import tqdm

from mxtaltools.constants.space_group_info import SPACE_GROUPS, SYM_OPS
from mxtaltools.dataset_utils.construction.featurization_utils import (
    check_zp_mol_alignment, cocrystal_check, crystal_filter, extract_zp1_pose_info,
    init_zp1_crystals, instantiate_crystal, overwrite_conformer_to_zp)
from mxtaltools.dataset_utils.construction.featurize_cif_chunks import (
    extract_crystal_info, init_reader)
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

#: How exactly a rebuilt unit cell must reproduce the one in the CIF, in Angstrom.
#: The same numbers `crystal_rebuild_checks` uses for its own position gate.
REBUILD_MAX_TOL = 5e-2
REBUILD_MEAN_TOL = 1e-2

#: Folder labels Nikos uses -> space group number. His labels are our symbols
#: with the '/' dropped; the exceptions are listed explicitly. A label in neither
#: is an ERROR, not a skip -- silently dropping an unrecognised group would read
#: downstream as "we searched there and found nothing".
_SG_BY_LABEL = {sym.replace('/', ''): num for num, sym in SPACE_GROUPS.items()}
_SG_BY_LABEL.update({
    'P21n': 14,  # P21/n, setting 2 of P21/c
})

#: acridine_P21c_rank0002_score111548.938718_cell26_var0_a90.00_b96.41_g90.00
#: Stage-1 files carry no rank/score:
#: acridine_P21c_cell10_var0_350_a90.00_b95.19_g90.00
_RANKED = re.compile(
    r'^acridine_(?P<sg>[^_]+)_rank(?P<rank>\d+)_score(?P<score>[-\d.eE+]+)'
    r'_cell(?P<cell>\d+)_var(?P<var>\d+)')
_UNRANKED = re.compile(
    r'^acridine_(?P<sg>[^_]+)_cell(?P<cell>\d+)_var(?P<var>\d+)_(?P<serial>\d+)')
#: the accepted_structures pool, which carries no rank or score at all:
#: struct_000053_cell38_gp0.3926_0.4295_0.0996
_STRUCT = re.compile(r'^struct_(?P<serial>\d+)_cell(?P<cell>\d+)')

#: 'Z_prime_2' (the staged export) and 'zprime_2' (the accepted pool)
_ZP_DIR = re.compile(r'^z_?prime_?(?P<zp>\d+)$', re.IGNORECASE)


def parse_provenance(root: str, path: str) -> dict:
    """
    Pull Z', stage, space group and any ranking out of the path.

    Handles both layouts he has sent:
        Z_prime_2/cif_exports/stage3/P21c/Unique/acridine_..._rank0002_score...
        zprime_2/Cc/struct_000009_cell31_gp...
    so the directory depth is not assumed -- the Z' and space-group components
    are identified by what they are, not by position.
    """
    rel = os.path.relpath(path, root).replace('\\', '/')
    parts = rel.split('/')
    name = os.path.basename(path)[:-len('.cif')]

    zps = [m.group('zp') for m in (_ZP_DIR.match(p) for p in parts) if m]
    if not zps:
        raise KeyError(f"no Z' directory (z_prime_N / zprimeN) in {rel!r}")
    stages = [p for p in parts if p.lower().startswith('stage')]

    #: the space group is the directory holding the file, unless that is the
    #: 'Unique' subfolder, in which case it is one level up.
    sg_part = parts[-2] if len(parts) >= 2 else None
    if sg_part == 'Unique' and len(parts) >= 3:
        sg_part = parts[-3]

    rec = {
        'rel_path': rel,
        'file_name': name,
        'zp_dir': int(zps[0]),
        'stage': int(stages[0].lower().replace('stage', '')) if stages else None,
        'sg_label': sg_part,
        'unique': 'Unique' in parts,
        'rank': None, 'nikos_score': None, 'cell_id': None, 'var': None,
    }

    label = rec['sg_label'].rstrip('_')
    if label not in _SG_BY_LABEL:
        raise KeyError(f"unrecognised space group folder {rec['sg_label']!r} in {rel}")
    rec['sg_from_label'] = _SG_BY_LABEL[label]

    #: His 'score' is not one quantity -- the Cc and Cc_ folders carry values
    #: three orders of magnitude apart (~1e5 vs ~5-8) -- so it is kept purely as
    #: provenance. Everything comparable is recomputed on our energy function.
    m = _RANKED.match(name) or _UNRANKED.match(name) or _STRUCT.match(name)
    if m is not None:
        g = m.groupdict()
        rec['cell_id'] = int(g['cell'])
        if 'var' in g:
            rec['var'] = int(g['var'])
        if 'rank' in g:
            rec['rank'] = int(g['rank'])
            rec['nikos_score'] = float(g['score'])
    return rec


def collect(root: str, pattern: str) -> list:
    """Every CIF matching `pattern` under `root`, with its provenance."""
    paths = sorted(glob.glob(os.path.join(root, pattern.replace('/', os.sep)),
                             recursive=True))
    if not paths:
        raise FileNotFoundError(f"no CIFs matched {pattern!r} under {root}")
    return [parse_provenance(root, p) for p in paths]


def read_one(cif_path: str, identifier: str, max_z_prime: int,
             protonation_state: str = 'protonated'):
    """
    One CIF -> (crystal, diagnostics). `crystal` is None if it could not be read.

    Mirrors `featurize_cif_chunks.process_chunk` step for step, and differs only
    in the acceptance criterion -- see the module docstring.
    """
    diag = {'rebuild_nn_max': None, 'rebuild_nn_mean': None, 'mirror_flip': None,
            'nonstandard_symmetry': None, 'reject': None}

    reader = io_crystal_reader(cif_path)
    if reader is None:
        diag['reject'] = 'reader failed'
        return None, diag
    csd_crystal, reduced_crystal = init_reader(0, protonation_state, reader)

    passed, unit_cell, rd_mols, failure_mode = crystal_filter(
        csd_crystal, reduced_crystal, max_heavy_atoms=100,
        protonation_state=protonation_state, max_atomic_number=100,
        max_z_prime=max_z_prime)
    if failure_mode is not None:
        diag['reject'] = f'filter: {failure_mode}'
        return None, diag

    crystal_dict, molecules = extract_crystal_info(
        cif_path, csd_crystal, identifier, protonation_state, rd_mols,
        reduced_crystal, unit_cell)

    try:
        sym_ops_are_standard = bool(np.all(
            np.stack(SYM_OPS[crystal_dict['space_group_number']])
            == np.stack(crystal_dict['symmetry_operators'])))
    except ValueError:
        diag['reject'] = 'incommensurate symmetry multiplicity'
        return None, diag
    diag['nonstandard_symmetry'] = not sym_ops_are_standard
    sym_ops = np.stack(crystal_dict['symmetry_operators'])

    mols = [MolData(z=torch.tensor(m['atom_atomic_numbers'], dtype=torch.long),
                    pos=torch.tensor(m['atom_coordinates'], dtype=torch.float32),
                    x=torch.tensor(m['atom_partial_charge'], dtype=torch.float32),
                    fingerprint=torch.tensor(m['molecule_fingerprint'],
                                             dtype=torch.float32)[None, ...],
                    smiles=m['molecule_smiles'], do_mol_analysis=True)
            for m in molecules]
    if len(mols) > 1 and not cocrystal_check(mols):
        diag['reject'] = 'non-identical cocrystals unsupported'
        return None, diag

    z_prime = torch.tensor(int(crystal_dict['z_prime']), dtype=torch.long)
    zp1_crystals = init_zp1_crystals(crystal_dict, mols, sym_ops, sym_ops_are_standard)
    (aunit_centroid, aunit_handedness, aunit_orientation,
     crystal_batch, is_well_defined, pos) = extract_zp1_pose_info(
        crystal_dict['unit_cell_coordinates'], z_prime, zp1_crystals)

    for ind, fc in enumerate(zp1_crystals):
        fc.aunit_centroid[:, :3] = aunit_centroid[ind][None, ...]
        fc.aunit_orientation[:, :3] = aunit_orientation[ind][None, ...]
        fc.aunit_handedness[:, :1] = aunit_handedness[ind]
        fc.is_well_defined = torch.tensor(is_well_defined, dtype=torch.bool)[ind]
        fc.pos = pos[crystal_batch.batch == ind]
        fc.box_analysis()

    rebuild_batch = collate_data_list(zp1_crystals, max_z_prime=1)
    rebuild_batch.pose_aunit()
    rebuild_batch.build_unit_cell()
    re_centroid, re_orientation, re_handedness, re_well_defined, _ = \
        rebuild_batch.reparameterize_unit_cell()

    #: The label flip. Recorded, not rejected -- the geometry below is the judge.
    diag['mirror_flip'] = bool(
        not torch.all(rebuild_batch.aunit_handedness.flatten() == re_handedness.flatten())
        or not torch.all(torch.isclose(rebuild_batch.aunit_orientation,
                                       re_orientation, rtol=1e-2)))

    #: THE acceptance test: does the parameterization we are about to store
    #: rebuild the unit cell that was in the file?
    uc_build = rebuild_batch.unit_cell_pos
    uc_orig = torch.tensor(np.concatenate(crystal_dict['unit_cell_coordinates']),
                           dtype=torch.float32)
    T_fc = rebuild_batch.T_fc[0]
    T_cf = torch.linalg.inv(T_fc)
    df = (uc_orig @ T_cf.T)[:, None, :] - (uc_build @ T_cf.T)[None, :, :]
    df -= torch.round(df)  # minimum image
    distmat = torch.linalg.norm(df @ T_fc.T, dim=-1)
    nearest = torch.argmin(distmat, dim=1)
    nn_dists = distmat[torch.arange(len(distmat)), nearest]
    diag['rebuild_nn_max'] = float(nn_dists.amax())
    diag['rebuild_nn_mean'] = float(nn_dists.mean())

    if len(torch.unique(nearest)) != len(nearest):
        diag['reject'] = 'rebuilt atoms do not match the file one-to-one'
        return None, diag
    if not (diag['rebuild_nn_max'] < REBUILD_MAX_TOL
            and diag['rebuild_nn_mean'] < REBUILD_MEAN_TOL):
        diag['reject'] = (f"rebuild off by {diag['rebuild_nn_max']:.3f} A "
                          f"(mean {diag['rebuild_nn_mean']:.3f})")
        return None, diag
    try:
        rebuild_batch.validate_cell_params(check_crystal_system=True)
    except AssertionError as e:
        diag['reject'] = f'invalid cell params: {str(e)[:80]}'
        return None, diag

    if z_prime > 1:
        # enforce identical atom ordering and conformers across the Z' images
        aligned = overwrite_conformer_to_zp(mols, rebuild_batch, z_prime)
        if not check_zp_mol_alignment(aligned, rebuild_batch):
            diag['reject'] = "Z'>1 image writing failed"
            return None, diag
        rebuild_batch = aligned

    crystal = instantiate_crystal(re_centroid, re_handedness, re_orientation,
                                  max_z_prime, mols, rebuild_batch, z_prime)
    crystal.identifier = identifier
    return crystal, diag


def io_crystal_reader(cif_path):
    from ccdc import io
    try:
        return io.CrystalReader(cif_path, format='cif')
    except RuntimeError:  # refine_bonds timeout
        return None


def ingest(nikos_root: str, nikos_glob: str, max_z_prime: int = 3,
           protonation_state: str = 'protonated', reference_mol=None):
    """CIFs -> (list of MolCrystalData, manifest keyed by our short key)."""
    records = collect(nikos_root, nikos_glob)
    print(f"matched {len(records)} CIFs under {nikos_root}")
    for k, n in sorted(Counter((r['zp_dir'], r['sg_label']) for r in records).items()):
        print(f"   Z'={k[0]}  {k[1]:<10s} {n}")

    manifest, crystals = {}, []
    for i, rec in enumerate(tqdm.tqdm(records)):
        rec['key'] = key = f'nik{i:05d}'
        manifest[key] = rec
        path = os.path.join(nikos_root, rec['rel_path'].replace('/', os.sep))
        try:
            crystal, diag = read_one(path, key, max_z_prime, protonation_state)
        except Exception as e:  # noqa: BLE001 - one bad file must not sink the batch
            crystal, diag = None, {'reject': f'{type(e).__name__}: {str(e)[:80]}',
                                   'rebuild_nn_max': None, 'rebuild_nn_mean': None,
                                   'mirror_flip': None, 'nonstandard_symmetry': None}
        rec.update(diag)
        if crystal is not None:
            crystals.append(crystal)

    _audit(crystals, manifest)
    if reference_mol is not None:
        _check_composition(crystals, reference_mol)
    return crystals, manifest


def _check_composition(crystals: list, reference_mol) -> None:
    """
    His molecule must BE our molecule, atom for atom.

    `protonation_state='deprotonated'` runs Chem.RemoveAllHs, which turns
    acridine into C13N. Nothing downstream announces that: the envwise RDF simply
    builds a different number of channels (36 instead of 91, from 8 WL labels
    instead of 13), and every comparison against our landscapes then fails or --
    worse, if the counts had happened to agree -- silently compares different
    descriptors. Checked here, at the point where it is fixable.
    """
    from collections import Counter
    want = Counter(reference_mol.z.long().tolist())
    bad = []
    for c in crystals:
        zp = int(c.z_prime)
        got = Counter(c.z.long().tolist())
        expected = Counter({k: v * zp for k, v in want.items()})
        if got != expected:
            bad.append((c.identifier, dict(got), dict(expected)))
    if bad:
        raise AssertionError(
            f"{len(bad)} of {len(crystals)} ingested molecules do not match the "
            f"reference conformer's composition, e.g. {bad[:2]} "
            f"(key, got, expected). Check `protonation_state`: 'deprotonated' "
            f"strips all hydrogens.")
    n = int(reference_mol.num_atoms)
    print(f"composition matches the reference conformer "
          f"({n} atoms per molecule) for all {len(crystals)} structures")


def _audit(crystals: list, manifest: dict) -> None:
    """
    Judge what came back, including what did NOT.

    A rejected CIF is indistinguishable downstream from a structure we matched to
    nothing, so every rejection is named here with its reason.
    """
    got = {c.identifier for c in crystals}
    lost = [r for r in manifest.values() if r['key'] not in got]
    print(f"\ningested {len(crystals)} / {len(manifest)} structures")
    if lost:
        print(f"REJECTED {len(lost)}:")
        for r in lost:
            print(f"   {r['key']}  {r['rel_path']}\n        {r['reject']}")

    bad_sg, bad_zp = [], []
    for c in crystals:
        rec = manifest[c.identifier]
        if int(c.sg_ind) != rec['sg_from_label']:
            bad_sg.append((c.identifier, rec['sg_label'], int(c.sg_ind)))
        if int(c.z_prime) != rec['zp_dir']:
            bad_zp.append((c.identifier, rec['zp_dir'], int(c.z_prime)))
    if bad_sg:
        raise AssertionError(
            f"space group disagrees with the containing folder for {len(bad_sg)} "
            f"structures, e.g. {bad_sg[:3]} (key, folder label, ingested sg_ind)")
    if bad_zp:
        raise AssertionError(
            f"Z' disagrees with the containing folder for {len(bad_zp)} structures, "
            f"e.g. {bad_zp[:3]} (key, folder Z', ingested z_prime)")
    print("space group and Z' agree with the folder layout for all ingested structures")

    res = [manifest[c.identifier]['rebuild_nn_max'] for c in crystals]
    print(f"unit-cell rebuild residual: max {max(res):.2e} A, "
          f"mean {sum(res) / len(res):.2e} A  (tolerance {REBUILD_MAX_TOL:.0e})")

    flips = [c.identifier for c in crystals if manifest[c.identifier]['mirror_flip']]
    print(f"{len(flips)} / {len(crystals)} carry a planar-mirror label flip "
          f"(handedness/orientation); geometry is unaffected, "
          f"cell-parameter-space comparison would not be")
    nonstd = [c.identifier for c in crystals
              if manifest[c.identifier]['nonstandard_symmetry']]
    print(f"{len(nonstd)} / {len(crystals)} are in a nonstandard setting (e.g. P21/n)")


def main():
    from energy_sampling.utils import load_yaml
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=os.path.join(os.path.dirname(__file__),
                                                     'config.yaml'))
    ap.add_argument('--max-z-prime', type=int, default=3)
    ap.add_argument('--protonation-state', default='protonated',
                    choices=['protonated', 'deprotonated'],
                    help="'deprotonated' strips ALL hydrogens and makes his "
                         "molecules incomparable to our reference conformer")
    cli = ap.parse_args()

    cfg = load_yaml(cli.config)
    os.makedirs(cfg['out_dir'], exist_ok=True)
    reference_mol = torch.load(cfg['conformer'], weights_only=False,
                               map_location='cpu')
    if isinstance(reference_mol, list):
        reference_mol = reference_mol[0]
    crystals, manifest = ingest(cfg['nikos_root'], cfg['nikos_glob'],
                                cli.max_z_prime, cli.protonation_state,
                                reference_mol)
    out = os.path.join(cfg['out_dir'], 'nikos_structures.pt')
    torch.save({'crystals': crystals, 'manifest': manifest,
                'protonation_state': cli.protonation_state,
                'nikos_root': cfg['nikos_root'], 'nikos_glob': cfg['nikos_glob']}, out)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
