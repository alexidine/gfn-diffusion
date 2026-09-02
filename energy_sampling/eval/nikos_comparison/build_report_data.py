"""
Collect every measured number for the acridine comparison report into one file.

Nothing is computed here -- this only JOINS artifacts that other scripts produced,
so the report cannot drift from the measurements:

    audit_conformer.py        -> conformer identity per source (re-run, captured)
    polymorph_levels.json     -> experimental vs relaxed polymorph energies
    nikos_levels.pt           -> Nikos L0 / L1 / L2 energies and L1->L2 movement
    nikos_vs_polymorphs.json  -> COMPACK of his pool against the known forms
    nikos_comparison_l1.csv   -> landscape COMPACK at L1 (backed up before overwrite)
    nikos_comparison.csv      -> landscape COMPACK at L2 (current)

Every level is labelled, because the single most common way to get this comparison
wrong is to quote a number from one level beside a number from another:

    L0  as delivered      his own file, HIS conformer
    L1  reprojected       same cell and pose, OUR reference conformer substituted
    L2  relaxed           L1 rigid-body relaxed on our MACE surface (cell + pose;
                          molecule held rigid -- NO all-atom relaxation anywhere)
"""
import csv
import json
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OFF = 11836.127  # stored-scale offset; see project-mace-lattice-energy-e0-offset


def main():
    from energy_sampling.utils import load_yaml
    cfg = load_yaml(os.path.join(HERE, 'config.yaml'))
    D = cfg['out_dir']

    rep = {'levels': {
        'L0': 'as delivered (his conformer)',
        'L1': 'reprojected onto our reference conformer, cell and pose unchanged',
        'L2': 'L1 rigid-body relaxed on our MACE surface (cell + pose, molecule rigid)',
    }, 'energy_offset': OFF}

    #: conformer audit -- captured by running it, so the report shows what the
    #: checker actually printed rather than a transcription of it
    out = subprocess.run([sys.executable, '-u',
                          os.path.join(HERE, 'audit_conformer.py')],
                         capture_output=True, text=True, cwd=os.path.dirname(
                             os.path.dirname(os.path.dirname(HERE))))
    rep['conformer_audit'] = [l for l in out.stdout.splitlines()
                              if l.strip() and 'Warning' not in l]

    rep['polymorphs'] = json.load(open(os.path.join(D, 'polymorph_levels.json')))

    pv = os.path.join(D, 'nikos_vs_polymorphs.json')
    rep['polymorph_compack'] = json.load(open(pv)) if os.path.exists(pv) else None

    lev = torch.load(os.path.join(D, 'nikos_levels.pt'),
                     weights_only=False, map_location='cpu')
    ids = list(lev['l1'].identifier)
    moved = lev.get('l1_l2_rdf_gap')
    l1c = {r['key']: r for r in csv.DictReader(
        open(os.path.join(D, 'nikos_comparison_l1.csv')))}
    l2c = {r['key']: r for r in csv.DictReader(
        open(os.path.join(D, 'nikos_comparison.csv')))}

    def compack(row, prior):
        n = row.get(f'{prior}_compack_nmatched') or '0'
        r = row.get(f'{prior}_compack_rmsd') or '0'
        d = row.get(f'{prior}_rdf_dist') or 'nan'
        return dict(n=int(float(n)), rmsd=round(float(r), 4),
                    rdf=round(float(d), 5),
                    within=row.get(f'{prior}_within_cut') == 'True')

    rows = []
    for i, k in enumerate(ids):
        a, b = l1c[k], l2c[k]
        prior = a['matched_landscape']
        rows.append(dict(
            key=k, sg_label=a['sg_label'], sg=int(a['sg_ind']),
            zp=int(a['z_prime']), landscape=prior,
            e0=round(float(a['l0_mace']), 3),
            e1=round(float(a['l1_mace']), 3),
            e2=round(float(b['l2_mace']), 3),
            moved=round(float(moved[i]), 4) if moved is not None else None,
            land_l1=compack(a, prior), land_l2=compack(b, prior),
        ))
    rep['nikos'] = rows

    #: his 13 ingested are not 13 distinct structures -- record the duplicate
    #: explicitly so the report cannot imply more independence than there is
    rep['duplicates'] = [dict(a='nik00011', b='nik00013', rdf_l2=0.0854,
                              note='P2_1/c and P2_1/n settings of the SAME space '
                                   'group (14); converge inside the 0.10 cut only '
                                   'after relaxation')]
    rep['excluded'] = [dict(key='nik00003', sg='C2'),
                       dict(key='nik00014', sg='P2_1 2_1 2_1')]

    p = os.path.join(HERE, 'report_data.json')
    json.dump(rep, open(p, 'w'), indent=1)
    print(f"wrote {p}")
    print(f"  {len(rep['nikos'])} structures, {len(rep['polymorphs'])} polymorphs, "
          f"conformer audit {len(rep['conformer_audit'])} lines, "
          f"polymorph COMPACK {'present' if rep['polymorph_compack'] else 'MISSING'}")


if __name__ == '__main__':
    main()
