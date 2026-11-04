"""Census of rigid C/H/N/O/F molecules in the local SMILES sources (read-only), keyed for the training pool.

A molecule is kept when it is one connected component, net-neutral and closed-shell, made only of C/H/N/O/F, has
2 or more heavy atoms, MW < 230 g/mol and at most one RDKit rotatable bond (CalcNumRotatableBonds). n_rot (0 or 1)
and rotor (a CH3 / OH / NH2 group present) are recorded, so the strict Day-like subset (n_rot 0, rotor 0) and the
rigid (n_rot 0) and one-torsion (n_rot 1) tiers can be cut afterwards.

Key: the non-standard InChIKey with relative stereochemistry and fixed hydrogens (InChI options -SRel -FixedH).
Mirror images share it (energies are mirror-invariant); tautomers, diastereomers and E/Z isomers do not, since the
conditioning reads 3D coordinates (owner 2026-11-03).

Charges (owner 2026-11-03, no stray charges): net charge 0, and every charged atom has an oppositely charged atom
within two bonds (nitro, N-oxide, azide, diazo, nitrone, furoxan). A separated zwitterion ([NH3+]...[O-], betaines,
mesoionics) is kept only from the CSD, where a crystal shows it. A molecule whose source SMILES leaves a stereocentre
unassigned gets its own key (n_stereo_unassigned > 0); training_pool_build.py enumerates it. ik14 is the standard
InChIKey's first block (constitution, any stereo, any tautomer), the key the Day exclusion uses. smiles is the first spelling met, with its stereo.

Sources under SRC: csd/csd_hcnof_unique.tsv (every C/H/N/O/F component of every entry of the local CSD release
as600be, canonicalised stereo-free; scan scripts in csd/provenance/), zinc20_2d_A and zinc20_2d_B (ZINC20 2D
tranches, MW bins A <= 200 and B 200-250), pubchem/CID-SMILES.gz.

usage, from energy_sampling/: python -m data_processing.training_pool_census SOURCE OUT_DIR [workers]
SOURCE in csd, zinc20A, zinc20B, pubchem. Writes OUT_DIR/<source>_v3.csv.gz and <source>_v3_counts.json.
"""
import collections
import glob
import gzip
import json
import os
import sys
from multiprocessing import Pool

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')
SRC = 'D:/crystal_datasets/smiles_sources'
CHARS = set('CNOFHcno()[]=#+-0123456789%@/\\.')
ALLOWED = {'C', 'H', 'N', 'O', 'F'}
ROTOR = None


def pre(s):
    return 0 < len(s) <= 90 and '.' not in s and all(c in CHARS for c in s)


def local_dipoles(m):
    """True when every charged atom has an oppositely charged atom within two bonds (or nothing is charged)."""
    ch = [(a.GetIdx(), a.GetFormalCharge()) for a in m.GetAtoms() if a.GetFormalCharge()]
    if not ch:
        return True
    D = Chem.GetDistanceMatrix(m)
    return all(any(q * q2 < 0 and D[i, j] <= 2 for j, q2 in ch) for i, q in ch)


ALLOW_SEPARATED = False  # set per source in main(): True for the CSD only


def judge(s):
    """(fields, 'ok') for a kept molecule, else (None, reason)."""
    global ROTOR
    if ROTOR is None:
        ROTOR = Chem.MolFromSmarts('[$([CH3]),$([OX2H1]),$([NX3H2])]')
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None, 'parse'
    if len(Chem.GetMolFrags(m)) != 1:
        return None, 'multi'
    if any(a.GetSymbol() not in ALLOWED for a in m.GetAtoms()):
        return None, 'elements'
    if Chem.GetFormalCharge(m) != 0 or any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
        return None, 'charge_radical'
    if not ALLOW_SEPARATED and not local_dipoles(m):
        return None, 'separated_zwitterion'
    nh = m.GetNumHeavyAtoms()
    if nh < 2:
        return None, 'tiny'
    mw = Descriptors.MolWt(m)
    if mw >= 230:
        return None, 'mw'
    nrot = rdMolDescriptors.CalcNumRotatableBonds(m)
    if nrot > 1:
        return None, 'rotatable_gt1'
    try:
        key = Chem.MolToInchiKey(m, options='-SRel -FixedH')
        ik14 = Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None, 'inchi'
    if not key:
        return None, 'inchi'
    centres = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        scaf = ''
    return dict(key=key, smiles=Chem.MolToSmiles(m), ik14=ik14, n_rot=nrot, rotor=int(m.HasSubstructMatch(ROTOR)),
                n_heavy=nh, mw=round(mw, 3), n_rings=rdMolDescriptors.CalcNumRings(m),
                n_arom_rings=rdMolDescriptors.CalcNumAromaticRings(m), n_stereo=len(centres),
                n_stereo_unassigned=sum(1 for _, t in centres if t == '?'), scaffold=scaf), 'ok'


def work(batch):
    out, cnt = [], collections.Counter()
    for s, extra in batch:
        r, why = judge(s)
        cnt[why if r is None else f"rot{r['n_rot']}_rotor{r['rotor']}"] += 1
        if r is not None:
            r['extra'] = extra
            out.append(r)
    return out, cnt


TOTAL = [0]


def _smi_files(pattern):
    for fn in sorted(glob.glob(pattern)):
        with open(fn) as f:
            f.readline()
            for ln in f:
                TOTAL[0] += 1
                s = ln.split(' ', 1)[0]
                if pre(s):
                    yield s, ''


def lines(source):
    """(smiles, extra) after the character prefilter; every input line is counted in TOTAL."""
    if source == 'csd':
        with open(f'{SRC}/csd/csd_hcnof_unique.tsv') as f:
            hdr = f.readline().rstrip('\n').split('\t')
            ix = {k: i for i, k in enumerate(hdr)}
            for ln in f:
                p = ln.rstrip('\n').split('\t')
                TOTAL[0] += 1
                if pre(p[0]):
                    yield p[0], f"{p[ix['first_refcode']]}|{p[ix['n_refcodes']]}|{p[ix['has_3d_any']]}"
    elif source == 'zinc20A':
        yield from _smi_files(f'{SRC}/zinc20_2d_A/*/*.smi')
    elif source == 'zinc20B':
        yield from _smi_files(f'{SRC}/zinc20_2d_B/*/*.smi')
    elif source == 'pubchem':
        with gzip.open(f'{SRC}/pubchem/CID-SMILES.gz', 'rt') as f:
            for ln in f:
                TOTAL[0] += 1
                s = ln.rstrip('\n').split('\t', 1)[-1]
                if pre(s):
                    yield s, ''
    else:
        raise SystemExit(f'unknown source {source}')


def batches(it, n=5000):
    b = []
    for x in it:
        b.append(x)
        if len(b) == n:
            yield b
            b = []
    if b:
        yield b


def _set_allow(v):
    global ALLOW_SEPARATED
    ALLOW_SEPARATED = v


def main():
    import pandas as pd
    source, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    global ALLOW_SEPARATED
    ALLOW_SEPARATED = source == 'csd'
    os.makedirs(out, exist_ok=True)
    rows, cnt = {}, collections.Counter()
    with Pool(workers, initializer=_set_allow, initargs=(ALLOW_SEPARATED,)) as p:
        for k, (o, c) in enumerate(p.imap_unordered(work, batches(lines(source)), chunksize=2)):
            cnt.update(c)
            for r in o:
                rows.setdefault(r['key'], r)
            if k % 500 == 0:
                print(source, k, TOTAL[0], len(rows), flush=True)
    df = pd.DataFrame(list(rows.values()))
    df.to_csv(f'{out}/{source}_v3.csv.gz', index=False, compression='gzip')
    json.dump(dict(input_lines=TOTAL[0], passed_prefilter=sum(cnt.values()), reasons=dict(cnt),
                   distinct_keys=len(df)), open(f'{out}/{source}_v3_counts.json', 'w'), indent=1)
    print(source, 'done', TOTAL[0], dict(cnt), len(df))


if __name__ == '__main__':
    main()
