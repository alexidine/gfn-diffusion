"""
A pool of small-molecule SMILES in the QM9 regime (H, C, N, O, F; 4-9 heavy atoms by default), from SMILES sources
of any size, with every molecule's source and parent statistics kept so the mixture can be chosen afterwards.

Each input line gives a SMILES (and an identifier, ignored). A line is kept when its characters can only spell
H/C/N/O/F molecules (a cheap prefilter before RDKit), then each connected component is kept when it is H/C/N/O/F
only, net-neutral and closed-shell. A component inside the heavy-atom window enters the pool as itself ("direct"). A
larger component, up to --max-parent heavy atoms, is a PARENT: it is cut into rigid pieces at every cuttable bond and
every connected union of pieces inside the window enters the pool as a fragment, capped with hydrogens where bonds
were cut. A parent is not pared down once: it gives every fragment it holds, up to --max-per-parent (a seeded
subsample of its fragments when it holds more, so that no large scaffold dominates).

A cuttable bond is a single bond in no ring whose two sides each hold more than one atom counting hydrogens: so a
terminal F (or any terminal heavy atom with no H) is never cut off, while CH3, NH2 and OH groups are. Ring bonds and
double and triple bonds are never cut. The Mo3ENet paring rule (mxtaltools mol_building.pare_molecule_skeleton) cut any
bridge of the H-explicit graph whose smaller side has more than one atom, double-bond bridges included, and only ever
removed the smaller side; this module keeps either side and cuts single bonds only.

Molecules are compared as canonical SMILES without stereochemistry. Outputs, per source, in --out:
  <source>/chunks/: one TSV per input chunk; <source>/pool.tsv after --reduce: smiles, heavy atoms, n_direct (input
  components equal to it), n_parents (parents that gave it), min_parent_heavy, plus a header. Nothing is overwritten.

    python -m data_processing.smiles_pool --source zinc20A --files D:/.../zinc20_2d_A/*/*.smi --out D:/.../pool
    python -m data_processing.smiles_pool --source zinc20A --out D:/.../pool --reduce
"""
import argparse
import glob
import gzip
import hashlib
import os
import random
import sys
from collections import defaultdict
from multiprocessing import Pool

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

ALLOWED_Z = {1, 6, 7, 8, 9}
# characters that can appear in a SMILES of an H/C/N/O/F molecule (organic subset, brackets, charges, H counts,
# bonds, ring closures, stereo marks, component separator); any other character (Cl, Br, S, P, metals, ...) rejects
PREFILTER_CHARS = set('CNOFHcno()[]=#+-0123456789%@/\\.')


def prefilter(smi):
    return bool(smi) and all(ch in PREFILTER_CHARS for ch in smi)


def components(smi):
    """The H/C/N/O/F, net-neutral, closed-shell connected components of a SMILES, as heavy-atom molecules without
    stereochemistry; [] when RDKit cannot read it."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    out = []
    for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True):
        if any(a.GetAtomicNum() not in ALLOWED_Z for a in frag.GetAtoms()):
            continue
        if sum(a.GetFormalCharge() for a in frag.GetAtoms()) != 0:
            continue
        if any(a.GetNumRadicalElectrons() for a in frag.GetAtoms()):
            continue
        frag = Chem.RemoveHs(frag)
        Chem.RemoveStereochemistry(frag)
        out.append(frag)
    return out


def canon(mol):
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def cuttable_bonds(mol):
    """Indices of the bonds this module may cut: single, in no ring, each side holding more than one atom counting
    hydrogens."""
    n = mol.GetNumAtoms()
    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        adj[b.GetBeginAtomIdx()].append((b.GetEndAtomIdx(), b.GetIdx()))
        adj[b.GetEndAtomIdx()].append((b.GetBeginAtomIdx(), b.GetIdx()))
    out = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        ok = True
        for start, other in ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()), (b.GetEndAtomIdx(), b.GetBeginAtomIdx())):
            seen, stack = {start}, [start]
            while stack:
                x = stack.pop()
                for y, bi in adj[x]:
                    if bi == b.GetIdx() or y in seen:
                        continue
                    seen.add(y)
                    stack.append(y)
            if len(seen) + sum(mol.GetAtomWithIdx(i).GetTotalNumHs() for i in seen) <= 1:
                ok = False
        if ok:
            out.append(b.GetIdx())
    return out


def pieces(mol, cut):
    """The rigid pieces left when the bonds `cut` are removed: (list of atom-index sets, piece adjacency as a list of
    (piece i, piece j, atom in i, atom in j) per cut bond)."""
    n = mol.GetNumAtoms()
    cut = set(cut)
    piece_of = [-1] * n
    sets = []
    for s in range(n):
        if piece_of[s] >= 0:
            continue
        k = len(sets)
        piece_of[s] = k
        members, stack = {s}, [s]
        while stack:
            x = stack.pop()
            for b in mol.GetAtomWithIdx(x).GetBonds():
                if b.GetIdx() in cut:
                    continue
                y = b.GetOtherAtomIdx(x)
                if piece_of[y] < 0:
                    piece_of[y] = k
                    members.add(y)
                    stack.append(y)
        sets.append(members)
    links = []
    for bi in cut:
        b = mol.GetBondWithIdx(bi)
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        links.append((piece_of[a], piece_of[c], a, c))
    return sets, links


def connected_unions(sizes, links, lo, hi):
    """Every connected set of pieces whose heavy-atom total is in [lo, hi], as frozensets of piece indices."""
    nbr = defaultdict(set)
    for i, j, _, _ in links:
        nbr[i].add(j)
        nbr[j].add(i)
    seen, out = set(), []
    frontier = [frozenset([i]) for i in range(len(sizes)) if sizes[i] <= hi]
    seen.update(frontier)
    while frontier:
        nxt = []
        for s in frontier:
            w = sum(sizes[i] for i in s)
            if w >= lo:
                out.append(s)
            for i in s:
                for j in nbr[i]:
                    if j in s:
                        continue
                    t = s | {j}
                    if t in seen or w + sizes[j] > hi:
                        continue
                    seen.add(t)
                    nxt.append(t)
        frontier = nxt
    return out


def fragment_smiles(mol, atoms):
    """The fragment made of `atoms`, every bond to an atom outside it cut and capped with one H; canonical SMILES
    without stereochemistry, or None when the capped fragment does not sanitise."""
    keep = set(atoms)
    rw = Chem.RWMol(mol)
    for a in atoms:
        at = rw.GetAtomWithIdx(a)
        lost = sum(1 for nb in at.GetNeighbors() if nb.GetIdx() not in keep)
        if lost:
            at.SetNumExplicitHs(at.GetNumExplicitHs() + lost)
    for i in sorted(set(range(mol.GetNumAtoms())) - keep, reverse=True):
        rw.RemoveAtom(i)
    frag = rw.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except Exception:
        return None
    if sum(a.GetFormalCharge() for a in frag.GetAtoms()) != 0 or any(a.GetNumRadicalElectrons() for a in frag.GetAtoms()):
        return None
    return canon(frag)


def fragments(mol, lo=4, hi=9, max_per_parent=32, seed_key=''):
    """The distinct capped fragments of a parent inside [lo, hi] heavy atoms, at most max_per_parent of them (a
    subsample seeded by seed_key when it holds more)."""
    sets, links = pieces(mol, cuttable_bonds(mol))
    sizes = [len(s) for s in sets]
    found = set()
    for union in connected_unions(sizes, links, lo, hi):
        smi = fragment_smiles(mol, [a for i in union for a in sets[i]])
        if smi is not None:
            found.add(smi)
    found = sorted(found)
    if len(found) > max_per_parent:
        rng = random.Random(int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16))
        found = sorted(rng.sample(found, max_per_parent))
    return found


def heavy_atoms_of(smi):
    return Chem.MolFromSmiles(smi).GetNumHeavyAtoms()


def process_lines(lines, lo, hi, max_parent, max_per_parent, smiles_col=0):
    """smiles -> [n_direct, n_parents, min_parent_heavy] over one chunk of input lines (whitespace-separated, the
    SMILES in column smiles_col), and the chunk's counters."""
    agg = {}
    stats = defaultdict(int)
    for line in lines:
        cols = line.split()
        smi = cols[smiles_col] if len(cols) > smiles_col else ''
        stats['lines'] += 1
        if not prefilter(smi):
            continue
        stats['prefilter_kept'] += 1
        for comp in components(smi):
            n = comp.GetNumHeavyAtoms()
            if lo <= n <= hi:
                stats['direct'] += 1
                rec = agg.setdefault(canon(comp), [0, 0, 10 ** 6])
                rec[0] += 1
            elif hi < n <= max_parent:
                stats['parents'] += 1
                key = canon(comp)
                frs = fragments(comp, lo, hi, max_per_parent, key)
                stats['fragments'] += len(frs)
                for f in frs:
                    rec = agg.setdefault(f, [0, 0, 10 ** 6])
                    rec[1] += 1
                    rec[2] = min(rec[2], n)
            elif n > max_parent:
                stats['too_large'] += 1
    return agg, dict(stats)


def _work(job):
    path_out, lines, lo, hi, max_parent, max_per_parent, smiles_col = job
    agg, stats = process_lines(lines, lo, hi, max_parent, max_per_parent, smiles_col)
    tmp = path_out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for smi, (nd, npar, mph) in agg.items():
            f.write(f'{smi}\t{nd}\t{npar}\t{mph}\n')
    os.replace(tmp, path_out)
    return stats


def iter_lines(files, skip_header):
    for p in files:
        opener = gzip.open if p.endswith('.gz') else open
        with opener(p, 'rt', encoding='utf-8', errors='replace') as f:
            first = True
            for line in f:
                if first and skip_header and line.lower().startswith('smiles'):
                    first = False
                    continue
                first = False
                yield line


def build(args):
    out = os.path.join(args.out, args.source, 'chunks')
    os.makedirs(out, exist_ok=True)
    files = sorted(f for pat in args.files for f in glob.glob(pat))
    if not files:
        raise SystemExit(f'no input files match {args.files}')

    def jobs():
        buf, k = [], 0
        for line in iter_lines(files, args.skip_header):
            buf.append(line)
            if len(buf) >= args.chunk:
                yield (os.path.join(out, f'chunk{k:06d}.tsv'), buf, args.lo, args.hi, args.max_parent,
                       args.max_per_parent, args.smiles_col)
                buf, k = [], k + 1
        if buf:
            yield (os.path.join(out, f'chunk{k:06d}.tsv'), buf, args.lo, args.hi, args.max_parent, args.max_per_parent,
                   args.smiles_col)

    def todo():
        for j in jobs():
            if os.path.exists(j[0]):   # resumable: a finished chunk is never redone or overwritten
                continue
            yield j

    total = defaultdict(int)
    with Pool(args.workers) as pool:
        for i, st in enumerate(pool.imap_unordered(_work, todo(), chunksize=1)):
            for k, v in st.items():
                total[k] += v
            if i % 20 == 0:
                print(f'{args.source}: {i + 1} chunks done; {dict(total)}', flush=True)
    print(f'{args.source}: done; {dict(total)}', flush=True)


def reduce(args):
    src = os.path.join(args.out, args.source)
    path = os.path.join(src, 'pool.tsv')
    if os.path.exists(path):
        raise SystemExit(f'{path} exists')
    agg = {}
    for p in sorted(glob.glob(os.path.join(src, 'chunks', 'chunk*.tsv'))):
        with open(p, encoding='utf-8') as f:
            for line in f:
                smi, nd, npar, mph = line.rstrip('\n').split('\t')
                rec = agg.get(smi)
                if rec is None:
                    agg[smi] = [int(nd), int(npar), int(mph)]
                else:
                    rec[0] += int(nd)
                    rec[1] += int(npar)
                    rec[2] = min(rec[2], int(mph))
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        f.write('smiles\theavy_atoms\tn_direct\tn_parents\tmin_parent_heavy\n')
        for smi in sorted(agg):
            nd, npar, mph = agg[smi]
            f.write(f'{smi}\t{heavy_atoms_of(smi)}\t{nd}\t{npar}\t{mph if npar else ""}\n')
    os.replace(path + '.tmp', path)
    print(f'{args.source}: {len(agg)} unique molecules -> {path}', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, help='a short name: the subfolder of --out')
    p.add_argument('--files', nargs='*', default=[], help='input globs (.smi, .txt or .gz; first column = SMILES)')
    p.add_argument('--out', required=True)
    p.add_argument('--skip-header', action='store_true', help="drop a first line starting with 'smiles' (ZINC)")
    p.add_argument('--smiles-col', type=int, default=0,
                   help='whitespace-separated column holding the SMILES (PubChem CID-SMILES: 1)')
    p.add_argument('--lo', type=int, default=4)
    p.add_argument('--hi', type=int, default=9)
    p.add_argument('--max-parent', type=int, default=40, help='largest parent (heavy atoms) cut into fragments')
    p.add_argument('--max-per-parent', type=int, default=32)
    p.add_argument('--chunk', type=int, default=200000, help='input lines per chunk file')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--reduce', action='store_true', help='merge the chunk files into pool.tsv')
    args = p.parse_args(argv)
    (reduce if args.reduce else build)(args)


if __name__ == '__main__':
    main(sys.argv[1:])
