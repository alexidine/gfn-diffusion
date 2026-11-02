"""Condition-set statistics: molecules and crystals of every condition / reference source, coloured by source.

Read-only on every dataset. Native stored quantities only: nothing is re-scored, relaxed or re-embedded. Two stages,
each cached so re-plotting is cheap:

  extract  per-source features -> CACHE/<source>_*.{npz,csv.gz}   (CPU; heavy files read with a lazy-tensor
           unpickler that loads only the named tensors)
  rdf      within-molecule pairwise envwise-RDF distances (eval/day_bench route: Day CIF -> P1 crystal of Day's own
           rigid conformer; rc held-out crystals as stored) -> CACHE/rdf_<source>.npz
  plot     figures -> FIG/*.png and summary numbers -> STATS (json)

Sources (key: what is read):
  day     Day-1000 FIT+DMA landscapes (Taylor, Butler, Day 2025): every CSV of each refcode dir, union of ids;
          molecule = first CIF block's asymmetric unit (Z'=1), bonds by RDKit rdDetermineBonds.
  csd     csd_fd_v5 (crystal_fd.dataset/1, Z'=1 CSD crystals; mol_pos/mol_z/cell/sg; RDKit SMILES column).
  qm9     qm9_dataset.pt (133,728 MolData, DFT geometries, QM9 properties in y).
  pool    the proposed Day-like training pool (POOL_FILE): molecules only, read as SMILES.
  qm9rc, csdrc, zincrc, gdbrc, pubrc
          rc_oct22 built families: conditions (one crystal per molecule, train) and test_conditions (held out) for
          molecules; prior / test_prior 'equalized_prior' for crystals (eLJ, sg 2, Z'=1). Every family is the 4-9
          heavy-atom window of smiles_pool.py; outside qm9rc most molecules are capped fragments of larger compounds.
  single  MIPCAS / NEHZOR / acridine single-system priors (energy vs density only).

usage (from energy_sampling/, csd_mxt_gfn venv, MXtalTools on PYTHONPATH):
    CUDA_VISIBLE_DEVICES=-1 python -m data_processing.condition_set_stats --steps extract,rdf,plot \
        --cache D:/crystal_datasets/paper_review_nov02/data --fig OUT/fig --stats OUT/stats.json
"""
import argparse
import collections
import gzip
import io
import json
import os
import pickle
import sys
import textwrap
import time
import warnings
import zipfile

import numpy as np

warnings.filterwarnings('ignore')

D = 'D:/crystal_datasets'
DAY_ROOT = f'{D}/day1000_fitdma/FIT-DMA_Landscapes'
CSD_FD = f'{D}/csd_fd_v5'
QM9_FILE = f'{D}/qm9_dataset.pt'
RC_DIR = f'{D}/conditional/priors/rc_oct22'
RC_FAMILIES = ('qm9rc', 'csdrc', 'zincrc', 'gdbrc', 'pubrc')
SINGLE = {  # system: (file, energy field, energy label)
    # labels checked 2026-11-02: the k10_v2 file stores elj already multiplied by its lj_coeff 0.3636 (its minimum
    # equals the pooled source's raw minimum x 0.3636); the NEHZOR pooled file stores raw elj and carries
    # thermal_scaling_factor 0.156 separately; 'mace' is the lattice energy per molecule in kJ/mol per
    # crystal_analysis.analyze's docstring (unit READ from code, not re-measured)
    'MIPCAS (sg 2)': (f'{D}/conditional/priors/noised_oct28/mipcas_elj_oct05a_k10_v2.pt', 'elj',
                      'stored elj (eLJ x 0.3636, scaled as stored; dimensionless)'),
    'NEHZOR (sg 14)': (f'{D}/conditional/priors/nehzor_elj_pooled_oct05_prior.pt', 'elj',
                       'stored elj (raw eLJ, dimensionless; file thermal_scaling_factor 0.156 not applied)'),
    'acridine (sg 14)': (f'{D}/conditional/priors/acridine_oldmace_oct23_prior.pt', 'mace',
                         'stored mace (MACE acr_112025_mh1_stagetwo lattice energy, kJ/mol per molecule)'),
}

# ---------------------------------------------------------------- source registry: label, colour (fixed order)
# Colours are the dataviz reference categorical palette (light mode), assigned in fixed order; the neutral grey is
# reserved for "the five rc_oct22 families pooled" and is never a source of its own.
SOURCES = collections.OrderedDict([
    ('day', ('Day-1000 (FIT+DMA)', '#2a78d6')),
    ('csd', ('CSD exp. (csd_fd_v5, Z\'=1)', '#eb6834')),
    ('qm9', ('QM9 (DFT geometries)', '#1baf7a')),
    ('pool', ('proposed Day-like pool (no crystals)', '#0f9fb0')),
    ('qm9rc', ('rc qm9rc: QM9 molecules', '#eda100')),
    ('csdrc', ('rc csdrc: CSD-derived pieces', '#e87ba4')),
    ('zincrc', ('rc zincrc: ZINC-derived', '#008300')),
    ('gdbrc', ('rc gdbrc: GDB-11-derived', '#6250d6')),
    ('pubrc', ('rc pubrc: PubChem-derived', '#e34948')),
])
# the proposed pool: CSD + ZINC20-A + PubChem molecules passing the Day-like filter, Day's own removed by SMILES and
# InChIKey skeleton (census of 2026-11-02; one row per molecule, no conformer, no crystal)
POOL_FILE = f'{D}/paper_review_nov02/pool/daylike_pool_v0.csv.gz'
POOL_SOURCES = (('csd', 'pool: in CSD', '#eb6834'), ('zinc20A', 'pool: in ZINC20-A', '#008300'),
                ('pubchem', 'pool: in PubChem', '#6250d6'))
WITH_RC = False  # --with-rc: also read and plot the rc_oct22 families (the 4-9 heavy-atom generated set)
RC_POOLED = ('rc_oct22, 5 families pooled', '#6b6a66')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

ELEMENTS = ('C', 'H', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'P', 'I', 'other')
FG_SMARTS = collections.OrderedDict([
    ('carbonyl C=O', '[CX3]=[OX1]'),
    ('amide', '[NX3][CX3]=[OX1]'),
    ('carboxylic acid', '[CX3](=[OX1])[OX2H1]'),
    ('ester', '[#6][CX3](=[OX1])[OX2H0][#6]'),
    ('ether', '[OX2H0]([#6;!$([CX3]=[O,S,N])])[#6;!$([CX3]=[O,S,N])]'),
    ('hydroxyl', '[OX2H1][#6;!$([CX3]=[O,S])]'),
    ('amine (non-amide)', '[NX3;!$(N[CX3]=[O,S,N]);!$(N-[N,O,S]);!$([N+]);!$(N-a)]'),
    ('aryl amine', '[NX3;!$(N[CX3]=[O,S,N]);!$([N+])]-a'),
    ('nitrile', '[CX2]#[NX1]'),
    ('nitro', '[$([NX3](=O)=O),$([NX3+](=O)[O-])]'),
    ('N-oxide / zwitterion', '[#7+]~[#8-]'),
    ('imine C=N', '[CX3]=[NX2]'),
    ('alkene C=C', '[CX3]=[CX3]'),
    ('alkyne C#C', '[CX2]#[CX2]'),
    ('aromatic ring', 'a'),
    ('aromatic N heterocycle', '[n]'),
    ('fluorine', '[F]'),
    ('3-membered ring', '[r3]'),
    ('4-membered ring', '[r4]'),
    ('methyl CH3', '[CH3]'),
    ('any O-H', '[OX2H1]'),
    ('NH2', '[NX3H2]'),
])
ROTOR_GROUPS = '[$([CH3]),$([OX2H1]),$([NX3H2]),$([SX2H1])]'  # terminal rotors RDKit's strict count ignores
COV_R = {1: .31, 5: .84, 6: .76, 7: .71, 8: .66, 9: .57, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39}
SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
BOND_PAIRS = ('C-C', 'C-N', 'C-O', 'N-N', 'N-O', 'C-F', 'C-H', 'N-H', 'O-H')
ANGLE_CENTRES = ('C', 'N', 'O')
SEED = 0


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ================================================================ lazy torch-zip reader
# A torch .pt (zip) is unpickled with tensors replaced by stubs that read only their own storage on demand, and with
# unknown classes (PyG Batch, MolCrystalData, MolData) replaced by plain state holders. This reads a few columns of a
# multi-GB collated batch without materialising the rest.
class _Storage:
    __slots__ = ('key', 'dtype', 'numel')

    def __init__(self, key, dtype, numel):
        self.key, self.dtype, self.numel = key, dtype, numel


class TStub:
    __slots__ = ('st', 'offset', 'size', 'stride')
    ZF = None
    PREFIX = None

    def __init__(self, st, offset, size, stride):
        self.st, self.offset, self.size, self.stride = st, offset, tuple(size), tuple(stride)

    def load(self):
        import torch
        raw = TStub.ZF.read(f'{TStub.PREFIX}/data/{self.st.key}')
        if len(raw) == 0:
            return torch.empty(self.size, dtype=self.st.dtype)
        flat = torch.frombuffer(bytearray(raw), dtype=self.st.dtype)
        return torch.as_strided(flat, self.size, self.stride, self.offset).clone()

    def np(self):
        return self.load().numpy()


def _rebuild(storage, offset, size, stride, *a, **k):
    return TStub(storage, offset, size, stride)


class Generic:
    __slots__ = ('cls', 'state', 'args')

    def __init__(self, cls=None, *args):
        self.cls, self.args, self.state = cls, args, None

    def __setstate__(self, state):
        self.state = state


def _generic_factory(name):
    class G(Generic):
        __slots__ = ()

        def __new__(cls, *a, **k):
            o = object.__new__(cls)
            o.cls, o.args, o.state = name, a, None
            return o

        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return _generic_factory(name + '()')()
    G.__name__ = name.split('.')[-1]
    return G


_GEN = {}


def _storage_dtypes():
    import torch
    return {'FloatStorage': torch.float32, 'DoubleStorage': torch.float64, 'HalfStorage': torch.float16,
            'BFloat16Storage': torch.bfloat16, 'LongStorage': torch.int64, 'IntStorage': torch.int32,
            'ShortStorage': torch.int16, 'CharStorage': torch.int8, 'ByteStorage': torch.uint8,
            'BoolStorage': torch.bool}


class _Unp(pickle.Unpickler):
    DT = None

    def find_class(self, module, name):
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor', '_rebuild_parameter'):
            return _rebuild
        if module == 'torch' and name.endswith('Storage'):
            return name
        if module.startswith(('builtins', 'collections', 'numpy', 'copyreg', '_codecs', 'datetime', 'pathlib')):
            return super().find_class(module, name)
        if module == 'torch' or (module.startswith('torch.') and not module.startswith('torch_geometric')):
            try:
                return super().find_class(module, name)
            except Exception:
                pass
        key = f'{module}.{name}'
        if key not in _GEN:
            _GEN[key] = _generic_factory(key)
        return _GEN[key]

    def persistent_load(self, pid):
        import torch
        typ, key, numel = pid[1], pid[2], pid[4]
        dt = self.DT.get(typ, torch.uint8) if isinstance(typ, str) else getattr(typ, 'dtype', torch.uint8)
        return _Storage(key, dt, numel)


def open_pt(path):
    _Unp.DT = _storage_dtypes()
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith('/data.pkl')][0]
    TStub.ZF, TStub.PREFIX = zf, pkl[:-len('/data.pkl')]
    with zf.open(pkl) as f:
        return _Unp(io.BufferedReader(f, 1 << 22)).load()


def mapping_of(o):
    if not isinstance(o, Generic) or not isinstance(o.state, dict):
        return None
    store = o.state.get('_store')
    if isinstance(store, Generic) and isinstance(store.state, dict) and isinstance(store.state.get('_mapping'), dict):
        return store.state['_mapping']
    return {k: v for k, v in o.state.items() if not k.startswith('_')}


# ================================================================ molecule descriptors (RDKit, from SMILES)
_PATTS = None
_ROTOR = None


def canon(s):
    from rdkit import Chem
    m = Chem.MolFromSmiles(s) if s else None
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m, isomericSmiles=False)


def describe_smiles(s):
    """Per-molecule scalar descriptors of a SMILES (stereo removed); None-filled dict when RDKit refuses it."""
    global _PATTS
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDLogger.DisableLog('rdApp.*')
    global _ROTOR
    if _PATTS is None:
        _PATTS = {k: Chem.MolFromSmarts(v) for k, v in FG_SMARTS.items()}
        _ROTOR = Chem.MolFromSmarts(ROTOR_GROUPS)
    out = {'key': None}
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
    if m is None:
        return out
    Chem.RemoveStereochemistry(m)
    out['key'] = Chem.MolToSmiles(m, isomericSmiles=False)
    mh = Chem.AddHs(m)
    cnt = collections.Counter(a.GetSymbol() for a in mh.GetAtoms())
    out['n_heavy'] = m.GetNumHeavyAtoms()
    out['n_atoms'] = mh.GetNumAtoms()
    out['mw'] = Descriptors.MolWt(m)
    out['n_rot'] = rdMolDescriptors.CalcNumRotatableBonds(m)
    out['n_rings'] = rdMolDescriptors.CalcNumRings(m)
    out['charged'] = int(any(a.GetFormalCharge() != 0 for a in m.GetAtoms()))
    out['n_frag'] = len(Chem.GetMolFrags(m))
    for e in ELEMENTS[:-1]:
        out[f'n_{e}'] = cnt.get(e, 0)
    out['n_other'] = sum(v for k, v in cnt.items() if k not in ELEMENTS)
    for k, p in _PATTS.items():
        out[f'fg_{k}'] = len(m.GetSubstructMatches(p))
    els = set(cnt)
    out['day_elements'] = int(els <= {'C', 'H', 'N', 'O', 'F'})
    out['day_mw'] = int(out['mw'] < 230)
    out['day_rot'] = int(out['n_rot'] == 0)
    out['day_pass'] = int(out['day_elements'] and out['day_mw'] and out['day_rot'])
    out['day_no_rotor_group'] = int(not m.HasSubstructMatch(_ROTOR))
    out['day_strict'] = int(out['day_pass'] and out['day_no_rotor_group'])
    return out


def describe_many(smiles, workers):
    from multiprocessing import Pool
    import pandas as pd
    smiles = list(smiles)
    if workers <= 1 or len(smiles) < 5000:
        rows = [describe_smiles(s) for s in smiles]
    else:
        with Pool(workers) as p:
            rows = list(p.imap(describe_smiles, smiles, chunksize=1000))
    df = pd.DataFrame(rows)
    df.insert(0, 'smiles_raw', smiles)
    return df


# ================================================================ geometry from 3D coordinates
def geometry_of(z, pos):
    """Bond lengths (pair code, A), bond angles (centre code, degrees) and the heavy-atom rms out-of-plane deviation
    (A) of one molecule. Bonds: d < 1.15 x (sum of covalent radii); no bond orders (element pairs only)."""
    z = np.asarray(z, dtype=int)
    pos = np.asarray(pos, dtype=float)
    n = len(z)
    r = np.array([COV_R.get(int(a), 1.0) for a in z])
    d = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))
    adj = (d < 1.15 * (r[:, None] + r[None])) & ~np.eye(n, dtype=bool)
    adj &= ~((z[:, None] == 1) & (z[None] == 1))
    bl, bc = [], []
    ii, jj = np.where(np.triu(adj, 1))
    for i, j in zip(ii, jj):
        a, b = SYM.get(int(z[i]), 'X'), SYM.get(int(z[j]), 'X')
        pair = '-'.join(sorted((a, b), key=lambda s: 'CNOFHX'.index(s) if s in 'CNOFHX' else 9))
        if pair in BOND_PAIRS:
            bl.append(d[i, j])
            bc.append(BOND_PAIRS.index(pair))
    al, ac = [], []
    for c in range(n):
        s = SYM.get(int(z[c]), 'X')
        if s not in ANGLE_CENTRES:
            continue
        nb = np.where(adj[c])[0]
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                u, v = pos[nb[x]] - pos[c], pos[nb[y]] - pos[c]
                cosv = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
                al.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
                ac.append(ANGLE_CENTRES.index(s))
    h = pos[z > 1]
    oop = np.nan
    if len(h) >= 4:
        sv = np.linalg.svd(h - h.mean(0), compute_uv=False)
        oop = float(sv[-1] / np.sqrt(len(h)))
    elif len(h) >= 1:
        oop = 0.0
    return bl, bc, al, ac, oop


def geometry_many(mols):
    """mols: iterable of (z, pos). Returns the concatenated arrays."""
    BL, BC, AL, AC, OOP = [], [], [], [], []
    for z, pos in mols:
        bl, bc, al, ac, oop = geometry_of(z, pos)
        BL += bl
        BC += bc
        AL += al
        AC += ac
        OOP.append(oop)
    return dict(bond_len=np.array(BL, np.float32), bond_code=np.array(BC, np.int8),
                angle=np.array(AL, np.float32), angle_code=np.array(AC, np.int8), oop=np.array(OOP, np.float32))


# ================================================================ extraction per source
def _save_mol(cache, src, df):
    df.to_csv(os.path.join(cache, f'{src}_molecules.csv.gz'), index=False, compression='gzip')


def extract_day(cache, workers):
    import pandas as pd
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds
    RDLogger.DisableLog('rdApp.*')
    refs = sorted(x for x in os.listdir(DAY_ROOT) if os.path.isdir(os.path.join(DAY_ROOT, x)))
    log('day: refcodes', len(refs))
    mol_rows, geo_in, xt = [], [], []
    for k, ref in enumerate(refs):
        d = os.path.join(DAY_ROOT, ref)
        files = sorted(os.listdir(d))
        frames = []
        for f in files:
            if f.endswith('.csv'):
                frames.append(pd.read_csv(os.path.join(d, f), usecols=['id', 'spacegroup', 'density', 'energy']))
        df = pd.concat(frames).drop_duplicates('id')
        df['ref'] = k
        xt.append(df[['ref', 'spacegroup', 'density', 'energy']].to_numpy(np.float64))
        cifs = [f for f in files if f.endswith('.cif')]
        main = [f for f in cifs if f == ref + '.cif'] or [f for f in cifs if 'sohncke' in f] or cifs
        z, xyz = _day_first_block(os.path.join(d, main[0]))
        smi, smi_err = None, ''
        try:
            block = f'{len(z)}\n\n' + '\n'.join(f'{SYM.get(a, "X")} {x:.6f} {y:.6f} {w:.6f}'
                                                 for a, (x, y, w) in zip(z, xyz))
            m = Chem.MolFromXYZBlock(block)
            rdDetermineBonds.DetermineBonds(m, charge=0)
            if len(Chem.GetMolFrags(m)) > 1 or any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
                # a Z'=1 asymmetric unit is one molecule: fragments or radicals mean bond perception failed
                # (e.g. lovwoi -> 'F[C][C][C+](F)F.[F-]'); keep the row, drop the key
                raise ValueError('perception_fragmented_or_radical')
            smi = Chem.MolToSmiles(Chem.RemoveHs(m))
            ik = Chem.MolToInchiKey(m)
        except Exception as e:
            smi, smi_err, ik = None, f'{type(e).__name__}: {e}'[:80], None
        mol_rows.append(dict(refcode=ref, smiles_day=smi, inchikey=ik, bond_error=smi_err, n_structures=len(df)))
        geo_in.append((z, xyz))
        if k % 100 == 0:
            log('day', k, ref, len(df))
    meta = pd.DataFrame(mol_rows)
    desc = describe_many(meta.smiles_day.fillna(''), 1)
    df = pd.concat([meta, desc.drop(columns=['smiles_raw'])], axis=1)
    df['split'] = 'all'
    _save_mol(cache, 'day', df)
    np.savez_compressed(os.path.join(cache, 'day_geometry.npz'), **geometry_many(geo_in))
    X = np.concatenate(xt)
    np.savez_compressed(os.path.join(cache, 'day_crystals.npz'), mol=X[:, 0].astype(np.int32),
                        sg=X[:, 1].astype(np.int16), density=X[:, 2].astype(np.float32),
                        energy=X[:, 3].astype(np.float32))
    log('day done', len(df), 'molecules', len(X), 'structures')


def _day_first_block(cif):
    lines, started = [], False
    with open(cif, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('data_'):
                if started:
                    break
                started = True
            if started:
                lines.append(line)
    cell, sites, mode = {}, [], None
    for line in lines:
        t = line.strip()
        if t.startswith('_cell_length_') or t.startswith('_cell_angle_'):
            k, v = t.split()[:2]
            cell[k] = float(v)
        elif t == 'loop_':
            mode = 'loop'
        elif t.startswith('_symmetry_equiv_pos_as_xyz'):
            mode = 'ops'
        elif t.startswith('_atom_site_occupancy'):
            mode = 'sites'
        elif t.startswith('_') or not t or t.startswith('#'):
            continue
        elif mode == 'sites':
            f = t.split()
            sites.append((f[1], [float(x) for x in f[2:5]]))
    a, b, c = cell['_cell_length_a'], cell['_cell_length_b'], cell['_cell_length_c']
    al, be, ga = np.radians([cell['_cell_angle_alpha'], cell['_cell_angle_beta'], cell['_cell_angle_gamma']])
    v = np.sqrt(1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2 + 2 * np.cos(al) * np.cos(be) * np.cos(ga))
    T = np.array([[a, b * np.cos(ga), c * np.cos(be)],
                  [0, b * np.sin(ga), c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)],
                  [0, 0, c * v / np.sin(ga)]])
    inv = {v_: k_ for k_, v_ in SYM.items()}
    z = np.array([inv.get(s[0], 0) for s in sites])
    xyz = np.array([s[1] for s in sites]) @ T.T
    return z, xyz


def extract_csd(cache, workers, n_geom):
    import pandas as pd
    import torch
    from rdkit import Chem
    from mxtaltools.constants.space_group_info import SYM_OPS
    d = torch.load(os.path.join(CSD_FD, 'data.pt'), weights_only=True, mmap=True)
    st = d['status'].numpy()
    smiles = list(d['columns']['smiles'])
    ident = list(d['identifier'])
    sg = d['sg'].numpy()
    cell = d['cell'].numpy()
    ptr = d['mol_ptr'].numpy()
    mz = d['mol_z'].numpy().astype(np.int64)
    log('csd rows', len(sg))
    pt = Chem.GetPeriodicTable()
    mass_of = np.array([pt.GetAtomicWeight(i) if i > 0 else 0 for i in range(119)])
    mol_mass = np.add.reduceat(mass_of[mz], ptr[:-1])
    n_heavy_xyz = np.add.reduceat((mz > 1).astype(np.int64), ptr[:-1])
    nz = np.zeros(231, dtype=np.int64)
    for k_, v_ in SYM_OPS.items():
        nz[int(k_)] = len(v_)
    zmult = nz[sg]  # Z = general-position multiplicity of the standard setting (Z' = 1 rows only)
    a, b, c = cell[:, 0], cell[:, 1], cell[:, 2]
    al, be, ga = cell[:, 3], cell[:, 4], cell[:, 5]
    vol = a * b * c * np.sqrt(1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
                              + 2 * np.cos(al) * np.cos(be) * np.cos(ga))
    density = zmult * mol_mass * 1.66054 / vol
    uniq = sorted(set(s for s in smiles if isinstance(s, str)))
    log("csd rows without SMILES", sum(not isinstance(s, str) for s in smiles))
    log('csd unique smiles', len(uniq))
    desc = describe_many(uniq, workers)
    desc['split'] = 'all'
    raw2key = dict(zip(desc.smiles_raw, desc.key))
    keys = sorted(set(k for k in desc.key if isinstance(k, str)))
    kidx = {k: i for i, k in enumerate(keys)}
    mol = np.array([kidx.get(raw2key.get(s), -1) for s in smiles], np.int32)
    # one row per distinct key (first spelling)
    desc = desc[desc.key.notna()].drop_duplicates('key').reset_index(drop=True)
    _save_mol(cache, 'csd', desc)
    with open(os.path.join(cache, 'csd_keys.json'), 'w') as f:
        json.dump(keys, f)
    np.savez_compressed(os.path.join(cache, 'csd_crystals.npz'), mol=mol, sg=sg.astype(np.int16),
                        density=density.astype(np.float32), status=st.astype(np.int8),
                        n_heavy=n_heavy_xyz.astype(np.int16), identifier=np.array(ident))
    rng = np.random.default_rng(SEED)
    first = {}
    for i, k in enumerate(mol):
        if k >= 0 and k not in first:
            first[k] = i
    rows = rng.choice(np.array(list(first.values())), min(n_geom, len(first)), replace=False)
    pos = d['mol_pos']
    geo = geometry_many((mz[ptr[i]:ptr[i + 1]], pos[ptr[i]:ptr[i + 1]].numpy()) for i in rows)
    np.savez_compressed(os.path.join(cache, 'csd_geometry.npz'), **geo)
    manifest = json.load(open(os.path.join(CSD_FD, 'meta.json')))['manifest']['codes']
    with open(os.path.join(cache, 'csd_manifest_codes.json'), 'w') as f:
        json.dump(manifest, f)
    log('csd done')


def extract_qm9(cache, workers, n_geom):
    lst = open_pt(QM9_FILE)
    maps = [mapping_of(o) for o in lst]
    smiles = [m['smiles'] for m in maps]
    log('qm9 rows', len(smiles))
    desc = describe_many(smiles, workers)
    desc['split'] = 'all'
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(len(maps), min(n_geom, len(maps)), replace=False))
    y = np.stack([maps[i]['y'].np().reshape(-1) for i in rows])
    geo = geometry_many((maps[i]['z'].np(), maps[i]['pos'].np()) for i in rows)
    desc_ = desc.drop_duplicates('key').reset_index(drop=True)
    _save_mol(cache, 'qm9', desc_)
    np.savez_compressed(os.path.join(cache, 'qm9_geometry.npz'), **geo)
    np.savez_compressed(os.path.join(cache, 'qm9_props.npz'), rows=rows, y=y.astype(np.float64),
                        n_heavy=desc.n_heavy.to_numpy()[rows])
    log('qm9 done')


def extract_pool(cache, workers, n_geom=5000):
    import pandas as pd
    p = pd.read_csv(POOL_FILE, keep_default_na=False)
    log('pool rows', len(p))
    desc = describe_many(p.key, workers)
    desc['split'] = 'all'
    desc['pool_sources'] = p.srcs.values
    _save_mol(cache, 'pool', desc)
    # the pool stores no geometry: embed a seeded sample the way the proposal's conformer option (a) would
    # (ETKDGv3, then MMFF94 to convergence), for the bond, angle and planarity figures only
    from rdkit import Chem
    from rdkit.Chem import AllChem
    rng = np.random.default_rng(SEED)
    keys = p.key.values[rng.choice(len(p), min(n_geom, len(p)), replace=False)]
    mols = []
    for k in keys:
        m = Chem.AddHs(Chem.MolFromSmiles(k))
        ps = AllChem.ETKDGv3()
        ps.randomSeed = SEED
        if AllChem.EmbedMolecule(m, ps) != 0:
            continue
        AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
        mols.append((np.array([a.GetAtomicNum() for a in m.GetAtoms()]), m.GetConformer().GetPositions()))
    np.savez_compressed(os.path.join(cache, 'pool_geometry.npz'), **geometry_many(mols))
    log('pool done', len(mols), 'embedded of', len(keys))


def _rc_batch(path, entry):
    obj = open_pt(path)
    return mapping_of(obj[entry]), {k: v for k, v in obj.items() if not isinstance(v, Generic)}


def extract_rc(fam, cache, workers, n_geom):
    import pandas as pd
    rng = np.random.default_rng(SEED)
    mol_frames, geo_parts = [], []
    for split, tag in (('train', ''), ('held_out', 'test_')):
        m, hdr = _rc_batch(os.path.join(RC_DIR, f'rcoct22i1_{fam}_{tag}conditions.pt'), 'prior')
        smiles = list(m['smiles'])
        log(fam, split, 'conditions', len(smiles))
        desc = describe_many(smiles, workers)
        desc['split'] = split
        mol_frames.append(desc)
        ptr = m['ptr'].np()
        z = m['z'].np()
        pos = m['pos'].np()
        ng = n_geom if split == 'train' else max(1, n_geom // 4)
        rows = rng.choice(len(smiles), min(ng, len(smiles)), replace=False)
        g = geometry_many((z[ptr[i]:ptr[i + 1]], pos[ptr[i]:ptr[i + 1]]) for i in rows)
        geo_parts.append((split, g))
        del m, z, pos
        # crystals
        pm, phdr = _rc_batch(os.path.join(RC_DIR, f'rcoct22i1_{fam}_{tag}prior.pt'), 'equalized_prior')
        ident = [s[:-3] if s.endswith('_c0') else s for s in pm['identifier']]
        raw2key = dict(zip(desc.smiles_raw, desc.key))
        uk = sorted(set(raw2key.values()) - {None})
        kidx = {k: i for i, k in enumerate(uk)}
        mol = np.array([kidx.get(raw2key.get(s), -1) for s in ident], np.int32)
        out = dict(mol=mol)
        for f in ('density', 'packing_coeff', 'elj', 'num_atoms'):
            out[f] = pm[f].np().reshape(-1).astype(np.float32)
        np.savez_compressed(os.path.join(cache, f'{fam}_{split}_crystals.npz'), keys=np.array(uk), **out)
        log(fam, split, 'prior rows', len(mol), 'unmapped', int((mol < 0).sum()))
        del pm
    _save_mol(cache, fam, pd.concat(mol_frames, ignore_index=True))
    for split, g in geo_parts:
        np.savez_compressed(os.path.join(cache, f'{fam}_{split}_geometry.npz'), **g)
    log(fam, 'done')


def extract_single(cache):
    out = {}
    for name, (path, field, lab) in SINGLE.items():
        m, hdr = _rc_batch(path, 'equalized_prior')
        out[name] = dict(density=m['density'].np().reshape(-1).astype(np.float32),
                         energy=m[field].np().reshape(-1).astype(np.float32))
        log('single', name, len(out[name]['density']))
    np.savez_compressed(os.path.join(cache, 'single_systems.npz'),
                        **{f'{k}|{f}': v[f] for k, v in out.items() for f in v})


# ================================================================ within-molecule RDF distances
def rdf_step(cache, n_mol, k):
    import pandas as pd
    import torch
    torch.set_num_threads(6)
    from eval.day_bench.day_landscape import blocks, block_to_atoms
    from eval.day_bench.p1_relax import p1_crystal
    from eval.day_bench.rdf_embed import rdfs
    from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat
    rng = np.random.default_rng(SEED)

    def pairs(D):
        iu = np.triu_indices(len(D), 1)
        return D[iu]

    # Day: stratified molecule sample (<= 9 heavy atoms / > 9), two structure sets per molecule
    dm = pd.read_csv(os.path.join(cache, 'day_molecules.csv.gz'))
    X = np.load(os.path.join(cache, 'day_crystals.npz'))
    small = dm.index[(dm.n_heavy <= 9) & dm.key.notna()].to_numpy()
    large = dm.index[(dm.n_heavy > 9) & dm.key.notna()].to_numpy()
    pick = np.concatenate([rng.choice(small, min(n_mol, len(small)), replace=False),
                           rng.choice(large, min(n_mol, len(large)), replace=False)])
    order = np.argsort(X['mol'], kind='stable')
    starts = np.searchsorted(X['mol'][order], np.arange(len(dm) + 1))
    rec = {'lowest': [], 'window8': []}
    t0 = time.time()
    for j, mi in enumerate(pick):
        ref = dm.refcode[mi]
        d = os.path.join(DAY_ROOT, ref)
        rows = order[starts[mi]:starts[mi + 1]]
        e = X['energy'][rows]
        frames = [pd.read_csv(os.path.join(d, f), usecols=['id', 'energy']) for f in sorted(os.listdir(d))
                  if f.endswith('.csv')]
        df = pd.concat(frames).drop_duplicates('id').sort_values('energy').reset_index(drop=True)
        bl = {}
        for f in sorted(os.listdir(d)):
            if f.endswith('.cif'):
                bl.update(blocks(os.path.join(d, f)))
        df = df[df.id.astype(str).isin(bl.keys())].reset_index(drop=True)
        if len(df) < 3:
            continue
        lowest = list(df.id.astype(str)[:k])
        win = df[df.energy <= df.energy.iloc[0] + 8.0]
        w_ids = list(win.id.astype(str).sample(min(k, len(win)), random_state=SEED))
        # Day's own rigid conformer: first molecule of the lowest structure (as day_landscape.day_reference)
        from eval.day_bench.p1_ingest import make_reference, split_molecules
        refm = make_reference(split_molecules(block_to_atoms(bl[lowest[0]]))[0], smiles=dm.key[mi])
        for name, ids in (('lowest', lowest), ('window8', w_ids)):
            cr = []
            for i in ids:
                c, res = p1_crystal(block_to_atoms(bl[i]), refm)
                if c is not None:
                    cr.append(c)
            if len(cr) < 2:
                continue
            r, b = rdfs(cr)
            Dm = compute_rdf_distmat(r, b, show_tqdm=False).numpy()
            rec[name].append(dict(mol=int(mi), n_heavy=int(dm.n_heavy[mi]), n=len(cr), d=pairs(Dm)))
        del bl
        if j % 10 == 0:
            log('rdf day', j, ref, f'{time.time() - t0:.0f}s')
    _save_rdf(cache, 'day_lowest', rec['lowest'])
    _save_rdf(cache, 'day_window8', rec['window8'])
    # rc families: held-out crystals as stored (all of each sampled molecule's rows)
    for fam in RC_FAMILIES:
        dd = torch.load(os.path.join(RC_DIR, f'rcoct22i1_{fam}_test_prior.pt'), weights_only=False)
        b = dd['equalized_prior']
        idf = np.array([s[:-3] if s.endswith('_c0') else s for s in b.identifier])
        uniq = np.unique(idf)
        sel = rng.choice(uniq, min(2 * n_mol, len(uniq)), replace=False)
        out = []
        for u in sel:
            rows = np.where(idf == u)[0][:k]
            if len(rows) < 2:
                continue
            lst = [b.get_example(int(i)) for i in rows]
            r, bb = rdfs(lst)
            Dm = compute_rdf_distmat(r, bb, show_tqdm=False).numpy()
            out.append(dict(mol=-1, n_heavy=int((lst[0].z > 1).sum()), n=len(rows), d=pairs(Dm)))
        _save_rdf(cache, fam, out)
        log('rdf', fam, len(out))
        del dd, b


def _save_rdf(cache, name, recs):
    if not recs:
        return
    np.savez_compressed(os.path.join(cache, f'rdf_{name}.npz'),
                        d=np.concatenate([r['d'] for r in recs]).astype(np.float32),
                        owner=np.concatenate([np.full(len(r['d']), j) for j, r in enumerate(recs)]).astype(np.int32),
                        n_heavy=np.array([r['n_heavy'] for r in recs]), n=np.array([r['n'] for r in recs]),
                        median=np.array([np.median(r['d']) for r in recs]))


# ================================================================ plotting
def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
                         'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': GRID, 'grid.linewidth': 0.6, 'figure.dpi': 150, 'savefig.dpi': 150,
                         'lines.linewidth': 1.4, 'figure.facecolor': 'white', 'axes.facecolor': '#fcfcfb'})
    return plt


class _Arrays(dict):
    @property
    def files(self):
        return list(self.keys())


class Data:
    """Cached per-source tables, loaded lazily."""

    def __init__(self, cache):
        import pandas as pd
        self.cache = cache
        self.pd = pd
        self.mol = {}
        for s in SOURCES:
            if s in RC_FAMILIES and not WITH_RC:
                continue
            p = os.path.join(cache, f'{s}_molecules.csv.gz')
            if os.path.exists(p):
                self.mol[s] = pd.read_csv(p, low_memory=False)

    def npz(self, name):
        """Fully materialised npz (an NpzFile re-decompresses an array on every key access)."""
        p = os.path.join(self.cache, name)
        if not os.path.exists(p):
            return None
        with np.load(p, allow_pickle=True) as z:
            return _Arrays({k: z[k] for k in z.files})

    def mols(self, s, split=None, distinct=False):
        """distinct=True: one row per canonical key. Only Day needs it (1,017 refcode rows, 989 constitutions);
        every other source's table is already one row per key (per split for the rc families)."""
        df = self.mol[s]
        df = df[df.key.notna()]
        if split is not None and s in RC_FAMILIES:
            df = df[df.split == split]
        if distinct:
            df = df.drop_duplicates('key')
        return df

    def pool_from(self, src):
        """Proposed-pool molecules found in census source src (csd, zinc20A, pubchem)."""
        df = self.mols('pool', distinct=True)
        return df[df.pool_sources.astype(str).str.contains(src)]

    def rc_pooled(self, split='train'):
        return self.pd.concat([self.mols(f, split) for f in RC_FAMILIES if f in self.mol], ignore_index=True)

    def geom(self, s):
        if s in RC_FAMILIES:
            return self.npz(f'{s}_train_geometry.npz')
        return self.npz(f'{s}_geometry.npz')


def _lbl(s):
    return SOURCES[s][0]


def _col(s):
    return SOURCES[s][1]


def _hist(ax, x, bins, color, label, lw=1.4, ls='-', integer=False):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return
    h, e = np.histogram(np.clip(x, bins[0], bins[-1] - 1e-9), bins=bins, density=True)
    ax.stairs(h, e, color=color, lw=lw, ls=ls, label=f'{label} (N={len(x):,})')


def _two_row(plt, feature, bins, xlabel, D, title, fname, fig_dir, integer=False, transform=None, extra=None):
    """Left: Day vs CSD vs QM9 vs the proposed pool (+ rc pooled when rc is loaded). Right: Day vs each rc family
    (train) when rc is loaded, else Day vs the pool split by source. Density-normalised."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    f = transform or (lambda df: df[feature])
    for s in ('day', 'csd', 'qm9', 'pool'):
        if s in D.mol:
            _hist(axs[0], f(D.mols(s, distinct=True)), bins, _col(s), _lbl(s), lw=2.0 if s == 'day' else 1.4)
    with_rc = any(s in D.mol for s in RC_FAMILIES)
    if with_rc:
        _hist(axs[0], f(D.rc_pooled()), bins, RC_POOLED[1], RC_POOLED[0] + ' (train)', ls='--')
    _hist(axs[1], f(D.mols('day', distinct=True)), bins, _col('day'), _lbl('day'), lw=2.0)
    if with_rc:
        for s in RC_FAMILIES:
            if s in D.mol:
                _hist(axs[1], f(D.mols(s, 'train')), bins, _col(s), _lbl(s) + ' train')
        axs[1].set_title('rc_oct22 condition molecules: conditions file, train split\n(4-9 heavy atoms by '
                         'construction)', fontsize=8)
    elif 'pool' in D.mol:
        for src, lab, c in POOL_SOURCES:
            _hist(axs[1], f(D.pool_from(src)), bins, c, lab)
        axs[1].set_title('proposed pool by source (a molecule in several sources counts in each)', fontsize=8)
    axs[0].set_title('reference, experimental and molecule pools\n(one row per distinct molecule)', fontsize=8)
    for ax in axs:
        ax.set_xlabel(xlabel)
        ax.legend(frameon=False, loc='upper right')
    axs[0].set_ylabel('probability density (per unit of x)')
    fig.suptitle(title, fontsize=9.5, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, fname))
    plt.close(fig)
    return fname


def _heat(plt, mat, rows, cols, title, fname, fig_dir, fmt='{:.0f}', cbar='% of molecules', vmax=None):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('blue', ['#f4f8fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab',
                                                      '#0d366b'])
    fig, ax = plt.subplots(figsize=(1.2 + 0.95 * len(cols), 0.9 + 0.3 * len(rows)))
    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=vmax if vmax is not None else np.nanmax(mat))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha='right')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.grid(False)
    vm = vmax if vmax is not None else np.nanmax(mat)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=6.5,
                        color='white' if v > 0.55 * vm else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar)
    ax.set_title(title, loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, fname))
    plt.close(fig)
    return fname


def _col_sources(D):
    cols, frames = [], []
    for s in SOURCES:
        if s in D.mol:
            df = D.mols(s, 'train', distinct=True)
            cols.append(f'{_lbl(s)}\nN={len(df):,}')
            frames.append(df)
    return cols, frames


def plot_all(cache, fig_dir, stats_path):
    plt = _style()
    D = Data(cache)
    os.makedirs(fig_dir, exist_ok=True)
    stats = {'cache': cache, 'sources': {s: _lbl(s) for s in SOURCES}, 'figures': {}}
    S = stats

    def note(fname, caption):
        S['figures'][fname] = caption

    # ---- per-source molecule counts
    S['molecule_counts'] = {s: {sp: int((D.mol[s].split == sp).sum()) for sp in D.mol[s].split.unique()}
                            for s in D.mol}
    S['molecule_parse_failures'] = {s: int(D.mol[s].key.isna().sum()) for s in D.mol}

    # ---- fig01 sizes
    for feat, bins, xl, fn in (('n_heavy', np.arange(0.5, 40.5, 1), 'heavy atoms per molecule (count; >39 clipped)',
                                'fig01a_heavy_atoms.png'),
                               ('n_atoms', np.arange(0.5, 70.5, 1), 'atoms per molecule incl. H (count; >69 clipped)',
                                'fig01b_total_atoms.png'),
                               ('mw', np.arange(0, 505, 5), 'molecular weight (g/mol; >500 clipped)',
                                'fig01c_molecular_weight.png')):
        _two_row(plt, feat, bins, xl, D, f'Molecule size: {xl.split(" (")[0]}; N molecules per source in legend',
                 fn, fig_dir)
        note(fn, f'Distribution of {xl} per source (distinct canonical molecules; rc families: train split; Day: '
                 'distinct constitutions of the 1,017 refcodes, keys from RDKit bond perception of the CIF, '
                 'refcodes whose perception failed dropped). Day-1000 drawn bold in both panels as the benchmark '
                 'reference.')
    S['size'] = {s: {f: {q: float(np.nanpercentile(D.mols(s, 'train', distinct=True)[f], p)) for q, p in
                         (('p05', 5), ('median', 50), ('p95', 95))} for f in ('n_heavy', 'n_atoms', 'mw')}
                 for s in D.mol}
    S['frac_heavy_le9'] = {s: float((D.mols(s, 'train', distinct=True).n_heavy <= 9).mean()) for s in D.mol}
    S['day_refcode_rows'] = int(len(D.mol['day']))
    S['day_distinct_molecules'] = int(D.mols('day', distinct=True).shape[0])
    S['day_bond_perception_failures'] = D.mol['day'].loc[D.mol['day'].key.isna(), 'refcode'].tolist()

    # ---- fig02 composition
    cols, frames = _col_sources(D)
    mat = np.array([[100 * (df[f'n_{e}'] > 0).mean() for df in frames] for e in ELEMENTS])
    note(_heat(plt, mat, [f'contains {e}' for e in ELEMENTS], cols,
               'Element presence: % of distinct molecules containing the element (rc families: train split)',
               'fig02a_element_presence.png', fig_dir, vmax=100),
         'Percent of each source\'s distinct molecules that contain each element. Day-1000 is C/H/N/O/F only by '
         'construction; compare which training sources carry elements outside that set. CSD: csd_fd_v5 is not '
         'organic-only (about 40% of its rows carry B, Si or a metal); the molecule-level CSD columns cover only '
         'rows with an RDKit SMILES; the rows without one (stats crystal_counts.csd_rows_without_smiles, mostly '
         'metal-bearing) are absent here.')
    S['element_presence_pct'] = {c.split('\n')[0]: {e: float(mat[i, j]) for i, e in enumerate(ELEMENTS)}
                                 for j, c in enumerate(cols)}
    het = lambda df: (df.n_N + df.n_O + df.n_F) / df.n_heavy.clip(lower=1)
    _two_row(plt, None, np.linspace(0, 1, 26), 'heteroatom fraction (N+O+F)/heavy atoms (unitless)', D,
             'Composition: heteroatom fraction per molecule; N molecules per source in legend',
             'fig02b_heteroatom_fraction.png', fig_dir, transform=het)
    note('fig02b_heteroatom_fraction.png', 'Per-molecule fraction of heavy atoms that are N, O or F.')

    # ---- fig03 functional groups
    fgs = list(FG_SMARTS)
    mat = np.array([[100 * (df[f'fg_{g}'] > 0).mean() for df in frames] for g in fgs])
    note(_heat(plt, mat, fgs, cols, 'Functional groups: % of distinct molecules with >= 1 match (RDKit SMARTS)',
               'fig03_functional_groups.png', fig_dir, vmax=100),
         'Percent of each source\'s molecules matching each SMARTS pattern (definitions in FG_SMARTS of the '
         'script). rc families: train split.')
    S['functional_group_pct'] = {c.split('\n')[0]: {g: float(mat[i, j]) for i, g in enumerate(fgs)}
                                 for j, c in enumerate(cols)}

    # ---- fig04 rigidity + Day criteria
    _two_row(plt, 'n_rot', np.arange(-0.5, 12.5, 1), 'rotatable bonds per molecule (RDKit strict count; >11 clipped)',
             D, 'Rigidity: rotatable bonds; N molecules per source in legend', 'fig04a_rotatable_bonds.png', fig_dir)
    note('fig04a_rotatable_bonds.png', 'RDKit CalcNumRotatableBonds per molecule. Day-1000 selected 0 rotatable '
                                       'bonds (their definition may differ from RDKit\'s; Day mass shown at 0).')
    crit = [('elements in {C,H,N,O,F}', 'day_elements'), ('MW < 230 g/mol', 'day_mw'),
            ('0 rotatable bonds', 'day_rot'), ('all three (Day criteria, RDKit count)', 'day_pass'),
            ('no CH3 / OH / NH2 / SH rotor group', 'day_no_rotor_group'),
            ('all three + no rotor group (Day-like)', 'day_strict')]
    mat = np.array([[100 * df[c].mean() for df in frames] for _, c in crit])
    note(_heat(plt, mat, [c for c, _ in crit], cols, 'Fraction of each source passing Day-1000 selection criteria',
               'fig04b_day_criteria.png', fig_dir, vmax=100),
         'Percent of distinct molecules meeting each Day criterion and all three; Z\'=1 and the 26-space-group '
         'restriction are crystal-level and not included.')
    S['day_criteria_pct'] = {c.split('\n')[0]: {k: float(mat[i, j]) for i, (k, _) in enumerate(crit)}
                             for j, c in enumerate(cols)}

    # ---- fig05/06 bond lengths and angles, fig04c planarity
    gsrc = [('day', _lbl('day'), _col('day')), ('csd', _lbl('csd'), _col('csd')), ('qm9', _lbl('qm9'), _col('qm9')),
            ('pool', _lbl('pool') + ', ETKDG + MMFF94 sample', _col('pool'))]
    G = {s: D.geom(s) for s, _, _ in gsrc}
    gsrc = [t for t in gsrc if G[t[0]] is not None]
    gl = list(gsrc)
    if WITH_RC:
        pooled = {}
        for f in RC_FAMILIES:
            g = D.geom(f)
            if g is not None:
                for k in g.files:
                    pooled.setdefault(k, []).append(g[k])
        G['rc'] = {k: np.concatenate(v) for k, v in pooled.items()}
        gl.append(('rc', RC_POOLED[0] + ' (train conformers)', RC_POOLED[1]))
    fig, axs = plt.subplots(3, 3, figsize=(10, 8.0))
    ranges = {'C-C': (1.15, 1.65), 'C-N': (1.1, 1.55), 'C-O': (1.15, 1.5), 'N-N': (1.05, 1.5), 'N-O': (1.15, 1.45),
              'C-F': (1.25, 1.45), 'C-H': (0.9, 1.15), 'N-H': (0.85, 1.1), 'O-H': (0.8, 1.05)}
    S['bond_length_median_A'] = {}
    for p, ax in zip(BOND_PAIRS, axs.flat):
        lo, hi = ranges[p]
        bins = np.linspace(lo, hi, 61)
        code = BOND_PAIRS.index(p)
        for s, lab, c in gl:
            g = G[s]
            if g is None or len(g['bond_len']) == 0:
                continue
            x = g['bond_len'][g['bond_code'] == code]
            if len(x) < 20:
                continue
            _hist(ax, x, bins, c, lab, lw=2.0 if s == 'day' else 1.2, ls='--' if s == 'rc' else '-')
            S['bond_length_median_A'].setdefault(p, {})[lab] = float(np.median(x))
        ax.set_title(f'{p} bonds')
        ax.set_xlabel('bond length (Angstrom)')
        ax.set_ylabel('density (1/Angstrom)')
    nmol = {s: (len(G[s]['oop']) if G[s] is not None else 0) for s, _, _ in gl}
    _fig_legend(fig, gl, nmol)
    fig.suptitle('Bond lengths by element pair (bonds from 3D: d < 1.15 x sum of covalent radii;\n'
                 'out-of-range values clipped into end bins). N = molecules sampled per source.',
                 fontsize=8.5, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(fig_dir, 'fig05_bond_lengths.png'))
    plt.close(fig)
    note('fig05_bond_lengths.png', 'Bond-length histograms by element pair from each source\'s stored 3D '
         'geometry: Day = rigid conformer in the first CIF block; CSD = deposited heavy atoms with regenerated H; '
         'QM9 = DFT; rc = ETKDG embed with random torsions, no force-field relaxation. No bond orders.')
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
    S['angle_median_deg'] = {}
    for cc, ax in zip(ANGLE_CENTRES, axs):
        code = ANGLE_CENTRES.index(cc)
        for s, lab, c in gl:
            g = G[s]
            if g is None:
                continue
            x = g['angle'][g['angle_code'] == code]
            if len(x) < 20:
                continue
            _hist(ax, x, np.linspace(50, 185, 136), c, lab, lw=2.0 if s == 'day' else 1.2,
                  ls='--' if s == 'rc' else '-')
            S['angle_median_deg'].setdefault(cc, {})[lab] = float(np.median(x))
        ax.set_title(f'angles centred on {cc}')
        ax.set_xlabel('bond angle (degrees)')
        ax.set_ylabel('density (1/degree)')
    _fig_legend(fig, gl, nmol)
    fig.suptitle('Bond angles by central element (same molecules as fig05). N = molecules sampled per source.',
                 fontsize=8.5, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(fig_dir, 'fig06_bond_angles.png'))
    plt.close(fig)
    note('fig06_bond_angles.png', 'Bond-angle histograms by central element, same molecules as fig05.')
    fig, ax = plt.subplots(figsize=(6, 3.2))
    S['planar_frac_oop_lt_0.05A'] = {}
    for s, lab, c in gl:
        g = G[s]
        if g is None:
            continue
        x = g['oop']
        _hist(ax, x, np.linspace(0, 1.0, 51), c, lab, lw=2.0 if s == 'day' else 1.2, ls='--' if s == 'rc' else '-')
        S['planar_frac_oop_lt_0.05A'][lab] = float(np.mean(x[np.isfinite(x)] < 0.05))
    ax.set_xlabel('heavy-atom rms out-of-plane deviation (Angstrom; >1 clipped)')
    ax.set_ylabel('density (1/Angstrom)')
    ax.set_title('Planarity of the stored conformer; N molecules in legend', loc='left')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig04c_planarity.png'))
    plt.close(fig)
    note('fig04c_planarity.png', 'Smallest principal rms deviation of heavy atoms from their best plane, per '
                                 'molecule; < 0.05 A counts as planar.')

    # ---- fig07 QM9 molecular energies (only source with a molecular native energy)
    qp = D.npz('qm9_props.npz')
    if qp is not None:
        y = qp['y']
        u0 = y[:, 10]
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.0))
        axs[0].hist(u0, bins=80, color=_col('qm9'), histtype='step', lw=1.4)
        axs[0].set_xlabel('QM9 U0 (Hartree; y[10])')
        axs[0].set_ylabel('molecules (count)')
        hb7 = axs[1].hexbin(qp['n_heavy'], u0, gridsize=40, cmap='Blues', bins='log', mincnt=1)
        fig.colorbar(hb7, ax=axs[1], fraction=0.04, pad=0.02).set_label('molecules per hexagon (log scale)')
        axs[1].set_xlabel('heavy atoms (count)')
        axs[1].set_ylabel('QM9 U0 (Hartree)')
        fig.suptitle(f'Native molecular energy: QM9 only (DFT B3LYP/6-31G(2df,p) U0), N={len(u0):,} sampled '
                     'molecules.\nDay, CSD and rc molecules store no molecular energy.', fontsize=8.5, x=0.01,
                     ha='left')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'fig07_qm9_molecular_energy.png'))
        plt.close(fig)
        note('fig07_qm9_molecular_energy.png', 'QM9 internal energy at 0 K (U0, Hartree) of a random 20k sample '
             'and its scaling with heavy-atom count. No other source stores a molecular energy.')
        S['qm9_U0_hartree'] = {'median': float(np.median(u0)), 'min': float(u0.min()), 'max': float(u0.max()),
                               'n': int(len(u0))}

    # ---- crystals
    day = D.npz('day_crystals.npz')
    dm = D.mol['day']
    emin = np.full(len(dm), np.inf)
    np.minimum.at(emin, day['mol'], day['energy'])
    erel = day['energy'] - emin[day['mol']]
    rc = {}
    for f in (RC_FAMILIES if WITH_RC else ()):
        for sp in ('train', 'held_out'):
            z = D.npz(f'{f}_{sp}_crystals.npz')
            if z is not None:
                rc[(f, sp)] = {k: z[k] for k in z.files}
    csd = D.npz('csd_crystals.npz')
    S['crystal_counts'] = {'day_structures': int(len(day['mol'])),
                           'day_within_8kJ': int((erel <= 8).sum()),
                           'csd_rows': int(len(csd['mol'])) if csd is not None else 0,
                           # rows with no molecule key: csd_fd_v5 SMILES None (39,236; RDKit features fell back,
                           # ~92% carry a metal) or not re-parsable by RDKit (the rest): in every crystal-level CSD
                           # panel, absent from every molecule-level one
                           'csd_rows_without_smiles': int((csd['mol'] < 0).sum()) if csd is not None else 0,
                           **{f'{f}_{sp}_rows': int(len(v['mol'])) for (f, sp), v in rc.items()}}
    S['day_energy_span_kJmol_median'] = float(np.nanmedian(_per_group(day['mol'], erel, np.max, len(dm))))
    # CSD experimental crystals of Day-like molecules (the population Day was drawn from), Day's refcodes removed
    csd_dl = None
    CSD_DL = 'CSD exp. Day-like (Day refcodes removed)'
    if csd is not None:
        ckeys = json.load(open(os.path.join(cache, 'csd_keys.json')))
        strict = D.mol['csd'].set_index('key').day_strict.reindex(ckeys).fillna(0).to_numpy().astype(bool)
        dayref = set(dm.refcode.str.upper())
        not_day = np.array([str(i).upper() not in dayref for i in csd['identifier']])
        csd_dl = (csd['mol'] >= 0) & strict[np.maximum(csd['mol'], 0)] & not_day
        S['crystal_counts']['csd_daylike_rows_day_removed'] = int(csd_dl.sum())
        S['crystal_counts']['csd_daylike_molecules_day_removed'] = int(len(set(csd['mol'][csd_dl].tolist())))

    from matplotlib.colors import LogNorm
    fig, axs = plt.subplots(2, 4, figsize=(13, 7.0)) if WITH_RC else plt.subplots(1, 3, figsize=(12, 3.8))
    axs = np.atleast_2d(axs)
    ax = axs[0, 0]
    keep = erel <= 40
    hb = ax.hexbin(day['density'][keep], day['energy'][keep], gridsize=70, cmap='Blues', bins='log', mincnt=1)
    ax.set_xlabel('density (g/cm$^3$)')
    ax.set_ylabel('FIT+DMA lattice energy (kJ/mol)')
    ax.set_title(f'Day-1000, all structures\n(N={keep.sum():,}; {(~keep).sum():,} with E-Emin>40 omitted)')
    ax = axs[0, 1]
    ax.hexbin(day['density'][keep], erel[keep], gridsize=70, cmap='Blues', bins='log', mincnt=1)
    ax.set_xlabel('density (g/cm$^3$)')
    ax.set_ylabel('E - E_min of its molecule (kJ/mol)')
    ax.set_title(f'Day-1000, relative to each\nmolecule\'s minimum ({len(dm)} refcodes)')
    for ax, f in zip(list(axs[0, 2:]) + list(axs[1, :3]) if WITH_RC else [], RC_FAMILIES):
        v = rc.get((f, 'train'))
        if v is None:
            continue
        ok = np.isfinite(v['elj'])
        ax.hexbin(v['density'][ok], v['elj'][ok], gridsize=70, cmap='Blues', bins='log', mincnt=1,
                  extent=(0, 2.6, np.percentile(v['elj'][ok], 0.01), np.percentile(v['elj'][ok], 99.99)))
        ax.set_xlabel('density (g/cm$^3$)')
        ax.set_ylabel('eLJ (raw, dimensionless; lj_coeff 1)')
        ax.set_title(f'{_lbl(f)} train prior\n(N={ok.sum():,}; sg 2)')
    ax = axs[1, 3] if WITH_RC else axs[0, 2]
    bins = np.linspace(0, 2.6, 66)
    dmin_idx = _argmin_per_group(day['mol'], day['energy'], len(dm))
    _hist(ax, day['density'], bins, _col('day'), 'Day all structures', lw=1.2, ls=':')
    _hist(ax, day['density'][dmin_idx[dmin_idx >= 0]], bins, _col('day'), 'Day global minimum per refcode', lw=2)
    if csd is not None:
        _hist(ax, csd['density'], bins, _col('csd'), 'CSD exp. Z\'=1 (all elements)', lw=1.0, ls=':')
        _hist(ax, csd['density'][csd_dl], bins, _col('csd'), CSD_DL, lw=1.6)
    allrc = np.concatenate([rc[(f, 'train')]['density'] for f in RC_FAMILIES if (f, 'train') in rc] or [[]])
    if WITH_RC:
        _hist(ax, allrc, bins, RC_POOLED[1], RC_POOLED[0] + ' train prior', ls='--')
    ax.set_xlabel('density (g/cm$^3$)')
    ax.set_ylabel('density of rows (per g/cm$^3$)')
    ax.set_title('density, all sources\n(CSD: no stored energy)')
    ax.legend(frameon=False, fontsize=5.5)
    fig.suptitle('Native lattice energy vs density (Day FIT+DMA, kJ/mol; CSD stores no energy). Colour = log10 count '
                 'per hexagon.' + (' rc panels: raw eLJ, not comparable.' if WITH_RC else ''), fontsize=9, x=0.01,
                 ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig08_crystal_energy_density.png'))
    plt.close(fig)
    note('fig08_crystal_energy_density.png', 'Native energy vs density per crystal source. Day: FIT+DMA lattice '
         'energy of every landscape structure (absolute and relative to the molecule minimum); rc families: stored '
         'eLJ of the train prior rows (equalized_prior); last panel: density only, all sources.')
    S['density_median_gcm3'] = {'Day all': float(np.median(day['density'])),
                                'Day global minima': float(np.median(day['density'][dmin_idx[dmin_idx >= 0]])),
                                'CSD exp': float(np.median(csd['density'])) if csd is not None else None,
                                CSD_DL: float(np.median(csd['density'][csd_dl])) if csd is not None else None,
                                **{f'{f} {sp}': float(np.median(v['density'])) for (f, sp), v in rc.items()}}
    S['rc_low_packing_frac_pc_lt_0.55'] = {f'{f} {sp}': float(np.mean(v['packing_coeff'] < 0.55))
                                           for (f, sp), v in rc.items()}
    # the same rows counted per molecule: share of molecules with at least one such row (fig10's high-std mode)
    S['rc_molecules_with_low_packing_row_frac'] = {
        f'{f} {sp}': float(np.mean(np.bincount(v['mol'][(v['mol'] >= 0) & (v['packing_coeff'] < 0.55)],
                                               minlength=len(v['keys']))[np.bincount(v['mol'][v['mol'] >= 0],
                                                                                     minlength=len(v['keys'])) > 0]
                                   > 0)) for (f, sp), v in rc.items()}

    # ---- fig09 space groups and Z'
    day_sg = day['sg'].astype(int)
    day_min_sg = day_sg[dmin_idx[dmin_idx >= 0]]
    sgs_day = [s for s, _ in collections.Counter(day_sg).most_common()]
    top = sorted(set(sgs_day[:26]))
    order_sg = [s for s, _ in collections.Counter(day_sg).most_common() if s in top]

    def frac(arr):
        c = collections.Counter(arr.tolist())
        n = len(arr)
        return [100 * c.get(s, 0) / n for s in order_sg] + [100 * sum(v for k, v in c.items() if k not in top) / n]

    lab_sg = [str(s) for s in order_sg] + ['other']
    series = [('Day all structures', frac(day_sg), len(day_sg)),
              ('Day 8 kJ/mol window', frac(day_sg[erel <= 8]), int((erel <= 8).sum())),
              ('Day global minimum per refcode', frac(day_min_sg), len(day_min_sg))]
    cols_s = [_col('day'), _col('day'), _col('day')]
    if csd is not None:
        series.append(('CSD exp. Z\'=1 (fd_v5, all elements)', frac(csd['sg'].astype(int)), len(csd['sg'])))
        series.append((CSD_DL, frac(csd['sg'][csd_dl].astype(int)), int(csd_dl.sum())))
        cols_s += [_col('csd'), _col('csd')]
    if WITH_RC:
        allrc_sg = np.full(len(allrc), 2)
        series.append(('rc_oct22 train priors', frac(allrc_sg), len(allrc_sg)))
        cols_s.append(RC_POOLED[1])
    fig, axs = plt.subplots(len(series) + 1, 1, figsize=(11, 1.6 * (len(series) + 1)), sharex=False)
    for ax, (name, fr, n), c in zip(axs, series, cols_s):
        ax.bar(range(len(lab_sg)), fr, color=c, width=0.8)
        ax.set_xticks(range(len(lab_sg)))
        ax.set_xticklabels(lab_sg, fontsize=6.5)
        ax.set_ylabel('% of rows')
        ax.set_title(f'{name} (N={n:,})', loc='left')
    axs[-2].set_xlabel('space group number (ordered by frequency over all Day structures; "other" = not among them)')
    S['sg_pct'] = {name: dict(zip(lab_sg, map(float, fr))) for name, fr, _ in series}
    S['day_global_min_n_sgs'] = int(len(set(day_min_sg.tolist())))
    ax = axs[-1]
    codes = json.load(open(os.path.join(cache, 'csd_manifest_codes.json'))) if os.path.exists(
        os.path.join(cache, 'csd_manifest_codes.json')) else {}
    zp = collections.OrderedDict([('Z\'=1 in fd_v5', codes.get('canonical', 0) + codes.get('special_metric', 0)),
                                  ('refused Z\'>1', codes.get('refused:z_prime_gt1', 0)),
                                  ('Z\' fractional', codes.get('skipped:z_prime_fractional', 0)),
                                  ('special position (Z\'<1)', codes.get('skipped:special_position', 0)),
                                  ('multi-component', codes.get('skipped:multi_component', 0)),
                                  ('other skips', sum(v for k, v in codes.items() if k not in (
                                      'canonical', 'special_metric', 'refused:z_prime_gt1',
                                      'skipped:z_prime_fractional', 'skipped:special_position',
                                      'skipped:multi_component')))])
    tot = sum(zp.values()) or 1
    ax.barh(list(zp)[::-1], [100 * v / tot for v in list(zp.values())[::-1]], color=_col('csd'))
    ax.set_xlabel('% of CSD ingest entries (csd_v1_r2)')
    ax.set_title(f'Z\' and why CSD entries are not in fd_v5 (N={tot:,} ingested CIFs). Day-1000 is '
                 'Z\'=1 by construction.', loc='left')
    S['csd_ingest_codes'] = dict(zp)
    fig.suptitle('Space-group distribution per source and Z\' coverage', fontsize=9.5, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig09_spacegroups_zprime.png'))
    plt.close(fig)
    note('fig09_spacegroups_zprime.png', 'Percent of rows per space group: Day (all structures, the 8 kJ/mol '
         'window, and the global minimum of each refcode), CSD experimental Z\'=1 crystals, and the rc train priors '
         '(sg 2 only). Bottom: CSD ingest outcome codes, which bound what Z\' the experimental set represents.')

    # ---- fig10 within-molecule diversity
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axs[0]
    cpm_day = np.bincount(day['mol'], minlength=len(dm))
    cpm_day8 = np.bincount(day['mol'][erel <= 8], minlength=len(dm))
    lb = np.logspace(0, 5, 41)
    ax.hist(cpm_day, bins=lb, histtype='step', color=_col('day'), lw=2, label=f'Day all (N={len(dm)} refcodes)')
    ax.hist(cpm_day8, bins=lb, histtype='step', color=_col('day'), lw=1.4, ls='--', label='Day within 8 kJ/mol')
    if csd is not None:
        cm = csd['mol'][csd['mol'] >= 0]
        cpm_csd = np.bincount(cm)
        cpm_csd = cpm_csd[cpm_csd > 0]
        ax.hist(cpm_csd, bins=lb, histtype='step', color=_col('csd'), lw=1.4,
                label=f'CSD entries per molecule (N={len(cpm_csd):,})')
    for f in RC_FAMILIES:
        v = rc.get((f, 'train'))
        if v is None:
            continue
        c = np.bincount(v['mol'][v['mol'] >= 0])
        c = c[c > 0]
        ax.hist(c, bins=lb, histtype='step', color=_col(f), lw=1.1, label=f'{_lbl(f)} train (N={len(c):,})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('crystals per molecule (count)')
    ax.set_ylabel('molecules (count)')
    ax.set_title('crystals stored per molecule', loc='left')
    ax.legend(frameon=False, fontsize=5.5)
    S['crystals_per_molecule_median'] = {'Day all': float(np.median(cpm_day)), 'Day 8kJ': float(np.median(cpm_day8)),
                                         'CSD': float(np.median(cpm_csd)) if csd is not None else None}
    ax = axs[1]
    bins = np.linspace(0, 0.3, 61)
    sd_day = _per_group(day['mol'], day['density'], np.std, len(dm))
    sd_day8 = _per_group(day['mol'][erel <= 8], day['density'][erel <= 8], np.std, len(dm))
    _hist(ax, sd_day, bins, _col('day'), 'Day all structures', lw=2)
    _hist(ax, sd_day8, bins, _col('day'), 'Day within 8 kJ/mol', ls='--')
    for f in RC_FAMILIES:
        v = rc.get((f, 'train'))
        if v is None:
            continue
        ok = v['mol'] >= 0
        sd = _per_group(v['mol'][ok], v['density'][ok], np.std, int(v['mol'].max()) + 1)
        _hist(ax, sd, bins, _col(f), f'{_lbl(f)} train', lw=1.1)
        S['crystals_per_molecule_median'][f'{f} train'] = float(np.median(np.bincount(v['mol'][ok])))
    ax.set_xlabel('per-molecule std of crystal density (g/cm$^3$; >0.3 clipped)')
    ax.set_ylabel('density of molecules')
    ax.set_title('density spread within a molecule (not a structural distance)', loc='left')
    ax.legend(frameon=False, fontsize=5.5)
    ax = axs[2]
    S['rdf_within_molecule'] = {}
    rbins = np.linspace(0, 1.2, 61)
    for name, lab, c, ls, lw in (('day_lowest', 'Day 10 lowest-energy', _col('day'), '-', 2.0),
                                 ('day_window8', 'Day 10 random in 8 kJ/mol', _col('day'), '--', 1.4)):
        r = D.npz(f'rdf_{name}.npz')
        if r is None:
            continue
        for sub, msk_fn, sls in (('<=9 heavy', lambda nh: nh <= 9, ls), ('>9 heavy', lambda nh: nh > 9, None)):
            own = msk_fn(r['n_heavy'])
            sel = own[r['owner']]
            if sel.sum() == 0:
                continue
            if sub == '>9 heavy' and name == 'day_window8':
                continue
            _hist(ax, r['d'][sel], rbins, c, f'{lab}, {sub} ({own.sum()} mol)',
                  lw=lw if sub == '<=9 heavy' else 1.0, ls=sls if sls else ':')
            S['rdf_within_molecule'][f'{lab}, {sub}'] = {'n_molecules': int(own.sum()), 'n_pairs': int(sel.sum()),
                                                         'median_pair_distance': float(np.median(r['d'][sel]))}
    for f in (RC_FAMILIES if WITH_RC else ()):
        r = D.npz(f'rdf_{f}.npz')
        if r is None:
            continue
        _hist(ax, r['d'], rbins, _col(f), f'{_lbl(f)} held-out ({len(r["n"])} mol)', lw=1.1)
        S['rdf_within_molecule'][f'{f} held-out'] = {'n_molecules': int(len(r['n'])), 'n_pairs': int(len(r['d'])),
                                                     'median_pair_distance': float(np.median(r['d']))}
    ax.set_xlabel('pairwise envwise-RDF distance within a molecule\n(mean per-channel EMD, unitless; >1.2 clipped)')
    ax.set_ylabel('density of pairs')
    ax.set_title('structural diversity within a molecule (<=10 crystals each)', loc='left')
    ax.legend(frameon=False, fontsize=5.2)
    fig.suptitle('Within-condition diversity. Day uses its own rigid conformer in P1 (eval/day_bench route); rc rows '
                 'as stored (sg 2). RDF: envwise, 10 A cutoff, 100 bins.', fontsize=9, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig10_within_molecule_diversity.png'))
    plt.close(fig)
    note('fig10_within_molecule_diversity.png', 'Left: crystals stored per molecule. Middle: per-molecule density '
         'standard deviation (a scalar spread, not a structural distance). Right: pairwise envwise-RDF distances '
         'among up to 10 crystals per sampled molecule (Day stratified by size; rc held-out files).')

    # ---- fig11 corner matrix (molecule level: lowest-energy / experimental crystal)
    def mol_table_day():
        t = dm[['n_heavy', 'mw', 'n_N', 'n_O', 'n_rot']].copy()
        t['density'] = np.where(dmin_idx >= 0, day['density'][np.maximum(dmin_idx, 0)], np.nan)
        return t

    def mol_table_rc():
        parts = []
        for f in RC_FAMILIES:
            v = rc.get((f, 'train'))
            if v is None:
                continue
            m = D.mols(f, 'train').set_index('key')
            keys = v['keys']
            ok = v['mol'] >= 0
            idx = _argmin_per_group(v['mol'][ok], v['elj'][ok], len(keys))
            dens = np.full(len(keys), np.nan)
            pcf = np.full(len(keys), np.nan)
            g = idx >= 0
            dens[g] = v['density'][ok][idx[g]]
            pcf[g] = v['packing_coeff'][ok][idx[g]]
            t = m.reindex(keys)[['n_heavy', 'mw', 'n_N', 'n_O', 'n_rot']].copy()
            t['density'] = dens
            t['packing'] = pcf
            parts.append(t)
        return self_concat(parts)

    def self_concat(parts):
        return D.pd.concat(parts, ignore_index=True) if parts else None

    def mol_table_csd():
        if csd is None:
            return None
        m = D.mols('csd')
        keys = json.load(open(os.path.join(cache, 'csd_keys.json')))
        first = {}
        for i, k in enumerate(csd['mol']):
            if k >= 0 and k not in first:
                first[k] = i
        dens = np.full(len(keys), np.nan)
        for k, i in first.items():
            dens[k] = csd['density'][i]
        t = m.set_index('key').reindex(keys)[['n_heavy', 'mw', 'n_N', 'n_O', 'n_rot']].copy()
        t['density'] = dens
        return t.reset_index(drop=True)

    tabs = [('Day-1000 (global-min crystal, one row per refcode)', _col('day'), mol_table_day()),
            ('CSD exp. (first entry)', _col('csd'), mol_table_csd()),
            (CSD_DL + ' (first entry)', '#9c3d12', None if csd is None else mol_table_csd()[
                strict & ~np.isin(np.array(ckeys, dtype=object), list(set(D.mols('day').key)))].reset_index(drop=True)),
            (RC_POOLED[0] + ' (lowest-eLJ crystal)', RC_POOLED[1], mol_table_rc())]
    tabs = [t for t in tabs if t[2] is not None]
    for _, _, t in tabs:
        t['n_NO'] = t.n_N + t.n_O
    var = [('n_heavy', 'heavy atoms (count)', (0, 30)), ('mw', 'MW (g/mol)', (0, 450)),
           ('n_NO', 'N + O atoms (count)', (0, 10)), ('n_rot', 'rotatable bonds (count)', (0, 10)),
           ('density', 'density (g/cm$^3$)', (0.6, 2.2))]
    rng = np.random.default_rng(SEED)
    nv = len(var)
    fig, axs = plt.subplots(nv, nv, figsize=(11, 11))
    nshow = 2500
    for i in range(nv):
        for j in range(nv):
            ax = axs[i, j]
            if j > i:
                ax.axis('off')
                continue
            vi, li, ri = var[i]
            vj, lj, rj = var[j]
            for name, c, t in tabs:
                if i == j:
                    x = t[vi].to_numpy(float)
                    bins = np.arange(ri[0] - 0.5, ri[1] + 1.5, 1) if vi in ('n_heavy', 'n_NO', 'n_rot') else \
                        np.linspace(ri[0], ri[1], 41)
                    _hist(ax, x, bins, c, name, lw=1.6 if 'Day' in name else 1.1)
                else:
                    tt = t[[vi, vj]].dropna()
                    if len(tt) > nshow:
                        tt = tt.sample(nshow, random_state=SEED)
                    xj = tt[vj].to_numpy(float)
                    yi = tt[vi].to_numpy(float)
                    if vj in ('n_heavy', 'n_NO', 'n_rot'):
                        xj = xj + rng.uniform(-0.3, 0.3, len(xj))
                    if vi in ('n_heavy', 'n_NO', 'n_rot'):
                        yi = yi + rng.uniform(-0.3, 0.3, len(yi))
                    ax.scatter(xj, yi, s=2, color=c, alpha=0.35, lw=0)
            if i != j:
                ax.set_xlim(rj)
                ax.set_ylim(ri)
            else:
                ax.set_xlim(ri)
            if i == nv - 1:
                ax.set_xlabel(lj)
            if j == 0 and i != 0:
                ax.set_ylabel(li)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=3, label=f'{n} (N={len(t):,}; <= {nshow:,} points per panel)')
               for n, c, t in tabs]
    fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=8, bbox_to_anchor=(0.98, 0.9))
    fig.suptitle('Two-body correlates of molecule-level scalars (one row per molecule; integer variables jittered '
                 '+-0.3). Energies omitted here: see fig12 (not comparable across sources).', fontsize=9, x=0.01,
                 ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(fig_dir, 'fig11_corner_matrix.png'))
    plt.close(fig)
    note('fig11_corner_matrix.png', 'Pairwise scatter (lower triangle) and marginals (diagonal) of heavy atoms, MW, '
         'N+O count, rotatable bonds and the density of one crystal per molecule: Day global minimum, CSD first '
         'entry, rc lowest-eLJ train row. rc families pooled (grey). Marginals clip values outside the axis range '
         'into the end bin (the CSD spikes at 30 heavy atoms and 450 g/mol).')
    S['spearman_molecule_level'] = {}
    from scipy.stats import spearmanr
    for name, _, t in tabs:
        tt = t[['n_heavy', 'mw', 'n_NO', 'n_rot', 'density']].dropna()
        S['spearman_molecule_level'][name] = {f'{a}~{b}': float(spearmanr(tt[a], tt[b]).correlation)
                                              for a in tt for b in tt if a < b}

    # ---- fig12 energy correlates (within source)
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4)) if WITH_RC else plt.subplots(1, 2, figsize=(9, 3.4))
    emin_ok = np.isfinite(emin)
    axs[0].hexbin(dm.n_heavy[emin_ok], emin[emin_ok], gridsize=25, cmap='Blues', bins='log', mincnt=1)
    axs[0].set_xlabel('heavy atoms (count)')
    axs[0].set_ylabel('Day E_min FIT+DMA (kJ/mol)')
    axs[0].set_title(f'Day-1000 global-min energy vs size (N={emin_ok.sum()})', loc='left')
    if rc:
        e_min_rc, nh_rc, pc_all, d_all = [], [], [], []
        for f in RC_FAMILIES:
            v = rc.get((f, 'train'))
            if v is None:
                continue
            m = D.mols(f, 'train').set_index('key').reindex(v['keys'])
            ok = v['mol'] >= 0
            e = _per_group(v['mol'][ok], v['elj'][ok], np.min, len(v['keys']))
            e_min_rc.append(e)
            nh_rc.append(m.n_heavy.to_numpy(float))
            sub = rng.choice(len(v['density']), min(200000, len(v['density'])), replace=False)
            pc_all.append(v['packing_coeff'][sub])
            d_all.append(v['density'][sub])
        e_min_rc = np.concatenate(e_min_rc)
        nh_rc = np.concatenate(nh_rc)
        ok = np.isfinite(e_min_rc) & np.isfinite(nh_rc)
        axs[1].hexbin(nh_rc[ok], e_min_rc[ok], gridsize=25, cmap='Blues', bins='log', mincnt=1)
        axs[1].set_xlabel('heavy atoms (count)')
        axs[1].set_ylabel('min eLJ per molecule (raw)')
        axs[1].set_title(f'rc_oct22 train, lowest eLJ vs size (N={ok.sum():,})', loc='left')
        pc_all = np.concatenate(pc_all)
        d_all = np.concatenate(d_all)
        axs[2].hexbin(d_all, pc_all, gridsize=60, cmap='Blues', bins='log', mincnt=1, extent=(0, 2.6, 0, 0.9))
        axs[2].set_xlabel('density (g/cm$^3$)')
        axs[2].set_ylabel('packing coefficient (unitless)')
        axs[2].set_title(f'rc_oct22 train rows (N={len(d_all):,} sampled)', loc='left')
        S['spearman_elj_min_vs_heavy_rc'] = float(spearmanr(nh_rc[ok], e_min_rc[ok]).correlation)
    sp_ok = emin_ok & dm.n_heavy.notna().to_numpy()  # lovwoi: bond perception failed, no heavy-atom count
    S['spearman_Emin_vs_heavy_day'] = float(spearmanr(dm.n_heavy[sp_ok], emin[sp_ok]).correlation)
    keep = erel <= 30
    sub = rng.choice(np.where(keep)[0], min(400000, keep.sum()), replace=False)
    axs[-1].hexbin(day['density'][sub], erel[sub], gridsize=50, cmap='Blues', bins='log', mincnt=1)
    axs[-1].set_xlabel('density (g/cm$^3$)')
    axs[-1].set_ylabel('E - E_min (kJ/mol)')
    axs[-1].set_title(f'Day: E - E_min vs density\n(N={len(sub):,} sampled, E - E_min <= 30)', loc='left')
    fig.suptitle('Energy correlates, each within its own energy function (colour = log10 count)', fontsize=9, x=0.01,
                 ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig12_energy_correlates.png'))
    plt.close(fig)
    note('fig12_energy_correlates.png', 'Energy scaling with molecule size and density, per energy function: Day '
         'FIT+DMA global minimum vs heavy atoms; rc lowest eLJ vs heavy atoms; rc packing coefficient vs density; '
         'Day relative energy vs density.')

    # ---- fig13 train vs held-out
    if WITH_RC:  # rc_oct22 only
        from scipy.stats import ks_2samp
        fig, axs = plt.subplots(3, 5, figsize=(14, 7.5))
        S['train_vs_heldout_ks'] = {}
        for j, f in enumerate(RC_FAMILIES):
            if f not in D.mol:
                continue
            tr, ho = D.mols(f, 'train'), D.mols(f, 'held_out')
            vt, vh = rc.get((f, 'train')), rc.get((f, 'held_out'))
            panels = [('heavy atoms per molecule (count)', tr.n_heavy, ho.n_heavy, np.arange(0.5, 12.5, 1)),
                      ('crystal density (g/cm$^3$)', vt['density'], vh['density'], np.linspace(0.2, 2.2, 61)),
                      ('eLJ per crystal (raw)', vt['elj'], vh['elj'],
                       np.linspace(np.nanpercentile(vt['elj'], 0.5), np.nanpercentile(vt['elj'], 99.5), 61))]
            S['train_vs_heldout_ks'][f] = {}
            for i, (xl, a, b, bins) in enumerate(panels):
                ax = axs[i, j]
                _hist(ax, a, bins, _col(f), 'train')
                _hist(ax, b, bins, INK2, 'held out', ls='--')
                ks = ks_2samp(np.asarray(a, float)[np.isfinite(a)], np.asarray(b, float)[np.isfinite(b)])
                S['train_vs_heldout_ks'][f][xl.split(' (')[0]] = {'ks_D': float(ks.statistic),
                                                                  'n_train': int(len(a)), 'n_held_out': int(len(b))}
                ax.set_xlabel(xl)
                ax.set_ylabel('probability density')
                ax.set_title(f'{_lbl(f)}: KS D={ks.statistic:.3f}', loc='left')
                ax.legend(frameon=False, fontsize=5.5)
        fig.suptitle('Train vs held-out by rc family (molecule-level split; rows 2-3 are all prior rows of each split). '
                     'N per split in legends.\nRows 2-3: values outside the bin range (eLJ: train 0.5-99.5 percentile) '
                     'are clipped into the end bins, hence the end spikes.', fontsize=9, x=0.01, ha='left')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'fig13_train_vs_heldout.png'))
        plt.close(fig)
        note('fig13_train_vs_heldout.png', 'Per rc family: heavy-atom count of condition molecules, and density and '
             'eLJ of all prior rows, train (colour) vs held out (grey dashed), with the two-sample KS statistic. The '
             'held-out side is not an iid draw: build_rc_prior holds a molecule out if ANY of its spellings is held out '
             'in either selection file, which adds smaller, multi-source molecules (csdrc 721 of 3,373 held-out '
             'molecules, qm9rc 563 of 6,652; verified against the selection TSVs, 2026-11-02) and drives the larger '
             'csdrc/qm9rc heavy-atom KS D.')

    # ---- fig14 overlap table
    day_keys = set(D.mols('day').key)
    rows = []
    for s in SOURCES:
        if s == 'day' or s not in D.mol:
            continue
        for sp in (['train', 'held_out'] if s in RC_FAMILIES else ['all']):
            ks_ = set(D.mols(s, sp if s in RC_FAMILIES else None).key)
            hit = day_keys & ks_
            dsub = D.mols('day')
            nref = int(dsub.key.isin(hit).sum())
            nref9 = int((dsub.key.isin(hit) & (dsub.n_heavy <= 9)).sum())
            rows.append((f'{_lbl(s)}{" " + sp if s in RC_FAMILIES else ""}', len(ks_), len(hit), nref, nref9))
    dsub = D.mols('day')
    for sp in (('train', 'held_out') if WITH_RC else ()):  # what a model trained on all five families has seen
        ks_ = set().union(*(set(D.mols(f, sp).key) for f in RC_FAMILIES if f in D.mol))
        hit = day_keys & ks_
        rows.append((f'rc_oct22, 5 families pooled {sp}', len(ks_), len(hit), int(dsub.key.isin(hit).sum()),
                     int((dsub.key.isin(hit) & (dsub.n_heavy <= 9)).sum())))
    # cross-family leak: held-out keys of one family present in any family's train keys
    leak = {}
    trains = {f: set(D.mols(f, 'train').key) for f in RC_FAMILIES if f in D.mol}
    alltrain = set().union(*trains.values()) if trains else set()
    for f in RC_FAMILIES:
        if f in D.mol:
            ho = set(D.mols(f, 'held_out').key)
            leak[f] = {'held_out': len(ho), 'in_own_train': len(ho & trains[f]), 'in_any_rc_train': len(ho & alltrain),
                       'in_qm9': len(ho & set(D.mols('qm9').key)) if 'qm9' in D.mol else None}
    S['rc_heldout_leak'] = leak
    csd_ref = set()
    if csd is not None:
        ids = [str(x).upper() for x in csd['identifier']]
        dayref = set(dm.refcode.str.upper())
        exact = dayref & set(ids)
        prefix = dayref & set(i[:6] for i in ids)
        S['csd_refcode_overlap'] = {'day_refcodes': len(dayref), 'exact_refcode_in_csd_fd_v5': len(exact),
                                    'refcode_family_in_csd_fd_v5': len(prefix)}
        ex9 = int((dm.refcode.str.upper().isin(exact) & (dm.n_heavy <= 9)).sum())
        # column 2 here counts distinct CSD refcodes (crystal entries), not molecules: say so in the row name;
        # column 3 counts the distinct Day molecules among the matched refcodes (refcodes sharing a molecule merge)
        exact_mols = dm.loc[dm.refcode.str.upper().isin(exact), 'key'].dropna().nunique()
        rows.append(('CSD exp. by exact refcode (col. 2 = refcodes)', len(set(ids)), int(exact_mols), len(exact), ex9))
    S['overlap_with_day'] = [dict(source=r[0], molecules_in_source=r[1], day_molecules_hit=r[2],
                                  day_refcodes_hit=r[3], day_refcodes_hit_le9_heavy=r[4]) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.28 * (len(rows) + 1)))
    ax.axis('off')
    hdr = ['source (split)', 'distinct molecules in source (count)', 'Day molecules hit (count)',
           'Day refcodes hit (count)', 'Day refcodes hit, <=9 heavy (count)']
    tb = ax.table(cellText=[[r[0]] + [f'{x:,}' for x in r[1:]] for r in rows], colLabels=hdr, loc='center',
                  cellLoc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    tb.scale(1, 1.25)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_width(0.28 if c_ == 0 else 0.18)
    ax.set_title(f'Molecule overlap with Day-1000 (canonical stereo-free SMILES; Day {len(day_keys)} distinct '
                 f'molecules / {len(dm)} refcodes,\n{int((dm.n_heavy <= 9).sum())} refcodes <= 9 heavy atoms; '
                 'lovwoi, whose bond perception failed, is in neither count). '
                 '\nLast row: CSD crystals whose identifier equals a Day refcode (6-letter refcode family: '
                 f'{S.get("csd_refcode_overlap", {}).get("refcode_family_in_csd_fd_v5", "n/a")}).', loc='left',
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig14_overlap_with_day.png'))
    plt.close(fig)
    note('fig14_overlap_with_day.png', 'Day molecules present in each source, by canonical stereo-free SMILES '
         '(a lower bound: charge-separated Day perceptions miss), and CSD crystals of Day refcodes.')

    # ---- fig15 single systems
    if WITH_RC:  # rc_oct22 only
        ss = D.npz('single_systems.npz')
        if ss is not None:
            names = sorted(set(k.split('|')[0] for k in ss.files))
            fig, axs = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.6))
            for ax, nm in zip(np.atleast_1d(axs), names):
                d_, e_ = ss[f'{nm}|density'], ss[f'{nm}|energy']
                ok = np.isfinite(e_)
                lo, hi = np.percentile(e_[ok], [0.1, 99.9])
                ax.hexbin(d_[ok], e_[ok], gridsize=50, cmap='Blues', bins='log', mincnt=1, extent=(d_[ok].min(),
                                                                                                  d_[ok].max(), lo, hi))
                ax.set_xlabel('density (g/cm$^3$)')
                lab = SINGLE[nm][2]
                ax.set_ylabel({'MIPCAS (sg 2)': 'eLJ x 0.3636 (dimensionless)', 'NEHZOR (sg 14)': 'raw eLJ (dimensionless)',
                               'acridine (sg 14)': 'MACE lattice energy (kJ/mol)'}.get(nm, lab))
                ax.set_title(f'{nm}: N={ok.sum():,}\n{os.path.basename(SINGLE[nm][0])}\n{lab}', loc='left', fontsize=6.5,
                             wrap=True)
            fig.suptitle('Single-system priors: stored energy vs density (each its own energy function; 0.1-99.9 '
                         'percentile shown)', fontsize=9, x=0.01, ha='left')
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'fig15_single_systems.png'))
            plt.close(fig)
            note('fig15_single_systems.png', 'Stored energy vs density of the MIPCAS, NEHZOR and acridine priors used '
                 'by unconditional runs.')

    plot_shift(plt, D, S, fig_dir)
    plot_sources(plt, D, S, fig_dir)
    with open(stats_path, 'w') as f:
        json.dump(S, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    missing = [fn for fn in S['figures'] if not os.path.exists(os.path.join(fig_dir, fn))]
    log('figures written', len(S['figures']), 'missing', missing)
    return S


RC_SELECTION = f'{D}/smiles_sources/selection/new_oct22/selected.tsv'
RC_ORIGIN = {  # what each rc family's SMILES pool was cut from (smiles_pool.py inputs), chunks built of searched
    'qm9rc': ('QM9 molecules as such (selection qm9_oct22)', '66/66'),
    'csdrc': ('CSD H/C/N/O/F components (csd_hcnof_unique.tsv)', '28/28'),
    'zincrc': ('ZINC20 2D tranche A', '92/92'),
    'gdbrc': ('GDB-11, sizes 1-11', '100/477'),
    'pubrc': ('PubChem CID-SMILES (1,000,000 drawn)', '100/284'),
}


def plot_sources(plt, D, S, fig_dir):
    """fig00: one row per source saying what it is, its unit counts, geometry and energy. rc rows also give the share
    of molecules that are capped fragments of a larger compound (n_direct = 0 in the selection file)."""
    import pandas as pd
    frag = {}
    if os.path.exists(RC_SELECTION):
        sel = pd.read_csv(RC_SELECTION, sep='\t', usecols=['smiles', 'n_direct', 'min_parent_heavy'],
                          keep_default_na=False).set_index('smiles')
        sel['min_parent_heavy'] = pd.to_numeric(sel.min_parent_heavy, errors='coerce')
        for f in RC_FAMILIES[1:]:
            if WITH_RC and f in D.mol:
                j = D.mol[f][['smiles_raw']].join(sel, on='smiles_raw')
                j = j[j.n_direct.notna()]
                fr = j.n_direct == 0
                frag[f] = (float(fr.mean()), float(j.loc[fr, 'min_parent_heavy'].median()))
    S['rc_fragment_share'] = {f: {'fragment_share': v[0], 'median_smallest_parent_heavy': v[1]} for f, v in frag.items()}
    cc = S.get('crystal_counts', {})
    mc = S['molecule_counts']

    def n(s, sp):
        if s == 'pool':  # census keys re-canonicalised by describe_smiles: about 2% collapse onto each other
            return f'{len(D.mols(s, distinct=True)):,}'
        return f'{mc.get(s, {}).get(sp, 0):,}'

    rows = [
        ['Day-1000', 'benchmark: rigid C/H/N/O/F CSD molecules, MW < 230; FIT+DMA landscapes',
         f'{S["day_distinct_molecules"]:,} ({S["day_refcode_rows"]:,} refcodes)', f'{cc.get("day_structures", 0):,}',
         'Day rigid conformer', 'FIT+DMA (kJ/mol)', '-'],
        ['CSD csd_fd_v5', 'experimental Z\'=1 crystals, all elements and sizes', n('csd', 'all'),
         f'{cc.get("csd_rows", 0):,}', 'deposited, H regenerated', '-', '-'],
        ['QM9', 'GDB-17 molecules <= 9 heavy atoms', n('qm9', 'all'), '0', 'DFT', 'U0 (Hartree)', '-'],
        ['proposed pool', 'CSD + ZINC-A + PubChem passing the Day-like filter, Day removed', n('pool', 'all'), '0',
         '-', '-', 'not yet'],
    ]
    for f in (RC_FAMILIES if WITH_RC else ()):
        org, chunks = RC_ORIGIN[f]
        what = f'4-9 heavy atoms from {org}'
        if f in frag:
            what += f'; {100 * frag[f][0]:.0f}% fragments (median parent {frag[f][1]:.0f} heavy)'
        rows.append([f'rc {f}', textwrap.fill(what, 62), f'{n(f, "train")} / {n(f, "held_out")}',
                     f'{cc.get(f + "_train_rows", 0):,} / {cc.get(f + "_held_out_rows", 0):,}',
                     'ETKDG, random torsions', 'eLJ raw, sg 2', f'5% by hash; chunks {chunks}'])
    hdr = ['source', 'what it is', 'molecules (train / held out)', 'crystal rows (train / held out)',
           'molecular geometry', 'native energy', 'split']
    fig, ax = plt.subplots(figsize=(15, 0.7 + 0.42 * (len(rows) + 1)))
    ax.axis('off')
    tb = ax.table(cellText=rows, colLabels=hdr, loc='center', cellLoc='left', colLoc='left',
                  colWidths=[0.08, 0.37, 0.12, 0.13, 0.11, 0.08, 0.11])
    tb.auto_set_font_size(False)
    tb.set_fontsize(6.8)
    tb.scale(1, 2.1)
    ax.set_title('Sources compared in fig01-fig15. A "molecule" is one distinct canonical stereo-free SMILES; a '
                 '"crystal row" is one stored structure. rc families: rc_oct22 built files (conditions / '
                 'equalized_prior).', loc='left', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig00_sources.png'))
    plt.close(fig)
    S['figures']['fig00_sources.png'] = (
        'What each source is: origin, molecule and crystal counts, geometry, native energy and split. rc fragment '
        'share = molecules with no identical component in their source pool (selection column n_direct = 0), i.e. '
        'capped pieces of a larger compound; the family name is the pool a molecule was drawn from first.')


SHIFT_FEATURES = ['n_heavy', 'n_atoms', 'mw', 'n_rings', 'n_C', 'n_H', 'n_N', 'n_O', 'n_F'] + \
    [f'fg_{k}' for k in FG_SMARTS]
SHIFT_POOL_N = 5000      # pool molecules per classifier comparison
SHIFT_NN_QUERIES = 1000  # pool queries for the nearest-neighbour baselines


def _count_fps(keys):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetCountFingerprint(Chem.MolFromSmiles(k)) for k in keys]


def _nn_sims(q_keys, ref_fps, exclude_self=False):
    """Largest count-Morgan Tanimoto of each query to the reference set (exclude_self: ignore similarity 1 to an
    identical key, for queries drawn from the reference)."""
    from rdkit import DataStructs
    out = []
    for f in _count_fps(q_keys):
        s = np.array(DataStructs.BulkTanimotoSimilarity(f, ref_fps))
        if exclude_self:
            s[np.argmax(s)] = -1
        out.append(s.max())
    return np.array(out)


def shift_step(cache):
    """Two-sample tests of Day against the proposed pool and its parts: (1) a gradient-boosted classifier asked to
    tell Day from the other set, 5-fold cross-validated ROC AUC (0.5 = indistinguishable), on RDKit descriptors and,
    separately, on count-Morgan fingerprints; (2) nearest-neighbour Tanimoto in both directions. -> CACHE/shift.json"""
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    D = Data(cache)
    rng = np.random.default_rng(SEED)
    day = D.mols('day', distinct=True)
    pool = D.mols('pool', distinct=True).reset_index(drop=True)
    sets = {'pool (all)': pool}
    for src, lab, _ in POOL_SOURCES:
        sets[lab] = D.pool_from(src).reset_index(drop=True)

    def sample(df, n):
        return df.iloc[rng.choice(len(df), min(n, len(df)), replace=False)].reset_index(drop=True)

    def fp_matrix(keys):
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return np.stack([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(k)) for k in keys]).astype(np.float32)

    def auc(a, b, kind):
        X = (np.concatenate([a[SHIFT_FEATURES].to_numpy(float), b[SHIFT_FEATURES].to_numpy(float)]) if kind == 'desc'
             else np.concatenate([fp_matrix(a.key), fp_matrix(b.key)]))
        y = np.r_[np.ones(len(a)), np.zeros(len(b))]
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight='balanced',
                                             random_state=SEED)
        sc = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='roc_auc')
        return float(sc.mean()), float(sc.std())

    out = {'auc': {}, 'smd': {}, 'nn': {}, 'n': {}}
    half = sample(pool, 2 * SHIFT_POOL_N)
    null_a, null_b = half.iloc[:len(day)], half.iloc[SHIFT_POOL_N:]
    for kind in ('desc', 'fp'):
        out['auc'][f'null: pool vs pool ({kind})'] = auc(null_a, null_b, kind)
        for name, df in sets.items():
            out['auc'][f'Day vs {name} ({kind})'] = auc(day, sample(df, SHIFT_POOL_N), kind)
            log('shift', kind, name, out['auc'][f'Day vs {name} ({kind})'])
    for name, df in sets.items():
        out['n'][name] = int(len(df))
        a, b = day[SHIFT_FEATURES].astype(float), df[SHIFT_FEATURES].astype(float)
        sd = np.sqrt((a.var() + b.var()) / 2).replace(0, np.nan)
        out['smd'][name] = ((a.mean() - b.mean()) / sd).dropna().to_dict()
    # nearest neighbours, count-Morgan Tanimoto
    ref = _count_fps(pool.key)
    dayfp = _count_fps(day.key)
    q = sample(pool, SHIFT_NN_QUERIES)
    out['nn']['Day -> pool'] = _nn_sims(day.key, ref).tolist()
    out['nn']['pool sample -> rest of pool'] = _nn_sims(q.key, ref, exclude_self=True).tolist()
    out['nn']['pool sample -> Day'] = _nn_sims(q.key, dayfp).tolist()
    out['nn']['Day -> rest of Day'] = _nn_sims(day.key, dayfp, exclude_self=True).tolist()
    with open(os.path.join(cache, 'shift.json'), 'w') as f:
        json.dump(out, f)
    log('shift done')


def plot_shift(plt, D, S, fig_dir):
    p = os.path.join(D.cache, 'shift.json')
    if not os.path.exists(p):
        return
    sh = json.load(open(p))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={'width_ratios': [1.1, 1.3, 1.1]})
    ax = axs[0]
    names = [k for k in sh['auc'] if k.endswith('(desc)')]
    y = np.arange(len(names))
    for off, kind, c in ((-0.18, 'desc', _col('day')), (0.18, 'fp', _col('pool'))):
        m = [sh['auc'][n.replace('(desc)', f'({kind})')][0] for n in names]
        e = [sh['auc'][n.replace('(desc)', f'({kind})')][1] for n in names]
        ax.barh(y + off, m, height=0.34, xerr=e, color=c, label='descriptors' if kind == 'desc' else
                'count-Morgan fingerprint')
    ax.axvline(0.5, color=INK2, lw=0.8, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace(' (desc)', '') for n in names], fontsize=7)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel('classifier ROC AUC, 5-fold CV (0.5 = indistinguishable)')
    ax.set_title('Can a classifier tell Day from each set?', loc='left')
    ax.legend(frameon=False, fontsize=6.5, loc='lower right')
    ax = axs[1]
    smd = sh['smd']['pool (all)']
    top = sorted(smd, key=lambda k: -abs(smd[k]))[:14]
    cols = {lab: c for _, lab, c in POOL_SOURCES}
    yy = np.arange(len(top))
    for j, (name, c) in enumerate([('pool (all)', _col('pool'))] + [(lab, c) for _, lab, c in POOL_SOURCES]):
        ax.scatter([sh['smd'][name].get(k, np.nan) for k in top], yy + (j - 1.5) * 0.15, s=14, color=c,
                   label=name, zorder=3)
    ax.axvline(0, color=INK2, lw=0.8)
    for x in (-0.2, 0.2):
        ax.axvline(x, color=GRID, lw=0.8, ls='--')
    ax.set_yticks(yy)
    ax.set_yticklabels([k.replace('fg_', '') for k in top], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('standardised mean difference, Day minus set (pooled SD units)')
    ax.set_title('Largest per-feature shifts (|SMD| < 0.2 is small)', loc='left')
    ax.legend(frameon=False, fontsize=6.5)
    ax = axs[2]
    bins = np.linspace(0, 1, 41)
    for name, c, ls in (('Day -> pool', _col('day'), '-'), ('pool sample -> rest of pool', _col('pool'), '-'),
                        ('pool sample -> Day', _col('pool'), ':'), ('Day -> rest of Day', _col('day'), ':')):
        x = np.array(sh['nn'][name])
        _hist(ax, x, bins, c, f'{name} (median {np.median(x):.2f})', ls=ls, lw=1.6 if ls == '-' else 1.2)
    ax.set_xlabel('nearest-neighbour Tanimoto (count Morgan r2, 2048)')
    ax.set_ylabel('density of queries')
    ax.set_title('Coverage in fingerprint space', loc='left')
    ax.legend(frameon=False, fontsize=6.3, loc='upper left')
    fig.suptitle(f'Day-1000 against the proposed pool and its sources. Classifier: Day (988) vs up to '
                 f'{SHIFT_POOL_N:,} molecules per set, class-balanced gradient boosting. NN: {SHIFT_NN_QUERIES:,} pool '
                 'queries; Day queries all 988.', fontsize=8.5, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig16_day_vs_pool_shift.png'))
    plt.close(fig)
    S['shift'] = {'auc': sh['auc'], 'n_sets': sh['n'],
                  'smd_pool_top': {k: sh['smd']['pool (all)'][k] for k in top},
                  'nn_median': {k: float(np.median(v)) for k, v in sh['nn'].items()}}
    S['figures']['fig16_day_vs_pool_shift.png'] = (
        'Left: 5-fold cross-validated ROC AUC of a class-balanced gradient-boosted classifier separating Day from '
        'each set, on RDKit descriptors (counts of heavy atoms, elements, rings and the fig03 groups) and on count '
        'Morgan fingerprints; the null row separates two random pool samples. Middle: standardised mean difference '
        'of the descriptors with the largest shift between Day and the pool, per pool source. Right: nearest-'
        'neighbour Tanimoto distributions in both directions; "rest of" excludes the query itself.')


def _fig_legend(fig, gl, nmol):
    from matplotlib.lines import Line2D
    # Day geometry is one conformer per refcode (1,017; refcodes sharing a molecule repeat it)
    h = [Line2D([0], [0], color=c, lw=2.0 if s == 'day' else 1.4, ls='--' if s == 'rc' else '-',
                label=f'{lab} (N={nmol[s]:,} {"refcodes" if s == "day" else "molecules"})') for s, lab, c in gl]
    fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, fontsize=7)


def _per_group(g, x, fn, n):
    """fn over x within each integer group g in [0, n); NaN for empty groups."""
    g = np.asarray(g)
    x = np.asarray(x, float)
    out = np.full(n, np.nan)
    if len(g) == 0:
        return out
    o = np.argsort(g, kind='stable')
    gs, xs = g[o], x[o]
    cuts = np.flatnonzero(np.diff(gs)) + 1
    for grp, part in zip(np.split(gs, cuts), np.split(xs, cuts)):
        if len(part):
            out[grp[0]] = fn(part)
    return out


def _argmin_per_group(g, x, n):
    g = np.asarray(g)
    x = np.asarray(x, float)
    out = np.full(n, -1, dtype=np.int64)
    o = np.lexsort((x, g))
    gs = g[o]
    first = np.r_[True, gs[1:] != gs[:-1]]
    out[gs[first]] = o[first]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', default='extract,rdf,plot')
    ap.add_argument('--sources', default='day,csd,qm9,pool,' + ','.join(RC_FAMILIES) + ',single')
    ap.add_argument('--cache', default=f'{D}/paper_review_nov02/data')
    ap.add_argument('--fig', required=True, help='figure output directory')
    ap.add_argument('--stats', required=True, help='summary json path')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--n_geom', type=int, default=20000, help='molecules per source for bond/angle geometry')
    ap.add_argument('--rdf_mols', type=int, default=40, help='Day molecules per size stratum; rc: 2x per family')
    ap.add_argument('--rdf_k', type=int, default=10, help='crystals per molecule for RDF distances')
    ap.add_argument('--refresh', action='store_true', help='re-extract sources already cached')
    ap.add_argument('--with-rc', action='store_true', help='also plot the rc_oct22 families and single systems')
    a = ap.parse_args()
    global WITH_RC
    WITH_RC = a.with_rc
    os.makedirs(a.cache, exist_ok=True)
    steps = a.steps.split(',')
    if 'extract' in steps:
        for s in a.sources.split(','):
            done = os.path.exists(os.path.join(a.cache, f'{s}_molecules.csv.gz')) or (
                s == 'single' and os.path.exists(os.path.join(a.cache, 'single_systems.npz')))
            if done and not a.refresh:
                log('cached', s)
                continue
            t0 = time.time()
            if s == 'day':
                extract_day(a.cache, a.workers)
            elif s == 'csd':
                extract_csd(a.cache, a.workers, a.n_geom)
            elif s == 'qm9':
                extract_qm9(a.cache, a.workers, a.n_geom)
            elif s == 'pool':
                extract_pool(a.cache, a.workers)
            elif s in RC_FAMILIES:
                extract_rc(s, a.cache, a.workers, a.n_geom)
            elif s == 'single':
                extract_single(a.cache)
            log('extracted', s, f'{time.time() - t0:.0f}s')
    if 'rdf' in steps:
        rdf_step(a.cache, a.rdf_mols, a.rdf_k)
    if 'shift' in steps:
        shift_step(a.cache)
    if 'plot' in steps:
        plot_all(a.cache, a.fig, a.stats)


if __name__ == '__main__':
    main()
