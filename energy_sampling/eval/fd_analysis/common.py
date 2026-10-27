"""Shared pieces of the FD-identity test (eval/fd_analysis): campaigns, output folders, shard states matched to
registry hits, the coordinator's RDF route, and COMPACK on pairs of repaired CIFs with a keyed cache in the output
folder.

Read-only on the campaign folders. Every write goes to a NEW folder OUT_ROOT/<name> (new_out_dir refuses an existing
one) -- CIFs and the COMPACK cache included; campaign_compack's calibrate/dedup (which write into CAMPAIGN_DIR/compack)
are never called, only its pure helpers (crystal_key, write_cifs into our folder, the RULE functions).

Run with gfn_diffusion and mxtaltools on PYTHONPATH, CPU only (CUDA_VISIBLE_DEVICES=-1).
"""
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

OUT_ROOT = 'D:/crystal_datasets/fd_analysis_oct26'
CAMPAIGNS = {
    'mipcas': dict(dir='D:/crystal_datasets/mipcas_elj_sep28', mol='D:/crystal_datasets/mipcas/MIPCAS_standardized.pt'),
    'nehzor': dict(dir='D:/crystal_datasets/nehzor_elj_sep28', mol='D:/crystal_datasets/nehzor/NEHZOR0_std_conf.pt',
                   old_compack='D:/crystal_datasets/campaigns/nehzor_elj_sep28/compack'),
}
READ_ONLY = ['D:/crystal_datasets/mipcas_elj_sep28', 'D:/crystal_datasets/nehzor_elj_sep28',
             'D:/crystal_datasets/campaigns']
THREADS = 2
RDF_BATCH = 16


def setup():
    torch.set_num_threads(THREADS)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise SystemExit('run with CUDA_VISIBLE_DEVICES=-1 (CPU only)')


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for x in self.streams:
            x.write(s)
            x.flush()

    def flush(self):
        for x in self.streams:
            x.flush()


def new_out_dir(name):
    """OUT_ROOT/name, created; refuses an existing folder and any path under a campaign folder. stdout is copied to
    log.txt there."""
    path = os.path.abspath(os.path.join(OUT_ROOT, name))
    for ro in READ_ONLY:
        if os.path.normcase(path).startswith(os.path.normcase(os.path.abspath(ro))):
            raise SystemExit(f'{path} is inside the read-only {ro}')
    if os.path.exists(path):
        raise SystemExit(f'{path} exists: refusing to overwrite (pick a new name)')
    os.makedirs(path)
    sys.stdout = _Tee(sys.__stdout__, open(os.path.join(path, 'log.txt'), 'w', encoding='utf-8'))
    return path


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def machine_note():
    try:
        import psutil
        return (f'{os.cpu_count()} logical CPUs, system CPU load {psutil.cpu_percent(interval=1):.0f}% at the time '
                f'(other jobs running), torch threads {torch.get_num_threads()}, this process RSS '
                f'{psutil.Process().memory_info().rss / 1e9:.2f} GB, peak working set '
                f'{getattr(psutil.Process().memory_info(), "peak_wset", float("nan")) / 1e9:.2f} GB')
    except Exception:  # noqa: BLE001
        return f'torch threads {torch.get_num_threads()}'


# ----------------------------------------------------------------------------
# campaign, registry, shard states
# ----------------------------------------------------------------------------
def load_campaign(name):
    """(coordinator module, CampaignConfig with the local conformer, registry, energy reference)."""
    from mxtaltools.crystal_search import coordinator as co
    c = CAMPAIGNS[name]
    cfg = co.CampaignConfig.load(os.path.join(c['dir'], 'coord.yaml'))
    cfg.mol_path = c['mol']
    # compute_rdfs' batch: 64 (coord.yaml) peaks near 3 GB on NEHZOR; 16 gives bit-identical RDFs (checked
    # 2026-10-26 on 64 NEHZOR basins: max |diff| 0.0) at about 1 GB less
    cfg.rdf_batch = RDF_BATCH
    reg = torch.load(os.path.join(c['dir'], 'registry.pt'), map_location='cpu', weights_only=False)
    ref = co._energy_ref(cfg, reg)
    return co, cfg, reg, ref


def shard_states(name, co, cfg, reg, ref, band_kT):
    """Every physical shard row (this energy model) within band_kT of the reference, with its registry basin.

    The registry admitted every physical row within window_kT (3) of the reference as it stood when the row was
    ingested; the reference only falls, so every row within band_kT <= window_kT of the final reference is a registry
    hit. Its basin is the hit with the same (run, cursor) and the same float32 energy (repeated energies are taken in
    order). Returns dict(params [n,12] float32, hand [n,1], energy [n] float64, basin [n] long (-1 unmatched),
    stream [n] (0 random, 1 hops), run, cursor, shard_row)."""
    h = reg['hits']
    emap = {}
    for b, e, run, cur in zip(h['basin'], h['energy'], h['run'], h['cursor']):
        emap.setdefault((run, int(cur)), {}).setdefault(float(e), []).append(int(b))
    P, H, E, B, S, RUN, CUR, ROW = [], [], [], [], [], [], [], []
    files = sorted(glob.glob(os.path.join(CAMPAIGNS[name]['dir'], 'shards', '*', '*.pt')))
    for f in files:
        s = torch.load(f, map_location='cpu', weights_only=False)
        if s.get('energy_model_id') != cfg.energy_model_id or len(s['energy']) == 0:
            continue
        en = s['energy']
        keep = co.physical(s['params'], s['lj']) & torch.isfinite(en) & (en.double() <= ref + band_kT * cfg.kT)
        idx = torch.nonzero(keep).flatten()
        if not len(idx):
            continue
        m = emap.get((s['run'], int(s['cursor'])), {})
        lab = []
        for i in idx.tolist():
            lst = m.get(float(en[i]))
            lab.append(lst.pop(0) if lst else -1)
        P.append(s['params'][idx].float())
        H.append(s['handedness'][idx].float())
        E.append(en[idx].double())
        B.append(torch.tensor(lab, dtype=torch.long))
        S.append(torch.full((len(idx),), 0 if s['stream'] == 'random' else 1, dtype=torch.long))
        RUN += [s['run']] * len(idx)
        CUR.append(torch.full((len(idx),), int(s['cursor']), dtype=torch.long))
        ROW.append(idx)
    return dict(params=torch.cat(P), hand=torch.cat(H), energy=torch.cat(E), basin=torch.cat(B),
                stream=torch.cat(S), run=np.array(RUN), cursor=torch.cat(CUR), shard_row=torch.cat(ROW),
                n_shards=len(files))


def subset(st, idx):
    idx = torch.as_tensor(idx, dtype=torch.long)
    out = {}
    for k, v in st.items():
        if torch.is_tensor(v) and v.ndim and len(v) == len(st['energy']):
            out[k] = v[idx]
        elif isinstance(v, np.ndarray) and len(v) == len(st['energy']):
            out[k] = v[idx.numpy()]
        elif isinstance(v, list) and len(v) == len(st['energy']):
            out[k] = [v[i] for i in idx.tolist()]
        else:
            out[k] = v
    return out


def distinct(st):
    """Rows with distinct crystal_key (exact repeats removed, first kept); adds 'key'."""
    from energy_sampling.eval.campaign_compack import crystal_key
    keys, keep, seen = [], [], set()
    for i, (p, h) in enumerate(zip(st['params'], st['hand'])):
        k = crystal_key(p, h)
        if k not in seen:
            seen.add(k)
            keep.append(i)
            keys.append(k)
    out = subset(st, keep)
    out['key'] = keys
    return out


# ----------------------------------------------------------------------------
# RDFs (the coordinator's route: rebuild, standardize_cells, compute_rdfs in the campaign's rdf_mode)
# ----------------------------------------------------------------------------
def rebuild(co, cfg, params, hand):
    return co.rebuild_crystals(cfg, params, hand)


def rdfs(co, cfg, params, hand, batch=256):
    """[n, channels, bins] float32 RDFs, as coordinator._ingest_rows computes them before _assign."""
    from mxtaltools.crystal_search.standardize import standardize_cells
    from mxtaltools.dataset_utils.utils import collate_data_list
    out = []
    for lo in range(0, len(params), batch):
        cr = co.rebuild_crystals(cfg, params[lo:lo + batch], hand[lo:lo + batch])
        std, info = standardize_cells(collate_data_list(cr), on_failure='flag')
        out.append(co.compute_rdfs(std.batch_to_list(), cfg.rdf_batch, cfg.rdf_mode))
    return torch.cat(out) if out else torch.zeros(0)


def rdf_prep(r):
    """Per-channel normalised CDFs [C, n, bins] and active masks [n, C], as coordinator.rdf_distance_matrix prepares."""
    s = r.sum(-1, keepdim=True)
    return torch.cumsum(r / (s + 1e-10), -1).permute(1, 0, 2).contiguous(), (s[..., 0] > 1e-12).float()


def rdf_dist_prepped(ca, aa, cb, ab, bin_width=10 / 99, chunk=64):
    """coordinator.rdf_distance_matrix on prepared inputs (identical arithmetic); the row chunk shrinks so that the
    [C, chunk, nb] temporary stays near 2e7 values."""
    out = torch.empty(ca.shape[1], cb.shape[1])
    chunk = max(1, min(chunk, int(2e7 // max(1, ca.shape[0] * cb.shape[1]))))
    na, nb = aa.sum(1), ab.sum(1)
    for i in range(0, ca.shape[1], chunk):
        s = torch.cdist(ca[:, i:i + chunk], cb, p=1).sum(0)
        inter = aa[i:i + chunk] @ ab.T
        union = (na[i:i + chunk, None] + nb[None] - inter).clamp_min(1)
        out[i:i + chunk] = bin_width * s / union
    return out


# ----------------------------------------------------------------------------
# COMPACK
# ----------------------------------------------------------------------------
# The CSD reader perceives bonds from distances. On NEHZOR it reads Br...Br contacts of 2.06-2.84 A between molecules
# as bonds, adds Br images to close the polymer and shifts atoms by lattice vectors, so a unit cell of 4 molecules is
# read as 2 components or a polymer and COMPACK fails or under-matches (a state against itself in another setting:
# 9/20). Each CIF is therefore read and repaired before the comparison: one atom per CIF label at the file's own
# (whole-molecule) coordinates, and each molecule (CIF atoms written molecule by molecule, n_atoms each, in the
# conformer's atom order) given exactly the conformer's bond graph as the same reader perceives it on the isolated
# molecule. The repaired crystal must hold Z components of n_atoms atoms each, each with the conformer's internal
# distances (1e-3 A), or the pair is 'repair_failed'. PackingSimilarity then runs as single_compack_run does (shell 20,
# CCDC defaults, the first CIF the reference). On a CIF the reader does not fuse the repair changes nothing. Cache
# entries carry RUNNER; results of another runner are never reused.
RUNNER = 'repaired-v1'


def write_cifs(co, cfg, crystals, keys, cif_dir):
    """One unit-cell CIF per crystal, <key>.cif in cif_dir (campaign_compack.write_cifs' route); existing kept."""
    from energy_sampling.eval.campaign_compack import write_cifs as _w
    return _w(co, cfg, None, None, keys, cif_dir, crystals=crystals)


def pair_key(a, b):
    return f'{a}|{b}' if a <= b else f'{b}|{a}'


def cif_atom_sites(path):
    """(labels, fractional coordinates [n, 3]) of the _atom_site loop, in file order."""
    lines = open(path, encoding='utf-8').read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == 'loop_':
            heads, j = [], i + 1
            while j < len(lines) and lines[j].strip().startswith('_'):
                heads.append(lines[j].strip())
                j += 1
            if '_atom_site_label' in heads:
                cl, cx = heads.index('_atom_site_label'), heads.index('_atom_site_fract_x')
                lab, fr = [], []
                while j < len(lines) and len(lines[j].split()) == len(heads):
                    t = lines[j].split()
                    lab.append(t[cl])
                    fr.append([float(t[cx]), float(t[cx + 1]), float(t[cx + 2])])
                    j += 1
                return lab, np.array(fr)
            i = j
        else:
            i += 1
    raise ValueError(f'no _atom_site loop in {path}')


def molecule_template(mi):
    """The conformer's bond graph as the CSD reader perceives it on the isolated molecule (one molecule in a 60 A P1
    box), its internal distances, and whether mi['perm'] (the mirror's atom matching) preserves that graph."""
    from ccdc.crystal import Crystal
    from ccdc.molecule import Atom
    X, z = mi['X'].numpy(), mi['z'].numpy()
    L = 60.0
    sym = [Atom(atomic_number=int(v)).atomic_symbol for v in z]
    rows = [f'{s} {s}{i + 1} {(x[0] + L / 2) / L:.10f} {(x[1] + L / 2) / L:.10f} {(x[2] + L / 2) / L:.10f}'
            for i, (s, x) in enumerate(zip(sym, X))]
    txt = '\n'.join(['data_template'] + [f'_cell_length_{a} {L}' for a in 'abc'] +
                    [f'_cell_angle_{a} 90' for a in ('alpha', 'beta', 'gamma')] +
                    ["_space_group_name_H-M_alt 'P 1'", '_space_group_IT_number 1', 'loop_',
                     '_space_group_symop_operation_xyz', "'x, y, z'", 'loop_', '_atom_site_type_symbol',
                     '_atom_site_label', '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z'] + rows) + '\n'
    m = Crystal.from_string(txt, 'cif').molecule
    li = {f'{s}{i + 1}': i for i, s in enumerate(sym)}
    bonds = sorted(tuple(sorted((li[b.atoms[0].label], li[b.atoms[1].label]))) for b in m.bonds)
    if len(m.components) != 1:
        raise SystemExit(f'conformer read as {len(m.components)} components')
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    perm = np.asarray(mi['perm'])
    bset = set(bonds)
    auto = all(tuple(sorted((int(perm[p]), int(perm[q])))) in bset for p, q in bonds)
    return dict(na=len(z), bonds=bonds, D=D, perm_is_automorphism=bool(auto))


def repair_crystal(path, tmpl):
    """The CIF's crystal with its molecules repaired (see the note above); (crystal, info). Raises ValueError when the
    repaired crystal is not Z whole molecules of the conformer."""
    from ccdc.io import CrystalReader
    from ccdc.molecule import Bond
    na, bonds = tmpl['na'], tmpl['bonds']
    c = CrystalReader(path)[0]
    lab, fr = cif_atom_sites(path)
    if len(set(lab)) != len(lab) or len(lab) % na:
        raise ValueError(f'{len(lab)} atom sites, not a multiple of {na} distinct labels')
    Z = len(lab) // na
    m = c.molecule
    n_comp_read = len(m.components)
    labset, seen, extra = set(lab), set(), []
    for a in m.atoms:
        if a.label in seen or a.label not in labset:
            extra.append(a)
        seen.add(a.label)
    if extra:
        m.remove_atoms(extra)
    at = {a.label: a for a in m.atoms}
    if len(at) != len(lab) or len(m.atoms) != len(lab):
        raise ValueError(f'{len(m.atoms)} atoms after removing images, {len(lab)} sites')
    Fc = np.array([a.fractional_coordinates for a in m.atoms])
    Cc = np.array([a.coordinates for a in m.atoms])
    Mx = np.linalg.lstsq(np.c_[Fc, np.ones(len(Fc))], Cc, rcond=None)[0]   # the reader's fractional -> Cartesian map
    target = np.c_[fr, np.ones(len(fr))] @ Mx
    moved = 0
    for l, x in zip(lab, target):
        if np.abs(np.array(at[l].coordinates) - x).max() > 1e-4:
            at[l].coordinates = tuple(float(v) for v in x)
            moved += 1
    idx = {l: i for i, l in enumerate(lab)}
    want = {(k * na + p, k * na + q) for k in range(Z) for p, q in bonds}
    have, rem = set(), []
    for b in m.bonds:
        e = tuple(sorted((idx[b.atoms[0].label], idx[b.atoms[1].label])))
        have.add(e)
        if e not in want:
            rem.append(b)
    n_inter = sum(1 for b in rem if idx[b.atoms[0].label] // na != idx[b.atoms[1].label] // na)
    add = sorted(want - have)
    if rem:
        m.remove_bonds(rem)
    for p, q in add:
        m.add_bond(Bond.BondType('Unknown'), at[lab[p]], at[lab[q]])
    changed = bool(extra or moved or rem or add)
    if changed:
        c.molecule = m
    comps = c.molecule.components
    info = dict(Z=Z, components_as_read=n_comp_read, image_atoms_removed=len(extra), atoms_moved=moved,
                inter_bonds_removed=n_inter, intra_bonds_removed=len(rem) - n_inter, intra_bonds_added=len(add),
                changed=changed)
    if len(comps) != Z or any(len(x.atoms) != na for x in comps):
        raise ValueError(f'repaired: {len(comps)} components of sizes {sorted(len(x.atoms) for x in comps)}')
    P = {a.label: np.array(a.coordinates) for a in c.molecule.atoms}
    for k in range(Z):
        B = np.stack([P[l] for l in lab[k * na:(k + 1) * na]])
        dev = float(np.abs(np.linalg.norm(B[:, None] - B[None], axis=-1) - tmpl['D']).max())
        if dev > 1e-3:
            raise ValueError(f'repaired molecule {k}: internal distances off the conformer by {dev:.3g} A')
    return c, info


def compack_pair_repaired(args):
    """(key, rmsd, n_matched, status, seconds, info_ref, info_test): both CIFs repaired, then PackingSimilarity with
    packing_shell_size 20 and CCDC defaults, the first CIF the reference (single_compack_run's settings). Status 'ok';
    'failed' when reading the result raises AttributeError (single_compack_run's (0, 0)); 'repair_failed ...';
    'error ...'."""
    key, ref_path, test_path, tmpl = args
    t = time.time()
    try:
        ref, ia = repair_crystal(ref_path, tmpl)
        test, ib = repair_crystal(test_path, tmpl)
    except ValueError as e:
        return key, float('nan'), -1, f'repair_failed: {str(e)[:200]}', time.time() - t, None, None
    try:
        from ccdc.crystal import PackingSimilarity
        eng = PackingSimilarity()
        eng.settings.packing_shell_size = 20
        r = eng.compare(ref, test)
        try:
            rmsd, nm, status = float(r.rmsd), int(r.nmatched_molecules), 'ok'
        except AttributeError:
            rmsd, nm, status = 0.0, 0, 'failed'
        return key, rmsd, nm, status, time.time() - t, ia, ib
    except Exception as e:  # noqa: BLE001 - recorded, never read as a result
        return key, float('nan'), -1, f'error {type(e).__name__}: {str(e)[:200]}', time.time() - t, ia, ib


def compack(pairs, cif_dir, cache, cache_path, tmpl, timeout=120.0, extra_caches=()):
    """COMPACK on repaired CIFs (compack_pair_repaired; lower key the reference) on each pair of keys, one worker
    process at a time. cache (dict) is updated and saved to cache_path every 25 pairs. A pair with a final result
    ('ok' or 'failed') of this RUNNER in cache or in an extra cache (read-only dicts) is not rerun. A pair that runs
    past timeout seconds is recorded 'timeout', its worker is terminated and a fresh one started, so every pair
    returns within timeout."""
    want = list(dict.fromkeys(pair_key(a, b) for a, b in pairs))
    for ec in extra_caches:
        for k in want:
            if k not in cache and k in ec and ec[k].get('runner') == RUNNER and ec[k]['status'] in ('ok', 'failed'):
                cache[k] = dict(ec[k], reused=True)
    todo = [k for k in want if k not in cache]
    log(f'COMPACK ({RUNNER}): {len(want)} pairs, {len(want) - len(todo)} cached, {len(todo)} to run (1 worker, '
        f'timeout {timeout} s)')
    if not todo:
        return {k: cache[k] for k in want}
    pool = mp.Pool(1)
    t0 = time.time()
    try:
        for n, k in enumerate(todo):
            a, b = k.split('|')
            hnd = pool.apply_async(compack_pair_repaired, ((k, os.path.join(cif_dir, f'{a}.cif'),
                                                            os.path.join(cif_dir, f'{b}.cif'), tmpl),))
            try:
                _, rmsd, nm, status, dt, ia, ib = hnd.get(timeout=timeout)
            except mp.TimeoutError:
                rmsd, nm, status, dt, ia, ib = float('nan'), -1, 'timeout', float(timeout), None, None
                pool.terminate()
                pool.join()
                pool = mp.Pool(1)
            cache[k] = dict(rmsd=rmsd, n_matched=nm, status=status, seconds=dt, runner=RUNNER, repair_a=ia,
                            repair_b=ib)
            if (n + 1) % 25 == 0 or n + 1 == len(todo):
                torch.save(cache, cache_path)
                log(f'   {n + 1}/{len(todo)} compared ({(time.time() - t0) / (n + 1):.2f} s per pair wall)')
    finally:
        pool.terminate()
        pool.join()
        torch.save(cache, cache_path)
    return {k: cache[k] for k in want}


# ----------------------------------------------------------------------------
# cut statistics
# ----------------------------------------------------------------------------
def pr_table(d, y, cuts):
    """Rows (cut, TP, FP, FN, TN, precision, recall, F1) for 'same' = d <= cut against y (1 = COMPACK 20/20)."""
    rows = []
    for c in cuts:
        pred = d <= c
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        tn = int((~pred & (y == 0)).sum())
        p = tp / (tp + fp) if tp + fp else float('nan')
        r = tp / (tp + fn) if tp + fn else float('nan')
        f1 = 2 * p * r / (p + r) if (tp and p + r) else 0.0
        rows.append((float(c), tp, fp, fn, tn, p, r, f1))
    return rows


def best_f1_cut(d, y):
    """The compared distance maximising F1 of 'same' = d <= cut; ties: the smallest such distance."""
    o = np.sort(np.unique(d))
    rows = pr_table(d, y, o)
    f1 = np.array([r[-1] for r in rows])
    i = int(np.argmax(f1))
    return float(o[i]), rows[i]


def recall_at_fpr(d, y, fpr):
    """Share of matched pairs with d <= c, at the largest compared c whose share of unmatched pairs with d <= c is
    <= fpr ('same' = d <= c); (recall, c)."""
    o = np.sort(np.unique(d))
    neg = d[y == 0]
    pos = d[y == 1]
    best = (0.0, float('nan'))
    for c in o:
        if (neg <= c).mean() <= fpr:
            best = (float((pos <= c).mean()), float(c))
    return best


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(np.asarray(a), np.asarray(b)))
