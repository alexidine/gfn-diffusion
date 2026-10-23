"""Capped Metropolis walkers from search minima, to cover each basin's thermal band for a prior buffer.

NOT a thermal sampler, and not meant to be one: the walkers only have to visit the states the prior should cover.

Walkers
    One walker per (seed, temperature rung). Seeds are the crystals of --prior (a prior file's 'prior' key, a batch or a
    list of crystals, e.g. a search export) within --seed_window_kT of the lowest one, ranked by their STORED energies
    when the file carries them. --n_shards/--shard split the ranked seeds round-robin, so array jobs share one window
    and one energy reference without re-scoring every seed.

Energy
    The PHYSICAL energy only, in the training currency, on the training route's analyze call (['reduction_en', ef],
    cutoff 10, supercell_size 10, std_orientation False; conditioning molecule = the file's first crystal):
      elj        eLJ x the prior's thermal_scaling_factor (stamped as lj_coeff, as train.py does);
      uma, mace  the MLIP lattice energy (kJ/mol per molecule). The gas-phase leg is computed ONCE per run on the
                 first seed and attached as <ef>_gas_pot, as the training energy does (it depends on the molecule only).
    No Jacobians, no density, bounding or reduction terms. reduction_en is recorded, and used only with --red_max: a
    proposal whose reduction penalty exceeds it is rejected (counted as 'red'), which
    keeps a chain inside the reduced-cell domain the trainer penalises leaving (without it 65% of the states of the
    2026-09-29 MIPCAS eLJ flood carry a non-zero penalty). With --pc_max, a
    proposal denser than that packing coefficient is rejected before the energy call (MLIPs can show spurious
    low-energy holes at overlapping geometries).

Moves
    In the GFN's own latent chart. Period-2 wraps: the model's (the space group's full-width centroid axes, orientation
    phi and r) plus, for P-1 only, centroid x, whose half-cell aunit width is a pure translation of the whole crystal
    there (E(u) = E(u + 2) exactly; the model does NOT wrap it, but both faces are inside its chart). The chain is kept
    UNFOLDED across the orientation-axis equator (theta = 1) and pole (theta = -1), which are seams of the chart, not
    edges: (theta, phi, r) is the same rotation as (2 - theta, phi + 1, -r) and as (-2 - theta, phi + 1, r). Each
    proposal is folded back into the chart before the box test, the energy call and the record. Every other dim is held
    inside the latent_to_cell_params clamp box (including r >= -0.99): a proposal whose folded point leaves it is
    rejected, so no state is ever clamped. Dead rows (fixed cell angles) stay at the seed value. A triclinic proposal
    whose angles admit no cell is rejected before scoring.
    Proposal: x' = x + exp(log_scale_c) * chol(Sigma_c) z, per chain c, in the unfolded coordinates. Sigma_c starts
    at init_step^2 I. Until --adapt_end, log_scale_c moves toward --acc_target acceptance and Sigma_c is re-estimated
    from the chain's own states at --cov_updates (keep the last one well before --adapt_end: each update re-zeroes
    log_scale); from --adapt_end on both are frozen, so the rest is a plain Metropolis chain.

Seeds and walkers
    The windowed minima are THINNED first (greedy atomwise-RDF cover in ascending energy at --dedupe_radius, default
    the same-packing radius --rdf_dcut: the lowest member of each group of near-duplicates survives), then each distinct
    minimum gets --replicas walkers per temperature rung.

Exploration
    At every checkpoint the walkers' current states are added to a greedy RDF cover per radius (--explore_radii, in
    units of --rdf_dcut), seeded with the minima. The share landing farther than r from everything seen before is the
    new fraction; exploration % = 1 - new fraction (a Good-Turing-style coverage estimate at that resolution). Rows go
    to exploration.jsonl against cumulative walker-steps, energy evaluations and wall time.

Acceptance
    min(1, exp(-(E' - E) / T_c)) with T_c = temper_c * T_train, and a HARD cap: E' > E_spawn + cap_local or
    E' > E_min_seeds + cap_global is rejected outright (caps in units of kT_train). Non-finite energies are rejected.

Output (--out)
    seeds.pt     seed rows: latent, cell params, handedness, E (fresh), stored E, packing coeff, reduction_en, row index.
    shard_*.pt   every ACCEPTED move: latent (folded into the chart), cell params, handedness, E, dE_spawn, packing
                 coeff, reduction_en, chain, seed, rung, step. Folding by symmetry, dedup and symmetrisation happen
                 downstream.
    state.pt     resume point, written at start and with each shard (chains in unfolded coordinates).
    exploration.jsonl  one row per checkpoint: effort so far and, per radius, the new fraction and cover size (the
                 cover restarts from the minima and the current states after a resume; rows carry `resumed`).
    meta.json    settings and provenance.

OOM
    The energy call halves its chunk on CUDA OOM and retries, and regrows it after 20 clean chunks. An OOM that
    escapes it restarts the loop from state.pt with a halved chunk, at most --max_restarts times; then it raises.

    python data_processing/capped_mc.py --prior <file> --energy_function uma --mlip_path <esen_s.pt> --out <dir> --resume
    python data_processing/capped_mc.py --system nehzor --out D:/crystal_datasets/capped_mc/nehzor_elj   (local ELJ)
"""
import argparse
import gc
import glob
import json
import math
import os
import time

import numpy as np
import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('GPU_MEM_FRACTION', 0.25)))

from mxtaltools.dataset_utils.utils import collate_data_list
from energy_sampling.models.aunit_periodicity import sg_periodic_centroid_axes
from energy_sampling.models.dead_latent_rows import resolve_dead_rows

SYSTEMS = {  # local ELJ shortcuts
    'mipcas': dict(prior='D:/crystal_datasets/conditional/priors/mipcas_sg2_zp1_elj_200k_prior_dataset_niggli_v2.pt'),
    'nehzor': dict(prior='D:/crystal_datasets/conditional/priors/nehzor_sg14_zp1_elj_prior_dataset_w3.pt'),
}
DIM = 12  # Z'=1: [3 lengths | 3 angles | 3 centroid | 3 orientation (theta, phi, r)]
THETA, PHI, RMAG = 9, 10, 11
ANG_LO, ANG_SPAN = 0.2 * math.pi, 0.6 * math.pi
MLIPS = ('uma', 'mace')


def is_oom(e):
    return isinstance(e, torch.OutOfMemoryError) or 'out of memory' in str(e).lower()


def latent_box():
    """The latent_to_cell_params clamp box (mxtaltools crystal_ops), Z'=1."""
    lo = torch.full((DIM,), -1.0, dtype=torch.float64)
    hi = torch.full((DIM,), 1.0, dtype=torch.float64)
    lo[:3] = -0.99
    hi[:2] = 1 - 1e-4
    lo[9] = -0.99   # orientation theta
    lo[11] = -0.99  # orientation r
    return lo, hi


def wrap(x, per):
    x = x.clone()
    x[:, per] = torch.remainder(x[:, per] + 1, 2) - 1
    return x


def fold(x, per):
    """Unfolded chain coordinates -> the chart point of the same crystal. theta past the equator (> 1) or the pole
    (< -1) maps to the same rotation about the antipodal axis: (2 - theta, phi + 1, -r) and (-2 - theta, phi + 1, r).
    One reflection each way, so a theta more than 2 past either seam folds to a point the box test rejects."""
    x = x.clone()
    eq, pole = x[:, THETA] > 1, x[:, THETA] < -1
    x[eq, THETA] = 2 - x[eq, THETA]
    x[eq, RMAG] = -x[eq, RMAG]
    x[pole, THETA] = -2 - x[pole, THETA]
    x[eq | pole, PHI] += 1
    return wrap(x, per)


def save_atomic(obj, path, tries=10):
    """torch.save to a temp file, then os.replace; retries a bounded number of times if another process holds the
    target open (Windows PermissionError), then raises."""
    torch.save(obj, path + '.tmp')
    for k in range(tries):
        try:
            os.replace(path + '.tmp', path)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(1.0)


def triclinic_ok(L, eps=1e-3):
    """Angle triple admits a cell: 1 - sum cos^2 + 2 prod cos > eps (volume factor squared)."""
    ang = (L[:, 3:6] / 2 + 0.5) * ANG_SPAN + ANG_LO
    c = torch.cos(ang)
    return (1 - (c ** 2).sum(1) + 2 * c.prod(1)) > eps


def load_predictor(ef, path, device):
    if ef == 'uma':
        from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
        return init_uma_crystal_predictor(path, device=device)
    if ef == 'mace':
        from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
        return load_mace_model(path, device=device, dtype=torch.float32)
    return None


class Energy:
    """The physical energy on the training route, chunked, OOM-safe. Every chunk is padded to the current chunk size
    so a few collated templates serve every call. On CUDA OOM the chunk halves; after REGROW_AFTER consecutive
    successful chunks it doubles again, up to the starting size, so one heavy batch cannot pin the run at a tiny chunk.
    For an MLIP the gas-phase leg is computed once (set_gas_reference) and attached to every batch."""
    REGROW_AFTER = 20

    def __init__(self, mol, sg, ef, lj_coeff, device, chunk, predictor=None, pc_max=None, rdf_mode='atomwise'):
        self.mol, self.sg, self.ef, self.lj_coeff, self.dev = mol, sg, ef, float(lj_coeff), device
        self.chunk = self.max_chunk = int(chunk)
        self.pred, self.pc_max, self.gas, self.rdf_mode = predictor, pc_max, None, rdf_mode
        self.tmpl, self.n_eval, self.n_oom, self.n_ok = {}, 0, 0, 0
        if ef in MLIPS and predictor is None:
            raise ValueError(f'energy_function {ef!r} needs an MLIP predictor')

    def _template(self, n):
        if n not in self.tmpl:
            if len(self.tmpl) >= 6:
                self.tmpl.pop(next(iter(self.tmpl)))
            b = collate_data_list([self.mol.clone() for _ in range(n)], max_z_prime=1)
            b.reset_sg_info(torch.full((n,), self.sg, dtype=torch.long))
            self.tmpl[n] = b
        return self.tmpl[n].clone()

    @torch.no_grad()
    def set_gas_reference(self, L1):
        """the MLIP gas-phase leg (isolated molecule), once, on one real crystal of this molecule"""
        if self.ef not in MLIPS:
            return
        b = self._template(1).to(self.dev)
        b.latent_to_cell_params(torch.as_tensor(L1, dtype=torch.float32, device=self.dev).reshape(1, -1))
        fn = b.compute_lattice_gas_phase_uma if self.ef == 'uma' else b.compute_lattice_gas_phase_mace
        v = fn(self.pred).detach().double().flatten()
        if not torch.isfinite(v).all():
            raise RuntimeError(f'{self.ef} gas-phase reference is non-finite ({v.tolist()}); refusing to cache it')
        self.gas = float(v[0])

    @torch.no_grad()
    def _eval(self, L):
        n = len(L)
        if n < self.chunk:  # pad with row 0 so the template size never changes
            L = torch.cat([L, L[:1].expand(self.chunk - n, -1)])
        b = self._template(self.chunk).to(self.dev)
        b.latent_to_cell_params(L.to(self.dev, torch.float32))
        pc_all = b.packing_coeff.double().flatten()
        cp_all = b.full_cell_parameters().double()
        hand_all = torch.as_tensor(b.aunit_handedness).double().reshape(b.num_graphs, -1)[:, 0]
        valid = torch.isfinite(pc_all)
        if self.pc_max is not None:
            valid &= pc_all <= self.pc_max
        E = torch.full((b.num_graphs,), float('inf'), dtype=torch.float64, device=self.dev)
        red = torch.full((b.num_graphs,), float('nan'), dtype=torch.float64, device=self.dev)
        vi = torch.nonzero(valid).flatten()
        if len(vi):
            bb = b if len(vi) == b.num_graphs else b.subsample_new_batch(vi)
            if self.ef == 'elj':
                bb.lj_coeff = torch.full((bb.num_graphs,), self.lj_coeff, dtype=torch.float32, device=self.dev)
            else:
                setattr(bb, f'{self.ef}_gas_pot', torch.full((bb.num_graphs,), self.gas, dtype=torch.float32,
                                                             device=self.dev))
            out = bb.analyze(['reduction_en', self.ef], cutoff=10, supercell_size=10, std_orientation=False,
                             predictor=self.pred)
            E[vi] = out[self.ef].double().flatten()
            red[vi] = out['reduction_en'].double().flatten()
            del bb, out
        res = dict(E=E[:n], red=red[:n], pc=pc_all[:n], cp=cp_all[:n], hand=hand_all[:n])
        res = {k: v.to(L.device) for k, v in res.items()}
        del b
        return res

    @torch.no_grad()
    def _rdf_eval(self, L):
        """RDF features of chart points: the per-channel normalised CDFs, flattened (fp16), and the active-channel
        mask. Distance between two crystals = (10/99) x L1(CDF_a, CDF_b) / |channels active in either| (the pipeline
        metric; analysis route: cutoff 10, rdf_cutoff 10, 100 bins, std_orientation True)."""
        n = len(L)
        if n < self.chunk:
            L = torch.cat([L, L[:1].expand(self.chunk - n, -1)])
        b = self._template(self.chunk).to(self.dev)
        b.latent_to_cell_params(L.to(self.dev, torch.float32))
        o = b.analyze(['rdf'], cutoff=10, rdf_cutoff=10, supercell_size=10, bins=100, rdf_mode=self.rdf_mode,
                      std_orientation=True)
        r = (o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf'])[:n].float()
        sm = r.sum(-1, keepdim=True)
        res = dict(C=torch.cumsum(r / (sm + 1e-10), -1).flatten(1).half(), A=(sm[..., 0] > 1e-12))
        del b, o, r
        return res

    def rdf(self, L):
        return self._chunked(self._rdf_eval, L, count=False)

    def __call__(self, L):
        return self._chunked(self._eval, L)

    def _chunked(self, fn, L, count=True):
        outs, s = [], 0
        while s < len(L):
            n = min(self.chunk, len(L) - s)
            oom = False
            try:
                outs.append(fn(L[s:s + n]))
                s += n
                self.n_ok += 1
                if self.n_ok >= self.REGROW_AFTER and self.chunk < self.max_chunk:
                    self.chunk = min(self.max_chunk, self.chunk * 2)
                    self.n_ok = 0
            except (RuntimeError, torch.OutOfMemoryError) as e:
                if not is_oom(e):
                    raise
                oom = True
            if oom:
                # outside the except block: the exception's traceback holds the failed attempt's frames (and their
                # GPU tensors) until the block ends, so emptying the cache inside it frees nothing
                gc.collect()
                torch.cuda.empty_cache()
                if self.chunk <= 1:
                    raise torch.OutOfMemoryError('CUDA out of memory at chunk 1 in the energy call')
                self.chunk = max(1, self.chunk // 2)
                self.n_oom += 1
                self.n_ok = 0
                print(f'  OOM in energy call: chunk -> {self.chunk}', flush=True)
        if count:
            self.n_eval += len(L)
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


BW = 10 / 99


def rdf_nn(C, A, blocks):
    """min RDF distance from each row of (C, A) to the rows held in `blocks` [(C_blk, A_blk), ...] (device tensors)"""
    d = torch.full((len(C),), float('inf'), device=C.device)
    if not blocks:
        return d
    Cq, Aq = C.float(), A.float()
    aq = Aq.sum(1)
    for Cb, Ab in blocks:
        union = (aq[:, None] + Ab.sum(1)[None] - Aq @ Ab.T).clamp_min(1)
        d = torch.minimum(d, (torch.cdist(Cq, Cb.float(), p=1) * BW / union).min(1).values)
    return d


class Cover:
    """A greedy RDF cover at radius r, grown in arrival order: a row joins as a representative when it is farther than
    r from every representative so far (and from those admitted earlier in the same batch). `add` returns the new-row
    mask, so the new fraction of a batch is the share of it that landed outside everything seen before."""

    def __init__(self, r, cap=200000, block=4096):
        self.r, self.cap, self.block, self.blocks, self.n, self.full = r, cap, block, [], 0, False

    def add(self, C, A, q=512):
        new = torch.zeros(len(C), dtype=torch.bool, device=C.device)
        for s in range(0, len(C), q):
            c, a = C[s:s + q], A[s:s + q].float()
            cand = torch.nonzero(rdf_nn(c, a, self.blocks) > self.r).flatten()
            if len(cand):
                cc, ca = c[cand].float(), a[cand]
                union = (ca.sum(1)[:, None] + ca.sum(1)[None] - ca @ ca.T).clamp_min(1)
                Dc = (torch.cdist(cc, cc, p=1) * BW / union).cpu()
                keep = []
                for k in range(len(cand)):
                    if all(Dc[k, j] > self.r for j in keep):
                        keep.append(k)
                kk = cand[keep]
                new[s + kk] = True
                if not self.full:
                    self._append(c[kk].half(), a[kk])
        return new

    def _append(self, c, a):
        while len(c):
            if self.n >= self.cap:
                self.full = True
                return
            if not self.blocks or len(self.blocks[-1][0]) >= self.block:
                self.blocks.append((c[:0], a[:0]))
            room = min(self.block - len(self.blocks[-1][0]), self.cap - self.n)
            Cb, Ab = self.blocks[-1]
            self.blocks[-1] = (torch.cat([Cb, c[:room]]), torch.cat([Ab, a[:room]]))
            self.n += len(c[:room])
            c, a = c[room:], a[room:]


def load_seeds(path, ef):
    """(molecule, lj_coeff, latents [n,12], stored energies in training units or None, identifiers, sg) from a prior
    file ({'prior': batch, 'thermal_scaling_factor': ...}), a crystal batch, or a list of crystals."""
    d = torch.load(path, map_location='cpu', weights_only=False)
    tsf = None
    if isinstance(d, dict):
        tsf = d.get('thermal_scaling_factor')
        pr = d['prior'] if 'prior' in d else next(v for v in d.values() if hasattr(v, 'batch_to_list') or isinstance(v, list))
    else:
        pr = d
    if isinstance(pr, list):
        pr = collate_data_list(pr)
    if ef == 'elj':
        if tsf is None:
            raise ValueError(f'{path}: an eLJ run needs the file\'s thermal_scaling_factor (the training lj_coeff)')
        lj = float(tsf)
    else:
        if tsf is not None and abs(float(tsf) - 1.0) > 1e-9:
            raise ValueError(f'{path} carries thermal_scaling_factor={tsf} on a {ef} run (mirrors train.py)')
        lj = 1.0
    sgs = torch.unique(pr.sg_ind.flatten())
    if len(sgs) != 1:
        raise ValueError(f'{path}: seeds span space groups {sgs.tolist()}; one per run')
    mol = pr.batch_to_list()[0]  # the conditioning molecule train.py uses (new_analysis.resolve_molecule)
    L = pr.latent_transform(pr.full_cell_parameters()).double()  # unclipped
    stored = None
    if ef in pr.keys():
        stored = getattr(pr, ef).double().flatten() * (lj if ef == 'elj' else 1.0)  # stored eLJ is the RAW sum
    ident = list(pr.identifier) if hasattr(pr, 'identifier') else [None] * len(L)
    return mol, lj, L, stored, ident, int(sgs[0])


def run(a, resume):
    dev = a.device
    prior = a.prior or SYSTEMS[a.system]['prior']
    ef = a.energy_function
    os.makedirs(a.out, exist_ok=True)
    mol, lj, L_raw, stored, ident, sg = load_seeds(prior, ef)
    per = sorted(set([6 + ax for ax in sg_periodic_centroid_axes(sg)] + ([6] if sg == 2 else []) + [PHI, RMAG]))
    dead = list(resolve_dead_rows(sg, True, 1))
    act = [j for j in range(DIM) if j not in dead]
    bnd = [j for j in act if j not in per]  # includes THETA: tested on the folded point
    r_floor = -0.99  # latent_to_cell_params clamps r below this; r is wrapped, so test it explicitly
    lo, hi = latent_box()
    lo, hi = lo.to(dev), hi.to(dev)
    tempers = [float(t) for t in a.tempers.split(',')]
    T_rung = torch.tensor(tempers, dtype=torch.float64, device=dev) * a.t_train
    d_act = len(act)
    cap_local = a.cap_local if a.cap_local is not None else a.cap_local_kT * a.t_train
    cap_global = a.cap_global if a.cap_global is not None else a.cap_global_kT * a.t_train
    window = a.seed_window if a.seed_window is not None else a.seed_window_kT * a.t_train
    pc_max = a.pc_max if a.pc_max is not None else (0.9 if ef in MLIPS else None)
    predictor = load_predictor(ef, a.mlip_path, dev) if ef in MLIPS else None
    en = Energy(mol, sg, ef, lj, dev, a.chunk, predictor, pc_max, rdf_mode=a.rdf_mode)
    radii = [float(x) * a.rdf_dcut for x in a.explore_radii.split(',') if x]
    ded_r = a.rdf_dcut if a.dedupe_radius is None else a.dedupe_radius
    state_path = os.path.join(a.out, 'state.pt')

    if resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=dev, weights_only=False)
        S = st['seeds']
        X, E, log_scale, Sig = st['X'], st['E'], st['log_scale'], st['Sig']
        acc_ema, s1, s2, nwin, ref = st['acc_ema'], st['s1'], st['s2'], st['nwin'], st['ref']
        step0, shard_i = st['step'] + 1, st['shard_i']
        gen = torch.Generator(device=dev)
        gen.set_state(st['gen'].cpu() if hasattr(st['gen'], 'cpu') else st['gen'])
        en.chunk = min(en.chunk, st.get('chunk', en.chunk))
        en.gas = st.get('gas')
        effort0 = st.get('effort', dict(walker_steps=0, evals=0, wall=0.0))
        if ef in MLIPS and en.gas is None:
            en.set_gas_reference(S['L'][0].cpu())
        print(f'resumed at step {step0} (shard {shard_i}), {len(X)} chains, chunk {en.chunk}', flush=True)
    else:
        # a fresh start owns the directory: drop old shards AND any old resume point, so an OOM restart before this
        # run's first checkpoint can never resume someone else's chains
        for f in glob.glob(os.path.join(a.out, 'shard_*.pt')) + [state_path, state_path + '.tmp']:
            if os.path.exists(f):
                os.remove(f)
        gen = torch.Generator(device=dev).manual_seed(a.rng + 1000 * a.shard)
        L0 = wrap(L_raw, per)
        out_box = ((L0 < lo.cpu()) | (L0 > hi.cpu())).any(1)
        dev_clip = (L0 - L0.clamp(lo.cpu(), hi.cpu())).abs().max(1).values
        L0 = L0.clamp(lo.cpu(), hi.cpu())
        en.set_gas_reference(L0[0])
        print(f'{ef}: prior {os.path.basename(prior)}, sg {sg}, lj_coeff {lj:.6f}, periodic dims {per}, dead {dead}, '
              f'active {act}; caps: local {cap_local:.2f}, global {cap_global:.2f}, seed window {window:.2f} (kT '
              f'{a.t_train}); pc_max {pc_max}; gas reference {en.gas}', flush=True)
        # the seed window and ranking: from STORED energies when the file carries them (every shard sees the same
        # window without re-scoring all seeds), else from a fresh pass over every seed
        if stored is not None:
            Er = stored.clone()
        else:
            print('  no stored energies: scoring every seed', flush=True)
            Er = en(L0.to(dev))['E'].cpu()
        fin = torch.isfinite(Er)
        Emin_rank = float(Er[fin].min())
        keep = torch.nonzero(fin & (Er <= Emin_rank + window)).flatten()
        keep = keep[torch.argsort(Er[keep])]
        # THIN THE MINIMA, THEN ASSIGN WALKERS: a greedy RDF cover in ascending energy at the same-packing radius keeps
        # the lowest member of each group of near-duplicates (a prior file re-finds the same crystal many times and in
        # several pose copies; their walkers would re-walk one band). Deterministic, so every shard gets the same list.
        n_window = len(keep)
        if ded_r > 0:
            fk = en.rdf(L0[keep].to(dev))
            keep = keep[Cover(ded_r, cap=len(keep) + 1).add(fk['C'], fk['A']).cpu()]
            del fk
        print(f'  minima in the window: {n_window}; distinct at RDF radius {ded_r:g} ({a.rdf_mode}): {len(keep)}',
              flush=True)
        idx = keep[a.shard::a.n_shards]
        if a.max_seeds and len(idx) > a.max_seeds:
            pick = torch.randperm(len(idx), generator=torch.Generator().manual_seed(a.rng))[:a.max_seeds]
            idx = idx[pick.sort().values]
        r0 = en(L0[idx].to(dev))
        E0 = r0['E'].cpu()
        ok0 = torch.isfinite(E0)
        # fresh vs stored: the same route should reproduce the file's energies; a constant offset is absorbed into
        # the global reference, a SPREAD means a different model, molecule or route
        off, spread = 0.0, 0.0
        if stored is not None:
            dif = (E0 - Er[idx])[ok0]
            off, spread = float(dif.median()), float((dif - dif.median()).abs().median())
            print(f'  fresh vs stored energy over this shard\'s seeds: median offset {off:+.4g}, median |deviation| '
                  f'{spread:.4g}, max |diff| {float(dif.abs().max()):.4g}', flush=True)
            if spread > a.max_spread:
                raise RuntimeError(f'fresh energies deviate from the stored ones by a median {spread:.3g} beyond a '
                                   f'constant offset (> --max_spread {a.max_spread}): wrong model, molecule or route?')
        E_glob = Emin_rank + off
        idx, E0 = idx[ok0], E0[ok0]
        r0 = {k: v[ok0.to(v.device)] for k, v in r0.items()}
        print(f'  seeds: {len(L_raw)} rows, {int(out_box.sum())} outside the clamp box (max excursion '
              f'{float(dev_clip.max()):.4f}; clipped); window {window:.2f} above {Emin_rank:.3f} holds {len(keep)}; '
              f'shard {a.shard}/{a.n_shards} uses {len(idx)}', flush=True)
        S = dict(row=idx, L=L0[idx].to(dev), E=E0.to(dev), E_glob=E_glob,
                 cp=r0['cp'], hand=r0['hand'], pc=r0['pc'], red=r0['red'])
        torch.save(dict(row=idx.numpy(), identifier=[ident[i] for i in idx.tolist()],
                        latent=S['L'].float().cpu().numpy(), cell_params=S['cp'].float().cpu().numpy(),
                        handedness=S['hand'].float().cpu().numpy(), E=S['E'].cpu().numpy(),
                        E_stored=(Er[idx].numpy() if stored is not None else None),
                        pc=S['pc'].float().cpu().numpy(), reduction_en=S['red'].float().cpu().numpy(),
                        E_glob=E_glob, lj_coeff=lj, clipped=out_box[idx].numpy()),
                   os.path.join(a.out, 'seeds.pt'))
        R = len(tempers)
        n_seed = len(idx)
        S['chain_seed'] = torch.arange(n_seed, device=dev).repeat_interleave(R * a.replicas)
        S['chain_rung'] = torch.arange(R, device=dev).repeat_interleave(a.replicas).repeat(n_seed)
        effort0 = dict(walker_steps=0, evals=0, wall=0.0)
        X = S['L'][S['chain_seed']].clone()
        E = S['E'][S['chain_seed']].clone()
        N = len(X)
        log_scale = torch.zeros(N, dtype=torch.float64, device=dev)
        Sig = torch.eye(d_act, dtype=torch.float64, device=dev).expand(N, -1, -1) * a.init_step ** 2
        acc_ema = torch.full((N,), a.acc_target, dtype=torch.float64, device=dev)
        s1 = torch.zeros(N, d_act, dtype=torch.float64, device=dev)
        s2 = torch.zeros(N, d_act, d_act, dtype=torch.float64, device=dev)
        nwin, ref = 0, X[:, act].clone()
        step0, shard_i = 0, 0
        mlip = None
        if ef in MLIPS:
            mlip = dict(path=a.mlip_path, bytes=os.path.getsize(a.mlip_path))
        meta = dict(vars(a), prior=prior, sg=sg, periodic_dims=per, dead_rows=dead, active_dims=act, lj_coeff=lj,
                    E_glob=E_glob, E_min_stored_or_ranked=Emin_rank, fresh_minus_stored=off, n_seeds=n_seed,
                    n_seeds_in_window=n_window, n_distinct_minima=len(keep), dedupe_radius=ded_r, n_chains=N, T_rungs=T_rung.tolist(), cap_local_abs=cap_local,
                    cap_global_abs=cap_global, seed_window_abs=window, pc_max_used=pc_max, gas_reference=en.gas,
                    mlip=mlip, energy=f'{ef}; analyze([reduction_en, {ef}], cutoff=10, supercell_size=10, '
                                      f'std_orientation=False); molecule = first crystal of the file',
                    started=time.strftime('%Y-%m-%d %H:%M:%S'))
        json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)

    N = len(X)
    cs, cr = S['chain_seed'], S['chain_rung']
    E_sp = S['E'][cs]
    cap = torch.minimum(E_sp + cap_local, torch.full_like(E_sp, S['E_glob'] + cap_global))
    T_c = T_rung[cr]
    cov_updates = sorted(int(s) for s in a.cov_updates.split(',') if s)
    buf = []
    chol = torch.linalg.cholesky(Sig + 1e-12 * torch.eye(d_act, dtype=torch.float64, device=dev))

    def flush(step):
        nonlocal buf, shard_i
        if buf:
            cat = {k: torch.cat([b[k] for b in buf]).cpu().numpy() for k in buf[0]}
            torch.save(cat, os.path.join(a.out, f'shard_{shard_i:04d}.pt'))
            shard_i += 1
            buf = []
        save_atomic(dict(seeds=S, X=X, E=E, log_scale=log_scale, Sig=Sig, acc_ema=acc_ema, s1=s1, s2=s2, nwin=nwin,
                         ref=ref, step=step, shard_i=shard_i, gen=gen.get_state(), chunk=en.chunk, gas=en.gas,
                         effort=effort()), state_path)

    # EXPLORATION: a greedy RDF cover per radius, seeded with the minima; at every checkpoint the walkers' current
    # states are added and the share landing farther than r from everything seen before is the new fraction.
    # exploration % = 1 - new fraction: the Good-Turing-style estimate of how much of what the walkers now visit was
    # already covered at that resolution. One jsonl row per checkpoint against walker-steps, evaluations and wall time.
    covers = [Cover(r) for r in radii]
    fs = en.rdf(S['L'])
    for cv in covers:
        cv.add(fs['C'], fs['A'])
    del fs
    xlog = os.path.join(a.out, 'exploration.jsonl')
    t_start = time.time()

    def effort():
        return dict(walker_steps=effort0['walker_steps'] + N * max(0, cur_step[0] + 1 - step0),
                    evals=effort0['evals'] + en.n_eval - n_eval0, wall=effort0['wall'] + time.time() - t_start)

    def explore(step):
        ft = en.rdf(fold(X, per))
        row = dict(step=step, **effort(), resumed=bool(step0 > 0), n_walkers=N)
        parts = []
        for cv in covers:
            nf = float(cv.add(ft['C'], ft['A']).float().mean())
            row[f'new_{cv.r:.4g}'] = nf
            row[f'reps_{cv.r:.4g}'] = cv.n
            row[f'full_{cv.r:.4g}'] = cv.full
            parts.append(f'r {cv.r:.3g}: {100 * (1 - nf):5.1f}% ({cv.n}{" FULL" if cv.full else ""})')
        del ft
        with open(xlog, 'a') as fh:
            print(json.dumps(row), file=fh)
        print(f'  exploration at step {step} ({row["walker_steps"]} walker-steps, {row["evals"]} evals, '
              f'{row["wall"]:.0f} s): ' + ' | '.join(parts), flush=True)

    cur_step = [step0 - 1]
    n_eval0 = en.n_eval
    if step0 == 0:
        flush(-1)  # this run's own resume point exists before the first step, so an early restart resumes THIS run

    t0 = time.time()
    tot = dict(prop=0, box=0, cap=0, nonfin=0, acc=0)
    for step in range(step0, a.steps):
        cur_step[0] = step
        z = torch.randn(N, d_act, generator=gen, device=dev, dtype=torch.float64)
        dx = torch.exp(log_scale)[:, None] * torch.einsum('nij,nj->ni', chol, z)
        Xn = X.clone()  # unfolded chain coordinates
        Xn[:, act] += dx
        Xn = wrap(Xn, per)
        Xe = fold(Xn, per)  # the chart point of the same crystal: box test, energy and record all use it
        ok = ((Xe[:, bnd] >= lo[bnd]) & (Xe[:, bnd] <= hi[bnd])).all(1) & (Xe[:, RMAG] >= r_floor)
        if sg in (1, 2):
            ok &= triclinic_ok(Xe)
        En = torch.full((N,), float('inf'), dtype=torch.float64, device=dev)
        rn = None
        ii = torch.nonzero(ok).flatten()
        if len(ii):
            rn = en(Xe[ii])
            En[ii] = rn['E']
        fin = torch.isfinite(En)
        under = En <= cap
        if a.red_max is not None and rn is not None:  # keep the chain inside the reduced-cell domain
            red_n = torch.zeros(N, dtype=torch.float64, device=dev)
            red_n[ii] = rn['red'].double()
            tot['red'] = tot.get('red', 0) + int((ok & fin & under & (red_n > a.red_max)).sum())
            under = under & (red_n <= a.red_max)
        u = torch.rand(N, generator=gen, device=dev, dtype=torch.float64)
        acc = ok & fin & under & (torch.log(u) < -(En - E) / T_c)
        tot['prop'] += N; tot['box'] += int((~ok).sum()); tot['nonfin'] += int((ok & ~fin).sum())
        tot['cap'] += int((ok & fin & ~under).sum()); tot['acc'] += int(acc.sum())
        if acc.any():
            X[acc] = Xn[acc]
            E[acc] = En[acc]
            pos = torch.full((N,), -1, dtype=torch.long, device=dev)
            pos[ii] = torch.arange(len(ii), device=dev)
            j = pos[acc]
            buf.append(dict(latent=Xe[acc].float(), cell_params=rn['cp'][j].float(), handedness=rn['hand'][j].float(),
                            E=En[acc], dE_spawn=(En[acc] - E_sp[acc]).float(), pc=rn['pc'][j].float(),
                            reduction_en=rn['red'][j].float(), chain=torch.nonzero(acc).flatten().int(),
                            seed=cs[acc].int(), rung=cr[acc].to(torch.int8),
                            step=torch.full((int(acc.sum()),), step, dtype=torch.int32, device=dev)))
        # adaptation (frozen from adapt_end on)
        acc_ema = 0.9 * acc_ema + 0.1 * acc.double()
        if step < a.adapt_end:
            log_scale = (log_scale + a.adapt_gain * (acc_ema - a.acc_target)).clamp(-12, 4)
            dl = X[:, act] - ref
            pa = [k for k, j in enumerate(act) if j in per]
            dl[:, pa] = torch.remainder(dl[:, pa] + 1, 2) - 1
            s1 += dl
            s2 += dl[:, :, None] * dl[:, None, :]
            nwin += 1
            if step + 1 in cov_updates and nwin > 10:
                m = s1 / nwin
                C = s2 / nwin - m[:, :, None] * m[:, None, :]
                C = C + (1e-4 ** 2) * torch.eye(d_act, dtype=torch.float64, device=dev)
                # the Haario factor goes into Sig and log_scale is re-zeroed, so the step after an update is the
                # optimal-scale guess for the chain's own covariance; log_scale re-adapts from there only if adaptation
                # continues after this update, hence the last update belongs well before adapt_end
                Sig = 0.5 * Sig * torch.exp(2 * log_scale)[:, None, None] + 0.5 * (2.38 ** 2 / d_act) * C
                log_scale = torch.zeros_like(log_scale)
                chol = torch.linalg.cholesky(Sig + 1e-12 * torch.eye(d_act, dtype=torch.float64, device=dev))
                s1.zero_(); s2.zero_(); nwin = 0; ref = X[:, act].clone()
                print(f'  step {step}: proposal covariance re-estimated per chain', flush=True)
        if step % a.log_every == 0 or step == a.steps - 1:
            dE = E - E_sp
            msg = ' | '.join(
                f'T{tempers[r]:g}: acc {float(acc_ema[cr == r].mean()):.2f} dE med {float(dE[cr == r].median()):5.1f} '
                f'q90 {float(dE[cr == r].quantile(0.9)):5.1f} step {float(torch.exp(log_scale[cr == r]).median() * torch.sqrt(torch.diagonal(Sig[cr == r], dim1=1, dim2=2)).median()):.3g}'
                for r in range(len(tempers)))
            rate = (en.n_eval - n_eval0) / max(time.time() - t0, 1e-9)
            print(f'step {step:5d} {time.time() - t0:7.0f}s  {rate:6.0f} evals/s  chunk {en.chunk}  '
                  f'rejected: box {tot["box"] / max(tot["prop"], 1):.3f} cap {tot["cap"] / max(tot["prop"], 1):.3f} '
                  f'nonfinite {tot["nonfin"]}  | {msg}', flush=True)
        if (step + 1) % a.shard_every == 0 or step == a.steps - 1:
            flush(step)
            explore(step)
    print(f'done: {en.n_eval - n_eval0} MC evaluations, {tot["acc"]} accepted moves, {time.time() - t0:.0f} s, '
          f'{en.n_oom} OOM halvings', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--prior', default=None, help='prior file, crystal batch or list of crystals (the seeds)')
    p.add_argument('--system', default=None, choices=list(SYSTEMS), help='local ELJ shortcut for --prior')
    p.add_argument('--energy_function', default='elj', choices=['elj', 'uma', 'mace'])
    p.add_argument('--mlip_path', default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--adapt_end', type=int, default=120)
    p.add_argument('--cov_updates', default='40,80')
    p.add_argument('--t_train', type=float, default=2.5)
    p.add_argument('--tempers', default='1.0', help='one walker per seed per rung; T = temper x t_train')
    p.add_argument('--cap_local_kT', type=float, default=10.0, help='hard cap above the spawning seed, in kT_train')
    p.add_argument('--cap_global_kT', type=float, default=14.0, help='hard cap above the lowest seed, in kT_train')
    p.add_argument('--seed_window_kT', type=float, default=6.0, help='seeds within this of the lowest, in kT_train')
    p.add_argument('--cap_local', type=float, default=None, help='absolute override (training units)')
    p.add_argument('--cap_global', type=float, default=None, help='absolute override (training units)')
    p.add_argument('--seed_window', type=float, default=None, help='absolute override (training units)')
    p.add_argument('--pc_max', type=float, default=None, help='reject denser proposals unscored (default 0.9 for MLIPs)')
    p.add_argument('--red_max', type=float, default=None,
                   help='reject a proposal whose reduction penalty (reduction_en) exceeds this; default: not used')
    p.add_argument('--max_spread', type=float, default=1.0,
                   help='max median |fresh - stored - offset| over the seeds before refusing (training units)')
    p.add_argument('--replicas', type=int, default=2, help='walkers per distinct minimum per temperature rung')
    p.add_argument('--rdf_mode', default='atomwise', choices=['atomwise', 'envwise'])
    p.add_argument('--rdf_dcut', type=float, default=0.12,
                   help='same-packing RDF radius (COMPACK-calibrated per system): seed thinning and exploration unit')
    p.add_argument('--dedupe_radius', type=float, default=None,
                   help='thin the minima at this RDF radius before assigning walkers (default --rdf_dcut; 0 = off)')
    p.add_argument('--explore_radii', default='1,2,3', help='exploration radii, in units of --rdf_dcut')
    p.add_argument('--n_shards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--max_seeds', type=int, default=0, help='0 = every seed of this shard')
    p.add_argument('--chunk', type=int, default=256)
    p.add_argument('--init_step', type=float, default=0.01)
    p.add_argument('--acc_target', type=float, default=0.3)
    p.add_argument('--adapt_gain', type=float, default=0.1)
    p.add_argument('--shard_every', type=int, default=50)
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--rng', type=int, default=0)
    p.add_argument('--max_restarts', type=int, default=10)
    p.add_argument('--resume', action='store_true', help='continue from <out>/state.pt when it exists')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if not (a.prior or a.system):
        p.error('give --prior (or a local --system shortcut)')
    if a.energy_function in MLIPS and not a.mlip_path:
        p.error(f'--energy_function {a.energy_function} needs --mlip_path')
    if not 0 <= a.shard < a.n_shards:
        p.error('need 0 <= --shard < --n_shards')
    for attempt in range(a.max_restarts + 1):
        escaped = False
        try:
            run(a, resume=a.resume or attempt > 0)
            return
        except (RuntimeError, torch.OutOfMemoryError) as e:
            if not is_oom(e) or attempt == a.max_restarts:
                raise
            escaped = True
        if escaped:  # cleanup outside the except block, where the failed frames are released
            a.chunk = max(1, a.chunk // 2)
            gc.collect()
            torch.cuda.empty_cache()
            print(f'OOM escaped the chunked energy call (restart {attempt + 1}/{a.max_restarts}); resuming from '
                  f'state.pt with chunk {a.chunk}', flush=True)


if __name__ == '__main__':
    main()
