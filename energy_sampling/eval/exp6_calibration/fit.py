"""Fit exp-6 to the MLIP (docs/design/exp6_calibration_plan.md, step 4).

Parameters: the FIT set's per-type (A, B, C) for the types the frames contain (C, HC, HN, HO, N, O, F), as log A, B,
log C, and one scale on the MMFF partial charges (log s). Cl and S stay at FIT. Every energy is evaluated through
``minima.StageEnergy`` with the 'exp6' target -- the search's own path -- with the parameter tables built from the
current values (``exp6_ff.pair_tables`` is redirected for one registered name; each pair's splice radius is computed
at the current values and held fixed within a step).

Loss, per minibatch of molecules (fit-split molecules only), each term within a molecule:
    energy    weighted mean of (d - weighted mean of d)^2, d = E_exp6 - E_mlip, weight
              max(exp(-(E_mlip - lowest E_mlip of the molecule) / T_W), W_FLOOR): the low-energy region counts most
              and a molecule's constant offset (its gas leg) cancels
    gradient  same weights, mean over cell coordinates of ((g_exp6 - g_mlip) / sigma_k)^2, sigma_k the robust spread
              of the MLIP gradient in coordinate k over the fit frames (coordinates the cell clean holds are left out)
    order     over pairs of the molecule's MLIP-minimum frames more than 0.5 kJ/mol apart: softplus of the exp-6
              difference against the MLIP order, in units of TAU_RANK
    prior     ((theta - theta_FIT) / scale)^2, so a type the data barely constrain stays near FIT
Each move from FIT is bounded smoothly (BOUND). After every step, the types in any pair that lost its repulsive wall
or its well (pair_ok, the checks a parameter set must pass) are put back to their values before the step (wall_safe);
the other types keep their step and the learning rate never changes.
Frames are dropped when not labelled, when their shortest intermolecular contact is below common.MIN_CONTACT_A (MLIP
holes), or when more than E_MAX above their molecule's lowest MLIP frame.

Molecules are split once (seeded) into fit and validation (--val_frac); the validation readout, for FIT and for the
current values, is written at every --eval_every steps; the step (FIT included) with the lowest validation SELECT
(the RMS energy residual below 10 kJ/mol, the region the search surrogate must get right) is kept.

usage: python -m eval.exp6_calibration.fit FRAMES_PT OUT_JSON [--steps 1500] [--lr 0.02] [--molecules 6]
           [--frames 16] [--w_grad 0.1] [--w_rank 1.0] [--w_prior 0.01] [--val_frac 0.15] [--device cuda]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from eval.exp6_calibration import common as C

NAME = 'exp6_calibration_fitting'
FITTED = ('C', 'HC', 'HN', 'HO', 'N', 'O', 'F')
T_W, W_FLOOR, E_MAX = 10.0, 0.02, 80.0
TAU_RANK, RANK_MIN_GAP = 1.0, 0.5
PRIOR_SCALE = dict(logA=0.5, B=0.3, logC=0.5, logs=0.3)
# largest move from FIT: A and C within a factor e, B within 0.5 / A, the charge scale within a factor e^0.5. Without a
# bound the prototype (2026-11-25) drove one pair to A 1832, B 4.75, C 21.4 by step ~110: no repulsive wall
BOUND = dict(logA=1.0, B=0.5, logC=1.0, logs=0.5)
# the readout that picks the kept step; the all-frames weighted RMS rose while this fell in the prototype (2026-11-25)
SELECT = 'low_rms_kJ'


class Params(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from mxtaltools.analysis import exp6_ff as ff
        base = ff.PARAMETER_SETS['fit']['types']
        self.names = list(base)
        self.el = [base[k][0] for k in self.names]
        A, B, Cc = (torch.tensor([float(base[k][j]) for k in self.names], dtype=torch.float64) for j in (1, 2, 3))
        self.register_buffer('logA0', A.log())
        self.register_buffer('B0', B)
        self.register_buffer('logC0', Cc.log())
        self.free = torch.tensor([k in FITTED for k in self.names])
        n = len(self.names)
        self.dlogA = torch.nn.Parameter(torch.zeros(n, dtype=torch.float64))
        self.dB = torch.nn.Parameter(torch.zeros(n, dtype=torch.float64))
        self.dlogC = torch.nn.Parameter(torch.zeros(n, dtype=torch.float64))
        self.dlogs = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def deltas(self):
        """Moves from FIT, each bounded smoothly to +-BOUND (b tanh(raw / b): the identity near FIT), zero for the
        types held at FIT."""
        m = self.free.to(self.dlogA.dtype)
        b = lambda raw, k: BOUND[k] * torch.tanh(raw / BOUND[k])  # noqa: E731
        return m * b(self.dlogA, 'logA'), m * b(self.dB, 'B'), m * b(self.dlogC, 'logC'), b(self.dlogs, 'logs')

    def values(self):
        dA, dB, dC, ds = self.deltas()
        return self.logA0 + dA, self.B0 + dB, self.logC0 + dC, ds

    def prior(self):
        dA, dB, dC, ds = self.deltas()
        return ((dA / PRIOR_SCALE['logA']) ** 2).sum() + ((dB / PRIOR_SCALE['B']) ** 2).sum() \
            + ((dC / PRIOR_SCALE['logC']) ** 2).sum() + (ds / PRIOR_SCALE['logs']) ** 2

    def tables(self, device, dtype):
        """exp6_ff tables for the 'exp6' form, differentiable in the parameters at fixed splice radii."""
        from mxtaltools.analysis import exp6_ff as ff
        logA, B, logC, _ = self.values()
        Aij = torch.exp(0.5 * (logA[:, None] + logA[None, :]))
        Bij = 0.5 * (B[:, None] + B[None, :])
        Cij = torch.exp(0.5 * (logC[:, None] + logC[None, :]))
        n = len(self.names)
        rs = torch.tensor([[ff._inner_inflection(float(Aij[i, j]), float(Bij[i, j]), float(Cij[i, j]))
                            for j in range(n)] for i in range(n)], dtype=torch.float64)
        Es = Aij * torch.exp(-Bij * rs) - Cij / rs ** 6
        dEs = -Aij * Bij * torch.exp(-Bij * rs) + 6 * Cij / rs ** 7
        out = {k: v.to(device=device, dtype=dtype) for k, v in dict(A=Aij, B=Bij, C=Cij, r_s=rs, E_s=Es, dE_s=dEs).items()}
        out['names'], out['h_shift'] = self.names, 0.0
        return out

    def spec(self):
        logA, B, logC, logs = (v.detach() for v in self.values())
        return dict(format='exp6_params/1', typing='fit', h_shift=0.0, charge_scale=float(torch.exp(logs)),
                    types={k: [self.el[i], float(torch.exp(logA[i])), float(B[i]), float(torch.exp(logC[i]))]
                           for i, k in enumerate(self.names)})


def pair_ok(A, B, C):
    """The checks exp6_ff applies to every pair of a parameter set: a repulsive wall (an inner inflection) and a well
    (an exp-6 minimum beyond it)."""
    from mxtaltools.analysis import exp6_ff as ff
    try:
        ff._exp6_minimum(A, B, C, ff._inner_inflection(A, B, C))
        return True
    except ValueError:
        return False


@torch.no_grad()
def failing_types(params):
    """Indices of the types in any pair that fails pair_ok at the current values."""
    logA, B, logC, _ = (v.double() for v in params.values())
    n, bad = len(params.names), set()
    for i in range(n):
        for j in range(i, n):
            if not pair_ok(float(torch.exp(0.5 * (logA[i] + logA[j]))), float(0.5 * (B[i] + B[j])),
                           float(torch.exp(0.5 * (logC[i] + logC[j])))):
                bad |= {i, j}
    return bad


@torch.no_grad()
def wall_safe(params, before):
    """After an optimiser step: while any pair fails pair_ok, put the (A, B, C) of the types in failing pairs back to
    ``before`` (the values before the step, where every pair passed). Ends within one pass per type, since with every
    type put back the step is undone. Returns the names of the types held back."""
    held = set()
    for _ in range(len(params.names) + 1):
        bad = failing_types(params)
        if not bad:
            break
        for i in bad:
            for v, b in zip((params.dlogA, params.dB, params.dlogC), before[:3]):
                v[i] = b[i]
        held |= bad
    return sorted(params.names[i] for i in held)


def install(params, device):
    """Register NAME and point exp6_ff.pair_tables at the live tables for it."""
    from mxtaltools.analysis import exp6_ff as ff
    ff.register_parameter_set(NAME, dict(typing='fit', h_shift=0.0,
                                         types={k: list(v) for k, v in ff.PARAMETER_SETS['fit']['types'].items()}))
    orig = ff.pair_tables
    state = {}

    def pair_tables(p, device='cpu', dtype=torch.float32):
        return state['tables'] if p == NAME else orig(p, device=device, dtype=dtype)

    ff.pair_tables = pair_tables

    def refresh():
        state['tables'] = params.tables(device, torch.float32)
    return refresh


class Frames:
    def __init__(self, path, device, val_frac):
        from mxtaltools.analysis import exp6_ff as ff
        from mxtaltools.dataset_utils.utils import collate_data_list
        d = torch.load(path, weights_only=False)
        fr, self.templates = d['frames'], d['templates']
        e = fr['e_mlip'].numpy()
        ok = np.isfinite(e) & np.isfinite(fr['e_fit'].numpy()) & torch.isfinite(fr['g_mlip']).all(1).numpy() \
            & (fr['contact'].numpy() >= C.MIN_CONTACT_A)
        names = np.array(fr['name'])
        for n in set(names[ok]):
            k = ok & (names == n)
            ok &= ~((names == n) & (e > e[k].min() + E_MAX))
        self.idx = np.flatnonzero(ok)
        self.name, self.source = names, np.array(fr['source'])
        self.params, self.hand = fr['params'], fr['hand']
        self.e_mlip, self.g_mlip, self.e_fit = fr['e_mlip'], fr['g_mlip'], fr['e_fit']
        self.emin = {n: float(e[self.idx][names[self.idx] == n].min()) for n in set(names[self.idx])}
        mols = sorted(self.emin)
        rng = np.random.default_rng(C.SEED)
        val = set(rng.choice(mols, max(1, int(round(val_frac * len(mols)))), replace=False).tolist())
        self.fit_mols, self.val_mols = [m for m in mols if m not in val], sorted(val)
        g = self.g_mlip[self.idx].numpy()
        q75, q25 = np.percentile(g, [75, 25], axis=0)
        self.sigma = torch.tensor(np.maximum((q75 - q25) / 1.349, 1e-12))
        self.gmask = torch.tensor((q75 - q25) / 1.349 > 1e-6)
        self.device = device
        self.q = {}
        for n in mols:  # MMFF charges of each molecule, in its atom order
            self.q[n] = ff.batch_charges(collate_data_list([self.templates[n].clone()]), 'mmff').cpu()
        print(f'{len(self.idx)} usable frames of {len(e)}; {len(self.fit_mols)} fit, {len(self.val_mols)} validation '
              f'molecules; gradient coordinates used {int(self.gmask.sum())}/12', flush=True)

    def of(self, mol, kind=None):
        k = self.idx[self.name[self.idx] == mol]
        if kind is not None:
            k = k[np.char.endswith(self.source[k].astype(str), kind)]
        return k


def evaluate(frames, idx, params, refresh, create_graph):
    """exp-6 energy [n] (kJ/mol per molecule) and gradient [n, 12] at frames ``idx``, through StageEnergy."""
    from mxtaltools.crystal_search import minima as mn
    refresh()
    s = torch.exp(params.values()[3])
    rows, q = [], []
    for i in idx:
        n = frames.name[i]
        rows += C.rows_at(frames.templates[n], frames.params[[i]], frames.hand[[i]])
        q.append(frames.q[n])
    charges = (torch.cat(q).double() * s).float().to(frames.device)
    se = mn.StageEnergy(rows, 'exp6', 1.0, exp6_kwargs={'params': NAME, 'charges': charges}, device=frames.device,
                        chunk=len(rows))
    x = frames.params[idx].float().to(frames.device).requires_grad_(True)
    e = se._energy(x, list(range(len(rows)))) * se.kT
    g, = torch.autograd.grad(e.sum(), x, create_graph=create_graph)
    return e.double().cpu() if not create_graph else e.double(), g.double().cpu() if not create_graph else g.double()


def molecule_terms(frames, idx, e, g, rank_idx=None, e_rank=None):
    """(energy term, gradient term, order term) of one molecule's frames."""
    dev = e.device
    em = frames.e_mlip[idx].to(dev)
    w = torch.clamp(torch.exp(-(em - frames.emin[frames.name[idx[0]]]) / T_W), min=W_FLOOR)
    d = e - em
    dbar = (w * d).sum() / w.sum()
    le = (w * (d - dbar) ** 2).sum() / w.sum()
    m = frames.gmask.to(dev)
    gg = ((g - frames.g_mlip[idx].to(dev)) / frames.sigma.to(dev))[:, m]
    lg = (w[:, None] * gg ** 2).sum() / (w.sum() * gg.shape[1])
    lr = torch.zeros((), dtype=e.dtype, device=dev)
    if rank_idx is not None and len(rank_idx) > 1:
        a = frames.e_mlip[rank_idx].to(dev)
        da = a[:, None] - a[None, :]
        df = e_rank[:, None] - e_rank[None, :]
        pair = da > RANK_MIN_GAP
        if pair.any():
            lr = torch.nn.functional.softplus(-df[pair] / TAU_RANK).mean()
    return le, lg, lr


@torch.no_grad()
def readout(frames, mols, params, refresh):
    """Validation metrics: weighted within-molecule RMS of the energy residual (kJ/mol), RMS of it below 10 kJ/mol,
    normalised gradient RMS, and the fraction of MLIP-minimum pairs (> 0.5 kJ/mol apart) in the MLIP's order."""
    res = dict(w_rms=[], low_rms=[], grad_rms=[], order_right=0, order_pairs=0)
    for mol in mols:
        idx = frames.of(mol)
        if not len(idx):
            continue
        with torch.enable_grad():
            e, g = evaluate(frames, idx, params, refresh, create_graph=False)
        em = frames.e_mlip[idx]
        w = torch.clamp(torch.exp(-(em - frames.emin[mol]) / T_W), min=W_FLOOR)
        d = e - em
        d = d - (w * d).sum() / w.sum()
        res['w_rms'].append(float(torch.sqrt((w * d ** 2).sum() / w.sum())))
        low = (em - frames.emin[mol]) < 10.0
        if low.any():
            res['low_rms'].append(float(torch.sqrt(((d[low] - d[low].mean()) ** 2).mean())))
        m = frames.gmask
        res['grad_rms'].append(float(torch.sqrt((((g - frames.g_mlip[idx]) / frames.sigma)[:, m] ** 2).mean())))
        mins = [j for j, i in enumerate(idx) if str(frames.source[i]).endswith('_min')]
        for a in range(len(mins)):
            for b in range(a + 1, len(mins)):
                da = float(em[mins[a]] - em[mins[b]])
                if abs(da) > RANK_MIN_GAP:
                    res['order_pairs'] += 1
                    res['order_right'] += int(np.sign(da) == np.sign(float(e[mins[a]] - e[mins[b]])))
    return dict(w_rms_kJ=float(np.median(res['w_rms'])) if res['w_rms'] else None,
                low_rms_kJ=float(np.median(res['low_rms'])) if res['low_rms'] else None,
                grad_rms=float(np.median(res['grad_rms'])) if res['grad_rms'] else None,
                order_right=res['order_right'] / max(res['order_pairs'], 1), order_pairs=res['order_pairs'])


def check_path(frames, params, refresh, n=8):
    """At the FIT values the fit's path must reproduce the labelled FIT exp-6 energies (StageEnergy with params
    'fit' and MMFF charges): the redirected tables and the charge tensor change nothing else."""
    idx = frames.idx[:n]
    e, _ = evaluate(frames, idx, params, refresh, create_graph=False)
    worst = float((e - frames.e_fit[idx]).abs().max())
    print(f'path check at FIT: worst |E - labelled FIT exp-6| {worst:.2e} kJ/mol over {len(idx)} frames', flush=True)
    if not worst < 1e-2:
        raise SystemExit('the fit path does not reproduce the FIT exp-6 energies')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames')
    ap.add_argument('out_json')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--molecules', type=int, default=6, help='molecules per step')
    ap.add_argument('--frames_per', type=int, default=16, help='random frames per molecule per step')
    ap.add_argument('--w_grad', type=float, default=0.1)
    ap.add_argument('--w_rank', type=float, default=1.0)
    ap.add_argument('--w_prior', type=float, default=0.01)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    if os.path.exists(a.out_json):
        raise SystemExit(f'{a.out_json} exists; choose a new name')
    if a.device == 'cuda':
        from eval.exp6_recall.mirror_test import cap_gpu
        cap_gpu()
    torch.manual_seed(C.SEED)
    params = Params()
    refresh = install(params, a.device)
    frames = Frames(a.frames, a.device, a.val_frac)
    check_path(frames, params, refresh)
    opt = torch.optim.Adam(params.parameters(), lr=a.lr)
    rng = np.random.default_rng(C.SEED)
    fit0 = readout(frames, frames.val_mols, params, refresh)
    print(f'validation at FIT: {fit0}', flush=True)
    # FIT itself is a candidate: if no step beats it on validation, the file is FIT (best step 0)
    best = (fit0[SELECT] if fit0[SELECT] is not None else math.inf, params.spec(), 0)
    history, t0, cut = [], time.time(), [0, {}]
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        le = lg = lr = 0.0
        for mol in rng.choice(frames.fit_mols, min(a.molecules, len(frames.fit_mols)), replace=False):
            pool = frames.of(mol)
            pick = rng.choice(pool, min(a.frames_per, len(pool)), replace=False)
            mins = frames.of(mol, '_min')
            both = np.concatenate([pick, mins])
            e, g = evaluate(frames, both, params, refresh, create_graph=True)
            a_, b_, c_ = molecule_terms(frames, both[:len(pick)], e[:len(pick)], g[:len(pick)],
                                        mins if len(mins) > 1 else None, e[len(pick):])
            le, lg, lr = le + a_, lg + b_, lr + c_
        k = min(a.molecules, len(frames.fit_mols))
        loss = (le + a.w_grad * lg + a.w_rank * lr) / k + a.w_prior * params.prior()
        loss.backward()
        if step == 1:  # the parameters must reach the energy: a silent detach would fit nothing
            dead = [k for k, v in (('A', params.dlogA), ('B', params.dB), ('C', params.dlogC), ('charge scale', params.dlogs))
                    if v.grad is None or not bool(v.grad.abs().sum() > 0)]
            if dead:
                raise SystemExit(f'no gradient reaches {dead}')
        before = [v.detach().clone() for v in (params.dlogA, params.dB, params.dlogC)]
        opt.step()
        held = wall_safe(params, before)
        if held:
            cut[0] += 1
            for k in held:
                cut[1][k] = cut[1].get(k, 0) + 1
            if cut[0] <= 5 or cut[0] % 100 == 0:
                print(f'step {step}: held {held} at their previous values (a pair would lose its wall or well); '
                      f'{cut[0]} steps so far', flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            v = readout(frames, frames.val_mols, params, refresh)
            score = v[SELECT] if v[SELECT] is not None else math.inf
            history.append(dict(step=step, loss=float(loss), val=v, s=time.time() - t0))
            print(f'step {step}: loss {float(loss):.3f}; validation {v}; {time.time() - t0:.0f} s', flush=True)
            if score < best[0]:
                best = (score, params.spec(), step)
    spec = best[1] or params.spec()
    spec['provenance'] = dict(frames=os.path.abspath(a.frames), best_step=best[2], args=vars(a),
                              validation_at_fit=fit0, history=history, fit_molecules=frames.fit_mols,
                              validation_molecules=frames.val_mols, steps_with_held_types=cut[0], held_type_counts=cut[1], loss=dict(T_W=T_W, W_FLOOR=W_FLOOR, E_MAX=E_MAX,
                              TAU_RANK=TAU_RANK, RANK_MIN_GAP=RANK_MIN_GAP, PRIOR_SCALE=PRIOR_SCALE))
    from mxtaltools.analysis import exp6_ff as ff
    ff.register_parameter_set('exp6_calibration_check', spec)  # validates; tables build (each pair keeps a wall)
    ff._pair_tables('exp6_calibration_check', 'cpu', torch.float32)
    json.dump(spec, open(a.out_json, 'w'), indent=1)
    print(f'wrote {a.out_json} (best validation step {best[2]})', flush=True)


if __name__ == '__main__':
    main()
