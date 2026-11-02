"""
Fit and overfit readings for a conditional AR flow policy (models/nf_policy.py::NFPolicy), offline, on fixed rows.

The trainer's density readouts (train.py::Modeller.density_readouts) give one mean -log q per row set. This harness
reads an archive on fixed row sets and splits that number into what it is made of, so a held-out deficit can be put
on the encoder, the decoder or the data:

  nll        -log q per row, exact, with its per-row-of-the-chart terms (one block: the AR factorisation, so each
             term is the conditional NLL of that chart row given the earlier ones), and against the chart's reference
             measure (`_ref`, -log q + log rho, as density_readouts). Per molecule, then over molecules.
  scramble   the gain G(c) = NLL(baseline) - NLL(model) per molecule, where the baseline is the same architecture
             trained with scramble_conditions (it learns the condition-free mixture), and the transfer ratio
             median G(held-out set) / median G(training minima). A ratio near 1 says the conditioning transfers; near
             0 says it is a lookup of the training molecules. Needs --baseline.
  swap       each molecule's rows scored under the condition of its nearest training molecule (Morgan r2 Tanimoto,
             a fixed fingerprint, so the neighbour does not depend on the encoder) and of a random training molecule.
             The model must beat lookup: NLL(own) < NLL(nearest).
  amortise   the decoder frozen, one free context vector per molecule fitted on half its rows (Adam) and read on the
             other half, both halves in turn: the amortisation gap NLL(encoder's context) - NLL(free context).
             A gap on held-out molecules that the training molecules do not show is the encoder failing to place a
             new molecule; a held-out deficit that survives a free context is the decoder's.
  phase2     K draws per condition from q, scored by the run's reward: Var(log w), the effective sample fraction and
             the importance-sampling log Z(c) = logmeanexp(log R - log q). None of these reads the level head. One
             energy call per draw.
  c2st       a two-sample classifier test, training molecules against held-out molecules, on the decoder's context
             vector and, as the reference, on Morgan fingerprints. Context AUC well above fingerprint AUC says the
             encoder maps held-out molecules off the training manifold beyond their chemical shift.

THE ROW SETS (built once, cached under --out, never read by any trainer buffer):
  train_rows          rows the run trains on (prior_dataset as loaded: a pre-noised prior's children of kinds
                      noised_prior.train_kinds inside its windows)
  heldout_basin_rows  the same molecules' held-out anchor groups as the trainer keeps them (prior_heldout_batch: the
                      same kinds under the same windows): basins of training molecules never trained on
  train_minima        the clean anchors of those molecules (prior_anchor_batch)
  heldout_mols        rows of --heldout-rows whose molecule is in the run's test_molecules_path (never trained on),
                      inside the run's window_kT x T of the molecule's own floor and its packing window, re-scored by
                      the run's energy; anchors only where the file tags kinds
Like for like: heldout_basin_rows against train_rows (both kernel children), heldout_mols against train_minima (both
minima); PAIRS holds the pairing every gap and transfer ratio uses.
The data come through the trainer's own load chain (Modeller.init_energy_function .. init_identifiers), so the rows,
windows, chart and conditions are the run's.

    cd energy_sampling
    CUDA_VISIBLE_DEVICES=-1 python -m eval.nf_fit --config <run>.yaml --checkpoint <ck>.pt \
        --baseline <scrambled ck>.pt --heldout-rows <held-out molecules' prior>.pt --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

LATENT_NAMES = ('a', 'b', 'c', 'alpha', 'beta', 'gamma', 'u', 'v', 'w', 'theta', 'phi', 'r')
SET_NAMES = ('train_rows', 'heldout_basin_rows', 'train_minima', 'heldout_mols')
# each held-out set and the training set it is read against
PAIRS = {'heldout_basin_rows': 'train_rows', 'heldout_mols': 'train_minima'}
PROBES = ('nll', 'scramble', 'swap', 'amortise', 'phase2', 'c2st')
CHUNK = 512


# -- the run ---------------------------------------------------------------------------------------------------------

@dataclass
class Run:
    m: object            # train.Modeller after its data chain
    policy: object       # the archive's NFPolicy
    baseline: object     # the scrambled-condition archive's NFPolicy, or None
    step: int


def load(config, checkpoint, baseline=None, device='cpu'):
    """The run's data through the trainer's own chain, then the archive's policy (and the baseline's) by
    Checkpointer.load_weights_only, which asserts the archive solves this config's problem."""
    import train
    from utils import get_train_args
    args = get_train_args(['--config', config])
    args.device = args.buffer_device = device
    m = train.Modeller(args)
    m.init_energy_function()
    ef = m.energy_function
    m.init_mol_dataset()
    with torch.no_grad(), ef.allow_unreferenced():
        m.init_prior_dataset()
    m.init_identifiers()
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if getattr(ef, 'energy_reference_mode', None) is not None:
        # the table training scored against, installed, never recomputed (as eval/cond_panel/sampler.py::load_run)
        ef.set_energy_reference(ck['energy_reference']['table'])
    base = None
    if baseline is not None:
        m.checkpointer.load_weights_only(baseline)
        base = m.gfn_model.eval()
    m.checkpointer.load_weights_only(checkpoint)
    policy = m.gfn_model.eval()
    if not hasattr(policy, 'sampler') or not policy.conditional:
        raise ValueError(f"{checkpoint}: not a conditional flow policy (model.kind nf on a conditional route)")
    return Run(m=m, policy=policy, baseline=base, step=int(ck['modeller_state']['step_ind']))


def row_names(run):
    fd = getattr(run.m.energy_function, 'fd_layout', None)
    names = list(fd.names) if fd is not None else list(LATENT_NAMES[:run.policy.dim])
    return [names[i] for i in run.policy.live_idx.tolist()]


# -- the row sets ----------------------------------------------------------------------------------------------------

def _pick(batch, idx):
    return batch.subsample_new_batch(torch.as_tensor(idx, dtype=torch.long, device=batch.ptr.device)).cpu()


def _keys(batch):
    """One condition key per row: molecule and space group."""
    sg = batch.sg_ind.reshape(-1).tolist()
    return [f'{i}|{int(s)}' for i, s in zip(batch.identifier, sg)]


def _cap_per_molecule(batch, mols, cap, rng, extra=None):
    """Rows of `batch` whose molecule is in `mols` (and `extra` is True), at most `cap` per molecule, seeded."""
    by = {}
    for i, ident in enumerate(batch.identifier):
        if ident in mols and (extra is None or bool(extra[i])):
            by.setdefault(ident, []).append(i)
    idx = []
    for ident in sorted(by):
        rows = by[ident]
        idx += sorted(rng.choice(rows, size=min(cap, len(rows)), replace=False).tolist())
    return idx


def _anchors(batch):
    from noised_prior import KIND_ANCHOR, tag
    kinds = getattr(batch, 'noised_kind', None)
    return None if kinds is None else (tag(batch, 'noised_kind') == KIND_ANCHOR)


def build_sets(run, heldout_rows, n_mols, cap, seed):
    m, ef = run.m, run.m.energy_function
    rng = np.random.default_rng(seed)
    train = m.prior_dataset.batch
    held = getattr(m, 'prior_heldout_batch', None)
    anchors = getattr(m, 'prior_anchor_batch', None)
    train_mols = set(train.identifier)
    # training molecules with held-out basins, so the within-molecule comparison is on one set of molecules
    pool = sorted(train_mols & set(held.identifier)) if held is not None else sorted(train_mols)
    mols = set(rng.choice(pool, size=min(n_mols, len(pool)), replace=False).tolist())
    sets = {'train_rows': _pick(train, _cap_per_molecule(train, mols, cap, rng))}
    if held is not None:
        sets['heldout_basin_rows'] = _pick(held, _cap_per_molecule(held, mols, cap, rng))
    if anchors is not None:
        sets['train_minima'] = _pick(anchors, _cap_per_molecule(anchors, mols, cap, rng))

    if heldout_rows is not None:
        if m.test_mol_dataset is None:
            raise ValueError("--heldout-rows needs the run's test_molecules_path: held-out molecules are those")
        data = torch.load(heldout_rows, map_location='cpu', weights_only=False)
        rows = data.get('equalized_prior', data.get('prior')) if isinstance(data, dict) else data
        m._refuse_mirrored_rows(rows, heldout_rows, 'heldout_rows')
        test_mols = set(m.test_mol_dataset.batch.identifier)
        seen = train_mols | set(m.mol_dataset.batch.identifier)
        if held is not None:
            seen |= set(held.identifier)
        ok = np.array([i in test_mols and i not in seen for i in rows.identifier])
        sg_ok = np.isin(rows.sg_ind.reshape(-1).cpu().numpy(), np.asarray(m.args.space_groups))
        anchor = _anchors(rows)
        keep = ok & sg_ok & (anchor.numpy() if anchor is not None else True)
        rows = _pick(rows, np.nonzero(keep)[0])
        cand = sorted(set(rows.identifier))
        # 2x the molecules asked for, so molecules the window empties can be replaced
        pre = set(rng.choice(cand, size=min(2 * n_mols, len(cand)), replace=False).tolist())
        rows = _pick(rows, [i for i, ident in enumerate(rows.identifier) if ident in pre])
        # the run's windows, on a fresh scoring through the run's energy (a held-out file's stored terms need not be
        # in the run's currency): within window_kT x T of the molecule's floor, inside the packing window
        import noised_prior
        s = noised_prior.settings(m.args)
        T = float(m.args.energy_config.temperature)
        _, rows = m._score_prior_rows(rows, announce=False)
        rows = rows.cpu()
        seed_e, _ = m._stored_reward_terms(rows)
        pc = torch.as_tensor(rows.packing_coeff).reshape(-1).double()
        keys = _keys(rows)
        floor = {}
        for k, e in zip(keys, seed_e.tolist()):
            floor[k] = min(floor.get(k, math.inf), e)
        win = np.array([(s['window_kT'] is None or e - floor[k] <= s['window_kT'] * T)
                        and s['packing_window'][0] <= p <= s['packing_window'][1]
                        for k, e, p in zip(keys, seed_e.tolist(), pc.tolist())])
        rows = _pick(rows, np.nonzero(win)[0])
        cand = sorted(set(rows.identifier))
        hmols = set(rng.choice(cand, size=min(n_mols, len(cand)), replace=False).tolist())
        sets['heldout_mols'] = _pick(rows, _cap_per_molecule(rows, hmols, cap, rng))
        registry = m.identifier_registry
        b = sets['heldout_mols']
        b.add_graph_attr(torch.tensor([registry[i] for i in b.identifier], dtype=torch.long), 'mol_id')
    for name, b in list(sets.items()):
        print(f"set {name}: {int(b.num_graphs)} rows of {len(set(b.identifier))} molecules", flush=True)
        if int(b.num_graphs) == 0:
            del sets[name]
    return sets


# -- scoring ---------------------------------------------------------------------------------------------------------

def _ctx_vec(ctx):
    return ctx[0] if isinstance(ctx, tuple) else ctx


def _with_vec(ctx, vec):
    return (vec,) + tuple(ctx[1:]) if isinstance(ctx, tuple) else vec


def rows_given_ctx(sampler, x, ctx):
    """[n, d] log q terms per live chart row of a ConditionalLatentFlow at its block context `ctx`, x [n, dim]
    wrapped and pinned: base log-density plus every block's log-derivative of that row, less log sd on standardised
    rows. Their sum over rows is sampler.log_prob(x, c) at ctx = context_input(c)[0] (the harness asserts it). With
    one block, row j's term is log q(x_order[j] | x_order[<j], c)."""
    from models.spline_flow import _rq_spline, _wrap
    f = sampler.flow
    xl = x[:, sampler.live_idx]
    z = torch.where(f.is_wrap, _wrap(xl), (xl - f.mu) / f.sd)
    terms = torch.where(f.is_wrap, torch.zeros_like(f.sd), -torch.log(f.sd)).expand_as(z).clone()
    for blk in f.blocks:
        shift, sp = blk.spline_params(z, ctx)
        if blk.has_bounded:
            u = z + torch.where(blk.is_bounded, torch.zeros_like(shift), shift)
            u = torch.where(blk.is_wrap, _wrap(u), u)
            z, ld = _rq_spline(u, sp, blk.bound, blk.is_wrap, blk.is_bounded, shift, blk.min_height, blk.tied)
        else:
            u = torch.where(blk.is_wrap, _wrap(z + shift), z + shift)
            z, ld = _rq_spline(u, sp, blk.bound, blk.is_wrap, min_height=blk.min_height, tied=blk.tied)
        terms = terms + ld
    base = torch.where(f.unit_base, torch.full_like(z, -math.log(2.0)), -0.5 * z * z - 0.5 * math.log(2 * math.pi))
    return base + terms


def policy_inputs(run, rows, cond_rows=None):
    """(x [n, dim], condition, mol_batch, log_ref [n]) for crystals `rows`, conditioned on the molecules of
    `cond_rows` (default: the rows' own), each at its crystal row's space group and Z'. As density_readouts reads a
    stored row: the latent or FD state of the stored crystal and the condition built by condition_samples."""
    m, ef = run.m, run.m.energy_function
    dev = run.policy.device
    sub = rows.to(dev)
    fd = getattr(ef, 'crystal_coordinates', 'latent') == 'fd'
    if fd and not hasattr(sub, 'fd_state'):
        m._attach_fd_reference(sub)
    x = m._batch_latents(sub).to(dev).float()
    if fd:
        log_ref = sub.fd_log_reference.reshape(-1).float()
    else:
        T = float(m.args.energy_config.temperature)
        jac, _ = ef.compute_jacobian(sub, torch.full((int(sub.num_graphs),), T, device=dev))
        log_ref = (-jac / T).reshape(-1).float()
    cb = sub if cond_rows is None else cond_rows.to(dev)
    cb, _, condition, _ = ef.condition_samples(cb, sg_inds=sub.sg_ind.reshape(-1).long(),
                                               z_primes=sub.z_prime.reshape(-1).long())
    x = run.policy._pin_dead(run.policy._wrap_ang(x))
    return x, condition, cb, log_ref.cpu()


def block_context(policy, condition, mol_batch, n):
    c = policy._context(condition, mol_batch, n)
    return policy.sampler.flow.context_input(c)[0]


@torch.no_grad()
def score(run, rows, policy=None, cond_rows=None, check=True):
    """{'terms': [n, d] -log q per chart row, 'nll': [n], 'ref': [n] -log q + log rho, 'vec': [n, context_dim] the
    decoder's context vector}, in chunks."""
    policy = policy or run.policy
    out = {'terms': [], 'nll': [], 'ref': [], 'vec': []}
    n = int(rows.num_graphs)
    for lo in range(0, n, CHUNK):
        idx = torch.arange(lo, min(lo + CHUNK, n))
        sub = rows.subsample_new_batch(idx.to(rows.ptr.device))
        csub = None if cond_rows is None else cond_rows.subsample_new_batch(idx.to(cond_rows.ptr.device))
        x, cond, cb, log_ref = policy_inputs(run, sub, csub)
        ctx = block_context(policy, cond, cb, len(idx))
        t = rows_given_ctx(policy.sampler, x, ctx)
        if check and lo == 0:
            exact = policy.log_density(x, cond, cb).float()
            gap = float((t.sum(-1) - exact).abs().max())
            if not gap < 1e-3 * max(1.0, float(exact.abs().max())):
                raise AssertionError(f"per-row terms sum to log q only within {gap:.3g}: rows_given_ctx no longer "
                                     f"mirrors ConditionalLatentFlow.to_base")
        out['terms'].append(-t.cpu())
        out['nll'].append(-t.sum(-1).cpu())
        out['ref'].append(-t.sum(-1).cpu() + log_ref)
        out['vec'].append(_ctx_vec(ctx).float().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def per_molecule(keys, values):
    """{key: mean of values over that key's rows}."""
    acc = {}
    for k, v in zip(keys, values.tolist()):
        acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _mean_se(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return float('nan'), float('nan'), 0
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else float('nan'), int(a.size)


def _bootstrap_ratio(num, den, rng, n=1000):
    num, den = np.asarray(num), np.asarray(den)
    r = [np.median(rng.choice(num, num.size)) / np.median(rng.choice(den, den.size)) for _ in range(n)]
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


# -- probes ----------------------------------------------------------------------------------------------------------

def probe_nll(run, sets, scored):
    names = row_names(run)
    res = {'row_names': names, 'n_blocks': len(run.policy.sampler.flow.blocks), 'sets': {}}
    for name, b in sets.items():
        s = scored[name]
        keys = _keys(b)
        fin = torch.isfinite(s['nll'])
        mol = per_molecule([k for k, f in zip(keys, fin) if f], s['nll'][fin])
        molref = per_molecule([k for k, f in zip(keys, fin) if f], s['ref'][fin])
        mean, se, nm = _mean_se(mol.values())
        rmean, rse, _ = _mean_se(molref.values())
        res['sets'][name] = dict(rows=int(b.num_graphs), molecules=nm, nonfinite=int((~fin).sum()),
                                 nll=mean, nll_se=se, nll_ref=rmean, nll_ref_se=rse,
                                 nll_p90=float(torch.quantile(s['nll'][fin].double(), 0.9)) if fin.any() else None,
                                 per_row=[float(v) for v in s['terms'][fin].mean(0)])
    return res


def probe_scramble(run, sets, scored, rng):
    if run.baseline is None:
        return None
    res = {'sets': {}}
    gains = {}
    for name, b in sets.items():
        base = score(run, b, policy=run.baseline)
        keys = _keys(b)
        g = per_molecule(keys, base['nll'] - scored[name]['nll'])
        gv = np.array([v for v in g.values() if np.isfinite(v)])
        gains[name] = gv
        mean, se, nm = _mean_se(gv)
        res['sets'][name] = dict(molecules=nm, gain_median=float(np.median(gv)), gain_mean=mean, gain_se=se,
                                 baseline_nll=_mean_se(per_molecule(keys, base['nll']).values())[0])
    for name, ref in PAIRS.items():
        if name in gains and ref in gains and gains[name].size and gains[ref].size:
            r = float(np.median(gains[name]) / np.median(gains[ref]))
            res['sets'][name]['transfer_ratio'] = r
            res['sets'][name]['transfer_ratio_ci95'] = _bootstrap_ratio(gains[name], gains[ref], rng)
            res['sets'][name]['ratio_reference'] = ref
    return res


def _fingerprints(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        out.append(None if mol is None else gen.GetFingerprint(mol))
    return out


def probe_swap(run, sets, scored, rng):
    """Score each molecule's rows under its nearest training molecule's condition and a random one's. The
    conditions come from the training conditions file's rows (mol_dataset), as the model's own condition for a
    training molecule does; that own-condition route is checked against the row's own condition first."""
    from rdkit import DataStructs
    mb = run.m.mol_dataset.batch
    train_ids = list(mb.identifier)
    train_fps = _fingerprints(list(mb.smiles))
    valid = [i for i, f in enumerate(train_fps) if f is not None]
    row_of = {ident: i for i, ident in enumerate(train_ids)}
    res = {'sets': {}}
    smiles_of = dict(zip(mb.identifier, mb.smiles))
    if run.m.test_mol_dataset is not None:
        tb = run.m.test_mol_dataset.batch
        smiles_of.update(zip(tb.identifier, tb.smiles))
    for name in ('train_minima', 'heldout_mols'):
        if name not in sets:
            continue
        b = sets[name]
        mols = sorted(set(b.identifier) & set(smiles_of))
        fps = _fingerprints([smiles_of[i] for i in mols])
        nn, rnd, sim = {}, {}, {}
        for ident, fp in zip(mols, fps):
            if fp is None:
                continue
            cand = [j for j in valid if train_ids[j] != ident]
            s = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, [train_fps[j] for j in cand]))
            nn[ident], sim[ident] = cand[int(s.argmax())], float(s.max())
            rnd[ident] = int(rng.choice(cand))
        keep = [i for i, ident in enumerate(b.identifier) if ident in nn]
        rows = b.subsample_new_batch(torch.as_tensor(keep, dtype=torch.long))
        own = scored[name]['nll'][keep]
        reads = {}
        for label, pick in (('nearest', nn), ('random', rnd)):
            cond = mb.subsample_new_batch(torch.as_tensor([pick[i] for i in rows.identifier], dtype=torch.long,
                                                          device=mb.ptr.device))
            reads[label] = score(run, rows, cond_rows=cond, check=False)['nll']
        if name == 'train_minima':
            # the training molecule's own conditions-file row must give the row's own condition
            cond = mb.subsample_new_batch(torch.as_tensor([row_of[i] for i in rows.identifier], dtype=torch.long,
                                                          device=mb.ptr.device))
            via = score(run, rows, cond_rows=cond, check=False)['nll']
            fin = torch.isfinite(via) & torch.isfinite(own)
            res['own_condition_route_max_abs_diff'] = float((via - own)[fin].abs().max())
        keys = list(rows.identifier)
        o = per_molecule(keys, own)
        dn, dr = per_molecule(keys, reads['nearest'] - own), per_molecule(keys, reads['random'] - own)
        ids = [i for i in o if np.isfinite(dn[i]) and np.isfinite(dr[i])]
        sims = np.array([sim[i] for i in ids])
        dnv, drv = np.array([dn[i] for i in ids]), np.array([dr[i] for i in ids])
        terciles = np.quantile(sims, [1 / 3, 2 / 3]) if sims.size >= 3 else None
        strata = []
        if terciles is not None:
            for lo, hi, lab in ((-1, terciles[0], 'lowest third'), (terciles[0], terciles[1], 'middle third'),
                                (terciles[1], 2, 'highest third')):
                sel = (sims > lo) & (sims <= hi)
                strata.append(dict(label=lab, tanimoto_lo=float(sims[sel].min()), tanimoto_hi=float(sims[sel].max()),
                                   molecules=int(sel.sum()), nearest_penalty_median=float(np.median(dnv[sel]))))
        res['sets'][name] = dict(molecules=len(ids), tanimoto_median=float(np.median(sims)),
                                 nearest_penalty_median=float(np.median(dnv)),
                                 random_penalty_median=float(np.median(drv)),
                                 beats_nearest_frac=float((dnv > 0).mean()), strata=strata)
    return res


def probe_amortise(run, sets, scored, steps=200, lr=1e-2, min_rows=4):
    """Two-fold per molecule: a free context vector per molecule, fitted on one half of its rows with the decoder
    frozen, read on the other half; the gap is the encoder's context against the best one, in nats per row."""
    policy = run.policy
    res = {'steps': steps, 'lr': lr, 'sets': {}}
    params = list(policy.parameters())
    flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        for name, b in sets.items():
            keys = _keys(b)
            # the halves split by anchor group (a kernel child and its anchor share noised_parent), so a basin read
            # out of sample was not fitted; rows without the tag are their own group
            parent = getattr(b, 'noised_parent', None)
            groups = (torch.as_tensor(parent).reshape(-1).tolist() if parent is not None
                      else list(range(int(b.num_graphs))))
            n_groups = {}
            for k, g in set(zip(keys, groups)):
                n_groups[k] = n_groups.get(k, 0) + 1
            mols = sorted({k for k in keys if keys.count(k) >= min_rows and n_groups[k] >= 2})
            if not mols:
                continue
            keep = [i for i, k in enumerate(keys) if k in mols]
            rows = b.subsample_new_batch(torch.as_tensor(keep, dtype=torch.long))
            kk = [keys[i] for i in keep]
            mol_ix = torch.tensor([mols.index(k) for k in kk])
            fold = torch.zeros(len(kk), dtype=torch.long)
            order = {}
            for i, (k, g) in enumerate(zip(kk, [groups[j] for j in keep])):
                gs = order.setdefault(k, {})
                fold[i] = gs.setdefault(g, len(gs)) % 2
            x, cond, cb, _ = policy_inputs(run, rows)
            with torch.no_grad():
                ctx = block_context(policy, cond, cb, len(kk))
            ctx = tuple(t.detach() for t in ctx) if isinstance(ctx, tuple) else ctx.detach()
            vec0 = _ctx_vec(ctx)
            init = torch.stack([vec0[mol_ix == j][0] for j in range(len(mols))])
            amort = -rows_given_ctx(policy.sampler, x, ctx).sum(-1).detach()
            free = torch.full_like(amort, float('nan'))
            insample = torch.full_like(amort, float('nan'))
            for f in (0, 1):
                fit, read = fold == f, fold != f
                v = init.clone().requires_grad_(True)
                opt = torch.optim.Adam([v], lr=lr)
                with torch.enable_grad():
                    for _ in range(steps):
                        opt.zero_grad()
                        nll = -rows_given_ctx(policy.sampler, x[fit], _with_vec(
                            tuple(t[fit] for t in ctx) if isinstance(ctx, tuple) else ctx[fit], v[mol_ix[fit]])).sum(-1)
                        nll[torch.isfinite(nll)].sum().backward()
                        opt.step()
                with torch.no_grad():
                    sl = lambda t, s: tuple(u[s] for u in t) if isinstance(t, tuple) else t[s]
                    free[read] = -rows_given_ctx(policy.sampler, x[read], _with_vec(sl(ctx, read),
                                                                                  v[mol_ix[read]])).sum(-1)
                    insample[fit] = -rows_given_ctx(policy.sampler, x[fit], _with_vec(sl(ctx, fit),
                                                                                    v[mol_ix[fit]])).sum(-1)
            gap = per_molecule(kk, amort - free)
            gap_in = per_molecule(kk, amort - insample)
            mean, se, nm = _mean_se(gap.values())
            res['sets'][name] = dict(molecules=nm, gap_mean=mean, gap_se=se,
                                     gap_median=float(np.nanmedian(list(gap.values()))),
                                     gap_insample_mean=_mean_se(gap_in.values())[0],
                                     context_shift_median=float(np.median(
                                         (v.detach() - init).norm(dim=-1).numpy() / init.norm(dim=-1).numpy())))
    finally:
        for p, f in zip(params, flags):
            p.requires_grad_(f)
    return res


@torch.no_grad()
def probe_phase2(run, sets, n_mols, k, rng):
    """K draws per condition, through the trainer's eval sampling function and reward (eval/utils.py::
    sample_eval_fwd_trajs): per condition Var(log w), the effective sample fraction (sum w)^2 / (K sum w^2) and the
    importance-sampling log Z(c) = logmeanexp(log w), log w = log R - log q (log P_B = 0)."""
    from energy_sampling.eval.utils import sample_eval_fwd_trajs
    from utils import get_gfn_init_state, logmeanexp, uniform_discretizer
    m, ef, policy = run.m, run.m.energy_function, run.policy
    T = float(m.args.energy_config.temperature)
    mb = m.mol_dataset.batch
    tb = m.test_mol_dataset.batch if m.test_mol_dataset is not None else None
    res = {'draws_per_condition': k, 'sets': {}}
    for name, cfile in (('train_minima', mb), ('heldout_mols', tb)):
        if name not in sets or cfile is None:
            continue
        b = sets[name]
        sg_of = dict(zip(b.identifier, b.sg_ind.reshape(-1).tolist()))
        row_of = {ident: i for i, ident in enumerate(cfile.identifier)}
        mols = [i for i in sorted(sg_of) if i in row_of]
        mols = rng.choice(mols, size=min(n_mols, len(mols)), replace=False).tolist()
        per = []
        for ident in mols:
            rows = cfile.subsample_new_batch(torch.full((k,), row_of[ident], dtype=torch.long, device=cfile.ptr.device))
            rows = rows.to(policy.device)
            rows.orient_molecule(mode='standard')
            out = sample_eval_fwd_trajs(get_gfn_init_state(k, ef.data_ndim, policy.device),
                                        policy, lambda bsz: uniform_discretizer(bsz, 1), ef, rows,
                                        sg_inds=torch.full((k,), int(sg_of[ident]), dtype=torch.long),
                                        z_primes=torch.ones(k, dtype=torch.long),
                                        temperatures=T * torch.ones(k, device=policy.device))
            lw = (out['log_r'].flatten() + out['log_pb'] - out['log_pf']).double()
            fin = torch.isfinite(lw)
            lwf = lw[fin]
            w = torch.softmax(lwf, 0)
            per.append(dict(identifier=ident, finite=int(fin.sum()), var_logw=float(lwf.var()),
                            ess_frac=float(1.0 / (w.pow(2).sum() * lwf.numel())),
                            is_log_z=float(logmeanexp(lwf)), head_log_z=float(out['log_flow'][:, 0].mean())))
        v = np.array([p['var_logw'] for p in per])
        e = np.array([p['ess_frac'] for p in per])
        d = np.array([p['is_log_z'] - p['head_log_z'] for p in per])
        res['sets'][name] = dict(conditions=len(per), var_logw_median=float(np.median(v)),
                                 ess_frac_median=float(np.median(e)),
                                 is_minus_head_spread=float(np.std(d - d.mean())), per_condition=per)
    return res


def probe_c2st(run, n_max, rng, folds=5, shuffles=20):
    """Training molecules against held-out molecules, on the decoder's context vector (each molecule at the run's
    first space group) and on Morgan fingerprints; cross-validated AUC of a logistic regression and of a 15-nearest-
    neighbour classifier, with the AUC of label-shuffled fits as the null."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    m = run.m
    if m.test_mol_dataset is None:
        return None
    sg = int(m.args.space_groups[0])
    feats, fps = [], []
    for cfile in (m.mol_dataset.batch, m.test_mol_dataset.batch):
        n = int(cfile.num_graphs)
        idx = np.sort(rng.choice(n, size=min(n_max, n), replace=False))
        b = cfile.subsample_new_batch(torch.as_tensor(idx, dtype=torch.long, device=cfile.ptr.device))
        vec = []
        with torch.no_grad():
            for lo in range(0, int(b.num_graphs), CHUNK):
                sub = b.subsample_new_batch(torch.arange(lo, min(lo + CHUNK, int(b.num_graphs))).to(b.ptr.device))
                sub = sub.to(run.policy.device)
                ns = int(sub.num_graphs)
                sub, _, cond, _ = m.energy_function.condition_samples(
                    sub, sg_inds=torch.full((ns,), sg, dtype=torch.long), z_primes=torch.ones(ns, dtype=torch.long))
                vec.append(_ctx_vec(block_context(run.policy, cond, sub, ns)).float().cpu())
        fp = _fingerprints(list(b.smiles))
        ok = [i for i, f in enumerate(fp) if f is not None]
        feats.append(torch.cat(vec)[ok].numpy())
        fps.append(np.array([list(fp[i]) for i in ok], dtype=np.float32))
    nmin = min(len(feats[0]), len(feats[1]))
    sel = [rng.choice(len(f), nmin, replace=False) for f in feats]
    y = np.r_[np.zeros(nmin), np.ones(nmin)]
    res = {'molecules_per_side': nmin, 'space_group': sg, 'spaces': {}}
    for label, (a, b) in (('context', (feats[0][sel[0]], feats[1][sel[1]])),
                          ('fingerprint', (fps[0][sel[0]], fps[1][sel[1]]))):
        X = np.r_[a, b]
        row = {}
        for clf_name, clf in (('logistic', make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                              ('knn15', make_pipeline(StandardScaler(), KNeighborsClassifier(15)))):
            cv = StratifiedKFold(folds, shuffle=True, random_state=0)
            auc = roc_auc_score(y, cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1])
            null = [roc_auc_score(ys, cross_val_predict(clf, X, ys, cv=cv, method='predict_proba')[:, 1])
                    for ys in (rng.permutation(y) for _ in range(shuffles))]
            row[clf_name] = dict(auc=float(auc), null_mean=float(np.mean(null)), null_sd=float(np.std(null)))
        res['spaces'][label] = row
    return res


# -- report ----------------------------------------------------------------------------------------------------------

def report(res):
    lines = []
    p = res.get('nll')
    if p:
        lines.append(f"\nNLL per row set at step {res['step']} (-log q, nats per crystal; mean over molecules of each "
                     f"molecule's mean over its rows, +- SE over molecules; `ref` adds the chart's log reference "
                     f"density, comparable across charts). Like for like: heldout_basin_rows against train_rows (kernel "
                     f"children), heldout_mols against train_minima (minima).")
        lines.append('row set | rows | molecules | NLL (nats) | NLL ref (nats) | NLL P90 over rows (nats) | non-finite')
        for name, s in p['sets'].items():
            lines.append(f"{name} | {s['rows']} | {s['molecules']} | {s['nll']:.2f} +- {s['nll_se']:.2f} | "
                         f"{s['nll_ref']:.2f} +- {s['nll_ref_se']:.2f} | {s['nll_p90']:.2f} | {s['nonfinite']}")
        exact = 'the exact AR factorisation' if p['n_blocks'] == 1 else (
            f"NOT a factorisation ({p['n_blocks']} blocks: each row's terms summed over blocks)")
        lines.append(f"\nPer chart row, -log q term (nats per crystal, mean over rows), in the chart's row order; with "
                     f"one block each column is that row's conditional NLL given the rows before it, {exact}. "
                     f"Compare columns across sets: a row whose held-out term rises most is where transfer fails.")
        lines.append('row set | ' + ' | '.join(p['row_names']))
        for name, s in p['sets'].items():
            lines.append(f"{name} | " + ' | '.join(f'{v:.2f}' for v in s['per_row']))
    p = res.get('scramble')
    if p:
        lines.append("\nConditioning gain over the scrambled-condition baseline (nats per crystal; per molecule "
                     "NLL(baseline) - NLL(model), median and mean +- SE over molecules). The transfer ratio is the "
                     "set's median gain over its training pair's (heldout_basin_rows: train_rows; heldout_mols: "
                     "train_minima), with a 95% bootstrap interval over molecules: "
                     "1 = the conditioning transfers, 0 = it does not.")
        lines.append('row set | molecules | baseline NLL (nats) | gain median (nats) | gain mean (nats) | '
                     'transfer ratio [95% CI]')
        for name, s in p['sets'].items():
            tr = (f"{s['transfer_ratio']:.2f} [{s['transfer_ratio_ci95'][0]:.2f}, {s['transfer_ratio_ci95'][1]:.2f}]"
                  if 'transfer_ratio' in s else '--')
            lines.append(f"{name} | {s['molecules']} | {s['baseline_nll']:.2f} | {s['gain_median']:.2f} | "
                         f"{s['gain_mean']:.2f} +- {s['gain_se']:.2f} | {tr}")
    p = res.get('swap')
    if p:
        lines.append(f"\nCondition swap (nats per crystal, per molecule then median over molecules): each molecule's "
                     f"rows under its nearest training molecule's condition (Morgan r2 Tanimoto) and a random training "
                     f"molecule's, minus under its own. Positive = its own condition is better. The own-condition route "
                     f"through the conditions file differs from the row's own by at most "
                     f"{p.get('own_condition_route_max_abs_diff', float('nan')):.2g} nats.")
        lines.append('row set | molecules | nearest Tanimoto, median | nearest penalty, median (nats) | '
                     'random penalty, median (nats) | share of molecules own < nearest')
        for name, s in p['sets'].items():
            lines.append(f"{name} | {s['molecules']} | {s['tanimoto_median']:.2f} | {s['nearest_penalty_median']:.2f} "
                         f"| {s['random_penalty_median']:.2f} | {s['beats_nearest_frac']:.2f}")
        for name, s in p['sets'].items():
            for st in s['strata']:
                lines.append(f"  {name}, nearest Tanimoto {st['label']} ({st['tanimoto_lo']:.2f}-{st['tanimoto_hi']:.2f}"
                             f", {st['molecules']} molecules): nearest penalty median {st['nearest_penalty_median']:.2f}")
    p = res.get('amortise')
    if p:
        lines.append(f"\nAmortisation gap (nats per crystal): the decoder frozen, one free context vector per molecule "
                     f"fitted by Adam ({p['steps']} steps, lr {p['lr']:g}) on half its anchor groups, read on the other "
                     f"half, both halves in turn; gap = NLL(encoder's context) - NLL(free context). Out-of-sample is "
                     f"the reading; in-sample shows how much a free context can overfit half the rows. Molecules with "
                     f"at least 4 rows in at least 2 anchor groups.")
        lines.append('row set | molecules | gap, out of sample (nats) | gap median (nats) | gap, in sample (nats) | '
                     'context moved, median relative norm')
        for name, s in p['sets'].items():
            lines.append(f"{name} | {s['molecules']} | {s['gap_mean']:.2f} +- {s['gap_se']:.2f} | "
                         f"{s['gap_median']:.2f} | {s['gap_insample_mean']:.2f} | {s['context_shift_median']:.2f}")
    p = res.get('phase2')
    if p:
        lines.append(f"\nPhase-2 probes, level-free ({p['draws_per_condition']} draws per condition, one energy call "
                     f"each; log w = log R - log q): median over conditions of Var(log w) (nats^2) and of the effective "
                     f"sample fraction; spread (SD over conditions, centred) of IS log Z(c) minus the level head.")
        lines.append('row set | conditions | Var(log w), median (nats^2) | ESS fraction, median | '
                     'IS log Z - head, SD over conditions (nats)')
        for name, s in p['sets'].items():
            lines.append(f"{name} | {s['conditions']} | {s['var_logw_median']:.2f} | {s['ess_frac_median']:.4f} | "
                         f"{s['is_minus_head_spread']:.2f}")
    p = res.get('c2st')
    if p:
        lines.append(f"\nTwo-sample classifier, training against held-out molecules ({p['molecules_per_side']} per side,"
                     f" SG {p['space_group']}): cross-validated AUC, and the AUC of label-shuffled fits (mean +- SD) "
                     f"as the null. Context AUC above the fingerprint AUC = the encoder separates held-out molecules "
                     f"beyond their chemistry.")
        lines.append('feature space | logistic AUC | logistic null | 15-NN AUC | 15-NN null')
        for name, s in p['spaces'].items():
            lg, kn = s['logistic'], s['knn15']
            lines.append(f"{name} | {lg['auc']:.3f} | {lg['null_mean']:.3f} +- {lg['null_sd']:.3f} | {kn['auc']:.3f} | "
                         f"{kn['null_mean']:.3f} +- {kn['null_sd']:.3f}")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help="the run's yaml")
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--baseline', default=None, help='the same run trained with scramble_conditions (scramble probe)')
    ap.add_argument('--heldout-rows', default=None, help="a prior file with rows of the run's held-out molecules")
    ap.add_argument('--out', required=True)
    ap.add_argument('--mols', type=int, default=100, help='molecules per set')
    ap.add_argument('--rows-per-mol', type=int, default=16)
    ap.add_argument('--probes', default=','.join(PROBES))
    ap.add_argument('--phase2-mols', type=int, default=32)
    ap.add_argument('--phase2-draws', type=int, default=64)
    ap.add_argument('--amortise-steps', type=int, default=200)
    ap.add_argument('--c2st-mols', type=int, default=1000)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=20261102)
    a = ap.parse_args(argv)
    probes = [p for p in a.probes.split(',') if p]
    bad = set(probes) - set(PROBES)
    if bad:
        raise SystemExit(f'unknown probes {sorted(bad)}; known {PROBES}')
    if a.device == 'cpu':
        assert not torch.cuda.is_available(), 'CPU run with a visible GPU: set CUDA_VISIBLE_DEVICES=-1'
    os.makedirs(a.out, exist_ok=True)
    run = load(a.config, a.checkpoint, a.baseline, a.device)
    cache = os.path.join(a.out, f'sets_{a.seed}_{a.mols}x{a.rows_per_mol}.pt')
    if os.path.exists(cache):
        sets = torch.load(cache, map_location='cpu', weights_only=False)
        print(f'row sets from {cache}', flush=True)
    else:
        sets = build_sets(run, a.heldout_rows, a.mols, a.rows_per_mol, a.seed)
        torch.save(sets, cache)
    rng = np.random.default_rng(a.seed)
    scored = {name: score(run, b) for name, b in sets.items()}
    res = {'checkpoint': os.path.abspath(a.checkpoint), 'baseline': a.baseline, 'config': os.path.abspath(a.config),
           'step': run.step, 'sets_file': cache}
    if 'nll' in probes:
        res['nll'] = probe_nll(run, sets, scored)
    if 'scramble' in probes:
        res['scramble'] = probe_scramble(run, sets, scored, rng)
    if 'swap' in probes:
        res['swap'] = probe_swap(run, sets, scored, rng)
    if 'amortise' in probes:
        res['amortise'] = probe_amortise(run, sets, scored, steps=a.amortise_steps)
    if 'phase2' in probes:
        res['phase2'] = probe_phase2(run, sets, a.phase2_mols, a.phase2_draws, rng)
    if 'c2st' in probes:
        res['c2st'] = probe_c2st(run, a.c2st_mols, rng)
    stem = os.path.splitext(os.path.basename(a.checkpoint))[0]
    with open(os.path.join(a.out, f'nf_fit_{stem}.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1)
    text = report(res)
    with open(os.path.join(a.out, f'nf_fit_{stem}.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
