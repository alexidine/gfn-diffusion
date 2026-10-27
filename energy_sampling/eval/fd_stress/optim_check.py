"""Full campaign-stage relaxations from FD starts in a set of space groups: do the walks keep to the search rules?

Per group: n FD-uniform starts (``stress.run_group``'s draw: ``FDStartSampler``, on sg 6-9 the K-cut chart since
2026-10-26 (the double cover only if the sampler refuses a group), the 3 A
floor), centres wrapped into [0, 1) before the build (the builder clips, not wraps, a centre outside it). Then the
campaign random stream's two eLJ stages (streams/random.yaml ``opt``: 350 steps with compression, then 150 annealed
steps), wrap mode, without the stage-2 early-stop cascade (no reference energy outside the campaign's group).
MXtalTools' end-of-stage ``canonicalize_aunit`` is wrapped so a raise is counted instead of losing the batch (as in
``stress.relax_group``); the next stage starts from the rows the optimizer returned.

Recorded per row over every recorded step of both stages (the record holds the cleaned parameters, centres unwrapped):
a cell length on the 3 A floor; a free cell angle on the [0.2 pi, 0.8 pi] clamp; an angle triple with no real cell
((V/abc)^2 <= 0); a rotation-vector norm on its clamp; at any step and at the returned (best) step. At the end: eLJ
start -> stage 1 -> stage 2, packing coefficient, steps used against the cap, the crystal_fd fold of the endpoint
(status, moved, eLJ kept within stress.ELJ_RTOL).

``--fd-refold`` passes ``fd_refold=True`` to every stage (MXtalTools ``gradient_descent_optimization``: rows at the
angle clamp or near / past F = 0 re-described during the walk, the end of a stage folded by crystal_fd); the wrapped
end-of-stage call is then ``fd_fold_aunit`` instead of ``canonicalize_aunit``.

Run (CPU, from energy_sampling/):
  CUDA_VISIBLE_DEVICES=-1 python -m eval.fd_stress.optim_check --name optim_c1 --groups 1,2,14 [--fd-refold]
Outputs go to a new directory D:/crystal_datasets/fd_stress_oct26/<name>/ (refused if it exists).
"""
import argparse
import json
import math
import os
import time
import traceback

import torch

from eval.fd_stress import stress as S

DEFAULT_GROUPS = '1,2,4,7,14,15,19,33,61,70,85,88,92,142,146,148,169,176,198,205,212,227,230'
ANGLE_LO, ANGLE_HI = 0.2 * math.pi, 0.8 * math.pi  # clean_cell_parameters at angle_pad 0.6
NORM_LO, NORM_HI = 0.001 * math.pi, 1.999 * math.pi  # clean_cell_parameters' rotation-vector norm bound
EDGE_TOL = 1e-5
CP_SUPPORT = (0.45, 0.95)


def volume_factor(angles):
    c = torch.cos(angles.double())
    return 1 - c[..., 0] ** 2 - c[..., 1] ** 2 - c[..., 2] ** 2 + 2 * c[..., 0] * c[..., 1] * c[..., 2]


def wall_flags(P, floor):
    """[T, n, 12] cleaned parameter record -> dict of [T, n] bool wall contacts."""
    P = P.double()
    ang = P[..., 3:6]
    norm = P[..., 9:12].norm(dim=-1)
    return dict(floor=(P[..., :3] <= floor + EDGE_TOL).any(-1),
                angle_clamp=(((ang - ANGLE_LO).abs() < EDGE_TOL) | ((ang - ANGLE_HI).abs() < EDGE_TOL)).any(-1),
                unreal=volume_factor(ang) <= 0,
                norm_clamp=(norm <= NORM_LO + EDGE_TOL) | (norm >= NORM_HI - EDGE_TOL))


def run_stage(mol, sg, params, hand, oc, chunk, scratch):
    """One stage over all rows in chunks. Returns (next params, next hand, best params (wrapped, before MXtalTools'
    re-expression), best hand, records list per chunk, canonicalize_aunit raise messages)."""
    from mxtaltools.crystal_search import crystal_opt_utils as cou
    from mxtaltools.dataset_utils.utils import collate_data_list
    calls, outs, recs = [], [], []
    end_fn = 'fd_fold_aunit' if oc.get('fd_refold') else 'canonicalize_aunit'  # the end-of-stage call to wrap
    orig = getattr(cou, end_fn)

    def wrapped(cb):
        pre = (cb.full_cell_parameters().detach().float().clone(), cb.aunit_handedness.detach().reshape(-1).float())
        try:
            out = orig(cb)
            calls.append((pre, None))
            return out
        except ValueError as e:
            calls.append((pre, f'{type(e).__name__}: {str(e)[:200]}'))
    setattr(cou, end_fn, wrapped)
    try:
        for lo in range(0, len(params), chunk):
            b = S.rows(mol, sg, params[lo:lo + chunk], hand[lo:lo + chunk])
            out, rec = b.optimize_crystal_parameters(return_record=True, intermediates_path=scratch, **oc)
            outs += out
            recs.append(dict(params=rec['params'], loss=rec['loss'], cp=rec['cp'], refold=rec.get('fd_refold')))
    finally:
        setattr(cou, end_fn, orig)
    if len(calls) != len(recs) or len(outs) != len(params):
        raise RuntimeError(f'{len(calls)} {end_fn} calls for {len(recs)} chunks, {len(outs)} rows out')
    post = collate_data_list(outs)
    best_p = torch.cat([c[0][0] for c in calls])
    best_h = torch.cat([c[0][1] for c in calls])
    return (post.full_cell_parameters().detach().float().clone(), post.aunit_handedness.detach().reshape(-1).float(),
            best_p, best_h, recs, [c[1] for c in calls if c[1] is not None])


def stage_walls(recs, floor):
    """Per row: any-step and best-step wall contacts, steps used, best-step cp, over a stage's chunk records."""
    any_, best, steps, cp_best = {}, {}, [], []
    for r in recs:
        P, L = r['params'], r['loss']
        T = min(P.shape[0], L.shape[0])
        P, L, cp = P[:T], L[:T], r['cp'][:T]
        w = wall_flags(P, floor)
        bi = torch.argmin(L, 0)
        ar = torch.arange(P.shape[1])
        for k, v in w.items():
            any_.setdefault(k, []).append(v.any(0))
            best.setdefault(k, []).append(v[bi, ar])
        steps.append(T)
        cp_best.append(cp[bi, ar])
    return ({k: torch.cat(v) for k, v in any_.items()}, {k: torch.cat(v) for k, v in best.items()}, max(steps),
            torch.cat(cp_best))


def run_group(sg, mol, n, seed, chunk_cap, scratch, log, fd_refold=False):
    from eval.fd_search import campaign, fd_starts, run_fd_search
    from mxtaltools.crystal_fd import canon, tables
    from mxtaltools.constants.space_group_info import SYM_OPS
    tab = tables.group_table(sg)
    sym_mult = len(SYM_OPS[sg])
    chunk = min(chunk_cap, S.energy_chunk(sym_mult) * 2)
    rec = dict(sg=sg, symbol=tab.symbol, family=tab.lattice_type, sym_mult=sym_mult, n=n, chunk=chunk)
    g = torch.Generator().manual_seed(int(seed) * 7919 + sg)
    try:
        sampler = fd_starts.FDStartSampler(sg, mol, generator=g)
        rec['sampler'] = 'fd'
    except NotImplementedError:
        sampler = S.make_double_cover_sampler(sg, mol, g)
        rec['sampler'] = 'double_cover'
    run_fd_search.SG = sg  # fd_start_rows' check reads this module global
    floor = run_fd_search.length_floor()
    batch, cp_t, desc, info, draw, clipped = run_fd_search.fd_start_rows(sampler, n, seed, 0, floor)
    rec['n_floored_start'] = int(clipped.any(1).sum())
    p0, h0, clip0 = S.desc_params(fd_starts.rows_to_desc(sg, batch))  # centres wrapped into [0, 1)
    del batch
    echunk = S.energy_chunk(sym_mult)
    E0 = S.elj_of(mol, sg, p0, h0, echunk)

    cfg = campaign.stream_config('random')
    t0 = time.time()
    p, h = p0, h0
    E_stage, walls_any, walls_best, steps, caps, raises, refolds, evals = [], [], [], [], [], [], [], []
    for st in cfg.opt:
        oc = run_fd_search._stage_config(dict(st), cfg)  # no early_stop_ref passed: the cascade is off
        if fd_refold:
            oc['fd_refold'] = True
        p, h, bp, bh, recs, msgs = run_stage(mol, sg, p, h, oc, chunk, scratch)
        a, b, T, cp_best = stage_walls(recs, floor)
        walls_any.append(a)
        walls_best.append(b)
        steps.append(T)
        evals.append(sum(int(torch.isfinite(r['loss']).sum()) for r in recs))  # row-evaluations (retired rows: inf)
        caps.append(int(oc['max_num_steps']))
        raises.append(msgs)
        if fd_refold:  # refolds applied per row in this stage (records['fd_refold'] summed over steps)
            refolds.append(torch.cat([r['refold'].sum(0) for r in recs]))
        E_stage.append(S.elj_of(mol, sg, bp, bh, echunk))
    rec['wall_s'] = time.time() - t0
    E1, E2 = E_stage
    rec['elj'] = dict(start_median=S._f(E0.median()), stage1_median=S._f(E1.median()), end_median=S._f(E2.median()),
                      end_min=S._f(E2.min()), finite_end=int(torch.isfinite(E2).sum()),
                      lowered=int((E2 < E0).sum()), stage2_not_worse=int((E2 <= E1 + 1e-3 * E1.abs().clamp(min=1)).sum()))
    rec['steps'] = steps
    rec['row_evals'] = evals
    rec['step_caps'] = caps
    rec['mxt_canon_raised_chunks'] = [len(m) for m in raises]
    rec['mxt_canon_message'] = next((m[0] for m in raises if m), None)
    for i, r in enumerate(refolds):
        rec[f'stage{i + 1}_refolds'] = dict(rows=int((r > 0).sum()), total=int(r.sum()), max_per_row=int(r.max()))
    for i, (a, b) in enumerate(zip(walls_any, walls_best)):
        rec[f'stage{i + 1}_walls_any'] = {k: int(v.sum()) for k, v in a.items()}
        rec[f'stage{i + 1}_walls_best'] = {k: int(v.sum()) for k, v in b.items()}
    rec['cp_end'] = dict(min=S._f(cp_best.min()), median=S._f(cp_best.median()), max=S._f(cp_best.max()),
                         out_of_support=int(((cp_best < CP_SUPPORT[0]) | (cp_best > CP_SUPPORT[1])).sum()))

    # the endpoint: the stage-2 best rows (wrapped, before MXtalTools' re-expression), folded by crystal_fd
    c = canon.canonicalize(fd_starts.rows_to_desc(sg, S.rows(mol, sg, bp, bh)))
    names = c.names()
    rec['fold_status'] = {s: names.count(s) for s in sorted(set(names))}
    rec['fold_moved'] = int(c.moved.sum())
    good = torch.tensor([s == 'canonical' for s in names])
    if bool(good.any()):
        gi = torch.nonzero(good).flatten()
        pc, hc, clipc = S.desc_params(c.desc[gi])
        Ec = S.elj_of(mol, sg, pc[~clipc], hc[~clipc], echunk)
        rel = S.elj_rel(Ec, E2[gi][~clipc])
        rec['fold_elj_kept'] = int((rel <= S.ELJ_RTOL).sum())
        rec['fold_elj_n'] = int((~clipc).sum())
        rec['fold_elj_worst_rel'] = S._max(rel)
        rec['fold_fd_status'] = int(canon.fd_status(c.desc[gi]).is_fd.sum())
    rec['_rows'] = dict(start=p0, start_hand=h0, end=bp, end_hand=bh, elj_start=E0, elj_stage1=E1, elj_end=E2,
                        cp_end=cp_best, walls_any={k: walls_any[0][k] | walls_any[1][k] for k in walls_any[0]},
                        walls_best=walls_best[1], fold_status=names,
                        refolds=torch.stack(refolds) if refolds else None)  # [stage, row] refolds applied
    bad = torch.nonzero(~good).flatten().tolist()[:3]
    rec['fold_fail_examples'] = [dict(row=i, status=names[i], end=[round(float(x), 6) for x in bp[i]],
                                      volume_factor=S._f(volume_factor(bp[i, 3:6])),
                                      angle_sum_deg=S._f(torch.rad2deg(bp[i, 3:6].double()).sum()),
                                      elj=S._f(E2[i])) for i in bad]
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--name', required=True)
    ap.add_argument('--groups', default=DEFAULT_GROUPS)
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--molecule-from', default=os.path.join(S.OUT_ROOT, 'c1_all', 'molecule.pt'))
    ap.add_argument('--fd-refold', action='store_true',
                    help="pass fd_refold=True to MXtalTools' optimiser in every stage (crystal_fd re-descriptions)")
    args = ap.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise SystemExit('run with CUDA_VISIBLE_DEVICES=-1 (CPU only)')
    torch.set_num_threads(args.threads)
    out_dir = S.out_path(args.name)
    os.makedirs(out_dir, exist_ok=False)  # refuses an existing directory
    os.makedirs(os.path.join(out_dir, 'groups'))
    scratch = os.path.join(out_dir, '_opt_intermediates.pt')
    log_path = os.path.join(out_dir, 'run.log')

    def log(*a):
        print(*a, flush=True)
        with open(log_path, 'a') as fh:
            print(*a, file=fh)
    saved = torch.load(args.molecule_from, weights_only=False)
    mol = saved['mol']
    S._atomic_json(dict(args=vars(args), started=time.strftime('%Y-%m-%d %H:%M:%S'),
                        molecule=saved['record'].get('chosen', {}).get('smiles')), os.path.join(out_dir, 'meta.json'))
    for sg in S.parse_groups(args.groups):
        t0 = time.time()
        try:
            rec = run_group(sg, mol, args.n, args.seed, args.chunk, scratch, log, fd_refold=args.fd_refold)
        except Exception as e:  # noqa: BLE001 -- recorded as a failure of the group, never a pass
            rec = dict(sg=sg, exception=dict(type=type(e).__name__, message=str(e)[:600],
                                             traceback=traceback.format_exc().strip().splitlines()[-14:]))
        rows_rec = rec.pop('_rows', None)
        if rows_rec is not None:  # per-row start, endpoint, energies, cp and wall flags
            torch.save(rows_rec, os.path.join(out_dir, 'groups', f'sg_{sg:03d}_rows.pt'))
        S._atomic_json(rec, os.path.join(out_dir, 'groups', f'sg_{sg:03d}.json'))
        if 'exception' in rec:
            log(f'sg {sg}: EXCEPTION {rec["exception"]["type"]}: {rec["exception"]["message"][:200]}')
        else:
            log(f'sg {sg} {rec["symbol"]}: {time.time() - t0:.0f}s steps {rec["steps"]}/{rec["step_caps"]} '
                f'eLJ {rec["elj"]["start_median"]:.0f} -> {rec["elj"]["stage1_median"]:.0f} -> '
                f'{rec["elj"]["end_median"]:.0f} lowered {rec["elj"]["lowered"]}/{args.n} '
                f'walls1 {rec["stage1_walls_any"]} walls2@best {rec["stage2_walls_best"]} cp {rec["cp_end"]} '
                f'fold {rec["fold_status"]} kept {rec.get("fold_elj_kept")}/{rec.get("fold_elj_n")} '
                f'mxt_raise {rec["mxt_canon_raised_chunks"]}')
    if os.path.exists(scratch):
        os.remove(scratch)
    log(f'{time.strftime("%Y-%m-%d %H:%M:%S")} done')


if __name__ == '__main__':
    main()
