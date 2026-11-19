"""Per-space-group readouts on a run over several space groups (cfg:space_groups). Every function returns {} on a
run over fewer than two groups, as train.py::group_split_means does.

- ``condition_groups(condition_id, space_groups, n_zp)``: each row's space group, decoded from the mixed-radix
  condition_id = mol_id * (n_sg * n_zp) + sg_local * n_zp + zp_local (MolecularCrystal.condition_samples).
- ``branch_group_stats(loss_dict, sub_type, space_groups, n_zp)``: train-cadence per-group health of one branch's
  loss dict, {'sg<g>': {'<sub_type>/<name>': value}}, for train.py::Modeller._update_rolling to put through the
  MetricTracker (so the wandb keys are sg<g>/<sub_type>/<name>, EMA'd on the pooled keys' clock):
    tb_err            RMS over the group's rows of the TB residual log_pf + log Z - log_pb - log_r, with log Z the
                      centre the loss uses (log_z_target where the loss carries one, else the head) -- the pooled
                      tb_err's definition (utils.quick_tb_stats) restricted to the group
    logw_std_within   (fwd and replay) the pooled within-condition RMS of log w about its condition's mean, over the
                      group's rows whose condition holds >= 2 of the batch's rows; absent when none does
    mle               the mean over the group's rows of the per-row MLE loss -log q (loss_dict 'mle_rows')
    log_Z_learned     the mean of the head's level over the group's rows
- ``buffer_group_share(sg_rows, space_groups, key)``: {'sg<g>/<key>': the share of the rows in group g}, every run
  group present (a group with no row reads 0).
- ``head_panel_stats(logits, groups, prefix)`` / ``is_panel_stats(...)``: the Z(sg) panel aggregates
  (train.py::Modeller.log_zsg_panels).

Per-group levels live on different FD charts (different cell-coordinate counts and pose normalisation), so compare each
group's curve over time, not levels across groups; p(g) here is a sampling allocation, not a thermodynamic group
preference.
"""
import math

import torch

from models.group_draw import group_weights_is

ZSG_CAVEAT = ("per-group levels live on different FD charts (different cell-coordinate counts and pose "
              "normalisation), so compare each group's curve over time, not levels across groups; p(g) here is a "
              "sampling allocation, not a thermodynamic group preference")


def _multi_group(space_groups) -> bool:
    return len({int(g) for g in (space_groups or [])}) >= 2


def condition_groups(condition_id, space_groups, n_zp) -> torch.Tensor:
    """[n] long: the space group number of each condition_id (cpu)."""
    cid = torch.as_tensor(condition_id).detach().reshape(-1).long().cpu()
    sg = torch.as_tensor([int(g) for g in space_groups], dtype=torch.long)
    return sg[(cid // int(n_zp)) % len(sg)]


def _aligned(values, groups):
    """`groups` cut to one entry per element of `values` [B]: as is when the counts agree, the first of each
    contiguous tile when `groups` is the K-tiled [B * K] layout (terminal-major), else None."""
    n, m = values.numel(), groups.numel()
    if n == m:
        return groups
    if n > 0 and m % n == 0:
        return groups.view(n, -1)[:, 0]
    return None


def _flat(t):
    return torch.as_tensor(t).detach().reshape(-1).double().cpu()


def branch_group_stats(loss_dict, sub_type, space_groups, n_zp) -> dict:
    """{'sg<g>': {'<sub_type>/<name>': float}} for the groups the batch holds (module docstring); {} on a single-group
    run or a loss dict without condition_id."""
    if not _multi_group(space_groups) or not loss_dict or loss_dict.get('condition_id') is None:
        return {}
    cid = torch.as_tensor(loss_dict['condition_id']).detach().reshape(-1).long().cpu()
    grp = condition_groups(cid, space_groups, n_zp)
    out = {}

    def put(g, name, v):
        if math.isfinite(v):
            out.setdefault(f'sg{int(g)}', {})[f'{sub_type}/{name}'] = float(v)

    if all(k in loss_dict for k in ('log_pf', 'log_pb', 'log_r', 'log_Z')):
        log_pf, log_pb, log_r = _flat(loss_dict['log_pf']), _flat(loss_dict['log_pb']), _flat(loss_dict['log_r'])
        z = loss_dict.get('log_z_target')
        z = _flat(loss_dict['log_Z'] if z is None else z)
        n = log_pf.numel()
        if z.numel() == 1:
            z = z.expand(n)
        if log_pb.numel() == n and log_r.numel() == n and z.numel() == n and grp.numel() == n:
            resid = log_pf + z - log_pb - log_r
            log_w = log_r + log_pb - log_pf
            fin = torch.isfinite(resid) & torch.isfinite(log_w)
            within = None
            if sub_type in ('fwd', 'replay') and bool(fin.any()):
                uniq, inv = torch.unique(cid[fin], return_inverse=True)
                cnt = torch.zeros(uniq.numel(), dtype=torch.float64).scatter_add_(
                    0, inv, torch.ones(inv.numel(), dtype=torch.float64))
                mean = torch.zeros(uniq.numel(), dtype=torch.float64).scatter_add_(0, inv, log_w[fin]) / cnt
                within = torch.full_like(log_w, math.nan)
                multi = cnt[inv] >= 2
                within[torch.nonzero(fin).flatten()[multi]] = (log_w[fin] - mean[inv])[multi]
            for g in torch.unique(grp).tolist():
                sel = (grp == g) & fin
                if not bool(sel.any()):
                    continue
                put(g, 'tb_err', float(resid[sel].pow(2).mean().sqrt()))
                if within is not None:
                    w = within[sel]
                    w = w[torch.isfinite(w)]
                    if w.numel():
                        put(g, 'logw_std_within', float(w.pow(2).mean().sqrt()))

    for key, name in (('mle_rows', 'mle'), ('log_Z', 'log_Z_learned')):
        v = loss_dict.get(key)
        if v is None:
            continue
        v = _flat(v)
        g_rows = _aligned(v, grp)
        if g_rows is None or v.numel() < 2:   # a scalar head has one level for every group
            continue
        for g in torch.unique(g_rows).tolist():
            sel = (g_rows == g) & torch.isfinite(v)
            if bool(sel.any()):
                put(g, name, float(v[sel].mean()))
    return out


def buffer_group_share(sg_rows, space_groups, key) -> dict:
    """{'sg<g>/<key>': share of `sg_rows` in group g} for every run group; {} on a single-group run or no rows."""
    if not _multi_group(space_groups) or sg_rows is None:
        return {}
    s = torch.as_tensor(sg_rows).detach().reshape(-1).long().cpu()
    if s.numel() == 0:
        return {}
    return {f'sg{int(g)}/{key}': float((s == int(g)).double().mean()) for g in sorted({int(g) for g in space_groups})}


def _entropy(p):
    """[M] entropy in nats of each row of p [M, G] (0 log 0 = 0)."""
    return -(p * torch.log(p.clamp_min(1e-300))).masked_fill(p <= 0, 0.0).sum(-1)


def head_panel_stats(logits, groups, prefix='zsg/') -> dict:
    """The head panel from the levels logits [M, G] = L(m, g) (models/group_draw.py::level_group_logits): per molecule
    p(g | m) = softmax_g L(m, g); returns <prefix>head_p_sg<g> (panel mean of p), <prefix>head_level_sg<g> (panel
    mean of L) and <prefix>head_entropy_nats (panel mean of the entropy of p). {} on fewer than two groups."""
    if not _multi_group(groups):
        return {}
    L = torch.as_tensor(logits).detach().double().cpu()
    p = torch.softmax(L, -1)
    out = {f'{prefix}head_entropy_nats': float(_entropy(p).mean())}
    for k, g in enumerate(groups):
        out[f'{prefix}head_p_sg{int(g)}'] = float(p[:, k].mean())
        out[f'{prefix}head_level_sg{int(g)}'] = float(L[:, k].mean())
    return out


def is_panel_stats(log_w, mol_index, group_index, n_mol, groups, level=None, prefix='zsg/') -> dict:
    """The importance-sampling panel from per-draw log w [N] = log R - log q, each draw's panel molecule mol_index [N]
    in [0, n_mol) and group column group_index [N] in [0, G). Per molecule, models/group_draw.py::group_weights_is over
    its finite draws gives log Z_g, the normalised weights Z_g / sum Z and ESS_g. Returns, as panel means over the
    molecules where each is finite: <prefix>is_log_z_sg<g>, <prefix>is_p_sg<g>, <prefix>is_ess_sg<g>,
    <prefix>is_entropy_nats (entropy of the normalised weights) and, given the head's level [n_mol, G],
    <prefix>head_minus_is_sg<g> = L(m, g) - log Z_g(m). {} on fewer than two groups."""
    if not _multi_group(groups):
        return {}
    G = len(groups)
    lw = _flat(log_w)
    mi = torch.as_tensor(mol_index).reshape(-1).long().cpu()
    gi = torch.as_tensor(group_index).reshape(-1).long().cpu()
    fin = torch.isfinite(lw)
    log_z = torch.full((int(n_mol), G), -math.inf, dtype=torch.float64)
    w = torch.zeros(int(n_mol), G, dtype=torch.float64)
    ess = torch.zeros(int(n_mol), G, dtype=torch.float64)
    for m in range(int(n_mol)):
        sel = (mi == m) & fin
        if bool(sel.any()):
            log_z[m], w[m], ess[m] = group_weights_is(lw[sel], gi[sel], G)
    has = torch.isfinite(log_z).any(-1)
    out = {}
    if bool(has.any()):
        out[f'{prefix}is_entropy_nats'] = float(_entropy(w[has]).mean())
    L = None if level is None else torch.as_tensor(level).detach().double().cpu()
    for k, g in enumerate(groups):
        g = int(g)
        ok = torch.isfinite(log_z[:, k])
        if bool(ok.any()):
            out[f'{prefix}is_log_z_sg{g}'] = float(log_z[ok, k].mean())
            if L is not None:
                out[f'{prefix}head_minus_is_sg{g}'] = float((L[ok, k] - log_z[ok, k]).mean())
        if bool(has.any()):
            out[f'{prefix}is_p_sg{g}'] = float(w[has, k].mean())
            out[f'{prefix}is_ess_sg{g}'] = float(ess[has, k].mean())
    return out


def panel_line(stats, groups, prefix, n_mol, draws) -> str:
    """One captioned line of a panel's numbers for the eval log."""
    def f(key, fmt='.3g'):
        v = stats.get(f'{prefix}{key}')
        return '-' if v is None else format(v, fmt)
    ran = any(k.startswith(f'{prefix}is_') for k in stats)
    cells = []
    for g in groups:
        g = int(g)
        cell = f"sg{g} head p {f(f'head_p_sg{g}')} L {f(f'head_level_sg{g}')}"
        if ran:
            cell += (f" | IS logZ {f(f'is_log_z_sg{g}')} p {f(f'is_p_sg{g}')} ess {f(f'is_ess_sg{g}')}"
                     f" L-IS {f(f'head_minus_is_sg{g}')}")
        cells.append(cell)
    ent = f"H head {f('head_entropy_nats')}" + (f" IS {f('is_entropy_nats')}" if ran else '')
    what = (f"{n_mol} panel molecules, each in all {len(groups)} groups; head p = panel mean of softmax_g L(m, g)"
            + (f", IS from {draws} forward draws per (molecule, group), logZ / p / ess panel means" if ran
               else ", IS panel not on this eval"))
    return (f"{prefix.rstrip('/')} ({what}; entropies in nats, uniform {math.log(len(groups)):.3g}; {ZSG_CAVEAT}): "
            + '; '.join(cells) + f'; {ent}')
