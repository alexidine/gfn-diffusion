"""
Topline eval metrics, training vs held-out molecules, recomputed offline from archives.

The trainer's two eval streams are not like-for-like: eval_fwd spreads 2,500 draws over
5,265 molecules (about 0.5 per molecule) and eval_test 1,000 over 585 (about 1.7), so any
within-molecule statistic sees different group sizes on the two sides, and train-minus-
held-out partly reads sampling geometry (the comment in Modeller.log_test_metrics says as
much). Here both streams get the same design: `--conditions` molecules each (all 585
held-out molecules and the same number of training molecules, one fixed seeded draw shared
by every checkpoint) and `--draws` draws per molecule, so the difference is the model's.

Every number goes through the trainer's own functions: utils.py::quick_tb_stats called the
way Modeller._eval_conditional_stats calls it (residuals centred by
gflownet_losses.py::tb_z_per_traj: the tracker's ema_logw where a condition is trusted,
elsewhere the tracker's all-condition level under condition_log_z.untrusted_z 'global' and
the head under 'head'; clip_beta and tb_z_source from the active stage's forward
coefficients, the config's conditional_worst_quantile), the reasonable-crystal mask and
utils.py::per_condition_fraction. The one departure is the reward ramp: it needs the anchor
buffer at that step, which a step archive does not carry, so under_coverage and the
relative_under family are the uniform versions. over_coverage, jensen_z and the TB fit
family do not read the ramp.

EXCESS ENERGY, both streams on one floor. The trainer's Excess Energy family measures each
draw against its condition's running best energy, a record only training molecules have. On
a run with energy_config.energy_reference the reference table covers held-out molecules too
(each molecule's lowest seed energy), so the excess here is
(MolecularCrystal.seed_energy_from(draw) - E_ref(c)) / T on both streams: the crystal energy
without the Jacobian or the penalties, above the molecule's best search minimum, against the
trainer's bar u* = data_ndim * nonthermal_entropy_per_dim. The trainer's own definition is
printed beside it for the training stream, where it exists. jensen_z is also split into its
energy term (mean log R) and its path term (mean log P_B - log P_F).

    cd energy_sampling
    python -m eval.cond_panel.topline --config configs/cond_tb_sep25/ctb25_extreme_l1_cont3.yaml \
        --checkpoints D:/.../<stem>_step10000.pt D:/.../<stem>_step20000.pt ... --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
from types import SimpleNamespace

import torch

from eval.cond_panel.sampler import draw, load_run, pooled_levels, reasonable_mask
from gflownet_losses import tb_z_per_traj
from utils import per_condition_fraction, quick_tb_stats

# (label, key, unit) in print order; keys come from quick_tb_stats unless marked
ROWS = (
    ('forward level jensen_z', 'jensen_z', 'nats'),
    ('  its energy term, mean log R', 'logr_mean', 'nats'),
    ('  its path term, mean log P_B - log P_F', 'path_mean', 'nats'),
    ('excess over best search minimum, median', 'excess_p50', 'kT'),
    ('excess over best search minimum, mean', 'excess_mean', 'kT'),
    ('excess over best search minimum, 90th pct', 'excess_p90', 'kT'),
    ('draws more than u* above it', 'excess_hot_frac', 'fraction'),
    ('draws below the best search minimum', 'below_ref_frac', 'fraction'),
    ('excess, trainer definition, median', 'excess_trainer_p50', 'kT'),
    ('pooled log-mean-exp emp_z', 'emp_z', 'nats'),
    ('within-molecule log w std', 'logw_std_within', 'nats'),
    ('pooled log w std', 'logw_std', 'nats'),
    ('typical-molecule RMS residual cond_tb_err', 'cond_tb_err', 'nats'),
    ('worst-quantile RMS residual tb_err_worst', 'tb_err_worst', 'nats'),
    ('worst-quantile level error z_grad_worst', 'z_grad_worst', 'nats'),
    ('clipped mean residual tb_resid_clipped', 'tb_resid_clipped', 'nats'),
    ('over_coverage', 'over_coverage', 'nats'),
    ('under_coverage (uniform)', 'under_coverage', 'nats'),
    ('reasonable fraction', 'reasonable', 'fraction'),
    ('Cond Reasonable Spread', 'cond_reasonable_spread', 'fraction'),
    ('Cond Reasonable Worst', 'cond_reasonable_worst', 'fraction'),
    ('Cond Reasonable Failing Frac', 'cond_reasonable_failing', 'fraction'),
    ('mol_energy median', 'mol_energy_median', 'raw ELJ'),
    ('mol_energy 5th percentile', 'mol_energy_p05', 'raw ELJ'),
    ('packing coefficient median', 'cp_median', 'dimensionless'),
    ('draws with reduction penalty > 1e-6', 'reduction_frac', 'fraction'),
    ('head log Z, mean over molecules', 'head_mean', 'nats'),
    ('head - tracker, RMS over molecules', 'head_minus_tracker_rms', 'nats'),
)


def stage_coeffs(run, direction='fwd'):
    """The active stage's coefficients for `direction`: the base block with the stage's
    overrides on top, as Modeller.set_loss_coeffs installs them before an eval."""
    cfg = run.config
    base = dict(cfg[f'{direction}_loss_coeffs'])
    ck_stage = run.stage
    for stage in cfg['protocols'][cfg['protocol']]['stages']:
        if stage['name'] == ck_stage:
            base.update((stage.get('loss_coeffs') or {}).get(direction) or {})
            break
    else:
        raise KeyError(f'stage {ck_stage!r} is not in protocol {cfg["protocol"]!r}')
    return SimpleNamespace(**base)


def untrusted_fallback(run):
    """ConditionLogZTracker.untrusted_fallback, read off the stored tracker: the
    all-condition level under untrusted_z 'global' once it is set, else None (the head)."""
    level = run.tracker.get('global_logw')
    if run.tracker.get('untrusted_z') != 'global' or level is None or not math.isfinite(float(level)):
        return None
    return float(level)


def tracker_centre(run, cid, head, coeffs):
    """Modeller._eval_conditional_stats' centring: the tracker's ema_logw where the
    condition is trusted (ConditionLogZTracker.lookup: count >= min_visits, finite), and
    elsewhere -- which is every held-out molecule -- what tb_z_per_traj gives an untrusted
    row: the tracker's all-condition level, or the head when there is none."""
    if getattr(coeffs, 'tb_z_source', 'learned') != 'persistent' or 'ema_logw' not in run.tracker:
        return head, torch.zeros_like(head, dtype=torch.bool)
    min_visits = int(run.tracker.get('min_visits', run.config.get('condition_log_z', {}).get('min_visits', 20)))
    zt = run.tracker['ema_logw'][cid].to(head.dtype)
    zm = (run.tracker['count'][cid] >= min_visits) & torch.isfinite(zt)
    return tb_z_per_traj(head, zt, zm, untrusted_fallback(run)), zm


def excess_energy(run, d, cid):
    """Per draw, nats above the molecule's best search minimum; None without a reference."""
    ef = run.energy_function
    if ef.energy_reference is None:
        return None
    return ((ef.seed_energy_from(d) - ef.energy_reference_for(cid)) / run.temperature).double()


def stream_metrics(run, rows, coeffs, batch_size, seed):
    d = draw(run, rows, batch_size=batch_size, seed=seed)
    cid = d['condition_id']
    log_z, trusted = tracker_centre(run, cid, d['head_log_z'], coeffs)
    q = float(run.config['conditional_worst_quantile'])
    m = quick_tb_stats(d['log_pf'], d['log_pb'], log_z, d['log_r'],
                       clip_beta=getattr(coeffs, 'beta', None), condition_id=cid, worst_quantile=q)
    m = {k: (float(v) if torch.is_tensor(v) or isinstance(v, (int, float)) else v) for k, v in m.items()}
    lv = pooled_levels(d['log_w'])
    m['emp_z'], m['jensen_z'], m['ess_frac'] = lv['emp_z'], lv['jensen_z'], lv['ess_frac']
    m['jensen_z_se'] = lv['jensen_z_se']
    m['logr_mean'] = float(d['log_r'].double().mean())
    m['path_mean'] = float((d['log_pb'] - d['log_pf']).double().mean())

    u = excess_energy(run, d, cid)
    if u is not None:
        s_per_dim = run.config.get('nonthermal_entropy_per_dim', 4.0)  # the trainer's default
        u_star = float(s_per_dim or 0.0) * run.energy_function.data_ndim
        m['excess_p50'], m['excess_mean'] = float(u.median()), float(u.mean())
        m['excess_p90'] = float(torch.quantile(u, 0.9))
        m['excess_hot_frac'] = float((u > u_star).float().mean())
        m['below_ref_frac'] = float((u < 0).float().mean())
        m['u_star'] = u_star
    # the trainer's Excess Energy: -T log R against the tracker's best energy for the
    # condition, clamped at 0, over the rows whose condition has a record
    floor = run.tracker.get('best_energy')
    if floor is not None:
        floor = floor[cid].double()
        seen = torch.isfinite(floor)
        if bool(seen.any()):
            ut = ((-d['log_r'].double() * run.temperature - floor) / run.temperature)[seen].clamp_min(0.0)
            m['excess_trainer_p50'] = float(ut.median())

    good = reasonable_mask(run, d['sample_batch'])
    m['reasonable'] = float(good.float().mean())
    frac = per_condition_fraction(good.float(), cid,
                                  float(run.config.get('reasonable_cond_bar', 0.5)),
                                  worst_quantile=q, higher_is_worse=False)
    if frac is not None:
        m['cond_reasonable_spread'] = float(frac['spread'])
        m['cond_reasonable_worst'] = float(frac['worst'])
        m['cond_reasonable_failing'] = float(frac['failing_frac'])
    e = d['mol_energy']
    m['mol_energy_median'] = float(e.median())
    m['mol_energy_p05'] = float(torch.quantile(e.double(), 0.05))
    m['cp_median'] = float(d['packing_coeff'].median())
    m['reduction_frac'] = float((d['reduction_en'] > 1e-6).float().mean())
    # one head value per molecule (the head reads only the condition)
    uc = torch.unique(cid)
    head_by_c = {int(c): float(h) for c, h in zip(cid, d['head_log_z'])}
    heads = torch.tensor([head_by_c[int(c)] for c in uc], dtype=torch.float64)
    m['head_mean'] = float(heads.mean())
    if 'ema_logw' in run.tracker:
        tz = run.tracker['ema_logw'][uc].double()
        ok = torch.isfinite(tz)
        m['head_minus_tracker_rms'] = (float((heads[ok] - tz[ok]).pow(2).mean().sqrt())
                                       if ok.any() else float('nan'))
    m['n_draws'], m['n_molecules'], m['trusted_frac'] = cid.numel(), int(uc.numel()), float(trusted.float().mean())
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--conditions', type=int, default=None,
                    help='molecules per stream; defaults to the held-out set size')
    ap.add_argument('--draws', type=int, default=8, help='draws per molecule')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch-size', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='directory for topline.json')
    args = ap.parse_args()
    if args.device == 'cpu':
        assert not torch.cuda.is_available(), 'CPU run with a visible GPU: set CUDA_VISIBLE_DEVICES=-1'

    results = {}
    for path in args.checkpoints:
        run = load_run(path, args.config, device=args.device)
        n_c = args.conditions or run.test_conditions.num_graphs
        g = torch.Generator().manual_seed(args.seed)
        train_idx = torch.randperm(run.conditions.num_graphs, generator=g)[:n_c]
        test_idx = torch.randperm(run.test_conditions.num_graphs, generator=g)[:n_c]
        tile = lambda idx: idx.repeat_interleave(args.draws)
        coeffs = stage_coeffs(run, 'fwd')
        res = {
            'train': stream_metrics(run, run.conditions.subsample_new_batch(tile(train_idx)), coeffs,
                                    args.batch_size, args.seed),
            'held_out': stream_metrics(run, run.test_conditions.subsample_new_batch(tile(test_idx)), coeffs,
                                       args.batch_size, args.seed + 1),
        }
        results[run.step] = res
        tr, te = res['train'], res['held_out']
        fb = untrusted_fallback(run)
        elsewhere = 'the head' if fb is None else f'the all-condition level {fb:.2f}'
        print(f'\nStep {run.step} ({run.stage}): {tr["n_molecules"]} training and {te["n_molecules"]} held-out '
              f'molecules, {args.draws} draws each; residuals centred as the trainer\'s eval centres them '
              f'(tracker on {tr["trusted_frac"]:.0%} of training draws and {te["trusted_frac"]:.0%} of held-out '
              f'draws, {elsewhere} elsewhere), clip beta {getattr(coeffs, "beta", None)}, worst quantile '
              f'{run.config["conditional_worst_quantile"]}. Excess is in kT above each molecule\'s best search '
              f'minimum (u* = {tr.get("u_star", float("nan")):.0f}); jensen_z standard errors '
              f'{tr["jensen_z_se"]:.2f} (train) and {te["jensen_z_se"]:.2f} (held-out).')
        print(f'{"metric":44s} {"unit":>13s} {"train":>9s} {"held-out":>9s} {"held-out - train":>17s}')
        for label, key, unit in ROWS:
            a, b = tr.get(key, float('nan')), te.get(key, float('nan'))
            print(f'{label:44s} {unit:>13s} {a:9.3f} {b:9.3f} {b - a:17.3f}')

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, 'topline.json'), 'w') as fh:
            json.dump({str(k): v for k, v in results.items()}, fh, indent=1, default=float)


if __name__ == '__main__':
    main()
