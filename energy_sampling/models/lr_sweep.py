"""Learning-rate bracket and training-recipe fix for the encoder probe battery.

WHY THIS EXISTS. Every battery run before 2026-09-01 was contaminated by a training
COLLAPSE: per-task loss falls for ~1,500 steps, a gradient spike hits, and both tasks snap
back to their starting loss and stay there. Same architecture, same data, same seed reaches
0% or 100% depending only on the recipe, so no architecture claim measured under it survives.

METHOD, following the project's brute-force bracket
(docs/lr_controller_spec.md, project_lr_brute_force_bracket):

  * FULL-HORIZON TRIALS, not short ones. The recorded finding is that a 150-step trial
    overestimates the sustained boundary by ~2x because slow destabilisation lives at 400+
    steps -- and our collapse lands at ~1,750. A short probe here would certify a rate that
    detonates later, which is the exact failure that protocol was written to stop.
  * NO RAMP UNDER A CANDIDATE in phase 1. "A continuous multiplier under a candidate rung
    means the rate under test is not the rate applied." Schedules are tested separately, in
    phase 2, as recipes in their own right.
  * SELECTION IS NOT SURVIVAL-ONLY. The recorded finding is that survival-only selection is
    blind to loss-level regression. A rung is scored on its FINAL loss, and separately
    flagged for collapse (final materially worse than its own best).
  * CONFIRMATION ON A DIFFERENT SEED. A same-seed re-run restores an identical RNG state and
    confirms nothing.

The chirality tripwire doubles as the collapse canary: ~100% when healthy, 0% when collapsed,
with nothing in between, because a +/-1 target scored by rounding is a step function.

    python -m models.lr_sweep --phase 1
    python -m models.lr_sweep --phase 2 --lr 3e-4
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

import models.encoder_probe as ep

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
PROBES_USED = ('eccentricity', 'spd_to_marked', 'cip_code')


def build(n_mol, k=16, encoding='none'):
    smis = ep.load_qm9_stereo(n_mol)
    S = []
    for s in smis:
        try:
            S.append(ep.build_sample(s, encoding, k))
        except Exception:
            pass
    order = np.random.default_rng(12345).permutation(len(S))
    return [S[i] for i in order]


def train(arm, tr, te, steps, lr, seed, device, warmup=0, cosine=False, clip=5.0,
          hidden=128, layers=4, k=16, batch=128):
    """One trial. Returns the loss trace and the tripwire accuracy at intervals."""
    cfg = ep.ARMS[arm]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = ep.target_stats(tr, device)
    m = ep.ProbeModel(tr[0].x.shape[1], k, tr[0].edge_attr.shape[1], hidden=hidden,
                      layers=layers, attention=cfg['attention']).to(device)
    m.weighting = 'none'
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    eb = ep.collate(te, device, cfg['spd'])
    trace, best_state, best_loss = [], None, math.inf

    for step in range(steps):
        if warmup and step < warmup:
            scale = (step + 1) / warmup
        elif cosine:
            t = (step - warmup) / max(1, steps - warmup)
            scale = 0.5 * (1 + math.cos(math.pi * t))
        else:
            scale = 1.0
        for g in opt.param_groups:
            g['lr'] = lr * scale
        idx = rng.choice(len(tr), min(batch, len(tr)), replace=False)
        b = ep.collate([tr[i] for i in idx], device, cfg['spd'])
        opt.zero_grad()
        total, parts = m.loss(m(b), b, mu, sd)
        total.backward()
        gn = float(nn.utils.clip_grad_norm_(m.parameters(), clip))
        opt.step()
        cur = float(total.detach())
        if cur < best_loss:
            # KEEP THE BEST, not the last. Half the seeds in the 2026-08-31 sweep ended
            # worse than their own minimum, so 'final' was never the right thing to score.
            best_loss = cur
            best_state = {kk: v.detach().clone() for kk, v in m.state_dict().items()}
        if step % 100 == 0 or step == steps - 1:
            trace.append({'step': step, 'loss': cur, 'grad': gn, 'parts': parts})

    def score(state):
        m.load_state_dict(state)
        m.eval()
        out = ep.score(m, [eb], mu, sd)
        m.train()
        return {p: out[p]['exact'] for p in out}

    return {'arm': arm, 'lr': lr, 'seed': seed, 'warmup': warmup, 'cosine': cosine,
            'clip': clip, 'steps': steps, 'trace': trace,
            'final_loss': trace[-1]['loss'], 'best_loss': best_loss,
            'final_scores': score({kk: v for kk, v in m.state_dict().items()}),
            'best_scores': score(best_state)}


def verdict(r):
    """Collapse = the run ended materially worse than its own best."""
    span = max(r['trace'][0]['loss'] - r['best_loss'], 1e-9)
    regress = (r['final_loss'] - r['best_loss']) / span
    return {'collapsed': regress > 0.25, 'regress_frac': regress}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', type=int, default=1, choices=[1, 2, 3])
    ap.add_argument('--arm', default='mp')
    ap.add_argument('--n-mol', type=int, default=3000)
    ap.add_argument('--n-test', type=int, default=600)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--lrs', type=float, nargs='+',
                    default=[3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)

    ep.PROBES = [p for p in ep.PROBES if p.name in PROBES_USED]
    pool = build(a.n_mol + a.n_test)
    te, tr = pool[:a.n_test], pool[a.n_test:]
    print(f'{len(tr)} train / {len(te)} test molecules; probes {[p.name for p in ep.PROBES]}')

    rows = []
    if a.phase == 1:
        print('\nPHASE 1 -- fixed-LR bracket, FULL horizon, no ramp under the candidate')
        for lr in a.lrs:
            for seed in a.seeds:
                r = train(a.arm, tr, te, a.steps, lr, seed, a.device)
                v = verdict(r)
                r.update(v)
                rows.append(r)
                cip = 100 * r['best_scores'].get('cip_code', 0)
                print(f"  lr {lr:8.1e} seed {seed}: best loss {r['best_loss']:7.3f}  "
                      f"final {r['final_loss']:7.3f}  regress {100*v['regress_frac']:5.1f}%  "
                      f"{'COLLAPSED' if v['collapsed'] else 'stable   '}  "
                      f"tripwire {cip:5.1f}%")
    else:
        print(f'\nPHASE {a.phase} -- schedule variants at lr {a.lr:.1e}')
        variants = [('flat', 0, False), ('warmup', 300, False),
                    ('cosine', 0, True), ('warmup+cosine', 300, True)]
        for name, wu, cos in variants:
            for seed in a.seeds:
                r = train(a.arm, tr, te, a.steps, a.lr, seed, a.device, warmup=wu, cosine=cos)
                v = verdict(r)
                r.update(v, name=name)
                rows.append(r)
                cip = 100 * r['best_scores'].get('cip_code', 0)
                print(f"  {name:<14} seed {seed}: best {r['best_loss']:7.3f}  "
                      f"final {r['final_loss']:7.3f}  regress {100*v['regress_frac']:5.1f}%  "
                      f"{'COLLAPSED' if v['collapsed'] else 'stable   '}  tripwire {cip:5.1f}%")

    print(f"\n{'recipe':<22}{'best loss':>11}{'final':>9}" +
          ''.join(f'{p:>16}' for p in PROBES_USED))
    for r in rows:
        tag = r.get('name', f"lr {r['lr']:.0e}")
        print(f"{tag:<22}{r['best_loss']:>11.3f}{r['final_loss']:>9.3f}" +
              ''.join(f"{100*r['best_scores'].get(p,0):>15.1f}%" for p in PROBES_USED))

    out = a.out or os.path.join(RESULTS, f'lr_sweep_phase{a.phase}.json')
    os.makedirs(RESULTS, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'rows': rows, 'arm': a.arm, 'steps': a.steps}, f, indent=2)
    print(f'wrote {out}')
    return rows


if __name__ == '__main__':
    main()
