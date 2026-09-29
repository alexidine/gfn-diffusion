"""Exploration curve of a capped_mc run: coverage % against effort, merged over its shards.

    python data_processing/capped_mc_explore.py <run dir>     (a dir of shard_* outputs, or one shard's own dir)

Reads every exploration.jsonl under the dir (written by capped_mc.py at each checkpoint). Per checkpoint step reached
by every shard: walker-steps and energy evaluations summed over shards, wall time the maximum (shards run in parallel),
and per radius the exploration % = 100 x (1 - new fraction), the new fraction weighted by each shard's walkers. Each
shard keeps its own cover, so a state new to one shard may already be covered by another's minima: the merged figure
is a slight UNDER-estimate of coverage where neighbouring minima fell into different shards.
Writes exploration.png beside the logs when matplotlib is available.
"""
import glob
import json
import os
import sys

run = sys.argv[1]
logs = sorted(glob.glob(os.path.join(run, '**', 'exploration.jsonl'), recursive=True))
if not logs:
    raise SystemExit(f'no exploration.jsonl under {run}')
per_shard = []
for f in logs:
    rows = [json.loads(ln) for ln in open(f) if ln.strip()]
    per_shard.append({r['step']: r for r in rows})
radii = sorted({k[4:] for s in per_shard for r in s.values() for k in r if k.startswith('new_')}, key=float)
steps = sorted(set.intersection(*[set(s) for s in per_shard]))
print(f'{run}: {len(logs)} shard log(s); checkpoints reached by every shard: {len(steps)}')
print('Exploration % = share of the walkers\' current states already within r (atomwise/envwise RDF distance) of '
      'something seen before, merged over shards; effort columns are totals over shards (wall time = slowest shard).')
hdr = f"{'step':>6} {'walker-steps':>13} {'evals':>11} {'wall (h)':>9} " + ' '.join(f"{'r ' + r:>12}" for r in radii)
print(hdr)
curve = []
for st in steps:
    rs = [s[st] for s in per_shard]
    ws, ev, wall = sum(r['walker_steps'] for r in rs), sum(r['evals'] for r in rs), max(r['wall'] for r in rs)
    nw = sum(r['n_walkers'] for r in rs)
    cov = {rad: 100 * (1 - sum(r[f'new_{rad}'] * r['n_walkers'] for r in rs) / nw) for rad in radii}
    full = any(r.get(f'full_{rad}') for r in rs for rad in radii)
    flag = ' (resumed)' if any(r.get('resumed') for r in rs) else ''
    print(f'{st:>6} {ws:>13} {ev:>11} {wall / 3600:>9.2f} ' + ' '.join(f'{cov[rad]:>11.1f}%' for rad in radii)
          + (' [a cover hit its cap]' if full else '') + flag)
    curve.append((ws, wall, cov))
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for rad in radii:
        axs[0].plot([c[0] for c in curve], [c[2][rad] for c in curve], marker='o', ms=3, label=f'r = {rad}')
        axs[1].plot([c[1] / 3600 for c in curve], [c[2][rad] for c in curve], marker='o', ms=3, label=f'r = {rad}')
    axs[0].set_xlabel('walker-steps (all shards)')
    axs[1].set_xlabel('wall time (h, slowest shard)')
    for ax in axs:
        ax.set_ylabel('exploration % (states already covered)')
        ax.set_ylim(0, 101)
        ax.legend()
    fig.suptitle(os.path.basename(os.path.normpath(run)))
    fig.tight_layout()
    out = os.path.join(run, 'exploration.png')
    fig.savefig(out, dpi=120)
    print(f'wrote {out}')
except Exception as e:  # plotting is a convenience; the table above is the result
    print(f'(no figure: {e})')
