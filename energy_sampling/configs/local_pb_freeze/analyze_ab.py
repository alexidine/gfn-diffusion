"""(f)+(g): read the pbab arms off wandb -- curves, not summaries.

    python configs/local_pb_freeze/analyze_ab.py [cache_dir]

Pulls every run tagged `pbab` (full scan_history, cached as pickle in cache_dir),
draws ab_curves.png (one panel per channel, one line per arm) and writes
ab_summary.csv (median over the last 25% of equilibration per channel per arm).
The per-branch per-submodel gradient norms (fused_grad/{branch}_norm_{submodel})
exist only where grad_geometry ran, i.e. on the trainable arm they are the (f)
measurement; on the frozen arms backward_policy must read 0.
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = os.path.dirname(os.path.abspath(__file__))
cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(out, '_cache')
os.makedirs(cache, exist_ok=True)

ARMS = ['pbab_train', 'pbab_headfrozen', 'pbab_frozen']

def pull():
    import wandb
    api = wandb.Api(timeout=120)
    runs = {}
    for r in api.runs("mkilgour/GFN Energy", filters={"config.tag": "pbab"}, order="-created_at"):
        rn = r.config.get('run_name')
        if rn not in ARMS or rn in runs:
            continue  # newest run per arm wins (a killed earlier launch is ignored)
        fn = os.path.join(cache, f'{rn}.pkl')
        if not os.path.exists(fn) or r.state == 'running':
            df = pd.DataFrame(list(r.scan_history(keys=None, page_size=5000)))
            df.to_pickle(fn)
        df = pd.read_pickle(fn)
        if '_step' not in df:
            print(rn, r.state, 'no history yet -- skipped'); continue
        runs[rn] = df.sort_values('_step')
        print(rn, r.state, len(runs[rn]))
    return runs

runs = pull()

CH = [  # (key, title, log-scale)
    ('fwd/tb_err', 'fwd/tb_err', True),
    ('fwd/tb_err_worst', 'fwd/tb_err_worst', True),
    ('fwd/scatter_err', 'fwd/scatter_err', False),
    ('fwd/over_coverage', 'fwd/over_coverage', False),
    ('bwd/under_coverage', 'bwd/under_coverage', False),
    ('replay/resid_vs_intake', 'replay/resid_vs_intake (memo sensor)', False),
    ('replay/tb_err', 'replay/tb_err', True),
    ('bwd/tb_err', 'bwd/tb_err', True),
    ('fwd/logw_std_within', 'fwd/logw_std_within', False),
    ('bwd/logw_std_within', 'bwd/logw_std_within', False),
    ('fwd/jensen_z', 'fwd/jensen_z (mean log w)', False),
    ('log Z learned', 'log Z learned', False),
    ('Mean Sample Energy', 'Mean Sample Energy (fwd)', False),
    ('fwd/logr_mean', 'fwd/logr_mean', False),
    ('gradnorm/backward_policy', 'gradnorm/backward_policy (fused loss)', True),
    ('gradnorm/forward_policy', 'gradnorm/forward_policy (fused loss)', True),
    ('fused_grad/bwd_norm_backward_policy', 'bwd branch grad on backward_policy', True),
    ('fused_grad/replay_norm_backward_policy', 'replay branch grad on backward_policy', True),
    ('fused_grad/bwd_norm_forward_policy', 'bwd branch grad on forward_policy', True),
    ('fused_grad/replay_norm_forward_policy', 'replay branch grad on forward_policy', True),
    ('fused_grad/cos_bwd_replay', 'cos(bwd, replay) whole-model', False),
    ('pb_gap', 'replay/logpb - fwd/logpb', False),
    ('pf_gap', 'replay/logpf - fwd/logpf', False),
    ('fwd/step_var', 'fwd/step_var', False),
]

def roll(s, w=9):
    return s.rolling(w, min_periods=1, center=True).median()

for rn, df in runs.items():
    if {'replay/logpb_mean', 'fwd/logpb_mean'} <= set(df.columns):
        df['pb_gap'] = df['replay/logpb_mean'] - df['fwd/logpb_mean']
        df['pf_gap'] = df['replay/logpf_mean'] - df['fwd/logpf_mean']

n = len(CH); cols = 4; rows_ = (n + cols - 1) // cols
fig, axes = plt.subplots(rows_, cols, figsize=(5 * cols, 3.2 * rows_))
summary = []
for (key, title, logy), ax in zip(CH, axes.flat):
    for rn in ARMS:
        if rn not in runs or key not in runs[rn]:
            continue
        d = runs[rn][['_step', key]].dropna()
        if len(d) == 0:
            continue
        ax.plot(d._step, roll(d[key]), label=rn, alpha=.85)
    ax.set_title(title, fontsize=9); ax.legend(fontsize=6)
    if logy: ax.set_yscale('log')
for ax in list(axes.flat)[n:]:
    ax.axis('off')
plt.tight_layout(); plt.savefig(os.path.join(out, 'ab_curves.png'), dpi=100); plt.close()

for key, title, _ in CH:
    row = {'channel': key}
    for rn in ARMS:
        if rn in runs and key in runs[rn]:
            d = runs[rn][['_step', key]].dropna()
            d = d[d._step >= 300]  # equilibration engages at ~200 + burn-in 100
            q = max(1, len(d) // 4)
            row[rn + '_last25'] = float(d[key].iloc[-q:].median()) if len(d) else np.nan
            row[rn + '_first25'] = float(d[key].iloc[:q].median()) if len(d) else np.nan
    summary.append(row)
s = pd.DataFrame(summary)
s.to_csv(os.path.join(out, 'ab_summary.csv'), index=False)
pd.set_option('display.width', 250)
print(s.round(3).to_string(index=False))

# (f): the split of P_B's gradient by branch, trainable arm only
if 'pbab_train' in runs:
    df = runs['pbab_train']
    keys = [c for c in df.columns if c.startswith('fused_grad/') and '_norm_' in c]
    if keys:
        g = df[['_step'] + keys].dropna()
        print('\n(f) per-branch per-submodel gradient norms, pbab_train, median over equilibration:')
        print(g[g._step >= 300][keys].median().round(2).to_string())
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
        for i, sub in enumerate(['backward_policy', 'forward_policy', 's_model']):
            for br in ['bwd', 'replay', 'fwd']:
                k = f'fused_grad/{br}_norm_{sub}'
                if k in g:
                    ax[i].plot(g._step, roll(g[k]), label=br)
            ax[i].set_yscale('log'); ax[i].set_title(f'branch gradient norm on {sub}'); ax[i].legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(out, 'f_branch_split.png'), dpi=110); plt.close()
