"""(e) What phase 1 does to P_B and what phase 2 does after -- read off wandb.

    python configs/local_pb_freeze/analyze_cluster.py <dir with *.pkl histories>

pt100_*_lr4p0 runs are the PHASE-1 sources (train_prior only, MLE on dataset
terminals); the p02 arms warm-start from their phase1_exit and run
equilibration (fused fwd/bwd/replay, fixed fracs, fixed LR scale). gradnorm/*
is the per-submodel L2 norm of the CURRENT loss's gradient, so across the
boundary the loss changes (MLE -> fused TB); the pb/pf RATIO is the comparable
quantity, the absolute level is not.
"""
import sys, os, re, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = sys.argv[1]
out = os.path.dirname(os.path.abspath(__file__))
fams = ['mip', 'neh', 'mipu', 'nehu', 'acr']

def load(name):
    df = pd.read_pickle(os.path.join(src, name + '.pkl'))
    return df.sort_values('_step')

def roll(s, w):
    return s.rolling(w, min_periods=max(1, w // 3), center=True).median()

# ---------------- phase 1 sources ----------------
fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex='col')
rows = []
for j, fam in enumerate(fams):
    df = load(f'pt100_pt100_{fam}_lr4p0')
    g = df[['_step', 'gradnorm/backward_policy', 'gradnorm/forward_policy', 'gradnorm/s_model', 'bwd/mle']].dropna(subset=['gradnorm/backward_policy'])
    pb, pf = g['gradnorm/backward_policy'], g['gradnorm/forward_policy']
    ax = axes[0, j]
    ax.plot(g._step, roll(pb, 15), label='backward_policy')
    ax.plot(g._step, roll(pf, 15), label='forward_policy')
    ax.plot(g._step, roll(g['gradnorm/s_model'], 15), label='s_model', alpha=.6)
    ax.set_yscale('log'); ax.set_title(f'{fam}: phase-1 (MLE) gradnorm'); ax.legend(fontsize=7)
    ax2 = axes[1, j]
    ax2.plot(g._step, roll(pb / pf, 15), color='k'); ax2.set_ylim(0, 1.2)
    ax2.set_title('pb/pf gradnorm ratio'); ax2.set_xlabel('step')
    m = df[['_step', 'bwd/mle']].dropna()
    ax3 = ax2.twinx(); ax3.plot(m._step, roll(m['bwd/mle'], 15), color='C3', alpha=.5); ax3.set_ylabel('bwd/mle', color='C3')
    n = len(g); q = max(1, n // 10)
    rows.append(dict(fam=fam, steps=int(g._step.max()),
                     pb_first10=float(pb.iloc[:q].median()), pb_last10=float(pb.iloc[-q:].median()),
                     pf_first10=float(pf.iloc[:q].median()), pf_last10=float(pf.iloc[-q:].median()),
                     ratio_first10=float((pb / pf).iloc[:q].median()), ratio_last10=float((pb / pf).iloc[-q:].median()),
                     mle_first=float(m['bwd/mle'].iloc[:5].median()), mle_last=float(m['bwd/mle'].iloc[-5:].median())))
plt.tight_layout(); plt.savefig(os.path.join(out, 'e1_phase1_pb_gradnorm.png'), dpi=110); plt.close()
p1 = pd.DataFrame(rows); print('PHASE 1 (pt100 lr4p0 sources)'); print(p1.round(3).to_string(index=False))
p1.to_csv(os.path.join(out, 'e1_phase1_pb_gradnorm.csv'), index=False)

# ---------------- phase 2 arms ----------------
arms = sorted(glob.glob(os.path.join(src, 'p02_p02_*.pkl')))
rows = []
fig, axes = plt.subplots(2, 5, figsize=(20, 7))
famax = {f: i for i, f in enumerate(fams)}
for fn in arms:
    name = os.path.basename(fn)[:-4]
    m = re.match(r'p02_p02_(\w+?)_lr(\S+)', name)
    fam, sc = m.group(1), float(m.group(2).replace('p', '.'))
    df = pd.read_pickle(fn)
    if len(df) < 100 or '_step' not in df: continue
    df = df.sort_values('_step')
    g = df[['_step', 'gradnorm/backward_policy', 'gradnorm/forward_policy', 'gradnorm/s_model', 'lr_fused']].dropna(subset=['gradnorm/backward_policy'])
    pb, pf = g['gradnorm/backward_policy'], g['gradnorm/forward_policy']
    s0 = g._step.iloc[0]
    rel = g._step - s0
    ax = axes[0, famax[fam]]
    ax.plot(rel, roll(pb / pf, 25), label=f'scale {sc:g}')
    ax.set_title(f'{fam}: phase-2 pb/pf gradnorm ratio'); ax.set_ylim(0, 0.8); ax.legend(fontsize=7)
    ax = axes[1, famax[fam]]
    ax.plot(rel, roll(pb, 25), label=f'scale {sc:g}'); ax.set_yscale('log'); ax.set_title('gradnorm/backward_policy'); ax.set_xlabel('steps since phase-1 exit')
    # logpb gaps
    lp = df[['_step', 'fwd/logpb_mean', 'bwd/logpb_mean', 'replay/logpb_mean', 'fwd/logpf_mean', 'replay/logpf_mean']].dropna()
    n = len(g); q = max(1, n // 5)
    lr = 1.25e-4 * sc  # seed_lr is 1.25e-4 on every p02 arm; lr = seed_lr * fixed_scale
    memo = df['replay/resid_vs_intake'].dropna()
    rows.append(dict(fam=fam, scale=sc, lr=lr, steps=int(g._step.max() - s0),
                     ratio_first20=float((pb / pf).iloc[:q].median()), ratio_last20=float((pb / pf).iloc[-q:].median()),
                     pb_first20=float(pb.iloc[:q].median()), pb_last20=float(pb.iloc[-q:].median()),
                     pf_last20=float(pf.iloc[-q:].median()),
                     replay_minus_fwd_logpb=float((lp['replay/logpb_mean'] - lp['fwd/logpb_mean']).iloc[-q:].median()) if len(lp) else np.nan,
                     bwd_minus_fwd_logpb=float((lp['bwd/logpb_mean'] - lp['fwd/logpb_mean']).iloc[-q:].median()) if len(lp) else np.nan,
                     replay_minus_fwd_logpf=float((lp['replay/logpf_mean'] - lp['fwd/logpf_mean']).iloc[-q:].median()) if len(lp) else np.nan,
                     memo_last=float(memo.iloc[-max(1, len(memo)//5):].median()) if len(memo) else np.nan))
plt.tight_layout(); plt.savefig(os.path.join(out, 'e2_phase2_pb_gradnorm.png'), dpi=110); plt.close()
p2 = pd.DataFrame(rows).sort_values(['fam', 'scale']); print('\nPHASE 2 (p02 arms)'); print(p2.round(3).to_string(index=False))
p2.to_csv(os.path.join(out, 'e2_phase2_pb_gradnorm.csv'), index=False)

# LR dependence of the P_B share and of its absolute motion
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for fam in fams:
    d = p2[p2.fam == fam]
    ax[0].plot(d.lr, d.ratio_last20, 'o-', label=fam)
    ax[1].plot(d.lr, d.pb_last20, 'o-', label=fam)
ax[0].set_xscale('log'); ax[0].set_xlabel('lr_fused'); ax[0].set_ylabel('pb/pf gradnorm ratio (last 20%)'); ax[0].legend()
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('lr_fused'); ax[1].set_ylabel('gradnorm/backward_policy (last 20%)'); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(out, 'e3_pb_share_vs_lr.png'), dpi=110); plt.close()

# ---------------- the boundary: phase-1 tail -> phase-2 head, per family ----------------
# pt100 ran train_prior ONLY at scale 4.0 (T=100); p02 arms warm-start from its exit.
fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
for j, fam in enumerate(fams):
    src1 = load(f'pt100_pt100_{fam}_lr4p0')
    g1 = src1[['_step', 'gradnorm/backward_policy', 'gradnorm/forward_policy']].dropna()
    ax = axes[j]
    ax.plot(g1._step - g1._step.max(), roll(g1['gradnorm/backward_policy'] / g1['gradnorm/forward_policy'], 15), color='k', label='phase 1 (MLE, scale 4.0)')
    for fn in arms:
        name = os.path.basename(fn)[:-4]
        m = re.match(r'p02_p02_(\w+?)_lr(\S+)', name)
        if m.group(1) != fam: continue
        df = pd.read_pickle(fn)
        if len(df) < 100 or '_step' not in df: continue
        df = df.sort_values('_step')
        g = df[['_step', 'gradnorm/backward_policy', 'gradnorm/forward_policy']].dropna()
        ax.plot(g._step - g._step.iloc[0], roll(g['gradnorm/backward_policy'] / g['gradnorm/forward_policy'], 15), label=f'phase 2 scale {m.group(2).replace("p", ".")}', alpha=.8)
    ax.axvline(0, color='r', ls='--'); ax.set_xlim(-4000, 8000); ax.set_ylim(0, 1.0)
    ax.set_title(f'{fam}: pb/pf gradnorm ratio across the exit'); ax.set_xlabel('steps from phase-1 exit'); ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(out, 'e4_boundary_pb_ratio.png'), dpi=110); plt.close()
