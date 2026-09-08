import pickle, glob, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
CANCELLED = {'acr_lr0p025','mipu_lr0p015625','mipu_lr0p03125','mipu_lr0p0625','mipu_lr0p125','mipu_lr0p25','neh_lr0p5'}
def roll(ev, col, w):
    s = ev[['_runtime', col]].dropna().sort_values('_runtime')
    ser = pd.Series(s[col].values, index=pd.to_timedelta(s['_runtime'].values, unit='s'))
    return s['_runtime'].values/3600, ser.rolling(w).mean().values
for fn in sorted(glob.glob('data/*.pkl')):
    d = pickle.load(open(fn,'rb')); m=d['meta']; h=d['hist']; ev=d['events']
    if h is None or len(h)==0: continue
    nm = m['name'].replace('p02_p02_','')
    t0 = h['_timestamp'].min()
    h = h.copy(); h['hr'] = (h['_timestamp']-t0)/3600
    ev = ev.copy(); ev['hr'] = ev['_runtime']/3600
    # events _runtime is relative to run start; history _timestamp-t0 too (t0 first logged row, at step0). small offset ok
    fig, axes = plt.subplots(7,1, figsize=(16,20), sharex=True)
    evals = h.loc[h['eval_step_time'].notna(),'hr'].values if 'eval_step_time' in h else []
    def evlines(ax):
        for e in evals: ax.axvline(e, color='k', alpha=0.15, lw=0.8)
    ax=axes[0]; ax.plot(h['hr'], h['train_step_time'], '.', ms=2, color='C0', label='train_step_time (s)')
    if 'energy/seconds_in_step' in h:
        ax.plot(h['hr'], h['energy/seconds_in_step']/10, '.', ms=2, color='C3', label='energy s per step (energy/seconds_in_step /10)')
        ax.plot(h['hr'], h['train_step_time'] - h['energy/seconds_in_step']/10, '.', ms=2, color='C2', label='non-energy s per step')
    if 'z_cal/seconds' in h: ax.plot(h['hr'], h['z_cal/seconds']/10, '.', ms=2, color='C4', label='z_cal s per step')
    ax.set_ylim(0, np.nanpercentile(h['train_step_time'],99.5)*1.6); ax.legend(fontsize=7, ncol=4); ax.set_ylabel('s/step'); evlines(ax)
    ax.set_title(f"{nm}  [{'CANCELLED' if nm in CANCELLED else 'survived/running'}]  host {m['host']}  batch {m['config'].get('batch_size')}  steps {int(h['_step'].min())}-{int(h['_step'].max())}")
    ax=axes[1]
    g = ev[['hr','system.gpu.0.gpu']].dropna(); ax.plot(g['hr'], g['system.gpu.0.gpu'], '-', lw=0.3, color='0.6', label='system.gpu.0.gpu raw')
    for w,c in (('3600s','C1'),('7200s','C3')):
        x,y = roll(ev,'system.gpu.0.gpu',w); ax.plot(x,y,color=c,lw=1.5,label=f'trailing {w} mean')
    if 'gpu/util_policy' in h: ax.plot(h['hr'], h['gpu/util_policy'], color='C0', lw=1, label='gpu/util_policy (in-process 2h)')
    if 'gpu/util_recent' in h: ax.plot(h['hr'], h['gpu/util_recent'], color='C9', lw=0.7, label='gpu/util_recent (15 min)')
    ax.axhline(55, color='r', ls='--', lw=0.8); ax.set_ylim(0,100); ax.legend(fontsize=7, ncol=5); ax.set_ylabel('GPU util %'); evlines(ax)
    ax=axes[2]
    for col,c in (('system.gpu.0.memory','C1'),('system.gpu.0.memoryAllocated','C2'),('system.gpu.0.powerPercent','C3')):
        if col in ev: s=ev[['hr',col]].dropna(); ax.plot(s['hr'], s[col], lw=0.5, color=c, label=col)
    ax.set_ylim(0,100); ax.legend(fontsize=7, ncol=3); ax.set_ylabel('%'); evlines(ax)
    ax=axes[3]
    s=ev[['hr','system.cpu']].dropna(); ax.plot(s['hr'], s['system.cpu'], lw=0.5, color='C0', label='system.cpu (process CPU %)')
    ax2=ax.twinx(); s=ev[['hr','system.proc.cpu.threads']].dropna(); ax2.plot(s['hr'], s['system.proc.cpu.threads'], lw=0.7, color='C3', label='proc threads'); ax2.set_ylabel('threads')
    ax.legend(fontsize=7, loc='upper left'); ax2.legend(fontsize=7, loc='upper right'); ax.set_ylabel('CPU %'); evlines(ax)
    ax=axes[4]
    s=ev[['hr','system.proc.memory.rssMB']].dropna(); ax.plot(s['hr'], s['system.proc.memory.rssMB']/1024, lw=0.7, color='C0', label='proc RSS (GB)')
    ax2=ax.twinx(); s=ev[['hr','system.memory_percent']].dropna(); ax2.plot(s['hr'], s['system.memory_percent'], lw=0.7, color='C3', label='system memory %'); ax2.set_ylim(0,100)
    ax.legend(fontsize=7, loc='upper left'); ax2.legend(fontsize=7, loc='upper right'); ax.set_ylabel('GB'); evlines(ax)
    ax=axes[5]
    for col,c in (('system.disk.nvme0n1p2.in','C0'),('system.disk.nvme0n1p2.out','C1'),('system.network.sent','C2'),('system.network.recv','C3')):
        if col in ev: s=ev[['hr',col]].dropna(); v=s[col].values; ax.plot(s['hr'], v, lw=0.6, color=c, label=col)
    ax.set_yscale('symlog'); ax.legend(fontsize=7, ncol=4); ax.set_ylabel('MB (cumulative?)'); evlines(ax)
    ax=axes[6]
    for col,c in (('replay/absorption_n','C0'),('replay/absorbed_frac','C1'),('samples_per_sec','C2'),('Batch Size','C3')):
        if col in h:
            v=h[col]; v = v/ np.nanmax(np.abs(v)) if np.nanmax(np.abs(v))>0 else v
            ax.plot(h['hr'], v, '.', ms=2, color=c, label=col+' (normalised)')
    ax.legend(fontsize=7, ncol=4); ax.set_xlabel('hours since first logged step'); evlines(ax)
    fig.tight_layout(); fig.savefig(f'figs/run_{nm}.png', dpi=80); plt.close(fig)
    print('wrote', nm)
