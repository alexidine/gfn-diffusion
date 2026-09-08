import pickle, glob, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
CANCELLED = {'acr_lr0p025','mipu_lr0p015625','mipu_lr0p03125','mipu_lr0p0625','mipu_lr0p125','mipu_lr0p25','neh_lr0p5'}
THR = 54.0
pd.set_option('display.width',260)
runs={}
for fn in sorted(glob.glob('data/*.pkl')):
    d = pickle.load(open(fn,'rb')); m=d['meta']; h=d['hist']; ev=d['events']
    if h is None or len(h)==0: continue
    nm = m['name'].replace('p02_p02_','')
    h=h.copy(); h['hr']=(h['_timestamp']-h['_timestamp'].min())/3600
    ev=ev.copy(); ev['hr']=ev['_runtime']/3600
    runs[nm]=(m,h,ev)
rows=[]; fam = {}
for nm,(m,h,ev) in runs.items():
    s = ev[['hr','system.gpu.0.gpu']].dropna().sort_values('hr')
    ser = pd.Series(s['system.gpu.0.gpu'].values, index=pd.to_timedelta(s['hr'].values, unit='h'))
    u2 = ser.rolling('7200s').mean(); u2[ser.index < pd.Timedelta(hours=2)] = np.nan
    u1 = ser.rolling('3600s').mean(); u1[ser.index < pd.Timedelta(hours=1)] = np.nan
    end = s['hr'].iloc[-1]
    # step-time predictor: B = median(util*t) over hours 1..3 of the run (fast/base regime), predicted util = B / trailing-1h median step time
    st = h[['hr','train_step_time']].dropna()
    st = st[st['hr']>1.0]
    # per-sample util at the step's time
    ui = np.interp(st['hr'].values, s['hr'].values, s['system.gpu.0.gpu'].values)
    B_run = np.nanmedian(ui*st['train_step_time'].values)   # GPU-busy s/step over the run
    stser = pd.Series(st['train_step_time'].values, index=pd.to_timedelta(st['hr'].values, unit='h'))
    t1h = stser.rolling('3600s').median(); t15 = stser.rolling('900s').median()
    pred1h = B_run / t1h; pred15 = B_run / t15
    def first_below(series, thr, min_hr):
        x = series[(series.index > pd.Timedelta(hours=min_hr)) & (series < thr)]
        return x.index[0].total_seconds()/3600 if len(x) else np.nan
    r = dict(run=nm, canc=nm in CANCELLED, batch=int(m['config'].get('batch_size')), end_h=round(end,1),
             B_gpu_s_per_step=round(B_run/100,2), t_crit_s=round(B_run/THR,1),
             t_base=round(np.nanmedian(stser[stser.index<pd.Timedelta(hours=4)]),2), t_end1h=round(t1h.iloc[-1],2),
             u2h_end=round(u2.iloc[-1],1), first_u2h_below=round(first_below(u2,THR,2),1), first_u1h_below=round(first_below(u1,THR,1),1),
             first_pred1h_below=round(first_below(pred1h,THR,1),1), first_pred15_below=round(first_below(pred15,THR,1),1),
             pred1h_min=round(np.nanmin(pred1h),1))
    r['lead_u1h_h'] = round(end - r['first_u1h_below'],1) if np.isfinite(r['first_u1h_below']) else None
    r['lead_pred15_h'] = round(end - r['first_pred15_below'],1) if np.isfinite(r['first_pred15_below']) else None
    rows.append(r)
    fam.setdefault(nm.split('_')[0], []).append((nm, s, u2, pred1h, end))
df=pd.DataFrame(rows).sort_values(['canc','run'])
print(df.to_string(index=False)); df.to_csv('predictor.csv', index=False)

# FIGURE 1: trailing-2h device util per family, kills marked
fams = ['mipu','nehu','acr','mip','neh']
fig, axes = plt.subplots(len(fams),1, figsize=(15,3.2*len(fams)), sharex=False)
for ax,f in zip(axes,fams):
    for nm,s,u2,pred,end in fam.get(f,[]):
        c = 'C3' if nm in CANCELLED else 'C0'
        x = u2.index.total_seconds()/3600
        ax.plot(x, u2.values, color=c, lw=1.2, alpha=0.9, label=nm+(' KILLED' if nm in CANCELLED else ''))
        ax.plot(x, pred.reindex(u2.index, method='nearest').values if False else np.nan*x, lw=0)
        if nm in CANCELLED: ax.plot([end],[u2.dropna().iloc[-1] if len(u2.dropna()) else np.nan],'x',color=c,ms=10,mew=2)
    ax.axhline(THR, color='k', ls='--', lw=0.8); ax.set_ylim(35,90); ax.set_ylabel('trailing 2h mean\nsystem.gpu.0.gpu (%)')
    ax.legend(fontsize=7, ncol=3, loc='upper right'); ax.grid(alpha=0.3); ax.set_title(f'{f} arms  (x = kill)', fontsize=10)
axes[-1].set_xlabel('hours since run start')
fig.tight_layout(); fig.savefig('figs/A_trailing2h_util_by_family.png', dpi=90); plt.close(fig)

# FIGURE 2: util vs 1/step_time -- constancy of GPU-busy seconds per step
fig, axes = plt.subplots(1,3, figsize=(16,5))
groups = {'UMA (mipu, nehu)': [n for n in runs if n.startswith(('mipu','nehu'))], 'MACE (acr)': [n for n in runs if n.startswith('acr')], 'ELJ (mip, neh)': [n for n in runs if n.startswith(('mip_','neh_'))]}
for ax,(title,names) in zip(axes, groups.items()):
    for i,nm in enumerate(names):
        m,h,ev = runs[nm]
        st = h[['hr','train_step_time']].dropna(); st = st[st['hr']>1.0]
        s = ev[['hr','system.gpu.0.gpu']].dropna().sort_values('hr')
        # 30-min bins
        b = (st['hr']//0.5)
        tt = st.groupby(b)['train_step_time'].median()
        ss = pd.Series(s['system.gpu.0.gpu'].values, index=(s['hr']//0.5)).groupby(level=0).mean()
        j = tt.index.intersection(ss.index)
        ax.plot(1/tt[j], ss[j], '.', ms=5, alpha=0.7, color=plt.cm.tab20(i%20), label=f"{nm} b{m['config'].get('batch_size')}"+(' K' if nm in CANCELLED else ''))
    xs = np.linspace(0.02,0.16,50)
    for B in (4.6, 6.5, 8.0, 10.6, 16.3, 5.5):
        ax.plot(xs, np.clip(100*B*xs,0,100), '-', color='0.8', lw=0.6)
    ax.axhline(THR, color='k', ls='--', lw=0.8); ax.set_xlabel('1 / step time  (1/s), 30-min bins'); ax.set_ylabel('system.gpu.0.gpu 30-min mean (%)'); ax.set_ylim(30,95); ax.set_title(title+'  -- grey lines: util = B / t for fixed GPU-busy s/step B'); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('figs/B_util_vs_inverse_steptime.png', dpi=90); plt.close(fig)

# FIGURE 3: onset alignment -- for each run with a slow regime, the step-time components around the first sustained onset
