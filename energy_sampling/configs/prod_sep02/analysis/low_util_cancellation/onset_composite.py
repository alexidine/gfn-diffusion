"""figs/C_onset_alignment.png: step-time components, device util and host memory +-4 h around the
first sustained step-time rise on each run that had one. Onsets are the regime boundaries from
regimes.csv, hand-checked against the per-run figures (the automatic pick lands on startup wobble
for mipu_lr0p125)."""
import pickle, glob, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
onsets = {'neh_lr0p5':(19.0,8.04,10.05),'mipu_lr0p0625':(32.6,9.73,12.46),'mipu_lr0p125':(27.8,9.69,11.74),'mipu_lr0p03125':(19.0,9.67,14.97),
          'mip_lr1':(21.8,7.99,8.99),'nehu_lr0p125':(18.1,14.25,20.99),'nehu_lr0p0625':(13.5,14.28,17.9),'acr_lr0p1':(10.9,10.45,13.0)}
CANC = {'neh_lr0p5','mipu_lr0p0625','mipu_lr0p125','mipu_lr0p03125'}
files = {pickle.load(open(f,'rb'))['meta']['name'].replace('p02_p02_',''): f for f in glob.glob('data/*.pkl')}
order = list(onsets)
fig, axes = plt.subplots(len(order), 3, figsize=(17, 2.6*len(order)))
for r,nm in enumerate(order):
    d = pickle.load(open(files[nm],'rb')); h=d['hist'].copy(); ev=d['events'].copy()
    h['hr']=(h['_timestamp']-h['_timestamp'].min())/3600; ev['hr']=ev['_runtime']/3600
    t_on, t_before, t_after = onsets[nm]
    w = (h['hr']>t_on-4)&(h['hr']<t_on+4); e = (ev['hr']>t_on-4)&(ev['hr']<t_on+4)
    ax=axes[r,0]; hh=h[w]
    ax.plot(hh['hr']-t_on, hh['train_step_time'], '.', ms=3, color='C0', label='step time')
    ax.plot(hh['hr']-t_on, hh['energy/seconds_in_step']/10, '.', ms=3, color='C3', label='energy (MLIP/ELJ) s')
    ax.plot(hh['hr']-t_on, hh['train_step_time']-hh['energy/seconds_in_step']/10, '.', ms=3, color='C2', label='non-energy s')
    for et in h.loc[h['eval_step_time'].notna(),'hr']:
        if abs(et-t_on)<4: ax.axvline(et-t_on, color='k', alpha=0.2)
    ax.axvline(0, color='r', lw=0.8); ax.set_ylim(0, t_after*1.5); ax.set_ylabel(f"{nm}\n{'KILLED' if nm in CANC else 'survived'}\ns/step", fontsize=8)
    if r==0: ax.legend(fontsize=7, ncol=3); ax.set_title('step-time components around onset (grey = eval)', fontsize=9)
    ax=axes[r,1]; ee=ev[e]
    ax.plot(ee['hr']-t_on, ee['system.gpu.0.gpu'], lw=0.3, color='0.6')
    ser = pd.Series(ee['system.gpu.0.gpu'].values, index=pd.to_timedelta(ee['hr'].values, unit='h'))
    ax.plot(ee['hr']-t_on, ser.rolling('1800s').mean().values, color='C1', lw=1.5, label='30-min mean')
    ax.plot(ee['hr']-t_on, ee['system.gpu.0.smClock']/1410*100, color='C4', lw=0.8, label='SM clock (% of 1410)')
    ax.axvline(0, color='r', lw=0.8); ax.axhline(54, color='k', ls='--', lw=0.7); ax.set_ylim(0,105)
    if r==0: ax.legend(fontsize=7); ax.set_title('device util (wandb system stream) + SM clock', fontsize=9)
    ax=axes[r,2]
    ax.plot(ee['hr']-t_on, ee['system.memory_percent'], color='C3', lw=1, label='host memory % (node-wide)')
    ax2=ax.twinx(); ax2.plot(ee['hr']-t_on, ee['system.proc.memory.rssMB']/1024, color='C0', lw=0.8, label='our RSS GB'); ax2.set_ylim(2,7)
    ax.axvline(0, color='r', lw=0.8); ax.set_ylim(0,40)
    if r==0: ax.legend(fontsize=7, loc='upper left'); ax2.legend(fontsize=7, loc='upper right'); ax.set_title('host-wide memory vs our process RSS', fontsize=9)
for ax in axes[-1]: ax.set_xlabel('hours from onset')
fig.tight_layout(); fig.savefig('figs/C_onset_alignment.png', dpi=85); plt.close(fig)
