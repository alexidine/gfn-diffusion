import pickle, glob, pandas as pd, numpy as np
CANCELLED = {'acr_lr0p025','mipu_lr0p015625','mipu_lr0p03125','mipu_lr0p0625','mipu_lr0p125','mipu_lr0p25','neh_lr0p5'}
pd.set_option('display.width',260); pd.set_option('display.max_columns',40)
def ev_at(ev, col, t0, t1, fn=np.nanmean):
    s = ev[(ev['_runtime']>=t0*3600)&(ev['_runtime']<t1*3600)][col].dropna()
    return fn(s.values) if len(s) else np.nan
def net_rate(ev, col, t0, t1):
    s = ev[(ev['_runtime']>=t0*3600)&(ev['_runtime']<t1*3600)][['_runtime',col]].dropna()
    if len(s)<2: return np.nan
    return (s[col].iloc[-1]-s[col].iloc[0])/max(1e-9,(s['_runtime'].iloc[-1]-s['_runtime'].iloc[0]))/1e6  # MB/s if bytes
allrows=[]
for fn in sorted(glob.glob('data/*.pkl')):
    d = pickle.load(open(fn,'rb')); m=d['meta']; h=d['hist']; ev=d['events']
    if h is None or len(h)==0: continue
    nm = m['name'].replace('p02_p02_','')
    h=h.copy(); h['hr']=(h['_timestamp']-h['_timestamp'].min())/3600
    st = h[['hr','_step','train_step_time','energy/seconds_in_step','Batch Size']].dropna(subset=['train_step_time']).reset_index(drop=True)
    # skip the first hour (startup, warmup, compile)
    st = st[st['hr']>1.0].reset_index(drop=True)
    if len(st)<20: 
        print(nm,'too short'); continue
    med = st['train_step_time'].rolling(9, center=True, min_periods=5).median()
    # regime segmentation: greedy -- new regime when rolling median deviates >10% from current regime's median for 5 consecutive points
    regimes=[]; start=0; cur=med.iloc[4] if len(med)>4 else med.iloc[0]
    i=5; run=0
    seg_vals=[]
    while i < len(st):
        v = med.iloc[i]
        if np.isfinite(v) and abs(v-cur)/cur > 0.10:
            run+=1
            if run>=5:
                regimes.append((start, i-5)); start=i-5; cur=med.iloc[i]; run=0
        else:
            run=0
            # slowly track the current regime median
            cur = np.nanmedian(st['train_step_time'].iloc[start:i+1])
        i+=1
    regimes.append((start, len(st)-1))
    rows=[]
    for (a,b) in regimes:
        seg = st.iloc[a:b+1]
        if len(seg)<3: continue
        t0,t1 = seg['hr'].iloc[0], seg['hr'].iloc[-1]
        stt = seg['train_step_time'].median(); en = seg['energy/seconds_in_step'].median()/10
        util = ev_at(ev,'system.gpu.0.gpu',t0,t1)
        rows.append(dict(run=nm, canc=nm in CANCELLED, t0=round(t0,1), t1=round(t1,1), dur_h=round(t1-t0,1), steps=f"{int(seg['_step'].iloc[0])}-{int(seg['_step'].iloc[-1])}",
            batch=int(seg['Batch Size'].median()), step_s=round(stt,2), energy_s=round(en,2), nonenergy_s=round(stt-en,2),
            gpu_util=round(util,1), util_x_t=round(util*stt/100,2),
            smclk=round(ev_at(ev,'system.gpu.0.smClock',t0,t1),0), pwrW=round(ev_at(ev,'system.gpu.0.powerWatts',t0,t1),0), tempC=round(ev_at(ev,'system.gpu.0.temp',t0,t1),0),
            host_mem_pct=round(ev_at(ev,'system.memory_percent',t0,t1),1), rss_gb=round(ev_at(ev,'system.proc.memory.rssMB',t0,t1)/1024,2),
            net_sent_MBs=round(net_rate(ev,'system.network.sent',t0,t1),2), net_recv_MBs=round(net_rate(ev,'system.network.recv',t0,t1),2)))
    df=pd.DataFrame(rows); allrows.append(df)
    print(df.to_string(index=False)); print()
big=pd.concat(allrows); big.to_csv('regimes.csv', index=False)
