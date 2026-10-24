"""The kernel scales of the landscape figures (generate_figs.SCALE_RADII): per system, the RDF distance at which a
logistic fit gives P(at least N of 20 molecules matched by COMPACK) = 0.95, for N = 20, 15 and 10, with a 200-resample
bootstrap interval, from the pairs and COMPACK results p23_compack.py wrote; and, per scale, the 100k bundles' ball
counts and Gaussian sums at the experimental forms (diagnostics: kernel d_cut = 3 r / 2.6 here, the reach used when this
was run; the figures use reach 3, so d_cut = r). usage: python compack_scales.py <dir of p23_compack.py outputs>"""
import sys
import os, json, numpy as np, torch
from scipy.optimize import minimize
from energy_sampling.eval.paper1_results.dens_engine import kernel_sums_from_hist, SETS
rng=np.random.default_rng(0); RES=r'D:\crystal_datasets\gfn_results\dens'; IN=sys.argv[1]
def fit(d,y):
    def nll(p):
        z=p[1]*(d-p[0]); return np.sum(np.logaddexp(0,z)*y+np.logaddexp(0,-z)*(1-y))
    best=None
    for d0 in np.linspace(0.05,d.max(),12):
        r=minimize(nll,[d0,20.0],method='Nelder-Mead',options={'xatol':1e-6,'fatol':1e-8,'maxiter':4000}); best=r if best is None or r.fun<best.fun else best
    return best.x
def at(d,y,p):
    d50,s=fit(d,y); v=d50+np.log(1/p-1)/s; b=[]
    for _ in range(200):
        i=rng.integers(0,len(d),len(d))
        if y[i].min()!=y[i].max():
            a=fit(d[i],y[i]); b.append(a[0]+np.log(1/p-1)/a[1])
    return v, np.quantile(b,.025), np.quantile(b,.975)
def pct(draws,pts):
    s=np.sort(draws); return 100*0.5*(np.searchsorted(s,pts,'left')+np.searchsorted(s,pts,'right'))/len(s)
def ball(out, r):
    rec=json.load(open(os.path.join(out,'sigma.json'))); h=np.load(os.path.join(out,'hist.npy'),mmap_mode='r'); ci=SETS.index('window_reduced')
    nb=int(round(r/rec['hist_max']*rec['hist_bins']))
    return np.concatenate([np.asarray(h[r0:r0+8192,ci,:nb],dtype=np.int64).sum(1) for r0 in range(0,h.shape[0],8192)])
for run,lj,mol in (('mipu','mip','MIPCAS'),('nehu','neh','NEHZOR')):
    P=torch.load(os.path.join(IN, f'p23_{run}_pairs.pt'),weights_only=False)['pairs']; c20=torch.load(os.path.join(IN, f'p23_{run}_compack.pt'),weights_only=False)
    ok=[k for k in range(len(P)) if c20[k]['status']=='ok']
    d=np.array([P[k]['d'] for k in ok]); n20=np.array([c20[k]['n_matched'] for k in ok]); rm=np.array([c20[k]['rmsd'] for k in ok])
    print(f"\n== {mol}")
    for thr in (20,15,10):
        v,lo,hi=at(d,(n20>=thr).astype(float),0.95)
        near=(np.abs(d-v)<0.03)&(n20>=thr)
        print(f"   >= {thr} of 20 at 95%: r = {v:.3f} [{lo:.3f}, {hi:.3f}]; median RMSD of matching pairs near r: {np.median(rm[near]):.2f} A (n={near.sum()})")
        for tag,rn in (('UMA',run),('LJ',lj)):
            out=os.path.join(RES,f'p23_{rn}_ft_running_10k_100000_k_red0')
            dr=torch.load(os.path.join(out,'draws.pt'),weights_only=False); refs=torch.load(os.path.join(out,'refs.pt'),weights_only=False); N=len(dr['energy']); m=dr['window_reduced'].numpy()
            c=ball(out,v); cd,cr=c[:N][m],c[N:]
            g=kernel_sums_from_hist(out,3*v/2.6); gd,gr=g[:N][m],g[N:]
            print(f"      {tag}: ball -> "+', '.join(f"{n} {int(x)} draws pct {p:.0f}" for n,x,p in zip(refs['labels'],cr,pct(cd,cr)))+f"; draws with 0 neighbours {100*np.mean(cd==0):.0f}%, median count {int(np.median(cd))} | gaussian -> "+', '.join(f"pct {p:.0f} (sum {x:.1f})" for x,p in zip(gr,pct(gd,gr)))+f"; draws<LOD {100*np.mean(gd<1):.0f}%; kernel d_cut {3*v/2.6:.3f}")