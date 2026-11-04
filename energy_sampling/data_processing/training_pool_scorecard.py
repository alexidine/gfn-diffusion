"""Scorecard of the proposed training pool against four desiderata (read-only).

Pool: pool_v1.csv.gz from training_pool_build.py (census_v3 union, isotopes stripped, stereo enumerated, Day removed by
InChIKey skeleton). Section 4 is repeated on the rigid tier alone (Day is all rigid; most of the pool has one torsion).

  1 HCNOF        every pool SMILES re-parsed: element set within {C,H,N,O,F}
  2 MW range     MW quantiles and bands, Day against the pool, with the source of each band
  3 coverage     (a) overlap between sources by ik14; (b) leave-CSD-out: CSD molecules absent from PubChem and ZINC,
                 nearest-neighbour Tanimoto to the PubChem+ZINC part; (c) GDB-11 enumerated rigid molecules
                 (census v1, strict tier) found in the pool, by heavy-atom count
  4 Day not OOD  nearest-neighbour Tanimoto (count Morgan r2, 2048) to a fixed random reference of the pool;
                 in-distribution threshold = 5th / 1st percentile over pool molecules outside the reference;
                 Day's share below it. Descriptor support: Day's share inside the pool's 0.5-99.5 percentile range

usage, from energy_sampling/: python -m data_processing.training_pool_scorecard
  -> HERE/scorecard_v1.json, FIG/fig17_pool_scorecard.png
"""
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*')
HERE = 'D:/crystal_datasets/paper_review_nov02/pool'
FIG = 'D:/crystal_datasets/paper_review_nov02/fig'
SEED = 0
N_REF, N_Q = 500_000, 2_000
rng = np.random.default_rng(SEED)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
out = {}


def fps(smiles):
    return [gen.GetCountFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def nn(q_smiles, q_ik, ref, ref_ik):
    """Largest Tanimoto of each query to the reference, ignoring reference molecules with the query's own skeleton
    (ik14: its stereoisomers and tautomers, which the fingerprint cannot tell apart). Day queries have none by
    construction, so pool queries must not either."""
    by_ik = {}
    for i, k in enumerate(ref_ik):
        by_ik.setdefault(k, []).append(i)
    out = []
    for f, k in zip(fps(q_smiles), q_ik):
        sim = np.array(DataStructs.BulkTanimotoSimilarity(f, ref))
        sim[by_ik.get(k, [])] = -1
        out.append(sim.max())
    return np.array(out)


# ---- pool
pool = pd.read_csv(f'{HERE}/pool_v1.csv.gz', keep_default_na=False)
pool['srcs'] = pool.sources
built = json.load(open(f'{HERE}/pool_v1_counts.json'))
out['build'] = {k: built[k] for k in ('1_union_keys', '3_keys_after_isotope_and_stereo', '4_day_excluded_keys')}
out['pool'] = built['final'] | {'stereo enumerated (molecules)': int(pool.stereo_enumerated.sum()),
                                'split': built['6_split']}

# Day molecules (perceived SMILES; lovwoi has none)
day = pd.read_csv('D:/crystal_datasets/paper_review_nov02/data/day_molecules.csv.gz')
day = day[day.key.notna()].drop_duplicates('key')

# ---- 1 HCNOF
ok = 0
for s in pool.smiles:
    m = Chem.MolFromSmiles(s)
    ok += int(m is not None and {a.GetSymbol() for a in Chem.AddHs(m).GetAtoms()} <= set('CHNOF'))
out['1_hcnof'] = {'pool molecules C/H/N/O/F only': ok, 'of': int(len(pool))}

# ---- 2 MW
bands = [0, 100, 150, 200, 230]
def band_share(x):
    h, _ = np.histogram(x, bands)
    return {f'{a}-{b}': round(100 * v / len(x), 1) for a, b, v in zip(bands[:-1], bands[1:], h)}
hi = pool[pool.mw >= 200]
out['2_mw'] = {'Day': {'min': float(day.mw.min()), 'p05': float(day.mw.quantile(.05)), 'median': float(day.mw.median()),
                       'p95': float(day.mw.quantile(.95)), 'max': float(day.mw.max()), 'bands_pct': band_share(day.mw)},
               'pool': {'min': float(pool.mw.min()), 'p05': float(pool.mw.quantile(.05)),
                        'median': float(pool.mw.median()), 'p95': float(pool.mw.quantile(.95)),
                        'max': float(pool.mw.max()), 'bands_pct': band_share(pool.mw)},
               'band 200-230 by source (molecules found in)': {s: int(hi.srcs.str.contains(s).sum())
                                                               for s in ('csd', 'zinc20A', 'zinc20B', 'pubchem')}}

# ---- 3 coverage
ik = {s: set(pool.loc[pool.srcs.str.contains(s), 'ik14']) for s in ('csd', 'zinc20A', 'pubchem')}
pz = set().union(*(ik[s] for s in ik if s != 'csd'))
out['3a_overlap'] = {'CSD molecules (skeletons)': len(ik['csd']),
                     'CSD skeletons also in PubChem or ZINC (%)': round(100 * len(ik['csd'] & pz) / len(ik['csd']), 1),
                     'ZINC-A skeletons also in PubChem (%)': round(100 * len(ik['zinc20A'] & ik['pubchem'])
                                                                   / len(ik['zinc20A']), 1)}
g = pd.read_csv(f'{HERE}/gdb11_daylike.csv.gz', keep_default_na=False, usecols=['ik14', 'n_heavy'])     if os.path.exists(f'{HERE}/gdb11_daylike.csv.gz') else None  # census v1: GDB-11, strict tier only
if g is not None:
    strict_ik = set(pool.loc[(pool.n_rot == 0) & (pool.rotor == 0), 'ik14'])
    g['in_pool'] = g.ik14.isin(strict_ik)
    out['3c_gdb11_strict_in_pool_pct_by_heavy'] = {int(k): round(100 * v, 2) for k, v in
                                                   g.groupby('n_heavy').in_pool.mean().items()}
    out['3c_gdb11_strict_in_pool_pct'] = round(100 * g.in_pool.mean(), 2)

# fixed reference for every nearest-neighbour measurement: N_REF random pool molecules from PubChem/ZINC
nonc = np.where(pool.srcs != 'csd')[0]
ref_idx = rng.choice(nonc, min(N_REF, len(nonc)), replace=False)
in_ref = np.zeros(len(pool), bool)
in_ref[ref_idx] = True
ref = fps(pool.smiles.values[ref_idx])
ref_ik = pool.ik14.values[ref_idx]
q_idx = rng.choice(np.where(~in_ref & (pool.srcs != 'csd'))[0], N_Q, replace=False)
nn_q = nn(pool.smiles.values[q_idx], pool.ik14.values[q_idx], ref, ref_ik)
csd_only = pool[(pool.srcs == 'csd')]
csd_q = csd_only.sample(min(N_Q, len(csd_only)), random_state=SEED)
nn_csd = nn(csd_q.smiles.values, csd_q.ik14.values, ref, ref_ik)
nn_day = nn(day.key.values, [''] * len(day), ref, ref_ik)
t5, t1 = np.percentile(nn_q, 5), np.percentile(nn_q, 1)
out['3b_leave_csd_out'] = {'CSD-only molecules': int(len(csd_only)), 'queried': int(len(csd_q)),
                           'NN median': float(np.median(nn_csd)),
                           'share below pool 5th percentile (%)': round(100 * float((nn_csd < t5).mean()), 1)}
# ---- 4 Day not OOD
out['4_day_ood'] = {'reference molecules': int(len(ref)), 'pool queries outside reference': int(N_Q),
                    'pool NN median': float(np.median(nn_q)), 'Day NN median': float(np.median(nn_day)),
                    'threshold p5': float(t5), 'threshold p1': float(t1),
                    'Day below p5 (%), expected 5 if in-distribution': round(100 * float((nn_day < t5).mean()), 1),
                    'Day below p1 (%), expected 1': round(100 * float((nn_day < t1).mean()), 1),
                    'Day refcodes below p1': day.loc[nn_day < t1, 'refcode'].tolist()}
sup = {}
for c in ('n_heavy', 'mw', 'n_rings'):
    lo, hi_ = np.percentile(pool[c], [0.5, 99.5])
    sup[c] = round(100 * float(((day[c] >= lo) & (day[c] <= hi_)).mean()), 1)
out['4_day_inside_pool_0.5_99.5_pct_range'] = sup


# ---- 4, rigid view: Day is all rigid while most of the pool has one torsion, so repeat the test with reference and
# pool queries drawn from the rigid tier only (same reference size, same skeleton exclusion)
rig = np.where((pool.srcs != 'csd').to_numpy() & (pool.n_rot == 0).to_numpy())[0]
r_idx = rng.choice(rig, min(N_REF, len(rig) - N_Q), replace=False)
r_in = np.zeros(len(pool), bool)
r_in[r_idx] = True
rref = fps(pool.smiles.values[r_idx])
rref_ik = pool.ik14.values[r_idx]
rq = rng.choice(np.setdiff1d(rig, r_idx), N_Q, replace=False)
nn_rq = nn(pool.smiles.values[rq], pool.ik14.values[rq], rref, rref_ik)
nn_rday = nn(day.key.values, [''] * len(day), rref, rref_ik)
r5, r1 = np.percentile(nn_rq, 5), np.percentile(nn_rq, 1)
out['4_day_ood_rigid_view'] = {'reference molecules (rigid tier)': int(len(rref)), 'pool NN median': float(np.median(nn_rq)),
                               'Day NN median': float(np.median(nn_rday)), 'threshold p5': float(r5), 'threshold p1': float(r1),
                               'Day below p5 (%), expected 5': round(100 * float((nn_rday < r5).mean()), 1),
                               'Day below p1 (%), expected 1': round(100 * float((nn_rday < r1).mean()), 1),
                               'Day refcodes below p1': day.loc[nn_rday < r1, 'refcode'].tolist()}
json.dump(out, open(f'{HERE}/scorecard_v1.json', 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))


# ---- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 4, figsize=(19, 4.2))
b = np.arange(0, 235, 5)
axs[0].hist(pool.mw, b, density=True, histtype='step', color='#0f9fb0', lw=1.4, label=f'pool (N={len(pool):,})')
axs[0].hist(day.mw, b, density=True, histtype='step', color='#2a78d6', lw=2, label=f'Day-1000 (N={len(day):,})')
axs[0].set_xlabel('molecular weight (g/mol)')
axs[0].set_ylabel('probability density (per g/mol)')
axs[0].set_title('2. molecular weight', loc='left')
axs[0].legend(frameon=False, fontsize=7)
bb = np.linspace(0, 1, 51)
for ax, curves, thr, title, nref in (
        (axs[1], ((nn_q, 'pool molecules outside the reference', '#0f9fb0', 1.4), (nn_day, 'Day-1000', '#2a78d6', 2.0),
                  (nn_csd, 'CSD-only molecules (left out of the reference)', '#eb6834', 1.4)), t5,
         '3b, 4. fingerprint coverage: whole pool', len(ref)),
        (axs[2], ((nn_rq, 'rigid pool molecules outside the reference', '#0f9fb0', 1.4),
                  (nn_rday, 'Day-1000', '#2a78d6', 2.0)), r5, '4. fingerprint coverage: rigid tier only', len(rref))):
    for x, lab, c, lw in curves:
        ax.hist(x, bb, density=True, histtype='step', color=c, lw=lw, label=f'{lab} (median {np.median(x):.2f})')
    ax.axvline(thr, color='#52514e', ls='--', lw=0.8, label='pool 5th percentile')
    ax.set_xlabel(f'nearest-neighbour Tanimoto to {nref:,} reference molecules (count Morgan r2)', fontsize=8)
    ax.set_ylabel('density of queries')
    ax.set_title(title, loc='left')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(frameon=False, fontsize=6.5, loc='upper left')
if g is not None:
    k = sorted(out['3c_gdb11_strict_in_pool_pct_by_heavy'])
    axs[3].bar(k, [out['3c_gdb11_strict_in_pool_pct_by_heavy'][i] for i in k], color='#6250d6')
    axs[3].set_xlabel('heavy atoms (count)')
    axs[3].set_ylabel('% of GDB-11 strict molecules in the pool')
    axs[3].set_title('3c. GDB-11 rigid molecules found in the pool', loc='left')
fig.suptitle('Training pool v1 scorecard (rigid + one-torsion, C/H/N/O/F, MW < 230, Day removed). NN references: fixed random '
             'PubChem/ZINC pool molecules (whole pool, or rigid tier only); same-skeleton neighbours excluded.', fontsize=8.5, x=0.01, ha='left')
fig.tight_layout()
fig.savefig(f'{FIG}/fig17_pool_scorecard.png', dpi=150)

