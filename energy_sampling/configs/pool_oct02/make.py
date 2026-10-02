"""pool_oct02: polish and re-merge the pooled anchors of the four zp1 systems on the cluster.

    python configs/pool_oct02/make.py      (locally: reads each system's starts.pt for its row count)

The pooled anchors are the lowest state of every basin within 10 kT, from the zp1_sep28 campaigns plus every old prior
and search file, written by `python -m data_processing.pool_anchors starts` in the trainer's chart (handedness +1;
NEHZOR -1 rows embedded with the +1 molecule). Measured 2026-10-02: the campaigns' end states are not converged (400
more Rprop steps lower 60 MIPCAS eLJ anchors a median 0.66 kT and move 49 of them past the identity cut; a further 400
steps move none). So every anchor is relaxed again here:

  polish  run_search, init_sample_method 'data', one stage (Rprop, lr 0.001 annealed, no compression, wrap, up to
          POLISH_STEPS steps, convergence_eps 1e-6), writing shards to a campaign directory per system (stream
          'polish'); array tasks split the starts by mol_seed.
  merge   one coordinator curate pass over that directory (RDF leader clustering at the system's identity cut), then
          `pool_anchors export`: the lowest state of every basin within 12 kT, in the trainer's chart, as a prior file
          (anchors_polished.pt) -- the seeds of the capped-MC flood that follows.

Writes, per system, <name>/coord.yaml and <name>/polish.yaml; INDEX.tsv (one row per polish task); and the --array
lines of submit_polish.sbatch and submit_merge.sbatch. launch.sh submits both (merge after every polish task ends).
"""
import math
import os
import re

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = 'D:/crystal_datasets/pooled_oct02'
DATA = '/scratch/mk8347/data/crystal_datasets'
UMA = '/scratch/mk8347/models/uma/esen_s.pt'
POLISH_STEPS = 500
# name, space group, energy key, molecule on the cluster, thermal_scaling_factor (eLJ: the current prior files'),
# polish tasks, polish batch size
SYSTEMS = [
    ('mipcas_elj', 2, 'elj', f'{DATA}/mipcas/MIPCAS_standardized.pt', 0.3635836825309169, 2, 2000),
    ('nehzor_elj', 14, 'elj', f'{DATA}/nehzor/NEHZOR0_std_conf.pt', 0.15557874739170074, 2, 2000),
    ('mipcas_uma', 2, 'uma', f'{DATA}/mipcas/MIPCAS_standardized.pt', 1.0, 4, 500),
    ('nehzor_uma', 14, 'uma', f'{DATA}/nehzor/NEHZOR0_std_conf.pt', 1.0, 8, 500),
]
DRIVE = re.compile(r'(^|[\s\'"=])[A-Za-z]:[\\/]')


def assert_no_local_paths(text, where):
    bad = [ln for ln in text.splitlines() if DRIVE.search(ln)]
    if bad:
        raise SystemExit(f'{where}: local path(s) would ship to the cluster:\n  ' + '\n  '.join(bad))


def dump(obj, path):
    text = yaml.safe_dump(obj, sort_keys=False)
    assert_no_local_paths(text, path)
    open(path, 'w', newline='\n').write(text)


ONLY = [x for x in os.environ.get('ONLY', '').split(',') if x]  # e.g. ONLY=mipcas_elj,mipcas_uma
SYSTEMS = [s for s in SYSTEMS if not ONLY or s[0] in ONLY]
rows = ['\t'.join(['task', 'name', 'sub', 'n_sub', 'num_samples', 'n_starts', 'sg', 'key', 'tsf', 'mol'])]
t = 0
for name, sg, key, mol, tsf, n_sub, bs in SYSTEMS:
    n = torch.load(f'{LOCAL}/{name}/starts.pt', weights_only=False).num_graphs
    camp = f'{DATA}/pooled_oct02/{name}/polish'
    d = os.path.join(HERE, name)
    os.makedirs(d, exist_ok=True)
    coord = yaml.safe_load(open(f'{LOCAL}/{name}/coord.yaml'))
    coord.update(mol_path=mol, window_kT=12.0, streams={}, hops=None, priors=[], energy_ref=None,
                 identity_note=coord.get('identity_note', '') + ' | pool_oct02 polish of the pooled anchors')
    dump(coord, os.path.join(d, 'coord.yaml'))
    stream = yaml.safe_load(open(f'D:/crystal_datasets/{name}_sep28/streams/random.yaml'))
    for k in ('coord_hop_wait_s', 'init_reduced', 'init_target_cp', 'sgs_to_search', 'zp_to_search'):
        stream.pop(k, None)
    stream.update(mol_path=mol, out_dir=f'{camp}/runs', run_name='polish_TASK', init_sample_method='data',
                  dataset_path=f'{DATA}/pooled_oct02/{name}/starts.pt', num_samples=math.ceil(n / n_sub), mol_seed=0,
                  batch_size=bs, grow_batch_size=True, coord_dir=camp, coord_stream='polish',
                  coord_curate_every_s=None, opt_seed=7_000_000)
    stream['opt'] = [dict(optim_target=key, enforce_reduced=False, compression_factor=0.0, cutoff=10, init_lr=0.001,
                          convergence_eps=1e-6, optimizer_func='rprop', anneal_lr=True, grad_norm_clip=0.1,
                          show_tqdm=False, max_num_steps=POLISH_STEPS, target_packing_coeff=None,
                          centroid_boundary='wrap')]
    dump(stream, os.path.join(d, 'polish.yaml'))
    for k in range(n_sub):
        rows.append('\t'.join(str(v) for v in (t, name, k, n_sub, math.ceil(n / n_sub), n, sg, key, tsf, mol)))
        t += 1
    print(f'{name}: {n} starts, {n_sub} polish tasks of {math.ceil(n / n_sub)}')
index = '\n'.join(rows) + '\n'
assert_no_local_paths(index, 'INDEX.tsv')
open(os.path.join(HERE, 'INDEX.tsv'), 'w', newline='\n').write(index)
for fn, last in (('submit_polish.sbatch', t - 1), ('submit_merge.sbatch', len(SYSTEMS) - 1)):
    p = os.path.join(HERE, fn)
    text = re.sub(r'#SBATCH --array=\S+', f'#SBATCH --array=0-{last}', open(p, encoding='utf8').read())
    assert_no_local_paths(text, fn)
    open(p, 'w', encoding='utf8', newline='\n').write(text)
print(f'{t} polish tasks, {len(SYSTEMS)} merge tasks')
