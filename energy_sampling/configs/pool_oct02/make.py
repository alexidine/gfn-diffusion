"""pool_oct02: the pooled prior rebuild of the four zp1 systems on the cluster -- premerge, polish, merge, flood,
assemble.

    python configs/pool_oct02/make.py      (locally: reads the zp1_sep28 stream configs and the pooled coord.yaml files)

Source: for each system, D:/crystal_datasets/pooled_oct02/<name>/ holds the zp1_sep28 campaign's shards plus one stream
per old prior / search file, every old row re-scored on the campaign molecule (session scripts make_pool.py). Stages,
each an sbatch array here, chained by launch.sh:

  premerge  (systems whose pooled merge was not done locally: PREMERGE) one coordinator curate pass over the uploaded
            pool (<name>/pool: RDF leader clustering of every row within 10 kT), then `pool_anchors starts`: the lowest
            state of every basin, in the trainer's chart, as <name>/starts.pt.
  polish    run_search, init_sample_method 'data', one stage (Rprop, lr 0.001 annealed, no compression, wrap, up to
            POLISH_STEPS steps, convergence_eps 1e-6) into <name>/polish (a campaign directory, stream 'polish'); the
            array tasks split starts.pt by mol_seed, the block size computed from the file at run time. Why: the zp1
            end states are not converged (2026-10-02: 400 more Rprop steps lower 60 MIPCAS eLJ anchors a median 0.66 kT
            and move 49 of them past the identity cut; a further 400 move none).
  merge     a curate pass over <name>/polish, then `pool_anchors export`: basins within 12 kT, trainer's chart, as
            <name>/anchors_polished.pt (prior layout).
  flood     data_processing/capped_mc.py from anchors_polished.pt, FLOOD settings below, into <name>/flood/shard_<k>.
  assemble  data_processing/pool_assemble.py: anchors + flood states, latent de-dupe at DEDUPE, normaliser images,
            written as <name>/<name>_pooled_oct02_prior.pt ('prior' and 'equalized_prior' hold the same rows).

Writes <name>/coord.yaml, <name>/polish.yaml, <name>/pool_coord.yaml (PREMERGE systems), INDEX.tsv (polish tasks),
FLOOD.tsv (flood tasks), SYSTEMS.tsv, and the --array lines of the five sbatch files.
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
DEDUPE = 0.01
# capped_mc: one global level 10 kT above the lowest seed, no per-seed cap, T = 1 x training, 400 steps, one walker per
# distinct minimum (seeds thinned at the identity cut), no packing-coefficient filter, and proposals that leave the
# reduced-cell domain rejected (--red_max 0)
FLOOD = ('--steps 400 --adapt_end 120 --cov_updates 40,80 --tempers 1.0 --cap_local_kT 1e6 --cap_global_kT 10 '
         '--seed_window_kT 10 --replicas 1 --pc_max 100 --red_max 0 --rdf_mode atomwise')
# name, space group, energy key, molecule on the cluster, thermal_scaling_factor (eLJ: the current prior files'),
# polish tasks, polish batch size, flood shards
SYSTEMS = [
    ('mipcas_elj', 2, 'elj', f'{DATA}/mipcas/MIPCAS_standardized.pt', 0.3635836825309169, 2, 2000, 1),
    ('mipcas_uma', 2, 'uma', f'{DATA}/mipcas/MIPCAS_standardized.pt', 1.0, 4, 500, 4),
    ('nehzor_elj', 14, 'elj', f'{DATA}/nehzor/NEHZOR0_std_conf.pt', 0.15557874739170074, 4, 2000, 2),
    ('nehzor_uma', 14, 'uma', f'{DATA}/nehzor/NEHZOR0_std_conf.pt', 1.0, 16, 500, 8),
]
PREMERGE = ['nehzor_elj', 'nehzor_uma']
DRIVE = re.compile(r'(^|[\s\'"=])[A-Za-z]:[\\/]')


def assert_no_local_paths(text, where):
    bad = [ln for ln in text.splitlines() if DRIVE.search(ln)]
    if bad:
        raise SystemExit(f'{where}: local path(s) would ship to the cluster:\n  ' + '\n  '.join(bad))


def dump(obj, path):
    text = yaml.safe_dump(obj, sort_keys=False)
    assert_no_local_paths(text, path)
    open(path, 'w', newline='\n').write(text)


def write(name, text):
    assert_no_local_paths(text, name)
    open(os.path.join(HERE, name), 'w', newline='\n').write(text)


# the ten columns are the layout the first launch's job scripts read (its merge is still queued): do not reorder
pol = ['\t'.join(['task', 'name', 'sub', 'n_sub', 'num_samples', 'n_starts', 'sg', 'key', 'tsf', 'mol'])]
fl = ['\t'.join(['task', 'name', 'shard', 'n_shards', 'key', 'cut', 'args'])]
sy = ['\t'.join(['task', 'name', 'sg', 'key', 'tsf', 'mol', 'premerge'])]
t = f = 0
for k, (name, sg, key, mol, tsf, n_sub, bs, n_fl) in enumerate(SYSTEMS):
    d = os.path.join(HERE, name)
    os.makedirs(d, exist_ok=True)
    camp = f'{DATA}/pooled_oct02/{name}/polish'
    pooled = yaml.safe_load(open(f'{LOCAL}/{name}/coord.yaml'))
    pooled.update(mol_path=mol)
    if name in PREMERGE:
        dump(pooled, os.path.join(d, 'pool_coord.yaml'))
    coord = dict(pooled)
    coord.update(window_kT=12.0, streams={}, hops=None, priors=[], energy_ref=None,
                 identity_note=str(pooled.get('identity_note', '')) + ' | pool_oct02 polish of the pooled anchors')
    dump(coord, os.path.join(d, 'coord.yaml'))
    stream = yaml.safe_load(open(f'D:/crystal_datasets/{name}_sep28/streams/random.yaml'))
    for kk in ('coord_hop_wait_s', 'init_reduced', 'init_target_cp', 'sgs_to_search', 'zp_to_search'):
        stream.pop(kk, None)
    n_block = n_starts = 0  # set from starts.pt by the job
    if name not in PREMERGE:  # the first launch froze this value into the campaign directory
        n_starts = torch.load(f'{LOCAL}/{name}/starts.pt', weights_only=False).num_graphs
        n_block = math.ceil(n_starts / n_sub)
    stream.update(mol_path=mol, out_dir=f'{camp}/runs', run_name='polish_TASK', init_sample_method='data',
                  dataset_path=f'{DATA}/pooled_oct02/{name}/starts.pt', num_samples=n_block, mol_seed=0,
                  batch_size=bs, grow_batch_size=True, coord_dir=camp, coord_stream='polish',
                  coord_curate_every_s=None, opt_seed=7_000_000)
    stream['opt'] = [dict(optim_target=key, enforce_reduced=False, compression_factor=0.0, cutoff=10, init_lr=0.001,
                          convergence_eps=1e-6, optimizer_func='rprop', anneal_lr=True, grad_norm_clip=0.1,
                          show_tqdm=False, max_num_steps=POLISH_STEPS, target_packing_coeff=None,
                          centroid_boundary='wrap')]
    dump(stream, os.path.join(d, 'polish.yaml'))
    for j in range(n_sub):
        pol.append('\t'.join(str(v) for v in (t, name, j, n_sub, n_block, n_starts, sg, key, tsf, mol)))
        t += 1
    for j in range(n_fl):
        fl.append('\t'.join(str(v) for v in (f, name, j, n_fl, key, pooled['identity_cut'], FLOOD)))
        f += 1
    sy.append('\t'.join(str(v) for v in (k, name, sg, key, tsf, mol, int(name in PREMERGE))))
    print(f'{name}: {n_sub} polish tasks, {n_fl} flood shards' + (', premerge on the cluster' if name in PREMERGE else ''))
write('INDEX.tsv', '\n'.join(pol) + '\n')
write('FLOOD.tsv', '\n'.join(fl) + '\n')
write('SYSTEMS.tsv', '\n'.join(sy) + '\n')
for fn, last in (('submit_polish.sbatch', t - 1), ('submit_flood.sbatch', f - 1), ('submit_premerge.sbatch', len(SYSTEMS) - 1),
                 ('submit_merge.sbatch', len(SYSTEMS) - 1), ('submit_assemble.sbatch', len(SYSTEMS) - 1)):
    p = os.path.join(HERE, fn)
    text = re.sub(r'#SBATCH --array=\S+', f'#SBATCH --array=0-{last}', open(p, encoding='utf8').read())
    assert_no_local_paths(text, fn)
    open(p, 'w', encoding='utf8', newline='\n').write(text)
print(f'{t} polish tasks, {f} flood tasks, {len(SYSTEMS)} systems; dedupe {DEDUPE}')
