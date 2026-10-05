"""pool_oct02: the pooled prior rebuild of the four zp1 systems on the cluster -- premerge, export, flood, assemble.

    python configs/pool_oct02/make.py      (locally: reads the zp1_sep28 stream configs and the pooled coord.yaml files)

Source: for each system, D:/crystal_datasets/pooled_oct02/<name>/ holds the zp1_sep28 campaign's shards plus one stream
per old prior / search file, every old row re-scored on the campaign molecule (session scripts make_pool.py). The
pooled registry's basins are the anchors. Their lowest states are unconverged search end states: a further relaxation
drains them into far fewer minima (2026-10-03, MIPCAS UMA: 7,624 -> 394), but every such minimum within 5 kT already
has an anchor within the identity cut at the same energy, so the anchors hold the floors and a spread up the walls,
which is what a coverage prior wants. Stages, each an sbatch array here, chained by launch.sh
(prep = premerge -> export; flood = flood -> assemble):

  premerge  (systems whose pooled merge was not done on the dev box: PREMERGE) one coordinator curate pass over the
            uploaded pool (<name>/pool: RDF leader clustering of every row within 10 kT).
  export    `pool_anchors export` of the pooled registry: the lowest state of every basin within WINDOW kT, in the
            chart the trainer reads, as <name>/anchors.pt (prior layout). For a system merged on the dev box the file
            is uploaded and the task only checks that it is there.
  flood     data_processing/capped_mc.py from anchors.pt, FLOOD settings below, into <name>/flood/shard_<k>.
  assemble  data_processing/pool_assemble.py: anchors + flood states under CEILING, thinned in the trainer latent at
            the system's radius (RADIUS_MULT x its KICK), out-of-box and density-penalised states left out, normaliser
            images, written as <name>/<name>_pooled_oct05_prior.pt ('prior' and 'equalized_prior' hold the same rows)
            with a .summary.json, both copied into conditional/priors/. `launch.sh assemble` runs this stage alone.
  polish, merge   not in the chain; kept for the converged-minima census (`launch.sh polish`): run_search in data mode
            over starts.pt (Rprop, lr 0.001 annealed, up to POLISH_STEPS steps, convergence_eps 1e-6), then a curate
            pass and an export of the polished basins (anchors_polished.pt).

Writes <name>/coord.yaml, <name>/polish.yaml, <name>/pool_coord.yaml (PREMERGE systems), INDEX.tsv (polish tasks),
FLOOD.tsv (flood tasks), SYSTEMS.tsv, and the --array lines of the six sbatch files.
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
# assemble (owner 2026-10-05): no row budget; thin at a physical radius per system, the same for anchors and walk
# states; leave out everything more than CEILING kT above the lowest anchor. KICK = the latent size of a hop-style kick
# (log_noise_latent_parameters, keep_start_representable) that raises the energy of a minimum by a median 1 kT, the
# median over 6 of the system's lowest minima > 0.25 apart in atomwise RDF, 32 kicks per size (measured 2026-10-05;
# the eLJ minima re-relaxed to convergence first, the UMA ones campaign end states). Face states (a centre coordinate
# on a wall of the latent box, 16-33% of the anchors, all from the old files) ship like any other row.
CEILING = 10.0
RADIUS_MULT = 1.0
KICK = {'mipcas_elj': 0.0244, 'mipcas_uma': 0.0232, 'nehzor_elj': 0.0262, 'nehzor_uma': 0.0252}
# capped_mc (owner 2026-10-03): the walkers' own distribution is the data. From every start, walk at the training
# temperature and keep every accepted move: no burn-in, no claim of equilibrium. Why T x1: in 12 dimensions a walker
# settles about 6 T above its floor and does not come back down (measured, MIPCAS eLJ: +5.6 kT at x1 from starts near
# the floor, reached in about 150 steps and held; at x3 96% of the covered cells sit in the top 4 kT under the ceiling),
# so a hot walk only inflates into the ceiling. Starts: the anchors within WINDOW kT of the lowest (thinned at the
# identity cut), one walker each, 300 steps. The ceiling is a safety rail (15 kT above the lowest start); proposals
# that leave the reduced-cell domain are rejected (--red_max 0); no packing-coefficient filter.
WINDOW = 5.0
FLOOD = ('--steps 300 --adapt_end 90 --cov_updates 30,60 --tempers 1.0 --cap_local_kT 1e6 --cap_global_kT 15 '
         f'--seed_window_kT {WINDOW:g} --replicas 1 --pc_max 100 --red_max 0 --rdf_mode atomwise')
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
sy = ['\t'.join(['task', 'name', 'sg', 'key', 'tsf', 'mol', 'premerge', 'window', 'radius', 'ceiling'])]
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
    sy.append('\t'.join(str(v) for v in (k, name, sg, key, tsf, mol, int(name in PREMERGE), WINDOW,
                                         round(RADIUS_MULT * KICK[name], 6), CEILING)))
    print(f'{name}: {n_sub} polish tasks, {n_fl} flood shards' + (', premerge on the cluster' if name in PREMERGE else ''))
write('INDEX.tsv', '\n'.join(pol) + '\n')
write('FLOOD.tsv', '\n'.join(fl) + '\n')
write('SYSTEMS.tsv', '\n'.join(sy) + '\n')
for fn, last in (('submit_polish.sbatch', t - 1), ('submit_flood.sbatch', f - 1), ('submit_premerge.sbatch', len(SYSTEMS) - 1),
                 ('submit_merge.sbatch', len(SYSTEMS) - 1), ('submit_export.sbatch', len(SYSTEMS) - 1),
                 ('submit_assemble.sbatch', len(SYSTEMS) - 1)):
    p = os.path.join(HERE, fn)
    text = re.sub(r'#SBATCH --array=\S+', f'#SBATCH --array=0-{last}', open(p, encoding='utf8').read())
    assert_no_local_paths(text, fn)
    open(p, 'w', encoding='utf8', newline='\n').write(text)
print(f'{t} polish tasks, {f} flood tasks, {len(SYSTEMS)} systems; ceiling {CEILING:g} kT, radius {RADIUS_MULT:g} x kick: '
      + ', '.join(f'{n} {RADIUS_MULT * KICK[n]:g}' for n in KICK))
