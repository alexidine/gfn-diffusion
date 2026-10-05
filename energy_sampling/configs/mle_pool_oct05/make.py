"""mle_pool_oct05 -- phase 1 (bwd MLE) from FRESH weights on the pooled priors of 2026-10-05, for mip / mipu / neh / nehu.

    python configs/mle_pool_oct05/make.py

THE ARMS are mle_fresh_sep17's (its build_arm: mle_w3_sep16's arm shape on today's committed mk_dev -- rate scale 2.0,
burn-in, batch policy, eval, archives, terminal MLE stage, DPLR off, fresh weights) with two changes:

  prior_path = molecules_path = <family>_pooled_oct02_prior.pt
      Built by configs/pool_oct02: the zp1_sep28 campaign pooled with every older prior and search file, the pooled
      basins within 5 kT as anchors, a training-temperature capped-MC walk from each (every accepted move kept),
      thinned in latent space and written with every normaliser image the +1 chart holds (x4 in P-1, x8 in P2_1/c).
      `prior` and `equalized_prior` are the same rows (owner 2026-10-05: everything points at the full set).
  buffers.prior_buffer.max_size = buffers.anchor_buffer.max_size = the file's row count
      mk_dev's 250,000 / 200,000 are below every file; a smaller cap subsamples at seeding and breaks up the image sets.

Not acridine: its pooled prior is the pool_acr_oct03 battery's.

  prior_scan_cache = true
      The first launch scores every prior row and writes <prior>.scan-<hash>.pt beside the file; a requeue or relaunch
      loads it after re-scoring 512 rows (prior_scan_cache.py).

NOTE ON START-UP: train.py re-scores every prior row at the FIRST launch. The UMA files hold 263,060 (mipu) and 334,968 (nehu)
rows, so those two arms spend their first minutes to an hour in that scan.

OUT-OF-BOX STATES: kept states outside the trainer latent box were filtered from the files, all images (owner
2026-10-05): 176 states from neh, 13 from nehu, none from mip / mipu.

DENSITY: states the trainer's density penalty touches (energies.molecular_crystal.density_penalty > 0, i.e. a packing
coefficient below 0.55 or above 0.95) were filtered from the files, all images (owner 2026-10-05): 3 states from mipu,
109 from neh, 27 from nehu, none from mip; every one was below 0.55.
"""
import copy
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / rel / 'make.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fr = _load('mle_fresh_sep17', 'frmake')
w3, nig = fr.w3, fr.nig

TAG = 'mlepl'
BATTERY = 'mle_pool_oct05'
#: family -> (prior file, its size in bytes on the cluster, its row count)
PRIORS = {
    'mip':  ('mipcas_elj_pooled_oct02_prior.pt', 200_815_040, 263_172),
    'mipu': ('mipcas_uma_pooled_oct02_prior.pt', 200_729_472, 263_060),
    'neh':  ('nehzor_elj_pooled_oct02_prior.pt', 346_122_055, 289_872),
    'nehu': ('nehzor_uma_pooled_oct02_prior.pt', 399_966_599, 334_968),
}


def build_arm(base, fam):
    fr.TAG = TAG
    name, cfg, dropped = fr.build_arm(base, fam)
    fname, _nbytes, rows = PRIORS[fam]
    prior = f'{w3.CLUSTER_DATA}/{fname}'
    cfg['prior_path'] = prior
    cfg['molecules_path'] = prior
    cfg['buffers']['prior_buffer']['max_size'] = rows
    cfg['buffers']['anchor_buffer']['max_size'] = rows
    cfg['prior_scan_cache'] = True
    return name, cfg, dropped


def check(cfg, name, fam):
    spec, _to_prior = fr.FAM[fam]
    seed = w3.load(spec['seed'])
    mine, theirs = w3.problem_def(cfg), w3.problem_def(seed)
    moved = sorted(k for k in set(mine) | set(theirs) if k != 'prior_path' and mine.get(k) != theirs.get(k))
    assert not moved, f'{name}: problem identity moved from the seed problem on {moved}'
    fname, _nbytes, rows = PRIORS[fam]
    assert cfg['prior_path'] == cfg['molecules_path'] == f'{w3.CLUSTER_DATA}/{fname}', name
    assert cfg['buffers']['prior_buffer']['max_size'] == cfg['buffers']['anchor_buffer']['max_size'] == rows, name
    assert cfg['buffers']['prior_buffer']['seed_source'] == cfg['buffers']['anchor_buffer']['seed_source'] == \
        'prior_dataset', name
    assert cfg['prior_scan_cache'] is True, name
    assert cfg['checkpoint_name'] is None and cfg['continue_from_checkpoint'] == w3.CONT_PLACEHOLDER, name
    assert cfg['load_weights_only'] is False and cfg['model']['dplr_rank'] == 0, name
    assert cfg['lr_control']['fixed_scale'] == w3.SCALE == 2.0, name
    st = cfg['protocols'][w3.PROTOCOL]['stages']
    assert len(st) == 1 and 'exit' not in st[0] and st[0]['train_mode'] == 'bwd', name
    assert st[0]['bwd_sampling_mode'] == 'dataset', name
    assert cfg['integrator']['T'] == w3.SHIP_T == cfg['eval_T'], name
    w3._scan_local_paths(cfg, name)


def main(argv):
    dirty = w3.dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted:\n  ' + '\n  '.join(dirty))
    base = yaml.safe_load(w3.MK_DEV.read_text(encoding='utf-8'))
    arms = {}
    for fam in PRIORS:
        name, cfg, dropped = build_arm(copy.deepcopy(base), fam)
        check(cfg, name, fam)
        w3.load_check(cfg, name)
        arms[name] = (cfg, fam, dropped)
        local = w3.LOCAL_DATA / PRIORS[fam][0]
        if local.exists():
            assert local.stat().st_size == PRIORS[fam][1], \
                f'{fam}: {local} is {local.stat().st_size} bytes, PRIORS says {PRIORS[fam][1]}'
        else:
            print(f'NOTE: {local} not on this machine; its byte size is unchecked')

    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n',
                                               encoding='utf-8')
    for name, (cfg, *_r) in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tstart\twarm_src\tprior\tprior_bytes\n')
        for name, (cfg, fam, _d) in arms.items():
            f.write(f"{name}\t{fam}\tfresh\t-\t{PRIORS[fam][0]}\t{PRIORS[fam][1]}\n")
    anchor = "SYM=${PROJECT_ROOT}/MXtalTools/mxtaltools/common/sym_utils.py\n"
    assert anchor in nig.SBATCH
    sb = (nig.SBATCH.replace(anchor, anchor + fr.WALL_GUARD)
          .replace('phase-1 MLE on the Niggli P-1 priors, warm (mle09 best, weights-only) and fresh.',
                   'phase-1 MLE from fresh weights, DPLR off, on the pooled priors of 2026-10-05.')
          .replace('__LAST__', str(len(arms) - 1)).replace('__TAG__', TAG).replace('__BATTERY__', BATTERY)
          .replace('__CKPTS__', w3.CLUSTER_CKPTS).replace('__DATA__', w3.CLUSTER_DATA))
    assert '__' not in sb.replace('__pycache__', ''), 'an unfilled placeholder in the sbatch'
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(sb)
    for i, (name, (cfg, fam, dropped)) in enumerate(arms.items()):
        lc = cfg['lr_control']
        print(f"[{i}] {name:<16} prior={PRIORS[fam][0]} rows={PRIORS[fam][2]:,} fixed_scale={lc['fixed_scale']} "
              f"burn_in={lc.get('burn_in_steps')}@{lc.get('burn_in_scale')} excursion_k="
              f"{lc['hard_failure']['loss_excursion_k']} cut={lc.get('fire_cut_factor')} lr={lc['fixed_scale'] * lc['seed_lr']:g} "
              f"epochs={cfg['epochs']} batch={cfg.get('batch_size')}")


if __name__ == '__main__':
    main(sys.argv[1:])
