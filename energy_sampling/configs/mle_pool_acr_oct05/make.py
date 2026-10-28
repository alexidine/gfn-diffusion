"""mle_pool_acr_oct05 -- phase 1 (bwd MLE) from FRESH weights on the pooled acridine prior of 2026-10-05.

    python configs/mle_pool_acr_oct05/make.py

THE ARM is mle_pool_oct05's (its build_arm, family 'acr': mle_fresh_sep17's arm shape on today's committed mk_dev --
rate scale 2.0, burn-in, batch policy, eval, archives, terminal MLE stage, DPLR off, fresh weights, prior_path =
molecules_path = the pooled file, both buffer caps = its row count) with one change:

  mlip_path = /scratch/mk8347/data/acr_newmodel.model
      The MACE checkpoint the prior was searched, walked and scored under (acr_m2_sep26 made the same switch for the
      sampler). mlip_path is NOT part of the trainer's problem identity (utils.get_problem_definition), so nothing in a
      checkpoint load tells this model from the older one: a generator that builds on family 'acr' must set mlip_path
      itself, as this one does (the family default is the older, more strongly binding checkpoint).

THE PRIOR, acridine_mace_pooled_oct05_prior.pt, is built by configs/pool_acr_oct03 (acridine P2_1/c Z'=1, the
universal conformer = the gas-phase minimum under acr_newmodel) on pool_oct02's final rule: the basins of the
acr_zp1_sep30 campaign within 5 kT as anchors, a training-temperature capped-MC walk from each, every row in the
trainer's chart at handedness +1 (exact for planar acridine); candidates more than 10 kT above the lowest anchor, outside
the trainer latent box, or under the trainer's density penalty left out; one thinning radius for every pair, 0.0312
(the latent kick that raises a relaxed basin by a median 1 kT), no row budget; every kept state written with its 8
normaliser images. The molecule's own C2 relabelling is not applied (owner 2026-10-05).

ITS SIZE AND ROW COUNT are read from the assembly's own summary, acridine_mace_pooled_oct05_prior.pt.summary.json,
copied beside this file from the priors directory once the assemble job has run. Without it the generator refuses:
the arm's buffer caps and the job's size guard are the file's, not numbers typed here.

START-UP: prior_scan_cache is true (mle_pool_oct05): the first launch scores every prior row with MACE and writes the
cache beside the prior; later launches and requeues load it after re-scoring 512 rows.
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


pl = _load('mle_pool_oct05', 'plmake')
fr, w3, nig = pl.fr, pl.w3, pl.nig

TAG = pl.TAG
BATTERY = 'mle_pool_acr_oct05'
FAM = 'acr'
MLIP_OLD = '/scratch/mk8347/data/acr_112025_mh1_stagetwo.model'
MLIP_NEW = '/scratch/mk8347/data/acr_newmodel.model'
PRIOR_FILE = 'acridine_mace_pooled_oct05_prior.pt'
SUMMARY = HERE / (PRIOR_FILE + '.summary.json')


def prior():
    """(file name, bytes, rows) from the assemble job's summary; exits when the summary has not been copied here."""
    import json
    if not SUMMARY.exists():
        sys.exit(f'REFUSING: {SUMMARY.name} is not here. Run configs/pool_acr_oct03/launch.sh assemble on the cluster, '
                 f'then copy the summary it writes into {HERE.name}/.')
    d = json.loads(SUMMARY.read_text(encoding='utf-8'))
    assert d['file'] == PRIOR_FILE and d['energy_key'] == 'mace' and d['folded'] is True, d
    return PRIOR_FILE, int(d['bytes']), int(d['rows'])


def build_arm(base):
    pl.PRIORS[FAM] = prior()
    name, cfg, dropped = pl.build_arm(base, FAM)
    assert cfg['mlip_path'] == MLIP_OLD, cfg['mlip_path']
    cfg['mlip_path'] = MLIP_NEW
    return name, cfg, dropped


def check(cfg, name):
    pl.check(cfg, name, FAM)  # problem identity = the seed arm's; prior, caps, fresh start, rate, terminal MLE stage
    assert cfg['energy_function'] == 'mace' and cfg['mlip_path'] == MLIP_NEW, name
    assert cfg['prior_scan_cache'] is True, name
    assert cfg['run_name'] == name == f'{TAG}_{FAM}_lr2', name


def main(argv):
    dirty = w3.dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted:\n  ' + '\n  '.join(dirty))
    base = yaml.safe_load(w3.MK_DEV.read_text(encoding='utf-8'))
    name, cfg, dropped = build_arm(copy.deepcopy(base))
    check(cfg, name)
    w3.load_check(cfg, name)
    PRIOR = prior()

    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n',
                                               encoding='utf-8')
    with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tstart\twarm_src\tprior\tprior_bytes\n')
        f.write(f"{name}\t{FAM}\tfresh\t-\t{PRIOR[0]}\t{PRIOR[1]}\n")
    anchor = "SYM=${PROJECT_ROOT}/MXtalTools/mxtaltools/common/sym_utils.py\n"
    assert anchor in nig.SBATCH
    sb = (nig.SBATCH.replace(anchor, anchor + fr.WALL_GUARD)
          .replace('phase-1 MLE on the Niggli P-1 priors, warm (mle09 best, weights-only) and fresh.',
                   'phase-1 MLE from fresh weights, DPLR off, on the pooled acridine prior of 2026-10-05.')
          .replace('__LAST__', '0').replace('__TAG__', TAG).replace('__BATTERY__', BATTERY)
          .replace('__CKPTS__', w3.CLUSTER_CKPTS).replace('__DATA__', w3.CLUSTER_DATA))
    assert '__' not in sb.replace('__pycache__', ''), 'an unfilled placeholder in the sbatch'
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(sb)
    lc = cfg['lr_control']
    print(f"[0] {name:<16} prior={PRIOR[0]} rows={PRIOR[2]:,} mlip={cfg['mlip_path'].rsplit('/', 1)[1]} "
          f"fixed_scale={lc['fixed_scale']} burn_in={lc.get('burn_in_steps')}@{lc.get('burn_in_scale')} excursion_k="
          f"{lc['hard_failure']['loss_excursion_k']} cut={lc.get('fire_cut_factor')} "
          f"lr={lc['fixed_scale'] * lc['seed_lr']:g} epochs={cfg['epochs']} batch={cfg.get('batch_size')}->"
          f"{cfg.get('max_batch_size')}; energy_config keys dropped to match the seed identity: {dropped}")


if __name__ == '__main__':
    main(sys.argv[1:])
