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

THE PRIOR, acridine_mace_pooled_oct03_pc055_prior.pt, is built by configs/pool_acr_oct03 (acridine P2_1/c Z'=1, the
universal conformer = the gas-phase minimum under acr_newmodel): the basins of the acr_zp1_sep30 campaign within 5 kT
as anchors, a training-temperature capped-MC walk from each, every row in the trainer's chart at handedness +1
(exact for planar acridine), states outside the trainer latent box left out (64 anchors), states outside the density
penalty's free range 0.55-0.95 left out (374 anchors, 617,326 of 1,131,800 walk states: the walk's 15 kT ceiling is
above this model's whole binding energy of 13.1 kT, so walkers went diffuse), walk states thinned in latent space to
fit 400,000 rows (radius 0.146), every kept state written with its 8 normaliser images. 10,527 anchors + 26,200 walk
states = 36,727 states x 8 = 293,816 rows. The molecule's own C2 relabelling is not applied (owner 2026-10-05).

NOTE ON START-UP: train.py re-scores every prior row at init: 293,816 MACE evaluations before the first step.
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
#: the prior file, its size in bytes, its row count
PRIOR = ('acridine_mace_pooled_oct03_pc055_prior.pt', 314_402_033, 293_816)


def build_arm(base):
    pl.PRIORS[FAM] = PRIOR
    name, cfg, dropped = pl.build_arm(base, FAM)
    assert cfg['mlip_path'] == MLIP_OLD, cfg['mlip_path']
    cfg['mlip_path'] = MLIP_NEW
    return name, cfg, dropped


def check(cfg, name):
    pl.check(cfg, name, FAM)  # problem identity = the seed arm's; prior, caps, fresh start, rate, terminal MLE stage
    assert cfg['energy_function'] == 'mace' and cfg['mlip_path'] == MLIP_NEW, name
    assert cfg['run_name'] == name == f'{TAG}_{FAM}_lr2', name


def main(argv):
    dirty = w3.dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted:\n  ' + '\n  '.join(dirty))
    base = yaml.safe_load(w3.MK_DEV.read_text(encoding='utf-8'))
    name, cfg, dropped = build_arm(copy.deepcopy(base))
    check(cfg, name)
    w3.load_check(cfg, name)
    local = w3.LOCAL_DATA / PRIOR[0]
    if local.exists():
        assert local.stat().st_size == PRIOR[1], f'{local} is {local.stat().st_size} bytes, PRIOR says {PRIOR[1]}'
    else:
        print(f'NOTE: {local} not on this machine; its byte size is unchecked')

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
