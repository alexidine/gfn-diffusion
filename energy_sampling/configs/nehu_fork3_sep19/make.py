"""nehu_fork3_sep19 -- weights-only forks of mlefr_nehu_lr2 at its step-120000 archive at MLE rates 1.0 and
4.0: burn-in 500 steps at 0.05, then the geometric promotion ramp (1000 steps) to the rate; batch pinned at 1600.

    python configs/nehu_fork3_sep19/make.py

Parent config, sbatch and guards as nehu_fork_sep18; only the seed step, rates, ramp and batch differ.
"""
import copy
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('nfkmake', HERE.parent / 'nehu_fork_sep18' / 'make.py')
nfk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nfk)
w3 = nfk.w3

TAG = 'nfk3'
BATTERY = 'nehu_fork3_sep19'
SEED_STEP = 120000
SCALES = (1.0, 4.0)
RAMP_STEPS = 1000
BATCH = 1600


def build():
    parent = w3.load(nfk.PARENT)
    arms = {}
    for s in SCALES:
        name = f'{TAG}_nehu_lr{s:g}'
        cfg = copy.deepcopy(parent)
        cfg['run_name'] = name
        cfg['tag'] = TAG
        cfg['checkpoint_name'] = w3.CK_PLACEHOLDER
        cfg['load_weights_only'] = True
        cfg['continue_from_checkpoint'] = w3.CONT_PLACEHOLDER
        cfg['warm_start_ignore_problem_keys'] = None
        lc = cfg['lr_control']
        lc['fixed_scale'] = float(s)
        lc['burn_in_scale'] = 0.05
        lc['burn_in_steps'] = 500
        lc['promotion_ramp_steps'] = RAMP_STEPS
        cfg['batch_size'] = BATCH
        cfg['grow_batch_size'] = False
        cfg['batch_util_target'] = 0.0
        assert w3.problem_def(cfg) == w3.problem_def(parent), name
        assert cfg['model']['dplr_rank'] == 0 and lc['fire_cut_factor'] == 1.0, name
        w3._scan_local_paths(cfg, name)
        w3.load_check(cfg, name)
        arms[name] = cfg
    return arms


def main(argv):
    arms = build()
    sb = (HERE.parent / 'nehu_fork_sep18' / 'submit_nehu_fork_sep18.sbatch').read_text(encoding='utf-8')
    assert sb.count(f'_step{nfk.SEED_STEP}.pt') >= 3 and '--array=0-1' in sb and '--job-name=nfk\n' in sb
    sb = (sb.replace(f'_step{nfk.SEED_STEP}.pt', f'_step{SEED_STEP}.pt')
          .replace('--array=0-1', f'--array=0-{len(arms) - 1}')
          .replace('--job-name=nfk\n', f'--job-name={TAG}\n')
          .replace('nehu_fork_sep18', BATTERY))
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n',
                                               encoding='utf-8')
    row = (HERE.parent / 'nehu_fork_sep18' / 'INDEX.tsv').read_text(encoding='utf-8').splitlines()[1].split('\t')
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tstart\twarm_src\tprior\tprior_bytes\n')
        for name in arms:
            f.write(f'{name}\tnehu\tfork\t{row[3]}\t{row[4]}\t{row[5]}\n')
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(sb)
    for i, (name, cfg) in enumerate(arms.items()):
        lc = cfg['lr_control']
        print(f"[{i}] {name}  {lc['burn_in_steps']} @ {lc['burn_in_scale']} -> ramp {lc['promotion_ramp_steps']} -> "
              f"{lc['fixed_scale']}  batch {cfg['batch_size']}  seed=*{row[3]}_*_step{SEED_STEP}.pt")


if __name__ == '__main__':
    main(sys.argv[1:])
