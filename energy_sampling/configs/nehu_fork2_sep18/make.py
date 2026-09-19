"""nehu_fork2_sep18 -- weights-only fork of mlefr_nehu_lr2 at its step-70000 archive at MLE rate 4.0 with
NO rate jump (burn_in_scale = fixed_scale) and a PINNED batch (1600, no sizer, no retests).

    python configs/nehu_fork2_sep18/make.py

Otherwise nehu_fork_sep18's lr4 arm: same parent config, seed archive and sbatch.
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

TAG = 'nfk2'
BATTERY = 'nehu_fork2_sep18'
SCALE = 4.0
BATCH = 1600


def build():
    parent = w3.load(nfk.PARENT)
    name = f'{TAG}_nehu_lr{SCALE:g}'
    cfg = copy.deepcopy(parent)
    cfg['run_name'] = name
    cfg['tag'] = TAG
    cfg['checkpoint_name'] = w3.CK_PLACEHOLDER
    cfg['load_weights_only'] = True
    cfg['continue_from_checkpoint'] = w3.CONT_PLACEHOLDER
    cfg['warm_start_ignore_problem_keys'] = None
    lc = cfg['lr_control']
    lc['fixed_scale'] = SCALE
    lc['burn_in_scale'] = SCALE
    cfg['batch_size'] = BATCH
    cfg['grow_batch_size'] = False
    cfg['batch_util_target'] = 0.0
    assert w3.problem_def(cfg) == w3.problem_def(parent), name
    assert cfg['model']['dplr_rank'] == 0 and lc['fire_cut_factor'] == 1.0, name
    w3._scan_local_paths(cfg, name)
    w3.load_check(cfg, name)
    return {name: cfg}


def main(argv):
    arms = build()
    sb = (HERE.parent / 'nehu_fork_sep18' / 'submit_nehu_fork_sep18.sbatch').read_text(encoding='utf-8')
    assert '--array=0-1' in sb and '--job-name=nfk\n' in sb
    sb = (sb.replace('--array=0-1', f'--array=0-{len(arms) - 1}')
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
        print(f"[{i}] {name}  burn_in {lc['burn_in_scale']} -> {lc['fixed_scale']}  batch {cfg['batch_size']} "
              f"grow={cfg['grow_batch_size']}  seed=*{row[3]}_*_step{nfk.SEED_STEP}.pt")


if __name__ == '__main__':
    main(sys.argv[1:])
