"""nehu_fork_sep18 -- two weights-only forks of mlefr_nehu_lr2 at its step-70000 archive, bracketing
the MLE rate: 4.0 and 1.0 (the running lr2 arm is the control).

    python configs/nehu_fork_sep18/make.py

Each arm is mle_fresh_sep17's nehu arm with: run_name/tag, lr_control.fixed_scale, and a weights-only
start from *mlefr_nehu_lr2_*_step70000.pt (fresh optimizer, burn-in 500 at 0.05, then the rate).
"""
import copy
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('frmake', HERE.parent / 'mle_fresh_sep17' / 'make.py')
fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fr)
w3 = fr.w3

TAG = 'nfk'
BATTERY = 'nehu_fork_sep18'
PARENT = 'mle_fresh_sep17/mlefr_nehu_lr2.yaml'
SRC = 'mlefr_nehu_lr2'
SEED_STEP = 70000
SCALES = (4.0, 1.0)


def build():
    parent = w3.load(PARENT)
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
        cfg['lr_control']['fixed_scale'] = float(s)
        assert w3.problem_def(cfg) == w3.problem_def(parent), name
        assert cfg['model']['dplr_rank'] == 0 and cfg['lr_control']['burn_in_scale'] == 0.05, name
        assert cfg['lr_control']['fire_cut_factor'] == 1.0, name
        w3._scan_local_paths(cfg, name)
        w3.load_check(cfg, name)
        arms[name] = cfg
    return arms


def main(argv):
    arms = build()
    fr_sb = (HERE.parent / 'mle_fresh_sep17' / 'submit_mle_fresh_sep17.sbatch').read_text(encoding='utf-8')
    seed_glob = '*${SRC}_*_best.pt'
    assert fr_sb.count(seed_glob) >= 3 and '--array=0-4' in fr_sb and 'mle_fresh_sep17' in fr_sb
    sb = (fr_sb.replace(seed_glob, f'*${{SRC}}_*_step{SEED_STEP}.pt')
          .replace('--array=0-4', f'--array=0-{len(arms) - 1}')
          .replace('--job-name=mlefr', f'--job-name={TAG}')
          .replace('mle_fresh_sep17', BATTERY))
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n',
                                               encoding='utf-8')
    parent_index = (HERE.parent / 'mle_fresh_sep17' / 'INDEX.tsv').read_text(encoding='utf-8').splitlines()
    nehu = next(l.split('\t') for l in parent_index if l.startswith('mlefr_nehu_lr2\t'))
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tstart\twarm_src\tprior\tprior_bytes\n')
        for name in arms:
            f.write(f'{name}\tnehu\tfork\t{SRC}\t{nehu[4]}\t{nehu[5]}\n')
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(sb)
    for i, (name, cfg) in enumerate(arms.items()):
        print(f"[{i}] {name}  scale={cfg['lr_control']['fixed_scale']}  seed=*{SRC}_*_step{SEED_STEP}.pt")


if __name__ == '__main__':
    main(sys.argv[1:])
