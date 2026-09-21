"""nehu_mlefan_sep21 -- a 4-hour MLE learning-rate fan on nehu (owner 2026-09-21): weights-only forks of the
running mlefr_nehu_lr2 MLE arm from its CURRENT best-MLE checkpoint at rates 1.0, 2.0 (the parent's), 4.0, 8.0,
each through burn-in 500 at 0.05 and the 1000-step geometric promotion ramp, batch pinned 1600.

    python configs/nehu_mlefan_sep21/make.py

WHY. The nehu MLE at 2.0 is crawling: bwd/mle -27.1 -> -27.8 over 50k steps (125k -> 175k), nonthermal 2.2%,
and phase 2 inherits that fit (p20_nehu_n5 had the worst start of the five). 4.0 detonated on the OLD promotion
jump (mlefr_*_lr4, 17916557); the geometric ramp has since replaced the jump and has not been tried on nehu at
4.0 or above. Read at 4 h: bwd/mle level and slope, bwd/tb_err, Nonthermal Fraction, lr_ctrl fires.

Shape and guards as nehu_fork3_sep19 (mle_fresh_sep17's nehu arm: DPLR off, W3 prior, protocol mle_w3); seed
resolved by the sbatch as *mlefr_nehu_lr2_*_best.pt at launch (the parent keeps running; best.pt is swapped
atomically). Wall 4:00:00.
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

TAG = 'nlrm'
BATTERY = 'nehu_mlefan_sep21'
SRC = 'mlefr_nehu_lr2'
SCALES = (1.0, 2.0, 4.0, 8.0)
RAMP_STEPS = 1000
BATCH = 1600
WALL = '4:00:00'


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
        cfg['max_batch_size'] = BATCH
        cfg['grow_batch_size'] = False
        cfg['batch_util_target'] = 0.0
        # detach_pb was NEVER READ (retired 2026-09-20 in the working tree; committed code ignores it): dropped
        for blk in ('bwd_loss_coeffs', 'replay_loss_coeffs'):
            cfg[blk].pop('detach_pb', None)
        assert w3.problem_def(cfg) == w3.problem_def(parent), name
        assert cfg['model']['dplr_rank'] == 0 and lc['fire_cut_factor'] == 1.0 and lc['mode'] == 'fixed', name
        assert cfg['protocol'] == 'mle_w3', name
        w3._scan_local_paths(cfg, name)
        w3.load_check(cfg, name)
        arms[name] = cfg
    return arms


def main(argv):
    arms = build()
    sb = (HERE.parent / 'mle_fresh_sep17' / 'submit_mle_fresh_sep17.sbatch').read_text(encoding='utf-8')
    assert sb.count('*${SRC}_*_best.pt') >= 3 and '--array=0-4' in sb and '--job-name=mlefr\n' in sb and '--time=2-00:00:00' in sb
    sb = (sb.replace('--array=0-4', f'--array=0-{len(arms) - 1}')
          .replace('--job-name=mlefr\n', f'--job-name={TAG}\n')
          .replace('--time=2-00:00:00', f'--time={WALL}')
          .replace('mle_fresh_sep17', BATTERY)
          .replace('phase-1 MLE from fresh weights, DPLR off, on the W3 P2_1/c and Niggli P-1 priors.',
                   'nehu MLE learning-rate fan: weights-only forks of mlefr_nehu_lr2 from its current best.pt at rates 1/2/4/8, 4 h.'))
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
        lc = cfg['lr_control']
        print(f"[{i}] {name:<16} {lc['burn_in_steps']} @ {lc['burn_in_scale']} -> ramp {lc['promotion_ramp_steps']} -> "
              f"{lc['fixed_scale']:g}  batch {cfg['batch_size']}  seed=*{SRC}_*_best.pt  wall {WALL}")


if __name__ == '__main__':
    main(sys.argv[1:])
