"""nehu_eqfan_sep21 -- a 4-hour equilibration learning-rate fan on nehu (owner 2026-09-21): the prod_sep20 shape
(phase 2 from the current mlefr_nehu_lr2 best-MLE checkpoint, rollout every 5th step, P_B frozen at entry, replay
pinned 0.3 / bwd 0.7, tau 600, batch 1600) at rates 0.125, 0.25, 1.0, 2.0. p20_nehu_n5 (0.5) is the fifth point,
already running.

    python configs/nehu_eqfan_sep21/make.py

WHY. At 0.5 p20_nehu_n5's gradient norm rose 88 -> 122 over its first 2k steps with the forward error creeping up,
while p20_nehu_n10 at 0.25 (every 10th) sat at the same log Z with flat gradients -- rate or rollout period? This
fan holds N=5 and moves only the rate, two rungs below and two above production. Read at 4 h: gradnorm and
lr_ctrl fires (stability), replay/resid_vs_intake and replay/val_gap_nats (memorisation), fwd/log_Z_learned and
eval_fwd/tb_err (progress), all at matched steps against p20_nehu_n5.

Everything else exactly prod_sep20 (imports its builder). Wall 4:00:00.
"""
import copy
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
_spec = importlib.util.spec_from_file_location('p20make', ROOT / 'prod_sep20' / 'make.py')
p20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p20)
fin = p20.fin

TAG = 'nlre'
BATTERY = 'nehu_eqfan_sep21'
FAM = 'nehu'
N = 5
RATES = (0.125, 0.25, 1.0, 2.0)
WALL = '4:00:00'


def main(argv):
    dirty = fin.w3.dirty_files()
    if dirty:
        print('WARNING: the working tree is dirty (base read from git HEAD; the cluster runs HEAD):\n  ' + '\n  '.join(dirty))
    base = fin.committed_mk_dev()
    arms = {}
    for rate in RATES:
        p20.RATE_N5[FAM] = float(rate)          # p20.build_arm / p20.check read the family rate from here
        p20.TAG = TAG
        name, cfg = p20.build_arm(base, FAM, N)
        name = f"{TAG}_nehu_lr{('%g' % rate).replace('.', 'p')}"
        cfg['run_name'] = name
        p20.check(cfg, name, FAM, N)
        fin.load_check(cfg, name, ['train_prior', 'equilibration'])
        assert cfg['lr_control']['fixed_scale'] == rate and cfg['tag'] == TAG, name
        arms[name] = cfg
    prior_index = {l.split('\t')[1]: l.split('\t') for l in (ROOT / 'mle_fresh_sep17' / 'INDEX.tsv').read_text(encoding='utf-8').splitlines()[1:]}
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n', encoding='utf-8')
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    row = prior_index[FAM]
    fin._write_index(HERE / 'INDEX_a.tsv', [(name, FAM, 'seed', fin.SRC[FAM], row[4], row[5]) for name in arms])
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(fin.SBATCH.format(wall=WALL, last=len(arms) - 1, tag=TAG, battery=BATTERY, leg='a',
                                  ckpts=fin.w3.CLUSTER_CKPTS, data=fin.w3.CLUSTER_DATA, seed_block=fin.SEED_A,
                                  what='nehu equilibration learning-rate fan: the prod_sep20 every-5th shape at rates 0.125/0.25/1/2, 4 h.'))
    for i, (name, cfg) in enumerate(arms.items()):
        print(f"[{i}] {name:<18} N={N} rate={cfg['lr_control']['fixed_scale']:g} batch={cfg['batch_size']} tau=600 "
              f"pinned 0.3/0.7 frozen-at-entry seed=*{fin.SRC[FAM]}_*_best.pt  wall {WALL}")


if __name__ == '__main__':
    main(sys.argv[1:])
