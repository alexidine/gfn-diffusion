"""rr10_sep22 -- every-10th rollouts at the FULL rate (owner 2026-09-22): is the every-10th lag a rate effect or a
rollout effect, and what does the memorisation gap do at ~10 passes per row?

    python configs/rr10_sep22/make.py

BACKGROUND. prod_sep20's every-10th arms run at HALF the every-5th rate (0.25 vs 0.5) to hold the per-row replay
dose. On mip the N=5 and N=10 curves superimpose on the ROLLOUT axis, which two stories explain: progress limited
by fresh data (the rate is irrelevant) or by the optimizer budget (ten steps at half = five at full; the halving
threw away half the steps). On nehu the halved every-10th arm is far behind every-5th per rollout (Z 16.9 at 31k
steps vs 23.2 at 21.5k), so on the hard surface the step budget does real work. The fin19 hold cells put the gap
at 0.15 / 0.20 / 0.27 for N = 5 / 10 / 20 at matched dose with no visible harm to log Z, energies or delta_mean.

ARMS (the prod_sep20 shape: from the mlefr best-MLE checkpoint, P_B frozen at entry, replay pinned 0.3 / bwd
0.7, batch 1600, no forces; entry mechanics IDENTICAL to p20 so the comparisons are clean):
  r10_mip_n10        every 10th, rate 0.5, tau 600     rate or rollouts, on the rollout axis vs p20_mip_n5 / p20_mip_n10
  r10_mip_n10_t1200  every 10th, rate 0.5, tau 1200    what the pool lever buys on top (same passes, rows spread 2x)
  r10_nehu_n10       every 10th, rate 0.5, tau 600     the hard surface: does full rate at every-10th track every-5th
                                                        per rollout. NB seeds from mlefr_nehu_lr2's CURRENT best (~10k
                                                        MLE steps later than p20_nehu_n5's 205.6k seed) -- a small confound.
READ at matched steps and on the rollout axis: fwd/log_Z_learned, eval_fwd/tb_err, zmatch/delta_mean (progress vs
every-5th); replay/val_gap_nats and replay/resid_vs_intake (the price, vs the fin19 hold cells). Tracks every-5th
per rollout with the gap under ~0.3 -> every-10th at full rate is a clean cost lever; gap runs away or progress
lags -> the halving was right and every-5th is simply the answer. mip reads at ~5 h, nehu at ~1 day. Wall 2 days.
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

TAG = 'r10'
BATTERY = 'rr10_sep22'
RATE = 0.5
ARMS = [('mip', 10, 600), ('mip', 10, 1200), ('nehu', 10, 600)]   # (family, fwd_rollout_every, tau)


def main(argv):
    dirty = fin.w3.dirty_files()
    if dirty:
        print('WARNING: the working tree is dirty (base read from git HEAD; the cluster runs HEAD):\n  ' + '\n  '.join(dirty))
    base = fin.committed_mk_dev()
    arms = {}
    for fam, n, tau in ARMS:
        p20.TAG = TAG
        p20.RATE_N5[fam] = RATE * n / 5.0          # p20.build_arm applies RATE_N5 x 5/N -> the FULL rate at every N
        p20.TAU = tau
        name, cfg = p20.build_arm(base, fam, n)
        name = f'{TAG}_{fam}_n{n}' + (f'_t{tau}' if tau != 600 else '')
        cfg['run_name'] = name
        p20.check(cfg, name, fam, n)
        fin.load_check(cfg, name, ['train_prior', 'equilibration'])
        assert cfg['lr_control']['fixed_scale'] == RATE and cfg['buffers']['replay_buffer']['mean_residence_steps'] == tau, name
        assert fin._stage(cfg, 'equilibration')['fwd_rollout_every'] == n and cfg['tag'] == TAG, name
        arms[name] = (cfg, fam, n, tau)
    prior_index = {l.split('\t')[1]: l.split('\t') for l in (ROOT / 'mle_fresh_sep17' / 'INDEX.tsv').read_text(encoding='utf-8').splitlines()[1:]}
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n', encoding='utf-8')
    for name, (cfg, *_r) in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    fin._write_index(HERE / 'INDEX_a.tsv', [(name, fam, 'seed', fin.SRC[fam], prior_index[fam][4], prior_index[fam][5])
                                           for name, (cfg, fam, n, tau) in arms.items()])
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(fin.SBATCH.format(wall=fin.WALL, last=len(arms) - 1, tag=TAG, battery=BATTERY, leg='a',
                                  ckpts=fin.w3.CLUSTER_CKPTS, data=fin.w3.CLUSTER_DATA, seed_block=fin.SEED_A,
                                  what='every-10th rollouts at the FULL rate 0.5 (mip tau 600 / tau 1200, nehu tau 600): rate or rollouts, and the memorisation price.'))
    for i, (name, (cfg, fam, n, tau)) in enumerate(arms.items()):
        print(f"[{i}] {name:<18} N={n} rate={cfg['lr_control']['fixed_scale']:g} tau={tau} batch={cfg['batch_size']} "
              f"pinned 0.3/0.7 frozen-at-entry seed=*{fin.SRC[fam]}_*_best.pt")


if __name__ == '__main__':
    main(sys.argv[1:])
