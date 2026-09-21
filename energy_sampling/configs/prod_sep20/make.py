"""prod_sep20 -- PRODUCTION phase 2 for the five main systems under RARE ROLLOUTS, from the mle_fresh_sep17
best-MLE checkpoints (owner 2026-09-20 evening: "N=1 is just never going to work on mlip runs ... no reason to
keep it at all if we are doing N=5 or more anyway").

    python configs/prod_sep20/make.py            # base = the COMMITTED mk_dev (git show HEAD:), warns on a dirty tree

THE SHAPE, per family, straight from the MLE checkpoint (no N=1 trunk): a rollout every 5th step, P_B frozen at
the phase-2 entry (freeze_pb:full on equilibration's on_enter, AFTER the stub's last MLE steps), replay 0.3 /
bwd 0.7 PINNED, tau 600, no forces, batch pinned (1600; acr 1000 for MACE memory), rate = the family's rate
(prod_sep12's paper slots: mip 1.0, neh 0.5, mipu 0.0625, nehu 0.125, acr 0.05) x 5/N.
  p20_<fam>_n5   the five production arms
  p20_mip_n10    every 10th step at half rate: does the rare climber reach the ceiling? judged against the
                 final_sep19 mip trunk (N=1, converged 35.7 at ~26k) at matched steps
  p20_nehu_n10   the same question on the production MLIP family

WHAT final_sep19 SETTLED (mip, 2026-09-20): a converged sampler HOLDS under a rollout every 5th/10th/20th step
with P_B frozen (log Z 35.7, better delta_mean, cooler tails, 1/4-1/15 the energy time); the memorisation gap
scales with N (0.075 / 0.15 / 0.20 / 0.27 at N = 1 / 5 / 10 / 20); unfreezing P_B after convergence collapses
within 5k steps; the stored force changes nothing at 2x the energy time; the free gated ramp walks the replay
share to the floor once coverage flattens and the model degrades -- the 0.3 pin wins. What it did NOT settle:
whether a rare climber reaches the ceiling from MLE. The dose ladder's closest evidence (from a half-trained
archive): N=5 full rate 35.25 @13.5k vs N=1 35.5; N=20 quarter rate 34.5 @10.7k still rising. The failure mode
is slow, not broken; the _n10 arms measure it.

MECHANICS: as final_sep19 leg A (FULL resume of *<src>_*_best.pt in stage train_prior; the config's train_prior
is the stub that exits at the first eval and snapshots the prior; the transition fires on_enter for real:
rebuild_prior_by_churn on anchors, bootstrap_z, freeze_pb:full; identity + model + energy_config VERBATIM from the
seed yaml; the W3 / Niggli wall guards and the prior byte-size guard in the sbatch; requeue-safe).
"""
import copy
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
_spec = importlib.util.spec_from_file_location('finmake', ROOT / 'final_sep19' / 'make.py')
fin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fin)

TAG = 'p20'
BATTERY = 'prod_sep20'
TAU = 600
PIN_REPLAY = 0.3
FAMS = ['mip', 'neh', 'mipu', 'nehu', 'acr']
fin.SEED_ARM.update({'neh': 'mle_fresh_sep17/mlefr_neh_lr2.yaml', 'mipu': 'mle_fresh_sep17/mlefr_mipu_lr2.yaml',
                     'acr': 'mle_fresh_sep17/mlefr_acr_lr2.yaml'})
fin.SRC.update({'neh': 'mlefr_neh_lr2', 'mipu': 'mlefr_mipu_lr2', 'acr': 'mlefr_acr_lr2'})
fin.MLIP.update({'neh': False, 'mipu': True, 'acr': True})
RATE_N5 = {'mip': 1.0, 'neh': 0.5, 'mipu': 0.0625, 'nehu': 0.125, 'acr': 0.05}
BATCH = {'mip': 1600, 'neh': 1600, 'mipu': 1600, 'nehu': 1600, 'acr': 1000}
ARMS = [(fam, 5) for fam in FAMS] + [('mip', 10), ('nehu', 10)]


def build_arm(base, fam, n):
    name = f'{TAG}_{fam}_n{n}'
    cfg = fin.common(copy.deepcopy(base), fam, name)
    cfg['tag'] = TAG
    eq = fin._stage(cfg, 'equilibration')
    eq['fwd_rollout_every'] = int(n)
    fin._pin(eq, PIN_REPLAY)
    eq['on_enter'] = list(eq['on_enter']) + ['freeze_pb:full']
    cfg['buffers']['replay_buffer']['mean_residence_steps'] = TAU
    fin._rate(cfg, RATE_N5[fam] * 5.0 / n)
    cfg['batch_size'] = BATCH[fam]
    cfg['max_batch_size'] = BATCH[fam]
    if fam == 'acr':
        # MACE sits at the memory ceiling and is unmeasured with a per-branch checkpoint split: every branch (prod_sep12)
        cfg['traj_checkpoint_modes'] = None
    return name, cfg


def check(cfg, name, fam, n):
    seed = fin.load(fin.SEED_ARM[fam])
    assert fin.w3.problem_def(cfg) == fin.w3.problem_def(seed), name + ': problem identity moved from the seed'
    assert cfg['prior_path'] == seed['prior_path'] == cfg['molecules_path'], name
    assert cfg['model'] == seed['model'] and cfg['model']['dplr_rank'] == 0, name
    assert cfg['integrator']['T'] == 100 == cfg['eval_T'], name
    assert cfg['checkpoint_name'] == fin.PLACEHOLDER and cfg['prior_model_name'] == fin.PRIOR_PLACEHOLDER, name
    assert cfg['load_weights_only'] is False and cfg['continue_from_checkpoint'] is False, name
    assert cfg['epochs'] >= 500_000 and cfg['tag'] == TAG and cfg['run_name'] == name, name
    stages = cfg['protocols']['unconditional_tb']['stages']
    assert cfg['protocol'] == 'unconditional_tb' and [s['name'] for s in stages] == ['train_prior', 'equilibration'], name
    stub, eq = stages
    assert stub['exit'] == [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}] and 'skip_if' not in stub, name
    assert eq['fwd_rollout_every'] == n and 'exit' not in eq, name
    assert eq['fracs'] == {'fwd': 0.0, 'bwd': 0.7, 'replay': 0.3} and eq['balance']['bounds'] == fin.PINNED_BOUNDS, name
    assert eq['on_enter'][-1] == 'freeze_pb:full' and 'rebuild_prior_by_churn' in eq['on_enter'], name
    assert eq['loss_coeffs']['fwd']['freeze_policy'] == 1.0, name
    assert cfg.get('freeze_backward_policy') in (False, None) and cfg['buffers']['prior_buffer']['source'] == 'anchors', name
    rb = cfg['buffers']['replay_buffer']
    assert rb['mean_residence_steps'] == TAU and rb['max_size'] >= 5 * BATCH[fam] * TAU / n and rb['val_frac'] == fin.VAL_FRAC, name
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == lc['burn_in_scale'] == RATE_N5[fam] * 5.0 / n, name
    assert cfg['batch_size'] == cfg['max_batch_size'] == BATCH[fam] and cfg['grow_batch_size'] is False, name
    fc, rc = cfg['fwd_loss_coeffs'], cfg['replay_loss_coeffs']
    assert fc['reward_grads'] == 0.0 and fc['traj_grads'] == 0.0 and fc['path_grad_last_k'] == 0, name
    assert rc['stored_force_k'] == 0 and rc['resample_last_k'] == 0, name
    assert cfg['traj_checkpoint'] is fin.MLIP[fam] and cfg['energy_config']['internal_oom_recovery'] is fin.MLIP[fam], name
    if fam == 'acr':
        assert cfg['traj_checkpoint_modes'] is None and cfg['energy_function'] == 'mace', name
    fin.w3._scan_local_paths(cfg, name)


def main(argv):
    dirty = fin.w3.dirty_files()
    if dirty:
        print('WARNING: the working tree is dirty (base read from git HEAD; the cluster runs HEAD):\n  ' + '\n  '.join(dirty))
    base = fin.committed_mk_dev()
    arms = {}
    for fam, n in ARMS:
        name, cfg = build_arm(base, fam, n)
        check(cfg, name, fam, n)
        fin.load_check(cfg, name, ['train_prior', 'equilibration'])
        arms[name] = (cfg, fam, n)
    prior_index = {l.split('\t')[1]: l.split('\t') for l in (ROOT / 'mle_fresh_sep17' / 'INDEX.tsv').read_text(encoding='utf-8').splitlines()[1:]}
    for stale in HERE.glob(f'{TAG}_*.yaml'):
        stale.unlink()
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n', encoding='utf-8')
    for name, (cfg, fam, n) in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    rows = []
    for name, (cfg, fam, n) in arms.items():
        row = prior_index[fam]
        assert cfg['prior_path'].rsplit('/', 1)[1] == row[4], name
        rows.append((name, fam, 'seed', fin.SRC[fam], row[4], row[5]))
    fin._write_index(HERE / 'INDEX_a.tsv', rows)
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(fin.SBATCH.format(wall=fin.WALL, last=len(arms) - 1, tag=TAG, battery=BATTERY, leg='a',
                                  ckpts=fin.w3.CLUSTER_CKPTS, data=fin.w3.CLUSTER_DATA, seed_block=fin.SEED_A,
                                  what='PRODUCTION phase 2 under rare rollouts from the mlefr best-MLE checkpoints: every 5th step, P_B frozen at entry, replay pinned 0.3; plus every-10th test arms on mip and nehu.'))
    for i, (name, (cfg, fam, n)) in enumerate(arms.items()):
        print(f"[{i}] {name:<14} N={n:2d} rate={cfg['lr_control']['fixed_scale']:g} batch={cfg['batch_size']} tau={TAU} "
              f"{'MLIP' if fin.MLIP[fam] else 'ELJ '} pinned 0.3/0.7 frozen-at-entry seed=*{fin.SRC[fam]}_*_best.pt")


if __name__ == '__main__':
    main(sys.argv[1:])
