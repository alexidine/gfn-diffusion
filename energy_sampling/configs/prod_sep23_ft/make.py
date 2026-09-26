"""prod_sep23_ft -- fine-tuning legs on the converged prod_sep20 samplers with the box and reduction penalties
RAMPED 100x (owner 2026-09-23: "fine-tuning runs on the converged mip/mipu/neh/nehu checkpoints where we ramp /
boost the reduction and box penalties ... a somewhat smooth ramp ... 100x over 10k or 20k steps").

    python configs/prod_sep23_ft/make.py

THE ARMS: p23_<fam>_ft for mip / mipu / neh / nehu / acr -- the prod_sep20 production config of the family (every
5th step, P_B frozen, replay pinned 0.3 / bwd 0.7, tau 600, rate 0.5, batch 1600; acr: MACE, batch 1000,
traj_checkpoint_modes None, added 2026-09-25 once the owner judged p20_acr_n5 converged enough) plus, on the
equilibration stage,
    coeff_schedule: {bounding_coeff: {target: 1000, steps: 20000}, reduction_coeff: {target: 1000, steps: 20000}}
a GEOMETRIC ramp of both penalty coefficients from the config's 10 to 1000 over 20k train steps, anchored at
the leg's first tick (stage_ctrl.coeff_sched_entry rides the checkpoint, so a requeue continues the ramp),
then held. Each arm is a FULL resume of the p20 arm's newest step archive (`_stepN.pt` + its frozen
`_stepN_buffers.pt`): the coefficients are NOT part of the problem identity (utils._NON_IDENTITY_ENERGY_CONFIG_KEYS),
so the load is the ordinary continuation -- stage `equilibration` continues, P_B snapshot restored, LR at the
restored cruise scale, replay buffer restored.

THE TRANSIENT: stored replay rows carry the reward they were admitted with, so rows touching a penalty are
under-penalised until they turn over (tau 600; ~1.15x per residence at this ramp rate). At coefficient 10 the
mean penalties per sample were mip 0.006 box / 0.055 reduction with 10% of forward samples in box contact;
mipu 0.0001 / 0.031 (5%); neh 0.00002 / 0.021 (1%); nehu 0.00002 / 0.004 (1.4%); acr 0.0007 / 0.0066 with 14.4% in
box contact (p20_acr_n5 at step 267.7k, 2026-09-25) -- mip and acr are where the ramp bites.

READ: energy_func//bounding_coeff and //reduction_coeff (the live ramp), protocol/coeff_sched_p_*, fwd/box_contact_frac,
Mean bounding_energy / reduction_energy (should FALL as the coefficients rise), fwd/log_Z_learned (will drop as
mass leaves the walls; the new landscape has its own Z), eval_fwd/tb_err, zmatch/delta_mean, replay/val_gap_nats.
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

TAG = 'p23'
BATTERY = 'prod_sep23_ft'
FAMS = ['mip', 'mipu', 'neh', 'nehu', 'acr']     # array index = position here; acr (4) was added after the first four launched
N = 5
TARGET = 1000.0
STEPS = 20000
SCHEDULE = {'bounding_coeff': {'target': TARGET, 'steps': STEPS, 'kind': 'geometric'},
            'reduction_coeff': {'target': TARGET, 'steps': STEPS, 'kind': 'geometric'}}


def main(argv):
    dirty = fin.w3.dirty_files()
    if dirty:
        print('WARNING: the working tree is dirty (base read from git HEAD; the cluster runs HEAD):\n  ' + '\n  '.join(dirty))
    base = fin.committed_mk_dev()
    arms = {}
    for fam in FAMS:
        p20.TAG = TAG
        name, cfg = p20.build_arm(base, fam, N)
        name = f'{TAG}_{fam}_ft'
        cfg['run_name'] = name
        eq = fin._stage(cfg, 'equilibration')
        eq['coeff_schedule'] = copy.deepcopy(SCHEDULE)
        p20.check(cfg, name, fam, N)
        assert eq['coeff_schedule'] == SCHEDULE and cfg['energy_config']['bounding_coeff'] == 10.0 == cfg['energy_config']['reduction_coeff'], name
        fin.load_check(cfg, name, ['train_prior', 'equilibration'])
        arms[name] = (cfg, fam)
    prior_index = {l.split('\t')[1]: l.split('\t') for l in (ROOT / 'mle_fresh_sep17' / 'INDEX.tsv').read_text(encoding='utf-8').splitlines()[1:]}
    for stale in HERE.glob(f'{TAG}_*.yaml'):
        stale.unlink()
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n', encoding='utf-8')
    for name, (cfg, fam) in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    # warm_src = the p20 production arm; the sbatch seeds from ITS newest step archive (SEED_B block)
    fin._write_index(HERE / 'INDEX_a.tsv', [(name, fam, 'switch', f'p20_{fam}_n5', prior_index[fam][4], prior_index[fam][5])
                                           for name, (cfg, fam) in arms.items()])
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(fin.SBATCH.format(wall=fin.WALL, last=len(arms) - 1, tag=TAG, battery=BATTERY, leg='a',
                                  ckpts=fin.w3.CLUSTER_CKPTS, data=fin.w3.CLUSTER_DATA, seed_block=fin.SEED_B,
                                  what='fine-tuning legs on the converged p20 samplers: box and reduction penalties ramped 10 -> 1000 over 20k steps (coeff_schedule), seeded from each p20 arm\'s newest step archive.'))
    for i, (name, (cfg, fam)) in enumerate(arms.items()):
        print(f"[{i}] {name:<14} N={N} rate={cfg['lr_control']['fixed_scale']:g} batch={cfg['batch_size']} "
              f"ramp bounding+reduction 10 -> {TARGET:g} over {STEPS} steps  switch-from=*p20_{fam}_n5_*_step*.pt")


if __name__ == '__main__':
    main(sys.argv[1:])
