"""acr_m2_sep26 -- the acridine sampler switched to a NEW MACE checkpoint (owner 2026-09-26: "we have another
acridine mace checkpoint. I'd like to start it from the live acridine run and let it converge. Hopefully it will not
be too disruptive and it will re-converge nicely").

    python configs/acr_m2_sep26/make.py

MEASURED 2026-09-26 (24 anchors of the w3 prior, local card, the trainer's own analyze call): new - old lattice
energy +43.7 kJ/mol per molecule (row-to-row sd 0.95; old mean -75.3, new -31.6), i.e. +17.5 nat at kT 2.5 with
0.4 nat of row-to-row disagreement. So log Z must fall by ~17 nat at the switch and every reward stored under the
old model is ~17 nat too high: the buffers cannot be carried across, and Z cannot be left to TB.

THE ARM: am2_acr_ft = p23_acr_ft's config (the prod_sep20 production shape on acridine: MACE, batch 1000, every 5th
step, P_B frozen, replay pinned 0.3 / bwd 0.7, tau 600, rate 0.5) with four changes:
  * mlip_path -> MLIP_NEW, the new checkpoint. mlip_path is NOT part of the problem identity
    (utils.get_problem_definition), so the seed loads as an ordinary FULL resume: same stage, P_B snapshot restored,
    LR at the restored cruise scale.
  * buffers.fresh_on_switch: true -- the seed is ANOTHER run's checkpoint, so no buffer sidecar is restored
    (checkpointing.fresh_buffers_on_switch prints that it fired): the prior and anchor buffers seed from the prior
    dataset the new model re-analyses at startup (14,671 rows; every launch pays that scan anyway) and the replay
    buffer refills from live rollouts. Every stored row the run trains on is scored by the new model; nothing is
    re-scored in place. A requeue loads the arm's OWN checkpoint and restores its sidecar as usual.
  * the box and reduction penalties at their POST-RAMP value 1000 as plain config values (non-identity keys) and no
    coeff_schedule. Submit once p23_acr_ft's ramp has completed (protocol/coeff_sched_p_bounding_coeff = 1) and the
    switch changes nothing but the model; submitted earlier, the penalties jump from the ramp's current value to 1000
    at the switch (a copied stage starts a fresh stage_ctrl, so a ramp could not be continued through it anyway).
  * a copied-stage transition: `equilibration` exits at its first eval (always-true exit) into `equilibration_m2`, an
    exact copy, so the copy's on_enter fires for real against the new model at the first eval tick (an armed exit
    trigger pulls the eval forward; final_sep19 leg B's identical switch fired within its first tick):
      rebuild_prior_by_churn   the dataset-seeded prior buffer is replaced by the ordinary churned composition:
                               init_fraction x max_size noised anchors scored LIVE by the new model
                               (top_up_prior_from_anchors -> energy_function.log_reward).
      bootstrap_z:rollout:4000 log Z re-estimated from 4000 fresh rollouts under the new model -- the ~17 nat drop
                               in one step instead of TB walking it down through a Z-dominated residual.
      freeze_pb:full           a no-op on a resumed leg (Modeller.set_pb_freeze keeps the snapshot already installed).

AUTOMATIC on any startup: init_prior_dataset re-analyses the whole prior dataset (the anchor archive) through the live
energy function, and the gas-phase reference (mace_gas_pot) is computed per molecule by the live predictor and
cached, never checkpointed -- both carry the new model's numbers from step 0.

NOTHING STALE: with fresh_on_switch nothing scored by the old model survives the load. The cost is the entry
transient the p20 arms already went through from MLE: a replay buffer that fills from ~0 to its tau-600 equilibrium
(~120k rows) over ~2k steps, the first steps before the transition trained against the old Z.

SEED: the sbatch seeds from p23_acr_ft -- its newest 5000-step archive by default, or its _running.pt with
SRC_RUNNING=1 in the environment (cancel p23_acr_ft first). The buffers are not used either way. The new weights
must sit at MLIP_NEW on the cluster; load_mace_model fails loudly otherwise.

READ: the trainlog's 'buffers.fresh_on_switch: ... NOT restored' line and 'bootstrap_z: log_Z <-' at the transition;
fwd/log_Z_learned (the ~17 nat drop, then the settle), eval_fwd/tb_err, bwd/tb_err, replay/tb_err, Mean Sample
Energy (the new model's scale, ~-32 at the anchors), fwd/box_contact_frac, replay_buffer_length (0 -> ~120k over
~2k steps), prior_buffer_length (14.7k at load, ~62k after the rebuild, then regrows), zmatch/delta_mean.
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

TAG = 'am2'
BATTERY = 'acr_m2_sep26'
FAM = 'acr'
N = 5
SRC_ARM = 'p23_acr_ft'
SRC_YAML = 'prod_sep23_ft/p23_acr_ft.yaml'
MLIP_OLD = '/scratch/mk8347/data/acr_112025_mh1_stagetwo.model'
#: the new checkpoint's cluster path. Local copy on 2026-09-26: C:\Users\mikem\Downloads\acr_newmodel.model
#: (57,094,353 bytes); the old one lives beside it as D:\crystal_datasets\acr_112025_mh1_stagetwo.model locally.
MLIP_NEW = '/scratch/mk8347/data/acr_newmodel.model'
PENALTY = 1000.0   # prod_sep23_ft's ramp target, held as the plain config value
SWITCH_EXIT = [{'metric': 'fwd/log_Z_learned', 'above': -1.0e9, 'patience': 1}]


def build(base):
    p20.TAG = TAG
    name, cfg = p20.build_arm(base, FAM, N)
    name = f'{TAG}_{FAM}_ft'
    cfg['run_name'] = name
    assert cfg['mlip_path'] == MLIP_OLD, cfg['mlip_path']
    cfg['mlip_path'] = MLIP_NEW
    assert cfg['buffers']['fresh_on_switch'] is False, name + ': mk_dev default moved'
    cfg['buffers']['fresh_on_switch'] = True
    ec = cfg['energy_config']
    assert ec['bounding_coeff'] == 10.0 == ec['reduction_coeff'], name
    ec['bounding_coeff'] = PENALTY
    ec['reduction_coeff'] = PENALTY
    pre_switch = copy.deepcopy(cfg)          # the production shape, checked by p20.check below
    stages = cfg['protocols']['unconditional_tb']['stages']
    eq = fin._stage(cfg, 'equilibration')
    assert not eq.get('coeff_schedule'), name + ': the base config carries a ramp'
    eq['exit'] = copy.deepcopy(SWITCH_EXIT)
    m2 = copy.deepcopy(eq)
    m2['name'] = 'equilibration_m2'
    m2.pop('exit')
    m2.pop('on_exit', None)
    stages.append(m2)
    return name, cfg, pre_switch


def check(cfg, name, pre_switch):
    p20.check(pre_switch, name, FAM, N)
    seed = fin.load(SRC_YAML)
    assert fin.w3.problem_def(cfg) == fin.w3.problem_def(seed), name + ': problem identity moved from ' + SRC_ARM
    assert cfg['mlip_path'] == MLIP_NEW and seed['mlip_path'] == MLIP_OLD and MLIP_NEW != MLIP_OLD, name
    assert cfg['buffers']['fresh_on_switch'] is True and 'fresh_on_switch' not in seed['buffers'], name
    ec = cfg['energy_config']
    assert ec['bounding_coeff'] == PENALTY == ec['reduction_coeff'], name
    st = cfg['protocols']['unconditional_tb']['stages']
    assert [s['name'] for s in st] == ['train_prior', 'equilibration', 'equilibration_m2'], name
    eq, m2 = st[1], st[2]
    assert eq['exit'] == SWITCH_EXIT and 'exit' not in m2 and 'on_exit' not in m2, name
    assert m2['on_enter'] == eq['on_enter'] == ['rebuild_prior_by_churn', 'bootstrap_z:rollout:4000', 'freeze_pb:full'], name
    for s in (eq, m2):
        assert not s.get('coeff_schedule'), name + ': no ramp -- the penalties are plain config values'
        assert s['fwd_rollout_every'] == N and s['fracs'] == {'fwd': 0.0, 'bwd': 0.7, 'replay': 0.3}, name
        assert s['balance']['bounds'] == fin.PINNED_BOUNDS, name
    # everything else IS p23_acr_ft: strip the three intended differences and the seed's own ramp, then compare whole
    a, b = copy.deepcopy(seed), copy.deepcopy(cfg)
    b['protocols']['unconditional_tb']['stages'].pop()
    fin._stage(b, 'equilibration').pop('exit')
    fin._stage(a, 'equilibration')['coeff_schedule'] = fin._stage(b, 'equilibration')['coeff_schedule']
    for d in (a, b):
        d['run_name'] = d['tag'] = d['mlip_path'] = d['buffers']['fresh_on_switch'] = None
        d['energy_config']['bounding_coeff'] = d['energy_config']['reduction_coeff'] = None
    assert a == b, name + ': differs from ' + SRC_ARM + ' beyond the model path, fresh buffers, the penalties and the switch'


def main(argv):
    dirty = fin.w3.dirty_files()
    if dirty:
        print('WARNING: the working tree is dirty (base read from git HEAD; the cluster runs HEAD):\n  ' + '\n  '.join(dirty))
    base = fin.committed_mk_dev()
    name, cfg, pre_switch = build(base)
    check(cfg, name, pre_switch)
    fin.load_check(cfg, name, ['train_prior', 'equilibration', 'equilibration_m2'])
    prior_index = {l.split('\t')[1]: l.split('\t') for l in (ROOT / 'mle_fresh_sep17' / 'INDEX.tsv').read_text(encoding='utf-8').splitlines()[1:]}
    for stale in HERE.glob(f'{TAG}_*.yaml'):
        stale.unlink()
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n', encoding='utf-8')
    with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    row = prior_index[FAM]
    assert cfg['prior_path'].rsplit('/', 1)[1] == row[4], name
    fin._write_index(HERE / 'INDEX_a.tsv', [(name, FAM, 'switch', SRC_ARM, row[4], row[5])])
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(fin.SBATCH.format(wall=fin.WALL, last=0, tag=TAG, battery=BATTERY, leg='a',
                                  ckpts=fin.w3.CLUSTER_CKPTS, data=fin.w3.CLUSTER_DATA, seed_block=fin.SEED_B,
                                  what=f'the acridine sampler switched to the new MACE checkpoint {MLIP_NEW}: seeded from {SRC_ARM}, penalties held at {PENALTY:g}, prior buffer rebuilt and log Z bootstrapped at the copied-stage transition.'))
    print(f"[0] {name:<12} N={N} rate={cfg['lr_control']['fixed_scale']:g} batch={cfg['batch_size']} penalties {PENALTY:g} "
          f"fresh_on_switch=True mlip_path={MLIP_NEW}  switch-from=*{SRC_ARM}_*  (SRC_RUNNING=1 for its _running.pt)")
    print(f"the new weights must be at {MLIP_NEW} on the cluster before submitting")


if __name__ == '__main__':
    main(sys.argv[1:])
