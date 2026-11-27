"""pool4_nov26b: weight decay on MLE, multi-temperature TB, and training-molecule subsets, on pool4_nov26's arms
(configs/pool4_nov26/make.py).

  wd5e2       as pool4_nov26's mle4_ctrl (mle4's final weights, MLE_WARM_STEPS more MLE steps) with decoupled weight
              decay 5e-2 (AdamW on the policy, none on the Z head): compare with mle4_ctrl
  wd1e1       wd5e2 at 1e-1
  mt_ctrl     TB only, weights-only from pool4_nov26 tcond_c2's phase1_exit (TCOND_C2_EXIT), with
              temperature_conditioning (pool4_nov26's KAPPA_PER_ATOM and LOG_T_RANGE) and every row at the run's
              temperature (no `temperatures` key): the reference for mt_all and mt_ladder
  mt_all      mt_ctrl with the TB stage's `temperatures` ALL_TEMPS (every row draws one, stratified)
  mt_ladder   mt_ctrl with the TB stage split into LADDER's rungs of TB_STEPS / 4 steps each, each rung drawing over its
              own list; each rung enters with the TB stage's on_enter and ramps its LR (pool4_nov24's ramp)
  mol1k       fresh MLE exactly as pool4_nov26's cap_* arms' base (pool4_nov24 mle_live), MLE_FRESH_STEPS steps, with
              noised_prior.train_molecules 1000: the 1,000 molecules_path molecules with the lowest molecule_u and every
              prior row of them; the energy clip and the held-out panel (test_molecules_path) are the full arm's
  mol10k      mol1k at 10,000 molecules
  mol1k_cap   mol1k with pool4_nov26 cap_policy's flow (tf_width 768, tf_layers 12)
              Compare mol1k and mol10k with pool4_nov24 mle_live, and mol1k_cap with pool4_nov26 cap_policy, at matched
              steps. At small N equal steps are many more epochs: the batch sizer sets the rows per step by GPU use,
              not by N, so each molecule is drawn about 68,405 / N times as often as at full N (68,405 molecules).
  smoke_*     `smoke: true`, one per arm, at the first array indices: SMOKE_STEPS MLE steps (wd, mol); SMOKE_MT_STEPS TB
              steps (mt_ctrl, mt_all: pool4_nov24's LR ramp and 1,000 steps at the full TB LR); the ladder's smoke is
              SMOKE_MT_STEPS over its four rungs, so it crosses every rung boundary but no rung finishes its ramp.
The mt arms are ON HOLD: a local smoke (2026-11-26, eLJ, pool4_test, mle4 weights) put the per-temperature bootstrap
roots 27-35 nats apart at [2.5, 25], over the spec's 10-nat bar for the smallest-root levelling. Submit them only once
the owner has chosen how Z(c, T) is levelled at entry.
The mt arms replay by `visits` (duty is refused with `temperatures`), run pool4_nov24's TB stage otherwise and ramp the
LR at every stage entry (pool4_nov24's ramp). They NEED tcond_c2's phase1_exit to exist first: submit them after
pool4_nov26 tcond_c2 has written it (the sbatch stops on a missing source checkpoint). Every wd and mol arm stops at its
MLE stage's exit (snapshot:phase1_exit); every mt arm at its last TB stage's (snapshot:tb_exit). The mol arms need the
code that reads noised_prior.train_molecules on the cluster: noised_prior.settings raises on an unknown key."""
import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('n26', HERE.parent / 'pool4_nov26' / 'make.py')
n26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n26)
p4 = n26.p4

TAG = 'pool4_nov26b'
SMOKE_STEPS = n26.SMOKE_STEPS
#: tcond_c2's MLE exit (pool4_nov26/tcond_c2.yaml, snapshot:phase1_exit), named as the trainer names a snapshot:
#: <tag>_<run_name>_<problem slug>_<snapshot>.pt. The slug is utils.problem_slug of that yaml: mle4's stem
#: (pool4_nov24 STEM, hash 1d2dd7) with the hash fa971a, since temp_cond and energy_config.log_temperature_range are
#: identity (utils.get_problem_definition; computed 2026-11-26 from the committed yaml, which reproduces 1d2dd7 for
#: mle4_ctrl). CHECK the file is in the cluster CKDIR under this name before submitting the mt arms.
TCOND_STEM = 'elj-pool8_4sg_train_k4_metric-T2.5-fa971a'
TCOND_C2_EXIT = f'{n26.TAG}_tcond_c2_{TCOND_STEM}_phase1_exit.pt'
TB_STEPS = 40000                 # TB steps per mt arm, the ladder's rungs together
SMOKE_MT_STEPS = p4.RAMP_STEPS + 1000   # the TB LR ramp, then 1,000 steps at its full value
T_RUN = 2.5                      # energy_config.temperature of the pool4 arms: the store's currency and the evals'
ALL_TEMPS = [2.5, 5.0, 10.0, 25.0]
LADDER = [[25.0], [25.0, 10.0], [25.0, 10.0, 5.0], [25.0, 10.0, 5.0, 2.5]]


def decayed(name, steps, wd):
    cfg = n26.warm(name, steps)
    cfg.update(use_weight_decay=True, weight_decay=wd, weight_decay_kind='decoupled', tag=TAG)
    return cfg


def mt_base(name, tb_steps):
    """pool4_nov24's TB arm (MLE stage skipped on the loaded weights) from tcond_c2's phase1_exit, with temperature
    conditioning as tcond_c2 trained it, replay by visits and the TB stage's LR ramped."""
    cfg = p4.tb_arm(name, [('buffers.store.replay_weight', 'visits'), ('tb.lr_schedule', p4.ramp(0.2))],
                    tb_steps=tb_steps)
    p4._set(cfg, 'checkpoint_name', TCOND_C2_EXIT)
    assert cfg['load_weights_only'] is True and cfg['energy_config']['temperature'] == T_RUN
    cfg['temperature_conditioning'] = True
    cfg['noised_prior']['kick_kappa_per_atom'] = n26.KAPPA_PER_ATOM
    cfg['energy_config']['log_temperature_range'] = list(n26.LOG_T_RANGE)
    cfg['tag'] = TAG
    return cfg


def mt_ctrl(name, tb_steps):
    return mt_base(name, tb_steps)


def mt_all(name, tb_steps):
    cfg = mt_base(name, tb_steps)
    p4._tb(cfg)['temperatures'] = list(ALL_TEMPS)
    return cfg


def subset(name, steps, n_mol, nf=()):
    """pool4_nov26's fresh MLE arm (pool4_nov24 mle_live with model.nf changed by `nf`) on n_mol training molecules."""
    cfg = n26.fresh(name, steps, dict(nf))
    assert cfg['noised_prior'].get('train_molecules') is None and cfg['buffers']['store']['enabled'] is True
    cfg['noised_prior']['train_molecules'] = n_mol
    cfg['tag'] = TAG
    return cfg


def mt_ladder(name, tb_steps):
    """The TB stage as len(LADDER) consecutive stages tb1..tbK of tb_steps / K steps, rung k drawing over LADDER[k-1];
    each rung repeats the TB stage's on_enter and LR ramp (a transition rebuilds the optimizers), the last keeps its
    on_exit."""
    cfg = mt_base(name, tb_steps)
    stages = cfg['protocols'][p4.uni.PROTOCOL]['stages']
    tb = p4._tb(cfg)
    at = stages.index(tb)
    K = len(LADDER)
    assert tb_steps % K == 0, (tb_steps, K)
    rungs = []
    for k, temps in enumerate(LADDER, start=1):
        rung = copy.deepcopy(tb)
        rung.update(name=f'tb{k}', temperatures=list(temps), max_steps=tb_steps // K)
        if k < K:
            rung['on_exit'] = [f'snapshot:tb{k}_exit']
        rungs.append(rung)
    stages[at:at + 1] = rungs
    return cfg


ARMS = {
    'wd5e2': lambda s: decayed('wd5e2', s, 5.0e-2),
    'wd1e1': lambda s: decayed('wd1e1', s, 1.0e-1),
    'mt_ctrl': lambda s: mt_ctrl('mt_ctrl', s),
    'mt_all': lambda s: mt_all('mt_all', s),
    'mt_ladder': lambda s: mt_ladder('mt_ladder', s),
    'mol1k': lambda s: subset('mol1k', s, 1000),
    'mol10k': lambda s: subset('mol10k', s, 10000),
    'mol1k_cap': lambda s: subset('mol1k_cap', s, 1000, dict(tf_width=768, tf_layers=12)),
}
STEPS = {'wd5e2': n26.MLE_WARM_STEPS, 'wd1e1': n26.MLE_WARM_STEPS, 'mt_ctrl': TB_STEPS, 'mt_all': TB_STEPS,
         'mt_ladder': TB_STEPS, 'mol1k': n26.MLE_FRESH_STEPS, 'mol10k': n26.MLE_FRESH_STEPS,
         'mol1k_cap': n26.MLE_FRESH_STEPS}
SMOKE = {'wd5e2': SMOKE_STEPS, 'wd1e1': SMOKE_STEPS, 'mt_ctrl': SMOKE_MT_STEPS, 'mt_all': SMOKE_MT_STEPS,
         'mt_ladder': SMOKE_MT_STEPS, 'mol1k': SMOKE_STEPS, 'mol10k': SMOKE_STEPS, 'mol1k_cap': SMOKE_STEPS}


def configs():
    out = {}
    for name, build in ARMS.items():
        sm = build(SMOKE[name])
        sm.update(run_name=f'smoke_{name}', smoke=True, eval_period=100, archive_period=10 ** 6)
        out[sm['run_name']] = sm
    for name, build in ARMS.items():
        out[name] = build(STEPS[name])
        out[name]['run_name'] = name
    return out


if __name__ == '__main__':
    cfgs = configs()
    for name, cfg in cfgs.items():
        assert cfg['run_name'] == name and cfg['tag'] == TAG
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    text = p4.sbatch(list(cfgs)).replace(p4.TAG, TAG)
    (HERE / f'submit_{TAG}.sbatch').write_text(text, encoding='utf-8', newline='\n')
    full = [n for n in cfgs if not n.startswith('smoke_')]
    for name in full:
        cfg = copy.deepcopy(cfgs[name])
        cfg.update(continue_from_checkpoint=True, checkpoint_name=None, load_weights_only=False)
        (HERE / f'{name}_resume.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    (HERE / f'submit_{TAG}_resume.sbatch').write_text(p4.sbatch([f'{n}_resume' for n in full]).replace(p4.TAG, TAG),
                                                      encoding='utf-8', newline='\n')
    print(' '.join(cfgs))
