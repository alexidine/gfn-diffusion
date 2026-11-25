"""pool4_nov24: the 4-space-group pool on the unified store route. TB arms warm-start from mle4's final weights, plus one
fresh MLE arm on the live physical kernel and two smokes.

  base      unified_nov23's arm (pool8sg4 prior, store on, physical live kernel, TB stage 'tb'), with the batch sizer on
            (grow_batch_size, batch_util_target UTIL_TARGET), and the score and pathwise terms present at 0 so a stage
            may set them
  tb_m4     weights-only from mle4 (MLE4_CK; cross-attention `split`, as mle4 trained); the MLE stage is skipped
  tb_m4_*   tb_m4 with one change, named in ARMS
  mle_live  fresh, MLE stage only on fresh physical kicks, `split_pooled` (no cross-attention), stops at phase1_exit
  smoke_*   `smoke: true`, SMOKE_TB_STEPS TB steps: one per TB arm (same change), and mle_live's shape followed by TB.
            The smokes take the first array indices: submit them first (sbatch --array=0-<last smoke>), the rest after.
  resume    `--resume ARM` writes ARM_resume.yaml: continue_from_checkpoint true, checkpoint_name null,
            load_weights_only false (a TB arm resubmitted with its checkpoint_name would restart from mle4 at step 0).

  legs      every job asks for LEG_TIME (12 h), so full arms run as chained legs. Each full arm also gets
            ARM_resume.yaml, run by submit_pool4_nov24_resume.sbatch (indices 0-8 = main indices 9-17):
            sbatch --dependency=afterany:<previous leg> --array=0-8 configs/pool4_nov24/submit_pool4_nov24_resume.sbatch

  aug       augmented MLE before augmented TB, in their own sbatch files (submit_pool4_nov24_aug.sbatch, index order
            smoke_mle4_aug, smoke_tb_m4aug_aug, mle4_aug, tb_m4aug_aug; each TB arm needs its MLE arm's phase1_exit,
            so submit in that order with afterok): mle4_aug trains the MLE stage MLE_AUG_STEPS steps from mle4's final
            weights with random frames (rotate, mirror; replay rule visits, since duty is refused with frames) and
            stops; tb_m4aug_aug is tb_m4_aug started from mle4_aug's phase1_exit instead of mle4.

usage: python configs/pool4_nov24/make.py              (arm yamls, resume yamls and the sbatch files)
       python configs/pool4_nov24/make.py --resume ARM (ARM_resume.yaml)"""
import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('unified_nov23', HERE.parent / 'unified_nov23' / 'make.py')
uni = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uni)

TAG = 'pool4_nov24'
MLE4_CK = 'pool8sg4_nov20_mle4_elj-pool8_4sg_train_k4_metric-T2.5-1d2dd7_final.pt'   # in the sbatch CKDIR
UTIL_TARGET = 0.7           # mle4's batch_util_target
TB_STEPS = 100000
MLE_STEPS = 150000
SMOKE_TB_STEPS, SMOKE_MLE_STEPS = 600, 300
AUX_KEYS = {'bwd': ('score', 'score_beta', 'score_rows', 'score_force'),
            'replay': ('score', 'score_beta', 'score_rows', 'score_force'),
            'fwd': ('score', 'score_beta', 'score_rows', 'pathwise', 'pathwise_grad_clip', 'pathwise_entropy')}
AUX_OFF = {'score': 0.0, 'score_beta': 10.0, 'score_rows': 0, 'score_force': 'live', 'pathwise': 0.0,
           'pathwise_grad_clip': 100.0, 'pathwise_entropy': 1.0}   # entropy 1.0 = the code default: reverse KL
# config paths an arm may change: every one is read on the store route (a legacy buffers.* key would sweep nothing)
LIVE_PREFIXES = ('protocols.', 'model.nf.rotate_frames', 'model.nf.mirror_frames', 'model.nf.atom_tokens',
                 'buffers.store.replay_weight', 'buffers.store.keep_rollouts', 'buffers.store.growth_every', 'fwd_loss_coeffs.tb_detach_z', 'energy_config.temperature', 'energy_config.density_coeff', 'warm_start_ignore_problem_keys', 'lr_control.hard_failure.grad_abs', 'smoke',
                 'checkpoint_name',
                 'load_weights_only', 'epochs')


def _tb(cfg):
    return next(s for s in cfg['protocols'][uni.PROTOCOL]['stages'] if s['name'] == 'tb')


def _mle(cfg):
    return next(s for s in cfg['protocols'][uni.PROTOCOL]['stages'] if s['name'] == 'train_prior')


def base(tb_steps=TB_STEPS, mle_steps=MLE_STEPS):
    cfg = uni.unified({'prior': f'{uni.ref.sg4.DATA}/{uni.ref.sg4.PRIOR}',
                       'conditions': f'{uni.ref.sg4.DATA}/{uni.ref.sg4.CONDITIONS}',
                       'test': f'{uni.ref.sg4.DATA}/{uni.ref.sg4.TEST}'}, mle_steps=mle_steps, tb_steps=tb_steps)
    assert cfg['grow_batch_size'] is False and cfg['batch_util_target'] == 0.0
    cfg.update(tag=TAG, grow_batch_size=True, batch_util_target=UTIL_TARGET)
    for branch, keys in AUX_KEYS.items():
        block = cfg[f'{branch}_loss_coeffs']
        for k in keys:
            block.setdefault(k, AUX_OFF[k])
            assert block[k] == AUX_OFF[k] or k == 'score_force', (branch, k, block[k])
    return cfg


def _set(cfg, path, value):
    assert path.startswith(LIVE_PREFIXES), f'{path} is not a store-route setting'
    *head, last = path.split('.')
    node = cfg
    for k in head:
        node = node[k]
    node[last] = value


def from_mle4(name, cfg):
    """TB only, from mle4's final weights: the MLE stage's skip_if weights_loaded skips it."""
    assert _mle(cfg).get('skip_if') == 'weights_loaded'
    _set(cfg, 'checkpoint_name', MLE4_CK)
    _set(cfg, 'load_weights_only', True)
    _set(cfg, 'model.nf.atom_tokens', 'split')
    cfg['run_name'] = name
    return cfg


def tb_arm(name, changes=(), tb_steps=TB_STEPS):
    cfg = from_mle4(name, base(tb_steps=tb_steps))
    for path, value in changes:
        if path.startswith('tb.'):
            node, keys = _tb(cfg), path.split('.')[1:]
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = value
        else:
            _set(cfg, path, value)
    return cfg


def mle_live(name='mle_live', mle_steps=MLE_STEPS):
    cfg = base(mle_steps=mle_steps)
    assert cfg['model']['nf']['atom_tokens'] == 'split_pooled'
    mle = _mle(cfg)
    mle['on_exit'] = ['snapshot:phase1_exit', 'stop']
    cfg['protocols'][uni.PROTOCOL]['stages'] = [mle]
    cfg.update(run_name=name, epochs=mle_steps + uni.LOOP_MARGIN)
    return cfg


MLE_AUG_STEPS, SMOKE_MLE_AUG_STEPS = 20000, 300     # 20k: one 12 h leg with room; TB needs its phase1_exit
FRAMES = [('buffers.store.replay_weight', 'visits'), ('model.nf.rotate_frames', True), ('model.nf.mirror_frames', True)]
STEM = 'elj-pool8_4sg_train_k4_metric-T2.5-1d2dd7'      # the problem stem of this prior's checkpoint names


def mle4_aug(name='mle4_aug', mle_steps=MLE_AUG_STEPS):
    """MLE only, weights-only from mle4's final weights, with random frames; stops at phase1_exit."""
    cfg = from_mle4(name, base(mle_steps=mle_steps))
    for path, value in FRAMES:
        _set(cfg, path, value)
    mle = _mle(cfg)
    del mle['skip_if']                                  # the weights are loaded and this arm trains the MLE stage
    mle.update(max_steps=mle_steps, on_exit=['snapshot:phase1_exit', 'stop'])
    cfg['protocols'][uni.PROTOCOL]['stages'] = [mle]
    cfg.update(epochs=mle_steps + uni.LOOP_MARGIN)
    return cfg


def tb_from_mle_aug(name, source_run, tb_steps=TB_STEPS):
    """tb_m4_aug's settings, weights-only from source_run's phase1_exit snapshot."""
    cfg = tb_arm(name, FRAMES, tb_steps=tb_steps)
    _set(cfg, 'checkpoint_name', f'{TAG}_{source_run}_{STEM}_phase1_exit.pt')
    return cfg


def aug_configs():
    sm_mle = mle4_aug('smoke_mle4_aug', SMOKE_MLE_AUG_STEPS)
    sm_tb = tb_from_mle_aug('smoke_tb_m4aug_aug', 'smoke_mle4_aug', tb_steps=SMOKE_TB_STEPS)
    for c in (sm_mle, sm_tb):
        c.update(smoke=True, eval_period=100, archive_period=10 ** 6)
    return {'smoke_mle4_aug': sm_mle, 'smoke_tb_m4aug_aug': sm_tb, 'mle4_aug': mle4_aug(),
            'tb_m4aug_aug': tb_from_mle_aug('tb_m4aug_aug', 'mle4_aug')}



#: LR changes ramp, never jump (owner rule, nov25): a stage's schedule starts at 1% of its level and rises
#: geometrically to it over RAMP_STEPS
RAMP_STEPS = 2000


def ramp(to, steps=RAMP_STEPS):
    return [{'to': to / 100, 'steps': 0, 'kind': 'hold'}, {'to': to, 'steps': steps, 'kind': 'geometric'}]


#: on-policy in place of replay: the forward branch trains the policy by TB on a fresh rollout every step at replay's
#: weight, as replay runs it (against the detached learned head, Huber knee 10: tb_detach_z), while emp_z fits the
#: Z(c) head on the same rollouts; no rollout cadence or sidecar; the store writes no rollout row (so no replay pool),
#: and growth kicks are owed on every 20th rollout, tb_m4's kick rate. Its LR ramps to tb_m4's level over RAMP_STEPS.
ONPOL = [('fwd_loss_coeffs.tb_detach_z', 0.0),         # the base block carries the key the stage overrides
         ('tb.fracs', {'fwd': 0.5, 'bwd': 0.5, 'replay': 0.0}), ('tb.fwd_rollout_every', 0),
         ('tb.fwd_z_sidecar', False), ('tb.loss_coeffs.fwd.freeze_policy', 0.0), ('tb.loss_coeffs.fwd.tb', 1.0),
         ('tb.loss_coeffs.fwd.tb_detach_z', 1.0), ('tb.lr_schedule', ramp(0.2)),
         ('buffers.store.keep_rollouts', False), ('buffers.store.growth_every', 20)]


def onpol_configs():
    sm = tb_arm('smoke_tb_m4_onpol', ONPOL, tb_steps=SMOKE_TB_STEPS)
    sm.update(smoke=True, eval_period=100, archive_period=10 ** 6)
    return {'smoke_tb_m4_onpol': sm, 'tb_m4_onpol': tb_arm('tb_m4_onpol', ONPOL)}

#: hot TB: tb_m4 at a higher temperature (energy_config.temperature; tb_m4 runs 2.5). The MLE stage is reward-free, so
#: mle4's weights may start another temperature, but T is part of the checkpoint's problem identity: each arm exempts
#: it (warm_start_ignore_problem_keys, honoured on the weights-only load only). The energy window (window_kT x T)
#: widens with T and the reward cache keys on T. Each LR change ramps (RAMP_STEPS).
#: energy_config.density_coeff scales with T (BASE_T, BASE_DENSITY): the density term enters the energy beside the eLJ
#: energy and is divided by T with it, while the bounding and reduction terms are pre-multiplied by T, so without the
#: scaling the packing wall would soften by T / BASE_T in nats and a hot target would open loose, gas-like cells.
HOT = {'tb_m4_T10': 10.0, 'tb_m4_T25': 25.0}
BASE_T, BASE_DENSITY = 2.5, 10.0
HOT_COMMON = [('warm_start_ignore_problem_keys', ['energy_config.temperature']), ('tb.lr_schedule', ramp(0.2))]


def hot_changes(T):
    return [('energy_config.temperature', T), ('energy_config.density_coeff', BASE_DENSITY * T / BASE_T)] + HOT_COMMON


def hot_configs():
    base_cfg = base()
    assert base_cfg['energy_config']['temperature'] == BASE_T, base_cfg['energy_config']['temperature']
    assert base_cfg['energy_config']['density_coeff'] == BASE_DENSITY, base_cfg['energy_config']['density_coeff']
    out = {}
    for name, T in HOT.items():                         # smokes first: they take the first array indices
        sm = tb_arm(f'smoke_{name}', hot_changes(T), tb_steps=SMOKE_TB_STEPS)
        sm.update(smoke=True, eval_period=100, archive_period=10 ** 6)
        out[sm['run_name']] = sm
    for name, T in HOT.items():
        out[name] = tb_arm(name, hot_changes(T))
    return out


ARMS = {
    'tb_m4': [],
    'tb_m4_lr4': [('tb.lr_schedule', [{'to': 0.8, 'steps': 0, 'kind': 'hold'}])],
    'tb_m4_visits': [('buffers.store.replay_weight', 'visits')],
    'tb_m4_aug': [('buffers.store.replay_weight', 'visits'), ('model.nf.rotate_frames', True),
                  ('model.nf.mirror_frames', True)],
    'tb_m4_every4': [('tb.fwd_rollout_every', 4)],
    'tb_m4_knee20': [('tb.loss_coeffs.bwd.beta', 20.0)],
    'tb_m4_sm': [('tb.loss_coeffs.bwd.score', 0.1), ('tb.loss_coeffs.bwd.score_rows', 0),        # 0: every bwd row
                 ('tb.loss_coeffs.bwd.score_beta', 10.0), ('tb.loss_coeffs.bwd.score_force', 'live')],
    'tb_m4_pw': [('tb.loss_coeffs.fwd.pathwise', 0.1), ('tb.loss_coeffs.fwd.pathwise_grad_clip', 100.0),
                 ('lr_control.hard_failure.grad_abs', None)],
}


def configs():
    out = {}
    for name, ch in ARMS.items():                       # smokes first: they take the first array indices
        sm = tb_arm(f'smoke_{name}', ch, tb_steps=SMOKE_TB_STEPS)
        sm.update(smoke=True, eval_period=100, archive_period=10 ** 6)
        out[sm['run_name']] = sm
    smoke_mle = base(tb_steps=SMOKE_TB_STEPS, mle_steps=SMOKE_MLE_STEPS)
    smoke_mle.update(run_name='smoke_mle', smoke=True, eval_period=100, archive_period=10 ** 6)
    out['smoke_mle'] = smoke_mle
    out.update({name: tb_arm(name, ch) for name, ch in ARMS.items()})
    out['mle_live'] = mle_live()
    return out


def resume(name):
    cfg = yaml.safe_load((HERE / f'{name}.yaml').read_text(encoding='utf-8'))
    cfg.update(continue_from_checkpoint=True, checkpoint_name=None, load_weights_only=False)
    return cfg


GUARDS = '''    if [ "$(awk '/^continue_from_checkpoint:/{print $2}' ${ARMS}/${ARM}.yaml)" = "true" ] && [ -n "${SRC}" ] && [ "${SRC}" != "null" ]; then
        echo "FATAL: ${ARM} resumes but still names checkpoint_name ${SRC}: it would restart from those weights (make.py --resume)" >&2; exit 1
    fi
    if [ "$(awk '/^continue_from_checkpoint:/{print $2}' ${ARMS}/${ARM}.yaml)" != "true" ]; then
        RUN0=$(awk '/^run_name:/{print $2}' ${ARMS}/${ARM}.yaml)
        if ls ${CKDIR}/@TAG@_${RUN0}_*_running.pt > /dev/null 2>&1; then
            echo "FATAL: ${ARM} starts fresh, but ${CKDIR} already holds a running checkpoint of @TAG@_${RUN0}: resume it (make.py --resume) or rename the run" >&2; exit 1
        fi
    fi
'''


LEG_TIME = '12:00:00'       # the cluster has been kind to 12 h jobs: full arms run as chained legs


def sbatch(arms):
    text = (HERE.parent / 'unified_nov23' / 'submit_template.sbatch').read_text(encoding='utf-8')
    assert text.count('#SBATCH --time=2-00:00:00') == 1
    text = text.replace('#SBATCH --time=2-00:00:00', f'#SBATCH --time={LEG_TIME}')
    anchor = "    SRC=$(awk '/^checkpoint_name:/{print $2}' ${ARMS}/${ARM}.yaml)\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + GUARDS)
    return (text.replace('@TAG@', TAG).replace('@ARMS@', ' '.join(arms)).replace('@NARMS@', str(len(arms) - 1))
            .replace('@RAM_GB@', str(uni.RAM_GB)))


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == '--resume':
        out = HERE / f'{sys.argv[2]}_resume.yaml'
        out.write_text(yaml.safe_dump(resume(sys.argv[2]), sort_keys=False), encoding='utf-8', newline='\n')
        print(out)
        sys.exit(0)
    cfgs = configs()
    for name, cfg in cfgs.items():
        assert cfg['run_name'] == name and cfg['tag'] == TAG
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(list(cfgs)), encoding='utf-8', newline='\n')
    full = [n for n in cfgs if not n.startswith('smoke_')]     # resume legs: same order as the main array's 9-17
    for name in full:
        (HERE / f'{name}_resume.yaml').write_text(yaml.safe_dump(resume(name), sort_keys=False), encoding='utf-8',
                                                  newline='\n')
    (HERE / f'submit_{TAG}_resume.sbatch').write_text(sbatch([f'{n}_resume' for n in full]), encoding='utf-8',
                                                      newline='\n')
    print(f'{len(cfgs)} arms in {HERE}: ' + ' '.join(cfgs) + f'; resume legs for {len(full)}')
    aug = aug_configs()
    for name, cfg in aug.items():
        assert cfg['run_name'] == name and cfg['tag'] == TAG and name not in cfgs
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    (HERE / f'submit_{TAG}_aug.sbatch').write_text(sbatch(list(aug)), encoding='utf-8', newline='\n')
    aug_full = [n for n in aug if not n.startswith('smoke_')]
    for name in aug_full:
        (HERE / f'{name}_resume.yaml').write_text(yaml.safe_dump(resume(name), sort_keys=False), encoding='utf-8',
                                                  newline='\n')
    (HERE / f'submit_{TAG}_aug_resume.sbatch').write_text(sbatch([f'{n}_resume' for n in aug_full]),
                                                          encoding='utf-8', newline='\n')
    print(f'aug: ' + ' '.join(aug) + f'; resume legs for {len(aug_full)}')
    onpol = onpol_configs()
    for name, cfg in onpol.items():
        assert cfg['run_name'] == name and cfg['tag'] == TAG and name not in cfgs and name not in aug
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    (HERE / f'submit_{TAG}_onpol.sbatch').write_text(sbatch(list(onpol)), encoding='utf-8', newline='\n')
    (HERE / 'tb_m4_onpol_resume.yaml').write_text(yaml.safe_dump(resume('tb_m4_onpol'), sort_keys=False),
                                                  encoding='utf-8', newline='\n')
    (HERE / f'submit_{TAG}_onpol_resume.sbatch').write_text(sbatch(['tb_m4_onpol_resume']), encoding='utf-8',
                                                            newline='\n')
    print('onpol: ' + ' '.join(onpol) + '; resume leg for tb_m4_onpol')
    hot = hot_configs()
    for name, cfg in hot.items():
        assert cfg['run_name'] == name and cfg['tag'] == TAG and name not in cfgs and name not in aug
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    (HERE / f'submit_{TAG}_hot.sbatch').write_text(sbatch(list(hot)), encoding='utf-8', newline='\n')
    hot_full = [n for n in hot if not n.startswith('smoke_')]
    for name in hot_full:
        (HERE / f'{name}_resume.yaml').write_text(yaml.safe_dump(resume(name), sort_keys=False), encoding='utf-8',
                                                  newline='\n')
    (HERE / f'submit_{TAG}_hot_resume.sbatch').write_text(sbatch([f'{n}_resume' for n in hot_full]),
                                                          encoding='utf-8', newline='\n')
    print('hot: ' + ' '.join(hot) + f'; resume legs for {len(hot_full)}')
