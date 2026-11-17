"""pool8_big2_nov17: continuations of the pool8_big_nov15 run (wandb vtba4cf5; the big NF on the pool8 P-1 K=4 noised set),
weights-only from its step archives. Owner 2026-11-17: TB continuation, unaugmented, from step 50,000 (held-out
likelihood peaked at 42k-51k; later steps overfit); and short augmented MLE runs from the 10k-increment archives, to find
how much unaugmented pretraining augmentation needs. Augmented TB comes later, from whichever augmented arm reads best.

Arms (one GPU each; every arm = pool8_big_nov15/make.py's config, the same data, model and GPU settings, then):
  tb        mk_dev's conditional_vargrad protocol (its train_prior stage is skipped on a weights load; var_conditioning
            runs), the tbhead recipe of the local phase-2 runs: fwd pooled_center 'head', bootstrap_z removed (a weights
            load carries no condition_log_z state), stage LR held at 0.2 x seed from step 0, the conditional buffer
            globals (COND_GLOBALS), fixed batch. Rotation and mirror augmentation off. Source: step 50,000.
  aug<N>k   augmented MLE (the local rotation recipe, semiS1): rotate_frames on, mirror off, canonical frame none,
            orient_grad_balance norm, rho 1; the MLE stage kept on the weights load (skip_if removed), its LR held at
            0.2 x seed from step 0 (the source's post-hold LR). AUG_STEPS steps. Source: step N x 1000.

usage: python configs/pool8_big2_nov17/make.py             (arm yamls + submit_pool8_big2_nov17.sbatch, an array)
       python configs/pool8_big2_nov17/make.py --local DIR [--source CKPT] [--steps N]
                                                             (local smoke configs on the norbornane files)"""
import argparse
import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
_spec = importlib.util.spec_from_file_location('big', HERE.parent / 'pool8_big_nov15' / 'make.py')
big = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(big)
nov10 = big.nov10

TAG = 'pool8_big2_nov17'
SOURCE_RUN = 'pool8_big_nov15_big'
SOURCE_SLUG = 'elj-pool8_p1_train_k4_metric-T2.5-7dae6f'   # utils.problem_slug of pool8_big_nov15/big.yaml
TB_SOURCE_STEP = 50000
AUG_SOURCE_STEPS = (10000, 20000, 30000, 40000, 50000)
TB_STEPS = 60000            # a cap: the 2-day limit ends the run first
AUG_STEPS = 15000
TB_BATCH = 1412             # the source run's landed batch on an A100-80
COND_GLOBALS = {            # the conditional-arm buffer globals of the local phase-2 (tbhead) runs
    'grow_batch_size': False, 'batch_util_target': 0.0,
    'buffers.replay_buffer.churn_rate': 0, 'buffers.replay_buffer.mean_residence_steps': 1200,
    'buffers.replay_buffer.max_size': 50000, 'buffers.replay_buffer.val_frac': 0.0,
    'buffers.replay_buffer.admit_from_eval': False, 'buffers.replay_buffer.prioritise.enabled': False,
    'buffers.anchor_buffer.frozen': True, 'buffers.anchor_buffer.thin_every_n_evals': 0,
    'buffers.anchor_buffer.refresh_every_n_evals': 0, 'buffers.anchor_buffer.topup_admit_record_breakers': True,
    'buffers.prior_buffer.source': 'anchors', 'condition_log_z.rollout_condition_draw': 'cycle',
    'z_calibration.fill_threshold': 0.0, 'z_calibration.fill_from_eval': 'off',
}


def source_name(step):
    return f'{SOURCE_RUN}_{SOURCE_SLUG}_step{step}.pt'


def put(cfg, dotted, value):
    *head, last = dotted.split('.')
    node = cfg
    for k in head:
        node = node[k]
    assert last in node, dotted
    node[last] = value


def tb_config(base, source):
    cfg = copy.deepcopy(base)
    mk = yaml.safe_load((REPO / 'configs' / 'mk_dev.yaml').read_text(encoding='utf-8'))
    for k, v in COND_GLOBALS.items():
        put(cfg, k, v)
    proto = copy.deepcopy(mk['protocols']['conditional_vargrad'])
    stages = proto['stages']
    assert [s['name'] for s in stages] == ['train_prior', 'var_conditioning'], [s['name'] for s in stages]
    assert stages[0].get('skip_if') == 'weights_loaded'
    vc = stages[1]
    vc['on_enter'] = [x for x in vc['on_enter'] if x != 'bootstrap_z']
    vc['loss_coeffs']['fwd']['pooled_center'] = 'head'
    vc['lr_schedule'] = [{'to': 0.2, 'steps': 0, 'kind': 'hold'}]
    cfg['protocols']['conditional_vargrad'] = proto
    nf = cfg['model']['nf']
    nf.update(rotate_frames=False, mirror_frames=False, canonical_frame='none')
    cfg.update(protocol='conditional_vargrad', checkpoint_name=source, load_weights_only=True,
               continue_from_checkpoint=False, archive_buffers=False, archive_period=5000, eval_period=500,
               batch_size=TB_BATCH, epochs=TB_STEPS)
    return cfg


def aug_config(base, source):
    cfg = copy.deepcopy(base)
    nf = cfg['model']['nf']
    nf.update(rotate_frames=True, mirror_frames=False, canonical_frame='none', orient_grad_balance='norm',
              orient_grad_rho=1.0)
    stage = cfg['protocols'][cfg['protocol']]['stages'][0]
    assert stage['name'] == 'train_prior' and stage.pop('skip_if') == 'weights_loaded'
    stage['lr_schedule'] = [{'to': 0.2, 'steps': 0, 'kind': 'hold'}]
    stage['max_steps'] = AUG_STEPS
    cfg.update(checkpoint_name=source, load_weights_only=True, continue_from_checkpoint=False, archive_period=2500,
               epochs=AUG_STEPS)
    return cfg


def arms(files, source_for):
    base = big.config(files)
    out = {'tb': tb_config(base, source_for(TB_SOURCE_STEP))}
    for s in AUG_SOURCE_STEPS:
        out[f'aug{s // 1000}k'] = aug_config(base, source_for(s))
    for name, cfg in out.items():
        cfg.update(run_name=name, tag=TAG)
    return out


def sbatch(names):
    text = (HERE / 'submit_template.sbatch').read_text(encoding='utf-8')
    files = (big.PRIOR, big.CONDITIONS, big.TEST)
    return (text.replace('@TAG@', TAG).replace('@ARMS@', ' '.join(names)).replace('@NARMS@', str(len(names) - 1))
            .replace('@DATA@', big.DATA).replace('@FILES@', ' '.join(files)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--local', default=None, help='write local smoke configs (norbornane files) into this directory')
    p.add_argument('--source', default=None, help='local smoke: checkpoint file (in the local checkpoints dir) to load')
    p.add_argument('--steps', type=int, default=200)
    a = p.parse_args()
    if a.local:
        out = Path(a.local)
        out.mkdir(parents=True, exist_ok=True)
        for name, cfg in arms(nov10.LOCAL, lambda s: a.source).items():
            cfg.update(checkpoints_dir='D:/crystal_datasets/gfn_checkpoints', tag=f'{TAG}_local', smoke=True,
                       cuda_memory_fraction=0.5, epochs=a.steps, batch_size=min(cfg['batch_size'], 64),
                       eval_period=max(50, a.steps // 2), archive_period=10 ** 6)
            proto = cfg['protocols'][cfg['protocol']]['stages']
            for st in proto:
                if 'max_steps' in st:
                    st['max_steps'] = min(st['max_steps'], a.steps)
            (out / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
        print(f'local smoke configs in {out}')
    else:
        files = {'prior': f'{big.DATA}/{big.PRIOR}', 'conditions': f'{big.DATA}/{big.CONDITIONS}',
                 'test': f'{big.DATA}/{big.TEST}'}
        made = arms(files, source_name)
        for name, cfg in made.items():
            (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
        (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(list(made)), encoding='utf-8', newline='\n')
        print(', '.join(made), f'and submit_{TAG}.sbatch in {HERE}')
