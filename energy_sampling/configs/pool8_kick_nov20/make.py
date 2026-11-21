"""pool8_kick_nov20 training arms: the kick-scale MLE test (owner 2026-11-20, "let's try a bigger kick run"). Two
unaugmented MLE continuations of pool8_big_nov15/big (wandb vtba4cf5), weights only from its step-50,000 archive (the
held-out optimum, 42-51k), identical except for the prior file:

  mle_ctrl  the current prior, pool8_nov09/priors/pool8_p1_train_k4_metric_switched.pt (children at 10^U(-2.0, -0.375) A)
  mle_kick  this folder's build, pool8_kick_nov20/priors/pool8_p1_train_k4_metric_loglo-0.675.pt (children at
            10^U(-0.675, -0.375) A; the same anchors, held-out descent groups, K, switched eLJ and forces; README.md)

Each arm = pool8_big_nov15/make.py's config (data, model, GPU settings; switched eLJ through mk_dev) with the MLE stage
kept on the weights load (skip_if removed), its LR held at 0.2 x seed from step 0 (the source's post-hold LR, as
pool8_big2_nov17's aug arms), MLE_STEPS steps, no augmentation. The source checkpoint names the unswitched prior, so
prior_path is ignored in the warm-start problem check (as pool8_smpw_nov19).

Read (one run each, against each other): the overfit gaps within each arm (eval_bwd/mle_heldout_gap, eval_test/mle_gap),
eval_fwd/jensen_z, Excess Energy Nats Per DoF P50, Nonthermal Fraction (equipartition), and -log q on the rows both
arms share: eval_bwd/mle_anchors (the clean training anchors) and eval_test/mle (held-out molecules' crystals). The
arms' own eval_bwd/mle and eval_bwd/mle_heldout_rows are read on different rows and are not comparable across arms.

usage: python configs/pool8_kick_nov20/make.py   (mle_ctrl.yaml, mle_kick.yaml, submit_pool8_kick_nov20.sbatch)
Submit after merge.sbatch has written the kick prior: sbatch submit_pool8_kick_nov20.sbatch (both arms)."""
import copy
import os
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('big2', HERE.parent / 'pool8_big2_nov17' / 'make.py')
big2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(big2)
big = big2.big

TAG = 'pool8_kick_nov20'
SOURCE_STEP = 50000
MLE_STEPS = 15000
PRIORS = {'mle_ctrl': big.PRIOR[:-len('.pt')] + '_switched.pt',
          # big.DATA is .../conditional/pool8_nov09/priors; the kick build writes under .../conditional/pool8_kick_nov20
          'mle_kick': '../../pool8_kick_nov20/priors/pool8_p1_train_k4_metric_loglo-0.675.pt'}


def arm(prior, run_name):
    files = {'prior': os.path.normpath(f'{big.DATA}/{prior}').replace(os.sep, '/'), 'conditions': f'{big.DATA}/{big.CONDITIONS}', 'test': f'{big.DATA}/{big.TEST}'}
    cfg = copy.deepcopy(big.config(files))
    assert cfg['energy_config']['analyze_kwargs'].get('lj_envelope') == 1.0, cfg['energy_config']['analyze_kwargs']
    nf = cfg['model']['nf']
    assert not nf.get('rotate_frames') and not nf.get('mirror_frames'), (nf.get('rotate_frames'), nf.get('mirror_frames'))
    stage = cfg['protocols'][cfg['protocol']]['stages'][0]
    assert stage['name'] == 'train_prior' and stage.pop('skip_if') == 'weights_loaded'
    stage['lr_schedule'] = [{'to': 0.2, 'steps': 0, 'kind': 'hold'}]
    stage['max_steps'] = MLE_STEPS
    cfg.update(checkpoint_name=big2.source_name(SOURCE_STEP), load_weights_only=True, continue_from_checkpoint=False,
               warm_start_ignore_problem_keys=['prior_path'], archive_period=2500, epochs=MLE_STEPS,
               run_name=run_name, tag=TAG)
    return cfg


def arms():
    out = {name: arm(prior, name) for name, prior in PRIORS.items()}
    a, b = (copy.deepcopy(out[k]) for k in PRIORS)
    for c in (a, b):
        c.pop('prior_path'), c.pop('run_name')
    assert a == b, 'the two arms may differ only in prior_path and run_name'
    return out


def sbatch(names):
    text = (HERE / 'submit_template.sbatch').read_text(encoding='utf-8')
    files = (*PRIORS.values(), big.CONDITIONS, big.TEST)
    return (text.replace('@TAG@', TAG).replace('@ARMS@', ' '.join(names)).replace('@NARMS@', str(len(names) - 1))
            .replace('@DATA@', big.DATA).replace('@FILES@', ' '.join(files)))


if __name__ == '__main__':
    made = arms()
    for name, cfg in made.items():
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(list(made)), encoding='utf-8', newline='\n')
    print(', '.join(made), f'and submit_{TAG}.sbatch in {HERE}')
