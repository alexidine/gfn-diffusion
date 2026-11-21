"""pool8_smooth_nov21: conditional MLE on the pool8 P-1 eLJ set, exploring three smoothing levers (owner 2026-11-21: "MLE,
vector embeddings only (no atomwise cross attention), buffer noising (efficient live kernel), condition embedding
noising; maybe 8 12 hour runs", here nine). Fresh policies: an architecture without cross-attention cannot warm-start from the
big run's weights, so every arm starts from scratch for a like-for-like comparison.

Every arm = pool8_big_nov15/make.py's config on the switched prior, one batch size (BATCH, no growth), MLE_STEPS steps:

  x_fixed         atomwise cross-attention, the stored K=4 children (the reference)
  x_kick          cross-attention, a fresh chart kick of each row's parent anchor every draw
  v_fixed         vector embeddings only (atom_tokens split_pooled), the stored children
  v_kick          vector only, fresh kick
  v_kick_bn       vector only, fresh kick, the condition bottleneck with no noise (the control for the noise arms)
  v_noise05       vector only, stored children, condition noise 0.5
  v_kick_noise05  vector only, fresh kick, condition noise 0.5
  v_kick_noise10  vector only, fresh kick, condition noise 1.0
  v_wide_noise05  vector only, fresh kick at 10^U(-1, -0.2) (0.10-0.63 A), condition noise 0.5

Fresh kick: noised_prior.live_kernel 'chart', at the prior's own radius law 10^U(-2, -0.375) A unless the arm says
otherwise, centred on the row's parent anchor, or on the row itself when it lies across a chart wall from its parent. Condition noise: model.nf.cond_bottleneck BOTTLENECK with cond_noise sigma on each pooled molecule vector, in
training only, at most BOTTLENECK x 1/2 log(1 + 1/sigma^2) nats per vector (sigma 0.5: 25.8; sigma 1: 11.1, about
log of the 68k training molecules). The prefix row features stay noise-free.

All arms keep train_kinds ['child'], so the stored rows, the held-out rows and every density readout are the same rows
in all nine. A fresh-kick arm never trains on the stored children, so its eval_bwd/mle reads rows it has not seen.

Read at matched steps: eval_bwd/mle_heldout_rows, eval_test/mle, eval_bwd/mle_anchors, their gaps, eval_fwd/jensen_z,
eval_test/jensen_z, Excess Energy Nats Per DoF P50 (every arm has the same floors: one prior).

usage: python configs/pool8_smooth_nov21/make.py   (nine arm yamls, submit_pool8_smooth_nov21.sbatch)"""
import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('big', HERE.parent / 'pool8_big_nov15' / 'make.py')
big = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(big)

TAG = 'pool8_smooth_nov21'
PRIOR = big.PRIOR[:-len('.pt')] + '_switched.pt'
BATCH = 1400
MLE_STEPS = 60000
BOTTLENECK = 32
WIDE = [-1.0, -0.2]
#: name -> (atom_tokens, fresh kick, radius law or None for the prior's, condition noise or None)
ARMS = {
    'x_fixed': ('split', False, None, None),
    'x_kick': ('split', True, None, None),
    'v_fixed': ('split_pooled', False, None, None),
    'v_kick': ('split_pooled', True, None, None),
    'v_kick_bn': ('split_pooled', True, None, 0.0),
    'v_noise05': ('split_pooled', False, None, 0.5),
    'v_kick_noise05': ('split_pooled', True, None, 0.5),
    'v_kick_noise10': ('split_pooled', True, None, 1.0),
    'v_wide_noise05': ('split_pooled', True, WIDE, 0.5),
}


def arm(name, files):
    tokens, kick, law, noise = ARMS[name]
    cfg = copy.deepcopy(big.config(files))
    assert cfg['energy_config']['analyze_kwargs'].get('lj_envelope') == 1.0, cfg['energy_config']['analyze_kwargs']
    nf = cfg['model']['nf']
    assert nf['atom_tokens'] == 'split' and nf['row_tokens'] == 'prefix', (nf['atom_tokens'], nf['row_tokens'])
    assert not nf.get('rotate_frames') and not nf.get('mirror_frames')
    nf['atom_tokens'] = tokens
    if noise is not None:
        nf.update(cond_bottleneck=BOTTLENECK, cond_noise=float(noise))
    np_cfg = cfg['noised_prior']
    assert np_cfg['train_kinds'] == ['child'] and np_cfg['live_kernel'] is None
    if kick:
        np_cfg['live_kernel'] = 'chart'
        if law is not None:
            np_cfg['live_log_range'] = list(law)
    stage = cfg['protocols'][cfg['protocol']]['stages'][0]
    assert stage['name'] == 'train_prior'
    stage['max_steps'] = MLE_STEPS
    cfg.update(batch_size=BATCH, grow_batch_size=False, batch_util_target=0.0, epochs=MLE_STEPS, archive_period=5000, eval_period=500,
               run_name=name, tag=TAG)
    return cfg


def sbatch(names):
    text = (HERE / 'submit_template.sbatch').read_text(encoding='utf-8')
    return (text.replace('@TAG@', TAG).replace('@ARMS@', ' '.join(names)).replace('@NARMS@', str(len(names) - 1))
            .replace('@DATA@', big.DATA).replace('@FILES@', ' '.join((PRIOR, big.CONDITIONS, big.TEST))))


if __name__ == '__main__':
    files = {'prior': f'{big.DATA}/{PRIOR}', 'conditions': f'{big.DATA}/{big.CONDITIONS}', 'test': f'{big.DATA}/{big.TEST}'}
    for name in ARMS:
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(arm(name, files), sort_keys=False), encoding='utf-8', newline='\n')
    (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(list(ARMS)), encoding='utf-8', newline='\n')
    print(', '.join(ARMS), f'and submit_{TAG}.sbatch in {HERE}')
