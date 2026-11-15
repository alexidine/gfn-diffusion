"""pool8_big_nov15: one unaugmented conditional MLE run on the pool8 P-1 noised set (K = 4 children per minimum), with a
larger NF model and usage-targeted batch growth. Stage 0 of the rotation recipe: its checkpoints are the warm start for
augmented training later (rotate_frames + orient_grad_balance), once the encoder reads pose.

Base: configs/pool8_mle_nov10/make.py's base (the committed mk_dev, NF policy, molecule conditioning, prefix FD rows,
the prior's training children inside a 10 kT window, mk_dev's LR schedule, MLE only, then stop), changed as follows:
  model     split atom encoder (the encoder the rotation recipe was tested on), tf_width 256 -> 512,
            tf_layers 4 -> 8, atom_node_dim 128 -> 256, atom_convs 3 -> 4, context_dim 128 -> 256
  batch     batch_size 1000 (= fused_grad_accum_min_samples), grow_batch_size true, batch_util_target 0.7: the sizer
            climbs only as far as the GPU occupancy target needs (train.py select_batch_size), which keeps the card
            above Torch's low-utilisation cancellation
  card      cuda_memory_fraction 0.97 (one run per card)
  length    STEPS is a cap; the 2-day time limit ends the run first. Archives every ARCHIVE steps.
Readouts: eval_test/mle and eval_test/mle_gap (held-out molecules), eval_bwd/mle_heldout_gap (held-out crystals).

usage: python configs/pool8_big_nov15/make.py            (writes big.yaml and submit_pool8_big_nov15.sbatch here)
       python configs/pool8_big_nov15/make.py --local DIR (the same run on the local norbornane files, as a smoke)"""
import argparse
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('nov10', HERE.parent / 'pool8_mle_nov10' / 'make.py')
nov10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nov10)

TAG = 'pool8_big_nov15'
STEPS = 400000
ARCHIVE = 10000
ARM = 'big'
DATA, PRIOR, CONDITIONS, TEST = nov10.DATA, nov10.PRIOR, nov10.CONDITIONS, nov10.TEST
BIG = {
    'model.nf.atom_tokens': 'split',
    'model.nf.tf_width': 512, 'model.nf.tf_layers': 8,
    'model.nf.atom_node_dim': 256, 'model.nf.atom_convs': 4, 'model.nf.context_dim': 256,
    'batch_size': 1000, 'grow_batch_size': True, 'batch_util_target': 0.7,
    'cuda_memory_fraction': 0.97,
    'epochs': STEPS, 'archive_period': ARCHIVE, 'tag': TAG,
}


def config(files):
    nov10.ARMS[ARM] = ({}, 'the pool8_big_nov15 run')
    cfg = nov10.arm_config(ARM, files)
    for k, v in BIG.items():
        nov10.put(cfg, k, v)
    stage = cfg['protocols'][cfg['protocol']]['stages'][0]
    stage['max_steps'] = int(cfg['epochs'])
    return cfg


def sbatch():
    template = (HERE / 'submit_template.sbatch').read_text(encoding='utf-8')
    files = (PRIOR, CONDITIONS, TEST)
    return (template.replace('@TAG@', TAG).replace('@GROUPS_DOC@', f'#   {ARM}: one run (make.py docstring)')
            .replace('@DATA@', DATA).replace('@CASES@', ARM).replace('@FILES@', ' '.join(files))
            .replace('@BYTES@', ' '.join('-1' for _ in files)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--local', default=None, help='write the run on the local norbornane files into this directory')
    a = p.parse_args()
    if a.local:
        out = Path(a.local)
        out.mkdir(parents=True, exist_ok=True)
        cfg = config(nov10.LOCAL)
        cfg.update(checkpoints_dir='D:/crystal_datasets/gfn_checkpoints', tag=f'{TAG}_local', smoke=True,
                   cuda_memory_fraction=0.9, epochs=400)
        cfg['protocols'][cfg['protocol']]['stages'][0]['max_steps'] = 400
        (out / f'{ARM}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
        print(f'local smoke config in {out / (ARM + ".yaml")}')
    else:
        files = {'prior': f'{DATA}/{PRIOR}', 'conditions': f'{DATA}/{CONDITIONS}', 'test': f'{DATA}/{TEST}'}
        (HERE / f'{ARM}.yaml').write_text(yaml.safe_dump(config(files), sort_keys=False), encoding='utf-8',
                                          newline='\n')
        (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(), encoding='utf-8', newline='\n')
        print(f'{ARM}.yaml and submit_{TAG}.sbatch in {HERE}')
