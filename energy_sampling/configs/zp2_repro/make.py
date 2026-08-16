"""
Local repro of the two prod0810 Z'>1 arms (5 = acridine sg14 zp2 mace,
6 = acridine sg9 zp2 mace), which failed on the cluster 2026-08-11.

Derives each arm verbatim from configs/prod0810/<i>.yaml and rewrites ONLY:
  - the four cluster paths -> their local D:\ equivalents (verified present)
  - run_name / tag -> zp2repro_* so nothing collides with a real run identity
  - checkpoint_read_only: true, checkpoints_dir -> scratchpad (no writes to
    checkpoints/ at all; see checkpointing.py:94)
  - batch/epoch/eval cadence shrunk to laptop scale

Everything that could plausibly carry the failure -- z_primes, space_groups,
model layout, protocol, buffers, energy_config -- is left untouched.

Arm 4 (sg14 ZP1, same molecule + same energy + same everything else) is emitted
as the control: if 4 runs and 5/6 don't, the break is in the Z'>1 path itself
and not in the acridine/mace setup.
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, os.pardir, 'prod0810')
SCRATCH = (r'C:\Users\mikem\AppData\Local\Temp\claude'
           r'\C--Users-mikem-Projects-mxt-gfn-gfn-diffusion-energy-sampling'
           r'\fda8ba2a-5670-466f-98e8-0ab7f5a0151d\scratchpad\ckpt')

PRIORS = r'D:\crystal_datasets\conditional\priors'
MLIP = r'D:\crystal_datasets\acr_112025_mh1_stagetwo.model'

ARMS = {
    4: ('zp1_ctrl', 'acridine_sg14_zp1_mace_prior_dataset.pt'),
    5: ('zp2_sg14', 'acridine_sg14_zp2_mace_prior_dataset.pt'),
    6: ('zp2_sg9', 'acridine_sg9_zp2_mace_prior_dataset.pt'),
}


def build(index, name, dataset):
    with open(os.path.join(SRC, f'{index}.yaml')) as f:
        cfg = yaml.safe_load(f)

    cfg['run_name'] = f'zp2repro_{name}'
    cfg['tag'] = 'zp2repro'

    cfg['mlip_path'] = MLIP
    cfg['prior_path'] = os.path.join(PRIORS, dataset)
    cfg['molecules_path'] = os.path.join(PRIORS, dataset)
    cfg['checkpoints_dir'] = SCRATCH + os.sep
    cfg['checkpoint_read_only'] = True

    # laptop scale -- a startup/shape failure fires long before any of this matters
    cfg['batch_size'] = 100
    cfg['grow_batch_size'] = False
    cfg['max_batch_size'] = 100
    cfg['epochs'] = 60
    cfg['eval_period'] = 25
    cfg['figs_period'] = 50
    cfg['eval_num_samples'] = 200
    cfg['archive_period'] = 100000
    cfg['ray_calibration']['enabled'] = False

    out = os.path.join(HERE, f'{name}.yaml')
    with open(out, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return out


if __name__ == '__main__':
    for index, (name, dataset) in ARMS.items():
        path = build(index, name, dataset)
        assert os.path.exists(os.path.join(PRIORS, dataset)), dataset
        print('wrote', path)
    assert os.path.exists(MLIP), MLIP
    os.makedirs(SCRATCH, exist_ok=True)
