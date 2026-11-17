"""pool8_tbrate_nov17: what sets the pace of TB on pool8. Owner 2026-11-17, after pool8_big2_nov17_tb (wandb dh0y8yvg):
once its log Z head reached the forward level (~25k steps) the policy's measures improved, but slowly (held-out
within-molecule spread ~0.05 nats per 1k steps). Four short arms from the same source, each the pool8_big2_nov17 'tb' arm
(make.py there: tb_config) with bootstrap_z:offset added to var_conditioning's on_enter, so no arm spends its start
walking the head in, and ONE change each:

  control   none
  lr4       the stage LR hold 0.8 x seed instead of 0.2 (policy LR 1.0e-4 instead of 2.5e-5); lr_flow unchanged
  fresh5    fwd_rollout_every 5 instead of 20 (replay rows four times fresher, four times the energy calls)
  cmean     fwd pooled_center 'mean' (mk_dev's) instead of 'head'

Source: pool8_big_nov15_big step 50,000 (weights only; curves and archives continue from 50,000). STEPS steps each.
Read, per 1k steps against control over the same steps: eval_test/logw_std_within, Excess Energy Nats P50,
eval_bwd/jensen_z - eval_fwd/jensen_z; fresh5 also per energy call (rollout/n).

usage: python configs/pool8_tbrate_nov17/make.py            (arm yamls + submit_pool8_tbrate_nov17.sbatch, an array)
       python configs/pool8_tbrate_nov17/make.py --local DIR --source CKPT --ckdir DIR [--steps N]
                                                            (local smoke configs on the norbornane files)"""
import argparse
import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('big2', HERE.parent / 'pool8_big2_nov17' / 'make.py')
big2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(big2)
big = big2.big
nov10 = big2.nov10

TAG = 'pool8_tbrate_nov17'
SOURCE_STEP = 50000
STEPS = 10000


def _stage(cfg):
    stages = cfg['protocols'][cfg['protocol']]['stages']
    vc = [s for s in stages if s['name'] == 'var_conditioning']
    assert len(vc) == 1, [s['name'] for s in stages]
    return vc[0]


def control(base, source):
    cfg = big2.tb_config(base, source)
    vc = _stage(cfg)
    assert 'bootstrap_z' not in ' '.join(vc['on_enter']), vc['on_enter']
    vc['on_enter'] = vc['on_enter'] + ['bootstrap_z:offset']
    cfg.update(epochs=STEPS, archive_period=5000)
    return cfg


def arms(files, source):
    base = big.config(files)
    out = {'control': control(base, source)}

    lr4 = copy.deepcopy(out['control'])
    vc = _stage(lr4)
    assert vc['lr_schedule'] == [{'to': 0.2, 'steps': 0, 'kind': 'hold'}], vc['lr_schedule']
    vc['lr_schedule'] = [{'to': 0.8, 'steps': 0, 'kind': 'hold'}]
    out['lr4'] = lr4

    fresh5 = copy.deepcopy(out['control'])
    vc = _stage(fresh5)
    assert vc['fwd_rollout_every'] == 20, vc['fwd_rollout_every']
    vc['fwd_rollout_every'] = 5
    out['fresh5'] = fresh5

    cmean = copy.deepcopy(out['control'])
    vc = _stage(cmean)
    assert vc['loss_coeffs']['fwd']['pooled_center'] == 'head'
    vc['loss_coeffs']['fwd']['pooled_center'] = 'mean'
    out['cmean'] = cmean

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
    p.add_argument('--source', default=None, help='local smoke: checkpoint file name (in --ckdir) to load')
    p.add_argument('--ckdir', default=None, help='local smoke: checkpoints directory (reads the source, writes there)')
    p.add_argument('--steps', type=int, default=60)
    a = p.parse_args()
    if a.local:
        assert a.source and a.ckdir, '--local needs --source and --ckdir'
        out = Path(a.local)
        out.mkdir(parents=True, exist_ok=True)
        for name, cfg in arms(nov10.LOCAL, a.source).items():
            cfg.update(checkpoints_dir=a.ckdir, tag=f'{TAG}_local', smoke=True, cuda_memory_fraction=0.5,
                       epochs=a.steps, batch_size=min(cfg['batch_size'], 64), eval_period=max(30, a.steps // 2),
                       eval_num_samples=2000, test_eval_num_samples=500, archive_period=10 ** 6)
            (out / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
        print(f'local smoke configs in {out}')
    else:
        files = {'prior': f'{big.DATA}/{big.PRIOR}', 'conditions': f'{big.DATA}/{big.CONDITIONS}',
                 'test': f'{big.DATA}/{big.TEST}'}
        made = arms(files, big2.source_name(SOURCE_STEP))
        for name, cfg in made.items():
            (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
        (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(list(made)), encoding='utf-8', newline='\n')
        print(', '.join(made), f'and submit_{TAG}.sbatch in {HERE}')
