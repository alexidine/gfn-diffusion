"""pool8_tbrate_nov17: what sets the pace of TB on pool8. Owner 2026-11-17, after pool8_big2_nov17_tb (wandb dh0y8yvg):
once its log Z head reached the forward level (~25k steps) the policy's measures improved, but slowly (held-out
within-molecule spread ~0.05 nats per 1k steps). Four short arms from the same source, each the pool8_big2_nov17 'tb' arm
(make.py there: tb_config) with bootstrap_z:offset added to var_conditioning's on_enter, so no arm spends its start
walking the head in, and ONE change each:

  control   none
  lr4       the stage LR hold 0.8 x seed instead of 0.2 (policy LR 1.0e-4 instead of 2.5e-5); lr_flow unchanged
  fresh5    fwd_rollout_every 5 instead of 20 (replay rows four times fresher, four times the energy calls)
  cmean     fwd pooled_center 'mean' (mk_dev's) instead of 'head'

Added 2026-11-18, after only fresh5 helped: control and fresh5 differ in replay row AGE (mean 1,087 vs 279 steps) and
in REUSE (each row drawn ~20x vs ~5x), at the same 50k rows. Two arms complete the 2 x 2 in rollout rate x row age:
  young     control with replay mean_residence_steps YOUNG_TAU: control's rate and reuse, fresh5's age, ~1/4 the rows
  bigbuf    fresh5 with replay max_size BIGBUF_ROWS: fresh5's rate and reuse, control's age, ~4x the rows
Age effect: young - control and fresh5 - bigbuf. Reuse effect: bigbuf - control and fresh5 - young. Submit only these
two: sbatch --array=4,5 submit_pool8_tbrate_nov17.sbatch.

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
YOUNG_TAU = 280         # fresh5's measured mean replay row age (replay_buffer_mean_age 279)
BIGBUF_ROWS = 200000    # 4 x 50k: fresh5's admissions held for control's tau 1200 without overflow eviction


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

    # the 2 x 2 in rollout rate x replay row age (owner 2026-11-18, after fresh5 alone helped): buffer size = admission
    # rate x row age (Little), reuse per row = replay draws per step / admissions per step, so control (every 20, tau
    # 1200) and fresh5 (every 5, the 50k cap evicting at random: mean age ~280) differ in BOTH age and reuse
    young = copy.deepcopy(out['control'])      # control's rate and reuse, fresh5's age (~1/4 the rows)
    rb = young['buffers']['replay_buffer']
    assert rb['mean_residence_steps'] == 1200 and rb['max_size'] == 50000, rb
    rb['mean_residence_steps'] = YOUNG_TAU
    out['young'] = young

    bigbuf = copy.deepcopy(out['fresh5'])      # fresh5's rate and reuse, control's age (~4x the rows)
    bigbuf['buffers']['replay_buffer']['max_size'] = BIGBUF_ROWS
    out['bigbuf'] = bigbuf

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
