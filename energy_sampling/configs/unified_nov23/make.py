"""unified_nov23: the unified crystal training workflow, one arm on the store route.

  unified  pool8sg4_next_nov22 v4_kick's model, prior and MLE stage, with
           store     buffers.store enabled, ram_gb from RAM_GB (also #SBATCH --mem), ram_frac RAM_FRAC, mle_draw anchor,
                     replay_weight uniform
           MLE       live physical kernel on anchors, MLE_STEPS
           TB        fused, TB_STEPS: bwd and replay 50:50, TB against the detached Z(c) head at Huber knees BWD_KNEE and
                     REPLAY_KNEE; Z(c) on the forward sidecar at knee Z_KNEE, FWD_REPEATS rows per condition
           scoring   fwd_rollout_every = max(1, round(2 B R / SCORES_PER_STEP)); each rollout row owes one kick

The base is the committed mk_dev (git show HEAD), whatever the working tree holds.

usage: python configs/unified_nov23/make.py    (unified.yaml, submit_unified_nov23.sbatch)
       python configs/unified_nov23/make.py --local DIR --ckdir CK --batch B --mle-steps M --tb-steps T
                                            [--rollout-every N] [--ram-gb G] [--eval-samples E] [--eval-period P]
                                            (a smoke config on the pool4 test prior)"""
import argparse
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('next_nov22', HERE.parent / 'pool8sg4_next_nov22' / 'make.py')
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

TAG = 'unified_nov23'
ARM = 'unified'
PROTOCOL = 'unified'
RAM_GB = 224                # GiB: #SBATCH --mem and buffers.store.ram_gb
RAM_FRAC = 0.5
SCORES_PER_STEP = 200       # energy calls per train step, half on rollouts and half on kicks
BATCH = 1000
FWD_REPEATS = 2
MLE_STEPS = 150000
TB_STEPS = 100000
BWD_KNEE, REPLAY_KNEE, Z_KNEE = 80.0, 10.0, 10.0
TEST_EVAL_SAMPLES = 1000
LOOP_MARGIN = ref.LOOP_MARGIN
LOCAL_DIR = 'D:/crystal_datasets/cluster_pull_nov22/pool4_test'
LOCAL_FILES = {'prior': f'{LOCAL_DIR}/pool4_test_train_k4_metric.pt',
               'conditions': f'{LOCAL_DIR}/pool4_test_train_conditions.pt',
               'test': f'{LOCAL_DIR}/pool4_test_test_w5.pt'}
LOCAL_RAM_GB = 4.0          # CrystalStore allocates its columns and index at the row limit when it is built


def committed_mk_dev() -> str:
    out = subprocess.run(['git', 'show', 'HEAD:./mk_dev.yaml'], cwd=HERE.parent, capture_output=True, check=True)
    return out.stdout.decode('utf-8')


class _CommittedConfigs:
    """pool8_mle_nov10's HERE.parent: the one file it reads there is the committed mk_dev."""

    def __truediv__(self, name):
        assert name == 'mk_dev.yaml', name
        return SimpleNamespace(read_text=lambda encoding='utf-8': committed_mk_dev())


ref.sg4.big.nov10.HERE = SimpleNamespace(parent=_CommittedConfigs())


def rollout_every(batch, repeats=FWD_REPEATS, scores=SCORES_PER_STEP) -> int:
    return max(1, round(2 * batch * repeats / scores))


def tb_stage(steps, every):
    return {
        'name': 'tb',
        'train_mode': 'fused',
        'bwd_sampling_mode': 'prior',
        'flags': {'update_log_z': True, 'buffers_active': True, 'weighted_bwd_sampling': False,
                  'weighted_condition_sampling': False},
        'on_enter': ['set_lr_flow:1.0e-4', 'bootstrap_z:offset'],
        'lr_schedule': [{'to': 0.2, 'steps': 0, 'kind': 'hold'}],
        'fracs': {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5},
        'min_fracs': {'fwd': 0.0},
        'deactivate_threshold': 0.01,
        'fwd_rollout_every': every,
        'z_pin_rollout_every': 0,
        'fwd_rollout_triggers': {'occupancy_min_batches': 0.0, 'val_gap_max': 0.0, 'drift_std_max': 0.0, 'ess_min': 0.0},
        'fwd_z_sidecar': True,
        'max_steps': steps,
        'on_exit': ['snapshot:tb_exit', 'stop'],
        'loss_coeffs': {
            'fwd': {'tb': 0.0, 'vg_lb': 0.0, 'vg_by_condition': 1.0, 'emp_z': 1.0, 'freeze_policy': 1.0,
                    'repeats': float(FWD_REPEATS), 'beta': Z_KNEE, 'pooled_vg': 0.0, 'tb_z_source': 'learned'},
            'bwd': {'tb': 1.0, 'beta': BWD_KNEE, 'freeze_z': 1.0, 'vg_lb': 0.0, 'condition_block_m': 0, 'repeats': 1.0,
                    'level_gap': 0.0, 'tb_z_source': 'learned'},
            'replay': {'tb': 1.0, 'beta': REPLAY_KNEE, 'freeze_z': 1.0, 'vg_by_condition': 0.0, 'condition_block_m': 0,
                       'tb_z_source': 'learned'},
        },
    }


def unified(files, batch=BATCH, mle_steps=MLE_STEPS, tb_steps=TB_STEPS, every=None, ram_gb=RAM_GB):
    cfg = ref.v4_kick(files)
    every = rollout_every(batch) if every is None else every

    np_cfg = cfg['noised_prior']
    assert np_cfg['live_kernel'] == 'chart' and np_cfg['live_log_range'] == ref.KICK_LAW, np_cfg
    assert np_cfg['train_kinds'] == ['child'] and np_cfg['window_kT'] == 10.0, np_cfg
    assert np_cfg.get('window_by', 'row') == 'row', np_cfg
    np_cfg.update(live_kernel='physical', train_kinds=['anchor'], window_by='row')

    buffers = cfg['buffers']
    assert buffers['store']['enabled'] is False and buffers['store']['ram_gb'] is None, buffers['store']
    assert buffers['anchor_buffer']['frozen'] is True and buffers['anchor_buffer']['seed_floors_only'] is True
    assert buffers['replay_buffer']['admit_from_eval'] is True and buffers['replay_buffer']['val_frac'] == 0.0
    assert buffers['replay_buffer']['prioritise']['enabled'] is False, buffers['replay_buffer']['prioritise']
    buffers['store'] = {'enabled': True, 'ram_gb': ram_gb, 'ram_frac': RAM_FRAC, 'mle_draw': 'anchor',
                        'replay_weight': 'uniform'}

    assert cfg['protocol'] == 'unconditional_tb', cfg['protocol']
    stages = ref._stages(cfg)
    assert len(stages) == 1, [s['name'] for s in stages]
    mle = stages[0]
    assert mle['name'] == 'train_prior' and mle['train_mode'] == 'bwd' and mle['bwd_sampling_mode'] == 'dataset', mle
    assert mle['max_steps'] == ref.STEPS and mle['on_exit'] == ['snapshot:phase1_exit', 'stop'], mle
    mle.update(max_steps=mle_steps, on_exit=['snapshot:phase1_exit'])
    cfg['protocol'] = PROTOCOL
    cfg['protocols'] = {PROTOCOL: {'stages': [mle, tb_stage(tb_steps, every)]}}

    assert cfg['zsg_panel'] == {'enabled': True, 'molecules': 64, 'draws': 32, 'every_n_evals': 4}, cfg['zsg_panel']
    cfg['zsg_panel'] = {'enabled': True, 'molecules': 64}
    assert cfg['archive_buffers'] is False and cfg['test_eval_num_samples'] == 4000
    assert cfg['batch_size'] == cfg['fused_grad_accum_min_samples'] == BATCH and cfg['grow_batch_size'] is False
    cfg.update(tag=TAG, run_name=ARM, batch_size=batch, fused_grad_accum_min_samples=batch,
               test_eval_num_samples=TEST_EVAL_SAMPLES, epochs=mle_steps + tb_steps + LOOP_MARGIN)
    return cfg


def localize(cfg, ckdir, eval_samples, eval_period):
    """A smoke run on the pool4 test prior with every buffer in host RAM and no step archives."""
    cfg.update(checkpoints_dir=ckdir, tag=f'{TAG}_local', smoke=True, cuda_memory_fraction=0.4, buffer_device='cpu',
               eval_num_samples=eval_samples, test_eval_num_samples=min(TEST_EVAL_SAMPLES, eval_samples),
               eval_period=eval_period, archive_period=10 ** 6)
    cfg['zsg_panel']['molecules'] = 8
    return cfg


def sbatch():
    text = (HERE / 'submit_template.sbatch').read_text(encoding='utf-8')
    return (text.replace('@TAG@', TAG).replace('@ARMS@', ARM).replace('@NARMS@', '0')
            .replace('@RAM_GB@', str(RAM_GB)))


def write(cfg, out=HERE):
    (out / f'{ARM}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--local', default=None, help='write the smoke config into this directory')
    p.add_argument('--ckdir', default=None, help='local: checkpoints directory')
    p.add_argument('--batch', type=int, default=None, help='local: batch size')
    p.add_argument('--mle-steps', type=int, default=None, help='local: MLE stage steps')
    p.add_argument('--tb-steps', type=int, default=None, help='local: TB stage steps')
    p.add_argument('--rollout-every', type=int, default=None, help='local: fwd_rollout_every (default from the budget)')
    p.add_argument('--ram-gb', type=float, default=LOCAL_RAM_GB, help='local: buffers.store.ram_gb')
    p.add_argument('--eval-samples', type=int, default=None, help='local: eval_num_samples (default 10 x batch)')
    p.add_argument('--eval-period', type=int, default=None, help='local: eval_period (default a quarter of the steps)')
    a = p.parse_args()
    if a.local:
        assert a.ckdir and a.batch and a.mle_steps and a.tb_steps, '--local needs --ckdir, --batch, --mle-steps, --tb-steps'
        for path in (a.local, a.ckdir):
            assert not path.upper().startswith(('D:', '/D/')), f'local outputs stay off D: ({path})'
        out = Path(a.local)
        out.mkdir(parents=True, exist_ok=True)
        c = unified(LOCAL_FILES, batch=a.batch, mle_steps=a.mle_steps, tb_steps=a.tb_steps, every=a.rollout_every,
                    ram_gb=a.ram_gb)
        c = localize(c, a.ckdir, a.eval_samples or 10 * a.batch,
                     a.eval_period or max(10, (a.mle_steps + a.tb_steps) // 4))
        write(c, out)
        print(f'{out / (ARM + ".yaml")}: batch {a.batch}, MLE {a.mle_steps}, TB {a.tb_steps}, '
              f'fwd_rollout_every {ref._stages(c)[1]["fwd_rollout_every"]}, ram_gb {a.ram_gb}')
    else:
        write(unified({'prior': f'{ref.sg4.DATA}/{ref.sg4.PRIOR}', 'conditions': f'{ref.sg4.DATA}/{ref.sg4.CONDITIONS}',
                       'test': f'{ref.sg4.DATA}/{ref.sg4.TEST}'}))
        (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(), encoding='utf-8', newline='\n')
        print(f'{ARM}.yaml and submit_{TAG}.sbatch in {HERE}')
