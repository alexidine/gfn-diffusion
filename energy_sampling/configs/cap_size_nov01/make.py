"""cap_size_nov01: conditional NF MLE (train_prior alone) on the rc_oct22 all-family anchors, crossing crystals per
molecule (2, 4, all ~10: build_rc_prior.thin_rows of rcoct22i1_prior.pt) with model size (W768 L8 H12, 62M; W1024 L12 H16).
The recipe of the local run cond_oct31_big_allk2_w768 (W&B zo2jwtz4): the COMMITTED mk_dev, make_condanchor's globals
(batch 5000 pinned, buffer_device cpu, anchors only, clip 10, LR hold 1.0, lr_flow 1e-4), T 6.9. Every arm runs ~30-37
passes over its rows and evaluates every ~1.8 passes. Held-out molecules: rcoct22i1_qm9rc_test_conditions.pt (6,652; the
all-family held-out file is not on the cluster). Two jobs, one A100 each, three arms per card under MPS at 0.30 each.
The cross-condition matrix (xcond_eval) runs every 2 evals. The first launch carried tag 'dev' and xcond off; its w768 k2/k4
arms died loading their prior (thin_prior.py) and are relaunched as array index 2 with this file's settings.
usage: python configs/cap_size_nov01/make.py   (writes <arm>.yaml beside this file)"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = '/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling'
DATA = '/scratch/mk8347/data/crystal_datasets/conditional/priors/rc_oct22'
PRIORS = {2: f'{DATA}/caps_nov01/rcoct22i1_cap2_prior.pt', 4: f'{DATA}/caps_nov01/rcoct22i1_cap4_prior.pt',
          10: f'{DATA}/rcoct22i1_prior.pt'}
ROWS = {2: 1363104, 4: 2726000, 10: 6675860}   # thin_rows(k) of the 6,675,860 rows (cap 4 approximate); 10 = uncapped
STEPS = {2: 10000, 4: 20000, 10: 40000}          # 36.7 / 36.7 / 30.0 passes at batch 5000
EVAL = {2: 500, 4: 1000, 10: 2000}               # 1.8 / 1.8 / 1.5 passes per eval
ARCHIVE = {2: 2000, 4: 4000, 10: 8000}           # five weight-only archives per arm
MODELS = {'w768': dict(tf_width=768, tf_layers=8, tf_heads=12), 'w1024': dict(tf_width=1024, tf_layers=12, tf_heads=16)}
GLOBALS = {
    'batch_size': 5000, 'grow_batch_size': False, 'batch_util_target': 0.0, 'buffer_device': 'cpu',
    'buffers.replay_buffer.churn_rate': 0, 'buffers.replay_buffer.mean_residence_steps': 1200,
    'buffers.replay_buffer.max_size': 150000, 'buffers.replay_buffer.val_frac': 0.0,
    'buffers.replay_buffer.admit_from_eval': False, 'buffers.replay_buffer.prioritise.enabled': False,
    'buffers.anchor_buffer.frozen': False, 'buffers.anchor_buffer.thin_every_n_evals': 0,
    'buffers.anchor_buffer.refresh_every_n_evals': 0, 'buffers.anchor_buffer.topup_admit_record_breakers': True,
    'buffers.prior_buffer.source': 'anchors', 'condition_log_z.rollout_condition_draw': 'cycle',
    'z_calibration.fill_threshold': 0.0, 'z_calibration.fill_from_eval': 'off',
    'model.nf.context_dim': 256, 'model.nf.context_hidden': 1024,
}


def put(cfg, dotted, value):
    *head, last = dotted.split('.')
    node = cfg
    for k in head:
        node = node[k]
    node[last] = value


def arm_config(model, cap):
    cfg = yaml.safe_load((HERE.parent / 'mk_dev.yaml').read_text(encoding='utf-8'))
    for k, v in GLOBALS.items():
        put(cfg, k, v)
    for k, v in MODELS[model].items():
        put(cfg, f'model.nf.{k}', v)
    cfg.update(run_name=f'cap_nov01_{model}_k{cap}', tag='cap_size_nov01', cuda_memory_fraction=0.30, checkpoints_dir=f'{ROOT}/checkpoints',
               checkpoint_read_only=False, archive_period=ARCHIVE[cap], archive_buffers=False, gradient_norm_clip=10.0,
               figs_every_n_evals=0, embedding_conditioning=True, lr_flow=1.0e-4, epochs=STEPS[cap],
               eval_period=EVAL[cap], compile_policy=False, protocol='conditional_vargrad', prior_path=PRIORS[cap],
               molecules_path=f'{DATA}/rcoct22i1_conditions.pt',
               test_molecules_path=f'{DATA}/rcoct22i1_qm9rc_test_conditions.pt')
    cfg['energy_config']['temperature'] = 6.9
    cfg['harvest']['enabled'] = False
    cfg['model']['kind'] = 'nf'
    cfg['xcond_eval'].update(enabled=True, every_n_evals=2)   # NF: one exact pass per cell, no energy calls
    cfg['noised_prior'].update(train_kinds=['anchor'], live_kernel=None)
    cfg['lr_control']['schedule'] = [{'kind': 'hold', 'to': 1.0, 'steps': 1000}]
    st = cfg['protocols']['conditional_vargrad']['stages'][0]
    st.pop('exit', None)
    st['max_steps'] = STEPS[cap]
    st['on_exit'] = ['snapshot:phase1_exit', 'stop']
    cfg['protocols']['conditional_vargrad']['stages'] = [st]
    return cfg


if __name__ == '__main__':
    for model in MODELS:
        for cap in PRIORS:
            cfg = arm_config(model, cap)
            out = HERE / f"{cfg['run_name']}.yaml"
            out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
            print(out.name, f'rows {ROWS[cap]:,} steps {STEPS[cap]:,} passes {STEPS[cap] * 5000 / ROWS[cap]:.1f}')
