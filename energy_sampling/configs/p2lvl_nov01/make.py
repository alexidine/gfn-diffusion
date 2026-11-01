"""p2lvl_nov01: phase 2 (var_conditioning) from one phase-1 checkpoint, comparing the level the pooled term scores its
rows against (fwd_loss_coeffs.pooled_center). Source: the step-6000 weights of cap_size_nov01's finished w768 cap-2 arm
(W&B 89vq5jmq, run dev_cap_nov01_w768_k2; held-out NLL -9.40 there, its minimum -9.56 at step 5500). Phase 1 is skipped
by the weights-only load (train_prior skip_if: weights_loaded), so var_conditioning starts at step 0 on the same
1,363,104 rows, held static (buffers.prior_buffer.static: no in-run noising), 2 per molecule.

  pooledvg      pooled_center mean          the committed term: the 4-row group mean, in the graph (VarGrad)
  tbhead        pooled_center head          per-row TB against the detached Z(c) head
  tbpool        pooled_center head_stored   per-row TB against the detached 1/2 head + 1/2 stored-row mean
  pooledvg_s2   pooled_center mean, seed 12346: the noise reference for pooledvg

Everything else is the committed mk_dev's var_conditioning on top of cap_size_nov01's arm settings, except: batch 2500
(1,250 molecules x (2 replay + 2 stored) = 5,000 rows a step, phase 1's row count; 5,000-crystal rollouts), no
bootstrap_z on entry (no condition is ever trusted at 681k conditions), replay max_size 200,000, eval every 250 steps,
mk_dev's LR schedule (0.05 x for 3,000 steps, then 0.2 x over 1,000). Two arms per A100 under MPS at 0.45 each.
usage: python configs/p2lvl_nov01/make.py   (writes <arm>.yaml beside this file)"""
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('cap_make', HERE.parent / 'cap_size_nov01' / 'make.py')
cap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cap)

SOURCE = 'dev_cap_nov01_w768_k2_elj-rcoct22i1_cap2_prior-T6.9-d7ef7b_step6000.pt'   # in checkpoints_dir
STEPS = 20000
ARMS = {'pooledvg': ('mean', 12345), 'tbhead': ('head', 12345), 'tbpool': ('head_stored', 12345),
        'pooledvg_s2': ('mean', 12346)}


def arm_config(arm):
    center, seed = ARMS[arm]
    cfg = yaml.safe_load((HERE.parent / 'mk_dev.yaml').read_text(encoding='utf-8'))
    for k, v in cap.GLOBALS.items():
        cap.put(cfg, k, v)
    for k, v in cap.MODELS['w768'].items():
        cap.put(cfg, f'model.nf.{k}', v)
    cap.put(cfg, 'batch_size', 2500)
    cap.put(cfg, 'buffers.replay_buffer.max_size', 200000)
    cap.put(cfg, 'buffers.anchor_buffer.frozen', True)
    cfg.update(run_name=arm, tag='p2lvl_nov01', seed=seed, cuda_memory_fraction=0.45,
               checkpoints_dir=f'{cap.ROOT}/checkpoints', checkpoint_read_only=False,
               checkpoint_name=SOURCE, load_weights_only=True,
               archive_period=2000, archive_buffers=False, gradient_norm_clip=10.0, figs_every_n_evals=0,
               embedding_conditioning=True, lr_flow=1.0e-4, epochs=STEPS, eval_period=250, compile_policy=False,
               protocol='conditional_vargrad', prior_path=cap.PRIORS[2],
               molecules_path=f'{cap.DATA}/rcoct22i1_conditions.pt',
               test_molecules_path=f'{cap.DATA}/rcoct22i1_qm9rc_test_conditions.pt')
    cfg['energy_config']['temperature'] = 6.9
    cfg['harvest']['enabled'] = False
    cfg['model']['kind'] = 'nf'
    cfg['xcond_eval'].update(enabled=True, every_n_evals=2)
    cfg['noised_prior'].update(train_kinds=['anchor'], live_kernel=None)
    stages = cfg['protocols']['conditional_vargrad']['stages']
    assert [s['name'] for s in stages] == ['train_prior', 'var_conditioning'], [s['name'] for s in stages]
    vc = stages[1]
    vc['on_enter'] = [a for a in vc['on_enter'] if a != 'bootstrap_z']
    vc['loss_coeffs']['fwd']['pooled_center'] = center
    return cfg


if __name__ == '__main__':
    for arm in ARMS:
        cfg = arm_config(arm)
        out = HERE / f'p2lvl_nov01_{arm}.yaml'   # train.py names the run (W&B, checkpoints) tag_run_name
        out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
        print(out.name, 'pooled_center', ARMS[arm][0], 'seed', ARMS[arm][1])
