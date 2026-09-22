"""force_sep18d -- the T=100 leg from the TIGHTLY CONVERGED MLE checkpoint, LOCAL (RTX 5080).

The production entry point: mle09_mip_lr4p0's best checkpoint (the seed of p12_mip_lr1),
T=100, still in the train_prior stage. force_sep18 (T=10, pg14_p1_cont at 80k MLE steps)
measured the search regime; force_sep18c (the converged TB archive) measures flight from
basin floors. This leg watches one run pass from search into flight from a real production
start, which is what a schedule has to detect.

Same shape as force_sep18c: forward seat (fwd 0.30 / bwd 0.50 / replay 0.20, rollout every
step, fwd trains the policy), P_B frozen from the MLE weights at load, prior-buffer churn from
noised anchors only, batch 400. The dose16 base's train_prior stub already carries the
always-true exit (bwd/mle above -1e9), so the run exits MLE at its first eval, enters
equilibration (rebuild_prior_by_churn from anchors, bootstrap_z) and the seat takes over.

  ctrl   reward_grads 0                          the seat alone
  raw    k 1, reward_grads 1, no gate, no clip   dose16_fs_k1's term
  g      k 1, gate 1.0, no clip                  the search-phase winner
  gc     k 1, gate 1.0, |F| clip 250             the reshaped form

IDENTITY as force_sep18c: prior_path keeps the cluster string (mirrored under C:\scratch),
molecules_path local, hash checked against the seed's stored problem_hash at generation.
The seed file is resolved by glob at generation time (`*mle09_mip_lr4p0_*_best.pt` in
D:\crystal_datasets\checkpoints); its modeller_state.step_ind sets the resume step.
"""
import glob
import os
import sys
from argparse import Namespace
from pathlib import Path

import yaml

HERE = Path(__file__).parent
ES = HERE.parent.parent
BASE = HERE.parent / 'dose_sep16' / 'dose16_fs_k1.yaml'
TAG = 'fs18d'
CKPT_DIR = 'D:\\crystal_datasets\\checkpoints\\'
SEED_GLOB = 'D:/crystal_datasets/checkpoints/*mle09_mip_lr4p0_*_best.pt'
STUB_STEPS = 500     # the base's eval_period; the stub exit fires at the first eval
STEPS = 3000
BATCH = 400
LOCAL_MOLECULES = 'D:\\crystal_datasets\\conditional\\priors\\mipcas_sg2_zp1_elj_200k_prior_dataset.pt'
FORCE_CLIP = 250.0


def _ns(obj):
    if isinstance(obj, dict):
        return Namespace(**{k: _ns(v) for k, v in obj.items()})
    return obj


def problem_hash_of(cfg):
    sys.path.insert(0, str(ES.parent))
    from energy_sampling.utils import get_problem_definition, normalize_problem_def, problem_hash
    return problem_hash(normalize_problem_def(get_problem_definition(_ns(cfg))))


def resolve_seed():
    import torch
    hits = sorted(glob.glob(SEED_GLOB))
    assert len(hits) == 1, f'expected one seed matching {SEED_GLOB}, found {hits}'
    seed = hits[0]
    # a _best.pt is a hardlink off _running.pt; the resume restores buffers from the run PREFIX's rolling
    # sidecar (<prefix>_buffers.pt), exactly as pg14_p1_cont_best resumed this morning
    sidecar = seed.replace('_best.pt', '_buffers.pt')
    assert os.path.exists(sidecar), f'{seed}: no rolling sidecar {sidecar} beside it'
    ck = torch.load(seed, map_location='cpu', weights_only=False)
    ms = ck['modeller_state']
    assert ms['stage'] == 'train_prior', ms['stage']
    assert int(ck['train_T']) == 100, ck['train_T']
    return os.path.basename(seed), int(ms['step_ind']), ck['problem_hash']


def base():
    with BASE.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def common(cfg, name, seed_file, resume_step, seed_hash, steps=STEPS):
    cfg['run_name'] = name
    cfg['tag'] = TAG
    cfg['cuda_memory_fraction'] = 0.75
    cfg['checkpoints_dir'] = CKPT_DIR
    cfg['checkpoint_name'] = seed_file
    cfg['prior_model_name'] = None      # anchors-only churn; snapshot_prior at the stub exit writes one anyway
    assert cfg['load_weights_only'] is False and cfg['continue_from_checkpoint'] is False
    assert cfg['prior_path'].startswith('/scratch/mk8347/'), cfg['prior_path']
    cfg['molecules_path'] = LOCAL_MOLECULES
    cfg['batch_size'] = BATCH
    cfg['max_batch_size'] = BATCH
    assert cfg['grow_batch_size'] is False and cfg['batch_sizer_retest_steps'] == 0
    cfg['epochs'] = resume_step + STUB_STEPS + steps
    cfg['eval_num_samples'] = 2500
    cfg['figs_period'] = 1000
    cfg['archive_period'] = 0
    cfg['freeze_backward_policy'] = 'full'
    cfg['buffers']['prior_buffer']['source'] = 'anchors'
    assert int(cfg['integrator']['T']) == 100
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == lc['burn_in_scale']
    stages = cfg['protocols']['unconditional_tb']['stages']
    tp = [s for s in stages if s['name'] == 'train_prior'][0]
    assert tp['exit'] == [{'above': -1000000000.0, 'metric': 'bwd/mle', 'patience': 1}], tp['exit']
    assert cfg['eval_period'] == STUB_STEPS, cfg['eval_period']
    h = problem_hash_of(cfg)
    assert h == seed_hash, f'{name}: problem hash {h} != the seed\'s {seed_hash}'
    return cfg


def arm(name, k, rg, seed, gate=None, force_clip=0.0, steps=STEPS):
    seed_file, resume_step, seed_hash = seed
    cfg = common(base(), name, seed_file, resume_step, seed_hash, steps=steps)
    eq = [s for s in cfg['protocols']['unconditional_tb']['stages'] if s['name'] == 'equilibration'][0]
    assert eq['fwd_rollout_every'] == 0 and eq['loss_coeffs']['fwd'] == {'tb': 1.0, 'freeze_policy': 0.0}, name
    fc = cfg['fwd_loss_coeffs']
    fc['path_grad_last_k'] = int(k)
    fc['reward_grads'] = float(rg)
    fc['reward_grad_clip'] = 0.0
    fc['reward_grad_gate'] = None if gate is None else float(gate)
    fc['reward_grad_force_clip'] = float(force_clip)
    fc['path_grad_scale'] = 1
    fc['traj_grads'] = 0.0
    out = HERE / f'{name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f'wrote {out.name:14s} k={k} reward_grads={rg} gate={gate} force_clip={force_clip} steps={steps} '
          f'seed step {resume_step} batch={BATCH} T=100 hash ok')
    return out


if __name__ == '__main__':
    seed = resolve_seed()
    arm('ctrl', 0, 0.0, seed)
    arm('raw', 1, 1.0, seed)
    arm('g', 1, 1.0, seed, gate=1.0)
    arm('gc', 1, 1.0, seed, gate=1.0, force_clip=FORCE_CLIP)
    arm('smoke_g', 1, 1.0, seed, gate=1.0, steps=40)
