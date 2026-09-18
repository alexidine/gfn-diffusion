"""force_sep18c -- the NEAR-CONVERGED leg of the terminal-force test, LOCAL (RTX 5080).

force_sep18 (T=10, off an MLE seed) measured the SEARCH regime: every force form helped and
the residual gate cost nothing. The harm the cluster force arms showed -- relocation turning
into flight once the basins are found -- needs a sampler that already sits at basin floors.
That is the dose_sep16 seed: p12_mip_lr1's step-160000 archive (T=100, Z 28.7, phase 2,
buffers in its sidecar), downloaded to D:\crystal_datasets\checkpoints. These arms resume
it on the FORWARD SEAT exactly as dose16_fs_k1 did (fwd 0.30 / bwd 0.50 / replay 0.20,
rollout every step, fwd trains the policy), with P_B frozen, prior-buffer churn from noised
anchors only (no prior model), at batch 400 so the card holds T=100 with three branches.

  ctrl   reward_grads 0                          the seat alone
  raw    k 1, reward_grads 1, no gate, no clip   dose16_fs_k1's term (turned over at ~1k steps on the cluster)
  g      k 1, gate 1.0, no clip                  the search-phase winner
  gc     k 1, gate 1.0, |F| clip 250             the reshaped form scored offline on this very checkpoint

IDENTITY. load_full compares the stored problem_def, which hashes prior_path as the FULL
cluster string, so the config keeps that string verbatim and a copy of the prior file sits at
C:\scratch\mk8347\data\...\ (what /scratch/... resolves to on this box). molecules_path is
not in the identity and points at the D: copy. batch_size/max_batch_size 400 clamps the
restored 1600 (a resume never exceeds the config's max_batch_size). freeze_backward_policy
is a top-level key, so it applies on resume; stage on_enter actions would not. The ELJ
energy runs under MXT_LEGACY_TRICLINIC_WALLS=1 (the archive predates the Niggli walls).

READ. Against ctrl at matched steps: Z, eval_fwd/tb_err (the gain); bwd/tb_err, bwd/tb_resid,
bwd/under_coverage, zmatch/delta_mean (the damage: forward improving while these degrade is
the relocation fingerprint); Excess Energy P50/P90/P99; rewardgrad/gate_frac and push stats.
"""
import sys
from argparse import Namespace
from pathlib import Path

import yaml

HERE = Path(__file__).parent
ES = HERE.parent.parent
BASE = HERE.parent / 'dose_sep16' / 'dose16_fs_k1.yaml'
TAG = 'fs18c'
CKPT_DIR = 'D:\\crystal_datasets\\checkpoints\\'
SEED = 'p12_p12_mip_lr1_elj-mipcas_sg2_zp1_elj_200k_prior_dataset-T2.5-a8b328_step160000.pt'
PRIOR_MODEL = 'p12_p12_mip_lr1_elj-mipcas_sg2_zp1_elj_200k_prior_dataset-T2.5-a8b328_prior.pt'
SEED_HASH = 'a8b328'
RESUME_STEP = 160000
STEPS = 2000
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


def base():
    with BASE.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def common(cfg, name, steps=STEPS):
    cfg['run_name'] = name
    cfg['tag'] = TAG
    cfg['cuda_memory_fraction'] = 0.75
    cfg['checkpoints_dir'] = CKPT_DIR
    cfg['checkpoint_name'] = SEED
    cfg['prior_model_name'] = PRIOR_MODEL
    assert cfg['load_weights_only'] is False and cfg['continue_from_checkpoint'] is False
    # prior_path stays the cluster string: it is part of the stored problem identity
    assert cfg['prior_path'].startswith('/scratch/mk8347/'), cfg['prior_path']
    cfg['molecules_path'] = LOCAL_MOLECULES
    cfg['batch_size'] = BATCH
    cfg['max_batch_size'] = BATCH
    assert cfg['grow_batch_size'] is False and cfg['batch_sizer_retest_steps'] == 0
    cfg['epochs'] = RESUME_STEP + steps
    cfg['eval_period'] = 250
    cfg['eval_num_samples'] = 2500
    cfg['figs_period'] = 1000
    cfg['archive_period'] = 0
    cfg['freeze_backward_policy'] = 'full'
    # owner 2026-09-18: no prior-MODEL samples in the backward branch; churn from noised anchors only
    cfg['buffers']['prior_buffer']['source'] = 'anchors'
    assert int(cfg['integrator']['T']) == 100
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == lc['burn_in_scale']
    h = problem_hash_of(cfg)
    assert h == SEED_HASH, f'{name}: problem hash {h} != the archive\'s {SEED_HASH}'
    return cfg


def arm(name, k, rg, gate=None, force_clip=0.0, steps=STEPS):
    cfg = common(base(), name, steps=steps)
    stages = cfg['protocols']['unconditional_tb']['stages']
    eq = [s for s in stages if s['name'] == 'equilibration'][0]
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
    print(f'wrote {out.name:14s} k={k} reward_grads={rg} gate={gate} force_clip={force_clip} steps={steps} batch={BATCH} T=100 hash ok')
    return out


if __name__ == '__main__':
    arm('ctrl', 0, 0.0)
    arm('raw', 1, 1.0)
    arm('g', 1, 1.0, gate=1.0)
    arm('gc', 1, 1.0, gate=1.0, force_clip=FORCE_CLIP)
    arm('smoke_g', 1, 1.0, gate=1.0, steps=30)
