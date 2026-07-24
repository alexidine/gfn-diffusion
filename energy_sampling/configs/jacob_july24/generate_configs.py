"""
jacob_july24 -- 5 diagnostic runs (0-4 for slurm arrays) chasing the buildout
variance blowup, all resuming FROM stab_july21c index 6's (o06f3c2z,
elj_h512x4_T60_lr5.0e-5) z_matched snapshot -- the buildout-entry injection
point -- and running the failing stage directly. Base = stab_july21c's base_elj.yaml verbatim; every run keeps that
run's exact geometry (512x4, T60, lr 5e-5, T-scaled tripwires) so the ONLY
difference vs the parent is the single intervention per run.

Code prerequisite: the 2026-07-24 molecular_crystal.py (jacobian component
logging + MXT_JACOBIAN_DELTA softening + rot_*_jacobian_peak_energy channels).
ONLY molecular_crystal.py ships to the cluster -- no other file changed
behavior-wise, and the peak channels deliberately ride through the existing
'Mean <key>' logging path so train.py stays frozen.

Runs:
  0 : jacob_control       -- measure only: jacobian components under the
                             unmodified failure. Control arm for 1-4.
  1 : jacob_soften_d0.05  -- identical config to 0; REQUIRES the env var
                             MXT_JACOBIAN_DELTA=0.05 at launch (see below).
                             Softens the rotational Haar log-singularities
                             (r-term cap ~37 -> ~6 nats).
  2 : jacob_beta2         -- Huber beta 10 -> 2 on fwd/bwd/replay loss
                             defaults: per-sample TB gradient magnitude capped
                             at 2 instead of 10 (z_match already runs fwd
                             beta 2; its stage override is unchanged).
  3 : jacob_lossclip100   -- loss_clip 1e9 -> 100 on all three defaults:
                             soft (log-tail) per-sample per-term loss cap,
                             mainly squashing the replay-garbage terms
                             (replay/tb ran 240-7000 during the parent ramp).
  4 : jacob_gradclip100   -- gradient_norm_clip 600 -> 100 (global norm).

Why the env var for run 1: any NEW energy_config yaml key enters problem_def
(only the _NON_IDENTITY_ENERGY_CONFIG_KEYS are popped), and assert_problem_match
then hard-refuses the parent checkpoint. utils.py can't ship, so the knob rides
in the environment. Launch run 1 as:
    MXT_JACOBIAN_DELTA=0.05 python train.py --config configs/jacob_july24/1.yaml
Forgetting it degrades run 1 to a duplicate of run 0 -- the init log prints
"MXT_JACOBIAN_DELTA=0.05: ..." when active, and Mean/peak rot_r channels cap
at ~6 nats, so a silent miss is visible both ways.

Resume mechanics (what the checkpoint carries vs what the config owns):
  - Injection point = the parent's z_matched SNAPSHOT (z_match exit = buildout
    entry), NOT 'best'. Snapshots save with_buffers=True, so weights and
    buffers come as a consistent frozen pair (..._z_matched_buffers.pt);
    'z_matched' is not in CHECKPOINT_TAGS, so sidecar_candidates never falls
    back to the parent's rolling sidecar (the corrupted pre-death state) --
    it finds the frozen pair or nothing. The snapshot also carries
    request_eval stamped with the exit streaks, so the first post-resume eval
    re-fires the z_match exit through the normal path and buildout's on_enter
    (reseed_prior_from_dataset) actually runs. Each run therefore traverses
    the WHOLE buildout failure cycle under its intervention (parent: entry
    ~13.6k, first ramp ~22k-29.5k, death 49.6k -- epochs 60000 covers it).
  - checkpoint_name + load_weights_only: false = FULL resume: weights, EMA,
    optimizers, step_ind, stage NAME, stage_ctrl, batch controller state, and
    lr_ctrl (phase_seen/scale). Stage BEHAVIOR (loss coeffs, balance rules,
    exits, tripwire bars) always re-derives from THIS config, so the
    beta/loss_clip/gradient_norm_clip overrides bind on resume with no extra
    flag.
  - lr_ctrl.scale is restored from the snapshot (~0.01-0.08, a transition-era
    fire had just cut it) and the controller re-stamps LRs from it every
    tick -- all five runs inherit the SAME scale, so they stay comparable; no
    attempt to reset it (the parent's second ramp ran entirely at min_lr
    anyway, LR level only sets the walk's clock).
  - reuse_prior: true resolves the frozen prior model through
    find_shared_prior (problem_def match, any run_name) -- the parent's own
    *_prior.pt qualifies.

eval_period stays 250 (not the usual 500): the collapse's fast phase is a
couple hundred steps and the whole point is component resolution through it;
figs_period 1000 keeps the upload budget sane. epochs 60000 bounds runtime
(the parent died at ~49.6k from a ~13.6k buildout entry).
"""

from copy import deepcopy
from pathlib import Path

import yaml

OUTDIR = Path(__file__).parent

# o06f3c2z's buildout-entry snapshot (frozen sidecar ..._z_matched_buffers.pt
# alongside, same-minute mtime). NOT the _fresh_ variant -- that's index 15's
# fresh-warm-start control run.
CKPT_FILE = 'stab_july21c_elj_h512x4_T60_lr5.0e-5_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-68890c_z_matched.pt'

# parent run geometry: stab_july21c index 6
WIDTH, LAYERS, TRAJ_LEN, PEAK_LR = 512, 4, 60, 5.0e-5
WIDTH_KEYS = ('s_emb_dim', 't_hidden_dim', 's_hidden_dim',
              'policy_hidden_dim', 'flow_hidden_dim', 'cond_hidden_dim')
LAYER_KEYS = ('s_layers', 'policy_layers', 'flow_layers', 'cond_layers')
LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')
LOSS_COEFF_BLOCKS = ('fwd_loss_coeffs', 'bwd_loss_coeffs', 'replay_loss_coeffs')

# stab_july21c's T=25 baseline tripwire bars, scaled linearly by T/25 per that
# battery's (unvalidated-extrapolation) convention -- kept IDENTICAL here so
# the control arm reproduces the parent run's protection exactly
T_BASELINE = 25
CLIP_BASELINE = 250.0
CUT_LOSS_BASELINE = 2.5e+3
CUT_GRAD_BASELINE = 7.5e+3
RESET_LOSS_BASELINE = 2.5e+4
RESET_GRAD_BASELINE = 7.5e+4


def make_base():
    with (OUTDIR / 'base_elj.yaml').open('r') as f:
        config = yaml.safe_load(f)

    for key in WIDTH_KEYS:
        config['model'][key] = WIDTH
    for key in LAYER_KEYS:
        config['model'][key] = LAYERS
    config['integrator']['T'] = TRAJ_LEN
    config['integrator']['min_traj_length'] = TRAJ_LEN
    config['integrator']['max_traj_length'] = TRAJ_LEN
    config['eval_T'] = TRAJ_LEN
    for key in LR_KEYS:
        config[key] = PEAK_LR

    factor = TRAJ_LEN / T_BASELINE
    config['gradient_norm_clip'] = round(CLIP_BASELINE * factor, 1)
    config['adaptive_lr']['cut_loss_abs'] = round(CUT_LOSS_BASELINE * factor, 1)
    config['adaptive_lr']['cut_grad_abs'] = round(CUT_GRAD_BASELINE * factor, 1)
    config['adaptive_lr']['reset_loss_abs'] = round(RESET_LOSS_BASELINE * factor, 1)
    config['adaptive_lr']['reset_grad_abs'] = round(RESET_GRAD_BASELINE * factor, 1)
    config['max_batch_size'] = 50000

    # battery-common resume + cadence settings
    config['tag'] = 'jacob_july24'
    config['checkpoint_name'] = CKPT_FILE
    config['load_weights_only'] = False  # FULL resume (base_elj default is true)
    config['epochs'] = 60000
    config['eval_period'] = 250
    config['figs_period'] = 1000
    return config


def emit(ind, config, name, note, launch_env=None):
    config['run_name'] = name
    with open(OUTDIR / f'{ind}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return {'index': ind, 'run_name': name, 'note': note,
            'launch_env': launch_env,
            'checkpoint_name': config['checkpoint_name'],
            'gradient_norm_clip': config['gradient_norm_clip'],
            'huber_beta': config['fwd_loss_coeffs']['beta'],
            'loss_clip': config['fwd_loss_coeffs']['loss_clip']}


if __name__ == '__main__':
    log = []

    log.append(emit(0, make_base(), 'jacob_control',
                    'measure jacobian components under the unmodified failure'))

    log.append(emit(1, make_base(), 'jacob_soften_d0.05',
                    'rotational Haar singularities softened; config identical to 0',
                    launch_env='MXT_JACOBIAN_DELTA=0.05'))

    config = make_base()
    for block in LOSS_COEFF_BLOCKS:
        config[block]['beta'] = 2.0
    log.append(emit(2, config, 'jacob_beta2',
                    'Huber beta 10 -> 2 on all loss-coeff defaults'))

    config = make_base()
    for block in LOSS_COEFF_BLOCKS:
        config[block]['loss_clip'] = 100.0
    log.append(emit(3, config, 'jacob_lossclip100',
                    'per-sample soft loss clip 1e9 -> 100 on all defaults'))

    config = make_base()
    config['gradient_norm_clip'] = 100.0
    log.append(emit(4, config, 'jacob_gradclip100',
                    'global gradient norm clip 600 -> 100'))

    with open(OUTDIR / 'experiment_log.yaml', 'w') as f:
        yaml.dump(log, f, sort_keys=False)

    print(f'Generated {len(log)} configs with log at {OUTDIR / "experiment_log.yaml"}')
