"""
stab_july21c -- 6 toy + 10 elj runs (0-15 for slurm arrays), successor to
stab_july21/stab_july21b. Built on mk_dev.yaml as of 2026-07-21 (carries the
user's manual buffer/protocol edits: prior_buffer.churn_batch_ref decouple,
250k/200k/10k buffer max_sizes, mk_dev's current buildout/terminal balance
rules) rather than the older stab_july21 bases.

Three changes from stab_july21/stab_july21b:

1. ALL trajectory lengths > 50 (previous batteries topped out at T40/T100).
   This is unexplored territory for this codebase's LR tripwires: mk_dev.yaml's
   adaptive_lr bars (gradient_norm_clip, cut/reset_{loss,grad}_abs) are
   explicitly calibrated for T=25 ("revisit if integrator.T changes" -- see
   mk_dev.yaml's own comment). scale_tripwires() below rescales all five
   LINEARLY by T/25 for every run in this battery. This is an UNVALIDATED
   EXTRAPOLATION -- no healthy-window data exists above T=25 in this
   codebase -- flagged here and in the per-run configs' comments.

2. eval_T = integrator.T always (no exceptions). The original stab_july21
   elj battery ran eval_T = 2T, which turned out to be a pure integration-dt
   mismatch artifact (see stab_july21b's docstring / reference_stab_july21
   _elj_battery_T_dominates), not a real T-resolution effect.

3. Every config preloads a prior via reuse_prior (base_toy.yaml/base_elj.yaml
   both set reuse_prior: true, and add the 'warmstart' stage between
   train_prior and buildout -- see those files). find_shared_prior's scan
   matches on problem_def only (any run_name/tag), and both toy and elj
   priors already exist in checkpoints_dir from stab_july21_toy/stab_july21
   -- so the reuse_prior: true runs below will pick one of those up
   immediately, no bootstrap ordering needed. ONE run per family (index 5
   for toy, index 15 for elj) sets reuse_prior: false anyway -- not for
   bootstrapping, but as a genuine fresh-forward-policy ablation (does a
   cold-started policy at THIS battery's specific T/lr also converge
   stably, same question as the preload runs but without warmstart's head
   start).

4. z_match protocol (regenerated 2026-07-21 evening): both bases replace the
   'warmstart' stage with the validated anchor_seed -> z_match handoff chain
   (mk_dev 2026-07-21, runs ona8i747/fxr4h4zy/wdsb4ylp). base_toy keeps the
   additive/active anchor_seed (junk prior + anchor-transfusion churn is the
   toys' localization engine; anchors now seed from the conditions file per
   the domain rule -- VERIFY toy_2harm_conditions.pt has baked target
   latents); base_elj runs the LOCALIZED variant (seed_prior_from_anchors:
   10000:flush + buffers_active false through the handoff +
   reseed_prior_from_dataset at buildout entry -- see 1wwyjp6x for why
   whole-buffer anchor_seed fails on molecule problems). REQUIRES the
   2026-07-21 evening code drop (ConditionLogZTracker per-mode level
   streams, gates/delta_worst, flush/reseed protocol actions) -- configs
   fail at parse on older code. Runs 0-2 launched under the pre-z_match
   bases are the old-handoff control arm; their yamls here no longer match
   what they ran (wandb carries the as-run configs).

Focus is stable convergence, not ceiling probing: every peak LR below is a
DOWN-scaled extrapolation from a previously-observed-safe T40 point (LR ~
1/T scaling, matching this codebase's T-compounding principle for TB
detonation risk -- see tb-surface-lr-ceiling), not an overshoot probe. Sweep
axes are T, model width/depth, and LR; no deliberate ceiling probes this
round.

Toy family (T40 reference points from stab_july21_toy zcckykx4/gejezmjg,
this session): 1024x4 ~8e-5 (shaved slightly below the observed-OK-but-
excursion-prone 1e-4), 512x4 5e-5 (clean).
  0 : 1024x4 T60 lr 5.3e-5  -- T-scale-up anchor
  1 : 1024x4 T80 lr 4.0e-5  -- T-scale-up, wider gap
  2 : 512x4  T60 lr 3.3e-5  -- width x T-scale-up
  3 : 512x4  T80 lr 2.5e-5  -- width x T-scale-up
  4 : 1024x4 T70 lr 4.6e-5, warmup_steps 300  -- faster per-stage ramp
  5 : 512x4  T70 lr 2.9e-5, reuse_prior: false  -- fresh warm-start control

Elj family (T40 reference points from mk_dev/stab_july21b: 512x4 7.5e-5
flat-stable, 1024x4 2e-5 toy-scaled guess, 512x6 4e-5 depth-transfer guess):
  6  : 512x4  T60  lr 5.0e-5  -- scaled anchor
  7  : 512x4  T80  lr 3.75e-5 -- T transfer
  8  : 512x4  T100 lr 3.0e-5  -- T transfer
  9  : 1024x4 T60  lr 1.3e-5  -- width transfer
  10 : 1024x4 T100 lr 8.0e-6  -- width x long-T corner
  11 : 512x6  T60  lr 2.7e-5  -- depth transfer
  12 : 512x6  T100 lr 1.6e-5  -- depth x long-T
  13 : 512x4  T80  lr 3.75e-5, warmup_steps 300  -- faster ramp at long T
  14 : 1024x4 T80  lr 1.0e-5  -- width x mid-T
  15 : 512x4  T60  lr 5.0e-5, reuse_prior: false  -- fresh warm-start control

dplr_rank stays 6 everywhere. max_batch_size 50000 flat (A100 utilization
policy; the adaptive growth finds each config's real ceiling).
"""

from copy import deepcopy
from pathlib import Path

import yaml

OUTDIR = Path(__file__).parent
MAX_BATCH = 50000
WIDTH_KEYS = ('s_emb_dim', 't_hidden_dim', 's_hidden_dim',
              'policy_hidden_dim', 'flow_hidden_dim', 'cond_hidden_dim')
LAYER_KEYS = ('s_layers', 'policy_layers', 'flow_layers', 'cond_layers')
LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')

# mk_dev.yaml's T=25 baseline tripwire bars (see adaptive_lr comment there)
T_BASELINE = 25
CLIP_BASELINE = 250.0
CUT_LOSS_BASELINE = 2.5e+3
CUT_GRAD_BASELINE = 7.5e+3
RESET_LOSS_BASELINE = 2.5e+4
RESET_GRAD_BASELINE = 7.5e+4


def load_base(name):
    with (OUTDIR / name).open('r') as f:
        return yaml.safe_load(f)


def fmt_lr(x):
    return f"{x:.1e}".replace('e-0', 'e-')


def scale_tripwires(config, traj_len):
    factor = traj_len / T_BASELINE
    config['gradient_norm_clip'] = round(CLIP_BASELINE * factor, 1)
    config['adaptive_lr']['cut_loss_abs'] = round(CUT_LOSS_BASELINE * factor, 1)
    config['adaptive_lr']['cut_grad_abs'] = round(CUT_GRAD_BASELINE * factor, 1)
    config['adaptive_lr']['reset_loss_abs'] = round(RESET_LOSS_BASELINE * factor, 1)
    config['adaptive_lr']['reset_grad_abs'] = round(RESET_GRAD_BASELINE * factor, 1)


def make_config(ind, base, family, width, layers, traj_len, peak_lr,
                warmup_steps=None, fresh_warmstart=False, seed=None, note=''):
    config = deepcopy(base)

    for key in WIDTH_KEYS:
        config['model'][key] = width
    for key in LAYER_KEYS:
        config['model'][key] = layers

    config['integrator']['T'] = traj_len
    config['integrator']['min_traj_length'] = traj_len
    config['integrator']['max_traj_length'] = traj_len
    config['eval_T'] = traj_len  # ALWAYS == train T for this battery
    if seed is not None:
        config['seed'] = seed

    for key in LR_KEYS:
        config[key] = round(float(peak_lr), 12)

    scale_tripwires(config, traj_len)

    if warmup_steps is not None:
        config['adaptive_lr']['warmup_steps'] = warmup_steps

    if fresh_warmstart:
        config['reuse_prior'] = False

    config['max_batch_size'] = MAX_BATCH

    name = f"{family}_h{width}x{layers}_T{traj_len}_lr{fmt_lr(peak_lr)}"
    if warmup_steps is not None:
        name += f"_w{warmup_steps}"
    if fresh_warmstart:
        name += "_fresh"
    if seed is not None:
        name += f"_s{seed % 100}"
    config['run_name'] = name

    with open(OUTDIR / f'{ind}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return {'index': ind, 'run_name': name, 'family': family,
            'width': width, 'layers': layers, 'T': traj_len,
            'eval_T': config['eval_T'], 'seed': config['seed'],
            'peak_lr': float(peak_lr), 'warmup_steps': warmup_steps,
            'fresh_warmstart': fresh_warmstart,
            'gradient_norm_clip': config['gradient_norm_clip'],
            'cut_loss_abs': config['adaptive_lr']['cut_loss_abs'],
            'cut_grad_abs': config['adaptive_lr']['cut_grad_abs'],
            'note': note}


if __name__ == '__main__':
    toy = load_base('base_toy.yaml')
    elj = load_base('base_elj.yaml')
    log = []

    # ---- toy stabilization battery: 0-5 ----
    log.append(make_config(0, toy, 'toy', 1024, 4, 60, 5.3e-5,
                           note='T-scale-up anchor'))
    log.append(make_config(1, toy, 'toy', 1024, 4, 80, 4.0e-5,
                           note='T-scale-up, wider gap'))
    log.append(make_config(2, toy, 'toy', 512, 4, 60, 3.3e-5,
                           note='width x T-scale-up'))
    log.append(make_config(3, toy, 'toy', 512, 4, 80, 2.5e-5,
                           note='width x T-scale-up'))
    log.append(make_config(4, toy, 'toy', 1024, 4, 70, 4.6e-5,
                           warmup_steps=300, note='faster per-stage ramp'))
    log.append(make_config(5, toy, 'toy', 512, 4, 70, 2.9e-5,
                           fresh_warmstart=True, note='fresh warm-start control'))

    # ---- real-problem transfer battery: 6-15 ----
    log.append(make_config(6, elj, 'elj', 512, 4, 60, 5.0e-5,
                           note='scaled anchor'))
    log.append(make_config(7, elj, 'elj', 512, 4, 80, 3.75e-5,
                           note='T transfer'))
    log.append(make_config(8, elj, 'elj', 512, 4, 100, 3.0e-5,
                           note='T transfer'))
    log.append(make_config(9, elj, 'elj', 1024, 4, 60, 1.3e-5,
                           note='width transfer'))
    log.append(make_config(10, elj, 'elj', 1024, 4, 100, 8.0e-6,
                           note='width x long-T corner'))
    log.append(make_config(11, elj, 'elj', 512, 6, 60, 2.7e-5,
                           note='depth transfer'))
    log.append(make_config(12, elj, 'elj', 512, 6, 100, 1.6e-5,
                           note='depth x long-T'))
    log.append(make_config(13, elj, 'elj', 512, 4, 80, 3.75e-5,
                           warmup_steps=300, note='faster ramp at long T'))
    log.append(make_config(14, elj, 'elj', 1024, 4, 80, 1.0e-5,
                           note='width x mid-T'))
    log.append(make_config(15, elj, 'elj', 512, 4, 60, 5.0e-5,
                           fresh_warmstart=True, note='fresh warm-start control'))

    with open(OUTDIR / 'experiment_log.yaml', 'w') as f:
        yaml.dump(log, f, sort_keys=False)

    print(f'Generated {len(log)} configs with log at {OUTDIR / "experiment_log.yaml"}')
