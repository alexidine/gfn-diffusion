"""
tsched_july24 -- 9 runs (slurm array 0-8) testing an in-rollout schedule on the
forward noise rate. All resume from stab_july21c index 6's (o06f3c2z,
elj_h512x4_T60_lr5.0e-5) phase1_exit snapshot and traverse
anchor_seed -> z_match -> buildout under their own schedule. Base =
jacob_july24's base_elj.yaml verbatim (itself stab_july21c's), and every run
keeps that geometry (512x4, T60, lr 5e-5, T-scaled tripwires) so the only
difference vs the control is the variance schedule.

WHY phase1_exit AND NOT z_matched (which jacob_july24 used): a schedule moves
E[log P_F] by tens of nats on the terminal step, so log Z moves with it. A
z_matched snapshot is calibrated to the constant-rate policy and is stale the
moment the schedule is on. Variance changes go BEFORE z_match, never after.

Code prerequisite: the 2026-07-24 gfn.py (in-rollout variance schedule, P_B
bridge re-expressed in accumulated-variance time) and checkpointing.py
(RECONFIGURABLE_GFN_KEYS, so the schedule follows THIS config on resume rather
than being inherited from the parent's stored gfn_config -- without it every
arm silently runs as the control). Both ship together; nothing else changed.

WHAT THE SCHEDULE IS. The forward SDE's noise rate (variance per unit
trajectory time) becomes

    sigma^2(t) = sigma^2(0) * ratio ** (t ** power)

decaying from t=0 to t=1. `ratio` = sigma^2(1)/sigma^2(0); null = constant =
the code that produced every previous run, bit-for-bit (verified: full
fwd/bwd/replay trajectories, 30 tensors, zero mismatches). `power` sets how
late the drop concentrates: 1 spreads it over the whole rollout, 16 confines it
to roughly the last 15%.

Everything downstream runs on accumulated variance V(t) = int_0^t sigma^2,
which is what the constant-rate code was already using t as a proxy for. P_B is
this process's Brownian bridge, so its contraction toward the pin is
V(t_prev)/V(t_next), not t_prev/t_next; under a p=4 ramp those differ by 7x
over the last tenth of the rollout, far outside the +-40% pb_drift_range, so
scheduling P_F alone would have left P_B describing a different process exactly
where the experiment lives.

preserve_budget rescales sigma^2(0) so V(1) = t_scale exactly. x_0 = 0 is
deterministic and the drift is deterministic, so V(1) is the whole stock of
randomness available to build the terminal distribution: without preservation a
schedule shrinks the reachable support at the same time as the terminal step,
and the two are not separable afterwards.

WHAT IS BEING TESTED. Terminal mode width is floored by the last step's noise:
Var(x_1) >= sigma^2(1)*dt in every direction. At T=60 under the constant rate
that floor is ~4x the measured stiff-direction width of the reheated thermal
mode (~0.005 latent units), so the mode cannot be resolved at any drift
quality. Arms 3-8 remove the floor while holding the noise budget fixed.

Arms 1-2 are the null hypothesis and need no schedule at all -- just a smaller
constant t_scale. If a flat cut resolves the mode as well as a schedule does,
none of the schedule machinery is needed. Arm 2's t_scale is set to arm 4's
terminal rate exactly, so the two share a terminal floor and differ only in
budget: arm 2 has ~1/50th the support. The prediction that separates them is
that arm 2 resolves the mode AND loses coverage (under_coverage, buffer reach
degrading in the tails), while arm 4 resolves it with coverage intact.

  0 : tsched_control     -- schedule off; bit-identical to the parent code.
                            Also the anchor for reading arms 1-8.
  1 : tsched_flat_r10    -- constant rate cut 10x (t_scale 5.0e-3), no schedule
  2 : tsched_flat_match  -- constant rate cut to arm 4's TERMINAL rate, no
                            schedule: matched floor, ~1/50th the budget
  3 : tsched_p16_r50     -- power 16, ratio 1/50 -- surgical, last ~15% only
  4 : tsched_p4_r50      -- power 4,  ratio 1/50 -- grid centre
  5 : tsched_p1_r50      -- power 1,  ratio 1/50 -- decay across whole rollout
  6 : tsched_p4_r15      -- power 4,  ratio 1/15 -- shallow
  7 : tsched_p4_r150     -- power 4,  ratio 1/150 -- deep
  8 : tsched_p4_r50_raw  -- power 4,  ratio 1/50, preserve_budget OFF; control
                            on the budget knob itself

Ratios are chosen so the p=4 arms bracket the 0.005 target: the emitted
experiment_log.yaml carries each arm's realized rate(0), rate(1) and
last-step std at T=60, computed with the same trapezoid the model uses, so
calibration is read off the log rather than re-derived.

Resume mechanics: checkpoint_name + load_weights_only false = FULL resume
(weights, EMA, optimizers, step_ind, stage name, stage_ctrl, batch controller,
lr_ctrl). Stage BEHAVIOUR re-derives from this config, as do the three schedule
keys. reuse_prior resolves the frozen prior through find_shared_prior: the
schedule keys live under model: and are not part of problem_def, so all nine
arms share one prior and phase 1 stays skipped.

REPLAY ARMS 9-13 (appended 2026-07-24, slurm array now 0-13). These test the
replay-buffer redesign and REQUIRE the 2026-07-24 train.py + buffer.py
(softmax admission/purge over min(|resid|, cap)/T, health-modulated cap
cap(h) = cap_min + (cap_max-cap_min)/(1 + h/h0) with h = the fwd/scatter_err
EMA, residence TTL, TTL-cohort telemetry). Shipping those REPLACES the old
replay mechanism for EVERY arm -- there is no old-behavior fallback -- so
base_elj.yaml now carries the new replay keys and arms 0-8 run the new
mechanism at reference settings (T 5, cap 16-50 @ h0 8, TTL 500,
churn_rate 50). Arm 0's "bit-identical to the parent code" claim is
accordingly narrowed to the VARIANCE SCHEDULE: the replay subsystem differs
from the parent run in all arms. Motivation for accepting that: fresh-replay-
buffer runs (2026-07-24) got total stability from discarding accumulated
buffer gunk alone, so the reference settings are expected stabilizing --
the schedule comparison 0-vs-1-8 stays internally clean because every arm
shares them.

Arms 9-13 are single-knob deltas vs arm 0 (schedule OFF, control geometry),
so the battery is two independent one-factor scans sharing one control:

   9 : tsched_replay_t2       -- admit_temperature 2: sharp tail focus
                                 (T->0 recovers argsort-like admission)
  10 : tsched_replay_t20      -- admit_temperature 20: near-uniform reservoir
  11 : tsched_replay_ttl150   -- TTL 150: aggressive freshness, the
                                 continuous version of the fresh-buffer result
  12 : tsched_replay_ttl2000  -- TTL 2000: staleness vs supersampling trade
  13 : tsched_replay_fixedcap -- cap_min = cap_max = 50: health modulation
                                 OFF, static cap only -- is cap(h) load-bearing
                                 or is a fixed belt enough?

Read them on fwd convergence speed plus the TTL-cohort scalars
(replay_buffer_absorbed_frac / expired_undrawn_frac / expired_delta /
expired_draws) and the replay_buffer_admit_cap / admit_health traces. The
~15k transition kick (deterministic across all jacob_july24 arms) is the
expected excursion that actually exercises the cap -- if admit_cap never
leaves cap_max on any arm, the health modulation was never engaged and arm
13 is uninformative on that axis.
"""

from copy import deepcopy
from math import exp, log
from pathlib import Path

import yaml

OUTDIR = Path(__file__).parent

# o06f3c2z's phase1_exit snapshot -- train_prior's on_exit, i.e. anchor_seed
# entry. CONFIRM THIS FILE EXISTS on the cluster before submitting: jacob_july24
# used the _z_matched sibling, and if phase-1 snapshots were not retained this
# battery needs a fresh-run fallback instead.
CKPT_FILE = ('stab_july21c_elj_h512x4_T60_lr5.0e-5_elj-mipcas_sg2_zp1_'
             'elj_prior_dataset-T2.5-68890c_phase1_exit.pt')

# parent run geometry: stab_july21c index 6
WIDTH, LAYERS, TRAJ_LEN, PEAK_LR = 512, 4, 60, 5.0e-5
WIDTH_KEYS = ('s_emb_dim', 't_hidden_dim', 's_hidden_dim',
              'policy_hidden_dim', 'flow_hidden_dim', 'cond_hidden_dim')
LAYER_KEYS = ('s_layers', 'policy_layers', 'flow_layers', 'cond_layers')
LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')

# stab_july21c's T=25 baseline tripwire bars, scaled linearly by T/25 per that
# battery's convention -- identical to jacob_july24 so the control reproduces
# the parent's protection exactly
T_BASELINE = 25
CLIP_BASELINE = 250.0
CUT_LOSS_BASELINE = 2.5e+3
CUT_GRAD_BASELINE = 7.5e+3
RESET_LOSS_BASELINE = 2.5e+4
RESET_GRAD_BASELINE = 7.5e+4

VAR_SCHEDULE_GRID = 8192  # must match GFN.VAR_SCHEDULE_GRID


def schedule_profile(t_scale, ratio, power, preserve_budget, traj_len=TRAJ_LEN):
    """
    Realized schedule quantities, using the same cumulative trapezoid on the
    same grid the model builds, so the logged numbers are the ones training
    will actually use rather than an analytic approximation to them.

    `last_step_var` is the floor that matters: the variance the final step
    injects, V(1) - V(1 - 1/traj_len), which is what Var(x_1) is bounded below
    by. For a decaying schedule that exceeds the instantaneous endpoint rate
    times dt, since the step averages over an interval where the rate is still
    falling -- quoting sigma^2(1)*dt instead would understate the floor.
    """
    n = VAR_SCHEDULE_GRID
    if ratio is None:
        accum = [t_scale * i / n for i in range(n + 1)]
        s0 = rate1 = t_scale
    else:
        rate = [exp(log(ratio) * (i / n) ** power) for i in range(n + 1)]
        integral = sum(0.5 * (rate[i] + rate[i - 1]) / n for i in range(1, n + 1))
        s0 = t_scale / integral if preserve_budget else t_scale
        accum, running = [0.0], 0.0
        for i in range(1, n + 1):
            running += 0.5 * (rate[i] + rate[i - 1]) / n
            accum.append(running * s0)
        rate1 = s0 * ratio

    lo = accum[round(n * (1 - 1 / traj_len))]
    return {'rate_at_0': s0, 'rate_at_1': rate1, 'variance_budget_V1': accum[-1],
            'last_step_var': accum[-1] - lo}


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

    # schedule off by default: identical to the parent code
    config['model']['t_scale_ratio'] = None
    config['model']['t_scale_power'] = 4.0
    config['model']['t_scale_preserve_budget'] = True

    config['tag'] = 'tsched_july24'
    config['checkpoint_name'] = CKPT_FILE
    config['load_weights_only'] = False
    config['epochs'] = 60000
    config['eval_period'] = 500
    config['figs_period'] = 1000
    return config


def emit(ind, config, name, note):
    config['run_name'] = name
    with open(OUTDIR / f'{ind}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    model = config['model']
    profile = schedule_profile(
        model['t_scale'], model['t_scale_ratio'],
        model['t_scale_power'], model['t_scale_preserve_budget'])
    rb = config['buffers']['replay_buffer']
    return {'index': ind, 'run_name': name, 'note': note,
            't_scale': model['t_scale'],
            't_scale_ratio': model['t_scale_ratio'],
            't_scale_power': model['t_scale_power'],
            't_scale_preserve_budget': model['t_scale_preserve_budget'],
            'rate_at_0': float(f'{profile["rate_at_0"]:.6g}'),
            'rate_at_1': float(f'{profile["rate_at_1"]:.6g}'),
            'variance_budget_V1': float(f'{profile["variance_budget_V1"]:.6g}'),
            'last_step_std_at_T60': float(f'{profile["last_step_var"] ** 0.5:.4g}'),
            'replay_admit_T': rb['admit_temperature'],
            'replay_ttl': rb['max_residence_steps'],
            'replay_cap_min_max_h0': [rb['admit_cap_min'], rb['admit_cap_max'],
                                      rb['admit_cap_health_h0']]}


def scheduled(ratio, power, preserve_budget=True):
    config = make_base()
    config['model']['t_scale_ratio'] = ratio
    config['model']['t_scale_power'] = float(power)
    config['model']['t_scale_preserve_budget'] = preserve_budget
    return config


def flat(t_scale):
    config = make_base()
    config['model']['t_scale'] = float(f'{t_scale:.6g}')
    return config


def replay_arm(**overrides):
    """Control geometry (schedule OFF), replay-buffer knobs overridden."""
    config = make_base()
    config['buffers']['replay_buffer'].update(overrides)
    return config


if __name__ == '__main__':
    entries = []
    base_t_scale = make_base()['model']['t_scale']

    # arm 4 is the grid centre; arm 2 is the flat rate giving the SAME final-step
    # variance, so the two share a floor exactly and differ only in budget
    centre = scheduled(1 / 50, 4)
    centre_rate1 = schedule_profile(base_t_scale, 1 / 50, 4.0, True)['last_step_var'] * TRAJ_LEN

    entries.append(emit(0, make_base(), 'tsched_control',
                    'schedule off; bit-identical to the parent code'))
    entries.append(emit(1, flat(base_t_scale / 10), 'tsched_flat_r10',
                    'constant rate cut 10x, no schedule'))
    entries.append(emit(2, flat(centre_rate1), 'tsched_flat_match',
                    "constant rate at arm 4's terminal rate: matched floor, ~1/50th the budget"))
    entries.append(emit(3, scheduled(1 / 50, 16), 'tsched_p16_r50',
                    'power 16, ratio 1/50 -- drop confined to the last ~15%'))
    entries.append(emit(4, deepcopy(centre), 'tsched_p4_r50',
                    'power 4, ratio 1/50 -- grid centre'))
    entries.append(emit(5, scheduled(1 / 50, 1), 'tsched_p1_r50',
                    'power 1, ratio 1/50 -- decay across the whole rollout'))
    entries.append(emit(6, scheduled(1 / 15, 4), 'tsched_p4_r15',
                    'power 4, ratio 1/15 -- shallow'))
    entries.append(emit(7, scheduled(1 / 150, 4), 'tsched_p4_r150',
                    'power 4, ratio 1/150 -- deep'))
    entries.append(emit(8, scheduled(1 / 50, 4, preserve_budget=False), 'tsched_p4_r50_raw',
                    'power 4, ratio 1/50, preserve_budget off -- control on the budget knob'))

    # replay arms: single-knob deltas vs arm 0 (see docstring, REPLAY ARMS 9-13)
    entries.append(emit(9, replay_arm(admit_temperature=2.0), 'tsched_replay_t2',
                    'replay softmax T 5 -> 2: sharp tail focus'))
    entries.append(emit(10, replay_arm(admit_temperature=20.0), 'tsched_replay_t20',
                    'replay softmax T 5 -> 20: near-uniform reservoir'))
    entries.append(emit(11, replay_arm(max_residence_steps=150), 'tsched_replay_ttl150',
                    'replay TTL 500 -> 150: continuous fresh-buffer limit'))
    entries.append(emit(12, replay_arm(max_residence_steps=2000), 'tsched_replay_ttl2000',
                    'replay TTL 500 -> 2000: staleness vs supersampling trade'))
    entries.append(emit(13, replay_arm(admit_cap_min=50.0), 'tsched_replay_fixedcap',
                    'admit_cap_min = admit_cap_max = 50: health modulation off, static cap only'))

    with open(OUTDIR / 'experiment_log.yaml', 'w') as f:
        yaml.dump(entries, f, sort_keys=False)

    print(f'Generated {len(entries)} configs with log at {OUTDIR / "experiment_log.yaml"}')
    print(f'{"idx":>3s} {"run":>20s} {"rate(0)":>10s} {"rate(1)":>10s} '
          f'{"V(1)":>9s} {"std@T60":>9s}')
    for row in entries:
        print(f'{row["index"]:>3d} {row["run_name"]:>20s} {row["rate_at_0"]:10.4g} '
              f'{row["rate_at_1"]:10.4g} {row["variance_budget_V1"]:9.5f} '
              f'{row["last_step_std_at_T60"]:9.5f}')
