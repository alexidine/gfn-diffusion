"""
synth_real_aug19 -- two ~20 minute REAL runs, everything current turned on.

WHAT THIS IS FOR, and how it differs from `configs/synth_aug19/`. That battery
was diagnostic: single-key toggles, truncated priors, one terminal stage,
whatever it took to isolate a mechanism. This one is the opposite. It changes
NOTHING from the canonical config except the run identity, the resume point and
the budget -- so it answers "does the integrated system, with every 2026-08-19
default live, train cleanly on both routes" rather than any question about a
part.

Live in these runs, none of it overridden:

  * the occupancy ladder ARMED (`batch_util_target: 0.6` as a fraction of the
    card, `grow_batch_size: true`, `max_batch_size: 20000`) -- state 9;
  * the MLIP fast paths, which are in-code defaults and take no config at all
    (they are inert on the ELJ route both arms use, and that is fine: this
    battery is about the training stack, not the energy backend);
  * the canonical eval/figure cadence, so figure logging and held-out eval are
    exercised rather than tuned out.

WARM-STARTED FROM PHASE-1 EXITS, deliberately: the MLE warm start is a solved
problem and re-running it would spend the whole budget before reaching the
stage where anything interesting happens. Both arms therefore resume full state
(weights, optimizers, buffers) and land directly in their terminal stage.

    python configs/synth_real_aug19/make.py

TWO THINGS TO READ FIRST when these come back, because they are how each route
lies:

  * unconditional -- `log_Z_learned` and the fwd/bwd/replay `tb_err_worst`
    trio. A log Z dive at the resume step is a failed handoff, not a bad run.
  * conditional -- `eval_test_*` BEFORE `eval_fwd_*`. Train r2, tb_err and
    scatter_err can all improve on the same evaluation where the held-out set
    blows up, which is the entire point of having a held-out set.

BUDGETS ARE ABSOLUTE STEP INDICES. `epochs` counts from zero, not from the
resume point, so each arm's budget is written as its archive's step plus the
steps wanted and asserted below. A warm start with a small `epochs` runs zero
steps and reports as a clean run.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'configs'))

import generate                                    # noqa: E402

TAG = 'synth_real_aug19'

#: Both archives verified by reading `modeller_state.step_ind` out of the file,
#: not by trusting the name. Both are T=10, which is what makes a FULL-state
#: resume legal at all: the replay buffer stores T+1-length trajectories, so a
#: resume that changes T dies the first time the replay branch draws (F-046).
UNCOND_ARCHIVE = ('dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92'
                  '_phase1_exit.pt')
UNCOND_STEP = 430
COND_ARCHIVE = ('qm9a13_qm9a98_elj-qm9split_prior-T6.9-b3483b_phase1_exit.pt')
COND_STEP = 18450

#: ~20 minutes each, sized from MEASURED step rates.
#:
#: The conditional budget was wrong the first time and the error is worth
#: recording, because the reasoning was backwards. It was set to 600 steps on
#: the assumption that a live conditioner makes the step SLOWER than the
#: unconditional route. Measured: the conditional route ran 601 steps in 2:12
#: (~4.55 steps/s) against the unconditional route's 901 steps in 15:26
#: (~0.97 steps/s) -- roughly 4.7x FASTER, because at batch 500 it makes no MLIP
#: call and the conditioner is a frozen embedding, while the unconditional arm's
#: ladder was growing its batch the whole time. So the first conditional
#: "20 minute" run was a 2 minute run.
UNCOND_STEPS = 900
COND_STEPS = 5500
#: The level_gap comparison is matched to how far its control actually got
#: before being stopped by hand -- 1096 steps into var_conditioning -- so the
#: two are read over the same window rather than one being given more rope.
COND_MATCHED_STEPS = 1096

#: READ-ONLY. Neither arm needs to write: both resume from an archive that
#: already exists and neither crosses a phase-1 exit, so there is no snapshot
#: worth keeping and a clobbered archive would cost more than the run.
COMMON = dict(continue_from_checkpoint=False, load_weights_only=False,
              prior_model_name=None, checkpoint_read_only=True)


def arms():
    return {
        # ---- the canonical unconditional route ---------------------------
        f'{TAG}_uncond': generate.arm(
            f'{TAG}_uncond', problem='mipcas_elj', tag=TAG,
            checkpoint_name=UNCOND_ARCHIVE,
            epochs=UNCOND_STEP + UNCOND_STEPS, **COMMON),

        # ---- the conditional route ---------------------------------------
        # The problem block selects `protocol: conditional_vargrad`, and with it
        # the three Z settings whose inheritance detonates var_conditioning
        # within ~30 steps (F-042). That is the point of selecting a problem
        # rather than hand-editing keys, and it is why nothing about Z appears
        # in this file.
        f'{TAG}_cond_20min': generate.arm(
            f'{TAG}_cond_20min', problem='qm9_conditional', tag=TAG,
            checkpoint_name=COND_ARCHIVE,
            epochs=COND_STEP + COND_STEPS,
            # PHASE 1 IS RE-ENTERED ON A RESUME, and the first attempt spent its
            # entire budget there. Two things combine:
            #   * `skip_if: prior_loaded` only fires on a FRESH run
            #     (protocol.py:1335, `if m.step_ind != 0: return` -- a resumed
            #     run is left wherever its checkpoint says), so warm-starting
            #     from a phase-1 EXIT archive still re-enters train_prior;
            #   * the stage's `exit` is an AND-list of three terms, so a loose
            #     MLE gate alone cannot release it -- `eval/wass_debiased < 0.015`
            #     and `bwd/tbc < 2.0` are quality bars this route need not meet,
            #     and `wass` is not interpretive on crystal targets anyway.
            # So the whole block is replaced for this arm: one trivially
            # satisfiable term, on a gate configured to call any descent flat.
            # The MLE is ALREADY DONE -- the archive is its exit -- so this is
            # releasing a stage that has nothing left to do, not skipping work.
            # Scoped to this shakeout arm; canonical is untouched.
            **{'protocols.conditional_vargrad.stages[0].mle_gate':
                   {'slope_t': 0.1, 'min_rate': 1.0e9, 'window': 50},
               'protocols.conditional_vargrad.stages[0].exit':
                   [{'metric': 'gates/mle_flat', 'above': 0.5, 'patience': 1}],
               # PINNED TO 0 EXPLICITLY, and it has to be. Canonical adopted
               # level_gap: 1 on the strength of the comparison this arm is the
               # CONTROL for, so leaving it unset would silently make the control
               # identical to the treatment -- an arm that differs by omission is
               # a duplicate, and this one would have looked like a replication.
               'protocols.conditional_vargrad.stages[1].loss_coeffs.bwd.level_gap': 0},
            **COMMON),
        # ---- the level_gap toggle -------------------------------------------
        # SINGLE KEY against `_cond_20min`, and matched in LENGTH to it (the
        # comparison run was stopped by hand at 1096 steps into
        # var_conditioning, so this one gets exactly that budget and no more).
        #
        # WHY: the 72 h cluster run `qm9a13_qm9anchor_aug14_b020_bwd80` carried
        # `bwd.level_gap: 1` at this stage and never showed the vertical loss
        # excursion the local run shows. level_gap adds
        # `gap * (log_r + log_pb - log_pf)` on the bwd branch, where gap is the
        # per-condition Z-level discrepancy clamped to +/-10 nats -- a
        # PROPORTIONAL CONTROLLER on J_B (gflownet_losses.py:514), self-limiting
        # because the gap is re-read each step. It is the only restoring force
        # on the per-condition Z LEVEL, and a Z excursion is the documented
        # driver of this failure, so the hypothesis is that it damps the cascade
        # rather than the initial kick.
        #
        # PREDICTION, written before the run: smaller PEAK `bwd/tb_err_worst`
        # and `zmatch/delta_worst` through the post-transition window, and
        # faster recovery. Bounded, not eliminated -- the clamp caps the force.
        # Read `level_gap_coeff_rms` (rms |gap|) as the convergence signal; the
        # term's own VALUE is `gap * log w`, either sign, and is NOT one.
        # If the peak is unchanged, the tether is not what separates the two
        # runs and the remaining suspects are `bwd.repeats` (2 vs 1, which
        # halves the bwd VarGrad group) and base `bwd.beta` (80 vs 10).
        f'{TAG}_cond_levelgap': generate.arm(
            f'{TAG}_cond_levelgap', problem='qm9_conditional', tag=TAG,
            checkpoint_name=COND_ARCHIVE,
            epochs=COND_STEP + COND_MATCHED_STEPS,
            **{'protocols.conditional_vargrad.stages[0].mle_gate':
                   {'slope_t': 0.1, 'min_rate': 1.0e9, 'window': 50},
               'protocols.conditional_vargrad.stages[0].exit':
                   [{'metric': 'gates/mle_flat', 'above': 0.5, 'patience': 1}],
               # THE ONE VARIED KEY
               'protocols.conditional_vargrad.stages[1].loss_coeffs.bwd.level_gap': 1.0},
            **COMMON),
    }


def check(cfgs):
    """Assertions at generation time, where they cost nothing to fix."""
    budgets = {f'{TAG}_uncond': (UNCOND_ARCHIVE, UNCOND_STEP),
               f'{TAG}_cond_20min': (COND_ARCHIVE, COND_STEP),
               f'{TAG}_cond_levelgap': (COND_ARCHIVE, COND_STEP)}
    for name, cfg in cfgs.items():
        archive, step = budgets[name]
        assert cfg['checkpoint_name'] == archive, name
        # epochs is ABSOLUTE: the budget must clear the archive's own step, or
        # the run trains nothing and says so nowhere
        assert cfg['epochs'] > step + 100, (
            f'{name}: epochs {cfg["epochs"]} against a step-{step} archive')
        assert cfg['continue_from_checkpoint'] is False, name
        assert cfg['checkpoint_read_only'] is True, name
        # THE POINT OF THIS BATTERY: the ladder is live, at canonical settings.
        assert cfg['grow_batch_size'] is True, name
        assert 0 < float(cfg['batch_util_target']) <= 1, (
            f'{name}: batch_util_target {cfg["batch_util_target"]} is not a '
            f'fraction -- state 9 reinterpreted this key')
        assert cfg['figs_period'] % cfg['eval_period'] == 0, name
    for archive, _ in budgets.values():
        p = os.path.join(r'D:\crystal_datasets\gfn_checkpoints', archive)
        assert os.path.exists(p), f'missing archive: {p}'
    print(f'  checks passed on {len(cfgs)} arms')


if __name__ == '__main__':
    cfgs = arms()
    check(cfgs)
    generate.emit(cfgs, outdir=HERE)
