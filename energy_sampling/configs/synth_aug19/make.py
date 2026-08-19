"""
synth_aug19 -- the LOCAL synthesis shakeout for the MLIP fast paths and the
state-8 batch sizer.

WHAT THIS BATTERY IS FOR, and what it cannot do. It runs on the dev box (RTX
5080 laptop, torch.compile OFF on Windows), so it proves CORRECTNESS and
LIVENESS only: that each route loads and steps, that every fast path actually
EXECUTES (the flag metrics report the executed fraction, not the environment
variable), that the sizer's control law behaves, and that a stage transition
crosses cleanly. It proves NOTHING about occupancy levels, throughput rankings,
compile behaviour or OOM ceilings at production batch -- the rollout is
dispatch-bound locally and every registry floor is an A100 number. Those are
questions for the cluster step, and they are written down in RUNSHEET.md rather
than answered here.

    python configs/synth_aug19/make.py

Arms, and the question each answers (the design review is in RUNSHEET.md):

  a1_elj_base       canonical unconditional ELJ, warm-started at the shipped
                    phase1_exit archive. The control for every toggle: does the
                    canonical route still run clean end to end, and does the
                    sizer hold the base batch with no target (S3)?
  a2_mace_fast_eq2      MACE route, acridine sg14/zp1. Do mace_flag_batched_nl,
                    mace_flag_gpu_batch and nl_fastpath_frac all read 1.0, and
                    is mace_host_frac far below the pre-fix 0.68?
  a3_uma_ext_eq2        UMA route, mipcas sg2/zp1. Does uma_flag_external_graph read
                    1.0 (F-047's default), does uma_ext_graph_s populate, and
                    are the energies plausible?
  a4_sizer_ladder   a1 plus batch_util_target: SINGLE-KEY toggle against a1.
                    Does the calibration walk climb, dwell, table per-rung
                    occupancy, and conclude?
  a5_transition_s9     latent_gaussian from step 0, long enough to cross
                    train_prior -> equilibration. Transitions are where the OOM
                    and LR-reset pathology lives.
  a6_cond_elj_s9       QM9-conditional ELJ. Does the conditional route load with
                    the F-042 Z trio travelling with the protocol, and do the
                    held-out eval_test metrics appear?

=============================================================================
THE LOCAL PRIOR TRUNCATION, stated loudly because it changes what a1-a3 measure
=============================================================================
init_prior_dataset re-scores the WHOLE prior dataset through the energy function
before training starts. On the MLIP routes that is 205k (mace) / 176k (uma) rows
through an MLIP, which is hours on this box -- so a2/a3 point at TRUNCATED prior
files (256 / 512 rows) built into the session scratchpad by
`make_truncated_priors.py`.

That is a deliberate scope cut with two consequences, and neither is hidden:
  * these arms measure the PER-CALL energy properties (which path executed, its
    host/graph split, energy plausibility) -- exactly the per-call quantities
    the handoff §3 asks about -- and NOT anything about buffer statistics,
    composition, or the init pass's own memory behaviour;
  * the init OOM transient the handoff §4.4 blames for the cluster arms'
    collapse is precisely what truncation removes, so its absence here is not
    evidence.
a1/a4/a6 use the REAL priors: ELJ is cheap enough to score in full.

=============================================================================
epochs IS AN ABSOLUTE STEP INDEX
=============================================================================
A warm start at step 430 with `epochs: 200` runs ZERO steps and proves nothing
(this has happened). Every warm arm's budget is therefore written as
ARCHIVE_STEP + n and asserted below, and the executed step count is verified
from the run afterwards, not assumed.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # energy_sampling/
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'configs'))

import generate                                    # noqa: E402
import tierc_smoke                                 # noqa: E402

TAG = 'synth_aug19'

CKPT_DIR = r'D:\crystal_datasets\gfn_checkpoints'
#: the shipped ELJ warm start named by configs/mk_dev.yaml, and its step index
#: read from the archive itself (modeller_state.step_ind) rather than assumed
ELJ_ARCHIVE = ('dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92'
               '_phase1_exit.pt')
ELJ_ARCHIVE_STEP = 430

#: Truncated local MLIP priors -- see the module docstring. Session scratchpad,
#: NOT a project artifact: these are fixtures for one shakeout, and a battery
#: that quietly grew a permanent dependency on a 256-row prior would be worse
#: than one that says so.
SCRATCH = (r'C:\Users\mikem\AppData\Local\Temp\claude'
           r'\C--Users-mikem-Projects-mxt-gfn-gfn-diffusion-energy-sampling'
           r'\785135ac-7125-4e5c-9e58-c72c507aecb7\scratchpad')
MACE_PRIOR = os.path.join(SCRATCH, 'LOCAL_acridine_sg14_zp1_mace_prior_256.pt')
UMA_PRIOR = os.path.join(SCRATCH, 'LOCAL_mipcas_sg2_zp1_uma_prior_512.pt')
MACE_MODEL = r'D:\crystal_datasets\acr_112025_mh1_stagetwo.model'
UMA_MODEL = r'D:\crystal_datasets\esen_s.pt'

#: LOCAL cadence, not the cluster's. mk_dev ships eval 250 / figs 500 against
#: 100k epochs; these arms are hundreds of steps, so an inherited cadence would
#: either never fire or spend the whole arm in eval.
LOCAL_EVAL = dict(eval_period=50, figs_period=100, archive_period=100000,
                  eval_num_samples=1000)

#: READ-ONLY BY DEFAULT. Only a5 needs to write: it crosses a phase-1 exit,
#: whose on_exit snapshot is part of the machinery under test.
READ_ONLY = dict(checkpoint_read_only=True)

#: Pinned resume, every arm. The mk_dev defaults resolve to this run identity's
#: own _running.pt, which for a fresh run_name silently means "retrain phase 1"
#: -- a different wrong answer, equally invisible in the results.
FRESH = dict(checkpoint_name=None, continue_from_checkpoint=False,
             prior_model_name=None)
WARM = dict(checkpoint_name=ELJ_ARCHIVE, continue_from_checkpoint=False,
            load_weights_only=False, prior_model_name=None)

#: The occupancy ladder OFF on every arm but a4. The canonical config now ships
#: it armed (target 60), so a4-vs-a1 is a single-key toggle only if a1 turns it
#: off explicitly -- otherwise both arms calibrate and the comparison measures
#: nothing.
LADDER_OFF = dict(batch_util_target=0, grow_batch_size=False,
                  max_batch_size=1000)

#: The occupancy target as a FRACTION of the card (state 9). The canonical
#: config ships 0.6; the arm that exercises the ladder says so explicitly.
LADDER_TARGET = 0.6


def terminal_equilibration():
    """The canonical `equilibration` stage as the run's ONLY stage.

    WHY THE MLIP ARMS NEED THIS, measured rather than assumed. The first
    attempt ran them fresh on the full `unconditional_tb` protocol, so they
    spent their whole budget in `train_prior` -- a bwd/dataset MLE stage that
    makes NO energy call inside a training step. The result: 5-10 MLIP calls
    for the entire run, all of them from the init prior re-analysis and eval,
    and `energy/frac_of_step` at exactly 0. The executed-path flags were still
    readable (the calls happened), but nothing measured the MLIP where the
    handoff asks about it, which is the fused training step.

    This is also the F-046-safe shape for a fresh MLIP arm: one terminal stage
    from step 0, so the replay buffer fills at the run's own T rather than
    inheriting a trajectory length fixed at some archive's write time.

    The stage is TAKEN FROM CANONICAL, not retyped, so it cannot drift from the
    route it is supposed to represent -- with the two entry actions dropped,
    since `rebuild_prior_by_churn` would push the whole prior through the MLIP
    again at entry and `bootstrap_z` has no phase-1 product to bootstrap from.
    """
    stage = None
    for s in generate.canonical()['protocols']['unconditional_tb']['stages']:
        if s['name'] == 'equilibration':
            stage = {k: v for k, v in s.items() if k != 'on_enter'}
    assert stage is not None, 'canonical has no equilibration stage'
    return [stage]


def arms():
    out = {}

    # ---- a1: the canonical route, and the control for a4 --------------------
    out[f'{TAG}_a1_elj_base'] = generate.arm(
        f'{TAG}_a1_elj_base', problem='mipcas_elj', tag=TAG,
        epochs=ELJ_ARCHIVE_STEP + 300,
        **{**WARM, **LOCAL_EVAL, **READ_ONLY, **LADDER_OFF})

    # ---- a2: MACE with the fast paths as in-code defaults -------------------
    # FRESH, single terminal stage: no MACE archive exists locally, and a warm
    # start whose T differs from the archive's dies on the replay buffer's
    # stored trajectory length (F-046). Fresh at the run's own T is immune.
    # BATCH 2, on the owner's instruction: acridine sg14 through MACE does not
    # admit 100 on this card.
    out[f'{TAG}_a2_mace_fast_eq2'] = generate.arm(
        f'{TAG}_a2_mace_fast_eq2', problem='mipcas_elj', tag=TAG,
        energy_function='mace', mlip_path=MACE_MODEL,
        space_groups=[14], z_primes=[1],
        prior_path=MACE_PRIOR, molecules_path=MACE_PRIOR,
        # MEASURED, not chosen: batch 2 with traj_checkpoint false OOM'd in the
        # FUSED stage on this card -- 4.58 GiB short -- and then cascaded,
        # because the shrink path reaches batch 1 and gives up. The bwd/MLE
        # stage never showed it: that stage makes no in-step energy call.
        # traj_checkpoint is the documented lever (trajectory activation memory
        # O(1) in T instead of O(T), values bitwise identical), and batch 1 is
        # the floor. NB this is a LOCAL memory accommodation and says nothing
        # about the A100's ceiling.
        batch_size=1, fused_grad_accum_min_samples=1, traj_checkpoint=True,
        epochs=120, protocol='unconditional_tb',
        **{**FRESH, **READ_ONLY,
           **dict(LOCAL_EVAL, eval_period=50, figs_period=100,
                  eval_num_samples=20),
           **dict(LADDER_OFF, max_batch_size=1),
           'protocols.unconditional_tb.stages': terminal_equilibration()})

    # ---- a3: UMA with the external graph default (F-047) --------------------
    out[f'{TAG}_a3_uma_ext_eq2'] = generate.arm(
        f'{TAG}_a3_uma_ext_eq2', problem='mipcas_elj', tag=TAG,
        energy_function='uma', mlip_path=UMA_MODEL,
        prior_path=UMA_PRIOR, molecules_path=UMA_PRIOR,
        # traj_checkpoint for the same reason as a2 -- the fused stage
        # backpropagates through T MLIP calls on supercells
        batch_size=2, fused_grad_accum_min_samples=2, traj_checkpoint=True,
        epochs=120, protocol='unconditional_tb',
        **{**FRESH, **READ_ONLY,
           **dict(LOCAL_EVAL, eval_period=50, figs_period=100,
                  eval_num_samples=20),
           **dict(LADDER_OFF, max_batch_size=2),
           'protocols.unconditional_tb.stages': terminal_equilibration()})

    # ---- a4: the sizer ladder, one key off a1 -------------------------------
    # PREDICTION (written before the run, and sized against a1's MEASURED
    # 0.70 s/step at batch 1000): the walk climbs from 1000 in capped-geometric
    # rungs (1000 -> 1600 -> 2600 -> 4160 -> ...). Each rung needs BOTH a
    # 50-step dwell and >=3 raw occupancy samples at the 60 s sample period, so
    # the SAMPLE requirement binds -- ~180 s a rung, i.e. ~257 steps at the base
    # rung and fewer as the step time grows with batch. 900 steps should
    # therefore buy 3-4 rungs. Expect either 'target_met' or an INFEASIBLE
    # conclusion naming max_batch_size, the per-rung table populated with
    # (batch, med_s, util, n_util), and the batch NEVER below 1000, since
    # nothing walks downward.
    out[f'{TAG}_a4_sizer_ladder'] = generate.arm(
        f'{TAG}_a4_sizer_ladder', problem='mipcas_elj', tag=TAG,
        epochs=ELJ_ARCHIVE_STEP + 900,
        batch_util_target=LADDER_TARGET, grow_batch_size=True,
        max_batch_size=8000,
        # 50 steps is the shipped local dwell; the sample period is what
        # actually gates a rung here (3 samples at 60 s), so leave both alone
        # and let the run show which binds.
        **{**WARM, **LOCAL_EVAL, **READ_ONLY})

    # ---- a5: a stage transition, and the only arm that writes ---------------
    # latent_gaussian from step 0: deterministic, cheap, and its phase-1 exit
    # fires inside a few hundred steps (measured at 381 on the tier-C pair).
    gap_fill, gap_notes = tierc_smoke.problem_gap_fill('latent_gaussian')
    for note in gap_notes:
        print(f'  a5 gap-fill: {note}')
    out[f'{TAG}_a5_transition_s9'] = generate.arm(
        f'{TAG}_a5_transition_s9', problem='latent_gaussian', tag=TAG,
        epochs=900, batch_size=300, fused_grad_accum_min_samples=300,
        **{**FRESH, **LOCAL_EVAL, **LADDER_OFF,
           **dict(max_batch_size=300),
           # WRITES, deliberately: on_exit ['snapshot:phase1_exit',
           # 'snapshot_prior'] is part of what this arm verifies.
           'checkpoint_read_only': False,
           **gap_fill})

    # ---- a6: the conditional route -----------------------------------------
    out[f'{TAG}_a6_cond_elj_s9'] = generate.arm(
        f'{TAG}_a6_cond_elj_s9', problem='qm9_conditional', tag=TAG,
        epochs=300, batch_size=500, fused_grad_accum_min_samples=500,
        **{**FRESH, **LOCAL_EVAL, **READ_ONLY,
           **dict(LADDER_OFF, max_batch_size=500),
           'eval_num_samples': 1000, 'test_eval_num_samples': 500})

    return out


def check(cfgs):
    """Assertions at GENERATION time, where they cost nothing to fix."""
    for name, cfg in cfgs.items():
        # epochs is an ABSOLUTE index: a warm arm must budget past its archive
        if cfg.get('checkpoint_name') == ELJ_ARCHIVE:
            assert cfg['epochs'] > ELJ_ARCHIVE_STEP + 50, (
                f'{name}: epochs {cfg["epochs"]} against a step-'
                f'{ELJ_ARCHIVE_STEP} archive runs ~nothing')
        # the resume point is PINNED, never left to the _running.pt default
        assert cfg.get('continue_from_checkpoint') is False, name
        # MLIP arms carry a model path; non-MLIP arms must not
        assert (cfg.get('mlip_path') is None) == \
               (cfg['energy_function'] not in ('uma', 'mace')), name
        # the ladder is armed on exactly one arm, or a4 is not a single-key toggle
        armed = float(cfg.get('batch_util_target') or 0) > 0
        assert armed == name.endswith('a4_sizer_ladder'), (
            f'{name}: batch_util_target armed={armed}; only a4 may arm it')
        # only a5 writes checkpoints
        assert cfg.get('checkpoint_read_only') is (not name.endswith(
            'a5_transition_s9')), name
        # figs must be a multiple of eval, or no figure is ever logged
        assert cfg['figs_period'] % cfg['eval_period'] == 0, name
    # every fixture the arms name must exist NOW, not at launch
    for path in (os.path.join(CKPT_DIR, ELJ_ARCHIVE), MACE_PRIOR, UMA_PRIOR,
                 MACE_MODEL, UMA_MODEL):
        assert os.path.exists(path), f'missing input: {path}'
    print(f'  checks passed on {len(cfgs)} arms')


if __name__ == '__main__':
    cfgs = arms()
    check(cfgs)
    generate.emit(cfgs, outdir=HERE)
