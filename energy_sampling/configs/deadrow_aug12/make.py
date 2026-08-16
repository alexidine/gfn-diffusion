"""
deadrow_aug12 -- smoke + functionality battery for the dead-latent-row change. 2026-08-12.

WHAT THIS IS. A confidence battery for docs/decisions.md D33: latent rows that
`enforce_crystal_system` overwrites with a constant are now held out of the SDE.
`test_dead_latent_rows.py` proves the invariants on CPU, bitwise; this battery
proves the same thing END TO END on real problems, on GPU, through the training
loop -- the layer unit tests cannot reach.

WHAT THIS IS NOT. Not a convergence study. No arm is expected to reach a good
log Z, and none should be read as a science result. The question is only "did we
do what we intended, and did we break anything -- in code or statistically".
Budgets are sized for one overnight pass, not for asymptotes.

=============================================================================
THE RISK BEING TESTED
=============================================================================
A wrong index set here does NOT crash. It yields a model that trains, converges
and reports a plausible but WRONG log Z. So every arm below is paired with a tell
in verify.py that would move if the change misfired, and the two structural
controls (A and D) are the ones that matter most: they must show NO change.

  A  sg 2, triclinic, n_dead = 0.  THE CONTROL.
     Triclinic is enforce_crystal_system's "anything goes" branch, so nothing is
     dead and the knob must be inert. a_sg2_on and a_sg2_off differ ONLY in
     hold_dead_latent_rows, so their metric streams must agree to numerical
     noise. (Bitwise equality is proven on CPU in test_dead_latent_rows.py; on
     GPU, cuDNN/atomic nondeterminism puts a floor under this, so read it as
     "indistinguishable", not "identical".) Any systematic gap means the knob is
     doing something on a space group where it has nothing to do.

     These two arms RESUME mk_dev's pinned phase-1 exit. That is deliberate:
     for triclinic the resolved dead rows are () and a pre-change checkpoint
     stores nothing, so _assert_dead_rows_match passes -- which makes this arm
     also the end-to-end test that COMPATIBLE resumes still load.

  B  sg 14, monoclinic, n_dead = 2 (alpha, gamma).  THE PRIMARY ARM.
     b_sg14_on holds them; b_sg14_off is the same problem with the old
     behaviour. This is the only pair that can measure the fix, and it is the
     measurement F-009's tb_err/fwd-bwd-gap CONJECTURE needs. Both train from
     scratch: their expanded_dim differs (14 vs 16), so they cannot share a
     checkpoint, and no pre-change sg-14 checkpoint is loadable any more by
     design.

  C  sg 9, monoclinic, Z' = 2.  THE INTERACTION ARM.
     Two things only this arm reaches: dead rows at Z' > 1 (the dead angle rows
     are 3 and 5 regardless of Z', while the state grows to 18), and the
     zp_order_penalty, which indexes raw_latents[:, 6:6+3k] by ABSOLUTE
     position. That penalty is exactly what the "states stay full width" design
     was chosen to protect, so this arm is the test of that choice. sg 9 also
     carries 2 FREE centroid axes, which this change does NOT yet handle --
     see the NOT-COVERED note below.

  D  toy latent_multiharmonic, sg 1 placeholder.  THE GATE CONTROL.
     Toys carry space_groups: [1] as a PLACEHOLDER. P1 is triclinic so the
     angle rows are empty today and this looks safe either way -- but P1 has all
     three aunit centroid axes FREE, so once free axes join the table an
     ungated resolver would freeze three dims a toy genuinely uses. This arm
     pins the is_crystal gate now, while the failure is still cheap.

  E  conformer (TorsionGFN).  THE REGRESSION ARM.
     Adding the dead-row kwargs to GFN.get_periodic_dimensions broke every
     conformer run at construction -- the subclass overrides that method. Found
     by adversarial review AFTER the unit suite was green, because the suite had
     no subclass coverage. This arm exists so that class of break cannot recur
     silently.

=============================================================================
NOT COVERED -- read before trusting a clean result
=============================================================================
  * ORTHORHOMBIC (n_dead = 3) has NO prior on disk. Unit-tested only. To add it,
    generate one first:
        python data_processing/generate_sg_prior.py   (see its args for sg 19)
    Then copy arm B with space_groups: [19]. sg 19 is the most common Sohncke
    group, so this gap is worth closing before any chiral-molecule work.
  * FREE AUNIT AXES are now implemented, but ONLY AT Z'=1, and NO ARM COVERS THEM.
    canonicalize_free_axes pins them to the aunit box centre (latent 0) and they
    join the dead-row table -- verified energy- and RDF-invariant on physical
    structures: energy-invariant to <=1.2e-06 relative on 40 structures per space group.
    (The RDF is cutoff-sensitive and NOT a reliable witness -- findings.md F-010b.)
    But every polar/Sohncke Z'=1 group needs a prior that does not exist on disk,
    so this battery exercises the free-axis path in UNIT TESTS ONLY. sg 9's two
    free axes stay live in arm C because that arm is Z'=2, where the mechanism is
    deliberately gated off: the free translation there is one GLOBAL shift, so
    fixing it needs a common offset, and the leftover units then fall outside the
    box where re-wrapping one unit by auv_d is a symmetry only if auv_d == 1.
    Closing this needs a Z'=1 polar prior (sg 4 or sg 9); generate one and copy
    arm B. UNTIL THEN THE FREE-AXIS WORK HAS NO END-TO-END COVERAGE.
  * HEXAGONAL would trip the startup canonical-value assert by design
    (gamma = 2pi/3 does not map to latent 0). Deliberate, unreachable from any
    config here.
  * CONVERGENCE. Nothing here runs long enough to say a space group trains well.

=============================================================================
CONVENTIONS
=============================================================================
Derived from mk_dev.yaml, which is USER-OWNED and is never written by this
generator. Every arm states its own knob rather than inheriting it, because a
value written by OMISSION is mk_dev's default and would silently make two arms
duplicates (project_arms_written_by_omission_are_duplicates).

    python configs/deadrow_aug12/make.py

Launch order and the one manual negative test are in launch.txt.
"""
import copy
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIGS = HERE.parent
MK_DEV = CONFIGS / 'mk_dev.yaml'
CONFORMER_DEV = CONFIGS / 'conformer_dev.yaml'
TOY_BASE = CONFIGS / 'blocks_aug11' / 'base.yaml'

TAG = 'deadrow0812'
PRIOR_DIR = r'D:\crystal_datasets\conditional\priors'

# Smoke budgets. Arms A resume a phase-1 exit so their steps are all useful;
# arms B/C/D start cold and spend most of the budget in phase 1, which is itself
# part of the test (does MLE run with dead rows held?).
RESUME_STEP = 6680          # mk_dev's pinned phase-1 exit step
WARM_BUDGET = 900           # arms A: steps past the resume point
COLD_BUDGET = 2500          # arms B/C: from scratch, phase 1 included
TOY_BUDGET = 1500           # arm D: toys are cheap

EVAL_PERIOD = 150           # ~6-16 eval points per arm at these budgets
FIGURE_PERIOD = 600         # keep uploads small (feedback_figure_parsimony)

MK_DEV_PHASE1_EXIT = ('dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset'
                      '-T2.5-573c92_phase1_exit.pt')


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def toy_base():
    with TOY_BASE.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def common(cfg, name, budget):
    cfg['run_name'] = name
    cfg['tag'] = TAG
    cfg['epochs'] = budget
    cfg['eval_period'] = EVAL_PERIOD
    cfg['figure_period'] = FIGURE_PERIOD
    cfg['wandb_mode'] = 'online'      # smoke runs still get read; no offline arms

    # PINNED, not inherited. base() reads mk_dev.yaml live and mk_dev is user-owned:
    # it moved between this battery's generation (condition_block_m 1) and the next
    # regeneration (2), which would have silently changed a RULER on logw_std_within
    # under arms whose whole job is an A/B. Anything this battery must hold fixed is
    # written here explicitly -- the same rule as the by-omission trap documented
    # above, applied to drift over time rather than across arms.
    #
    # Both batch keys are INERT here (grow_batch_size is false, so
    # increment_batch_size is never called; and at batch 1000 an elj/toy energy is a
    # single chunk either way). They are pinned so the battery stays schema-current
    # and so flipping grow_batch_size later does not silently run without a ceiling.
    cfg['max_step_seconds'] = 10
    cfg.setdefault('energy_config', {})['internal_oom_recovery'] = False
    cfg.setdefault('buffers', {}).setdefault('prior_buffer', {})['condition_block_m'] = 1
    return cfg


# NB `reuse_prior` used to be written False by both helpers below, defensively. It is a
# RETIRED key, so preflight_config hard-raised on every arm and this battery could not
# have started -- caught by loading all six configs, not by generating them. The intent
# it encoded (never auto-refind a prior by run_name) is now the code's only behaviour:
# checkpoint_name/prior_model_name are the sole ways in, and both helpers set them
# explicitly. assert_start_pinned's reuse_prior check goes with it.
def fresh_start(cfg):
    """From scratch. Explicit, because the mk_dev defaults do something else."""
    cfg['checkpoint_name'] = None
    cfg['continue_from_checkpoint'] = False
    cfg['load_weights_only'] = False
    return cfg


def warm_start(cfg, checkpoint):
    cfg['checkpoint_name'] = checkpoint
    cfg['continue_from_checkpoint'] = False
    cfg['load_weights_only'] = False
    return cfg


def set_problem(cfg, sg, zp, prior_basename, energy='elj'):
    path = os.path.join(PRIOR_DIR, prior_basename + '.pt')
    cfg['space_groups'] = [int(sg)]
    cfg['z_primes'] = [int(zp)]
    cfg['energy_function'] = energy
    cfg['prior_path'] = path
    cfg['molecules_path'] = path
    return cfg


def set_knob(cfg, hold):
    cfg.setdefault('model', {})['hold_dead_latent_rows'] = bool(hold)
    return cfg


# ---------------------------------------------------------------- assertions
def assert_start_pinned(cfg, name):
    """
    Mirror of size_aug03's assert_pinned_resume, for a battery whose arms mostly
    start COLD. The mk_dev defaults resolve checkpoint_name: null +
    continue_from_checkpoint: true to '{tag}_{run_name}_..._running.pt', so an arm
    that forgets to override them silently picks up its own rolling checkpoint on a
    relaunch instead of restarting -- a different wrong answer, and invisible.
    Either a checkpoint is NAMED, or the start is explicitly cold. Never ambiguous.
    """
    named = bool(cfg.get('checkpoint_name'))
    if cfg.get('continue_from_checkpoint'):
        raise ValueError(f'{name}: continue_from_checkpoint must be False')
    if 'reuse_prior' in cfg:
        raise ValueError(f'{name}: reuse_prior is a RETIRED key -- preflight_config will '
                         f'hard-raise on it at load. Remove it, do not set it False.')
    if not named and cfg.get('checkpoint_name') is not None:
        raise ValueError(f'{name}: checkpoint_name must be a real name or explicitly null')
    return cfg


def assert_knob_explicit(cfg, name):
    if 'hold_dead_latent_rows' not in (cfg.get('model') or {}):
        raise ValueError(f'{name}: hold_dead_latent_rows must be stated explicitly -- '
                         f'omitting it inherits the code default and makes the A/B a duplicate')
    return cfg


def assert_distinct(arms):
    """Two arms that serialise identically are the same experiment run twice."""
    seen = {}
    for name, cfg in arms.items():
        body = dict(cfg)
        body.pop('run_name', None)
        key = yaml.safe_dump(body, sort_keys=True)
        if key in seen:
            raise ValueError(f'{name} is byte-identical to {seen[key]} apart from run_name')
        seen[key] = name


def assert_prior_exists(cfg, name, required):
    p = cfg['prior_path']
    if not os.path.exists(p):
        msg = f'{name}: prior not found at {p}'
        if required:
            raise FileNotFoundError(msg + ' -- run prep_priors.py first')
        print(f'  WARNING {msg} (arm cannot run)')
    return cfg


# ---------------------------------------------------------------------- arms
def arm_sg2(hold):
    """A: triclinic control. Resumes mk_dev's exit -- also tests compatible resume."""
    name = f"a_sg2_{'on' if hold else 'off'}"
    cfg = common(base(), name, RESUME_STEP + WARM_BUDGET)
    warm_start(cfg, MK_DEV_PHASE1_EXIT)
    set_knob(cfg, hold)
    return name, cfg


def arm_sg14(hold):
    """B: monoclinic, the primary pair. Cold: expanded_dim differs between arms."""
    name = f"b_sg14_{'on' if hold else 'off'}"
    cfg = common(base(), name, COLD_BUDGET)
    fresh_start(cfg)
    set_problem(cfg, 14, 1, 'deadrow10k_sg14_zp1_elj')
    set_knob(cfg, hold)
    return name, cfg


def arm_sg9_zp2():
    """C: Z'=2 monoclinic. Reaches zp_order_penalty's absolute raw_latents slice."""
    name = 'c_sg9_zp2_on'
    cfg = common(base(), name, COLD_BUDGET)
    fresh_start(cfg)
    set_problem(cfg, 9, 2, 'deadrow10k_sg9_zp2_elj')
    set_knob(cfg, True)
    return name, cfg


def arm_toy():
    """D: is_crystal gate. sg 1 here is a PLACEHOLDER, not a crystal."""
    name = 'd_toy_on'
    cfg = common(toy_base(), name, TOY_BUDGET)
    fresh_start(cfg)
    set_knob(cfg, True)
    return name, cfg


def main():
    arms = {}
    for hold in (True, False):
        n, c = arm_sg2(hold)
        arms[n] = c
    for hold in (True, False):
        n, c = arm_sg14(hold)
        arms[n] = c
    for fn in (arm_sg9_zp2, arm_toy):
        n, c = fn()
        arms[n] = c

    required = {'a_sg2_on', 'a_sg2_off', 'b_sg14_on', 'b_sg14_off'}
    for name, cfg in arms.items():
        assert_start_pinned(cfg, name)
        assert_knob_explicit(cfg, name)
        assert_prior_exists(cfg, name, required=name in required)
    assert_distinct(arms)

    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, width=120)
        print(f'  wrote {name}.yaml')

    rows = [
        ('a_sg2_on', RESUME_STEP + WARM_BUDGET,
         'CONTROL. sg 2 triclinic, n_dead=0, knob ON. Resumes mk_dev phase-1 exit, so it also '
         'proves a COMPATIBLE pre-change resume still loads (stored None vs resolved ()).'),
        ('a_sg2_off', RESUME_STEP + WARM_BUDGET,
         'CONTROL PARTNER. Identical but knob OFF. Must be indistinguishable from a_sg2_on to '
         'numerical noise -- a systematic gap means the knob acts where nothing is dead. '
         'Bitwise equality is the CPU unit test; on GPU expect nondeterminism-floor agreement.'),
        ('b_sg14_on', COLD_BUDGET,
         'PRIMARY. sg 14 monoclinic, 2 dead rows held. Tell: startup prints dead rows (3,5) and '
         'the probe confirms them; live_dim 10 of 12.'),
        ('b_sg14_off', COLD_BUDGET,
         'PRIMARY A/B. Same problem, old behaviour. THE measurement F-009 needs: does holding the '
         'rows change the fwd/bwd gap and the tb_err floor? Compare at matched step, not at '
         'convergence -- neither arm converges here.'),
        ('c_sg9_zp2_on', COLD_BUDGET,
         "INTERACTION. sg 9 monoclinic Z'=2: dead rows stay (3,5) while the state is 18 wide, and "
         'this is the only arm that reaches zp_order_penalty\'s absolute raw_latents[:, 6:6+3k] '
         'slice. Its 2 free centroid axes are NOT handled and stay live.'),
        ('d_toy_on', TOY_BUDGET,
         'GATE CONTROL. latent_multiharmonic with the sg 1 placeholder. Must resolve to NO dead '
         'rows via is_crystal. Tell: startup prints the non-crystal line, never a dead-row list.'),
        ('e_conformer', 'see launch.txt',
         'REGRESSION. TorsionGFN construction, which this change broke outright and the unit suite '
         'missed. Run configs/conformer_dev.yaml unchanged -- if it reaches step 1 the subclass '
         'delegation works. NOT generated here: conformer_dev.yaml is user-owned.'),
    ]
    (HERE / 'INDEX.tsv').write_text(
        'name\tbudget\tasks\n' + ''.join(f'{n}\t{b}\t{a}\n' for n, b, a in rows),
        encoding='utf-8')
    print(f'  wrote INDEX.tsv ({len(rows)} rows)')
    print()
    print(f'{len(arms)} arms generated. Run prep_priors.py first; see launch.txt.')


if __name__ == '__main__':
    main()
