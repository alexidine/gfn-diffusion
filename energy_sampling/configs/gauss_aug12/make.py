"""
gauss_aug12 -- latent_gaussian dead-row battery. 2026-08-12.

WHAT THIS IS. The first END-TO-END CORRECTNESS test of D33 (dead latent rows held
out of the SDE). Every arm has a CLOSED-FORM log Z, so the question is not "does it
look healthy" but "did it land on the number". `deadrow_aug12` is the smoke battery
on physical energies; this is the one that can be right or wrong.

WHY A NEW ENERGY WAS NEEDED. Physical energies have never converged perfectly, so
they cannot certify a log Z. `latent_gaussian` is a real crystal parameterization
(is_crystal: dead rows, periodic angles, the box) scored by an analytic gaussian on
the latent (latent_energy: no packing, no pressure, no mol_energy, and structurally
no reduction and no jacobian). That combination did not previously exist -- the old
toys were is_crystal FALSE, so they could not exercise dead rows at all.

WHAT IT ALSO CLOSES. deadrow_aug12's own NOT-COVERED list says it cannot reach
orthorhombic (no prior on disk) and covers FREE AUNIT AXES in unit tests only,
because a physical prior must be a real crystal. Nothing builds a cell here, so any
space group can be synthesised -- arms C, D and E close all three gaps.

=============================================================================
THE PREDICTION  (spec.py owns the arithmetic; verified on CPU before generating)
=============================================================================
  rows HELD:  log Z = (n_live/2) log(2 pi T) + n_live log w
  rows LIVE:  log Z = <above> + n_dead * log(2 + sqrt(pi/k))

  sg  dead        n_live   HELD       LIVE      delta
   2  ()            12   -16.6038  -16.6038   +0.0000   <- control, knob inert
  14  (3,5)         10   -13.8365  -11.1810   +2.6555
  19  (3,4,5)        9   -12.4528   -8.4696   +3.9832
   4  (3,5,7)        9   -12.4528   -8.4696   +3.9832
   1  (6,7,8)        9   -12.4528   -8.4696   +3.9832

The `delta` column is the FICTITIOUS VOLUME D33 removes. It is NOT n_dead*log 2:
the box wall is soft (quadratic, zero-slope onset), so the reachable volume per
live-but-dead row is 2 + sqrt(pi/k), which at k=1 is 3.77 -- the leak is nearly as
big as the box. The log-2 form was the first prediction and it is wrong by
+0.63/dim at k=1. Measured across a 20x sweep of k in findings.md F-011.

  ARM A  sg 2, triclinic, n_dead = 0.  THE CONTROL.
      Triclinic is enforce_crystal_system's "anything goes" branch, so the dead set
      is empty and the knob must be INERT. On CPU the two arms are bitwise identical
      (_pin_dead returns the same object when there is nothing to pin); on GPU read
      it as indistinguishable. A systematic gap means the knob acts where it has
      nothing to do.
  ARM B  sg 14, monoclinic, dead (3,5).  THE PRIMARY PAIR.
      Two clobbered angles. Smallest non-trivial signal, and the most common
      space group in the CSD.
  ARM C  sg 19, orthorhombic, dead (3,4,5).  LARGEST ANGLE SIGNAL.
      3 of 12 rows dead. Most common Sohncke group; had no prior on disk before now.
  ARM D  sg 4, monoclinic polar, dead (3,5,7).  THE FREE-AXIS ARM.
      2 clobbered angles + 1 FREE AUNIT AXIS. Must land on the SAME log Z as arm C
      by a DIFFERENT mechanism -- canonicalize_free_axes inside latent_params,
      not enforce_crystal_system inside latent_to_cell_params. If C agrees with
      theory and D does not, the free-axis half of D33 is wrong; nothing else in
      the battery separates those two code paths.
  ARM E  sg 1, P1, dead (6,7,8).  PURE FREE AXIS, NO ANGLE ROWS.
      The only arm where every dead row is a free axis. P1 is also the space group
      whose three free axes are the reason resolve_dead_rows is gated on is_crystal
      (toys carry space_groups: [1] as a placeholder), so this arm is the positive
      case of that gate: here sg 1 IS a crystal and the rows SHOULD be held.

=============================================================================
WHAT IS PINNED, AND WHY IT MATTERS HERE
=============================================================================
  periodic_centroids: FALSE on every arm, and not negotiable. Arms D and E have free
      centroid axes; 26 space groups have an axis that is both free (dead) and
      auv == 1 (angular). With centroid wrapping ON, a live-but-dead axis in the
      rows-LIVE arm would be wrapped onto a period of exactly 2 instead of diffusing
      against the soft wall, so its fictitious volume would be log 2, not
      log(2 + sqrt(pi/k)), and arms D/E OFF would miss the prediction for a reason
      having nothing to do with D33.
  reward_range: null. Inert today (set_reward_clip has no callers, so energy_clip
      stays None) but log_rescale_positive would deform the target if it were ever
      wired up, and a deformed target has no closed form.
  temperature_conditioning: false, temperature 1.0. T enters log Z directly as
      (n_live/2) log(2 pi T); a sampled T would make the target a mixture.
  bounding_coeff 1.0 -- deliberately SOFT. It maximises the rows-live penalty, so
      the A/B signal is as large as this energy can produce.

Derived from mk_dev.yaml, which is USER-OWNED and never written by this generator.
Every knob the battery depends on is written explicitly rather than inherited: a
value written by OMISSION is mk_dev's live default and can drift between
regenerations, which would silently change a ruler under an A/B
(project_arms_written_by_omission_are_duplicates).

    python configs/gauss_aug12/prep_prior.py      # first -- writes the priors
    python configs/gauss_aug12/make.py

Launch order is in launch.txt.
"""
import argparse
import copy
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIGS = HERE.parent
MK_DEV = CONFIGS / 'mk_dev.yaml'
for p in (r'C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion',
          r'C:\Users\mikem\Projects\mxt_gfn\mxtaltools'):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(HERE))

import spec  # noqa: E402

# The SHIPPING stage resolver, not a local copy. `protocol` became a SELECTOR
# naming an entry in `protocols` (config_state state 4), so the old
# `cfg['protocol']['stages']` reads a string here and dies; keeping a private
# copy of the new lookup would just queue up the same break at state 6.
from energy_sampling.config_invariants import active_stages  # noqa: E402

# Budgets. Every arm starts COLD: expanded_dim differs between held and live
# (e.g. sg 19: 21 vs 24 with 3 angular rows), so no two arms can share a
# checkpoint, and no pre-D33 checkpoint is loadable by design.
BUDGET = 4000
EVAL_PERIOD = 150
FIGURE_PERIOD = 600
BATCH = 1000
TRAJ_T = 10

# ------------------------------------------------------- co-tenancy (--cotenants N)
# This toy is the one problem here where several arms genuinely can share the card:
# the target is a 9-12 dim gaussian, the energy is analytic (no cluster, no unit cell)
# and the model is grossly over-specified for it. Sharing is opt-in, and the numbers
# below are what makes the claim COHERENT rather than merely asserted -- gpu_guard
# refuses `cotenants=N` whose per-arm ceiling does not actually divide the card.
CARD_MB = 16303          # RTX 5080 Laptop, measured
# Desktop compositing load (Chrome/Slack/Teams/explorer). A FALLBACK only: this is not
# a constant. Hardcoding 1800 sized the T=25 pair for a desktop that was actually using
# ~3160 MiB, so 2 x 6144 + 3160 = 15448 of 16303 left 855 MiB -- inside the margin, and
# gpu_guard correctly refused the second arm at launch. Prefer a LIVE reading; the
# hardcoded value is only for when nvidia-smi will not answer.
DESKTOP_MB_FALLBACK = 2600


def desktop_mb():
    """
    Measured non-training GPU load right now, or the fallback. Subtracting a stale
    guess is how a co-tenancy plan ends up one arm too wide.
    """
    try:
        sys.path.insert(0, str(CONFIGS.parent))
        from gpu_guard import gpu_memory, training_processes
        mem = gpu_memory()
        if mem and not training_processes():
            return int(mem[0])          # nothing training: used IS the desktop
    except Exception:
        pass
    return DESKTOP_MB_FALLBACK


DESKTOP_MB = desktop_mb()

# What a co-tenant arm gives up, and why each is acceptable HERE and nowhere else:
#
#   buffer_device: cpu    Buffers are the prime suspect for the ~11 GB a full-size arm
#                         holds (prior 250k / anchor 200k caps, resident on cuda). The
#                         cost is a host->device copy per draw, which for a 10-atom
#                         toy graph is negligible. NOT safe to assume for a physical
#                         run, where the buffer rows are large and drawn every step.
#   buffer caps cut       250k/200k are sized for physical campaigns; the toy's whole
#                         prior dataset is 20000 rows, so those caps can never be
#                         approached and only reserve address space.
#   eval_num_samples      The measurement precision, so this is the one to guard. The
#                         emp_z noise floor is sqrt(Var/N): at Var~0.1, N=10000 gives
#                         0.003 and N=2500 gives 0.007 -- both well inside the ~0.01
#                         tolerance a rows-held arm needs. Quartering it buys 4x eval
#                         headroom for a precision cost that does not change a verdict.
#   batch_size            Left ALONE by default. The A/B is within-pair so a smaller
#                         batch would still be valid, but batch is the training signal
#                         and shrinking it is the change most likely to stop an arm
#                         reaching its analytic value at all -- which would make the
#                         battery uninformative rather than merely noisier. Pass
#                         --batch to override deliberately.
COTENANT_EVAL_SAMPLES = 2500
COTENANT_PRIOR_MAX = 40000
COTENANT_ANCHOR_MAX = 20000


def cotenant_fraction(n):
    """
    Per-arm cuda_memory_fraction for n-way sharing.

    MEASUREMENT-DRIVEN where a measurement exists. Dividing the card by n and handing
    each arm the quotient is what the first version did, and it produced a ceiling of
    2282 MiB against a MEASURED peak of 2264 -- 18 MiB of headroom, on a HARD cap
    (`cuda_memory_fraction` is enforced, not advisory), with internal_oom_recovery
    false. That peak was taken at step 150, before stage-2 fused training, before any
    z_calibration burst (~11 extra full-batch rollouts at a transition) and before
    figures. Those arms would have died mid-run, thousands of steps in.

    So: give each arm its measured peak PLUS gpu_guard's own fragmentation margin, and
    let n be constrained by that rather than the reverse. Falls back to the naive
    division only when nothing has been measured yet.
    """
    import math
    usable = CARD_MB - DESKTOP_MB
    naive = usable / float(n)

    need = None
    try:
        sys.path.insert(0, str(CONFIGS.parent))
        from gpu_guard import load_registry, _with_margin
        peaks = [row.get('peak_reserved_mb') for row in load_registry().values()
                 if row.get('peak_reserved_mb')]
        if peaks:
            need = _with_margin(max(peaks))
    except Exception:
        need = None

    per_arm = max(naive, need) if need else naive
    if need and need * n > usable:
        raise ValueError(
            f'{n}-way sharing does not fit a measured peak of {need} MiB per arm '
            f'(with margin): {n} x {need} = {need * n} MiB against {usable} MiB '
            f'usable. Use --cotenants {int(usable // need)} or shrink the job.')
    return max(0.05, math.floor(100.0 * per_arm / CARD_MB) / 100.0)


# --------------------------------------------------- --pin-lr and --cap-stage1
# Both exist because wave 1 showed the A/B was confounded two ways, neither of which
# has anything to do with D33.
#
# PINNED LR. `lr = base_lr x peak_scale x envelope(t)` (controller.py). Wave 1's held
# arms sat pinned at cal_status `below_range` -- the SATURATION reading, not "too hot":
# once the loss is ~0.003 a ray probe cannot bracket alpha* and falls to a bound, and
# the controller kept cutting on it, ending 16x down. The rows-live arms, still far
# from converged, kept a high peak. Net effect: the two arms of a pair trained at LRs
# 8x apart, so "rows-live converges more slowly" was partly "rows-live ran hotter".
# Killing the confound needs all three factors held, not just the sensor:
#   ray_calibration.enabled false   -> peak_scale never moves
#   lr_sensor: none per stage       -> no sensor verdicts at all
#   lr_warmup_ratio 1, warmup 0     -> envelope(t) is flat, and it RESTARTS at every
#                                      stage transition, so leaving it on would put a
#                                      ramp exactly where stage 2 begins
# PINNED_LR is not invented: it is where wave 1's held arms' own controller settled
# (lr_fused 7.8e-6 to 1.1e-5) and held err ~0.001 for 3000 steps. The only LR MEASURED
# stable on this problem.
PINNED_LR = 1.0e-5

# CAPPED STAGE 1. Exit terms are ANDed (protocol.should_exit returns False on any
# failing term), and `gates/mle_flat` can NEVER flatten on a rows-live arm: every prior
# row holds the dead rows at exactly 0.0, so MLE buys unbounded likelihood by
# collapsing a delta onto a zero-variance direction -- c_sg19_off reached bwd/mle
# -20.8 vs its partner's -7.8, i.e. 13 nats over 3 dims, and the gain grows as
# log(1/sigma) without limit. So the arm was trapped in stage 1 for 3000 of 4000 steps
# by the very defect it exists to measure. Dropping that ONE term is the whole fix;
# bwd/tbc was already at 0.04 against a 2.0 bar. The mle_gate FLAG stays on so
# gates/mle_flat is still published as a diagnostic -- we want to keep watching it
# fail to flatten, just not to gate on it.
DROP_EXIT_METRICS = ('gates/mle_flat',)


def apply_pinned_lr(cfg):
    for key in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        cfg[key] = PINNED_LR
    cfg['lr_warmup_ratio'] = 1
    cfg.setdefault('adaptive_lr', {})['warmup_steps'] = 0
    # No `ray_calibration.enabled: False` here. That key is retired
    # (utils._RETIRED_KEYS) and, worse, `setdefault` would CREATE the retired
    # top-level block on a config that does not have one -- the gate fires on
    # PRESENCE, so the arm would be refused at load. The loop below is now the
    # whole switch: with every stage on `kind: none` no stage asks for ray, and
    # train.py arms the probe on `bool(self._ray_askers())`.
    stages = active_stages(cfg)
    # active_stages returns [] on an unresolvable selector rather than raising.
    # Silence there is the dangerous answer now that the loop below IS the ray
    # switch: no stages touched means no stage set to `none`, which means the
    # probe stays armed at whatever mk_dev declares, on an arm whose whole point
    # is that nothing actuates the LR.
    assert stages, (f"protocol {cfg.get('protocol')!r} resolves to no stages, so "
                    f"the pinned-LR arm would keep mk_dev's lr_sensor declarations")
    for stage in stages:
        # a MAPPING, not the bare string: Stage._parse_lr_sensor raises TypeError on a
        # str, so `lr_sensor: none` kills every arm at startup. Caught by loading the
        # generated configs, not by generating them.
        stage['lr_sensor'] = {'kind': 'none'}
    return cfg


def apply_stage1_cap(cfg):
    for stage in active_stages(cfg):
        if not stage.get('exit'):
            continue
        kept = [t for t in stage['exit'] if t.get('metric') not in DROP_EXIT_METRICS]
        if not kept:
            raise ValueError(f"stage {stage.get('name')}: dropping "
                             f"{DROP_EXIT_METRICS} would leave no exit condition, "
                             f"making the stage terminal")
        stage['exit'] = kept
    return cfg


def apply_cotenancy(cfg, n, batch=None):
    cfg['cuda_memory_fraction'] = cotenant_fraction(n)
    cfg['buffer_device'] = 'cpu'
    cfg['eval_num_samples'] = COTENANT_EVAL_SAMPLES
    buffers = cfg.setdefault('buffers', {})
    buffers.setdefault('prior_buffer', {})['max_size'] = COTENANT_PRIOR_MAX
    buffers.setdefault('anchor_buffer', {})['max_size'] = COTENANT_ANCHOR_MAX
    if batch:
        cfg['batch_size'] = int(batch)
        cfg['max_batch_size'] = int(batch)
    return cfg


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def arm(sg, hold):
    cfg = base()
    name = f'{"abcde"[spec.SPACE_GROUPS.index(sg)]}_sg{sg}_{"on" if hold else "off"}'
    path = spec.prior_path(sg)

    cfg['run_name'] = name
    cfg['tag'] = spec.TAG
    cfg['seed'] = 12345
    cfg['epochs'] = BUDGET
    cfg['eval_period'] = EVAL_PERIOD
    cfg['figure_period'] = FIGURE_PERIOD
    cfg['figs_period'] = 500
    cfg['wandb_mode'] = 'online'

    # ---- cold start, explicitly. The mk_dev defaults resolve
    # checkpoint_name: null + continue_from_checkpoint: true to
    # '{tag}_{run_name}_..._running.pt', so an arm that forgets to override them
    # silently resumes its own rolling checkpoint on a relaunch.
    cfg['checkpoint_name'] = None
    cfg['prior_model_name'] = None
    cfg['continue_from_checkpoint'] = False
    cfg['load_weights_only'] = False

    # ---- the problem
    cfg['energy_function'] = 'latent_gaussian'
    cfg['space_groups'] = [int(sg)]
    cfg['z_primes'] = [1]
    cfg['prior_path'] = path
    cfg['molecules_path'] = path
    cfg['test_molecules_path'] = None
    cfg['sg_conditioning'] = False
    cfg['zp_conditioning'] = False
    cfg['vector_conditioning'] = False
    cfg['molecule_conditioning'] = False
    cfg['temperature_conditioning'] = False

    ec = cfg.setdefault('energy_config', {})
    ec['temperature'] = spec.T
    ec['log_temperature_range'] = [0.0, 0.0]     # T fixed at 1; a range makes a mixture
    ec['bounding_coeff'] = spec.BOUNDING_COEFF
    ec['reduction_coeff'] = spec.REDUCTION_COEFF  # inert: structurally zero here
    ec['density_coeff'] = 0.0                     # unused by a latent energy
    ec['lj_coeff'] = 1.0                          # unused by a latent energy
    ec['lj_rescale'] = None
    ec['reward_range'] = None                     # never deform an analytic target
    ec['internal_oom_recovery'] = False
    ec['analyze_kwargs'] = {'c': spec.target_c(sg), 'width': spec.WIDTH}

    # ---- the knob under test, plus the one interaction that would invalidate the
    # closed form (see the module docstring)
    m = cfg.setdefault('model', {})
    m['hold_dead_latent_rows'] = bool(hold)
    m['periodic_centroids'] = False

    # ---- pinned mechanics
    cfg['batch_size'] = BATCH
    cfg['max_batch_size'] = BATCH
    cfg['grow_batch_size'] = False
    cfg['eval_num_samples'] = 10000     # IS log Z error ~ sqrt(Var(log w)/N)
    cfg['test_eval_num_samples'] = 2000
    cfg['eval_T'] = TRAJ_T
    cfg.setdefault('integrator', {})['T'] = TRAJ_T
    cfg['max_step_seconds'] = 10
    cfg.setdefault('bwd_loss_coeffs', {})['condition_block_m'] = 1
    return name, cfg


# ------------------------------------------------------------------ assertions
def assert_knob_explicit(name, cfg):
    for key in ('hold_dead_latent_rows', 'periodic_centroids'):
        if key not in (cfg.get('model') or {}):
            raise ValueError(f'{name}: model.{key} must be stated explicitly -- omitting it '
                             f'inherits mk_dev\'s live default and can drift between regenerations')


def assert_cold(name, cfg):
    if cfg.get('continue_from_checkpoint'):
        raise ValueError(f'{name}: continue_from_checkpoint must be False')
    if cfg.get('checkpoint_name') is not None:
        raise ValueError(f'{name}: every arm starts cold -- expanded_dim differs across arms')
    if 'reuse_prior' in cfg:
        raise ValueError(f'{name}: reuse_prior is a RETIRED key; preflight_config hard-raises '
                         f'on its PRESENCE, so `false` is worse than absent')


def assert_target_consistent(name, cfg, sg):
    """
    The config's `c` must be MODE on live rows and 0.0 on dead rows, resolved by the
    SHIPPING resolver. This is the check that catches the failure this battery cannot
    otherwise see: a `c` with MODE on a dead row adds a constant
    0.5*(MODE/w)^2 = 12.5 nats per row to every energy, and the run trains fine and
    reports a wrong log Z.
    """
    c = cfg['energy_config']['analyze_kwargs']['c']
    dead = spec.dead_rows(sg)
    if len(c) != spec.DIM:
        raise ValueError(f'{name}: c has {len(c)} entries, expected {spec.DIM}')
    for r in range(spec.DIM):
        want = 0.0 if r in dead else spec.MODE
        if abs(c[r] - want) > 1e-12:
            raise ValueError(f'{name}: c[{r}] = {c[r]}, expected {want} '
                             f'(dead rows for sg {sg} are {dead})')
    if abs(cfg['energy_config']['analyze_kwargs']['width'] - spec.WIDTH) > 1e-12:
        raise ValueError(f'{name}: width disagrees with spec')
    if abs(cfg['energy_config']['temperature'] - spec.T) > 1e-12:
        raise ValueError(f'{name}: temperature disagrees with spec -- log Z depends on it')
    if abs(cfg['energy_config']['bounding_coeff'] - spec.BOUNDING_COEFF) > 1e-12:
        raise ValueError(f'{name}: bounding_coeff disagrees with spec -- the rows-live '
                         f'prediction depends on it')


def assert_prior_exists(name, cfg):
    if not os.path.exists(cfg['prior_path']):
        raise FileNotFoundError(f'{name}: prior missing at {cfg["prior_path"]} -- '
                                f'run prep_prior.py first')


def assert_distinct(arms):
    seen = {}
    for name, cfg in arms.items():
        body = dict(cfg)
        body.pop('run_name', None)
        key = yaml.safe_dump(body, sort_keys=True)
        if key in seen:
            raise ValueError(f'{name} is byte-identical to {seen[key]} apart from run_name')
        seen[key] = name


def assert_cotenancy_coherent(name, cfg, n):
    """
    The same arithmetic gpu_guard enforces at launch, checked at GENERATION time so a
    battery cannot be written that the pre-flight will then refuse arm by arm.
    """
    ceiling = cfg['cuda_memory_fraction'] * CARD_MB
    if ceiling * n > CARD_MB - DESKTOP_MB:
        raise ValueError(
            f'{name}: {n} x {ceiling:.0f} MiB = {ceiling * n:.0f} MiB exceeds the '
            f'{CARD_MB - DESKTOP_MB} MiB left after the desktop; lower the fraction')
    if cfg.get('buffer_device') == 'cuda':
        raise ValueError(f'{name}: co-tenant arms must keep buffers off the card')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cotenants', type=int, default=1,
                    help='generate a variant calibrated to run N arms in parallel')
    ap.add_argument('--batch', type=int, default=None,
                    help='override batch_size (co-tenant mode only; see the notes above)')
    ap.add_argument('--pin-lr', action='store_true',
                    help='constant LR: no calibration, no sensor, flat envelope')
    ap.add_argument('--cap-stage1', action='store_true',
                    help='drop the un-flattenable gates/mle_flat exit term')
    ap.add_argument('--traj-t', type=int, default=None,
                    help="trajectory length (integrator.T and eval_T together). The "
                         "analytic log Z does NOT depend on it -- log Z uses the "
                         "SAMPLING temperature, not the chain length -- so a T sweep "
                         "tests the finite-T representational claim in F-024/F-026 "
                         "against an UNCHANGED target. t_scale_preserve_budget defaults "
                         "True, so the total noise budget is held fixed and only the "
                         "temporal resolution changes.")
    ap.add_argument('--suffix', default='',
                    help='output dir suffix, so a new condition does not overwrite '
                         'configs whose runs already completed')
    cli = ap.parse_args()
    n = max(1, cli.cotenants)

    out_dir = (HERE if n == 1 else HERE / f'par{n}')
    if cli.suffix:
        out_dir = out_dir.parent / (out_dir.name + cli.suffix) if n > 1 else HERE / cli.suffix
    out_dir.mkdir(exist_ok=True)

    arms = {}
    for sg in spec.SPACE_GROUPS:
        for hold in (True, False):
            name, cfg = arm(sg, hold)
            if n > 1:
                apply_cotenancy(cfg, n, cli.batch)
                assert_cotenancy_coherent(name, cfg, n)
            if cli.traj_t:
                # eval_T must track train T: a mismatch is a known artifact that
                # dominates the metrics (reference_stab_july21_elj_battery_T_dominates)
                cfg['integrator']['T'] = int(cli.traj_t)
                cfg['eval_T'] = int(cli.traj_t)
            if cli.pin_lr:
                apply_pinned_lr(cfg)
            if cli.cap_stage1:
                apply_stage1_cap(cfg)
            assert_knob_explicit(name, cfg)
            assert_cold(name, cfg)
            assert_target_consistent(name, cfg, sg)
            assert_prior_exists(name, cfg)
            arms[name] = cfg

    # sg 2 on/off are the ONE legitimate near-duplicate: the dead set is empty, so
    # apart from the knob they describe the same experiment. That is the point of the
    # control, so exempt exactly that pair rather than weakening the check.
    assert_distinct({k: v for k, v in arms.items() if not k.startswith('a_sg2')})

    for name, cfg in arms.items():
        out = out_dir / f'{name}.yaml'
        with out.open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        print(f'wrote {out.name}')

    rows = ['arm\tsg\tdead\tn_live\thold\tpredicted_log_Z\tprior']
    for sg in spec.SPACE_GROUPS:
        for hold in (True, False):
            name = f'{"abcde"[spec.SPACE_GROUPS.index(sg)]}_sg{sg}_{"on" if hold else "off"}'
            d = spec.dead_rows(sg)
            rows.append(f'{name}\t{sg}\t{"|".join(map(str, d)) or "-"}\t'
                        f'{spec.DIM - len(d) if hold else spec.DIM}\t{hold}\t'
                        f'{spec.analytic_log_z(sg, hold):.4f}\t'
                        f'{os.path.basename(spec.prior_path(sg))}')
    (out_dir / 'INDEX.tsv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    if n > 1:
        f = cotenant_fraction(n)
        print(f'wrote {out_dir.name}/ -- {len(arms)} arms calibrated for {n}-way sharing')
        print(f'  cuda_memory_fraction {f}  ->  {f * CARD_MB:.0f} MiB each, '
              f'{f * CARD_MB * n:.0f} MiB for all {n}, against '
              f'{CARD_MB - DESKTOP_MB} MiB usable')
        print(f'  buffer_device cpu, eval_num_samples {COTENANT_EVAL_SAMPLES}, '
              f'prior/anchor caps {COTENANT_PRIOR_MAX}/{COTENANT_ANCHOR_MAX}, '
              f'batch_size {arms[next(iter(arms))]["batch_size"]}')
        print(f'  pre-flight each launch with: python gpu_guard.py --cotenants {n} '
              f'--config configs/gauss_aug12/{out_dir.name}/<arm>.yaml')
    else:
        print(f'wrote INDEX.tsv ({len(arms)} arms)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
