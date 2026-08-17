"""
a100_stab_aug16 -- the first CLUSTER battery of the infrastructure-stabilization
plan: functionality checks on the real hardware, noise floors for the
a100-throughput suite, and the utilization-proxy measurement. Not science runs.

    python configs/a100_stab_aug16/make.py                 # WAVE 1 configs + INDEX + sbatch
    python configs/a100_stab_aug16/make.py --wave2 ...     # WAVE 2, from wave-1 artifacts
    python configs/a100_stab_aug16/make.py --preflight     # file checks, ON THE CLUSTER

base_uncond.yaml / base_cond.yaml are SNAPSHOTS of configs/shakeout_aug16/
{mipcas_elj,qm9_cond}.yaml (2026-08-16, the configs that actually ran the local
shakeout). Re-snapshot by hand, deliberately, so this battery does not drift.

=============================================================================
THE SHORT-RUN / UTILIZATION-WINDOW TENSION, RESOLVED EXPLICITLY
=============================================================================
gpu/util_policy is a 7200 s trailing window and the scheduler judges over its
own ~2 h window (observed: it cancels ~60% mean violations -- prod0810 uma arm
4r351oqm at 5.2 h; every qm9anchor_aug14 v1 arm, pinned batch 1000 ELJ T=10).
A short run CANNOT produce a valid reading from that proxy: below 7200 s the
window never fills, a partial window still prints a number, and util_recent ==
util_policy by construction (one deque, two windows), so their agreement on a
short run carries ZERO information.

The battery therefore splits into tiers, and each tier states what its
utilization numbers can support:

  F (functionality, wave 1)  SHORT. gpu/util_policy is NOT VALID on any F arm
       and must not be quoted. What a short job CAN report: the external
       nvidia-smi sidecar CSV over the whole job (a valid statement about what
       THAT JOB used over ITS OWN duration) and sacct Elapsed. Neither stands
       in for the 7200 s policy statistic.
  B (throughput floors, wave 2)  SHORT -- and legitimately so: the four
       step-cost benchmarks declare gpu/util_policy UNUSABLE in the registry
       itself. Their step budgets deliver minutes-to-an-hour, not the 2 h the
       "5 benchmarks x 2 h" arithmetic assumed; min_wallclock_s: 7200 exists
       only on a100-batch-scaling-elj.
  U (utilization, wave 2)  LONG, the only tier allowed to report
       gpu/util_policy. Step budgets are sized from wave-1 MEASURED step times
       so wallclock >= 1.15 x the window even if steps run 25% faster than the
       measured median; the SLURM cap bounds the slow side. Window fill is
       VERIFIED per run from the sidecar CSV duration, because the in-process
       sample count (G2 of phase6_measurement_request.md) is not instrumented.

Proxy bookkeeping (Phase 4 acceptance): gpu/util_policy is adopted under
CASE 2 (PARTIAL), leaning case 3. The cluster-visible evidence is cancellation
behaviour, not documentation: threshold ~60%, window ~2 h, statistic assumed
mean -- all inferred from kills. The error has a DIRECTION: the in-process
sampler takes zero samples during eval, so util_policy OVERSTATES what the
scheduler sees; U3 (a100-util-production-shape) measures that overstatement.
nvidia-smi corroborates only sampling artifacts -- it reads the SAME NVML
counter torch.cuda.utilization() wraps -- so case-2 weight comes from
sacct/scontrol and cancellations, which the sbatch epilogue captures.

=============================================================================
TWO WAVES, BECAUSE THE SIZES ARE UNMEASURED
=============================================================================
Every A100 absolute is unmeasured (benchmarks.md section 10): step time, batch
ceilings, whether compile engages. Wave 1's fresh functionality arms produce
(a) the pinned, immutable, step-tagged archives wave 2 warm-starts from, and
(b) measured step times wave 2's U budgets are sized from. Wave 2 refuses to
generate without them -- guessing here is how a 500-step budget got declared
against a 7200 s minimum (phase6_measurement_request.md section 2 box).

WAVE 1 (fresh; checkpoint_name null asserted; each covers paths that broke in
the 2026-08-16 local shakeout):
  f1_mipcas_elj    uncond route @1000. MLE -> gate -> phase-1->2 transition
                   (on_enter runs OUTSIDE the OOM try/except -- fatal if it
                   OOMs, which is the point of running it at production batch),
                   rebuild_prior_by_churn PURGE-DOWN (159k -> 62.5k), ray
                   sensor stage, cluster eval/figs cadence, archives.
                   Measures t_elj(1000) for wave-2 sizing.
  f2_qm9cond_b500  conditional route @500 (the batch the local shakeout
                   survived). Transition into var_conditioning (fwd .5/bwd .5,
                   repeats 2.0 both branches: SEVERAL TIMES equilibration's
                   memory peak), churn GENERATE-UP (4.9k seed -> 62.5k target
                   through crystal-energy evaluation, the
                   internal_oom_recovery forwarding path), embedding
                   conditioning, held-out eval_test. EXPECT the VarGrad
                   detonation ~30 steps after entry -- locally it fired from a
                   190-step AND a 3.2k-step MLE prior alike -- and everything
                   this arm exists to check happens BEFORE that point;
                   surviving on the A100 would itself be a finding.
  f2b_qm9cond_b1000  same @1000: does the production batch survive the
                   var_conditioning entry on an A100. A fatal on_enter OOM
                   here is the finding, cheaply.
  f3_mipcas_uma    UMA @250 (traj_checkpoint on). Init whole-dataset UMA scan,
                   transition churn with UMA scoring, fused MLIP steps.
                   Measures t_uma.
  f4_acridine_mace MACE @250, sg14 (monoclinic -- dead latent rows (3,5)).
                   FIRST MACE cost numbers on record anywhere; the
                   batched-neighbour fast path must be shown TAKEN (silent
                   torch_cluster fallback reads as a slow GPU).

WAVE 2 (warm; checkpoint_name/prior_model_name asserted against wave-1 names;
every group missing its artifacts is DROPPED LOUDLY, never silently):
  B1..B4  the four step-cost a100-throughput benchmarks x 5 repeat launches
          (registry noise_floor.repeats: 5 -- the validator refuses a floor
          recorded from fewer). Warm-started INSIDE the measured stage from a
          step-tagged archive, because a stage transition is not steady state
          and _running.pt is not immutable. lr sensors are STRIPPED and the
          four lr_* keys pinned to adaptive_lr.seed_lr (the exact translation
          tierc_smoke validated): the ray probe fires every 500 steps and
          would alias with a 500-step window; overrides cannot reach a stage
          declaration, so neutralising it is this generator's job (see the
          note at registry.yaml defaults.overrides).
  U1      a100-batch-scaling-elj rung 1000 x3 + one cadence probe
          (gpu_util_sample_period_s 60 -> 10; NOT a benchmark). Rung 1000 is
          the KNOWN-CANCELLED regime -- an instrumented cancellation with the
          sidecar + sacct epilogue is threshold data (disagreement-table row
          2b), and if the arms survive the three repeats are the rung floor.
  U2      rung 7410 x3: the plausible-survivor anchor (grown prod0810 elj
          runs lived for days in the 1650-7410 range). Middle rungs 1650/2722/
          4491 are DEFERRED to the phase-6 submission -- dropped coverage,
          logged here and in RUNSHEET.md.
  U3      a100-util-production-shape @4491, eval ON at 500/1000, x3: the
          eval-blindness direction and magnitude -- the number that decides
          whether util_policy overstates.

=============================================================================
WHAT THE GENERATOR ASSERTS (requirement: pin resume, validate every config)
=============================================================================
- checkpoint_name / continue_from_checkpoint / prior_model_name /
  load_weights_only are set DELIBERATELY on every arm and asserted per tier:
  wave 1 all-fresh, wave 2 all-warm with name-pattern checks (an archive must
  be _step<N>-tagged or _phase1_exit; _best.pt and _running.pt are REFUSED --
  rb0808 lost a battery to that fallback). `reuse_prior` is retired; these
  four keys are its replacement.
- config_invariants.check() runs over EVERY emitted config; any ERROR aborts
  generation and every BASELINE is printed. This is live, not decorative: on
  the raw cond template, exit_bar_is_within_measured_range reports the
  var_conditioning exit bar (logw_std_within < 6.0) as sitting under the
  metric's measured minimum 17.1 -- and names deletion as the remedy for a
  stage that is terminal by design. The battery arms apply exactly that:
  exit block removed, stage declared terminal (verified: the report fires on
  the template and not on the generated arms).
- warm epochs are computed through registry.epochs_for() where its hardcoded
  formula applies (B tier), and computed here for the U tier, where
  epochs_for() would silently return a 500-step budget against a 7200 s
  minimum (phase6_measurement_delta.md D10.1).
"""
import argparse
import copy
import math
import os
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
ROOT = HERE.parents[1]                      # energy_sampling/
sys.path.insert(0, str(ROOT))

import config_invariants                    # noqa: E402
from benchmarks import registry             # noqa: E402

TAG = 'a100_stab_aug16'

# --- cluster paths (prod0810/qm9anchor_aug14 conventions) --------------------
CLUSTER_CKPT_DIR = '/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints/'
CLUSTER_PRIOR_DIR = '/scratch/mk8347/data/crystal_datasets/conditional/priors/'
CLUSTER_UMA_MLIP = '/scratch/mk8347/models/uma/esen_s.pt'
CLUSTER_MACE_MLIP = '/scratch/mk8347/data/acr_112025_mh1_stagetwo.model'

SEED_LR = 1.25e-4       # adaptive_lr.seed_lr: what `auto` resolves to with no sensor.
                        # Pinning to it is a translation, not a guess (tier-C result).

# The LOOSE handoff gate -- fires ~150-200 steps locally on BOTH routes.
# DELIBERATE REVERSAL of the first draft's interpolation ({1.5, 0.2, 250}),
# on late-2026-08-16 shakeout evidence that arrived after that draft:
#   - the strict family's nearest measured point {1.5, 0.1, 250} did NOT fire
#     in 3186 cold steps (qm9_mle run; recorded in shakeout_aug16/qm9_cond.yaml
#     line 518), so a strict gate risks an F arm that never transitions -- and
#     the transition is the thing the F arms exist to exercise;
#   - the var_conditioning detonation is INDIFFERENT to MLE quality: a warm
#     start from the 3.2k-step MLE prior (qm9_warm run) detonated at the same
#     steps (170-220) as the 190-step prior, so holding MLE longer buys the
#     conditional route nothing a functionality check needs.
# The unconditional route COMPLETED off this gate locally (mipcas_final).
MLE_GATE = {'slope_t': 1.0, 'min_rate': 0.5, 'window': 100}

# The conditional arms' WEIGHTS-ONLY MLE warm start. From qm9_anchor_aug13, the
# later of the two runs on this exact problem; slug `elj-qm9split_prior-T6.9-
# b3483b` matches what these arms resolve to, which is what makes it legal
# (load_weights_only calls assert_problem_match). Override with --cond-mle-seed.
# MUST BE COPIED TO THE CLUSTER checkpoints dir -- it currently exists locally
# only; --preflight is what catches its absence before a queue slot is spent.
DEFAULT_COND_MLE_SEED = 'qm9a13_qm9a98b_elj-qm9split_prior-T6.9-b3483b_phase1_exit.pt'

FLOOR_REPEATS = 5       # matches registry noise_floor.repeats on the four step-cost benchmarks
UTIL_REPEATS = 3        # matches a100-batch-scaling-elj / a100-util-production-shape

# SLURM time limits per submit class: sized to LIKELY NEED (bracket top + a
# margin on the unmeasured side), not to the 48 h maximum -- user call,
# 2026-08-16. Up to 16 A100s run concurrently, so a class is one queue batch.
# If wave 1 moves a bracket (e.g. MACE slower than ~20 s/step), bump the class
# here and regenerate rather than editing the sbatch by hand.
TIME_CLASSES = {
    'wave1_elj': '03:00:00',      # f1/f2/f2b bracket tops ~2.5 h
    'wave1_mlip': '06:00:00',     # f3/f4 top ~4.5 h; MACE has NO prior measurement
    'wave2_floors': '02:00:00',   # 500/225-step budgets: ~8-70 min + resume init
    'wave2_util': '05:00:00',     # U1/U2 sized ~3.1 h at measured t; cap >> window
    'wave2_prodshape': '08:00:00',  # U3 sized ~6 h incl. eval share
}

POLICY_WINDOW_S = registry.POLICY_WINDOW_S          # 7200
U_TARGET_FACTOR = 1.15  # aim past the window, not at it
U_FAST_MARGIN = 0.75    # size steps as if they might run 25% faster than measured


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_yaml(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge(d1, d2):
    """Deep-merge d2 onto d1 (prod0810's overwrite_nested_dict)."""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            d1[k] = merge(d1[k], v)
        else:
            d1[k] = v
    return d1


def stage(cfg, protocol_name, stage_name):
    for s in cfg['protocols'][protocol_name]['stages']:
        if s.get('name') == stage_name:
            return s
    raise SystemExit(f'no stage {stage_name!r} in protocol {protocol_name!r}')


def steps_for_wallclock(target_s, step_time_s):
    """Steps so wallclock >= target even if steps run U_FAST_MARGIN x faster
    than measured; multiple of 10 (the fused force-refresh period)."""
    steps = math.ceil(target_s / (U_FAST_MARGIN * step_time_s))
    return int(math.ceil(steps / 10.0) * 10)


def cluster_layer(cfg, run_name, batch, epochs, archive_period):
    """The edits every arm shares: cluster paths/cadence + a pinned batch."""
    merge(cfg, {
        'run_name': run_name,
        'tag': TAG,
        'checkpoints_dir': CLUSTER_CKPT_DIR,
        'eval_period': 500,          # cluster cadence (prod0810 asserts 500/1000)
        'figs_period': 1000,
        'batch_size': batch,
        'max_batch_size': batch,     # independent hard stops; BOTH must move
        'grow_batch_size': False,
        'auto_batch_throughput_opt': False,   # inert under grow=False, pinned for clarity
        'epochs': epochs,
        'archive_period': archive_period,
        'archive_buffers': True,
    })
    return cfg


def pin_fresh(cfg, mle_seed=None):
    """Wave-1 resume pinning. `mle_seed` is a WEIGHTS-ONLY MLE warm start.

    WHY WEIGHTS-ONLY AND NOT A FULL RESUME. load_weights_only takes the model
    weights and nothing else -- step count restarts at 0, optimizers and buffers
    are fresh, and the run RE-ENTERS train_prior. So the arm still exercises the
    MLE stage, the gate, the transition and the churn rebuild (which is most of
    what these arms are for), but starts them from a policy that is already
    trained rather than from noise. A full resume would skip the very paths
    under test.

    WHY AT ALL. The MLE budget these arms can afford is ~150-250 steps, and the
    older qm9 conditional runs that started from a good MLE point began
    var_conditioning at markedly lower losses. Re-deriving that in 200 steps is
    not possible; loading it costs nothing.

    NOT A FIX FOR THE DETONATION, and the evidence is direct: the 2026-08-16
    `qm9_warm` shakeout did exactly this from a 3.2k-step MLE prior and
    detonated at the same steps as the cold run, because it still carried the
    unconditional Z trio. Warm start and conditionalise_z are independent
    changes and this battery makes both.

    IDENTITY IS CHECKED, NOT ASSUMED. checkpointing.load_weights_only calls
    assert_problem_match, so the seed must share the live config's problem
    identity -- same energy function, prior_path, T, space group. The qm9split
    seeds carry the slug `elj-qm9split_prior-T6.9-b3483b`, which is the slug
    these arms resolve to; that is why they are legal here and an aug14
    (qm9c100k) checkpoint would not be.
    """
    cfg['checkpoint_name'] = mle_seed
    cfg['continue_from_checkpoint'] = False
    cfg['prior_model_name'] = None
    cfg['load_weights_only'] = mle_seed is not None
    cfg['checkpoint_read_only'] = False     # wave 1 MUST write: wave 2 resumes from it
    return cfg


def pin_warm(cfg, archive, prior):
    cfg['checkpoint_name'] = archive
    cfg['continue_from_checkpoint'] = False  # inert while checkpoint_name is set; pinned anyway
    cfg['prior_model_name'] = prior          # a resume INSIDE a churn stage needs the frozen
    cfg['load_weights_only'] = False         # prior sampler; without it churn refills 100%
    return cfg                               # from anchors (train.py warns)


def strip_grad_geometry(cfg):
    """Benchmark arms: turn the fused gradient-geometry diagnostic OFF.

    It is periodic work whose period (50 fused steps) is not the report period,
    and each firing costs one extra backward PER ACTIVE BRANCH -- three on a
    fused stage. A 400-step window contains 8 firings, so it is a real, roughly
    5-10% addition to the measured step cost that has nothing to do with the
    training step being timed. Same argument as z_calibration and the ray probe
    in registry.yaml's defaults.overrides; this key simply postdates that list.

    Left ON for the wave-1 F arms: there it is production shape, and proving it
    survives a compiled trunk is part of what those arms check (it did not --
    2026-08-16, see maybe_compile_policy's donated-buffer note).
    """
    cfg.setdefault('grad_geometry', {})['enabled'] = False
    return cfg


def strip_lr_sensors(cfg):
    """Benchmark arms: drop every stage's lr_sensor and pin the four lr keys.

    The ray probe costs n_sub=8 paired sub-batches every 500 steps INSIDE the
    timing window and shifts the RNG stream even when it refuses a reading
    (findings F-039); hyper moves the LR mid-window. Neither belongs in a cost
    measurement. With the sensors gone, `auto` would be refused at load, so the
    keys are pinned to the seed -- the value they would have trained at anyway.
    """
    for proto in cfg.get('protocols', {}).values():
        for st in proto.get('stages', []):
            st.pop('lr_sensor', None)
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        assert k in cfg, f'missing {k}'
        cfg[k] = SEED_LR
    return cfg


def conditionalise_z(cfg):
    """Switch the three Z-side blocks from their UNCONDITIONAL defaults.

    THE CANONICAL CONFIG SAYS TO DO THIS AND NOTHING ENFORCED IT. mk_dev's
    condition_log_z block is annotated "MONITORING ONLY ... it becomes
    load-bearing on the conditional route", and its z_calibration parameters
    "are set for the UNCONDITIONAL route". A conditional config spawned from the
    canonical one inherits all three unchanged, which is what these arms did.

    The values are not a derivation -- they are what every conditional battery
    that RAN carries (qm9_aug11, qm9_anchor_aug13, qm9anchor_aug14, all three
    identical on all three keys), against a detonation in every conditional run
    that carried the unconditional trio, including one with a 3.2k-step MLE warm
    start. `config_invariants.conditional_z_settings_are_conditional` now
    reports the departure at load; this function is what makes these arms not
    depart. See findings F-042.
    """
    cfg['z_calibration']['enabled'] = False
    for key in ('fwd_tb_z_source', 'bwd_tb_z_source', 'replay_tb_z_source'):
        cfg['condition_log_z'][key] = 'persistent'
    cfg['condition_log_z']['half_life_visits'] = 28.0
    return cfg


def make_var_conditioning_terminal(cfg):
    """Remove var_conditioning's exit + on_exit: the declared bar
    (fwd/logw_std_within < 6.0) sits under the metric's measured minimum 17.1,
    so config_invariants correctly rejects it as an exit that cannot fire and
    the stage is terminal in practice. Declaring it terminal makes the config
    honest and the invariants pass for the RIGHT reason."""
    st = stage(cfg, 'conditional_vargrad', 'var_conditioning')
    st.pop('exit', None)
    st.pop('on_exit', None)
    return cfg


def apply_benchmark(cfg, bid):
    """Deep-merge the registry's resolved overrides (defaults + benchmark)."""
    return merge(cfg, registry.resolved_overrides(bid))


ARCHIVE_RE = re.compile(r'_(step\d+|phase1_exit)\.pt$')


def check_warm_names(kind, archive, prior, expect_prefix):
    assert archive and ARCHIVE_RE.search(archive), (
        f'{kind}: {archive!r} is not a step-tagged archive or phase1_exit snapshot. '
        f'_best.pt / _running.pt are REFUSED: _best without phase1_exit is proof the '
        f'exit never fired (fatal at step 0, rb0808 arm 0), and _running.pt is '
        f'overwritten by every run of the config -- not immutable, not a benchmark base.')
    assert '_best' not in archive and '_running' not in archive, f'{kind}: refused {archive!r}'
    assert archive.startswith(expect_prefix), (
        f'{kind}: archive {archive!r} does not start with {expect_prefix!r} -- wrong arm\'s '
        f'checkpoint. The problem identity would reject it at load, but not before '
        f'burning the queue slot.')
    assert prior and prior.endswith('_prior.pt'), f'{kind}: {prior!r} is not a *_prior.pt'
    assert prior.startswith(expect_prefix), f'{kind}: prior {prior!r} is not from {expect_prefix!r}'


def archive_step(archive):
    m = re.search(r'_step(\d+)\.pt$', archive)
    assert m, f'{archive!r}: warm benchmark arms must resume a _step<N> archive INSIDE ' \
              f'the measured stage, not phase1_exit -- a transition is not steady state'
    return int(m.group(1))


# ---------------------------------------------------------------------------
# wave 1
# ---------------------------------------------------------------------------

def build_wave1(base_uncond, base_cond, cond_mle_seed=None):
    arms = []

    def f_arm(name, base, batch, epochs, archive_period, extra=None, cond=False):
        cfg = copy.deepcopy(base)
        cluster_layer(cfg, name, batch, epochs, archive_period)
        pin_fresh(cfg, mle_seed=cond_mle_seed if cond else None)
        # production eval size on the cheap-energy arms; MLIP F arms keep 2000
        # so a 10k MLIP-scored eval does not dominate a functionality check
        cfg['eval_num_samples'] = 2000 if (extra or {}).get('_mlip') else 10000
        for proto in ('unconditional_tb', 'conditional_vargrad'):
            stage(cfg, proto, 'train_prior')['mle_gate'] = dict(MLE_GATE)
        if cond:
            make_var_conditioning_terminal(cfg)
            conditionalise_z(cfg)
            cfg['prior_path'] = CLUSTER_PRIOR_DIR + 'qm9split_prior.pt'
            cfg['molecules_path'] = CLUSTER_PRIOR_DIR + 'qm9split_conditions.pt'
            cfg['test_molecules_path'] = CLUSTER_PRIOR_DIR + 'qm9split_test_conditions.pt'
        else:
            p = CLUSTER_PRIOR_DIR + 'mipcas_sg2_zp1_elj_prior_dataset.pt'
            cfg['prior_path'] = p
            cfg['molecules_path'] = p
        if extra:
            merge(cfg, {k: v for k, v in extra.items() if not k.startswith('_')})
        arms.append((name, 'wave1_mlip' if (extra or {}).get('_mlip') else 'wave1_elj',
                     'func', '-', cfg))

    # Budgets sized to the LOOSE gate: transition ~step 200, then enough of the
    # target stage for archives >= transition+400 and a stable step-time median.
    f_arm('f1_mipcas_elj', base_uncond, batch=1000, epochs=2500, archive_period=500)
    f_arm('f2_qm9cond_b500', base_cond, batch=500, epochs=1500, archive_period=500, cond=True)
    f_arm('f2b_qm9cond_b1000', base_cond, batch=1000, epochs=1000, archive_period=500, cond=True)
    mlip_shared = {'_mlip': True, 'traj_checkpoint': True, 'max_step_seconds': 300}
    f_arm('f3_mipcas_uma', base_uncond, batch=250, epochs=1200, archive_period=250,
          extra=dict(mlip_shared,
                     energy_function='uma', mlip_path=CLUSTER_UMA_MLIP,
                     prior_path=CLUSTER_PRIOR_DIR + 'mipcas_sg2_zp1_uma_prior_dataset.pt',
                     molecules_path=CLUSTER_PRIOR_DIR + 'mipcas_sg2_zp1_uma_prior_dataset.pt'))
    f_arm('f4_acridine_mace', base_uncond, batch=250, epochs=1200, archive_period=250,
          extra=dict(mlip_shared,
                     energy_function='mace', mlip_path=CLUSTER_MACE_MLIP,
                     space_groups=[14],
                     prior_path=CLUSTER_PRIOR_DIR + 'acridine_sg14_zp1_mace_prior_dataset.pt',
                     molecules_path=CLUSTER_PRIOR_DIR + 'acridine_sg14_zp1_mace_prior_dataset.pt'))
    return arms


# ---------------------------------------------------------------------------
# wave 2
# ---------------------------------------------------------------------------

def build_wave2(base_uncond, base_cond, a):
    arms, dropped = [], []

    def bench_base(bid, cond=False):
        cfg = copy.deepcopy(base_cond if cond else base_uncond)
        apply_benchmark(cfg, bid)
        strip_lr_sensors(cfg)
        strip_grad_geometry(cfg)
        if cond:
            make_var_conditioning_terminal(cfg)
            conditionalise_z(cfg)
            cfg['prior_path'] = CLUSTER_PRIOR_DIR + 'qm9split_prior.pt'
            cfg['molecules_path'] = CLUSTER_PRIOR_DIR + 'qm9split_conditions.pt'
            cfg['test_molecules_path'] = CLUSTER_PRIOR_DIR + 'qm9split_test_conditions.pt'
        else:
            p = CLUSTER_PRIOR_DIR + 'mipcas_sg2_zp1_elj_prior_dataset.pt'
            cfg['prior_path'] = p          # uma/mace groups override via their extra
            cfg['molecules_path'] = p
        return cfg

    def floor_group(gname, bid, archive, prior, transition_step, prefix,
                    cond=False, extra=None):
        if not (archive and prior and transition_step):
            dropped.append(f'{gname} ({bid}): missing --{gname}-archive/--{gname}-prior/'
                           f'--{gname}-transition-step -- ALL {FLOOR_REPEATS} floor '
                           f'launches dropped; the floor stays unmeasured and the '
                           f'registry validator will keep refusing comparisons on it')
            return
        check_warm_names(gname, archive, prior, prefix)
        s = archive_step(archive)
        assert s >= transition_step + 400, (
            f'{gname}: archive step {s} is under {transition_step}+400 -- the window '
            f'would sit in the stage-entry transient (rebuild churn, LR re-warm, '
            f'allocator growth). Pick a later archive.')
        w = registry.benchmark(bid)['work']
        epochs = registry.epochs_for(bid, s)
        batch = w['batch_size']
        for i in range(FLOOR_REPEATS):
            cfg = bench_base(bid, cond=cond)
            cluster_layer(cfg, f'{gname}_r{i}', batch, epochs, archive_period=0)
            # cluster_layer writes the cadence AFTER the benchmark overrides, so
            # re-assert the benchmark's eval-off here rather than trusting order.
            cfg['eval_period'] = 100000000
            cfg['figs_period'] = 100000000
            pin_warm(cfg, archive, prior)
            cfg['checkpoint_read_only'] = True   # a benchmark must not clobber its base
            if extra:
                merge(cfg, copy.deepcopy(extra))
            arms.append((f'{gname}_r{i}', 'wave2_floors', 'floor', bid, cfg))
        print(f'  {gname}: {FLOOR_REPEATS} launches, resume step {s}, epochs {epochs} '
              f'({epochs - s} steps), batch {batch}')

    # --- B tier: the four step-cost a100-throughput floors -------------------
    p1 = f'{TAG}_f1_mipcas_elj_'
    floor_group('elj', 'elj-fused-uncond', a.elj_archive, a.elj_prior,
                a.elj_transition_step, p1)
    floor_group('cond', 'elj-fused-cond', a.cond_archive, a.cond_prior,
                a.cond_transition_step, f'{TAG}_f2_qm9cond_b500_', cond=True)
    floor_group('uma', 'uma-fused-uncond', a.uma_archive, a.uma_prior,
                a.uma_transition_step, f'{TAG}_f3_mipcas_uma_',
                extra={'mlip_path': CLUSTER_UMA_MLIP,
                       'prior_path': CLUSTER_PRIOR_DIR + 'mipcas_sg2_zp1_uma_prior_dataset.pt',
                       'molecules_path': CLUSTER_PRIOR_DIR + 'mipcas_sg2_zp1_uma_prior_dataset.pt'})
    floor_group('mace', 'mace-fused-uncond', a.mace_archive, a.mace_prior,
                a.mace_transition_step, f'{TAG}_f4_acridine_mace_',
                extra={'mlip_path': CLUSTER_MACE_MLIP,
                       'prior_path': CLUSTER_PRIOR_DIR + 'acridine_sg14_zp1_mace_prior_dataset.pt',
                       'molecules_path': CLUSTER_PRIOR_DIR + 'acridine_sg14_zp1_mace_prior_dataset.pt'})

    # --- U tier: the long, window-filling utilization arms -------------------
    if not (a.elj_archive and a.elj_prior and a.elj_transition_step and a.t_elj1000):
        dropped.append('U tier (a100-batch-scaling-elj anchors + a100-util-production-shape): '
                       'missing --elj-archive/--elj-prior/--elj-transition-step/--t-elj1000 '
                       '-- NO valid utilization reading will exist in this battery')
    else:
        s = archive_step(a.elj_archive)
        t1000 = a.t_elj1000
        t7410 = a.t_elj7410 or 4.0 * t1000     # ~B^0.8 scaling guess; override when measured
        t4491 = a.t_elj4491 or 2.8 * t1000
        target = U_TARGET_FACTOR * POLICY_WINDOW_S

        def util_arm(name, bid, batch, steps, extra=None, kind='util'):
            cfg = bench_base(bid)
            cluster_layer(cfg, name, batch, s + steps, archive_period=0)
            cfg['eval_period'] = 100000000
            cfg['figs_period'] = 100000000
            pin_warm(cfg, a.elj_archive, a.elj_prior)
            cfg['checkpoint_read_only'] = True
            if extra:
                merge(cfg, copy.deepcopy(extra))
            arms.append((name, 'wave2_util', kind, bid, cfg))

        n1000 = steps_for_wallclock(target, t1000)
        n7410 = steps_for_wallclock(target, t7410)
        for i in range(UTIL_REPEATS):
            util_arm(f'u_scale1000_r{i}', 'a100-batch-scaling-elj', 1000, n1000)
        # the cadence probe (phase6 request section 6): shipped 60 s vs 10 s, else
        # identical -- decides "cadence too coarse" vs "structurally biased".
        # A probe, not a benchmark: it is excluded from any floor.
        util_arm('u_scale1000_probe10s', 'a100-batch-scaling-elj', 1000, n1000,
                 extra={'gpu_util_sample_period_s': 10}, kind='probe')
        for i in range(UTIL_REPEATS):
            util_arm(f'u_scale7410_r{i}', 'a100-batch-scaling-elj', 7410, n7410)

        # U3: eval ON, the eval-blindness measurement. min_wallclock 14400.
        bid3 = 'a100-util-production-shape'
        w3 = registry.benchmark(bid3)
        t_eff = t4491 + a.eval_s / w3['overrides']['eval_period']
        n4491 = steps_for_wallclock(1.10 * w3['work']['min_wallclock_s'], t_eff)
        for i in range(UTIL_REPEATS):
            cfg = bench_base(bid3)
            cluster_layer(cfg, f'u_prodshape_r{i}', w3['work']['batch_size'], s + n4491,
                          archive_period=0)
            cfg['eval_period'] = w3['overrides']['eval_period']       # eval ON here
            cfg['figs_period'] = w3['overrides']['figs_period']
            cfg['eval_num_samples'] = w3['overrides']['eval_num_samples']
            pin_warm(cfg, a.elj_archive, a.elj_prior)
            cfg['checkpoint_read_only'] = True
            arms.append((f'u_prodshape_r{i}', 'wave2_prodshape', 'util', bid3, cfg))

        est = (UTIL_REPEATS * n1000 * t1000 + n1000 * t1000 + UTIL_REPEATS * n7410 * t7410
               + UTIL_REPEATS * n4491 * t_eff) / 3600.0
        print(f'  U tier: rung1000 {n1000} steps x{UTIL_REPEATS}+probe, rung7410 {n7410} '
              f'steps x{UTIL_REPEATS}, prodshape {n4491} steps x{UTIL_REPEATS}; '
              f'~{est:.1f} GPU-h at the measured/extrapolated step times')
        print('  DEFERRED (logged, not silent): a100-batch-scaling-elj middle rungs '
              '1650/2722/4491 eval-off -- they belong to the phase-6 submission, whose '
              'pre-flight also owns B_max bisection (delta D5)')

    return arms, dropped


# ---------------------------------------------------------------------------
# validation + emission
# ---------------------------------------------------------------------------

def validate(name, wave, kind, bid, cfg):
    # resume pinning is DELIBERATE on every arm (requirement 3)
    for key in ('checkpoint_name', 'continue_from_checkpoint', 'prior_model_name',
                'load_weights_only', 'checkpoint_read_only'):
        assert key in cfg, f'{name}: {key} missing entirely'
    conditional = cfg.get('embedding_conditioning') is True
    if wave.startswith('wave1'):
        assert cfg['prior_model_name'] is None, \
            f'{name}: wave-1 arms build their own prior in-run'
        if cfg['checkpoint_name'] is None:
            assert cfg['load_weights_only'] is False, f'{name}: no seed, so nothing to load'
        else:
            # a WEIGHTS-ONLY MLE seed. Full-state would restore the step count and
            # stage position and skip the paths this arm exists to exercise.
            assert cfg['load_weights_only'] is True, \
                f'{name}: an MLE seed must be weights-only -- a full-state resume ' \
                f'restores step/stage and skips the MLE stage, the gate, the ' \
                f'transition and the churn rebuild, which is everything this arm tests'
            assert conditional, f'{name}: only the conditional arms take an MLE seed'
            # assert_problem_match fires at load; catch the mismatch HERE, where it
            # costs nothing rather than a queue slot.
            assert 'qm9split_prior' in cfg['checkpoint_name'], \
                (f"{name}: MLE seed {cfg['checkpoint_name']!r} is not a qm9split_prior "
                 f"checkpoint. The problem identity is keyed on prior_path among other "
                 f"fields, so a seed from another condition library (e.g. qm9c100k) is "
                 f"refused by assert_problem_match at load.")
        assert cfg['checkpoint_read_only'] is False, \
            f'{name}: wave 1 must WRITE checkpoints; wave 2 resumes from them'
    else:
        assert cfg['checkpoint_name'] and cfg['prior_model_name'], \
            f'{name}: wave-2 arms are WARM -- a fresh start silently retrains phase 1 ' \
            f'({cfg["checkpoint_name"]!r}, {cfg["prior_model_name"]!r})'
        assert cfg['checkpoint_read_only'] is True, f'{name}: benchmark arms must not write'
        # wave 2 is the opposite case from wave 1's MLE seed: a benchmark measures
        # INSIDE a stage, so it needs the full state -- optimizers, buffers, step
        # count. Weights-only would restart at step 0 in train_prior and measure
        # the wrong stage entirely.
        assert cfg['load_weights_only'] is False, \
            f'{name}: a benchmark resume needs full state, not weights-only'
        assert cfg['epochs'] > archive_step(cfg['checkpoint_name']), \
            f'{name}: epochs {cfg["epochs"]} <= resume step -- runs ZERO steps and ' \
            f'reports a clean empty result (epochs is an ABSOLUTE index)'
    assert cfg['continue_from_checkpoint'] is False, f'{name}: pinned false everywhere'
    assert cfg['batch_size'] == cfg['max_batch_size'], f'{name}: batch not pinned'
    assert cfg['grow_batch_size'] is False, f'{name}: growth must be off'
    assert (cfg.get('mlip_path') is None) == (cfg['energy_function'] not in ('uma', 'mace')), \
        f'{name}: mlip_path must be set iff energy_function is an MLIP'
    assert str(cfg['checkpoints_dir']).startswith('/'), f'{name}: local checkpoints_dir'
    for key in ('prior_path', 'molecules_path'):
        assert str(cfg[key]).startswith('/'), f'{name}: local {key}: {cfg[key]}'
    assert cfg['tag'] == TAG and cfg['run_name'] == name

    if kind in ('floor', 'util', 'probe'):
        for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
            assert cfg[k] == SEED_LR, f'{name}: benchmark arms pin {k} to the seed'
        assert cfg['z_calibration']['enabled'] is False, f'{name}: z_cal corrupts the denominator'
        assert cfg['max_step_seconds'] == 0, f'{name}: the runaway guard cuts the batch mid-window'
        assert cfg['grad_geometry']['enabled'] is False, \
            f'{name}: the grad-geometry probe adds a backward per branch inside the window'
    elif not conditional:
        assert cfg['z_calibration']['enabled'] is True, \
            f'{name}: unconditional F arms are production-shaped'

    if conditional:
        # the three keys documented as unconditional-route settings, which every
        # conditional battery that RAN switched and every one that detonated did not
        assert cfg['z_calibration']['enabled'] is False, f'{name}: z_cal off on conditional'
        for k in ('fwd_tb_z_source', 'bwd_tb_z_source', 'replay_tb_z_source'):
            assert cfg['condition_log_z'][k] == 'persistent', f'{name}: {k} must be persistent'
        assert cfg['condition_log_z']['half_life_visits'] == 28.0, f'{name}: half_life 28'

    # requirement 4: config_invariants over every generated config
    violations = config_invariants.check(cfg)
    errors = [v for v in violations if v.severity == config_invariants.ERROR]
    for v in violations:
        print(f'    {name}: {v}')
    assert not errors, f'{name}: config_invariants ERRORs -- refusing to write'


def emit(arms, dropped, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'joblogs').mkdir(exist_ok=True)
    names = [n for n, *_ in arms]
    assert len(set(names)) == len(names), 'duplicate arm names'

    print('validating:')
    for name, wave, kind, bid, cfg in arms:
        validate(name, wave, kind, bid, cfg)
    for name, wave, kind, bid, cfg in arms:
        with (outdir / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    by_wave = {}
    for name, wave, kind, bid, cfg in arms:
        by_wave.setdefault(wave, []).append((name, kind, bid, cfg))

    master = ['name\twave\tkind\tbenchmark_id\tbatch\tepochs\tcheckpoint_name']
    for wave, rows in by_wave.items():
        lines = ['name\tkind\tbenchmark_id\tbatch\tepochs']
        for name, kind, bid, cfg in rows:
            lines.append(f'{name}\t{kind}\t{bid}\t{cfg["batch_size"]}\t{cfg["epochs"]}')
            master.append(f'{name}\t{wave}\t{kind}\t{bid}\t{cfg["batch_size"]}'
                          f'\t{cfg["epochs"]}\t{cfg.get("checkpoint_name") or "-"}')
        (outdir / f'INDEX_{wave}.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        (outdir / f'submit_{wave}.sbatch').write_text(
            SBATCH_TEMPLATE.format(label=wave, time=TIME_CLASSES[wave],
                                   array=f'0-{len(rows) - 1}'), encoding='utf-8')
    (outdir / 'INDEX.tsv').write_text('\n'.join(master) + '\n', encoding='utf-8')

    print(f'\nwrote {len(arms)} configs in {outdir}')
    for wave, rows in by_wave.items():
        print(f'  submit_{wave}.sbatch  --array=0-{len(rows) - 1}  ({len(rows)} arms)')
    if dropped:
        print('\nDROPPED COVERAGE (deliberate, per run sheet -- never silent):')
        for d in dropped:
            print(f'  - {d}')
    print('\nNEXT: python configs/a100_stab_aug16/make.py --preflight   ON THE CLUSTER')


# ---------------------------------------------------------------------------
# sbatch: one template, three uses. The sidecar is the load-bearing half --
# requirement 2 (all three utilization readings CONCURRENT) and the
# survives-scancel epilogue (wandb uploads no console log for a crashed run,
# and a cancellation is precisely the record we need).
# ---------------------------------------------------------------------------

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array={array}
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=a100stab_{label}
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/a100_stab_aug16/joblogs/%x_%A_%a.out

# a100_stab_aug16 :: {label}. Arm = row of INDEX_{label}.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index; a short
# range drops tail arms with no error at all.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/a100_stab_aug16
LOGS=${{ARMS}}/joblogs

ARM=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $1}}' ${{ARMS}}/INDEX_{label}.tsv)
if [ -z "${{ARM}}" ]; then echo "no arm at row ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; fi
CONFIG=${{ARMS}}/${{ARM}}.yaml
if [ ! -f "${{CONFIG}}" ]; then echo "missing config ${{CONFIG}}" >&2; exit 1; fi
echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}"
J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}

# ---- one-shot environment record (MIG turns the in-process sensor off
# SILENTLY and permanently; the UUID is what proves both samplers read the
# same card; scontrol records the partition/QoS the policy may vary by) ----
{{ nvidia-smi -L
  nvidia-smi --query-gpu=mig.mode.current,uuid,name,memory.total,driver_version --format=csv
  scontrol show job ${{SLURM_JOB_ID}}
  echo "nodelist: ${{SLURM_NODELIST}}  host: $(hostname)"
}} > ${{J}}.info 2>&1

# ---- concurrent samplers, spanning the WHOLE job (startup + eval + wrapup),
# so the denominator matches the scheduler's. 10 s cadence ~= 720 samples per
# 7200 s window against the in-process ~120: enough margin to adjudicate.
# stdbuf keeps lines flushed so a scancel does not eat the tail. These read the
# SAME NVML counter as torch.cuda.utilization() -- an independent SAMPLER, not
# an independent INSTRUMENT; throttle reasons + power separate "idle" from
# "throttled", and compute-apps makes co-tenanted intervals EXCLUDABLE. ----
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \
    --format=csv,nounits -l 10 > ${{J}}_smi.csv &
SMI_PID=$!
stdbuf -oL nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,nounits \
    -l 30 > ${{J}}_apps.csv &
APPS_PID=$!
if command -v dcgmi >/dev/null 2>&1; then
    # SM_ACTIVE (field 1002): the only genuinely DIFFERENT instrument here --
    # utilization.gpu is an any-kernel-resident duty cycle, blind to how much
    # of the GPU a kernel uses.
    stdbuf -oL dcgmi dmon -e 1002 -d 10000 > ${{J}}_dcgm.txt 2>&1 &
    DCGM_PID=$!
else
    echo "dcgmi not available" > ${{J}}_dcgm.txt
    DCGM_PID=""
fi

# ---- epilogue that survives scancel: on a cancelled job THIS is the record.
epilogue() {{
    kill ${{SMI_PID}} ${{APPS_PID}} ${{DCGM_PID}} 2>/dev/null
    sacct -j ${{SLURM_JOB_ID}} --format=JobID,State,ExitCode,Elapsed,NodeList,Reason,Comment%64 \
        > ${{J}}_sacct.txt 2>&1
}}
trap epilogue EXIT TERM

srun singularity exec --nv \\
    --overlay ${{OVERLAY}}:ro \\
    --bind ${{PROJECT_ROOT}}:${{PROJECT_ROOT}} \\
    --bind /scratch/mk8347/data:/scratch/mk8347/data \\
    --pwd ${{WORKDIR}} \\
    ${{IMAGE}} \\
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${{PROJECT_ROOT}}/MXtalTools:${{PROJECT_ROOT}}/gfn-diffusion:\\$PYTHONPATH
        python -u train.py --config ${{CONFIG}}
    "
"""


# ---------------------------------------------------------------------------
# preflight (run ON THE CLUSTER)
# ---------------------------------------------------------------------------

def preflight(outdir):
    paths = sorted(p for p in outdir.glob('*.yaml')
                   if p.name not in ('base_uncond.yaml', 'base_cond.yaml'))
    if not paths:
        print('no arm configs -- run make.py first'); return 1
    bad = 0
    ckpt_dir = load_yaml(paths[0])['checkpoints_dir']
    if not os.path.isdir(ckpt_dir):
        print(f'!! checkpoints_dir missing: {ckpt_dir}'); bad += 1
    for path in paths:
        cfg = load_yaml(path)
        for key in ('prior_path', 'molecules_path', 'test_molecules_path', 'mlip_path'):
            v = cfg.get(key)
            if v and not os.path.isfile(v):
                print(f'  MISSING  {path.name:<28} {key} -> {v}'); bad += 1
        ckpt = cfg.get('checkpoint_name')
        if ckpt:
            full = os.path.join(ckpt_dir, ckpt)
            if not os.path.isfile(full):
                print(f'  MISSING  {path.name:<28} checkpoint -> {full}'); bad += 1
            # A WEIGHTS-ONLY seed has no sidecar BY DESIGN: buffers re-seed from
            # prior_path, which is what keeps the churn test real. Requiring one
            # here would reject a correct config. A FULL-STATE resume without its
            # sidecar rebuilds empty buffers silently, so that stays a hard fail --
            # rb0808's fallback was itself the warning nobody read.
            if not cfg.get('load_weights_only'):
                sidecar = full.replace('.pt', '_buffers.pt')
                if not os.path.isfile(sidecar):
                    print(f'  MISSING  {path.name:<28} buffer sidecar -> {sidecar}'); bad += 1
        prior = cfg.get('prior_model_name')
        if prior and not os.path.isfile(os.path.join(ckpt_dir, prior)):
            print(f'  MISSING  {path.name:<28} prior model -> {prior}'); bad += 1
    print('preflight OK -- safe to submit' if not bad else f'{bad} missing file(s); do NOT submit')
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--wave2', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--outdir', default=None, help=argparse.SUPPRESS)  # self-test hook
    for grp in ('elj', 'cond', 'uma', 'mace'):
        ap.add_argument(f'--{grp}-archive', help=f'wave-1 _step<N> archive for the {grp} group')
        ap.add_argument(f'--{grp}-prior', help=f'wave-1 *_prior.pt for the {grp} group')
        ap.add_argument(f'--{grp}-transition-step', type=int,
                        help=f'step at which the wave-1 {grp} arm logged its stage transition '
                             f'(read "protocol: stage ... ->" in the joblogs .out)')
    ap.add_argument('--t-elj1000', type=float,
                    help='MEASURED wave-1 median train_step_time at batch 1000, seconds')
    ap.add_argument('--t-elj7410', type=float, help='override the 4.0x t1000 extrapolation')
    ap.add_argument('--t-elj4491', type=float, help='override the 2.8x t1000 extrapolation')
    ap.add_argument('--eval-s', type=float, default=150.0,
                    help='measured wave-1 eval_step_time at cluster cadence, seconds')
    ap.add_argument('--cond-mle-seed', default=DEFAULT_COND_MLE_SEED,
                    help='WEIGHTS-ONLY MLE warm start for the conditional F arms. Must be a '
                         'qm9split_prior checkpoint (problem identity is checked at load). '
                         'Pass "" to run them cold.')
    a = ap.parse_args()

    outdir = Path(a.outdir) if a.outdir else HERE
    if a.preflight:
        raise SystemExit(preflight(outdir))

    base_uncond = load_yaml(HERE / 'base_uncond.yaml')
    base_cond = load_yaml(HERE / 'base_cond.yaml')
    for b, name in ((base_uncond, 'base_uncond'), (base_cond, 'base_cond')):
        assert b.get('project_state_version') == 5, f'{name}: re-snapshot against current state'

    if a.wave2:
        arms, dropped = build_wave2(base_uncond, base_cond, a)
        if not arms:
            raise SystemExit('wave 2: nothing to generate -- pass the wave-1 artifact args')
    else:
        seed = a.cond_mle_seed or None
        arms, dropped = build_wave1(base_uncond, base_cond, cond_mle_seed=seed), []
        if seed:
            print(f'conditional MLE seed (weights-only): {seed}\n'
                  f'  -> must exist in the CLUSTER checkpoints dir; --preflight checks it.\n'
                  f'  -> a weights-only seed needs NO _buffers.pt sidecar (buffers re-seed '
                  f'from prior_path, which is what keeps the churn test real).')
        else:
            dropped.append('conditional MLE warm start: --cond-mle-seed "" -- f2/f2b run '
                           'cold, so they enter var_conditioning from ~200 steps of MLE '
                           'rather than a trained policy')
    emit(arms, dropped, outdir)


if __name__ == '__main__':
    main()
