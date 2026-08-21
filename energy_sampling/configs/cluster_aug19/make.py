"""
cluster_aug19 -- the battery `docs/design/cluster_plan_aug19.md` specifies.

    python configs/cluster_aug19/make.py      # configs + per-wave INDEX + sbatch

Every arm here answers a numbered question in that plan and carries a prediction
it can FAIL. Read the plan first; this file is its executable half and does not
restate the reasoning.

=============================================================================
EVERY ARM IS FRESH. No warm starts, deliberately.
=============================================================================
Two reasons, and the second is the one that bites:

  * no cluster-side archive has been verified to exist from this machine, and an
    arm whose `checkpoint_name` points at a missing file is a launch failure
    dressed as a config;
  * warm-starting from a `_phase1_exit.pt` archive does NOT skip phase 1
    (`skip_if: prior_loaded` only fires at step_ind 0, protocol.py:1335), so a
    resumed arm silently re-enters the MLE stage and measures the wrong thing --
    cluster_plan §6b, found the hard way locally.

The MLIP and sizer arms therefore run a SINGLE TERMINAL STAGE from step 0, which
is also the F-046-safe shape: the replay buffer fills at the run's own T instead
of inheriting a trajectory length fixed at some archive's write time.

=============================================================================
WAVES, and why they are separate submissions
=============================================================================
`gate` is the only true barrier (plan §6). Three switches in the last battery
reported success without reaching the code, so nothing long should burn hours
before the gate reads clean. Everything after it may be submitted at once.

  gate   G1-G3   3 arms   01:00:00   torch_cluster + executed-path flags + compile
                            (was 00:30:00 -- too short: the MLIP arms spend ~22 min
                             re-analysing the full prior before step 1, and g1 died
                             at step 190/200 with its OOM count frozen since step 10)
  prod   P1-P5   3 arms   12:00:00   production shakeout, the whole stack, hours
  sizer  B1-B5   5 arms   04:00:00   the occupancy ladder, incl. a REPEAT for B2
  mlip   M1-M2   4 arms   03:00:00   the two A/Bs the handoff left unrun
  cost   C1-C2   3 arms   02:00:00   what this week's changes actually cost
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'configs'))

import generate                                    # noqa: E402

TAG = 'cluster_aug19'

# Cluster paths, transcribed from configs/a100_stab_aug16/make.py.
CLUSTER_PRIOR_DIR = '/scratch/mk8347/data/crystal_datasets/conditional/priors/'
CLUSTER_CKPT_DIR = ('/scratch/mk8347/projects/gfn_cond/gfn-diffusion/'
                    'energy_sampling/checkpoints/')
CLUSTER_UMA_MLIP = '/scratch/mk8347/models/uma/esen_s.pt'
CLUSTER_MACE_MLIP = '/scratch/mk8347/data/acr_112025_mh1_stagetwo.model'

UMA_PRIOR = CLUSTER_PRIOR_DIR + 'mipcas_sg2_zp1_uma_prior_dataset.pt'
MACE_PRIOR = CLUSTER_PRIOR_DIR + 'acridine_sg14_zp1_mace_prior_dataset.pt'
ELJ_PRIOR = CLUSTER_PRIOR_DIR + 'mipcas_sg2_zp1_elj_prior_dataset.pt'
QM9_PRIOR = CLUSTER_PRIOR_DIR + 'qm9split_prior.pt'
QM9_CONDS = CLUSTER_PRIOR_DIR + 'qm9split_conditions.pt'
QM9_TEST = CLUSTER_PRIOR_DIR + 'qm9split_test_conditions.pt'

BASE = dict(checkpoints_dir=CLUSTER_CKPT_DIR, continue_from_checkpoint=False,
            checkpoint_name=None, prior_model_name=None,
            load_weights_only=False, checkpoint_read_only=False,
            cuda_memory_fraction=0.9)

#: The ladder OFF. Used wherever the arm measures something else -- a moving
#: batch changes work per step, the compile shape and the memory profile at once.
#: Written explicitly because canonical now ships it ARMED (state 9): an arm that
#: means "no ladder" and says nothing gets one.
LADDER_OFF = dict(batch_util_target=0, grow_batch_size=False)


def terminal(protocol: str, stage: str):
    """The named stage as the run's ONLY stage -- see the module docstring."""
    st = None
    for s in generate.canonical()['protocols'][protocol]['stages']:
        if s['name'] == stage:
            st = {k: v for k, v in s.items() if k != 'on_enter'}
    assert st is not None, f'{protocol} has no stage {stage}'
    return [st]


EQ = 'protocols.unconditional_tb.stages'
EQUIL = terminal('unconditional_tb', 'equilibration')


def arms():
    out = {}

    # ================================================================ gate ===
    # G1/G2: torch_cluster + the executed-path flags, on both MLIP routes.
    # G3: does anything break only under compile.
    out[f'{TAG}_g1_mace_gate'] = generate.arm(
        f'{TAG}_g1_mace_gate', problem='mipcas_elj', tag=TAG,
        energy_function='mace', mlip_path=CLUSTER_MACE_MLIP,
        space_groups=[14], z_primes=[1],
        prior_path=MACE_PRIOR, molecules_path=MACE_PRIOR,
        batch_size=100, max_batch_size=100, fused_grad_accum_min_samples=100,
        epochs=200, eval_period=100, figs_period=200, archive_period=0,
        **{**BASE, **LADDER_OFF, EQ: EQUIL})

    out[f'{TAG}_g2_uma_gate'] = generate.arm(
        f'{TAG}_g2_uma_gate', problem='mipcas_elj', tag=TAG,
        energy_function='uma', mlip_path=CLUSTER_UMA_MLIP,
        prior_path=UMA_PRIOR, molecules_path=UMA_PRIOR,
        batch_size=250, max_batch_size=250, fused_grad_accum_min_samples=250,
        epochs=200, eval_period=100, figs_period=200, archive_period=0,
        **{**BASE, **LADDER_OFF, EQ: EQUIL})

    out[f'{TAG}_g3_elj_compile'] = generate.arm(
        f'{TAG}_g3_elj_compile', problem='mipcas_elj', tag=TAG,
        prior_path=ELJ_PRIOR, molecules_path=ELJ_PRIOR,
        batch_size=1000, max_batch_size=1000, epochs=400,
        eval_period=200, figs_period=400, archive_period=0,
        **{**BASE, **LADDER_OFF, EQ: EQUIL})

    # ================================================================ prod ===
    # P1-P5. Production, not diagnostic: full prior, production T, sizer ARMED,
    # 0.9 of the card, checkpoints writing, hours. B4 (the S2 audit) rides here
    # because this is the only arm held past a 7200 s policy window.
    #
    # THE LADDER IS SHORT ON MLIP, and that is the plan's P3 decision made: at
    # ~181 s/step a 50-step dwell is 2.5 h per rung and 21 rungs cannot finish,
    # so max_batch_size sits near the route's real memory ceiling (the batch is
    # energy-call bound there; 20000 is fiction) and the dwell drops to 10.
    # The 3-sample occupancy requirement is met by a SINGLE step at that speed.
    for name, ef, mlip, prior, sg, batch, maxb in (
            ('p1_uma_prod', 'uma', CLUSTER_UMA_MLIP, UMA_PRIOR, [2], 250, 2000),
            ('p2_mace_prod', 'mace', CLUSTER_MACE_MLIP, MACE_PRIOR, [14], 100, 800)):
        out[f'{TAG}_{name}'] = generate.arm(
            f'{TAG}_{name}', problem='mipcas_elj', tag=TAG,
            energy_function=ef, mlip_path=mlip, space_groups=sg, z_primes=[1],
            prior_path=prior, molecules_path=prior,
            batch_size=batch, fused_grad_accum_min_samples=batch,
            max_batch_size=maxb, grow_batch_size=True, batch_util_target=0.6,
            batch_growth_interval=10,
            epochs=200000, eval_period=500, figs_period=1000,
            archive_period=5000, traj_checkpoint=True,
            # eval_T MUST track integrator.T -- the policy learns drift and
            # variance per step at one dt, so evaluating at another integrates a
            # different SDE and the wass/r2 numbers become a dt artifact. The
            # loader refuses the mismatch; this is that refusal obeyed.
            **{'integrator.T': 60, 'eval_T': 60, **BASE, EQ: EQUIL})

    # The conditional route in production. Included because canonical adopted
    # `level_gap: 1` on the strength of ONE 1000-step local arm, and that is a
    # thin basis for a default on the project's main experimental line.
    out[f'{TAG}_p3_qm9_cond_prod'] = generate.arm(
        f'{TAG}_p3_qm9_cond_prod', problem='qm9_conditional', tag=TAG,
        prior_path=QM9_PRIOR, molecules_path=QM9_CONDS,
        test_molecules_path=QM9_TEST,
        batch_size=1000, fused_grad_accum_min_samples=1000,
        max_batch_size=20000, grow_batch_size=True, batch_util_target=0.6,
        epochs=200000, eval_period=500, figs_period=1000, archive_period=5000,
        **BASE)

    # =============================================================== sizer ===
    # B1 + B2: the SAME arm twice. Reproducibility of the selection is the
    # question -- locally two runs of one route disagreed by >11 occupancy points
    # and picked different rungs.
    for rep in ('b1_ladder_r1', 'b2_ladder_r2'):
        out[f'{TAG}_{rep}'] = generate.arm(
            f'{TAG}_{rep}', problem='mipcas_elj', tag=TAG,
            prior_path=ELJ_PRIOR, molecules_path=ELJ_PRIOR,
            batch_size=1000, fused_grad_accum_min_samples=1000,
            max_batch_size=20000, grow_batch_size=True, batch_util_target=0.6,
            epochs=40000, eval_period=500, figs_period=1000, archive_period=5000,
            **{**BASE, EQ: EQUIL})

    # B5: does batch buy occupancy at the rung F-045 names as missing? FIXED
    # batches, no ladder -- the ladder would confound the very axis being swept.
    for b in (7410, 12000, 20000):
        out[f'{TAG}_b5_fixed{b}'] = generate.arm(
            f'{TAG}_b5_fixed{b}', problem='mipcas_elj', tag=TAG,
            prior_path=ELJ_PRIOR, molecules_path=ELJ_PRIOR,
            batch_size=b, max_batch_size=b, fused_grad_accum_min_samples=1000,
            epochs=20000, eval_period=500, figs_period=1000, archive_period=5000,
            **{**BASE, **LADDER_OFF, EQ: EQUIL})

    # ================================================================ mlip ===
    # M1: batched NL alone vs batched NL + device-built dict. The handoff's
    # retracted arm -- it ran the second flag with the first OFF, so the branch
    # was never entered. The switches are ENV vars, so they live in <arm>.env.
    for name in ('m1_mace_nl_only', 'm1_mace_nl_gpubatch'):
        out[f'{TAG}_{name}'] = generate.arm(
            f'{TAG}_{name}', problem='mipcas_elj', tag=TAG,
            energy_function='mace', mlip_path=CLUSTER_MACE_MLIP,
            space_groups=[14], z_primes=[1],
            prior_path=MACE_PRIOR, molecules_path=MACE_PRIOR,
            batch_size=100, max_batch_size=100, fused_grad_accum_min_samples=100,
            epochs=600, eval_period=300, figs_period=600, archive_period=0,
            **{**BASE, **LADDER_OFF, EQ: EQUIL})

    # M2: does the UMA external graph buy speed, or is it correctness only?
    for name in ('m2_uma_extgraph_on', 'm2_uma_extgraph_off'):
        out[f'{TAG}_{name}'] = generate.arm(
            f'{TAG}_{name}', problem='mipcas_elj', tag=TAG,
            energy_function='uma', mlip_path=CLUSTER_UMA_MLIP,
            prior_path=UMA_PRIOR, molecules_path=UMA_PRIOR,
            batch_size=250, max_batch_size=250, fused_grad_accum_min_samples=250,
            epochs=600, eval_period=300, figs_period=600, archive_period=0,
            **{**BASE, **LADDER_OFF, EQ: EQUIL})

    # ================================================================ cost ===
    # C1: fused Adam, on vs off. Locally worth -10.4% step time and +10.2%
    # occupancy, but n=1 and ORDER-CONFOUNDED (the replicate was never run), and
    # compile may absorb or amplify it. Switch is an env var.
    for name in ('c1_fused_on', 'c1_fused_off'):
        out[f'{TAG}_{name}'] = generate.arm(
            f'{TAG}_{name}', problem='mipcas_elj', tag=TAG,
            prior_path=ELJ_PRIOR, molecules_path=ELJ_PRIOR,
            batch_size=1000, max_batch_size=1000, epochs=3000,
            eval_period=1000, figs_period=2000, archive_period=0,
            **{**BASE, **LADDER_OFF, EQ: EQUIL})

    # C2: WHERE are the other ~98% of device->host syncs? ~11 600/step measured
    # locally, only ~220 visible from Python. with_stack is what attributes them,
    # and it is now affordable for many steps because the 748 MB chrome trace
    # became opt-in (write_trace, state 9).
    out[f'{TAG}_c2_syncs'] = generate.arm(
        f'{TAG}_c2_syncs', problem='mipcas_elj', tag=TAG,
        prior_path=ELJ_PRIOR, molecules_path=ELJ_PRIOR,
        batch_size=1000, max_batch_size=1000, epochs=1200,
        eval_period=600, figs_period=1200, archive_period=0,
        **{**BASE, **LADDER_OFF, EQ: EQUIL,
           'profiling.enabled': True,
           'profiling.trace.enabled': True,
           'profiling.trace.start_step': 400,
           'profiling.trace.active_steps': 4,
           'profiling.trace.with_stack': True,
           'profiling.trace.write_trace': False})

    # ================================================================= mem ===
    # WHY THIS WAVE EXISTS. g1_mace_gate OOM'd three times in its first ten
    # steps and cut batch 100 -> 23, then ran 180 steps stably. The memory trace
    # says it was never short of memory: LIVE allocation was flat at 890 MiB
    # (1.1% of the card) while RESERVED sat at 57.7 GB, of which 98.5% was
    # cached-but-held. cuda_memory_fraction is a HARD cap that counts those
    # unusable blocks, so the allocation failed with the card effectively empty.
    #
    # Historically this route ran an ENERGY batch of 79 against a POLICY batch of
    # 3000, with the energy call chunking internally. That is disabled by design
    # (grad accumulation is the simpler equivalent), so the only lever the OOM
    # path has is the policy batch -- which is why 100 became 23.
    #
    # THE QUESTION: is the ceiling fragmentation, trajectory activations, or the
    # cap itself? Each arm isolates one, all at batch 100, all otherwise
    # identical to g1. mem0 is the control and must reproduce the crash, or the
    # wave has no power and every other arm is uninterpretable.
    #
    # NB none of this was runnable before today: train.py ASSIGNED
    # PYTORCH_CUDA_ALLOC_CONF rather than defaulting it, so anything exported
    # here was clobbered by the process it was meant to configure.
    for name, extra in (
            ('mem0_mace_control', {}),
            ('mem1_mace_gc', {}),
            ('mem2_mace_split', {}),
            ('mem3_mace_trajckpt', dict(traj_checkpoint=True)),
            ('mem4_mace_cap97', dict(cuda_memory_fraction=0.97))):
        out[f'{TAG}_{name}'] = generate.arm(
            f'{TAG}_{name}', problem='mipcas_elj', tag=TAG,
            energy_function='mace', mlip_path=CLUSTER_MACE_MLIP,
            space_groups=[14], z_primes=[1],
            prior_path=MACE_PRIOR, molecules_path=MACE_PRIOR,
            batch_size=100, max_batch_size=100, fused_grad_accum_min_samples=100,
            epochs=200, eval_period=100, figs_period=200, archive_period=0,
            **{**BASE, **LADDER_OFF, EQ: EQUIL, **extra})

    # =============================================================== nlcap ===
    # THE DIRECT TEST OF THE SHIFT CAP, and the reason it is its own wave: the
    # mem wave established that MACE at batch 100 cascades 100 -> 23 with three
    # OOMs, and local profiling found why. `lattice_shift_range` clamps only
    # from BELOW, so a flattening cell's interplanar spacing -> 0 makes the
    # requested shift range unbounded -- and `batched_pbc_neighbour_list` takes
    # `.max(dim=0)` across the batch, ghost-expanding EVERY graph on the worst
    # cell's grid. Measured locally at 128 acridine graphs / 2944 atoms:
    #
    #     physical   grid [3,3,2]   K=245      peak  254 MiB
    #     x0.01      grid [3,3,52]  K=5,145    peak 1725 MiB
    #     x0.001     grid [3,3,510] K=50,029   OOM on a 16 GB card
    #
    # while the EDGE COUNT rose only 2.5x -- so it is ghost expansion the sane
    # cells never needed. A fresh policy emits exactly those cells, which is why
    # the failure is an early transient.
    #
    # Both arms are batch 100 -- the size that cascaded -- and short, because
    # all three OOMs landed inside the first ten steps. `off` restores the old
    # unbounded behaviour via the env override, so this is a single-key toggle
    # rather than a comparison between two builds.
    #
    # PREDICTIONS. off: reproduces the mem wave -- 3 OOM events, batch collapses
    # to 23. on: `energy/nl_shift_capped_frac` > 0 in the first reports (the
    # degenerate cells are there and the cap is catching them), `batch/oom_events`
    # 0, and the batch HOLDS at 100. If `capped_frac` stays 0 the cluster's
    # degeneracy never reached the cap and this is not the cluster's mechanism --
    # a real negative, and the arm is built to show it.
    for name in ('nl0_shiftcap_off', 'nl1_shiftcap_on'):
        out[f'{TAG}_{name}'] = generate.arm(
            f'{TAG}_{name}', problem='mipcas_elj', tag=TAG,
            energy_function='mace', mlip_path=CLUSTER_MACE_MLIP,
            space_groups=[14], z_primes=[1],
            prior_path=MACE_PRIOR, molecules_path=MACE_PRIOR,
            batch_size=100, max_batch_size=100, fused_grad_accum_min_samples=100,
            epochs=150, eval_period=75, figs_period=150, archive_period=0,
            **{**BASE, **LADDER_OFF, EQ: EQUIL})

    # THE CEILING-FINDER, and the arm that answers the objection to F-049. The
    # mem wave is easy to misread as "MACE cannot hold a coupled batch", but this
    # hardware has previously run energy batch 73 against policy batch 3000 at
    # T=100, so the ceiling is an ENGINEERING NUMBER, not a wall -- and nothing
    # in the battery measures it. This arm does: cap on, ladder ARMED from a
    # deliberately small rung, and the settling batch is the answer.
    #
    # Starts at 25, not 100. If the cap works the ladder climbs in ~10 steps, and
    # starting BELOW the size that cascaded means a failure to climb is
    # informative rather than a repeat of the crash. traj_checkpoint on, since
    # the ceiling worth knowing is the one under the memory settings prod uses.
    #
    # PREDICTION: settles above 100 -- somewhere in 100-400. Below 100 means the
    # cap is not the binding constraint and the activation transient is, which
    # sends the next round at the energy-call peak rather than the neighbour list.
    out[f'{TAG}_nl2_shiftcap_ladder'] = generate.arm(
        f'{TAG}_nl2_shiftcap_ladder', problem='mipcas_elj', tag=TAG,
        energy_function='mace', mlip_path=CLUSTER_MACE_MLIP,
        space_groups=[14], z_primes=[1],
        prior_path=MACE_PRIOR, molecules_path=MACE_PRIOR,
        batch_size=25, fused_grad_accum_min_samples=25,
        max_batch_size=2000, grow_batch_size=True, batch_util_target=0.6,
        batch_growth_interval=10, traj_checkpoint=True,
        epochs=400, eval_period=200, figs_period=400, archive_period=0,
        **{**BASE, EQ: EQUIL})

    return out


#: Per-arm environment, written beside the config. The MLIP construction paths
#: and the Adam implementation are ENV switches, not config keys, so an A/B on
#: them cannot be expressed in the YAML at all -- and an arm differing from its
#: control only by an unrecorded variable is unreadable after the fact.
ENV = {
    f'{TAG}_m1_mace_nl_only': {'MXT_BATCHED_MACE_NEIGHBOURS': '1',
                               'MXT_GPU_MACE_BATCH': '0'},
    f'{TAG}_m1_mace_nl_gpubatch': {'MXT_BATCHED_MACE_NEIGHBOURS': '1',
                                   'MXT_GPU_MACE_BATCH': '1'},
    f'{TAG}_m2_uma_extgraph_on': {'MXT_UMA_EXTERNAL_GRAPH': '1'},
    f'{TAG}_m2_uma_extgraph_off': {'MXT_UMA_EXTERNAL_GRAPH': '0'},
    f'{TAG}_c1_fused_on': {'MXT_FUSED_ADAM': '1'},
    f'{TAG}_c1_fused_off': {'MXT_FUSED_ADAM': '0'},
    # the shift cap, as a toggle. 100000 is "no cap" -- large enough that the
    # clamp can never bind, restoring the pre-2026-08-20 behaviour exactly.
    f'{TAG}_nl0_shiftcap_off': {'MXT_MAX_SHIFT_RANGE': '100000'},
    f'{TAG}_nl1_shiftcap_on': {'MXT_MAX_SHIFT_RANGE': '8'},
    f'{TAG}_nl2_shiftcap_ladder': {'MXT_MAX_SHIFT_RANGE': '8'},
    f'{TAG}_g2_uma_gate': {'MXT_UMA_GRAPH_TIMER': '1'},
    # ---- the allocator arms. mem0 sets NOTHING, so it takes train.py's
    # setdefault floor (expandable_segments) and is the control for the two
    # below it. mem3/mem4 vary a config key instead and inherit the same floor,
    # which is what keeps each arm a single-variable change.
    f'{TAG}_mem1_mace_gc': {
        'PYTORCH_CUDA_ALLOC_CONF':
            'expandable_segments:True,garbage_collection_threshold:0.8'},
    f'{TAG}_mem2_mace_split': {
        'PYTORCH_CUDA_ALLOC_CONF':
            'expandable_segments:True,garbage_collection_threshold:0.8,'
            'max_split_size_mb:128'},
}

WAVES = {
    'gate':  ('01:00:00', ['g1_mace_gate', 'g2_uma_gate', 'g3_elj_compile']),
    'prod':  ('12:00:00', ['p1_uma_prod', 'p2_mace_prod', 'p3_qm9_cond_prod']),
    'sizer': ('04:00:00', ['b1_ladder_r1', 'b2_ladder_r2',
                           'b5_fixed7410', 'b5_fixed12000', 'b5_fixed20000']),
    'mlip':  ('03:00:00', ['m1_mace_nl_only', 'm1_mace_nl_gpubatch',
                           'm2_uma_extgraph_on', 'm2_uma_extgraph_off']),
    'cost':  ('02:00:00', ['c1_fused_on', 'c1_fused_off', 'c2_syncs']),
    # 01:30:00, not 00:30:00: g1 spent ~22 of its 24 minutes on the init prior
    # re-analysis (205k rows through MACE) and ~1.6 min actually training, then
    # died at step 190 of 200 -- almost certainly the walltime, since
    # batch/oom_events had been frozen at 3 since step 10.
    # 01:00:00: 150 steps, and the whole question is decided in the first ten.
    # The MACE init prior re-analysis is the bulk of the wall clock either way.
    'nlcap': ('01:30:00', ['nl0_shiftcap_off', 'nl1_shiftcap_on',
                           'nl2_shiftcap_ladder']),
    'mem':   ('01:30:00', ['mem0_mace_control', 'mem1_mace_gc',
                           'mem2_mace_split', 'mem3_mace_trajckpt',
                           'mem4_mace_cap97']),
}

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
#SBATCH --job-name=clu19_{label}
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/cluster_aug19/joblogs/%x_%A_%a.out

# cluster_aug19 :: {label}. Arm = row of INDEX_{label}.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index; a short
# range drops tail arms with no error at all.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/cluster_aug19
LOGS=${{ARMS}}/joblogs

ARM=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $1}}' ${{ARMS}}/INDEX_{label}.tsv)
if [ -z "${{ARM}}" ]; then echo "no arm at row ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; fi
CONFIG=${{ARMS}}/${{ARM}}.yaml
if [ ! -f "${{CONFIG}}" ]; then echo "missing config ${{CONFIG}}" >&2; exit 1; fi
echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}"
J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}

# ---- one-shot environment record. MIG turns the in-process sensor off SILENTLY
# and permanently; the UUID is what proves both samplers read the same card.
{{ nvidia-smi -L
  nvidia-smi --query-gpu=mig.mode.current,uuid,name,memory.total,driver_version --format=csv
  scontrol show job ${{SLURM_JOB_ID}}
  echo "nodelist: ${{SLURM_NODELIST}}  host: $(hostname)"
}} > ${{J}}.info 2>&1

# ---- concurrent samplers spanning the WHOLE job, so the denominator matches the
# scheduler's. FILTER TO OUR GPU INDEX when reading these: sample output shows
# 4-GPU nodes at wildly different utilizations, and a co-tenant is the cluster's
# version of the confound that muddied the local sizer comparison.
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \\
    --format=csv,nounits -l 10 > ${{J}}_smi.csv &
SMI_PID=$!
stdbuf -oL nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,nounits \\
    -l 30 > ${{J}}_apps.csv &
APPS_PID=$!
if command -v dcgmi >/dev/null 2>&1; then
    # SM_ACTIVE (field 1002): the only genuinely DIFFERENT instrument here.
    # utilization.gpu is an any-kernel-resident duty cycle, blind to how MUCH of
    # the GPU a kernel uses -- and F-048 notes these files have never been read.
    stdbuf -oL dcgmi dmon -e 1002 -d 10000 > ${{J}}_dcgm.txt 2>&1 &
    DCGM_PID=$!
else
    echo "dcgmi not available" > ${{J}}_dcgm.txt
    DCGM_PID=""
fi

# ---- epilogue that survives scancel: on a cancelled job THIS is the record.
epilogue() {{
    kill ${{SMI_PID}} ${{APPS_PID}} ${{DCGM_PID}} 2>/dev/null
    sacct -j ${{SLURM_JOB_ID}} --format=JobID,State,ExitCode,Elapsed,NodeList,Reason,Comment%64 \\
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
        # PER-ARM ENV, one KEY=VAL per line in <arm>.env. SINGLE quotes only in
        # this block, comments included: it is interpolated inside the srun bash
        # -c string, so a double quote truncates the command and train.py never
        # starts, with no wandb run and a file that still passes bash -n.
        if [ -f ${{ARMS}}/${{ARM}}.env ]; then
            set -a; . ${{ARMS}}/${{ARM}}.env; set +a
            echo 'arm env:'; cat ${{ARMS}}/${{ARM}}.env
        fi
        python -u train.py --config ${{CONFIG}}
    "
"""


def check(cfgs):
    for name, cfg in cfgs.items():
        assert cfg['continue_from_checkpoint'] is False, name
        assert cfg.get('checkpoint_name') is None, f'{name}: no warm starts here'
        assert (cfg.get('mlip_path') is None) == \
               (cfg['energy_function'] not in ('uma', 'mace')), name
        assert cfg['figs_period'] % cfg['eval_period'] == 0, name
        t = float(cfg.get('batch_util_target') or 0)
        assert 0 <= t <= 1, f'{name}: batch_util_target {t} is not a fraction'
        if t > 0:
            assert cfg['grow_batch_size'] is True, f'{name}: armed but growth off'
            assert cfg['max_batch_size'] > cfg['batch_size'], f'{name}: one rung'
    covered = {f'{TAG}_{a}' for _, arms_ in WAVES.values() for a in arms_}
    assert covered == set(cfgs), (
        f'waves and arms disagree: {covered ^ set(cfgs)}')
    print(f'  checks passed on {len(cfgs)} arms')


if __name__ == '__main__':
    cfgs = arms()
    check(cfgs)
    generate.emit(cfgs, outdir=HERE, index=False)

    from pathlib import Path
    outdir = Path(HERE)
    (outdir / 'joblogs').mkdir(exist_ok=True)
    for arm, env in ENV.items():
        (outdir / f'{arm}.env').write_text(
            ''.join(f'{k}={v}\n' for k, v in env.items()), encoding='utf-8')

    print()
    for wave, (limit, names) in WAVES.items():
        rows = [f'{TAG}_{n}' for n in names]
        idx = ['name\tproblem\tbatch\tepochs'] + [
            f'{r}\t{cfgs[r]["energy_function"]}\t{cfgs[r]["batch_size"]}'
            f'\t{cfgs[r]["epochs"]}' for r in rows]
        (outdir / f'INDEX_{wave}.tsv').write_text('\n'.join(idx) + '\n',
                                                  encoding='utf-8')
        txt = SBATCH_TEMPLATE.format(label=wave, time=limit,
                                     array=f'0-{len(rows) - 1}')
        assert '"' not in txt.split('/bin/bash -c "')[1].split('python -u')[0] \
            .replace('\\$PYTHONPATH', ''), f'{wave}: double quote inside srun block'
        (outdir / f'submit_{wave}.sbatch').write_text(txt, encoding='utf-8')
        print(f'  submit_{wave}.sbatch   --array=0-{len(rows) - 1}   '
              f'{len(rows)} arms   time {limit}')
