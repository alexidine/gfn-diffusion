"""acr_b_aug31: acridine phase-2 LR scan at a PINNED, MEASURED batch, plus two batch probes.

    python configs/acr_b_aug31/make.py

Six arms, all seeded from the same frozen `pt100_acr_lr4p0` phase-1 exit (9010):

  arm        batch   scale   purpose
  lr0p25      2560    0.25   LR scan
  lr0p5       2560    0.5    LR scan
  lr1p0       2560    1.0    LR scan
  lr2p0       2560    2.0    LR scan
  b1500       1500    0.5    batch probe
  b2000       2000    0.5    batch probe

WHY THIS EXISTS RATHER THAN AN EDIT TO prod_t100_p2. Two reasons, both about not
destroying live state: `prod_t100_p2/make.py` writes all 20 of that battery's
YAMLs, and the nehu2 arms there have been hand-edited on the cluster to a fixed
batch -- regenerating would silently overwrite that. And the acr2 arms there are
LIVE, so a new fan cannot seed from their `_running.pt` files: four array tasks
start at different times and would glob the same file at different steps, which
is the failure the sbatch's phase1_exit-only rule exists to prevent. Seeding all
six from the frozen phase-1 exit instead costs ~2400 equilibration steps, ~1400
of which were run at batch 8000 -- a size this battery exists to abandon.

------------------------------------------------------- WHY BATCH 2560, MEASURED

`internal_oom_recovery: True` (shipped 3cf6b95) was dispositive for acridine: it
took the batch from 240 to 8000 by handing the ceiling back to the ROLLOUT, which
`traj_checkpoint` already makes ~O(1) in T. Before it, MACE evaluated the whole
rollout in ONE call, so the MLIP's per-call memory set the TRAINING batch -- 64
GiB allocations at batch 243, occupancy 32-38%, and every arm cancelled.

But 8000 is far past the useful point. The ladder walked 11 rungs on four
independent arms, 20 occupancy samples each, and they agree:

  batch   step_s   samples/s   steps/s   util(ours)   energy frac
   1000     10.7          93     0.093         38.2          0.46
   1600     13.7         117     0.073         49.0          0.58
   2560     18.9         135     0.053         64.5          0.67
   3560     25.7         138     0.039         78.3          0.70
   4560     30.2         151     0.033         84.5          0.73
   8000     54.6         146     0.018         91.0          0.76

Sample throughput SATURATES by ~4560 and never improves. Optimizer updates --
the decided objective (mk_dev.yaml: "occupancy constraint first, then
optimizer-step throughput, whose answer is the CONSTANT batch_size =
fused_grad_accum_min_samples") -- fall monotonically. 2560 buys 2.9x the updates
of 8000 for 8% less sample throughput.

Occupancy at 2560 is defensible against the only bracket that has been measured
(a100_stab_aug16: cancelled at <=40% external, survived at >=49.4%; qm9anchor_aug14
ran 34-48 h uncancelled at 57-68%). Our in-process sensor reads 64.5 there, and
the out-of-process smi sidecar calibrates it at batch 8000 as ours 91.0 against
external 93.9-97.1 -- i.e. ours runs 3-6 points LOW on this route, the OPPOSITE
of the ELJ table, because `energy/frac_of_step` is 0.76 here and the step is
dense MLIP work rather than dispatch-bound. That maps 2560 to external ~64-75.
OPEN: the offset is measured only at 8000; at 2560 the energy fraction is 0.67,
so re-run the smi check once these are up.

-------------------------------------------------- WHY PINNED AND NOT SERVO'D

`batch_util_target` cannot pick between adjacent rungs here. Per-arm occupancy at
2560 is 60.0 / 65.9 / 60.9 / 69.0 -- so a target of 0.65 holds 2560 on two arms
and pushes the other two to 3560, and an LR fan split across two batch sizes is
confounded (batch moves the stable LR). The window that selects 2560 for ALL arms
is (52.4, 60.0], 7.6 points wide, against 9 points of arm-to-arm spread at a
single rung. The threshold is narrower than the noise it must discriminate.

So the batch is a DECISION: base and ceiling equal, target 0 -- the shipping
"hold the base, no probe" path. `grow_batch_size` stays true so select_batch_size
still RESTORES the base after an OOM cut; nothing else can move the size.

The two probes exist because the 1600 and 2560 readings above came from 50-step
ladder dwells mid-climb. A pinned run gives thousands of steps and a real
out-of-process number at each size. Expect 1500 to land near the measured 1600
rung (ours ~49 -> external ~53): above the survival bracket, but only ~4 points
clear of it, on 24 h jobs.

Both probes run scale 0.5 so the three batch points share a rate. 0.5 sits inside
the predicted phase-2 optimum (0.55-1.1 of seed) and is the safer central pick:
a smaller batch is noisier, which LOWERS the stable LR, so the probes are the
arms most exposed to detonation.

READ `lr_ctrl/scale`, NOT THE ARM NAME. `hot_lr_sensor.action` is 'fire' here,
matching prod_t100_p2 so this stays comparable with the mip2/neh2 families, and a
fire both rewinds AND halves the rate permanently (fixed mode never re-races).
Any arm that fires stops naming its own rate.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIGS = HERE.parent
MK_DEV = CONFIGS / 'mk_dev.yaml'

CLUSTER_DATA = '/scratch/mk8347/data/crystal_datasets/conditional/priors'
CLUSTER_CKPTS = '/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints/'
PLACEHOLDER = 'WARM_CHECKPOINT_PLACEHOLDER'

SHIP_T = 100
EXCURSION_K = 60.0
LEVEL_WINDOW = 5000
PHASE2_STEPS = 12000

WARM_SRC = 'pt100_acr_lr4p0'
EXIT_STEP = 9010
PRIOR = f'{CLUSTER_DATA}/acridine_sg14_zp1_mace_prior_dataset.pt'
MLIP = '/scratch/mk8347/data/acr_112025_mh1_stagetwo.model'
FAN_BATCH = 2560

#: (name, batch, fixed_scale)
ARMS = [
    ('lr0p25', FAN_BATCH, 0.25),
    ('lr0p5',  FAN_BATCH, 0.5),
    ('lr1p0',  FAN_BATCH, 1.0),
    ('lr2p0',  FAN_BATCH, 2.0),
    ('b1500',  1500,      0.5),
    ('b2000',  2000,      0.5),
]


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build():
    arms = {}
    for name, batch, scale in ARMS:
        cfg = base()
        run = f'acrb_{name}'

        cfg['run_name'] = run
        cfg['tag'] = 'acrb31'
        cfg['checkpoints_dir'] = CLUSTER_CKPTS
        cfg['prior_path'] = PRIOR
        # one molecule, unconditional: the condition set IS the prior file.
        # mk_dev ships a LOCAL D:\ path here, which passes local preflight and
        # 404s on the cluster (prod_aug26, 2026-08-27).
        cfg['molecules_path'] = PRIOR
        cfg['test_molecules_path'] = None
        cfg['space_groups'] = [14]
        cfg['energy_function'] = 'mace'
        cfg['mlip_path'] = MLIP

        cfg['checkpoint_name'] = PLACEHOLDER
        cfg['load_weights_only'] = False
        cfg['continue_from_checkpoint'] = False
        cfg['prior_model_name'] = None

        cfg['integrator']['T'] = SHIP_T
        cfg['eval_T'] = SHIP_T
        cfg['traj_checkpoint'] = True
        # THE DISPOSITIVE SETTING for acridine occupancy -- see the docstring.
        # With it False, MACE evaluates the whole rollout in one call and its
        # per-call memory caps the TRAINING batch at ~240.
        cfg['energy_config']['internal_oom_recovery'] = True

        cfg['epochs'] = EXIT_STEP + PHASE2_STEPS
        cfg['eval_period'] = 1000
        cfg['eval_num_samples'] = 2500
        cfg['figs_period'] = 1000
        cfg['progress_gate']['level_window'] = LEVEL_WINDOW

        # PINNED. base == ceiling and target 0 is the shipping
        # "hold the base, no probe" path. grow_batch_size stays TRUE so
        # select_batch_size still restores the base after an OOM cut.
        cfg['batch_size'] = batch
        cfg['max_batch_size'] = batch
        cfg['grow_batch_size'] = True
        cfg['batch_util_target'] = 0

        cfg['max_reloads_per_1k_steps'] = 1.0

        lc = cfg['lr_control']
        lc['mode'] = 'fixed'
        lc['fixed_scale'] = float(scale)
        lc['burn_in_steps'] = 500
        lc['burn_in_scale'] = 0.05
        lc['repeat_every'] = 0
        lc['hard_failure']['loss_excursion_k'] = EXCURSION_K

        cfg['protocol'] = 'prod_eq'
        cfg['protocols']['prod_eq'] = {'stages': [
            # A RE-ENTRY STUB, NOT A TRAINING STAGE. It must exist and must be
            # named train_prior: the checkpoint carries its stage BY NAME and
            # StageProtocol raises otherwise, and a single-stage protocol would
            # never fire equilibration's on_enter (begin() returns early on a
            # resume).
            {
                'name': 'train_prior',
                'train_mode': 'bwd',
                'bwd_sampling_mode': 'dataset',
                'flags': {'update_log_z': True, 'scramble_conditions': True},
                'hot_lr_sensor': {'channel': 'bwd/mle', 'form': 'absolute',
                                  'rows': 31, 'above': 5.0, 'action': 'fire'},
                'loss_coeffs': {'bwd': {'mle': 1.0, 'tbc': 0.0, 'repeats': 1.0,
                                        'tb_z_source': 'persistent'}},
                # KEYED ON A TICK METRIC, NOT THE GATE. `_progress_history` is
                # not checkpointed, so a gate-keyed exit publishes 0 at the
                # first post-resume eval and wipes the streak restored from the
                # phase-1 exit -- which stranded all four nehzor pt100p2 arms in
                # this stub (2026-08-29).
                'exit': [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}],
                # snapshot_prior is MANDATORY: prior_model is not in the
                # checkpoint and only this creates it. No 'stop'.
                'on_exit': ['snapshot_prior'],
            },
            {
                'name': 'equilibration',
                'train_mode': 'fused',
                'bwd_sampling_mode': 'prior',
                'hot_lr_sensor': {'channel': 'fwd/scatter_err', 'rows': 11,
                                  'above': 2.0, 'action': 'fire'},
                'flags': {'update_log_z': True, 'buffers_active': True,
                          'z_calibration': True},
                'on_enter': ['rebuild_prior_by_churn',
                             'bootstrap_z:train_conditioner'],
                'fracs': {'fwd': 0.05, 'bwd': 0.93, 'replay': 0.02},
                'min_fracs': {'fwd': 0.02, 'bwd': 0.02, 'replay': 0.02},
                'deactivate_threshold': 0.01,
                'loss_coeffs': {
                    'fwd': {'tb': 1.0, 'freeze_policy': 1.0},
                    'bwd': {'tb': 1.0, 'beta': 80},
                    'replay': {'tb': 1.0, 'beta': 80},
                },
                'balance': {
                    'kind': 'ratio', 'pinned': {'fwd': 0.05},
                    'metrics': {'replay': 'fwd/over_coverage',
                                'bwd': 'bwd/relative_under_wcen'},
                    'numerator': 'replay', 'setpoint': 5.0, 'gain': 0.05,
                    'max_step': 0.05,
                    'bounds': {'replay': [0.02, 0.93], 'bwd': [0.02, 0.93]},
                    'converge_floor': 1.0,
                },
                'buffer_servo': {
                    'numerator': 'replay/ema_loss_mean',
                    'denominator': 'replay/birth_loss_mean',
                    'bar': 0.368, 'release': 0.6, 'scale': 0.15,
                    'gain': 0.05, 'relax': 0.5, 'max_step': 0.05,
                    'max_boost': 8.0,
                },
            },
        ]}

        check(cfg, name, batch, scale)
        arms[run] = cfg
    return arms


def check(cfg, name, batch, scale):
    assert cfg['integrator']['T'] == SHIP_T
    assert cfg['eval_T'] == cfg['integrator']['T'], \
        'utils hard-fails on eval_T != integrator.T at load'
    assert cfg['traj_checkpoint'] is True, \
        'with the MLIP chunking itself the ROLLOUT sets the batch, so the ' \
        'trajectory memory saving is load-bearing'
    assert cfg['energy_config']['internal_oom_recovery'] is True, \
        'without it MACE evaluates the whole rollout in one call and caps the ' \
        'training batch at ~240 -- the whole reason this battery exists'

    # THE BATCH IS A DECISION, and it must bind from BOTH sides: a base under
    # the ceiling with a live target would let the ladder walk up to it.
    assert cfg['batch_size'] == batch and cfg['max_batch_size'] == batch, \
        f'batch must be pinned at {batch} on both base and ceiling'
    assert cfg['batch_util_target'] == 0, \
        'a nonzero target selects the smallest rung clearing it, or walks to ' \
        'max_batch_size -- neither is the decision recorded in the docstring'
    assert cfg['grow_batch_size'] is True, \
        'growth stays on so select_batch_size restores the base after an OOM cut'

    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == float(scale)
    assert lc['hard_failure']['loss_excursion_k'] == EXCURSION_K
    assert cfg['max_reloads_per_1k_steps'] == 1.0

    stages = cfg['protocols']['prod_eq']['stages']
    assert len(stages) == 2
    st, eq = stages
    assert st['name'] == 'train_prior'
    assert st['on_exit'] == ['snapshot_prior'] and 'stop' not in st['on_exit']
    assert st['exit'] == [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}], \
        'the stub must exit on a TICK metric; a gate-keyed exit loses the race ' \
        'against the first post-resume progress_gate publish'
    assert eq['name'] == 'equilibration'
    assert eq.get('exit') is None, 'equilibration is terminal by design'
    assert eq['balance']['pinned']['fwd'] == eq['fracs']['fwd']
    assert eq['hot_lr_sensor']['channel'] == 'fwd/scatter_err', \
        "sensor bars are PER-STAGE; the stub's bwd/mle bar watches a channel " \
        'this stage does not live on'
    for br in ('fwd', 'bwd', 'replay'):
        terms = [k for k in eq['loss_coeffs'][br]
                 if k in ('tb', 'db', 'subtb', 'mle', 'tbc', 'vg_lb', 'vg_lme',
                          'emp_z', 'level_gap', 'z_level')
                 and float(eq['loss_coeffs'][br][k]) > 0]
        assert terms == ['tb'], f'{br} must be single-term tb, got {terms}'

    assert cfg['checkpoint_name'] == PLACEHOLDER
    assert cfg['load_weights_only'] is False, \
        'phase 2 must resume FULL state: the restored exit streak is what ' \
        'carries the arm through the train_prior stub'
    assert cfg['continue_from_checkpoint'] is False
    assert cfg['prior_model_name'] is None

    assert cfg['epochs'] == EXIT_STEP + PHASE2_STEPS
    assert cfg['epochs'] > EXIT_STEP + 1000, \
        'epochs is ABSOLUTE and the arm resumes at EXIT_STEP; too low and it ' \
        'runs (almost) nothing and reports state=finished'

    ep = cfg['eval_period']
    assert cfg['figs_period'] % ep == 0
    assert cfg['prior_path'] == PRIOR and cfg['molecules_path'] == PRIOR
    assert cfg['test_molecules_path'] is None
    assert cfg['energy_function'] == 'mace' and cfg['mlip_path'] == MLIP
    assert cfg['grad_clip_guard']['enabled'] is True
    assert_no_local_paths(cfg)


def assert_no_local_paths(node, trail='cfg'):
    """No Windows drive path may survive into a cluster arm -- it exists on the
    dev box, so local preflight passes, and 404s on the cluster."""
    if isinstance(node, dict):
        for k, v in node.items():
            assert_no_local_paths(v, f'{trail}.{k}')
    elif isinstance(node, list):
        for i, v in enumerate(node):
            assert_no_local_paths(v, f'{trail}[{i}]')
    elif isinstance(node, str):
        assert not (len(node) > 2 and node[1] == ':' and node[2] in '\\/'), \
            f'local drive path leaked into a cluster arm at {trail}: {node!r}'


SBATCH = """#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array={first}-{last}
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=acrb31
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/acr_b_aug31/joblogs/%x_%A_%a.out

# acr_b_aug31. Arm = row of INDEX.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/acr_b_aug31
LOGS=${{ARMS}}/joblogs
CKPTS={ckpts}
mkdir -p ${{LOGS}}

ARM=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $1}}' ${{ARMS}}/INDEX.tsv)
SRC=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $2}}' ${{ARMS}}/INDEX.tsv)
if [ -z "${{ARM}}" ]; then echo "no arm at row ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; fi
CONFIG=${{ARMS}}/${{ARM}}.yaml
if [ ! -f "${{CONFIG}}" ]; then echo "missing config ${{CONFIG}}" >&2; exit 1; fi

J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}
RESOLVED=${{J}}.yaml

# REQUEUE SAFETY. 12000 steps at ~19 s/step is ~63 h, so every arm is EXPECTED
# to hit the 24 h wall about three times. train.py's loader is
# `if checkpoint_name ... elif continue_from_checkpoint`, so checkpoint_name
# ALWAYS wins -- without this an extended arm would silently restart from the
# phase-1 exit and discard its own progress. Resubmit this same sbatch.
OWN=$(ls -t ${{CKPTS}}/*${{ARM}}_*_running.pt 2>/dev/null | head -1)
if [ -n "${{OWN}}" ]; then
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  REQUEUE: resuming own $(basename ${{OWN}})"
    CK=${{OWN}}
else
    # REFUSES AN AMBIGUOUS MATCH rather than taking the newest: every arm must
    # provably seed from the SAME file or it is not a comparison. phase1_exit
    # only -- never a live _running.pt, which array tasks would glob at
    # different steps.
    N=$(ls ${{CKPTS}}/*${{SRC}}_*_phase1_exit.pt 2>/dev/null | wc -l)
    if [ "${{N}}" -eq 0 ]; then
        echo "FATAL: no phase-1 exit matches *${{SRC}}_*_phase1_exit.pt in ${{CKPTS}}" >&2
        exit 1
    fi
    if [ "${{N}}" -gt 1 ]; then
        echo "FATAL: ${{N}} checkpoints match *${{SRC}}_*_phase1_exit.pt -- ambiguous, refusing:" >&2
        ls ${{CKPTS}}/*${{SRC}}_*_phase1_exit.pt >&2
        exit 1
    fi
    CK=$(ls ${{CKPTS}}/*${{SRC}}_*_phase1_exit.pt)
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  seed <- $(basename ${{CK}})"
fi
sed "s|{placeholder}|$(basename ${{CK}})|" ${{CONFIG}} > ${{RESOLVED}}

{{ nvidia-smi -L
  nvidia-smi --query-gpu=mig.mode.current,uuid,name,memory.total,driver_version --format=csv
  scontrol show job ${{SLURM_JOB_ID}}
  echo "nodelist: ${{SLURM_NODELIST}}  host: $(hostname)"
}} > ${{J}}.info 2>&1

# THE OCCUPANCY RECORD THAT MATTERS. The in-process sensor reads 3-6 points LOW
# on this route (calibrated at batch 8000: ours 91.0 vs external 93.9-97.1), and
# the offset is unmeasured at 1500-2560. This is the out-of-process number the
# scheduler cancels on -- check it once these are up:
#   tail -n 360 <arm>_smi.csv | awk -F, '$3+0==$3 {{s+=$3;n++}} END {{print s/n}}'
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \\
    --format=csv,nounits -l 10 > ${{J}}_smi.csv &
SMI_PID=$!

epilogue() {{
    kill ${{SMI_PID}} 2>/dev/null
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
        python -u train.py --config ${{RESOLVED}}
    "
"""


def main():
    arms = build()
    (HERE / 'joblogs').mkdir(exist_ok=True)
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\tbatch\tscale\n')
        for (nm, batch, scale) in ARMS:
            f.write(f'acrb_{nm}\t{WARM_SRC}\t{batch}\t{scale}\n')

    with (HERE / 'submit_acr_b_aug31.sbatch').open(
            'w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(first=0, last=len(ARMS) - 1,
                              ckpts=CLUSTER_CKPTS.rstrip('/'),
                              placeholder=PLACEHOLDER))

    print(f'{len(arms)} arms written to {HERE}\n')
    print(f"  {'arm':<16}{'batch':>7}{'scale':>7}")
    for (nm, batch, scale) in ARMS:
        print(f'  acrb_{nm:<11}{batch:>7}{scale:>7}')
    print(f'\n  all seed <- {WARM_SRC}_*_phase1_exit.pt  (step {EXIT_STEP})')
    print(f'  epochs {EXIT_STEP + PHASE2_STEPS}, array 0-{len(ARMS) - 1}')


if __name__ == '__main__':
    main()
