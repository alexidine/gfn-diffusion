"""acr_b_aug31: acridine phase-2 LR scan at batch 1000, plus a 1500 control.

    python configs/acr_b_aug31/make.py

  arm         batch   scale   purpose
  lr0p125      1000   0.125   LR scan
  lr0p25       1000   0.25    LR scan
  lr0p5        1000   0.5     LR scan
  lr1p0        1000   1.0     LR scan
  b1500        1500   0.5     throughput / stability control

All five seed from the same frozen `pt100_acr_lr4p0` phase-1 exit (step 9010).

------------------------------------------------ WHY 1000: OCCUPANCY IS NOT BINDING

`internal_oom_recovery: True` (3cf6b95) was dispositive for acridine -- batch 240
-> 8000 -- because MACE had been evaluating the whole rollout in ONE call, so the
MLIP's per-call memory was setting the TRAINING batch (64 GiB allocations at
batch 243, occupancy 32-38%, every arm cancelled).

That same chunking is why occupancy then stops being a constraint at all.
MEASURED on the first wave of this battery, wandb's out-of-process sampler:

  batch    ours    EXTERNAL    step_s   steps/s
   1500    47.7      85-100      13.6     0.074
   2560    64-68     85-100      17.8     0.056

The chunk loop keeps a dense MACE kernel resident essentially all the time, and
`utilization.gpu` measures the FRACTION OF TIME A KERNEL IS RUNNING, not how much
of the card is used. Smaller batch means fewer loop iterations, not more idle. So
external occupancy saturates near the top at every size we can run, sits far above
the measured survival bracket (a100_stab_aug16: cancelled <=40%, survived >=49.4%),
and carries no information about batch.

THE INSTRUMENT LESSON, because it cost this battery a submit cycle. The in-process
sensor reads LOW here by +3-6 points at batch 8000, +35 at 2560, and +52 at 1500.
That is not a smooth offset to calibrate against -- the first version of this file
extrapolated the batch-8000 point downward and concluded external ~64-75 at 2560,
when it is ~100. Do not infer external occupancy from `gpu/util_recent` on a
chunked-MLIP route; read the smi sidecar or wandb system metrics.

With the occupancy constraint unbinding, mk_dev's decided objective is the whole
answer: "occupancy constraint first, then optimizer-step throughput -- whose
answer is the CONSTANT batch_size = fused_grad_accum_min_samples", which is 1000.
Below it `fused_grad_accum_min_samples` accumulates micro-steps to 1000 samples
anyway, so a smaller batch buys no extra updates and pays more overhead per
update. The measured ladder agrees it is near the practical floor: step time at
batch 240 and 385 is IDENTICAL (7.4 s), i.e. a ~7 s fixed per-step cost that
dominates below ~1000.

  batch   step_s   samples/s   steps/s
   1000     10.7          93     0.093     <- 1.66x the updates of 2560
   1600     13.7         117     0.073
   2560     18.9         135     0.053
   8000     54.6         146     0.018

Sample throughput saturates by ~4560 and never improves; optimizer updates fall
monotonically. `b1500` is retained as the control: the one other size with live
external data, at a scale the fan also runs, so it confirms the 1.66x in situ and
is the fallback if 1000 destabilises.

--------------------------------------------------- PINNED, AND THE GRID SHIFTED

The batch is a DECISION: base == ceiling, `batch_util_target: 0` (the shipping
"hold the base, no probe" path). `grow_batch_size` stays true only so
select_batch_size still RESTORES the base after an OOM cut. A live target cannot
pick between adjacent rungs anyway -- per-arm occupancy at 2560 measured 60.0 /
65.9 / 60.9 / 69.0, so a 0.65 target would hold 2560 on two arms and push two to
3560, and an LR fan split across two batch sizes is confounded.

THE GRID IS SHIFTED DOWN ONE RUNG from the 2560 version (0.25-2.0 -> 0.125-1.0).
Batch 2560 -> 1000 is 2.6x fewer samples, so gradient noise rises ~1.6x and the
stable rate falls with it. Carrying the old grid across would repeat
uma_stab_aug30, whose arms ran 2-8x too hot and improved on nothing.

RUN PREFIX IS `acrb1k_`, NOT `acrb_`. An arm reusing an old name would glob the
previous wave's `_running.pt` and resume it -- and in fixed mode a mid-cruise
resume keeps the CHECKPOINTED scale, silently ignoring this config's
fixed_scale, so the fan would not run the rates it names.

READ `lr_ctrl/scale`, NOT THE ARM NAME. `hot_lr_sensor.action` is 'fire' here,
matching prod_t100_p2 for comparability with mip2/neh2, and a fire both rewinds
AND permanently halves the rate (fixed mode never re-races).

NOT AN EDIT TO prod_t100_p2, deliberately: that make.py writes all 20 of its
YAMLs and its nehu2 arms have been hand-edited on the cluster, so regenerating
would silently overwrite them.
"""
import subprocess
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
DROPPED = []
MLIP = '/scratch/mk8347/data/acr_112025_mh1_stagetwo.model'
FAN_BATCH = 1000

#: (name, batch, fixed_scale). The grid is shifted DOWN one rung from the 2560
#: version: batch 2560 -> 1000 is 2.6x fewer samples, so gradient noise rises
#: ~1.6x and the stable rate falls with it. Carrying 0.25-2.0 across would be
#: the same mistake uma_stab_aug30 made (arms 2-8x too hot, nothing improved).
ARMS = [
    ('lr0p125', FAN_BATCH, 0.125),
    ('lr0p25',  FAN_BATCH, 0.25),
    ('lr0p5',   FAN_BATCH, 0.5),
    ('lr1p0',   FAN_BATCH, 1.0),
    # throughput/stability control: the one batch we have live external data
    # for, at a scale the fan also runs. Confirms the 1.66x in situ and is the
    # fallback if 1000 destabilises.
    ('b1500',   1500,      0.5),
]


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def committed_energy_config():
    """`energy_config` as COMMITTED, or None if git cannot answer.

    A generator that snapshots the WORKING-TREE mk_dev bakes in whatever keys
    are mid-development there. Most are harmless -- they land on a Namespace and
    nothing reads them. `energy_config` is not: train.py does
    `MolecularCrystal(**energy_config)`, so a key whose handler is uncommitted is
    a TypeError at startup on every arm. That is how acr_b_aug31's first wave
    died on `prior_flow_path` (2026-08-31), with `lambda_mix` queued behind it.
    """
    out = subprocess.run(
        ['git', 'show', 'HEAD:energy_sampling/configs/mk_dev.yaml'],
        capture_output=True, text=True, cwd=str(CONFIGS.parent))
    if out.returncode != 0:
        return None
    return (yaml.safe_load(out.stdout) or {}).get('energy_config')


def drop_uncommitted_energy_keys(cfg, ref):
    """Strip energy_config keys absent from the COMMITTED mk_dev, loudly.

    SELF-HEALING BY DESIGN: the moment the handler and the mk_dev key are both
    committed, `ref` contains the key and nothing is stripped. So this does not
    have to be revisited when prior_flow lands -- it just stops firing.
    """
    if ref is None:
        raise SystemExit('cannot read committed mk_dev; refusing to generate '
                         'cluster configs against an unverifiable base')
    dropped = [k for k in list(cfg['energy_config']) if k not in ref]
    for k in dropped:
        cfg['energy_config'].pop(k)
    return dropped


def build():
    arms = {}
    ref = committed_energy_config()
    global DROPPED
    for name, batch, scale in ARMS:
        cfg = base()
        DROPPED = drop_uncommitted_energy_keys(cfg, ref)
        run = f'acrb1k_{name}'

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
            f.write(f'acrb1k_{nm}\t{WARM_SRC}\t{batch}\t{scale}\n')

    # THE INDEX IS WHAT THE SBATCH RESOLVES `${ARM}.yaml` FROM. A prefix that
    # drifts from the generated filenames does not fail here -- it fails on
    # the cluster as "missing config" for every arm (caught 2026-08-31).
    idx = [l.split('	')[0] for l in
           (HERE / 'INDEX.tsv').read_text(encoding='utf-8').splitlines()[1:] if l]
    missing = [a for a in idx if not (HERE / f'{a}.yaml').exists()]
    assert not missing, f'INDEX names with no matching yaml: {missing}'
    assert idx == list(arms), f'INDEX order != arm order: {idx} vs {list(arms)}'

    with (HERE / 'submit_acr_b_aug31.sbatch').open(
            'w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(first=0, last=len(ARMS) - 1,
                              ckpts=CLUSTER_CKPTS.rstrip('/'),
                              placeholder=PLACEHOLDER))

    print(f'{len(arms)} arms written to {HERE}\n')
    print(f"  {'arm':<16}{'batch':>7}{'scale':>7}")
    for (nm, batch, scale) in ARMS:
        print(f'  acrb1k_{nm:<11}{batch:>7}{scale:>7}')
    print(f'\n  all seed <- {WARM_SRC}_*_phase1_exit.pt  (step {EXIT_STEP})')
    print(f'  epochs {EXIT_STEP + PHASE2_STEPS}, array 0-{len(ARMS) - 1}')


if __name__ == '__main__':
    main()
