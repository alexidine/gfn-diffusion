"""acr_c_aug31: acridine phase-2, rate re-centred on the ONE rate that worked, x bwd floor.

    python configs/acr_c_aug31/make.py

  arm               batch   scale     bwd floor
  b50_lr0p00625      1000   0.00625      0.50
  b50_lr0p0125       1000   0.0125       0.50
  b50_lr0p025        1000   0.025        0.50
  b50_lr0p05         1000   0.05         0.50
  b25_lr0p00625      1000   0.00625      0.25
  b25_lr0p0125       1000   0.0125       0.25
  b25_lr0p025        1000   0.025        0.25

Three rates shared across both floors, so the floor's effect is separable at a
fixed rate rather than confounded with it. All seed from the frozen
`pt100_acr_lr4p0` phase-1 exit (9010). Batch 1000, pinned -- see acr_c_aug31.

-------------------------------------------------- WHY THE GRID MOVED THIS FAR DOWN

acr_c_aug31 ran 0.125 / 0.25 / 0.5 / 1.0 and only ONE arm improved on anything:

  arm        fwd/tb_err     bwd/tb_err   rel_under      log Z   Mean Sample E   LIVE scale
  lr0p125   1315 -> 2032    9.6 -> 15.7  3.6 -> 5.2      2.09     261 -> 226        0.125
  lr0p25    1427 ->  654    9.6 ->  9.1  3.5 -> 4.1      3.95     261 ->  63.8     0.0125
  lr0p5     1338 -> 3175    9.5 -> 21.6  3.5 -> 5.5     -2.75     261 -> 618         0.5
  lr1p0     1818 -> 4966    9.6 -> 25.2  3.4 -> 5.9     -4.12     261 -> 227         0.5

READ THE LAST COLUMN. `lr0p25` was not running 0.25 -- it fired TWICE during
burn-in (0.05 -> 0.025 -> 0.0125) and never promoted to its nominal rate. The
only arm that improved was ~20x colder than its label and ~10x below the coldest
rung anyone intended. The fan was bracketed from above only; 0.0125 is the
empirical anchor and this grid straddles it.

BURN-IN SCALE DROPPED 0.05 -> 0.003, AND THAT IS THE FIX FOR THE ABOVE. At 0.05
the burn-in ran 1-8x HOTTER than every target rate in this grid, so the warm-up
was the most dangerous part of the run -- which is exactly how lr0p25 fired
during it. A burn-in below the coldest rung lets every arm promote cleanly and
actually run the rate it names. Without this the fan measures accidents.

------------------------------------------------- THE BWD FLOOR IS THE ONLY ACTUATOR

`balance.kind: ratio` keys replay on `fwd/over_coverage` and bwd on
`bwd/relative_under_wcen`, setpoint 5.0. MEASURED at the operating point:

  route          over_cov   rel_under    ratio   err vs setpoint    bwd   replay
  ELJ mipcas         8.97        2.10     4.27        -0.16        0.93     0.02
  ELJ nehzor        10.57        2.18     4.84        -0.03        0.92     0.03
  MACE acr best    373.8         4.08    91.65        +2.91        0.25     0.70
  MACE acr worst  3813           5.87   649.3         +4.87        0.25     0.70

The ELJ arms sit ON the setpoint -- which is why their Bwd Frac never leaves
0.93; the loop is satisfied and does nothing. On MACE the ratio is 92-649x the
setpoint, so the integrator is RAILED: it pins replay at its ceiling and bwd at
its floor and contributes no regulation at all. The floor is therefore not a
guard rail here, it IS the setting, which is why this battery varies it.

WHY THE TWO METRICS DIVERGE BY ROUTE, since it bears on any future fix.
`over_coverage` is UNIFORM over positive residuals -- utils.py:2011, "replay must
see over-weighted junk". `relative_under` is reward-ramp weighted and
self-normalised, explicitly so a heavy low-reward tail cannot inflate it. So the
ratio divides a junk-SENSITIVE statistic by a junk-INSENSITIVE one. On ELJ there
is little junk and they are commensurable; on MACE the forward energy tail drives
over_coverage 40x while rel_under moves 2x. Switching the bwd side to Z-anchored
`under_coverage` does NOT fix this -- that is ramp-gated too, so the asymmetry is
in the WEIGHTING, not the centring. Setpoint left at 5.0 here deliberately:
changing it would alter what the floor means and confound the floor comparison.

WHAT A HIGHER FLOOR CAN AND CANNOT DO. bwd owns SPREAD, not level, and
[[bwd_stationarity_low_T]] bounds it further: E_buffer[log P_F] has a ceiling of
-H(buffer) that phase-1 MLE already attains, so a post-MLE init -- which every
arm here is -- starts AT that ceiling with ~zero budget to translate the cloud.
Multiplying bwd's weight multiplies a batch-mean gradient near zero. That is why
the 0.25 floor stopped acr_b's collapse to 0.02 and still left three of four arms
degrading. The budget regenerates through the virtuous cycle, slowly, and is paid
for by short-term forward-calibration degradation. So the expectation here is
that the floor is necessary and not sufficient, and the rate does the rest.

READ `lr_ctrl/scale`, NOT THE ARM NAME.
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
#: Floor on the BACKWARD share. Must land on balance.bounds --
#: min_fracs is INERT under kind: ratio. See the docstring.
BWD_FLOOR = 0.25

#: (name, batch, fixed_scale, bwd_floor). Three rates are shared across both
#: floors so the floor's effect is separable at fixed rate. 0.0125 is the
#: empirical anchor -- the live rate of acr_b's only improving arm.
ARMS = [
    ('b50_lr0p00625', 1000, 0.00625, 0.50),
    ('b50_lr0p0125',  1000, 0.0125,  0.50),
    ('b50_lr0p025',   1000, 0.025,   0.50),
    ('b50_lr0p05',    1000, 0.05,    0.50),
    ('b25_lr0p00625', 1000, 0.00625, 0.25),
    ('b25_lr0p0125',  1000, 0.0125,  0.25),
    ('b25_lr0p025',   1000, 0.025,   0.25),
]
#: BELOW the coldest rung. At acr_b's 0.05 the warm-up ran 1-8x hotter than the
#: targets and fired arms before they could promote -- see the docstring.
BURN_IN_SCALE = 0.003


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def committed_energy_config():
    """`energy_config` as COMMITTED, or None if git cannot answer.

    A generator that snapshots the WORKING-TREE mk_dev bakes in whatever keys
    are mid-development there. Most are harmless -- they land on a Namespace and
    nothing reads them. `energy_config` is not: train.py does
    `MolecularCrystal(**energy_config)`, so a key whose handler is uncommitted is
    a TypeError at startup on every arm. That is how acr_c_aug31's first wave
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
    for name, batch, scale, bwd_floor in ARMS:
        cfg = base()
        DROPPED = drop_uncommitted_energy_keys(cfg, ref)
        run = f'acrb3_{name}'

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
        lc['burn_in_scale'] = BURN_IN_SCALE
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
                # min_fracs stays at the default on EVERY arm: it is inert
                # under kind: ratio (which reads _share_interval(bounds)
                # only), and config_invariants caps it below 1/3 so it
                # cannot express a 0.5 floor even nominally. `bounds`
                # below is the operative floor.
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
                    # THE OPERATIVE FLOOR. `_ratio_tick` reads
                    # _share_interval(bounds) and never calls
                    # _nudge_mode_fracs, so min_fracs above is inert here and
                    # matches only so the two cannot read as disagreeing. fwd
                    # is pinned at 0.05, so this implies a replay ceiling of 0.70.
                    'bounds': {'replay': [0.02, 0.93],
                               'bwd': [bwd_floor, 0.93]},
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

        check(cfg, name, batch, scale, bwd_floor)
        arms[run] = cfg
    return arms


def check(cfg, name, batch, scale, bwd_floor):
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
    assert lc['burn_in_scale'] < min(a[2] for a in ARMS), (
        'burn-in must be COLDER than every target rate, or the warm-up is the '
        'most dangerous part of the run and arms fire before they promote')
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
    assert eq['balance']['bounds']['bwd'][0] == bwd_floor, (
        'the bwd floor MUST land on balance.bounds -- kind: ratio reads '
        '_share_interval(bounds) and never touches min_fracs')
    assert eq['min_fracs']['bwd'] < 1/3, (
        'config_invariants requires min_fracs.bwd in [0, 1/3); the operative '
        'floor is balance.bounds, which has no such cap')
    assert bwd_floor + eq['fracs']['fwd'] + eq['min_fracs']['replay'] < 1.0
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
#SBATCH --job-name=acrc31
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/acr_c_aug31/joblogs/%x_%A_%a.out

# acr_c_aug31. Arm = row of INDEX.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/acr_c_aug31
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
        for (nm, batch, scale, bf) in ARMS:
            f.write(f'acrb3_{nm}\t{WARM_SRC}\t{batch}\t{scale}\n')

    # THE INDEX IS WHAT THE SBATCH RESOLVES `${ARM}.yaml` FROM. A prefix that
    # drifts from the generated filenames does not fail here -- it fails on
    # the cluster as "missing config" for every arm (caught 2026-08-31).
    idx = [l.split('	')[0] for l in
           (HERE / 'INDEX.tsv').read_text(encoding='utf-8').splitlines()[1:] if l]
    missing = [a for a in idx if not (HERE / f'{a}.yaml').exists()]
    assert not missing, f'INDEX names with no matching yaml: {missing}'
    assert idx == list(arms), f'INDEX order != arm order: {idx} vs {list(arms)}'

    with (HERE / 'submit_acr_c_aug31.sbatch').open(
            'w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(first=0, last=len(ARMS) - 1,
                              ckpts=CLUSTER_CKPTS.rstrip('/'),
                              placeholder=PLACEHOLDER))

    print(f'{len(arms)} arms written to {HERE}\n')
    print(f"  {'arm':<20}{'batch':>7}{'scale':>9}{'bwd_floor':>11}")
    for (nm, batch, scale, bf) in ARMS:
        print(f'  acrb3_{nm:<15}{batch:>7}{scale:>9}{bf:>11}')
    print(f'\n  all seed <- {WARM_SRC}_*_phase1_exit.pt  (step {EXIT_STEP})')
    print(f'  epochs {EXIT_STEP + PHASE2_STEPS}, array 0-{len(ARMS) - 1}')


if __name__ == '__main__':
    main()
