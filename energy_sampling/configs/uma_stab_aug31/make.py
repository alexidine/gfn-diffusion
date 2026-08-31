"""uma_stab_aug31: at a rate we KNOW is survivable, does either fix let UMA improve?

    python configs/uma_stab_aug31/make.py

SUPERSEDES uma_stab_aug31, WHICH WAS VOID. That battery put every arm at
6.25e-5 or 1.25e-4 against 1.56e-5 -- the only rate anything on this route has
been observed to hold -- so all five ran 2-8x too hot and none improved at all.
Measured at ~7000 steps: `Mean Sample Energy` 0.6 -> +24.8 to +57.4 on every
arm, i.e. progressively more unbound. The comparison was between degrees of
failure.

Two errors produced that, both worth keeping straight:

  1. The owner asked for a base of 3.125e-5 with "one higher and one lower
     rung". aug30 substituted a descent-instrument design starting at 6.25e-5
     -- every arm rewinds-and-halves its own way down to a terminal rate --
     and built the higher rung while DROPPING the lower one. Removing the low
     rung was a change to the design, not a detail, and went unflagged.
  2. THE DESCENT NEVER RAN. Four of five arms took exactly ONE fire and sat at
     their starting rate for the whole run. Partly self-inflicted: aug30 set
     `hot_lr_sensor.action: report` for assay cleanliness, which disables one
     of the two mechanisms driving the descent -- in a design whose readout
     depended on the descent happening.

So the question returns to the owner's original one. Start INSIDE the
survivable regime and ask whether the intervention lets the arm improve,
rather than starting outside it and asking where the arm lands.

  arm              LR         bounds.bwd[0]   fwd beta
  ctrl_1p56        1.56e-5        0.02           10
  bwd25_1p56       1.56e-5        0.25           10
  beta80_1p56      1.56e-5        0.02           80
  bwd25_3p12       3.125e-5       0.25           10
  beta80_3p12      3.125e-5       0.02           80

NAMES ENCODE THE INTERVENTION. aug30 used `b` and `h`, unreadable off a legend.

`hot_lr_sensor.action` is back to `fire`. There is no descent to keep clean now
-- the arms start where they are meant to stay -- so the sensor is a guard
again and worth having.

READOUT: `Mean Sample Energy` and `eval_fwd/*` over the run. The aug30 arms all
rose from 0.6 into the +20s..+50s; anything that HOLDS near its entry value is
already better, anything that FALLS is the result. Read `lr_ctrl/scale` too --
an arm that fires has moved off its label and stops being the rate it names.

CARRIED OVER FROM aug30, unchanged and still deliberate:
  * batch PINNED at 1600 (base and ceiling) with `batch_util_target: 0`. A
    nonzero target selects the smallest rung CLEARING it and otherwise walks to
    `max_batch_size`, so 0.95 was never aiming at 95% -- it was unreachable and
    the ceiling did all the work. 1600 is measured: mid-to-high 60s external
    occupancy, clear of the 38-49% cancellation band.
  * the bwd floor lands on `balance.bounds`, NOT `min_fracs`, which is inert
    under `kind: ratio` (`_ratio_tick` reads `_share_interval(bounds)` only).
    Asserted at generation time. It WORKED on aug30 -- those arms held bwd at
    exactly 0.250 while ctrl fell to the 0.020 floor.
  * fwd beta 10 -> 80 is the RAISE direction, corroborated twice: the recorded
    huber-basin prediction, and the jacob_july24 result that beta 10 -> 2
    degraded monotonically under "the tail is the restoring force -- cap its
    magnitude, never its relative weight". The magnitude cap that principle
    prescribes instead is already live at `gradient_norm_clip: auto` -> 643.9.
  * beta also feeds `winsorized_z_root`, and `fill_threshold` stays 20 (= 2 x
    the BASE beta), so the beta arms still test more than the TB knee. Flagged,
    not fixed.
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
BATCH = 1600
LEVEL_WINDOW = 5000

#: The seed, shared by every arm -- same file the pt100p2 nehu2 fan used.
WARM_SRC = 'pt100_nehu_lr4p0'
EXIT_STEP = 5010
#: `epochs` is ABSOLUTE (trange(init_step, epochs+1)), so it is derived from
#: the resume step, never written as a constant. 6000 steps at ~18.7 s/step is
#: ~31 h, i.e. one requeue; the sbatch resumes an arm's own _running.pt.
#: Sized for the descent (checkpoint-cadence-limited, ~1000 steps per
#: effective halving) plus a cruise long enough to show the settled rate holds.
PHASE2_STEPS = 10000

PRIOR = f'{CLUSTER_DATA}/nehzor_sg14_zp1_uma_f047_prior_dataset.pt'
MLIP = '/scratch/mk8347/models/uma/esen_s.pt'

#: (name, fixed_scale, bwd floor, fwd huber beta). seed_lr is 1.25e-4, so
#: scale 0.125 = 1.56e-5 (the rate the pt100p2 survivors converged on) and
#: scale 0.25 = 3.125e-5. NAMES CARRY THE INTERVENTION, not a letter code.
ARMS = [
    ('ctrl_1p56',   0.125, 0.02, 10.0),
    ('bwd25_1p56',  0.125, 0.25, 10.0),
    ('beta80_1p56', 0.125, 0.02, 80.0),
    ('bwd25_3p12',  0.25,  0.25, 10.0),
    ('beta80_3p12', 0.25,  0.02, 80.0),
]
BWD_HI = 0.93


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build():
    arms = {}
    for name, scale, bwd_lo, fwd_beta in ARMS:
        cfg = base()
        run = f'us31_{name}'

        cfg['run_name'] = run
        cfg['tag'] = 'umastab31'
        cfg['checkpoints_dir'] = CLUSTER_CKPTS
        cfg['prior_path'] = PRIOR
        # one molecule, unconditional: the condition set IS the prior file.
        # mk_dev ships a LOCAL D:\ path here, which passes local preflight and
        # 404s on the cluster (prod_aug26, 2026-08-27).
        cfg['molecules_path'] = PRIOR
        cfg['test_molecules_path'] = None
        cfg['space_groups'] = [14]
        cfg['energy_function'] = 'uma'
        cfg['mlip_path'] = MLIP

        cfg['checkpoint_name'] = PLACEHOLDER
        cfg['load_weights_only'] = False
        cfg['continue_from_checkpoint'] = False
        cfg['prior_model_name'] = None

        cfg['integrator']['T'] = SHIP_T
        cfg['eval_T'] = SHIP_T
        cfg['traj_checkpoint'] = True
        cfg['energy_config']['internal_oom_recovery'] = True

        cfg['epochs'] = EXIT_STEP + PHASE2_STEPS
        cfg['eval_period'] = 1000
        cfg['eval_num_samples'] = 2500
        cfg['figs_period'] = 1000
        cfg['progress_gate']['level_window'] = LEVEL_WINDOW

        # the batch is a DECISION here, not a servo output -- see the docstring.
        # growth stays enabled so select_batch_size still restores the base
        # after an OOM cut; target 0 is its "hold the base, no probe" path.
        cfg['batch_size'] = BATCH
        cfg['max_batch_size'] = BATCH
        cfg['grow_batch_size'] = True
        cfg['batch_util_target'] = 0

        # mk_dev's default, restated so the arm records the budget it ran under
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
            # A RE-ENTRY STUB, NOT A TRAINING STAGE -- identical to pt100p2's.
            # It must exist and must be named train_prior: the checkpoint
            # carries its stage BY NAME and StageProtocol raises otherwise, and
            # a single-stage protocol would never fire equilibration's on_enter
            # (begin() returns early on a resume).
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
                # phase-1 exit -- which left all four nehzor pt100p2 arms
                # grinding MLE inside the cancellation band (2026-08-29).
                'exit': [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}],
                # snapshot_prior is MANDATORY: prior_model is not in the
                # checkpoint and only this creates it. No 'stop'.
                'on_exit': ['snapshot_prior'],
            },
            {
                'name': 'equilibration',
                'train_mode': 'fused',
                'bwd_sampling_mode': 'prior',
                # FIRE. aug30 ran this at 'report' so its LR descent would
                # have a single driver; there is no descent here -- the arms
                # start at the rate they are meant to hold -- so the sensor
                # goes back to being a guard. It routes into fire_loss_spike,
                # the same rewind seat as a divergence bar.
                'hot_lr_sensor': {'channel': 'fwd/scatter_err', 'rows': 11,
                                  'above': 2.0, 'action': 'fire'},
                'flags': {'update_log_z': True, 'buffers_active': True,
                          'z_calibration': True},
                'on_enter': ['rebuild_prior_by_churn',
                             'bootstrap_z:train_conditioner'],
                'fracs': {'fwd': 0.05, 'bwd': BWD_HI, 'replay': 0.02},
                'min_fracs': {'fwd': 0.02, 'bwd': bwd_lo, 'replay': 0.02},
                'deactivate_threshold': 0.01,
                'loss_coeffs': {
                    'fwd': {'tb': 1.0, 'freeze_policy': 1.0, 'beta': fwd_beta},
                    'bwd': {'tb': 1.0, 'beta': 80},
                    'replay': {'tb': 1.0, 'beta': 80},
                },
                'balance': {
                    'kind': 'ratio', 'pinned': {'fwd': 0.05},
                    'metrics': {'replay': 'fwd/over_coverage',
                                'bwd': 'bwd/relative_under_wcen'},
                    'numerator': 'replay', 'setpoint': 5.0, 'gain': 0.05,
                    'max_step': 0.05,
                    # THE OPERATIVE FLOOR. `min_fracs` above is inert under
                    # kind: ratio and is set to match only so the two cannot
                    # be read as disagreeing.
                    'bounds': {'replay': [0.02, BWD_HI],
                               'bwd': [bwd_lo, BWD_HI]},
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

        check(cfg, name, scale, bwd_lo, fwd_beta)
        arms[run] = cfg
    return arms


def check(cfg, name, scale, bwd_lo, fwd_beta):
    assert cfg['integrator']['T'] == SHIP_T
    assert cfg['eval_T'] == cfg['integrator']['T'], \
        'utils hard-fails on eval_T != integrator.T at load'
    assert cfg['traj_checkpoint'] is True
    assert cfg['energy_config']['internal_oom_recovery'] is True, \
        'UMA sets its own ceiling through the chunk loop; the rollout sets the batch'

    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == float(scale)
    assert lc['hard_failure']['loss_excursion_k'] == EXCURSION_K

    stages = cfg['protocols']['prod_eq']['stages']
    assert len(stages) == 2
    st, eq = stages
    assert st['name'] == 'train_prior'
    assert 'stop' not in st['on_exit'] and st['on_exit'] == ['snapshot_prior']
    assert st['exit'] == [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}], \
        'the stub must exit on a TICK metric; a gate-keyed exit loses the race ' \
        'against the first post-resume progress_gate publish'
    assert eq['name'] == 'equilibration'
    assert eq.get('exit') is None, 'equilibration is terminal by design'
    assert eq['balance']['pinned']['fwd'] == eq['fracs']['fwd']

    # THE ASSAY'S OWN VARIABLES -- each asserted where it actually acts.
    assert eq['hot_lr_sensor']['action'] == 'fire', \
        'aug30 ran this at report to keep an LR descent attributable; these ' \
        'arms do not descend, so the guard goes back on'
    assert eq['loss_coeffs']['fwd']['beta'] == fwd_beta
    assert eq['loss_coeffs']['bwd']['beta'] == 80 and \
        eq['loss_coeffs']['replay']['beta'] == 80, \
        'bwd/replay are the unchanged reference; only fwd is an arm variable'
    assert eq['balance']['bounds']['bwd'][0] == bwd_lo, \
        'the bwd floor MUST land on balance.bounds -- kind: ratio reads ' \
        '_share_interval(bounds) and never touches min_fracs'
    assert eq['min_fracs']['bwd'] == bwd_lo, \
        'inert under kind: ratio, but it must not read as a different number'
    assert bwd_lo + eq['fracs']['fwd'] + eq['min_fracs']['replay'] < 1.0

    for br in ('fwd', 'bwd', 'replay'):
        terms = [k for k in eq['loss_coeffs'][br]
                 if k in ('tb', 'db', 'subtb', 'mle', 'tbc', 'vg_lb', 'vg_lme',
                          'emp_z', 'level_gap', 'z_level')
                 and float(eq['loss_coeffs'][br][k]) > 0]
        assert terms == ['tb'], f'{br} must be single-term tb, got {terms}'

    assert cfg['checkpoint_name'] == PLACEHOLDER
    assert cfg['load_weights_only'] is False
    assert cfg['continue_from_checkpoint'] is False
    assert cfg['prior_model_name'] is None

    assert cfg['epochs'] == EXIT_STEP + PHASE2_STEPS
    assert cfg['epochs'] > EXIT_STEP + 1000, \
        'epochs is ABSOLUTE and the arm resumes at EXIT_STEP; too low and it ' \
        'runs (almost) nothing and reports state=finished'

    # the batch is pinned, and the pin has to bind from BOTH sides: a base
    # under the ceiling with a live target would let the ladder walk up to it
    assert cfg['batch_size'] == BATCH and cfg['max_batch_size'] == BATCH
    assert cfg['batch_util_target'] == 0, \
        'a nonzero target here would select the smallest rung clearing it, or ' \
        'run the walk to max_batch_size -- neither is a decision we made'
    assert cfg['max_reloads_per_1k_steps'] == 1.0
    # THE aug30 FAILURE, MADE UNREPEATABLE. 1.56e-5 is the only rate anything
    # on this route has held; aug30 ran 2-8x above it and no arm improved.
    lr = float(lc['seed_lr']) * float(scale)
    assert lr <= 3.2e-5, (
        f'arm {name} runs at {lr:.3g}, above the 3.125e-5 ceiling this '
        f'battery is scoped to -- aug30 died of exactly this')
    assert cfg['grad_clip_guard']['enabled'] is True

    ep = cfg['eval_period']
    assert cfg['figs_period'] % ep == 0
    assert cfg['prior_path'] == PRIOR and cfg['molecules_path'] == PRIOR
    assert cfg['test_molecules_path'] is None
    assert cfg['energy_function'] == 'uma' and cfg['mlip_path'] == MLIP
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
#SBATCH --job-name=us31
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/uma_stab_aug31/joblogs/%x_%A_%a.out

# uma_stab_aug31. Arm = row of INDEX.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/uma_stab_aug31
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

# REQUEUE SAFETY. 6000 steps at ~18.7 s/step is ~31 h, so these arms are
# EXPECTED to hit the 24 h wall once. train.py's loader is
# `if checkpoint_name ... elif continue_from_checkpoint`, so checkpoint_name
# ALWAYS wins -- without this an extended arm would silently restart from the
# phase-1 exit and discard its own descent. Resubmit this same sbatch.
OWN=$(ls -t ${{CKPTS}}/*${{ARM}}_*_running.pt 2>/dev/null | head -1)
if [ -n "${{OWN}}" ]; then
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  REQUEUE: resuming own $(basename ${{OWN}})"
    CK=${{OWN}}
else
    # REFUSES AN AMBIGUOUS MATCH rather than taking the newest: every arm must
    # provably seed from the SAME file or it is not a comparison.
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

# the out-of-process occupancy record. The in-process sensor disagrees with it
# by a batch-dependent, sign-flipping error, and THIS is the number the
# scheduler cancels on -- at batch 1600 it is also the calibration point the
# recorded table (1000 / 4491 / 7410) does not cover.
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

    names = list(arms)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\tscale\tbwd_floor\tfwd_beta\n')
        for (nm, scale, bwd_lo, fwd_beta) in ARMS:
            f.write(f'us31_{nm}\t{WARM_SRC}\t{scale}\t{bwd_lo}\t{fwd_beta}\n')

    with (HERE / 'submit_uma_stab_aug31.sbatch').open(
            'w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(first=0, last=len(names) - 1,
                              ckpts=CLUSTER_CKPTS.rstrip('/'),
                              placeholder=PLACEHOLDER))

    print(f'{len(arms)} arms written to {HERE}\n')
    print(f"  {'arm':<18} {'scale':>6} {'bwd_floor':>10} {'fwd_beta':>9}")
    for (nm, scale, bwd_lo, fwd_beta) in ARMS:
        print(f'  us31_{nm:<15} {scale:>6} {bwd_lo:>10} {fwd_beta:>9}')
    print(f'\n  all seed <- {WARM_SRC}_*_phase1_exit.pt  (step {EXIT_STEP})')
    print(f'  epochs {EXIT_STEP + PHASE2_STEPS}, batch pinned {BATCH}, '
          f'array 0-{len(names) - 1}')


if __name__ == '__main__':
    main()
