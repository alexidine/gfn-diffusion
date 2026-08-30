"""uma_stab_aug30: why does UMA phase 2 detonate, and what raises its stable rate?

    python configs/uma_stab_aug30/make.py

FIVE ARMS on nehzor UMA, seeded from the same phase-1 exit as the pt100p2
nehu2 fan (`pt100_nehu_lr4p0`, gated at 5010).

  arm            init scale   bounds.bwd[0]   fwd beta
  ctrl_lr0p5        0.5           0.02           10
  b_lr0p5           0.5           0.25           10
  h_lr0p5           0.5           0.02           80
  b_lr1p0           1.0           0.25           10
  h_lr1p0           1.0           0.02           80

THE READOUT IS THE TERMINAL RATE, NOT SURVIVAL. Recovery stays ON: a fire
rewinds AND halves the scale, so each arm performs its own descending LR
search and settles somewhere. Comparing where four configs settle answers
"does fixing the root cause raise the stable rate?" directly, with no LR grid.
The pt100p2 fan already ran this experiment by accident -- every arm that
fired descended, and the survivors converged on 1.56e-5 regardless of the
scale in their name.

Read `lr_ctrl/scale`, NOT the arm name, when ranking these.

WHY nehzor RATHER THAN mipcas. Both UMA families are unstable, but nehzor's
four pt100p2 arms all either died on the rewind budget or were visibly heading
into another excursion, whereas mipcas' lr0p25 was recovering. A threshold
assay wants the pathology to fire reliably, and nehzor's does. nehu2 had also
already found ~1597 as its own batch, so 1600 is not an imposition here.

------------------------------------------------------------------ the arms

Both interventions target the SAME observation from the pt100p2 fan: the
excursion is in the FORWARD policy (mass leaves the box) and it correlates
with bwd_frac collapsing.

`fwd` beta 10 -> 80. The loss is `beta * smooth_l1(x; beta)`, whose per-row
gradient is `x` below the knee and exactly `beta*sign(x)` above it -- so past
|x| ~ beta the restoring force is CONSTANT while the error grows unbounded.
That basin-escape mechanism was recorded on the conditional route (2026-08-19,
3yifcbrb vs 696ns1fz) with fwd beta as the PRIMARY predicted fix, direction
RAISE. This stage already raised bwd and replay to 80 and left fwd behind at
the mk_dev base, so 80 removes an asymmetry rather than inventing a value --
it is a rate already proven survivable on two branches of this same stage.
It is also why UMA and not ELJ: ELJ's analytic energy rarely produces
residuals past 10; UMA's out-of-distribution tail does so constantly.

`bounds.bwd[0]` 0.02 -> 0.25. NOT `min_fracs`. This stage runs
`balance.kind: ratio`, and `_ratio_tick` never calls `_nudge_mode_fracs` --
it takes its limits from `_share_interval`, which reads ONLY
`balance.bounds`. Setting `min_fracs.bwd` here would be a silent no-op and
the arm would duplicate the control.

`fwd` is PINNED at 0.05, so the ratio controller trades bwd against replay
alone: bwd collapsing to its floor means replay climbing to 0.93. Two reasons
to expect the floor to matter, and they are independent -- bwd draws from the
prior (real data) while replay draws from the model's own past samples, so the
collapse swaps a fixed anchor for a self-referential one; and it maximises the
weight on the one piece of state a rewind does not restore.

--------------------------------------------------------------- assay hygiene

`hot_lr_sensor.action` -> 'report' on equilibration. At 'fire' it routes into
`fire_loss_spike()` (train.py:6416), the same seat as a divergence bar, so
`fwd/scatter_err` drawdowns and real detonations would both drive the LR
descent and the terminal rate would be set by two mechanisms at once.

`max_reloads_per_1k_steps` stays 1.0. This is the mk_dev DEFAULT and the
pt100p2 arms already carry it -- an earlier reading of 0.2 came from
configs/a100_stab_aug16, a different battery, and does not apply here. The
budget is therefore max(3, step/1000) = 5-11 over this arm's span, not the
flat 3 that reading implied. It matters for interpretation: an arm that died
on this budget fired five or more times, so it was not killed one cut short of
recovery -- its descent was not sticking.

WHY A DESCENT MIGHT NOT STICK, stated as an open question rather than a
finding. `fire_loss_spike` calls `set_state_dict(checkpoint['modeller_state'])`
and `lr_ctrl` (which holds `scale`) is in MODELLER_STATE_DEFAULTS, so the
rewind restores the rate recorded in the TARGET checkpoint and `on_divergence`
halves from there. The descent can then only advance once a healthy checkpoint
is written at the lower rate, i.e. at eval cadence. The pt100p2 arms are
consistent with this being partial rather than total (lr0p5 reached 1.56e-5,
two halvings), so it is offered as a mechanism to WATCH in `lr_ctrl/scale`,
not as a diagnosis. `controller.fire_loss_spike`'s own docstring describes an
earlier version of this failure (djr13t0j) as fixed by holding the ceiling
where the rewind cannot reach it; whether v10 fixed mode still has that
protection is unverified.

--------------------------------------------------------------------- batch

PINNED at 1600, not servo'd. `batch_util_target` selects the smallest rung
CLEARING the target and, if none does, runs the walk to `max_batch_size` and
stops there -- so 0.95 was never aiming at 95%, it was an unreachable bar that
left `max_batch_size` doing all the work (the pt100p2 UMA arms at 4000). The
target is set to 0 here, which is the shipping default's "hold the base, no
probe" path, and the base and ceiling are both 1600.

1600 is a MEASURED operating point, not a guess: mipcas UMA sat there at
mid-to-high 60s external occupancy, well clear of the 38-49% cancellation
band, at roughly half the step time of the 4000-batch arms. Tuning the target
to reach it was rejected because the in-process sensor disagrees with the
out-of-process one by a batch-dependent, SIGN-FLIPPING error (-5..-8 points at
batch 1000, +38..+42 at 7410), so no threshold on it means the same thing at
two batches. At 1600 we are on the under-reporting side, which is the safe one.

CONSEQUENCE, stated because it bounds what these arms prove: batch and stable
rate move together, so the terminal rates measured here are the rates AT 1600
and do not transfer to 4000.
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
PHASE2_STEPS = 6000

PRIOR = f'{CLUSTER_DATA}/nehzor_sg14_zp1_uma_f047_prior_dataset.pt'
MLIP = '/scratch/mk8347/models/uma/esen_s.pt'

#: (name, fixed_scale, bwd floor, fwd huber beta)
ARMS = [
    ('ctrl_lr0p5', 0.5, 0.02, 10.0),
    ('b_lr0p5',    0.5, 0.25, 10.0),
    ('h_lr0p5',    0.5, 0.02, 80.0),
    ('b_lr1p0',    1.0, 0.25, 10.0),
    ('h_lr1p0',    1.0, 0.02, 80.0),
]
BWD_HI = 0.93


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build():
    arms = {}
    for name, scale, bwd_lo, fwd_beta in ARMS:
        cfg = base()
        run = f'us30_{name}'

        cfg['run_name'] = run
        cfg['tag'] = 'umastab'
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
                # REPORT, not fire: at 'fire' this drives the same rewind seat
                # as a divergence bar and would set the terminal rate jointly
                # with it. The descent must have one driver.
                'hot_lr_sensor': {'channel': 'fwd/scatter_err', 'rows': 11,
                                  'above': 2.0, 'action': 'report'},
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
    assert eq['hot_lr_sensor']['action'] == 'report', \
        "at 'fire' the drawdown sensor shares the rewind seat with the " \
        'divergence bars and the terminal rate stops being attributable'
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
#SBATCH --job-name=us30
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/uma_stab_aug30/joblogs/%x_%A_%a.out

# uma_stab_aug30. Arm = row of INDEX.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/uma_stab_aug30
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
            f.write(f'us30_{nm}\t{WARM_SRC}\t{scale}\t{bwd_lo}\t{fwd_beta}\n')

    with (HERE / 'submit_uma_stab_aug30.sbatch').open(
            'w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(first=0, last=len(names) - 1,
                              ckpts=CLUSTER_CKPTS.rstrip('/'),
                              placeholder=PLACEHOLDER))

    print(f'{len(arms)} arms written to {HERE}\n')
    print(f"  {'arm':<18} {'scale':>6} {'bwd_floor':>10} {'fwd_beta':>9}")
    for (nm, scale, bwd_lo, fwd_beta) in ARMS:
        print(f'  us30_{nm:<13} {scale:>6} {bwd_lo:>10} {fwd_beta:>9}')
    print(f'\n  all seed <- {WARM_SRC}_*_phase1_exit.pt  (step {EXIT_STEP})')
    print(f'  epochs {EXIT_STEP + PHASE2_STEPS}, batch pinned {BATCH}, '
          f'array 0-{len(names) - 1}')


if __name__ == '__main__':
    main()
