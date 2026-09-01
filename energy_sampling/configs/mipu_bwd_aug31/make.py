"""mipu_bwd_aug31: does holding the backward fraction up let mipcas UMA improve?

    python configs/mipu_bwd_aug31/make.py

MIPCAS UMA, phase-2 entry, 8 arms: 4 backward floors x 2 learning rates, both
rates at or below the only rate this route has been observed to hold.

                      lr 1.56e-5   lr 7.81e-6
  bounds.bwd[0] 0.02      x            x        <- control, controller runs free
  bounds.bwd[0] 0.25      x            x
  bounds.bwd[0] 0.50      x            x
  bounds.bwd[0] 0.75      x            x

FORWARD HUBER BETA IS NOT AN ARM VARIABLE HERE (owner, 2026-08-31). The stage
sets no `beta` for fwd, so every arm inherits mk_dev's 10. check() asserts the
absence rather than the value, so a later edit cannot reintroduce it silently.

WHY MIPCAS, having run the two previous batteries on nehzor. The nehzor choice
was made when the readout was a threshold assay -- "which arm survives" -- and
nehzor's pathology fires more reliably. That readout is gone, and for "does the
intervention let it improve" nehzor is the wrong testbed on three counts:
  * its samples are UNBOUND. The nehu2 phase-2 arms ended at Mean Sample Energy
    +15.7 / +55.6 / +15.8 / +179.5 -- positive lattice energies, i.e. crystals
    less stable than the isolated molecules. mipu2 ended at -84.8 / -71.4 /
    -71.2. Asking whether a fix helps is better posed from a bound starting
    point.
  * mipcas has a convergence target on record (prod0810_mipcas_elj, 9inim617);
    nehzor UMA has none.
  * every diagnostic conversation about these runs has been about mipu2, and
    the batch size of 1600 was measured on mipu2_lr0p25 in the first place.

THE HYPOTHESIS, and it is a recorded one rather than a new guess. In the
prod_t100_p2 phase-2 fan the ratio controller drove Bwd Frac to its 0.02 floor
on EVERY UMA arm and left it at 0.93 on EVERY ELJ arm:

  mip2_lr0p5   (ELJ)   0.930      mipu2_lr2p0    0.065 (min 0.049)
  neh2_lr0p5   (ELJ)   0.918      mipu2_lr0p5    0.047 (min 0.020)
  ...                             mipu2_lr0p25   0.038 (min 0.020)

Both routes ENTER at bwd 0.93; `fwd` is pinned at 0.05, so the loop trades bwd
against replay alone, keyed on fwd/over_coverage against
bwd/relative_under_wcen. On UMA the forward policy spreads, over_coverage
rises, replay wins share, bwd starves -- and per stab_july21c BWD OWNS SPREAD,
so spread goes unowned and the policy spreads further. That is the buildout
death condition (bwd pinned at min_frac) reproduced dynamically by a
controller. stab_july21c's fix list heads with "raise bwd min_frac to 0.1-0.3",
and replay_july26's best arm in a 13-arm battery was bwd pinned at 0.3.

The floors span from "controller free" (0.02) to "near the entry value" (0.75,
against ELJ's observed 0.93), so the ladder can show whether the effect
saturates or keeps improving toward the ELJ regime.

THE FLOOR MUST LAND ON `balance.bounds`, NOT `min_fracs`. This stage runs
`kind: ratio`, and `_ratio_tick` never calls `_nudge_mode_fracs` -- it takes
its limits from `_share_interval`, which reads ONLY `balance.bounds`. Setting
`min_fracs` alone is a silent no-op and the arm would duplicate the control.
The actuator is known to work: the aug30 arms held bwd at exactly 0.250 while
their control fell to 0.020.

`min_fracs.bwd` is therefore left at the shipping 0.02 on EVERY arm rather than
mirrored to the floor -- protocol validates it into [0, 1/3), so 0.50 and 0.75
are not even expressible there (caught by config_invariants at generation
time, not by hand). The two keys therefore read differently on six of eight
arms, which is only safe while the kind stays `ratio`; check() pins it, because
under any other kind min_fracs goes LIVE and all eight arms would collapse onto
the same floor.

RATES. 1.56e-5 is where every surviving pt100p2 UMA arm converged after its
divergence cuts; 7.81e-6 is one rung below it. Both are at or under that rate
on purpose -- uma_stab_aug30 ran 2-8x above it and NO arm improved at all
(Mean Sample Energy 0.6 -> +24.8..+57.4 on all five). check() enforces a
1.6e-5 ceiling so that cannot recur.

24-HOUR WALL, per the owner. At ~18.7 s/step that is ~4600 steps, landing near
step 9600. `epochs` is set well past it so the WALL binds and a resubmit always
extends; equilibration is terminal by design, so epochs is only an ending.
For scale: the ratio controller took ~1000 steps to drive bwd below 0.25 on the
pt100p2 UMA arms (measured: 1030 / 1000 / 1140 / 990 / 990 from stage entry),
so the floor contrast becomes readable around step 6000 and energy needs
considerably longer.
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

#: MIPCAS UMA. Its phase-1 exit gated at 5010, same as nehzor's.
WARM_SRC = 'pt100_mipu_lr4p0'
EXIT_STEP = 5010
PRIOR = f'{CLUSTER_DATA}/mipcas_sg2_zp1_uma_f047_prior_dataset.pt'
MLIP = '/scratch/mk8347/models/uma/esen_s.pt'
SPACE_GROUPS = [2]

#: `epochs` is ABSOLUTE and these arms resume at EXIT_STEP, so it is derived,
#: never a constant. Far past what 24 h reaches, so the wall binds.
PHASE2_STEPS = 15000

#: seed_lr is 1.25e-4. scale 0.125 -> 1.56e-5, scale 0.0625 -> 7.81e-6.
RATES = [('lr1p56', 0.125), ('lr0p78', 0.0625)]
#: bounds.bwd[0]. 0.02 is the shipping floor = the controller runs free.
FLOORS = [('bwd02', 0.02), ('bwd25', 0.25), ('bwd50', 0.50), ('bwd75', 0.75)]
BWD_HI = 0.93
#: the base fwd huber the stage must INHERIT -- beta is not an arm variable
BASE_FWD_BETA = 10.0


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def committed_energy_kwargs():
    """Which `energy_config` keys `MolecularCrystal.__init__` accepts IN THE
    COMMITTED TREE -- i.e. in the code the cluster will actually pull.

    `mk_dev.yaml` is live and tracks whatever the owner is working on, so it
    carries keys whose consumer may still be an uncommitted edit. Generating
    from it therefore ships a config the cluster's code cannot construct:
    train.py does `MolecularCrystal(**energy_config)` and dies with
    `TypeError: unexpected keyword argument`, at startup, before wandb.

    Returns None (check skipped, loudly) if git is unavailable.
    """
    import ast
    import subprocess
    for path in ('energy_sampling/energies/molecular_crystal.py',
                 'energies/molecular_crystal.py'):
        try:
            src = subprocess.run(['git', 'show', f'HEAD:{path}'],
                                 capture_output=True, text=True, check=True,
                                 cwd=str(CONFIGS.parent)).stdout
        except Exception:
            continue
        if not src.strip():
            continue
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.ClassDef)
                    and node.name == 'MolecularCrystal'):
                for fn in node.body:
                    if (isinstance(fn, ast.FunctionDef)
                            and fn.name == '__init__'):
                        return {a.arg for a in fn.args.args} - {'self'}
    return None


def build():
    arms = {}
    accepted = committed_energy_kwargs()
    if accepted is None:
        print('WARNING: could not read the committed MolecularCrystal signature; '
              'energy_config keys are UNCHECKED and may not exist on the cluster')
    for ftag, bwd_lo in FLOORS:
        for rtag, scale in RATES:
            cfg = base()
            name = f'{ftag}_{rtag}'
            run = f'mb31_{name}'

            cfg['run_name'] = run
            cfg['tag'] = 'mipubwd'
            cfg['checkpoints_dir'] = CLUSTER_CKPTS
            cfg['prior_path'] = PRIOR
            # one molecule, unconditional: the condition set IS the prior file.
            # mk_dev ships a LOCAL D:\ path here, which passes local preflight
            # and 404s on the cluster (prod_aug26, 2026-08-27).
            cfg['molecules_path'] = PRIOR
            cfg['test_molecules_path'] = None
            cfg['space_groups'] = SPACE_GROUPS
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

            # the batch is a DECISION, not a servo output. A nonzero
            # batch_util_target selects the smallest rung CLEARING it and
            # otherwise walks to max_batch_size, so 0.95 never aimed at 95% --
            # it was unreachable and the ceiling did all the work. 1600 is
            # measured on mipu2: mid-to-high 60s external occupancy, clear of
            # the 38-49% cancellation band. Growth stays ON so select_batch_size
            # still restores the base after an OOM cut; target 0 is its
            # "hold the base, no probe" path.
            cfg['batch_size'] = BATCH
            cfg['max_batch_size'] = BATCH
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
                # A RE-ENTRY STUB, NOT A TRAINING STAGE. It must exist and must
                # be named train_prior: the checkpoint carries its stage BY NAME
                # and StageProtocol raises otherwise, and a single-stage
                # protocol would never fire equilibration's on_enter (begin()
                # returns early on a resume).
                {
                    'name': 'train_prior',
                    'train_mode': 'bwd',
                    'bwd_sampling_mode': 'dataset',
                    'flags': {'update_log_z': True, 'scramble_conditions': True},
                    'hot_lr_sensor': {'channel': 'bwd/mle', 'form': 'absolute',
                                      'rows': 31, 'above': 5.0, 'action': 'fire'},
                    'loss_coeffs': {'bwd': {'mle': 1.0, 'tbc': 0.0,
                                            'repeats': 1.0,
                                            'tb_z_source': 'persistent'}},
                    # KEYED ON A TICK METRIC, NOT THE GATE. `_progress_history`
                    # is not checkpointed, so a gate-keyed exit publishes 0 at
                    # the first post-resume eval and wipes the streak restored
                    # from the phase-1 exit -- which left all four nehzor
                    # pt100p2 arms grinding MLE inside the cancellation band.
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
                    'fracs': {'fwd': 0.05, 'bwd': BWD_HI, 'replay': 0.02},
                    # NOT mirrored to bwd_lo: protocol validates min_fracs.bwd
                    # into [0, 1/3), so floors of 0.50 and 0.75 are not even
                    # EXPRESSIBLE there. Held at the shipping value; it is inert
                    # under kind: ratio, and check() pins the kind so that
                    # stays true.
                    'min_fracs': {'fwd': 0.02, 'bwd': 0.02, 'replay': 0.02},
                    'deactivate_threshold': 0.01,
                    # NO fwd `beta` -- it inherits mk_dev's 10 and is NOT an arm
                    # variable in this battery (owner, 2026-08-31). check()
                    # asserts its ABSENCE so it cannot be reintroduced quietly.
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
                        # THE OPERATIVE FLOOR -- `min_fracs` above is inert
                        # under kind: ratio and matches only so the two cannot
                        # read as disagreeing.
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

            # DROP energy_config keys the COMMITTED code cannot construct.
            # mk_dev currently ships `prior_flow_path: None` and
            # `lambda_mix: 1.0` for the prior-flow work, whose consumer in
            # molecular_crystal.py is still an uncommitted edit -- so the
            # cluster raises TypeError at init. Both are inert here
            # (prior_flow_path None = the flow is off), and neither is in
            # problem_def: mb31 and prod_t100_p2/pt100mipu2 hash IDENTICALLY
            # to 587e6c with and without them, so dropping cannot orphan the
            # phase-1 exit this battery resumes from.
            if accepted is not None:
                dropped = sorted(set(cfg['energy_config']) - accepted)
                for k in dropped:
                    del cfg['energy_config'][k]
                if dropped and not arms:
                    print(f'energy_config: dropped {dropped} -- not accepted by '
                          f'the COMMITTED MolecularCrystal.__init__')

            check(cfg, name, scale, bwd_lo, accepted)
            arms[run] = cfg
    return arms


def check(cfg, name, scale, bwd_lo, accepted=None):
    assert cfg['integrator']['T'] == SHIP_T
    assert cfg['eval_T'] == cfg['integrator']['T'], \
        'utils hard-fails on eval_T != integrator.T at load'
    assert cfg['traj_checkpoint'] is True
    assert cfg['energy_config']['internal_oom_recovery'] is True, \
        'UMA sets its own ceiling through the chunk loop; the rollout sets the batch'
    # EVERY energy_config key must exist in the code the CLUSTER pulls. train.py
    # does MolecularCrystal(**energy_config), so one key that only the working
    # tree understands is a TypeError at startup, before wandb -- which is how
    # this battery died on its first submit (prior_flow_path, from mk_dev).
    if accepted is not None:
        unknown = sorted(set(cfg['energy_config']) - accepted)
        assert not unknown, (
            f'{name}: energy_config carries {unknown}, which the COMMITTED '
            f'MolecularCrystal.__init__ does not accept. Commit the code that '
            f'consumes them, or drop them from the arm.')
    assert cfg['energy_function'] == 'uma' and cfg['mlip_path'] == MLIP
    assert cfg['prior_path'] == PRIOR and cfg['molecules_path'] == PRIOR
    assert cfg['test_molecules_path'] is None
    assert cfg['space_groups'] == SPACE_GROUPS

    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == float(scale)
    assert lc['hard_failure']['loss_excursion_k'] == EXCURSION_K
    # THE aug30 FAILURE, MADE UNREPEATABLE. 1.56e-5 is the only rate anything on
    # this route has held; aug30 ran 2-8x above it and no arm improved at all.
    lr = float(lc['seed_lr']) * float(scale)
    assert lr <= 1.6e-5, (
        f'arm {name} runs at {lr:.3g}, above the 1.56e-5 ceiling this battery '
        f'is scoped to -- uma_stab_aug30 died of exactly this')

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
    assert eq['hot_lr_sensor']['action'] == 'fire'

    # THE ARM VARIABLE, asserted where it actually acts
    assert eq['balance']['bounds']['bwd'][0] == bwd_lo, \
        'the bwd floor MUST land on balance.bounds -- kind: ratio reads ' \
        '_share_interval(bounds) and never touches min_fracs'
    assert eq['balance']['kind'] == 'ratio', \
        'min_fracs.bwd is left at 0.02 BECAUSE kind: ratio ignores it; under ' \
        'any other kind that value goes live and all eight arms collapse onto ' \
        'the same floor'
    assert eq['min_fracs']['bwd'] == 0.02, \
        'min_fracs.bwd validates into [0, 1/3), so it cannot carry a 0.50 or ' \
        '0.75 floor -- balance.bounds is the only route'
    # fwd is pinned at 0.05, so bwd and replay share 0.95; the floor has to
    # leave replay its own floor or _share_interval collapses to a point
    assert bwd_lo + eq['fracs']['fwd'] + eq['min_fracs']['replay'] < 1.0
    assert bwd_lo <= BWD_HI - eq['min_fracs']['replay'], \
        f'bwd floor {bwd_lo} leaves replay under its own 0.02 floor'

    # BETA IS NOT AN ARM VARIABLE HERE -- assert its ABSENCE, not its value
    assert 'beta' not in eq['loss_coeffs']['fwd'], \
        'fwd huber beta is deliberately NOT an arm variable in this battery ' \
        '(owner 2026-08-31); the stage must inherit the base value'
    assert float(cfg['fwd_loss_coeffs']['beta']) == BASE_FWD_BETA, \
        f'the inherited fwd beta moved off {BASE_FWD_BETA}; every arm would ' \
        f'silently change together'
    assert eq['loss_coeffs']['bwd']['beta'] == 80 and \
        eq['loss_coeffs']['replay']['beta'] == 80

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

    assert cfg['batch_size'] == BATCH and cfg['max_batch_size'] == BATCH
    assert cfg['batch_util_target'] == 0, \
        'a nonzero target would select the smallest rung clearing it, or walk ' \
        'to max_batch_size -- neither is a decision we made'
    assert cfg['max_reloads_per_1k_steps'] == 1.0
    assert cfg['grad_clip_guard']['enabled'] is True
    assert cfg['figs_period'] % cfg['eval_period'] == 0
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
#SBATCH --job-name=mb31
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/mipu_bwd_aug31/joblogs/%x_%A_%a.out

# mipu_bwd_aug31 -- MIPCAS UMA, bwd floor x learning rate, 8 arms.
# Arm = row of INDEX.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/mipu_bwd_aug31
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

# REQUEUE SAFETY. 24 h reaches ~4600 steps at ~18.7 s/step against a 15000-step
# budget, so these arms are EXPECTED to hit the wall repeatedly. train.py's
# loader is `if checkpoint_name ... elif continue_from_checkpoint`, so
# checkpoint_name ALWAYS wins -- without this an extended arm would silently
# restart from the phase-1 exit and discard its own progress. Resubmit this
# same sbatch to extend.
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

# the out-of-process occupancy record -- the in-process sensor disagrees with it
# by a batch-dependent, sign-flipping error, and THIS is the number the
# scheduler cancels on.
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
    # joblogs must reach the CLUSTER, not just exist locally: SLURM cannot
    # create the --output directory, so a missing one kills the job at launch
    # -- before python, before wandb, a few seconds with no run to inspect.
    # git does not track empty directories, so a brand-new battery ships
    # without it unless something inside is committed.
    logs = HERE / 'joblogs'
    logs.mkdir(exist_ok=True)
    keep = logs / '.gitkeep'
    if not keep.exists():
        keep.write_text('ships this directory to the cluster; see make.py\n',
                        encoding='utf-8')
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    names = list(arms)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\tscale\tlr\tbwd_floor\n')
        for ftag, bwd_lo in FLOORS:
            for rtag, scale in RATES:
                f.write(f'mb31_{ftag}_{rtag}\t{WARM_SRC}\t{scale}'
                        f'\t{1.25e-4 * scale:.4g}\t{bwd_lo}\n')

    with (HERE / 'submit_mipu_bwd_aug31.sbatch').open(
            'w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(first=0, last=len(names) - 1,
                              ckpts=CLUSTER_CKPTS.rstrip('/'),
                              placeholder=PLACEHOLDER))

    print(f'{len(arms)} arms written to {HERE}   FAMILY: MIPCAS UMA\n')
    print(f"  {'#':<3}{'arm':<22}{'LR':>11}{'bwd floor':>11}")
    for i, (ftag, bwd_lo) in enumerate(FLOORS):
        for j, (rtag, scale) in enumerate(RATES):
            k = i * len(RATES) + j
            print(f'  {k:<3}mb31_{ftag}_{rtag:<15}{1.25e-4 * scale:>11.3g}'
                  f'{bwd_lo:>11}')
    print(f'\n  all seed <- {WARM_SRC}_*_phase1_exit.pt  (step {EXIT_STEP})')
    print(f'  epochs {EXIT_STEP + PHASE2_STEPS}, batch pinned {BATCH}, '
          f'24 h wall, array 0-{len(names) - 1}')
    print('  fwd huber beta: NOT an arm variable -- inherits the base 10')


if __name__ == '__main__':
    main()
