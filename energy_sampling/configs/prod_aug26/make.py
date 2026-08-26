"""prod_aug26: phase-1 (MLE-only) fixed-LR probe fans for the production tasks.

    python configs/prod_aug26/make.py     # configs + INDEX + sbatch

WHAT THIS IS (owner design, confirmed 2026-08-26). Goal #1/#2 phase 1: for each
task, blast warm-started MLE-only runs at a grid of FIXED learning rates, let
the LEVEL gate end each arm the moment its w1r bars are met (protocol 'stop' --
the GPU is yielded immediately), and hand-pick the best phase-1 exit checkpoint
for the phase-2 fan. This replaces in-run rate search for batteries: the fan IS
the search, and the per-arm machinery is only ASSERT + GUARD:

  * fixed LR         lr_control mode 'fixed' (burn-in hold 500 @ 0.05, then the
                     arm's scale; Adam's bias correction on the fresh optimizers
                     is the only 'ramp', by doctrine)
  * hot sensor       bwd/mle absolute, bar 5.0, action: FIRE -- the reviewed
                     actuation: a fire rewinds to the freshest same-stage 'best'
                     (phase-dependent selector: bwd/mle here) and cuts, sharing
                     the bar-fire cooldown. False positive ~ 50 steps.
  * derived bars     fixed mode derives hard-failure bars from burn-in; the
                     unified fire response (rewind + cut) guards the cruise.
  * level gate       w1r/median < 5 AND w1r/worst < 10 on the trailing median
                     ("decent" tier). CONVERGED -> snapshot phase1_exit +
                     snapshot_prior -> STOP.
  * auto batch sizer grow_batch_size on, canonical bounds.
  * auto grad clip   grad_clip_guard canonical.
  * reward clip      inert in phase 1 (the MLE loss never reads the reward);
                     left at canonical values.

WARM STARTS (owner: "for the sake of time"), weights-only so optimizers start
fresh at each arm's rate, and identity-guarded -- each family's identity fields
(prior_path, space_groups, energy_config) are written to MATCH its warm source
exactly, so assert_problem_match passes by construction:

  mipcas elj   <- base24_u2_fixed12      (bwd/mle -21.0 @30k, 2026-08-25 battery,
                                          exact current problem def)
  nehzor elj   <- prod0810_nehzor_elj    (bwd/mle -17.3 @30k; best matching-
                                          identity MLE on record, 2026-08-10)

The checkpoint FILENAME embeds a problem hash this generator does not
recompute; the sbatch resolves it on the cluster by run-name glob
(*<warm_src>*_running.pt, newest first) into a per-job COPY of the config, and
exits loudly if nothing matches. The repo configs carry the placeholder.

WAVE 2 (not generated here): UMA (mipcas + nehzor) and acridine MACE. Blocked
on (a) the ELJ winners -- owner 2.4: MLIP optima are the same or slightly
below ELJ's, so their grids seed from these results -- and (b) the UPDATED UMA
prior filenames (owner: the old UMA priors carried energy errors; every config
on record still references the old names).

Scales: [0.2, 0.4, 0.8, 1.2, 1.6] x seed 1.25e-4. 10 arms, one A100 each,
8 h walls (warm-started phase 1; the battery reached 30k steps in <12 h cold).
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIGS = HERE.parent
MK_DEV = CONFIGS / 'mk_dev.yaml'

CLUSTER_DATA = '/scratch/mk8347/data/crystal_datasets/conditional/priors'
CLUSTER_CKPTS = '/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints/'
PLACEHOLDER = 'WARM_CHECKPOINT_PLACEHOLDER'

SCALES = [0.2, 0.4, 0.8, 1.2, 1.6]

FAMILIES = {
    # identity fields copied VERBATIM from the warm source's logged config so
    # the weights-only load passes assert_problem_match by construction
    'mip': {
        'prior_path': f'{CLUSTER_DATA}/mipcas_sg2_zp1_elj_prior_dataset.pt',
        'space_groups': [2],
        'warm_src': 'base24_u2_fixed12',
    },
    'neh': {
        'prior_path': f'{CLUSTER_DATA}/nehzor_sg14_zp1_elj_prior_dataset.pt',
        'space_groups': [14],
        'warm_src': 'prod0810_nehzor_elj',
    },
}


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def scale_tag(s):
    return str(s).replace('.', 'p')


def build():
    arms = {}
    for fam, spec in FAMILIES.items():
        for s in SCALES:
            cfg = base()
            name = f'prod26_{fam}_lr{scale_tag(s)}'

            # -- identity + location ------------------------------------------
            cfg['run_name'] = name
            cfg['tag'] = 'prod26'
            cfg['checkpoints_dir'] = CLUSTER_CKPTS
            cfg['prior_path'] = spec['prior_path']
            cfg['space_groups'] = spec['space_groups']
            # mk_dev pins a local dev warm start; the fan resolves its own
            cfg['checkpoint_name'] = PLACEHOLDER
            cfg['load_weights_only'] = True
            cfg['continue_from_checkpoint'] = False
            cfg['prior_model_name'] = None

            # -- run shape ----------------------------------------------------
            cfg['epochs'] = 30000            # cap; the level gate + stop is the
            cfg['eval_period'] = 500         # designed ending (cluster cadence)
            cfg['figs_period'] = 1000

            # -- the fixed rate under test ------------------------------------
            lc = cfg['lr_control']
            lc['mode'] = 'fixed'
            lc['fixed_scale'] = float(s)
            lc['burn_in_steps'] = 500        # bars + clip guard calibrate here
            lc['burn_in_scale'] = 0.05
            lc['repeat_every'] = 0

            # -- MLE only: single stage, level-gated, STOPS -------------------
            # Built fresh rather than filtered from mk_dev's protocols so every
            # battery-relevant key is written EXPLICITLY (arms by omission are
            # duplicates -- baseline_aug24 lesson).
            cfg['protocol'] = 'prod_mle'
            cfg['protocols']['prod_mle'] = {'stages': [{
                'name': 'train_prior',
                'train_mode': 'bwd',
                'bwd_sampling_mode': 'dataset',
                'flags': {'update_log_z': True, 'scramble_conditions': True},
                # THE REVIEWED ACTUATION: fires rewind to the freshest
                # same-stage best (bwd/mle selector on this stage) and cut
                'hot_lr_sensor': {'channel': 'bwd/mle', 'form': 'absolute',
                                  'rows': 31, 'above': 5.0, 'action': 'fire'},
                'loss_coeffs': {'bwd': {'mle': 1.0, 'tbc': 0.0, 'repeats': 1.0,
                                        'tb_z_source': 'persistent'}},
                'exit': [{'metric': 'gates/progress_done', 'above': 0.5,
                          'patience': 1}],
                'on_exit': ['snapshot:phase1_exit', 'snapshot_prior', 'stop'],
            }]}

            check(cfg, fam, s)
            arms[name] = cfg
    return arms


def check(cfg, fam, s):
    """Re-assert every owner-confirmed battery property on the FINISHED dict --
    a generator that silently drops an override writes 10 wrong arms at once."""
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == float(s)
    assert lc['repeat_every'] == 0
    st = cfg['protocols']['prod_mle']['stages'][0]
    assert st['on_exit'][-1] == 'stop', 'arms must yield the GPU on exit'
    assert st['hot_lr_sensor']['action'] == 'fire'
    assert cfg['progress_gate']['mode'] == 'level', 'mk_dev must carry the level gate'
    bars = {m['key']: m['bar'] for m in cfg['progress_gate']['metrics']}
    assert bars == {'w1r/median': 5.0, 'w1r/worst': 10.0}
    assert cfg['load_weights_only'] is True
    assert cfg['checkpoint_name'] == PLACEHOLDER
    assert cfg['continue_from_checkpoint'] is False
    assert cfg['grow_batch_size'] is True, 'auto batch sizer is a battery property'
    assert cfg['grad_clip_guard']['enabled'] is True
    assert cfg['prior_path'] == FAMILIES[fam]['prior_path']


SBATCH = """#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-{last}
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=prod26
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prod_aug26/joblogs/%x_%A_%a.out

# prod_aug26 phase-1 fan. Arm = row of INDEX.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/prod_aug26
LOGS=${{ARMS}}/joblogs
CKPTS={ckpts}
mkdir -p ${{LOGS}}

ARM=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $1}}' ${{ARMS}}/INDEX.tsv)
SRC=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $2}}' ${{ARMS}}/INDEX.tsv)
if [ -z "${{ARM}}" ]; then echo "no arm at row ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; fi
CONFIG=${{ARMS}}/${{ARM}}.yaml
if [ ! -f "${{CONFIG}}" ]; then echo "missing config ${{CONFIG}}" >&2; exit 1; fi

# WARM-START RESOLUTION. The checkpoint filename embeds a problem hash the
# generator does not recompute; resolve by run-name glob, newest first, into a
# per-job COPY of the config (the repo config keeps its placeholder). LOUD on
# a miss: a fan arm silently cold-starting would corrupt the whole comparison.
CK=$(ls -t ${{CKPTS}}/*${{SRC}}*_running.pt 2>/dev/null | head -1)
if [ -z "${{CK}}" ]; then
    echo "FATAL: no warm checkpoint matches *${{SRC}}*_running.pt in ${{CKPTS}}" >&2
    exit 1
fi
echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  warm <- $(basename ${{CK}})"
J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}
RESOLVED=${{J}}.yaml
sed "s|{placeholder}|$(basename ${{CK}})|" ${{CONFIG}} > ${{RESOLVED}}

{{ nvidia-smi -L
  nvidia-smi --query-gpu=mig.mode.current,uuid,name,memory.total,driver_version --format=csv
  scontrol show job ${{SLURM_JOB_ID}}
  echo "nodelist: ${{SLURM_NODELIST}}  host: $(hostname)"
}} > ${{J}}.info 2>&1

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
    HERE.mkdir(exist_ok=True)
    (HERE / 'joblogs').mkdir(exist_ok=True)
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\n')
        for name, cfg in arms.items():
            fam = name.split('_')[1]
            f.write(f"{name}\t{FAMILIES[fam]['warm_src']}\n")
    with (HERE / 'submit_prod_aug26.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(last=len(arms) - 1, ckpts=CLUSTER_CKPTS.rstrip('/'),
                              placeholder=PLACEHOLDER))
    print(f'{len(arms)} arms written to {HERE}')
    for name in arms:
        print(' ', name)


if __name__ == '__main__':
    main()
