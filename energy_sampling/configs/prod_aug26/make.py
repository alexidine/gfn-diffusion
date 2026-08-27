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

WAVE 2 (not generated here): the UMA fans (mipcas + nehzor), blocked on (a)
the ELJ winners -- owner 2.4: MLIP optima are the same or slightly below
ELJ's, so their grids seed from these results -- and (b) the UPDATED UMA prior
filenames (the old UMA priors carried energy errors; every config on record
still references the old names). Acridine MACE rides WAVE 1 (see FAMILIES).

ELJ scales [0.2, 0.4, 0.8, 1.2, 1.6], acridine [0.1, 0.2, 0.4, 0.8], all on
seed 1.25e-4. 14 arms, one A100 each, 8 h walls (warm-started phase 1; the
battery reached 30k steps in <12 h cold).
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
    # acridine rides WAVE 1 after all (owner 2026-08-26: goal #2 is ASAP, and
    # the seed-from-ELJ-winners leverage cannot apply -- there is no acridine
    # ELJ task to seed from). MACE phase 1 is still MLE (the energy reaches
    # buffers and eval, not the loss), so the fan mechanics are identical; the
    # grid shifts one rung colder per the owner's 2.4 (MLIP optima at or
    # slightly below ELJ's). Identity fields verbatim from
    # prod0810_acridine_sg14_zp1_mace (bwd/mle -19.1 @ 5,280).
    'acr': {
        'prior_path': f'{CLUSTER_DATA}/acridine_sg14_zp1_mace_prior_dataset.pt',
        'space_groups': [14],
        'warm_src': 'prod0810_acridine_sg14_zp1_mace',
        'energy_function': 'mace',
        'mlip_path': '/scratch/mk8347/data/acr_112025_mh1_stagetwo.model',
        'scales': [0.1, 0.2, 0.4, 0.8],
        'max_batch_size': 50000,
    },
    # UMA on the UPDATED priors (owner uploaded 2026-08-26: *_uma_f047_*, the
    # energy-corrected files). COLD STARTS, necessarily: a new prior filename
    # is a new problem identity, so no historical checkpoint can pass the
    # weights-only guard -- and every old UMA checkpoint is bound to the
    # erroneous-energy prior anyway, the exact lineage the update severs.
    # Grid per owner: physical crystals of this type share roughly similar
    # optima ("direct neighbors on a fat enough grid") -- 3 neighbors around
    # the expected center, MLIP at-or-below ELJ (owner 2.4). Cold UMA phase 1
    # is the slow case: these rows ride the 12 h sbatch.
    'mipu': {
        'prior_path': f'{CLUSTER_DATA}/mipcas_sg2_zp1_uma_f047_prior_dataset.pt',
        'space_groups': [2],
        'warm_src': None,
        'energy_function': 'uma',
        'mlip_path': '/scratch/mk8347/models/uma/esen_s.pt',
        'scales': [0.2, 0.4, 0.8],
        'max_batch_size': 50000,
    },
    'nehu': {
        'prior_path': f'{CLUSTER_DATA}/nehzor_sg14_zp1_uma_f047_prior_dataset.pt',
        'space_groups': [14],
        'warm_src': None,
        'energy_function': 'uma',
        'mlip_path': '/scratch/mk8347/models/uma/esen_s.pt',
        'scales': [0.2, 0.4, 0.8],
        'max_batch_size': 50000,
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
        for s in spec.get('scales', SCALES):
            cfg = base()
            name = f'prod26_{fam}_lr{scale_tag(s)}'

            # -- identity + location ------------------------------------------
            cfg['run_name'] = name
            cfg['tag'] = 'prod26'
            cfg['checkpoints_dir'] = CLUSTER_CKPTS
            cfg['prior_path'] = spec['prior_path']
            # one molecule, unconditional: the condition set IS the prior file.
            # mk_dev's value is the LOCAL dev path -- left alone it ships a
            # D:\ path that exists on the dev box (so local preflight passes)
            # and kills every arm on-cluster at init_mol_dataset. 2026-08-27.
            cfg['molecules_path'] = spec['prior_path']
            cfg['test_molecules_path'] = None
            cfg['space_groups'] = spec['space_groups']
            if 'energy_function' in spec:
                cfg['energy_function'] = spec['energy_function']
            if 'mlip_path' in spec:
                cfg['mlip_path'] = spec['mlip_path']
            if 'max_batch_size' in spec:
                cfg['max_batch_size'] = spec['max_batch_size']
            # mk_dev pins a local dev warm start; the fan resolves its own.
            # A family with no warm source (the updated-prior UMA lineage
            # break) cold-starts: no placeholder, nothing to resolve.
            if spec.get('warm_src'):
                cfg['checkpoint_name'] = PLACEHOLDER
                cfg['load_weights_only'] = True
            else:
                cfg['checkpoint_name'] = None
                cfg['load_weights_only'] = False
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
    if FAMILIES[fam].get('warm_src'):
        assert cfg['load_weights_only'] is True
        assert cfg['checkpoint_name'] == PLACEHOLDER
    else:
        assert cfg['checkpoint_name'] is None and cfg['load_weights_only'] is False
    assert cfg['continue_from_checkpoint'] is False
    assert cfg['grow_batch_size'] is True, 'auto batch sizer is a battery property'
    assert cfg['grad_clip_guard']['enabled'] is True
    assert cfg['prior_path'] == FAMILIES[fam]['prior_path']
    assert cfg['molecules_path'] == FAMILIES[fam]['prior_path']
    assert cfg['test_molecules_path'] is None
    assert cfg['energy_function'] == FAMILIES[fam].get('energy_function', 'elj')
    if 'mlip_path' in FAMILIES[fam]:
        assert cfg['mlip_path'] == FAMILIES[fam]['mlip_path']
    assert_no_local_paths(cfg)


def assert_no_local_paths(node, trail='cfg'):
    """No Windows drive path may survive into a cluster arm. mk_dev is a LOCAL
    dev config: any path key the generator forgets to override ships a value
    that exists on the dev box -- so local preflight passes -- and 404s on the
    cluster. That exact shape killed all 20 prod_aug26 arms in ~10 s on
    2026-08-27 (molecules_path)."""
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
#SBATCH --time={time}
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
J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}
RESOLVED=${{J}}.yaml
if [ "${{SRC}}" = "-" ]; then
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  (cold start)"
    cp ${{CONFIG}} ${{RESOLVED}}
else
    CK=$(ls -t ${{CKPTS}}/*${{SRC}}*_running.pt 2>/dev/null | head -1)
    if [ -z "${{CK}}" ]; then
        echo "FATAL: no warm checkpoint matches *${{SRC}}*_running.pt in ${{CKPTS}}" >&2
        exit 1
    fi
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  warm <- $(basename ${{CK}})"
    sed "s|{placeholder}|$(basename ${{CK}})|" ${{CONFIG}} > ${{RESOLVED}}
fi

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
            f.write(f"{name}\t{FAMILIES[fam].get('warm_src') or '-'}\n")
    names = list(arms)
    warm_rows = [i for i, n in enumerate(names)
                 if FAMILIES[n.split('_')[1]].get('warm_src')]
    cold_rows = [i for i, n in enumerate(names)
                 if not FAMILIES[n.split('_')[1]].get('warm_src')]
    assert warm_rows == list(range(len(warm_rows))), 'warm rows must be contiguous first'
    assert cold_rows == list(range(len(warm_rows), len(names)))
    with (HERE / 'submit_prod_aug26.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(last=len(warm_rows) - 1, time='08:00:00',
                              ckpts=CLUSTER_CKPTS.rstrip('/'),
                              placeholder=PLACEHOLDER))
    # the cold UMA rows: same INDEX, their own array range and a 12 h wall
    with (HERE / 'submit_prod_aug26_uma.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        body = SBATCH.format(last=len(names) - 1, time='12:00:00',
                             ckpts=CLUSTER_CKPTS.rstrip('/'),
                             placeholder=PLACEHOLDER)
        body = body.replace('#SBATCH --array=0-' + str(len(names) - 1),
                            '#SBATCH --array=' + str(len(warm_rows)) + '-' + str(len(names) - 1))
        f.write(body)
    print(f'{len(arms)} arms written to {HERE}')
    for name in arms:
        print(' ', name)


if __name__ == '__main__':
    main()
