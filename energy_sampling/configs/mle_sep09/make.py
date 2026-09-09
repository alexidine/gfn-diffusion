"""mle_sep09: keep training phase 1, on the rebuilt priors, at a batch that buys
optimizer updates instead of occupancy.

WHAT THIS BATTERY IS FOR. The prod_t100 phase-1 fan exited on
`gates/progress_done` at 5010-19010 steps with nonthermal fractions of 0.017
(mip) to 0.286 (nehu). Nothing about those exits said "converged" -- the gate was
loosened and it called the short families done. These arms carry the same five
problems further, from the same weights, with three things changed.

  1. THE BATCH STOPS CHASING 95% OCCUPANCY. prod_t100 set batch_util_target 0.95
     on acr/mipu/nehu and 0.6 on mip/neh -- one line per family, and the whole
     reason the MLIP arms ran at batch 6560-8560 while the ELJ arms sat at 1000.
     select_batch_size grows ONLY to buy occupancy (train.py:980), and it
     delivered: 65% -> 96% util. It also cost 2.2-2.5x the optimizer updates per
     second, measured from the arms' own growth ladders:

         batch    1500    2500    3500    6500      (mipu, lived util / updates-s)
         util      65%     70%     75%     96%
         upd/s   0.700   0.634   0.497   0.334

     65% already clears the ~54% cancellation line with margin, so the growth
     from 1500 to 6500 bought headroom nothing used. Target 0.65 here, and
     max_batch_size 4000 as a bound rather than a destination -- at 4000 the step
     time is already the problem, but a bounded slow arm beats a cancelled one.

  2. THE PRIORS ARE REBUILT. mipcas UMA held 43,735 noised rows and nehzor UMA
     53,896, against 158,998-206,998 for their ELJ siblings -- not the energy
     function, just `tot_noised_samples` left at its 50,000 default. mipu was
     drawing 15% of its entire dataset every step and had made 751 passes over
     it. Rebuilt to ~200k around the SAME anchors at the SAME calibrated
     log_noise_range; new rows match the originals to within 0.010 IQR on every
     latent and energy column. neh and acr were already above 200k and keep the
     files they had.

  3. NO EXIT GATE. The stage omits `exit` entirely, which _parse_exit maps to
     None = terminal. `exit: []` would NOT do this: _exit_satisfied does
     `all(...)` over the terms, which is True on an empty list, so an empty list
     fires the exit on the first eval.

WHAT NO EXIT COSTS, AND THE ONE THING THAT COVERS IT. `on_exit` is the only
writer of phase1_exit.pt and the prior snapshot (checkpointing.py:26), so a stage
that never exits produces `_best.pt` and `_running.pt` and nothing else -- and a
_best.pt without a phase1_exit is the known-fatal warm-start source. That is what
`archive_period` is for; its own docstring names this case ("the single-stage
naive protocol fires no on_exit snapshots"). Every 5000 steps an archive is
hardlinked off `running`, ~30 MB and no extra serialization. archive_buffers is
OFF: those are a real ~880 MB copy each, and a weights-only warm start never
reads them.

THE RATE. Held at 4.0 on every family (owner, 2026-09-09), which won in all five
prod_t100 fans. The caveat is real and deliberately taken: cutting batch ~3x
raises gradient noise ~1.7x, and the stable rate moves down with it, so 4.0 at
batch ~2500 is not the same bet 4.0 at batch 7560 was. The hot sensor is armed
(`action: fire`) and the prod_t100 failures were loud -- neh/acr at scale 8.0
died at step ~670 -- so a bad rate is expected to announce itself rather than
quietly underperform. acridine fans 1/2/4/8 across it and is the arm that
measures this for the MLIP route.

WARM START ACROSS A REBUILT PRIOR. prior_path is hashed into the problem
definition BY FILENAME (utils.py:897), so mip/mipu/nehu -- the three whose priors
were rebuilt -- do not match their own phase-1 exits any more, even though the
anchors, the noise calibration and the energy function are identical.
`warm_start_ignore_problem_keys: [prior_path]` waives that ONE key, and only on
the weights-only path, which restores no buffers, no optimizer and no step count.
neh and acr kept their prior files and need no waiver, so they do not get one.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIGS = HERE.parent
MK_DEV = CONFIGS / 'mk_dev.yaml'

CLUSTER_DATA = '/scratch/mk8347/data/crystal_datasets/conditional/priors'
CLUSTER_CKPTS = '/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints/'

SHIP_T = 100
EXCURSION_K = 40.0
WALL = '2-00:00:00'

#: EPOCHS IS AN ABSOLUTE CAP, not a budget -- a resume starting past it runs zero
#: iterations. Set far beyond anything 48 h can reach so the WALL is the only
#: thing that ends an arm.
EPOCHS = 1_000_000

#: Held at 4.0 everywhere; acridine alone fans around it.
SCALE = 4.0
ACR_SCALES = [1.0, 2.0, 4.0, 8.0]

#: sed targets. The sbatch resolves both on-cluster: which checkpoint to seed
#: from, and whether this is a first launch or a requeue.
CK_PLACEHOLDER = 'WARM_CHECKPOINT_PLACEHOLDER'
CONT_PLACEHOLDER = 'CONTINUE_PLACEHOLDER'

FAMILIES = {
    'mip': {
        # rebuilt: 158,998 -> 201,503 noised rows, 13,505 anchors unchanged
        'prior_path': f'{CLUSTER_DATA}/mipcas_sg2_zp1_elj_200k_prior_dataset.pt',
        'prior_rebuilt': True,
        'space_groups': [2],
        'energy_function': 'elj',
        'mlip_path': None,
        'warm_src': 'pt100_mip_lr4p0',
    },
    'neh': {
        # NOT rebuilt: already 206,998 rows, so the problem identity is unchanged
        # and this family needs no warm-start waiver.
        'prior_path': f'{CLUSTER_DATA}/nehzor_sg14_zp1_elj_prior_dataset.pt',
        'prior_rebuilt': False,
        'space_groups': [14],
        'energy_function': 'elj',
        'mlip_path': None,
        'warm_src': 'pt100_neh_lr4p0',
    },
    'mipu': {
        # rebuilt: 43,735 -> 203,953 noised rows, 9,173 anchors unchanged
        'prior_path': f'{CLUSTER_DATA}/mipcas_sg2_zp1_uma_f047_200k_prior_dataset.pt',
        'prior_rebuilt': True,
        'space_groups': [2],
        'energy_function': 'uma',
        'mlip_path': '/scratch/mk8347/models/uma/esen_s.pt',
        'warm_src': 'pt100_mipu_lr4p0',
    },
    'nehu': {
        # rebuilt: 53,896 -> 203,609 noised rows, 5,990 anchors unchanged
        'prior_path': f'{CLUSTER_DATA}/nehzor_sg14_zp1_uma_f047_200k_prior_dataset.pt',
        'prior_rebuilt': True,
        'space_groups': [14],
        'energy_function': 'uma',
        'mlip_path': '/scratch/mk8347/models/uma/esen_s.pt',
        'warm_src': 'pt100_nehu_lr4p0',
    },
    'acr': {
        # NOT rebuilt: already 205,359 rows.
        'prior_path': f'{CLUSTER_DATA}/acridine_sg14_zp1_mace_prior_dataset.pt',
        'prior_rebuilt': False,
        'space_groups': [14],
        'energy_function': 'mace',
        'mlip_path': '/scratch/mk8347/data/acr_112025_mh1_stagetwo.model',
        'warm_src': 'pt100_acr_lr4p0',
        'scales': ACR_SCALES,
    },
}


def base():
    """The COMMITTED mk_dev, never the working tree -- a generator that reads an
    edited tree ships arms nobody can reproduce, and they die at startup with no
    wandb run to show for it."""
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def scale_tag(s):
    return str(s).replace('.', 'p')


def build():
    arms = {}
    for fam, spec in FAMILIES.items():
        for s in spec.get('scales', [SCALE]):
            cfg = base()
            name = f'mle09_{fam}_lr{scale_tag(s)}'

            # -- identity + location ------------------------------------------
            cfg['run_name'] = name
            cfg['tag'] = 'mle09'
            cfg['checkpoints_dir'] = CLUSTER_CKPTS
            cfg['prior_path'] = spec['prior_path']
            # one molecule, unconditional: the condition set IS the prior file.
            # mk_dev ships a LOCAL D:\ path here, which passes every local
            # preflight and then kills the arm on the cluster at init_mol_dataset.
            cfg['molecules_path'] = spec['prior_path']
            cfg['test_molecules_path'] = None
            cfg['space_groups'] = spec['space_groups']
            cfg['energy_function'] = spec['energy_function']
            cfg['mlip_path'] = spec['mlip_path']

            # -- warm start ----------------------------------------------------
            # Weights only: optimizers, schedulers, buffers, metrics and the step
            # count all start fresh. The sbatch substitutes the actual filename.
            cfg['checkpoint_name'] = CK_PLACEHOLDER
            cfg['load_weights_only'] = True
            cfg['continue_from_checkpoint'] = CONT_PLACEHOLDER
            cfg['prior_model_name'] = None
            if spec['prior_rebuilt']:
                cfg['warm_start_ignore_problem_keys'] = ['prior_path']

            # -- the ship length ------------------------------------------------
            # both together: utils hard-fails on eval_T != integrator.T at load.
            cfg['integrator']['T'] = SHIP_T
            cfg['eval_T'] = SHIP_T
            # owner rule: grad checkpointing rides the MLIPs in phase > 1 only,
            # and phase-1 MLE makes no energy call at all.
            cfg['traj_checkpoint'] = False

            # -- run shape -------------------------------------------------------
            cfg['epochs'] = EPOCHS
            cfg['eval_period'] = 500
            cfg['figs_period'] = 1000
            # the ONLY durable artifact this battery produces -- see the module
            # docstring. Hardlinked off `running`, so ~30 MB and no extra I/O.
            cfg['archive_period'] = 5000
            cfg['archive_buffers'] = False

            # -- batch: occupancy target, not a throughput walk -------------------
            cfg['batch_size'] = 1000
            cfg['grow_batch_size'] = True
            cfg['max_batch_size'] = 4000
            cfg['batch_util_target'] = 0.65
            # The rung dwell is measured in wall clock against _UTIL_MIN_SPAN_S
            # (60 s), and occupancy is bimodal with an autocorrelation time of
            # ~24 s. At the shipped 60 s sample period a 50-step rung held 1-3
            # samples and the ladder was deciding on noise; at 2 s it holds ~35.
            cfg['gpu_util_sample_period_s'] = 2
            cfg['gpu_util_policy_window_s'] = 7200

            # -- the fixed rate ---------------------------------------------------
            lc = cfg['lr_control']
            lc['mode'] = 'fixed'
            lc['fixed_scale'] = float(s)
            lc['burn_in_steps'] = 500
            lc['burn_in_scale'] = 0.05
            lc['repeat_every'] = 0
            lc['hard_failure']['loss_excursion_k'] = float(EXCURSION_K)

            # -- MLE only: one stage, TERMINAL ------------------------------------
            # `exit` is OMITTED, which _parse_exit maps to None = terminal stage.
            # Do NOT write `exit: []`: _exit_satisfied is all() over the terms,
            # which is True on an empty list, so [] fires on the first eval.
            # on_exit is kept for the shape of the thing but can never run here;
            # archive_period is what actually produces a warm-startable artifact.
            cfg['protocol'] = 'mle09'
            cfg['protocols']['mle09'] = {'stages': [{
                'name': 'train_prior',
                'train_mode': 'bwd',
                'bwd_sampling_mode': 'dataset',
                'flags': {'update_log_z': True, 'scramble_conditions': True},
                'hot_lr_sensor': {'channel': 'bwd/mle', 'form': 'absolute',
                                  'rows': 31, 'above': 5.0, 'action': 'fire'},
                'loss_coeffs': {'bwd': {'mle': 1.0, 'tbc': 0.0, 'repeats': 1.0,
                                        'tb_z_source': 'persistent'}},
                'on_exit': ['snapshot:phase1_exit', 'snapshot_prior', 'stop'],
            }]}

            check(cfg, fam, s)
            arms[name] = cfg
    return arms


def check(cfg, fam, s):
    """Re-assert every battery property on the FINISHED dict. A property checked
    on the inputs is not checked: base() reads a file that changes."""
    spec = FAMILIES[fam]

    assert cfg['integrator']['T'] == SHIP_T == cfg['eval_T'], \
        'utils hard-fails on eval_T != integrator.T at load'
    assert cfg['traj_checkpoint'] is False

    # the stage must be TERMINAL, and terminal means the key is absent
    stages = cfg['protocols']['mle09']['stages']
    assert len(stages) == 1
    st = stages[0]
    assert st['name'] == 'train_prior'
    assert 'exit' not in st, \
        "an `exit` key at all defeats the point; [] is WORSE than a term -- " \
        "_exit_satisfied is all() over an empty list, i.e. True on eval 1"
    assert st['hot_lr_sensor']['action'] == 'fire'
    terms = [k for k, v in st['loss_coeffs']['bwd'].items()
             if k not in ('repeats', 'tb_z_source') and float(v) > 0]
    assert terms == ['mle'], f'phase 1 is MLE-only, got {terms}'

    # the durability substitute for the on_exit snapshot that can never fire
    assert cfg['archive_period'] == 5000 and cfg['archive_buffers'] is False, \
        'without archives this arm produces no warm-startable checkpoint at all'

    # batch
    assert cfg['batch_size'] == 1000 and cfg['max_batch_size'] == 4000
    assert cfg['grow_batch_size'] is True and cfg['batch_util_target'] == 0.65
    assert cfg['gpu_util_sample_period_s'] == 2, \
        'at 60 s a 50-step rung holds 1-3 occupancy samples and the ladder is noise'

    # warm start
    assert cfg['checkpoint_name'] == CK_PLACEHOLDER
    assert cfg['continue_from_checkpoint'] == CONT_PLACEHOLDER
    assert cfg['load_weights_only'] is True, \
        'fresh phase 1 from old WEIGHTS: buffers/optimizer/step must start clean'
    assert cfg['prior_model_name'] is None
    waiver = cfg.get('warm_start_ignore_problem_keys')
    if spec['prior_rebuilt']:
        assert waiver == ['prior_path'], \
            f'{fam} has a rebuilt prior, so its own phase-1 exit no longer matches'
    else:
        assert waiver is None, \
            f'{fam} kept its prior file -- a waiver it does not need is a waiver ' \
            f'nobody will re-examine when it DOES need one'

    assert cfg['epochs'] == EPOCHS
    assert float(cfg['lr_control']['fixed_scale']) == float(s)
    assert cfg['lr_control']['mode'] == 'fixed'
    assert cfg['lr_control']['hard_failure']['loss_abs'] >= 1e6
    assert cfg['lr_control']['hard_failure']['grad_abs'] >= 1e6

    assert_no_local_paths(cfg)


def assert_no_local_paths(node, trail='cfg'):
    """A D:\\ or C:\\ path passes every local preflight and kills the arm on the
    cluster, after the queue slot is spent."""
    if isinstance(node, dict):
        for k, v in node.items():
            assert_no_local_paths(v, f'{trail}.{k}')
    elif isinstance(node, list):
        for i, v in enumerate(node):
            assert_no_local_paths(v, f'{trail}[{i}]')
    elif isinstance(node, str):
        low = node.lower()
        assert not (low.startswith('d:') or low.startswith('c:') or '\\' in node), \
            f'{trail} carries a local path: {node!r}'


SBATCH = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array={first}-{last}
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=mle09
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/mle_sep09/joblogs/%x_%A_%a.out

# mle_sep09: phase 1 continued on the rebuilt priors. Arm = row of INDEX.tsv
# (line 1 is the header). DO NOT EDIT --array BY HAND: make.py rewrites it.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/mle_sep09
LOGS=${{ARMS}}/joblogs
CKPTS={ckpts}
mkdir -p ${{LOGS}}

ARM=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $1}}' ${{ARMS}}/INDEX.tsv)
SRC=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $4}}' ${{ARMS}}/INDEX.tsv)
if [ -z "${{ARM}}" ]; then echo "no arm at row ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; fi
CONFIG=${{ARMS}}/${{ARM}}.yaml
if [ ! -f "${{CONFIG}}" ]; then echo "missing config ${{CONFIG}}" >&2; exit 1; fi

J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}
RESOLVED=${{J}}.yaml

# REQUEUE SAFETY. These arms have no exit gate and run to the 48 h wall, so every
# one of them is EXPECTED to be resubmitted. train.py's loader is
# `if checkpoint_name ... elif continue_from_checkpoint`, so checkpoint_name
# ALWAYS wins -- left pointing at the seed, a requeued arm would silently restart
# from its ancestor's phase-1 exit and throw away everything this wave bought.
# On a requeue we therefore BLANK checkpoint_name and continue our own run:
# that path takes load_full, so optimizer state and the step count carry, and
# load_weights_only is read only on the checkpoint_name branch.
OWN=$(ls -t ${{CKPTS}}/*${{ARM}}_*_running.pt 2>/dev/null | head -1)
if [ -n "${{OWN}}" ]; then
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  REQUEUE: continuing own $(basename ${{OWN}})"
    sed -e "s|{ck_ph}|null|" -e "s|{cont_ph}|true|" ${{CONFIG}} > ${{RESOLVED}}
else
    # REFUSES AN AMBIGUOUS MATCH rather than taking the newest: an arm that seeds
    # from a file nobody named is not a result. phase1_exit ONLY -- never a live
    # _running.pt, which array tasks would glob at different steps.
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
    sed -e "s|{ck_ph}|$(basename ${{CK}})|" -e "s|{cont_ph}|false|" ${{CONFIG}} > ${{RESOLVED}}
fi

# A placeholder that survived substitution would reach train.py as a literal
# filename and fail deep in the loader; catch it here, where the message is about
# the substitution that did not happen.
if grep -q '{ck_ph}\\|{cont_ph}' ${{RESOLVED}}; then
    echo "FATAL: placeholder left in ${{RESOLVED}}" >&2
    grep -n '{ck_ph}\\|{cont_ph}' ${{RESOLVED}} >&2
    exit 1
fi

{{ nvidia-smi -L
  nvidia-smi --query-gpu=mig.mode.current,uuid,name,memory.total,driver_version --format=csv
  scontrol show job ${{SLURM_JOB_ID}}
  echo "nodelist: ${{SLURM_NODELIST}}  host: $(hostname)"
}} > ${{J}}.info 2>&1

# The out-of-process occupancy record -- this is the number the scheduler cancels
# on, and the in-process sensor reads several points low on the MLIP routes.
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \\
    --format=csv,nounits -l 10 > ${{J}}_smi.csv &
SMI=$!

singularity exec --nv \\
  --overlay ${{OVERLAY}}:ro \\
  ${{IMAGE}} \\
  /bin/bash -c "source /ext3/env.sh; cd ${{WORKDIR}}; python train.py --config ${{RESOLVED}}"
RC=$?
kill ${{SMI}} 2>/dev/null
exit ${{RC}}
"""


def main():
    arms = build()
    (HERE / 'joblogs').mkdir(exist_ok=True)
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    names = list(arms)
    fam_of = lambda n: n.split('_')[1]
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tscale\twarm_src\n')
        for name in names:
            fam = fam_of(name)
            f.write(f"{name}\t{fam}\t{arms[name]['lr_control']['fixed_scale']}"
                    f"\t{FAMILIES[fam]['warm_src']}\n")

    fname = 'submit_mle_sep09.sbatch'
    with (HERE / fname).open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(first=0, last=len(names) - 1, time=WALL,
                              ckpts=CLUSTER_CKPTS.rstrip('/'),
                              ck_ph=CK_PLACEHOLDER, cont_ph=CONT_PLACEHOLDER))

    print(f'{len(arms)} arms -> {HERE}')
    print(f"{'arm':<20}{'family':<7}{'scale':>6}  {'prior':<48}{'seed'}")
    for name in names:
        fam = fam_of(name)
        spec = FAMILIES[fam]
        tag = 'REBUILT ' if spec['prior_rebuilt'] else '        '
        print(f"{name:<20}{fam:<7}{arms[name]['lr_control']['fixed_scale']:>6}  "
              f"{tag}{spec['prior_path'].rsplit('/', 1)[1]:<40}{spec['warm_src']}")
    print(f'\n{fname}: array 0-{len(names) - 1}, wall {WALL}')


if __name__ == '__main__':
    main()
