"""prod_t100_p2: phase-2 (equilibration) fans at the ship length, T=100.

    python configs/prod_t100_p2/make.py

SEEDS -- the best GATED phase-1 exit per family, from prod_t100:

  mipcas elj  <- pt100_mip_lr4p0   mle -26.87, w1r 3.99/10.60, gated @ 19010
  nehzor elj  <- pt100_neh_lr4p0   mle -15.81, w1r 4.18/12.36, gated @ 14010

REVISED 2026-08-28, before submission. The first version of this file seeded
from mip_lr2p0 (-23.19) and neh_lr1p0 (-10.02) because the scale-4 arms had the
best mle but had NOT yet gated, and a phase-2 stub seeded from a non-gated
_running.pt strands the arm in train_prior -- the restored exit streak is what
carries it through on the first post-resume eval, and without one the tell is
`phase` never leaving 1. Both scale-4 arms subsequently gated (mipcas at step
19010, nehzor at 14010), so that objection is gone and the seeds move to them:
3.7 and 5.8 nats better respectively, and for mipcas better on BOTH w1r stats
as well (3.99/10.60 against 5.58/10.88).

The lesson worth keeping: "best gated" is a MOVING TARGET while phase 1 is
still running. Re-read it immediately before submitting rather than trusting a
seed chosen hours earlier -- mip_lr4p0 needed 19010 steps to gate against
mip_lr0p5's 14510, so the best arm was also the slowest to qualify.

THE GRID, 0.25/0.5/1.0/2.0 at 2x spacing. Phase-2 rates are a separate
measurement from phase-1 rates -- the loss is different (fused TB reads the
energy; MLE does not) -- so nothing here is inherited from the phase-1 answer.
Two independent routes put the T=100 phase-2 optimum near 0.55-1.1:

  (a) the only phase-2 bracket ever run (elj_selD_eq, mipcas sg2, T=10) plus
      where two long healthy phase-2 runs settled -> ~0.28 at T=10; phase 1's
      optimum rose 2-4x from T=10 to T=100 (measured, this battery), so ~0.6-1.1
  (b) at T=10 phase 2 ran ~3.6x below phase 1 (1.0 -> 0.28); phase 1 at T=100
      measures 2-4, so phase 2 ~0.55-1.1

Both land in the same place, which is why this grid is 4 rungs rather than the
5 phase 1 needed with no prior at all. 2x spacing still, not sqrt(2): the T=100
phase-1 fan showed a sqrt(2) grid would have been too narrow to bracket.

WALL CLOCK IS THE OPEN RISK. Equilibration is TERMINAL by design (no exit), so
`epochs` is the only ending -- but fused at T=100 is unmeasured on this cluster.
Phase-1 bwd measures 1.5 s/step at T=100; fused does fwd+bwd+replay plus
z_calibration and the energy, so 4-6x that is a reasonable guess, i.e. 4000-9000
steps in 24 h against an epochs cap of 14500. The arms are therefore EXPECTED to
hit the wall before the cap, and the sbatch is requeue-safe: an arm that has
written its own _running.pt resumes from that rather than restarting from the
phase-1 exit. Resubmit the same sbatch to extend.

traj_checkpoint: TRUE here. The owner's rule is "grad checkpointing on with the
MLIPs, but only in phase > 1"; these are ELJ, so this EXTENDS it, and the
extension is flagged for review rather than assumed. The reason is that the
phase-1 -> 2 transition is where memory jumps (bwd-only stub -> fused, ~3x the
trajectory allocation) and a TRANSITION OOM IS FATAL, not recoverable. At T=100
activation memory is 10x its T=10 value, and the stub cannot pre-discover the
fused ceiling because it never allocates it. 33.6x less activation VRAM for 1.7x
step time is the right trade against a fatal failure, and it is also what lets
the batch grow at all -- which is the occupancy lever (phase 1 sat at batch 1000
and its MLIP family was cancelled for low utilisation).
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
SCALES = [0.25, 0.5, 1.0, 2.0]
#: widened from 2500 (owner, 2026-08-28). The gate's trailing median is what
#: absorbs w1r noise, and that noise is INDEPENDENT eval-to-eval (lag-1
#: autocorrelation -0.39..+0.14 across ten series, i.e. measurement not drift),
#: so more evals in the window is the only free way to reduce it: 10 evals
#: instead of 5 improves the median's standard error by sqrt(2) at zero compute.
#: Raising eval_num_samples and switching to a p90 bar were both ruled out.
LEVEL_WINDOW = 5000

#: MLIP families take the eval settings established on acridine in phase 1:
#: eval cost there is per-SAMPLE and molecule-dependent (nehzor UMA spent 50-57%
#: of wall clock outside the training step at T=10 against mipcas UMA's 16%, on
#: the same energy function), and it is what got the acridine phase-1 wave
#: cancelled for low occupancy. figs_period stays 1000 for every arm, so the
#: wandb storage bill is identical across the battery.
MLIP_EVAL = {'eval_num_samples': 2500, 'eval_period': 1000}

FAMILIES = {
    'mip2': {
        'prior_path': f'{CLUSTER_DATA}/mipcas_sg2_zp1_elj_prior_dataset.pt',
        'space_groups': [2],
        'warm_src': 'pt100_mip_lr4p0',
        'energy_function': 'elj',
        'mlip_path': None,
        'max_batch_size': 8000,
    },
    'neh2': {
        'prior_path': f'{CLUSTER_DATA}/nehzor_sg14_zp1_elj_prior_dataset.pt',
        'space_groups': [14],
        'warm_src': 'pt100_neh_lr4p0',
        'energy_function': 'elj',
        'mlip_path': None,
        'max_batch_size': 8000,
    },
    # ---- MLIP families, added 2026-08-28 once their phase 1 gated -----------
    # acridine's phase-1 exit is the CLEANEST in the whole battery: w1r
    # 1.94/4.68 against bars of 5.0/10.0, where the ELJ families exit at
    # 4.0-4.2 / 10.6-12.4. Its arms also gated fastest (9010 steps) once the
    # eval cost was cut.
    'acr2': {
        'prior_path': f'{CLUSTER_DATA}/acridine_sg14_zp1_mace_prior_dataset.pt',
        'space_groups': [14],
        'warm_src': 'pt100_acr_lr4p0',
        'energy_function': 'mace',
        'mlip_path': '/scratch/mk8347/data/acr_112025_mh1_stagetwo.model',
        'max_batch_size': 50000,
        **MLIP_EVAL,
    },
    'mipu2': {
        'prior_path': f'{CLUSTER_DATA}/mipcas_sg2_zp1_uma_f047_prior_dataset.pt',
        'space_groups': [2],
        'warm_src': 'pt100_mipu_lr4p0',
        'energy_function': 'uma',
        'mlip_path': '/scratch/mk8347/models/uma/esen_s.pt',
        'max_batch_size': 50000,
        **MLIP_EVAL,
    },
    # NEHZOR UMA IS THE ONE THAT MAY REFUSE AT SUBMIT TIME, deliberately.
    # nehu_lr4p0 was still RUNNING when this was written -- best of its batch on
    # mle (-16.34) with both w1r stats already inside the bars (3.24/8.61), so
    # it is the right seed, but it had not yet written a phase1_exit. Seeding a
    # fan from a LIVE _running.pt is the failure this battery's sbatch exists to
    # prevent: four array tasks start at different times and would glob the file
    # at different steps, so the fan stops being a comparison. Worse, a tagged
    # snapshot gets its own FROZEN buffer sidecar and the rolling _running one
    # does not, so phase 2 could come up without the buffers it needs.
    # The sbatch therefore refuses loudly ("no phase-1 exit matches ...") until
    # the arm gates. That is the intended behaviour, not a bug -- resubmit once
    # it does. It was ~2000 steps from the earliest gate step when written.
    'nehu2': {
        'prior_path': f'{CLUSTER_DATA}/nehzor_sg14_zp1_uma_f047_prior_dataset.pt',
        'space_groups': [14],
        'warm_src': 'pt100_nehu_lr4p0',
        'energy_function': 'uma',
        'mlip_path': '/scratch/mk8347/models/uma/esen_s.pt',
        'max_batch_size': 50000,
        **MLIP_EVAL,
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
            name = f'pt100{fam}_lr{scale_tag(s)}'

            cfg['run_name'] = name
            cfg['tag'] = 'pt100p2'
            cfg['checkpoints_dir'] = CLUSTER_CKPTS
            cfg['prior_path'] = spec['prior_path']
            # one molecule, unconditional: the condition set IS the prior file.
            # mk_dev ships a LOCAL D:\ path here, which passes local preflight
            # and 404s on the cluster (prod_aug26, 2026-08-27).
            cfg['molecules_path'] = spec['prior_path']
            cfg['test_molecules_path'] = None
            cfg['space_groups'] = spec['space_groups']
            cfg['energy_function'] = spec['energy_function']
            cfg['mlip_path'] = spec['mlip_path']

            # FULL resume of the phase-1 exit: phase 2 needs the stage_ctrl
            # (its restored exit streak carries it through the train_prior stub)
            # and the frozen buffer sidecar. weights-only would strand it.
            cfg['checkpoint_name'] = PLACEHOLDER
            cfg['load_weights_only'] = False
            cfg['continue_from_checkpoint'] = False
            cfg['prior_model_name'] = None

            cfg['integrator']['T'] = SHIP_T
            cfg['eval_T'] = SHIP_T
            # see the module docstring -- this EXTENDS the owner's MLIP rule to
            # ELJ phase 2, because the fatal case is a transition OOM
            cfg['traj_checkpoint'] = True

            cfg['epochs'] = 14500
            # ELJ eval is 6-10% of wall clock and stays at 500; the MLIP
            # families take the cut that rescued acridine's phase 1.
            cfg['eval_period'] = int(spec.get('eval_period', 500))
            if 'eval_num_samples' in spec:
                cfg['eval_num_samples'] = spec['eval_num_samples']
            cfg['figs_period'] = 1000
            cfg['batch_util_target'] = 0.95
            cfg['max_batch_size'] = spec['max_batch_size']
            cfg['progress_gate']['level_window'] = LEVEL_WINDOW

            lc = cfg['lr_control']
            lc['mode'] = 'fixed'
            lc['fixed_scale'] = float(s)
            lc['burn_in_steps'] = 500
            lc['burn_in_scale'] = 0.05
            lc['repeat_every'] = 0
            lc['hard_failure']['loss_excursion_k'] = EXCURSION_K

            cfg['protocol'] = 'prod_eq'
            cfg['protocols']['prod_eq'] = {'stages': [
                # STAGE 0 -- A RE-ENTRY STUB, NOT A TRAINING STAGE. It must
                # exist and must be named train_prior: the checkpoint carries
                # its stage BY NAME and StageProtocol raises if the config does
                # not define it. A single-stage equilibration protocol is doubly
                # broken -- begin() returns early on a resume, so as stage[0]
                # its on_enter would never fire, silently.
                {
                    'name': 'train_prior',
                    'train_mode': 'bwd',
                    'bwd_sampling_mode': 'dataset',
                    'flags': {'update_log_z': True, 'scramble_conditions': True},
                    'hot_lr_sensor': {'channel': 'bwd/mle', 'form': 'absolute',
                                      'rows': 31, 'above': 5.0, 'action': 'fire'},
                    'loss_coeffs': {'bwd': {'mle': 1.0, 'tbc': 0.0, 'repeats': 1.0,
                                            'tb_z_source': 'persistent'}},
                    # EXACTLY ONE TERM AT INDEX 0, patience 1, verbatim from the
                    # phase-1 arm. Exit streaks restore BY INDEX, and at the
                    # first post-resume eval maybe_advance only re-judges eval/*
                    # terms, so the restored streak is what carries the arm
                    # through. A second term would have no restored streak and
                    # would strand the arm in the stub.
                    'exit': [{'metric': 'gates/progress_done', 'above': 0.5,
                              'patience': 1}],
                    # NO 'stop' -- with it every arm ends ~10 steps after resume
                    # as a FINISHED run having done no phase-2 work. NO
                    # re-snapshot of phase1_exit -- all four arms share one and a
                    # duplicate would pollute the warm glob. snapshot_prior is
                    # MANDATORY: prior_model is not in the checkpoint and only
                    # this creates it; without it the churn rebuild backfills
                    # from anchors alone and bwd_sampling_mode 'prior' trains on
                    # the wrong buffer.
                    'on_exit': ['snapshot_prior'],
                },
                # STAGE 1 -- the production stage. TERMINAL by design (no exit).
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

            check(cfg, fam, s)
            arms[name] = cfg
    return arms


def check(cfg, fam, s):
    spec = FAMILIES[fam]
    assert cfg['integrator']['T'] == SHIP_T
    assert cfg['eval_T'] == cfg['integrator']['T'], \
        'utils hard-fails on eval_T != integrator.T at load'
    assert cfg['traj_checkpoint'] is True, \
        'phase 2 at T=100: a transition OOM is FATAL and the stub cannot ' \
        'pre-discover the fused ceiling'
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == float(s)
    assert lc['hard_failure']['loss_excursion_k'] == EXCURSION_K
    assert lc['hard_failure']['loss_abs'] >= 1e6 and lc['hard_failure']['grad_abs'] >= 1e6

    assert cfg['protocol'] == 'prod_eq'
    stages = cfg['protocols']['prod_eq']['stages']
    assert len(stages) == 2, 'phase 2 needs the train_prior stub + equilibration'
    st, eq = stages
    assert st['name'] == 'train_prior', \
        "the checkpoint carries its stage BY NAME; StageProtocol raises without it"
    assert 'stop' not in st['on_exit'], 'a phase-2 stub must NOT stop'
    assert st['on_exit'] == ['snapshot_prior'], \
        'snapshot_prior is mandatory (prior_model is not in the checkpoint)'
    assert len(st['exit']) == 1 and st['exit'][0] == {
        'metric': 'gates/progress_done', 'above': 0.5, 'patience': 1}, \
        'exit streaks restore BY INDEX -- one term at index 0, verbatim'
    assert eq['name'] == 'equilibration'
    assert eq.get('exit') is None, 'equilibration is terminal by design'
    assert eq['on_enter'] == ['rebuild_prior_by_churn', 'bootstrap_z:train_conditioner']
    assert eq['flags']['buffers_active'] and eq['flags']['z_calibration']
    assert eq['hot_lr_sensor']['channel'] == 'fwd/scatter_err', \
        "sensor bars are PER-STAGE; phase 1's bwd/mle bar watches a channel " \
        'this stage does not live on'
    assert eq['balance']['pinned']['fwd'] == eq['fracs']['fwd']
    for br in ('fwd', 'bwd', 'replay'):
        terms = [k for k in eq['loss_coeffs'][br]
                 if k in ('tb', 'db', 'subtb', 'mle', 'tbc', 'vg_lb', 'vg_lme',
                          'emp_z', 'level_gap', 'z_level')
                 and float(eq['loss_coeffs'][br][k]) > 0]
        assert terms == ['tb'], f'{br} must be single-term tb, got {terms}'

    # FULL resume -- weights-only would drop the stage_ctrl the stub needs
    assert cfg['checkpoint_name'] == PLACEHOLDER
    assert cfg['load_weights_only'] is False, \
        'phase 2 must resume FULL state: the restored exit streak is what ' \
        'carries the arm through the train_prior stub'
    assert cfg['continue_from_checkpoint'] is False
    assert cfg['prior_model_name'] is None

    assert cfg['progress_gate']['mode'] == 'level'
    assert cfg['progress_gate']['level_window'] == LEVEL_WINDOW
    bars = {m['key']: m['bar'] for m in cfg['progress_gate']['metrics']}
    assert bars == {'w1r/median': 5.0, 'w1r/worst': 10.0}
    ep = cfg['eval_period']
    assert cfg['figs_period'] % ep == 0
    assert cfg['progress_gate']['level_window'] // ep + 1 >= 3, \
        'too few evals in the level window; the gate would never fire'

    assert cfg['grow_batch_size'] is True
    assert cfg['grad_clip_guard']['enabled'] is True
    assert cfg['prior_path'] == spec['prior_path']
    assert cfg['molecules_path'] == spec['prior_path']
    assert cfg['test_molecules_path'] is None
    assert cfg['energy_function'] == spec['energy_function']
    assert cfg['mlip_path'] == spec['mlip_path']
    assert cfg['eval_period'] == int(spec.get('eval_period', 500))
    if 'eval_num_samples' in spec:
        assert cfg['eval_num_samples'] == spec['eval_num_samples']
    assert cfg['max_batch_size'] == spec['max_batch_size']
    # traj_checkpoint is ON for every family here. For the MLIPs that IS the
    # owner's rule (phase > 1); for ELJ it extends it, because the fatal case
    # is a transition OOM the bwd-only stub cannot pre-discover.
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
#SBATCH --job-name=pt100p2{wave}
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prod_t100_p2/joblogs/%x_%A_%a.out

# prod_t100_p2 phase-2 fan. Arm = row of INDEX.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/prod_t100_p2
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

# REQUEUE SAFETY, and it is load-bearing here. Equilibration is terminal and
# fused at T=100 is unmeasured, so these arms are EXPECTED to hit the 24 h wall
# before the epochs cap. train.py's loader is `if checkpoint_name ... elif
# continue_from_checkpoint`, so checkpoint_name ALWAYS wins -- without this an
# extended arm would silently restart from the phase-1 exit and discard its own
# progress. Resubmit this same sbatch to extend a fan.
OWN=$(ls -t ${{CKPTS}}/*${{ARM}}_*_running.pt 2>/dev/null | head -1)
if [ -n "${{OWN}}" ]; then
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  REQUEUE: resuming own $(basename ${{OWN}})"
    CK=${{OWN}}
else
    # REFUSES AN AMBIGUOUS MATCH rather than taking the newest: every arm in a
    # fan must provably seed from the SAME file or it is not a comparison.
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
    fam_of = lambda n: n.split('_')[0].replace('pt100', '')
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\tscale\n')
        for name in names:
            spec = FAMILIES[fam_of(name)]
            f.write(f"{name}\t{spec['warm_src']}"
                    f"\t{arms[name]['lr_control']['fixed_scale']}\n")

    for fam in FAMILIES:
        rows = [i for i, n in enumerate(names) if fam_of(n) == fam]
        assert rows == list(range(rows[0], rows[-1] + 1)), \
            f'family {fam} rows are not contiguous: {rows}'
        fname = f'submit_prod_t100_p2_{fam}.sbatch'
        with (HERE / fname).open('w', encoding='utf-8', newline='\n') as f:
            f.write(SBATCH.format(first=rows[0], last=rows[-1], wave=fam,
                                  ckpts=CLUSTER_CKPTS.rstrip('/'),
                                  placeholder=PLACEHOLDER))
        print(f'{fname}: rows {rows[0]}-{rows[-1]}')

    print(f'\n{len(arms)} arms written to {HERE}')
    for name in names:
        print(f"  {name:<20} scale {arms[name]['lr_control']['fixed_scale']:<6} "
              f"seed <- {FAMILIES[fam_of(name)]['warm_src']}")


if __name__ == '__main__':
    main()
