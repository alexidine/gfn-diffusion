"""prod_sep12 -- phase 2 on the four mle09 families, seeded from each arm's
best-MLE checkpoint.

WHAT THIS IS. mle_sep09 carried phase 1 (bwd MLE on the rebuilt priors) far past
the prod_t100 exits -- mip/neh to the 48 h wall at ~112k steps, mipu flat from
~60k, nehu still descending at 24k. These four arms run the prod_sep02 phase-2
shape (`prod_eq`: re-entry stub, then fused equilibration at pinned fractions
0.05/0.475/0.475, frozen anchors, fixed rate) from those weights instead of the
prod_t100 phase-1 exits. ONE arm per family, at the rate prod_sep02 gave its
7-day paper slot: mip 1.0, neh 0.5, mipu 0.0625, nehu 0.125.

THE SEED IS `_best.pt`, NOT A phase1_exit. mle09's stage is terminal by design
(no `exit`, so on_exit never runs and no phase1_exit.pt exists). For a
train_mode `bwd` stage `_best` is the bwd/mle record (train.py
_best_metric_channels), hardlinked off `running` at the 50-step save cadence --
i.e. the best-MLE sample of the run, which is exactly what is wanted. Its buffer
sidecar resolves to the run's ROLLING sidecar (sidecar_candidates strips the
`_best` tag), so anchors and prior buffer come across; equilibration's on_enter
`rebuild_prior_by_churn` replaces the prior buffer anyway.

FULL RESUME, as in prod_sep02. load_full restores the step (~110k on the ELJ
arms) and the stage NAME `train_prior`, which this config's prod_eq protocol
also defines -- as the stub that exits at the first eval (bwd/mle > -1e9,
patience 1), fires snapshot_prior, and advances to equilibration. A `_best.pt`
carries no request_eval stamp, so the stub can run up to eval_period-1 MLE
steps before that eval; harmless on a converged prior. `epochs` is an ABSOLUTE
cap (trange(init_step, epochs+1)) and is set unreachable; the wall ends the leg.

IDENTITY IS INHERITED, NOT REBUILT. Each arm is derived from the mle09 YAML of
its family, so energy_config/prior_path/space_groups -- the checkpoint's stored
problem_def -- match by construction, and `check` asserts it through
get_problem_definition. load_full honours no `warm_start_ignore_problem_keys`,
so this is the only way a rebuilt-prior family can seed at all. The prod_eq
protocol block is copied verbatim from the prod_sep02 arm of the same family,
which ran it for 7 days.

WHAT DIFFERS PER ROUTE, and only this:
  ELJ (mip, neh)   traj_checkpoint False -- measured 1.7-2.2x per-step waste at
                   4-5 GB of an 80 GB card; batch 1000 -> 4000 at util 0.65,
                   the mle09 occupancy target, inherited.
  UMA (mipu, nehu) traj_checkpoint True; batch PINNED at 1600 (prod_sep02's
                   measured operating point), eval_period 1000 with 2500 eval
                   samples, internal_oom_recovery on. All prod_sep02 values.

REQUEUE-SAFE. 2-day wall so the arms backfill sooner than a 7-day request; the
sbatch resumes an arm's own `_running.pt` (and its own `*_prior.pt`) when one
exists, seeds from `*<src>_*_best.pt` otherwise, and honours the `.dead`
sentinel so an UNRECOVERABLE arm does not eat a resubmission. Resubmit the same
sbatch to continue.
"""
import copy
import pathlib
import sys
from argparse import Namespace

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PLACEHOLDER = 'WARM_CHECKPOINT_PLACEHOLDER'
PRIOR_PLACEHOLDER = 'PRIOR_MODEL_PLACEHOLDER'
TAG = 'p12'
WALL = '2-00:00:00'
#: ABSOLUTE step cap, not a length -- the ELJ seeds sit at ~112k already.
EPOCHS = 1_000_000
FRACS = {'fwd': 0.05, 'bwd': 0.475, 'replay': 0.475}
ANCHOR = {'frozen': True, 'online_loss_flow': False,
          'thin_every_n_evals': 0, 'refresh_every_n_evals': 0, 'replay_beta': 1.0}
UMA_BATCH = 1600

FAM = {
    'mip': dict(seed='mle_sep09/mle09_mip_lr4p0.yaml', p2='prod_sep02/p02_mip_lr1.yaml',
                src='mle09_mip_lr4p0', scale=1.0, mlip=False),
    'neh': dict(seed='mle_sep09/mle09_neh_lr4p0.yaml', p2='prod_sep02/p02_neh_lr0p5.yaml',
                src='mle09_neh_lr4p0', scale=0.5, mlip=False),
    'mipu': dict(seed='mle_sep09/mle09_mipu_lr4p0.yaml', p2='prod_sep02/p02_mipu_lr0p0625.yaml',
                 src='mle09_mipu_lr4p0', scale=0.0625, mlip=True),
    'nehu': dict(seed='mle_sep09/mle09_nehu_lr4p0.yaml', p2='prod_sep02/p02_nehu_lr0p125.yaml',
                 src='mle09_nehu_lr4p0', scale=0.125, mlip=True),
}


def tag_for(scale):
    return 'lr' + ('%g' % scale).replace('.', 'p')


def load(rel):
    return yaml.safe_load((ROOT / rel).read_text(encoding='utf-8'))


def _ns(obj):
    if isinstance(obj, dict):
        return Namespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def problem_def(cfg):
    """The identity load_full will compare against the checkpoint's stored one."""
    sys.path.insert(0, str(ROOT.parent.parent))
    from energy_sampling.utils import get_problem_definition, normalize_problem_def
    return normalize_problem_def(get_problem_definition(_ns(cfg)))


def deltas(cfg, fam, name):
    spec = FAM[fam]
    p2 = load(spec['p2'])

    cfg['run_name'] = name
    cfg['tag'] = TAG
    cfg['checkpoint_name'] = PLACEHOLDER
    # FULL resume on both roles (seed and own-running); the sbatch picks the file.
    cfg['load_weights_only'] = False
    cfg['continue_from_checkpoint'] = False
    # only the weights-only path reads this; leaving it would imply it does something
    cfg.pop('warm_start_ignore_problem_keys', None)
    # prior_model is not checkpointed; leg 1's stub writes it, later legs glob it
    cfg['prior_model_name'] = PRIOR_PLACEHOLDER
    cfg['epochs'] = EPOCHS
    # phase-2 buffers are dynamics; archive them with the model as prod_sep02 did
    cfg['archive_period'] = 5000
    cfg['archive_buffers'] = True

    # the validated phase-2 protocol, verbatim from the same family's prod_sep02 arm
    cfg['protocol'] = 'prod_eq'
    cfg['protocols'].pop('mle09', None)
    cfg['protocols']['prod_eq'] = copy.deepcopy(p2['protocols']['prod_eq'])

    lc = cfg['lr_control']
    lc.update(mode='fixed', fixed_scale=float(spec['scale']), fire_cut_factor=1.0,
              repeat_every=0, burn_in_steps=500)
    # == fixed_scale: a rewind onto a burn-in-era checkpoint restores lr_ctrl.scale
    lc['burn_in_scale'] = float(spec['scale'])
    lc['hard_failure']['loss_excursion_k'] = float(p2['lr_control']['hard_failure']['loss_excursion_k'])

    cfg['buffers']['anchor_buffer'].update(ANCHOR)

    if spec['mlip']:
        cfg['traj_checkpoint'] = True
        cfg['batch_size'] = UMA_BATCH
        cfg['max_batch_size'] = UMA_BATCH
        cfg['batch_util_target'] = 0
        cfg['grow_batch_size'] = True
        cfg['eval_period'] = int(p2['eval_period'])
        cfg['eval_num_samples'] = int(p2['eval_num_samples'])
        cfg['energy_config']['internal_oom_recovery'] = True
    else:
        cfg['traj_checkpoint'] = False
    return cfg


def _scan_local_paths(node, name, trail='cfg'):
    if isinstance(node, dict):
        for k, v in node.items():
            _scan_local_paths(v, name, trail + '.' + str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _scan_local_paths(v, name, trail + '[' + str(i) + ']')
    elif isinstance(node, str) and ('D:' in node or '\\' in node):
        raise AssertionError(name + ': local path at ' + trail + ': ' + repr(node))


def check(cfg, name, fam):
    spec = FAM[fam]
    seed = load(spec['seed'])
    # THE ONE ASSERT THAT MATTERS: load_full refuses a checkpoint whose stored
    # problem_def differs, with no exemption. Same identity as the seed arm, or no run.
    assert problem_def(cfg) == problem_def(seed), name + ': problem identity moved from the mle09 seed'
    assert cfg['prior_path'] == seed['prior_path'] == cfg['molecules_path'], name

    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == lc['burn_in_scale'] == spec['scale'], name
    assert lc['fire_cut_factor'] == 1.0 and lc['repeat_every'] == 0, name
    assert cfg['checkpoint_name'] == PLACEHOLDER and cfg['prior_model_name'] == PRIOR_PLACEHOLDER, name
    assert cfg['load_weights_only'] is False and cfg['continue_from_checkpoint'] is False, name
    assert 'warm_start_ignore_problem_keys' not in cfg, name
    assert cfg['epochs'] >= 500_000, name + ': epochs is an absolute cap and the seed is at ~112k'
    assert cfg['integrator']['T'] == 100 == cfg['eval_T'], name
    assert cfg['energy_config'].get('reward_range'), name + ': soft clip NOT armed'
    ab = cfg['buffers']['anchor_buffer']
    for k, v in ANCHOR.items():
        assert ab[k] == v, name + ': anchor_buffer.' + k
    assert cfg['traj_checkpoint'] is spec['mlip'], name + ': traj_checkpoint rides the MLIPs only'
    if spec['mlip']:
        assert cfg['batch_size'] == cfg['max_batch_size'] == UMA_BATCH, name
        assert cfg['batch_util_target'] == 0, name
    else:
        assert cfg['batch_size'] == 1000 and cfg['max_batch_size'] == 4000, name
        assert cfg['batch_util_target'] == 0.65, name

    assert cfg['protocol'] == 'prod_eq' and 'mle09' not in cfg['protocols'], name
    stages = cfg['protocols']['prod_eq']['stages']
    names = [s['name'] for s in stages]
    assert names[0] == 'train_prior', name + ': the checkpoint stage must re-enter by name'
    stub = stages[0]
    assert stub['exit'] == [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}], name + ': stub exit'
    assert 'snapshot_prior' in stub['on_exit'] and 'skip_if' not in stub, name
    fused = [s for s in stages if s.get('train_mode') == 'fused']
    assert fused, name + ': no fused stage'
    for s in fused:
        assert s['fracs'] == FRACS and 'balance' not in s, name
    for s in stages:
        sensor = s.get('hot_lr_sensor')
        if isinstance(sensor, dict):
            assert sensor.get('action') == 'report', name + ': hot_lr can still fire'
    _scan_local_paths(cfg, name)


def build():
    out = {}
    for fam, spec in FAM.items():
        name = TAG + '_' + fam + '_' + tag_for(spec['scale'])
        cfg = deltas(load(spec['seed']), fam, name)
        check(cfg, name, fam)
        out[name] = (cfg, fam)
    assert len(out) == 4
    return out


SBATCH = """#!/bin/bash
#SBATCH --time={wall}
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-{last}
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=p12
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prod_sep12/joblogs/%x_%A_%a.out

# prod_sep12: phase 2 from the mle09 best-MLE checkpoints. Arm = row of
# INDEX.tsv (line 1 is the header). DO NOT EDIT --array BY HAND: make.py
# rewrites it. Resubmit this same file to continue an arm past the wall.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/prod_sep12
LOGS=${{ARMS}}/joblogs
CKPTS=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints
mkdir -p ${{LOGS}}

ARM=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $1}}' ${{ARMS}}/INDEX.tsv)
SRC=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $2}}' ${{ARMS}}/INDEX.tsv)
if [ -z "${{ARM}}" ]; then echo "no arm at row ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; fi
CONFIG=${{ARMS}}/${{ARM}}.yaml
if [ ! -f "${{CONFIG}}" ]; then echo "missing config ${{CONFIG}}" >&2; exit 1; fi

J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}
RESOLVED=${{J}}.yaml

# A DEAD ARM MUST NOT EAT ITS RESUBMISSION. total_reloads is not checkpointed,
# so a leg that aborted UNRECOVERABLE would hand the next one a fresh rewind
# budget and the same poisoned _running.pt.
if [ -f ${{CKPTS}}/${{ARM}}.dead ]; then
    echo "arm ${{ARM}} aborted UNRECOVERABLE on an earlier leg -- skipping"
    exit 0
fi

# RESUME OR SEED. train.py's loader is `if checkpoint_name ... elif
# continue_from_checkpoint`, so checkpoint_name ALWAYS wins -- an arm with its
# own _running.pt continues it; only a first launch seeds from the mle09 arm.
OWN=$(ls -t ${{CKPTS}}/*${{ARM}}_*_running.pt 2>/dev/null | head -1)
if [ -n "${{OWN}}" ]; then
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  RESUME: $(basename ${{OWN}})"
    CK=${{OWN}}
else
    # THE SEED IS THE mle09 ARM'S _best.pt: its bwd/mle record, hardlinked off
    # `running` at the 50-step save cadence. mle09's stage is terminal, so no
    # phase1_exit.pt exists for these arms. REFUSES AN AMBIGUOUS MATCH.
    N=$(ls ${{CKPTS}}/*${{SRC}}_*_best.pt 2>/dev/null | wc -l)
    if [ "${{N}}" -eq 0 ]; then
        echo "FATAL: no seed matches *${{SRC}}_*_best.pt in ${{CKPTS}}" >&2; exit 1
    fi
    if [ "${{N}}" -gt 1 ]; then
        echo "FATAL: ${{N}} ambiguous matches for *${{SRC}}_*_best.pt:" >&2
        ls ${{CKPTS}}/*${{SRC}}_*_best.pt >&2; exit 1
    fi
    CK=$(ls ${{CKPTS}}/*${{SRC}}_*_best.pt)
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  SEED: $(basename ${{CK}}) (mtime $(stat -c %y ${{CK}}))"
fi

# THE PRIOR MODEL. Not in the checkpoint; only train_prior's snapshot_prior
# writes it, and protocol.begin() returns early on a resumed step, so a
# continued leg must be handed its own. On the first leg the stub writes it.
OWNPRIOR=$(ls -t ${{CKPTS}}/*${{ARM}}_*_prior.pt 2>/dev/null | head -1)
if [ -n "${{OWNPRIOR}}" ]; then
    PM=$(basename ${{OWNPRIOR}}); echo "  prior model <- ${{PM}}"
else
    PM=null; echo "  prior model <- null (leg 1 writes it)"
fi

sed -e "s|{placeholder}|$(basename ${{CK}})|" \\
    -e "s|{prior_placeholder}|${{PM}}|" ${{CONFIG}} > ${{RESOLVED}}
if grep -q '{placeholder}\\|{prior_placeholder}' ${{RESOLVED}}; then
    echo "FATAL: placeholder left in ${{RESOLVED}}" >&2; exit 1
fi

{{ nvidia-smi -L
  scontrol show job ${{SLURM_JOB_ID}}
  echo "nodelist: ${{SLURM_NODELIST}}  host: $(hostname)"
}} > ${{J}}.info 2>&1

# The out-of-process occupancy record -- the number the scheduler cancels on.
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \\
    --format=csv,nounits -l 10 > ${{J}}_smi.csv &
SMI_PID=$!
smi_epilogue() {{
    kill ${{SMI_PID}} 2>/dev/null
    sacct -j ${{SLURM_JOB_ID}} --format=JobID,State,ExitCode,Elapsed,NodeList,Reason,Comment%64 \\
        > ${{J}}_sacct.txt 2>&1
}}
trap smi_epilogue EXIT TERM

# THE LAUNCH LINE IS COPIED, NOT REWRITTEN: PYTHONPATH must hold the PARENT of
# energy_sampling or every arm dies at import; both --bind lines are required.
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
    " 2>&1 | tee ${{J}}.trainlog

if grep -q 'UNRECOVERABLE' ${{J}}.trainlog 2>/dev/null; then
    echo "arm ${{ARM}} exhausted its rewind budget -- sentinel set, resubmissions will skip"
    touch ${{CKPTS}}/${{ARM}}.dead
fi
"""


def main():
    arms = build()
    logs = HERE / 'joblogs'
    logs.mkdir(exist_ok=True)
    (logs / '.gitkeep').write_text(
        'ships this directory to the cluster; SLURM cannot create --output\n', encoding='utf-8')
    for name, (cfg, _fam) in arms.items():
        with (HERE / (name + '.yaml')).open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\tscale\tlr\n')
        for name, (cfg, fam) in arms.items():
            s = FAM[fam]['scale']
            f.write('%s\t%s\t%g\t%.4g\n' % (name, FAM[fam]['src'], s, 1.25e-4 * s))
    with (HERE / 'submit_prod_sep12.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(wall=WALL, last=len(arms) - 1, placeholder=PLACEHOLDER,
                              prior_placeholder=PRIOR_PLACEHOLDER))
    for name, (cfg, fam) in arms.items():
        print('%-18s seed=%-16s scale=%-7g traj_ckpt=%s batch=%s/%s'
              % (name, FAM[fam]['src'], FAM[fam]['scale'], cfg['traj_checkpoint'],
                 cfg['batch_size'], cfg['max_batch_size']))


if __name__ == '__main__':
    main()
