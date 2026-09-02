"""prod_sep02 -- the 16-arm production battery. Five of these become paper models.

WHAT IS BEING VARIED: the learning rate, and nothing else. Every other axis is
pinned to one value across all sixteen arms, because the point is defensible
models rather than a mechanism study:

  frozen anchors        membership fixed, surprise sweep off, unweighted draw
  fixed fractions       fwd/bwd/replay 0.05/0.475/0.475, ratio controller ABSENT
  armed soft clip       energy_config.reward_range 250
  mode: fixed           no bracket, no repeat, no mid-run rate search
  fire_cut_factor 1.0   a fire rewinds but must NOT move the rate
  hot_lr report         the drawdown sensor observes; only the hard bars fire

EACH ARM IS DERIVED FROM AN ALREADY-VALIDATED CLUSTER CONFIG, not regenerated
from mk_dev. Every path, MLIP setting and problem hash therefore starts at a
value this cluster has already run, and anything that breaks is attributable to
the deltas below. That is deliberate insurance against the failure family that
has bitten three times -- local state that passes every local check because the
local box has it, then dies at cluster startup with no wandb run.

TWO WALLS, TWO SBATCH FILES. `--time` is per job, so the 7-day arms cannot share
an array with the 2-day ones. They must also not share a dependency chain:
submit_chain.sh is whole-array afterany, so a chained leg would wait seven days
for the long arm to finish.

  submit_prod_sep02_long.sbatch    4 arms, --time=7-00:00:00, ONE leg, no chain
  submit_prod_sep02_short.sbatch  12 arms, --time=2-00:00:00, chain x3 = 6 days

The four long arms are the paper models and carry ZERO chaining risk, because
they have no chain. The worst case for the twelve short arms is that leg 1
completes and the rest fail, which still banks 2 days of training per arm.
"""
import pathlib
import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PLACEHOLDER = 'WARM_CHECKPOINT_PLACEHOLDER'
PRIOR_PLACEHOLDER = 'PRIOR_MODEL_PLACEHOLDER'

FRACS = {'fwd': 0.05, 'bwd': 0.475, 'replay': 0.475}
ANCHOR = {'frozen': True, 'online_loss_flow': False,
          'thin_every_n_evals': 0, 'refresh_every_n_evals': 0, 'replay_beta': 1.0}

FAM = {
    'mip': dict(base='prod_t100_p2/pt100mip2_lr1p0.yaml', src='pt100_mip_lr4p0',
                exit=19010, head=200000, batch=None),
    'neh': dict(base='prod_t100_p2/pt100neh2_lr1p0.yaml', src='pt100_neh_lr4p0',
                exit=14010, head=200000, batch=None),
    'mipu': dict(base='mipu_bwd_aug31/mb31_bwd25_lr1p56.yaml', src='pt100_mipu_lr4p0',
                 exit=5010, head=100000, batch=1600),
    'nehu': dict(base='uma_stab_aug30/us30_b_lr0p5.yaml', src='pt100_nehu_lr4p0',
                 exit=5010, head=100000, batch=1600),
    'acr': dict(base='acr_c_aug31/acrb3_b50_lr0p025.yaml', src='pt100_acr_lr4p0',
                exit=9010, head=200000, batch=1000),
}

# (family, scale, wall).  'long' = 7-day single leg; 'short' = 2-day x 3 legs.
#
# ELJ IS NOT A SCAN. 1.0 is the measured winner of a 4-rung fan (0.25/0.5/1.0/2.0)
# run on these exact systems at T=100 from these exact phase-1 exits. nehzor is
# shaded to 0.5 because its 0.5 and 1.0 sit inside noise on every axis while 2.0
# detonated -- free insurance on a 7-day unattended arm. mipcas stays at 1.0,
# where 1.0 is a clear energy win.
#
# The MLIP families ARE scans, 2x spacing. The long slot goes to the rate with the
# best evidence rather than the middle of the grid: mipu 0.0625 (the 0.125 anchor
# was measured at batch ~4000, not the 1600 shipped here), nehu 0.125 (the only
# rate visited on that route). acr has no long arm by design.
ARMS = (
    [('mip', 1.0, 'long'), ('neh', 0.5, 'long')]
    + [('mipu', s, 'long' if s == 0.0625 else 'short')
       for s in (0.015625, 0.03125, 0.0625, 0.125, 0.25)]
    + [('nehu', s, 'long' if s == 0.125 else 'short')
       for s in (0.015625, 0.03125, 0.0625, 0.125, 0.25)]
    + [('acr', s, 'short') for s in (0.0125, 0.025, 0.05, 0.1)]
)


def tag_for(scale):
    """0.0625 -> lr0p0625. Stable, filename-safe, and collision-free at 2x
    spacing (asserted in build)."""
    return 'lr' + ('%g' % scale).replace('.', 'p')


def deltas(cfg, fam, name, scale):
    spec = FAM[fam]
    cfg['run_name'] = name
    cfg['tag'] = 'p02'
    cfg['checkpoint_name'] = PLACEHOLDER
    # FULL resume, both roles. One static config serves leg 1 (seed from the
    # phase-1 exit) and every later leg (resume from the arm's own _running.pt),
    # because the SBATCH resolves the path, not the YAML. `true` would be
    # seed-only and would restart each later leg at step 0 with empty buffers.
    cfg['load_weights_only'] = False
    cfg['continue_from_checkpoint'] = False
    # `prior_model` is NOT in the checkpoint, and only train_prior's
    # snapshot_prior on_exit creates it -- but protocol.begin() returns early when
    # step_ind != 0, so a resumed leg re-enters INSIDE equilibration and the stub
    # never runs. Without this the arm comes up with no prior sampler and
    # bwd_sampling_mode: prior degrades to an anchors-only buffer across 95% of
    # the branch budget, announced as a single print line. The sbatch resolves
    # this the same way it resolves the warm checkpoint: glob, else null on leg 1.
    cfg['prior_model_name'] = PRIOR_PLACEHOLDER
    # ABSOLUTE step index, not a count: trange(init_step, epochs+1) is its ONLY
    # consumer, so an unreachable value costs nothing and the wall ends the job.
    # A carried-over per-leg value is a RECORDED failure -- prod_t100_p2's
    # epochs 14500 gave mip2 zero iterations, four arms finished in 36 seconds.
    cfg['epochs'] = spec['exit'] + spec['head']

    if spec['batch'] is not None:
        cfg['batch_size'] = spec['batch']
        cfg['max_batch_size'] = spec['batch']
        cfg['batch_util_target'] = 0
        cfg['grow_batch_size'] = True

    cfg.setdefault('buffers', {}).setdefault('anchor_buffer', {}).update(ANCHOR)

    lc = cfg.setdefault('lr_control', {})
    lc.update(mode='fixed', fixed_scale=float(scale), fire_cut_factor=1.0,
              repeat_every=0, burn_in_steps=500)
    # EQUAL TO fixed_scale, deliberately. lr_ctrl.scale is restored from whatever
    # checkpoint a fire rewinds to, so a rewind onto a burn-in-era checkpoint
    # would pin the arm at the burn-in rate for the rest of the run, and
    # fire_cut_factor 1.0 does not prevent it. Equal rates also make
    # _arm_cruise_bar a no-op, so the bars stay fitted at the operating rate and
    # there is no suspension window to cross.
    lc['burn_in_scale'] = float(scale)

    n = 0
    for proto in (cfg.get('protocols') or {}).values():
        for stage in (proto.get('stages') or []):
            if isinstance(stage.get('hot_lr_sensor'), dict):
                stage['hot_lr_sensor']['action'] = 'report'
            if stage.get('train_mode') == 'fused' and 'fracs' in stage:
                stage['fracs'] = dict(FRACS)
                # the ratio controller is disabled by ABSENCE: protocol.tick only
                # calls _balance_tick when stage.balance is not None.
                stage.pop('balance', None)
                stage.pop('min_fracs', None)
                n += 1
            # leg 1 must RUN the stub -- it is what writes the prior snapshot the
            # later legs load, so skip_if would skip the thing being relied on.
            if stage.get('name') == 'train_prior':
                stage.pop('skip_if', None)
    assert n >= 1, name + ': no fused stage found to pin fractions on'
    return cfg


def _scan_local_paths(node, name, trail='cfg'):
    """THE LOCAL-PATH TRAP: a D:\\ path passes every local check because the local
    box has the file, then dies at cluster startup with no wandb run."""
    if isinstance(node, dict):
        for k, v in node.items():
            _scan_local_paths(v, name, trail + '.' + str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _scan_local_paths(v, name, trail + '[' + str(i) + ']')
    elif isinstance(node, str) and ('D:' in node or '\\' in node):
        raise AssertionError(name + ': local path at ' + trail + ': ' + repr(node))


def check(cfg, name, fam, scale):
    """Refuse to emit rather than diagnose after a wasted job."""
    spec = FAM[fam]
    lc = cfg['lr_control']
    ab = cfg['buffers']['anchor_buffer']
    assert lc['mode'] == 'fixed', name
    assert lc['fixed_scale'] == lc['burn_in_scale'] == scale, name
    assert lc['fire_cut_factor'] == 1.0, name
    assert ab['frozen'] is True and ab['online_loss_flow'] is False, name
    assert ab['thin_every_n_evals'] == 0, name
    assert ab['refresh_every_n_evals'] == 0, name
    assert cfg['epochs'] > spec['exit'] + 50000, name + ': epochs not unreachable'
    assert cfg['checkpoint_name'] == PLACEHOLDER, name
    assert cfg['prior_model_name'] == PRIOR_PLACEHOLDER, name
    assert cfg['load_weights_only'] is False, name
    assert cfg['energy_config'].get('reward_range'), name + ': soft clip NOT armed'
    fused = [s for p in cfg['protocols'].values() for s in p['stages']
             if s.get('train_mode') == 'fused']
    assert fused, name + ': no fused stage'
    for s in fused:
        assert s['fracs'] == FRACS, name + ': fractions not pinned'
        assert 'balance' not in s, name + ': ratio controller still present'
        sensor = s.get('hot_lr_sensor')
        if isinstance(sensor, dict):
            assert sensor.get('action') == 'report', name + ': hot_lr can still fire'
    _scan_local_paths(cfg, name)


def build():
    out = {}
    for fam, scale, wall in ARMS:
        name = 'p02_' + fam + '_' + tag_for(scale)
        assert name not in out, 'arm name collision: ' + name
        cfg = yaml.safe_load((ROOT / FAM[fam]['base']).read_text(encoding='utf-8'))
        cfg = deltas(cfg, fam, name, scale)
        check(cfg, name, fam, scale)
        out[name] = (cfg, fam, scale, wall)
    assert len(out) == len(ARMS) == 16, 'expected 16 arms, got %d' % len(out)
    return out


# The two templates differ ONLY in the header, the index they read, and whether
# they honour the .dead sentinel. Everything below RESOLVE is shared.
_HEAD = """#!/bin/bash
#SBATCH --time={wall}
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-{last}
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name={jobname}
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prod_sep02/joblogs/%x_%A_%a.out

# prod_sep02 {label}. Arm = row of {index} (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/prod_sep02
LOGS=${{ARMS}}/joblogs
CKPTS=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints
mkdir -p ${{LOGS}}

ARM=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $1}}' ${{ARMS}}/{index})
SRC=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $2}}' ${{ARMS}}/{index})
if [ -z "${{ARM}}" ]; then echo "no arm at row ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; fi
CONFIG=${{ARMS}}/${{ARM}}.yaml
if [ ! -f "${{CONFIG}}" ]; then echo "missing config ${{CONFIG}}" >&2; exit 1; fi

J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}
RESOLVED=${{J}}.yaml
{sentinel}
# RESUME OR SEED. train.py's loader is `if checkpoint_name ... elif
# continue_from_checkpoint`, so checkpoint_name ALWAYS wins -- without this an
# extended arm would silently restart from the phase-1 exit and discard its own
# progress.
OWN=$(ls -t ${{CKPTS}}/*${{ARM}}_*_running.pt 2>/dev/null | head -1)
if [ -n "${{OWN}}" ]; then
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  RESUME: $(basename ${{OWN}})"
    CK=${{OWN}}
else
    # REFUSES AN AMBIGUOUS MATCH rather than taking the newest: every arm must
    # provably seed from the same file or it is not a comparison.
    N=$(ls ${{CKPTS}}/*${{SRC}}_*_phase1_exit.pt 2>/dev/null | wc -l)
    if [ "${{N}}" -eq 0 ]; then
        echo "FATAL: no phase-1 exit matches *${{SRC}}_*_phase1_exit.pt" >&2; exit 1
    fi
    if [ "${{N}}" -gt 1 ]; then
        echo "FATAL: ${{N}} ambiguous matches for *${{SRC}}_*_phase1_exit.pt:" >&2
        ls ${{CKPTS}}/*${{SRC}}_*_phase1_exit.pt >&2; exit 1
    fi
    CK=$(ls ${{CKPTS}}/*${{SRC}}_*_phase1_exit.pt)
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  SEED: $(basename ${{CK}})"
fi

# THE PRIOR MODEL, resolved the same way and for the same reason. `prior_model`
# is not in the checkpoint and only train_prior's snapshot_prior creates it, but
# protocol.begin() returns early when step_ind != 0 -- so a resumed leg would
# come up with no prior sampler and bwd_sampling_mode: prior would silently
# degrade to an anchors-only buffer across 95% of the branch budget. On leg 1
# there is nothing to find and `null` is correct: the stub runs and writes it.
OWNPRIOR=$(ls -t ${{CKPTS}}/*${{ARM}}_*_prior.pt 2>/dev/null | head -1)
if [ -n "${{OWNPRIOR}}" ]; then
    PM=$(basename ${{OWNPRIOR}}); echo "  prior model <- ${{PM}}"
else
    PM=null; echo "  prior model <- null (leg 1 writes it)"
fi

sed -e "s|{placeholder}|$(basename ${{CK}})|" \\
    -e "s|{prior_placeholder}|${{PM}}|" ${{CONFIG}} > ${{RESOLVED}}

{{ nvidia-smi -L
  scontrol show job ${{SLURM_JOB_ID}}
  echo "nodelist: ${{SLURM_NODELIST}}  host: $(hostname)"
}} > ${{J}}.info 2>&1

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
{epilogue}"""

# Only the chained arms need this. A 7-day arm has no next leg to protect.
_SENTINEL = """
# A DEAD ARM MUST NOT EAT ITS REMAINING LEGS. total_reloads is not checkpointed,
# so a leg that aborted UNRECOVERABLE hands the next leg a fresh rewind budget,
# the same poisoned _running.pt, and an identical re-detonation. This is a static
# LR scan, so some arms are MEANT to be too hot and cannot self-recover.
if [ -f ${{CKPTS}}/${{ARM}}.dead ]; then
    echo "arm ${{ARM}} aborted UNRECOVERABLE on an earlier leg -- skipping"
    exit 0
fi
"""

_EPILOGUE = """
if grep -q 'UNRECOVERABLE' ${{J}}.trainlog 2>/dev/null; then
    echo "arm ${{ARM}} exhausted its rewind budget -- sentinel set, later legs will skip"
    touch ${{CKPTS}}/${{ARM}}.dead
fi
"""


def main():
    arms = build()
    # joblogs must reach the CLUSTER, not just exist locally: SLURM cannot create
    # the --output directory, so a missing one kills the job at launch, before
    # python and before wandb. git does not track empty directories.
    logs = HERE / 'joblogs'
    logs.mkdir(exist_ok=True)
    (logs / '.gitkeep').write_text(
        'ships this directory to the cluster; SLURM cannot create --output\n',
        encoding='utf-8')

    for name, (cfg, _fam, _scale, _wall) in arms.items():
        with (HERE / (name + '.yaml')).open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    for wall, index, jobname, sbatch, label, walltime in (
            ('long', 'INDEX_long.tsv', 'p02long', 'submit_prod_sep02_long.sbatch',
             '7-day single-leg arms', '7-00:00:00'),
            ('short', 'INDEX_short.tsv', 'p02short', 'submit_prod_sep02_short.sbatch',
             '2-day chained arms', '2-00:00:00')):
        rows = [(n, FAM[f]['src'], s) for n, (_c, f, s, w) in arms.items() if w == wall]
        with (HERE / index).open('w', encoding='utf-8', newline='\n') as f:
            f.write('arm\twarm_src\tscale\tlr\n')
            for n, src, s in rows:
                f.write('%s\t%s\t%g\t%.4g\n' % (n, src, s, 1.25e-4 * s))
        # THE PREFIX ASSERT: an index naming an arm whose YAML is absent dies as
        # "missing config" at launch. This has happened.
        for n, _src, _s in rows:
            assert (HERE / (n + '.yaml')).exists(), 'index names a missing config: ' + n
        with (HERE / sbatch).open('w', encoding='utf-8', newline='\n') as f:
            f.write(_HEAD.format(
                wall=walltime, last=len(rows) - 1, jobname=jobname, index=index,
                label=label, placeholder=PLACEHOLDER,
                prior_placeholder=PRIOR_PLACEHOLDER,
                sentinel=_SENTINEL.format(CKPTS='{CKPTS}', ARM='{ARM}')
                if wall == 'short' else '',
                epilogue=_EPILOGUE.format(J='{J}', ARM='{ARM}', CKPTS='{CKPTS}')
                if wall == 'short' else ''))
        print('%-6s %2d arms -> %s' % (wall, len(rows), sbatch))
        for n, _src, s in rows:
            print('        %-22s scale=%-9g lr=%.4g' % (n, s, 1.25e-4 * s))


if __name__ == '__main__':
    main()
