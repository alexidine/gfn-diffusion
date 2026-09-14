"""flk_sep14 -- the mip Z-flicker battery: eight 8-hour arms off p12_mip_lr1's
live checkpoint, one knob each.

    python configs/flk_sep14/make.py

WHAT IS BEING FIXED. p12_mip_lr1 runs a ~2000-step limit cycle in replay share,
log Z (5 nats) and bwd/under_coverage. The gated ramp's ratchet compares the
level to its stage-wide minimum, that minimum was set at a log Z peak (the
level carries Z at ~-0.2 nat/nat), and the Schmitt band [best+0.25, best+0.5]
sits inside the level's operating range -- so every Z relaxation trips it,
replay is cut to the floor, the policy sharpens, Z falls, and the cut holds
(memory: project_p12_mip_ramp_limit_cycle). The high-Z side of the cycle is the
better model on the forward oracle, on held-out forward, on both Z estimators
and on every Z-free bwd channel; its one cost is replay memorisation 0.71->0.62,
still above the 1/e line. The owner wants a robust TB minimum without the
flicker, and to see what the replay bounds, an anchors-only prior churn and the
anchor noise do on this checkpoint.

THE SEED IS THE PARENT'S LIVE `_running.pt` under a FULL resume, plus the
parent's `_prior.pt` (the stub wrote it; source: prior_model arms need it).
Step ~135k, stage equilibration; a resume re-enters INSIDE the stage, so
on_enter does not rerun and the buffers come from the parent's rolling sidecar.
Every arm therefore starts from the identical state, mid-cycle wherever the
parent happens to be, which is the point: the arms differ only in the knob.

WALL 8 h: mip runs 2.6 s/step at its settled batch of 1600, so ~11k steps, five
periods of the cycle -- enough to see it persist or vanish, and enough for a
pinned arm to show where Z and the forward TB error settle.

ARMS (base = p12_mip_lr1.yaml, committed):
  ctrl            unchanged -- the cycle, as the reference
  band            the guard sized to the Z swing: slope bar 1.0, ratchet_tol 1.0,
                  release 0.5. Transparent in descent, ignores a 2-nat Z
                  relaxation at the plateau, still trips on a 1-nat coverage loss
  band_cap50      band + replay bounded at 0.5 (bounds.replay [0.1, 0.5]) -- the
                  upper bound of the ramp, tested as a knob
  pin30           fracs 0/0.7/0.3, PINNED by collapsing the ramp's bounds to a point
                  (a stage without balance would inherit the parent's 0.10 on resume)
  pin50           fracs 0/0.5/0.5 (the rr07 shape)
  band_anch       band + prior_buffer.source anchors: churn from noised anchors,
                  the prior model out of the loop (the owner's intended default)
  band_anch_n3x   band_anch + anchor noise x3 (noise_log_range [-2.0, -1.0])
  band_anch_nd3   band_anch + anchor noise /3 ([-3.0, -2.0])
The noise ladder rides on the anchors source because under prior_model only the
10% anchor floor share is ever noised, and the knob would be nearly inert.

WHAT TO READ: fwd/log_Z_learned sd over the last 4k steps (ctrl ~1.1), the
count of protocol/gr_tripped onsets, Replay Frac path, fwd/tb_err and
eval_fwd/tb_err level, replay/resid_vs_intake (must stay above 0.37), and
bwd/logw_std_within as the Z-free coverage channel.
"""
import copy
import pathlib
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ES = ROOT.parent
BASE = ROOT / 'prod_sep12' / 'p12_mip_lr1.yaml'
PARENT = 'p12_mip_lr1'
PLACEHOLDER = 'WARM_CHECKPOINT_PLACEHOLDER'
PRIOR_PLACEHOLDER = 'PRIOR_MODEL_PLACEHOLDER'
TAG = 'flk14'
WALL = '8:00:00'
BAND = dict(bar=1.0, ratchet_tol=1.0, ratchet_release_tol=0.5)
EXECUTED = ('configs/mk_dev.yaml', 'train.py', 'protocol.py', 'buffer.py',
            'gflownet_losses.py', 'checkpointing.py', 'utils.py', 'config_invariants.py',
            'models/gfn.py', 'energies/molecular_crystal.py')


def _band(cfg):
    eq = _eq(cfg)
    eq['balance'].update(BAND)


def _eq(cfg):
    return [s for s in cfg['protocols']['unconditional_tb']['stages'] if s['name'] == 'equilibration'][0]


def _pin(cfg, replay):
    # A PIN THAT SURVIVES A RESUME. Removing `balance` would NOT pin: the
    # fractions are modeller state restored from the parent's checkpoint, and a
    # stage without balance never re-stamps its entry fracs (protocol.tick only
    # moves them through _balance_tick; entry fracs are applied at advance()).
    # The gated ramp clips its share to `bounds` on every tick, so bounds
    # collapsed to a point pin the split from the first tick, whatever was restored.
    eq = _eq(cfg)
    bwd = round(1.0 - replay, 3)
    eq['fracs'] = {'fwd': 0.0, 'bwd': bwd, 'replay': replay}
    eq['balance']['bounds'] = {'bwd': [bwd, bwd], 'replay': [replay, replay]}


def _anchors(cfg):
    cfg['buffers']['prior_buffer']['source'] = 'anchors'


def _noise(cfg, lo, hi):
    cfg['buffers']['anchor_buffer']['noise_log_range'] = [lo, hi]


ARMS = {
    'ctrl':          [],
    'band':          [_band],
    'band_cap50':    [_band, lambda c: _eq(c)['balance']['bounds'].__setitem__('replay', [0.1, 0.5])],
    'pin30':         [lambda c: _pin(c, 0.3)],
    'pin50':         [lambda c: _pin(c, 0.5)],
    'band_anch':     [_band, _anchors],
    'band_anch_n3x': [_band, _anchors, lambda c: _noise(c, -2.0, -1.0)],
    'band_anch_nd3': [_band, _anchors, lambda c: _noise(c, -3.0, -2.0)],
}


def dirty_files():
    out = subprocess.run(['git', 'status', '--porcelain', '--'] + [str(ES / p) for p in EXECUTED],
                         capture_output=True, text=True, cwd=str(ES), check=True).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def build():
    base = yaml.safe_load(BASE.read_text(encoding='utf-8'))
    out = {}
    for arm, deltas in ARMS.items():
        cfg = copy.deepcopy(base)
        name = TAG + '_' + arm
        cfg['run_name'] = name
        cfg['tag'] = TAG
        cfg['checkpoint_name'] = PLACEHOLDER
        cfg['prior_model_name'] = PRIOR_PLACEHOLDER
        cfg['load_weights_only'] = False
        cfg['continue_from_checkpoint'] = False
        for d in deltas:
            d(cfg)
        check(cfg, name, arm)
        out[name] = cfg
    return out


def check(cfg, name, arm):
    eq = _eq(cfg)
    assert cfg['checkpoint_name'] == PLACEHOLDER and cfg['prior_model_name'] == PRIOR_PLACEHOLDER, name
    assert cfg['load_weights_only'] is False, name + ': the seed is a full resume'
    assert cfg['epochs'] >= 500_000, name
    assert eq['fwd_rollout_every'] == 20, name
    if arm.startswith('pin'):
        b = eq['balance']; r = eq['fracs']['replay']
        assert b['bounds'] == {'bwd': [round(1 - r, 3), round(1 - r, 3)], 'replay': [r, r]}, name + ': bounds must be a point'
        assert abs(sum(eq['fracs'].values()) - 1.0) < 1e-9 and eq['fracs']['fwd'] == 0.0, name
        assert b['pinned'] == {'fwd': 0.0}, name
    else:
        b = eq['balance']
        assert b['kind'] == 'gated_ramp' and b['pinned'] == {'fwd': eq['fracs']['fwd']}, name
        if arm == 'ctrl':
            assert b['bar'] == 0.0 and b['ratchet_tol'] == 0.5 and b['ratchet_release_tol'] == 0.25, name
        else:
            assert b['bar'] == 1.0 and b['ratchet_tol'] == 1.0 and b['ratchet_release_tol'] == 0.5, name
        assert b['bounds']['replay'] == ([0.1, 0.5] if arm == 'band_cap50' else [0.1, 0.75]), name
    src = cfg['buffers']['prior_buffer']['source']
    assert src == ('anchors' if 'anch' in arm else 'prior_model'), name
    nr = cfg['buffers']['anchor_buffer']['noise_log_range']
    want = {'band_anch_n3x': [-2.0, -1.0], 'band_anch_nd3': [-3.0, -2.0]}.get(arm, [-2.5, -1.5])
    assert nr == want, name
    for s in cfg['protocols']['unconditional_tb']['stages']:
        sensor = s.get('hot_lr_sensor')
        if isinstance(sensor, dict):
            assert sensor.get('action', 'report') == 'report', name


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
#SBATCH --job-name=flk14
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/flk_sep14/joblogs/%x_%A_%a.out

# flk_sep14: eight knobs off p12_mip_lr1's live checkpoint. Arm = row of
# INDEX.tsv (line 1 is the header). DO NOT EDIT --array BY HAND.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/flk_sep14
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

if [ -f ${{CKPTS}}/${{ARM}}.dead ]; then
    echo "arm ${{ARM}} aborted UNRECOVERABLE on an earlier leg -- skipping"; exit 0
fi

# RESUME OWN, else SEED FROM THE PARENT'S LIVE running.pt. The parent is a running
# job that rewrites this file atomically every 50 steps; whichever version is
# on disk at launch is the seed, and the launch line prints its mtime so the
# seed step can be read back from the parent's log.
OWN=$(ls -t ${{CKPTS}}/*${{ARM}}_*_running.pt 2>/dev/null | head -1)
if [ -n "${{OWN}}" ]; then
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  RESUME: $(basename ${{OWN}})"; CK=${{OWN}}
else
    N=$(ls ${{CKPTS}}/*${{SRC}}_*_running.pt 2>/dev/null | wc -l)
    if [ "${{N}}" -ne 1 ]; then
        echo "FATAL: ${{N}} matches for *${{SRC}}_*_running.pt (need exactly 1):" >&2
        ls ${{CKPTS}}/*${{SRC}}_*_running.pt >&2; exit 1
    fi
    CK=$(ls ${{CKPTS}}/*${{SRC}}_*_running.pt)
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  SEED: $(basename ${{CK}}) (mtime $(stat -c %y ${{CK}}))"
fi

# THE PRIOR MODEL: own if a leg wrote one, else the PARENT'S (its stub wrote it).
OWNPRIOR=$(ls -t ${{CKPTS}}/*${{ARM}}_*_prior.pt 2>/dev/null | head -1)
if [ -n "${{OWNPRIOR}}" ]; then PM=$(basename ${{OWNPRIOR}})
else
    NP=$(ls ${{CKPTS}}/*${{SRC}}_*_prior.pt 2>/dev/null | wc -l)
    if [ "${{NP}}" -ne 1 ]; then
        echo "FATAL: ${{NP}} matches for *${{SRC}}_*_prior.pt (need exactly 1)" >&2; exit 1
    fi
    PM=$(basename $(ls ${{CKPTS}}/*${{SRC}}_*_prior.pt))
fi
echo "  prior model <- ${{PM}}"

sed -e "s|{placeholder}|$(basename ${{CK}})|" \\
    -e "s|{prior_placeholder}|${{PM}}|" ${{CONFIG}} > ${{RESOLVED}}
if grep -q '{placeholder}\\|{prior_placeholder}' ${{RESOLVED}}; then
    echo "FATAL: placeholder left in ${{RESOLVED}}" >&2; exit 1
fi

{{ nvidia-smi -L
  scontrol show job ${{SLURM_JOB_ID}}
  echo "nodelist: ${{SLURM_NODELIST}}  host: $(hostname)"
}} > ${{J}}.info 2>&1

stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \\
    --format=csv,nounits -l 10 > ${{J}}_smi.csv &
SMI_PID=$!
smi_epilogue() {{
    kill ${{SMI_PID}} 2>/dev/null
    sacct -j ${{SLURM_JOB_ID}} --format=JobID,State,ExitCode,Elapsed,NodeList,Reason,Comment%64 \\
        > ${{J}}_sacct.txt 2>&1
}}
trap smi_epilogue EXIT TERM

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
    touch ${{CKPTS}}/${{ARM}}.dead
fi
"""


def main(argv):
    dirty = dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted files the arms execute:\n  ' + '\n  '.join(dirty) +
                 '\nBuild from a clean worktree, or pass --allow-dirty for a LOCAL build.')
    arms = build()
    logs = HERE / 'joblogs'
    logs.mkdir(exist_ok=True)
    (logs / '.gitkeep').write_text('SLURM cannot create --output\n', encoding='utf-8')
    for name, cfg in arms.items():
        with (HERE / (name + '.yaml')).open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\n')
        for name in arms:
            f.write('%s\t%s\n' % (name, PARENT))
    with (HERE / 'submit_flk_sep14.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(wall=WALL, last=len(arms) - 1, placeholder=PLACEHOLDER,
                              prior_placeholder=PRIOR_PLACEHOLDER))
    for name, cfg in arms.items():
        eq = _eq(cfg); b = eq.get('balance')
        print('%-20s fracs=%s  guard=%s  source=%s  noise=%s' % (
            name, eq['fracs'], 'bar %.1f tol %.1f rel %.2f replay-bounds %s' % (
                b['bar'], b['ratchet_tol'], b['ratchet_release_tol'], b['bounds']['replay']),
            cfg['buffers']['prior_buffer']['source'], cfg['buffers']['anchor_buffer']['noise_log_range']))


if __name__ == '__main__':
    main(sys.argv[1:])
