"""prod_sep12 -- phase 2 on the four mle09 families, seeded from each arm's
best-MLE checkpoint, on the CURRENT DEFAULT (mk_dev's unconditional_tb with
rare forward rollouts).

    python configs/prod_sep12/make.py            # refuses on a dirty tree
    python configs/prod_sep12/make.py --allow-dirty

WHAT THIS IS. mle_sep09 carried phase 1 (bwd MLE on the rebuilt priors) far past
the prod_t100 exits. These four arms take those weights into equilibration under
the rare-rollout scheme the owner chose on 2026-09-12: the forward branch (the
only branch that calls the energy function) runs on 1 step in N=20, bwd and
replay train between, log Z is pinned by the absorber fill at each rollout, and
the replay buffer stores the whole rollout under a hazard clock. ONE arm per
family, at the rate prod_sep02 gave its paper slot: mip 1.0, neh 0.5,
mipu 0.0625, nehu 0.125.

THE BASE IS mk_dev.yaml AS IT IS ON DISK, by owner instruction, and that is a
departure from the rule that generators read the COMMITTED mk_dev. The reason
the rule exists still holds: the cluster runs committed code, and a key the
committed code does not read runs the arm silently as a control. So `main`
REFUSES when mk_dev.yaml or any module these arms execute is dirty, unless
--allow-dirty is passed -- and an --allow-dirty build is for local validation,
never for the cluster. Commit, regenerate clean, then push.

THE SEED IS `_best.pt`, NOT A phase1_exit. mle09's stage is terminal (no `exit`,
so on_exit never runs). For a train_mode `bwd` stage `_best` is the bwd/mle
record hardlinked off `running` at the 50-step cadence -- the best-MLE sample.
Loaded as a FULL resume (load_full): the step (~112k on ELJ) and the stage NAME
`train_prior` come across, and this protocol's `train_prior` is the stub that
exits at the first eval (bwd/mle > -1e9, patience 1), fires snapshot_prior --
which `buffers.prior_buffer.source: prior_model` needs -- and advances. The
sidecar resolves to the seed run's ROLLING buffers; equilibration's on_enter
`rebuild_prior_by_churn` replaces the prior buffer anyway. `epochs` is an
ABSOLUTE cap and is set unreachable; the wall ends a leg.

IDENTITY IS TAKEN FROM THE mle09 ARM, NOT REBUILT. load_full compares the stored
problem_def with no exemption, and mip/mipu/nehu hash the rebuilt prior's
filename. The family overlay copies prior_path/molecules_path/space_groups/
energy_function/mlip_path from the mle09 YAML, and `check` asserts the resulting
problem_def equals the seed's through get_problem_definition itself. Dead latent
rows are architectural and cannot be reconfigured on resume; HEAD's gfn.py
already carries them with the same resolver, so the mle09 checkpoints match.

THE CADENCE AND THE BUFFER, the two things the owner set:
  fwd_rollout_every 20      the backstop period; the occupancy trigger (2.0
                            batches, inherited) fires rollouts every step while
                            the buffer is below that, i.e. a built-in warm-up.
  mean_residence_steps 120  tau = 6N. Occupancy under store-all is B*tau/N, so
                            6 batches (24k rows at the ELJ ceiling 4000, 9.6k at
                            UMA 1600). Each rollout replaces 1/6 of the buffer;
                            v0 asked for <= 1/5. The dose analysis says
                            occupancy cancels out of exposure, so tau is a
                            memory/lag choice, not a fit lever.
  max_size 250000           the cap does NOT bind: >= 10x the equilibrium at the
                            batch ceiling, asserted. Occupancy is set by tau.
  churn_rate 0              store-all at the LIVE batch (inherited).

BATCH: the mle09 occupancy policy on every arm -- grow to 4000 at util 0.65,
2 s sampler. UMA enters at 1600 (the prod_sep02 operating point), ELJ at 1000.
Under rare rollouts the UMA step is cheaper and occupancy falls, and the sizer
is what buys the margin back over the ~54% cancellation line.

EVAL IS THE FIXED ENERGY FLOOR under rare rollouts. Training calls the MLIP on
B/N rows per step (~80 at UMA 1600); an eval at period 250 with 10000 samples
would call it 40 per step on top -- half the budget. UMA keeps prod_sep02's
1000/2500 (2.5 per step). ELJ energy is cheap and keeps 500/10000.

WHAT ELSE DIFFERS FROM mk_dev, and only this: the fixed rate (mode fixed is
already the default; burn_in 500 at the operating rate, loss_excursion_k 60 --
the values the p02 paper arms ran 7 days on; mk_dev's 10 is a bracket-mode
setting), traj_checkpoint on the MLIP arms only, internal_oom_recovery on the
MLIP arms, archive_buffers on. Everything else in `equilibration` -- the gated
ramp, its bars, the triggers, the fill -- is mk_dev's, by owner instruction.

REQUEUE-SAFE. 2-day wall; the sbatch resumes an arm's own `_running.pt` and its
own `*_prior.pt` when they exist, seeds from `*<src>_*_best.pt` otherwise, and
honours the `.dead` sentinel. Resubmit the same sbatch to continue.
"""
import copy
import pathlib
import subprocess
import sys
from argparse import Namespace

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ES = ROOT.parent                      # energy_sampling/
MK_DEV = ROOT / 'mk_dev.yaml'
PLACEHOLDER = 'WARM_CHECKPOINT_PLACEHOLDER'
PRIOR_PLACEHOLDER = 'PRIOR_MODEL_PLACEHOLDER'
TAG = 'p12'
WALL = '2-00:00:00'
#: ABSOLUTE step cap, not a length -- the ELJ seeds sit at ~112k already.
EPOCHS = 1_000_000

#: owner, 2026-09-12
N = 20
TAU_OVER_N = 6
TAU = N * TAU_OVER_N
REPLAY_MAX = 250_000
UTIL_TARGET = 0.65
MAX_BATCH = 4000
EXCURSION_K = 60.0

#: the modules a training step executes; dirty = the cluster cannot run this build
EXECUTED = ('configs/mk_dev.yaml', 'train.py', 'protocol.py', 'buffer.py',
            'gflownet_losses.py', 'checkpointing.py', 'utils.py', 'config_invariants.py',
            'models/gfn.py', 'models/set_policy.py', 'energies/molecular_crystal.py',
            'lr_larder.py', 'gpu_guard.py')

FAM = {
    'mip':  dict(seed='mle_sep09/mle09_mip_lr4p0.yaml',  src='mle09_mip_lr4p0',  scale=1.0,    mlip=False, batch=1000),
    'neh':  dict(seed='mle_sep09/mle09_neh_lr4p0.yaml',  src='mle09_neh_lr4p0',  scale=0.5,    mlip=False, batch=1000),
    'mipu': dict(seed='mle_sep09/mle09_mipu_lr4p0.yaml', src='mle09_mipu_lr4p0', scale=0.0625, mlip=True,  batch=1600),
    'nehu': dict(seed='mle_sep09/mle09_nehu_lr4p0.yaml', src='mle09_nehu_lr4p0', scale=0.125,  mlip=True,  batch=1600),
}
#: what the family overlay copies from the mle09 arm: the problem identity and
#: the paths that carry it. Nothing else.
IDENTITY_KEYS = ('prior_path', 'molecules_path', 'test_molecules_path', 'space_groups',
                 'energy_function', 'mlip_path', 'checkpoints_dir')
#: the p02/mle09 UMA eval budget; ELJ keeps mle09's 500/10000
UMA_EVAL = dict(eval_period=1000, eval_num_samples=2500, figs_period=1000)
ELJ_EVAL = dict(eval_period=500, eval_num_samples=10000, figs_period=1000)


def tag_for(scale):
    return 'lr' + ('%g' % scale).replace('.', 'p')


def load(rel):
    return yaml.safe_load((ROOT / rel).read_text(encoding='utf-8'))


def _ns(obj):
    if isinstance(obj, dict):
        return Namespace(**{k: _ns(v) for k, v in obj.items()})
    return obj


def problem_def(cfg):
    """The identity load_full will compare against the checkpoint's stored one."""
    sys.path.insert(0, str(ES.parent))
    from energy_sampling.utils import get_problem_definition, normalize_problem_def
    return normalize_problem_def(get_problem_definition(_ns(cfg)))


def dirty_files():
    out = subprocess.run(['git', 'status', '--porcelain', '--'] + [str(ES / p) for p in EXECUTED],
                         capture_output=True, text=True, cwd=str(ES), check=True).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def deltas(cfg, fam, name):
    spec = FAM[fam]
    seed = load(spec['seed'])

    # -- identity + location: the mle09 arm's, verbatim ------------------------
    for k in IDENTITY_KEYS:
        cfg[k] = copy.deepcopy(seed.get(k))
    cfg['run_name'] = name
    cfg['tag'] = TAG

    # -- warm start: FULL resume on both roles; the sbatch picks the file ------
    cfg['checkpoint_name'] = PLACEHOLDER
    cfg['load_weights_only'] = False
    cfg['continue_from_checkpoint'] = False
    cfg['warm_start_ignore_problem_keys'] = None     # weights-only path only; inert here
    cfg['prior_model_name'] = PRIOR_PLACEHOLDER
    cfg['epochs'] = EPOCHS
    cfg['archive_period'] = 5000
    cfg['archive_buffers'] = True

    # -- the ship length --------------------------------------------------------
    cfg['integrator']['T'] = 100
    cfg['eval_T'] = 100

    # -- protocol: mk_dev's unconditional_tb, train_prior turned into the stub --
    cfg['protocol'] = 'unconditional_tb'
    stages = cfg['protocols']['unconditional_tb']['stages']
    stub = [s for s in stages if s['name'] == 'train_prior'][0]
    stub.pop('skip_if', None)          # the checkpoint re-enters this stage by name
    stub['exit'] = [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}]
    stub['on_exit'] = ['snapshot_prior']   # prior_buffer.source prior_model needs it
    eq = [s for s in stages if s['name'] == 'equilibration'][0]
    eq['fwd_rollout_every'] = N

    # -- the replay buffer: tau sets occupancy, the cap never binds ------------
    rb = cfg['buffers']['replay_buffer']
    rb['churn_rate'] = 0
    rb['mean_residence_steps'] = TAU
    rb['max_size'] = REPLAY_MAX

    # -- the fixed rate ----------------------------------------------------------
    lc = cfg['lr_control']
    lc['mode'] = 'fixed'
    lc['fixed_scale'] = float(spec['scale'])
    lc['burn_in_scale'] = float(spec['scale'])   # == fixed: a rewind restores lr_ctrl.scale
    lc['burn_in_steps'] = 500
    lc['repeat_every'] = 0
    lc['fire_cut_factor'] = 1.0
    lc['hard_failure']['loss_excursion_k'] = EXCURSION_K

    # -- batch: the mle09 occupancy policy ------------------------------------
    cfg['batch_size'] = spec['batch']
    cfg['grow_batch_size'] = True
    cfg['max_batch_size'] = MAX_BATCH
    cfg['batch_util_target'] = UTIL_TARGET
    cfg['gpu_util_sample_period_s'] = 2
    cfg['gpu_util_policy_window_s'] = 7200

    # -- route-specific ----------------------------------------------------------
    cfg['traj_checkpoint'] = bool(spec['mlip'])
    cfg['energy_config']['internal_oom_recovery'] = bool(spec['mlip'])
    cfg.update(UMA_EVAL if spec['mlip'] else ELJ_EVAL)
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
    assert cfg['integrator']['T'] == 100 == cfg['eval_T'], name

    # warm start
    assert cfg['checkpoint_name'] == PLACEHOLDER and cfg['prior_model_name'] == PRIOR_PLACEHOLDER, name
    assert cfg['load_weights_only'] is False and cfg['continue_from_checkpoint'] is False, name
    assert cfg['epochs'] >= 500_000, name + ': epochs is an absolute cap and the seed is at ~112k'
    assert cfg['model']['hold_dead_latent_rows'] is True, name + ': the mle09 checkpoints hold dead rows'

    # protocol
    stages = cfg['protocols']['unconditional_tb']['stages']
    assert cfg['protocol'] == 'unconditional_tb' and [s['name'] for s in stages] == ['train_prior', 'equilibration'], name
    stub, eq = stages
    assert stub['exit'] == [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}] and 'skip_if' not in stub, name
    assert stub['on_exit'] == ['snapshot_prior'], name
    assert cfg['buffers']['prior_buffer']['source'] == 'prior_model', name + ': the stub snapshot exists for this'
    assert eq['train_mode'] == 'fused' and 'exit' not in eq, name + ': equilibration is terminal'
    assert eq['fwd_rollout_every'] == N, name
    assert eq['flags']['z_calibration'] is False, name + ': the servo would call the MLIP on every skipped step'
    assert float(cfg['z_calibration']['fill_threshold']) > 0, name + ': the fill is the only Z pin'
    assert cfg['z_calibration']['fill_mode'] == 'absorb', name
    assert eq['balance']['kind'] == 'gated_ramp' and eq['balance']['pinned'] == {'fwd': eq['fracs']['fwd']}, name
    assert (eq['fwd_rollout_triggers'].get('ess_min') or 0) == 0, name + ': ess_min latches at max rate'
    for s in stages:
        sensor = s.get('hot_lr_sensor')
        if isinstance(sensor, dict):
            assert sensor.get('action', 'report') == 'report', name + ': hot_lr can still fire'

    # replay buffer
    rb = cfg['buffers']['replay_buffer']
    assert rb['churn_rate'] == 0 and rb['mean_residence_steps'] == TAU, name
    equilibrium = MAX_BATCH * TAU / N
    assert rb['max_size'] >= 10 * equilibrium, \
        name + ': replay cap %d would bind against a %d-row equilibrium' % (rb['max_size'], equilibrium)
    assert rb['backstop_mult'] > 0, name
    ab = cfg['buffers']['anchor_buffer']
    assert ab['frozen'] is True and ab['online_loss_flow'] is False, name

    # rate
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == lc['burn_in_scale'] == spec['scale'], name
    assert lc['fire_cut_factor'] == 1.0 and lc['repeat_every'] == 0, name
    assert lc['hard_failure']['loss_excursion_k'] == EXCURSION_K, name

    # batch
    assert cfg['grow_batch_size'] is True and cfg['max_batch_size'] == MAX_BATCH, name
    assert cfg['batch_util_target'] == UTIL_TARGET and cfg['batch_size'] == spec['batch'], name
    assert cfg['gpu_util_sample_period_s'] == 2, name

    # route
    assert cfg['traj_checkpoint'] is spec['mlip'], name + ': traj_checkpoint rides the MLIPs only'
    assert cfg['energy_config']['internal_oom_recovery'] is spec['mlip'], name
    assert cfg['energy_config'].get('reward_range'), name + ': soft clip NOT armed'
    assert cfg['eval_period'] == (1000 if spec['mlip'] else 500), name
    assert cfg['compile_policy'] is False, name
    _scan_local_paths(cfg, name)


def build():
    out = {}
    for fam, spec in FAM.items():
        name = TAG + '_' + fam + '_' + tag_for(spec['scale'])
        cfg = deltas(yaml.safe_load(MK_DEV.read_text(encoding='utf-8')), fam, name)
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

# prod_sep12: phase 2 from the mle09 best-MLE checkpoints, rare rollouts N=20.
# Arm = row of INDEX.tsv (line 1 is the header). DO NOT EDIT --array BY HAND:
# make.py rewrites it. Resubmit this same file to continue an arm past the wall.
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


def main(argv):
    dirty = dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: the cluster runs committed code, and these arms read keys from '
                 'uncommitted files:\n  ' + '\n  '.join(dirty) +
                 '\nCommit them and regenerate, or pass --allow-dirty for a LOCAL build.')
    if dirty:
        print('WARNING: --allow-dirty build on %d uncommitted files; NOT for the cluster' % len(dirty))

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
        rb = cfg['buffers']['replay_buffer']
        print('%-18s seed=%-16s scale=%-7g N=%d tau=%d cap=%d batch=%d->%d@%.2f traj_ckpt=%s eval=%d/%d'
              % (name, FAM[fam]['src'], FAM[fam]['scale'], N, rb['mean_residence_steps'],
                 rb['max_size'], cfg['batch_size'], cfg['max_batch_size'], cfg['batch_util_target'],
                 cfg['traj_checkpoint'], cfg['eval_period'], cfg['eval_num_samples']))


if __name__ == '__main__':
    main(sys.argv[1:])
