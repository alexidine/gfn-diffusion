"""final_sep19 -- the final-protocol battery: phase 2 under RARE ROLLOUTS from the mle_fresh_sep17
best-MLE checkpoints, a 2 x 2 on the two optional pieces (P_B frozen, stored terminal force), on nehu
(UMA, the production target) and mip (ELJ, the family with the local ladder and the dose history).

    python configs/final_sep19/make.py            # base = the COMMITTED mk_dev (git show HEAD:), warns on a dirty tree
    python configs/final_sep19/make.py --families nehu

OWNER (2026-09-19): "we want to get to our final protocol. Must have: rare rollouts. Optional: Pb frozen,
forces." Same shape as the local rr_sep19 ladder: tau 600, holdout 5%, the gated ramp UNPINNED with its
ceiling at replay 0.95 / bwd 0.05, batch pinned, no fwd-branch forces (the forward branch is inert under
rare rollouts: fracs.fwd 0, freeze_policy 1; rollouts are admission events).

ARMS per family (N = fwd_rollout_every, one constant below; rate = the family's N=5 rate x 5/N so the
dose per replay row -- passes x pressure per pass -- is the one the dose_sep16 ladder found safe):
  <fam>       rare rollouts, P_B trainable, no forces           the must-have alone
  <fam>_pb    + P_B frozen at the phase-2 entry (freeze_pb:full on equilibration's on_enter)
  <fam>_f     + stored terminal force on replay rows (replay_loss_coeffs.stored_force_k 1, implied)
  <fam>_pbf   + both

SEED = the mlefr_<fam>_lr2 arm's `_best.pt` (the bwd/mle record; that stage is terminal so no phase1_exit
exists), loaded as a FULL resume exactly as prod_sep12 seeded from mle09: the step and the stage NAME
train_prior come across, this config's train_prior is the stub that exits at the first eval (bwd/mle >
-1e9, patience 1) and fires snapshot_prior; the transition into equilibration fires on_enter for real
(rebuild_prior_by_churn on anchors, bootstrap_z, and freeze_pb on the _pb arms -- the snapshot is taken
AFTER the stub, on the phase-1 P_B, and persists in the checkpoint so a requeued leg does not re-snapshot).
Until that first eval (eval_period steps) the stub trains bwd MLE at the restored cruise scale; the
transition re-enters burn-in at fixed_scale (= burn_in_scale) for 500 steps.

IDENTITY is the seed arm's: prior_path/molecules_path/space_groups/z_primes/energy_function/mlip_path/
checkpoints_dir, the `model` block (dplr_rank 0 -- mk_dev carries 6, and load_full compares the stored
problem_def with no exemption) and energy_config, copied verbatim; `check` asserts get_problem_definition
equality with the seed. Everything else is the committed mk_dev's unconditional_tb with the deltas below.

STORED FORCE on an MLIP: force_chunk_rows stays ABSENT (route-aware default: 8 rows per autograd chunk
on an MLIP, one call on ELJ -- train.py ~9940). n20_sf1 lost on the dose ladder at N=20 / full rate /
trainable P_B; k=1 + frozen P_B is the untested cell.

READ vs the family's own history: fwd/log_Z_learned level and sd, eval_fwd/tb_err, replay/val_gap_nats,
Mean Sample Energy, Replay Frac (where the controller parks), protocol/gr_tripped|gr_fired, and on the _f
arms rewardgrad/force_* plus energy/seconds (the admission-time force call). The cell that reaches the
highest Z with a flat held-out gap is the protocol.

REQUEUE-SAFE, 2-day wall, sbatch as prod_sep12 (own _running.pt wins, seeds from *<src>_*_best.pt
otherwise, .dead sentinel, prior model handed to later legs) plus the W3 guards from mle_fresh_sep17
(monoclinic class walls, the Niggli penalty, prior byte size).
"""
import copy
import importlib.util
import pathlib
import subprocess
import sys
import types
from argparse import Namespace

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                      # energy_sampling/configs
ES = ROOT.parent                        # energy_sampling/
REPO = ES.parent                        # gfn_diffusion/
TAG = 'fin19'
BATTERY = 'final_sep19'
PLACEHOLDER = 'WARM_CHECKPOINT_PLACEHOLDER'
PRIOR_PLACEHOLDER = 'PRIOR_MODEL_PLACEHOLDER'
WALL = '2-00:00:00'
EPOCHS = 1_000_000                      # ABSOLUTE step cap; the seeds sit at ~120k

#: the shape (owner 2026-09-19)
N = 10                                  # fwd_rollout_every; the local rr_sep19 ladder reads N=5 vs N=10 tonight
TAU = 600
REPLAY_MAX = 1_000_000                  # the cap must never bind: B x tau / N = 96k rows at 1600 / 600 / 10
VAL_FRAC, VAL_CAP = 0.05, 1024
BOUNDS = {'bwd': [0.05, 0.9], 'replay': [0.1, 0.95]}
BATCH = 1600
BURN_IN_STEPS = 500
EXCURSION_K = 60.0
#: the N=5 (full-pressure) rate per family: mip from the dose ladder (n5_t600_pb at 1.0), nehu the p02/p12
#: paper rate (0.125, the only rate ever visited there). Scaled by 5/N.
RATE_N5 = {'nehu': 0.125, 'mip': 1.0}
FAMILIES = ['nehu', 'mip']
SEED_ARM = {'nehu': 'mle_fresh_sep17/mlefr_nehu_lr2.yaml', 'mip': 'mle_fresh_sep17/mlefr_mip_lr2.yaml'}
SRC = {'nehu': 'mlefr_nehu_lr2', 'mip': 'mlefr_mip_lr2'}
MLIP = {'nehu': True, 'mip': False}
UMA_EVAL = dict(eval_period=1000, eval_num_samples=2500, figs_period=1000)
ELJ_EVAL = dict(eval_period=500, eval_num_samples=10000, figs_period=1000)
IDENTITY_KEYS = ('prior_path', 'molecules_path', 'test_molecules_path', 'space_groups', 'z_primes',
                 'energy_function', 'mlip_path', 'checkpoints_dir', 'model', 'energy_config')
CELLS = {'': (False, False), '_pb': (True, False), '_f': (False, True), '_pbf': (True, True)}

_spec = importlib.util.spec_from_file_location('w3make', ROOT / 'mle_w3_sep16' / 'make.py')
w3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w3)


def committed_mk_dev():
    out = subprocess.run(['git', 'show', 'HEAD:energy_sampling/configs/mk_dev.yaml'], capture_output=True,
                         text=True, cwd=str(REPO), check=True).stdout
    return yaml.safe_load(out)


def load(rel):
    return yaml.safe_load((ROOT / rel).read_text(encoding='utf-8'))


def _ns(obj):
    if isinstance(obj, dict):
        return Namespace(**{k: _ns(v) for k, v in obj.items()})
    return obj


def raw_problem_def(cfg):
    sys.path.insert(0, str(REPO))
    from energy_sampling.utils import get_problem_definition
    return get_problem_definition(_ns(cfg))


def _eq(cfg):
    return [s for s in cfg['protocols']['unconditional_tb']['stages'] if s['name'] == 'equilibration'][0]


def deltas(cfg, fam, name, pb, force):
    seed = load(SEED_ARM[fam])
    mlip = MLIP[fam]
    for k in IDENTITY_KEYS:
        cfg[k] = copy.deepcopy(seed.get(k))
    # energy_config is the SEED'S BLOCK VERBATIM, including its absences: get_problem_definition hashes
    # the block, so a key mk_dev has since gained (physical_energy_clip: null) that the seed lacks moves
    # the identity and load_full refuses the checkpoint (checked 2026-09-19). Absent = the code default,
    # which is what the seed arm itself runs with on the cluster.
    cfg['run_name'] = name
    cfg['tag'] = TAG

    # -- warm start: FULL resume; the sbatch picks the file ----------------------
    cfg['checkpoint_name'] = PLACEHOLDER
    cfg['load_weights_only'] = False
    cfg['continue_from_checkpoint'] = False
    cfg['warm_start_ignore_problem_keys'] = None
    cfg['prior_model_name'] = PRIOR_PLACEHOLDER
    cfg['epochs'] = EPOCHS
    cfg['archive_period'] = 5000
    cfg['archive_buffers'] = True
    cfg['integrator']['T'] = 100
    cfg['eval_T'] = 100

    # -- protocol: the stub, then equilibration under rare rollouts ---------------
    cfg['protocol'] = 'unconditional_tb'
    stages = cfg['protocols']['unconditional_tb']['stages']
    stub = [s for s in stages if s['name'] == 'train_prior'][0]
    stub.pop('skip_if', None)
    stub['exit'] = [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}]
    stub['on_exit'] = ['snapshot_prior']
    eq = _eq(cfg)
    eq['fwd_rollout_every'] = N
    assert eq['balance']['kind'] == 'gated_ramp' and eq['balance']['pinned'] == {'fwd': 0.0}, name
    eq['balance']['bounds'] = {k: list(v) for k, v in BOUNDS.items()}
    assert float(eq['deactivate_threshold']) <= BOUNDS['bwd'][0] and eq.get('min_fracs') is None, name
    on_enter = list(eq.get('on_enter') or [])
    assert 'rebuild_prior_by_churn' in on_enter, name
    if pb:
        on_enter.append('freeze_pb:full')
    eq['on_enter'] = on_enter
    assert cfg.get('freeze_backward_policy') in (False, None), name   # the freeze is the stage action, not the load-time key
    assert cfg['buffers']['prior_buffer']['source'] == 'anchors', name

    # -- the replay buffer -------------------------------------------------------
    rb = cfg['buffers']['replay_buffer']
    rb['churn_rate'] = 0
    rb['mean_residence_steps'] = TAU
    rb['max_size'] = REPLAY_MAX
    rb['val_frac'] = VAL_FRAC
    rb['val_cap'] = VAL_CAP

    # -- the rate ---------------------------------------------------------------
    scale = RATE_N5[fam] * 5.0 / N
    lc = cfg['lr_control']
    lc['mode'] = 'fixed'
    lc['fixed_scale'] = float(scale)
    lc['burn_in_scale'] = float(scale)
    lc['burn_in_steps'] = BURN_IN_STEPS
    lc['repeat_every'] = 0
    lc['fire_cut_factor'] = 1.0
    lc['hard_failure']['loss_excursion_k'] = EXCURSION_K

    # -- batch pinned ------------------------------------------------------------
    cfg['batch_size'] = BATCH
    cfg['max_batch_size'] = BATCH
    cfg['grow_batch_size'] = False
    cfg['batch_util_target'] = 0.0
    cfg['batch_sizer_retest_steps'] = 0

    # -- route ---------------------------------------------------------------------
    cfg['traj_checkpoint'] = bool(mlip)
    cfg['energy_config']['internal_oom_recovery'] = bool(mlip)
    cfg.update(UMA_EVAL if mlip else ELJ_EVAL)

    # -- forces: NONE on the forward branch; the stored terminal force on replay for the _f cells
    fc = cfg['fwd_loss_coeffs']
    fc['reward_grads'] = 0.0
    fc['traj_grads'] = 0.0
    fc['path_grad_last_k'] = 0
    rc = cfg['replay_loss_coeffs']
    rc['stored_force_k'] = 1 if force else 0
    rc['stored_force_mode'] = 'implied'
    rc['resample_last_k'] = 0
    rc['reward_grads'] = 0.0
    rc['force_chunk_rows'] = None      # route-aware default: 8 rows per autograd chunk on an MLIP, one call on ELJ
    return cfg


def check(cfg, name, fam, pb, force):
    seed = load(SEED_ARM[fam])
    pd_mine, pd_seed = raw_problem_def(cfg), raw_problem_def(seed)
    assert w3.problem_def(cfg) == w3.problem_def(seed), name + ': problem identity moved from the seed'
    assert cfg['prior_path'] == seed['prior_path'] == cfg['molecules_path'], name
    assert cfg['model'] == seed['model'] and cfg['model']['dplr_rank'] == 0, name
    assert cfg['integrator']['T'] == 100 == cfg['eval_T'], name
    assert cfg['checkpoint_name'] == PLACEHOLDER and cfg['prior_model_name'] == PRIOR_PLACEHOLDER, name
    assert cfg['load_weights_only'] is False and cfg['continue_from_checkpoint'] is False, name
    assert cfg['epochs'] >= 500_000, name
    stages = cfg['protocols']['unconditional_tb']['stages']
    assert cfg['protocol'] == 'unconditional_tb' and [s['name'] for s in stages] == ['train_prior', 'equilibration'], name
    stub, eq = stages
    assert stub['exit'] == [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}] and 'skip_if' not in stub, name
    assert eq['fwd_rollout_every'] == N and eq['balance']['bounds'] == BOUNDS, name
    assert ('freeze_pb:full' in eq['on_enter']) is pb, name
    assert eq['fracs'] == {'fwd': 0.0, 'bwd': 0.9, 'replay': 0.1} and eq['loss_coeffs']['fwd']['freeze_policy'] == 1.0, name
    rb = cfg['buffers']['replay_buffer']
    assert rb['mean_residence_steps'] == TAU and rb['max_size'] >= 10 * BATCH * TAU / N and rb['val_frac'] == VAL_FRAC, name
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == lc['burn_in_scale'] == RATE_N5[fam] * 5.0 / N, name
    assert cfg['batch_size'] == cfg['max_batch_size'] == BATCH and cfg['grow_batch_size'] is False, name
    rc, fc = cfg['replay_loss_coeffs'], cfg['fwd_loss_coeffs']
    assert rc['stored_force_k'] == (1 if force else 0) and rc['resample_last_k'] == 0 and rc['force_chunk_rows'] is None, name
    assert fc['reward_grads'] == 0.0 and fc['traj_grads'] == 0.0 and fc['path_grad_last_k'] == 0, name
    if force:
        # the stored legs are taken at the admission temperature; the runtime refuses this under T-conditioning
        assert not pd_mine.get('temp_cond'), name + ': stored force under temperature conditioning'
    assert cfg['traj_checkpoint'] is MLIP[fam] and cfg['energy_config']['internal_oom_recovery'] is MLIP[fam], name
    w3._scan_local_paths(cfg, name)


def load_check(cfg, name):
    """Load the arm the way train.py does (retired keys, invariants, derived values) and parse its
    protocol with the placeholders resolved as a first launch would resolve them."""
    sys.path.insert(0, str(ES))
    sys.path.insert(0, str(REPO))
    import config_invariants
    from energy_sampling.utils import dict2namespace, preflight_config, resolve_derived_config
    from protocol import StageProtocol
    raw = copy.deepcopy(cfg)
    raw['checkpoint_name'] = 'SEED_best.pt'
    raw['prior_model_name'] = None
    errs = config_invariants.errors(copy.deepcopy(raw))
    assert not errs, f'{name}: config_invariants ERRORS: {errs}'
    args = resolve_derived_config(preflight_config(dict2namespace(raw)))
    stages = StageProtocol(types.SimpleNamespace(args=args)).stages
    assert [s.name for s in stages] == ['train_prior', 'equilibration'], name
    return stages


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
#SBATCH --job-name={tag}
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/{battery}/joblogs/%x_%A_%a.out

# {battery}: phase 2 under rare rollouts from the mle_fresh_sep17 best-MLE checkpoints, 2x2 on P_B frozen x stored force.
# Arm = row of INDEX.tsv (line 1 is the header). DO NOT EDIT --array BY HAND: make.py rewrites it.
# Resubmit this same file to continue an arm past the wall.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/{battery}
LOGS=${{ARMS}}/joblogs
CKPTS={ckpts}
DATA={data}
mkdir -p ${{LOGS}}

ROW=$((SLURM_ARRAY_TASK_ID + 2))
ARM=$(awk -F'\\t' -v n=${{ROW}} 'NR==n {{print $1}}' ${{ARMS}}/INDEX.tsv)
SRC=$(awk -F'\\t' -v n=${{ROW}} 'NR==n {{print $4}}' ${{ARMS}}/INDEX.tsv)
PRIOR=$(awk -F'\\t' -v n=${{ROW}} 'NR==n {{print $5}}' ${{ARMS}}/INDEX.tsv)
PRIOR_BYTES=$(awk -F'\\t' -v n=${{ROW}} 'NR==n {{print $6}}' ${{ARMS}}/INDEX.tsv)
if [ -z "${{ARM}}" ]; then echo "no arm at row ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; fi
CONFIG=${{ARMS}}/${{ARM}}.yaml
if [ ! -f "${{CONFIG}}" ]; then echo "missing config ${{CONFIG}}" >&2; exit 1; fi

J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}
RESOLVED=${{J}}.yaml

# A DEAD ARM MUST NOT EAT ITS RESUBMISSION.
if [ -f ${{CKPTS}}/${{ARM}}.dead ]; then
    echo "arm ${{ARM}} aborted UNRECOVERABLE on an earlier leg -- skipping"
    exit 0
fi

# THE WALLS. The W3 / Niggli priors are filtered against the monoclinic class walls and the Niggli
# triclinic penalty; older MXtalTools reads MXT_NIGGLI_TRICLINIC, newer refuses the variable.
SYM=${{PROJECT_ROOT}}/MXtalTools/mxtaltools/common/sym_utils.py
if ! grep -q 'MONO_CLASS' ${{SYM}}; then
    echo "FATAL: ${{SYM}} lacks MONO_CLASS (the monoclinic class walls) -- git pull MXtalTools" >&2; exit 1
fi
if grep -q 'MXT_NIGGLI_TRICLINIC is retired' ${{SYM}}; then
    NIG_EXPORT=""
elif grep -q "os.environ.get('MXT_NIGGLI_TRICLINIC'" ${{SYM}}; then
    NIG_EXPORT="export MXT_NIGGLI_TRICLINIC=1;"
else
    echo "FATAL: ${{SYM}} has no triclinic Niggli penalty -- git pull MXtalTools" >&2; exit 1
fi
HAVE=$(stat -c %s ${{DATA}}/${{PRIOR}} 2>/dev/null || echo 0)
if [ "${{HAVE}}" != "${{PRIOR_BYTES}}" ]; then
    echo "FATAL: ${{DATA}}/${{PRIOR}} is ${{HAVE}} bytes, expected ${{PRIOR_BYTES}}" >&2; exit 1
fi

# RESUME OR SEED. checkpoint_name always wins in train.py: an arm with its own _running.pt continues it
# (full load); only a first launch seeds from the mlefr arm's _best.pt (its bwd/mle record). REFUSES AN
# AMBIGUOUS MATCH.
OWN=$(ls -t ${{CKPTS}}/*${{ARM}}_*_running.pt 2>/dev/null | head -1)
if [ -n "${{OWN}}" ]; then
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  RESUME: $(basename ${{OWN}})"
    CK=${{OWN}}
else
    NM=$(ls ${{CKPTS}}/*${{SRC}}_*_best.pt 2>/dev/null | wc -l)
    if [ "${{NM}}" -ne 1 ]; then
        echo "FATAL: ${{NM}} matches for *${{SRC}}_*_best.pt in ${{CKPTS}} (need exactly 1):" >&2
        ls ${{CKPTS}}/*${{SRC}}_*_best.pt >&2; exit 1
    fi
    CK=$(ls ${{CKPTS}}/*${{SRC}}_*_best.pt)
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  SEED: $(basename ${{CK}}) (mtime $(stat -c %y ${{CK}}))"
fi

# THE PRIOR MODEL: the stub's snapshot_prior writes it on leg 1; a continued leg is handed its own.
OWNPRIOR=$(ls -t ${{CKPTS}}/*${{ARM}}_*_prior.pt 2>/dev/null | head -1)
if [ -n "${{OWNPRIOR}}" ]; then
    PM=$(basename ${{OWNPRIOR}}); echo "  prior model <- ${{PM}}"
else
    PM=null; echo "  prior model <- null (leg 1 writes it)"
fi

sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${{CK}})|" \\
    -e "s|PRIOR_MODEL_PLACEHOLDER|${{PM}}|" ${{CONFIG}} > ${{RESOLVED}}
if grep -q 'WARM_CHECKPOINT_PLACEHOLDER\\|PRIOR_MODEL_PLACEHOLDER' ${{RESOLVED}}; then
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
        ${{NIG_EXPORT}}
        python -c \\"import mxtaltools.common.sym_utils as s; assert getattr(s, 'NIGGLI_TRICLINIC', True), 'Niggli triclinic penalty is OFF'\\" || exit 1
        python -u train.py --config ${{RESOLVED}}
    " 2>&1 | tee ${{J}}.trainlog

if grep -q 'UNRECOVERABLE' ${{J}}.trainlog 2>/dev/null; then
    echo "arm ${{ARM}} exhausted its rewind budget -- sentinel set, resubmissions will skip"
    touch ${{CKPTS}}/${{ARM}}.dead
fi
"""


def build(families):
    base = committed_mk_dev()
    arms = {}
    for fam in families:
        for suffix, (pb, force) in CELLS.items():
            name = f'{TAG}_{fam}{suffix}'
            cfg = deltas(copy.deepcopy(base), fam, name, pb, force)
            check(cfg, name, fam, pb, force)
            load_check(cfg, name)
            arms[name] = (cfg, fam, pb, force)
    return arms


def main(argv):
    families = argv[argv.index('--families') + 1].split(',') if '--families' in argv else FAMILIES
    dirty = w3.dirty_files()
    if dirty:
        print('WARNING: the working tree is dirty (base read from git HEAD; the cluster runs HEAD):\n  ' + '\n  '.join(dirty))
    arms = build(families)
    prior_index = {l.split('\t')[1]: l.split('\t') for l in (ROOT / 'mle_fresh_sep17' / 'INDEX.tsv').read_text(encoding='utf-8').splitlines()[1:]}
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n', encoding='utf-8')
    for name, (cfg, *_r) in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tstart\twarm_src\tprior\tprior_bytes\n')
        for name, (cfg, fam, pb, force) in arms.items():
            row = prior_index[fam]
            assert cfg['prior_path'].rsplit('/', 1)[1] == row[4], name
            f.write(f'{name}\t{fam}\tseed\t{SRC[fam]}\t{row[4]}\t{row[5]}\n')
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(wall=WALL, last=len(arms) - 1, tag=TAG, battery=BATTERY,
                              ckpts=w3.CLUSTER_CKPTS, data=w3.CLUSTER_DATA))
    for i, (name, (cfg, fam, pb, force)) in enumerate(arms.items()):
        print(f"[{i}] {name:<16} N={N} tau={TAU} rate={cfg['lr_control']['fixed_scale']:g} pb_frozen={pb} "
              f"stored_force={force} seed=*{SRC[fam]}_*_best.pt batch={BATCH}")


if __name__ == '__main__':
    main(sys.argv[1:])
