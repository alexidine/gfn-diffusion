"""mle_w3_sep16 -- phase 1 (bwd MLE) on the W3 P2_1/c priors, seeded from the mle09
best-MLE checkpoints of the same three problems.

    python configs/mle_w3_sep16/make.py                # refuses on a dirty tree
    python configs/mle_w3_sep16/make.py --allow-dirty  # local validation only

Needs the project venv (it imports energy_sampling.utils for the problem identity).

ONE ARM PER PROBLEM: the mle09 arm's configuration on today's mk_dev, with the prior
swapped for its `_w3` file.

    neh   nehzor   sg14 Z'=1  elj
    nehu  nehzor   sg14 Z'=1  uma   (f047, 200k)
    acr   acridine sg14 Z'=1  mace

THE PRIORS. `<name>_w3.pt` holds the rows of `equalized_prior` whose stored cell has
zero penalty under the three-wall P2_1/c domain; `prior` and `thermal_scaling_factor`
are the original file's. The sbatch refuses a cluster copy whose byte size differs from
PRIOR_BYTES, which is what an unfinished upload looks like.

THE WALLS ARE CODE. The penalty that matches these priors is mxtaltools WALLS_COMMIT.
The generator refuses a local mxtaltools HEAD without it, and the sbatch refuses a
cluster MXtalTools checkout whose common/sym_utils.py lacks WALLS_SYMBOL.

THE SEED is `*<src>_*_best.pt`: the bwd/mle record of mle09's terminal MLE stage (mle09
wrote no phase1_exit). It loads WEIGHTS-ONLY -- optimizers, buffers, metrics and the step
count start fresh, so every buffer is built from the W3 prior under the new walls.
`warm_start_ignore_problem_keys: [prior_path]` waives the one identity key that moved, and
`check` asserts the rest of the problem definition equals the seed arm's. An energy_config
key newer than the seed arm is dropped when mk_dev holds it at the committed
MolecularCrystal default (the same behaviour, and the seed's identity), and refused
otherwise.

NO EXIT. The stage omits `exit`, which makes it terminal (`exit: []` fires on the first
eval). Archives every 5000 steps, with buffers, are the warm-startable artifacts.

REQUEUE-SAFE. Resubmitting the same sbatch continues an arm's own `_running.pt` (auto-
resume, a full load: optimizer state and step carry); an arm that aborted UNRECOVERABLE
leaves a `.dead` sentinel and later submissions skip it.
"""
import ast
import copy
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
MXT = REPO.parent / 'mxtaltools'
MK_DEV = ROOT / 'mk_dev.yaml'

TAG = 'mlew3'
BATTERY = 'mle_w3_sep16'
PROTOCOL = 'mle_w3'
WALL = '2-00:00:00'
#: ABSOLUTE step cap (trange(init_step, epochs + 1)); the wall ends a leg.
EPOCHS = 1_000_000
SHIP_T = 100
SCALE = 4.0
BURN_IN_STEPS = 500
BURN_IN_SCALE = 0.05
EXCURSION_K = 40.0
UTIL_TARGET = 0.65
ENTRY_BATCH = 1000
MAX_BATCH = 4000
CUDA_MEMORY_FRACTION = 0.97

CK_PLACEHOLDER = 'WARM_CHECKPOINT_PLACEHOLDER'
CONT_PLACEHOLDER = 'CONTINUE_PLACEHOLDER'

CLUSTER_DATA = '/scratch/mk8347/data/crystal_datasets/conditional/priors'
CLUSTER_CKPTS = '/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints'
LOCAL_DATA = pathlib.Path(r'D:\crystal_datasets\conditional\priors')

WALLS_COMMIT = 'a71efd6b'
WALLS_SYMBOL = 'MONO_CLASS'

FAM = {
    'neh':  dict(seed='mle_sep09/mle09_neh_lr4p0.yaml',  src='mle09_neh_lr4p0',  prior_bytes=200_300_990),
    'nehu': dict(seed='mle_sep09/mle09_nehu_lr4p0.yaml', src='mle09_nehu_lr4p0', prior_bytes=252_737_846),
    'acr':  dict(seed='mle_sep09/mle09_acr_lr4p0.yaml',  src='mle09_acr_lr4p0',  prior_bytes=270_223_697),
}
#: copied verbatim from the seed arm: the problem identity and the paths that carry it
IDENTITY_KEYS = ('space_groups', 'z_primes', 'energy_function', 'mlip_path', 'checkpoints_dir')

_REQUIRED = object()
_UNPARSEABLE = object()


def load(rel):
    return yaml.safe_load((ROOT / rel).read_text(encoding='utf-8'))


def arm_name(fam):
    return f"{TAG}_{fam}_lr{('%g' % SCALE).replace('.', 'p')}"


def w3_path(path):
    assert path.startswith(CLUSTER_DATA + '/') and path.endswith('.pt') and not path.endswith('_w3.pt'), path
    return path[:-len('.pt')] + '_w3.pt'


# ----------------------------------------------------------------------------- git state

def _git(args, cwd):
    return subprocess.run(['git'] + args, capture_output=True, text=True, cwd=str(cwd), check=True).stdout


def dirty_files():
    """Uncommitted state the cluster would not have: mk_dev and every module outside
    configs/, tests/ and SCRATCH/ (modified or untracked)."""
    out = _git(['status', '--porcelain', '--untracked-files=all', '--', '.'], ES)
    dirty = []
    for line in out.splitlines():
        path = line[3:].strip().strip('"')
        rel = path[len('energy_sampling/'):] if path.startswith('energy_sampling/') else path
        if rel == 'configs/mk_dev.yaml' or (rel.endswith('.py') and not rel.startswith(('configs/', 'tests/', 'SCRATCH/'))):
            dirty.append(path)
    return dirty


def assert_walls_committed():
    rc = subprocess.run(['git', 'merge-base', '--is-ancestor', WALLS_COMMIT, 'HEAD'], cwd=str(MXT)).returncode
    if rc != 0:
        sys.exit(f'REFUSING: mxtaltools HEAD does not contain {WALLS_COMMIT} (the monoclinic class walls); '
                 f'the W3 priors are filtered against that penalty.')
    committed = _git(['show', 'HEAD:mxtaltools/common/sym_utils.py'], MXT)
    assert WALLS_SYMBOL in committed, f'{WALLS_SYMBOL} not in the committed sym_utils.py -- the sbatch guard would never pass'


def mxt_dirty():
    out = _git(['status', '--porcelain', '--', 'mxtaltools'], MXT)
    return [line[3:] for line in out.splitlines() if line.strip()]


# ----------------------------------------------------------------------------- identity

def committed_crystal_init():
    """MolecularCrystal.__init__ as COMMITTED, {arg: default}; energy_config is splatted into it."""
    src = _git(['show', 'HEAD:energy_sampling/energies/molecular_crystal.py'], REPO)
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MolecularCrystal')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    pos = init.args.args[1:]
    defaults = [_REQUIRED] * (len(pos) - len(init.args.defaults)) + list(init.args.defaults)
    pairs = list(zip(pos, defaults)) + list(zip(init.args.kwonlyargs, init.args.kw_defaults))
    out = {}
    for arg, node in pairs:
        if node is _REQUIRED or node is None:
            out[arg.arg] = _REQUIRED
            continue
        try:
            out[arg.arg] = ast.literal_eval(node)
        except ValueError:
            out[arg.arg] = _UNPARSEABLE
    return out


def _ns(obj):
    if isinstance(obj, dict):
        return Namespace(**{k: _ns(v) for k, v in obj.items()})
    return obj


def problem_def(cfg):
    """The identity a checkpoint load compares, built by the trainer's own function."""
    sys.path.insert(0, str(REPO))
    from energy_sampling.utils import get_problem_definition, normalize_problem_def
    return normalize_problem_def(get_problem_definition(_ns(cfg)))


def align_energy_identity(cfg, seed):
    """Drop the energy_config keys the seed arm never had, provided mk_dev holds each at
    the committed MolecularCrystal default; refuse any other."""
    sys.path.insert(0, str(REPO))
    from energy_sampling.utils import _NON_IDENTITY_ENERGY_CONFIG_KEYS
    init = committed_crystal_init()
    dropped = []
    for key in list(cfg['energy_config']):
        if key in seed['energy_config'] or key in _NON_IDENTITY_ENERGY_CONFIG_KEYS:
            continue
        value = cfg['energy_config'][key]
        default = init.get(key, _REQUIRED)
        if default is _REQUIRED or default is _UNPARSEABLE or default != value:
            raise AssertionError(
                f'energy_config.{key} = {value!r} is newer than the mle09 seed and is not the '
                f'MolecularCrystal default ({"none" if default in (_REQUIRED, _UNPARSEABLE) else repr(default)}): '
                f'keeping it moves the problem identity off the seed, dropping it changes the run')
        cfg['energy_config'].pop(key)
        dropped.append(key)
    return dropped


# ----------------------------------------------------------------------------- arms

def deltas(cfg, fam, name):
    spec = FAM[fam]
    seed = load(spec['seed'])

    # -- identity + location ---------------------------------------------------
    for k in IDENTITY_KEYS:
        cfg[k] = copy.deepcopy(seed.get(k))
    prior = w3_path(seed['prior_path'])
    cfg['prior_path'] = prior
    cfg['molecules_path'] = prior          # unconditional: the condition set IS the prior file
    cfg['test_molecules_path'] = None
    cfg['run_name'] = name
    cfg['tag'] = TAG
    cfg['cuda_memory_fraction'] = CUDA_MEMORY_FRACTION

    # -- warm start: weights only; the sbatch substitutes both placeholders -------
    cfg['checkpoint_name'] = CK_PLACEHOLDER
    cfg['continue_from_checkpoint'] = CONT_PLACEHOLDER
    cfg['load_weights_only'] = True
    cfg['warm_start_ignore_problem_keys'] = ['prior_path']
    cfg['prior_model_name'] = None
    dropped = align_energy_identity(cfg, seed)

    # -- the ship length ---------------------------------------------------------
    cfg['integrator']['T'] = SHIP_T
    cfg['eval_T'] = SHIP_T
    cfg['traj_checkpoint'] = False

    # -- run shape -----------------------------------------------------------------
    cfg['epochs'] = EPOCHS
    cfg['eval_period'] = 500
    cfg['figs_period'] = 1000
    cfg['archive_period'] = 5000
    cfg['archive_buffers'] = True

    # -- batch: grow to an occupancy target ----------------------------------------
    cfg['batch_size'] = ENTRY_BATCH
    cfg['grow_batch_size'] = True
    cfg['max_batch_size'] = MAX_BATCH
    cfg['batch_util_target'] = UTIL_TARGET
    cfg['gpu_util_sample_period_s'] = 2
    cfg['gpu_util_policy_window_s'] = 7200

    # -- the fixed rate --------------------------------------------------------------
    lc = cfg['lr_control']
    lc['mode'] = 'fixed'
    lc['fixed_scale'] = SCALE
    lc['burn_in_steps'] = BURN_IN_STEPS
    lc['burn_in_scale'] = BURN_IN_SCALE
    lc['repeat_every'] = 0
    lc['fire_cut_factor'] = 1.0
    lc['hard_failure']['loss_excursion_k'] = EXCURSION_K

    # -- protocol: mk_dev's train_prior, made terminal, hot sensor armed -----------
    stages = cfg['protocols']['unconditional_tb']['stages']
    stage = copy.deepcopy(next(s for s in stages if s['name'] == 'train_prior'))
    for k in ('exit', 'skip_if', 'on_exit'):
        stage.pop(k, None)
    stage['hot_lr_sensor'] = dict(stage['hot_lr_sensor'], action='fire')
    cfg['protocol'] = PROTOCOL
    cfg['protocols'][PROTOCOL] = {'stages': [stage]}
    return cfg, dropped


def _scan_local_paths(node, name, trail='cfg'):
    if isinstance(node, dict):
        for k, v in node.items():
            _scan_local_paths(v, name, f'{trail}.{k}')
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _scan_local_paths(v, name, f'{trail}[{i}]')
    elif isinstance(node, str):
        low = node.lower()
        if low.startswith(('c:', 'd:')) or '\\' in node:
            raise AssertionError(f'{name}: local path at {trail}: {node!r}')


def check(cfg, name, fam):
    """Every battery property, re-asserted on the FINISHED dict."""
    spec = FAM[fam]
    seed = load(spec['seed'])

    # identity: the seed's, bar the one waived key
    mine, theirs = problem_def(cfg), problem_def(seed)
    moved = sorted(k for k in set(mine) | set(theirs) if k != 'prior_path' and mine.get(k) != theirs.get(k))
    assert not moved, f'{name}: problem identity moved from the mle09 seed on {moved}'
    assert cfg['prior_path'] == cfg['molecules_path'] == w3_path(seed['prior_path']), name
    assert cfg['test_molecules_path'] is None, name
    assert cfg['warm_start_ignore_problem_keys'] == ['prior_path'], name
    init = committed_crystal_init()
    unknown = sorted(k for k in cfg['energy_config'] if k not in init)
    assert not unknown, f'{name}: energy_config keys the committed MolecularCrystal does not take: {unknown}'
    assert cfg['model']['hold_dead_latent_rows'] is True, f'{name}: the mle09 checkpoints hold dead rows'

    # warm start
    assert cfg['checkpoint_name'] == CK_PLACEHOLDER and cfg['continue_from_checkpoint'] == CONT_PLACEHOLDER, name
    assert cfg['load_weights_only'] is True and cfg['prior_model_name'] is None, name

    # length, shape
    assert cfg['integrator']['T'] == SHIP_T == cfg['eval_T'], name
    assert cfg['traj_checkpoint'] is False and cfg['compile_policy'] is False, name
    assert cfg['epochs'] == EPOCHS, name
    assert cfg['archive_period'] == 5000 and cfg['archive_buffers'] is True, name

    # protocol: one terminal MLE stage
    stages = cfg['protocols'][PROTOCOL]['stages']
    assert cfg['protocol'] == PROTOCOL and len(stages) == 1, name
    st = stages[0]
    assert st['name'] == 'train_prior' and st['train_mode'] == 'bwd' and st['bwd_sampling_mode'] == 'dataset', name
    assert not ({'exit', 'skip_if', 'on_exit'} & set(st)), f'{name}: the stage must be terminal and unskippable'
    terms = [k for k, v in st['loss_coeffs']['bwd'].items() if k != 'repeats' and float(v) > 0]
    assert terms == ['mle'], f'{name}: phase 1 is MLE-only, got {terms}'
    assert st['hot_lr_sensor']['channel'] == 'bwd/mle' and st['hot_lr_sensor']['action'] == 'fire', name

    # batch
    assert cfg['batch_size'] == ENTRY_BATCH and cfg['max_batch_size'] == MAX_BATCH, name
    assert cfg['grow_batch_size'] is True and cfg['batch_util_target'] == UTIL_TARGET, name
    assert cfg['gpu_util_sample_period_s'] == 2, name

    # rate
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == SCALE and lc['repeat_every'] == 0, name
    assert lc['burn_in_steps'] == BURN_IN_STEPS and lc['burn_in_scale'] == BURN_IN_SCALE, name
    assert lc['fire_cut_factor'] == 1.0 and lc['hard_failure']['loss_excursion_k'] == EXCURSION_K, name
    assert lc['hard_failure']['loss_abs'] >= 1e6 and lc['hard_failure']['grad_abs'] >= 1e6, name

    _scan_local_paths(cfg, name)


def load_check(cfg, name):
    """Load the arm the way train.py does -- retired keys, state version, invariants,
    derived values -- and parse its protocol, with the placeholders resolved as a first
    launch would resolve them."""
    sys.path.insert(0, str(ES))
    sys.path.insert(0, str(REPO))
    import config_invariants
    from energy_sampling.utils import dict2namespace, preflight_config, resolve_derived_config
    from protocol import StageProtocol

    raw = copy.deepcopy(cfg)
    raw['checkpoint_name'] = 'SEED_best.pt'
    raw['continue_from_checkpoint'] = False
    errs = config_invariants.errors(copy.deepcopy(raw))
    assert not errs, f'{name}: config_invariants ERRORS: {errs}'
    args = resolve_derived_config(preflight_config(dict2namespace(raw)))
    stages = StageProtocol(types.SimpleNamespace(args=args)).stages
    assert [s.name for s in stages] == ['train_prior'], name
    return stages


def build():
    base = yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))
    out = {}
    for fam in FAM:
        name = arm_name(fam)
        cfg, dropped = deltas(copy.deepcopy(base), fam, name)
        check(cfg, name, fam)
        out[name] = (cfg, fam, dropped)
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
#SBATCH --job-name={tag}
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/{battery}/joblogs/%x_%A_%a.out

# {battery}: phase-1 MLE on the W3 P2_1/c priors, weights-only from the mle09 best-MLE checkpoints.
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

# A DEAD ARM MUST NOT EAT ITS RESUBMISSION: total_reloads is not checkpointed.
if [ -f ${{CKPTS}}/${{ARM}}.dead ]; then
    echo "arm ${{ARM}} aborted UNRECOVERABLE on an earlier leg -- skipping"
    exit 0
fi

# THE PENALTY THESE PRIORS WERE FILTERED AGAINST lives in MXtalTools ({walls_commit}).
if ! grep -q '{walls_symbol}' ${{PROJECT_ROOT}}/MXtalTools/mxtaltools/common/sym_utils.py; then
    echo "FATAL: MXtalTools checkout lacks {walls_symbol} (commit {walls_commit}, the monoclinic class walls) -- git pull MXtalTools" >&2
    exit 1
fi

# THE PRIOR MUST BE THE WHOLE FILE: a partial upload has the right name and the wrong size.
HAVE=$(stat -c %s ${{DATA}}/${{PRIOR}} 2>/dev/null || echo 0)
if [ "${{HAVE}}" != "${{PRIOR_BYTES}}" ]; then
    echo "FATAL: ${{DATA}}/${{PRIOR}} is ${{HAVE}} bytes, expected ${{PRIOR_BYTES}} -- upload unfinished or a different file" >&2
    exit 1
fi

# RESUME OR SEED. train.py's loader is `if checkpoint_name ... elif continue_from_checkpoint`,
# so checkpoint_name ALWAYS wins. An arm with its own _running.pt blanks it and auto-resumes
# (full load: optimizer state and step carry; load_weights_only is read only on the
# checkpoint_name branch). A first launch seeds weights-only from the mle09 arm's _best.pt,
# and REFUSES an absent or ambiguous match.
OWN=$(ls -t ${{CKPTS}}/*${{ARM}}_*_running.pt 2>/dev/null | head -1)
if [ -n "${{OWN}}" ]; then
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  RESUME: $(basename ${{OWN}})"
    sed -e "s|{ck_ph}|null|" -e "s|{cont_ph}|true|" ${{CONFIG}} > ${{RESOLVED}}
else
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
    sed -e "s|{ck_ph}|$(basename ${{CK}})|" -e "s|{cont_ph}|false|" ${{CONFIG}} > ${{RESOLVED}}
fi
if grep -q '{ck_ph}\\|{cont_ph}' ${{RESOLVED}}; then
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
    assert_walls_committed()
    dirty = dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: the cluster runs committed code, and these are uncommitted:\n  '
                 + '\n  '.join(dirty) + '\nCommit them and regenerate, or pass --allow-dirty for a LOCAL build.')
    if dirty:
        print(f'WARNING: --allow-dirty build on {len(dirty)} uncommitted files; NOT for the cluster')
    mxt = mxt_dirty()
    if mxt:
        print('NOTE: mxtaltools has uncommitted changes the cluster will NOT run:\n  ' + '\n  '.join(mxt))

    arms = build()
    for name, (cfg, _fam, _dropped) in arms.items():
        load_check(cfg, name)

    for fam, spec in FAM.items():
        local = LOCAL_DATA / w3_path(load(spec['seed'])['prior_path']).rsplit('/', 1)[1]
        if local.exists():
            assert local.stat().st_size == spec['prior_bytes'], \
                f'{fam}: {local} is {local.stat().st_size} bytes, PRIOR_BYTES says {spec["prior_bytes"]}'
        else:
            print(f'NOTE: {local} not on this machine; PRIOR_BYTES for {fam} is unchecked')

    logs = HERE / 'joblogs'
    logs.mkdir(exist_ok=True)
    (logs / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n',
                                   encoding='utf-8')
    for name, (cfg, _fam, _dropped) in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tscale\twarm_src\tprior\tprior_bytes\n')
        for name, (cfg, fam, _dropped) in arms.items():
            f.write(f"{name}\t{fam}\t{SCALE:g}\t{FAM[fam]['src']}\t{cfg['prior_path'].rsplit('/', 1)[1]}"
                    f"\t{FAM[fam]['prior_bytes']}\n")
    fname = f'submit_{BATTERY}.sbatch'
    with (HERE / fname).open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(wall=WALL, last=len(arms) - 1, tag=TAG, battery=BATTERY,
                              ckpts=CLUSTER_CKPTS, data=CLUSTER_DATA,
                              walls_commit=WALLS_COMMIT, walls_symbol=WALLS_SYMBOL,
                              ck_ph=CK_PLACEHOLDER, cont_ph=CONT_PLACEHOLDER))

    print(f'{len(arms)} arms -> {HERE}  ({fname}: array 0-{len(arms) - 1}, wall {WALL})')
    for name, (cfg, fam, dropped) in arms.items():
        print(f"{name:<18} seed=*{FAM[fam]['src']}_*_best.pt  prior={cfg['prior_path'].rsplit('/', 1)[1]}  "
              f"{cfg['energy_function']}  scale={SCALE:g}  T={SHIP_T}  batch={cfg['batch_size']}->{cfg['max_batch_size']}"
              f"@{cfg['batch_util_target']}  eval={cfg['eval_period']}/{cfg['eval_num_samples']}"
              f"  dropped energy_config={dropped or '-'}")


if __name__ == '__main__':
    main(sys.argv[1:])
