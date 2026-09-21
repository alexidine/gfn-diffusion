"""final_sep19 -- the final-protocol battery, STAGED (owner 2026-09-19 evening): converge each family in the
known-good but expensive regime first, then switch it into rare rollouts and test the two optional pieces.

    python configs/final_sep19/make.py            # base = the COMMITTED mk_dev (git show HEAD:), warns on a dirty tree
    python configs/final_sep19/make.py --families nehu

LEG A (submit_final_sep19_a.sbatch): the PRODUCTION TRUNK fin19_<fam>_a -- phase 2 from the mle_fresh_sep17
best-MLE checkpoint under the shape that reached 35.5-35.9 on mip (dose_sep16 n1_t6_pbfrozen / dose16e
n1_pbfrozen): a rollout EVERY step, P_B frozen at the phase-2 entry (freeze_pb:full on equilibration's
on_enter), replay 0.3 / bwd 0.7 PINNED, tau 120, no forces, batch pinned 1600. It runs until the owner judges
it converged (log Z flat, held-out gap flat, zmatch/delta_mean settled). Beside it on ELJ families only
(RAMP_TRUNKS), fin19_<fam>_ramp_a: the same trunk with the gated ramp FREE from 0.3 up to replay 0.95 /
bwd 0.05 -- the share question, given the horizon it needs, without the production path depending on it.
Owner 2026-09-19 21:10: mip only for now.

LEG B (submit_final_sep19_b.sbatch, four cells per family): the SWITCH into rare rollouts, seeded by a FULL
resume of the leg-A arm's newest step archive (archive_period 5000, archive_buffers on: `_stepN.pt` pairs
with its frozen `_stepN_buffers.pt`). The resumed stage `equilibration` (N=1) carries an always-true exit,
so at the first eval the run transitions -- for real: fresh stage_ctrl, entry fracs applied, LR re-stamped
at fixed_scale -- into `equilibration_rare`: fwd_rollout_every N_RARE, tau 600, the share per SHARE_MODE,
rate = the family's N=5 rate x 5/N_RARE x RATE_B_FACTOR (half the dose-matched pressure: the local hold test
held log Z at the full dose but its held-out gap crept, see RATE_B_FACTOR). The cells:
  <fam>_pb      P_B stays frozen (the snapshot rides the checkpoint), no forces      the switch alone
  <fam>_unpb    unfreeze_pb on the rare stage's on_enter                              does a live P_B help or hurt once converged
  <fam>_pbf     frozen + stored terminal force on replay rows (stored_force_k 1)      the untested cell of the dose ladder
  <fam>_unpbf   live P_B + stored force
  <fam>_pb_n5, <fam>_pb_n20   the plain cell (frozen, no force) at N=5 and N=20, dose-matched rate (x 5/N):
                the fan over N -- how rare the rollouts can be while the trunk's state holds (rows after the 2x2)

WHY STAGED. The local rr_sep19 ladder (2026-09-19) showed that resuming a held state (replay 0.1 / bwd 0.9)
under rare rollouts is a deadlock: log Z flat, coverage level rising, the ramp's ratchet never releases. The
staged form never asks rare rollouts to climb; it asks them to HOLD a converged state, which is the cheap
regime's only job in production. The hold test (rr19 hold10: dose16e_n1_pbfrozen best -> N=10) is the local
evidence for this design.

SHARE_MODE. 'inherit' (shipped): the rare stage declares no fracs and no balance, so the transition leaves the
split exactly where leg A's controller left it and nothing moves it afterwards -- the switch changes N and
nothing else. 'pin30': replay 0.3 / bwd 0.7 pinned (the local hold test's shape). 'ramp50': enter at 0.5 with
the ramp free.

IDENTITY: prior_path/molecules_path/space_groups/z_primes/energy_function/mlip_path/checkpoints_dir, the
`model` block (dplr_rank 0; committed mk_dev carries 6) and energy_config VERBATIM from the seed yaml
(get_problem_definition hashes the block; a key the seed lacks moves the identity and load_full refuses).
Leg-B configs share leg A's identity exactly, asserted. Everything else is the committed mk_dev's
unconditional_tb with the deltas below. force_chunk_rows stays absent (route-aware default: 8-row autograd
chunks on an MLIP, one call on ELJ).

READ. Leg A: fwd/log_Z_learned vs the family's history, eval_fwd/tb_err, replay/val_gap_nats, Mean Sample
Energy; switch when flat. Leg B at matched steps past the switch: the same four, plus Replay Frac and
protocol/gr_tripped|gr_fired under 'ramp50', rewardgrad/force_* and energy/seconds on the _f cells. The cell
that holds leg A's level with a flat held-out gap at 1/N_RARE of the energy calls is the protocol.
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

#: ---- the shape -------------------------------------------------------------------------------------
N_RARE = 10                             # leg B rollout period for the 2x2 (the local hold test's N)
N_FAN = [5, 20]                         # the plain cell (P_B kept frozen, no force) repeated at these N, dose-matched rate
SHARE_MODE = 'inherit'                  # 'inherit' | 'pin30' | 'ramp50' -- see the docstring
TAU_A, TAU_B = 120, 600
REPLAY_MAX = 1_000_000                  # the cap must never bind: B x tau / N = 192k rows in leg A at 1600 / 120 / 1
VAL_FRAC, VAL_CAP = 0.05, 1024
RAMP_BOUNDS = {'bwd': [0.05, 0.9], 'replay': [0.1, 0.95]}
PIN_REPLAY = 0.3
PINNED_BOUNDS = {'bwd': [0.7, 0.7], 'replay': [0.3, 0.3]}
BATCH = 1600
BURN_IN_STEPS = 500
EXCURSION_K = 60.0
#: rates. Leg A at N=1: mip 1.0 (dose_sep16 n1 arms), nehu 0.125 (the p02/p12 paper rate, the only rate ever
#: visited there). Leg B: the family's N=5 rate x 5/N_RARE (mip: n5_t600_pb at 1.0; nehu: 0.125 assumed).
RATE_N1 = {'nehu': 0.125, 'mip': 1.0}
RATE_N5 = {'nehu': 0.125, 'mip': 1.0}
#: leg B runs at HALF the dose-matched rate. The local hold test (rr19 hold10: the converged N=1 frozen mip
#: arm switched into N=10 at the dose-matched 0.5, batch 400, 1000 steps) HELD log Z (34.84 -> 34.83, sd 0.03),
#: energies and on-policy errors flat at 1/30 the energy calls, but its held-out gap crept 0.10 -> 0.17 while
#: replay/tb_err fell 3.7 -> 3.1: memorisation at that pressure. Leg B's job is to hold, not to climb, so it
#: takes half the pressure per pass; the cluster's batch 1600 gives 4x the fresh rows per step on top.
RATE_B_FACTOR = 0.5
FAMILIES = ['mip']                      # owner 2026-09-19 21:10: mip only for now
RAMP_TRUNKS = ['mip']                   # families that also get the controller-free trunk (the leg-A share question)
SEED_ARM = {'nehu': 'mle_fresh_sep17/mlefr_nehu_lr2.yaml', 'mip': 'mle_fresh_sep17/mlefr_mip_lr2.yaml'}
SRC = {'nehu': 'mlefr_nehu_lr2', 'mip': 'mlefr_mip_lr2'}
MLIP = {'nehu': True, 'mip': False}
UMA_EVAL = dict(eval_period=1000, eval_num_samples=2500, figs_period=1000)
ELJ_EVAL = dict(eval_period=500, eval_num_samples=10000, figs_period=1000)
IDENTITY_KEYS = ('prior_path', 'molecules_path', 'test_molecules_path', 'space_groups', 'z_primes',
                 'energy_function', 'mlip_path', 'checkpoints_dir', 'model', 'energy_config')
CELLS = {'pb': (True, False), 'unpb': (False, False), 'pbf': (True, True), 'unpbf': (False, True)}   # (keep frozen, force)

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


def _stage(cfg, name):
    return [s for s in cfg['protocols']['unconditional_tb']['stages'] if s['name'] == name][0]


def _pin(stage, replay):
    bwd = round(1.0 - replay, 3)
    stage['fracs'] = {'fwd': 0.0, 'bwd': bwd, 'replay': replay}
    stage['balance']['bounds'] = {'bwd': [bwd, bwd], 'replay': [replay, replay]}


def common(cfg, fam, name):
    """The phase-2 localisation shared by both legs: identity from the seed, mk_dev's unconditional_tb with
    the stub train_prior, pinned batch, the route's eval budget, no forward-branch forces."""
    seed = load(SEED_ARM[fam])
    mlip = MLIP[fam]
    for k in IDENTITY_KEYS:
        cfg[k] = copy.deepcopy(seed.get(k))
    # energy_config is the SEED'S BLOCK VERBATIM, absences included (see the docstring: identity)
    cfg['run_name'] = name
    cfg['tag'] = TAG
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
    cfg['protocol'] = 'unconditional_tb'
    stub = _stage(cfg, 'train_prior')
    stub.pop('skip_if', None)
    stub['exit'] = [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}]
    stub['on_exit'] = ['snapshot_prior']
    eq = _stage(cfg, 'equilibration')
    assert eq['balance']['kind'] == 'gated_ramp' and eq['balance']['pinned'] == {'fwd': 0.0}, name
    assert 'rebuild_prior_by_churn' in (eq.get('on_enter') or []), name
    assert float(eq['deactivate_threshold']) <= RAMP_BOUNDS['bwd'][0] and eq.get('min_fracs') is None, name
    assert cfg.get('freeze_backward_policy') in (False, None), name   # the freeze is a stage action, never the load-time key
    assert cfg['buffers']['prior_buffer']['source'] == 'anchors', name
    rb = cfg['buffers']['replay_buffer']
    rb['churn_rate'] = 0
    rb['max_size'] = REPLAY_MAX
    rb['val_frac'] = VAL_FRAC
    rb['val_cap'] = VAL_CAP
    lc = cfg['lr_control']
    lc['mode'] = 'fixed'
    lc['burn_in_steps'] = BURN_IN_STEPS
    lc['repeat_every'] = 0
    lc['fire_cut_factor'] = 1.0
    lc['hard_failure']['loss_excursion_k'] = EXCURSION_K
    cfg['batch_size'] = BATCH
    cfg['max_batch_size'] = BATCH
    cfg['grow_batch_size'] = False
    cfg['batch_util_target'] = 0.0
    cfg['batch_sizer_retest_steps'] = 0
    cfg['traj_checkpoint'] = bool(mlip)
    cfg['energy_config']['internal_oom_recovery'] = bool(mlip)
    cfg.update(UMA_EVAL if mlip else ELJ_EVAL)
    fc = cfg['fwd_loss_coeffs']
    fc['reward_grads'] = 0.0
    fc['traj_grads'] = 0.0
    fc['path_grad_last_k'] = 0
    rc = cfg['replay_loss_coeffs']
    rc['stored_force_k'] = 0
    rc['stored_force_mode'] = 'implied'
    rc['resample_last_k'] = 0
    rc['reward_grads'] = 0.0
    rc['force_chunk_rows'] = None
    # detach_pb was NEVER READ by any executed module (retired 2026-09-20 in the working tree; committed code
    # ignores it): dropped so the config loads under either, and a requeue after the retirement lands still resolves
    for blk in ('bwd_loss_coeffs', 'replay_loss_coeffs'):
        cfg[blk].pop('detach_pb', None)
    return cfg


def _rate(cfg, scale):
    lc = cfg['lr_control']
    lc['fixed_scale'] = float(scale)
    lc['burn_in_scale'] = float(scale)


def stage_a(cfg, fam, name, share='pin'):
    common(cfg, fam, name)
    eq = _stage(cfg, 'equilibration')
    eq['fwd_rollout_every'] = 1
    if share == 'pin':
        # THE PRODUCTION TRUNK: replay 0.3 / bwd 0.7 pinned by point bounds -- the only split with long-horizon
        # evidence (dose_sep16 n1 arms, 8k-21k steps, 35.9 on mip). Nothing about the share is assumed here.
        _pin(eq, PIN_REPLAY)
    elif share == 'ramp':
        # THE SHARE QUESTION, beside the pinned trunk on ELJ where it is cheap: enter at the same 0.3 / 0.7 with the
        # gated ramp FREE up to 95:5 (owner: let the controller do its job). The local ramp50 arm ramped unvetoed
        # from a fresh reference and climbed faster over 1000 steps, but zmatch/delta_mean was not read and the runs
        # were far too short for convergence statistics; read this arm at 5k / 10k / 20k on zmatch/delta_mean,
        # replay/val_gap_nats and fwd/log_Z_learned against the pinned trunk.
        eq['fracs'] = {'fwd': 0.0, 'bwd': 0.7, 'replay': 0.3}
        eq['balance']['bounds'] = {k: list(v) for k, v in RAMP_BOUNDS.items()}
    else:
        raise ValueError(share)
    eq['on_enter'] = list(eq['on_enter']) + ['freeze_pb:full']
    cfg['buffers']['replay_buffer']['mean_residence_steps'] = TAU_A
    _rate(cfg, RATE_N1[fam])
    return cfg


def leg_b(cfg, fam, name, keep_frozen, force, n_rare=N_RARE):
    """Leg A's config (same identity, same stages) plus the switch: `equilibration` exits at its first eval
    into `equilibration_rare`."""
    stage_a(cfg, fam, name, share='pin')      # seeded from the PINNED trunk; 'inherit' then holds 0.3 / 0.7
    stages = cfg['protocols']['unconditional_tb']['stages']
    eq = _stage(cfg, 'equilibration')
    eq['exit'] = [{'metric': 'fwd/log_Z_learned', 'above': -1.0e9, 'patience': 1}]
    rare = copy.deepcopy(eq)
    rare['name'] = 'equilibration_rare'
    rare.pop('exit')
    rare.pop('on_exit', None)
    rare['on_enter'] = [] if keep_frozen else ['unfreeze_pb']
    rare['fwd_rollout_every'] = int(n_rare)
    if SHARE_MODE == 'inherit':
        # no fracs, no balance: advance() leaves the live split as leg A's controller left it and nothing moves it
        # afterwards, so the switch changes N and nothing else
        rare.pop('fracs', None)
        rare['balance'] = None
    elif SHARE_MODE == 'pin30':
        _pin(rare, PIN_REPLAY)
    elif SHARE_MODE == 'ramp50':
        rare['fracs'] = {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}
        rare['balance']['bounds'] = {k: list(v) for k, v in RAMP_BOUNDS.items()}
    else:
        raise ValueError(SHARE_MODE)
    stages.append(rare)
    cfg['buffers']['replay_buffer']['mean_residence_steps'] = TAU_B
    _rate(cfg, RATE_N5[fam] * 5.0 / n_rare * RATE_B_FACTOR)
    if force:
        cfg['replay_loss_coeffs']['stored_force_k'] = 1
    return cfg


def check_common(cfg, name, fam):
    seed = load(SEED_ARM[fam])
    assert w3.problem_def(cfg) == w3.problem_def(seed), name + ': problem identity moved from the seed'
    assert cfg['prior_path'] == seed['prior_path'] == cfg['molecules_path'], name
    assert cfg['model'] == seed['model'] and cfg['model']['dplr_rank'] == 0, name
    assert cfg['integrator']['T'] == 100 == cfg['eval_T'], name
    assert cfg['checkpoint_name'] == PLACEHOLDER and cfg['prior_model_name'] == PRIOR_PLACEHOLDER, name
    assert cfg['load_weights_only'] is False and cfg['continue_from_checkpoint'] is False, name
    assert cfg['epochs'] >= 500_000, name
    stub = _stage(cfg, 'train_prior')
    assert stub['exit'] == [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}] and 'skip_if' not in stub, name
    eq = _stage(cfg, 'equilibration')
    assert eq['fwd_rollout_every'] == 1 and eq['fracs'] == {'fwd': 0.0, 'bwd': 0.7, 'replay': 0.3}, name
    assert eq['balance']['bounds'] in (RAMP_BOUNDS, PINNED_BOUNDS) and 'freeze_pb:full' in eq['on_enter'], name
    assert eq['loss_coeffs']['fwd']['freeze_policy'] == 1.0, name
    rb = cfg['buffers']['replay_buffer']
    assert rb['max_size'] >= 5 * BATCH * TAU_A and rb['val_frac'] == VAL_FRAC, name
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == lc['burn_in_scale'], name
    assert cfg['batch_size'] == cfg['max_batch_size'] == BATCH and cfg['grow_batch_size'] is False, name
    fc = cfg['fwd_loss_coeffs']
    assert fc['reward_grads'] == 0.0 and fc['traj_grads'] == 0.0 and fc['path_grad_last_k'] == 0, name
    assert cfg['replay_loss_coeffs']['resample_last_k'] == 0 and cfg['replay_loss_coeffs']['force_chunk_rows'] is None, name
    assert cfg['traj_checkpoint'] is MLIP[fam] and cfg['energy_config']['internal_oom_recovery'] is MLIP[fam], name
    w3._scan_local_paths(cfg, name)


def check_a(cfg, name, fam, share):
    check_common(cfg, name, fam)
    assert [s['name'] for s in cfg['protocols']['unconditional_tb']['stages']] == ['train_prior', 'equilibration'], name
    eq = _stage(cfg, 'equilibration')
    assert 'exit' not in eq, name
    assert eq['balance']['bounds'] == (RAMP_BOUNDS if share == 'ramp' else PINNED_BOUNDS), name
    assert cfg['buffers']['replay_buffer']['mean_residence_steps'] == TAU_A, name
    assert cfg['lr_control']['fixed_scale'] == RATE_N1[fam] and cfg['replay_loss_coeffs']['stored_force_k'] == 0, name


def check_b(cfg, name, fam, keep_frozen, force, cfg_a, n_rare=N_RARE):
    check_common(cfg, name, fam)
    assert w3.problem_def(cfg) == w3.problem_def(cfg_a), name + ': identity differs from leg A'
    assert _stage(cfg, 'equilibration')['balance']['bounds'] == PINNED_BOUNDS, name
    st = cfg['protocols']['unconditional_tb']['stages']
    assert [s['name'] for s in st] == ['train_prior', 'equilibration', 'equilibration_rare'], name
    eq, rare = st[1], st[2]
    assert eq['exit'] == [{'metric': 'fwd/log_Z_learned', 'above': -1.0e9, 'patience': 1}], name
    assert rare['fwd_rollout_every'] == n_rare and 'exit' not in rare, name
    assert rare['on_enter'] == ([] if keep_frozen else ['unfreeze_pb']), name
    if SHARE_MODE == 'inherit':
        assert 'fracs' not in rare and rare['balance'] is None, name
    elif SHARE_MODE == 'pin30':
        assert rare['fracs'] == {'fwd': 0.0, 'bwd': 0.7, 'replay': 0.3} and rare['balance']['bounds'] == {'bwd': [0.7, 0.7], 'replay': [0.3, 0.3]}, name
    else:
        assert rare['fracs'] == {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5} and rare['balance']['bounds'] == RAMP_BOUNDS, name
    assert cfg['buffers']['replay_buffer']['mean_residence_steps'] == TAU_B, name
    assert cfg['lr_control']['fixed_scale'] == RATE_N5[fam] * 5.0 / n_rare * RATE_B_FACTOR, name
    assert cfg['replay_loss_coeffs']['stored_force_k'] == (1 if force else 0), name
    if force:
        assert not raw_problem_def(cfg).get('temp_cond'), name + ': stored force under temperature conditioning'


def load_check(cfg, name, expect):
    """Load the arm the way train.py does (retired keys, invariants, derived values) and parse its protocol."""
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
    assert [s.name for s in stages] == expect, (name, [s.name for s in stages])


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

# {battery} leg {leg}: {what}
# Arm = row of INDEX_{leg}.tsv (line 1 is the header). DO NOT EDIT --array BY HAND: make.py rewrites it.
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
INDEX=${{ARMS}}/INDEX_{leg}.tsv
ARM=$(awk -F'\\t' -v n=${{ROW}} 'NR==n {{print $1}}' ${{INDEX}})
SRC=$(awk -F'\\t' -v n=${{ROW}} 'NR==n {{print $4}}' ${{INDEX}})
PRIOR=$(awk -F'\\t' -v n=${{ROW}} 'NR==n {{print $5}}' ${{INDEX}})
PRIOR_BYTES=$(awk -F'\\t' -v n=${{ROW}} 'NR==n {{print $6}}' ${{INDEX}})
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
# (full load); only a first launch seeds. REFUSES AN AMBIGUOUS MATCH.
OWN=$(ls -t ${{CKPTS}}/*${{ARM}}_*_running.pt 2>/dev/null | head -1)
if [ -n "${{OWN}}" ]; then
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  RESUME: $(basename ${{OWN}})"
    CK=${{OWN}}
else
{seed_block}
fi

# THE PRIOR MODEL: written by the stub's snapshot_prior on leg A's first launch; every later launch is
# handed one (its own if it has one, else its source arm's).
OWNPRIOR=$(ls -t ${{CKPTS}}/*${{ARM}}_*_prior.pt 2>/dev/null | head -1)
SRCPRIOR=$(ls -t ${{CKPTS}}/*${{SRC}}_*_prior.pt 2>/dev/null | head -1)
if [ -n "${{OWNPRIOR}}" ]; then
    PM=$(basename ${{OWNPRIOR}}); echo "  prior model <- ${{PM}} (own)"
elif [ -n "${{SRCPRIOR}}" ]; then
    PM=$(basename ${{SRCPRIOR}}); echo "  prior model <- ${{PM}} (source arm)"
else
    PM=null; echo "  prior model <- null (the stub writes it)"
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

SEED_A = """    # THE SEED IS THE mlefr ARM'S _best.pt (its bwd/mle record; that stage is terminal, no phase1_exit exists).
    NM=$(ls ${CKPTS}/*${SRC}_*_best.pt 2>/dev/null | wc -l)
    if [ "${NM}" -ne 1 ]; then
        echo "FATAL: ${NM} matches for *${SRC}_*_best.pt in ${CKPTS} (need exactly 1):" >&2
        ls ${CKPTS}/*${SRC}_*_best.pt >&2; exit 1
    fi
    CK=$(ls ${CKPTS}/*${SRC}_*_best.pt)
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  SEED: $(basename ${CK}) (mtime $(stat -c %y ${CK}))\""""

SEED_B = """    # THE SEED IS LEG A'S NEWEST STEP ARCHIVE (archive_period 5000; its frozen _stepN_buffers.pt sidecar
    # pairs with it), never its live _running.pt, which the still-running leg-A job may be rewriting.
    CK=$(ls -t ${CKPTS}/*${SRC}_*_step[0-9]*.pt 2>/dev/null | grep -v '_buffers.pt$' | head -1)
    if [ -z "${CK}" ]; then
        echo "FATAL: leg A arm ${SRC} has no step archive yet in ${CKPTS} (needs archive_period steps)" >&2; exit 1
    fi
    if [ ! -f "${CK%.pt}_buffers.pt" ]; then
        echo "FATAL: ${CK} has no frozen buffers sidecar beside it" >&2; exit 1
    fi
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  SWITCH from: $(basename ${CK}) (mtime $(stat -c %y ${CK}))\""""


def build(families):
    base = committed_mk_dev()
    A, B = {}, {}
    for fam in families:
        name_a = f'{TAG}_{fam}_a'
        cfg_a = stage_a(copy.deepcopy(base), fam, name_a, share='pin')
        check_a(cfg_a, name_a, fam, 'pin')
        load_check(cfg_a, name_a, ['train_prior', 'equilibration'])
        A[name_a] = (cfg_a, fam)
        if fam in RAMP_TRUNKS:
            name_r = f'{TAG}_{fam}_ramp_a'   # NOT <fam>_a_ramp: leg B's seed glob *<fam>_a_*_step*.pt must not match it
            cfg_r = stage_a(copy.deepcopy(base), fam, name_r, share='ramp')
            check_a(cfg_r, name_r, fam, 'ramp')
            load_check(cfg_r, name_r, ['train_prior', 'equilibration'])
            A[name_r] = (cfg_r, fam)
        for cell, (keep_frozen, force) in CELLS.items():
            name = f'{TAG}_{fam}_{cell}'
            cfg = leg_b(copy.deepcopy(base), fam, name, keep_frozen, force)
            check_b(cfg, name, fam, keep_frozen, force, cfg_a)
            load_check(cfg, name, ['train_prior', 'equilibration', 'equilibration_rare'])
            B[name] = (cfg, fam, name_a, keep_frozen, force, N_RARE)
        for n in N_FAN:      # rows AFTER the 2x2, so --array=0-3 still submits the 2x2 alone
            name = f'{TAG}_{fam}_pb_n{n}'
            cfg = leg_b(copy.deepcopy(base), fam, name, True, False, n_rare=n)
            check_b(cfg, name, fam, True, False, cfg_a, n_rare=n)
            load_check(cfg, name, ['train_prior', 'equilibration', 'equilibration_rare'])
            B[name] = (cfg, fam, name_a, True, False, n)
    return A, B


def _write_index(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tstart\twarm_src\tprior\tprior_bytes\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')


def main(argv):
    families = argv[argv.index('--families') + 1].split(',') if '--families' in argv else FAMILIES
    dirty = w3.dirty_files()
    if dirty:
        print('WARNING: the working tree is dirty (base read from git HEAD; the cluster runs HEAD):\n  ' + '\n  '.join(dirty))
    A, B = build(families)
    prior_index = {l.split('\t')[1]: l.split('\t') for l in (ROOT / 'mle_fresh_sep17' / 'INDEX.tsv').read_text(encoding='utf-8').splitlines()[1:]}
    for stale in HERE.glob(f'{TAG}_*.yaml'):
        stale.unlink()
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n', encoding='utf-8')
    for name, (cfg, *_r) in list(A.items()) + list(B.items()):
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    rows_a = [(n, fam, 'seed', SRC[fam], prior_index[fam][4], prior_index[fam][5]) for n, (cfg, fam) in A.items()]
    rows_b = [(n, fam, 'switch', src, prior_index[fam][4], prior_index[fam][5]) for n, (cfg, fam, src, *_x) in B.items()]
    _write_index(HERE / 'INDEX_a.tsv', rows_a)
    _write_index(HERE / 'INDEX_b.tsv', rows_b)
    for stale in ('INDEX.tsv', f'submit_{BATTERY}.sbatch'):
        if (HERE / stale).exists():
            (HERE / stale).unlink()
    common_kw = dict(wall=WALL, tag=TAG, battery=BATTERY, ckpts=w3.CLUSTER_CKPTS, data=w3.CLUSTER_DATA)
    with (HERE / f'submit_{BATTERY}_a.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(last=len(A) - 1, leg='a', seed_block=SEED_A,
                              what='phase 2 from the mlefr best-MLE checkpoints, rollout every step, P_B frozen, replay pinned 0.3.', **common_kw))
    with (HERE / f'submit_{BATTERY}_b.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(last=len(B) - 1, leg='b', seed_block=SEED_B,
                              what=f'the switch into rare rollouts (N={N_RARE}, share {SHARE_MODE}) from leg A archives; 2x2 on P_B kept frozen x stored force.', **common_kw))
    for i, (name, (cfg, fam)) in enumerate(A.items()):
        share = 'ramp free 0.3->95:5' if name.endswith('_ramp_a') else 'pinned 0.3/0.7'
        print(f"[a{i}] {name:<16} N=1 tau={TAU_A} rate={cfg['lr_control']['fixed_scale']:g} {share} frozen-at-entry seed=*{SRC[fam]}_*_best.pt")
    for i, (name, (cfg, fam, src, kf, force, n)) in enumerate(B.items()):
        print(f"[b{i}] {name:<16} N={n} tau={TAU_B} rate={cfg['lr_control']['fixed_scale']:g} share={SHARE_MODE} "
              f"pb_frozen_kept={kf} stored_force={force} switch-from=*{src}_*_step*.pt")


if __name__ == '__main__':
    main(sys.argv[1:])
