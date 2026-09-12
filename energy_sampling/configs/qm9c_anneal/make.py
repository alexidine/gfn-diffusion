r"""
Build the lambda-anneal arm from the qm9c_lam003 base, adopting mk_dev's
conditional_vargrad protocol WHOLE: train_prior, then var_conditioning with the
REPLAY branch in the forward branch's seat -- a forward rollout on 1 step in 20
feeding the replay buffer, pooled VarGrad between replay and bwd over the
aligned per-condition draw, the forward loss as a Z(c)-only sidecar on rollout
steps, and lambda_mix ramped by the stage's own anneal_coeffs under a loss gate.
The route's GLOBAL values (GLOBALS below) are set here; mk_dev notes each beside
its key as "CONDITIONAL ARM:".

A FRESH RUN, NOT A RESUME. The policy is a weights-only load of the conditional
route's phase-1 train_prior exit, which is unconditional by design
(scramble_conditions); optimizers, buffers and the condition tracker start
fresh. prior_model_name loads the same run's prior model, so train_prior's
skip_if (prior_loaded) holds and protocol.begin() enters var_conditioning
through advance() at step 0 -- which is what runs its on_enter
(rebuild_prior_by_churn, set_lr_flow). A protocol holding
var_conditioning alone would START in it, and the first stage is never entered
through a transition, so its on_enter would never run.

WHY 0.01 IS THE START AND NOT 0. The ramp is multiplicative (`val <- val/rate`),
so it cannot leave exactly zero. And it should not start near zero anyway:
measured on 32000 policy draws x the real ELJ (scratch lambda_scan2.py), the
physical leg's within-condition spread is under a quarter of the flow leg's for
all lambda < 0.03, so the whole region below that costs wall clock and moves the
target almost not at all. lambda = 0.003, where the previous ladder sat, puts the
physical leg at 13% of the flow leg.

    python configs/qm9c_anneal/make.py
"""
import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.dirname(HERE)
REPO = os.path.dirname(CONFIGS)
for p in (REPO, os.path.dirname(REPO), os.path.join(os.path.dirname(REPO), 'mxtaltools')):
    if p not in sys.path:
        sys.path.insert(0, p)

BASE = os.path.join(CONFIGS, 'qm9c_lam003.yaml')
MK_DEV = os.path.join(CONFIGS, 'mk_dev.yaml')
OUT = os.path.join(CONFIGS, 'qm9c_anneal.yaml')

LAMBDA_START = 0.001   # 0 -> 0.001 is measured invisible; the ramp is multiplicative and cannot leave 0
EPOCHS = 60000

# THE LAMBDA=0 NULL ARM (`python configs/qm9c_anneal/make.py --null`): the same
# fresh start and the same stage at lambda 0, where the target is the prior
# flow's own density. qm9c_null_l0 is the same check on the forward-seat design,
# from bit-identical policy weights with the same seed, LR, T and eval, so the two
# compare directly. The ramp is val/rate and cannot leave 0, so the anneal block
# is dropped rather than left reading as armed.
NULL_OUT = os.path.join(CONFIGS, 'qm9c_null_replay.yaml')
NULL_EPOCHS = 3000                  # qm9c_null_l0's length
NULL_CUDA_FRACTION = 0.8            # a local run: the desktop holds ~2.5 GB of the 16 GB card
NULL_DROPPED = ('anneal_coeffs', 'anneal_cooldown_steps')   # var_conditioning.balance keys

# THE STEP-RESPONSE ARM (`python configs/qm9c_anneal/make.py --step 0.1`): a FULL
# resume of the null arm's final checkpoint -- policy, Z(c) head, optimizers,
# tracker and buffers as the null run left them, converged at lambda 0 -- with
# lambda jumped to the given value and HELD. A deliberately too-large step: what
# breaks, and in what order, is what defines an anneal failure. The anneal block
# is dropped as on --null, so the stage's rules still log their elevations
# (protocol/elev_*) while nothing moves lambda. The full resume skips on_enter
# (protocol.begin() returns early), which is intended: the stage continues.
NULL_RUN = 'qm9c_null_replay'
STEP_EPOCHS = 3000                  # steps AFTER the null run's; epochs is absolute

# the conditional route's T=20 phase-1 run (matches the base's T20 prior flow)
SEED = 'dev_qm9c_t20_elj-qm9c100k_prior-T6.9-5e5294'
SEED_EXIT = f'{SEED}_phase1_exit.pt'     # train_prior's exit: the policy weights
SEED_PRIOR = f'{SEED}_prior.pt'          # the same run's sampling-only prior model

# The route's GLOBAL values: the stage list cannot carry them. Every one is set
# explicitly, because the base carries older values (replay churn 80 /
# residence 50 / max_size 12000 / prioritised; anchor thin 5 / refresh 3; no
# prior_buffer.source; batch growth to 8000 at util target 0.6; fill 20.0).
#
# THE BATCH IS PINNED. condition_draw's C (batch_size // 2 conditions a step)
# and the replay occupancy below both scale with the live batch, so growth would
# move both mid-stage. With grow_batch_size false select_batch_size never runs
# (neither the growth ladder nor the max_step_seconds guard), and
# util_target_actuable refuses a batch_util_target beside it. max_batch_size is
# then inert at >= batch_size. An OOM still cuts the batch, and with growth off
# nothing restores it until a resume (checkpointing.reconcile_batch_size).
#
# REPLAY OCCUPANCY, by Little's law: admissions per step x mean_residence_steps.
# churn_rate 0 admits the LIVE batch_size rows per manage call -- not batch_size
# x repeats: a rollout's 2000 rows at fwd repeats 2 are drawn down to 1000 --
# and there is one call per rollout (1 step in fwd_rollout_every). The eval call
# is OFF on this arm (admit_from_eval false below), so the 1/eval_period term is
# gone: 1000 x (1/20) = 50 rows/step x 1200 = 60,000 rows (fewer if
# admit_reward_min rejects any). max_size is ~2.5x that, so the cap never binds
# and the hazard alone sets occupancy; the warm-up's 20,000 releases well below
# it. A row is ~2.6 KB on the GPU (buffer_device cuda: the crystal graph, ~1.6 KB
# measured on the qm9c prior batch, plus the 21 x 12 float32 trajectory; the
# per-row bookkeeping columns are CPU tensors): ~180 MiB at equilibrium, ~375 MiB
# at the cap.
#
# THE SINGLE-NUMBER Z FILL IS OFF. It cannot act on a Z(c) head
# (_z_fill_head_is_fillable), so a threshold > 0 read as armed while doing
# nothing; the Z sidecar's emp_z pins Z(c) on every rollout, and fwd_z_sidecar
# waives the cadence's fill requirement (fwd_rollout_cadence_is_well_formed).
GLOBALS = {
    'batch_size': 1000,
    'grow_batch_size': False,
    'batch_util_target': 0.0,
    'buffers.replay_buffer.churn_rate': 0,
    'buffers.replay_buffer.mean_residence_steps': 1200,
    'buffers.replay_buffer.max_size': 150000,
    'buffers.replay_buffer.val_frac': 0.0,
    # the eval rollout is the EMA model on the eval_T grid -- a different model
    # on a different grid -- and on this route replay carries half the loss
    'buffers.replay_buffer.admit_from_eval': False,
    'buffers.replay_buffer.prioritise.enabled': False,   # refused beside pooled_source 'replay' / condition_draw
    'buffers.anchor_buffer.frozen': False,               # anchor growth back on
    'buffers.anchor_buffer.thin_every_n_evals': 0,
    'buffers.anchor_buffer.refresh_every_n_evals': 0,
    'buffers.anchor_buffer.topup_admit_record_breakers': True,
    'buffers.prior_buffer.source': 'anchors',
    'condition_log_z.rollout_condition_draw': 'cycle',
    'z_calibration.fill_threshold': 0.0,
    'z_calibration.fill_from_eval': 'off',
}


def replay_equilibrium(cfg, stage):
    """(rows admitted per train step, equilibrium replay occupancy) under
    churn_rate 0: batch_size rows per manage call, one call per rollout and --
    only while admit_from_eval is on -- one per eval, times
    mean_residence_steps (GLOBALS note)."""
    rb = cfg['buffers']['replay_buffer']
    assert rb['churn_rate'] == 0, 'the arithmetic below is the churn_rate 0 contract'
    calls = 1.0 / stage['fwd_rollout_every']
    if rb['admit_from_eval']:
        calls += 1.0 / cfg['eval_period']
    per_step = cfg['batch_size'] * calls
    return per_step, per_step * rb['mean_residence_steps']


def stage_named(cfg, name):
    for prot in cfg['protocols'].values():
        for st in prot['stages']:
            if st['name'] == name:
                return st
    raise KeyError(name)


def set_dotted(cfg, dotted, value):
    """Set `a.b.c`. Every parent block exists in the base, so a KeyError here
    is a base that moved, not a block to create."""
    *head, last = dotted.split('.')
    node = cfg
    for k in head:
        node = node[k]
    node[last] = value


def get_dotted(cfg, dotted):
    node = cfg
    for k in dotted.split('.'):
        node = node[k]
    return node


def _var_conditioning(prot):
    return next(s for s in prot['stages'] if s['name'] == 'var_conditioning')


def _step_tag(lam):
    return 'l' + f'{lam:g}'.replace('.', 'p')


def _source_tag(source):
    """'null' for the null arm (or 'null_<suffix>' for a variant of it), else the
    step arm's own lambda tag."""
    if source.startswith(NULL_RUN):
        rest = source[len(NULL_RUN):].lstrip('_')
        return f'null_{rest}' if rest else 'null'
    return source[len('qm9c_step_'):].split('_after_')[0]


def main(null=False, step=None, source=NULL_RUN, length=None, suffix=''):
    cfg = yaml.safe_load(open(BASE))
    mk = yaml.safe_load(open(MK_DEV))
    ref = copy.deepcopy(mk['protocols']['conditional_vargrad'])   # the whole protocol, single source
    held = null or step is not None          # lambda fixed: the anneal block is dropped
    # each arm has its own default length; --length overrides either
    length = length if length is not None else (NULL_EPOCHS if null else STEP_EPOCHS)
    tail = f'_{suffix.lstrip("_")}' if suffix else ''
    if step is not None:
        lam = float(step)
        # the source is in the name, so two ladders through the same lambda
        # cannot share a run_name (a reused run_name overwrites its checkpoints);
        # `suffix` separates two runs of the SAME rung, e.g. a longer re-run
        run_name = f'qm9c_step_{_step_tag(lam)}_after_{_source_tag(source)}{tail}'
        out = os.path.join(CONFIGS, f'{run_name}.yaml')
    elif null:
        # a suffixed null arm is a VARIANT baseline (e.g. P_B learned): its own
        # run_name and config, so it cannot clobber the original's checkpoints
        lam, run_name = 0.0, f'{NULL_RUN}{tail}'
        out = NULL_OUT if not tail else os.path.join(CONFIGS, f'{run_name}.yaml')
    else:
        lam, run_name, out = LAMBDA_START, 'qm9c_anneal_r2gate', OUT

    cfg['run_name'] = run_name
    # absolute: a weights-only start is at step 0, so this is the whole budget;
    # a resume picks up where its source run's epochs ended
    if step is not None:
        cfg['epochs'] = yaml.safe_load(open(os.path.join(CONFIGS, f'{source}.yaml')))['epochs'] + length
    else:
        cfg['epochs'] = length if null else EPOCHS

    for name in (SEED_EXIT, SEED_PRIOR):
        path = os.path.join(cfg['checkpoints_dir'], name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{path} does not exist; the arm starts from it')
    cfg['prior_model_name'] = SEED_PRIOR
    cfg['continue_from_checkpoint'] = False
    if step is None:
        # weights-only start from the phase-1 exit, with its prior model loaded by
        # path so train_prior is skipped (module docstring)
        cfg['checkpoint_name'] = SEED_EXIT
        cfg['load_weights_only'] = True
    else:
        # FULL resume of the source run's final (STEP-RESPONSE ARM note);
        # exactly one must exist, or the arm would start from something else
        finals = sorted(f for f in os.listdir(cfg['checkpoints_dir'])
                        if f.startswith(f'dev_{source}_') and f.endswith('_final.pt'))
        if len(finals) != 1:
            raise FileNotFoundError(f'need exactly one dev_{source}_*_final.pt in '
                                    f'{cfg["checkpoints_dir"]}, found {finals}')
        cfg['checkpoint_name'] = finals[0]
        cfg['load_weights_only'] = False

    ec = cfg['energy_config']
    ec['lambda_mix'] = lam
    assert ec.get('prior_flow_path'), 'the anneal needs a fitted prior flow'
    # the clip is a nonlinear rescale of the PHYSICAL energy; applied to a
    # mixture it would make the lambda=0 endpoint something other than the flow,
    # and MolecularCrystal refuses the combination outright
    ec['reward_range'] = None

    # base pooled keys must EXIST (mk_dev owns the schema) but stay 0 in the base
    # block: runs_grouped_vargrad is not scoped by train_mode, so a nonzero base
    # pooled_vg flips condition_block_m 0 -> 2 in every bwd stage, train_prior
    # included. The stage below turns it on where it is actually computed.
    # pooled_source stays 'fwd' in the base for the same reason: 'replay' is
    # refused wherever pooled_vg is 0, so it is set on the stage only.
    for k, v in (('pooled_vg', 0.0), ('pooled_beta', 40.0), ('pooled_ratio', 0.5),
                 ('pooled_source', 'fwd')):
        cfg['fwd_loss_coeffs'][k] = v

    for dotted, v in GLOBALS.items():
        set_dotted(cfg, dotted, v)

    # both stages, copied whole: var_conditioning's on_enter runs only when it
    # is entered from train_prior (module docstring)
    cfg['protocols']['conditional_vargrad'] = copy.deepcopy(ref)

    # what the written protocol must equal: mk_dev's, less the anneal block on
    # the held-lambda arms (NULL_DROPPED)
    expected = copy.deepcopy(ref)
    if held:
        cfg['cuda_memory_fraction'] = NULL_CUDA_FRACTION
        for prot in (cfg['protocols']['conditional_vargrad'], expected):
            bal = _var_conditioning(prot)['balance']
            for k in NULL_DROPPED:
                bal.pop(k)

    with open(out, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, width=1000)

    # LOADING IS THE TEST, not generating. Re-read from disk and run the
    # invariants (configs/generate.py asserts on errors for the same reason),
    # then the real loader train.py uses.
    import config_invariants as ci
    import utils
    back = yaml.safe_load(open(out))
    errs = ci.errors(back)
    for v in ci.check(back):
        print('  ', v)
    assert not errs, f'{len(errs)} invariant ERROR(s) in the generated config'
    utils.resolve_derived_config(utils.preflight_config(utils.dict2namespace(utils.load_yaml(out))))

    prot = back['protocols']['conditional_vargrad']
    assert prot == expected, "conditional_vargrad must be mk_dev's, unedited bar NULL_DROPPED on --null"
    assert [s['name'] for s in prot['stages']] == ['train_prior', 'var_conditioning']
    assert prot['stages'][0]['skip_if'] == 'prior_loaded'
    got = prot['stages'][1]
    assert got['balance']['kind'] == 'lexicographic'
    if held:
        assert not any(k in got['balance'] for k in NULL_DROPPED)
    else:
        assert got['balance']['anneal_coeffs']['lambda_mix']['target'] == 1.0
        assert got['balance']['anneal_cooldown_steps'] == 1000
        assert got['balance']['anneal_coeffs']['lambda_mix']['rate'] == 0.8
    assert got['fracs'] == {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}
    assert got['min_fracs'] == {'fwd': 0.0}
    fwd, bwd, rep = (got['loss_coeffs'][m] for m in ('fwd', 'bwd', 'replay'))
    assert fwd['pooled_vg'] == 1.0 and fwd['pooled_source'] == 'replay'
    assert fwd['tb'] == 0.0 and fwd['freeze_policy'] == 1.0 and fwd['emp_z'] == 1.0
    assert fwd['vg_lb'] == 0.0 and bwd['vg_lb'] == 0.0
    assert bwd['tb'] == 0.0 and rep['tb'] == 0.0, 'base replay tb is 1.0'
    assert bwd['condition_block_m'] == 0 and rep['condition_block_m'] == 0
    assert got['fwd_rollout_every'] == 20 and got['fwd_z_sidecar'] is True
    assert got['replay_warmup_rows'] == 20000
    assert got['condition_draw'] == {'conditions': 0, 'replay_rows': 2, 'prior_rows': 2,
                                     'pick': 'uniform'}
    assert [r['metric'] for r in got['balance']['rules']] == \
        ['fwd/r2_unexplained', 'bwd/r2_unexplained']
    # P_B is FROZEN (full snapshot) at stage entry on the replay seat: every
    # learned-P_B variant diverged there (2026-09-12)
    assert 'freeze_pb' in got['on_enter']
    # absent = the code default (False); the base has never carried the key
    assert back.get('freeze_backward_policy', False) is False
    assert not any(a.startswith('bootstrap_z') for a in got['on_enter'])
    # the replay cap is headroom over the equilibrium, and the warm-up releases
    # well below it (GLOBALS note)
    per_step, occupancy = replay_equilibrium(back, got)
    rb = back['buffers']['replay_buffer']
    assert rb['max_size'] >= 2 * occupancy, (rb['max_size'], occupancy)
    assert got['replay_warmup_rows'] <= occupancy / 2, (got['replay_warmup_rows'], occupancy)
    if step is None:
        assert back['checkpoint_name'] == SEED_EXIT and back['load_weights_only'] is True
    else:
        assert back['checkpoint_name'].startswith(f'dev_{source}_')
        assert back['checkpoint_name'].endswith('_final.pt')
        assert back['load_weights_only'] is False
    assert back['continue_from_checkpoint'] is False
    assert back['prior_model_name'] == SEED_PRIOR
    for dotted, v in GLOBALS.items():
        assert get_dotted(back, dotted) == v, dotted
    assert back['fwd_loss_coeffs']['pooled_source'] == 'fwd'
    assert back['energy_config']['lambda_mix'] == lam
    assert back['energy_config']['reward_range'] is None
    # the pooled term reads bwd's condition groups; condition_draw's prior_rows
    # supplies them (vargrad_needs_groups, inside ci.errors above)
    assert ci._runs_vargrad(back, got, 'bwd'), 'the pooled term would not read bwd groups'
    for prot in back['protocols'].values():
        for s in prot['stages']:
            if s.get('train_mode') == 'bwd':
                assert not ci._runs_vargrad(back, s, 'bwd'), \
                    f"stage {s['name']}: pooled_vg leaked into a bwd stage"
    print(f'wrote {out}  (lambda_mix {"fixed at" if held else "starts at"} {lam}, epochs '
          f'{back["epochs"]}; replay equilibrium ~{occupancy:.0f} rows at {per_step:.0f} '
          f'admitted/step, cap {rb["max_size"]})')


def _cli(argv):
    """`--null [--length N] [--suffix S]`, or `--step LAMBDA [--from RUN] [--length N]
    [--suffix S]`; neither = the anneal
    arm. --from names the run whose final checkpoint the step arm resumes
    (default the null arm) -- so a ladder is a chain of step arms -- and --length
    is how many steps the arm runs past that run's end (default STEP_EPOCHS)."""
    def _value(flag):
        i = argv.index(flag)
        if i + 1 >= len(argv):
            raise SystemExit(f'{flag} needs a value')
        return argv[i + 1]

    null = '--null' in argv
    step = float(_value('--step')) if '--step' in argv else None
    source = _value('--from') if '--from' in argv else NULL_RUN
    length = int(_value('--length')) if '--length' in argv else None
    suffix = _value('--suffix') if '--suffix' in argv else ''
    if '--from' in argv and step is None:
        raise SystemExit('--from only means something with --step')
    if not null and step is None and any(f in argv for f in ('--length', '--suffix')):
        raise SystemExit('--length and --suffix need --null or --step')
    if step is not None and not 0.0 < step <= 1.0:
        raise SystemExit(f'--step {step}: lambda must be in (0, 1]')
    if null and step is not None:
        raise SystemExit('--null and --step are exclusive')
    if length is not None and length <= 0:
        raise SystemExit(f'--length {length}: must be > 0')
    return null, step, source, length, suffix


if __name__ == '__main__':
    _null, _step, _source, _length, _suffix = _cli(sys.argv[1:])
    main(null=_null, step=_step, source=_source, length=_length, suffix=_suffix)
