"""size_aug03 -- 2x2: replay buffer SIZING x fwd_frac. Derived from
configs/ctrl_aug03/fx_static.yaml (read-only), which is the (lo, 0.05) cell and
is ALREADY RUN -- only the three new cells are generated here.

WHY. ctrl_aug03 found replay/fwd scatter inverted (0.74-0.97) where it should be
structurally > 1, since replay is a |resid|-prioritized resample of fwd and is
therefore the hard tail. The buffer telemetry says why:

    draws/row/step = batch_size / occupancy = 2831 / ~3800 = 0.75

Every replay step trains on ~3/4 of the whole buffer, and the mean resident row
in fx_static had been trained on 33.1 times. At step 6690 -- ten steps into the
stage -- each row already had ~7-9 draws, which is why the ratio is already 0.91
at the FIRST reading rather than decaying into it. The reference run
postfix_july30 never inverts: batch 1000 into occupancy ~10000 (its residence
hazard is off) = 0.1 draws/row/step, at replay frac 0.048.

So the ctrl_aug03 servo suppressed a symptom of an undersized buffer. It holds
occupancy invariant by construction, so it never moved draws/row/step at all
(still 0.88-0.94); what it moved was CUMULATIVE exposure, by evicting rows
before they could be memorized (mean draws/row 33.1 -> 1.2-4.1). That also
explains its saturation: fx_servo_hi sat at 1.16 draws/row, and you cannot go
below 1.

FACTOR 1 -- SIZING. Raising max_size ALONE IS INERT: the hazard evicts n/tau per
call, so occupancy equilibrates at churn_rate * tau = 80 * 50 = 4000, which is
already exactly max_size (measured: 3789). max_size is not what binds. The lever
is churn_rate at FIXED tau, with max_size raised only so it stops binding:

                     lo (current)     hi (test)
    churn_rate                 80           700
    mean_residence_steps       50            50    <- unchanged
    max_size                 4000         40000
    occupancy                4000        ~35000
    mean row age          ~50 steps    ~50 steps   <- unchanged
    draws/row/step           0.75          0.08
    draws/row/lifetime       35.4           4.0

Mean AGE is what freshness means for policy lag, and it is held fixed -- only
reuse moves. This is a different axis from the ctrl_aug03 servo, which cut age
and left reuse alone. Admission lands at ~26% of the candidate pool, BELOW
fx_servo's 36%, so selectivity improves rather than degrades. toxic_min_draws
rides the lifetime-draw count (it was ~0.57x expected draws at 20/35.4).

FACTOR 2 -- fwd_frac. ctrl_aug03 ran fwd at 0.05; the reference runs use 0.20.
fwd is Z's ONLY gradient source, and across all five ctrl_aug03 arms
fwd/tb_resid_clipped sat at a persistent -0.31..-0.54 instead of hovering near
zero with sign crossings. Adam's scale-invariance says a 4x loss-weight cut
should not matter for a single-source parameter, so there is no mechanism in
hand -- this cell is the measurement, not a theory.

The extra fwd mass comes from bwd (0.75 -> 0.60), leaving REPLAY fixed at 0.20
across all four cells: replay is the variable factor 1 acts on, so it must not
move with factor 2.

No buffer servo and no frac controller anywhere in this battery -- the question
is whether correct sizing removes the need for either.
"""
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).parent
SRC = HERE.parent / 'ctrl_aug03' / 'fx_static.yaml'
TAG = 'size_aug03'
RESUME_STEP = 6680
STEP_BUDGET = 2000          # matched to the ctrl_aug03 comparison point (step 8680)

SIZING_HI = {'churn_rate': 700, 'max_size': 40000, 'mean_residence_steps': 50,
             'toxic_min_draws': 3}
FRACS_F05 = {'fwd': 0.05, 'bwd': 0.75, 'replay': 0.20}
FRACS_F20 = {'fwd': 0.20, 'bwd': 0.60, 'replay': 0.20}


def base():
    with SRC.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def naive_stage(cfg):
    for st in cfg['protocol']['stages']:
        if st['name'] == 'naive':
            return st
    raise KeyError("no 'naive' stage")


def assert_pinned_resume(cfg, name):
    """Every arm must start from ONE explicitly named checkpoint.

    The inherited mk_dev defaults are `continue_from_checkpoint: true` +
    `checkpoint_name: null`, which resolve to '{tag}_{run_name}_{problem}_running.pt'.
    That is right for a dev run you keep extending and wrong for a battery: a
    generator that forgets to override them gives each arm a DIFFERENT starting
    state, and the failure is invisible in the results. (Arms do not chain into
    each other -- run_name is unique, so each finds no 'running' and silently
    retrains phase 1 instead, which is a different wrong answer, not a safe one.)
    Cheap to assert, expensive to discover afterwards."""
    ck = cfg.get('checkpoint_name')
    if not ck:
        raise ValueError(f'{name}: checkpoint_name must name the shared resume point')
    if cfg.get('continue_from_checkpoint'):
        raise ValueError(f'{name}: continue_from_checkpoint must be False so a relaunch '
                         f'returns to {ck}, not to this arm\'s own rolling checkpoint')
    if cfg.get('reuse_prior'):
        raise ValueError(f'{name}: reuse_prior must be False -- it auto-refinds a '
                         f'per-run prior and bypasses the pinned resume')
    return cfg


def cell(sizing_hi, fracs, name, note):
    cfg = base()
    cfg['epochs'] = RESUME_STEP + STEP_BUDGET
    if sizing_hi:
        cfg['buffers']['replay_buffer'].update(copy.deepcopy(SIZING_HI))
    st = naive_stage(cfg)
    st['fracs'] = dict(fracs)
    st.pop('balance', None)          # no frac controller in this battery
    st.pop('buffer_servo', None)     # no servo either
    cfg['run_name'] = name
    cfg['tag'] = TAG
    assert_pinned_resume(cfg, name)
    with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)
    rb = cfg['buffers']['replay_buffer']
    occ = rb['churn_rate'] * rb['mean_residence_steps']
    print(f'wrote {name:14s} churn={rb["churn_rate"]:4d} tau={rb["mean_residence_steps"]} '
          f'max_size={rb["max_size"]:5d} occ~{occ:5d} draws/row/step={cfg["batch_size"] / occ:.3f} '
          f'fracs={fracs}')
    LOG.append({'run_name': name, 'tag': TAG, 'sizing': 'hi' if sizing_hi else 'lo',
                'fwd_frac': fracs['fwd'], 'fracs': dict(fracs),
                'churn_rate': rb['churn_rate'], 'max_size': rb['max_size'],
                'mean_residence_steps': rb['mean_residence_steps'],
                'predicted_occupancy': occ,
                'predicted_draws_per_row_step': round(cfg['batch_size'] / occ, 4),
                'epochs': cfg['epochs'], 'buffer_servo': False, 'balance': None,
                'note': note})


LOG = [{'run_name': 'fx_static', 'tag': 'ctrl_aug03', 'sizing': 'lo', 'fwd_frac': 0.05,
        'fracs': dict(FRACS_F05), 'churn_rate': 80, 'max_size': 4000,
        'mean_residence_steps': 50, 'predicted_occupancy': 4000,
        'predicted_draws_per_row_step': 0.708, 'epochs': 9080,
        'buffer_servo': False, 'balance': None,
        'note': 'ALREADY RUN as the ctrl_aug03 reference; the (lo, 0.05) cell'}]

cell(True, FRACS_F05, 'sz_hi_f05',
     'sizing fixed, fwd 0.05 -- does correct sizing alone lift the ratio above 1')
cell(False, FRACS_F20, 'sz_lo_f20',
     'current sizing, fwd 0.20 -- does fwd mass alone close the Z residual')
cell(True, FRACS_F20, 'sz_hi_f20',
     'both -- the reference-like configuration')

with (HERE / 'experiment_log.yaml').open('w', encoding='utf-8') as f:
    yaml.safe_dump(LOG, f, default_flow_style=False, sort_keys=False)
print(f'wrote experiment_log.yaml ({len(LOG)} cells, 1 pre-existing)')
