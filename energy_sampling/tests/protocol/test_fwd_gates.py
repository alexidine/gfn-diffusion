"""Modeller._fwd_gates -- the one place that decides whether a fused step runs
a forward ROLLOUT (fwd_ran) and whether the forward loss carries any WEIGHT
(fwd_active).

The two were separate expressions in two branches; the risk in unifying them is
that the off-cadence branch stops reproducing today's table. So the table is
pinned outright, both branches, including the force-refresh case where a branch
runs only to keep its rolling stats fresh and contributes zero weight.

The off-cadence drift trigger (stage.fwd_rollout_drift_max) ships at 0. Its
tests are the REFUSALS -- off by default, and the minimum two-step gap -- since
the gap is what stops a chronically high drift turning the cadence into
rollouts at 0 and 1 (mod N), a deterministic cadence change the key does not
claim to make.
"""
from types import MethodType, SimpleNamespace

import pytest

from energy_sampling.train import Modeller

THRESH = 0.01


def _m(every=0, fwd_frac=0.0, triggers=None, drift_std=None, dormant=False,
       prime_anchor=True):
    """prime_anchor pins the stage's cadence anchor at step 0, so `step_ind`
    reads as the absolute counter -- which is what the cadence tests below are
    about. The stage-entry tests at the bottom pass False to exercise the lazy
    derivation instead."""
    m = SimpleNamespace(
        step_ind=0, fwd_frac=fwd_frac, _replay_drift_std=drift_std,
        protocol=SimpleNamespace(
            stage=SimpleNamespace(name='equilibration',
                                  fwd_rollout_every=every,
                                  fwd_rollout_triggers=dict(triggers or {})),
            mode_dormant=lambda mode: dormant))
    m._rollout_trigger_fires = MethodType(Modeller._rollout_trigger_fires, m)
    m._rollout_trigger_reading = MethodType(Modeller._rollout_trigger_reading, m)
    m.ROLLOUT_TRIGGERS = Modeller.ROLLOUT_TRIGGERS
    m._fwd_gates = MethodType(Modeller._fwd_gates, m)
    if prime_anchor:
        m._cadence_anchor_stage, m._cadence_anchor = 'equilibration', 0
    return m


# ------------------------------------------------- no cadence: today's table

@pytest.mark.parametrize('fwd_frac,force_refresh,dormant,expected', [
    (0.5, False, False, (True, True)),      # trains
    (0.5, True, False, (True, True)),
    (0.0, False, False, (False, False)),    # below threshold, no refresh due
    (0.0, True, False, (True, False)),      # force-refresh only: runs, weight 0
    (0.0, True, True, (False, False)),      # dormant branch skips its refresh
])
def test_the_uncadenced_table_is_unchanged(fwd_frac, force_refresh, dormant, expected):
    m = _m(every=0, fwd_frac=fwd_frac, dormant=dormant)
    assert m._fwd_gates(THRESH, force_refresh) == expected


# ---------------------------------------------------------------- cadence

def test_the_cadence_runs_on_its_multiples_and_never_trains():
    m = _m(every=7, fwd_frac=0.0)
    ran = []
    for step in range(70):
        m.step_ind = step
        r, active = m._fwd_gates(THRESH, force_refresh=(step % 10 == 0))
        assert active is False, 'a cadenced stage pins fwd_frac at 0'
        if r:
            ran.append(step)
    assert ran == list(range(0, 70, 7))


def test_the_force_refresh_cannot_add_a_rollout_under_a_cadence():
    """At refresh_every 10 it would call the energy function every 10 steps
    regardless of the cadence, silently."""
    m = _m(every=7)
    m.step_ind = 3
    assert m._fwd_gates(THRESH, force_refresh=True)[0] is False


def test_a_cadenced_rollout_would_train_if_the_stage_gave_fwd_weight():
    """fwd_active is DERIVED from fwd_ran, so a stage that later unpins fwd
    trains on its rollout steps rather than needing a second gate. Inert today:
    every cadenced stage ships fracs.fwd 0."""
    m = _m(every=7, fwd_frac=0.5)
    m.step_ind = 14
    assert m._fwd_gates(THRESH, False) == (True, True)
    m.step_ind = 15
    assert m._fwd_gates(THRESH, False) == (False, False)


# ------------------------------------------------------------- drift trigger

def test_the_trigger_is_off_by_default_even_at_pathological_drift():
    m = _m(every=7, triggers=None, drift_std=99.0)
    ran = [s for s in range(70) if (setattr(m, 'step_ind', s) or m._fwd_gates(THRESH, False)[0])]
    assert ran == list(range(0, 70, 7))
    assert getattr(m, '_drift_trigger_count', 0) == 0


def test_the_trigger_holds_before_any_drift_has_been_measured():
    """_replay_drift_std is unset on step 1 of a fresh run and after a resume;
    a bare attribute read would abort the run, since the training loop's
    handler catches only RuntimeError/ValueError."""
    m = SimpleNamespace(
        step_ind=1, fwd_frac=0.0,
        protocol=SimpleNamespace(
            stage=SimpleNamespace(name='equilibration', fwd_rollout_every=7,
                                  fwd_rollout_triggers={'drift_std_max': 0.3}),
            mode_dormant=lambda mode: False))
    m._rollout_trigger_fires = MethodType(Modeller._rollout_trigger_fires, m)
    m._rollout_trigger_reading = MethodType(Modeller._rollout_trigger_reading, m)
    m.ROLLOUT_TRIGGERS = Modeller.ROLLOUT_TRIGGERS
    m._fwd_gates = MethodType(Modeller._fwd_gates, m)
    m._cadence_anchor_stage, m._cadence_anchor = 'equilibration', 0
    assert m._fwd_gates(THRESH, False) == (False, False)


def test_the_trigger_respects_the_two_step_gap():
    """The drift reading is taken BEFORE that step's admission, so without the
    gap a chronically high drift fires at step 1 of every window."""
    m = _m(every=7, triggers={'drift_std_max': 0.3}, drift_std=5.0)

    def n_fired():
        return sum((getattr(m, '_rollout_trigger_counts', None) or {}).values())

    ran, triggered, last = [], [], None
    for step in range(28):
        m.step_ind = step
        before = n_fired()
        if m._fwd_gates(THRESH, False)[0]:
            if n_fired() > before:
                assert last is not None and step - last >= 2, (step, last)
                triggered.append(step)
            ran.append(step)
            last = step
    assert set(range(0, 28, 7)) <= set(ran), 'the cadence itself is never blocked'
    assert triggered and n_fired() == len(triggered)


def test_the_trigger_holds_on_a_drift_under_the_bar():
    m = _m(every=7, triggers={'drift_std_max': 5.0}, drift_std=4.9)
    m.step_ind = 3
    assert m._fwd_gates(THRESH, False)[0] is False


def test_a_non_finite_reading_never_fires():
    m = _m(every=7, triggers={'drift_std_max': 0.3}, drift_std=float('nan'))
    m.step_ind = 3
    assert m._fwd_gates(THRESH, False)[0] is False


# ---------------------------------------------------------------------------
# The cadence is anchored to the STAGE, so its first fused step always rolls out.
# That step is the Z bootstrap and the replay buffer's first fill.
# ---------------------------------------------------------------------------

def test_the_first_step_of_a_stage_always_rolls_out_whatever_the_counter_says():
    """The regression this replaces: `step_ind % every` only looked right because
    the acceptance runs SKIP phase 1, putting stage entry at step 0 where
    0 % every == 0. A run that trains phase 1 and transitions at step 5000 gets
    5000 % 7 = 4 -- four fused steps against an MLE-era log Z before anything
    pinned it, and a replay buffer drawn from empty meanwhile."""
    m = _m(every=7, prime_anchor=False)
    m.step_ind = 5000
    assert m._fwd_gates(THRESH, False)[0] is True, 'stage entry must roll out'
    for offset in (1, 2, 6):
        m.step_ind = 5000 + offset
        assert m._fwd_gates(THRESH, False)[0] is False
    m.step_ind = 5007
    assert m._fwd_gates(THRESH, False)[0] is True, 'and the cadence is regular after it'


def test_a_new_stage_re_anchors_rather_than_inheriting_the_old_phase():
    m = _m(every=7, prime_anchor=False)
    m.step_ind = 5000
    m._fwd_gates(THRESH, False)
    m.protocol.stage.name = 'second_fused_stage'
    m.step_ind = 5003
    assert m._fwd_gates(THRESH, False)[0] is True, 'the incoming stage gets its own bootstrap'
    m.step_ind = 5004
    assert m._fwd_gates(THRESH, False)[0] is False


def test_a_mid_stage_resume_rolls_out_on_its_first_step():
    """Deliberate: the anchor is derived lazily, not checkpointed, so a restart
    re-pins log Z before training on it rather than waiting up to N-1 steps."""
    m = _m(every=20, prime_anchor=False)
    m.step_ind = 3117                      # wherever the checkpoint left off
    assert m._fwd_gates(THRESH, False)[0] is True


# ---------------------------------------------------------------------------
# The multi-reason trigger: fwd_rollout_every is the BACKSTOP, the bars set the
# cadence, and each bar reports under its own name.
# ---------------------------------------------------------------------------

def _trig(m):
    return dict(getattr(m, '_rollout_trigger_counts', None) or {})


def test_each_bar_fires_under_its_own_name():
    """'the cadence was cut short' and 'why' are different questions; a single
    counter answers neither."""
    m = _m(every=100, triggers={'drift_std_max': 5.0}, drift_std=9.0)
    m.step_ind = 40
    assert m._fwd_gates(THRESH, False)[0] is True
    assert _trig(m) == {'drift_std_max': 1}


def test_an_unreadable_bar_never_fires():
    """A missing measurement is not evidence of a problem. ess_min reads the
    metric tracker, which is empty before the first replay measurement."""
    m = _m(every=100, triggers={'ess_min': 0.5})
    m.metric_tracker = SimpleNamespace(get=lambda *a, **k: None)
    m.step_ind = 40
    assert m._fwd_gates(THRESH, False)[0] is False
    assert _trig(m) == {}


def test_below_bars_and_above_bars_have_opposite_senses():
    """ess_min fires when the reading FALLS under it; val_gap_max when it rises
    above. Getting one backwards would arm a trigger permanently."""
    low = _m(every=100, triggers={'ess_min': 0.5})
    low.metric_tracker = SimpleNamespace(get=lambda d, k, *a: 0.2 if k == 'policy_drift_ess_frac' else None)
    low.step_ind = 40
    assert low._fwd_gates(THRESH, False)[0] is True and _trig(low) == {'ess_min': 1}

    high = _m(every=100, triggers={'ess_min': 0.5})
    high.metric_tracker = SimpleNamespace(get=lambda d, k, *a: 0.9 if k == 'policy_drift_ess_frac' else None)
    high.step_ind = 40
    assert high._fwd_gates(THRESH, False)[0] is False


def test_there_is_no_z_bar_and_the_name_is_refused():
    """REGRESSION, twice over. A z bar was tried against sqrt(q*elapsed) -- a pure
    function of the step counter and a constant, i.e. the backstop period respelled
    in nats -- and then against the signed drift off birth_log_pf, which is only
    the P_F half: `learn_pb` re-scores log_pb on every draw too, so
    d log w = d log_pb - d log_pf and the second term alone is not it.

    Nothing measures the root between rollouts, because getting one IS the rollout.
    The unpinned Z interval is bounded open-loop by fwd_rollout_every, and an
    unknown bar name must be refused rather than silently ignored -- a bar that
    does nothing reads as a bar that is holding."""
    assert 'z_drift_max' not in dict(Modeller.ROLLOUT_TRIGGERS)
    from energy_sampling.protocol import Stage
    import copy, yaml, pathlib as _pl
    from energy_sampling import config_invariants
    cfg = yaml.safe_load((_pl.Path(__file__).resolve().parents[2] / 'configs'
                          / 'local_rr_sep07' / 'rr_n7_race.yaml').read_text(encoding='utf-8'))
    st = [s for s in config_invariants.active_stages(cfg)
          if s.get('train_mode') == 'fused'][0]
    assert 'z_drift_max' not in st['fwd_rollout_triggers'], 'no shipped z bar'
    bad = copy.deepcopy(st); bad['fwd_rollout_triggers'] = {'z_drift_max': 3.0}
    with pytest.raises(ValueError, match='unknown bars'):
        Stage(bad, 1)


def test_several_bars_tripping_at_once_each_count_once():
    m = _m(every=100, triggers={'drift_std_max': 5.0, 'ess_min': 0.5}, drift_std=9.0)
    m.metric_tracker = SimpleNamespace(get=lambda d, k, *a: 0.2 if k == 'policy_drift_ess_frac' else None)
    m.step_ind = 40
    assert m._fwd_gates(THRESH, False)[0] is True
    assert _trig(m) == {'drift_std_max': 1, 'ess_min': 1}


def test_an_empty_trigger_block_is_the_fixed_period_behaviour():
    m = _m(every=7, triggers={}, drift_std=99.0)
    for step in range(1, 7):
        m.step_ind = step
        assert m._fwd_gates(THRESH, False)[0] is False
    m.step_ind = 7
    assert m._fwd_gates(THRESH, False)[0] is True


def test_every_rollout_path_increments_the_cost_counter():
    """rollout/rate is energy calls per step -- the number the whole design is
    trying to cut -- so it must count the cadence, off-cadence triggers, and the
    un-cadenced baseline alike, or the savings are computed off a partial count."""
    m = _m(every=7, prime_anchor=False)
    m.step_ind = 0
    m._fwd_gates(THRESH, False)                              # stage-entry rollout
    for step in range(1, 15):
        m.step_ind = step
        m._fwd_gates(THRESH, False)
    assert m._rollout_count == 3, 'steps 0, 7, 14'

    trig = _m(every=1000, triggers={'drift_std_max': 5.0}, drift_std=9.0,
              prime_anchor=False)
    trig.step_ind = 0
    trig._fwd_gates(THRESH, False)                           # entry
    trig.step_ind = 40
    assert trig._fwd_gates(THRESH, False)[0] is True         # trigger, off cadence
    assert trig._rollout_count == 2, 'off-cadence rollouts count too'

    plain = _m(every=0, fwd_frac=1.0)                        # no cadence: every step
    for step in range(5):
        plain.step_ind = step
        plain._fwd_gates(THRESH, False)
    assert plain._rollout_count == 5, 'the 1.0-per-step baseline'
