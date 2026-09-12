"""balance.anneal_cooldown_steps: the lexicographic anneal's cooldown
(protocol.Stage._parse_balance, StageProtocol._balance_tick).

Two phases, both measured from the LATER of the stage's first tick and its last
anneal event, both stamped in stage_ctrl (which rides the checkpoint):
  entry        (no anneal yet this stage) the rules are NOT evaluated -- no
               running best is seeded from the entry transient -- the nudge
               takes default_boost and the clean streak stays 0;
  post-anneal  the rules ARE evaluated (running bests keep tracking) but the
               clean streak is held at 0, so no anneal can fire.
0 is the pre-cooldown behaviour. The engine is a real StageProtocol over a
one-stage lexicographic protocol; only the metric tracker is a dict.
"""
import copy
import pathlib
from types import SimpleNamespace

import pytest
import yaml

from energy_sampling.protocol import Stage, StageProtocol, fresh_stage_ctrl

MK_DEV = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'
#: relative to its own running best; drift 0 so a test can read the best exactly
REL = {'metric': 'fwd/x', 'relative': 'best', 'margin': 1.5, 'drift': 0.0,
       'if_missing': 'violated', 'boost': {'replay': 1.0}}
LAMBDA = {'lambda_mix': {'target': 1.0, 'rate': 0.5}}
_ABSENT = object()


def _balance(cooldown=500, rules=None, anneal_coeffs=None):
    """default_boost is bwd and every rule boosts replay, so the logged boost
    says whether a rule was read and won."""
    node = {'kind': 'lexicographic', 'default_boost': {'bwd': 1.0},
            'rules': copy.deepcopy([REL] if rules is None else rules),
            'anneal_coeffs': copy.deepcopy(LAMBDA if anneal_coeffs is None else anneal_coeffs)}
    if cooldown is not _ABSENT:
        node['anneal_cooldown_steps'] = cooldown
    return node


def _spec(balance):
    return {'name': 's', 'train_mode': 'fused',
            'fracs': {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}, 'balance': balance}


def _stage(balance):
    return Stage(_spec(balance), 0)


class _Tracker:
    def __init__(self):
        self.values = {}

    def get(self, direction, name, default=None):
        return self.values.get(f'{direction}/{name}', default)


def _engine(balance, patience=2):
    args = SimpleNamespace(
        protocol='p', protocols=SimpleNamespace(p=SimpleNamespace(stages=[_spec(balance)])),
        controller=SimpleNamespace(anneal_patience=patience, beta=0.025, min_mode_frac=0.001,
                                   decay_rate=0.95),
        energy_config=SimpleNamespace(lambda_mix=0.01))
    m = SimpleNamespace(args=args, stage='s', stage_ctrl=fresh_stage_ctrl(),
                        metric_tracker=_Tracker(), step_ind=0,
                        fwd_frac=0.0, bwd_frac=0.5, replay_frac=0.5)
    return StageProtocol(m), m


def _tick(p, m, step, value):
    """One balance tick at `step`, with fwd/x reading `value`."""
    m.step_ind = step
    m.metric_tracker.values['fwd/x'] = value
    p._balance_tick()
    return p.ctrl


# ---------------------------------------------------------------- parsing

def test_parses_and_defaults_to_zero():
    assert _stage(_balance(_ABSENT)).balance['anneal_cooldown_steps'] == 0
    assert _stage(_balance(500)).balance['anneal_cooldown_steps'] == 500


@pytest.mark.parametrize('bad', [-1, 2.5, 500.0, True, '500', None])
def test_a_non_integer_or_negative_cooldown_is_refused(bad):
    with pytest.raises(ValueError, match='anneal_cooldown_steps must be an integer'):
        _stage(_balance(bad))


@pytest.mark.parametrize('kind', ['proportional', 'constraint', 'ratio', 'gated_ramp'])
@pytest.mark.parametrize('cooldown', [0, 500])
def test_every_other_kind_refuses_the_key_even_at_zero(kind, cooldown):
    """A cooldown nothing reads would read as an armed one."""
    with pytest.raises(ValueError, match='needs kind: lexicographic'):
        _stage({'kind': kind, 'anneal_cooldown_steps': cooldown})


def test_a_cooldown_with_nothing_to_anneal_is_refused():
    with pytest.raises(ValueError, match='nothing to anneal'):
        _stage(_balance(500, anneal_coeffs={}))
    # _anneal tightens only ABSOLUTE rules: a relative one carrying `anneal`
    # changes nothing either
    with pytest.raises(ValueError, match='nothing to anneal'):
        _stage(_balance(500, rules=[dict(REL, anneal={'rate': 0.9})], anneal_coeffs={}))
    assert _stage(_balance(0, anneal_coeffs={})).balance['anneal_cooldown_steps'] == 0
    absolute = {'metric': 'fwd/x', 'above': 1.0, 'anneal': {'rate': 0.9}, 'boost': 'replay'}
    assert _stage(_balance(500, rules=[absolute], anneal_coeffs={})) \
        .balance['anneal_cooldown_steps'] == 500


def test_mk_devs_var_conditioning_carries_the_cooldown_and_the_branch_rules():
    cfg = yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))
    st = Stage(cfg['protocols']['conditional_vargrad']['stages'][1], 1)
    b = st.balance
    assert b['anneal_cooldown_steps'] == 1000
    # 1 - r2 (scale-free unexplained fraction) is the pacing metric: it separates
    # 'the target got harder' from 'the sampler fell behind' (2026-09-12)
    assert [r['metric'] for r in b['rules']] == ['fwd/r2_unexplained', 'bwd/r2_unexplained']
    for r in b['rules']:
        assert (r['relative'], r['margin'], r['drift'], r['if_missing']) == 
            ('best', 1.1, 0.001, 'violated')
    assert b['anneal_coeffs']['lambda_mix'] == {'target': 1.0, 'rate': 0.8}


# ---------------------------------------------------------------- the tick

def test_the_entry_transient_is_never_read():
    """THE LATCH. The stage enters at a level of 0.5 it never returns to; the
    operating level is 5.0. A best captured at entry would hold every later
    tick violated (5.0 > 1.5 x 0.5) and the anneal would never fire."""
    p, m = _engine(_balance(500))
    c = _tick(p, m, 100, 0.5)                   # the first tick stamps the entry
    assert c['anneal_cooling'] == 1.0 and c['gr_entry_step'] == 100
    for step in range(110, 600, 10):
        c = _tick(p, m, step, 0.5)
        assert c['rules'].get(0, {}).get('best') is None, step
        assert c['anneal_streak'] == 0 and c['boost'] == 'bwd'
    assert m.bwd_frac > 0.5, 'the nudge goes to default_boost'
    c = _tick(p, m, 600, 5.0)                   # 500 steps after the first tick
    assert c['anneal_cooling'] == 0.0
    assert c['rules'][0]['best'] == pytest.approx(5.0), 'seeded at the operating level'
    assert c['anneal_streak'] == 1


def test_a_violated_rule_does_not_steer_during_the_entry_cooldown():
    rule = {'metric': 'fwd/x', 'above': 1.0, 'boost': {'replay': 1.0}}
    p, m = _engine(_balance(500, rules=[rule]))
    for step in (0, 250, 490):
        assert _tick(p, m, step, 5.0)['boost'] == 'bwd'
    assert _tick(p, m, 500, 5.0)['boost'] == 'replay'


def test_the_entry_cooldown_is_anchored_on_the_first_tick_not_on_step_zero():
    """Stage entry is not step 0 on a transition or a resumed stage."""
    p, m = _engine(_balance(500))
    _tick(p, m, 6510, 1.0)
    assert _tick(p, m, 7000, 1.0)['anneal_cooling'] == 1.0
    assert _tick(p, m, 7010, 1.0)['anneal_cooling'] == 0.0


def test_events_are_spaced_by_the_cooldown_while_the_rules_keep_tracking():
    p, m = _engine(_balance(500), patience=2)
    _tick(p, m, 0, 5.0)                          # entry stamp
    c = _tick(p, m, 500, 5.0)                    # seeds the best; clean: streak 1
    assert c['anneal_streak'] == 1 and c['anneal_events'] == 0
    c = _tick(p, m, 510, 5.0)                    # streak 2 = patience: the event
    assert c['anneal_events'] == 1 and c['last_anneal_step'] == 510
    assert c['coeffs']['lambda_mix']['val'] == pytest.approx(0.02)
    # POST-ANNEAL: the rule is read (its best follows the level down) but the
    # streak is held at 0 for 500 steps
    c = _tick(p, m, 520, 4.0)
    assert c['anneal_cooling'] == 1.0
    assert c['rules'][0]['best'] == pytest.approx(4.0), 'the running best still tracks'
    for step in range(530, 1010, 10):
        c = _tick(p, m, step, 4.0)
        assert c['anneal_streak'] == 0 and c['anneal_events'] == 1, step
    c = _tick(p, m, 1010, 4.0)                   # 500 steps after the event
    assert c['anneal_cooling'] == 0.0 and c['anneal_streak'] == 1
    c = _tick(p, m, 1020, 4.0)
    assert c['anneal_events'] == 2 and c['last_anneal_step'] == 1020
    assert c['coeffs']['lambda_mix']['val'] == pytest.approx(0.04)


def test_zero_is_the_pre_cooldown_behaviour():
    """Absent and 0 run the same ticks: a best from the first tick, the streak
    from the first clean tick, back-to-back events at the patience."""
    runs = []
    for cooldown in (_ABSENT, 0):
        p, m = _engine(_balance(cooldown), patience=2)
        for step, value in ((0, 5.0), (10, 5.0), (20, 9.0), (30, 5.0), (40, 5.0), (50, 5.0)):
            _tick(p, m, step, value)
        runs.append((copy.deepcopy(m.stage_ctrl), m.fwd_frac, m.bwd_frac, m.replay_frac))
    assert runs[0] == runs[1]
    c = runs[0][0]
    assert c['rules'][0]['best'] == pytest.approx(5.0)
    assert c['anneal_events'] == 2 and c['anneal_cooling'] == 0.0
    assert 'gr_entry_step' not in c, 'no stamp is taken without a cooldown'


def test_a_resume_keeps_both_windows():
    """stage_ctrl rides the checkpoint, so a restored stage stays inside the
    window it was in."""
    p, m = _engine(_balance(500), patience=2)
    _tick(p, m, 0, 5.0)
    p2, m2 = _engine(_balance(500), patience=2)
    m2.stage_ctrl = copy.deepcopy(m.stage_ctrl)            # saved inside the entry window
    c = _tick(p2, m2, 400, 0.5)
    assert c['anneal_cooling'] == 1.0 and c['rules'].get(0, {}).get('best') is None
    for step in (500, 510):
        _tick(p, m, step, 5.0)                             # the event at 510
    p3, m3 = _engine(_balance(500), patience=2)
    m3.stage_ctrl = copy.deepcopy(m.stage_ctrl)            # saved inside the post-anneal window
    assert _tick(p3, m3, 900, 5.0)['anneal_cooling'] == 1.0
    assert _tick(p3, m3, 1010, 5.0)['anneal_cooling'] == 0.0


def test_the_report_logs_the_events_and_the_cooling_flag_only_where_set():
    p, m = _engine(_balance(500), patience=2)
    _tick(p, m, 0, 5.0)
    out = p.report()
    assert out['protocol/anneal_cooling'] == 1.0 and out['protocol/anneal_events'] == 0
    for step in (500, 510):
        _tick(p, m, step, 5.0)
    out = p.report()
    # the flag is the tick's own state, read before its event: the tick that
    # fires was open, and the post-anneal window starts on the next one
    assert out['protocol/anneal_events'] == 1 and out['protocol/anneal_cooling'] == 0.0
    _tick(p, m, 520, 5.0)
    assert p.report()['protocol/anneal_cooling'] == 1.0
    p0, m0 = _engine(_balance(0))
    _tick(p0, m0, 0, 5.0)
    out0 = p0.report()
    assert 'protocol/anneal_cooling' not in out0 and out0['protocol/anneal_events'] == 0
