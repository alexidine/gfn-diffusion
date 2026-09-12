"""The eval rollout's replay admission: the gate, the origin tags, the metrics.

WHAT THE EVAL SITE DOES. evaluation() admits its own rollout to BOTH buffers.
The prior buffer is off-policy by design and wants it. The replay buffer is a
store of the training policy's own trajectories, and what arrives here is a
different sample: the EMA model, at eval_T, over eval_num_samples. Admission is
capped at churn_rate, so at eval_num_samples 10000 with churn_rate = batch 1000
it puts 1000 rows per eval against a forward intake of 1000 per 20 steps -- ~17%
of the buffer. Those rows are born with a NaN birth_log_pf and drop out of the
drift statistic, so the only trace of them was
`replay/policy_drift_covered_frac` sitting at 0.84.

WHY THE GATE IS TESTED AT THE SITE, not as a predicate. A predicate that answers
correctly while the call site ignores it is the failure this is worth testing
for at all, so `_admit_eval_rollout` -- which holds both admissions and the gate
between them -- is the unit here, called with the REAL method bound to a stub.

WHY THE NEGATIVE NEEDS A MUTATION CHECK. "manage_replay_buffer was not called"
passes just as well on a fixture that could never have called it. The gateless
twin below runs the same body without the guard on the same stub and must
ADMIT -- so the negative is a reading of the gate, not of the fixture.

    python -m pytest tests/protocol/test_eval_admission_gate.py
"""
import ast
import os
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from buffer import ORIGIN_BOOTSTRAP, ORIGIN_EVAL, ORIGIN_NAMES, ORIGIN_ROLLOUT  # noqa: E402
from train import Modeller, _origin_fracs  # noqa: E402

TRAIN_PY = os.path.join(_here, 'train.py')

#: What each manage_replay_buffer call site in train.py must tag its rows with,
#: keyed by the ENCLOSING function. Read from the source below, because the only
#: way to reach these calls at runtime is a real rollout on a real energy.
EXPECTED_ORIGIN = {
    # the plain fwd dispatch and the fused step: the training policy's own
    # rollout at the training discretizer
    'train_step': 'ORIGIN_ROLLOUT',
    'fused_train_step': 'ORIGIN_ROLLOUT',
    # z-calibration's extra rollout. freeze_policy zeroes the policy's GRADIENT,
    # not its forward pass, so the sample is the same one the fused step admits
    '_z_rollout_step': 'ORIGIN_ROLLOUT',
    # a one-off entry draw at eval_num_samples on the eval_T grid
    'bootstrap_z_by_rollout': 'ORIGIN_BOOTSTRAP',
    # the EMA model on the eval grid
    '_admit_eval_rollout': 'ORIGIN_EVAL',
}


# --------------------------------------------------------------------- fixture

def modeller(admit_from_eval='absent'):
    """A stub carrying only what `_admit_eval_rollout` reads, with the REAL
    method bound. 'absent' omits the key entirely -- the shape every config
    written before this gate existed has, and the one that must still admit."""
    rb = SimpleNamespace()
    if admit_from_eval != 'absent':
        rb.admit_from_eval = admit_from_eval
    m = SimpleNamespace(args=SimpleNamespace(buffers=SimpleNamespace(replay_buffer=rb)),
                        prior_calls=[], replay_calls=[])
    m.manage_prior_buffer = lambda batch: m.prior_calls.append(batch)
    m.manage_replay_buffer = (
        lambda stats, batch, on_policy=True, origin=ORIGIN_ROLLOUT:
        m.replay_calls.append({'stats': stats, 'batch': batch,
                               'on_policy': on_policy, 'origin': origin}))
    m._admit_eval_rollout = MethodType(Modeller._admit_eval_rollout, m)
    return m


def gateless(m, fwd_stats, sample_batch):
    """`_admit_eval_rollout` with the guard deleted -- the pre-gate behaviour,
    and the proof that the fixture CAN admit."""
    m.manage_prior_buffer(sample_batch)
    m.manage_replay_buffer(fwd_stats, sample_batch, on_policy=False,
                           origin=ORIGIN_EVAL)


STATS, BATCH = {'log_r': 'stats'}, 'batch'


# ------------------------------------------------------------------- the gate

@pytest.mark.parametrize('setting', ['absent', True])
def test_the_eval_rollout_is_admitted_by_default(setting):
    """Today's behaviour on every unconditional route, and what an older config
    (no key at all) must keep doing."""
    m = modeller(setting)
    m._admit_eval_rollout(STATS, BATCH)
    assert len(m.replay_calls) == 1
    assert m.replay_calls[0]['on_policy'] is False, 'the EMA model is not the policy'


def test_admit_from_eval_false_skips_only_the_replay_call():
    m = modeller(False)
    m._admit_eval_rollout(STATS, BATCH)
    assert m.replay_calls == []
    assert m.prior_calls == [BATCH], 'the prior buffer is not what this gates'


@pytest.mark.parametrize('setting', ['absent', True, False])
def test_the_prior_buffer_admission_is_unconditional(setting):
    m = modeller(setting)
    m._admit_eval_rollout(STATS, BATCH)
    assert m.prior_calls == [BATCH]


def test_the_negative_is_a_reading_of_the_gate_not_of_the_fixture():
    """MUTATION CHECK. The same stub, the same body, the guard removed: it
    admits. Without this, `replay_calls == []` above would pass on a fixture
    that was never able to record a call at all."""
    m = modeller(False)
    gateless(m, STATS, BATCH)
    assert len(m.replay_calls) == 1


def test_a_falsy_non_bool_still_disables():
    """The key is read through bool(): a config written `admit_from_eval: 0`
    must not be admitted on the strength of not being `False`."""
    m = modeller(0)
    m._admit_eval_rollout(STATS, BATCH)
    assert m.replay_calls == []


# ------------------------------------------------------------- the origin tags

def _call_sites():
    """Every `self.manage_replay_buffer(...)` call in train.py, with the name of
    the function it sits in. AST, not grep: a call split across lines (three of
    the five are) has no single line to match on."""
    tree = ast.parse(open(TRAIN_PY, encoding='utf-8').read())
    found = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.fn = None

        def visit_FunctionDef(self, node):
            prev, self.fn = self.fn, node.name
            self.generic_visit(node)
            self.fn = prev

        def visit_Call(self, node):
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr == 'manage_replay_buffer'
                    and isinstance(f.value, ast.Name) and f.value.id == 'self'):
                kw = {k.arg: k.value for k in node.keywords}
                found.append((self.fn, kw))
            self.generic_visit(node)

    V().visit(tree)
    return found


def test_every_call_site_tags_its_origin_explicitly():
    """The default is ORIGIN_ROLLOUT, so an untagged eval-like site is wrong and
    SILENT -- its rows simply read as forward-rollout rows. Requiring the
    keyword everywhere is what makes a new call site a test failure rather than
    a mislabelled cohort."""
    sites = _call_sites()
    assert len(sites) == 5, [fn for fn, _ in sites]
    untagged = [fn for fn, kw in sites if 'origin' not in kw]
    assert not untagged, f'manage_replay_buffer call sites with no origin=: {untagged}'


def test_each_call_site_tags_the_right_origin():
    got = {}
    for fn, kw in _call_sites():
        node = kw['origin']
        assert isinstance(node, ast.Name), f'{fn}: origin is not an ORIGIN_* constant'
        got[fn] = node.id
    assert got == EXPECTED_ORIGIN


def test_the_expected_tags_name_real_constants():
    """EXPECTED_ORIGIN above is spelled in source names; a typo there would make
    the test assert a constant that does not exist."""
    live = {'ORIGIN_ROLLOUT': ORIGIN_ROLLOUT, 'ORIGIN_EVAL': ORIGIN_EVAL,
            'ORIGIN_BOOTSTRAP': ORIGIN_BOOTSTRAP}
    assert set(EXPECTED_ORIGIN.values()) <= set(live)


# ---------------------------------------------------------------- the metrics

def test_origin_fracs_is_a_composition():
    origin = torch.tensor([ORIGIN_ROLLOUT] * 6 + [ORIGIN_EVAL] * 3 + [ORIGIN_BOOTSTRAP],
                          dtype=torch.int8)
    got = _origin_fracs(origin, 'replay_buffer_origin_frac')
    assert got == {'replay_buffer_origin_frac/rollout': 0.6,
                   'replay_buffer_origin_frac/eval': 0.3,
                   'replay_buffer_origin_frac/bootstrap': 0.1}
    assert sum(got.values()) == pytest.approx(1.0)


def test_every_origin_name_is_emitted_even_at_zero():
    """A cohort that empties must read as 0.0, not vanish: a key that disappears
    when its share goes to zero renders the interesting case as a gap."""
    got = _origin_fracs(torch.zeros(4, dtype=torch.int8), 'p')
    assert set(got) == {f'p/{name}' for name in ORIGIN_NAMES.values()}
    assert got['p/rollout'] == 1.0 and got['p/eval'] == 0.0
    assert sum(got.values()) == pytest.approx(1.0)


@pytest.mark.parametrize('column', [None, torch.zeros(0, dtype=torch.int8)])
def test_the_keys_are_absent_without_the_column(column):
    """A store written before `origin` existed must emit NOTHING rather than a
    fabricated all-rollout composition."""
    assert _origin_fracs(column, 'replay_buffer_origin_frac') == {}


# ------------------------------------------------------- the last training draw

def _draw_stub(origin):
    m = SimpleNamespace(replay_buffer=SimpleNamespace(origin=origin))
    m._note_replay_draw_origins = MethodType(Modeller._note_replay_draw_origins, m)
    return m


def test_the_draw_composition_reads_the_drawn_rows():
    m = _draw_stub(torch.tensor([ORIGIN_ROLLOUT, ORIGIN_EVAL, ORIGIN_EVAL,
                                 ORIGIN_BOOTSTRAP], dtype=torch.int8))
    m._note_replay_draw_origins([1, 2])          # both eval rows
    assert m._replay_draw_origin['replay_draw_origin_frac/eval'] == 1.0
    assert m._replay_draw_origin['replay_draw_origin_frac/rollout'] == 0.0


def test_the_draw_composition_is_empty_without_the_column():
    m = _draw_stub(None)
    m._note_replay_draw_origins([0, 1])
    assert m._replay_draw_origin == {}
