"""The held-out plateau gate, gates/heldout_flat (cfg heldout_gate).

progress_metrics.heldout_plateau: a trailing median over `smooth` evals, then 1 when the best
smoothed value of the last `evals` improves on the best before them by at most `nats`.
train.py::Modeller.update_heldout_gate resolves the source against the eval's metrics, publishes
the verdict through protocol.publish_gate, and abstains (publishes nothing, prints once per stage)
when the source is absent.

    python -m pytest -q tests/protocol/test_heldout_gate.py
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from progress_metrics import (_HELDOUT_GATE, HELDOUT_AUTO_SOURCES, heldout_plateau,
                              parse_heldout_gate, resolve_heldout_source)
from protocol import StageProtocol, fresh_stage_ctrl
from utils import MetricTracker

pytestmark = pytest.mark.fast

SPEC = dict(evals=6, nats=0.25, smooth=3)
EXIT = [{'metric': 'gates/heldout_flat', 'above': 0.5, 'patience': 1}]


def verdicts(series, **spec):
    spec = {**SPEC, **spec}
    state = {}
    return [heldout_plateau(state, v, spec['evals'], spec['nats'], spec['smooth'])['flat'] for v in series], state


def test_fires_after_a_held_out_minimum_followed_by_a_rise():
    rng = np.random.RandomState(0)
    # falls 0.2 nats/eval to a minimum at eval 15, then rises 0.1 nats/eval (overfitting), noise sd 0.05
    t = np.arange(30)
    series = np.where(t <= 15, 10.0 - 0.2 * t, 7.0 + 0.1 * (t - 15)) + 0.05 * rng.randn(30)
    out, _ = verdicts(series)
    assert not any(out[:16]), out
    first = out.index(1.0)
    assert 16 <= first <= 15 + SPEC['evals'], first
    assert all(out[first:]), out


def test_fires_on_a_plateau():
    rng = np.random.RandomState(1)
    series = 5.0 + 0.1 * rng.randn(20)
    out, _ = verdicts(series)
    # first possible verdict at eval smooth + evals (1-based)
    assert not any(out[:SPEC['smooth'] + SPEC['evals'] - 1])
    assert out[SPEC['smooth'] + SPEC['evals'] - 1] == 1.0
    assert all(out[SPEC['smooth'] + SPEC['evals'] - 1:])


@pytest.mark.parametrize('seed', range(5))
def test_does_not_fire_on_a_steadily_falling_noisy_series(seed):
    rng = np.random.RandomState(seed)
    # 0.15 nats/eval fall (0.9 over 6 evals), noise sd 0.15, plus isolated single-eval spikes (at least
    # `smooth` evals apart) that the median absorbs; two outliers in one window of 3 are not absorbed
    series = 20.0 - 0.15 * np.arange(80) + 0.15 * rng.randn(80)
    series[rng.randint(0, 3) + np.arange(0, 78, 13)] -= 1.5
    out, _ = verdicts(series)
    assert not any(out), [i for i, v in enumerate(out) if v]


def test_state_is_bounded():
    _, state = verdicts(np.linspace(10.0, 0.0, 500))
    assert len(state['raw']) == SPEC['smooth'] and len(state['smoothed']) == SPEC['evals']
    assert set(state) == {'raw', 'smoothed', 'best_before'}


def test_auto_source_order_and_named_source():
    full = {k: float(i) for i, k in enumerate(HELDOUT_AUTO_SOURCES)}
    for i, k in enumerate(HELDOUT_AUTO_SOURCES):
        present = {kk: v for kk, v in full.items() if HELDOUT_AUTO_SOURCES.index(kk) >= i}
        assert resolve_heldout_source('auto', present) == k
    assert resolve_heldout_source('auto', {'eval_bwd/mle': 1.0}) is None
    assert resolve_heldout_source('eval_bwd/mle', {'eval_bwd/mle': 1.0}) == 'eval_bwd/mle'
    assert resolve_heldout_source('eval_bwd/mle', full) is None


@pytest.mark.parametrize('node', [{'evals': 0}, {'smooth': 1.5}, {'nats': -0.1}, {'nats': True},
                                  {'source': ''}, {'window': 3}, [1, 2]])
def test_parse_refuses_malformed_blocks(node):
    with pytest.raises((TypeError, ValueError)):
        parse_heldout_gate(node)


def test_parse_defaults_and_invariant_rule():
    import config_invariants as ci
    assert parse_heldout_gate(None) == _HELDOUT_GATE
    assert parse_heldout_gate(SimpleNamespace(evals=4)) == {**_HELDOUT_GATE, 'evals': 4}
    assert ci.heldout_gate_is_well_formed({}) == []
    bad = ci.heldout_gate_is_well_formed({'heldout_gate': {'evals': 0}})
    assert bad and bad[0].severity == ci.ERROR
    # published from the eval path, so exit patience on it is counted in eval_period
    assert ci._exit_metric_cadence({'eval_period': 250}, 'gates/heldout_flat') == 250.0


def test_mk_dev_carries_the_block_and_no_stage_exits_on_it():
    path = Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    assert parse_heldout_gate(cfg['heldout_gate']) == _HELDOUT_GATE
    stages = [st for proto in cfg['protocols'].values() for st in (proto or {}).get('stages') or []]
    assert stages
    assert not [st['name'] for st in stages for term in st.get('exit') or []
                if term.get('metric') == 'gates/heldout_flat']


# ------------------------------------------------- Modeller.update_heldout_gate on a real engine

def engine(exit_block):
    stages = [{'name': 's0', 'train_mode': 'bwd', 'bwd_sampling_mode': 'dataset', 'exit': exit_block},
              {'name': 's1', 'train_mode': 'bwd', 'bwd_sampling_mode': 'dataset'}]
    args = SimpleNamespace(protocol='p', grow_batch_size=False,
                           protocols=SimpleNamespace(p=SimpleNamespace(stages=stages)),
                           heldout_gate=dict(SPEC))
    m = SimpleNamespace(
        args=args, stage='s0', stage_ctrl=fresh_stage_ctrl(),
        metric_tracker=MetricTracker(period=25.0), step_ind=0,
        combo_loss_record=[], batch_sizer=None,
        batch_size_oom_ceiling=None, batch_size_oom_ceiling_at=None,
        batch_size_oom_min=None, _runaway_last_cut=None, _runaway_unresponsive_stage=None,
        _accum_floor_warned_stage=None, batch_size_last_grow=0,
        fwd_frac=0.0, bwd_frac=1.0, replay_frac=0.0,
        init_schedulers_optimizers=lambda: None, set_loss_coeffs=lambda: None,
        lr_controller=SimpleNamespace(on_stage_change=lambda: 0),
        grad_guard=SimpleNamespace(refresh=lambda reason=None: None),
        checkpointer=SimpleNamespace(save=lambda tag: None, save_buffers=lambda: None))
    p = StageProtocol(m)
    m.protocol = p
    return p, m


def run_evals(m, p, readouts, eval_period=250):
    """One evaluation per entry: update_heldout_gate on the readouts, then the tick and maybe_advance
    the eval would reach. Returns the eval index at which the stage advanced, or None."""
    from train import Modeller
    for i, metrics in enumerate(readouts):
        m.step_ind += eval_period
        Modeller.update_heldout_gate(m, dict(metrics))
        p.tick()
        if p.maybe_advance({}):
            return i
    return None


def test_a_held_out_minimum_advances_a_stage_exiting_on_heldout_flat():
    p, m = engine(EXIT)
    t = np.arange(40)
    # the auto source takes eval_test/mle_ref over eval_test/mle and the held-out rows
    series = np.where(t <= 15, 10.0 - 0.2 * t, 7.0 + 0.1 * (t - 15))
    readouts = [{'eval_test/mle_ref': v, 'eval_test/mle': 0.0, 'eval_bwd/mle_heldout_rows': 0.0} for v in series]
    at = run_evals(m, p, readouts)
    assert at is not None and 16 <= at <= 15 + SPEC['evals'], at
    assert p.stage.name == 's1'


def test_zero_verdicts_are_published():
    p, m = engine(EXIT)
    run_evals(m, p, [{'eval_bwd/mle_heldout_rows': 10.0 - i} for i in range(3)])
    assert p.ctrl['gates']['heldout_flat'] == 0.0
    assert p.ctrl['gate_written']['heldout_flat'] == m.step_ind
    assert p.ctrl['gate_state']['heldout']['source'] == 'eval_bwd/mle_heldout_rows'


def test_abstains_loudly_once_per_stage_on_a_missing_source(capsys):
    p, m = engine(EXIT)
    # an SDE policy writes no density readouts
    at = run_evals(m, p, [{'eval_bwd/mle_heldout_gap': 0.0}] * 30)
    assert at is None and p.stage.name == 's0'
    assert 'heldout_flat' not in p.ctrl['gates'] and 'heldout_flat' not in p.ctrl.get('gate_written', {})
    out = capsys.readouterr().out
    assert out.count('heldout_gate ABSTAINS') == 1, out
    assert 'cannot fire' in out
