"""
Two eval-cadence readouts on the anchor path. Both are logging only: no route
changes behaviour, and a route that never produces the underlying quantity logs
exactly the keys it logged before.

  condition_best_energy_phys_*   the lambda=1 per-condition minimum
                                 (ConditionLogZTracker.best_energy_phys), logged
                                 beside the MIXED condition_best_energy_* only
                                 while it is its own stream (phys_is_alias False).
  anchor_funnel/<gate>           rows surviving each gate of
                                 Modeller.screen_and_admit_anchors, accumulated
                                 across calls and drained by log_buffer_stats.

    pytest tests/crystal/test_anchor_diagnostics.py
"""
import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling.train import Modeller  # noqa: E402


def _load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(__file__), filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# the screen_and_admit_anchors stub harness (FakeBatch rows, stub Modeller)
A = _load_sibling('anchor_currency_harness', 'test_anchor_currency.py')

FUNNEL = ('calls', 'health_blocked', 'candidates', 'in_window', 'warm', 'screened', 'confirmed')


# ------------------------------------------------ physical conditional minima

def _cond_stats(t):
    return Modeller.log_condition_log_z_stats(SimpleNamespace(condition_log_z=t, step_ind=1))


def _visited(energy_phys=None):
    """Conditions 0, 1 warm (ema_logw finite); condition 2 has a minimum but no
    log_w yet, so it is outside the `valid` mask both medians are taken over."""
    t = A.tracker(min_visits=1)
    t.update(torch.tensor([0, 1]), torch.tensor([0.0, 0.0]), step=0)
    t.update_best_energy(torch.tensor([0, 1, 2]), torch.tensor([1.0, 3.0, -500.0]),
                         energy_phys=energy_phys)
    return t


def test_phys_minima_are_absent_while_the_stream_is_an_alias():
    t = _visited()
    assert t.phys_is_alias, 'precondition'
    out = _cond_stats(t)
    assert out['condition_best_energy_median'] == 2.0
    assert not [k for k in out if 'best_energy_phys' in k]


def test_phys_minima_are_logged_over_the_same_conditions_once_separate():
    t = _visited(energy_phys=torch.tensor([10.0, 30.0, -900.0]))
    assert not t.phys_is_alias, 'precondition'
    out = _cond_stats(t)
    assert out['condition_best_energy_median'] == 2.0          # the mixture, unchanged
    assert out['condition_best_energy_phys_median'] == 20.0     # condition 2 masked out
    assert 'condition_best_energy_phys_hist' in out


# ------------------------------------------------------ anchor admission funnel

def _lambda_free_stub(best):
    """Warm condition 0 with mixed minimum `best`; returns the stub Modeller."""
    t = A.tracker(min_visits=1)
    A._warm(t)
    t.update_best_energy(torch.tensor([0]), torch.tensor([best]))
    m, _ = A._screen_stub(t, A._screen_rows(), flow=False)
    return m


def _screen(m, energy):
    Modeller.screen_and_admit_anchors(m, A._screen_rows(), torch.full((2,), 100.0),
                                      torch.as_tensor(energy), torch.zeros(2))


def _funnel(**counts):
    return {k: counts.get(k, 0) for k in FUNNEL}


def _report(funnel):
    """The real log_buffer_stats on a stub carrying only the funnel (no buffers)."""
    r = SimpleNamespace(bwd_sampling_mode='fwd', prior_churn={}, _anchor_funnel=funnel)
    return {k: v for k, v in Modeller.log_buffer_stats(r).items() if k.startswith('anchor_funnel/')}


def test_a_health_gate_block_counts_only_the_block():
    m = _lambda_free_stub(0.0)
    m.args.buffers.anchor_buffer.health_gate_ceiling = 0.5
    m.metric_tracker = SimpleNamespace(get=lambda *a: 10.0)     # |tb_resid_clipped| > 0.5
    _screen(m, [0.5, 50.0])
    assert m._anchor_funnel == _funnel(calls=1, health_blocked=1)


def test_a_batch_outside_the_window_reaches_the_window_test_and_stops():
    m = _lambda_free_stub(-100.0)                              # window: E < -99
    _screen(m, [0.5, 50.0])
    assert m._anchor_funnel == _funnel(calls=1, candidates=2)


def test_counts_accumulate_across_calls_and_the_report_drains_them():
    m = _lambda_free_stub(0.0)                                 # window: E < 1
    _screen(m, [0.5, 50.0])                                    # row 0 admitted
    _screen(m, [50.0, 50.0])                                   # nothing in the window
    assert m.last_anchor_admitted == 1
    want = _funnel(calls=2, candidates=4, in_window=1, warm=1, screened=1, confirmed=1)
    assert m._anchor_funnel == want

    assert _report(m._anchor_funnel) == {f'anchor_funnel/{k}': v for k, v in want.items()}
    assert m._anchor_funnel == _funnel(), 'the report must zero the counts it read'
    assert _report(m._anchor_funnel) == {f'anchor_funnel/{k}': 0 for k in FUNNEL}


def test_no_funnel_keys_on_a_run_that_never_screens():
    r = SimpleNamespace(bwd_sampling_mode='fwd', prior_churn={})
    assert not [k for k in Modeller.log_buffer_stats(r) if k.startswith('anchor_funnel/')]


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
