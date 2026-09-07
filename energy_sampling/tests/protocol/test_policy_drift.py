"""replay/policy_drift_* -- how far the POLICY has moved off the rows it stored
(train.py Modeller._policy_drift_stats).

d_i = log_pf_now - birth_log_pf, over the drawn rows carrying a birth score.
The statistic is non-circular: neither side is touched by the Z fill or the
servo, unlike every Z-fit measure.

Two things here are easy to get wrong and silent when wrong. (1) The headline is
policy_drift_std, not ess_frac: d is a sum over T x data_ndim Gaussian factors,
so the Kish ESS is pinned at its 1/n floor across most of the operating range
and carries no gradation there. (2) The draw re-runs condition_samples, which
redraws log T / SG / Z'; on a route where any of those enters the condition
vector, birth and now are scored under DIFFERENT conditions and d measures the
redraw. The metric must close itself there rather than report the redraw as
drift.
"""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from energy_sampling.train import Modeller


def _bind(birth, redrawn=False):
    m = SimpleNamespace(
        replay_buffer=SimpleNamespace(
            birth_log_pf=torch.as_tensor(birth, dtype=torch.float32)),
        energy_function=SimpleNamespace(temperature_conditioning=redrawn),
        _replay_drift_std=None)
    m._condition_is_redrawn = MethodType(Modeller._condition_is_redrawn, m)
    m._policy_drift_stats = MethodType(Modeller._policy_drift_stats, m)
    return m


def _stats(m, inds, now):
    return m._policy_drift_stats(np.asarray(inds),
                                 torch.as_tensor(now, dtype=torch.float32))


def test_zero_drift_reads_as_zero_and_full_ess():
    birth = [-3.0, 1.5, 20.0, -100.0]
    m = _bind(birth)
    st = _stats(m, [0, 1, 2, 3], birth)
    assert st['policy_drift_std'] == pytest.approx(0.0)
    assert st['policy_drift_nats'] == pytest.approx(0.0)
    assert st['policy_drift_ess_frac'] == pytest.approx(1.0)
    assert st['policy_drift_covered_frac'] == pytest.approx(1.0)
    assert m._replay_drift_std == pytest.approx(0.0)


def test_a_constant_offset_is_not_drift():
    """Any UNIFORM scoring difference between get_traj_fwd and get_traj_replay
    shifts every d_i equally, so it cannot move the headline or the ESS."""
    birth = [-3.0, 1.5, 20.0, -100.0]
    st = _stats(_bind(birth), [0, 1, 2, 3], [b + 37.0 for b in birth])
    assert st['policy_drift_std'] == pytest.approx(0.0)
    assert st['policy_drift_ess_frac'] == pytest.approx(1.0)
    assert st['policy_drift_nats'] == pytest.approx(37.0)


def test_nan_births_are_excluded_and_covered_frac_says_so():
    """Eval-admitted rows are born NaN by design (the EMA model on the eval
    grid is not the training policy). They must drop out of d rather than read
    as permanent drift, and the coverage has to be visible."""
    birth = [0.0, float('nan'), 0.0, float('nan')]
    st = _stats(_bind(birth), [0, 1, 2, 3], [0.0, 5.0, 0.0, -5.0])
    assert st['policy_drift_covered_frac'] == pytest.approx(0.5)
    assert st['policy_drift_std'] == pytest.approx(0.0)


def test_thin_coverage_publishes_its_reason_rather_than_a_reading():
    """Below two covered rows nothing was measured, but 'uncovered' must not
    look like 'the replay branch is not running'."""
    m = _bind([0.0, float('nan'), float('nan'), float('nan')])
    st = _stats(m, [0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0])
    assert st == {'policy_drift_covered_frac': pytest.approx(0.25)}
    assert m._replay_drift_std is None, 'the trigger must not read a stale std'


def test_an_empty_draw_divides_by_nothing():
    m = _bind([0.0, 1.0])
    assert _stats(m, [], []) == {}
    assert m._replay_drift_std is None


def test_duplicate_draws_count_twice():
    """The prioritised draw is with replacement and _sample_indices repeat-tiles,
    so the statistic is a property of the BATCH THAT TRAINS, not of the buffer."""
    st = _stats(_bind([0.0, 0.0]), [0, 0, 0, 1], [0.0, 0.0, 0.0, 4.0])
    assert st['policy_drift_covered_frac'] == pytest.approx(1.0)
    assert st['policy_drift_std'] == pytest.approx(float(
        torch.tensor([0.0, 0.0, 0.0, 4.0], dtype=torch.float64).std(unbiased=True)))


def test_one_dominating_row_floors_the_ess_and_stays_finite():
    """d = 50 nat on one row of 100 -- a naive exp(d) is inf at production
    spreads; the max-shift keeps every reported value finite."""
    n = 100
    birth = [0.0] * n
    now = [0.0] * (n - 1) + [50.0]
    st = _stats(_bind(birth), list(range(n)), now)
    assert st['policy_drift_ess_frac'] == pytest.approx(1.0 / n, rel=0.01)
    assert all(np.isfinite(v) for v in st.values())


def test_std_separates_two_draws_where_the_ess_is_floored():
    """The merged correction: at production spreads ess_frac saturates at its
    1/n floor and stops carrying gradation, while std still resolves 2x."""
    g = torch.Generator().manual_seed(0)
    n = 400
    out = {}
    for scale in (3.0, 6.0):
        d = torch.randn(n, generator=g, dtype=torch.float32) * scale
        out[scale] = _stats(_bind([0.0] * n), list(range(n)), d)
    assert out[3.0]['policy_drift_ess_frac'] < 0.02
    assert out[6.0]['policy_drift_ess_frac'] < 0.02
    assert out[6.0]['policy_drift_std'] / out[3.0]['policy_drift_std'] == pytest.approx(2.0, rel=0.15)


def test_the_metric_closes_when_the_condition_is_redrawn():
    """With temperature_conditioning the draw re-samples log T into the
    condition vector, so d measures the redraw. The metric publishes nothing
    and the trigger stamp stays None even against heavy apparent drift."""
    m = _bind([0.0, 0.0, 0.0, 0.0], redrawn=True)
    assert _stats(m, [0, 1, 2, 3], [0.0, 9.0, -9.0, 30.0]) == {}
    assert m._replay_drift_std is None


@pytest.mark.parametrize('ef,expected', [
    (dict(temperature_conditioning=True), True),
    (dict(sg_conditioning=True, space_groups=[2, 14]), True),
    (dict(sg_conditioning=True, space_groups=[2]), False),
    (dict(zp_conditioning=True, z_primes=(1, 2)), True),
    (dict(zp_conditioning=True, z_primes=(1,)), False),
    # conditionality is not the test -- the REDRAW is
    (dict(embedding_conditioning=True, vector_conditioning=True), False),
])
def test_the_closure_predicate_tracks_the_redraw_not_conditionality(ef, expected):
    m = SimpleNamespace(energy_function=SimpleNamespace(**ef))
    m._condition_is_redrawn = MethodType(Modeller._condition_is_redrawn, m)
    assert m._condition_is_redrawn() is expected
