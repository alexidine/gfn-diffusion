"""The NF-route row store (row_store.py; docs/design/unified_store.md), on hand-built stores.

Draw tests check frequencies against the q of the spec recomputed here in numpy, not
against draw_probs, so a wrong formula in the module cannot pass by agreeing with itself.
"""
import io
import math
import os
import sys

import numpy as np
import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here)):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling.row_store import (  # noqa: E402
    DATA_TAGS, FORMAT_VERSION, TAG_ANCHOR, TAG_CHILD, TAG_GENERATED, TAG_ONPOLICY,
    RowStore, RowStoreError)

pytestmark = pytest.mark.fast   # torch imported; CPU tensors of a few hundred rows, nothing built

SET = dict(bar=1.0, eps=0.1, cap_per_condition=100, max_rows=1000, records_kept=2,
           kappa=1.0, floor_frac=0.25, uniform_frac=0.1, admit_reward_min=None)


def make(n_cond=2, dim=3, cond_dim=2, **kw):
    return RowStore(dim, n_cond, cond_dim, 'cpu', **{**SET, **kw})


def zero_q(x, c):
    return torch.zeros(x.shape[0])


def cond_of(cid, cond_dim=2):
    """A condition that is a function of cid."""
    cid = torch.as_tensor(cid, dtype=torch.float32).reshape(-1, 1)
    return cid.expand(-1, cond_dim) * torch.arange(1, cond_dim + 1, dtype=torch.float32)


def marked_x(ids, dim=3):
    """States whose first coordinate identifies the row."""
    ids = torch.as_tensor(ids, dtype=torch.float32).reshape(-1, 1)
    return torch.cat([ids, torch.zeros(ids.shape[0], dim - 1)], 1)


def seeded(log_r, cid=None, held_out=None, tag=TAG_CHILD, n_cond=2, **kw):
    """A store seeded with the given rows, refreshed once with log q = 0."""
    n = len(log_r)
    st = make(n_cond=n_cond, **kw)
    cid = torch.zeros(n, dtype=torch.int64) if cid is None else torch.as_tensor(cid)
    st.seed(cid, marked_x(range(n)), torch.tensor(log_r, dtype=torch.float32), tag,
            torch.zeros(n, dtype=torch.bool) if held_out is None else torch.as_tensor(held_out))
    st.set_conditions(cid, cond_of(cid))
    st.refresh(zero_q, None, step=0, chunk=4)
    return st


def spec_q(s, floor_frac, uniform_frac, kappa):
    """q of the spec, from |r| over the drawable rows."""
    s = np.asarray(s, dtype=np.float64)
    m = np.sort(s)[(len(s) - 1) // 2]          # lower median, as torch.median
    f = max(floor_frac * m, 1e-12)
    sp = np.maximum(s, f) ** kappa
    P = sp / sp.sum()
    return (1 - uniform_frac) * P + uniform_frac / len(s)


def chi2(counts, probs):
    exp = probs * counts.sum()
    return float(((counts - exp) ** 2 / exp).sum())


# ---- seeding, layout, held-out ---------------------------------------------------------

def test_seed_puts_held_out_last_protects_everything_and_ignores_held_out_for_records():
    st = make()
    held = torch.tensor([True, False, True, False, False])
    st.seed(torch.tensor([0, 0, 1, 1, 0]), marked_x(range(5)), torch.tensor([50., 1., 60., 2., 3.]),
            torch.tensor([TAG_CHILD, TAG_ANCHOR, TAG_CHILD, TAG_CHILD, TAG_ANCHOR]), held)
    assert st.n_mle == 3 and st.n_seed == 5 and len(st) == 5
    assert st.x[:, 0].tolist() == [1., 3., 4., 0., 2.]          # stable, held-out last
    assert st.held_out.tolist() == [False, False, False, True, True]
    assert bool(st.protected.all())
    assert bool(torch.isnan(st.log_q).all()) and st.q_step.tolist() == [-1] * 5
    assert st.best_log_r.tolist() == [3.0, 2.0]                 # 50 and 60 are held out


def test_seed_refuses_non_data_tags_and_a_second_seed():
    st = make()
    with pytest.raises(RowStoreError):
        st.seed([0], marked_x([0]), [1.0], TAG_ONPOLICY, False)
    st.seed([0], marked_x([0]), [1.0], TAG_ANCHOR, False)
    with pytest.raises(RowStoreError):
        st.seed([0], marked_x([1]), [1.0], TAG_ANCHOR, False)


def test_admit_refuses_data_tags():
    st = seeded([1.0, 2.0])
    for t in DATA_TAGS:
        with pytest.raises(RowStoreError):
            st.admit([0], marked_x([9]), [50.], [0.], None, t, 1, torch.zeros(2), None)
    assert len(st) == 2


def test_unscored_and_held_out_rows_are_never_drawn_for_training():
    st = make()
    held = torch.tensor([False, False, False, True, True])
    st.seed(torch.zeros(5, dtype=torch.int64), marked_x(range(5)), torch.tensor([1., 2., 3., 40., 80.]),
            TAG_CHILD, held)
    lv = torch.zeros(2)
    assert st.draw_tb(10, lv) is None                           # never scored: log q NaN
    st.refresh(zero_q, lv, step=3, chunk=2)
    g = torch.Generator().manual_seed(0)
    idx, w, stats = st.draw_tb(20000, lv, generator=g)
    assert not bool(st.held_out[idx].any())
    assert stats['replay/is_elig_frac'] == 1.0
    mle = st.draw_mle(20000, generator=g)
    assert int(mle.max()) < st.n_mle and not bool(st.held_out[mle].any())


def test_mle_draw_is_uniform_over_data_rows_only():
    st = seeded([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], held_out=[False, False, False, False, True, True])
    st.admit([0, 0, 0], marked_x([10, 11, 12]), [50., 60., -40.], [0., 0., 0.], None,
             TAG_ONPOLICY, 1, torch.zeros(2), None)
    assert len(st) == 9 and st.n_mle == 4
    idx = st.draw_mle(40000, generator=torch.Generator().manual_seed(1))
    assert bool(((st.tag[idx] == TAG_ANCHOR) | (st.tag[idx] == TAG_CHILD)).all())
    assert not bool(st.held_out[idx].any())
    counts = torch.bincount(idx, minlength=st.n_mle).numpy().astype(float)
    assert counts.shape == (4,)
    assert chi2(counts, np.full(4, 0.25)) < 16.3                # chi2(3) 0.999 quantile


# ---- TB priority draw --------------------------------------------------------------------

def test_priority_draw_frequencies_and_is_weights_match_the_spec():
    log_r = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    st = seeded(log_r + [30.0, 40.0], held_out=[False] * 6 + [True, True])
    lv = torch.zeros(2)
    q = spec_q(np.abs(log_r), 0.25, 0.1, 1.0)
    rows, q_mod = st.draw_probs(lv)
    assert np.allclose(q_mod.numpy(), q, rtol=0, atol=1e-12)
    N = 200000
    idx, w, stats = st.draw_tb(N, lv, generator=torch.Generator().manual_seed(2))
    counts = torch.bincount(idx, minlength=6).numpy().astype(float)
    assert counts.shape == (6,)
    assert chi2(counts, q) < 20.5                               # chi2(5) 0.999 quantile
    assert torch.allclose(w, torch.as_tensor((1.0 / 6) / q[idx.numpy()]), rtol=0, atol=1e-12)
    # the IS identity: sum_E q w f = mean_E f exactly; the empirical mean of w f approaches it
    f = np.array([3.0, -1.0, 7.0, 0.5, 2.0, 11.0])
    assert abs(float((q * ((1.0 / 6) / q) * f).sum()) - f.mean()) < 1e-12
    assert abs(float((w.numpy() * f[idx.numpy()]).mean()) - f.mean()) < 0.05
    wn = w.numpy()
    assert stats['replay/is_ess_frac'] == pytest.approx(wn.sum() ** 2 / (N * (wn ** 2).sum()), rel=1e-9)
    assert 0.0 < stats['replay/is_ess_frac'] <= 1.0
    assert stats['replay/is_w_max_ratio'] == pytest.approx(wn.max() / wn.mean(), rel=1e-9)


def test_floor_and_uniform_share_keep_every_drawable_row_drawable_and_bound_w():
    log_r = [0.0, 0.0, 1e-3, 3.0, 50.0]
    st = seeded(log_r, kappa=3.0, uniform_frac=0.2, floor_frac=0.5)
    lv = torch.zeros(2)
    rows, q = st.draw_probs(lv)
    assert float(q.min()) >= 0.2 / 5 - 1e-15
    _, w, _ = st.draw_tb(5000, lv, generator=torch.Generator().manual_seed(3))
    assert float(w.max()) <= 1 / 0.2 + 1e-9
    # no uniform share: the floor alone keeps the zero-residual rows drawable, at ratio (f / max s)^kappa
    st.configure(uniform_frac=0.0)
    rows, q = st.draw_probs(lv)
    f = 0.5 * 1e-3                                               # floor_frac x lower median (1e-3)
    assert float(q.min()) > 0.0
    assert float(q.min() / q.max()) == pytest.approx((f / 50.0) ** 3, rel=1e-9)
    # kappa 0 is uniform whatever the residuals
    st.configure(kappa=0.0, uniform_frac=0.1)
    rows, q = st.draw_probs(lv)
    assert torch.allclose(q, torch.full_like(q, 0.2))


def test_priority_follows_a_live_level_shift():
    st = seeded([0.0, 8.0], cid=[0, 1])
    rows, q0 = st.draw_probs(torch.tensor([0.0, 0.0]))
    assert float(q0[1]) > float(q0[0])                           # row 1 has |r| = 8
    rows, q1 = st.draw_probs(torch.tensor([-8.0, 8.0]))           # now row 0 has |r| = 8, row 1 has 0
    assert float(q1[0]) > float(q1[1])
    assert np.allclose(q1.numpy(), spec_q([8.0, 0.0], 0.25, 0.1, 1.0), atol=1e-12)
    rows, _ = st.draw_probs(torch.tensor([0.0, float('nan')]))  # no level for condition 1: not drawable
    assert rows.tolist() == [0]
    with pytest.raises(RowStoreError):
        st.draw_probs(torch.zeros(3))


# ---- refresh and touch ---------------------------------------------------------------------

def test_refresh_recomputes_log_q_in_chunks_and_residuals_follow_the_supplied_level():
    st = make(n_cond=3)
    cid = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    st.seed(cid, marked_x(range(10)), torch.arange(10, dtype=torch.float32), TAG_CHILD, False)
    calls = []

    def fn(x, c):
        calls.append(x.shape[0])
        return 2.0 * x[:, 0] + c.float()

    lv = torch.tensor([0.5, -1.0, 2.0])
    out = st.refresh(fn, lv, step=17, chunk=3)
    assert calls == [3, 3, 3, 1]
    assert st.eval_idx == 1 and st.q_step.tolist() == [17] * 10
    assert out['rows'] == 10 and out['nonfinite'] == 0 and out['seconds'] >= 0.0
    expect = st.log_r - (2.0 * st.x[:, 0] + st.cid.float()) - lv[st.cid]
    assert torch.allclose(st.resid(lv), expect)
    assert out['resid_med'] == pytest.approx(float(expect.abs().median()))
    st.refresh(lambda x, c: torch.where(x[:, 0] < 2, torch.full((x.shape[0],), math.nan), x[:, 0]), lv, 18, 4)
    assert st.eval_idx == 2 and st.last_refresh['nonfinite'] == 2
    with pytest.raises(RowStoreError):
        st.refresh(zero_q, lv, 0, chunk=0)


def test_touch_writes_log_q_and_refuses_a_stale_version():
    st = seeded([1.0, 2.0, 3.0])
    lv = torch.zeros(2)
    idx, _, _ = st.draw_tb(4, lv, generator=torch.Generator().manual_seed(4))
    st.touch(idx, st.version, torch.full((4,), 0.25), step=9)
    assert bool((st.log_q[idx] == 0.25).all()) and bool((st.q_step[idx] == 9).all())
    v0 = st.version
    st.admit([0], marked_x([10]), [-20.0], [0.0], None, TAG_ONPOLICY, 10, lv, None)   # append only
    assert st.version == v0
    st.configure(cap_per_condition=0)
    st.admit([0], marked_x([11]), [-30.0], [0.0], None, TAG_ONPOLICY, 11, lv, None)   # evicts row 3
    assert st.version == v0 + 1
    with pytest.raises(RowStoreError):
        st.touch(idx, v0, torch.zeros(4), step=12)


def test_evicting_only_this_calls_candidates_keeps_the_version():
    st = seeded([1.0, 2.0, 3.0], cap_per_condition=0)              # only protected rows stored
    lv = torch.zeros(2)
    idx, _, _ = st.draw_tb(4, lv, generator=torch.Generator().manual_seed(4))
    v0 = st.version
    out = st.admit([0, 0], marked_x([10, 11]), [-20.0, -30.0], [0.0, 0.0], None, TAG_ONPOLICY, 1, lv, None)
    assert out['admitted'] == 2 and out['evicted_cap'] == 2 and len(st) == 3
    assert st.version == v0                                         # no stored row moved
    st.touch(idx, v0, torch.zeros(4), step=2)


# ---- admission and records -----------------------------------------------------------------

def test_admission_bar_record_strictness_ties_floor_and_non_finite():
    st = seeded([5.0, 1.0], cid=[0, 1], bar=1.0)               # best log R: c0 = 5, c1 = 1
    lv = torch.zeros(2)
    log_r = torch.tensor([0.5, -2.0, 5.0, 4.0, 9.0, 9.0, 9.5, 3.0, 3.0])
    log_q = torch.tensor([0.0, 0.0, 0.0, 3.5, float('nan'), 8.5, 0.0, 2.5, 2.8])
    cid = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1])
    ok = torch.tensor([True, True, True, True, True, True, False, True, True])
    out = st.admit(cid, marked_x(range(100, 109)), log_r, log_q, ok, TAG_ONPOLICY, 1, lv, cond_of(cid))
    added = {int(v): bool(p) for v, p in zip(st.x[st.n_seed:, 0].tolist(), st.protected[st.n_seed:])}
    # 100: |r| 0.5 < bar, no record -> out; 101: |r| 2 -> in; 102: log R 5 == best, |r| 5 -> in, not a record;
    # 103: |r| 0.5, 4 < best -> out; 104: log q NaN -> out; 105: record (9 > 5, finite) with |r| 0.5 -> in, protected;
    # 106: 9.5 fails the floor -> out and not a record; 107/108: tie at 3 > 1 for c1 -> only 107 is the record
    assert added == {101: False, 102: False, 105: True, 107: True}
    assert out['admitted'] == 4 and out['records'] == 2 and out['rejected_floor'] == 1
    assert st.best_log_r.tolist() == [9.0, 3.0]
    st.admit([0], marked_x([200]), [9.0], [8.5], None, TAG_ONPOLICY, 2, lv, None)   # equal, not strict
    assert len(st) == 6 and st.best_log_r.tolist() == [9.0, 3.0]


def test_admission_bar_is_inclusive():
    st = seeded([5.0, 1.0], cid=[0, 1], bar=1.0)
    out = st.admit([0, 0], marked_x([100, 101]), [1.0, 0.75], [0.0, 0.0], None, TAG_ONPOLICY, 1,
                   torch.zeros(2), None)
    assert out['admitted'] == 1 and out['records'] == 0 and st.x[-1, 0] == 100.0   # |r| == bar is admitted


def test_condition_mismatch_raises_before_any_row_moves():
    st = seeded([1.0, 2.0], cid=[0, 1])
    with pytest.raises(RowStoreError):
        st.set_conditions([0], torch.tensor([[9.0, 9.0]]))
    with pytest.raises(RowStoreError):
        st.set_conditions([1, 1], torch.tensor([[1.0, 2.0], [1.0, 2.5]]))
    with pytest.raises(RowStoreError):
        st.admit([0], marked_x([5]), [50.], [0.], None, TAG_ONPOLICY, 1, torch.zeros(2),
                 torch.tensor([[3.0, 3.0]]))
    assert len(st) == 2


def test_records_survive_a_zero_cap_and_demotion_drops_the_oldest():
    st = seeded([1.0, 1.5], records_kept=2, cap_per_condition=0, bar=0.0)
    lv = torch.zeros(2)
    for k, lr in enumerate([6.0, 7.0]):
        out = st.admit([0, 0], marked_x([10 + k, 20 + k]), [lr, -5.0], [0.0, 0.0], None, TAG_ONPOLICY, k, lv, None)
        assert out['records'] == 1 and out['evicted_cap'] == 1      # the non-record goes, the record stays
    assert st.x[st.n_seed:, 0].tolist() == [10., 11.] and bool(st.protected.all())
    out = st.admit([0], marked_x([12]), [8.0], [0.0], None, TAG_ONPOLICY, 2, lv, None)
    assert out['demoted'] == 1 and out['evicted_cap'] == 1         # record 6 lost protection, then the cap took it
    assert st.x[st.n_seed:, 0].tolist() == [11., 12.]
    assert bool(st.protected[:st.n_seed].all())                     # data rows are never demoted


def test_caps_hold_and_admission_never_stalls_when_caps_sum_past_max_rows():
    n_cond, cap, max_rows, kept = 4, 30, 50, 2
    st = seeded([100.0] * n_cond, cid=list(range(n_cond)), n_cond=n_cond,
                cap_per_condition=cap, max_rows=max_rows, records_kept=kept, bar=1.0)
    lv = torch.zeros(n_cond)
    g = torch.Generator().manual_seed(5)
    for k in range(60):
        cid = torch.randint(n_cond, (20,), generator=g)
        if k % 3 == 0:
            cid[:] = 0                                              # one condition overloaded
        log_r = -1.0 - 10.0 * torch.rand(20, generator=g)
        if k % 7 == 0:
            log_r[0] = 100.0 + k                                    # a record now and then
        out = st.admit(cid, marked_x(torch.arange(20) + 1000 * k), log_r, torch.zeros(20), None,
                       TAG_ONPOLICY, k, lv, None, generator=g)
        assert out['admitted'] >= 19
        unprot = (~st.protected) & (~st.held_out)
        per = torch.bincount(st.cid[unprot], minlength=n_cond)
        assert int(per.max()) <= cap and int(unprot.sum()) <= max_rows
        assert int(st.protected.sum()) <= st.n_seed + kept * n_cond
        assert bool(st.protected[:st.n_seed].all())
    assert int(((~st.protected) & (~st.held_out)).sum()) == max_rows


def test_global_pass_runs_after_the_per_condition_pass():
    st = seeded([100.0, 100.0], cid=[0, 1], cap_per_condition=3, max_rows=4, bar=0.0)
    cid = torch.tensor([0, 0, 0, 0, 1, 1])
    out = st.admit(cid, marked_x(range(10, 16)), -torch.arange(1.0, 7.0), torch.zeros(6), None,
                   TAG_ONPOLICY, 1, torch.zeros(2), None, generator=torch.Generator().manual_seed(6))
    assert out['admitted'] == 6 and out['evicted_cap'] == 1 and out['evicted_global'] == 1
    assert len(st) == 2 + 4


def test_eviction_frequency_is_proportional_to_inverse_residual_plus_eps():
    r = np.array([0.02, 1.0, 2.0, 9.0])                          # 0.02 << eps: eps sets its weight
    eps = 0.1
    p = (1 / (r + eps)) / (1 / (r + eps)).sum()
    counts = np.zeros(4)
    trials = 3000
    for t in range(trials):
        st = seeded([100.0], n_cond=1, cap_per_condition=3, bar=0.0, eps=eps)
        st.admit(torch.zeros(4, dtype=torch.int64), marked_x(range(4)), torch.tensor(-r, dtype=torch.float32),
                 torch.zeros(4), None, TAG_ONPOLICY, 1, torch.zeros(1), None,
                 generator=torch.Generator().manual_seed(1000 + t))
        kept = set(st.x[st.n_seed:, 0].long().tolist())
        gone = ({0, 1, 2, 3} - kept)
        assert len(gone) == 1
        counts[gone.pop()] += 1
    assert chi2(counts, p) < 16.3                               # chi2(3) 0.999 quantile


# ---- monitoring --------------------------------------------------------------------------

def test_stats_on_a_hand_built_store():
    st = make(n_cond=2, bar=1.2, cap_per_condition=1)
    st.seed(torch.tensor([0, 0, 1, 1]), marked_x(range(4)), torch.tensor([3.0, 1.0, 2.0, 10.0]),
            torch.tensor([TAG_ANCHOR, TAG_ANCHOR, TAG_CHILD, TAG_CHILD]),
            torch.tensor([False, False, False, True]))
    st.refresh(zero_q, None, step=0, chunk=8)
    lv = torch.tensor([0.0, 0.5])
    st.admit([0, 0, 0], marked_x([10, 11, 12]), [4.0, 0.2, -2.0], [0.0, 0.0, 0.0], None,
             TAG_ONPOLICY, 1, lv, None)
    s = st.stats(lv)
    expect = {
        'store/rows': 6, 'store/rows_anchor': 2, 'store/rows_child': 2, 'store/rows_onpolicy': 2,
        'store/rows_generated': 0, 'store/rows_heldout': 1, 'store/rows_protected': 5, 'store/rows_records': 1,
        'store/cond_rows_min': 1.0, 'store/cond_rows_median': 1.0, 'store/cond_rows_max': 4.0,
        'store/conds': 2, 'store/conds_at_cap': 1,
        'store/admitted': 2, 'store/admitted_records': 1, 'store/rejected_floor': 0,
        'store/evicted_cap': 0, 'store/evicted_global': 0, 'store/demoted': 0,
        'store/admitted_resid_med': 2.0,
        # drawable |r| against the level: anchors 3, 1; child 1.5; admitted 4, 2 (held-out 9.5 excluded)
        'store/resid_med': 2.0, 'store/resid_p90': 3.6,
        # protected, not held out, |r| >= 1.2: anchor 3, child 1.5, record 4
        'store/protected_above_bar': 3, 'store/protected_above_bar_conds': 2,
        'store/protected_above_bar_cond_max': 2,
        'store/refresh_nonfinite': 0,
    }
    for k, v in expect.items():
        assert s[k] == pytest.approx(v), k
    for k in ('store/evicted_resid_med', 'store/evicted_age_med', 'store/evicted_age_max'):
        assert math.isnan(s[k]), k
    assert s['store/refresh_seconds'] >= 0.0
    s2 = st.stats(lv, prefix='x/')
    assert s2['x/admitted'] == 0 and math.isnan(s2['x/admitted_resid_med'])   # window drained


def test_stats_report_evicted_residuals_and_ages_in_evals():
    st = seeded([100.0], n_cond=1, bar=0.0)                     # refreshed once: eval_idx 1
    lv = torch.zeros(1)
    st.admit([0, 0], marked_x([10, 11]), [-3.0, -5.0], [0.0, 0.0], None, TAG_GENERATED, 1, lv, None)
    st.refresh(zero_q, lv, 2, 8)
    st.refresh(zero_q, lv, 3, 8)                                # eval_idx 3
    st.stats(lv)
    st.configure(cap_per_condition=0)
    st.admit([0], marked_x([12]), [-7.0], [0.0], None, TAG_ONPOLICY, 4, lv, None)
    s = st.stats(lv)
    assert s['store/evicted_cap'] == 3 and s['store/admitted'] == 1
    assert s['store/evicted_resid_med'] == 5.0                  # lower median of 3, 5, 7
    assert s['store/evicted_age_med'] == 2.0 and s['store/evicted_age_max'] == 2.0   # ages 2, 2, 0
    assert s['store/rows'] == 1 and s['store/rows_generated'] == 0


# ---- settings and persistence ------------------------------------------------------------

@pytest.mark.parametrize('bad', [dict(kappa=-1.0), dict(floor_frac=1.5), dict(uniform_frac=-0.1),
                                 dict(bar=-0.1), dict(eps=0.0), dict(cap_per_condition=-1),
                                 dict(max_rows=-1), dict(records_kept=0), dict(nonsense=1),
                                 dict(kappa=math.nan), dict(bar=math.nan), dict(max_rows=math.inf),
                                 dict(cap_per_condition=2.5), dict(records_kept=True),
                                 dict(admit_reward_min=math.nan), dict(uniform_frac='0.1')])
def test_configure_refuses_malformed_settings(bad):
    st = make()
    with pytest.raises(RowStoreError):
        st.configure(**bad)
    assert st.kappa == 1.0 and st.cap_per_condition == 100


def test_state_dict_round_trip():
    st = seeded([5.0, 1.0, 2.0, 30.0], cid=[0, 1, 1, 0], held_out=[False, False, False, True])
    lv = torch.tensor([0.0, 0.25])
    st.admit([0, 1, 1], marked_x([10, 11, 12]), [9.0, -3.0, 0.1], [0.0, 0.0, 0.0], None,
             TAG_ONPOLICY, 4, lv, cond_of([0, 1, 1]))
    st.refresh(lambda x, c: 0.1 * x[:, 0], lv, 5, 3)
    buf = io.BytesIO()
    torch.save(st.state_dict(stamp={'prior': 'abc'}), buf)
    buf.seek(0)
    state = torch.load(buf)
    assert state['stamp'] == {'prior': 'abc'} and state['format_version'] == FORMAT_VERSION
    back = RowStore.from_state_dict(state, 'cpu', **SET)
    assert len(back) == len(st)
    for c in ('cid', 'x', 'log_r', 'log_q', 'q_step', 'tag', 'protected', 'held_out', 'born_eval'):
        a, b = getattr(st, c), getattr(back, c)
        assert a.dtype == b.dtype, c
        assert torch.equal(torch.nan_to_num(a.float(), nan=-999.0), torch.nan_to_num(b.float(), nan=-999.0)), c
    assert torch.equal(torch.nan_to_num(st.cond_table, nan=-1.0), torch.nan_to_num(back.cond_table, nan=-1.0))
    assert torch.equal(st.best_log_r, back.best_log_r)
    assert (back.n_mle, back.n_seed, back.eval_idx, back.version) == (st.n_mle, st.n_seed, st.eval_idx, st.version)
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    a, b = st.draw_tb(64, lv, generator=g1), back.draw_tb(64, lv, generator=g2)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    back.touch(b[0], st.version, torch.zeros(64), 6)
    state['format_version'] = FORMAT_VERSION + 1
    with pytest.raises(RowStoreError):
        RowStore.from_state_dict(state, 'cpu', **SET)
