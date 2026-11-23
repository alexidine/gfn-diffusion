"""CrystalStore (crystal_store.py) on small hand-built stores with real template batches.

Draw laws are checked by chi-square against the law computed here, not by the module's own numbers.
"""
import io
import math
import os
import sys

import numpy as np
import pytest
import torch
from scipy.stats import chi2

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here)):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling import crystal_store as CS  # noqa: E402
from energy_sampling.crystal_store import (  # noqa: E402
    ANCHOR, CHILD, EVAL, FREE, ONPOLICY, ORIGIN_KICK, ORIGIN_ROLLOUT, ORIGIN_SEED, REPLAY_WEIGHTS, UNDRAWN,
    CrystalStore, CrystalStoreError, uniform)
from mxtaltools.dataset_utils.data_classes import MolData  # noqa: E402
from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402

pytestmark = pytest.mark.fast   # torch imported; CPU tensors of at most a few thousand rows
DIM = 3
P_MIN = 1e-5                    # chi-square p-value floor; every generator here is seeded


def _molecule(i):
    scale = 1.0 + 0.1 * i
    pos = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1]], dtype=torch.float32) * scale
    return MolData(z=torch.tensor([6, 7, 8], dtype=torch.long), pos=pos, x=torch.zeros((3, 1), dtype=torch.float32),
                   identifier=f'm{i}', mol_volume=torch.tensor(55.0 * scale ** 3), mass=torch.tensor(42.0),
                   radius=torch.tensor(1.2 * scale), z_prime=torch.tensor(1, dtype=torch.long))


def _templates(n, first=0):
    b = collate_data_list([_molecule(first + i) for i in range(n)])
    b.add_graph_attr(torch.arange(first, first + n, dtype=torch.long) % 4, 'sg_ind')
    return b


def _layout(spec):
    """spec: one (anchors, children) per condition. Rows: each condition's anchors, then its children, assigned to its
    anchors in turn. Returns (cid, kind, parent)."""
    cid, kind, parent = [], [], []
    for c, (na, nk) in enumerate(spec):
        first = len(cid)
        cid += [c] * (na + nk)
        kind += [ANCHOR] * na + [CHILD] * nk
        parent += [-1] * na + [first + j % na for j in range(nk)]
    return torch.tensor(cid), torch.tensor(kind), torch.tensor(parent)


def make(spec=((2, 6), (1, 3)), max_rows=100, H=5, seed=0, undrawn=None, energy=None, n_extra_cond=0):
    cid, kind, parent = _layout(spec)
    n = cid.numel()
    x = torch.linspace(-0.9, 0.9, n * DIM).reshape(n, DIM)
    e = torch.arange(n, dtype=torch.float32) + 10.0 if energy is None else energy
    und = torch.zeros(n, dtype=torch.bool) if undrawn is None else undrawn
    n_cond = len(spec) + n_extra_cond
    st = CrystalStore(DIM, n_cond, max_rows, H, seed=seed)
    st.seed(x, cid, kind, -e / 2.5, e, parent, und, _templates(len(spec)), torch.arange(len(spec)),
            torch.zeros(n_cond, dtype=torch.float64))
    return st


def add_onpolicy(st, cids, step=1, log_q=-1.0, energy=50.0, **kw):
    cids = torch.as_tensor(cids)
    n = cids.numel()
    return st.add(torch.zeros(n, DIM), cids, ONPOLICY, ORIGIN_ROLLOUT, torch.full((n,), -20.0),
                  torch.full((n,), energy), step, log_q=torch.full((n,), log_q), resid=torch.full((n,), 0.5), **kw)


def add_children(st, parents, step=1, energy=60.0, **kw):
    parents = torch.as_tensor(parents)
    n = parents.numel()
    return st.add(torch.zeros(n, DIM), st.cid[parents], CHILD, ORIGIN_KICK, torch.full((n,), -24.0),
                  torch.full((n,), energy), step, parent=parents, **kw)


def chi_p(counts, probs):
    counts, probs = np.asarray(counts, dtype=float), np.asarray(probs, dtype=float)
    exp = counts.sum() * probs / probs.sum()
    return float(chi2.sf(((counts - exp) ** 2 / exp).sum(), len(counts) - 1))


# ---- construction, seed ------------------------------------------------------------------------------------------


def test_origin_values_match_the_buffer():
    import buffer
    assert (CS.ORIGIN_ROLLOUT, CS.ORIGIN_EVAL, CS.ORIGIN_BOOTSTRAP) == (
        buffer.ORIGIN_ROLLOUT, buffer.ORIGIN_EVAL, buffer.ORIGIN_BOOTSTRAP)
    assert not {ORIGIN_SEED, ORIGIN_KICK} & {buffer.ORIGIN_ROLLOUT, buffer.ORIGIN_EVAL, buffer.ORIGIN_BOOTSTRAP}


def test_seed_writes_every_column_and_passes_check():
    st = make()
    st.check()
    assert (st.n_rows, st.n_anchors, st.n_drawable(CHILD), st.n_drawable(ANCHOR), st.n_drawable(ONPOLICY)) == (12, 3, 9, 3, 0)
    assert (st.origin[:12] == ORIGIN_SEED).all() and (st.birth_step[:12] == 0).all() and (st.draws[:12] == 0).all()
    assert torch.isnan(st.log_q[:12]).all() and torch.isnan(st.log_q_birth[:12]).all() and torch.isnan(st.resid[:12]).all()
    assert (st.q_step[:12] == -1).all()
    assert st.parent[:2].tolist() == [-1, -1] and st.parent[2:8].tolist() == [0, 1, 0, 1, 0, 1]
    assert st.best_energy.tolist() == [10.0, 18.0]
    assert st.cond_sg.tolist() == [0, 1]


def test_refusals_at_construction_and_seed():
    with pytest.raises(CrystalStoreError):
        CrystalStore(DIM, 1, 9, 5)
    with pytest.raises(CrystalStoreError):
        CrystalStore(DIM, 1, 2 ** 31, 5)
    with pytest.raises(CrystalStoreError, match='anchors'):
        make(spec=((5, 3),), max_rows=14, H=5)      # 5 anchors + 2 * 5 > 14
    st = make()
    with pytest.raises(CrystalStoreError, match='once'):
        st.seed(torch.zeros(1, DIM), [0], [ANCHOR], [0.0], [0.0], [-1], [False], _templates(2), [0, 1], torch.zeros(2))
    cid, kind, parent = _layout(((1, 2), (1, 1)))
    parent[2] = 3                                   # child of condition 0 pointing at condition 1's anchor
    fresh = CrystalStore(DIM, 2, 50, 5)
    with pytest.raises(CrystalStoreError, match='different conditions'):
        fresh.seed(torch.zeros(5, DIM), cid, kind, torch.zeros(5), torch.zeros(5), parent, torch.zeros(5, dtype=bool),
                   _templates(2), [0, 1], torch.zeros(2))


def test_seed_refuses_a_store_with_no_drawable_child():
    und = torch.tensor([False, True, True])
    with pytest.raises(CrystalStoreError, match='drawable child'):
        make(spec=((1, 2),), undrawn=und)


def test_seed_refuses_a_condition_with_no_template():
    cid, kind, parent = _layout(((1, 1), (1, 1)))
    st = CrystalStore(DIM, 2, 50, 5)
    with pytest.raises(CrystalStoreError, match='no template'):
        st.seed(torch.zeros(4, DIM), cid, kind, torch.zeros(4), torch.zeros(4), parent, torch.zeros(4, dtype=bool),
                _templates(1), [0], torch.zeros(2))


def test_seed_above_the_limit_keeps_anchors_and_the_true_minimum():
    spec = ((2, 40), (1, 20))
    cid, kind, parent = _layout(spec)
    n = cid.numel()
    e = torch.arange(n, dtype=torch.float32) + 10.0
    lowest = int(torch.nonzero(kind == CHILD)[-1])
    e[lowest] = -100.0
    st = make(spec=spec, max_rows=30, H=5, energy=e)
    st.check()
    assert st.n_rows == 25 and st.n_anchors == 3 and st.stats()['store/children'] == 22
    assert float(st.best_energy[int(cid[lowest])]) == -100.0
    assert st.n_seed_given == n and st.n_seed_kept == 25


def test_seed_thin_is_reproducible_and_ignores_reseed():
    def build(reseed):
        spec = ((2, 40), (1, 20))
        cid, kind, parent = _layout(spec)
        n = cid.numel()
        st = CrystalStore(DIM, 2, 30, 5, seed=7)
        if reseed:
            st.reseed(123, 456)
        x = torch.arange(n * DIM, dtype=torch.float32).reshape(n, DIM)
        st.seed(x, cid, kind, torch.zeros(n), torch.zeros(n), parent, torch.zeros(n, dtype=bool), _templates(2), [0, 1],
                torch.zeros(2))
        return st.x[:st.high_water].clone()
    assert torch.equal(build(False), build(True))


def test_seed_drops_nonfinite_rows_and_orphans_without_touching_the_minimum():
    spec = ((2, 4),)
    cid, kind, parent = _layout(spec)
    n = cid.numel()
    e = torch.arange(n, dtype=torch.float32) + 10.0
    e[0] = float('nan')                             # anchor 0: its children (positions 2, 4) are orphans
    e[3] = float('-inf')
    x = torch.zeros(n, DIM)
    st = CrystalStore(DIM, 1, 50, 5)
    st.seed(x, cid, kind, torch.zeros(n), e, parent, torch.zeros(n, dtype=bool), _templates(1), [0], torch.zeros(1))
    st.check()
    assert st.seed_nonfinite == 2 and st.seed_orphans == 2 and st.n_rows == 2
    assert float(st.best_energy[0]) == 11.0         # orphans count, non-finite rows do not: min(11, 12, 14, 15)


# ---- add -------------------------------------------------------------------------------------------------------


def test_add_is_slot_writes_with_no_column_rebuild(monkeypatch):
    st = make()
    pointers = {name: getattr(st, name).data_ptr() for name, _ in CS._COLUMNS}
    def refuse(*a, **k):
        raise AssertionError('a column was rebuilt')
    monkeypatch.setattr(torch, 'cat', refuse)
    monkeypatch.setattr(torch.Tensor, 'index_select', refuse)
    for _ in range(5):
        add_onpolicy(st, [0, 1, 1])
        add_children(st, [0, 8])
    assert all(getattr(st, name).data_ptr() == v for name, v in pointers.items())
    monkeypatch.undo()
    st.check()
    assert st.n_rows == 12 + 25


def test_add_writes_every_field_of_an_onpolicy_row():
    st = make()
    s = add_onpolicy(st, [1], step=7, log_q=-3.0, eval_row=True)
    r = int(s[0])
    assert r == 12
    assert (int(st.kind[r]), int(st.flags[r]), int(st.origin[r]), int(st.parent[r])) == (ONPOLICY, EVAL, ORIGIN_ROLLOUT, -1)
    assert float(st.log_q[r]) == -3.0 and float(st.log_q_birth[r]) == -3.0 and float(st.resid[r]) == 0.5
    assert (int(st.q_step[r]), int(st.birth_step[r]), int(st.draws[r])) == (7, 7, 0)
    st.check()


def test_child_links_are_validated():
    st = make()
    with pytest.raises(CrystalStoreError, match='not an anchor'):
        add_children(st, [3])                       # slot 3 is a child: a child is never a parent
    with pytest.raises(CrystalStoreError, match='different conditions'):
        st.add(torch.zeros(1, DIM), [0], CHILD, ORIGIN_KICK, [0.0], [0.0], 1, parent=[8])
    with pytest.raises(CrystalStoreError, match='only a child'):
        st.add(torch.zeros(1, DIM), [0], ONPOLICY, ORIGIN_ROLLOUT, [0.0], [0.0], 1, parent=[0], log_q=[0.0])
    with pytest.raises(CrystalStoreError, match='not a live slot'):
        st.add(torch.zeros(1, DIM), [0], CHILD, ORIGIN_KICK, [0.0], [0.0], 1, parent=[99])
    assert st.n_rows == 12
    st.check()


def test_add_refusals():
    st = make(H=2, max_rows=20)
    with pytest.raises(CrystalStoreError, match='headroom'):
        add_onpolicy(st, [0, 0, 0])                 # one add above H
    with pytest.raises(CrystalStoreError, match='need log_q'):
        st.add(torch.zeros(1, DIM), [0], ONPOLICY, ORIGIN_ROLLOUT, [0.0], [0.0], 1)
    with pytest.raises(CrystalStoreError, match='on-policy rows only'):
        st.add(torch.zeros(1, DIM), [0], ANCHOR, ORIGIN_SEED, [0.0], [0.0], 1, log_q=[0.0])
    with pytest.raises(CrystalStoreError, match='condition id'):
        add_onpolicy(st, [5])
    with pytest.raises(CrystalStoreError, match='ANCHOR, CHILD and ONPOLICY'):
        st.add(torch.zeros(1, DIM), [0], FREE, ORIGIN_SEED, [0.0], [0.0], 1)
    full = make(spec=((3, 3),), H=2, max_rows=8)    # 3 anchors + 2 * 2 = 7: one anchor more fits, two do not
    best = float(full.best_energy[0])
    with pytest.raises(CrystalStoreError, match='anchors'):
        full.add(torch.zeros(2, DIM), [0, 0], ANCHOR, ORIGIN_SEED, [0.0, 0.0], [-999.0, -999.0], 1)
    assert float(full.best_energy[0]) == best       # a refused add writes nothing


def test_nonfinite_rows_are_dropped_counted_and_never_reach_the_minimum():
    st = make()
    x = torch.zeros(3, DIM)
    x[1, 0] = float('nan')
    slots = st.add(x, [0, 0, 0], ONPOLICY, ORIGIN_ROLLOUT, [0.0, 0.0, float('inf')], [5.0, -999.0, -999.0], 1,
                   log_q=[0.0, 0.0, 0.0])
    assert slots.numel() == 1 and st.nonfinite_rows == 2
    assert float(st.best_energy[0]) == 5.0
    st.check()


def test_a_row_with_a_nonfinite_log_q_is_stored_undrawn():
    st = make()
    s = add_onpolicy(st, [0], log_q=float('nan'))
    r = int(s[0])
    assert int(st.flags[r]) & UNDRAWN and st.n_drawable(ONPOLICY) == 0 and st.n_rows == 13
    assert torch.isnan(st.log_q_birth[r]) and int(st.q_step[r]) == -1
    st.check()


def test_undrawn_rows_are_stored_and_never_drawn():
    st = make()
    s = add_children(st, [0, 0], undrawn=torch.tensor([True, False]))
    assert st.n_drawable(CHILD) == 10
    drawn = st.draw(CHILD, 4000)
    assert int(s[0]) not in drawn.tolist() and int(s[1]) in drawn.tolist()
    st.check()


# ---- draws -----------------------------------------------------------------------------------------------------


def test_draw_is_uniform_over_conditions_then_rows():
    st = make(spec=((1, 2), (1, 30), (1, 7)))
    B = 60000
    rows = st.draw(CHILD, B)
    assert (st.kind[rows] == CHILD).all() and rows.numel() == B
    assert chi_p(torch.bincount(st.cid[rows].long(), minlength=3).numpy(), [1, 1, 1]) > P_MIN
    in_big = rows[st.cid[rows] == 1]
    u, c = torch.unique(in_big, return_counts=True)
    assert u.numel() == 30 and chi_p(c.numpy(), np.ones(30)) > P_MIN


def test_a_condition_drawn_twice_gets_independent_rows():
    st = make(spec=((1, 50),))
    rows = st.draw(CHILD, 40)                       # one condition, with replacement
    assert torch.unique(rows).numel() > 1


def test_draw_on_an_empty_pool_raises():
    st = make()
    with pytest.raises(CrystalStoreError, match='no drawable onpolicy'):
        st.draw(ONPOLICY, 3)
    assert st.n_drawable(ONPOLICY) == 0


def test_draw_anchors_by_anchor_and_by_condition():
    st = make(spec=((4, 4), (1, 4)))
    B = 40000
    by_anchor = st.draw_anchors(B)
    assert (st.kind[by_anchor] == ANCHOR).all()
    assert chi_p(torch.bincount(st.cid[by_anchor].long(), minlength=2).numpy(), [4, 1]) > P_MIN
    by_cond = st.draw_anchors(B, by='condition')
    assert chi_p(torch.bincount(st.cid[by_cond].long(), minlength=2).numpy(), [1, 1]) > P_MIN
    with pytest.raises(CrystalStoreError, match='by must be'):
        st.draw_anchors(4, by='row')


def test_an_undrawn_anchor_is_never_drawn_but_its_child_link_holds():
    cid, kind, parent = _layout(((2, 4),))
    und = torch.zeros(cid.numel(), dtype=torch.bool)
    und[0] = True
    st = make(spec=((2, 4),), undrawn=und)
    assert set(st.draw_anchors(500).tolist()) == {1}
    st.check()


def test_new_anchors_enter_the_uniform_anchor_draw():
    st = make(spec=((1, 2),))
    assert set(st.draw_anchors(50).tolist()) == {0}
    s = st.add(torch.zeros(1, DIM), [0], ANCHOR, ORIGIN_SEED, [0.0], [12.0], 3)
    assert set(st.draw_anchors(400).tolist()) == {0, int(s[0])}
    st.check()


def test_the_store_never_touches_the_global_generators():
    torch.manual_seed(5)
    np.random.seed(5)
    t0, n0 = torch.get_rng_state().clone(), np.random.get_state()[1].copy()
    st = make(spec=((2, 40),), max_rows=30, H=5)
    add_onpolicy(st, [0, 0, 0])
    st.draw(CHILD, 10)
    st.draw_anchors(10)
    st.draw_weighted(ONPOLICY, 10, lambda s, r, step: torch.ones(len(r)), 0)
    st.thin(3)
    assert torch.equal(torch.get_rng_state(), t0) and np.array_equal(np.random.get_state()[1], n0)


def test_reseed_changes_the_draws_and_is_reproducible():
    a, b, c = make(seed=1), make(seed=1), make(seed=1)
    assert torch.equal(a.draw(CHILD, 50), b.draw(CHILD, 50))
    a.reseed(1, 100)
    b.reseed(1, 100)
    c.reseed(1, 200)
    ra, rb, rc = a.draw(CHILD, 50), b.draw(CHILD, 50), c.draw(CHILD, 50)
    assert torch.equal(ra, rb) and not torch.equal(ra, rc)
    with pytest.raises(CrystalStoreError):
        a.reseed(-1, 0)


# ---- the replay weight seam ----------------------------------------------------------------------------------


def _unequal_onpolicy():
    st = make(spec=((1, 1), (1, 1), (1, 1)), max_rows=400, H=60)
    for c, m in enumerate((2, 7, 41)):
        add_onpolicy(st, [c] * m)
    return st


def test_correction_is_exactly_one_for_the_uniform_rule_on_unequal_conditions():
    st = _unequal_onpolicy()
    assert REPLAY_WEIGHTS['uniform'] is uniform
    rows, w = st.draw_weighted(ONPOLICY, 500, uniform, 3)
    assert rows.numel() == 500 and (w == 1.0).all()
    def none_rule(store, rows, step):
        return None
    def ones_rule(store, rows, step):
        return torch.ones(len(rows))
    for rule in (none_rule, ones_rule):
        _, w = st.draw_weighted(ONPOLICY, 500, rule, 3)
        assert (w == 1.0).all()


def test_weighted_draw_follows_the_rule_within_a_uniform_condition():
    st = _unequal_onpolicy()
    def rule(store, rows, step):
        return (rows % 3 + 1).double()
    B = 90000
    rows, w = st.draw_weighted(ONPOLICY, B, rule, 3)
    cnt = torch.bincount(st.cid[rows].long(), minlength=3).numpy()
    assert chi_p(cnt, [1, 1, 1]) > P_MIN
    big = torch.nonzero((st.kind[:st.high_water] == ONPOLICY) & (st.cid[:st.high_water] == 2)).flatten()
    got = torch.bincount(rows[st.cid[rows] == 2], minlength=st.high_water)[big].numpy()
    assert chi_p(got, (big % 3 + 1).numpy()) > P_MIN
    probs = (big % 3 + 1).double() / (big % 3 + 1).double().sum()
    expect = {int(r): float(1 / (len(big) * p)) for r, p in zip(big, probs)}
    picked = rows[st.cid[rows] == 2]
    for r, v in zip(picked[:200].tolist(), w[st.cid[rows] == 2][:200].tolist()):
        assert v == pytest.approx(expect[r], rel=1e-6)


def test_correction_restores_the_uniform_within_condition_mean():
    st = _unequal_onpolicy()
    def rule(store, rows, step):
        return (rows % 5 + 1).double()
    rows, w = st.draw_weighted(ONPOLICY, 200000, rule, 0)
    in_c = st.cid[rows] == 2
    value = (rows[in_c] % 7).double()
    weighted = float((value * w[in_c].double()).sum() / w[in_c].double().sum())
    big = torch.nonzero((st.kind[:st.high_water] == ONPOLICY) & (st.cid[:st.high_water] == 2)).flatten()
    assert weighted == pytest.approx(float((big % 7).double().mean()), abs=0.03)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.0])
def test_a_bad_weight_raises_naming_the_rule_and_the_slot(bad):
    st = _unequal_onpolicy()
    def poison_rule(store, rows, step):
        w = torch.ones(len(rows))
        w[0] = bad
        return w
    with pytest.raises(CrystalStoreError, match=r"poison_rule.*slot \d+"):
        st.draw_weighted(ONPOLICY, 10, poison_rule, 0)


def test_a_wrong_length_weight_raises():
    st = _unequal_onpolicy()
    with pytest.raises(CrystalStoreError, match='weights'):
        st.draw_weighted(ONPOLICY, 10, lambda s, r, step: torch.ones(1), 0)


def test_an_all_zero_condition_falls_back_to_uniform_within_it():
    st = _unequal_onpolicy()
    c2 = st.cid == 2
    def rule(store, rows, step):
        return (~c2[rows]).double()                 # condition 2: all zero
    B = 60000
    rows, w = st.draw_weighted(ONPOLICY, B, rule, 0)
    assert chi_p(torch.bincount(st.cid[rows].long(), minlength=3).numpy(), [1, 1, 1]) > P_MIN
    u, cnt = torch.unique(rows[st.cid[rows] == 2], return_counts=True)
    assert u.numel() == 41 and chi_p(cnt.numpy(), np.ones(41)) > P_MIN
    assert (w[st.cid[rows] == 2] == 1.0).all()


def test_a_zero_weight_row_is_never_drawn():
    st = _unequal_onpolicy()
    def rule(store, rows, step):
        return (rows % 2).double()
    rows, _ = st.draw_weighted(ONPOLICY, 20000, rule, 0)
    in_mixed = st.cid[rows] != 0                    # condition 0's two rows may both be even
    assert (rows[in_mixed] % 2 == 1).all()


def test_mask_restricts_the_draw_to_a_subtype():
    st = make(spec=((1, 1), (1, 1)), max_rows=300, H=40)
    add_onpolicy(st, [0] * 10)
    evals = add_onpolicy(st, [0] * 4 + [1] * 4, eval_row=True)
    def is_eval(store, rows):
        return (store.flags[rows] & EVAL) != 0
    rows, w = st.draw_weighted(ONPOLICY, 20000, uniform, 0, mask=is_eval)
    assert set(rows.tolist()) == set(evals.tolist())
    assert chi_p(torch.bincount(st.cid[rows].long(), minlength=2).numpy(), [1, 1]) > P_MIN
    assert (w == 1.0).all()
    with pytest.raises(CrystalStoreError, match='pass the mask'):
        st.draw_weighted(ONPOLICY, 5, uniform, 0, mask=lambda s, r: torch.zeros(len(r), dtype=torch.bool))


# ---- touch and rewind ----------------------------------------------------------------------------------------


def test_touch_counts_occurrences_and_never_writes_birth():
    st = make()
    s = add_onpolicy(st, [0, 0], log_q=-2.0)
    a, b = int(s[0]), int(s[1])
    n = st.touch([a, a, b, a], st.epoch, [-5.0, -6.0, -7.0, -8.0], [1.0, 2.0, 3.0, 4.0], 9)
    assert n == 2
    assert (int(st.draws[a]), int(st.draws[b])) == (3, 1)
    assert (float(st.log_q[a]), float(st.resid[a]), int(st.q_step[a])) == (-8.0, 4.0, 9)
    assert float(st.log_q_birth[a]) == -2.0 and float(st.log_q_birth[b]) == -2.0
    st.check()


def test_touch_keeps_the_old_value_on_a_nonfinite_input_and_still_counts_the_draw():
    st = make()
    a = int(add_onpolicy(st, [0], log_q=-2.0)[0])
    assert st.touch([a], st.epoch, [float('nan')], [1.0], 4) == 0
    assert float(st.log_q[a]) == -2.0 and int(st.q_step[a]) == 1 and int(st.draws[a]) == 1


def test_touch_under_another_epoch_writes_nothing():
    st = make(spec=((1, 20),), max_rows=26, H=5)
    add_onpolicy(st, [0] * 4)
    e0 = st.epoch
    rows = st.draw(CHILD, 5)
    assert st.thin(2) > 0 and st.epoch == e0 + 1
    before_q, before_d = st.log_q.clone(), st.draws.clone()
    assert st.touch(rows, e0, torch.zeros(5), torch.zeros(5), 3) == 0
    assert torch.equal(torch.nan_to_num(before_q[:st.high_water], 7.0), torch.nan_to_num(st.log_q[:st.high_water], 7.0))
    assert torch.equal(before_d[:st.high_water], st.draws[:st.high_water])


def test_touch_refuses_an_anchor():
    st = make()
    with pytest.raises(CrystalStoreError, match='anchor'):
        st.touch([0], st.epoch, [1.0], [1.0], 2)


def test_invalidate_q_after_clears_only_later_writes():
    st = make()
    s = add_onpolicy(st, [0, 0, 0], step=2)
    a, b, c = s.tolist()
    st.touch([b], st.epoch, [-4.0], [0.2], 8)
    st.touch([c], st.epoch, [-4.0], [0.2], 12)
    assert st.invalidate_q_after(8) == 1
    assert float(st.log_q[c]) == -1.0 and int(st.q_step[c]) == 2 and torch.isnan(st.resid[c])
    assert float(st.log_q[b]) == -4.0 and int(st.q_step[b]) == 8 and int(st.q_step[a]) == 2
    assert float(st.log_q_birth[c]) == -1.0


def test_invalidate_q_after_clears_a_touched_child():
    st = make()
    kid = int(st.draw(CHILD, 1)[0])
    st.touch([kid], st.epoch, [-4.0], [0.2], 12)
    assert st.invalidate_q_after(8) == 1
    assert torch.isnan(st.log_q[kid]) and torch.isnan(st.resid[kid]) and int(st.q_step[kid]) == -1
    st.check()


def test_a_row_born_after_the_rewind_keeps_its_admission_log_q_and_a_repeat_changes_nothing():
    st = make()
    s = add_onpolicy(st, [0, 1], step=10, log_q=-2.0)
    assert st.invalidate_q_after(5) == 2
    assert st.log_q[s].tolist() == [-2.0, -2.0] and st.q_step[s].tolist() == [10, 10]
    assert torch.isnan(st.resid[s]).all()
    g = st.generation
    assert st.invalidate_q_after(5) == 0 and st.generation == g


def test_a_log_q_replay_rule_still_draws_after_a_rewind():
    st = make()
    s = add_onpolicy(st, [0, 0, 1, 1], step=10, log_q=-2.0)
    st.touch(s, st.epoch, torch.full((4,), -3.0), torch.full((4,), 0.1), 12)
    st.invalidate_q_after(5)
    st.check()
    rows, cor = st.draw_weighted(ONPOLICY, 64, lambda store, r, step: torch.exp(-store.log_q[r].double()), 13)
    assert set(rows.tolist()) <= set(s.tolist()) and torch.isfinite(cor).all()


def test_check_detects_a_drawable_onpolicy_row_without_log_q():
    st = make()
    s = add_onpolicy(st, [0])
    st.log_q[s] = math.nan
    with pytest.raises(CrystalStoreError, match='finite log_q'):
        st.check()


# ---- thinning ------------------------------------------------------------------------------------------------


def _thin_store(children=200, onpolicy=200, max_rows=1000, H=10):
    st = make(spec=((4, children),), max_rows=max_rows, H=H)
    for i in range(0, onpolicy, H):
        add_onpolicy(st, [0] * min(H, onpolicy - i))
    return st


def test_thin_takes_half_from_each_pool_and_keeps_anchors():
    st = _thin_store()
    anchors = st.x[:4].clone()
    freed = st.thin(100)
    s = st.stats()
    assert freed == 100 and s['store/children'] == 150 and s['store/onpolicy'] == 150 and s['store/anchors'] == 4
    assert torch.equal(st.x[:4], anchors) and (st.kind[:4] == ANCHOR).all()
    assert st.thinned_total == 100
    st.check()


def test_thin_frees_a_block_when_asked_for_less():
    st = _thin_store(max_rows=1000)
    assert st.thin(1) == 50                         # ceil(0.05 * max_rows)
    st.check()


def test_a_short_pool_passes_its_deficit_to_the_other():
    st = _thin_store(children=200, onpolicy=10)
    st.thin(100)
    s = st.stats()
    assert s['store/onpolicy'] == 0 and s['store/children'] == 110
    st = _thin_store(children=15, onpolicy=200, H=10)
    st.thin(100)
    s = st.stats()
    assert s['store/children'] == 10 and s['store/onpolicy'] == 105     # 5 children above the floor, 95 on-policy
    st.check()


def test_children_never_fall_below_headroom():
    st = _thin_store(children=12, onpolicy=0, H=10)
    assert st.thin(2) == 2 and st.stats()['store/children'] == 10
    with pytest.raises(CrystalStoreError, match='a thin of'):
        st.thin(1)
    st.check()


def test_thin_is_uniform_within_each_pool():
    n = 40
    counts = np.zeros(n)
    for seed in range(800):
        st = make(spec=((1, n),), max_rows=1000, H=2, seed=seed)
        kids = torch.nonzero(st.kind[:st.high_water] == CHILD).flatten()
        assert st.thin(12) == 20                    # block = min(ceil(0.05 * 1000), 40 // 2)
        counts[(st.kind[kids] == FREE).numpy()] += 1
    assert chi_p(counts, np.ones(n)) > P_MIN


def test_thin_at_the_exact_capacity_boundary():
    H, na = 6, 4
    max_rows = na + 2 * H
    st = make(spec=((na, H),), max_rows=max_rows, H=H)      # a seed fills to max_rows - H
    add_children(st, [0])
    assert st.n_rows == na + H + 1 and st.thinned_total == 0
    slots = add_children(st, torch.zeros(H, dtype=torch.long))
    assert slots.numel() == H and st.n_rows == max_rows
    s = st.stats()
    assert s['store/children'] == 2 * H and s['store/anchors'] == na and st.thinned_total == 1
    st.check()


def test_the_store_runs_full_for_many_adds():
    H, na = 8, 3
    max_rows = na + 2 * H + 5
    st = make(spec=((na, H),), max_rows=max_rows, H=H)
    for i in range(200):
        if i % 2:
            add_children(st, torch.zeros(H, dtype=torch.long), step=i)
        else:
            add_onpolicy(st, [0] * H, step=i)
        assert st.n_rows <= max_rows
        assert st.stats()['store/children'] >= H
        st.check()
    assert st.thinned_total > 0 and st.n_anchors == na


def test_thinned_rows_keep_their_minimum():
    st = make(spec=((1, 10),), max_rows=20, H=4)
    s = add_children(st, [0], energy=-500.0)
    st.remove(s)
    assert float(st.best_energy[0]) == -500.0
    st = make(spec=((1, 12),), max_rows=20, H=4)
    add_onpolicy(st, [0, 0, 0, 0], energy=-77.0)
    for _ in range(6):
        add_onpolicy(st, [0, 0, 0, 0], energy=40.0)
    assert float(st.best_energy[0]) == -77.0
    st.check()


def test_an_undrawn_row_is_thinned_and_counted():
    st = make(spec=((1, 10),), max_rows=20, H=2)
    for _ in range(2):
        add_children(st, [0, 0], undrawn=torch.ones(2, dtype=torch.bool))
    u0 = st.stats()['store/undrawn']
    assert st.thin(12) == 12                        # 14 children, floor 2
    assert st.stats()['store/undrawn'] < u0 and st.stats()['store/children'] == 2
    st.check()


# ---- remove and new conditions -------------------------------------------------------------------------------


def test_remove_refuses_an_anchor_with_a_live_child():
    st = make()
    with pytest.raises(CrystalStoreError, match='live children'):
        st.remove([0])
    assert st.n_rows == 12
    st.check()


def test_remove_an_anchor_with_all_its_children():
    st = make()
    kids = torch.nonzero(st.parent[:st.high_water] == 0).flatten()
    epoch = st.epoch
    st.remove(torch.cat([torch.tensor([0]), kids]))
    assert st.n_anchors == 2 and st.n_rows == 12 - 1 - kids.numel() and st.epoch == epoch + 1
    assert 0 not in st.draw_anchors(300).tolist()
    st.check()


def test_remove_refusals():
    st = make()
    with pytest.raises(CrystalStoreError, match='twice'):
        st.remove([3, 3])
    with pytest.raises(CrystalStoreError, match='not a slot'):
        st.remove([500])
    st.remove([3])
    with pytest.raises(CrystalStoreError, match='already free'):
        st.remove([3])


def test_removing_a_whole_condition_empties_its_pools_and_slots_are_reused():
    st = make()
    rows = torch.nonzero(st.cid[:st.high_water] == 1).flatten()
    st.remove(rows)
    st.check()
    assert set(st.cid[st.draw(CHILD, 400)].tolist()) == {0}
    s = add_onpolicy(st, [0, 0])
    assert set(s.tolist()) <= set(rows.tolist())
    assert st.high_water == 12
    st.check()


def test_add_conditions_appends_without_renumbering():
    st = make()
    x_before = st.x[:12].clone()
    ids = st.add_conditions(2, _templates(2, first=2))
    assert ids.tolist() == [2, 3] and st.n_cond == 4
    assert torch.equal(st.x[:12], x_before) and st.cid[:12].max() == 1
    assert len(st.index.count) == 12 and float(st.best_energy[3]) == math.inf
    assert st.tmpl_of[2:].tolist() == [2, 3]
    s = st.add(torch.zeros(1, DIM), [3], ANCHOR, ORIGIN_SEED, [0.0], [1.0], 1)
    kid = st.add(torch.zeros(1, DIM), [3], CHILD, ORIGIN_KICK, [0.0], [2.0], 1, parent=s)
    assert float(st.best_energy[3]) == 1.0 and int(st.parent[kid[0]]) == int(s[0])
    assert 3 in st.cid[st.draw(CHILD, 400)].tolist()
    mol = st.mol_batch(torch.tensor([3]))
    assert float(mol.radius[0]) == pytest.approx(1.2 * 1.3)
    st.check()


def test_add_conditions_without_a_template_leaves_it_unset():
    st = make()
    ids = st.add_conditions(1)
    assert int(st.tmpl_of[int(ids[0])]) == -1
    with pytest.raises(CrystalStoreError, match='no template'):
        st.mol_batch(ids)
    st.check()


# ---- templates -----------------------------------------------------------------------------------------------


def test_mol_batch_reads_the_template_and_tiles():
    st = make()
    b = st.mol_batch(torch.tensor([1, 0]), repeats=2)
    assert b.num_graphs == 4
    assert b.radius.reshape(-1).tolist() == pytest.approx([1.2 * 1.1, 1.2 * 1.1, 1.2, 1.2])
    assert torch.equal(b.pos[:3], st.templates.pos[3:6])


def test_condition_radius_and_rg_come_from_the_template():
    from energy_sampling import noised_prior as NP
    st = make()
    assert st.cond_radius.tolist() == pytest.approx([1.2, 1.32], rel=1e-6)
    assert st.cond_rg.tolist() == pytest.approx(NP.radius_of_gyration(st.templates).tolist(), rel=1e-6)


# ---- index under churn, round trip -----------------------------------------------------------------------------


def test_index_survives_random_churn_and_compaction():
    rng = np.random.default_rng(3)
    st = make(spec=((2, 8), (1, 2), (3, 5), (1, 1), (1, 9)), max_rows=120, H=12)
    n_cond = 5
    compactions = 0
    for i in range(1500):
        op = rng.integers(0, 4)
        if op == 0:
            add_onpolicy(st, rng.integers(0, n_cond, int(rng.integers(1, 12))).tolist(), step=i)
        elif op == 1:
            anchors = torch.nonzero(st.kind[:st.high_water] == ANCHOR).flatten()
            add_children(st, anchors[rng.integers(0, len(anchors), int(rng.integers(1, 12)))], step=i)
        elif op == 2:
            live = torch.nonzero(st.kind[:st.high_water] > ANCHOR).flatten()
            if len(live) > st.headroom + 2:
                st.remove(live[rng.choice(len(live), int(rng.integers(1, 5)), replace=False)])
        else:
            st.thin(int(rng.integers(1, 4)))
        if i % 25 == 0:
            st.check()
        compactions = st.index.compactions
    st.check()
    assert compactions > 0


def test_compaction_leaves_each_pool_room_for_its_add():
    idx = CS.PoolIndex(4, 50)
    idx.build(np.arange(30, dtype=np.int32), np.array([0] * 15 + [1] * 9 + [2] * 6, dtype=np.int64))
    idx.tail = idx.arena.size                       # the next move to the tail must compact
    # pool 3 is short; pool 1 fits its old capacity (21) but needs more than 1.5 x its count (13)
    idx.add(np.arange(30, 44, dtype=np.int32), np.array([1] * 5 + [3] * 9, dtype=np.int64))
    assert idx.compactions == 1
    assert (idx.count == [15, 14, 6, 9]).all() and (idx.cap >= idx.count).all()
    for p in range(4):
        members = idx.arena[idx.start[p]:idx.start[p] + idx.count[p]]
        assert (idx.where[members] == np.arange(idx.count[p])).all()


def test_state_round_trip_is_exact():
    st = make(spec=((2, 40), (1, 30)), max_rows=120, H=10, seed=5)
    for i in range(14):
        add_onpolicy(st, [0, 1, 1, 0, 1], step=i, log_q=-float(i))
        add_children(st, [0, 1, 42], step=i)
    st.thin(6)
    st.touch(st.draw(CHILD, 4), st.epoch, torch.full((4,), -3.0), torch.full((4,), 0.1), 20)
    state = st.state_dict(stamp={'k': 1})
    assert state['columns']['x'].shape[0] == st.high_water
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    state = torch.load(buf, weights_only=False)
    assert state['stamp'] == {'k': 1}

    other = CrystalStore(DIM, 2, 120, 10, seed=99)
    other.apply_state(state)
    other.check()
    a, b = st.state_dict(stamp={'k': 1}), other.state_dict(stamp={'k': 1})
    for name, _ in CS._COLUMNS:
        assert torch.equal(torch.nan_to_num(a['columns'][name].double(), 1e30),
                           torch.nan_to_num(b['columns'][name].double(), 1e30)), name
    for name in CS._TABLES:
        assert torch.equal(torch.nan_to_num(a['tables'][name].double(), 1e30),
                           torch.nan_to_num(b['tables'][name].double(), 1e30)), name
    for key in ('free', 'index_count', 'index_rows'):
        assert torch.equal(a[key], b[key]), key
    assert all(torch.equal(x, y) for x, y in zip(a['active'], b['active']))
    assert a['counters'] == b['counters']
    assert torch.equal(st.draw(CHILD, 200), other.draw(CHILD, 200))
    assert torch.equal(st.draw_anchors(200), other.draw_anchors(200))
    s1, s2 = add_onpolicy(st, [0, 1], step=30), add_onpolicy(other, [0, 1], step=30)
    assert torch.equal(s1, s2)
    assert st.thin(4) == other.thin(4)
    assert torch.equal(st.kind[:st.high_water], other.kind[:other.high_water])
    st.check()
    other.check()


def test_saved_size_follows_the_occupied_rows_not_the_capacity():
    def saved(max_rows):
        buf = io.BytesIO()
        torch.save(make(max_rows=max_rows).state_dict(), buf)
        return buf.getbuffer().nbytes
    assert saved(2_000_000) < saved(100) + 4096


def test_state_is_a_snapshot():
    st = make()
    state = st.state_dict()
    add_onpolicy(st, [0])
    st.log_r[0] = 123.0
    assert state['columns']['x'].shape[0] == 12 and float(state['columns']['log_r'][0]) != 123.0


def test_apply_state_refuses_a_different_store():
    st = make()
    state = st.state_dict()
    with pytest.raises(CrystalStoreError, match='max_rows'):
        CrystalStore(DIM, 2, 101, 5).apply_state(state)
    with pytest.raises(CrystalStoreError, match='format_version'):
        CrystalStore(DIM, 2, 100, 5).apply_state({**state, 'format_version': 2})
    with pytest.raises(CrystalStoreError, match='n_cond'):
        CrystalStore(DIM, 3, 100, 5).apply_state(state)


def test_apply_state_extends_conditions_added_after_seed():
    st = make()
    st.add_conditions(1, _templates(1, first=2))
    st.add(torch.zeros(1, DIM), [2], ANCHOR, ORIGIN_SEED, [0.0], [1.0], 1)
    other = CrystalStore(DIM, 2, 100, 5)
    other.apply_state(st.state_dict())
    other.check()
    assert other.n_cond == 3 and int(other.tmpl_of[2]) == 2


def test_generation_counts_durable_changes_only():
    st = make()
    g = st.generation
    add_onpolicy(st, [0], durable=False)
    assert st.generation == g
    add_onpolicy(st, [0])
    assert st.generation == g + 1
    st.touch([int(st.high_water) - 1], st.epoch, [0.0], [0.0], 2)
    assert st.generation == g + 2
    st.thin(1)
    assert st.generation == g + 3


def test_check_detects_a_corrupted_store():
    for corrupt, what in ((lambda s: s.parent.__setitem__(3, 9), 'I2'),
                          (lambda s: s.kind.__setitem__(3, FREE), 'I5'),
                          (lambda s: s.flags.__setitem__(3, UNDRAWN), 'undrawn'),
                          (lambda s: s.window_floor.__setitem__(0, 5.0), 'I7'),
                          (lambda s: s.energy.__setitem__(3, -1e6), 'I6'),
                          (lambda s: s.log_q_birth.__setitem__(3, 0.0), 'I12'),
                          (lambda s: setattr(s, 'best_energy', torch.zeros(5)), 'table best_energy'),
                          (lambda s: s.cond_stats.__setitem__('n', torch.zeros(5)), 'table n')):
        st = make()
        corrupt(st)
        with pytest.raises(CrystalStoreError, match=what):
            st.check()


# ---- refused writes leave the store unchanged ------------------------------------------------------------------


def _fingerprint(st):
    return (st.n_cond, st.n_rows, st.high_water, st.n_free, st.epoch, st.generation, st.nonfinite_rows, st.thinned_total,
            list(st._n_kind), list(st._n_drawable), st.best_energy.tolist(), st.tmpl_of.tolist(), len(st.cond_sg),
            len(st.window_floor), len(st._floor_written), len(st.index.count), st.index.arena.size,
            [s.pos.size for s in st.active], {k: tuple(v.shape) for k, v in st.cond_stats.items()},
            None if st.templates is None else int(st.templates.num_graphs), repr(st.rng.bit_generator.state))


def test_add_refuses_a_condition_with_no_template_and_changes_nothing():
    st = make(max_rows=19, H=8)                     # 12 rows + 8 would thin, and -5 would lower the minimum
    new = int(st.add_conditions(1)[0])
    before = _fingerprint(st)
    with pytest.raises(CrystalStoreError, match=f'conditions \\[{new}\\] have no template'):
        st.add(torch.zeros(8, DIM), [0] * 7 + [new], ONPOLICY, ORIGIN_ROLLOUT, torch.zeros(8), torch.full((8,), -5.0), 1,
               log_q=torch.zeros(8), resid=torch.zeros(8))
    assert _fingerprint(st) == before
    st.check()


def test_a_nonfinite_row_of_a_condition_with_no_template_is_dropped_and_counted():
    st = make()
    new = int(st.add_conditions(1)[0])
    got = st.add(torch.zeros(1, DIM), [new], ONPOLICY, ORIGIN_ROLLOUT, [math.nan], [1.0], 1, log_q=[0.0], resid=[0.0])
    assert got.numel() == 0 and st.nonfinite_rows == 1 and float(st.best_energy[new]) == math.inf
    st.check()


def test_a_weight_sum_that_overflows_raises_naming_the_rule():
    st = make()

    def huge(store, rows, step):
        return torch.full((rows.numel(),), 1.5e308, dtype=torch.float64)
    with pytest.raises(CrystalStoreError, match="'huge'.*not finite"):
        st.draw_weighted(CHILD, 32, huge, 0)


def test_correction_is_exact_for_one_huge_weight():
    st = make(spec=((1, 5),))
    kids = torch.nonzero(st.kind[:st.high_water] == CHILD).flatten()

    def one(store, rows, step):
        return (rows == kids[0]).double() * 1.5e308
    rows, cor = st.draw_weighted(CHILD, 16, one, 0)
    assert (rows == kids[0]).all() and cor.tolist() == pytest.approx([0.2] * 16)


@pytest.mark.parametrize('breaks', ['template', 'stat'])
def test_a_failed_add_conditions_changes_nothing(breaks):
    st = make()
    st.cond_stats['seen'] = torch.zeros(2, dtype=torch.int32)
    templates = _templates(2, first=2)
    if breaks == 'template':
        templates.radius = templates.radius[:1]
    else:
        st.cond_stats['bad'] = torch.tensor(1.0)
    before = _fingerprint(st)
    with pytest.raises((CrystalStoreError, RuntimeError, IndexError)):
        st.add_conditions(2, templates)
    assert _fingerprint(st) == before
    if breaks == 'stat':
        del st.cond_stats['bad']
    st.check()


def test_add_conditions_extends_every_cond_stats_dtype():
    st = make()
    st.cond_stats.update(seen=torch.ones(2, dtype=torch.int32), hit=torch.ones(2, dtype=torch.bool),
                         spread=torch.ones((2, 3), dtype=torch.float32))
    st.add_conditions(2, _templates(2, first=2))
    assert st.cond_stats['seen'].tolist() == [1, 1, 0, 0] and st.cond_stats['seen'].dtype == torch.int32
    assert st.cond_stats['hit'].tolist() == [True, True, False, False]
    assert st.cond_stats['spread'].shape == (4, 3) and torch.isnan(st.cond_stats['spread'][2:]).all()
    st.check()


def test_round_trip_keeps_the_index_layout_and_compaction_count():
    rng = np.random.default_rng(0)
    st = make(spec=((2, 30), (1, 300), (1, 5)), max_rows=500, H=40, seed=5)
    for i in range(30):
        add_onpolicy(st, rng.integers(0, 3, 40).tolist(), step=i + 1)
    assert st.index.compactions > 0
    other = CrystalStore(DIM, 3, 500, 40, seed=77)
    other.apply_state(st.state_dict())
    other.check()
    assert other.stats() == st.stats()
    assert (other.index.start == st.index.start).all() and (other.index.cap == st.index.cap).all()
    assert other.index.tail == st.index.tail
    for i in range(300):
        c = rng.integers(0, 3, 40).tolist()
        assert torch.equal(add_onpolicy(st, c, step=100 + i), add_onpolicy(other, c, step=100 + i))
        assert torch.equal(st.draw(CHILD, 16), other.draw(CHILD, 16))
    assert other.stats() == st.stats()
