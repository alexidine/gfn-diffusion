"""The trainer's draws from the crystal store (train.py Modeller._store_draw, _kick_anchors, _kick_fresh, _store_touch), on
a Modeller built without __init__ around a real CrystalStore with real template batches. CPU, no energy computed.

    python -m pytest -q tests/crystal/test_row_store_trainer.py
"""
import pytest
import torch

from energy_sampling import crystal_store as CS
from energy_sampling.tests.crystal import store_stub as S
from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast


def test_mle_draw_is_reward_free_tiled_and_writes_nothing(monkeypatch):
    m = S.build()
    S.shift_kick(monkeypatch)
    st = m.row_store
    before = st.generation
    cond, cid, inds, x, log_r, mol, traj = m.draw_bwd_sample(2)
    assert inds is None and log_r is None and traj is None
    assert x.shape == (16, S.DIM) and torch.equal(x[0::2], x[1::2])
    assert m._mle_rows_drawn == 16 and cid.numel() == 16 and torch.equal(cid[0::2], cid[1::2])
    assert st.generation == before and bool(torch.isnan(st.log_q[:st.high_water]).all())
    assert torch.equal(mol.sg_ind.reshape(-1), st.cond_sg[cid].long())


def test_mle_draw_centres_are_anchors_and_the_model_never_sees_one(monkeypatch):
    m = S.build()
    seen = []
    real = S.shift_kick(monkeypatch)
    kick = S.physical_kick.kick

    def spy(kits, centres, *a, **kw):
        seen.append(centres.clone())
        return kick(kits, centres, *a, **kw)
    monkeypatch.setattr(S.physical_kick, 'kick', spy)
    st = m.row_store
    _, _, _, x, *_ = m.draw_bwd_sample(1)
    anchors = st.x[:st.high_water][st.kind[:st.high_water] == CS.ANCHOR]
    assert bool((torch.cdist(seen[0], anchors).min(1).values == 0).all()) and len(real) == 1
    assert bool((torch.cdist(x.float(), anchors).min(1).values > 0).all())


def test_a_kick_that_kept_its_anchor_is_redrawn_then_dropped_and_counted(monkeypatch):
    m = S.build()
    st = m.row_store
    a0 = st.x[0].clone()
    S.shift_kick(monkeypatch, keep=lambda c: (c == a0).all(1))         # anchor 0 never leaves its point
    rows = torch.zeros(4, dtype=torch.long)
    monkeypatch.setattr(st, 'draw_anchors', lambda n, by='anchor': torch.zeros(n, dtype=torch.long))
    with pytest.raises(RuntimeError, match='no MLE draw'):
        m._kick_fresh(rows)
    assert m._physical['mle_kept_dropped'] == 4


def test_a_kept_row_is_replaced_by_a_kick_of_another_anchor(monkeypatch):
    m = S.build()
    st = m.row_store
    a0 = st.x[0].clone()
    S.shift_kick(monkeypatch, keep=lambda c: (c == a0).all(1))
    other = int(torch.nonzero(st.kind[:st.high_water] == CS.ANCHOR).flatten()[1])
    monkeypatch.setattr(st, 'draw_anchors', lambda n, by='anchor': torch.full((n,), other))
    rows, x = m._kick_fresh(torch.tensor([0, other]))
    assert rows.tolist() == [other, other] and m._physical['mle_kept_dropped'] == 0
    assert not bool((x == a0).all(1).any())


def test_the_kick_uses_its_own_generator_and_leaves_the_global_ones_alone(monkeypatch):
    m = S.build()
    calls = S.shift_kick(monkeypatch)
    state = torch.get_rng_state()
    m._kick_anchors(torch.tensor([0, 1]))
    assert calls[0]['generator'] is m._physical['gen']
    assert torch.equal(torch.get_rng_state(), state)


def test_a_stage_that_reads_log_r_cannot_take_the_mle_draw(monkeypatch):
    m = S.build()
    S.shift_kick(monkeypatch)
    m._bwd_draw_is_reward_free = lambda: False
    with pytest.raises(ValueError, match='reads log R'):
        m.draw_bwd_sample(1)


def test_reward_free_is_judged_on_the_store_route_too():
    m = S.build()
    del m._bwd_draw_is_reward_free
    m.gfn_model._stash_live_branches = False
    m.args.fwd_loss_coeffs = m.args.bwd_loss_coeffs
    assert Modeller._bwd_draw_is_reward_free(m)
    m.protocol.stage.bwd_sampling_mode = 'prior'
    assert not Modeller._bwd_draw_is_reward_free(m)


def test_bwd_draw_reads_stored_children():
    m = S.build()
    m.protocol.stage.bwd_sampling_mode = 'prior'
    st = m.row_store
    cond, cid, inds, x, log_r, mol, traj = m.draw_bwd_sample(1)
    idx, epoch = inds
    assert epoch == st.epoch and traj is None and bool((st.kind[idx] == CS.CHILD).all())
    assert torch.equal(x, st.x[idx]) and torch.equal(log_r, st.log_r[idx]) and torch.equal(cid, st.cid[idx].long())
    assert torch.equal(mol.pos[:3], st.templates.subsample_new_batch(st.tmpl_of[st.cid[idx[:1]]].long()).pos)


def test_replay_draw_reads_on_policy_rows_with_a_one_step_path():
    m = S.build()
    st = m.row_store
    fwd, batch = S.rollout([0, 1, 2, 3])
    m._store_admit(fwd, batch, CS.ORIGIN_ROLLOUT)
    cond, cid, inds, x, log_r, mol, traj = m.draw_replay_sample(1)
    idx, _ = inds
    assert bool((st.kind[idx] == CS.ONPOLICY).all()) and traj.shape == (8, 2, S.DIM)
    assert torch.equal(traj[:, 1], x) and bool((traj[:, 0] == 0).all()) and m._replay_is_w is None
    assert mol.replay_log_T.shape == (8,)


def test_a_replay_weight_rule_reaches_the_loss_weights(monkeypatch):
    m = S.build()
    st = m.row_store
    fwd, batch = S.rollout([0, 0, 1])
    m._store_admit(fwd, batch, CS.ORIGIN_ROLLOUT)

    def rule(store, rows, step):
        return torch.ones(len(rows)) * 2

    monkeypatch.setitem(CS.REPLAY_WEIGHTS, 'twos', rule)
    m._store_cfg['replay_weight'] = 'twos'
    m._store_draw('replay', 1)
    assert m._replay_is_w.shape == (8,) and torch.allclose(m._replay_is_w, torch.ones(8))
    assert st.n_drawable(CS.ONPOLICY) == 3


def test_touch_writes_log_q_and_residual_and_mle_inds_write_nothing():
    m = S.build()
    m.protocol.stage.bwd_sampling_mode = 'prior'
    st = m.row_store
    _, _, inds, *_ = m.draw_bwd_sample(1)
    m._store_touch(inds, {'log_pf': torch.full((8,), -3.0), 'resid': torch.full((8,), 0.5)})
    assert bool((st.log_q[inds[0]] == -3.0).all()) and bool((st.resid[inds[0]] == 0.5).all())
    assert bool((st.q_step[inds[0]] == 5).all())
    gen = st.generation
    m._store_touch(None, {})
    assert st.generation == gen


def test_replay_rows_and_occupancy_read_the_store():
    m = S.build()
    assert m._replay_rows() == 0
    fwd, batch = S.rollout([0, 1])
    m._store_admit(fwd, batch, CS.ORIGIN_ROLLOUT)
    assert m._replay_rows() == 2
    assert m._rollout_trigger_reading('occupancy_min_batches', 0) == 2 / 8
    m.row_store = None
    assert m._replay_rows() is None


# -- the load: which file rows the store takes ------------------------------------------------------------------------

def _file(kinds, parents, groups, splits=None, shift=None):
    """A prior file of real graphs, one molecule per condition group."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    b = collate_data_list([S.molecule(int(g)) for g in groups])
    n = len(kinds)
    if shift is not None:
        for row, d in shift.items():
            b.pos[b.ptr[row]:b.ptr[row + 1]] += d
    b.add_graph_attr(torch.tensor(parents, dtype=torch.long), 'noised_parent')
    b.add_graph_attr(torch.linspace(-0.5, 0.5, n * 12).reshape(n, 12), 'fd_state')
    b.add_graph_attr(torch.tensor([S.GROUPS[0]] * n, dtype=torch.long), 'sg_ind')
    return (b, torch.tensor(kinds), torch.zeros(n, dtype=torch.long) if splits is None else torch.tensor(splits),
            torch.tensor(groups))


def _take(monkeypatch, b, kinds, splits, groups, keep, fd_ok=None, cap=8):
    from energy_sampling import noised_prior as NP
    monkeypatch.setattr(S.physical_kick, 'group_kits', lambda layout: {})
    m = Modeller.__new__(Modeller)
    m.energy_function = S.Energy()
    m.energy_function.fd_layout = None
    m.device = torch.device('cpu')
    m.args = S.SimpleNamespace(energy_config=S.SimpleNamespace(temperature=S.T), eval_num_samples=cap // 4, seed=0)
    m._noised_prior_settings = dict(live_log_range=(-2.0, -1.0), window_kT=10.0, packing_window=(0.5, 0.9))
    n = len(kinds)
    keep = torch.tensor(keep)
    masks = dict(buffer=keep, heldout_children=torch.zeros(n, dtype=torch.bool),
                 anchors=keep & (kinds == NP.KIND_ANCHOR))
    log_r = torch.full((n,), -1.0, dtype=torch.float64)
    fd = torch.ones(n, dtype=torch.bool) if fd_ok is None else torch.tensor(fd_ok)
    out = m._store_take_rows(b, {}, torch.zeros(n, dtype=torch.float64), log_r, kinds, splits, groups, fd, masks,
                             torch.zeros(n, dtype=torch.float64))
    return m, out


def _kinds():
    from energy_sampling import noised_prior as NP
    return NP.KIND_ANCHOR, NP.KIND_CHILD


def test_the_load_stores_a_kept_childs_anchor_undrawn_when_the_anchor_left_the_window(monkeypatch):
    a, c = _kinds()
    b, kinds, splits, groups = _file([a, c, c, a, c], [0, 0, 0, 3, 3], [0, 0, 0, 1, 1])
    m, (energy, panel) = _take(monkeypatch, b, kinds, splits, groups, [False, True, True, True, True])
    load = m._store_load
    assert load['rows'].tolist() == [0, 1, 2, 3, 4] and load['undrawn'].tolist() == [True, False, False, False, False]
    assert load['parent'].tolist() == [-1, 0, 0, -1, 3]
    assert energy.numel() == 4 and panel.num_graphs == 3         # scored rows in the windows; the children
    assert m._physical['no_parent'] == 0


def test_the_load_drops_a_child_whose_anchor_has_no_chart_point(monkeypatch):
    a, c = _kinds()
    b, kinds, splits, groups = _file([a, c, a, c], [0, 0, 2, 2], [0, 0, 1, 1])
    m, _ = _take(monkeypatch, b, kinds, splits, groups, [True, True, True, True], fd_ok=[False, True, True, True])
    assert m._store_load['rows'].tolist() == [0, 2, 3] and m._physical['no_parent'] == 1
    assert m._store_load['parent'].tolist() == [-1, -1, 1]       # the parent column is remapped to stored positions


def test_the_load_refuses_a_row_off_its_conditions_template_frame(monkeypatch):
    a, c = _kinds()
    b, kinds, splits, groups = _file([a, c], [0, 0], [0, 0], shift={1: torch.tensor([0.01, 0.0, 0.0])})
    with pytest.raises(ValueError, match='share a frame'):
        _take(monkeypatch, b, kinds, splits, groups, [True, True])


def test_the_load_refuses_a_child_whose_parent_is_another_condition(monkeypatch):
    a, c = _kinds()
    b, kinds, splits, groups = _file([a, a, c], [0, 1, 0], [0, 1, 1])
    with pytest.raises(ValueError, match='another condition'):
        _take(monkeypatch, b, kinds, splits, groups, [True, True, True])


def test_a_training_condition_with_no_stored_row_takes_its_molecules_template():
    from mxtaltools.dataset_utils.utils import collate_data_list
    m = Modeller.__new__(Modeller)
    m.energy_function = S.Energy()
    t = S.templates(1).subsample_new_batch(torch.tensor([0]))        # molecule 0 in group 2 only
    mols = collate_data_list([S.molecule(0)])
    mols.add_graph_attr(torch.tensor([0]), 'mol_id')
    mols.add_graph_attr(torch.tensor([S.GROUPS[0]]), 'sg_ind')
    del mols['identifier']
    m.mol_dataset = S.SimpleNamespace(batch=mols)
    m._store_load = dict(templates=t, t_groups=torch.tensor([0]), groups=torch.tensor([0, 0]))
    m._store_after_identifiers()
    load = m._store_load
    assert load['template_cids'].tolist() == [0, 1] and load['cid'].tolist() == [0, 0]
    assert load['templates'].sg_ind.reshape(-1).tolist() == list(S.GROUPS)
    assert torch.equal(load['templates'].pos[:3], load['templates'].pos[3:6])
