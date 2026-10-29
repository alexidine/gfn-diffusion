"""The trainer's side of the row store (train.py Modeller._store_* and the sites that dispatch to them), on a
Modeller built without __init__ around a hand-built RowStore and a stub policy. What each test would catch:

  * the bwd draw reading anything but the data rows, or handing back a molecule batch or a trajectory;
  * the replay draw not carrying its IS weights and stats to replay_train_step, or a wrong trajectory shape;
  * the level table not being the replay branch's TB level (the head, or the persistent target where trusted);
  * a rollout on a condition the store has never held being refused for want of a level;
  * the admission floor judged on the referenced log R;
  * records judged on anything but the rollout batch's mol_energy, or a batch without it admitted silently;
  * seed rows given any energy but the stored rows' mol_energy, or best_energy read off held-out rows;
  * replay availability and the occupancy trigger not reading the store.

CPU, synthetic rows, no energy computed.

    python -m pytest -q tests/crystal/test_row_store_trainer.py
"""
from types import SimpleNamespace

import pytest
import torch

from energy_sampling.row_store import RowStore, TAG_ANCHOR, TAG_CHILD, TAG_ONPOLICY
from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast

STORE = dict(bar=1.0, eps=0.1, cap_per_condition=100, max_rows=1000, records_kept=2, kappa=1.0,
             floor_frac=0.25, uniform_frac=0.1, refresh_chunk=7, admit_reward_min=None)
DIM, N_COND = 3, 4


class _Head(torch.nn.Module):
    """log Z(c) = 10 * c[:, 0] (a conditional level head reading the condition vector)."""

    def forward(self, emb):
        return 10.0 * emb[:, 0]


class _Policy(torch.nn.Module):
    """A conditional stub with the NF policy's interface: log q(x | c) = -|x|^2 - c[:, 0]."""
    conditional = True

    def __init__(self):
        super().__init__()
        self.flow_model = _Head()

    def get_condition_embedding(self, condition, mol_batch):
        return condition

    def log_density(self, x, condition, mol_batch):
        return -(x ** 2).sum(-1) - condition[:, 0]


class _Ef:
    temperature = 1.0

    def unreferenced_log_r(self, log_r, cid):
        return log_r - 100.0      # a reference of -100 / T on every condition


def _modeller(tb_z_source='learned', tracker=None, **store_over):
    m = Modeller.__new__(Modeller)
    s = {**STORE, **store_over}
    m.args = SimpleNamespace(buffers=SimpleNamespace(store=SimpleNamespace(enabled=True, **s)),
                             replay_loss_coeffs=SimpleNamespace(tb_z_source=tb_z_source))
    m.gfn_model = _Policy()
    m.device = torch.device('cpu')
    m.batch_size = 32
    m.step_ind = 5
    m.energy_function = _Ef()
    if tracker is not None:
        m.condition_log_z = tracker
    st = RowStore(DIM, N_COND, 1, 'cpu', **{k: v for k, v in s.items() if k != 'refresh_chunk'})
    g = torch.Generator().manual_seed(0)
    n = 40
    cid = torch.arange(n) % 2                          # conditions 0 and 1 only
    held = torch.arange(n) >= 30
    st.set_conditions(cid, (cid.float() * 0.1)[:, None])
    x = torch.randn(n, DIM, generator=g)
    log_r = torch.randn(n, generator=g)
    st.seed(cid, x, log_r, TAG_CHILD, held, energy=torch.full((n,), -1.0))   # best physical energy -1 on c0, c1
    m.row_store = st
    m._store_refresh()
    return m


def test_bwd_draw_is_uniform_over_data_rows_and_carries_no_graph():
    m = _modeller()
    cond, cid, inds, lat, log_r, mol_batch, traj = m.draw_bwd_sample(2)
    idx, version = inds
    assert mol_batch is None and traj is None and version == m.row_store.version
    assert idx.numel() == 64 and bool((idx < m.row_store.n_mle).all())
    assert bool((idx[0::2] == idx[1::2]).all())        # repeats tile each drawn row in place
    st = m.row_store
    assert torch.equal(lat, st.x[idx]) and torch.equal(log_r, st.log_r[idx]) and torch.equal(cid, st.cid[idx])
    assert torch.equal(cond, st.cond_table[st.cid[idx]])


def test_replay_draw_sets_is_weights_and_a_one_step_trajectory():
    m = _modeller()
    m._replay_is_w = None
    cond, cid, inds, lat, log_r, mol_batch, traj = m.draw_replay_sample(1)
    idx, _ = inds
    assert mol_batch is None and traj.shape == (32, 2, DIM)
    assert torch.equal(traj[:, 1], lat) and bool((traj[:, 0] == 0).all())
    assert bool((~m.row_store.held_out[idx]).all())
    assert m._replay_is_w.shape == (32,) and set(m._replay_is_stats) == {
        'replay/is_ess_frac', 'replay/is_w_max_ratio', 'replay/is_elig_frac'}
    assert m._replay_draw_origin == {}


def test_level_table_is_the_head_and_nan_where_no_condition_is_stored():
    m = _modeller()
    lv = m._store_level_table()
    assert torch.allclose(lv[:2], torch.tensor([0.0, 1.0])) and bool(torch.isnan(lv[2:]).all())


def test_level_table_under_persistent_reads_the_trusted_target():
    tracker = SimpleNamespace(
        lookup=lambda u: (torch.full((len(u),), 7.0), torch.as_tensor(u) == 1),
        untrusted_fallback=lambda: None)
    m = _modeller(tb_z_source='persistent', tracker=tracker)
    lv = m._store_level_table()
    assert torch.allclose(lv[:2], torch.tensor([0.0, 7.0]))   # condition 0 untrusted: the head


def test_refresh_writes_log_q_under_eval_mode_and_restores_training():
    m = _modeller()
    m.gfn_model.train()
    st = m.row_store
    st.log_q.fill_(float('nan'))
    m._store_refresh()
    want = -(st.x ** 2).sum(-1) - st.cond_table[st.cid, 0]
    assert torch.allclose(st.log_q, want) and m.gfn_model.training


def _rollout(cid, log_r, x=None, mol_energy=None):
    n = len(cid)
    x = torch.zeros(n, DIM) if x is None else x
    fwd = {'condition_id': torch.as_tensor(cid), 'log_r': torch.as_tensor(log_r, dtype=torch.float32),
           'flow_states': torch.stack([torch.zeros_like(x), x], 1), 'log_pf': torch.zeros(n)}
    mol_energy = -torch.as_tensor(log_r, dtype=torch.float32) if mol_energy is None else mol_energy
    batch = SimpleNamespace(conditions=(torch.as_tensor(cid).float() * 0.1)[:, None],
                            mol_energy=torch.as_tensor(mol_energy, dtype=torch.float32)[:, None])
    return fwd, batch


def test_admission_on_a_condition_the_store_never_held():
    m = _modeller()
    fwd, batch = _rollout([3, 3], [50.0, 0.0])        # condition 3: no seed row, cond_table unset
    res = m._store_admit(fwd, batch, origin=0)
    st = m.row_store
    assert res['admitted'] == 2 and res['records'] == 1
    assert torch.allclose(st.cond_table[3], torch.tensor([0.3]))
    assert int((st.tag == TAG_ONPOLICY).sum()) == 2 and float(st.best_energy[3]) == -50.0


def test_records_are_judged_on_the_batch_mol_energy_not_log_r():
    m = _modeller(bar=1e6)                              # only records are admitted
    # c0 (seed best -1): row 0 has the higher log R but energy 5; row 1 the lower log R and energy -3 -> the record
    fwd, batch = _rollout([0, 0], [30.0, -20.0], mol_energy=[5.0, -3.0])
    res = m._store_admit(fwd, batch, origin=0)
    st = m.row_store
    assert res['records'] == 1 and res['admitted'] == 1
    assert float(st.log_r[-1]) == -20.0 and float(st.energy[-1]) == -3.0 and bool(st.protected[-1])
    assert float(st.best_energy[0]) == -3.0 and float(st.best_energy[1]) == -1.0


def test_admission_refuses_a_batch_without_matching_mol_energy():
    m = _modeller()
    fwd, batch = _rollout([0, 0], [1.0, 2.0])
    del batch.mol_energy
    with pytest.raises(RuntimeError, match='mol_energy'):
        m._store_admit(fwd, batch, origin=0)
    fwd, batch = _rollout([0, 0], [1.0, 2.0], mol_energy=[1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match='mol_energy'):
        m._store_admit(fwd, batch, origin=0)
    assert len(m.row_store) == 40


class _PriorRows:
    """A stored prior batch: per-row noised_kind, space group (read as the condition id), log R and mol_energy."""

    def __init__(self, kind, sg, log_r, mol_energy, x):
        self.noised_kind, self.sg_ind = torch.as_tensor(kind), torch.as_tensor(sg)
        self.log_r, self.mol_energy, self.x = torch.as_tensor(log_r), torch.as_tensor(mol_energy), torch.as_tensor(x)
        self.num_graphs = len(self.sg_ind)
        self.z_prime = torch.ones(self.num_graphs, dtype=torch.long)
        self.ptr = torch.arange(self.num_graphs + 1)

    def subsample_new_batch(self, idx):
        return _PriorRows(self.noised_kind[idx], self.sg_ind[idx], self.log_r[idx], self.mol_energy[idx], self.x[idx])

    def to(self, device):
        return self


class _SeedEf:
    """seed_energy_from differs from mol_energy by 1000, so a seed taking it is caught."""
    condition_library_size = N_COND

    def prebuilt_sample_to_reward(self, sub, T, return_ens_dict=False):
        return sub.log_r, {'mol_energy': sub.mol_energy}

    def seed_energy_from(self, ens):
        return ens['mol_energy'] + 1000.0

    def condition_samples(self, sub, sg_inds, z_primes):
        return sub, None, sg_inds.float()[:, None] * 0.1, sg_inds


def test_seeding_stores_mol_energy_and_best_energy_from_the_drawable_rows():
    m = Modeller.__new__(Modeller)
    m.args = SimpleNamespace(energy_config=SimpleNamespace(temperature=1.0), buffer_device='cpu')
    m.device = torch.device('cpu')
    m.energy_function = _SeedEf()
    m._batch_latents = lambda sub: sub.x
    n = 7
    m.prior_dataset = SimpleNamespace(batch=_PriorRows(
        [TAG_ANCHOR, TAG_CHILD, TAG_CHILD, TAG_CHILD, TAG_ANCHOR, TAG_CHILD, TAG_CHILD],
        [0, 0, 0, 1, 1, 1, 1], [1.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [-2.0, 7.0, -5.0, -1.0, -3.0, 4.0, 0.5], torch.arange(n * DIM, dtype=torch.float32).reshape(n, DIM)))
    m.prior_heldout_batch = _PriorRows([TAG_ANCHOR, TAG_ANCHOR], [0, 1], [50.0, 60.0], [-90.0, -80.0],
                                       -torch.ones(2, DIM))
    settings = {k: v for k, v in STORE.items() if k != 'refresh_chunk'}
    m._store_seed(settings, chunk=3)
    st = m.row_store
    assert st.n_mle == 7 and st.n_seed == 9 and bool(st.held_out[7:].all())
    assert st.energy.tolist() == [-2.0, 7.0, -5.0, -1.0, -3.0, 4.0, 0.5, -90.0, -80.0]   # mol_energy rides its row
    assert st.log_r.tolist() == [1.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 50.0, 60.0]
    assert st.best_energy.tolist()[:2] == [-5.0, -3.0]                       # the held-out -90 and -80 do not count
    assert Modeller._stored_reward_terms(m, m.prior_dataset.batch).__len__() == 2   # the default call is unchanged


def test_admission_floor_is_judged_on_the_unreferenced_log_r():
    m = _modeller(admit_reward_min=-60.0)
    fwd, batch = _rollout([0, 0], [50.0, 30.0])        # unreferenced: -50 (kept), -70 (refused)
    res = m._store_admit(fwd, batch, origin=0)
    assert res['rejected_floor'] == 1 and res['admitted'] == 1


def test_manage_replay_buffer_admits_to_the_store_and_builds_no_replay_buffer():
    m = _modeller()
    m.args.harvest = None
    m._replay_managed = True
    m.replay_in_play = lambda: True
    fwd, batch = _rollout([0, 1], [40.0, 41.0])
    n0 = len(m.row_store)
    m.manage_replay_buffer(fwd, batch)
    assert len(m.row_store) == n0 + 2 and not hasattr(m, 'replay_buffer')


def test_replay_rows_and_occupancy_read_the_store():
    m = _modeller()
    assert m._replay_rows() == 30                       # drawable: the non-held-out rows
    assert m._rollout_trigger_reading('occupancy_min_batches', 0) == 30 / 32
    m.row_store = None
    assert m._replay_rows() is None and m._rollout_trigger_reading('occupancy_min_batches', 0) is None
    m.replay_buffer = [0] * 7
    assert m._replay_rows() == 7


def test_touch_writes_log_pf_of_the_drawn_rows():
    m = _modeller()
    _, _, inds, *_ = m.draw_bwd_sample(1)
    m._store_touch(inds, {'log_pf': torch.full((32,), -3.0)})
    assert bool((m.row_store.log_q[inds[0]] == -3.0).all()) and bool((m.row_store.q_step[inds[0]] == 5).all())


def test_stats_are_published_through_log_buffer_stats_keys():
    m = _modeller()
    keys = m.row_store.stats(m._store_level_table())
    assert keys['store/rows'] == 40 and keys['store/rows_heldout'] == 10


@pytest.mark.parametrize('conditional', [True, False])
def test_store_residual_and_is_weights_match_the_replay_tb_loss_on_a_real_flow_policy(conditional):
    """On models/nf_policy.NFPolicy: the store's level table is the replay TB loss's log Z, its r is minus the loss's
    residual, the replay loss reduces as sum(w l) / sum(w) with the draw's IS weights, and touch writes the loss's
    log P_F."""
    import torch.nn.functional as F
    from types import SimpleNamespace as NS
    from models.nf_policy import NFPolicy
    from gflownet_losses import get_gfn_backward_loss
    torch.manual_seed(0)
    gcfg = dict(dim=12, s_emb_dim=32, conditions_dim=4 if conditional else 0, harmonics_dim=16, t_dim=8,
                t_hidden_dim=32, s_hidden_dim=32, s_layers=2, policy_hidden_dim=32, policy_layers=2,
                flow_hidden_dim=32, flow_layers=2, cond_hidden_dim=32, cond_layers=2, log_var_range=6.0, t_scale=0.05,
                learned_variance=True, condition_embedding_dim=16 if conditional else 0, conditions_type='vector',
                clipping=True, gfn_clip=200.0, pb_drift_range=0.4, pb_var_range=6.0, conditional=conditional,
                learn_pb=True, dropout=0, norm='layer', zero_init=False, device='cpu', max_z_prime=1,
                full_flow=False, do_periodic_angles=True, periodic_centroids=True, periodic_centroid_axes=(1, 2),
                dead_latent_rows=None, dead_latent_values=None, dplr_rank=0, pb_exact_reversal=True)
    pol = NFPolicy(gcfg, dict(conditioner='made', hidden=32, hidden_layers=1, n_bins=8, context_dim=16,
                              context_hidden=32, tied_derivatives=False, tf_fp32=False), seed=0)
    with torch.no_grad():                               # a level head away from its zero init
        for p in pol.flow_model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    m = _modeller()
    m.gfn_model = pol
    st = RowStore(12, 3, 4, 'cpu', **{k: v for k, v in STORE.items() if k != 'refresh_chunk'})
    n = 40
    cid = torch.arange(n) % 3
    st.set_conditions(cid, torch.randn(3, 4)[cid])
    st.seed(cid, torch.rand(n, 12) * 2 - 1, torch.randn(n) * 5, TAG_CHILD, torch.arange(n) >= 35,
            energy=torch.zeros(n))
    m.row_store = st
    m._store_refresh()
    lv = m._store_level_table()
    cond, c_id, inds, lat, log_r, _, traj = m.draw_replay_sample(1)
    beta = 10.0
    coeffs = NS(freeze_policy=0, freeze_z=0, traj_grads=1, beta=beta, vg_lb=0, vg_lme=0, pf_boost=0, db=0,
                subtb=0, emp_z=0, tb=1, mle=0, loss_clip=-1)
    loss, d = get_gfn_backward_loss(coeffs, lat, pol, log_r, None, None, condition=cond if conditional else None,
                                    repeats=1, report_losses=True, trajectories=traj, condition_id=c_id,
                                    sample_weights=m._replay_is_w)
    assert torch.allclose(lv[c_id], d['log_Z'], atol=1e-4)
    assert torch.allclose(st.resid(lv, inds[0]), -d['resid'], atol=1e-4)
    per = beta * F.smooth_l1_loss(d['resid'], torch.zeros_like(d['resid']), reduction='none', beta=beta)
    w = m._replay_is_w
    assert torch.allclose(loss.detach(), (w * per).sum() / w.sum(), rtol=1e-5)
    m._store_touch(inds, d)
    assert torch.allclose(st.log_q[inds[0]], d['log_pf'])
