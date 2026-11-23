"""Phase 2 on the crystal store (train.py Modeller._store_admit, _admit_eval_rollout, _store_heldout_panel,
_store_check_currency, _store_seed_keep, _store_rewind), on a Modeller built without __init__ around a real
CrystalStore. CPU, no energy computed but a stub's.

    python -m pytest -q tests/crystal/test_store_tb.py
"""
import pytest
import torch
from types import SimpleNamespace

from energy_sampling import crystal_store as CS
from energy_sampling.buffer import ORIGIN_BOOTSTRAP, ORIGIN_EVAL, ORIGIN_ROLLOUT
from energy_sampling.tests.crystal import store_stub as S
from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast


def test_every_scored_rollout_row_is_written_with_its_log_q_and_residual():
    m = S.build()
    st = m.row_store
    fwd, batch = S.rollout([0, 1, 2, 3], log_r=[-1.0, -2.0, -3.0, -4.0], log_pf=[-5.0, -6.0, -7.0, -8.0])
    slots = m._store_admit(fwd, batch, ORIGIN_ROLLOUT)
    assert slots.numel() == 4 and st.n_drawable(CS.ONPOLICY) == 4
    assert st.log_q[slots].tolist() == [-5.0, -6.0, -7.0, -8.0] == st.log_q_birth[slots].tolist()
    assert torch.allclose(st.resid[slots], torch.tensor([-1.0, -1.0, -1.0, -1.0]))    # log_pf - log_pb - log_r + log Z
    assert bool((st.q_step[slots] == 5).all()) and bool((st.birth_step[slots] == 5).all())
    assert m._growth_owed == 4
    st.check()


def test_non_finite_rows_are_dropped_or_stored_undrawn():
    m = S.build()
    st = m.row_store
    fwd, batch = S.rollout([0, 1, 2], log_r=[float('nan'), -1.0, -1.0], log_pf=[-1.0, float('nan'), -1.0])
    slots = m._store_admit(fwd, batch, ORIGIN_ROLLOUT)
    assert slots.numel() == 2 and st.nonfinite_rows == 1
    assert st.n_drawable(CS.ONPOLICY) == 1                  # the row with no finite log q is kept and never drawn
    assert m._growth_owed == 2
    st.check()


@pytest.mark.parametrize('drawable', [True, False])
def test_an_eval_rollout_is_tagged_and_drawable_only_when_a_tb_stage_scored_it(drawable):
    m = S.build()
    m.args.harvest = None
    st = m.row_store
    gen = st.generation
    fwd, batch = S.rollout([0, 1])
    m._admit_eval_rollout(fwd, batch, drawable=drawable)
    assert st.stats()['store/onpolicy'] == 2 and st.n_drawable(CS.ONPOLICY) == (2 if drawable else 0)
    slots = torch.nonzero(st.kind[:st.high_water] == CS.ONPOLICY).flatten()
    assert bool(((st.flags[slots] & CS.EVAL) != 0).all()) and bool((st.origin[slots] == ORIGIN_EVAL).all())
    assert (st.generation > gen) == drawable                # an MLE eval write is not durable
    assert m._growth_owed == 0                              # an eval rollout owes no kicks
    assert not hasattr(m, 'prior_buffer')


def test_a_rollout_is_admitted_in_a_stage_with_no_replay_in_play():
    m = S.build()
    m.args.harvest = None
    m.replay_in_play = lambda: False
    fwd, batch = S.rollout([0, 1, 2])
    m.manage_replay_buffer(fwd, batch, origin=ORIGIN_ROLLOUT)
    assert m.row_store.n_drawable(CS.ONPOLICY) == 3 and m._growth_owed == 3
    assert not hasattr(m, 'replay_buffer')


def test_a_bootstrap_routed_through_the_replay_manager_is_stored_undrawn():
    m = S.build()
    m.args.harvest = None
    fwd, batch = S.rollout([0, 1])
    m.manage_replay_buffer(fwd, batch, origin=ORIGIN_BOOTSTRAP)
    st = m.row_store
    assert st.stats()['store/onpolicy'] == 2 and st.n_drawable(CS.ONPOLICY) == 0 and m._growth_owed == 0


def test_the_heldout_panel_is_fixed_and_balanced_over_groups():
    m = S.build()
    mol_id = torch.tensor([4, 5, 6, 7, 8])
    m.test_mol_dataset = SimpleNamespace(batch=SimpleNamespace(mol_id=mol_id, num_graphs=5))
    rows, sg = m._store_heldout_panel(16)
    assert rows.numel() == 16 == sg.numel()                 # 4 molecules x 2 groups x 2 draws
    assert rows.unique().numel() == 4
    for r in rows.unique():
        assert sorted(sg[rows == r].tolist()) == sorted(list(S.GROUPS) * 2)
    again = m._store_heldout_panel(16)
    assert again[0] is rows and again[1] is sg
    other = S.build()
    other.test_mol_dataset = m.test_mol_dataset
    assert torch.equal(other._store_heldout_panel(16)[0], rows)     # deterministic in the seed


def test_a_panel_too_small_for_every_group_keeps_one_molecule():
    m = S.build()
    m.test_mol_dataset = SimpleNamespace(batch=SimpleNamespace(mol_id=torch.tensor([1, 2]), num_graphs=2))
    rows, sg = m._store_heldout_panel(1)
    assert rows.numel() == 4 and rows.unique().numel() == 1


def test_the_currency_check_passes_on_true_rows_and_raises_on_a_stale_one():
    m = S.build()
    st = m.row_store
    x = torch.full((2, S.DIM), 0.25)
    sq = (x ** 2).sum(-1)
    cid = torch.tensor([0, 1])
    fwd, batch = S.rollout(cid, x=x, log_r=-sq - cid / 10, energy=sq + cid)
    slots = m._store_admit(fwd, batch, ORIGIN_ROLLOUT)
    m._store_check_currency()
    st.log_r[slots[0]] += 1.0
    with pytest.raises(RuntimeError, match='currency check'):
        m._store_check_currency()


def test_the_currency_check_reads_no_global_generator():
    m = S.build()
    state = torch.get_rng_state()
    m._store_check_currency()
    assert torch.equal(torch.get_rng_state(), state)


def test_an_own_run_resume_seeds_from_the_saved_row_mask():
    first = S.build()
    first.run_name = 'run_a'
    first._row_store_state, first._row_store_source = None, (None, None)
    was = torch.tensor([True, False, True, True, False, False, True, False, True, True, False])
    assert torch.equal(first._store_seed_keep(was), was)

    moved = was.clone()
    moved[[1, 2]] = ~moved[[1, 2]]   # a window edge that moved by rounding on another node
    own = S.build()
    own.run_name = 'run_a'
    own._row_store_state, own._row_store_source = {'seed_keep': first._store_seed_mask}, ('run_a', 7)
    assert torch.equal(own._store_seed_keep(moved), was)
    assert torch.equal(own._store_seed_mask['bits'], first._store_seed_mask['bits'])

    other = S.build()
    other.run_name = 'run_b'
    other._row_store_state, other._row_store_source = {'seed_keep': first._store_seed_mask}, ('run_a', 7)
    assert torch.equal(other._store_seed_keep(moved), moved)

    short = S.build()
    short.run_name = 'run_a'
    short._row_store_state, short._row_store_source = {'seed_keep': first._store_seed_mask}, ('run_a', 7)
    with pytest.raises(ValueError, match='file rows'):
        short._store_seed_keep(moved[:-1])


def test_a_rewind_invalidates_only_later_log_q():
    m = S.build()
    m.protocol.stage.bwd_sampling_mode = 'prior'
    st = m.row_store
    _, _, inds, *_ = m.draw_bwd_sample(1)
    m._store_touch(inds, {'log_pf': torch.full((8,), -3.0), 'resid': torch.full((8,), 0.5)})
    assert m._store_rewind(5) == 0 and bool((st.log_q[inds[0]] == -3.0).all())
    assert m._store_rewind(4) > 0 and bool(torch.isnan(st.log_q[inds[0]]).all())


def test_store_tb_loss_matches_the_loss_of_the_same_rows():
    """On models/nf_policy.NFPolicy: the replay draw's rows through get_gfn_backward_loss, and the columns _store_touch
    writes are the loss's log P_F and signed residual."""
    import torch.nn.functional as F
    from types import SimpleNamespace as NS
    from models.nf_policy import NFPolicy
    from gflownet_losses import get_gfn_backward_loss
    torch.manual_seed(0)
    gcfg = dict(dim=12, s_emb_dim=32, conditions_dim=1, harmonics_dim=16, t_dim=8, t_hidden_dim=32, s_hidden_dim=32,
                s_layers=2, policy_hidden_dim=32, policy_layers=2, flow_hidden_dim=32, flow_layers=2,
                cond_hidden_dim=32, cond_layers=2, log_var_range=6.0, t_scale=0.05, learned_variance=True,
                condition_embedding_dim=16, conditions_type='vector', clipping=True, gfn_clip=200.0,
                pb_drift_range=0.4, pb_var_range=6.0, conditional=True, learn_pb=True, dropout=0, norm='layer',
                zero_init=False, device='cpu', max_z_prime=1, full_flow=False, do_periodic_angles=True,
                periodic_centroids=True, periodic_centroid_axes=(1, 2), dead_latent_rows=None,
                dead_latent_values=None, dplr_rank=0, pb_exact_reversal=True)
    pol = NFPolicy(gcfg, dict(conditioner='made', hidden=32, hidden_layers=1, n_bins=8, context_dim=16,
                              context_hidden=32, tied_derivatives=False, tf_fp32=False), seed=0)
    m = S.build(dim=12)
    st = m.row_store
    fwd, batch = S.rollout([0, 1, 2, 3], x=torch.full((4, 12), 0.25))
    m._store_admit(fwd, batch, ORIGIN_ROLLOUT)
    cond, c_id, inds, lat, log_r, mol, traj = m.draw_replay_sample(1)
    beta = 10.0
    coeffs = NS(freeze_policy=0, freeze_z=1, traj_grads=1, beta=beta, vg_lb=0, vg_lme=0, pf_boost=0, db=0, subtb=0,
                emp_z=0, tb=1, mle=0, loss_clip=-1)
    loss, d = get_gfn_backward_loss(coeffs, lat, pol, log_r, None, None, condition=cond, repeats=1, report_losses=True,
                                    trajectories=traj, condition_id=c_id, sample_weights=m._replay_is_w)
    per = beta * F.smooth_l1_loss(d['resid'], torch.zeros_like(d['resid']), reduction='none', beta=beta)
    assert torch.allclose(loss.detach(), per.mean(), rtol=1e-5)
    m._store_touch(inds, d)
    last = {int(r): i for i, r in enumerate(inds[0].tolist())}
    at = torch.tensor(list(last))
    pick = torch.tensor(list(last.values()))
    assert torch.allclose(st.log_q[at], d['log_pf'][pick].float()) and torch.allclose(st.resid[at], d['resid'][pick].float())
    assert bool(torch.isfinite(st.log_q_birth[at]).all())
