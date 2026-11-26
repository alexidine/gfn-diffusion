"""Temperature conditioning on the crystal store route: the weights-only warm start that adds the temperature column
(checkpointing.py Checkpointer._retempered_state, ._adds_temperature_column), the MLE draw's effective-temperature
labels (train.py Modeller._kick_temperature), and the store route's checks (config_invariants.row_store_problems). CPU.

    python -m pytest -q tests/models/test_temperature_column.py
"""
import math
from types import SimpleNamespace

import pytest
import torch

from energy_sampling import config_invariants as ci
from energy_sampling.checkpointing import Checkpointer
from energy_sampling.models.nf_policy import NFPolicy
from energy_sampling.tests.models.test_nf_policy import SMALL, gfn_config

pytestmark = pytest.mark.fast
RANGE = (-2.5, 1.5)


def molecule_policy(n_cond, seed=0):
    cfg = gfn_config(True, cond_dim=n_cond)
    cfg['conditions_type'] = 'molecule'
    torch.manual_seed(seed)
    return NFPolicy(cfg, {**SMALL, 'conditioner': 'transformer', 'tf_width': 32, 'tf_layers': 1, 'tf_heads': 2,
                          'atom_tokens': True, 'atom_node_dim': 16, 'atom_convs': 1}, seed=seed)


def molecules(n=3):
    counts = torch.tensor([3, 5, 4][:n])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    torch.manual_seed(1)
    return SimpleNamespace(z=torch.randint(1, 9, (int(counts.sum()),)), pos=torch.randn(int(counts.sum()), 3),
                           batch=torch.repeat_interleave(torch.arange(n), counts), ptr=ptr, num_graphs=n)


def ckpt(n_cond, rng=RANGE, tcond=True):
    mgr = Checkpointer.__new__(Checkpointer)
    mgr.modeller = SimpleNamespace(args=SimpleNamespace(energy_config=SimpleNamespace(log_temperature_range=list(rng)),
                                                        temperature_conditioning=tcond),
                                   get_conditioning_dim=lambda: n_cond)
    return mgr


def test_the_added_temperature_column_leaves_the_loaded_policy_unchanged_at_every_temperature():
    old, new = molecule_policy(0), molecule_policy(1, seed=5)
    mb = molecules()
    with torch.no_grad():
        states = old.get_traj_fwd(torch.zeros(3, 12), None, 0, None, mb)[0]
    state = ckpt(1)._retempered_state(new, old.state_dict(), 'test')
    new.load_state_dict(state, strict=True)
    _, want, _, flow_want = old.get_traj_bwd(states[:, 1], None, None, mb)
    for log_t in (-2.0, 0.4, 1.4):
        _, got, _, flow_got = new.get_traj_bwd(states[:, 1], None, torch.full((3, 1), log_t), mb)
        assert torch.allclose(got, want, atol=1e-5), log_t
        assert torch.allclose(flow_got[:, 0], flow_want[:, 0], atol=1e-5), log_t


def test_the_column_is_zero_weight_and_the_scaler_spans_the_declared_range():
    old, new = molecule_policy(0), molecule_policy(1)
    state = ckpt(1)._retempered_state(new, old.state_dict(), 'test')
    lo, hi = RANGE
    assert torch.allclose(state['sampler.flow.c_mu'][1:], torch.tensor([0.5 * (lo + hi)]))
    assert torch.allclose(state['sampler.flow.c_sd'][1:], torch.tensor([(hi - lo) / math.sqrt(12)]))
    for head in ('context', 'level'):
        w_old, w_new = old.state_dict()[f'sampler.flow.{head}.inp.weight'], state[f'sampler.flow.{head}.inp.weight']
        assert w_new.shape[1] == w_old.shape[1] + 1
        assert bool((w_new[:, -1] == 0).all()) and torch.equal(w_new[:, :-1], w_old)


def test_a_state_of_the_same_width_is_left_alone():
    pol = molecule_policy(1)
    state = pol.state_dict()
    assert ckpt(1)._retempered_state(pol, state, 'test') is state


@pytest.mark.parametrize('run_dim, held, tcond, adds', [(1, 0, True, True), (1, 1, True, False), (2, 0, True, False),
                                                        (1, 0, False, False)])
def test_the_temperature_column_is_added_only_for_one_column_more_under_temperature_conditioning(run_dim, held,
                                                                                                tcond, adds):
    assert ckpt(run_dim, tcond=tcond)._adds_temperature_column({'conditions_dim': held}) is adds


def test_temp_cond_and_the_temperature_range_may_be_exempted():
    mgr = Checkpointer.__new__(Checkpointer)
    mgr.modeller = SimpleNamespace(args=SimpleNamespace(
        warm_start_ignore_problem_keys=['temp_cond', 'energy_config.log_temperature_range']))
    assert mgr._warm_start_ignore_keys() == ('temp_cond', 'energy_config.log_temperature_range')


# -- the MLE draw's labels -------------------------------------------------------------------------------------------

def test_a_kick_is_labelled_kappa_atoms_rho_squared_over_the_live_coordinates():
    from energy_sampling.train import Modeller
    m = Modeller.__new__(Modeller)
    st = SimpleNamespace(cond_sg=torch.tensor([2, 14]), cid=torch.tensor([0, 1, 1]))
    m.row_store = st
    m._physical = {'kits': {2: SimpleNamespace(live=torch.ones(12)),
                            14: SimpleNamespace(live=torch.tensor([1.0] * 10 + [0.0] * 2))}}
    m._noised_prior_settings = {'kick_kappa_per_atom': 15.0}
    mol = SimpleNamespace(ptr=torch.tensor([0, 10, 30, 45]))   # 10, 20 and 15 atoms
    rho = torch.tensor([0.1, 0.2, 0.05], dtype=torch.float64)
    got = m._kick_temperature(torch.tensor([0, 1, 2]), rho, mol)
    want = torch.tensor([15 * 10 * 0.01 / 12, 15 * 20 * 0.04 / 10, 15 * 15 * 0.0025 / 10])
    assert torch.allclose(got, want.float())


# -- the store route's checks ----------------------------------------------------------------------------------------

def _problems(monkeypatch, stages, tcond=True, kappa=15.0, rng=(-2.5, 1.5)):
    store = dict(ci.ROW_STORE_DEFAULTS, enabled=True)
    cfg = {'buffers': {'store': store}, 'temperature_conditioning': tcond,
           'noised_prior': {'kick_kappa_per_atom': kappa}, 'energy_config': {'log_temperature_range': rng},
           'bwd_loss_coeffs': {'mle': 1.0, 'tb': 0.0}}
    monkeypatch.setattr(ci, 'active_stages', lambda c: stages)
    return [m for sev, m in ci.row_store_problems(cfg) if sev == ci.ERROR]


MLE = {'name': 'train_prior', 'fracs': {'fwd': 0.0, 'bwd': 1.0, 'replay': 0.0}, 'max_steps': 10,
       'loss_coeffs': {'bwd': {'mle': 1.0, 'tb': 0.0}}}
TB = {'name': 'tb', 'fracs': {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}, 'max_steps': 10, 'fwd_rollout_every': 20,
      'loss_coeffs': {'bwd': {'tb': 1.0}}}


def test_temperature_conditioning_passes_on_an_mle_only_protocol(monkeypatch):
    assert not [m for m in _problems(monkeypatch, [MLE]) if 'temperature' in m or 'kick_kappa' in m]


def test_temperature_conditioning_is_refused_with_a_stage_that_reads_rewards(monkeypatch):
    assert any('reads rewards' in m for m in _problems(monkeypatch, [MLE, TB]))


def test_temperature_conditioning_needs_the_kick_stiffness_and_a_range(monkeypatch):
    assert any('kick_kappa_per_atom' in m for m in _problems(monkeypatch, [MLE], kappa=None))
    assert any('log_temperature_range' in m for m in _problems(monkeypatch, [MLE], rng=None))
    assert any('log_temperature_range' in m for m in _problems(monkeypatch, [MLE], rng=(1.0, 0.0)))


def test_the_kick_stiffness_without_temperature_conditioning_is_a_dead_key(monkeypatch):
    assert any('dead key' in m for m in _problems(monkeypatch, [MLE], tcond=False))
