"""The stored-reward pass and its sidecar cache (train.py Modeller._stored_reward_terms, _stored_reward_terms_fresh) on a
stub batch and energy function. CPU, no energy computed.

    python -m pytest -q tests/crystal/test_reward_cache.py
"""
from types import SimpleNamespace

import pytest
import torch

from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast


class Batch:
    """The slice of a torch_geometric batch the pass reads: one value per row, a per-row list, subsample, move."""

    def __init__(self, vals, names=None):
        self.store = {'vals': vals, 'identifier': list(names or [f'm{i}' for i in range(len(vals))])}
        self.ptr = torch.arange(len(vals) + 1)
        self.moved_with = []

    @property
    def num_graphs(self):
        return len(self.store['vals'])

    def keys(self):
        return list(self.store)

    def __getitem__(self, k):
        return self.store[k]

    def __delitem__(self, k):
        del self.store[k]

    def subsample_new_batch(self, at):
        out = Batch(self.store['vals'][at])
        if 'identifier' not in self.store:
            del out.store['identifier']
        out.moved_with = self.moved_with
        return out

    def to(self, device):
        self.moved_with.append(sorted(self.store))
        return self

    def add_graph_attr(self, v, name):
        self.store[name] = v


class Energy:
    energy_clip = None
    physical_energy_clip = None
    energy_reference = None

    def __init__(self):
        self.rows_scored = 0

    def prebuilt_sample_to_reward(self, sub, T, return_ens_dict=False):
        v = sub['vals'].double()
        self.rows_scored += len(v)
        off = sub.store['condition_id'].double() if 'condition_id' in sub.store else 0.0
        return -(v + off) / T, {'mol_energy': v * 2}

    @staticmethod
    def seed_energy_from(ens):
        return ens['mol_energy'] / 2


def modeller(tmp_path, n=50):
    m = Modeller.__new__(Modeller)
    m.energy_function = Energy()
    m.device = torch.device('cpu')
    m.args = SimpleNamespace(energy_config=SimpleNamespace(temperature=2.0), energy_function='stub')
    prior = tmp_path / 'prior.pt'
    prior.write_bytes(b'x' * 100)
    return m, Batch(torch.arange(n, dtype=torch.float32)), str(prior)


def test_each_chunk_drops_the_row_lists_before_its_move(tmp_path):
    m, b, _ = modeller(tmp_path)
    m._stored_reward_terms(b, chunk=16)
    assert b.moved_with and all('identifier' not in keys for keys in b.moved_with)
    assert 'identifier' in b.store                       # the caller's batch keeps its lists


def test_the_cache_returns_what_the_pass_computed_and_rescores_only_its_check_rows(tmp_path):
    m, b, path = modeller(tmp_path, n=2000)
    rows = torch.arange(1999, -1, -1)
    cids = torch.arange(2000) % 7
    first = m._stored_reward_terms(b, chunk=128, with_mol_energy=True, rows=rows, cids=cids, prior_path=path)
    assert m.energy_function.rows_scored == 2000
    m.energy_function.rows_scored = 0
    second = m._stored_reward_terms(b, chunk=128, with_mol_energy=True, rows=rows, cids=cids, prior_path=path)
    assert all(torch.equal(x, y) for x, y in zip(first, second))
    assert 0 < m.energy_function.rows_scored <= Modeller.REWARD_CACHE_CHECK_ROWS + 32


@pytest.mark.parametrize('change', ['rows', 'cids', 'clip', 'temperature', 'mol'])
def test_a_changed_input_misses_the_cache(tmp_path, change):
    m, b, path = modeller(tmp_path, n=300)
    rows, cids, mol = torch.arange(300), torch.zeros(300, dtype=torch.long), False
    m._stored_reward_terms(b, rows=rows, cids=cids, with_mol_energy=mol, prior_path=path)
    if change == 'rows':
        rows = torch.arange(299, -1, -1)
    elif change == 'cids':
        cids = torch.ones(300, dtype=torch.long)
    elif change == 'clip':
        m.energy_function.energy_clip = 400.0
    elif change == 'temperature':
        m.args.energy_config.temperature = 3.0
    else:
        mol = True
    m.energy_function.rows_scored = 0
    m._stored_reward_terms(b, rows=rows, cids=cids, with_mol_energy=mol, prior_path=path)
    assert m.energy_function.rows_scored == 300                  # recomputed in full


def test_a_stale_cache_is_recomputed(tmp_path):
    m, b, path = modeller(tmp_path, n=400)
    m._stored_reward_terms(b, prior_path=path)
    b.store['vals'] = b.store['vals'] + 1.0                      # the scoring changed under the same key
    m.energy_function.rows_scored = 0
    out = m._stored_reward_terms(b, prior_path=path)
    assert m.energy_function.rows_scored > 400                  # the check rows, then every row
    assert torch.equal(out[0], (torch.arange(400, dtype=torch.float32) + 1.0).double())


def test_no_prior_path_means_no_cache(tmp_path):
    m, b, _ = modeller(tmp_path, n=100)
    m._stored_reward_terms(b)
    m.energy_function.rows_scored = 0
    m._stored_reward_terms(b)
    assert m.energy_function.rows_scored == 100
    assert not list(tmp_path.glob('*.rewards-*'))


def test_a_sidecar_that_cannot_be_written_leaves_the_run_running(tmp_path, monkeypatch):
    from energy_sampling import noised_prior
    m, b, path = modeller(tmp_path, n=120)
    real = noised_prior.cache_path

    def blocked(prior_path, kind, key):                 # a directory where the temporary file must go
        side = real(prior_path, kind, key)
        import os
        os.makedirs(f'{side}.tmp{os.getpid()}', exist_ok=True)
        return side
    monkeypatch.setattr(noised_prior, 'cache_path', blocked)
    out = m._stored_reward_terms(b, prior_path=path)
    assert torch.equal(out[0], torch.arange(120, dtype=torch.float32).double())
    assert not list(tmp_path.glob('*.rewards-*.pt'))


class TermEnergy(Energy):
    """Energy with the store's T-free terms (MolecularCrystal.store_terms_from): four columns of the row's value."""

    @staticmethod
    def store_terms_from(ens):
        v = ens['mol_energy'].float().flatten() / 2
        return torch.stack([v, v + 1, v + 2, v + 3], 1)


def test_store_terms_round_trip_through_the_cache_under_their_own_key(tmp_path):
    m, b, path = modeller(tmp_path, n=300)
    m.energy_function = TermEnergy()
    plain = m._stored_reward_terms(b, with_mol_energy=True, prior_path=path)
    assert len(plain) == 3
    m.energy_function.rows_scored = 0
    first = m._stored_reward_terms(b, with_mol_energy=True, prior_path=path, with_store_terms=True)
    assert m.energy_function.rows_scored == 300                   # its key is not the plain call's: scored in full
    assert len(first) == 4 and first[3].shape == (300, 4) and first[3].dtype == torch.float32
    assert torch.equal(first[3][:, 0], torch.arange(300, dtype=torch.float32))
    assert all(torch.equal(x, y) for x, y in zip(plain, first[:3]))
    m.energy_function.rows_scored = 0
    second = m._stored_reward_terms(b, with_mol_energy=True, prior_path=path, with_store_terms=True)
    assert all(torch.equal(x, y) for x, y in zip(first, second))
    assert 0 < m.energy_function.rows_scored <= Modeller.REWARD_CACHE_CHECK_ROWS + 32   # read from the cache
    assert len(list(tmp_path.glob('*.rewards-*.pt'))) == 2


class Coded(TermEnergy):
    """A reward path with every method the code digest may hash."""

    def _tempered_density(self, density_energy, temperature):
        return density_energy

    @staticmethod
    def store_terms_from(ens):
        return 1

    def log_r_from_terms(self, energy, terms, log_T):
        return 1


class TermsEdited(Coded):
    @staticmethod
    def store_terms_from(ens):
        return 2

    def log_r_from_terms(self, energy, terms, log_T):
        return 2


class DensityEdited(Coded):
    def _tempered_density(self, density_energy, temperature):
        return density_energy * 1


def test_the_code_digest_hashes_the_term_code_only_when_the_cache_holds_terms():
    """A cache without term columns misses on an edit to the code computing its values (with _tempered_density, which
    generator_energy calls on every run), not on an edit to the term code; one with them misses on both."""
    def digest(ef, terms):
        m = Modeller.__new__(Modeller)
        m.energy_function = ef
        return m._reward_code_digest(with_store_terms=terms)
    assert digest(Coded(), False) == digest(TermsEdited(), False)
    assert digest(Coded(), True) != digest(TermsEdited(), True)
    assert digest(Coded(), False) != digest(Coded(), True)
    assert digest(Coded(), False) != digest(DensityEdited(), False)
    assert digest(Coded(), True) != digest(DensityEdited(), True)
