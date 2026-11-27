"""The store's temperature-free terms and the rebuild of log R from them (energies/molecular_crystal.py
MolecularCrystal.store_terms_from, .log_r_from_terms, ._tempered_density, the penalty_nats / log_reference_nats keys of
generator_energy), on synthetic eLJ crystals of the FD chart scored on the CPU. The bar is 1e-4 + 1e-6 |log R| nats.

    python -m pytest -q tests/crystal/test_log_r_from_terms.py
"""
import math
from pathlib import Path

import pytest
import torch

import utils
from energies.molecular_crystal import STORE_TERM_KEYS, MolecularCrystal
from mxtaltools.common.utils import log_rescale_positive
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

CANONICAL = Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'
N = 16


def _molecule(i):
    pos = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1]], dtype=torch.float32) * 1.6
    return MolData(z=torch.tensor([6, 7, 8], dtype=torch.long), pos=pos, x=torch.zeros((3, 1), dtype=torch.float32),
                   identifier=f'synthetic-{i}', mol_volume=torch.tensor(55.0 * 1.6 ** 3), mass=torch.tensor(42.0),
                   radius=torch.tensor(1.2 * 1.6), z_prime=torch.tensor(1, dtype=torch.long))


def _molecules(n=N):
    return collate_data_list([_molecule(i) for i in range(n)])


@pytest.fixture(scope='module')
def ef():
    args = utils.get_train_args(['--config', str(CANONICAL)])
    cfg = {'device': 'cpu', 'energy_function': args.energy_function, 'mlip_path': args.mlip_path,
           'space_groups': [2], 'z_primes': args.z_primes, 'sg_conditioning': args.sg_conditioning,
           'temperature_conditioning': args.temperature_conditioning, 'zp_conditioning': args.zp_conditioning}
    cfg.update(vars(args.energy_config))
    cfg.update(crystal_coordinates='fd', space_groups=[2])
    return MolecularCrystal(**cfg)


@pytest.fixture(scope='module')
def states(ef):
    g = torch.Generator().manual_seed(3)
    s = torch.rand(N, ef.data_ndim, generator=g) * 1.8 - 0.9
    s[:4, 1] = 1.05            # four rows outside the box: a bounding penalty in the live score
    return s


@pytest.fixture(autouse=True)
def reset(ef):
    yield
    ef.energy_clip = None
    ef.physical_energy_clip = None
    ef.density_reference_temperature = None


def live(ef, s, T):
    """(log R [N], scored batch) of the rows at kT T, through energy()'s chain (T = 10 ** log10 T)."""
    log_T = torch.full((len(s),), math.log10(T))
    energy, batch = ef.analyze_crystal_batch(s, _molecules(len(s)), 10 ** log_T, return_batch=True)
    return -energy / 10 ** log_T, batch, log_T


def close(a, b):
    a, b = a.double(), b.double()
    bar = 1e-4 + 1e-6 * b.abs()
    return bool(torch.isfinite(a).all() and torch.isfinite(b).all() and ((a - b).abs() <= bar).all())


def _arm_clip(ef, s):
    """An energy clip low enough that both log rescales fire on some rows and not on others."""
    _, batch, _ = live(ef, s, 2.5)
    u = (batch.mol_energy + ef.density_coeff * batch.density_energy + batch.pressure_energy).double()
    lo = u.sort().values
    ef.energy_clip = float(0.5 * (lo[0] + lo[1]))               # between the two lowest: the rest clipped twice
    first = log_rescale_positive(u, ef.energy_clip) + 25.0 * batch.penalty_nats.double()
    second = first > ef.energy_clip + 0.1 * abs(ef.energy_clip)
    assert bool((u > ef.energy_clip).any()) and bool((u <= ef.energy_clip).any())
    assert bool(second.any()) and not bool(second.all())


def test_scored_rows_carry_the_four_terms(ef, states):
    _, batch, _ = live(ef, states, 2.5)
    terms = ef.store_terms_from(batch)
    assert terms.shape == (N, 4) and terms.dtype == torch.float32 and bool(torch.isfinite(terms).all())
    assert STORE_TERM_KEYS == ('density_energy', 'pressure_energy', 'penalty_nats', 'log_reference_nats')
    assert float(terms[:4, 2].min()) > 0 and float(terms[4:, 2].abs().max()) < 1e-3    # bounding on the 4 outside
    assert torch.equal(terms[:, 3], batch.fd_log_reference.float())
    torch.testing.assert_close(terms[:, 3] * -2.5, batch.jacobian_energy.float(), rtol=1e-6, atol=1e-5)


def test_a_saved_batch_drops_the_analysis_only_terms_and_merges_with_an_older_file(ef, states):
    """noised_prior.strip_analysis_helpers (every prior build, merge and split writes through it) drops penalty_nats
    and log_reference_nats, so a file built now merges (merge_batches: the same keys on both sides) with one built
    before generator_energy wrote them; density_energy and pressure_energy are row data and stay."""
    import noised_prior
    from energies.molecular_crystal import ANALYSIS_ONLY_KEYS, merge_batches
    _, batch, _ = live(ef, states, 2.5)
    assert all(hasattr(batch, k) for k in ANALYSIS_ONLY_KEYS)
    new = noised_prior.strip_analysis_helpers(batch.clone())
    assert not any(k in new.keys() for k in ANALYSIS_ONLY_KEYS)
    assert {'density_energy', 'pressure_energy'} <= set(new.keys())
    old = new.clone()                    # a file built before the two keys existed: the same keys as a stripped one
    assert int(merge_batches([old, new]).num_graphs) == 2 * N


@pytest.mark.parametrize('T', [2.5, 5.0, 25.0])
@pytest.mark.parametrize('clip', [None, 'armed'])
@pytest.mark.parametrize('physical', [None, 50.0])
@pytest.mark.parametrize('ref', [None, 2.5])
def test_terms_rebuild_the_live_reward(ef, states, T, clip, physical, ref):
    ef.density_reference_temperature = ref
    if clip:
        _arm_clip(ef, states)
    ef.physical_energy_clip = physical
    log_r, batch, log_T = live(ef, states, T)
    rebuilt = ef.log_r_from_terms(batch.mol_energy, ef.store_terms_from(batch), log_T)
    assert close(rebuilt, log_r), (rebuilt - log_r).abs().max()


@pytest.mark.parametrize('ref', [None, 2.5])
@pytest.mark.parametrize('scored_at,rebuilt_at', [(2.5, 25.0), (25.0, 2.5), (5.0, 10.0)])
def test_terms_scored_at_one_temperature_give_the_reward_at_another(ef, states, ref, scored_at, rebuilt_at):
    ef.density_reference_temperature = ref
    _arm_clip(ef, states)
    _, batch, _ = live(ef, states, scored_at)
    target, _, log_T = live(ef, states, rebuilt_at)
    rebuilt = ef.log_r_from_terms(batch.mol_energy, ef.store_terms_from(batch), log_T)
    assert close(rebuilt, target), (rebuilt - target).abs().max()


@pytest.mark.parametrize('T1,T2', [(2.5, 2.5), (2.5, 25.0), (25.0, 5.0)])
def test_prebuilt_rows_rebuild_with_no_bounding(ef, states, T1, T2):
    ef.density_reference_temperature = 2.5
    _arm_clip(ef, states)
    _, batch, _ = live(ef, states, 2.5)
    prebuilt, ens = ef.prebuilt_sample_to_reward(batch, T1, return_ens_dict=True)
    terms = ef.store_terms_from(ens)
    assert float(terms[:, 2].abs().max()) < 1e-3                 # no raw latents: no bounding term
    want = ef.prebuilt_sample_to_reward(batch, T2)
    rebuilt = ef.log_r_from_terms(ens['mol_energy'], terms, math.log10(T2))
    assert close(rebuilt, want), (rebuilt - want).abs().max()
    if T1 == T2:
        assert close(rebuilt, prebuilt)


def test_tempered_wall_costs_the_same_nats_at_every_temperature(ef, states):
    ef.density_reference_temperature = 2.5
    _, batch, _ = live(ef, states, 2.5)
    terms = ef.store_terms_from(batch)
    loose = terms[:, 0] > 0                                      # packing under the window: a density wall
    assert bool(loose.any()), batch.packing_coeff
    no_wall = terms.clone()
    no_wall[:, 0] = 0.0
    for T in (2.5, 25.0):
        lt = math.log10(T)
        gap = ef.log_r_from_terms(batch.mol_energy, terms, lt) - ef.log_r_from_terms(batch.mol_energy, no_wall, lt)
        want = -ef.density_coeff * terms[:, 0] / 2.5
        torch.testing.assert_close(gap[loose].double(), want[loose].double(), rtol=1e-4, atol=1e-3)


def test_no_reference_temperature_is_todays_formula(ef, states):
    d = torch.rand(5)
    assert ef.density_reference_temperature is None and ef._tempered_density(d, 25.0) is d
    _, batch, _ = live(ef, states, 2.5)
    today = ef.prebuilt_sample_to_reward(batch, 2.5)
    ef.density_reference_temperature = 2.5                       # w = 2.5 / 2.5 = 1.0 exactly at the run's T
    assert torch.equal(ef.prebuilt_sample_to_reward(batch, 2.5), today)


def test_rebuild_raises_where_the_terms_do_not_determine_log_r(ef):
    e, t = torch.zeros(2), torch.zeros(2, 4)
    with pytest.raises(ValueError, match='shape'):
        ef.log_r_from_terms(e, torch.zeros(2, 3), 0.4)
    for attr, value, match in (('energy_reference_mode', 'seed_min', 'energy reference'),
                               ('prior_flow', object(), 'prior_flow'),
                               ('latent_energy', True, 'physical crystal energy'),
                               ('crystal_coordinates', 'latent', "crystal_coordinates 'fd'")):
        old = getattr(ef, attr)
        setattr(ef, attr, value)
        try:
            with pytest.raises(ValueError, match=match):
                ef.log_r_from_terms(e, t, 0.4)
        finally:
            setattr(ef, attr, old)


def test_rebuild_raises_under_an_energy_reference(capsys):
    args = utils.get_train_args(['--config', str(CANONICAL)])
    capsys.readouterr()
    cfg = {'device': 'cpu', 'energy_function': args.energy_function, 'mlip_path': args.mlip_path,
           'space_groups': [2], 'z_primes': args.z_primes, 'temperature_conditioning': False}
    cfg.update(vars(args.energy_config))
    cfg.update(crystal_coordinates='fd', space_groups=[2], energy_reference='seed_min')
    ref = MolecularCrystal(**cfg)
    with pytest.raises(ValueError, match='energy reference'):
        ref.log_r_from_terms(torch.zeros(1), torch.zeros(1, 4), 0.4)
