"""
The lambda mix between the prior-flow density and the physical energy.

WHAT MUST HOLD, and why each is load-bearing:

  lambda=0  ->  the target is EXACTLY the fitted prior density. Not
                approximately: the null test's whole value is that the correct
                outcome is known in advance, so any leftover term (a jacobian, a
                reduction penalty, a temperature factor) turns a clean prediction
                into an unexplained drift that reads as a sampler failure.
  lambda=1  ->  bitwise the pre-existing physical energy, so every shipped config
                is untouched by this feature existing.
  TEMPERATURE-INVARIANT at lambda=0. energy() divides the total by temperature
                before use. A log-density is not a Boltzmann energy and must not
                be tempered, or the lambda=0 target becomes q^(1/T) -- a flatter
                distribution than the prior, and not a null test at all.

    python test_lambda_mix.py
"""
import math
import os
import sys

import torch

CPU = torch.device('cpu')
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402
from energy_sampling.energies.molecular_crystal import MolecularCrystal  # noqa: E402
from energy_sampling.energies.prior_flow import PriorFlow  # noqa: E402
from energy_sampling.models.dead_latent_rows import resolve_dead_rows  # noqa: E402

DATASET = os.path.abspath(os.path.join(_here, '..', '..', 'mxtaltools',
                                       'mini_datasets', 'mini_new_csd.pt'))
SG, DIM = 2, 12
MASK = [i in (10, 11) for i in range(DIM)]     # periodic_centroids OFF -> phi, r only
_MOL = None


def mol_batch(n):
    global _MOL
    if _MOL is None:
        data = torch.load(DATASET, weights_only=False)
        c = [e for e in data if int(e.z_prime) == 1 and bool(e.is_well_defined)
             and not bool(e.cocrystal)]
        c.sort(key=lambda e: int(e.num_nodes))
        _MOL = c[0]
    b = collate_data_list([_MOL.clone() for _ in range(n)])
    b.reset_sg_info(SG)
    return b


def fitted_flow(path, seed=0):
    """A small, quickly-fitted flow. Quality is irrelevant here -- only the wiring."""
    g = torch.Generator().manual_seed(seed)
    x = 0.3 * torch.randn(20000, DIM, generator=g)
    x[:, MASK] = 2.0 * torch.rand(20000, 2, generator=g) - 1.0
    x = x.clamp(-0.95, 0.95)
    f = PriorFlow(n_blocks=2, n_bins=6, hidden=(64, 64), steps=400,
                  time_budget=12.0, seed=seed, device=CPU)
    f.fit(x, MASK, period=2.0)
    f.save(path, traj_T=20,
           dead_rows=resolve_dead_rows(SG, is_crystal=True, max_z_prime=1))
    return f


def energy_fn(path=None, lam=1.0, temperature=1.0):
    # lambda_mix is passed ALWAYS, including with no flow -- that combination is
    # the one the loud-failure guard exists for, and a helper that omitted it
    # would leave the guard untested while looking covered.
    kw = dict(lambda_mix=lam)
    if path is not None:
        kw['prior_flow_path'] = path
    return MolecularCrystal(
        device=CPU, energy_function='latent_gaussian', space_groups=[SG], z_primes=(1,),
        temperature=temperature, bounding_coeff=1.0, reduction_coeff=1.0,
        reward_range=None, internal_oom_recovery=False,
        host_gas_phase_reference=False,
        analyze_kwargs={'c': [0.5] * DIM, 'width': 0.4}, **kw)


def in_box(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return 0.6 * (2.0 * torch.rand(n, DIM, generator=g) - 1.0)


# --------------------------------------------------------------------------


def test_lambda_zero_is_exactly_the_flow(tmp_path):
    path = str(tmp_path / 'flow.pt')
    f = fitted_flow(path)
    ef = energy_fn(path, lam=0.0, temperature=1.0)
    n = 16
    x = in_box(n)
    total, _ = ef.analyze_crystal_batch(x, mol_batch(n), temperature=torch.ones(n))
    cb = ef.instantiate_crystals(x, mol_batch(n))
    lat = cb.latent_params(gauge_fix_free_axes=ef.is_crystal)
    expect = f.energy(lat)
    assert torch.allclose(total, expect, atol=1e-4), \
        (total[:3].tolist(), expect[:3].tolist())


def test_lambda_zero_is_temperature_invariant(tmp_path):
    """A log-density must not be tempered, or the target becomes q^(1/T)."""
    path = str(tmp_path / 'flow.pt')
    fitted_flow(path)
    n = 12
    x = in_box(n, seed=2)
    rewards = []
    for T in (1.0, 6.9):
        ef = energy_fn(path, lam=0.0, temperature=T)
        rewards.append(ef.log_reward(x, mol_batch(n), torch.full((n,), math.log10(T))))
    assert torch.allclose(rewards[0], rewards[1], atol=1e-4), \
        (rewards[0][:3].tolist(), rewards[1][:3].tolist())
    # negative control: at lambda=1 the PHYSICAL energy must still temper
    phys = []
    for T in (1.0, 6.9):
        ef = energy_fn(path, lam=1.0, temperature=T)
        phys.append(ef.log_reward(x, mol_batch(n), torch.full((n,), math.log10(T))))
    assert not torch.allclose(phys[0], phys[1], atol=1e-2)


def test_lambda_one_is_bitwise_the_old_energy(tmp_path):
    """Shipped configs must be untouched by this feature existing."""
    path = str(tmp_path / 'flow.pt')
    fitted_flow(path)
    n = 16
    x = in_box(n, seed=3)
    with_flow = energy_fn(path, lam=1.0)
    without = energy_fn(None)
    a, _ = with_flow.analyze_crystal_batch(x, mol_batch(n), temperature=torch.ones(n))
    b, _ = without.analyze_crystal_batch(x, mol_batch(n), temperature=torch.ones(n))
    assert torch.equal(a, b), (a[:3].tolist(), b[:3].tolist())


def test_lambda_without_a_flow_is_fatal():
    """Otherwise the knob is silently inert -- the failure mode this repo bleeds from."""
    try:
        energy_fn(None, lam=0.0)
    except ValueError as exc:
        assert 'lambda_mix' in str(exc)
    else:
        raise AssertionError('lambda_mix was accepted with no prior_flow_path')
    energy_fn(None)          # the no-op value stays legal without a flow


def test_flow_rejects_a_trajectory_length_mismatch(tmp_path):
    path = str(tmp_path / 'flow.pt')
    fitted_flow(path)
    loaded = PriorFlow.load(path, device=CPU)
    dead = resolve_dead_rows(SG, is_crystal=True, max_z_prime=1)
    loaded.verify_against_policy(MASK, dead_rows=dead, traj_T=20)      # matches

    try:
        loaded.verify_against_policy(MASK, dead_rows=dead, traj_T=100)
    except ValueError as exc:
        assert 'T=' in str(exc) and 'problem_def' in str(exc)
    else:
        raise AssertionError('a T mismatch was accepted; nothing else would catch it')

    try:
        loaded.verify_against_policy([True] * DIM, dead_rows=dead, traj_T=20)
    except ValueError as exc:
        assert 'wrapped' in str(exc)
    else:
        raise AssertionError('a wrap-mask mismatch was accepted')


def test_energy_clip_and_mixing_are_mutually_exclusive(tmp_path):
    path = str(tmp_path / 'flow.pt')
    fitted_flow(path)
    ef = MolecularCrystal(
        device=CPU, energy_function='latent_gaussian', space_groups=[SG], z_primes=(1,),
        temperature=1.0, reward_range=10.0, internal_oom_recovery=False,
        host_gas_phase_reference=False, analyze_kwargs={'c': [0.5] * DIM, 'width': 0.4},
        prior_flow_path=path, lambda_mix=0.0)
    ef.set_reward_clip([0.0, 1.0, 2.0])       # switches energy_clip on
    n = 8
    try:
        ef.analyze_crystal_batch(in_box(n, seed=4), mol_batch(n),
                                 temperature=torch.ones(n))
    except ValueError as exc:
        assert 'energy_clip' in str(exc)
    else:
        raise AssertionError('a clipped mixture was scored; the lambda=0 endpoint '
                             'would not have been the flow')


def test_jacobian_and_reduction_scale_with_lambda(tmp_path):
    """The shipped route (elj) has both LIVE, and neither belongs to the flow.

    A change of measure from box-latent to physical coordinates, and a physical
    validity penalty, are properties of the PHYSICAL target. Left unscaled they
    would sit on top of the lambda=0 fixed point, which would then not be the
    fitted density -- and the null test would report a drift with no cause.
    The latent energies used elsewhere in this file zero both structurally, so
    the physical branch is forced on here to exercise the arithmetic.
    """
    path = str(tmp_path / 'flow.pt')
    f = fitted_flow(path)
    n = 12
    x = in_box(n, seed=7)

    ef0 = energy_fn(path, lam=0.0)
    ef0.latent_energy = False                 # force jacobian + reduction live
    tot0, d0 = ef0.analyze_crystal_batch(x, mol_batch(n), temperature=torch.ones(n))

    ef1 = energy_fn(path, lam=1.0)
    ef1.latent_energy = False
    tot1, d1 = ef1.analyze_crystal_batch(x, mol_batch(n), temperature=torch.ones(n))

    cb = ef0.instantiate_crystals(x, mol_batch(n))
    lat = cb.latent_params(gauge_fix_free_axes=ef0.is_crystal)
    assert torch.allclose(tot0, f.energy(lat), atol=1e-4),         'lambda=0 picked up a physical term; the fixed point is not the flow'
    assert not torch.allclose(tot0, tot1, atol=1e-2),         'lambda made no difference with the physical branch live'


def test_jacobian_and_reduction_do_not_fire_at_lambda_zero(tmp_path):
    """Both belong to the PHYSICAL leg and must vanish at lambda=0.

    The jacobian is a change of measure from box-latent to physical coordinates,
    needed only because a physical target is DEFINED in physical space. The flow
    is fitted in latent space, so at lambda=0 there is no measure to correct, and
    leaving it on would move the fixed point off the fitted density -- surfacing
    later as a drift with no cause, which is the one reading the null test exists
    to exclude. Same argument for the reduction penalty.

    The latent energies used above zero both structurally, so the physical branch
    is forced on here; the shipped elj route has both live and needs real cluster
    construction, which no test fixture in this repo can currently drive.
    """
    path = str(tmp_path / 'flow.pt')
    f = fitted_flow(path)
    n = 12
    x = in_box(n, seed=7)

    ef0 = energy_fn(path, lam=0.0)
    ef0.latent_energy = False                    # force jacobian + reduction live
    tot0, _ = ef0.analyze_crystal_batch(x, mol_batch(n), temperature=torch.ones(n))

    ef1 = energy_fn(path, lam=1.0)
    ef1.latent_energy = False
    tot1, _ = ef1.analyze_crystal_batch(x, mol_batch(n), temperature=torch.ones(n))

    cb = ef0.instantiate_crystals(x, mol_batch(n))
    lat = cb.latent_params(gauge_fix_free_axes=ef0.is_crystal)
    assert torch.allclose(tot0, f.energy(lat), atol=1e-4),         'lambda=0 picked up a physical term; the fixed point is not the flow'
    assert not torch.allclose(tot0, tot1, atol=1e-2),         'lambda made no difference with the physical branch live'


def _legs(ef, x, n):
    """(total, batch) via the REAL path. analyze_crystal_batch copies every
    ens_dict key onto the batch, so reading the legs off the returned batch tests
    the delivery mechanism the buffers will actually use -- not just that
    generator_energy computed them. generator_energy cannot be called directly
    here: it needs a batch that crystal_batch.analyze() has already scored."""
    return ef.analyze_crystal_batch(x, mol_batch(n), temperature=torch.ones(n),
                                    return_batch=True)


def test_legs_reconstruct_the_total_exactly(tmp_path):
    """The whole buffer-currency design rests on this: the mix is LINEAR, so a
    row carrying both endpoints can be re-scored at any lambda with no rescore."""
    path = str(tmp_path / 'flow.pt')
    fitted_flow(path)
    n, lam = 16, 0.3
    x = in_box(n, seed=7)
    total, cb = _legs(energy_fn(path, lam=lam), x, n)
    e0, e1 = cb.flow_energy, cb.physical_energy
    # bounding sits OUTSIDE the legs deliberately: it is a penalty on the policy's
    # RAW output, and a stored row re-scored through prebuilt_sample_to_reward has
    # none. Keeping it out is what makes a stored pair of legs re-mixable into the
    # row's live energy by a plain weighted sum, with no correction term.
    bound = cb.bounding_energy * 1.0        # bounding_coeff is 1.0 in energy_fn
    assert torch.allclose((1.0 - lam) * e0 + lam * e1 + bound, total, atol=1e-5), \
        'legs + bounding do not reconstruct the mixed total'


def test_the_legs_do_not_move_with_lambda(tmp_path):
    """A STORED leg must stay valid at every lambda the run later reaches -- if
    the endpoints themselves drifted, storing them would buy nothing over storing
    the mixture, and the buffer would go stale exactly as it does today."""
    path = str(tmp_path / 'flow.pt')
    fitted_flow(path)
    n = 16
    x = in_box(n, seed=8)
    _, a = _legs(energy_fn(path, lam=0.2), x, n)
    _, b = _legs(energy_fn(path, lam=0.9), x, n)
    for key in ('flow_energy', 'physical_energy'):
        assert torch.equal(getattr(a, key), getattr(b, key)),             f'{key} depends on lambda; it must not'


def test_lambda_free_runs_publish_only_the_physical_leg(tmp_path):
    """Every shipped config is lambda-free, so no lambda=0 endpoint may appear to
    confuse a consumer that finds one, and the physical leg plus bounding must be
    the whole total. (Bitwise identity of the lambda-free TOTAL against the
    pre-feature energy is pinned separately, by
    test_lambda_one_is_bitwise_the_old_energy.)"""
    del tmp_path
    n = 16
    x = in_box(n, seed=9)
    total, cb = _legs(energy_fn(None), x, n)
    assert not hasattr(cb, 'flow_energy'), \
        'a lambda-free run published a lambda=0 endpoint it has no flow to define'
    assert torch.allclose(cb.physical_energy + cb.bounding_energy, total.cpu(),
                          atol=1e-5), \
        'the physical leg plus bounding is not the total on a lambda-free run'


def test_physical_leg_matches_the_lambda_free_total(tmp_path):
    """The leg a consumer reads instead of the mixture (anchor filtering) must be
    the SAME number a lambda-free run would have produced, or every energy bar in
    the repo -- all calibrated at lambda=1 -- silently changes meaning."""
    path = str(tmp_path / 'flow.pt')
    fitted_flow(path)
    n = 16
    x = in_box(n, seed=10)
    _, mixed = _legs(energy_fn(path, lam=0.05), x, n)
    _, free = _legs(energy_fn(None), x, n)
    assert torch.equal(mixed.physical_energy, free.physical_energy), \
        'the physical leg diverges between a mixing run and a lambda-free one'


if __name__ == '__main__':
    import tempfile
    from pathlib import Path
    n = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith('test_') or not callable(fn):
            continue
        if 'tmp_path' in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        n += 1
        print(f'  ok  {name}')
    print(f'\n{n} passed')
