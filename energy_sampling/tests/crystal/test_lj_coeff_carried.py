"""`lj_coeff` rides on the data, is applied ONCE, and an unstamped row is fatal.

WHY THIS FILE EXISTS. The coefficient used to be applied by the consumer, at
`molecular_crystal.generator_energy`, while `.elj` held a RAW lattice sum. Every
other reader of `.elj` -- buffer `y`, the progress gate's energy marginal, the
prior-buffer admission/expiry/reach gates -- therefore saw an uncalibrated
number, and the gates compared it against `Emin(c)`, which is a composite total.
Measured on mipcas the two differ by ~217 energy units against a `ramp_floor` of
100, so those channels could never fire.

It now applies inside `compute_eLJ_energy`, from a per-graph attribute stamped
at `instantiate_crystals`. The failure modes that buys are all silent, so each
gets a test:

  SCALED       `.elj` must carry the coefficient, or the move accomplished
               nothing and every downstream reader is still uncalibrated.
  ONCE         the composite must scale LINEARLY in the coefficient. Reinstating
               the old multiply would square it -- 0.132 rather than 0.364 on
               mipcas -- and the result stays finite and plausible. This is the
               regression the arithmetic assertion below exists to catch.
  UNSTAMPED    a row STORED before the coefficient rode with the data has a raw
               `.elj` and no attribute, and `generator_energy` READS that value
               rather than recomputing it. Nothing in the number says which
               currency it is in, so absence must raise rather than default.
  MISMATCHED   two sources can both be stamped and DISAGREE (a prior calibrated
               at 0.3636 mixed with anchors built at 1.0). Presence alone would
               pass that; the value is checked too.

NOTE the sibling energy tests all use `energy_function='latent_gaussian'`, which
never reads `.elj` -- so they are structurally blind to every case here.
"""
import os
import sys

import pytest
import torch

CPU = torch.device('cpu')
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402
from mxtaltools.dataset_utils.data_classes import MolData  # noqa: E402
from energy_sampling.energies.molecular_crystal import MolecularCrystal  # noqa: E402

SG, N = 2, 2


def _synthetic_molecule(identifier, scale):
    """Same tiny rigid molecule the crystal-boundary test uses.

    Synthetic rather than loaded from the mini dataset: those entries carry
    string attrs that set_mol_attrs cannot collate on the analyze path
    ('too many dimensions str'), and nothing here needs a real molecule -- the
    assertions are about how a coefficient propagates, not about chemistry.
    """
    pos = torch.tensor([[-0.8, -0.2, 0.0],
                        [0.7, -0.1, 0.1],
                        [0.1, 0.8, -0.1]], dtype=torch.float32) * scale
    return MolData(
        z=torch.tensor([6, 7, 8], dtype=torch.long),
        pos=pos,
        x=torch.zeros((3, 1), dtype=torch.float32),
        identifier=identifier,
        mol_volume=torch.tensor(55.0 * scale ** 3),
        mass=torch.tensor(42.0),
        radius=torch.tensor(1.2 * scale),
        z_prime=torch.tensor(1, dtype=torch.long),
    )


def mol_batch(n=N):
    return collate_data_list([_synthetic_molecule('syn-%d' % i, 1.0 + 0.1 * i)
                              for i in range(n)])


def energy_fn(lj_coeff):
    return MolecularCrystal(
        device=CPU, energy_function='elj', space_groups=[SG], z_primes=(1,),
        temperature=1.0, lj_coeff=lj_coeff, bounding_coeff=1.0,
        reduction_coeff=1.0, density_coeff=1.0, reward_range=None,
        internal_oom_recovery=False, host_gas_phase_reference=False)


def score(coeff, seed=0):
    """Build, stamp, analyze, score -- the path every fresh sample takes."""
    ef = energy_fn(coeff)
    b = mol_batch()
    g = torch.Generator().manual_seed(seed)
    x = 0.4 * (2.0 * torch.rand(b.num_graphs, 12, generator=g) - 1.0)
    x[:, 3:6] = 0.0                      # keep the cell angles benign
    T = torch.ones(b.num_graphs)
    with torch.no_grad():
        energy, built = ef.analyze_crystal_batch(x, b, T, return_batch=True)
    return energy.double(), built.elj.double(), built


def test_elj_carries_the_coefficient():
    """SCALED: doubling the coefficient doubles the stored `.elj`."""
    _, elj_1, _ = score(1.0)
    _, elj_2, _ = score(2.0)
    assert torch.allclose(elj_2, 2.0 * elj_1, rtol=1e-5), (
        'stored .elj did not scale with lj_coeff -- downstream readers (buffer y, '
        'the energy marginal, the prior gates) are still seeing a raw sum')


def test_composite_scales_linearly_not_quadratically():
    """ONCE: the composite must be linear in the coefficient.

    Only the mol_energy term carries it, so the difference between two
    coefficients is exactly (k2 - k1) * elj_raw / z_prime. If the old
    `self.lj_coeff *` were reinstated alongside the new one the difference would
    pick up a k^2 term and this fails -- which is the whole point, because a
    doubly-scaled energy is finite and plausible.
    """
    e1, elj_1, built = score(1.0)          # elj_1 IS the raw sum, coefficient 1
    e2, _, _ = score(2.0)
    zp = built.z_prime.double().flatten()
    expected = (2.0 - 1.0) * (elj_1 / zp)
    assert torch.allclose(e2 - e1, expected, rtol=1e-5, atol=1e-6), (
        f'composite is not linear in lj_coeff: got {(e2 - e1).tolist()}, '
        f'expected {expected.tolist()} -- suspect a second multiply')


def test_unstamped_batch_raises():
    """UNSTAMPED: a stored row with no attribute must be fatal, never defaulted."""
    ef = energy_fn(0.3636)
    _, _, built = score(0.3636)
    del built.lj_coeff
    with pytest.raises(AttributeError, match='unknown currency'):
        ef.generator_energy(built, torch.ones(built.num_graphs))


def test_mismatched_coefficient_raises():
    """MISMATCHED: both stamped but disagreeing is its own error, not a migration."""
    ef = energy_fn(0.3636)
    _, _, built = score(0.3636)
    built.lj_coeff = torch.full_like(built.lj_coeff, 1.0)
    with pytest.raises(ValueError, match='two energy currencies'):
        ef.generator_energy(built, torch.ones(built.num_graphs))


def test_library_defaults_to_one_without_the_attribute():
    """The mxtaltools half stays usable by consumers that never stamp anything."""
    ef = energy_fn(1.0)
    b = mol_batch()
    g = torch.Generator().manual_seed(0)
    x = 0.4 * (2.0 * torch.rand(b.num_graphs, 12, generator=g) - 1.0)
    x[:, 3:6] = 0.0
    # replicate what analyze() does internally, so compute_eLJ_energy is called
    # on a CLUSTER with edges -- the same object the library builds for itself
    cb = ef.instantiate_crystals(x, b)
    cluster = cb.mol2cluster(10, 10, std_orientation=False)
    cluster.construct_radial_graph(cutoff=10)
    cluster._init_computes(override=True)
    assert not hasattr(cluster, 'lj_coeff')
    unstamped = cluster.compute_eLJ_energy()
    cluster.lj_coeff = torch.full((cluster.num_graphs,), 3.0)
    stamped = cluster.compute_eLJ_energy()
    assert torch.allclose(stamped.double(), 3.0 * unstamped.double(), rtol=1e-5), (
        'absent lj_coeff must behave as 1.0 and a present one must scale')
