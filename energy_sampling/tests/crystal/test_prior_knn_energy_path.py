"""
The `latent_knn` energy driven through the REAL MolecularCrystal path.

Companion to test_prior_knn.py, which tests the density in isolation. Nothing
there touches instantiate_crystals, latent_to_cell_params, analyze(), or the
generator_energy assembly, and that is where the config-shaped failures live:
a jacobian correction that should not be there, a reduction penalty
contaminating the target, `computes` asking mxtaltools for an attribute that
does not exist, or the dispatch scoring raw policy output instead of
gauge-fixed latents.

THE LOAD-BEARING ASSERTION is that the total energy equals the kNN term EXACTLY
on in-box samples. Every contaminant this file is meant to catch is additive, so
exact equality rules all of them out at once -- and the out-of-box test below
keeps that equality from being vacuous by showing the bounding term is still
live and still capable of moving the total.

    python test_prior_knn_energy_path.py
"""
import math
import os
import sys

import torch

CPU = torch.device('cpu')

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402
from energy_sampling.energies.molecular_crystal import MolecularCrystal  # noqa: E402
from energy_sampling.energies.prior_knn import PriorKNN, reference_digest, LATENT_PERIOD  # noqa: E402
from energy_sampling.models.dead_latent_rows import resolve_dead_rows  # noqa: E402
from build_prior_knn_reference import build_wrap_mask  # noqa: E402

DATASET = os.path.abspath(os.path.join(_here, '..', '..', 'mxtaltools',
                                       'mini_datasets', 'mini_new_csd.pt'))
SG = 2
DIM = 12
T = 1.0

_MOL = None


def mol_batch(n):
    global _MOL
    if _MOL is None:
        data = torch.load(DATASET, weights_only=False)
        cands = [e for e in data if int(e.z_prime) == 1 and bool(e.is_well_defined)
                 and not bool(e.cocrystal)]
        cands.sort(key=lambda e: int(e.num_nodes))
        _MOL = cands[0]
    b = collate_data_list([_MOL.clone() for _ in range(n)])
    b.reset_sg_info(SG)
    return b


def dead():
    return tuple(resolve_dead_rows(SG, is_crystal=True, max_z_prime=1))


def wrap_mask():
    # periodic_centroids off, matching the energy_fn below: rotational phi/r only
    return build_wrap_mask(1, ())


def write_reference(path, n=600, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    ref = 1.2 * (2.0 * torch.rand(n, DIM, generator=g) - 1.0)
    ref = ref.clamp(-0.98, 0.98)
    for r in dead():
        ref[:, r] = 0.0
    blob = {'reference': ref, 'wrap_mask': wrap_mask(), 'dead_rows': dead(),
            'k': k, 'period': LATENT_PERIOD, 'min_radius': 1e-4,
            'provenance': {'source': 'integration test'},
            'sha256': reference_digest(ref)}
    torch.save(blob, path)
    return ref


def energy_fn(path, bounding_coeff=1.0):
    return MolecularCrystal(
        device=CPU, energy_function='latent_knn',
        space_groups=[SG], z_primes=(1,),
        temperature=T, bounding_coeff=bounding_coeff, reduction_coeff=1.0,
        reward_range=None, internal_oom_recovery=False,
        host_gas_phase_reference=False,
        prior_knn_path=path)


def roundtripped_latents(ef, x, batch):
    """The latents generator_energy actually scores, mirroring its own call."""
    cb = ef.instantiate_crystals(x, batch)
    return cb.latent_params(gauge_fix_free_axes=ef.is_crystal)


# --------------------------------------------------------------------------


def test_total_energy_is_exactly_the_knn_term(tmp_path):
    path = str(tmp_path / 'ref.pt')
    write_reference(path)
    ef = energy_fn(path)

    assert ef.is_crystal is True
    assert ef.latent_energy is True
    assert 'latent_knn' not in ef.computes          # nothing for analyze() to build

    g = torch.Generator().manual_seed(1)
    n = 24
    x = 0.9 * (2.0 * torch.rand(n, DIM, generator=g) - 1.0)   # strictly in-box
    batch = mol_batch(n)

    total, _ = ef.analyze_crystal_batch(x, batch, temperature=torch.full((n,), T))
    expect = ef.prior_knn.energy(roundtripped_latents(ef, x, mol_batch(n)))

    assert torch.allclose(total, expect, atol=1e-4), \
        (total[:4].tolist(), expect[:4].tolist())


def test_out_of_box_adds_bounding_energy(tmp_path):
    """Negative control for the equality above: the bounding term is still live."""
    path = str(tmp_path / 'ref.pt')
    write_reference(path)
    ef = energy_fn(path, bounding_coeff=5.0)

    n = 8
    x = torch.zeros(n, DIM)
    x[:, 0] = 1.5                                     # 0.5 outside the box
    batch = mol_batch(n)

    total, _ = ef.analyze_crystal_batch(x, batch, temperature=torch.full((n,), T))
    knn_only = ef.prior_knn.energy(roundtripped_latents(ef, x, mol_batch(n)))

    excess = (total - knn_only)
    assert bool((excess > 1.0).all()), excess.tolist()
    # quadratic wall, coeff 5, violation 0.5 -> 5 * 0.25
    assert math.isclose(float(excess.mean()), 5.0 * 0.25, rel_tol=0.05), float(excess.mean())


def test_energy_varies_with_the_state(tmp_path):
    """A dispatch that silently returned a constant would satisfy nothing above."""
    path = str(tmp_path / 'ref.pt')
    write_reference(path)
    ef = energy_fn(path)

    g = torch.Generator().manual_seed(2)
    n = 32
    x = 0.9 * (2.0 * torch.rand(n, DIM, generator=g) - 1.0)
    total, _ = ef.analyze_crystal_batch(x, mol_batch(n), temperature=torch.full((n,), T))
    assert float(total.std()) > 0.1, float(total.std())
    assert bool(torch.isfinite(total).all())


def test_log_reward_is_negative_energy_over_temperature(tmp_path):
    path = str(tmp_path / 'ref.pt')
    write_reference(path)
    ef = energy_fn(path)

    n = 8
    g = torch.Generator().manual_seed(3)
    x = 0.9 * (2.0 * torch.rand(n, DIM, generator=g) - 1.0)
    log_t = torch.full((n,), math.log10(2.0))

    log_r = ef.log_reward(x, mol_batch(n), log_t)
    raw, _ = ef.analyze_crystal_batch(x, mol_batch(n), temperature=torch.full((n,), 2.0))
    assert torch.allclose(log_r, -raw / 2.0, atol=1e-4)


def test_missing_reference_path_raises():
    try:
        energy_fn(None)
    except ValueError as exc:
        assert 'prior_knn_path' in str(exc)
    else:
        raise AssertionError('latent_knn constructed with no reference at all')


def test_reference_of_the_wrong_width_raises(tmp_path):
    path = str(tmp_path / 'ref.pt')
    ref = torch.randn(400, 18)                      # a max_z_prime=2 layout
    blob = {'reference': ref, 'wrap_mask': build_wrap_mask(2, ()), 'dead_rows': (),
            'k': 8, 'period': LATENT_PERIOD, 'min_radius': 1e-4,
            'provenance': {}, 'sha256': reference_digest(ref)}
    torch.save(blob, path)
    try:
        energy_fn(path)
    except ValueError as exc:
        assert 'dimensional' in str(exc)
    else:
        raise AssertionError('an 18-dim reference was accepted for a 12-dim problem')


def test_a_mistyped_energy_config_key_is_fatal(tmp_path):
    """The whole design leans on energy_config: being the LOUD surface.

    train.py::init_energy_function does energy_config.update(args.energy_config
    .__dict__) then MolecularCrystal(**energy_config), and the constructor takes
    no **kwargs -- so a typo raises at load rather than training for a week
    against a default nobody chose. Every other placement in the YAML swallows an
    unknown key silently, which is why prior_knn_path lives here and not, say,
    at top level or under a *_loss_coeffs block.
    """
    path = str(tmp_path / 'ref.pt')
    write_reference(path)

    base = dict(device=CPU, energy_function='latent_knn', space_groups=[SG],
                z_primes=(1,), temperature=T, reward_range=None,
                internal_oom_recovery=False, host_gas_phase_reference=False)

    MolecularCrystal(**base, prior_knn_path=path)          # the spelling that works

    try:
        MolecularCrystal(**base, prior_knn_pathh=path)      # one keystroke off
    except TypeError as exc:
        assert 'prior_knn_pathh' in str(exc), str(exc)
    else:
        raise AssertionError(
            'a mistyped energy_config key was accepted; the constructor has grown '
            '**kwargs and every prior_knn_* knob can now be silently ignored')


def test_verify_against_a_real_policy(tmp_path):
    """The startup guard in train.py's init_gfn, exercised against a real GFN."""
    from energy_sampling.models.gfn import GFN

    path = str(tmp_path / 'ref.pt')
    write_reference(path)
    ef = energy_fn(path)

    def build(periodic_centroid_axes):
        torch.manual_seed(0)
        return GFN(dim=DIM, s_emb_dim=64, conditions_dim=0, harmonics_dim=16,
                   t_dim=16, t_hidden_dim=32, s_hidden_dim=32, s_layers=2,
                   policy_hidden_dim=32, policy_layers=2, flow_hidden_dim=32,
                   flow_layers=2, conditional=False, learn_pb=True,
                   learned_variance=True, t_scale=0.05, log_var_range=6.0,
                   pb_var_range=6.0, clipping=True, gfn_clip=200.0, device=CPU,
                   max_z_prime=1, do_periodic_angles=True,
                   periodic_centroids=bool(periodic_centroid_axes),
                   periodic_centroid_axes=periodic_centroid_axes,
                   hold_dead_latent_rows=True, dead_latent_rows=dead(),
                   dplr_rank=0, pb_exact_reversal=True)

    matching = build(None)
    ef.prior_knn.verify_against_policy(matching.ang_mask, dead_rows=matching.dead_rows)

    # SG2 has full-width aunit axes on b and c, so turning periodic_centroids on
    # wraps dims the reference was not built with. That must be fatal, not silent.
    mismatched = build((1, 2))
    try:
        ef.prior_knn.verify_against_policy(mismatched.ang_mask,
                                           dead_rows=mismatched.dead_rows)
    except ValueError as exc:
        assert 'wrapped' in str(exc), str(exc)
    else:
        raise AssertionError('a policy wrapping different dims was accepted')


if __name__ == '__main__':
    import tempfile
    from pathlib import Path

    passed = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith('test_') or not callable(fn):
            continue
        if 'tmp_path' in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        passed += 1
        print(f'  ok  {name}')
    print(f'\n{passed} passed')
