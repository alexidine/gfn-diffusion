"""
CPU tests for the kNN prior energy (energies/prior_knn.py).

WHAT THIS FILE IS FOR. The energy is a null-test instrument: it exists so that
VarGrad can be run against the prior's OWN implied energy, where the correct
answer is known in advance. An instrument whose failures are silent is worse
than no instrument, and the two ways this one fails silently are both geometric:

  1. the minimum-image metric is dropped, so points across a wrap boundary read
     as maximally far apart instead of adjacent, and
  2. the wrap mask or dead rows describe a different problem than the policy,
     so every distance is uniformly wrong.

Neither shows up as an error, a NaN, or an obviously bad number -- both just
produce a plausible energy landscape that is not the prior's. So every
assertion below is paired with a NEGATIVE CONTROL: a variant of the same setup
that must come out DIFFERENT. A test that only checks the wrapped case passes
just as happily when wrapping is deleted.

    python test_prior_knn.py
"""
import math
import os
import sys

import torch

CPU = torch.device('cpu')

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
for p in (_here, os.path.dirname(_here)):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energies.prior_knn import PriorKNN, reference_digest, LATENT_PERIOD  # noqa: E402


def _ref_blob(reference, wrap_mask, dead_rows=(), k=8):
    return {'reference': reference,
            'wrap_mask': list(wrap_mask),
            'dead_rows': tuple(dead_rows),
            'k': int(k),
            'period': LATENT_PERIOD,
            'min_radius': 1e-4,
            'provenance': {'source': 'unit test'},
            'sha256': reference_digest(reference)}


# --------------------------------------------------------------------------
# 1. the minimum-image metric


def test_wrapped_dim_wraps():
    """A query across the +-1 seam must read as NEAR, and must not without wrapping."""
    g = torch.Generator().manual_seed(0)
    ref = torch.zeros(400, 3)
    ref[:, 0] = 0.95 + 0.001 * torch.randn(400, generator=g)
    ref[:, 1:] = 0.001 * torch.randn(400, 2, generator=g)

    query = torch.tensor([[-0.95, 0.0, 0.0]])

    wrapped = PriorKNN(ref, wrap_mask=[True, False, False], k=8)
    plain = PriorKNN(ref, wrap_mask=[False, False, False], k=8)

    e_wrapped = float(wrapped.energy(query))
    e_plain = float(plain.energy(query))

    # true separations: 0.1 across the seam, 1.9 the long way round
    assert math.isclose(e_wrapped, 3 * math.log(0.1), abs_tol=0.15), e_wrapped
    assert math.isclose(e_plain, 3 * math.log(1.9), abs_tol=0.15), e_plain
    # the negative control: dropping the wrap must change the answer, a lot
    assert e_plain - e_wrapped > 5.0, (e_plain, e_wrapped)


def test_unwrapped_dim_does_not_wrap():
    """Guards the inverse error -- wrapping every dim instead of the masked ones."""
    ref = torch.zeros(400, 3)
    ref[:, 1] = 0.95                      # dim 1 is NOT in the mask
    query = torch.tensor([[0.0, -0.95, 0.0]])

    knn = PriorKNN(ref, wrap_mask=[True, False, False], k=8)
    e = float(knn.energy(query))
    assert math.isclose(e, 3 * math.log(1.9), abs_tol=0.15), e
    # and the same displacement on the WRAPPED dim reads as near
    knn2 = PriorKNN(ref[:, [1, 0, 2]].contiguous(), wrap_mask=[True, False, False], k=8)
    e2 = float(knn2.energy(torch.tensor([[-0.95, 0.0, 0.0]])))
    assert e2 < e - 5.0, (e2, e)


def test_wrap_is_minimum_image_not_modulo():
    """Displacement must fold to [-period/2, period/2], not to [0, period)."""
    ref = torch.zeros(200, 2)
    ref[:, 0] = 0.6
    knn = PriorKNN(ref, wrap_mask=[True, False], k=4)
    # 0.6 -> 0.4 is a displacement of 0.2 either signed or folded; 0.6 -> -0.9 is
    # 1.5 raw, which minimum-image folds to 0.5. A modulo-into-[0,2) fold would
    # leave it at 1.5 and invert the ordering of these two queries.
    near = float(knn.energy(torch.tensor([[0.4, 0.0]])))
    far = float(knn.energy(torch.tensor([[-0.9, 0.0]])))
    assert math.isclose(near, 2 * math.log(0.2), abs_tol=0.05), near
    assert math.isclose(far, 2 * math.log(0.5), abs_tol=0.05), far
    assert near < far


# --------------------------------------------------------------------------
# 2. it is actually a density


def test_energy_is_lower_in_dense_regions():
    g = torch.Generator().manual_seed(1)
    dense = torch.tensor([[-0.5, 0.0]]) + 0.01 * torch.randn(300, 2, generator=g)
    sparse = torch.tensor([[0.5, 0.0]]) + 0.30 * torch.randn(300, 2, generator=g)
    ref = torch.cat([dense, sparse], dim=0)

    knn = PriorKNN(ref, wrap_mask=[False, False], k=10)
    e_dense = float(knn.energy(torch.tensor([[-0.5, 0.0]])))
    e_sparse = float(knn.energy(torch.tensor([[0.5, 0.0]])))
    assert e_dense < e_sparse - 2.0, (e_dense, e_sparse)


def test_uniform_reference_is_flatter_than_clustered():
    """A uniform prior should give a near-flat energy; a clustered one should not.

    Comparative rather than absolute: an absolute bound on the std would be a
    number tuned until it passed, and would keep passing if the estimator broke
    in a way that made everything constant.
    """
    g = torch.Generator().manual_seed(2)
    n = 8000
    uniform = 2.0 * torch.rand(n, 2, generator=g) - 1.0
    clustered = torch.cat([0.05 * torch.randn(n // 2, 2, generator=g) - 0.5,
                           0.05 * torch.randn(n // 2, 2, generator=g) + 0.5], dim=0)
    query = 2.0 * torch.rand(500, 2, generator=g) - 1.0

    mask = [True, True]
    e_uniform = PriorKNN(uniform, wrap_mask=mask, k=20).energy(query)
    e_clustered = PriorKNN(clustered, wrap_mask=mask, k=20).energy(query)
    assert float(e_uniform.std()) < 0.5 * float(e_clustered.std()), \
        (float(e_uniform.std()), float(e_clustered.std()))


def test_energy_grows_with_distance_off_support():
    """Tails must rise monotonically -- the confinement, weak as it is, must exist."""
    ref = 0.02 * torch.randn(500, 2, generator=torch.Generator().manual_seed(3))
    knn = PriorKNN(ref, wrap_mask=[False, False], k=10)
    radii = torch.tensor([0.05, 0.1, 0.2, 0.4, 0.8])
    e = knn.energy(torch.stack([radii, torch.zeros_like(radii)], dim=1))
    assert bool((e[1:] > e[:-1]).all()), e.tolist()


# --------------------------------------------------------------------------
# 3. implementation invariants


def test_chunking_does_not_change_the_answer():
    """The running top-k merge across reference chunks must be exact."""
    g = torch.Generator().manual_seed(4)
    ref = 2.0 * torch.rand(1000, 4, generator=g) - 1.0
    query = 2.0 * torch.rand(37, 4, generator=g) - 1.0
    mask = [True, False, True, False]

    full = PriorKNN(ref, wrap_mask=mask, k=12)
    full.max_pair_elems = 10 ** 12          # one chunk
    baseline = full.energy(query)

    for elems in (2000, 500, 111, 13):
        chunked = PriorKNN(ref, wrap_mask=mask, k=12)
        chunked.max_pair_elems = elems
        assert torch.allclose(chunked.energy(query), baseline, atol=1e-5), elems


def test_dead_rows_are_excluded():
    g = torch.Generator().manual_seed(5)
    ref = 2.0 * torch.rand(600, 4, generator=g) - 1.0
    ref[:, 2] = 0.0                                   # dim 2 pinned
    knn = PriorKNN(ref, wrap_mask=[False] * 4, dead_rows=(2,), k=8)
    assert knn.d_live == 3

    query = torch.tensor([[0.1, 0.2, 0.0, 0.3]])
    moved_dead = torch.tensor([[0.1, 0.2, 0.7, 0.3]])   # dead row perturbed
    moved_live = torch.tensor([[0.1, 0.2, 0.0, 0.9]])   # live row perturbed

    base = float(knn.energy(query))
    assert math.isclose(float(knn.energy(moved_dead)), base, abs_tol=1e-6)
    # negative control: the live perturbation must matter, or the test above is
    # passing because nothing matters
    assert abs(float(knn.energy(moved_live)) - base) > 1e-3


def test_coincident_query_is_finite():
    ref = 2.0 * torch.rand(200, 3, generator=torch.Generator().manual_seed(6)) - 1.0
    knn = PriorKNN(ref, wrap_mask=[False] * 3, k=1, min_radius=1e-4)
    e = knn.energy(ref[:5])
    assert bool(torch.isfinite(e).all()), e
    assert float(e.min()) >= 3 * math.log(1e-4) - 1e-6


def test_batch_rows_are_independent():
    g = torch.Generator().manual_seed(7)
    ref = 2.0 * torch.rand(400, 3, generator=g) - 1.0
    query = 2.0 * torch.rand(16, 3, generator=g) - 1.0
    knn = PriorKNN(ref, wrap_mask=[True, False, False], k=6)
    batched = knn.energy(query)
    singly = torch.cat([knn.energy(query[i:i + 1]) for i in range(query.shape[0])])
    assert torch.allclose(batched, singly, atol=1e-6)


# --------------------------------------------------------------------------
# 4. loud failure on a mismatched or tampered reference


def test_digest_mismatch_raises(tmp_path):
    ref = torch.randn(300, 3)
    blob = _ref_blob(ref, [True, False, False])
    blob['reference'] = ref + 0.01          # coordinates changed after digesting
    path = str(tmp_path / 'ref.pt')
    torch.save(blob, path)
    try:
        PriorKNN.load(path)
    except ValueError as exc:
        assert 'digest mismatch' in str(exc)
    else:
        raise AssertionError('a tampered reference loaded without complaint')


def test_load_roundtrip_and_k_override(tmp_path):
    ref = 2.0 * torch.rand(500, 3, generator=torch.Generator().manual_seed(8)) - 1.0
    path = str(tmp_path / 'ref.pt')
    torch.save(_ref_blob(ref, [True, False, True], dead_rows=(), k=8), path)

    knn = PriorKNN.load(path)
    assert knn.k == 8 and knn.n_reference == 500
    assert sorted(torch.nonzero(knn.wrap_mask).flatten().tolist()) == [0, 2]

    wider = PriorKNN.load(path, k=32)
    assert wider.k == 32
    # a larger k is a strictly larger radius, so a strictly higher energy
    query = 2.0 * torch.rand(20, 3, generator=torch.Generator().manual_seed(9)) - 1.0
    assert bool((wider.energy(query) >= knn.energy(query) - 1e-6).all())


def test_missing_key_raises(tmp_path):
    ref = torch.randn(200, 2)
    blob = _ref_blob(ref, [True, False])
    del blob['wrap_mask']
    path = str(tmp_path / 'ref.pt')
    torch.save(blob, path)
    try:
        PriorKNN.load(path)
    except ValueError as exc:
        assert 'wrap_mask' in str(exc)
    else:
        raise AssertionError('a reference file with no wrap mask loaded anyway')


def test_verify_against_policy():
    ref = torch.randn(300, 4)
    knn = PriorKNN(ref, wrap_mask=[False, False, True, True], dead_rows=(1,), k=8)

    knn.verify_against_policy([False, False, True, True], dead_rows=(1,))   # matches

    for bad_mask, bad_dead, needle in (
            ([False, True, True, True], (1,), 'wrapped'),
            ([False, False, True, True], (0,), 'dead rows'),
            ([False, False, True, True], (), 'dead rows'),
    ):
        try:
            knn.verify_against_policy(bad_mask, dead_rows=bad_dead)
        except ValueError as exc:
            assert needle in str(exc), (needle, str(exc))
        else:
            raise AssertionError(f'mismatch accepted: {bad_mask} {bad_dead}')


def test_construction_rejects_incoherent_geometry():
    ref = torch.randn(100, 3)
    for kwargs, needle in (
            (dict(wrap_mask=[True, False], k=4), 'wrap_mask has'),
            (dict(wrap_mask=[True, False, False], k=100), 'more than k points'),
            (dict(wrap_mask=[True, False, False], k=0), 'k must be'),
            (dict(wrap_mask=[True, False, False], dead_rows=(7,), k=4), 'dead_rows must index'),
            (dict(wrap_mask=[True, False, False], dead_rows=(0, 1, 2), k=4), 'every dim is dead'),
    ):
        try:
            PriorKNN(ref, **kwargs)
        except ValueError as exc:
            assert needle in str(exc), (needle, str(exc))
        else:
            raise AssertionError(f'accepted incoherent geometry: {kwargs}')

    try:
        PriorKNN(torch.randn(10, 3, 2), wrap_mask=[True] * 3, k=2)
    except ValueError as exc:
        assert 'must be [N, D]' in str(exc)
    else:
        raise AssertionError('accepted a 3-D reference')


def test_energy_rejects_wrong_width():
    knn = PriorKNN(torch.randn(200, 3), wrap_mask=[False] * 3, k=4)
    try:
        knn.energy(torch.randn(5, 4))
    except ValueError as exc:
        assert 'expected latents' in str(exc)
    else:
        raise AssertionError('scored a batch of the wrong width')


# --------------------------------------------------------------------------
# 5. calibration -- whether the energy MEANS what the null test needs
#
# Everything above checks that the estimator is built correctly. These two check
# something else: that exp(-E) is actually proportional to the density, which is
# the only property that makes a lambda=0 null result interpretable. It holds at
# low dimension and FAILS at the shipped d=12, so the second test pins a known
# limitation rather than a bug. If it ever starts passing, that is a real change
# and the docstring in prior_knn.py needs rewriting -- which is why it asserts
# the failure instead of being deleted or skipped.


def test_calibration_holds_at_low_dimension():
    from energies.density_calibration import calibrate, wrapped_gaussian_draw

    g = torch.Generator().manual_seed(11)
    mask = [True, True, False]
    ref, _ = wrapped_gaussian_draw(20000, 3, mask, 0.45, generator=g)
    query, log_p = wrapped_gaussian_draw(3000, 3, mask, 0.45, generator=g)

    cal = calibrate(PriorKNN(ref, wrap_mask=mask, k=32).energy, query, log_p)
    assert cal.passes, str(cal)


def test_calibration_fails_at_the_shipped_dimension():
    """d=12 is where the kNN ball stops being local. KNOWN LIMITATION, pinned."""
    from energies.density_calibration import calibrate, wrapped_gaussian_draw

    g = torch.Generator().manual_seed(12)
    mask = [i in (7, 8, 10, 11) for i in range(12)]
    ref, _ = wrapped_gaussian_draw(20000, 12, mask, 0.45, generator=g)
    query, log_p = wrapped_gaussian_draw(3000, 12, mask, 0.45, generator=g)

    cal = calibrate(PriorKNN(ref, wrap_mask=mask, k=32).energy, query, log_p)
    assert not cal.passes, (
        'the d=12 slope now sits within tolerance -- if this is real, prior_knn.py '
        f'is usable as a null-test target and its docstring is stale. {cal}')
    assert cal.slope < 0.9, str(cal)
    # the trap this gate exists to defeat: the fit LOOKS excellent
    assert cal.corr > 0.95, str(cal)


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
