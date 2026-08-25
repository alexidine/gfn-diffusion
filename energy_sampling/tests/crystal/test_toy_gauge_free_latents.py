"""
P1 (2026-08-24): toy routes must read latents WITHOUT the crystal gauge-fix.

latent_params() gauge-fixes the free centroid axes -- correct for crystals (pure
translation gauge, held dead in the SDE), destructive for toys (on P1 all THREE
centroid axes are 'free', so the multiharmonic toy's u,v,w MLE targets were
silently pinned to delta functions from 2026-08-11 onward; the model learned the
deltas, and the likelihood sharpness they induce is what made the toy's LR
boundary read 0.07x seed). The fix threads `gauge_fix_free_axes` from the one
place that knows the route (energy_function.is_crystal) through every live read:
buffers (x_fn), _batch_latents, init_prior_dataset, and the latents figure.

These tests pin the WIRING with recording stubs against the real methods; the
data-level fact (flag False preserves the stored centroid spread, std ~0.35 on
all 12 dims, bitwise round-trip) was verified directly against
toy_hard_uncond_multi_prior.pt on 2026-08-24.

`pytest tests/crystal/test_toy_gauge_free_latents.py -q`
"""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.fast


class RecordingBatch:
    """Records the gauge kwarg latent_params was called with."""

    def __init__(self):
        self.calls = []

    def latent_params(self, gauge_fix_free_axes=True):
        self.calls.append(bool(gauge_fix_free_axes))
        return 'latents'


def test_toy_latent_params_reads_gauge_free():
    # energy_sampling.buffer, NOT bare buffer: train.py imports the package
    # spelling, and the bare spelling is a SECOND module object whose function
    # fails an identity check (reference_dual_import_module_identity)
    from energy_sampling.buffer import toy_latent_params
    b = RecordingBatch()
    assert toy_latent_params(b) == 'latents'
    assert b.calls == [False]


def test_buffer_kwargs_routes_x_fn_by_is_crystal():
    """Every churned buffer is built through _buffer_kwargs, so this one seam
    decides whether stored rows keep their real u,v,w or get pinned."""
    from train import Modeller
    # energy_sampling.buffer, NOT bare buffer: train.py imports the package
    # spelling, and the bare spelling is a SECOND module object whose function
    # fails an identity check (reference_dual_import_module_identity)
    from energy_sampling.buffer import toy_latent_params

    def kwargs(is_crystal):
        fake = SimpleNamespace(args=SimpleNamespace(z_primes=[1]),
                               energy_function=SimpleNamespace(is_crystal=is_crystal))
        return Modeller._buffer_kwargs(fake)

    assert kwargs(True)['x_fn'] is None            # crystal: latent_params(), gauge-fixed
    assert kwargs(False)['x_fn'] is toy_latent_params  # toy: gauge-free


def test_batch_latents_passes_the_route_flag():
    from train import Modeller
    for is_crystal in (True, False):
        fake = SimpleNamespace(energy_function=SimpleNamespace(is_crystal=is_crystal))
        b = RecordingBatch()
        Modeller._batch_latents(fake, b)
        assert b.calls == [is_crystal]


def test_mxtaltools_latent_params_skips_the_gauge_fix_when_asked():
    """The real MolCrystalOps.latent_params body, on a recording stub: the flag
    must gate canonicalize_free_axes and ONLY canonicalize_free_axes."""
    from mxtaltools.dataset_utils.data_class_methods.crystal_ops import MolCrystalOps

    class Probe:
        def __init__(self):
            self.canonicalized = []

        def canonicalize_zp_aunits(self):
            self.canonicalized.append('zp')

        def canonicalize_free_axes(self):
            self.canonicalized.append('free')

        def full_cell_parameters(self):
            return 'cp'

        def latent_transform(self, cell_params):
            assert cell_params == 'cp'
            return SimpleNamespace(clip=lambda min, max: 'latents')

    p = Probe()
    assert MolCrystalOps.latent_params(p, gauge_fix_free_axes=False) == 'latents'
    assert p.canonicalized == ['zp'], 'free-axes gauge fix ran despite the flag'

    p = Probe()
    assert MolCrystalOps.latent_params(p) == 'latents'
    assert p.canonicalized == ['zp', 'free'], 'crystal default must still gauge-fix'


def test_is_crystal_energy_matches_the_instance_rule():
    from energies.molecular_crystal import TOY_ENERGY_FUNCTIONS, is_crystal_energy
    assert set(TOY_ENERGY_FUNCTIONS) == {'latent_harmonic', 'latent_multiharmonic'}
    assert not is_crystal_energy('latent_multiharmonic')
    assert is_crystal_energy('latent_gaussian')   # deliberately crystal-gated (dead-row toy)
    assert is_crystal_energy('elj')
