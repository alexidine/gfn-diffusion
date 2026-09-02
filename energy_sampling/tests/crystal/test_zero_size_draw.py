"""A zero-size buffer draw must return nothing, not crash on a negative dimension.

WHY THIS EXISTS. `_sample_indices` splits a prioritised draw as

    n_uniform  = max(1, int(batch_size * beta))
    n_weighted = batch_size - n_uniform

so `batch_size == 0` with any `beta > 0` asks numpy for -1 samples and dies:

    ValueError: negative dimensions are not allowed

raised inside np.random.choice, naming neither the buffer nor the caller. It
killed a run 50 steps into equilibration.

HOW A ZERO REQUEST ARISES. `reach_topup_size: 0` is the config spelling of
"never top up the prior buffer from anchors on the reach trigger". Two of the
three top_up_prior_from_anchors call sites gate on a positive size; the reach
site gated only on the trigger and passed the configured 0 straight through.

WHY IT SURFACED ONLY NOW, two independent reasons. The committed reach trigger
reads `prior_buffer.y` -- the raw elj sum -- against a composite `Emin(c)`, two
currencies hundreds of units apart, so it never fired. An uncommitted one-line
switch to `_prior_row_energy()` put both sides in the composite currency and
woke it up (measured: q90 excess 81.97 -> reach 0.180 against a 0.75 bar). And
no shipped config sets `reach_topup_size` to 0 -- mk_dev ships 1000 -- so the
zero-size path needed a live trigger AND a config asking for nothing.

NOT caused by moving `lj_coeff` onto the data: that left the composite
bit-identical, and this trigger reads the composite.

The guard is on the PRIMITIVE as well as the caller: any future caller that
legitimately asks for nothing must get nothing back.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from energy_sampling.buffer import CrystalBuffer


class _Sized:
    """Exercises _sample_indices' size arithmetic without a crystal batch.

    Only `len()` is consulted before the guard, so binding the unbound method is
    enough and keeps this test fast and independent of MolCrystalData.
    """

    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    _sample_indices = CrystalBuffer._sample_indices
    _target_row_pool = lambda self, cids: None          # noqa: E731

    def _draw_aligned(self, n, k, target_rows):
        """Stubbed: alignment is not what these tests cover.

        With no target_rows the real method is a plain uniform draw, and the
        thing under test is the SIZE arithmetic that splits the request into
        uniform and weighted halves -- not which rows come back.
        """
        return np.random.choice(n, size=k, replace=True)


@pytest.mark.parametrize('beta', [0.0, 0.1, 0.5, 1.0])
def test_zero_size_draw_returns_empty(beta):
    """The exact call that died: size 0 with a priority vector and a beta."""
    d = _Sized(32)
    p = np.full(32, 1.0 / 32)
    inds = d._sample_indices(0, p=p, beta=beta)
    assert isinstance(inds, np.ndarray)
    assert inds.size == 0, f'zero-size draw returned {inds.size} indices'


def test_negative_size_draw_returns_empty():
    """A negative request is the same bug one step further along, not a new one."""
    d = _Sized(32)
    assert d._sample_indices(-5, p=np.full(32, 1.0 / 32), beta=0.5).size == 0


def test_positive_draw_still_splits_uniform_and_weighted():
    """The guard must not swallow ordinary draws: 10 rows still come back."""
    d = _Sized(32)
    inds = d._sample_indices(10, p=np.full(32, 1.0 / 32), beta=0.5)
    assert inds.size == 10, f'expected 10 indices, got {inds.size}'
    assert inds.min() >= 0 and inds.max() < 32
