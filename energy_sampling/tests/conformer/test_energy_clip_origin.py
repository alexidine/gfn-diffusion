"""`energy_clip_origin` 'reference': the soft clip starts `energy_clip` above each member's own
reference-conformer potential, not at an absolute energy.

Raw force-field minima differ between molecules by hundreds of kcal/mol, so an absolute
cutoff engages at a different excess for each. Each claim is checked against the unclipped
potential of the same rows, and the set path against each member's own `potential_energy`.
Level `full`, force field `mmff`.
"""
import types

import numpy as np
import pytest
import torch

from conformer_modeller import ConformerModeller
from energies.conformer_torsions import ConformerTorsions
from test_multi_energy_vectorized import SIZES, SMIS, _Set

CLIP = 20.0          # small, so rows uniform in the box land on both sides of every cutoff


def _member(smi='CCCO', clip=CLIP):
    return ConformerTorsions(smiles=smi, level='full', device='cpu', force_field='mmff',
                             energy_clip=clip)


def _rows(en, n=256, seed=0):
    """Rows from near the reference (under the cutoff) out to a wide box draw (over it)."""
    rng = np.random.default_rng(seed)
    scale = np.geomspace(0.005, 0.6, n)[:, None]
    return torch.as_tensor(scale * rng.uniform(-1.0, 1.0, (n, en.ndim)), dtype=en.dtype)


def test_default_is_the_absolute_cutoff():
    en = _member()
    assert en.energy_clip_origin == 'absolute' and en.energy_clip_floor is None
    assert en.clip_cutoff == CLIP
    assert _member(clip=None).clip_cutoff is None


def test_floor_is_the_unclipped_potential_at_the_reference():
    en, raw = _member(), _member(clip=None)
    one = torch.tensor(1.0, dtype=en.dtype)
    want = float(raw.potential_energy(torch.zeros(1, raw.ndim, dtype=raw.dtype), one)[0])
    assert en.install_clip_floor() == pytest.approx(want, abs=1e-9)
    assert en.energy_clip == CLIP, 'reference_potential must restore the clip'
    assert en.energy_clip_origin == 'reference'
    assert en.clip_cutoff == pytest.approx(want + CLIP, abs=1e-9)


def test_member_clips_above_its_own_floor():
    en, raw = _member(), _member(clip=None)
    floor = en.install_clip_floor()
    one = torch.tensor(1.0, dtype=en.dtype)
    x = _rows(en)
    u = raw.potential_energy(x, one).double()        # inside the box: no wall
    e = en.potential_energy(x, one).double()
    cut = floor + CLIP
    above = u > cut
    assert bool(above.any()) and bool((~above).any()), 'rows must straddle the cutoff'
    assert torch.equal(e[~above], u[~above])
    assert torch.allclose(e[above], cut + torch.log1p(u[above] - cut), rtol=0, atol=1e-9)


def test_install_refuses_without_a_clip():
    with pytest.raises(ValueError, match='needs energy_clip'):
        _member(clip=None).install_clip_floor()


def test_set_reads_each_rows_own_floor():
    s = _Set(SMIS, SIZES, torch.float64, scale=0.6, energy_clip=CLIP)
    multi = s.multi
    absolute = multi.energy(s.X, s.batch, s.logT, return_exp=True)[1].conformer_energy.clone()
    floors = multi.install_clip_floor()
    assert multi.energy_clip_origin == 'reference'
    assert [m.energy_clip_floor for m in multi._members.values()] == pytest.approx(
        floors.tolist(), abs=1e-12)
    assert float(floors.max() - floors.min()) > 5.0, 'members must differ in floor'
    baked = multi.energy(s.X, s.batch, s.logT, return_exp=True)[1].conformer_energy.flatten()
    want = s.potential_oracle()                      # each member's own potential_energy
    assert torch.allclose(baked.double(), want, rtol=0, atol=1e-8)
    assert not torch.equal(baked, absolute.flatten()), 'the origin is INERT on this batch'
    # and the whole reward path, not only the baked field
    assert torch.allclose(multi.energy(s.X, s.batch, s.logT).double(), s.oracle().double(),
                          rtol=0, atol=1e-8)


def _stub(origin, en):
    cfg = types.SimpleNamespace(energy_clip=en.energy_clip)
    if origin is not None:
        cfg.energy_clip_origin = origin
    return types.SimpleNamespace(args=types.SimpleNamespace(energy_config=cfg),
                                 energy_function=en)


@pytest.mark.parametrize('origin', [None, 'absolute'])
def test_modeller_leaves_the_absolute_cutoff(origin):
    en = _member()
    ConformerModeller._install_clip_origin(_stub(origin, en))
    assert en.energy_clip_origin == 'absolute' and en.clip_cutoff == CLIP


def test_modeller_installs_the_reference_origin(capsys):
    en = _member()
    ConformerModeller._install_clip_origin(_stub('reference', en))
    assert en.energy_clip_origin == 'reference' and en.energy_clip_floor is not None
    assert 'above each member' in capsys.readouterr().out


def test_modeller_refuses_an_unknown_origin_and_a_missing_clip():
    with pytest.raises(SystemExit, match='energy_clip_origin'):
        ConformerModeller._install_clip_origin(_stub('floor', _member()))
    with pytest.raises(SystemExit, match='needs'):
        ConformerModeller._install_clip_origin(_stub('reference', _member(clip=None)))
