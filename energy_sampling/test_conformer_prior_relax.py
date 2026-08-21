"""`prior_relax_steps` must be a no-op when off and CALIBRATED when on.

The prior draws each torsion independently, so a long chain self-intersects and the LJ term
runs away (91% of Gly6's median energy). A few Rprop steps in state space repair it. These
gates exist because both halves can fail silently: an "off" path that quietly relaxes would
change every existing run, and an "on" path that does nothing would look like success while
the peptide prior stayed clashed.
"""
import types
import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings('ignore')
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from conformer_modeller import ConformerModeller, _NON_ENERGY_KEYS

GLY4 = 'NCC(=O)NCC(=O)NCC(=O)NCC(=O)O'
PHENYL_THP = 'C1CCC(CO1)c1ccccc1'


def _stub(smiles, steps):
    """Minimum surface `_draw_prior_states` touches: energy fn, prior, and the knob."""
    from energies.conformer_torsions import ConformerTorsions

    torch.set_default_dtype(torch.float32)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = types.SimpleNamespace()
    m.energy_function = ConformerTorsions(smiles=smiles, level='full', device=dev,
                                          force_field='mmff')
    m.internal_prior = torch.load('conformer_prior_v2.pt', weights_only=False)
    cfg = types.SimpleNamespace()
    if steps is not None:
        cfg.prior_relax_steps = steps
    m.args = types.SimpleNamespace(energy_config=cfg)
    return m


def _teff(en, x):
    from energies.prior_baselines import descend

    one = torch.tensor(1.0, device=en.device)
    e = en.potential_energy(x, one).double()
    e = e[torch.isfinite(e)]
    seeds = x[torch.argsort(en.potential_energy(x, one).double())[:128]]
    floor = float(descend(en, seeds, 150)[1].detach().double().cpu().min())
    return 1.0 + 2.0 * (float(e.median()) - floor) / en.ndim


@pytest.mark.parametrize('steps', [None, 0])
def test_off_is_bitwise_identical_to_the_raw_draw(steps):
    """Absent OR zero must not touch the states -- not 'approximately', bitwise."""
    m = _stub(GLY4, steps)
    raw, _ = m.energy_function.sample_prior_states(
        m.internal_prior, 256, np.random.default_rng(0), report=False)
    got, _ = ConformerModeller._draw_prior_states(m, 256, np.random.default_rng(0))
    raw = torch.as_tensor(raw, dtype=m.energy_function.dtype, device=m.energy_function.device)
    got = torch.as_tensor(got, dtype=m.energy_function.dtype, device=m.energy_function.device)
    assert torch.equal(raw, got)


def test_on_actually_repairs_the_clash():
    """RE-INTRODUCING the problem must FAIL this: an inert knob has to be caught."""
    m = _stub(GLY4, 10)
    en = m.energy_function
    raw, _ = en.sample_prior_states(m.internal_prior, 2048, np.random.default_rng(1),
                                    report=False)
    raw = torch.as_tensor(raw, dtype=en.dtype, device=en.device)
    relaxed, _ = ConformerModeller._draw_prior_states(m, 2048, np.random.default_rng(1))
    relaxed = torch.as_tensor(relaxed, dtype=en.dtype, device=en.device)

    assert not torch.equal(raw, relaxed), 'knob is INERT -- states came back untouched'
    one = torch.tensor(1.0, device=en.device)
    assert (en.potential_energy(relaxed, one).median()
            < en.potential_energy(raw, one).median())
    # calibrated: T_eff/T = 2.0 is a correctly THERMAL sample (equipartition, median d/2)
    assert 1.6 <= _teff(en, relaxed) <= 2.6, 'relaxed prior is not at thermal calibration'


def test_thermal_molecule_is_left_alone_by_default():
    """phenyl-THP's raw draw already reads 2.05; the default must not over-cool it."""
    m = _stub(PHENYL_THP, None)
    x, _ = ConformerModeller._draw_prior_states(m, 2048, np.random.default_rng(2))
    x = torch.as_tensor(x, dtype=m.energy_function.dtype, device=m.energy_function.device)
    assert 1.7 <= _teff(m.energy_function, x) <= 2.4


def test_knob_is_consumed_not_forwarded_to_the_energy_function():
    """ConformerTorsions takes no **kwargs, so a stray key raises at construction."""
    assert 'prior_relax_steps' in _NON_ENERGY_KEYS
