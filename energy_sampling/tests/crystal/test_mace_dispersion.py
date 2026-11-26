"""cfg:energy_config.mlip_dispersion -- the MACE route's D3 (MXtalTools load_mace_model(dispersion=...), torch-dftd).

What the trainer consumes: the predictor MolecularCrystal loads carries the dispersion, and MXtalTools adds it to every
MACE energy, crystal and isolated legs alike (AL_mace_utils.compute_crystal_mace_on_mxt_batch). Pinned here: the key is
read on the mace route only; it enters the problem identity only when set (an unset key keeps every existing hash) and
the prior's scoring identity always; and, on the dev box's real files, the trainer's predictor reproduces the energy the
MXtalTools search recorded under MACE-CSP-25.2 + D3(BJ, PBE) for the acridine known forms (acr_mlip_nov19).
"""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import noised_prior
import utils
from energies.molecular_crystal import MolecularCrystal, mlip_dispersion_settings

HERE = Path(__file__).resolve().parents[2]
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'
D3 = {'xc': 'pbe', 'damping': 'bj'}
MACE_CSP = 'D:/crystal_datasets/day_models/mace_csp_25.2_large.model'
KNOWN = 'D:/crystal_datasets/acridine/campaigns/acr_mlip_nov19/macecspl_known_minima.pt'


def _args(capsys):
    args = utils.get_train_args(['--config', str(CANONICAL)])
    capsys.readouterr()
    return args


def test_dispersion_is_refused_off_the_mace_route():
    with pytest.raises(ValueError, match='mace route only'):
        MolecularCrystal(device='cpu', energy_function='elj', mlip_dispersion=D3)


def test_settings_are_a_sorted_plain_dict():
    assert mlip_dispersion_settings(None) is None
    assert list(mlip_dispersion_settings(SimpleNamespace(xc='pbe', damping='bj'))) == ['damping', 'xc']


def test_problem_identity_carries_dispersion_only_when_set(capsys):
    args = _args(capsys)
    assert args.energy_config.mlip_dispersion is None
    unset = utils.get_problem_definition(args)
    assert 'mlip_dispersion' not in unset['energy_config']
    args.energy_config.mlip_dispersion = SimpleNamespace(**D3)
    setd = utils.get_problem_definition(args)
    assert setd['energy_config']['mlip_dispersion'] == D3
    assert utils.problem_hash(setd) != utils.problem_hash(unset)


def test_prior_identity_carries_dispersion(capsys):
    args = _args(capsys)
    ef = SimpleNamespace(analyze_kwargs={}, lj_coeff=None)
    off = noised_prior.identity(args, ef)
    assert off['mlip_dispersion'] is None
    args.energy_config.mlip_dispersion = SimpleNamespace(**D3)
    on = noised_prior.identity(args, ef)
    assert on['mlip_dispersion'] == {'damping': 'bj', 'xc': 'pbe'}
    assert noised_prior.identity_mismatch(off, on) == [f"mlip_dispersion: file None, run {on['mlip_dispersion']!r}"]
    # a file written before the key existed reads as no dispersion
    old = {k: v for k, v in off.items() if k != 'mlip_dispersion'}
    assert noised_prior.identity_mismatch(old, off) == []


@pytest.mark.skipif(not (os.path.exists(MACE_CSP) and os.path.exists(KNOWN)), reason='dev-box MACE-CSP files')
def test_trainer_predictor_reproduces_the_search_energy():
    pytest.importorskip('torch_dftd')
    from mxtaltools.crystal_search import minima as mn
    rows = torch.load(KNOWN, weights_only=False)
    reps = [dict(x=torch.cat([r.cell_lengths[0], r.cell_angles[0], r.aunit_centroid[0], r.aunit_orientation[0]]),
                 hand=r.aunit_handedness.reshape(-1)) for r in rows]
    want = torch.tensor([float(r.mace) for r in rows], dtype=torch.float64)
    got = {}
    for disp in (D3, None):
        ef = MolecularCrystal(device='cpu', energy_function='mace', mlip_path=MACE_CSP, mlip_dispersion=disp,
                              space_groups=[14])
        out = mn._reported(rows, reps, 'mace', 10, 10, 1.0, ef.predictor, None, 'cpu', 8)
        got[disp is None] = torch.tensor([float(r.mace) for r in out], dtype=torch.float64)
    assert (got[False] - want).abs().max() < 0.05, (got[False], want)     # kJ/mol per molecule
    assert (got[True] - want).abs().min() > 10.0, (got[True], want)       # D3 is tens of kJ/mol here
