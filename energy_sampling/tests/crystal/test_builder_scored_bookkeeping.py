"""build_noised_prior.score: a scored batch keeps its own batch bookkeeping when the energy function scores another.

The analysis returned batches whose _slice_dict / _inc_dict objects a later analysis call wrote into. A K 0 build
(no children, so no final join that rebuilds the bookkeeping) then wrote its anchors with atom pointers of another
batch: after the calibration pilot scored its kicked rows, subsample_new_batch on the written file gathered atoms from
the wrong rows (pool8 held-out anchors, 2026-11-11: 250 atom positions for 240 atoms). This test would have caught it:
it scores rows of molecules with different atom counts, then scores a different batch with the same energy function,
and requires the first result's atom-level pointers to still equal its ptr and a subsample to gather its own atoms.

Run from energy_sampling/:  python -m pytest -q tests/crystal/test_builder_scored_bookkeeping.py
"""
from pathlib import Path

import pytest
import torch

import utils
from energies.molecular_crystal import MolecularCrystal
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

pytestmark = pytest.mark.fast

CANONICAL = Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'


def _molecule(identifier, n_atoms, scale=1.6):
    base = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1], [0.0, 0.0, 0.9], [-0.5, 0.6, 0.5]])
    pos = base[:n_atoms] * scale
    return MolData(z=torch.tensor([6, 7, 8, 6, 7][:n_atoms], dtype=torch.long), pos=pos,
                   x=torch.zeros((n_atoms, 1), dtype=torch.float32), identifier=identifier,
                   mol_volume=torch.tensor(18.0 * n_atoms * scale ** 3), mass=torch.tensor(13.0 * n_atoms),
                   radius=torch.tensor(1.2 * scale), z_prime=torch.tensor(1, dtype=torch.long))


def _energy(capsys):
    args = utils.get_train_args(['--config', str(CANONICAL)])
    capsys.readouterr()
    cfg = {'device': 'cpu', 'energy_function': args.energy_function, 'mlip_path': args.mlip_path,
           'space_groups': args.space_groups, 'z_primes': args.z_primes, 'sg_conditioning': args.sg_conditioning,
           'temperature_conditioning': args.temperature_conditioning, 'zp_conditioning': args.zp_conditioning}
    cfg.update(vars(args.energy_config))
    cfg['crystal_coordinates'] = 'latent'
    return MolecularCrystal(**cfg)


def _crystals(ef, sizes, seed):
    """Crystal rows of molecules with these atom counts (the latent route), as a stored prior holds them."""
    g = torch.Generator().manual_seed(seed)
    n = len(sizes)
    states = torch.rand(n, 12, generator=g) * 1.6 - 0.8
    states[:, :3] = 0.75 + 0.2 * torch.rand(n, 3, generator=g)
    mols = collate_data_list([_molecule(f'm{seed}-{i}', k) for i, k in enumerate(sizes)])
    _, b = ef.analyze_crystal_batch(states, mols, torch.full((n,), 2.5), return_batch=True)
    return b


def _pointers_follow_ptr(b):
    sd = b.__dict__.get('_slice_dict', {})
    n_nodes = int(b.ptr[-1])
    atom_keys = [k for k in ('pos', 'z', 'x') if k in sd and torch.as_tensor(getattr(b, k)).shape[0] == n_nodes]
    assert atom_keys, 'no atom-level bookkeeping to check'
    return all(torch.equal(sd[k].cpu(), b.ptr.cpu()) for k in atom_keys)


def test_a_scored_batch_keeps_its_bookkeeping_when_the_energy_function_scores_another(capsys):
    from data_processing.build_noised_prior import score
    ef = _energy(capsys)
    first_rows = _crystals(ef, [3, 5, 3, 5, 4, 3, 5, 4], seed=1)
    first, kept = score(ef, first_rows, 2.5, 'cpu', 4096)
    assert len(first) == 1 and bool(kept.all())
    first = first[0]
    assert _pointers_follow_ptr(first)
    other, _ = score(ef, _crystals(ef, [5, 5, 3, 4, 4, 3], seed=2), 2.5, 'cpu', 4096)   # another batch, other sizes
    assert _pointers_follow_ptr(first), 'scoring another batch rewrote this result\'s atom pointers'
    pick = torch.tensor([0, 2, 5, 6])
    sub = first.subsample_new_batch(pick)
    counts = (first.ptr[1:] - first.ptr[:-1])[pick]
    assert int(sub.pos.shape[0]) == int(sub.batch.shape[0]) == int(counts.sum())
    expect = torch.cat([first.pos[first.ptr[i]:first.ptr[i + 1]] for i in pick.tolist()])
    assert torch.equal(sub.pos, expect)
