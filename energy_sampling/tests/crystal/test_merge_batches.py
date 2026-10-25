"""energies/molecular_crystal.py::merge_batches and the chunked scoring pass that uses it
(MolecularCrystal.batched_analyze_crystal_batch with internal_oom_recovery on, the init-time scan of the prior).

merge_batches joins a list of batches by append_batch in a balanced tree. The result must be the batch the old
one-at-a-time appends built: the same rows in the same order, node tensors and graph lists alike, for any number of
parts. The chunked pass must return the same energies and the same rows, in order, as one unchunked call.

Nothing on disk is read: the molecules are synthetic, as in test_mxtaltools_crystal_boundary.py.

    python -m pytest -q tests/crystal/test_merge_batches.py
"""
from pathlib import Path

import pytest
import torch

import utils
from energies.molecular_crystal import MolecularCrystal, merge_batches
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

HERE = Path(__file__).resolve().parents[2]
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'


def _molecule(i: int) -> MolData:
    """a small rigid molecule; the atom count varies with i so node offsets differ between rows"""
    n = 3 + i % 3
    g = torch.Generator().manual_seed(i)
    pos = torch.randn(n, 3, generator=g) * (0.9 + 0.05 * i)
    return MolData(z=torch.tensor([6, 7, 8, 6, 7][:n], dtype=torch.long), pos=pos,
                   x=torch.zeros((n, 1), dtype=torch.float32), identifier=f'mol-{i}',
                   mol_volume=torch.tensor(55.0 + i), mass=torch.tensor(42.0 + i), radius=torch.tensor(1.2),
                   z_prime=torch.tensor(1, dtype=torch.long))


def _same(a, b):
    assert a.num_graphs == b.num_graphs
    assert list(a.identifier) == list(b.identifier)
    for key in ('z', 'pos', 'batch', 'mol_volume', 'mass'):
        torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=0, atol=0)


@pytest.mark.parametrize('n_parts', [1, 2, 3, 7, 8])
def test_tree_merge_equals_appending_one_part_at_a_time(n_parts):
    whole = collate_data_list([_molecule(i) for i in range(17)])
    cuts = torch.linspace(0, 17, n_parts + 1).round().long().tolist()
    parts = [whole.subsample_new_batch(list(range(cuts[k], cuts[k + 1]))) for k in range(n_parts)]
    seq = parts[0]
    for p in parts[1:]:
        seq = seq.append_batch(p)
    tree = merge_batches(list(parts))
    _same(tree, seq)
    _same(tree, whole)


def _energy_function():
    args = utils.get_train_args(['--config', str(CANONICAL)])
    cfg = {'device': 'cpu', 'energy_function': args.energy_function, 'mlip_path': args.mlip_path,
           'space_groups': args.space_groups, 'z_primes': args.z_primes, 'sg_conditioning': args.sg_conditioning,
           'temperature_conditioning': args.temperature_conditioning, 'zp_conditioning': args.zp_conditioning,
           'vector_conditioning': getattr(args, 'vector_conditioning', False),
           'vector_conditioning_dim': getattr(args, 'vector_conditioning_dim', None),
           'embedding_conditioning': getattr(args, 'embedding_conditioning', False),
           'embedding_conditioning_dim': getattr(args, 'embedding_conditioning_dim', None)}
    cfg.update(vars(args.energy_config))
    return MolecularCrystal(**cfg)


def test_the_chunked_pass_returns_what_one_call_returns(capsys):
    ef = _energy_function()
    capsys.readouterr()
    n = 9
    mols = collate_data_list([_molecule(i) for i in range(n)])
    g = torch.Generator().manual_seed(0)
    states = (torch.rand(n, 12, generator=g) - 0.5) * 0.6
    states[:, 3:6] = 0.0
    temperature = torch.full((n,), 2.5)

    e_one, b_one = ef.batched_analyze_crystal_batch(states, mols, temperature, return_batch=True,
                                                    internal_oom_recovery=False)
    ef.batch_size = 2   # chunks of 2, 3, 4 (the pass grows the chunk by one after each success)
    e_chunk, b_chunk = ef.batched_analyze_crystal_batch(states, mols, temperature, return_batch=True,
                                                        internal_oom_recovery=True)
    torch.testing.assert_close(e_chunk, e_one.detach().cpu(), equal_nan=True)
    assert list(b_chunk.identifier) == list(b_one.identifier) == [f'mol-{i}' for i in range(n)]
    torch.testing.assert_close(b_chunk.gfn_energy, b_one.gfn_energy.detach().cpu(), equal_nan=True)
