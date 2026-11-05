"""NFPolicy molecule mode, one atom-table row per DISTINCT molecule (NFPolicy._molecule_table).

When the rows carry identifiers, the policy encodes each molecule once and points every row at its molecule, and the
flow reuses the molecule set's atom graph (spline_flow._atom_graph). Contracts:
- the density is the one the per-row table gives (same molecules, same numbers), typed and prefix rows alike;
- a known molecule set builds no second graph;
- rows sharing an identifier but carrying other atoms switch the shortcut off for the run, and the result is still
  the per-row one.

    python -m pytest -q tests/models/test_nf_molecule_table.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nf_typed_rows import _rows, _typed_policy  # noqa: E402

pytestmark = pytest.mark.fast


def _molecules(seed=0):
    """Three distinct molecules: (z [k], pos [k, 3], radius, volume)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for k in (3, 5, 4):
        out.append((torch.randint(1, 9, (k,), generator=g), torch.randn(k, 3, generator=g, dtype=torch.float64) * 1.5,
                    float(torch.rand(1, generator=g)) * 2 + 2.5, float(torch.rand(1, generator=g)) * 100 + 80))
    return out


def _batch(order, mols, ids=True, jitter=None):
    """A mol_batch whose rows are the molecules `order` names (repeats allowed)."""
    zs, ps, counts = [], [], []
    for j, m in enumerate(order):
        z, p, _, _ = mols[m]
        zs.append(z)
        ps.append(p + (jitter[j] if jitter is not None else 0))
        counts.append(len(z))
    counts = torch.tensor(counts)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    mb = SimpleNamespace(z=torch.cat(zs), pos=torch.cat(ps).float(), ptr=ptr, num_graphs=len(order),
                         batch=torch.repeat_interleave(torch.arange(len(order)), counts),
                         radius=torch.tensor([mols[m][2] for m in order]),
                         mol_volume=torch.tensor([mols[m][3] for m in order]))
    if ids:
        mb.identifier = [f'mol{m}' for m in order]
    return mb


ORDER = [0, 1, 0, 2, 1, 0]


def _cond(n, seed=0):
    return torch.randn(n, 4, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _log_q(pol, mb, x, cond=None):
    cond = _cond(len(x)) if cond is None else cond
    c = pol._context(cond, mb, len(x)).double()
    with torch.no_grad():
        return pol.sampler.log_prob(x, c)


@pytest.mark.parametrize('row_tokens', ['typed', 'prefix'])
def test_the_per_molecule_table_gives_the_per_row_density(row_tokens):
    pol = _typed_policy(14, scale=0.1, row_tokens=row_tokens)
    mols = _molecules()
    x = _rows(pol.sampler.flow.d, len(ORDER), 5) * 0.97
    per_row = _log_q(pol, _batch(ORDER, mols, ids=False), x)
    assert pol.sampler.flow.atom_z.shape[0] == len(ORDER)
    per_mol = _log_q(pol, _batch(ORDER, mols), x)
    assert pol.sampler.flow.atom_z.shape[0] == 3 and pol.sampler.flow.atom_key == ('mol0', 'mol1', 'mol2')
    assert torch.allclose(per_mol, per_row, atol=1e-6), (per_mol - per_row).abs().max()


def test_a_known_molecule_set_builds_no_second_graph(monkeypatch):
    import mxtaltools.models.functions.radial_graph as rg
    pol = _typed_policy(14, scale=0.1)
    mols = _molecules()
    x = _rows(pol.sampler.flow.d, len(ORDER), 6) * 0.97
    calls = []
    real = rg.build_radial_graph
    monkeypatch.setattr(rg, 'build_radial_graph', lambda *a, **k: calls.append(1) or real(*a, **k))
    first = _log_q(pol, _batch(ORDER, mols), x)
    second = _log_q(pol, _batch(list(reversed(ORDER)), mols), x.flip(0), _cond(len(ORDER)).flip(0))   # rows reversed
    assert len(calls) == 1
    assert torch.allclose(second, first.flip(0), atol=1e-6)


def test_other_atoms_under_one_identifier_fall_back_to_rows(capsys):
    pol = _typed_policy(14, scale=0.1)
    mols = _molecules()
    x = _rows(pol.sampler.flow.d, len(ORDER), 7) * 0.97
    jitter = [torch.zeros(len(mols[m][0]), 3, dtype=torch.float64) for m in ORDER]
    jitter[2] = jitter[2] + 0.5     # the second row of mol0: another conformer under the same name
    want = _log_q(pol, _batch(ORDER, mols, ids=False, jitter=jitter), x)
    got = _log_q(pol, _batch(ORDER, mols, jitter=jitter), x)
    assert 'carry different atoms' in capsys.readouterr().out and pol._molecule_table_off
    assert pol.sampler.flow.atom_z.shape[0] == len(ORDER)
    assert torch.allclose(got, want, atol=1e-10)
