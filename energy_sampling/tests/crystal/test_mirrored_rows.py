"""A stored Z'=1 row at aunit_handedness != +1 is refused at load, before anything scores it.

The trainer builds every crystal from the stored molecule at handedness +1 and never reads the
flag; the crystal search applies it. So a Z'=1 row at -1 trains as a different crystal from the
one its file describes. `train.py::mirrored_rows` finds those rows and
`Modeller._refuse_mirrored_rows` refuses them at every load site. What each test would catch:

  * a Z'=1 row at -1 (or 0) not found, or a +1 row found;
  * a collate pad (the columns past a row's own Z', written as 0) read as a handedness;
  * a Z'>1 row with a unit at -1 refused, or counted with the Z'=1 rows (it is printed);
  * a load site that does not run the check -- prior_path, molecules_path,
    test_molecules_path, a seed_source path -- or that scores the prior before it;
  * the check firing on a route that never places the molecule (toy, latent-scored);
  * a refusal that does not name the file, the count and the conversion.

CPU, synthetic molecules, no energy computed.

    python -m pytest -q tests/crystal/test_mirrored_rows.py
"""
from types import SimpleNamespace

import pytest
import torch

import energy_sampling.train as train
from energy_sampling.train import TRAINER_CHART_CONVERSION, Modeller, mirrored_rows
from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list

PHYSICAL = SimpleNamespace(is_crystal=True, latent_energy=False)    # elj, uma, mace, ...
TOY = SimpleNamespace(is_crystal=False, latent_energy=True)         # latent_(multi)harmonic
LATENT_SCORED = SimpleNamespace(is_crystal=True, latent_energy=True)  # latent_gaussian, latent_knn


def _molecule(i):
    pos = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1]]) * (1.0 + 0.1 * i)
    return MolData(z=torch.tensor([6, 7, 8]), pos=pos, x=torch.zeros((3, 1)), identifier=f'syn-{i}',
                   mol_volume=torch.tensor(55.0), mass=torch.tensor(42.0), radius=torch.tensor(1.2),
                   z_prime=torch.tensor(1, dtype=torch.long))


def _crystal(handedness, i=0):
    """sg 2, one molecule per unit; Z' is the length of `handedness`."""
    zp = len(handedness)
    m = _molecule(i)
    return MolCrystalData(molecule=[m.clone() for _ in range(zp)] if zp > 1 else m, sg_ind=2,
                          z_prime=torch.tensor(zp), cell_lengths=torch.full((3,), 8.0),
                          cell_angles=torch.full((3,), torch.pi / 2),
                          aunit_centroid=torch.full((3 * zp,), 0.25),
                          aunit_orientation=torch.full((3 * zp,), 0.1),
                          aunit_handedness=torch.tensor(handedness, dtype=torch.float32),
                          identifier=f'syn-{i}', max_z_prime=zp)


def _batch(*rows):
    return collate_data_list([_crystal(h, i) for i, h in enumerate(rows)],
                             max_z_prime=max(len(h) for h in rows))


def _save(tmp_path, name, batch):
    path = tmp_path / name
    torch.save({'prior': batch, 'equalized_prior': batch}, path)
    return str(path)


class _NotScored(SimpleNamespace):
    """A physical energy function that fails the test if anything scores through it."""

    def batched_analyze_crystal_batch(self, *a, **k):
        raise AssertionError('the file was scored before the handedness check ran')


def _modeller(energy_function=PHYSICAL, **args):
    m = Modeller.__new__(Modeller)
    m.energy_function = energy_function
    m.device = 'cpu'                    # buffer_device reads args, default 'cpu'
    m._buffer_kwargs = lambda: {}
    base = dict(prior_path=None, molecules_path=None, test_molecules_path=None,
                energy_function='elj', z_primes=[1], energy_config=SimpleNamespace(temperature=1.0),
                buffers=SimpleNamespace(anchor_buffer=SimpleNamespace(seed_source='generated')))
    base.update(args)
    m.args = SimpleNamespace(**base)
    return m


@pytest.fixture
def no_buffers(monkeypatch):
    """CrystalBuffer is out of scope: stand in a holder so a clean file loads without one."""
    monkeypatch.setattr(train, 'CrystalBuffer', lambda batch, **_: SimpleNamespace(batch=batch))


# ---------------------------------------------------------------- which rows

def test_zp1_rows_off_plus_one_are_found():
    zp1, zpn = mirrored_rows(_batch([1.0], [-1.0], [1.0], [-1.0], [0.0]))
    assert zp1.tolist() == [1, 3, 4] and zpn.tolist() == []


def test_a_collate_pad_is_not_a_handedness():
    b = _batch([1.0], [1.0, 1.0])
    assert float(b.aunit_handedness[0, 1]) == 0.0      # the pad this test is about
    zp1, zpn = mirrored_rows(b)
    assert zp1.tolist() == [] and zpn.tolist() == []


def test_a_zp2_unit_at_minus_one_is_counted_apart_from_zp1():
    zp1, zpn = mirrored_rows(_batch([1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0]))
    assert zp1.tolist() == [4] and zpn.tolist() == [1, 2]


def test_a_batch_without_the_column_has_no_rows():
    zp1, zpn = mirrored_rows(SimpleNamespace(num_graphs=3))
    assert zp1.numel() == 0 and zpn.numel() == 0


# ---------------------------------------------------------------- what the check does

def test_the_refusal_names_the_file_the_count_and_the_conversion():
    with pytest.raises(ValueError) as e:
        _modeller()._refuse_mirrored_rows(_batch([1.0], [-1.0], [-1.0]), 'D:/x/prior.pt', 'prior_path')
    msg = str(e.value)
    assert 'prior_path D:/x/prior.pt' in msg and "2 of 3 Z'=1 rows" in msg
    assert TRAINER_CHART_CONVERSION in msg and TRAINER_CHART_CONVERSION.startswith('build_qm9_full_prior.py')


def test_zp2_units_at_minus_one_are_printed_not_refused(capsys):
    _modeller()._refuse_mirrored_rows(_batch([1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]), 'acr.pt', 'prior_path')
    out = capsys.readouterr().out
    assert "2 of 3 rows are Z'>1" in out and 'NOT refused' in out


def test_a_clean_file_passes_silently(capsys):
    _modeller()._refuse_mirrored_rows(_batch([1.0], [1.0], [1.0, 1.0]), 'ok.pt', 'prior_path')
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('route', [TOY, LATENT_SCORED], ids=['toy', 'latent_scored'])
def test_routes_that_never_place_the_molecule_are_not_checked(route, capsys):
    _modeller(route)._refuse_mirrored_rows(_batch([-1.0], [-1.0], [1.0, -1.0]), 'toy.pt', 'prior_path')
    assert capsys.readouterr().out == ''


# ---------------------------------------------------------------- every load site runs it

def test_prior_path_is_refused_before_it_is_scored(tmp_path):
    path = _save(tmp_path, 'prior.pt', _batch([1.0], [-1.0], [1.0]))
    m = _modeller(_NotScored(is_crystal=True, latent_energy=False), prior_path=path)
    with pytest.raises(ValueError, match=r"prior_path .*1 of 3 Z'=1 rows"):
        m.init_prior_dataset()


@pytest.mark.parametrize('key', ['molecules_path', 'test_molecules_path'])
def test_condition_files_are_refused(tmp_path, no_buffers, key):
    clean = _save(tmp_path, 'clean.pt', _batch([1.0], [1.0]))
    bad = _save(tmp_path, 'bad.pt', _batch([1.0], [-1.0]))
    m = _modeller(molecules_path=bad if key == 'molecules_path' else clean,
                  test_molecules_path=bad if key == 'test_molecules_path' else None)
    with pytest.raises(ValueError, match=rf"{key} .*bad\.pt: 1 of 2 Z'=1 rows"):
        m.init_mol_dataset()


def test_clean_condition_files_load(tmp_path, no_buffers):
    clean = _save(tmp_path, 'clean.pt', _batch([1.0], [1.0]))
    m = _modeller(molecules_path=clean, test_molecules_path=clean)
    m.init_mol_dataset()
    assert m.mol_dataset.batch.num_graphs == 2 and m.test_mol_dataset.batch.num_graphs == 2


def test_a_seed_source_path_is_refused(tmp_path):
    path = _save(tmp_path, 'seeds.pt', _batch([-1.0], [1.0]))
    m = _modeller(_NotScored(is_crystal=True, latent_energy=False),
                  buffers=SimpleNamespace(anchor_buffer=SimpleNamespace(seed_source=path)))
    with pytest.raises(ValueError, match=r"buffers\.anchor_buffer\.seed_source .*1 of 2 Z'=1 rows"):
        m.init_anchor_buffer_seed()
