"""cfg:noised_prior.train_molecules on the store route: the N molecules_path molecules with the lowest
(molecule_u, identifier) (data_processing/molecule_subset.py::lowest_molecules, train.py::
Modeller.subset_training_molecules) and every prior row of them, cut by mask in Modeller._store_take_rows; the clip
energies, floors and seed mask stay the whole file's; the store stamp; the held-out panel; the refusals. CPU, no energy
computed.

    python -m pytest -q -p no:cacheprovider tests/crystal/test_train_molecules.py
"""
import argparse
import pathlib
from types import SimpleNamespace

import pytest
import torch
import yaml

from data_processing import molecule_subset as MS
from energy_sampling import crystal_store as CS
from energy_sampling import noised_prior as NP
from energy_sampling.tests.crystal import store_stub as S
from energy_sampling.tests.crystal.test_row_store_trainer import _file
from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast

MK_DEV = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'


# -- 1. the key --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('value', [None, 1, 1000])
def test_settings_accept_null_and_a_positive_integer(value):
    assert NP.settings(SimpleNamespace(noised_prior={'train_molecules': value}))['train_molecules'] == value


@pytest.mark.parametrize('value', [0, -1, 1.5, 1000.0, True, '10'])
def test_settings_refuse_anything_else(value):
    with pytest.raises(ValueError, match='train_molecules must be an integer >= 1 or null'):
        NP.settings(SimpleNamespace(noised_prior={'train_molecules': value}))


def test_the_default_is_null_and_mk_dev_carries_it():
    assert NP.DEFAULTS['train_molecules'] is None
    assert NP.settings(SimpleNamespace(noised_prior=None))['train_molecules'] is None
    block = yaml.safe_load(MK_DEV.read_text(encoding='utf-8'))['noised_prior']
    assert 'train_molecules' in block and block['train_molecules'] is None


# -- 2. the choice -----------------------------------------------------------------------------------------------------

def test_the_choice_is_n_molecules_nested_and_blind_to_order_and_repeats():
    ids = [f'm{i}' for i in range(500)]
    ten, hundred = MS.lowest_molecules(ids, 10), MS.lowest_molecules(ids, 100)
    assert len(ten) == len(set(ten)) == 10 and len(hundred) == 100
    assert hundred[:10] == ten                                         # nested
    us = sorted((MS.molecule_u(m), m) for m in ids)
    assert ten == [m for _, m in us[:10]]                              # the lowest draws, in order
    shuffled = [ids[(7 * i) % 500] for i in range(500)] + ids[:200]    # another order, with repeats
    assert MS.lowest_molecules(shuffled, 100) == hundred


def test_ties_in_the_draw_are_broken_by_identifier(monkeypatch):
    monkeypatch.setattr(MS, 'molecule_u', lambda m: 0.5)
    assert MS.lowest_molecules(['c', 'a', 'b', 'a'], 2) == ['a', 'b']


@pytest.mark.parametrize('n', [0, -1, 2.0, True, None])
def test_a_bad_count_raises(n):
    with pytest.raises(ValueError, match='integer >= 1'):
        MS.lowest_molecules(['a', 'b'], n)


def test_more_molecules_than_available_raises():
    with pytest.raises(ValueError, match='3 molecules asked for, 2 available'):
        MS.lowest_molecules(['a', 'b', 'a'], 3)


# -- 3. the molecules_path hook ----------------------------------------------------------------------------------------

class _Rows:
    """The two things the hook reads off a crystal batch: identifier and subsample_new_batch."""

    def __init__(self, identifier):
        self.identifier = list(identifier)
        self.num_graphs = len(self.identifier)

    def subsample_new_batch(self, idx):
        return _Rows([self.identifier[i] for i in idx])


def _host(n_mol, store=True):
    m = Modeller.__new__(Modeller)
    m.args = SimpleNamespace(noised_prior={'train_molecules': n_mol}, molecules_path='conditions.pt',
                             buffers=SimpleNamespace(store=SimpleNamespace(enabled=store, ram_gb=16)))
    return m


def test_the_hook_keeps_whole_molecules_of_molecules_path():
    ids = [f'm{i // 3}' for i in range(3 * 50)]                       # 50 molecules, 3 rows each
    m = _host(7)
    sub = m.subset_training_molecules(_Rows(ids), 'molecules_path')
    kept = set(MS.lowest_molecules(ids, 7))
    assert set(sub.identifier) == kept and m._train_molecule_set == kept
    assert all(sub.identifier.count(x) == 3 for x in kept)


def test_the_prior_and_the_null_key_pass_through_as_the_same_object():
    rows = _Rows([f'm{i}' for i in range(20)])
    m = _host(5)
    assert m.subset_training_molecules(rows, 'prior_path') is rows
    m = _host(None, store=False)
    assert m.subset_training_molecules(rows, 'molecules_path') is rows and m._train_molecule_set is None


def test_the_hook_refuses_the_key_off_the_store_route_and_past_the_molecules_available():
    rows = _Rows([f'm{i}' for i in range(20)])
    with pytest.raises(ValueError, match='store route only'):
        _host(5, store=False).subset_training_molecules(rows, 'molecules_path')
    with pytest.raises(ValueError, match='21 molecules asked for, 20 available'):
        _host(21).subset_training_molecules(rows, 'molecules_path')


def test_a_prior_without_a_noised_manifest_refuses_the_key():
    m = SimpleNamespace(args=SimpleNamespace(noised_prior={'train_molecules': 5}, prior_path='p.pt'),
                        NOISED_PRIOR_RECORD=Modeller.NOISED_PRIOR_RECORD)
    with pytest.raises(ValueError, match='needs a pre-noised prior'):
        Modeller._noised_prior_record(m, None)


def test_the_inn_route_refuses_the_key():
    from inn_modeller import InnModeller
    args = SimpleNamespace(noised_prior={'train_molecules': 5})
    with pytest.raises(ValueError, match='--molecule_fraction'):
        InnModeller.check_buffer_options(SimpleNamespace(inn=SimpleNamespace(), args=args))


# -- 4. the cut, in the store load -------------------------------------------------------------------------------------

A, C = NP.KIND_ANCHOR, NP.KIND_CHILD
# molecule g = condition group g (store_stub.molecule(g), identifier 'm<g>'); anchors interleaved with children, and four
# held-out rows (an anchor and a child of m0 and of m1) at the end
KINDS = [A, A, C, A, C, C, C, C, A, C, A, C, A, C]
PARENTS = [0, 1, 0, 3, 1, 3, 0, 3, 8, 8, 10, 10, 12, 12]
GROUPS = [0, 1, 0, 2, 1, 2, 0, 2, 1, 1, 0, 0, 1, 1]
SPLITS = [0] * 10 + [1] * 4
FLOOR = {0: -5.0, 1: -7.0, 2: -3.0}


def _load(monkeypatch, n_mol=None, kept=None, fd_ok=None):
    """Modeller._store_take_rows on the file above: every training row in the windows but row 9 (an m1 child), the
    two held-out children as the held-out panel's rows, a distinct log R per row, the floor of each row's condition."""
    monkeypatch.setattr(S.physical_kick, 'group_kits', lambda layout: {})
    b, kinds, splits, groups = _file(KINDS, PARENTS, GROUPS, SPLITS)
    n = len(KINDS)
    m = Modeller.__new__(Modeller)
    m.energy_function = S.Energy()
    m.energy_function.fd_layout = None
    m.device = torch.device('cpu')
    m.args = SimpleNamespace(energy_config=SimpleNamespace(temperature=S.T), eval_num_samples=4, seed=0)
    m._noised_prior_settings = dict(live_log_range=(-2.0, -1.0), window_kT=10.0, packing_window=(0.5, 0.9),
                                    train_molecules=n_mol)
    m._train_molecule_set = kept
    train = splits == NP.SPLIT_TRAIN
    buffer = train.clone()
    buffer[9] = False
    masks = dict(floor_ok=train, buffer=buffer, heldout_children=~train & (kinds == C), anchors=buffer & (kinds == A))
    log_r = -0.1 * torch.arange(1, n + 1, dtype=torch.float64)
    floor = torch.tensor([FLOOR[g] for g in GROUPS], dtype=torch.float64)
    fd = torch.ones(n, dtype=torch.bool) if fd_ok is None else torch.tensor(fd_ok)
    energy, panel = m._store_take_rows(b, {}, torch.zeros(n, dtype=torch.float64), log_r, kinds, splits, groups, fd,
                                       masks, floor)
    return m, energy, panel


def test_only_the_chosen_molecules_rows_are_stored_with_their_own_anchors_as_parents(monkeypatch):
    m, _, panel = _load(monkeypatch, 2, frozenset({'m0', 'm2'}))
    load = m._store_load
    assert load['rows'].tolist() == [0, 2, 3, 5, 6, 7]
    assert load['kinds'].tolist() == [A, C, A, C, C, C]
    assert load['parent'].tolist() == [-1, 0, -1, 2, 0, 2]            # slots of anchors rows 0 (m0) and 3 (m2)
    assert load['t_groups'].tolist() == [0, 2]                         # templates of the chosen molecules only
    assert set(load['templates'].identifier) == {'m0', 'm2'}
    assert set(panel.identifier) == {'m0', 'm2'}
    assert set(m.prior_anchor_batch.identifier) == {'m0', 'm2'}
    assert set(m.prior_heldout_batch.identifier) == {'m0'}             # the held-out children of chosen molecules


def test_the_cut_leaves_floors_clip_energies_and_the_seed_mask_as_the_whole_file_has_them(monkeypatch):
    full, e_full, _ = _load(monkeypatch)
    cut, e_cut, _ = _load(monkeypatch, 2, frozenset({'m0', 'm2'}))
    assert full._store_load['t_groups'].tolist() == [0, 1, 2]
    assert full._store_load['floor'].tolist() == [FLOOR[0], FLOOR[1], FLOOR[2]]
    assert cut._store_load['floor'].tolist() == [FLOOR[0], FLOOR[2]]
    assert torch.equal(e_cut, e_full) and e_full.numel() == 9          # every scored training row in the windows
    assert torch.allclose(e_full, (0.1 * torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]) * S.T).float())
    assert torch.equal(cut._store_seed_mask['bits'], full._store_seed_mask['bits'])


def test_a_child_dropped_for_its_anchors_chart_point_leaves_the_clip_alike_and_is_counted_where_stored(monkeypatch):
    no_chart = [True, False] + [True] * 12                             # m1's first anchor has no chart point
    full, e_full, _ = _load(monkeypatch, fd_ok=no_chart)
    cut, e_cut, _ = _load(monkeypatch, 2, frozenset({'m0', 'm2'}), fd_ok=no_chart)
    assert full._physical['no_parent'] == 1 and cut._physical['no_parent'] == 0
    assert torch.equal(e_cut, e_full) and e_full.numel() == 8          # row 4 left out of both
    assert 4 not in full._store_load['rows'].tolist()


def test_the_row_mask_reads_identifiers_and_the_cut_spares_floor_ok():
    b, kinds, splits, groups = _file(KINDS, PARENTS, GROUPS, SPLITS)
    rows = NP.molecule_rows(b, {'m0', 'm2'})
    assert rows.dtype == torch.bool and rows.tolist() == [g in (0, 2) for g in GROUPS]
    masks = dict(floor_ok=torch.ones(len(KINDS), dtype=torch.bool), buffer=torch.ones(len(KINDS), dtype=torch.bool))
    out = NP.cut_masks(masks, rows)
    assert out['floor_ok'] is masks['floor_ok'] and torch.equal(out['buffer'], rows)


def test_a_set_key_without_the_hooks_choice_raises(monkeypatch):
    with pytest.raises(ValueError, match='chose None molecules'):
        _load(monkeypatch, 2, None)


# -- 5. the store stamp ------------------------------------------------------------------------------------------------

def _stamp(n_mol):
    m = Modeller.__new__(Modeller)
    m.energy_function = SimpleNamespace(energy_reference=None, crystal_coordinates='fd', data_ndim=12,
                                        condition_library_size=8, energy_clip=400.0, energy_reference_mode=None,
                                        lj_coeff=1.0, analyze_kwargs={})
    m.args = SimpleNamespace(energy_config=argparse.Namespace(temperature=2.5), prior_path='no/such/prior.pt',
                             energy_function='elj', mlip_path=None, space_groups=[2], z_primes=[1],
                             noised_prior={} if n_mol is None else {'train_molecules': n_mol})
    m.identifier_registry = {'m0': 0, 'm1': 1}
    m.prior_data_record = {'data/prior_prenoised': 1}
    m.row_store = SimpleNamespace(with_terms=False)
    return Modeller._store_stamp(m)


def test_a_null_key_leaves_a_stamp_written_before_it_valid():
    new = _stamp(None)
    assert new['noised_prior_settings']['train_molecules'] is None
    old = dict(new, noised_prior_settings={k: v for k, v in new['noised_prior_settings'].items()
                                           if k != 'train_molecules'})
    assert CS.stamp_diff(old, new) == []
    assert CS.stamp_diff(old, _stamp(1000)) == ['noised_prior_settings']


def test_a_store_written_at_one_count_is_refused_at_another():
    st = S.build().row_store
    state = st.state_dict(_stamp(1000), compact=True)
    assert st.state_mismatch(state, _stamp(1000)) is None
    why = st.state_mismatch(state, _stamp(10000))
    assert why is not None and 'noised_prior_settings' in why


# -- 6. the held-out panel ---------------------------------------------------------------------------------------------

def test_the_heldout_panel_picks_the_same_test_molecules_under_any_training_set():
    test = [f't{i:02d}' for i in range(40)]

    def chosen(train):
        reg = {ident: i for i, ident in enumerate(sorted(set(train) | set(test)))}
        m = Modeller.__new__(Modeller)
        m.args = SimpleNamespace(seed=7)
        m.energy_function = SimpleNamespace(space_groups=[2, 14])
        rows = [t for t in test for _ in range(2)]                    # two rows per test molecule
        m.test_mol_dataset = SimpleNamespace(batch=SimpleNamespace(
            mol_id=torch.tensor([reg[t] for t in rows]), num_graphs=len(rows)))
        idx, sg = m._store_heldout_panel(40)
        return [rows[i] for i in idx.tolist()], sg.tolist()

    full = chosen([f'a{i:03d}' for i in range(300)] + [f'u{i:03d}' for i in range(300)])
    small = chosen([f'u{i:03d}' for i in range(7)])
    assert full == small and len(set(full[0])) == 10                  # 40 // (2 x 2 groups) molecules
