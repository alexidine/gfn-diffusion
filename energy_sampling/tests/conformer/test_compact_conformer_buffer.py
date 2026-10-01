"""Compact conformer stores (buffer.py::ConformerCompactRows) against full ones.

A compact store keeps each row's per-row fields and a slot in a per-condition table
(buffer.py::ConformerStatics) and materialises graphs on draw. Every test here builds the
SAME rows into a full ConformerBuffer / ConformerAnchorBuffer and a compact one, drives
both through the same operations, and requires the drawn graphs to be equal field for
field: every stored value (tensor dtype, shape and bits, NaN == NaN; lists element-wise),
ptr/batch, and the _slice_dict / _inc_dict bookkeeping that to_data_list reads.

DATA. A carrier-padded conditions set of four small molecules, built here the way
build_conformer_conditions.py --carrier writes one (float64), with random frozen
embeddings, cast to float32 as ConformerModeller._as_run_dtype casts it. Prior rows are
attach_states copies of those graphs with random states and energies.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/conformer/test_compact_conformer_buffer.py
"""
import numpy as np
import pytest
import torch

from buffer import (CONFORMER_COMPACT_FORMAT, CompactRowsError, ConformerAnchorBuffer,
                    ConformerBuffer, ConformerCompactRows, ConformerStatics)

SMIS = ['C', 'CO', 'N', 'CCO']
MOL_DIM, ENC = 16, 8
BULKY = ('fingerprint', 'rdf')
CHURNED = ('symmetry_operators', 'smiles', 'identifier') + BULKY


def _as_float32(batch):
    for key, val in list(batch._store.items()):
        if torch.is_tensor(val) and val.is_floating_point() and val.dtype != torch.float32:
            batch[key] = val.to(torch.float32)
    return batch


@pytest.fixture(scope='module')
def conditions():
    """The conditions table: one carrier-padded graph per molecule, float32."""
    from energies.conformer_carrier import carrier_pad_condition
    from energies.conformer_data import collate_conditions, condition_from_energy
    from energies.dof_features import free_dof_atom_index
    from energies.multi_conformer import MultiConformerTorsions

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        en = MultiConformerTorsions(SMIS, identifiers=SMIS, device='cpu', level='full',
                                    force_field='mmff')
        g = torch.Generator().manual_seed(0)
        rows = []
        for ident, mem in en._members.items():
            c = condition_from_energy(mem, identifier=ident)
            a, msk = free_dof_atom_index(mem)
            cc = carrier_pad_condition(c, en.carrier, ident, mem, atoms=a, mask=msk,
                                       R=int(a.shape[1]))
            cc.atom_embedding = torch.randn(int(cc.num_nodes), ENC, generator=g)
            cc.embedding = torch.randn(1, MOL_DIM, generator=g)
            rows.append(cc)
        batch = collate_conditions(rows)
    finally:
        torch.set_default_dtype(old)
    return _as_float32(batch)


def _prior_rows(conditions, per=(5, 7, 3, 6), seed=0):
    """attach_states rows of every condition, shuffled so conditions interleave."""
    from energies.conformer_data import attach_states

    g = torch.Generator().manual_seed(seed)
    graphs = conditions.to_data_list()
    k = int(conditions.n_torsions[0])
    parts = []
    for graph, n in zip(graphs, per):
        states = torch.rand((n, k), generator=g) * 2 - 1
        energies = torch.randn(n, generator=g) * 10
        parts.append(attach_states(graph, states, energies, identifier=graph.identifier))
    batch = parts[0]
    for part in parts[1:]:
        batch = batch.append_batch(part)
    return _as_float32(batch.subsample_new_batch(torch.randperm(batch.num_graphs, generator=g)))


def _table(conditions):
    """A fresh copy of the conditions batch, as mol_dataset would hold it."""
    return conditions.subsample_new_batch(torch.arange(conditions.num_graphs))


def _attach_mol_id(batch, idents):
    reg = {ident: i for i, ident in enumerate(sorted(set(idents)))}
    batch.add_graph_attr(torch.tensor([reg[i] for i in batch.identifier], dtype=torch.long),
                         'mol_id')
    return batch


def _condition(batch):
    """What ConformerTorsions.condition_samples attaches, without an energy."""
    batch.conditions = batch.embedding.reshape(batch.num_graphs, -1).clone()
    batch.condition_id = batch.mol_id.clone()
    return batch


def _same(x, y, path='batch'):
    """Every path at which two drawn graph batches differ (empty = equal)."""
    out = []
    kx, ky = set(x._store.keys()), set(y._store.keys())
    if kx != ky:
        out.append(f'{path} keys: {sorted(kx ^ ky)}')
    for key in kx & ky:
        a, b = x._store[key], y._store[key]
        if torch.is_tensor(a):
            ok = (torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype
                  and a.device == b.device)
            if ok and a.is_floating_point():
                ok = torch.equal(torch.isnan(a), torch.isnan(b)) and \
                    torch.equal(a[~torch.isnan(a)], b[~torch.isnan(b)])
            elif ok:
                ok = torch.equal(a, b)
            if not ok:
                out.append(f'{path}.{key}')
        elif a != b:
            out.append(f'{path}.{key}')
    for name in ('_slice_dict', '_inc_dict'):
        dx, dy = x.__dict__.get(name, {}), y.__dict__.get(name, {})
        if set(dx) != set(dy):
            out.append(f'{path}.{name} keys: {sorted(set(dx) ^ set(dy))}')
        else:
            out += [f'{path}.{name}[{k}]' for k in dx if not torch.equal(dx[k], dy[k])]
    if x.num_graphs != y.num_graphs:
        out.append(f'{path}.num_graphs')
    return out


def _check_draws(full, compact, inds=None, seed=0):
    """The same rows drawn from both stores are the same graphs."""
    assert len(full) == len(compact)
    n = len(full)
    if inds is None:
        rng = np.random.default_rng(seed)
        inds = rng.integers(0, n, size=max(3 * n, 8))     # repeats and every order
    assert _same(full.batch.subsample_new_batch(inds),
                 compact.batch.subsample_new_batch(inds)) == []
    assert _same(full.batch.subsample_new_batch(np.arange(n)), compact.batch.clone()) == []
    assert torch.equal(full.x, compact.x)
    if full.y is not None:
        assert torch.equal(full.y, compact.y)


def _pair(rows, table, cls=ConformerBuffer, exclude=BULKY, **kw):
    statics = ConformerStatics(table)
    full = cls(rows, 'cpu', exclude_keys=exclude, **kw)
    compact = cls(rows, 'cpu', exclude_keys=exclude, statics=statics, **kw)
    return full, compact, statics


# ------------------------------------------------------------------- equality


def test_materialised_graphs_equal_full_graphs_on_a_mixed_condition_prior(conditions):
    rows = _prior_rows(conditions)
    assert len(set(rows.identifier)) == len(SMIS), 'the batch must mix every condition'
    full, compact, _ = _pair(rows, _table(conditions), y_fn='conformer_energy')
    assert compact.is_compact and not full.is_compact
    assert set(compact.batch._rows) == {'torsion_state', 'conformer_energy'}
    _check_draws(full, compact)
    # attribute reads through the resident batch, as train.py makes them
    for key in ('identifier', 'n_torsions', 'dof_static', 'atom_embedding', 'pos',
                'torsion_state', 'conformer_energy'):
        a, b = getattr(full.batch, key), getattr(compact.batch, key)
        assert (a == b) if isinstance(a, list) else torch.equal(a, b), key
    assert not hasattr(compact.batch, 'asym_unit_lut')


def test_mol_id_attached_later_is_read_from_the_table(conditions):
    """init_identifiers attaches mol_id to mol_dataset first, then to the prior dataset."""
    rows = _prior_rows(conditions)
    table = _table(conditions)
    full, compact, _ = _pair(rows, table, y_fn='conformer_energy')
    _attach_mol_id(table, SMIS)
    for buf in (full, compact):
        _attach_mol_id(buf.batch, SMIS)
    assert 'mol_id' in compact.batch._static and 'mol_id' not in compact.batch._rows
    _check_draws(full, compact)


def test_seeded_stores_admit_rows_without_identifier_by_mol_id(conditions):
    """The prior/anchor seed path: materialise, condition, drop the string keys, build."""
    table = _attach_mol_id(_table(conditions), SMIS)
    rows = _condition(_attach_mol_id(_prior_rows(conditions), SMIS))
    full, compact, statics = _pair(rows, table, exclude=CHURNED, y_fn='conformer_energy')
    assert not hasattr(full.batch, 'identifier') and not hasattr(compact.batch, 'identifier')
    assert set(compact.batch._rows) == {'torsion_state', 'conformer_energy', 'conditions',
                                        'condition_id'}
    more = _condition(_attach_mol_id(_prior_rows(conditions, per=(2, 2, 4, 3), seed=1), SMIS))
    for key in ('identifier', 'smiles'):
        del more[key]                              # sample_graphs drops them at draw time
    full.add(more)
    compact.add(more)
    _check_draws(full, compact)


def test_derived_condition_fields_are_read_from_the_table(conditions):
    """ConformerModeller._derive_condition_fields puts conditions / condition_id on the
    table; conditioned rows then store only state and energy, and a row whose conditions
    differ (a per-row temperature column would) keeps its own."""
    table = _attach_mol_id(_table(conditions), SMIS)
    statics = ConformerStatics(table)
    probe = _condition(_attach_mol_id(_table(conditions), SMIS))
    statics.derive('conditions', probe.conditions)
    statics.derive('condition_id', probe.condition_id)
    rows =_condition(_attach_mol_id(_prior_rows(conditions), SMIS))
    full = ConformerBuffer(rows, 'cpu', exclude_keys=CHURNED, y_fn='conformer_energy')
    compact = ConformerBuffer(rows, 'cpu', exclude_keys=CHURNED, y_fn='conformer_energy',
                              statics=statics)
    assert set(compact.batch._rows) == {'torsion_state', 'conformer_energy'}
    _check_draws(full, compact)
    hot = _condition(_attach_mol_id(_prior_rows(conditions, seed=8), SMIS))
    hot.conditions = hot.conditions + 0.5
    full.add(hot)
    compact.add(hot)
    assert 'conditions' in compact.batch._rows and 'condition_id' in compact.batch._static
    _check_draws(full, compact)
    restored = ConformerBuffer.from_state_dict(compact.state_dict(), device='cpu')
    restored.bind_statics(statics)
    _check_draws(full, restored)


# ------------------------------------------------------------ add / purge / churn


def test_add_purge_and_churn_keep_the_stores_equal(conditions):
    table = _attach_mol_id(_table(conditions), SMIS)
    rows = _condition(_attach_mol_id(_prior_rows(conditions), SMIS))
    full, compact, _ = _pair(rows, table, exclude=CHURNED, y_fn='conformer_energy')
    rng = np.random.default_rng(3)
    for step in range(12):
        new = _condition(_attach_mol_id(
            _prior_rows(conditions, per=tuple(rng.integers(2, 5, size=4)), seed=10 + step),
            SMIS))
        full.add(new, birth_step=step)
        compact.add(new, birth_step=step)
        drop = rng.choice(len(full), size=int(rng.integers(1, 6)), replace=False)
        full.purge_by_index(drop)
        compact.purge_by_index(drop)
        _check_draws(full, compact, seed=step)
        for col in ('birth_step', 'select_counts', 'origin', 'is_val'):
            assert torch.equal(getattr(full, col), getattr(compact, col)), col
    # the stored rows are the per-row fields and one key each, not the graphs
    per_row = compact.batch.nbytes() / len(compact)
    assert per_row < 0.25 * _graph_bytes(full.batch) / len(full)


def _graph_bytes(batch):
    return sum(v.numel() * v.element_size() for v in batch._store.values() if torch.is_tensor(v))


def test_a_per_graph_field_that_differs_is_stored_per_row(conditions):
    table = _table(conditions)
    rows = _prior_rows(conditions)
    rows.dof_static = rows.dof_static + torch.arange(rows.num_graphs, dtype=torch.float32)[:, None]
    full, compact, _ = _pair(rows, table, y_fn='conformer_energy')
    assert 'dof_static' in compact.batch._rows
    _check_draws(full, compact)
    # and a later admission that matches the table stays equal on the merged store
    clean = _prior_rows(conditions, seed=5)
    full.add(clean)
    compact.add(clean)
    _check_draws(full, compact)


def test_std_oriented_rows_keep_their_positions_per_row(conditions):
    """The trainer std-orients forward and anchor batches (orient_molecule), which moves
    `pos` off the table's reference; those rows are admitted to the replay and prior
    buffers."""
    table = _attach_mol_id(_table(conditions), SMIS)
    rows = _condition(_attach_mol_id(_prior_rows(conditions), SMIS))
    full, compact, _ = _pair(rows, table, exclude=CHURNED, y_fn='conformer_energy')
    assert 'pos' in compact.batch._static
    oriented = _condition(_attach_mol_id(_prior_rows(conditions, seed=4), SMIS))
    oriented.orient_molecule(mode='std')
    full.add(oriented)
    compact.add(oriented)
    assert set(compact.batch._node_rows) == {'pos'}
    _check_draws(full, compact)
    rng = np.random.default_rng(0)
    for _ in range(3):
        drop = rng.choice(len(full), size=4, replace=False)
        full.purge_by_index(drop)
        compact.purge_by_index(drop)
        _check_draws(full, compact)
    restored = ConformerBuffer.from_state_dict(compact.state_dict(), device='cpu')
    restored.bind_statics(ConformerStatics(table))
    _check_draws(full, restored)


def test_a_list_field_that_differs_is_refused(conditions):
    rows = _prior_rows(conditions)
    rows.smiles = ['X'] + list(rows.smiles[1:])
    with pytest.raises(CompactRowsError, match='smiles'):
        ConformerBuffer(rows, 'cpu', exclude_keys=BULKY, statics=ConformerStatics(_table(conditions)))


def test_a_row_of_an_unknown_condition_is_refused(conditions):
    rows = _prior_rows(conditions)
    rows.identifier = ['not-a-condition'] + list(rows.identifier[1:])
    with pytest.raises(CompactRowsError, match='not-a-condition'):
        ConformerBuffer(rows, 'cpu', exclude_keys=BULKY, statics=ConformerStatics(_table(conditions)))


def test_a_row_of_another_atom_count_is_refused(conditions):
    """Two conditions swapped in the identifier list: the atoms no longer line up."""
    rows = _prior_rows(conditions)
    idents = list(rows.identifier)
    swap = {'C': 'CCO', 'CCO': 'C'}
    rows.identifier = [swap.get(i, i) for i in idents]
    with pytest.raises(CompactRowsError, match='atoms'):
        ConformerBuffer(rows, 'cpu', exclude_keys=BULKY, statics=ConformerStatics(_table(conditions)))


# ------------------------------------------------------------- persistence


def test_state_dict_round_trip_and_bind(conditions):
    table = _attach_mol_id(_table(conditions), SMIS)
    rows = _condition(_attach_mol_id(_prior_rows(conditions), SMIS))
    full, compact, statics = _pair(rows, table, exclude=CHURNED, y_fn='conformer_energy')
    compact.update_losses(torch.arange(len(compact), dtype=torch.float32), np.arange(len(compact)))
    state = compact.state_dict()
    assert state['batch'] is None
    assert state['compact_rows']['format'] == CONFORMER_COMPACT_FORMAT
    assert set(state['compact_rows']['identifiers']) == set(SMIS)
    restored = ConformerBuffer.from_state_dict(state, device='cpu')
    assert restored.is_compact and not restored.batch.is_bound and len(restored) == len(full)
    with pytest.raises(CompactRowsError, match='not bound'):
        restored.batch.subsample_new_batch([0])
    # a FRESH table, as a resumed run rebuilds it from molecules_path, in another order
    order = torch.tensor([3, 1, 0, 2])
    table2 = _attach_mol_id(_table(conditions).subsample_new_batch(order), SMIS)
    assert restored.bind_statics(ConformerStatics(table2)) == 'bound'
    _check_draws(full, restored)
    assert torch.equal(restored.ema_loss, compact.ema_loss)
    # and it keeps working: admission after the resume
    more = _condition(_attach_mol_id(_prior_rows(conditions, seed=7), SMIS))
    full.add(more)
    restored.add(more)
    _check_draws(full, restored)


def test_a_full_row_sidecar_is_converted_on_bind(conditions):
    table = _attach_mol_id(_table(conditions), SMIS)
    rows = _condition(_attach_mol_id(_prior_rows(conditions), SMIS))
    full = ConformerBuffer(rows, 'cpu', exclude_keys=CHURNED, y_fn='conformer_energy')
    restored = ConformerBuffer.from_state_dict(full.state_dict(), device='cpu')
    assert not restored.is_compact
    assert restored.bind_statics(ConformerStatics(table)) == 'converted'
    assert restored.is_compact
    _check_draws(full, restored)


def test_bind_refuses_a_table_without_a_stored_condition(conditions):
    table = _attach_mol_id(_table(conditions), SMIS)
    rows = _condition(_attach_mol_id(_prior_rows(conditions), SMIS))
    _, compact, _ = _pair(rows, table, exclude=CHURNED, y_fn='conformer_energy')
    restored = ConformerBuffer.from_state_dict(compact.state_dict(), device='cpu')
    short = _table(conditions).subsample_new_batch(torch.tensor([0, 1, 2]))
    with pytest.raises(CompactRowsError, match='not in this run'):
        restored.bind_statics(ConformerStatics(short))


def test_anchor_store_round_trip_admit_and_thin(conditions):
    table = _attach_mol_id(_table(conditions), SMIS)
    rows = _condition(_attach_mol_id(_prior_rows(conditions), SMIS))
    n = rows.num_graphs
    energy = rows.conformer_energy.clone()
    full, compact, statics = _pair(rows, table, cls=ConformerAnchorBuffer, exclude=CHURNED,
                                   reward=-energy, energy=energy)
    cand = _condition(_attach_mol_id(_prior_rows(conditions, seed=9), SMIS))
    e = cand.conformer_energy.clone()
    assert full.admit(cand, -e, e, dup_cutoff=0.5) == compact.admit(cand, -e, e, dup_cutoff=0.5)
    _check_draws(full, compact)
    floor = torch.full((len(SMIS),), float(energy.min()))
    full.thin(floor, energy_window=15.0, max_size=n)
    compact.thin(floor, energy_window=15.0, max_size=n)
    _check_draws(full, compact)
    assert torch.equal(full.condition_id, compact.condition_id)
    restored = ConformerAnchorBuffer.from_state_dict(compact.state_dict(), device='cpu')
    restored.bind_statics(statics)
    _check_draws(full, restored)
    assert torch.equal(restored.energy, full.energy)


# ------------------------------------------------------------------ draws


def test_condition_blocked_and_aligned_draws_pick_the_same_rows(conditions):
    table = _attach_mol_id(_table(conditions), SMIS)
    rows = _condition(_attach_mol_id(_prior_rows(conditions), SMIS))
    full, compact, _ = _pair(rows, table, exclude=CHURNED, y_fn='conformer_energy')
    for kwargs in ({'condition_block_m': 2}, {'condition_block_m': 3, 'target_cids': [1, 2]},
                   {'target_cids': [0]}, {}):
        draws = []
        for buf in (full, compact):
            np.random.seed(11)
            graphs, inds, _ = buf.sample_graphs(12, **kwargs)
            draws.append((graphs, inds))
        assert np.array_equal(draws[0][1], draws[1][1]), kwargs
        assert _same(draws[0][0], draws[1][0]) == [], kwargs
    np.random.seed(4)
    a = full._sample_indices(6, draw_cids=[0, 3], rows_per_condition=3)
    np.random.seed(4)
    b = compact._sample_indices(6, draw_cids=[0, 3], rows_per_condition=3)
    assert np.array_equal(a, b)
    assert np.array_equal(full.condition_row_counts(4), compact.condition_row_counts(4))


def test_row_batches_hand_out_every_row_in_chunks(conditions):
    """train.py::_dataset_row_batches: chunks of a compact store equal the full store's
    clone-and-subsample, under the same torch seed."""
    import train

    rows = _prior_rows(conditions)
    full, compact, _ = _pair(rows, _table(conditions), y_fn='conformer_energy')
    for limit in (None, 9):
        torch.manual_seed(2)
        whole = train._dataset_row_batches(full, limit)
        torch.manual_seed(2)
        parts = list(compact.row_batches(limit, chunk=4))
        assert len(whole) == 1 and len(parts) > 1
        assert sum(p.num_graphs for p in parts) == whole[0].num_graphs
        merged = parts[0]
        for p in parts[1:]:
            merged = merged.append_batch(p)
        assert _same(whole[0], merged) == []


# ------------------------------------------------------- the compact prior FILE
#
# energies/conformer_data.py::compact_prior: per row its state, energy and condition index.
# ConformerModeller.init_prior_dataset joins it to the conditions table without a graph per
# row (ConformerCompactRows.from_columns).


def _masked_rows(conditions, **kw):
    """_prior_rows with pad columns zeroed (a stored state's pads are 0), grouped by
    condition in table order, as a prior file holds them."""
    rows = _prior_rows(conditions, **kw)
    slot = {s: j for j, s in enumerate(conditions.identifier)}
    at = torch.tensor([slot[s] for s in rows.identifier])
    n = rows.num_graphs
    mask = conditions.state_mask.reshape(conditions.num_graphs, -1).bool()[at]
    rows.torsion_state = rows.torsion_state.reshape(n, -1) * mask
    return rows.subsample_new_batch(torch.argsort(at, stable=True))


def _compact_blob(conditions, rows):
    from energies.conformer_data import compact_prior

    slot = {s: j for j, s in enumerate(conditions.identifier)}
    idents = list(dict.fromkeys(rows.identifier))
    at = [slot[s] for s in idents]
    return compact_prior(
        idents, [list(rows.identifier).count(s) for s in idents],
        rows.torsion_state.reshape(rows.num_graphs, -1).double(),      # a float64 builder
        rows.conformer_energy.reshape(-1).double(),
        conditions.state_mask.reshape(conditions.num_graphs, -1).bool()[at],
        (conditions.ptr[1:] - conditions.ptr[:-1])[at])


def _modeller(conditions, prior_path, k, anchor=None, table=True):
    import types

    from conformer_modeller import ConformerModeller

    m = ConformerModeller.__new__(ConformerModeller)
    m.device = 'cpu'
    m.args = types.SimpleNamespace(
        prior_path=str(prior_path), molecules_path='conditions.pt' if table else None,
        prior_dataset_noise=None, buffer_device='cpu',
        buffers=types.SimpleNamespace(anchor_buffer=types.SimpleNamespace(**(anchor or {}))))
    m.energy_function = types.SimpleNamespace(ndim=k, dtype=torch.float32)
    m.mol_dataset = ConformerBuffer(_table(conditions), 'cpu', exclude_keys=BULKY)
    return m


@pytest.fixture
def float32_default():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(old)


def test_a_compact_prior_file_loads_as_the_store_its_graph_rows_give(conditions, tmp_path,
                                                                     float32_default):
    from energies.conformer_data import (PRIOR_COMPACT_FORMAT, PRIOR_GRAPH_FORMAT,
                                         save_prior_file)

    rows = _masked_rows(conditions)
    k = int(conditions.n_torsions[0])
    path = save_prior_file(_compact_blob(conditions, rows), tmp_path / 'prior.pt', source='t')
    blob = torch.load(path, weights_only=False)
    assert blob['prior_format'] == PRIOR_COMPACT_FORMAT and blob['source'] == 't'
    assert 'prior' not in blob and 'equalized_prior' not in blob

    m = _modeller(conditions, path, k)
    m.init_prior_dataset()
    got = m.prior_dataset
    assert got.is_compact and len(got) == rows.num_graphs
    assert set(got.batch._rows) == {'torsion_state', 'conformer_energy'}
    assert got.x.dtype == torch.float32 and got.y.dtype == torch.float32
    # the same store the graph rows build: the same static fields, and equal draws
    full = ConformerBuffer(rows, 'cpu', exclude_keys=BULKY, y_fn='conformer_energy')
    from_graphs = ConformerBuffer(rows, 'cpu', exclude_keys=BULKY, y_fn='conformer_energy',
                                  statics=m.conformer_statics)
    assert set(got.batch._static) == set(from_graphs.batch._static)
    _check_draws(full, got)
    # and the mol_id init_identifiers attaches later is read from the table for these rows too
    _attach_mol_id(m.mol_dataset.batch, SMIS)
    _attach_mol_id(got.batch, SMIS)
    _attach_mol_id(full.batch, SMIS)
    assert 'mol_id' in got.batch._static
    _check_draws(full, got)

    # the graph form is still written for a graph batch, and names itself
    graph = save_prior_file(rows, tmp_path / 'graph.pt')
    assert torch.load(graph, weights_only=False)['prior_format'] == PRIOR_GRAPH_FORMAT


def _edit(conditions, tmp_path, edit, table=True):
    from energies.conformer_data import save_prior_file

    rows = _masked_rows(conditions)
    blob = _compact_blob(conditions, rows)
    edit(blob)
    path = save_prior_file(blob, tmp_path / 'prior.pt')
    return _modeller(conditions, path, int(conditions.n_torsions[0]), table=table)


def _set(key, value):
    def edit(blob):
        blob[key] = value(blob[key])
    return edit


def _nonzero_pad(blob):
    r, c = map(int, torch.nonzero(~blob['state_mask'][blob['condition_index']])[0])
    blob['torsion_state'][r, c] = 0.5


@pytest.mark.parametrize('edit, why', [
    (_set('identifiers', lambda v: ['CCCC'] + v[1:]), 'not in the conditions table'),
    (_set('state_mask', lambda v: ~v), 'free-column mask or an atom count'),
    (_set('n_atoms', lambda v: v + 1), 'free-column mask or an atom count'),
    (_nonzero_pad, 'non-zero PAD column'),
    (_set('conformer_energy', lambda v: v[1:]), 'inconsistent compact prior'),
    (_set('condition_index', lambda v: v + 100), 'inconsistent compact prior'),
])
def test_a_compact_prior_file_that_is_not_the_tables_is_refused(conditions, tmp_path, edit, why,
                                                                float32_default):
    m = _edit(conditions, tmp_path, edit)
    with pytest.raises(SystemExit, match=why):
        m.init_prior_dataset()


def test_a_compact_prior_file_needs_the_conditions_table(conditions, tmp_path, float32_default):
    m = _edit(conditions, tmp_path, lambda blob: None, table=False)
    with pytest.raises(SystemExit, match='compact prior file'):
        m.init_prior_dataset()


def test_from_columns_refuses_columns_that_are_not_rows(conditions):
    statics = ConformerStatics(_table(conditions))
    keys = torch.tensor([0, 1, 1, 3])
    k = int(conditions.n_torsions[0])
    cols = {'torsion_state': torch.zeros(4, k), 'conformer_energy': torch.zeros(4)}
    rows = ConformerCompactRows.from_columns(statics, keys, cols, exclude_keys=BULKY)
    assert rows.num_graphs == 4 and torch.equal(rows.keys, keys)
    with pytest.raises(CompactRowsError, match='outside the'):
        ConformerCompactRows.from_columns(statics, torch.tensor([0, 9]), {})
    with pytest.raises(CompactRowsError, match='one entry per row'):
        ConformerCompactRows.from_columns(statics, keys, {'torsion_state': torch.zeros(3, k)})
    with pytest.raises(CompactRowsError, match='not a per-graph field'):
        ConformerCompactRows.from_columns(statics, keys, {'pos': torch.zeros(4, 3)})


# ----------------------------------------------------------- the anchor seed's rows


def test_lowest_rows_per_condition():
    from buffer import lowest_rows_per_condition as pick

    keys = torch.tensor([7, 7, 7, 2, 2, 5, 7, 2])
    e = torch.tensor([3., 1., 2., 9., 8., 0., 0.5, 8.])
    # no count: every row, in row order
    rows, k, n = pick(keys, e)
    assert rows.tolist() == list(range(8)) and (k, n) == (4, 3)
    rows, k, n = pick(keys, e, 2)
    assert (k, n) == (2, 3) and rows.tolist() == [1, 4, 5, 6, 7]     # tie at 8.0: the earlier row
    rows, k, _ = pick(keys, e, 1)
    assert k == 1 and rows.tolist() == [4, 5, 6]
    rows, k, _ = pick(keys, e, 50)
    assert rows.tolist() == list(range(8))
    with pytest.raises(ValueError, match='>= 1'):
        pick(keys, e, 0)
    rows, k, n = pick(torch.zeros(0, dtype=torch.long), torch.zeros(0))
    assert rows.numel() == 0 and (k, n) == (0, 0)
    # NaN energies are taken last
    rows, _, _ = pick(torch.tensor([0, 0, 0]), torch.tensor([float('nan'), 2., 1.]), 2)
    assert rows.tolist() == [1, 2]


def test_the_anchor_seed_takes_every_row_of_a_compact_prior_dataset(conditions, tmp_path,
                                                                    float32_default, capsys,
                                                                    monkeypatch):
    import buffer
    from energies.conformer_data import save_prior_file

    # two rows per chunk, so the load's pad check and the seed both run over several chunks
    monkeypatch.setattr(buffer, 'COMPACT_CHUNK_ROWS', 2)
    rows = _masked_rows(conditions)                 # 5, 7, 3 and 6 rows over four conditions
    k = int(conditions.n_torsions[0])
    path = save_prior_file(_compact_blob(conditions, rows), tmp_path / 'prior.pt')
    e = rows.conformer_energy.reshape(-1)
    at = torch.tensor([list(conditions.identifier).index(s) for s in rows.identifier])

    def seed(**anchor):
        m = _modeller(conditions, path, k, anchor=anchor)
        m.init_prior_dataset()
        parts = list(m._anchor_seed_row_batches(m.prior_dataset))
        # chunks of two rows, a single leftover row joined to the last of them
        assert len(parts) > 1 and all(part.num_graphs in (2, 3) for part in parts)
        out = parts[0]
        for part in parts[1:]:
            out = out.append_batch(part)
        return out

    # every row, in order, whatever max_size says: it does not limit the seed
    for max_size in (1000, 14, 3):
        everything = seed(max_size=max_size)
        assert torch.equal(everything.conformer_energy.reshape(-1), e)
    assert '21 of 21 prior-dataset rows over 4 condition(s) (every row)' in capsys.readouterr().out
    # the thinning knob: each condition's lowest rows
    three = seed(max_size=3, seed_rows_per_condition=3)
    assert three.num_graphs == 12
    want = torch.cat([torch.sort(e[at == c]).values[:3] for c in range(4)])
    assert torch.equal(torch.sort(three.conformer_energy.reshape(-1)).values,
                       torch.sort(want).values)
    assert 'the lowest-energy 3 per condition' in capsys.readouterr().out
    one = seed(max_size=1000, seed_rows_per_condition=1)
    assert torch.equal(torch.sort(one.conformer_energy.reshape(-1)).values,
                       torch.sort(torch.stack([e[at == c].min() for c in range(4)])).values)
    with pytest.raises(SystemExit, match='seed_rows_per_condition'):
        seed(max_size=10, seed_rows_per_condition=0)


def test_the_anchor_capacity_is_the_larger_of_max_size_and_the_rows_held(capsys):
    """ConformerModeller._resolve_anchor_capacity, after a seed and after a restore: a set
    larger than the configured max_size raises the capacity to its size, a smaller one
    leaves the configured value, and no buffer leaves it untouched."""
    import types

    from conformer_modeller import ConformerModeller

    def resolved(max_size, held):
        m = ConformerModeller.__new__(ConformerModeller)
        m.args = types.SimpleNamespace(buffers=types.SimpleNamespace(
            anchor_buffer=types.SimpleNamespace(max_size=max_size)))
        if held is not None:
            m.anchor_buffer = [None] * held
        m._resolve_anchor_capacity('config seed')
        return m.args.buffers.anchor_buffer.max_size

    assert resolved(40_000, 216_950) == 216_950
    assert 'capacity 216,950 rows = max(configured buffers.anchor_buffer.max_size 40,000, ' \
           '216,950 rows held)' in capsys.readouterr().out
    assert resolved(40_000, 449) == 40_000
    assert 'capacity 40,000 rows' in capsys.readouterr().out
    assert resolved(40_000, None) == 40_000 and capsys.readouterr().out == ''


def test_the_anchor_seed_hands_the_base_method_its_rows_by_standing_in(conditions, tmp_path,
                                                                      float32_default,
                                                                      monkeypatch):
    """ConformerModeller.init_anchor_buffer_seed: a compact prior dataset stands in with its
    seed rows as `row_batches`, which train.py::_dataset_row_batches reads; a graph-row dataset
    reaches the base method as itself; prior_dataset is put back; and the capacity is resolved
    from the buffer the base method built, larger than the configured max_size or not."""
    import train
    from energies.conformer_data import save_prior_file

    rows = _masked_rows(conditions)
    path = save_prior_file(_compact_blob(conditions, rows), tmp_path / 'prior.pt')
    seen = []

    def base_seed(self):
        n = sum(b.num_graphs for b in train._dataset_row_batches(self.prior_dataset))
        seen.append(n)
        self.anchor_buffer = [None] * n            # a stand-in with the seeded length

    monkeypatch.setattr(train.Modeller, 'init_anchor_buffer_seed', base_seed)

    def modeller(**anchor):
        m = _modeller(conditions, path, int(conditions.n_torsions[0]),
                      anchor={'seed_source': 'prior_dataset', **anchor})
        m.init_prior_dataset()
        return m

    # a set LARGER than max_size seeds fully, and the capacity becomes its size
    m = modeller(max_size=14)
    compact = m.prior_dataset
    m.init_anchor_buffer_seed()
    assert seen == [21] and m.prior_dataset is compact
    assert m.args.buffers.anchor_buffer.max_size == 21
    # a smaller one leaves the configured capacity
    m = modeller(max_size=1000)
    m.init_anchor_buffer_seed()
    assert seen[-1] == 21 and m.args.buffers.anchor_buffer.max_size == 1000
    # the thinning knob, and the capacity from what it seeded
    m = modeller(max_size=5, seed_rows_per_condition=2)
    m.init_anchor_buffer_seed()
    assert seen[-1] == 8 and m.args.buffers.anchor_buffer.max_size == 8
    # the rows before the thermal noise, when they were kept, are the ones seeded
    m = modeller(max_size=14)
    m._prior_dataset_raw = rows                                # a graph batch: handed over whole
    m.init_anchor_buffer_seed()
    assert seen[-1] == rows.num_graphs and '_prior_dataset_raw' not in m.__dict__
    m = modeller(max_size=14)
    m.prior_dataset = full = ConformerBuffer(rows, 'cpu', exclude_keys=BULKY,
                                             y_fn='conformer_energy')
    m.init_anchor_buffer_seed()
    assert seen[-1] == rows.num_graphs and m.prior_dataset is full
    assert m.args.buffers.anchor_buffer.max_size == rows.num_graphs
    # a restored anchor buffer is not re-seeded
    m.init_anchor_buffer_seed()
    assert len(seen) == 5


def test_chunk_bounds_never_leave_one_row_alone():
    from buffer import chunk_bounds

    assert chunk_bounds(0, 4) == []
    assert chunk_bounds(1, 4) == [(0, 1)]
    assert chunk_bounds(8, 4) == [(0, 4), (4, 8)]
    assert chunk_bounds(9, 4) == [(0, 4), (4, 9)]             # the ninth row joins the second chunk
    assert chunk_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert chunk_bounds(5, 1) == [(0, 2), (2, 5)]             # a chunk is at least two rows


# ------------------------------------------------------------------- footprints


def test_store_footprint_counts_what_each_store_holds(conditions):
    from buffer import store_footprint, table_footprint

    rows = _prior_rows(conditions)
    n, k = rows.num_graphs, int(conditions.n_torsions[0])
    full, compact, statics = _pair(rows, _table(conditions), y_fn='conformer_energy')
    f, c = store_footprint(full), store_footprint(compact)
    assert (f['rows'], c['rows']) == (n, n) and c['compact'] and not f['compact']
    # a compact row on the device: its state and its energy (x and y share their storage)
    assert c['device_bytes'] == n * (4 * k + 4)
    # on the host: key and atom count, and the per-row bookkeeping columns
    columns = sum(v.numel() * v.element_size() for name, v in vars(compact).items()
                  if torch.is_tensor(v) and name not in ('x', 'y'))
    assert c['host_bytes'] == 16 * n + columns and f['host_bytes'] == columns
    # a full row carries its condition's graph as well
    assert f['device_bytes'] > 10 * c['device_bytes']
    assert table_footprint(statics) > 0
    # after an admission x and y are their own tensors: twice the state and energy
    compact.add(_prior_rows(conditions, seed=3))
    c2 = store_footprint(compact)
    assert c2['device_bytes'] == 2 * (2 * n) * (4 * k + 4)
