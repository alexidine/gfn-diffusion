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
