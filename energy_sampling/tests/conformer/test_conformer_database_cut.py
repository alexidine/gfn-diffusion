"""The conformer DATABASE as a source: the set builder's prior_train.pt cut from it
(``build_conformer_set.py --database``) and the reference table's floors taken from it
(``build_conformer_references.py --database``).

What each test would catch, at level `full`, MMFF94, stereo lock 300 (configs/conformer_mk.yaml),
CPU, float64 builds, on a database built here at tiny budgets (8 draws per batch, 2 batches, 60
descent steps) over four QM9 molecules plus water:

  * a prior row that is not the database's: each condition's rows are its lowest ``cap`` stored
    rows, in ascending energy, their states measured into the rung's own member rebuilding the
    stored positions, and their energies the member's re-score of the stored (float32) state;
  * a condition dropped without a trace: one the database lacks is recorded (code, manifest
    count, prior_refusals.tsv) and the file still verifies for the rest;
  * each refusal of ``match_rows`` -- absent, refused in the database, another signature, a row
    outside the member's box, a re-score gap -- firing on its own corruption;
  * a database of other member arguments, or a half-finished one, accepted;
  * a manifest that does not name the database (run hash, shard header digest) and the cap;
  * a prior file the modeller's ``prior_path`` loader cannot take;
  * a reference floor that is not the database's lowest row re-scored through the member, a
    stamp that does not say where the floor came from, and a database-floor table resuming a
    search table's parts (or a search table's resume key changing).

The fitted InternalPrior is untracked: point GFN_CONFORMER_PRIOR at conformer_prior_v2.pt from a
worktree.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider tests/conformer/test_conformer_database_cut.py
"""
import csv
import gzip
import json
import os
import types
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow

HERE = Path(__file__).resolve().parents[2]            # energy_sampling/
SLICE = [(25407, 'CCCO'), (41457, 'CC1CCC1'), (62851, 'CCC(C)O'), (105646, 'C1C2CC1C2'),
         (116340, 'O')]
TINY = ['--batch', '8', '--max-batches', '2', '--steps', '60']
CAP = 2


@pytest.fixture(autouse=True)
def _restore_torch_state():
    """The builders set a float64 default and a thread count PROCESS-WIDE (they are CLIs)."""
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


def _prior():
    p = Path(os.environ.get('GFN_CONFORMER_PRIOR', HERE / 'conformer_prior_v2.pt'))
    if not p.exists():
        pytest.fail(f'fitted InternalPrior not found at {p}: set GFN_CONFORMER_PRIOR')
    return p


def _energy_kw():
    from build_conformer_conditions import energy_kwargs_from_config
    kw, _ = energy_kwargs_from_config(HERE / 'configs' / 'conformer_mk.yaml')
    kw.setdefault('level', 'full')
    kw.setdefault('force_field', 'mmff')
    return kw


def _set(src, out, db, *extra):
    import build_conformer_set as bcs
    bcs.main(['--out-dir', str(out), '--source', str(src), '--n-train', 'all',
              '--n-heldout', 'all', '--no-encoder', '--heldout-permille', '0',
              '--database', str(db), '--rows-per-condition-cap', str(CAP), *map(str, extra)])
    with open(out / 'manifest.json', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope='module')
def cut(tmp_path_factory):
    import build_conformer_database as db
    d = tmp_path_factory.mktemp('dbcut')
    src = d / 'slice.tsv.gz'
    with gzip.open(src, 'wt', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['dataset_index', 'smiles'])
        w.writerows(SLICE)
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    try:
        db.main(['build', '--source', str(src), '--prior', str(_prior()),
                 '--out-dir', str(d / 'db'), *TINY])
        man = _set(src, d / 'set', d / 'db')
    finally:
        torch.set_default_dtype(dtype)
        torch.set_num_threads(threads)
    (shard,) = sorted((d / 'db').glob('shard_*.pt'))
    blob = torch.load(shard, weights_only=False, map_location='cpu')
    recs = {c['identifier']: c for r in blob['keys'].values() for c in r['conditions']}
    return types.SimpleNamespace(dir=d, src=src, db=d / 'db', set=d / 'set', man=man,
                                 blob=blob, recs=recs)


def _members(idents):
    from build_conformer_conditions import build_member
    kw = _energy_kw()
    return {i: build_member(i, i, kw, carrier=True) for i in idents}


def _prior_rows(path):
    b = torch.load(path, weights_only=False, map_location='cpu')['equalized_prior']
    rows = {}
    for j, s in enumerate(b.identifier):
        rows.setdefault(s, []).append(j)
    return b, rows


# ------------------------------------------------------------------ the cut


def test_rows_are_the_lowest_database_rows_measured_into_the_member(cut):
    from energies.conformer_carrier import CarrierLayout
    from energies.conformer_data import bake_energies
    b, rows = _prior_rows(cut.set / 'prior_train.pt')
    idents = [m['identifier'] for m in cut.man['members']['train']]
    assert idents and set(rows) == set(idents)            # every condition of this slice has rows
    members = _members(idents)
    layout = CarrierLayout({i: m.energy for i, m in members.items()})
    x = torch.as_tensor(b.torsion_state)
    e = torch.as_tensor(b.conformer_energy).reshape(-1)
    for ident, js in rows.items():
        rec, en = cut.recs[ident], members[ident].energy
        want = rec['basins']['energy'][:CAP]
        assert len(js) == min(CAP, len(rec['basins']['energy']))
        # lowest first, and the stored energy is the member's re-score of the stored state
        np.testing.assert_allclose(e[js].double().numpy(), want, rtol=0, atol=1e-5)
        xs = layout.from_carrier(ident, x[js].double())
        with torch.no_grad():
            assert torch.equal(bake_energies(en, xs).to(e.dtype), e[js])
            pos = en.build_positions(xs).reshape(len(js), -1, 3).numpy()
        # the state rebuilds the stored positions (placement order), to float32 rounding
        assert np.abs(pos - rec['basins']['pos'][:CAP]).max() < 1e-4


def test_manifest_names_the_database_and_the_cap(cut):
    import build_conformer_database as db
    p = cut.man['prior']
    assert p['source'] == 'database' and p['rows_per_condition_cap'] == CAP
    assert p['database']['run_hash'] == cut.blob['run_hash']
    assert p['database']['header_hashes_sha256'] == \
        db.database_identity([cut.blob['header_hash']])['header_hashes_sha256']
    n_train = len(cut.man['members']['train'])
    assert p['conditions_with_rows'] + p['conditions_refused'] == n_train
    assert p['conditions_refused'] == 0 and p['refused'] == {}
    assert p['rows'] == sum(min(CAP, len(cut.recs[m['identifier']]['basins']['energy']))
                            for m in cut.man['members']['train'])
    assert p['max_rescore_gap_kcal'] <= 1e-6 and p['reference_differs_from_database'] == 0
    assert cut.man['seeds']['prior_seed'] is None
    assert (cut.set / 'prior_refusals.tsv').exists()
    assert 'prior_train.pt' in cut.man['artifacts']


def test_a_condition_the_database_lacks_is_recorded_not_skipped(cut, tmp_path, monkeypatch):
    import build_conformer_database as db
    real = db.read_conditions
    gone = sorted(cut.recs)[0]

    def lacking(path, identifiers=None, **kw):
        info, recs = real(path, identifiers, **kw)
        recs.pop(gone, None)
        return info, recs

    monkeypatch.setattr(db, 'read_conditions', lacking)
    man = _set(cut.src, tmp_path / 'set', cut.db)
    assert man['prior']['refused'] == {'db_absent': 1}
    assert man['prior']['conditions_refused'] == 1
    with open(tmp_path / 'set' / 'prior_refusals.tsv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    assert [(r['identifier'], r['reason_code']) for r in rows] == [(gone, 'db_absent')]
    _, prior_rows = _prior_rows(tmp_path / 'set' / 'prior_train.pt')
    assert gone not in prior_rows and prior_rows          # the rest verified and written


def test_match_rows_refuses_each_corruption(cut):
    import copy

    import build_conformer_database as db
    ident = next(i for i, c in cut.recs.items() if c['refusal'] is None)
    en = _members([ident])[ident].energy
    rec = cut.recs[ident]
    x, stored, info = db.match_rows(en, ident, rec, CAP, 1e-6)
    assert x is not None and info['code'] is None and info['rescore_gap'] <= 1e-6

    def code(r, tol=1e-6):
        return db.match_rows(en, ident, r, CAP, tol)[2]['code']

    assert code(None) == 'db_absent'
    r = copy.deepcopy(rec)
    r['refusal'] = {'code': 'ring_shapes', 'message': 'x'}
    assert code(r) == 'db_refused'
    r = copy.deepcopy(rec)
    r['signature'] = '0' * 16
    assert code(r) == 'db_signature'
    r = copy.deepcopy(rec)
    r['basins']['energy'] = r['basins']['energy'] + 1e-3
    assert code(r) == 'db_rescore'
    # the last placed atom thrown out along its bond: its r column leaves the box
    r = copy.deepcopy(rec)
    p = r['basins']['pos'][0]
    j = int(en.spec.n_atoms) - 1
    anchor = p[int(np.asarray(en.spec.ref_c)[j])]
    p[j] = anchor + 3.0 * (p[j] - anchor)
    assert code(r, tol=1e9) == 'db_outside_box'


def test_database_and_prior_draws_are_exclusive(cut, tmp_path):
    with pytest.raises(SystemExit, match='pass one'):
        _set(cut.src, tmp_path / 's', cut.db, '--prior-rows-per-condition', 4)


def test_a_database_of_other_member_arguments_is_refused_before_the_walk(cut, tmp_path,
                                                                         monkeypatch):
    import build_conformer_database as db
    real = db.first_header

    def other(path):
        h = real(path)
        h['energy_kwargs'] = dict(h['energy_kwargs'], energy_clip=250.0)
        return h

    monkeypatch.setattr(db, 'first_header', other)
    with pytest.raises(SystemExit, match='energy_clip'):
        _set(cut.src, tmp_path / 's', cut.db)


def test_an_unfinished_database_is_refused(cut, tmp_path):
    import build_conformer_database as db
    (shard,) = sorted(cut.db.glob('shard_*.pt'))
    d = tmp_path / 'db'
    d.mkdir()
    blob = dict(cut.blob, complete=False)
    torch.save(blob, d / shard.name)
    with pytest.raises(SystemExit, match='incomplete'):
        db.read_conditions(d, log=lambda *a: None)
    blob = dict(cut.blob, run_hash='another run')
    torch.save(blob, d / shard.name)
    torch.save(cut.blob, d / 'shard_0001_of_0001.pt')
    with pytest.raises(SystemExit, match='2 runs'):
        db.read_conditions(d, identifiers=[], log=lambda *a: None)


def test_the_prior_file_loads_through_the_modellers_prior_path_branch(cut):
    from buffer import ConformerBuffer
    from conformer_modeller import ConformerModeller
    torch.set_default_dtype(torch.float32)
    m = ConformerModeller.__new__(ConformerModeller)
    m.args = types.SimpleNamespace(prior_path=str(cut.set / 'prior_train.pt'),
                                   buffer_device='cpu')
    k = int(cut.man['layout']['K'])
    m.energy_function = types.SimpleNamespace(stereo_coeff=300.0, dtype=torch.float32, ndim=k)
    m.init_prior_dataset()
    assert isinstance(m.prior_dataset, ConformerBuffer)
    assert len(m.prior_dataset) == cut.man['prior']['rows']
    assert m.prior_dataset.x.shape == (cut.man['prior']['rows'], k)
    assert m.prior_dataset.x.dtype == torch.float32


# ------------------------------------------------------------------ reference floors


def _refs(cut, out, **kw):
    import build_conformer_references as bcr
    return bcr.build_references(cut.set / 'conditions_train.pt', out, _energy_kw(),
                                workers=0, log=lambda *a: None, **kw)


def test_reference_floors_are_the_database_lowest_rows(cut, tmp_path):
    import build_conformer_references as bcr
    res = _refs(cut, tmp_path / 'r.pt', database=cut.db, database_tol=1e-6)
    assert not res['failures']
    st = res['stamp']
    assert st['search'] is None and st['floor']['source'] == 'database'
    assert st['floor']['database']['run_hash'] == cut.blob['run_hash']
    for ident, e in res['entries'].items():
        assert e['floor_source'] == 'database' and e['e_min_start_kind'] == 'database'
        assert abs(e['e_min'] - cut.recs[ident]['e_min']) <= 1e-6
        assert e['e_min_database'] == cut.recs[ident]['e_min']
        assert 'combos' in e['basin_ref'] or 'skipped' in e['basin_ref']
    t = bcr.load_references(tmp_path / 'r.pt', conditions_path=cut.set / 'conditions_train.pt',
                            energy_kwargs=_energy_kw())
    t.verify({i: m.energy for i, m in _members(list(res['entries'])).items()}, tol=1e-9)


def test_a_condition_without_a_database_floor_is_a_recorded_failure(cut, tmp_path,
                                                                    monkeypatch):
    import build_conformer_database as db
    import build_conformer_references as bcr
    real = db.read_conditions
    gone = sorted(cut.recs)[0]

    def lacking(path, identifiers=None, **kw):
        info, recs = real(path, identifiers, **kw)
        recs.pop(gone, None)
        return info, recs

    monkeypatch.setattr(db, 'read_conditions', lacking)
    res = _refs(cut, tmp_path / 'r.pt', database=cut.db, database_tol=1e-6)
    assert list(res['failures']) == [gone] and 'db_absent' in res['failures'][gone]
    with pytest.raises(bcr.IncompleteReferencesError):
        bcr.load_references(tmp_path / 'r.pt', conditions_path=cut.set / 'conditions_train.pt',
                            energy_kwargs=_energy_kw())


def test_the_floor_source_keys_the_resume():
    import build_conformer_references as bcr
    base = {'format': bcr.FORMAT, 'conditions': {'sha256': 'c'}, 'energy': {}, 'search': {},
            'basin': {}, 'internal_prior': None}
    search = dict(base, floor={'source': 'search'})
    dbf = dict(base, search=None, floor={'source': 'database', 'rescore_tol_kcal': 1e-6,
                                         'database': {'run_hash': 'r',
                                                      'header_hashes_sha256': 'h'}})
    # a search table keys exactly as one written before floors had a source
    assert bcr._resume_key(search) == bcr._resume_key(base)
    assert bcr._resume_key(dbf) != bcr._resume_key(search)
    other = dict(dbf, floor=dict(dbf['floor'], database={'run_hash': 'r2',
                                                          'header_hashes_sha256': 'h'}))
    assert bcr._resume_key(other) != bcr._resume_key(dbf)
