"""The offline conformer database builder (build_conformer_database.py) and its cluster packaging
(configs/conformer_db_sep29/make.py).

What each test would catch, at level `full`, MMFF94, stereo lock 300 (configs/conformer_mk.yaml),
CPU, float64, tiny budgets (8 draws per batch, 2 batches, 60 descent steps):

  * a universe or a condition that is not the set builder's: under both builders' defaults (no
    encoder pool, owner decision 2026-09-29) the database's keys, sides, split hashes and
    conditions equal build_conformer_set.py's split.tsv and molecules.tsv on the same source;
  * a file missing a key, a refusal without its code, counts that do not close, rows outside
    the window, a signature other than the run's;
  * nondeterminism: two builds must give identical arrays;
  * a resume that recomputes held keys, differs from one build, or accepts a changed argument;
  * a stored row that is not what it says: its positions, measured into a FRESH member of the
    condition, must re-score to the stored energy and carry the condition's stereo at every
    pinned element with no lock element on the wrong side -- which also pins the atom order
    (placement) the positions are stored in;
  * a chiral pick without its mirror;
  * make.py letting a drive-letter path into the sbatch.

DATA. ``SLICE`` is five rows of QM9 (qm9_dataset.pt, by file index, as in
configs/conformer_db_sep29/qm9_index.tsv.gz): acyclic CCCO, single-ring CC1CCC1, bridged
bicyclo[1.1.1]pentane, chiral CCC(C)O and water (refused, lt4_atoms). The fitted InternalPrior is
untracked: point GFN_CONFORMER_PRIOR at conformer_prior_v2.pt from a worktree.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider tests/conformer/test_build_conformer_database.py
"""
import csv
import gzip
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow

HERE = Path(__file__).resolve().parents[2]            # energy_sampling/
SLICE = [(25407, 'CCCO'), (41457, 'CC1CCC1'), (62851, 'CCC(C)O'), (105646, 'C1C2CC1C2'),
         (116340, 'O')]
TINY = ['--batch', '8', '--max-batches', '2', '--steps', '60']
STRATUM = {'CCCO': 'acyclic', 'CC1CCC1': 'single_ring', 'CCC(C)O': 'acyclic',
           'C1C2CC1C2': 'fused', 'O': 'acyclic'}


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


def _write_slice(path, rows=SLICE):
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['dataset_index', 'smiles'])
        w.writerows(rows)
    return path


def _run(src, out, *extra):
    import build_conformer_database as db
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    try:
        return db.main(['build', '--source', str(src), '--prior', str(_prior()),
                        '--out-dir', str(out), *TINY, *map(str, extra)])
    finally:
        torch.set_default_dtype(dtype)
        torch.set_num_threads(threads)


def _load(out):
    (path,) = sorted(Path(out).glob('shard_*_of_*.pt'))
    return torch.load(path, weights_only=False, map_location='cpu')


def _arrays(blob):
    return {c['identifier']: (c['basins']['pos'], c['basins']['energy'], c['basins']['hits'])
            for r in blob['keys'].values() for c in r['conditions']}


def _assert_same_arrays(a, b):
    a, b = _arrays(a), _arrays(b)
    assert list(a) == list(b)
    for ident in a:
        for x, y in zip(a[ident], b[ident]):
            assert x.dtype == y.dtype and np.array_equal(x, y), ident


def _energy_kw():
    from build_conformer_conditions import energy_kwargs_from_config
    kw, _ = energy_kwargs_from_config(HERE / 'configs' / 'conformer_mk.yaml')
    kw.setdefault('level', 'full')
    kw.setdefault('force_field', 'mmff')
    return kw


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp('db')
    src = _write_slice(d / 'slice.tsv.gz')
    _run(src, d / 'a')
    return src, d / 'a', _load(d / 'a')


# ------------------------------------------------------------------ small units


def test_still_falling_reads_the_tail_of_the_best_seen():
    from build_conformer_database import still_falling
    steps = np.arange(100, dtype=float)
    trace = np.stack([100.0 - 0.05 * steps,              # falls 0.05 per step to the end
                      np.where(steps < 50, 100.0 - steps, 50.0),   # flat for the last 50
                      np.full(100, np.inf)], axis=1)
    assert still_falling(trace, 20, 0.01).tolist() == [True, False, False]
    with pytest.raises(ValueError):
        still_falling(trace[:20], 20, 0.01)


def test_stop_waits_for_a_streak_of_empty_batches():
    """stop_after(new in-window basins per batch, min_batches, stop_streak)."""
    from build_conformer_database import stop_after
    assert not stop_after([3], 2, 2)
    assert not stop_after([3, 0], 2, 2)            # one empty batch is not a streak of two
    assert stop_after([3, 0, 0], 2, 2)
    assert not stop_after([3, 0, 1, 0], 2, 2)      # a batch that founds a basin resets it
    assert stop_after([3, 0, 1, 0, 0], 2, 2)
    assert stop_after([3, 0], 2, 1)                # streak 1: the one-batch rule
    assert not stop_after([0], 2, 1)               # --min-batches holds first
    assert not stop_after([5, 0, 0], 4, 2) and stop_after([5, 0, 0, 0], 4, 2)


def test_strata():
    from build_conformer_database import stratum_of
    for smi, s in STRATUM.items():
        assert stratum_of(smi) == s, smi
    assert stratum_of('C1CCC2CCCCC2C1') == 'fused' and stratum_of('C1CC1C1CC1') == 'single_ring'
    assert stratum_of('C1CCC11CC1') == 'fused'          # spiro


def test_read_source_reads_gzip(tmp_path):
    import build_conformer_set as bcs
    for name, delim in (('s.tsv', '\t'), ('s.csv', ',')):
        plain = tmp_path / name
        with open(plain, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter=delim)
            w.writerow(['dataset_index', 'smiles'])
            w.writerows(SLICE)
        gz = tmp_path / f'{name}.gz'
        gz.write_bytes(gzip.compress(plain.read_bytes()))
        assert bcs.read_source(plain) == bcs.read_source(gz) == SLICE


def test_pool_need_zero_is_no_pool():
    import build_conformer_set as bcs
    for rows in ([(0, 'C'), (1, 'CC'), (2, 'C')], SLICE):
        assert bcs.pool_rows(rows, 0) == (set(), 0, bcs.NO_POOL_RULE)
    sides, rej, n_pool = bcs.plan_split([(0, 'CCO'), (1, 'OCC'), (2, 'CCN')], index_min=0,
                                        index_max=None, pool_idx=set(), salt='t', permille=0)
    assert [e.index for e in sides['train']] and n_pool == 0
    assert {e.index for e in sides['train']} == {0, 2}     # index 1 duplicates index 0's key
    assert [r['reason_code'] for r in rej] == ['duplicate_key']


# ------------------------------------------------------------------ the universe


def test_universe_and_conditions_are_the_set_builders(built, tmp_path):
    """Both builders at their defaults (no encoder pool) on one source: the database holds
    every split.tsv key with the same side and hash, and every condition molecules.tsv
    writes, under the same identifier, isomer rank and mirror."""
    import json

    import build_conformer_set as bcs
    src, _, blob = built
    out = tmp_path / 'set'
    try:
        bcs.main(['--out-dir', str(out), '--source', str(src), '--n-train', 'all',
                  '--n-heldout', 'all', '--no-encoder'])
    except SystemExit:
        pass                                   # no held-out molecule built is not a failure
    rd = lambda n: list(csv.DictReader(open(out / n, encoding='utf-8'), delimiter='\t'))
    split = {r['key']: r for r in rd('split.tsv')}
    keys = blob['keys']
    assert set(split) == set(keys) == {bcs.constitution_key(s) for _, s in SLICE}
    for k, r in keys.items():
        assert (r['side'], r['hash'], str(r['dataset_index'])) == \
               (split[k]['side'], split[k]['hash'], split[k]['dataset_index'])
    man = json.load(open(out / 'manifest.json', encoding='utf-8'))
    assert man['encoder_pool']['need'] == 0 and man['encoder_pool']['end'] == 0
    assert man['encoder_pool']['selection'] == bcs.NO_POOL_RULE
    hdr = blob['header']
    assert hdr['set_builder']['pool_need'] == 0 and hdr['args']['pool_need'] == 0
    assert hdr['set_builder']['pool_selection'] == bcs.NO_POOL_RULE
    conds = {c['identifier']: (r, c) for r in keys.values() for c in r['conditions']}
    written = rd('molecules.tsv')
    assert written
    for m in written:
        r, c = conds[m['identifier']]
        assert (r['key'], str(r['dataset_index']), r['side']) == \
               (m['key'], m['dataset_index'], m['split'])
        assert (str(c['isomer_rank']), c['mirror_of']) == (m['isomer_rank'], m['mirror'])
    # a database condition the set did not write is a held-out molecule it dropped for width
    dropped = {r['key'] for r in rd('rejections.tsv')
               if r['reason_code'].startswith('heldout_')}
    for ident, (r, _) in conds.items():
        if ident not in {m['identifier'] for m in written}:
            assert r['side'] == 'heldout' and r['key'] in dropped, ident


# ------------------------------------------------------------------ the file


def test_file_structure(built):
    from conformer_modeller import ConformerModeller
    import build_conformer_database as db
    _, out, blob = built
    assert blob['format'] == db.FORMAT and blob['complete']
    assert set(blob['assigned']) == set(blob['keys']) and len(blob['keys']) == len(SLICE)
    for key, r in blob['keys'].items():
        assert r['stratum'] == STRATUM[key] and r['side'] in ('train', 'heldout')
        if r['refusal'] is not None:
            assert not r['conditions']
            continue
        for c in r['conditions']:
            n = c['n_atoms']
            assert c['z'].shape == (n,) and c['perm'].shape == (n,)
            assert c['ref_pos'].shape == (n, 3) and c['ref_pos'].dtype == np.float64
            assert c['refusal'] is None, c['refusal']
            k = c['counts']
            assert k['starts'] == 16 and k['batches'] == 2 and c['stop'] in ('converged', 'cap')
            # the loop stops exactly where stop_after first says so, never later
            new = k['new_in_window']
            assert (c['stop'] == 'converged') == db.stop_after(new, 2, 2)
            assert not any(db.stop_after(new[:j], 2, 2) for j in range(1, len(new)))
            assert k['starts'] == k['valid'] + sum(k['excluded'].values())
            b = c['basins']
            assert b['pos'].shape == (len(b['energy']), n, 3) and b['pos'].dtype == np.float64
            assert len(b['energy']) == k['basins_window'] <= k['basins_total']
            assert np.all(np.diff(b['energy']) >= 0) and (b['hits'] >= 1).all()
            assert c['e_min'] == b['energy'][0]
            assert (b['energy'] - c['e_min'] <= c['window_kcal']).all()
            assert c['window_kcal'] == 10.0 * c['kT'] and c['kT'] == 1.0
            assert k['rescore_gap'] <= db.RESCORE_TOL
            assert set(c['seconds']) >= {'ring_shapes', 'draw', 'relax', 'screen', 'cluster'}
    assert blob['header']['recipe']['signature'] == ConformerModeller._MEMBER_SIGNATURE
    assert blob['header']['args']['stop_streak'] == 2          # in the resume hash
    assert 'stop_streak' in db.hashed_view(blob['header'])['args']
    out_d = db.summarize(out, log=lambda *a: None)
    assert out_d['mirror_pairs'] == 1 and out_d['key_refusals'] == {'acyclic/lt4_atoms': 1}
    assert {s: out_d['strata'][s]['keys'] for s in db.STRATA} == \
           {'acyclic': 3, 'single_ring': 1, 'fused': 1}


def test_unbuildable_key_is_a_recorded_refusal(built):
    _, _, blob = built
    r = blob['keys']['O']
    assert r['refusal']['code'] == 'lt4_atoms' and r['conditions'] == []
    assert r['dataset_index'] == 116340 and r['side'] in ('train', 'heldout')


def test_chiral_pick_brings_its_mirror(built):
    from build_conformer_conditions import smiles_identity
    from models.encoder_probe import mirror_smiles
    _, _, blob = built
    a, b = blob['keys']['CCC(C)O']['conditions']
    assert (a['isomer_rank'], b['isomer_rank']) == (0, 1)
    assert a['mirror_of'] == b['identifier'] and b['mirror_of'] == a['identifier']
    assert smiles_identity(mirror_smiles(a['identifier'])) == smiles_identity(b['identifier'])
    assert not a['achiral'] and len(a['basins']['energy']) and len(b['basins']['energy'])
    assert blob['keys']['CC1CCC1']['conditions'][0]['achiral']


def test_stored_rows_rescore_in_a_fresh_member_and_keep_the_condition_stereo(built):
    import build_conformer_references as bcr
    from build_conformer_conditions import build_member
    from build_conformer_database import lock_wrong_side
    from conformer_modeller import ConformerModeller
    from energies.conformer_data import bake_energies
    from mxtaltools.conformers.builder import collate, measure
    _, _, blob = built
    kw = _energy_kw()
    n_rows = n_pinned = 0
    for r in blob['keys'].values():
        for c in r['conditions']:
            en = build_member(c['identifier'], c['identifier'], kw, carrier=True).energy
            assert np.array_equal(np.asarray(en.spec.z), c['z'])
            assert np.array_equal(np.asarray(en.spec.perm), c['perm'])
            assert np.abs(en.ref_pos.numpy() - c['ref_pos']).max() < 1e-9
            assert c['signature'] == ConformerModeller._member_signature(c['identifier'], en)
            tree = collate([en.spec], device=en.device)
            xs = []
            for p in c['basins']['pos']:                   # PLACEMENT order, measured as is
                rr, th, ph = measure(tree, torch.as_tensor(p), dummy_frame=en._dummy_t)
                xs.append(en.state_from_dof(rr.reshape(1, -1), th.reshape(1, -1),
                                            ph.reshape(1, -1)))
            x = torch.cat(xs)
            with torch.no_grad():
                e = bake_energies(en, x).numpy()
            np.testing.assert_allclose(e, c['basins']['energy'], rtol=0, atol=1e-6)
            assert not lock_wrong_side(en, x).any()
            pin = bcr.condition_stereo(en)
            if bcr._pinned(pin):
                n_pinned += 1
                assert all(tuple(lab) == tuple(pin['target'])
                           for lab in bcr.stereo_labels(en, x, pin))
            n_rows += len(x)
    assert n_rows >= 5 and n_pinned == 2


# ------------------------------------------------------------------ determinism and resume


def test_two_builds_give_identical_arrays(built, tmp_path):
    src, _, blob = built
    _run(src, tmp_path / 'b')
    _assert_same_arrays(blob, _load(tmp_path / 'b'))


def test_resume_skips_held_keys_and_refuses_a_changed_argument(built, tmp_path, monkeypatch):
    import build_conformer_set as bcs
    src, _, blob = built
    out = tmp_path / 'c'
    _run(src, out, '--max-molecules', 2)
    first = _load(out)
    assert len(first['keys']) == 2 and first['assigned'] == blob['assigned'][:2]
    calls = []
    real = bcs.build_molecule

    def counting(entry, *a, **k):
        calls.append(entry.key)
        return real(entry, *a, **k)

    monkeypatch.setattr(bcs, 'build_molecule', counting)
    _run(src, out)                                     # --max-molecules changes no row
    assert sorted(calls) == sorted(blob['assigned'][2:])
    resumed = _load(out)
    assert resumed['complete'] and resumed['header_hash'] == blob['header_hash']
    _assert_same_arrays(blob, resumed)
    with pytest.raises(SystemExit, match='steps'):
        _run(src, out, '--steps', 61)
    with pytest.raises(SystemExit, match='stop_streak'):
        _run(src, out, '--stop-streak', 3)


# ------------------------------------------------------------------ packaging


def _make_module():
    path = HERE / 'configs' / 'conformer_db_sep29' / 'make.py'
    spec = importlib.util.spec_from_file_location('conformer_db_sep29_make', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_make_refuses_a_drive_letter_path():
    mk = _make_module()
    values = {'WALL': mk.WALL, 'MEM': mk.MEM, 'LAST': mk.N_SHARDS - 1, 'N_SHARDS': mk.N_SHARDS,
              'WORKDIR': mk.WORKDIR, 'PROJECT_ROOT': mk.PROJECT_ROOT, 'BATTERY': mk.BATTERY,
              'POOL_NEED': mk.POOL_NEED, 'PRIOR_NAME': mk.PRIOR_NAME, 'PRIOR_BYTES': 1,
              'PRIOR_SHA256': 'ab' * 32, 'INDEX_NAME': mk.INDEX_NAME, 'INDEX_SHA256': 'cd' * 32,
              'OUTROOT': mk.OUTROOT, 'PILOT_OUT': mk.PILOT_OUT,
              'PILOT_MOLECULES': mk.PILOT_MOLECULES, 'CONFIG_REL': mk.CONFIG_REL}
    text = mk.render(values)
    assert '#SBATCH --array=0-399' in text and '@@' not in text and '--nv' not in text
    assert '--gres' not in text and 'CUDA_VISIBLE_DEVICES=-1' in text
    for bad in ({'OUTROOT': r'D:\conformer_datasets\qm9'}, {'PILOT_OUT': 'C:/tmp/pilot'},
                {'CONFIG_REL': 'd:/configs/conformer_mk.yaml'}):
        with pytest.raises(SystemExit, match='local path'):
            mk.render({**values, **bad})
    with pytest.raises(SystemExit, match='unfilled'):
        mk.render({k: v for k, v in values.items() if k != 'POOL_NEED'})
    mk.assert_no_local_paths('see https://example.org/x and --time=06:00:00', 'ok')
    sbatch = HERE / 'configs' / 'conformer_db_sep29' / 'submit_db.sbatch'
    if sbatch.exists():
        mk.assert_no_local_paths(sbatch.read_text(encoding='utf-8'), str(sbatch))
