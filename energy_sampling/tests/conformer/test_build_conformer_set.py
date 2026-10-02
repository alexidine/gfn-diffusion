"""The conformer set builder: split, ladder, stereoisomers, refusals, layout, manifest.

What each group pins, all at level `full`, force field `mmff`, CPU, float64 builds:

  * the SPLIT is a function of the row SET (a reordered source gives a byte-identical split
    table), no constitution straddles it, duplicates collapse onto their lowest index, and
    the encoder's pool -- replayed, not assumed -- never enters the universe;
  * the LADDER nests: a smaller rung's conditions are a subset of a larger one's, both sides;
  * STEREO: enumeration sees ring cis/trans AND imine E/Z (neither hydrogen convention alone
    does), gives one name per stereoisomer (``stereo_identity``, not SMILES equality) and
    calls a molecule with no open element fully specified; every built condition is the
    isomer its identifier names, judged without RDKit's SMILES writer; a chiral pick brings
    its mirror up to the per-molecule cap;
  * every per-molecule REFUSAL has a code, whatever raised it, and the molecule accounting
    closes on the 60-row QM9 slice below; a reference whose inferred bond graph differs from
    RDKit's is refused even when that graph stays connected; an embedding failure says
    whether the isomer is known to exist; an isomer-dependent refusal does not decide the
    molecule; a held-out molecule wider than the train maxima, or one whose presence moves a
    training member off the columns the run rebuilds from the training file, is dropped and
    named, and the training file is verified against that training-only layout;
  * nitriles and alkynes are ADMITTED (owner decision): their transverse u/v columns sit in
    the carrier's theta region, so a member's carrier columns are not in member order;
  * a written file whose re-read layout disagrees with its CarrierLayout is refused (K, k and
    pads, the owned columns, the ORDER of the reconstruction map, uncovered columns); the
    prior file is K wide, pads exactly 0, and its stored energies are bit-equal to the
    re-score of its stored states;
  * the old CLI (build_conformer_conditions.py) refuses per molecule instead of aborting.

DATA. ``SLICE`` is QM9 (D:/crystal_datasets/qm9_dataset.pt) by file index: 48 rows drawn with
numpy seed 20260926 from indices >= 25,364, plus curated rows -- two encoder-pool rows (8079,
24057) whose constitutions reappear at 25460 / 60163, an in-universe duplicate pair
(26984 / 51056), water (lt4), a theta-box refusal (25679), a disconnected reference graph
(60167), cages (25604, 26072), and C / CO / N. The InternalPrior and the encoder checkpoint are
untracked: point GFN_CONFORMER_PRIOR and GFN_ENCODER_CKPT at them from a worktree.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider tests/conformer/test_build_conformer_set.py
"""
import csv
import json
import os
import random
import types
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow

HERE = Path(__file__).resolve().parents[2]            # energy_sampling/
POOL_NEED = 25364

SLICE = [
    (8079, 'C#CC1=NC=CN=C1'), (24057, 'FC1=NC=C(N=C1)C#C'), (25460, 'C#CC1=CN=CC=N1'),
    (25604, 'C1OC2C3CC1C2O3'), (25679, 'OC1=C2CCN2C(=O)N1'), (26072, 'OC12C3C1C1C3C21C=O'),
    (26984, 'C#CCC1=NC=CN=C1'), (33732, 'CC1COC2=C1OC=C2'), (35140, 'O1C2C1C1C3C4C2C4N13'),
    (37992, 'CC(=O)N1C2CC(=O)C12'), (39946, 'CCC1OCOCC=C1'), (45037, 'OC1CC2C1OCC2O'),
    (46668, 'CCCC1C2NC2C1=O'), (47056, 'O=CC1=CC(=CN1)C#C'), (51056, 'C#CCC1=CN=CC=N1'),
    (52508, 'N#CC1CCC11CCC1'), (57258, 'CC1(C#C)C(O)C1CO'), (60163, 'FC1=CN=C(C=N1)C#C'),
    (60167, 'CNC1=NC=NC=C1F'), (60175, 'CO'), (60970, 'CC1C2C3NC3C(=O)N12'),
    (65128, 'CC1COC1(C)CC#N'), (67302, 'O=C1CN2C(C3CO3)C12'), (70176, 'OC1(C=O)C2CCCC12'),
    (73326, 'CCC1C=C2CC3C1C23'), (76648, 'O=C(CCC#N)CC#C'), (78979, 'CCC1C2CC1(C)O2'),
    (79185, 'OC1=CN=C(N=C1)C#C'), (80354, 'O=C1C2CC1(N2)C#N'), (80476, 'FC1=NC(=O)NN=N1'),
    (86906, 'NC1=CC=C(F)C(N)=N1'), (87422, 'N=C1CC=CCN1C=O'), (87490, 'CC(C)(C#C)C#CC=O'),
    (90654, 'OCC12OC(=N)C1C2O'), (91408, 'O=C1CNCCOCO1'), (93604, 'N'),
    (95528, 'CN(C)C1(CC1N)C#N'), (95994, 'OCC1(CC1O)C1CC1'), (97082, 'CC1C2CC(=O)N(C)C12'),
    (101677, 'CN1C=CC(=O)C=C1'), (101900, 'NC1=C(NC(=N)O1)C#C'),
    (104059, 'CCC1=CCCC(=O)N1'), (106187, 'NC(=O)C12CC(C1)CC2'), (108315, 'C'),
    (111191, 'C1C2CC=C3CN1C23'), (112366, 'NC1=C(N)N=NO1'), (112406, 'CC1CN1CC(C)=O'),
    (113698, 'O=CC1CC2OC12'), (114806, 'COC1CC2(CO2)C1C'), (116090, 'COC1=NC2C3CN1C23'),
    (116267, 'CCCOC1=CNN=C1'), (116340, 'O'), (116419, 'OC1CC(=O)C(=O)C1'),
    (116991, 'CC1(CO1)C12CCC1O2'), (119916, 'CC1CC23CCCC12O3'), (120515, 'OCC12CCCC1N2'),
    (121238, 'CC1CN1C1(CC1)C#C'), (123788, 'CC1(OC(=N)C1N)C#C'), (125600, 'CCOCCOC1CC1'),
    (125791, 'CN1C2CC(C)(O)C12'), (129619, 'C1NC=NC2=C1C=NO2'),
    (130615, 'CC1CC2(COC2)C=C1'),
]
N_UNIVERSE = sum(1 for i, _ in SLICE if i >= POOL_NEED)          # 60

ENERGY_KW = dict(level='full', force_field='mmff')


@pytest.fixture(autouse=True)
def _restore_torch_state():
    """The builders set a float64 default and a thread count PROCESS-WIDE (they are CLIs);
    restore both so no later test module inherits them."""
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


def _prior_path():
    p = Path(os.environ.get('GFN_CONFORMER_PRIOR', HERE / 'conformer_prior_v2.pt'))
    if not p.exists():
        pytest.fail(f'fitted InternalPrior not found at {p}: this test cannot run without it. '
                    f'Set GFN_CONFORMER_PRIOR to the local conformer_prior_v2.pt.')
    return p


def _encoder_ckpt():
    from models import encoder_cache
    p = Path(os.environ.get('GFN_ENCODER_CKPT', encoder_cache.DEFAULT_CKPT))
    if not p.exists():
        pytest.fail(f'encoder checkpoint not found at {p}: this test cannot run without it. '
                    f'Set GFN_ENCODER_CKPT to models/results/encoder_ckpt/'
                    f'mp+attn+spd_n20000_s0.pt.')
    return p


def _write_slice(path, rows=SLICE):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset_index', 'smiles'])
        w.writerows(rows)
    return path


def _build(tmp_path, name, n_train, n_heldout, *extra, rows=SLICE):
    import build_conformer_set as bcs
    src = _write_slice(tmp_path / f'{name}_slice.csv', rows)
    out = tmp_path / name
    bcs.main(['--out-dir', str(out), '--source', str(src), '--n-train', str(n_train),
              '--n-heldout', str(n_heldout), '--no-encoder', '--pool-need', str(POOL_NEED),
              '--heldout-permille', '200', *map(str, extra)])
    with open(out / 'manifest.json', encoding='utf-8') as f:
        return out, json.load(f)


def _tsv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def _entry(smiles, side='train'):
    import build_conformer_set as bcs
    key = bcs.constitution_key(smiles)
    return bcs.Entry(key, 0, smiles, side, bcs.split_hash(key, 't'))


# ------------------------------------------------------------------ split and pool


def test_split_is_a_function_of_the_row_set(tmp_path):
    import build_conformer_set as bcs
    kw = dict(index_min=POOL_NEED, index_max=None, pool_idx=set(range(POOL_NEED)),
              salt='conformer_set_v1', permille=200)
    sides, rej, n_pool = bcs.plan_split(SLICE, **kw)
    shuffled = list(SLICE)
    random.Random(7).shuffle(shuffled)
    sides2, rej2, _ = bcs.plan_split(shuffled, **kw)
    bcs.write_split_table(tmp_path / 'a.tsv', sides)
    bcs.write_split_table(tmp_path / 'b.tsv', sides2)
    assert (tmp_path / 'a.tsv').read_bytes() == (tmp_path / 'b.tsv').read_bytes()
    assert rej == rej2

    train = {e.key for e in sides['train']}
    held = {e.key for e in sides['heldout']}
    assert train and held and not (train & held)
    assert all(e.index >= POOL_NEED for s in sides.values() for e in s)
    codes = {r['dataset_index']: r['reason_code'] for r in rej}
    # 8079 / 24057 are pool rows: their constitutions reappear at 25460 / 60163
    assert codes[25460] == codes[60163] == 'encoder_pool_key'
    # the Kekule-variant duplicate keeps its LOWEST index
    assert codes[51056] == 'duplicate_key'
    assert any(e.index == 26984 for s in sides.values() for e in s)
    assert n_pool == 2
    assert len(train) + len(held) + len(rej) == N_UNIVERSE


def test_pool_rows_replay_load_qm9():
    import build_conformer_set as bcs
    # a duplicate inside the range pushes the pool's end past `need`, as load_qm9's dedupe does
    rows = [(0, 'C'), (1, 'CC'), (2, 'C'), (3, 'CO'), (4, 'N'), (5, 'CCO')]
    idx, end, rule = bcs.pool_rows(rows, 3)
    assert idx == {0, 1, 3} and end == 4 and rule.startswith('replayed')
    idx, end, rule = bcs.pool_rows(rows[2:], 3)
    assert idx == {0, 1, 2} and end == 3 and rule.startswith('index rule')


def test_universe_reaching_into_the_pool_is_refused(tmp_path):
    import build_conformer_set as bcs
    src = _write_slice(tmp_path / 's.csv')
    with pytest.raises(SystemExit, match='encoder pool'):
        bcs.main(['--out-dir', str(tmp_path / 'o'), '--source', str(src), '--n-train', '2',
                  '--n-heldout', '0', '--no-encoder', '--pool-need', str(POOL_NEED),
                  '--index-min', '100'])


# ------------------------------------------------------------------ stereo


@pytest.mark.parametrize('smiles, n', [
    ('CCC(C)O', 2), ('CC(O)C(C)N', 4), ('CC(O)C(C)O', 3),
    # ring cis/trans: all-explicit-H enumeration finds ONE here
    ('OC1CCC(O)CC1', 2), ('CC1CC(C)C1', 2),
    # aziridine: C centre kept, the N invertomers collapse
    ('CC1CN1C', 2),
    # =NH imine E/Z: all-implicit-H enumeration finds ONE here
    ('CC=N', 2), ('CC=NO', 2), ('CCO', 1),
    # no stereo element at all: bicyclo[1.1.1]pentane bridgeheads (QM9 has these whole),
    # and the 1,3-disubstituted one whose two "cis/trans" strings are one molecule
    ('OC12CC(C1)C2', 1), ('CC12CC(C1)C2', 1), ('OCC12CC(O)(C1)C2', 1),
    # bicyclo[6.1.0]nonan-9-ol: a trans-fused enantiomer pair whose apex is NOT a
    # stereocentre, and a cis pair whose apex is pseudo-asymmetric (r/s)
    ('OC1C2CCCCCC12', 4),
    # QM9 40520: RDKit writes one of its isomers as two different canonical SMILES
    ('C1C2CC1CC1(CO1)C2', 2),
])
def test_stereoisomer_enumeration(smiles, n):
    from build_conformer_conditions import (enumerate_stereoisomers, smiles_identity,
                                            unspecified_stereo)
    from build_conformer_set import constitution_key
    from models.encoder_probe import parent_skeleton
    isos = enumerate_stereoisomers(smiles)
    assert len(isos) == n, isos
    # one split key per molecule, and on raw input the key IS parent_skeleton
    assert {constitution_key(i) for i in isos} == {parent_skeleton(smiles)}
    # "fully specified" is what enumeration says: every isomer names exactly itself
    assert all(not unspecified_stereo(i) for i in isos)
    assert all(enumerate_stereoisomers(i) == [i] for i in isos)
    # one name per stereoisomer
    assert len({smiles_identity(i) for i in isos}) == n
    if smiles == 'CC=N':
        assert all('[H]' in i for i in isos)
    if smiles == 'OCC12CC(O)(C1)C2':
        assert isos == ['OCC12CC(O)(C1)C2']         # named without the spurious tags


def test_build_member_holds_stereo_to_the_identifier():
    from build_conformer_conditions import MemberRefused, build_member, realised_isomer
    for smiles, code in (('CCC(C)O', 'stereo_unspecified'),
                         # the reference embeds as the OTHER bridgehead configuration
                         ('C[C@H]1[C@H](C=O)[C@]2(C)CN12', 'stereo_verify_failed'),
                         ('C[C@@H]1C[N@]1C', 'stereo_n_tagged')):
        with pytest.raises(MemberRefused) as exc:
            build_member(smiles, smiles, ENERGY_KW)
        assert exc.value.code == code, exc.value.message
    for smiles in ('CC[C@H](C)O', 'C/C=N/O', '[H]/N=C\\C', 'O[C@H]1CC[C@@H](O)CC1',
                   # no open element; the 3D reference carries bridgehead tags regardless
                   'OC12CC(C1)C2'):
        mb = build_member(smiles, smiles, ENERGY_KW)
        assert realised_isomer(mb.energy.mol) == smiles


def test_stereo_check_does_not_trust_the_smiles_writer():
    """QM9 32293: re-writing this identifier yields a SMILES of ANOTHER isomer (different
    InChI t-layer), so a check against the re-canonicalised string refused a reference that
    reproduces the identifier exactly."""
    from build_conformer_conditions import build_member, smiles_identity
    from rdkit import Chem
    ident = 'C1[C@@H]2[C@H]3C[C@H]4[C@@H]2[C@@H]1[C@@H]34'
    rewritten = Chem.MolToSmiles(Chem.MolFromSmiles(ident))
    assert rewritten != ident and smiles_identity(rewritten) != smiles_identity(ident)
    build_member(ident, ident, ENERGY_KW)                   # does not raise


def test_embed_failure_says_whether_the_isomer_exists():
    """QM9 105841: ETKDG embeds the untagged constitution, always as ONE isomer, and cannot
    embed that isomer from its own tagged SMILES; the other seven do not appear untagged."""
    from build_conformer_conditions import MemberRefused, build_member, enumerate_stereoisomers
    real = 'C[C@H]1[C@H]2[C@@H]3CC[C@H]2[C@H]13'
    names = enumerate_stereoisomers('CC1C2C3CCC2C13')
    assert real in names and len(names) == 8
    for smiles, code in ((real, 'embed_failed_realisable'),
                         (next(n for n in names if n != real), 'embed_failed')):
        with pytest.raises(MemberRefused) as exc:
            build_member(smiles, smiles, ENERGY_KW, carrier=True)
        assert exc.value.code == code, exc.value.message


def test_every_member_step_is_guarded(monkeypatch):
    import build_conformer_conditions as bcc

    def boom(_):
        raise KeyError('not a MemberRefused')
    monkeypatch.setattr(bcc, '_bond_graph_mismatch', boom)
    with pytest.raises(bcc.MemberRefused) as exc:
        bcc.build_member('CCO', 'CCO', ENERGY_KW)
    assert exc.value.code == 'other' and 'not a MemberRefused' in exc.value.message


def test_stereo_walk_mirror_and_cap():
    import build_conformer_set as bcs
    walk = dict(bundle=None, stereo_salt='')
    b, _ = bcs.build_molecule(_entry('CCC(C)O'), ENERGY_KW, cap=2, **walk)
    ids = [m.identifier for m in b.members]
    assert sorted(ids) == ['CC[C@@H](C)O', 'CC[C@H](C)O']       # a pick and its mirror
    b, _ = bcs.build_molecule(_entry('CCC(C)O'), ENERGY_KW, cap=1, **walk)
    assert len(b.members) == 1
    b, _ = bcs.build_molecule(_entry('CC(O)C(C)N'), ENERGY_KW, cap=None, **walk)
    assert len(b.members) == 4
    # achiral E/Z pair: no mirror to add, so the cap is filled by the other isomer
    b, _ = bcs.build_molecule(_entry('CC=NO'), ENERGY_KW, cap=2, **walk)
    assert sorted(m.identifier for m in b.members) == ['C/C=N/O', 'C/C=N\\O']


def test_one_isomer_is_one_condition():
    """QM9 40520: the enumerated names are one per stereoisomer, the other string RDKit
    writes for the same isomer is recorded as its duplicate, and the built members are
    pairwise distinct stereoisomers."""
    import build_conformer_set as bcs
    from build_conformer_conditions import smiles_identity
    entry = _entry('C1C2CC1CC1(CO1)C2')
    b, rej = bcs.build_molecule(entry, ENERGY_KW, cap=None, bundle=None, stereo_salt='')
    ids = [m.identifier for m in b.members]
    assert len({smiles_identity(i) for i in ids}) == len(ids) >= 1
    dup = [r['identifier'] for r in rej if r['reason_code'] == 'duplicate_stereoisomer']
    names, _, _ = bcs.stereo_plan(entry.key)
    assert len(dup) == 1 and dup[0] not in names
    assert smiles_identity(dup[0]) in {smiles_identity(n) for n in names}


def test_isomer_dependent_refusal_does_not_decide_the_molecule():
    """QM9 132157: `closure_encoding` depends on atom numbering, which differs between the
    canonical SMILES of two stereoisomers, so some isomers fail it and others build. Before
    it was dropped from CONSTITUTION_CODES, the stereo salt decided whether the molecule was
    kept (refused under '' and 'a', built under 'c')."""
    import build_conformer_set as bcs
    assert 'closure_encoding' not in bcs.CONSTITUTION_CODES
    # the same reason, by construction rather than by a measured case: what is still held
    # at `full` since the dummy frame is a frame the TREE cannot give a dummy (see the
    # CONSTITUTION_CODES comment). Every alkyne of the 3000-molecule QM9 census now charts
    # complete at `full`, so no measured molecule reaches it either way
    assert 'incomplete_chart_full' not in bcs.CONSTITUTION_CODES
    for salt in ('', 'a'):
        b, rej = bcs.build_molecule(_entry('CCN1C2C3COC3C21'), ENERGY_KW, cap=2, bundle=None,
                                    stereo_salt=salt)
        assert b is not None and len(b.members) == 2, [r['reason_code'] for r in rej]


def test_constitution_refusal_stops_the_walk():
    """A constitution-level refusal on the first isomer ends the walk: one isomer row, not
    one per stereoisomer. The molecule was an alkyne (C#CC1OC1C(F)(F)F, `incomplete_chart_full`)
    until the transverse pair and the dummy frame admitted alkynes; it now builds, and that
    code left CONSTITUTION_CODES (it became a property of the tree). A selenide with a
    stereocentre is refused by MMFF typing, a function of the bond graph, for both
    enantiomers -- and the pick's mirror is exactly what the walk would try next."""
    import build_conformer_set as bcs
    entry = _entry('CC(O)[Se]C')
    assert len(bcs.stereo_plan(entry.key)[0]) == 2          # there IS a second isomer to skip
    b, rej = bcs.build_molecule(entry, ENERGY_KW, cap=2, bundle=None, stereo_salt='')
    assert b is None
    iso = [r for r in rej if r['level'] == 'isomer']
    mol = [r for r in rej if r['level'] == 'molecule']
    assert len(iso) == 1 and iso[0]['reason_code'] == 'mmff_typing'
    assert len(mol) == 1 and mol[0]['reason_code'] == 'mmff_typing'
    # the alkyne the test used to refuse is a molecule of the set now
    b, _ = bcs.build_molecule(_entry('C#CC1OC1C(F)(F)F'), ENERGY_KW, cap=2, bundle=None,
                              stereo_salt='')
    assert b is not None and len(b.members) == 2


def test_graph_broken_member_is_refused():
    from build_conformer_conditions import MemberRefused, _bond_graph_mismatch, build_member
    from energies.conformer_torsions import ConformerTorsions
    # the seed-0 reference loses a bond and DISCONNECTS the graph: the constructor refuses
    with pytest.raises(MemberRefused) as exc:
        build_member('CNc1ncncc1F', 'CNc1ncncc1F', ENERGY_KW)
    assert exc.value.code == 'graph_broken'
    en = ConformerTorsions(smiles='CCO', device='cpu', **ENERGY_KW)
    assert _bond_graph_mismatch(en) == ([], [])
    en.bond_index_slot = np.asarray(en.bond_index_slot)[:, 1:]
    missing, extra = _bond_graph_mismatch(en)
    assert len(missing) == 1 and not extra


def test_connected_graph_mismatch_is_refused_by_build_member(monkeypatch):
    """QM9 72100: this isomer's seed-0 reference stretches a C-O bond to 1.71 A, past the
    perception cutoff, and the inferred graph stays CONNECTED. The constructor accepts it and
    every later check passes, so the graph comparison in `_build_member` is the only thing
    between it and a member whose C-O pair carries nonbonded terms on top of its bond term."""
    import build_conformer_conditions as bcc
    ident = 'C1[C@H]2O[C@@H]3[C@H]4O[C@@]13[C@H]4O2'
    with pytest.raises(bcc.MemberRefused) as exc:
        bcc.build_member(ident, ident, ENERGY_KW, carrier=True)
    assert exc.value.code == 'graph_broken', exc.value.message
    assert 'geometry-inferred bond graph' in exc.value.message
    assert 'missing [(' in exc.value.message and 'extra []' in exc.value.message
    # the example still exercises the call site: with the comparison blinded, it builds
    monkeypatch.setattr(bcc, '_bond_graph_mismatch', lambda energy: ([], []))
    bcc.build_member(ident, ident, ENERGY_KW, carrier=True)


def test_failure_classification():
    # `transverse_no_carrier_block` ('... has no transverse block') is gone from the table: it
    # was CarrierLayout's refusal of any transverse column, and the layout now places them
    # in the theta region, so nothing raises that text (see the test below)
    from build_conformer_conditions import REASON_CODES, classify_failure
    assert 'transverse_no_carrier_block' not in REASON_CODES
    cases = {'could not embed CCC': 'embed_failed',
             "X does not have a complete chart at level 'full'": 'incomplete_chart_full',
             'ring-closure bond (4, 10) has no free endpoint': 'closure_encoding',
             'need at least 4 atoms for a full internal parameterisation, got 3': 'lt4_atoms',
             'delta_theta_max=0.5 puts the theta box at [0.9, 3.2] rad': 'theta_box',
             'something new': 'other'}
    for msg, code in cases.items():
        assert classify_failure(ValueError(msg)) == code


def test_linear_group_refusals_are_classified_from_their_producers():
    """The refusals a linear group can still meet, raised by the code that raises them rather
    than typed here: both transverse reach checks (rho >= pi on the disc; rho >= pi/2 on a
    bend that anchors a dummy frame, which the pattern table missed when the dummy frame
    landed) are `transverse_box`, and the carrier layout's one remaining refusal, a block
    code with no region, is a chart/layout defect recorded as `other` with its message."""
    import types

    from build_conformer_conditions import MemberRefused, build_member, classify_failure
    from energies.conformer_carrier import CarrierLayout
    for smiles, dtm, text in (('CC#N', 2.3, 'reach rho >= pi,'),
                              ('CC#C', 1.2, 'anchors a dummy frame reach rho >= pi/2')):
        with pytest.raises(MemberRefused) as exc:
            build_member(smiles, smiles, {**ENERGY_KW, 'delta_theta_max': dtm}, carrier=True)
        assert exc.value.code == 'transverse_box' and text in exc.value.message
    with pytest.raises(ValueError, match='no carrier region') as exc:
        CarrierLayout({'X': types.SimpleNamespace(_free_block=np.array([0, 1, 5]))})
    assert classify_failure(exc.value) == 'other'
    # and the linear groups themselves build, in the carrier form a set file carries
    for smiles in ('CC#N', 'CC#C'):
        en = build_member(smiles, smiles, ENERGY_KW, carrier=True).energy
        assert en.data_ndim == 3 * en.spec.n_atoms - 6


# ------------------------------------------------------------------ whole builds


def test_sixty_row_build_accounts_for_every_row(tmp_path):
    from build_conformer_set import _sha256, rebuilt_layout
    out, man = _build(tmp_path, 'all', 'all', 'all')
    acc = man['accounting']
    assert acc['universe_rows'] == N_UNIVERSE and acc['not_attempted'] == 0
    assert acc['kept_molecules'] + sum(man['reasons']['molecule'].values()) == N_UNIVERSE

    rej = {int(r['dataset_index']): r['reason_code'] for r in _tsv(out / 'rejections.tsv')
           if r['level'] == 'molecule'}
    assert rej[25460] == rej[60163] == 'encoder_pool_key'
    assert rej[51056] == 'duplicate_key'
    assert rej[116340] == 'lt4_atoms'
    assert rej[25679] == 'theta_box'
    assert rej[60167] == 'graph_broken'
    # a many-molecule training set is a carrier, which places by region rank, so nothing
    # inside its widths moves it: the placement rule of admit_heldout drops nothing here
    assert not man['layout']['train_only_is_identity']
    assert 'heldout_moves_train_layout' not in man['reasons']['molecule']

    for name, meta in man['artifacts'].items():
        assert _sha256(out / name) == meta['sha256'], name
        assert (out / name).stat().st_size == meta['bytes'], name

    mols = _tsv(out / 'molecules.tsv')
    # LINEAR GROUPS ARE ADMITTED (owner decision). These two rows were refused before the
    # transverse chart reached the carrier and the alkyne chart: 26984 as
    # incomplete_chart_full, 52508 as transverse_no_carrier_block. Each linear centre
    # carries one (u, v) pair: the alkyne's two sp carbons 4 columns, the nitrile carbon 2.
    assert 26984 not in rej and 52508 not in rej
    tv = {int(m['dataset_index']): int(m['n_transverse']) for m in mols
          if int(m['dataset_index']) in (26984, 52508)}
    assert tv == {26984: 4, 52508: 2}, tv
    assert all(int(m['dataset_index']) >= POOL_NEED for m in mols)
    assert not ({m['key'] for m in mols if m['split'] == 'train'}
                & {m['key'] for m in mols if m['split'] == 'heldout'})
    K = man['layout']['K']
    for name in ('conditions_train.pt', 'conditions_heldout.pt'):
        batch = torch.load(out / name, weights_only=False)['prior']
        k_file, cols = rebuilt_layout(batch)
        assert k_file == K
        for m in man['members']['train' if 'train' in name else 'heldout']:
            assert len(cols[m['identifier']]) == m['k'] == K - m['n_pad']
    # every mirror pair is recorded both ways round and both halves were built
    ids = {m['identifier'] for m in mols}
    for a, b in man['stereo']['mirror_pairs']:
        assert a in ids and b in ids
    # the pair's references are independent embeddings: their energies are recorded, and
    # the manifest says the log Z equality holds only in expectation
    assert all(np.isfinite(float(m['reference_energy'])) for m in mols)
    assert 'in expectation' in man['stereo']['mirror_pairs_note']


def test_rungs_nest(tmp_path):
    small, _ = _build(tmp_path, 'small', 3, 2)
    large, _ = _build(tmp_path, 'large', 8, 4)
    assert (small / 'split.tsv').read_bytes() == (large / 'split.tsv').read_bytes()
    s, l = _tsv(small / 'molecules.tsv'), _tsv(large / 'molecules.tsv')
    for side in ('train', 'heldout'):
        a = {m['identifier'] for m in s if m['split'] == side}
        b = {m['identifier'] for m in l if m['split'] == side}
        assert a <= b, (side, a - b)
    assert {m['identifier'] for m in s if m['split'] == 'train'}
    # the held-out WALK nests too, including the molecules admit_heldout then dropped (too
    # wide, or moving a training member); a larger training rung can only re-admit them
    walked = lambda d: ({m['key'] for m in _tsv(d / 'molecules.tsv') if m['split'] == 'heldout'}
                        | {r['key'] for r in _tsv(d / 'rejections.tsv')
                           if r['reason_code'] in ('heldout_wider_than_train',
                                                   'heldout_moves_train_layout')})
    assert walked(small) and walked(small) <= walked(large)


def test_heldout_wider_than_train_is_dropped_and_named():
    import build_conformer_set as bcs
    from build_conformer_conditions import build_member
    train = {'CCO': build_member('CCO', 'CCO', ENERGY_KW, carrier=True)}
    wide = bcs.Built(_entry('CCCCO', 'heldout'),
                     members=[build_member('CCCCO', 'CCCCO', ENERGY_KW, carrier=True)])
    narrow = bcs.Built(_entry('CO', 'heldout'),
                       members=[build_member('CO', 'CO', ENERGY_KW, carrier=True)])
    admitted, rej, width = bcs.admit_heldout(train, [wide, narrow])
    assert [b.entry.key for b in admitted] == ['CO']
    assert width == [8, 7, 6]
    assert len(rej) == 1 and rej[0]['reason_code'] == 'heldout_wider_than_train'
    assert rej[0]['key'] == 'CCCCO'


#: a nitrile whose two enantiomers share one `_free_block` with the transverse v among the phi
#: columns, and a molecule inside its region widths ([8, 7, 6] against [9, 9, 6]) with other
#: codes. The training-only layout of the enantiomers is the IDENTITY; one over both sides is a
#: region-ordered carrier that moves v into the theta region.
MOVER_TRAIN, MOVER_HELD = 'CC(O)C#N', 'CCO'


def test_heldout_that_moves_the_training_placement_is_dropped_and_named():
    """Admission is asked of the placement the RUN rebuilds, not of the widths alone.

    The run builds its layout from the training file alone. Width-only admission let CCO
    into the shared layout that wrote the training file, so the enantiomers' v was written to
    carrier column 17 -- a column the run's identity layout wraps as a phi -- and every file
    check, run against that same shared layout, passed."""
    import build_conformer_set as bcs
    from build_conformer_conditions import build_member, enumerate_stereoisomers
    from energies.conformer_carrier import CarrierLayout
    isos = enumerate_stereoisomers(MOVER_TRAIN)
    assert len(isos) == 2
    train = {s: build_member(s, s, ENERGY_KW, carrier=True) for s in isos}
    held = bcs.Built(_entry(MOVER_HELD, 'heldout'),
                     members=[build_member(MOVER_HELD, MOVER_HELD, ENERGY_KW, carrier=True)])
    alone = CarrierLayout({i: m.energy for i, m in train.items()})
    both = CarrierLayout({**{i: m.energy for i, m in train.items()},
                          MOVER_HELD: held.members[0].energy})
    # the premise: equal widths, identity alone, the shared layout moves the training rows
    assert alone.is_identity and not both.is_identity
    assert both.block_width == alone.block_width
    assert all(not np.array_equal(both.cols[i], alone.cols[i]) for i in train)
    assert bcs.train_placement_change(alone, both, train)
    assert bcs.train_placement_change(alone, alone, train) == ''
    # a move the region codes cannot see: two same-region columns swapped, K and codes equal
    import copy
    swapped = copy.deepcopy(alone)
    ident = isos[0]
    c = swapped.cols[ident].copy()
    c[[0, 1]] = c[[1, 0]]
    swapped.cols[ident] = c
    assert np.array_equal(swapped.free_block, alone.free_block)
    assert 'moved' in bcs.train_placement_change(alone, swapped, train)

    admitted, rej, width = bcs.admit_heldout(train, [held])
    assert admitted == [] and width == list(alone.block_width)
    assert [r['reason_code'] for r in rej] == ['heldout_moves_train_layout']
    assert rej[0]['key'] == bcs.constitution_key(MOVER_HELD)

    # the rule is the placement, not "the codes differ": once the training set is itself a
    # carrier (a third, wider molecule), the same held-out molecule moves nothing
    train['CC(C)CO'] = build_member('CC(C)CO', 'CC(C)CO', ENERGY_KW, carrier=True)
    admitted, rej, _ = bcs.admit_heldout(train, [held])
    assert [b.entry.key for b in admitted] == [bcs.constitution_key(MOVER_HELD)] and rej == []


def test_training_file_is_checked_against_the_layout_the_run_rebuilds(tmp_path, monkeypatch):
    """End to end on a 2-row source: the nitrile on the training side, CCO held out.

    With the placement rule, CCO is dropped and named, and the training file is the one the
    run's own layout reads. With it disabled -- width-only admission, the old behaviour --
    main refuses before writing; with main's check disabled too, the training file's
    verification, run against the training-only layout, refuses it."""
    import build_conformer_set as bcs
    from energies.conformer_carrier import CarrierLayout
    salt = next(s for s in (f'mover{k}' for k in range(500))
                if bcs.split_hash(bcs.constitution_key(MOVER_HELD), s) % 1000
                < bcs.split_hash(bcs.constitution_key(MOVER_TRAIN), s) % 1000)
    permille = bcs.split_hash(bcs.constitution_key(MOVER_HELD), salt) % 1000 + 1
    rows = [(POOL_NEED + 10, MOVER_TRAIN), (POOL_NEED + 11, MOVER_HELD)]
    src = _write_slice(tmp_path / 'mover.csv', rows)

    memo, real_build = {}, bcs.build_molecule           # build each molecule once, not 3 times

    def build_once(entry, *a, **kw):
        if entry.key not in memo:
            memo[entry.key] = real_build(entry, *a, **kw)
        return memo[entry.key]
    monkeypatch.setattr(bcs, 'build_molecule', build_once)

    def run(name):
        out = tmp_path / name
        bcs.main(['--out-dir', str(out), '--source', str(src), '--n-train', '1',
                  '--n-heldout', '1', '--no-encoder', '--pool-need', str(POOL_NEED),
                  '--heldout-permille', str(permille), '--split-salt', salt])
        return out

    out = run('fixed')
    with open(out / 'manifest.json', encoding='utf-8') as f:
        man = json.load(f)
    rej = {r['key']: r['reason_code'] for r in _tsv(out / 'rejections.tsv')}
    assert rej[bcs.constitution_key(MOVER_HELD)] == 'heldout_moves_train_layout'
    assert not (out / 'conditions_heldout.pt').exists()
    assert man['layout']['is_identity'] and man['layout']['train_only_is_identity']
    batch = torch.load(out / 'conditions_train.pt', weights_only=False)['prior']
    assert len(set(batch.identifier)) == 2
    # the file read through the layout the run builds from it: identity maps, v in place
    members = {mb.identifier: mb for mb in memo[bcs.constitution_key(MOVER_TRAIN)][0].members}
    run_layout = CarrierLayout({i: m.energy for i, m in members.items()})
    bcs.verify_conditions_file(out / 'conditions_train.pt', run_layout,
                               man['members']['train'],
                               {i: m.condition for i, m in members.items()}, covers_all=True)

    real_admit = bcs.admit_heldout

    def width_only(tm, held):
        with monkeypatch.context() as m:
            m.setattr(bcs, 'train_placement_change', lambda *a: '')
            return real_admit(tm, held)
    monkeypatch.setattr(bcs, 'admit_heldout', width_only)
    with pytest.raises(SystemExit, match='move the training placement'):
        run('width_only')
    monkeypatch.setattr(bcs, 'train_placement_change', lambda *a: '')
    with pytest.raises(SystemExit, match='reconstruction map reads the carrier columns in '
                                         'another order'):
        run('unguarded')
    assert not (tmp_path / 'unguarded' / 'conditions_train.pt').exists()


def test_file_disagreeing_with_the_manifest_is_refused(tmp_path):
    """The written file is checked against the CarrierLayout that wrote it, asked directly.

    The check used to compare each row's owned columns, ascending, with the layout's column
    list in MEMBER order. The two agree only while every member's columns are in region order;
    a transverse u/v sits in the theta region wherever it falls in the member's state, so on
    a set with a nitrile or an alkyne that comparison refused every correct file. The owned
    SET is now compared with `layout.valid`, and the ORDER with the reconstruction map."""
    import copy

    import build_conformer_set as bcs
    from build_conformer_conditions import build_member
    from energies.conformer_carrier import CarrierLayout
    out, man = _build(tmp_path, 'v', 3, 0)
    members = [dict(m) for m in man['members']['train']]
    built = {m['identifier']: build_member(m['identifier'], m['identifier'],
                                           man['energy']['kwargs'], carrier=True)
             for m in members}
    lay = CarrierLayout({i: b.energy for i, b in built.items()})
    own = {i: b.condition for i, b in built.items()}
    assert lay.K == man['layout']['K']
    # the set must hold a member whose carrier columns are NOT in member order, or the order
    # checks below prove nothing (the 3-molecule rung holds an alkyne)
    scrambled = [i for i in lay.cols if (np.diff(lay.cols[i]) < 0).any()]
    assert scrambled, 'the rung needs a linear-group member (transverse u/v in theta)'
    path = out / 'conditions_train.pt'
    bcs.verify_conditions_file(path, lay, members, own, covers_all=True)

    members[0]['k'] += 1
    with pytest.raises(SystemExit, match='disagrees with the manifest'):
        bcs.verify_conditions_file(path, lay, members, own, covers_all=True)
    members = [dict(m) for m in man['members']['train']]
    batch = torch.load(path, weights_only=False)['prior']
    # a refused write leaves nothing behind
    final = tmp_path / 'x.pt'
    with pytest.raises(SystemExit):
        bcs._write_then_verify(lambda p: torch.save({'prior': batch}, p), final,
                               lambda p: bcs.verify_conditions_file(
                                   p, lay, [dict(m, k=m['k'] + 1) for m in members], own,
                                   covers_all=True))
    assert not final.exists() and not (tmp_path / 'x.pt.tmp').exists()

    # SAME k, DIFFERENT columns: the owned-set comparison, not the count, refuses it
    j = max(range(len(members)), key=lambda i: members[i]['n_pad'])
    ident = members[j]['identifier']
    assert members[j]['n_pad'] > 0, 'the slice build needs members of different widths'
    mine = list(lay.cols[ident])
    shifted = copy.deepcopy(lay)
    shifted.cols[ident] = np.asarray(mine[1:] + [next(c for c in range(lay.K)
                                                      if c not in mine)])
    with pytest.raises(SystemExit, match='carrier columns differ'):
        bcs.verify_conditions_file(path, shifted, members, own, covers_all=True)

    # SAME columns, OTHER ORDER: a file whose reconstruction map places the member's columns
    # in ascending order -- the map the old check assumed -- owns exactly the right columns,
    # passes every mask and width, and reads u/v and theta out of each other's columns
    s = scrambled[0]
    row = list(batch.identifier).index(s)
    at = slice(int(batch.ptr[row]), int(batch.ptr[row + 1]))
    ascending = torch.as_tensor(np.sort(lay.cols[s]))
    for name in ('ctree_r_col', 'ctree_th_col', 'ctree_ph_col'):
        m = torch.as_tensor(getattr(own[s], name)).reshape(-1)
        getattr(batch, name)[at] = torch.where(m >= 0, ascending[m.clamp_min(0)], m)
    reordered = tmp_path / 'reordered.pt'
    torch.save({'prior': batch}, reordered)
    assert np.array_equal(bcs.rebuilt_layout(batch)[1][s], np.sort(lay.cols[s]))
    with pytest.raises(SystemExit, match='reconstruction map reads the carrier columns in '
                                         'another order'):
        bcs.verify_conditions_file(reordered, lay, members, own, covers_all=True)

    # a column no member owns: legal in the held-out file, refused in the training file
    batch = torch.load(path, weights_only=False)['prior']
    alone = tmp_path / 'alone.pt'
    torch.save({'prior': batch.subsample_new_batch(np.array([j]))}, alone)
    bcs.verify_conditions_file(alone, lay, [members[j]], own, covers_all=False)
    with pytest.raises(SystemExit, match='owned by no member'):
        bcs.verify_conditions_file(alone, lay, [members[j]], own, covers_all=True)


def test_prior_file_contract(tmp_path):
    from energies.conformer_carrier import CarrierLayout
    from energies.conformer_data import bake_energies
    from energies.conformer_torsions import ConformerTorsions
    out, man = _build(tmp_path, 'p', 3, 0, '--prior-rows-per-condition', 4,
                      '--internal-prior', _prior_path())
    from energies.conformer_data import PRIOR_COMPACT_FORMAT
    b = torch.load(out / 'prior_train.pt', weights_only=False)
    # the COMPACT form, named in the file and in the manifest; no graph per row
    assert b['prior_format'] == man['prior']['prior_format'] == PRIOR_COMPACT_FORMAT
    assert 'prior' not in b and 'equalized_prior' not in b
    x = torch.as_tensor(b['torsion_state'])
    e = torch.as_tensor(b['conformer_energy']).reshape(-1)
    assert x.dtype == torch.float32 and e.dtype == torch.float32
    K = man['layout']['K']
    assert x.shape[1] == K
    assert bool((x[~b['state_mask'][b['condition_index']]] == 0).all())
    idents = [b['identifiers'][c] for c in b['condition_index'].tolist()]
    # each condition's mask and atom count are the conditions file's own
    cond = torch.load(out / 'conditions_train.pt', weights_only=False)['prior']
    slot = {s: j for j, s in enumerate(cond.identifier)}
    at = [slot[s] for s in b['identifiers']]
    assert torch.equal(b['state_mask'], torch.as_tensor(cond.state_mask).bool()[at])
    assert torch.equal(b['n_atoms'], (cond.ptr[1:] - cond.ptr[:-1])[at])
    kw = man['energy']['kwargs']
    members = {m['identifier']: ConformerTorsions(smiles=m['identifier'], device='cpu', **kw)
               for m in man['members']['train']}
    lay = CarrierLayout(members)
    assert lay.K == K
    for ident, en in members.items():
        rows = [j for j, s in enumerate(idents) if s == ident]
        assert len(rows) == 4
        with torch.no_grad():
            e2 = bake_energies(en, lay.from_carrier(ident, x[rows].double()))
        # BIT-EQUAL after the cast: the stored energy is the energy of the stored state
        assert torch.equal(e[rows], e2.to(e.dtype)), (ident, e[rows], e2)

    # the refusal paths of verify_prior_file, each on a copy of the file
    import build_conformer_set as bcs
    wrapped = {i: types.SimpleNamespace(energy=en) for i, en in members.items()}
    bcs.verify_prior_file(out / 'prior_train.pt', lay, wrapped, 4)

    def tampered(name, edit):
        blob = torch.load(out / 'prior_train.pt', weights_only=False)
        blob = edit(blob)
        torch.save(blob, tmp_path / name)
        return tmp_path / name

    def one_ulp(pb):
        e = pb['conformer_energy']
        before = float(e[0])
        e[0] = torch.nextafter(e[0], torch.tensor(float('inf'), dtype=e.dtype))
        assert float(e[0]) != before     # the edit took (a float64 nextafter rounds back)
        return pb

    def pad(pb):
        sm = pb['state_mask'][pb['condition_index']]
        r, c = map(int, torch.nonzero(~sm)[0])
        pb['torsion_state'][r, c] = 0.5
        return pb

    def drop_first_row(pb):
        for key in ('condition_index', 'torsion_state', 'conformer_energy'):
            pb[key] = pb[key][1:]
        return pb

    def other_mask(pb):
        pb['state_mask'][0] = ~pb['state_mask'][0]
        return pb

    def other_atoms(pb):
        pb['n_atoms'][0] += 1
        return pb

    def ragged(pb):
        pb['conformer_energy'] = pb['conformer_energy'][1:]
        return pb

    def graph_form(pb):
        return {'prior': None, 'equalized_prior': None}

    cases = {'ulp.pt': (one_ulp, 'stored energies differ'),
             'pad.pt': (pad, 'non-zero pad'),
             'rows.pt': (drop_first_row, 'rows, expected 4'),
             'mask.pt': (other_mask, 'state_mask is not the layout'),
             'atoms.pt': (other_atoms, 'atoms recorded'),
             'ragged.pt': (ragged, 'inconsistent compact prior'),
             'graph.pt': (graph_form, 'prior_format')}
    for name, (edit, why) in cases.items():
        with pytest.raises(SystemExit, match=why):
            bcs.verify_prior_file(tampered(name, edit), lay, wrapped, 4)

    # a forced rebuild without a prior leaves no stale prior beside the new manifest
    with pytest.raises(SystemExit, match='already holds a manifest'):
        bcs.main(['--out-dir', str(out), '--source', str(tmp_path / 'p_slice.csv'),
                  '--n-train', '3', '--n-heldout', '0', '--no-encoder',
                  '--pool-need', str(POOL_NEED)])
    bcs.main(['--out-dir', str(out), '--source', str(tmp_path / 'p_slice.csv'),
              '--n-train', '3', '--n-heldout', '0', '--no-encoder',
              '--pool-need', str(POOL_NEED), '--force'])
    assert not (out / 'prior_train.pt').exists()
    assert 'prior_train.pt' not in json.loads((out / 'manifest.json').read_text())['artifacts']


def test_draw_member_prior_is_the_modeller_draw():
    from conformer_modeller import ConformerModeller
    from energies.conformer_prior_draw import draw_member_prior
    from energies.conformer_torsions import ConformerTorsions
    prior = torch.load(_prior_path(), weights_only=False)
    en = ConformerTorsions(smiles='CCCCO', device='cpu', **ENERGY_KW)
    stub = types.SimpleNamespace(energy_function=en, internal_prior=prior,
                                 args=types.SimpleNamespace(
                                     energy_config=types.SimpleNamespace()))
    for steps in (0, 3):
        a, _ = ConformerModeller._draw_prior_states(stub, 32, np.random.default_rng(5),
                                                    steps=steps, en=en)
        b, e, st = draw_member_prior(en, 32, np.random.default_rng(5), relax_steps=steps,
                                     prior=prior)
        assert torch.equal(torch.as_tensor(a), b)
        assert st['relax_steps'] == steps
        from energies.conformer_data import bake_energies
        assert torch.equal(bake_energies(en, b), e)
    with pytest.raises(ValueError, match='no uniform fallback'):
        draw_member_prior(en, 4, np.random.default_rng(0), prior=None)


def test_encoder_reads_the_stereo_tags():
    from build_conformer_conditions import build_member
    from build_conformer_set import encoder_pool
    from models import encoder_cache
    from models.graph_encodings import graph_from_smiles
    ckpt = _encoder_ckpt()
    need, info = encoder_pool(ckpt)
    assert need == POOL_NEED, info
    bundle = encoder_cache.load_encoder(str(ckpt), device='cpu')
    r = build_member('CC[C@@H](C)O', 'CC[C@@H](C)O', ENERGY_KW, bundle=bundle)
    s = build_member('CC[C@H](C)O', 'CC[C@H](C)O', ENERGY_KW, bundle=bundle)
    assert np.count_nonzero(graph_from_smiles('CC[C@@H](C)O')[2]) == 1
    assert not torch.equal(r.condition.embedding, s.condition.embedding)


# ------------------------------------------------------------------ the old CLI


def test_old_cli_refuses_per_molecule(tmp_path):
    """Refused molecules are named and the file is written from the rest.

    The two refused SMILES were refused for their linear groups (incomplete_chart_full,
    transverse_no_carrier_block) before nitriles and alkynes were admitted; the constitution
    check no longer stops them, and they are refused for what they leave open instead -- an
    untagged imine and an untagged oxime, each E/Z unspecified. A nitrile and an alkyne that
    name one stereoisomer are written, into a carrier whose transverse columns sit in the
    theta region."""
    import build_conformer_conditions as bcc
    out, rj = tmp_path / 'c.pt', tmp_path / 'r.tsv'
    bcc.main(['--smiles', 'CCCCO', 'CC(C)(OC=N)C#N', 'CC#CC(C)=NO', 'OCCC#N', 'CC#CCO',
              '--level', 'full', '--force-field', 'mmff', '--carrier', '--out', str(out),
              '--rejections-out', str(rj)])
    written = torch.load(out, weights_only=False)['prior']
    assert list(written.identifier) == ['CCCCO', 'OCCC#N', 'CC#CCO']
    assert tuple(written.state_mask.shape) == (3, int(written.n_torsions[0]))  # carrier form
    assert sorted(r['code'] for r in _tsv(rj)) == ['stereo_unspecified', 'stereo_unspecified']
    assert sorted(r['identifier'] for r in _tsv(rj)) == ['CC#CC(C)=NO', 'CC(C)(OC=N)C#N']
    bcc.main(['--smiles', 'CCO', 'CCC1C2C3CN2C13', 'CCCO', '--carrier', '--no-check',
              '--out', str(out), '--rejections-out', str(rj)])
    assert list(torch.load(out, weights_only=False)['prior'].identifier) == ['CCO', 'CCCO']
    assert [r['code'] for r in _tsv(rj)] == ['closure_encoding']


def test_old_cli_torsion_draw_is_refused_above_torsion():
    from build_conformer_conditions import draw_prior_states
    from energies.conformer_torsions import ConformerTorsions
    en = ConformerTorsions(smiles='CCCCO', device='cpu', **ENERGY_KW)
    with pytest.raises(ValueError, match='torsion-tier draw'):
        draw_prior_states(en, 4, None, 0.15, 0)
