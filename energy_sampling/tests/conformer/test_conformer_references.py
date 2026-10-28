"""The offline per-molecule reference table (build_conformer_references.py).

WHAT THESE GATES PIN, each constructed so the number has to move when the claim is false:

  * the floor IS a tier minimum: the stored e_min equals a fresh ``tier_minimum`` over the
    same starts, and equals ``bake_energies`` of its own state -- the raw-potential, T = 1
    currency of the baked ``conformer_energy``;
  * the stereo pin is the CONDITION's: the elements its SMILES specifies, at the
    reference's configuration. A stereo-free SMILES pins nothing (the floor is over every
    stereoisomer), =NH imine E/Z is open unless the SMILES writes the [H], and a reference
    that does not realise the SMILES is refused. On a tagged Z-2-butene the floor is Z's,
    above E's, and its state is Z;
  * the starts are what they claim: an ETKDG start decodes to its own embedding's
    geometry, the embedding is MMFF-optimised, the embedder enforces exactly the
    condition's tags, an embedding of another configuration is dropped, and a prior draw
    that raises is topped up with uniform starts and surfaces at table level;
  * the stamp names what the table means, and ``load_references`` refuses a table built
    for another conditions file, another member definition, another format, or one
    missing an identifier, while accepting changes that cannot move a floor;
  * ``verify`` re-scores through the RUN's members, so a changed force field is caught even
    where every stamped argument agrees, and it refuses a member of another width or one
    the table has no entry for;
  * a killed build resumes the molecules it finished, refuses parts from another stamp
    (another search or another energy), and the pool's wait is bounded.

Level `full`, force field `mmff`, energy_clip 300: the conformer training target.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings('ignore')
try:
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')
except Exception:
    pass

import build_conformer_references as bcr
from energies.conformer_data import (bake_energies, collate_conditions, condition_from_smiles,
                                     save_condition_file)
from energies.conformer_torsions import ConformerTorsions
from energies.prior_baselines import descend, draw_uniform, tier_minimum

KW = dict(level='full', force_field='mmff', energy_clip=300.0)
#: three molecules of one width (9 atoms, k = 21), so one plain conditions file holds them;
#: none has a stereo element, so nothing is pinned
SMIS = ['CCO', 'COC', 'C=CC']
#: small but real: the reference, 2 ETKDG re-embeddings and 7 uniform starts, 150 steps
SEARCH = dict(n_uniform=7, n_prior=0, n_seeds=2, steps=150)
Z_BUTENE, E_BUTENE, BUTENE = 'C/C=C\\C', 'C/C=C/C', 'CC=CC'


def _write_conditions(path, smiles, **kw):
    batch = collate_conditions([condition_from_smiles(s, **{**KW, **kw}) for s in smiles])
    return save_condition_file(batch, str(path))


def _member(smiles):
    with bcr._float64():
        return bcr.build_member(smiles, KW)


def _dist(p):
    p = np.asarray(p, dtype=np.float64)
    return np.linalg.norm(p[:, None] - p[None], axis=-1)


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp('refs')
    cond = _write_conditions(d / 'conditions.pt', SMIS)
    out = d / 'references.pt'
    res = bcr.build_references(cond, out, KW, search=SEARCH, workers=0, log=lambda *_: None)
    assert not res['failures'], res['failures']
    return dict(dir=d, cond=cond, out=out, res=res,
                table=bcr.load_references(out, conditions_path=cond, energy_kwargs=KW))


# ------------------------------------------------------------------ the floor itself


def test_every_identifier_has_an_entry_in_file_order(built):
    assert built['table'].identifiers == SMIS and built['table'].missing == []
    for ident in SMIS:
        e = built['table'][ident]
        assert e['k'] == 21 and e['n_starts'] == 1 + 2 + 7
        assert e['starts']['n_etkdg'] == 2 and e['starts']['n_uniform'] == 7
        assert e['starts']['etkdg_seeds'] == [1, 2]
        assert e['stereo_pin'] == 'none' and e['n_below_other_stereo'] == 0


def test_e_min_matches_a_fresh_tier_minimum_over_the_same_starts(built):
    """DoD: e_min equals prior_baselines.tier_minimum with the same starts and seed."""
    with bcr._float64():
        for ident in SMIS:
            member = bcr.build_member(ident, KW)
            starts, _, _ = bcr.search_starts(member, ident, n_uniform=SEARCH['n_uniform'],
                                             n_prior=0, n_seeds=SEARCH['n_seeds'])
            best, worst, n = tier_minimum(member, starts, steps=SEARCH['steps'])
            e = built['table'][ident]
            assert abs(best - e['e_min']) < 1e-4, (ident, best, e['e_min'])
            assert abs(worst - e['worst_start']) < 1e-4 and n == e['n_starts']


def test_e_min_is_the_raw_potential_at_T1_of_its_own_state(built):
    """The currency: bake_energies (potential_energy at T = 1), never -log_r * T."""
    with bcr._float64():
        for ident in SMIS:
            member = bcr.build_member(ident, KW)
            e = built['table'][ident]
            x = e['e_min_state']
            assert x.shape == (member.ndim,) and bool((x.abs() <= 1.0).all())
            u = float(bake_energies(member, x.reshape(1, -1))[0])
            assert abs(u - e['e_min']) < 1e-9
            # a floor, so not above the reference conformer's own energy
            assert e['e_min'] <= e['u_ref'] + 1e-9


def test_the_floor_is_the_lowest_candidate_that_keeps_the_pin(monkeypatch):
    """Selection logic: the lowest candidate is labelled another configuration, so the floor
    is the next one down, its STATE is that candidate's, and the unrestricted minimum and
    the rank are recorded beside it."""
    with bcr._float64():
        member = bcr.build_member('CCCO', KW)
        starts = torch.cat([torch.zeros(1, member.ndim),
                            draw_uniform(member, 11, 0)]).clamp(-1, 1)
        kinds = ['ref'] + ['uniform'] * 11
        bx, _ = descend(member, starts, 150)
        bu = bake_energies(member, bx).numpy()
        lowest = bx[int(np.argmin(bu))]

        def fake(m, states, pin):
            s = torch.as_tensor(states).reshape(-1, m.ndim)
            return [('other',) if torch.equal(r, lowest) else ('ref',) for r in s]

        monkeypatch.setattr(bcr, 'stereo_labels', fake)
        pin = dict(atoms=(0,), bonds=(), target=('ref',), pinned={'atoms': [0], 'bonds': []})
        fl = bcr.floor_search(member, starts, kinds, 150, pin)
        u_state = float(bake_energies(member, fl['e_min_state'].reshape(1, -1))[0])
    assert fl['e_min_unpinned'] == pytest.approx(float(bu.min()))
    assert fl['n_below_other_stereo'] >= 1 and fl['e_min'] > fl['e_min_unpinned']
    assert fl['e_min'] == pytest.approx(float(np.sort(bu)[fl['n_below_other_stereo']]))
    assert u_state == pytest.approx(fl['e_min'], abs=1e-9)


# --------------------------------------------------------------- the condition's pin


@pytest.mark.parametrize('smiles, pin, pinned, n_open_atoms, n_open_bonds, imine', [
    # stereo-free SMILES: nothing pinned, whatever the reference happens to be
    ('CC(O)C(C)O', 'none', {'atoms': [], 'bonds': []}, 2, 0, 0),
    (BUTENE, 'none', {'atoms': [], 'bonds': []}, 0, 1, 0),
    # an implicit-H SMILES cannot write =NH E/Z: open, and counted as an imine
    ('CC=N', 'none', {'atoms': [], 'bonds': []}, 0, 1, 1),
    ('C[C@H](O)C=N', 'partial', {'atoms': [1], 'bonds': []}, 0, 1, 1),
    # tagged: every element the reference carries is pinned
    ('C[C@@H](O)[C@@H](C)O', 'full', {'atoms': [1, 3], 'bonds': []}, 0, 0, 0),
    (Z_BUTENE, 'full', {'atoms': [], 'bonds': [[1, 2]]}, 0, 0, 0),
    ('[H]/N=C/C', 'full', {'atoms': [], 'bonds': [[1, 2]]}, 0, 0, 0),
])
def test_the_pin_is_what_the_condition_smiles_specifies(smiles, pin, pinned, n_open_atoms,
                                                       n_open_bonds, imine):
    member = _member(smiles)
    p = bcr.condition_stereo(member)
    assert p['pin'] == pin and p['pinned'] == pinned
    assert (len(p['open']['atoms']), len(p['open']['bonds'])) == (n_open_atoms, n_open_bonds)
    assert p['open']['imine_nh'] == imine
    # the reference realises the pin it defines
    assert bcr.stereo_labels(member, torch.zeros(1, member.ndim), p)[0] == p['target']
    if pin == 'full':
        from rdkit import Chem
        assert p['stereo'] == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_a_reference_that_is_not_the_conditions_stereoisomer_is_refused():
    member = _member('C[C@H](O)CC')
    member.smiles = 'C[C@@H](O)CC'          # the mirror's SMILES on this member's reference
    with pytest.raises(ValueError, match="not the condition's stereoisomer at atom 1"):
        bcr.condition_stereo(member)


def test_a_full_pin_whose_isomer_string_disagrees_with_the_smiles_is_refused(monkeypatch):
    """The whole-molecule backstop behind the per-element check: a stereo kind the element
    lists do not cover would show only in the isomeric SMILES."""
    member = _member('C[C@H](O)CC')
    monkeypatch.setattr(bcr, '_isomer_smiles', lambda m: 'CC[C@@H](C)O')
    with pytest.raises(ValueError, match='perceives as'):
        bcr.condition_stereo(member)


def test_a_tagged_nitrogen_is_refused_because_3d_perception_cannot_check_it():
    with pytest.raises(ValueError, match='tetrahedral N'):
        bcr.condition_stereo(_member('C[N@]1CC1C'))


def test_a_smiles_that_does_not_index_the_members_mol_is_refused():
    member = _member('CCO')
    member.smiles = 'OCC'
    with pytest.raises(ValueError, match='does not index'):
        bcr.condition_stereo(member)


def test_non_legacy_stereo_perception_is_refused():
    from rdkit import Chem

    member = _member('C[C@H](O)CC')
    Chem.SetUseLegacyStereoPerception(False)
    try:
        with pytest.raises(RuntimeError, match='non-legacy'):
            bcr.condition_stereo(member)
    finally:
        Chem.SetUseLegacyStereoPerception(True)


def test_the_mirror_image_breaks_the_pin_at_the_same_energy():
    """The pin is only as good as the perception: 2-butanol's mirror scores the SAME energy
    (an exact symmetry of the full-level target) and must break the pin."""
    with bcr._float64():
        member = bcr.build_member('C[C@H](O)CC', KW)
        pin = bcr.condition_stereo(member)
        mirror = member.state_from_dof(member.r0.reshape(1, -1), member.th0.reshape(1, -1),
                                       -member.ph0.reshape(1, -1))
        ref = torch.zeros(1, member.ndim, dtype=torch.float64)
        lab_ref, lab_mirror = bcr.stereo_labels(member, torch.cat([ref, mirror]), pin)
        u = bake_energies(member, torch.cat([ref, mirror]))
    assert lab_ref == pin['target'] and lab_mirror != pin['target']
    assert abs(float(u[0] - u[1])) < 1e-6


@pytest.mark.parametrize('smiles', [Z_BUTENE, 'C[C@@H](O)[C@@H](C)O'])
def test_the_pin_agrees_with_whole_molecule_perception_on_descended_candidates(smiles):
    """For a fully tagged condition, keeping the pin must be exactly being the condition's
    isomer as RDKit perceives the whole molecule. Uniform descents land on both sides."""
    with bcr._float64():
        member = bcr.build_member(smiles, KW)
        pin = bcr.condition_stereo(member)
        bx, _ = descend(member, draw_uniform(member, 48, 3), 150)
        keeps = [lab == pin['target'] for lab in bcr.stereo_labels(member, bx, pin)]
        same = [s == pin['stereo'] for s in bcr.stereo_signatures(member, bx)]
    assert keeps == same
    assert 0 < sum(keeps) < len(keeps)


def test_a_tagged_z_butene_floors_at_z_above_e_and_the_untagged_one_at_e():
    """End to end through compute_entry: the tagged condition's floor is its own isomer's,
    with e_min_state that isomer and bake(e_min_state) == e_min; the stereo-free condition
    of the same molecule floors over both isomers, i.e. at E."""
    search = dict(n_uniform=15, n_prior=0, n_seeds=2, steps=150)
    out = {}
    for smi in (Z_BUTENE, BUTENE):
        m = _member(smi)
        out[smi] = (m, bcr.compute_entry(smi, smi, np.asarray(m.spec.z), m.ref_pos.numpy(),
                                         KW, search))
    m, z = out[Z_BUTENE]
    assert z['stereo_pin'] == 'full' and z['e_min_stereo'] == Z_BUTENE
    assert z['n_below_other_stereo'] >= 1 and z['e_min'] > z['e_min_unpinned'] + 1.0
    with bcr._float64():
        u = float(bake_energies(m, z['e_min_state'].reshape(1, -1))[0])
        lab = bcr.stereo_labels(m, z['e_min_state'].reshape(1, -1), bcr.condition_stereo(m))[0]
    assert u == pytest.approx(z['e_min'], abs=1e-9)
    assert lab == bcr.condition_stereo(m)['target']
    _, free = out[BUTENE]
    assert free['stereo_pin'] == 'none' and free['n_below_other_stereo'] == 0
    assert free['e_min'] == free['e_min_unpinned'] and free['e_min_stereo'] == E_BUTENE
    assert free['e_min'] < z['e_min'] - 1.0


# ------------------------------------------------------------------------ the starts


def test_rdkit_positions_are_the_members_reference_in_rdkit_order():
    """The placement -> RDKit index map every perception rests on (placement slot i holds
    RDKit atom perm[i]); the molecule's perm is not the identity."""
    member = _member('CC(O)CC')
    assert not np.array_equal(np.asarray(member.spec.perm), np.arange(member.spec.n_atoms))
    rd = bcr.rdkit_positions(member, torch.zeros(1, member.ndim))[0]
    gap = np.abs(_dist(rd) - _dist(member.mol.GetConformer().GetPositions())).max()
    assert gap < 1e-6


def test_an_etkdg_start_decodes_to_its_own_embedding():
    """Each ETKDG start, decoded, is the MMFF-optimised embedding at its seed: the geometry is
    measured in the member's own tree with the atoms in placement order."""
    with bcr._float64():
        member = bcr.build_member('CC(O)CC', KW)
        pin = bcr.condition_stereo(member)
        xe, info = bcr.etkdg_starts(member, 3, 0, pin)
    assert info['n_etkdg'] == 3 and info['etkdg_outside_box'] == 0
    for seed, x in zip(info['etkdg_seeds'], xe):
        rd = bcr._embed(pin['template'], seed, True)
        dec = bcr.rdkit_positions(member, x.reshape(1, -1))[0]
        assert np.abs(_dist(dec) - _dist(rd)).max() < 1e-5, seed


def test_an_embedding_is_mmff_optimised_as_the_reference_is():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    member = _member('CC(O)CC')
    tmpl = bcr.condition_stereo(member)['template']

    def relaxes_by(rd):
        m = Chem.Mol(tmpl)
        AllChem.EmbedMolecule(m, randomSeed=1)
        m.GetConformer().SetPositions(np.ascontiguousarray(rd))
        ff = AllChem.MMFFGetMoleculeForceField(m, AllChem.MMFFGetMoleculeProperties(m))
        e0 = ff.CalcEnergy()
        ff.Minimize(maxIts=2000)
        return e0 - ff.CalcEnergy()

    assert relaxes_by(bcr._embed(tmpl, 1, True)) < 1e-3
    assert relaxes_by(bcr._embed(tmpl, 1, False)) > 0.1     # the check can fail


def test_the_embedder_enforces_exactly_the_conditions_tags():
    """Tagged: every seed re-places the condition's isomer. Stereo-free: the open centres
    are the embedder's to pick, so the seeds reach more than one isomer."""
    with bcr._float64():
        tagged = bcr.build_member('C[C@@H](O)[C@@H](C)O', KW)
        pin = bcr.condition_stereo(tagged)
        xe, info = bcr.etkdg_starts(tagged, 6, 0, pin)
        assert info['n_etkdg'] == 6 and info['etkdg_other_stereo'] == 0
        assert all(lab == pin['target'] for lab in bcr.stereo_labels(tagged, xe, pin))
        free = bcr.build_member('CC(O)C(C)O', KW)
        xf, _ = bcr.etkdg_starts(free, 6, 0, bcr.condition_stereo(free))
        assert len(set(bcr.stereo_signatures(free, xf))) >= 2


def test_an_embedding_of_another_configuration_is_dropped(monkeypatch):
    real = bcr._embed
    monkeypatch.setattr(bcr, '_embed', lambda tmpl, seed, mmff: -real(tmpl, seed, mmff))
    with bcr._float64():
        member = bcr.build_member('C[C@H](O)CC', KW)
        xe, info = bcr.etkdg_starts(member, 4, 0, bcr.condition_stereo(member))
    assert len(xe) == 0 and info['n_etkdg'] == 0 and info['etkdg_other_stereo'] == 4


def test_a_prior_draw_that_raises_is_topped_up_with_uniform_starts():
    with bcr._float64():
        member = bcr.build_member('CCO', KW)
        starts, kinds, info = bcr.search_starts(member, 'CCO', n_uniform=3, n_prior=4,
                                                n_seeds=0, prior=object())
    assert 'prior_error' in info and info['n_uniform'] == 7 and info['n_prior'] == 0
    assert len(starts) == 1 + 7 and kinds.count('uniform') == 7


def test_a_prior_fallback_is_visible_at_table_level(tmp_path, monkeypatch):
    cond = _write_conditions(tmp_path / 'conditions.pt', SMIS)
    prior = tmp_path / 'prior.pt'
    prior.write_bytes(b'not a prior')
    monkeypatch.setattr(bcr, '_load_prior', lambda path: object())
    out = tmp_path / 'references.pt'
    res = bcr.build_references(cond, out, KW, workers=0, internal_prior_path=prior,
                               search=dict(n_uniform=3, n_prior=2, n_seeds=0, steps=20),
                               log=lambda *_: None)
    assert res['prior_fallbacks'] == SMIS and not res['failures']
    tab = bcr.load_references(out, conditions_path=cond, energy_kwargs=KW)
    assert tab.prior_fallbacks == SMIS
    assert all(tab[i]['starts']['n_uniform'] == 5 for i in SMIS)


# ------------------------------------------------------------------ stamp and loading


def test_stamp_carries_level_force_field_clip_and_resolved_kwargs(built):
    st = built['table'].stamp
    assert st['format'] == bcr.FORMAT
    assert (st['level'], st['force_field'], st['energy_clip']) == ('full', 'mmff', 300.0)
    assert st['energy_kwargs'] == KW
    # resolved against the signature: keys the caller never passed carry their defaults
    assert st['energy']['delta_r_max'] == 0.30 and st['energy']['seed'] == 0
    assert st['energy']['energy_clip'] == 300.0
    assert not set(st['energy']) & bcr.NON_DEFINING_KWARGS
    assert st['conditions']['sha256'] == bcr.file_sha256(built['cond'])
    assert st['conditions']['identifiers'] == sorted(SMIS)
    assert st['conditions']['n_identifiers'] == len(SMIS)
    assert st['search']['n_seeds'] == 2 and st['search']['steps'] == 150
    assert 'condition SMILES' in st['search']['stereo']
    assert st['internal_prior'] is None
    assert {'gfn', 'rdkit', 'torch', 'mxtaltools'} <= set(st['code'])


def test_a_conditions_file_with_one_molecule_added_is_refused(built):
    grown = _write_conditions(built['dir'] / 'grown.pt', SMIS + ['C1CC1'])
    with pytest.raises(bcr.StaleReferencesError, match='sha256'):
        bcr.load_references(built['out'], conditions_path=grown, energy_kwargs=KW)


@pytest.mark.parametrize('change', [dict(energy_clip=100.0), dict(level='flex'),
                                    dict(force_field='reference'), dict(delta_r_max=0.25),
                                    dict(seed=1)])
def test_a_changed_member_definition_is_refused(built, change):
    with pytest.raises(bcr.StaleReferencesError, match=f'energy {next(iter(change))}'):
        bcr.load_references(built['out'], conditions_path=built['cond'],
                            energy_kwargs={**KW, **change})


def test_changes_that_cannot_move_a_floor_are_accepted(built):
    """Absent = code default, and the non-defining arguments are not compared."""
    run_cfg = {**KW, 'delta_r_max': 0.30, 'seed': 0, 'device': 'cuda',
               'temperature_conditioning': True, 'ring_jitter_scale': 0.2,
               'smiles': 'CCCO', 'internal_prior_path': 'x.pt', 'prior_sample_size': 5}
    tab = bcr.load_references(built['out'], conditions_path=built['cond'],
                              energy_kwargs=run_cfg)
    assert len(tab) == len(SMIS)


def test_another_format_is_refused(built, tmp_path):
    blob = torch.load(built['out'], weights_only=False)
    blob['stamp']['format'] = 'conformer_references/1'
    old = tmp_path / 'old.pt'
    torch.save(blob, old)
    with pytest.raises(bcr.StaleReferencesError, match='format'):
        bcr.load_references(old, conditions_path=built['cond'], energy_kwargs=KW)


def test_a_partial_table_is_refused_unless_the_caller_accepts_it(built, tmp_path):
    out = tmp_path / 'subset.pt'
    bcr.build_references(built['cond'], out, KW, identifiers=[SMIS[0]], workers=0,
                         search=dict(n_uniform=3, n_prior=0, n_seeds=0, steps=20),
                         log=lambda *_: None)
    with pytest.raises(bcr.IncompleteReferencesError, match='2 of 3'):
        bcr.load_references(out, conditions_path=built['cond'], energy_kwargs=KW)
    tab = bcr.load_references(out, conditions_path=built['cond'], energy_kwargs=KW,
                              require_complete=False)
    assert tab.missing == sorted(SMIS[1:]) and list(tab.eval_refs(SMIS)) == [SMIS[0]]
    with bcr._float64():
        members = {i: bcr.build_member(i, KW) for i in SMIS}
    with pytest.raises(bcr.StaleReferencesError, match='no entry'):
        tab.verify(members)
    assert tab.verify(members, allow_missing=True) < 1e-9


# ---------------------------------------------------------------------- the table API


def test_verify_rescores_through_the_runs_members(built):
    tab = built['table']
    with bcr._float64():
        members = {i: bcr.build_member(i, KW) for i in SMIS}
        assert tab.verify(members) < 1e-9
        # float32 members, as the trainer builds them, pass the 1e-2 bar with room to spare
        m32 = {i: ConformerTorsions(smiles=i, device='cpu', dtype=torch.float32, **KW)
               for i in SMIS}
        assert tab.verify(m32) < 1e-3
        # the same stamp but another force field: only the re-score can see it
        other = {i: bcr.build_member(i, {**KW, 'force_field': 'reference'}) for i in SMIS}
    with pytest.raises(bcr.StaleReferencesError, match='re-score'):
        tab.verify(other)


def test_verify_refuses_a_tampered_floor(built):
    tab = bcr.load_references(built['out'], conditions_path=built['cond'], energy_kwargs=KW)
    tab.entries[SMIS[0]] = {**tab.entries[SMIS[0]], 'e_min': tab[SMIS[0]]['e_min'] + 0.05}
    with bcr._float64():
        members = {SMIS[0]: bcr.build_member(SMIS[0], KW)}
    with pytest.raises(bcr.StaleReferencesError):
        tab.verify(members)


def test_verify_refuses_a_member_of_another_width(built):
    with bcr._float64():
        members = {SMIS[0]: bcr.build_member('CCCO', KW)}      # k 30 against the entry's 21
    with pytest.raises(bcr.StaleReferencesError, match='member k 30'):
        built['table'].verify(members)


def test_eval_refs_feed_the_per_molecule_block(built):
    """The boundary with the consumer: refs in per_molecule_block's own form."""
    import energies.conformer_eval_metrics as cm

    tab, ident = built['table'], SMIS[0]
    refs = tab.eval_refs([ident, 'absent'])
    assert set(refs) == {ident} and set(refs[ident]) == {'e_min', 'basin_ref', 'target_tc'}
    with bcr._float64():
        member = bcr.build_member(ident, KW)
        g = torch.Generator().manual_seed(0)
        x = (torch.rand(40, member.ndim, generator=g, dtype=torch.float64) * 2 - 1) * 0.05
        e = bake_energies(member, x)
        row = cm.per_molecule_block({ident: member}, {ident: x}, {ident: e}, refs=refs,
                                    n_min=8)[ident]
    assert row['E/e_min_reference'] == pytest.approx(tab[ident]['e_min'])
    assert row['cover/n_modes'] == tab[ident]['n_modes']


def test_column_is_ordered_and_refuses_a_missing_identifier_or_a_none(built):
    tab = built['table']
    col = tab.column('e_min', SMIS[::-1])
    assert np.allclose(col, [tab[i]['e_min'] for i in SMIS[::-1]])
    with pytest.raises(KeyError):
        tab.column('e_min', SMIS + ['absent'])
    fresh = bcr.load_references(built['out'], conditions_path=built['cond'], energy_kwargs=KW)
    fresh.entries[SMIS[1]] = {**fresh.entries[SMIS[1]], 'n_accessible': None}
    with pytest.raises(ValueError, match="'n_accessible' is None for 1"):
        fresh.column('n_accessible', SMIS)


# ------------------------------------------------------------------- inputs and basins


def test_member_reference_mismatch_is_refused():
    with bcr._float64():
        member = bcr.build_member('CCO', KW)
    z, pos = np.asarray(member.spec.z), member.ref_pos.numpy().copy()
    assert bcr.check_member_matches(member, z, pos) == 0.0
    pos[3, 0] += 1e-3
    with pytest.raises(ValueError, match='reference conformer differs'):
        bcr.check_member_matches(member, z, pos)
    with pytest.raises(ValueError, match='atoms'):
        bcr.check_member_matches(member, z[::-1], member.ref_pos.numpy())


def test_compute_entry_builds_its_member_on_the_conditions_reference():
    """The member is built FROM the conditions file's stored reference, as the run builds it:
    a reference no embedding produces is reproduced exactly, and one that is not this
    molecule's atoms is refused."""
    member = _member('CCO')
    z = np.asarray(member.spec.z)
    pos = member.ref_pos.numpy().copy()
    pos[2, 1] += 1e-3
    assert bcr.compute_entry('CCO', 'CCO', z, pos, KW, SEARCH)['ref_pos_gap'] == 0.0
    with pytest.raises(ValueError, match='atomic numbers'):
        bcr.compute_entry('CCO', 'CCO', z[::-1].copy(), pos, KW, SEARCH)


def test_an_identifier_naming_two_molecules_is_refused(tmp_path):
    batch = collate_conditions([condition_from_smiles('CCO', identifier='X', **KW),
                                condition_from_smiles('COC', identifier='X', **KW)])
    path = save_condition_file(batch, str(tmp_path / 'dup.pt'))
    with pytest.raises(ValueError, match='names two molecules'):
        bcr.read_conditions(path)


def test_basin_block_keeps_the_mode_count_when_the_enumeration_is_skipped():
    with bcr._float64():
        member = bcr.build_member('CCCO', KW)
        full = bcr.basin_block(member)
        skipped = bcr.basin_block(member, max_modes=1)
    assert not full['basin_skipped'] and full['n_modes'] == len(full['basin_ref']['combos'])
    assert full['n_accessible'] == int(full['basin_ref']['accessible'].sum())
    assert skipped['basin_skipped'] and skipped['n_modes'] == full['n_modes']
    assert skipped['n_accessible'] is None and np.isnan(skipped['target_tc'])


# ---------------------------------------------------------------- resume and bounds


class _Killed(BaseException):
    """Not an Exception, so _run_job does not record it as a failure: it stops the build
    the way a kill does."""


def test_a_killed_build_resumes_the_molecules_it_finished(tmp_path, monkeypatch):
    cond = _write_conditions(tmp_path / 'conditions.pt', SMIS)
    out = tmp_path / 'references.pt'
    real, calls = bcr.compute_entry, []

    def dies_on_third(identifier, **kw):
        if identifier == SMIS[2]:
            raise _Killed()
        calls.append(identifier)
        return real(identifier=identifier, **kw)

    monkeypatch.setattr(bcr, 'compute_entry', dies_on_third)
    with pytest.raises(_Killed):
        bcr.build_references(cond, out, KW, search=SEARCH, workers=0, log=lambda *_: None)
    parts = out.with_name(out.name + '.parts')
    assert len(list(parts.glob('*.pt'))) == 2 and not out.exists()

    # a rerun under ANOTHER search, or ANOTHER member definition, is refused rather than
    # mixed with these parts
    with pytest.raises(ValueError, match='different stamp'):
        bcr.build_references(cond, out, KW, search={**SEARCH, 'n_uniform': 5}, workers=0,
                             log=lambda *_: None)
    with pytest.raises(ValueError, match='different stamp'):
        bcr.build_references(cond, out, {**KW, 'energy_clip': 250.0}, search=SEARCH,
                             workers=0, log=lambda *_: None)

    def counting(identifier, **kw):
        calls.append(identifier)
        return real(identifier=identifier, **kw)

    calls.clear()
    monkeypatch.setattr(bcr, 'compute_entry', counting)
    res = bcr.build_references(cond, out, KW, search=SEARCH, workers=0, log=lambda *_: None)
    assert calls == [SMIS[2]] and res['n_ok'] == 3 and not res['failures']
    assert not parts.exists()
    tab = bcr.load_references(out, conditions_path=cond, energy_kwargs=KW)
    assert tab.identifiers == SMIS


def test_the_pool_wait_is_bounded_and_a_rerun_completes(tmp_path):
    """A window in which no molecule finishes stops the pool, names every unfinished
    molecule and still writes the table; the rerun computes them on the worker path."""
    cond = _write_conditions(tmp_path / 'conditions.pt', SMIS[:2])
    out = tmp_path / 'references.pt'
    res = bcr.build_references(cond, out, KW, search=SEARCH, workers=1, molecule_timeout=0.01,
                               log=lambda *_: None)
    assert res['stalled'] and set(res['failures']) == set(SMIS[:2]) and res['n_ok'] == 0
    assert out.exists()
    with pytest.raises(bcr.IncompleteReferencesError, match='not completed'):
        bcr.load_references(out, conditions_path=cond, energy_kwargs=KW)
    res = bcr.build_references(cond, out, KW, search=SEARCH, workers=1, molecule_timeout=600,
                               log=lambda *_: None)
    assert not res['stalled'] and res['n_ok'] == 2 and not res['failures']
    # the worker path computes what this process computes
    with bcr._float64():
        m = bcr.build_member(SMIS[0], KW)
    here = bcr.compute_entry(SMIS[0], SMIS[0], np.asarray(m.spec.z), m.ref_pos.numpy(), KW,
                             SEARCH)
    assert res['entries'][SMIS[0]]['e_min'] == pytest.approx(here['e_min'], abs=1e-9)
