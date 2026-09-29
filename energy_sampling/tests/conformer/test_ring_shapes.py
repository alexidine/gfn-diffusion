"""Per-molecule ring shapes (energies/ring_shapes.py) and a ring's rotation about the bond
attaching it (ConformerTorsions.sample_prior_states), at level `full`, MMFF94, stereo lock 300.

What each test would catch:
  * shapes measured in the wrong atom order or deduplicated wrongly: cyclohexane must give its
    two chairs, lowest and mirror-signed, then only shapes >= 4 kcal/mol above them;
  * a five-ring collapsed to one shape;
  * shapes written into the wrong rows, or without the extras that close a ring the tree enters
    from outside: each draw's ring torsions must sit on the shape it took;
  * a shape carrying the other labelling or the other stereoisomer: the lock must stay at zero
    on grafted draws, and switching the trial off must make it fire;
  * the dihedral placing a ring's entry atom held at its reference, as at HEAD 48c1241: a ring
    carrying another ring must not lock out, and a chain bond next to a ring must turn;
  * any change to the draw where it should not change: an empty shape list, and a molecule with
    no ring entered from outside, draw bit for bit as before;
  * the rotation still held, or drawn so that the ring opens.
"""
import numpy as np
import pytest
import torch

import build_conformer_references as bcr
import energies.prior_baselines as pb
import energies.ring_metrics as rmet
import energies.ring_shapes as rs
from energies.conformer_torsions import ConformerTorsions

PRIOR_PATH = 'conformer_prior_v2.pt'
#: the production member (configs/conformer_mk.yaml's energy_config), float64 on the CPU
KW = dict(level='full', force_field='mmff', mmff_reference=True, seed=0, log_temperature=0.0,
          energy_clip=300.0, stereo_coeff=300.0)

CYCLOHEXANE = 'C1CCCCC1'
PROLINE = 'OC(=O)[C@@H]1CCCN1'                  # root in the ring, stereocentre on it
DIOXANONE = 'O=C1CO[C@@H](CO)CO1'               # stereocentre on the ring
HEXYL = 'CCCCCCC1CCCCC1'                        # the tree enters the ring from the chain
BICYCLOPROPYL = 'C1CC1C1CC1'                    # two ring systems joined by a bond
CYHEX_CYPR = 'C1CCC(CC1)C1CC1'                  # the same, one of them a cyclohexane
DECALIN = 'C1CC[C@H]2CCCC[C@@H]2C1'             # fused
NORBORNANE = 'C1C[C@H]2CC[C@@H]1C2'             # bridged
ETHYLBENZENE = 'CCc1ccccc1'                     # aromatic


@pytest.fixture(scope='module')
def prior():
    import pathlib
    if not pathlib.Path(PRIOR_PATH).exists():
        pytest.skip('{} missing'.format(PRIOR_PATH))
    return pb.load_prior(PRIOR_PATH)[0]


_MEMBERS, _SHAPES = {}, {}


def _member(smi):
    if smi not in _MEMBERS:
        _MEMBERS[smi] = bcr.build_member(smi, KW)
    return _MEMBERS[smi]


def _shapes(smi, prior):
    if smi not in _SHAPES:
        _SHAPES[smi] = rs.ring_shapes(_member(smi), prior, seed=bcr.molecule_seed(smi))
    return _SHAPES[smi]


def _graft(en, shp):
    """``[m, d]``: the reference state with each shape's theta and phi rows written."""
    assert en._transverse_t is None, 'polar rows only'
    r, th, ph = en.dof_from_state(torch.zeros(1, en.ndim, dtype=en.dtype))
    m = len(shp)
    R, TH, PH = r.repeat(m, 1), th.repeat(m, 1).clone(), ph.repeat(m, 1).clone()
    for col, (k, j) in enumerate(shp.rows):
        if k != 'r':
            (TH if k == 'theta' else PH)[:, j] = torch.as_tensor(shp.values[:, col],
                                                                 dtype=en.dtype)
    return en.state_from_dof(R, TH, PH)


def _ring_torsions_deg(en, x, cycles):
    return [np.degrees(t) for t in rmet.ring_torsions(en, x, cycles)]


def _wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def _circ_sd_deg(rad):
    z = abs(np.exp(1j * np.asarray(rad)).mean())
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(z, 1e-12)))))


def _no_rotation(monkeypatch):
    """Switch the rotation off: its rows held with the other extras, as before it existed."""
    real = ConformerTorsions.ring_blocks

    def held(self, prior):
        out = real(self, prior)
        for rec in self.ring_block_info:
            rec['rotation_rows'] = []
        return out

    monkeypatch.setattr(ConformerTorsions, 'ring_blocks', held)


def _hold_entry(monkeypatch):
    """Hold the dihedral placing each system's entry atom again, as HEAD 48c1241 did: put it
    back among the block's extras."""
    real = ConformerTorsions.ring_blocks

    def held(self, prior):
        out = real(self, prior)
        return [(o, b, e + [('phi', j) for j in rec['entry_rows']])
                for (o, b, e), rec in zip(out, self.ring_block_info)]

    monkeypatch.setattr(ConformerTorsions, 'ring_blocks', held)


# ------------------------------------------------------------------ shapes are measured

def test_cyclohexane_gives_its_two_chairs_first_then_higher_shapes_only(prior):
    """The two chairs, deduplicated to one shape each, lowest in energy and mirror-signed."""
    en = _member(CYCLOHEXANE)
    (shp,) = _shapes(CYCLOHEXANE, prior)
    assert shp.reason is None and len(shp) >= 2, (shp.reason, len(shp), shp.info)
    assert shp.info['n_embedded'] == 64 and shp.info['n_candidates'] == int(shp.count.sum())
    assert np.all(np.diff(shp.mmff_energy) >= 0)
    (cyc,) = rmet.ring_cycles(en)
    (t,) = _ring_torsions_deg(en, _graft(en, shp), [cyc])
    sign = np.sign(t)
    chair = (np.abs(t) > 40).all(1) & (np.abs(t) < 70).all(1) & (sign * np.roll(sign, 1, 1) < 0).all(1)
    assert chair[:2].all() and chair.sum() == 2, np.round(t)
    assert np.array_equal(sign[0], -sign[1]), 'the two chairs are not the two mirror-signed ones'
    assert (shp.mmff_energy[2:] - shp.mmff_energy[0] >= 4.0).all(), shp.mmff_energy
    # deduplicated: no two kept shapes agree on every phi row to DEDUP_DEG
    phi = [c for c, (k, _) in enumerate(shp.rows) if k == 'phi']
    v = np.degrees(shp.values[:, phi])
    for i in range(len(shp)):
        for j in range(i):
            assert np.abs(_wrap_deg(v[i] - v[j])).max() > rs.DEDUP_DEG, (i, j)


def test_a_five_ring_keeps_several_distinct_shapes(prior):
    (shp,) = _shapes(PROLINE, prior)
    assert shp.reason is None and len(shp) >= 3, (len(shp), shp.info)
    phi = [c for c, (k, _) in enumerate(shp.rows) if k == 'phi']
    v = np.degrees(shp.values[:, phi])
    d = [np.abs(_wrap_deg(v[i] - v[j])).max() for i in range(len(shp)) for j in range(i)]
    assert min(d) > rs.DEDUP_DEG, d


def test_a_shape_carries_the_rows_that_set_the_ring_and_not_the_rigid_ones(prior):
    """Entered from the chain at n1 from c_out: the angle and dihedral rows at n1 set the ring,
    the bond, angle and dihedral that place n1 itself only move it whole. The shape also carries
    n1's other children, the rest of the rotation's sibling group."""
    en = _member(HEXYL)
    (order, _, extra), = en.ring_blocks(prior)
    (rec,) = en.ring_block_info
    atoms = set(rec['atoms'])
    bi, ti = np.asarray(en.spec.bond_index), np.asarray(en.spec.torsion_index)
    n1 = [a for a in atoms if a > 0 and int(bi[a - 1, 0]) not in atoms]
    assert len(n1) == 1, 'the premise: the tree enters the ring from outside, once'
    n1 = n1[0]
    # the rows placing slot n1: the bond and angle are rigid extras; the dihedral is no extra
    rigid = {('r', n1 - 1), ('theta', n1 - 2)}
    assert rigid <= set(extra) and ('phi', n1 - 3) not in extra
    assert rec['entry_rows'] == [n1 - 3]
    got = rs.internal_extra_rows(en, extra, atoms)
    assert set(got) == set(extra) - rigid, (sorted(got), sorted(extra))
    assert rec['rotation_rows']
    for j in rec['rotation_rows']:                    # the rotation rows: about (c_out, n1)
        assert int(ti[j, 2]) == n1 and int(ti[j, 1]) == int(bi[n1 - 1, 0])
    group = {j for g in en.torsion_groups() if set(g) & set(rec['rotation_rows']) for j in g}
    others = group - set(rec['rotation_rows'])
    assert others and all(int(ti[j, 2]) == n1 and int(ti[j, 3]) not in atoms for j in others)
    (shp,) = _shapes(HEXYL, prior)
    assert shp.rows[:shp.n_order] == [(str(k), int(j)) for k, j in order]
    assert set(shp.rows[shp.n_order:]) == set(got) | {('phi', j) for j in others}


# ---------------------------------------------------------------------- draws use them

@pytest.mark.parametrize('smi', [PROLINE, HEXYL, DECALIN])
def test_each_draw_sits_on_the_shape_it_took(prior, smi):
    en = _member(smi)
    shapes = _shapes(smi, prior)
    n = 400
    x, st = en.sample_prior_states(prior, n, np.random.default_rng(1), report=False,
                                   ring_shapes=shapes)
    cycles = [c for c in rmet.ring_cycles(en) if not en.atom_is_aromatic[list(c)].all()]
    (shp,) = shapes                                   # one ring system each
    rec = st['ring_shapes'][0]
    assert rec['available'] == len(shp) and int(rec['drawn'].sum()) == n
    assert (rec['drawn'] > 0).all(), 'a stored shape was never drawn'
    got = np.concatenate(_ring_torsions_deg(en, x, cycles), 1)                  # [n, T]
    want = np.concatenate(_ring_torsions_deg(en, _graft(en, shp), cycles), 1)  # [m, T]
    dist = np.abs(_wrap_deg(got[:, None] - want[None])).max(2)                  # [n, m]
    dev = dist[np.arange(n), rec['pick']]
    # the jitter (ring_jitter_scale of the thermal width on every row) moves a ring torsion by
    # about 1-2 deg, up to 8 over 400 draws; distinct shapes differ by tens of degrees
    assert np.median(dev) < 3.0 and np.quantile(dev, 0.99) < 10.0, (np.median(dev), dev.max())
    # where the shape a draw took stands clear of every other shape, it is the nearest one
    sep = np.abs(_wrap_deg(want[:, None] - want[None])).max(2) + 1e9 * np.eye(len(shp))
    clear = sep.min(1)[rec['pick']] > 20.0
    assert clear.any()
    assert (dist.argmin(1)[clear] == rec['pick'][clear]).all()
    assert st['n_ring_shaped'] == 1
    assert st['closure_sigma'] < 3.0, st['closure_sigma']


def _locked_out(en, x):
    """Per draw: some locked element on the wrong side."""
    with torch.no_grad():
        pos = en.build_positions(x).reshape(x.shape[0], -1, 3)
        _, _, sign, _ = en.stereo.tensors(pos.device, pos.dtype)
        return (sign * en.stereo.values(pos) <= 0).any(1).numpy(), pos


@pytest.mark.parametrize('smi', [PROLINE, DIOXANONE, BICYCLOPROPYL, CYHEX_CYPR])
def test_the_lock_stays_at_zero_on_grafted_draws(prior, smi):
    en = _member(smi)
    shapes = _shapes(smi, prior)
    assert any(len(s) for s in shapes)
    n = 512
    x, st = en.sample_prior_states(prior, n, np.random.default_rng(2), report=False,
                                   ring_shapes=shapes)
    wrong, pos = _locked_out(en, x)
    assert not wrong.any(), f'{int(wrong.sum())} of {n} grafted draws are locked out'
    if smi in (PROLINE, DIOXANONE):                  # the stereocentre-bearing rings
        lock = en.stereo.lock_energy(pos, en.stereo_coeff)
        assert float(lock.max()) == 0.0, float(lock.max())


def test_a_ring_carrying_another_ring_is_not_locked_out(prior, monkeypatch):
    """cyclohexylcyclopropane. The dihedral placing the cyclopropane's entry atom on the
    cyclohexane is drawn with its sibling group, so that atom follows whatever shape the
    cyclohexane takes. Held at its reference, as at HEAD, it stayed put while the cyclohexane
    flipped: 43% of unshaped draws locked out, and the trial dropped 22 of 64 cyclohexane
    embeddings. Checked unshaped, shaped, and against the held row."""
    en = _member(CYHEX_CYPR)
    shapes = _shapes(CYHEX_CYPR, prior)
    assert shapes[0].info['n_failed_trial'] == 0, shapes[0].info
    n = 2048
    for s in (None, shapes):
        x, _ = en.sample_prior_states(prior, n, np.random.default_rng(2), report=False,
                                      ring_shapes=s)
        wrong, _ = _locked_out(en, x)
        assert wrong.mean() <= 0.001, ('shaped' if s else 'unshaped', wrong.mean())
    _hold_entry(monkeypatch)
    x, _ = en.sample_prior_states(prior, n, np.random.default_rng(2), report=False)
    assert _locked_out(en, x)[0].mean() > 0.2, 'the held row no longer locks out; the premise'


@pytest.mark.parametrize('arm', ['unshaped', 'shaped'])
def test_the_chain_bond_next_to_a_ring_entered_from_the_chain_turns(prior, arm, monkeypatch):
    """hexylcyclohexane: the dihedral placing the ring's entry atom turns about the chain bond
    one further out, and spreads over the circle; held, as at HEAD, it stayed within a degree."""
    en = _member(HEXYL)
    en.ring_blocks(prior)
    (j,) = en.ring_block_info[0]['entry_rows']
    assert j not in en.held_phi_rows(), 'the premise: a rotor, not an improper'
    a, b, c, d = (int(v) for v in np.asarray(en.spec.torsion_index)[j])
    from mxtaltools.conformers.geometry import dihedral
    shapes = _shapes(HEXYL, prior) if arm == 'shaped' else None

    def spread():
        x, _ = en.sample_prior_states(prior, 400, np.random.default_rng(0), report=False,
                                      ring_shapes=shapes)
        with torch.no_grad():
            p = en.build_positions(x).reshape(400, -1, 3)
        return _circ_sd_deg(dihedral(p[:, a], p[:, b], p[:, c], p[:, d]).numpy())

    sd = spread()
    _hold_entry(monkeypatch)
    sd0 = spread()
    assert sd > 60.0 and sd0 < 5.0, (sd, sd0)


@pytest.mark.parametrize('smi', [PROLINE, BICYCLOPROPYL, CYHEX_CYPR, HEXYL, DECALIN, NORBORNANE])
def test_closure_stays_bounded_with_the_entry_row_released(prior, smi, monkeypatch):
    """Both arms under 3 bond-sigma, and the unshaped draw's closure within 1.1 times, plus
    0.002 A, of the draw that holds the entry row: the released row moves no distance inside
    the ring it places."""
    en = _member(smi)
    got = {}
    for arm, s in (('unshaped', None), ('shaped', _shapes(smi, prior))):
        _, st = en.sample_prior_states(prior, 400, np.random.default_rng(0), report=False,
                                       ring_shapes=s)
        assert st['closure_sigma'] < 3.0, (arm, st['closure_sigma'])
        got[arm] = st['closure_err']
    _hold_entry(monkeypatch)
    _, st0 = en.sample_prior_states(prior, 400, np.random.default_rng(0), report=False)
    assert got['unshaped'] <= 1.1 * st0['closure_err'] + 0.002, (got, st0['closure_err'])


def test_without_the_trial_a_graph_equivalent_labelling_locks_draws_out(prior, monkeypatch):
    """The negative control of the lock test: the trial is what keeps it at zero."""
    monkeypatch.setattr(rs, '_trial', lambda member, prior, blocks, bk, shp, seed, fixed:
                        np.ones(len(shp), dtype=bool))
    en = _member(BICYCLOPROPYL)
    shapes = rs.ring_shapes(en, prior, seed=bcr.molecule_seed(BICYCLOPROPYL))
    n = 512
    x, _ = en.sample_prior_states(prior, n, np.random.default_rng(2), report=False,
                                  ring_shapes=shapes)
    with torch.no_grad():
        lock = en.stereo.lock_energy(en.build_positions(x).reshape(n, -1, 3), en.stereo_coeff)
    assert float((lock > 1.0).float().mean()) > 0.2


def test_an_embedding_of_the_other_stereoisomer_is_dropped(prior, monkeypatch):
    real = rs._embed

    def mirrored(template, n, seed, timeout_s):
        pos, e, nc = real(template, n, seed, timeout_s)
        return pos * np.array([-1.0, 1.0, 1.0]), e, nc

    monkeypatch.setattr(rs, '_embed', mirrored)
    (shp,) = rs.ring_shapes(_member(PROLINE), prior, n_embed=16, seed=3)
    assert shp.reason == 'none_kept' and len(shp) == 0
    assert shp.info['n_other_stereo'] == shp.info['n_embedded'] == 16


def test_when_every_embedding_fails_the_draw_is_the_unshaped_one(prior, monkeypatch):
    en = _member(HEXYL)
    monkeypatch.setattr(rs, '_embed', lambda template, n, seed, timeout_s:
                        (np.zeros((0, en.spec.n_atoms, 3)), np.zeros(0), 0))
    shapes = rs.ring_shapes(en, prior, seed=5)
    assert [s.reason for s in shapes] == ['no_embedding'] and len(shapes[0]) == 0
    ra, rb = np.random.default_rng(4), np.random.default_rng(4)
    xa, sa = en.sample_prior_states(prior, 64, ra, report=False, ring_shapes=shapes)
    xb, sb = en.sample_prior_states(prior, 64, rb, report=False)
    assert torch.equal(xa, xb) and np.array_equal(sa['dof'], sb['dof'])
    assert ra.random() == rb.random()


def test_an_aromatic_ring_gets_no_shapes(prior):
    (shp,) = _shapes(ETHYLBENZENE, prior)
    assert shp.reason == 'aromatic' and len(shp) == 0


def test_shapes_from_another_chart_are_refused(prior):
    en = _member(PROLINE)
    with pytest.raises(ValueError, match='ring_blocks'):
        en.sample_prior_states(prior, 8, np.random.default_rng(0), report=False,
                               ring_shapes=_shapes(BICYCLOPROPYL, prior))
    (shp,) = _shapes(CYCLOHEXANE, prior)
    with pytest.raises(ValueError, match='another chart'):
        en.sample_prior_states(prior, 8, np.random.default_rng(0), report=False,
                               ring_shapes=[shp])


def test_a_dummy_frame_row_is_measured_against_its_dummy():
    """_state_of_positions, which ring_shapes measures through: the member's own reference
    must measure to the zero state, a dummy-frame row included."""
    from mxtaltools.conformers.builder import collate
    en = _member('CC#CC1CC1')
    assert en._dummy_t is not None, 'the premise: a dummy-frame row'
    x = bcr._state_of_positions(en, collate([en.spec]), en.mol.GetConformer().GetPositions())
    assert float(x.abs().max()) < 1e-9


# ------------------------------------------------ the rotation about the attaching bond

@pytest.mark.parametrize('smi', [CYCLOHEXANE, PROLINE, ETHYLBENZENE, DECALIN, NORBORNANE])
def test_no_ring_entered_from_outside_draws_bitwise_as_before(prior, smi, monkeypatch):
    """ring_shapes=None where no ring has a rotation row: the draw, its DoF and the generator's
    next value equal those with the rotation switched off, the pre-rotation draw."""
    en = _member(smi)
    en.ring_blocks(prior)
    assert not any(r['rotation_rows'] for r in en.ring_block_info), 'the premise'
    ra = np.random.default_rng(3)
    xa, sa = en.sample_prior_states(prior, 128, ra, report=False)
    _no_rotation(monkeypatch)
    rb = np.random.default_rng(3)
    xb, sb = en.sample_prior_states(prior, 128, rb, report=False)
    assert torch.equal(xa, xb) and np.array_equal(sa['dof'], sb['dof'])
    assert ra.random() == rb.random()


@pytest.mark.parametrize('smi', [BICYCLOPROPYL, CYHEX_CYPR, HEXYL])
def test_a_ring_entered_from_outside_turns_about_its_attaching_bond(prior, smi, monkeypatch):
    """The dihedral about the attaching bond spreads over the circle, where the hold kept it
    within a few degrees, and closure is no worse than the held draw's."""
    en = _member(smi)
    en.ring_blocks(prior)
    rot = [j for r in en.ring_block_info for j in r['rotation_rows']]
    assert rot, 'the premise: a ring the tree enters from outside'
    a, b, c, d = (int(v) for v in np.asarray(en.spec.torsion_index)[rot[0]])
    from mxtaltools.conformers.geometry import dihedral

    def draw():
        x, st = en.sample_prior_states(prior, 400, np.random.default_rng(0), report=False)
        with torch.no_grad():
            p = en.build_positions(x).reshape(400, -1, 3)
        return _circ_sd_deg(dihedral(p[:, a], p[:, b], p[:, c], p[:, d]).numpy()), st

    sd, st = draw()
    _no_rotation(monkeypatch)
    sd0, st0 = draw()
    assert st['n_ring_rotations'] >= 1 and st0['n_ring_rotations'] == 0
    assert sd > 60.0 and sd0 < 5.0, (sd, sd0)
    assert st['closure_sigma'] < 3.0
    assert st['closure_err'] <= 1.1 * st0['closure_err'] + 0.002, (st['closure_err'],
                                                                   st0['closure_err'])
