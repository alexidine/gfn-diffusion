"""Gates for the RING BLOCK of the reference table (energies/prior_baselines.py).

WHAT THESE ARE FOR. The table's prior arm used to call
``sample_prior_states(..., joint_rings=False)`` while the closure monitor was gated on the
ring-system count -- which is zero on exactly that path. So the arm sampled every ring DoF
independently, violated closure by ~75 bond-sigma, and reported closure_err 0.000 A. The
table's poor ring numbers described a path the sampler does not use, and nothing in the
suite could tell.

Every gate below is therefore written so that re-introducing that failure produces a
FAILURE rather than a quieter pass:

  1. The two arms must reach DIFFERENT sampling paths, asserted on the measurement.
  2. The negative control must measurably WORSEN closure -- and the control is the only
     thing that can show the ring columns are live at all.
  3. The closure monitor must be gated on the MOLECULE, not on joint_rings. This is the
     specific bug, pinned directly.
  4. A ring statistic over an EMPTY population must not read as a pass.
  5. The four ring classes must stay apart: banked, held-by-design, unsupported, stale.
  6. Density-dependent ring numbers must stay refused, not approximated.

Deliberately small: three molecules, n=64, one or two seeds. These test the BENCHMARK's
logic, not the prior's quality -- that is what the benchmark itself is for.

    python -m pytest -q test_prior_baselines_rings.py
"""
import numpy as np
import pytest
import torch

from energies.conformer_torsions import ConformerTorsions
import energies.prior_baselines as pb
import energies.ring_metrics as rmet

PRIOR_PATH = 'conformer_prior_v2.pt'
N = 64

# One of each contract. A set where every ring happened to be banked would report a
# generic pass and could not tell "held by design" from "no bank resolved".
SATURATED = ('ethylcyclohexane', 'CCC1CCCCC1')          # banked pucker subspace
AROMATIC = ('ethylbenzene', 'CCc1ccccc1')               # held planar BY DESIGN
HETERO = ('methyltetrahydropyran', 'CC1CCCCO1')         # substituted heterocycle, banked
UNSUPPORTED = ('proline', 'OC(=O)C1CCCN1')              # saturated, no bank resolves
RING_SET = [SATURATED, AROMATIC, HETERO, UNSUPPORTED]


@pytest.fixture(scope='module')
def prior():
    import pathlib
    if not pathlib.Path(PRIOR_PATH).exists():
        pytest.skip('{} missing'.format(PRIOR_PATH))
    p, ver = pb.load_prior(PRIOR_PATH)
    assert ver >= 2, 'the ring set needs a v2 prior; v1 resolves no ring key'
    return p


def _en(smi, level='full'):
    return ConformerTorsions(smiles=smi, level=level, force_field='mmff',
                             log_temperature=0.0, device='cpu')


# ------------------------------------------------------ 1/2: the control must separate

@pytest.mark.parametrize('name,smi', RING_SET)
def test_negative_control_worsens_closure(prior, name, smi):
    """The arms must reach different paths, and OFF must be measurably WORSE.

    Asserted on closure in bond-sigma rather than on a flag, because a flag can be wired
    correctly while the switch never reaches the sampler. The bar is 4x AND above 3 sigma
    -- 3 is the point where the ring is visibly open, so the control has to be a genuinely
    broken proposal rather than a slightly wider one.
    """
    en = _en(smi)
    on, off = [], []
    for seed in (0, 1):
        for jr, acc in ((True, on), (False, off)):
            x, st = en.sample_prior_states(prior, N, np.random.default_rng(seed),
                                           report=False, joint_rings=jr)
            acc.append(rmet.closure_error(en, x)[1])
    m_on, m_off = float(np.mean(on)), float(np.mean(off))
    assert m_off > 4 * m_on and m_off > 3.0, (
        '{}: joint-ring OFF barely changed closure ({:.2f} vs {:.2f} bond-sigma). Either '
        'both arms are on the same sampling path, or the closure measurement is not live '
        '-- and every other ring number in the table is then unsupported.'
        .format(name, m_off, m_on))
    assert m_on < 3.0, ('{}: the rings-ON arm is itself above 3 bond-sigma ({:.2f}), so '
                        'the ring is visibly open on the path the table calls correct'
                        .format(name, m_on))


def test_arms_are_not_the_same_path(prior):
    """The benchmark's own arm wiring, not the sampler's.

    A benchmark that passed joint_rings=True to both arms would produce two identical
    columns, and two agreeing columns read as corroboration. This drives the real cell()
    and requires the ring columns to differ.
    """
    en = _en(SATURATED[1])
    zero = 0.0
    cells = {s: pb.cell(en, prior, s, 0, N, [0], zero)
             for s in ('prior-rings-on', 'prior-rings-off')}
    a, b = cells['prior-rings-on']['rings'], cells['prior-rings-off']['rings']
    assert a is not None and b is not None
    assert b['closure_err_sigma'] > 4 * a['closure_err_sigma'], (
        'the two benchmark arms produced the same closure; they are wired to the same '
        'sampling path and the negative control is decorative')
    # and the independent-DoF count must show WHICH path each took
    assert a['n_ring_dof_independent'] == 0, a['n_ring_dof_independent']
    assert b['n_ring_dof_independent'] > 0, (
        'the OFF arm reports zero independently-sampled ring DoF, so it did not actually '
        'disable joint ring sampling')


# ------------------------------------------------- 3: the specific bug, pinned directly

def test_closure_monitor_runs_with_joint_rings_off(prior):
    """THE REGRESSION. The monitor must key on the MOLECULE, not on joint_rings.

    It used to be gated on ``stats['n_rings']``, which is 0 whenever joint rings are off --
    so the one configuration whose closure is catastrophic reported 0.000 A and read as
    perfect. A diagnostic that goes quiet exactly where the thing it monitors fails is
    worse than no diagnostic.
    """
    en = _en(SATURATED[1])
    _, st = en.sample_prior_states(prior, N, np.random.default_rng(0),
                                   report=False, joint_rings=False)
    assert st['n_rings'] == 0, 'joint rings off should process no ring block'
    assert st['n_closure_bonds'] > 0, 'the molecule has closure bonds; the monitor must see them'
    assert np.isfinite(st['closure_err']) and st['closure_err'] > 1.0, (
        'closure_err is {} with joint rings OFF -- the monitor is gated on the ring path '
        'again, so the negative control silently reports a perfect ring'
        .format(st['closure_err']))
    assert st['joint_rings'] is False


def test_acyclic_closure_is_nan_not_zero(prior):
    """An acyclic molecule has no closure error to report, and 0.0 is not that.

    0.0 is a passing measurement of nothing, and it is what let an empty ring population
    read as a healthy one.
    """
    en = _en('CCCO')
    _, st = en.sample_prior_states(prior, N, np.random.default_rng(0), report=False)
    assert st['n_closure_bonds'] == 0
    assert np.isnan(st['closure_err']) and np.isnan(st['closure_sigma'])


# ---------------------------------------------- 4: an empty population is not a pass

def test_empty_ring_population_is_not_a_pass(prior):
    """Ring statistics must be absent, and identifiable as absent, on an acyclic molecule."""
    en = _en('CCCO')
    c = pb.cell(en, prior, 'prior-rings-on', 0, N, [0], 0.0)
    assert c['rings'] is None, 'an acyclic molecule produced ring statistics'
    # and the control arm is reported N/A rather than silently duplicating the ON arm
    off = pb.cell(en, prior, 'prior-rings-off', 0, N, [0], 0.0)
    assert off.get('inapplicable') and 'acyclic' in off['blocked']

    # the ring table must SAY the population is empty rather than printing a clean table
    txt = pb.fmt_rings('full', [{'molecule': 'propanol', 'd': 30, 'has_rings': False,
                                 'arms': {}}])
    assert 'EMPTY population' in txt and 'not a pass' in txt

    # the verdict must refuse to certify when no molecule ran both arms
    assert 'NOT demonstrated live' in pb.ring_verdict('full', [])


def test_ring_measurements_carry_the_population_guard(prior):
    en = _en(SATURATED[1])
    x, st = en.sample_prior_states(prior, N, np.random.default_rng(0), report=False)
    m = rmet.ring_measurements(en, x, prior, st)
    assert m['n_ring_systems'] == 1 and m['n_ring_cycles'] == 1
    assert m['n_closure_bonds'] >= 1
    assert m['n_ring_block_dof'] > 0 and m['n_ring_extra_dof'] > 0, (
        'ethylcyclohexane has ring-positioning DoF outside the block; a zero here means '
        'the extras are not being counted and letting them float re-opens the ring')


# ------------------------------------- 5: the four classes must not collapse into one

def test_ring_classes_stay_distinct(prior):
    """banked / held-by-design / unsupported are three different states, not one pass."""
    got = {}
    for name, smi in RING_SET:
        recs = rmet.classify_ring_blocks(_en(smi), prior)
        got[name] = sorted(r['ring_class'] for r in recs)
    assert got[SATURATED[0]] == ['banked_modes'], got
    assert got[HETERO[0]] == ['banked_modes'], got
    assert got[AROMATIC[0]] == ['held_aromatic'], got
    assert got[UNSUPPORTED[0]] == ['held_unsupported'], got
    assert len({tuple(v) for v in got.values()}) == 3, (
        'the ring set no longer distinguishes banked from held-by-design from '
        'unsupported, so the class column cannot fail: ' + repr(got))


def test_mixed_molecule_reports_each_ring_against_its_own_contract(prior):
    """One molecule, two rings, two different contracts -- reported separately.

    Pooling them would let a planar aromatic ring's zero pucker read as a saturated ring's
    failure to sample, or the reverse.
    """
    en = _en('C1CCC(CO1)c1ccccc1')
    recs = rmet.classify_ring_blocks(en, prior)
    assert sorted(r['ring_class'] for r in recs) == ['banked_modes', 'held_aromatic'], recs
    x, st = en.sample_prior_states(prior, N, np.random.default_rng(0), report=False)
    sat, aro = rmet.pucker_occupancy(en, x)
    assert len(sat) == 1 and len(aro) == 1, (len(sat), len(aro))
    assert aro[0]['median_abs_torsion_deg'] < 5.0, (
        'the aromatic ring is not planar: {:.1f} deg -- it is held planar BY DESIGN, so '
        'this is a broken contract, not a diversity result'
        .format(aro[0]['median_abs_torsion_deg']))
    assert sat[0]['median_abs_torsion_deg'] > 20.0, (
        'the saturated ring is nearly planar, so its pucker is not being sampled')


def test_stale_prior_is_distinguished_and_refused():
    """'stale prior' must not read as 'this molecule has no bank'.

    Both end in held_unsupported, and only the flag says which -- so the loader refuses a
    stale prior outright rather than letting the ring block quietly describe nothing.
    """
    import pathlib
    if not pathlib.Path('conformer_prior.pt').exists():
        pytest.skip('conformer_prior.pt missing')
    with pytest.raises(SystemExit) as ex:
        pb.load_prior('conformer_prior.pt')
    assert 'ring_sig_version' in str(ex.value)
    p, ver = pb.load_prior('conformer_prior.pt', allow_stale=True)
    assert ver == 1
    en = _en(SATURATED[1])
    recs = rmet.classify_ring_blocks(en, p)
    assert all(r['stale_prior'] for r in recs)
    assert [r['ring_class'] for r in recs] == ['held_unsupported'], (
        'a stale prior must fall through to held_unsupported; if it banks, the signature '
        'check is not doing anything')


# ------------------------------------------- 6: the density limit must stay a refusal

def test_ring_density_stays_unavailable(prior):
    """No ring ESS, no D_avoidable, no IS log Z -- and no substitute for them."""
    en = _en(SATURATED[1])
    x, st = en.sample_prior_states(prior, N, np.random.default_rng(0), report=False)
    with pytest.raises(NotImplementedError):
        en.prior_log_prob(prior, st['dof'])
    m = rmet.ring_measurements(en, x, prior, st)
    assert 'UNAVAILABLE' in m['ring_density']
    for forbidden in ('ess', 'ess_fitted', 'D_avoidable', 'eta', 'log_z'):
        assert forbidden not in m, (
            '{} appeared in the ring measurements; a ring block has no matched density, so '
            'this can only be the acyclic density or an independent marginal wearing a '
            'ring label'.format(forbidden))
    # ...and the per-molecule report must SKIP rather than return a number
    rep = __import__('energies.prior_diagnostics', fromlist=['x']).prior_report(
        en, prior, n=N, oracle=False, blocks=False)
    assert 'skipped' in rep and 'ess_fitted' not in rep


# --------------------------------------------- the tier contract the table depends on

def test_ring_comparison_inapplicable_at_torsion(prior):
    """At a collective tier the control cannot run, and must say so rather than fake one.

    draw_states has no joint_rings switch and freezes ring DoF at the reference, so an
    'off' arm there would be the same draw under a different name.
    """
    en = _en(SATURATED[1], level='torsion')
    assert en.collective
    status = pb.ring_arm_status(en, 'prior-rings-off')
    assert status is not None and 'collective' in status
    c = pb.cell(en, prior, 'prior-rings-off', 0, N, [0], 0.0)
    assert c.get('inapplicable') and c['budget'] == 'N/A' and c.get('rings') is None
    # the ON arm still runs there
    assert pb.ring_arm_status(en, 'prior-rings-on') is None
    with pytest.raises(NotImplementedError):
        pb.draw_prior(en, prior, 8, 0, joint_rings=False)


@pytest.mark.parametrize('tier', ['dihedral', 'flex', 'full'])
def test_ring_columns_run_at_every_selection_tier(prior, tier):
    """Ring energy/closure/pucker must run at every tier where the control applies.

    Per tier, not pooled: freezing a DoF gives a different distribution, so a ring result
    at 'flex' does not describe one at 'full'.
    """
    en = _en(SATURATED[1], level=tier)
    assert pb.ring_arm_status(en, 'prior-rings-off') is None, tier
    c = pb.cell(en, prior, 'prior-rings-on', 0, N, [0, 1], 0.0)
    g = c['rings']
    assert g['n_ring_systems'] == 1 and np.isfinite(g['closure_err_sigma'])
    assert g['n_seeds'] == 2 and c['seeds'] == [0, 1]
    assert len(g['saturated']) == 1 and g['saturated'][0]['n_basins'] >= 1


def test_pucker_identity_has_one_definition():
    """build_ring_banks must label a basin the same way the benchmark counts occupancy.

    Two copies of the quantisation would let "how many basins exist" and "how many were
    reached" answer different questions while both looked right.
    """
    import build_ring_banks as brb
    t = [0.9, -0.9, 0.05, -0.05, 1.2, -1.2]
    ids, labels = rmet.basin_labels(np.array([t]))
    assert labels[ids[0]] == rmet.basin_label(t)
    assert rmet.basin_label(t) == (1, -1, 0, 0, 1, -1)
    assert brb.basin_key.__module__ == 'build_ring_banks'
    src = __import__('inspect').getsource(brb.basin_key)
    assert 'basin_label' in src, (
        'build_ring_banks.basin_key no longer delegates to ring_metrics.basin_label; '
        'pucker identity has two definitions again')


# ------------------------------- the ring-frame follower rule (substituents on ring atoms)

@pytest.mark.parametrize('name,smi', [SATURATED, AROMATIC, HETERO, UNSUPPORTED])
def test_ring_frame_groups_are_not_drawn_from_a_rotamer_histogram(prior, name, smi):
    """A torsion group about a RING bond must never take a free-rotor draw.

    Groups are keyed on the central bond, and the "ring member leads" rule only fires when
    the group HOLDS a ring-placed row. The last ring atom in placement order has both ring
    neighbours already placed, so its substituents form a group with no ring member -- and
    the default branch draws the leader from a rotamer histogram keyed on the central bond
    type. That bond is a ring bond and does not rotate. Measured on cyclohexane before the
    fix: the leader sat 81 deg from the reference while correctly-mixed groups sat within
    5 deg, and the redundant angles at that carbon carried most of the angle strain.

    Asserted on the ANGLE the tree does not expose, not on the dihedral, because that is
    the quantity being destroyed and it is what the force field scores.
    """
    from mxtaltools.conformers.geometry import bond_angle
    en = _en(smi)
    ring_rows = set()
    for order, _, extra in en.ring_blocks(prior):
        for k, j in order:
            ring_rows.add(en._global_row(k, j))
        for k, j in extra:
            ring_rows.add(en._global_row(k, j))
    fg = en.ring_frame_groups(ring_rows)
    assert fg, ('{} has no ring-bond group without a ring member, so it cannot exercise '
                'this rule -- pick another molecule'.format(name))

    x, st = en.sample_prior_states(prior, 256, np.random.default_rng(0), report=False)
    assert st['n_ring_frame_groups'] == len(fg)

    # redundant graph angles whose apex is a ring atom: not tree DoF, so their value is
    # implied by the sampled theta/phi and nothing else measures them
    tree_ang = {(min(int(a), int(c)), int(b), max(int(a), int(c)))
                for a, b, c in np.asarray(en.spec.angle_index)}
    n = 256
    _, ff = en._batch(n)
    pos = en.build_positions(x)
    th = bond_angle(pos[ff.angle_index[:, 0]], pos[ff.angle_index[:, 1]],
                    pos[ff.angle_index[:, 2]])
    dev = np.abs(np.degrees((th - ff.theta0).numpy()).reshape(n, -1))
    ai = ff.angle_index.reshape(n, -1, 3)[0].numpy() % en.spec.n_atoms
    red = [i for i, r in enumerate(ai)
           if (min(int(r[0]), int(r[2])), int(r[1]), max(int(r[0]), int(r[2]))) not in tree_ang
           and en.atom_in_ring[int(r[1])]]
    assert red, '{}: no redundant ring-apex angle to measure'.format(name)
    got = float(dev[:, red].mean())
    # tree angles ARE sampled at their own thermal width and are the honest yardstick: a
    # redundant angle should be the same order, not three times it
    tree_idx = [i for i in range(len(ai)) if i not in red]
    ref = float(dev[:, tree_idx].mean())
    assert got < 2.0 * ref, (
        '{}: redundant ring-apex angles deviate {:.2f} deg against {:.2f} deg for the '
        'tree angles the sampler actually draws. A ring-bond group is being treated as '
        'freely rotatable again.'.format(name, got, ref))


def test_ring_frame_correction_needs_the_ring_geometry_not_the_reference(prior):
    """Pinning those rows to the REFERENCE dihedral is not the fix, and must not pass.

    The substituents have to follow where the ring block actually put the ring, which is
    only the reference when the molecule's dominant pucker happens to be the reference
    conformer. Pinning to ph0 measured BETTER than the bug on cyclohexane and WORSE on
    ethylcyclohexane -- so a test that accepted either would certify the wrong rule.
    """
    en = _en(SATURATED[1])
    ring_rows = set()
    for order, _, extra in en.ring_blocks(prior):
        for k, j in order:
            ring_rows.add(en._global_row(k, j))
        for k, j in extra:
            ring_rows.add(en._global_row(k, j))
    fg = en.ring_frame_groups(ring_rows)
    ph0 = en.ph0.detach().cpu().numpy()
    n0 = en.n_r + en.n_th
    _, st = en.sample_prior_states(prior, 256, np.random.default_rng(0), report=False)
    dof = st['dof']
    for rows_j, a, b, c, p, gi in fg:
        for j in rows_j:
            spread = float(np.std(dof[:, n0 + j]))
            assert spread > 1e-3, (
                'row {} is constant across draws, so it was pinned to a fixed value '
                'rather than following the ring frame'.format(j))


# -------------------------- the bank's rows belong to the molecule it was FITTED on

MIXED = ('phenyltetrahydropyran', 'C1CCC(CO1)c1ccccc1')


def test_ring_bank_is_applied_positionally_not_by_stored_row(prior):
    """A RingModes bank must be mapped onto THIS molecule's block, not its own row indices.

    The lookup key is (signature, n_dof): it identifies the ring TYPE and says nothing
    about row numbering, and the tree numbers a ring's DoF differently depending on what
    else is attached. Writing the bank's columns into its own stored rows permutes the
    block whenever the two molecules disagree.

    Phenyl-THP is the case in the set where they disagree -- its block is theta 1,5,8,13 /
    phi 4,7,12 against the bank's theta 2,5,8,11 / phi 4,7,10, and two bank columns landed
    on DoF placing atoms outside the ring. Asserted on the RESULT (the ring must come out
    chair-like, as the bank's own weights demand) rather than on the index arithmetic,
    because the arithmetic can be rewritten while staying wrong.
    """
    from energies.conformer_data import RingModes
    en = _en(MIXED[1])
    blocks = [(o, b, e) for o, b, e in en.ring_blocks(prior) if isinstance(b, RingModes)]
    assert blocks, '{} no longer resolves a RingModes bank'.format(MIXED[0])
    order, bank, _ = blocks[0]
    own = [kj for kj in order if kj[0] != 'r']
    assert list(own) != list(bank.order), (
        'phenyl-THP block order now coincides with the bank it was fitted against, so this '
        'molecule can no longer exercise the remap -- find another mixed-ring molecule')

    x, st = en.sample_prior_states(prior, 512, np.random.default_rng(0), report=False)
    assert st['n_ring_remapped'] == 1, st['n_ring_remapped']

    # the bank puts ~99.5% of its weight on the two chair basins; the draw must agree
    w = bank.weights(float(en.temperature))
    assert float(np.sort(w)[-2:].sum()) > 0.9, 'this bank is no longer chair-dominated'
    cyc = [c for c in rmet.ring_cycles(en) if not en.atom_is_aromatic[list(c)].all()]
    t = rmet.ring_torsions(en, x, cyc)[0]
    ids, labels = rmet.basin_labels(t)

    def alternating(lbl):
        s = [v for v in lbl if v != 0]
        return len(s) >= 4 and all(s[i] * s[i + 1] < 0 for i in range(len(s) - 1))

    chair = sum((ids == i).sum() for i, l in enumerate(labels) if alternating(l)) / len(ids)
    assert chair > 0.9, (
        'only {:.1%} of draws are chair-like while the bank weights the chair at {:.1%} -- '
        'the bank is being written into the wrong rows again'
        .format(chair, float(np.sort(w)[-2:].sum())))


def test_ring_bank_refuses_a_block_it_cannot_map(prior):
    """A kind-sequence mismatch must RAISE, not silently permute.

    Positional mapping is only meaningful when both sequences are the same block with the
    r rows removed. If the kinds do not line up they are different objects and there is no
    correspondence to use.
    """
    from dataclasses import replace
    from energies.conformer_data import RingModes
    en = _en(SATURATED[1])
    blocks = [(o, b, e) for o, b, e in en.ring_blocks(prior) if isinstance(b, RingModes)]
    order, bank, _ = blocks[0]
    scrambled = replace(bank, order=[('phi', j) if k == 'theta' else ('theta', j)
                                     for k, j in bank.order])
    p = __import__('copy').deepcopy(prior)
    p.ring_modes = {k: (scrambled if v is bank else v) for k, v in prior.ring_modes.items()}
    for k, v in prior.ring_modes.items():
        if list(v.order) == list(bank.order):
            p.ring_modes[k] = scrambled
    with pytest.raises(RuntimeError, match='refusing rather than permuting'):
        en.sample_prior_states(p, 16, np.random.default_rng(0), report=False)
