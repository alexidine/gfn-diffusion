"""energies/invertible_centres.py: the prior draws both sides of every free invertible centre.

At level `full` the target holds both pyramids of a three-coordinate centre the stereo lock
leaves free (and, unlocked, both labelled parities of every four-coordinate centre), while the
fitted prior holds the SIGN of the offset that sets a centre's side. The module names the
centres once, from the chart's structure: those whose offset can be negated as an exact
inversion without turning their ring system, and marks the planar ones.
`ConformerTorsions.sample_prior_states` flips the free three-coordinate ones, planar or not, on
an independent half of its draws, `.prior_log_prob` scores the mixture, and the offline eval's
parity metric and the log Z check's coverage label read the same table, less its planar
entries. Pinned here, level `full`, MMFF, float64, T = 1 kcal/mol unless stated, CPU:

  * NH3's draws split between the pyramids within a binomial bound, the side is the sign of
    the pyramid's triple product, and each flipped draw scores exactly its unflipped twin (the
    flip is the global mirror there); the two halves' energies are one distribution;
  * an amine N is flipped, a locked C beside it never is, a free four-coordinate C (unlocked)
    is not either, and DABCO's caged N, with no exocyclic substituent, has no entry;
  * NO ENERGY BAR (a decision of 2026-09-28): the tertiary amines a 10 kT rule on the
    reflected reference left one-sided (CCCN(C)C scored about 150 kT there, CCN(CC)CC,
    CC(C)N(C)C(C)C) are flipped, their draws split between the invertomers with the lock at
    0, and the eval fails a model that never visits the other side;
  * NO WIDTH BAR on the prior's flip (the same decision): the planar carbonyl C of CC=O is
    flipped on half the draws and the IS identity holds, while neither the eval nor the log Z
    check's label reads it;
  * every flip in the table is an exact inversion -- every graph bond length and angle kept,
    at the reference and on perturbed states, re-measured here independently, and the side
    reversed -- of each kind: ROOT, SIBLING (a ring atom about its ring child) and FRAME (a
    ring N at its ring's closure, C1CN1, C1CCNC1, C1CCNCC1, about its ring frame, in the
    prior's draw too), planar aromatic and alkene centres included; the root flips the table
    leaves out (CN1CCOCC1, DABCO) open the ring;
  * the table's extent, pinned rather than required: empty at `torsion`; a ring N at the
    tree's root (C1COCCN1) and one the tree enters from its substituent (CCN1CC1) have no
    entry; a sulfoxide S is free and flipped;
  * the lock names a centre only through a TETRAHEDRAL element: the key atom of a locked
    double bond (C/C=C/C(=O)N(C)C) is free and flipped, and the lock stays 0 on its draws;
  * a molecule with no free three-coordinate centre (CH4, ethanol, 2-butanol, at
    stereo_coeff 0 and 300) draws, and scores its draws, bitwise as with the flip switched
    off, generator state included; with one, the only change is the flip of the marked draws;
  * the stereo lock stays exactly 0, and every locked element's indicator unchanged, on the
    flipped draws of a molecule with two locked stereocentres and a free N;
  * the density is the two-component mixture: ln 2 below the one-sided density on either side,
    and the prior-proposal IS log Z of NH3 now agrees with the converged quadrature, where
    with the flip switched off it reads ~ln 2 low; prior_diagnostics' oracle flips the same
    way, so its estimate moves by ln 2 too. Beyond NH3's ROOT centre, whose two components sit
    22 widths apart: a SIBLING centre whose components overlap (the amide N of CC(=O)NC at
    T = 10) integrates along its flipped row exactly as with its flip switched off, and on
    NCCN's two SIBLING centres the coins are independent and E_mix[q_one / q_mix] = 1;
  * the eval's parity metric, the log Z check's held centres and the prior read one table,
    and the three-coordinate centres the eval requires on both sides are the non-planar ones
    among those the prior flips.

THE PRIOR IS AN EMPTY InternalPrior, so the file needs no fitted table: every group leader
falls back to a uniform draw, a ring is held about its reference at a fraction of thermal
width, and every row the flip or these checks read (improper rows, siblings, ring-frame
substituents, r and theta) is drawn thermally without one. On NH3, which has no group, the
draw is the fitted prior's draw.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/conformer/test_invertible_centres.py
"""
import math
from contextlib import contextmanager

import numpy as np
import pytest
import torch

import energies.invertible_centres as ic
from energies.conformer_torsions import ConformerTorsions

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

KW = dict(device='cpu', level='full', force_field='mmff', dtype=torch.float64)
LOCK = 300.0
#: converged log Z of NH3 at level full, T = 1 (tests/conformer/test_logz_check.py's ANCHOR:
#: the (16, 12, 96) grid and an independent 2e6-draw IS agree to 4e-4)
NH3_LOG_Z = -10.4893


@pytest.fixture(scope='module')
def prior():
    from mxtaltools.conformers.prior import InternalPrior
    return InternalPrior()


@pytest.fixture(scope='module')
def mols():
    """NH3 and CH3NH2 unlocked; CH3NH2, DABCO, CH4, ethanol, H2CO and 3-aminobutan-2-ol
    under the lock."""
    m = {'N': ConformerTorsions(smiles='N', **KW),
         'CN': ConformerTorsions(smiles='CN', **KW)}
    for smi in ('CN', 'C1CN2CCN1CC2', 'C', 'CCO', 'C=O', 'C[C@H](N)[C@H](C)O'):
        m[smi + '/lock'] = ConformerTorsions(smiles=smi, stereo_coeff=LOCK, **KW)
    return m


def _draw(en, prior, n, seed, reflect=True):
    """``(x, stats, next generator value)``; ``reflect=False`` switches the reflection off by
    emptying the table, which is how the draw read before it existed."""
    rng = np.random.default_rng(seed)
    real = ic.invertible_centres
    if not reflect:
        ic.invertible_centres = lambda member, *a, **k: []
    try:
        x, stats = en.sample_prior_states(prior, n, rng, report=False)
    finally:
        ic.invertible_centres = real
    return x, stats, rng.random()


def _build(en, dof):
    """Positions ``[n, n_atoms, 3]`` of DoF rows ``dof [n, ndim]``, through the chart."""
    n_r, n0 = en.n_r, en.n_r + en.n_th
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)
    x = en.state_from_dof(t(dof[:, :n_r]), t(dof[:, n_r:n0]), t(dof[:, n0:]))
    return en.build_positions(x).reshape(dof.shape[0], -1, 3)


def _pos(en, x):
    return en.build_positions(x).reshape(x.shape[0], -1, 3)


def _ref(en):
    return np.concatenate([en.r0.numpy(), en.th0.numpy(), en.ph0.numpy()])


@contextmanager
def _flipping(keep):
    """Within the block the prior draws, and its density scores, only the flips of the centres
    `keep(centre)` accepts: `lambda c: False` is the one-sided prior, as it read before the
    flip existed."""
    real = ic.invertible_centres
    ic.invertible_centres = lambda member, *a, **k: [c for c in real(member) if keep(c)]
    try:
        yield
    finally:
        ic.invertible_centres = real


def _wrapped(en, dof):
    """``dof`` with its phi columns on (-pi, pi]. A flipped row is not wrapped, and the
    one-sided density's wrapped Gaussian sums three unwrapped images only, so it under-counts
    a flipped draw scored unwrapped; the mixture's density is the same either way."""
    n0 = en.n_r + en.n_th
    out = np.array(dof, dtype=np.float64, copy=True)
    out[:, n0:] = np.pi - (np.pi - out[:, n0:]) % (2.0 * np.pi)
    return out


def _is_identity(en, prior, dof):
    """``(mean, standard error)`` of q_one / q_mix over draws ``dof`` from the mixture, q_one
    the density with every flip switched off: 1 when q_mix is the draws' density."""
    d = _wrapped(en, dof)
    mixed = en.prior_log_prob(prior, d)
    with _flipping(lambda c: False):
        one = en.prior_log_prob(prior, d)
    r = np.exp(one - mixed)
    return float(r.mean()), float(r.std() / math.sqrt(len(r)))


# ------------------------------------------------------------------ the table


@pytest.mark.fast
def test_the_table_names_amine_ns_and_not_a_caged_n(mols):
    table = lambda k: {c.name: c for c in ic.centre_table(mols[k])}
    nh3 = table('N')
    assert list(nh3) == ['N0'] and nh3['N0'].invertible
    assert (nh3['N0'].kind, nh3['N0'].rows, nh3['N0'].n_bonded) == (ic.ROOT, (0,), 3)
    # unlocked, CH3NH2's tetrahedral C is free too, but the prior flips three-coordinate
    # centres only (module docstring, LOCK AND SCOPE); the lock takes the C
    free = table('CN')
    assert free['C0'].free and free['C0'].n_bonded == 4 and not free['C0'].invertible
    assert [c.name for c in ic.invertible_centres(mols['CN'])] == ['N4']
    locked = table('CN/lock')
    assert locked['C0'].lock == ic.LOCKED and not locked['C0'].invertible
    assert locked['N4'].lock == ic.FREE and locked['N4'].invertible
    assert locked['N4'].kind == ic.SIBLING, 'an interior centre: its siblings about a pivot'
    # DABCO's cage N: free under the lock, and no exocyclic substituent to flip
    dabco = mols['C1CN2CCN1CC2/lock']
    z = np.asarray(dabco.spec.z)
    assert [c for c in ic.centre_table(dabco) if z[c.slot] == 7] == []
    assert ic.free_centres(dabco) == [] and ic.invertible_centres(dabco) == []
    for k in ('C/lock', 'CCO/lock', 'C=O/lock'):
        assert ic.free_centres(mols[k]) == [], k
    # H2CO's planar C is in the table and flipped, but it is its own flip: no free centre
    (h,) = ic.centre_table(mols['C=O/lock'])
    assert h.planar and h.invertible and h.kind == ic.ROOT
    assert not any(c.planar for c in ic.centre_table(mols['CN/lock']))


@pytest.mark.fast
def test_ring_ns_the_table_covers_and_does_not(prior):
    """A ring N the tree reaches through one ring bond and leaves through a ring-closure bond
    hangs its H off the ring frame (ConformerTorsions.ring_frame_groups): a FRAME entry, whose
    flip mirrors the H through the plane of the N's two ring neighbours. The prior's draw
    flips it on half its draws, the lock stays at 0 and the side follows the coins. L-proline's
    interior ring N flips about its ring child. The extent, pinned rather than required
    (module docstring, NOT COVERED): the root N of C1COCCN1 and the N of CCN1CC1, which the tree
    enters from its ethyl, have no entry; a table that starts covering them fails here, and the
    wiki page's sentence on it is then out of date."""
    for smi in ('C1CN1', 'C1CCNC1', 'C1CCNCC1'):
        en = ConformerTorsions(smiles=smi, stereo_coeff=LOCK, **KW)
        z = np.asarray(en.spec.z)
        (n,) = [c for c in ic.centre_table(en) if z[c.slot] == 7]
        graph, tree = ic._neighbours(en)
        assert n.kind == ic.FRAME and n.invertible and n.frame[2] == n.slot, smi
        assert n.frame[3] in graph[n.slot] - tree[n.slot], 'p is the ring-closure neighbour'
    en = ConformerTorsions(smiles='C1CCNC1', stereo_coeff=LOCK, **KW)
    (n,) = ic.invertible_centres(en)
    m = 256
    x, stats, _ = _draw(en, prior, m, 12)
    flip = stats['reflected'][:, 0]
    assert stats['invertible_centres'] == [n.name]
    assert abs(int(flip.sum()) - m / 2) <= 5.0 * math.sqrt(m) / 2.0, 'binomial(m, 1/2), 5 sd'
    pos = _pos(en, x)
    assert (ic.reference_side(pos, n) == ~flip).all()
    assert float(en.stereo.lock_energy(pos, LOCK).abs().max()) == 0.0
    pro = ConformerTorsions(smiles='OC(=O)[C@@H]1CCCN1', stereo_coeff=LOCK, **KW)
    z = np.asarray(pro.spec.z)
    (n,) = [c for c in ic.centre_table(pro) if z[c.slot] == 7]
    assert n.invertible and n.kind == ic.SIBLING
    for smi in ('C1COCCN1', 'CCN1CC1'):
        en = ConformerTorsions(smiles=smi, stereo_coeff=LOCK, **KW)
        z = np.asarray(en.spec.z)
        assert [c for c in ic.centre_table(en) if z[c.slot] == 7] == [], smi


@pytest.mark.fast
def test_a_tertiary_amine_is_flipped_however_crowded(prior):
    """NO ENERGY BAR (energies/invertible_centres.py, QUALIFIED: a decision of 2026-09-28). A
    10 kT rule on the reflected reference geometry, every rotor held there, left the N of
    CCCN(C)C one-sided: its moved methyl lands where the reference conformer already puts
    another group, and scored about 150 kT, while the target holds half its mass on each side
    (its two methyls are equivalent). The structural table flips it: the prior draws it on
    half its draws, the lock stays at 0 on them, the eval FAILS draws that all sit on one
    side, and the log Z check carries no label. The same holds for the other two amines the
    earlier rule left one-sided, and for a sulfoxide S, three-coordinate and so left free by
    the lock."""
    import eval.conformer_logz_check as lzc
    import eval.conformer_model_eval as me
    en = ConformerTorsions(smiles='CCCN(C)C', stereo_coeff=LOCK, **KW)
    (n,) = ic.free_centres(en)
    assert n.name.startswith('N') and n.n_bonded == 3 and n.invertible
    assert ic.invertible_centres(en) == [n] and lzc.prior_coverage_bias(en) is None
    m = 400
    x, stats, _ = _draw(en, prior, m, 11)
    flip = stats['reflected'][:, 0]
    assert stats['invertible_centres'] == [n.name]
    assert abs(int(flip.sum()) - m / 2) <= 5.0 * math.sqrt(m) / 2.0, 'binomial(m, 1/2), 5 sd'
    pos = _pos(en, x)
    assert (ic.reference_side(pos, n) == ~flip).all()
    assert float(en.stereo.lock_energy(pos, LOCK).abs().max()) == 0.0
    one = me.parity_coverage(en, x[torch.as_tensor(~flip)])
    assert one['n_centres'] == one['n_missed'] == 1 and one['missed_names'] == [n.name]
    assert me.judge({'parity': one}, me.Bars())['parity']['status'] == me.FAIL
    both = me.parity_coverage(en, x)
    assert both['n_missed'] == 0
    assert me.judge({'parity': both}, me.Bars())['parity']['status'] == me.PASS
    for smi in ('CCN(CC)CC', 'CC(C)N(C)C(C)C', 'CS(C)=O'):
        (c,) = ic.invertible_centres(ConformerTorsions(smiles=smi, stereo_coeff=LOCK, **KW))
        assert c.name[0] in 'NS', smi


def _graph_change(en, dof, centre):
    """``(largest bond-length change (A), largest bond-angle change (rad), side reversed on
    every row)`` when `centre` is flipped on each DoF row of ``dof [n, ndim]``: measured from
    `build_positions` over `bond_index_slot`, the side as the sign of the triple product of the
    centre's three lowest-slot neighbours, independently of energies/invertible_centres.py."""
    import itertools
    n0 = en.n_r + en.n_th
    b = np.asarray(en.bond_index_slot).reshape(2, -1).T
    nb = {}
    for u, v in b:
        nb.setdefault(int(u), set()).add(int(v))
        nb.setdefault(int(v), set()).add(int(u))
    ang = np.asarray([(a, c, e) for c in nb for a, e in itertools.combinations(sorted(nb[c]), 2)])
    k = sorted(nb[centre.slot])[:3]

    def geom(p):
        lengths = np.linalg.norm(p[:, b[:, 0]] - p[:, b[:, 1]], axis=-1)
        u, w = p[:, ang[:, 0]] - p[:, ang[:, 1]], p[:, ang[:, 2]] - p[:, ang[:, 1]]
        angles = np.arccos(np.clip((u * w).sum(-1) / np.linalg.norm(u, axis=-1)
                                   / np.linalg.norm(w, axis=-1), -1.0, 1.0))
        d = p[:, k] - p[:, centre.slot][:, None]
        return lengths, angles, np.sign(np.einsum('ij,ij->i', d[:, 0], np.cross(d[:, 1], d[:, 2])))

    p0 = _build(en, dof)
    mirror = dof.copy()
    ic.reflect_phi(mirror[:, n0:], centre,
                   ring=ic.frame_dihedral(p0, centre) if centre.kind == ic.FRAME else None)
    (l0, a0, s0), (l1, a1, s1) = geom(p0.numpy()), geom(_build(en, mirror).numpy())
    return float(np.abs(l1 - l0).max()), float(np.abs(a1 - a0).max()), bool((s1 == -s0).all())


@pytest.mark.fast
def test_every_flip_in_the_table_is_an_exact_inversion():
    """Every entry's flip keeps every graph bond length and angle, at the reference and on
    perturbed states, and reverses the side, over ring and acyclic molecules, locked and not,
    every kind of flip, and planar centres (the aromatic ring carbons and ipso C of
    CNc1ccccc1, the alkene and carbonyl carbons of C/C=C/C(=O)N(C)C). A ring atom's SIBLING
    flip pivots on its ring child, and a FRAME flip's p is the centre's ring-closure
    neighbour; at C3 and C4 of C1(C2CC2)CC1C1CC1 the table's FRAME test and the draw's
    ring-frame rule differ (module docstring, THE FLIP), and the flips are exact there too.
    The root flips the table leaves out are no inversions: those of CN1CCOCC1 and of DABCO
    open the ring (module docstring, NOT COVERED)."""
    rng = np.random.default_rng(9)
    cases = [('C1CCCCC1', 0.0), ('C1COCCN1', 0.0), ('OC(=O)[C@@H]1CCCN1', LOCK),
             ('C1CN2CCN1CC2', LOCK), ('CN', 0.0), ('CCCN(C)C', LOCK), ('C1CCNC1', LOCK),
             ('CN1CCCCC1', LOCK), ('CN(C)C1CC1', LOCK), ('CNc1ccccc1', LOCK),
             ('C/C=C/C(=O)N(C)C', 0.0), ('C1(C2CC2)CC1C1CC1', 0.0)]
    kinds, planar = set(), 0
    for smi, lock in cases:
        en = ConformerTorsions(smiles=smi, stereo_coeff=lock, **KW)
        ti = np.asarray(en.spec.torsion_index)
        ring = ic._ring_bonds(en)
        graph, tree = ic._neighbours(en)
        ref = _ref(en)
        dof = np.concatenate([ref[None], ref[None] + 0.02 * rng.standard_normal((16, ref.size))])
        for c in ic.centre_table(en):
            dl, da, flipped = _graph_change(en, dof, c)
            assert max(dl, da) < 1e-9 and flipped, (smi, c.name, dl, da, flipped)
            kinds.add(c.kind)
            planar += int(c.planar)
            if c.kind == ic.SIBLING and en.atom_in_ring[c.slot]:
                assert frozenset((c.slot, int(ti[c.pivot, 3]))) in ring, (smi, c.name)
            if c.kind == ic.FRAME:
                assert c.frame[3] in graph[c.slot] - tree[c.slot], (smi, c.name)
    assert kinds == {ic.ROOT, ic.SIBLING, ic.FRAME}
    assert planar >= 6, 'planar centres are in the table and their flips exact'
    for smi in ('CN1CCOCC1', 'C1CN2CCN1CC2'):
        en = ConformerTorsions(smiles=smi, stereo_coeff=LOCK, **KW)
        assert int(en.spec.z[0]) == 7 and 0 not in {c.slot for c in ic.centre_table(en)}, smi
        root = ic.Centre(slot=0, name='N0', kind=ic.ROOT, rows=tuple(en.improper_phi_rows()),
                         pivot=None, frame=None, n_bonded=3, planar=False, lock=ic.FREE,
                         triple=(1, 2, 3), side_ref=1.0)
        dl, _, _ = _graph_change(en, _ref(en)[None], root)
        assert dl > 1.0, (smi, 'the excluded root flip opens the ring', dl)


@pytest.mark.fast
def test_the_torsion_tier_has_no_invertible_centre():
    """A `torsion` column turns a whole bond, so no row moves alone and the state cannot
    reach a centre's other side: the target there holds the reference's. The eval's parity
    metric says so, rather than blaming the lock."""
    import eval.conformer_model_eval as me
    en = ConformerTorsions(smiles='CCCN', device='cpu', level='torsion', force_field='mmff',
                           dtype=torch.float64)
    assert en.collective and ic.centre_table(en) == []
    par = me.parity_coverage(en, torch.zeros(4, en.ndim, dtype=torch.float64))
    assert par['n_centres'] == 0 and 'collective' in par['na'] and 'lock' not in par['na']


@pytest.mark.fast
def test_the_flip_is_its_own_inverse_and_the_side_test_reads_it(mols):
    rng = np.random.default_rng(0)
    for k in ('N', 'CN'):
        en = mols[k]
        n0 = en.n_r + en.n_th
        dof = _ref(en)[None] + 0.05 * rng.standard_normal((64, en.ndim))
        for c in ic.centre_table(en):
            ph = dof[:, n0:]
            twice = ic.reflect_phi(ic.reflect_phi(ph.copy(), c), c)
            assert np.allclose(twice, ph, atol=1e-12, rtol=0.0)
            assert ic.reference_side(_build(en, dof), c).all()
            half = np.arange(64) % 2 == 0
            flipped = dof.copy()
            ic.reflect_phi(flipped[:, n0:], c, half)
            assert (ic.reference_side(_build(en, flipped), c) == ~half).all(), (k, c.name)
            moved = np.flatnonzero((flipped != dof).any(0)) - n0
            assert set(moved) == set(c.rows), 'only the centre\'s own rows move'


# ------------------------------------------------------------------ the draw


@pytest.mark.fast
def test_nh3_prior_draws_split_between_the_pyramids(mols, prior):
    nh3, n = mols['N'], 4000
    x, stats, _ = _draw(nh3, prior, n, 0)
    x0, stats0, _ = _draw(nh3, prior, n, 0, reflect=False)
    (c,) = ic.invertible_centres(nh3)
    flip = stats['reflected'][:, 0]
    assert stats['invertible_centres'] == ['N0']
    assert abs(int(flip.sum()) - n / 2) <= 5.0 * math.sqrt(n) / 2.0, 'binomial(n, 1/2), 5 sd'
    assert stats['reflected_frac'] == [pytest.approx(float(flip.mean()))]
    side = ic.reference_side(_pos(nh3, x), c)
    assert (side == ~flip).all(), 'every flipped draw, and only those, on the other side'
    # the pyramid itself: sign of (H1 - N).[(H2 - N) x (H3 - N)], with N in slot 0
    pos = nh3.build_positions(x).reshape(n, -1, 3).numpy()
    d = pos[:, 1:] - pos[:, :1]
    tp = np.einsum('ij,ij->i', d[:, 0], np.cross(d[:, 1], d[:, 2]))
    ref_sign = np.sign(tp[side][0])
    assert (np.sign(tp) == np.where(side, ref_sign, -ref_sign)).all()
    # the reflection is NH3's global mirror, so each reflected draw scores its twin exactly
    e, e0 = nh3.potential_energy(x, 1.0).numpy(), nh3.potential_energy(x0, 1.0).numpy()
    assert np.abs(e - e0).max() < 1e-9
    from scipy.stats import ks_2samp
    assert ks_2samp(e[flip], e[~flip]).pvalue > 1e-3, 'one energy distribution on both sides'
    assert bool((x.abs() <= 1.0).all()) and stats['clip_frac'] == stats0['clip_frac']


@pytest.mark.fast
def test_an_amine_n_is_flipped_beside_a_locked_c_and_a_caged_n_is_not(mols, prior):
    en = mols['CN/lock']
    x, stats, _ = _draw(en, prior, 400, 1)
    assert stats['invertible_centres'] == ['N4']
    side = {c.name: ic.reference_side(_pos(en, x), c).mean() for c in ic.centre_table(en)}
    assert side['C0'] == 1.0, 'the locked C is never flipped'
    assert 0.35 < side['N4'] < 0.65
    dabco = mols['C1CN2CCN1CC2/lock']
    x, stats, nxt = _draw(dabco, prior, 64, 2)
    x0, _, nxt0 = _draw(dabco, prior, 64, 2, reflect=False)
    assert stats['invertible_centres'] == [] and torch.equal(x, x0) and nxt == nxt0


@pytest.mark.fast
def test_draw_member_prior_draws_through_the_same_reflection(mols, prior):
    """The set builder's draw is sample_prior_states plus a bake, so it reflects too."""
    from energies.conformer_prior_draw import draw_member_prior
    en = mols['CN/lock']
    states, e, stats = draw_member_prior(en, 32, np.random.default_rng(5), prior=prior)
    x, direct, _ = _draw(en, prior, 32, 5)
    assert torch.equal(states, x) and stats['invertible_centres'] == ['N4']
    assert np.array_equal(stats['reflected'], direct['reflected'])


@pytest.mark.fast
@pytest.mark.parametrize('smi,lock', [('C', 0.0), ('C', LOCK), ('CCO', 0.0), ('CCO', LOCK),
                                      ('C[C@@H](O)CC', 0.0), ('C[C@@H](O)CC', LOCK)])
def test_no_three_coordinate_centre_draws_bitwise_as_before(prior, smi, lock):
    """No free three-coordinate centre (module docstring, BITWISE WHERE NOTHING IS FLIPPED):
    `invertible_centres` returns [], and the draw then makes no call on the generator: the
    draw, its DoF, the generator's next value and the density equal those with the flip
    switched off, which is the draw as it read before the flip existed. Unlocked, CH4's,
    ethanol's and 2-butanol's four-coordinate centres are free, in the table, and stay
    one-sided (LOCK AND SCOPE). A molecule with a planar sp2 centre (H2CO) is not one of
    these: its planar centre is flipped too."""
    en = ConformerTorsions(smiles=smi, stereo_coeff=lock, **KW)
    assert ic.invertible_centres(en) == [], 'the premise: nothing to flip'
    x, stats, nxt = _draw(en, prior, 128, 3)
    logq = en.prior_log_prob(prior, stats['dof'])
    x0, stats0, nxt0 = _draw(en, prior, 128, 3, reflect=False)
    assert stats['invertible_centres'] == [] and stats['reflected'].shape == (128, 0)
    assert torch.equal(x, x0) and np.array_equal(stats['dof'], stats0['dof']) and nxt == nxt0
    real = ic.invertible_centres
    ic.invertible_centres = lambda member, *a, **k: []
    try:
        logq0 = en.prior_log_prob(prior, stats0['dof'])
    finally:
        ic.invertible_centres = real
    assert np.array_equal(logq, logq0)
    if lock == 0.0:
        assert [c for c in ic.free_centres(en) if c.n_bonded == 4], smi


@pytest.mark.fast
def test_with_a_free_centre_the_reflection_is_the_only_change(mols, prior):
    en = mols['CN/lock']
    x, stats, _ = _draw(en, prior, 256, 6)
    x0, stats0, _ = _draw(en, prior, 256, 6, reflect=False)
    flip = stats['reflected'][:, 0]
    assert 0 < int(flip.sum()) < 256
    (c,) = ic.invertible_centres(en)
    want = stats0['dof'].copy()
    n0 = en.n_r + en.n_th
    ic.reflect_phi(want[:, n0:], c, flip)
    assert np.array_equal(stats['dof'], want)
    assert torch.equal(x[~flip], x0[~flip])


@pytest.mark.fast
def test_the_lock_stays_zero_on_reflected_draws(mols, prior):
    """3-aminobutan-2-ol, QM9-like: two locked stereocentres, CH3 and CH labelled parities,
    and a free NH2. Its reflected draws keep every locked element exactly where their
    unreflected twins have it, and the lock at 0."""
    en = mols['C[C@H](N)[C@H](C)O/lock']
    assert int(en.stereo.stereocentre.sum()) == 2
    n = 400
    x, stats, _ = _draw(en, prior, n, 4)
    x0, stats0, _ = _draw(en, prior, n, 4, reflect=False)
    (name,) = stats['invertible_centres']
    assert name.startswith('N')
    flip = stats['reflected'][:, 0]
    assert abs(int(flip.sum()) - n / 2) <= 5.0 * math.sqrt(n) / 2.0
    pos = en.build_positions(x).reshape(n, -1, 3)
    pos0 = en.build_positions(x0).reshape(n, -1, 3)
    assert float(en.stereo.lock_energy(pos, LOCK).abs().max()) == 0.0
    assert float((en.stereo.values(pos) - en.stereo.values(pos0)).abs().max()) < 1e-9
    assert bool((x.abs() <= 1.0).all()) and stats['clip_frac'] == stats0['clip_frac']


@pytest.mark.fast
def test_a_locked_double_bonds_key_atom_is_free_and_flipped(prior):
    """The lock names a centre only through a TETRAHEDRAL element (module docstring, LOCK AND
    SCOPE). A DOUBLE_BOND element is keyed on one of its double-bond atoms, three-coordinate,
    and locks the bond's E/Z, not that atom's side: read as a name, it left C0 of
    C/C=C/C(=O)N(C)C LOCKED under the lock and FREE without it. Under the lock no
    three-coordinate centre is LOCKED, the key atom is flipped on half the draws, and the lock
    stays exactly 0 on the flipped draws, every tetrahedral element's indicator where its
    unflipped twin has it (the lock-zero test's pattern)."""
    from energies.stereo_lock import DOUBLE_BOND, TETRAHEDRAL
    en = ConformerTorsions(smiles='C/C=C/C(=O)N(C)C', stereo_coeff=LOCK, **KW)
    kinds, keys = np.asarray(en.stereo.kind), np.asarray(en.stereo.key)
    (key,) = (int(k) for k in keys[kinds == DOUBLE_BOND])
    table = {c.slot: c for c in ic.centre_table(en)}
    three = [c for c in table.values() if c.n_bonded == 3]
    assert three and all(c.lock == ic.FREE for c in three), [(c.name, c.lock) for c in three]
    assert [c.lock for c in table.values() if c.n_bonded == 4] == [ic.LOCKED] * 3
    assert table[key].invertible, 'the key atom of the locked double bond is flipped'
    n = 400
    x, stats, _ = _draw(en, prior, n, 4)
    x0, stats0, _ = _draw(en, prior, n, 4, reflect=False)
    flip = stats['reflected'][:, stats['invertible_centres'].index(table[key].name)]
    assert abs(int(flip.sum()) - n / 2) <= 5.0 * math.sqrt(n) / 2.0, 'binomial(n, 1/2), 5 sd'
    pos = en.build_positions(x).reshape(n, -1, 3)
    pos0 = en.build_positions(x0).reshape(n, -1, 3)
    assert float(en.stereo.lock_energy(pos, LOCK).abs().max()) == 0.0
    tet = torch.as_tensor(kinds == TETRAHEDRAL)
    assert float((en.stereo.values(pos) - en.stereo.values(pos0))[:, tet].abs().max()) < 1e-9
    assert bool((x.abs() <= 1.0).all()) and stats['clip_frac'] == stats0['clip_frac']


# ------------------------------------------------------------------ the density


@pytest.mark.fast
def test_the_density_is_the_mixture_and_the_is_log_z_closes(mols, prior):
    """On a draw on either side the mixture's density is the one-sided density of that draw's
    twin less ln 2 (the other component sits ~2|ph0| / sigma widths away), and IS with the
    prior as proposal now reaches NH3's converged log Z, where one-sided it reads ~ln 2 low."""
    from energies.prior_diagnostics import is_log_z
    nh3 = mols['N']
    _, stats, _ = _draw(nh3, prior, 2000, 7)
    _, stats0, _ = _draw(nh3, prior, 2000, 7, reflect=False)
    real = ic.invertible_centres
    ic.invertible_centres = lambda member, *a, **k: []
    try:
        one_sided = nh3.prior_log_prob(prior, stats0['dof'])
        biased = is_log_z(nh3, prior, n=20000, seed=0)
    finally:
        ic.invertible_centres = real
    mixed = nh3.prior_log_prob(prior, stats['dof'])
    assert np.abs(mixed - (one_sided - math.log(2.0))).max() < 1e-9
    got = is_log_z(nh3, prior, n=20000, seed=0)
    assert abs(got['log_z'] - NH3_LOG_Z) < 0.1, got
    assert biased['log_z'] - NH3_LOG_Z < -0.5, ('the one-sided proposal misses a pyramid', biased)
    assert got['clip_frac'] == 0.0


@pytest.mark.fast
def test_the_oracle_reflects_as_the_prior_does(mols):
    """prior_diagnostics.oracle_logw draws the same reflection and scores the same mixture, so
    eta and D_avoidable compare two proposals over one support. On NH3 its self-normalised
    estimate of the partition function rises by ln 2 against the oracle with the reflection
    switched off, which misses the other pyramid (about 5 standard errors of slack at 20000
    draws); on methylamine, unlocked, the N reflects about its group's leader and the free
    four-coordinate C stays one-sided, as in the fitted prior."""
    from energies.prior_diagnostics import oracle_logw
    lme = lambda w: float(w.max() + np.log(np.mean(np.exp(w - w.max()))))
    both = oracle_logw(mols['N'], n=20000, seed=0)
    real = ic.invertible_centres
    ic.invertible_centres = lambda member, *a, **k: []
    try:
        one = oracle_logw(mols['N'], n=20000, seed=0)
    finally:
        ic.invertible_centres = real
    assert abs((lme(both) - lme(one)) - math.log(2.0)) < 0.1, (lme(both), lme(one))
    w = oracle_logw(mols['CN'], n=512, seed=1)
    assert w.shape == (512,) and np.isfinite(w).all()


@pytest.mark.fast
def test_a_sibling_mixture_integrates_like_one_side_where_its_components_overlap(prior):
    """NH3's two components sit 22 widths apart, where scoring a draw by the component on its
    own side reads the same as the mixture. The amide N of CC(=O)NC at T = 10 is a SIBLING
    centre whose components overlap: its flipped row's reference offset sits under two widths
    from its flip. Along that row, every other coordinate held at a prior draw, the mixture's
    density integrates over (-pi, pi] to exactly what the density with the N's flip switched
    off does: the flip is a reflection of the row on the circle, and the mixture its two
    halves. A density that scored each draw by its own side's component alone integrates to
    the mass on that side, 0.8 of it here. Only the N's flip is switched off: the carbonyl C,
    planar, is flipped too, and its factor, constant along the row, differs between the two
    densities at a fixed draw. Draws on either side of the N are integrated. With every flip
    switched off, E_mix[q_one / q_mix] = 1 on the draws within 4 standard errors."""
    en = ConformerTorsions(smiles='CC(=O)NC', stereo_coeff=LOCK, log_temperature=1.0, **KW)
    assert en.temperature == pytest.approx(10.0)
    (n,) = [c for c in ic.invertible_centres(en) if c.name.startswith('N')]
    assert n.kind == ic.SIBLING and not n.planar
    groups = en.torsion_groups()
    width = dict(zip(map(tuple, groups), en.sibling_jitter_sigma(groups, en.temperature)))
    ph0 = en.ph0.numpy()
    for j in n.rows:
        (w,) = [w for g, w in width.items() if j in g]
        sep = abs(ic._wrap(2.0 * (ph0[j] - ph0[n.pivot]))) / w
        assert 1.0 < sep < 2.0, ('the components overlap', sep)
    x, stats, _ = _draw(en, prior, 20000, 5)
    flip = stats['reflected'][:, stats['invertible_centres'].index(n.name)]
    n0 = en.n_r + en.n_th
    grid = np.linspace(-np.pi, np.pi, 2001)[1:]
    for k in [*np.flatnonzero(flip)[:2], *np.flatnonzero(~flip)[:2]]:   # both sides of the N
        for j in n.rows:
            d = np.repeat(stats['dof'][k:k + 1], len(grid), 0)
            d[:, n0 + j] = grid
            mixed = np.exp(en.prior_log_prob(prior, d)).sum()
            with _flipping(lambda c: c.slot != n.slot):
                one = np.exp(en.prior_log_prob(prior, d)).sum()
            assert mixed / one == pytest.approx(1.0, abs=1e-6), (k, j, mixed / one)
    mean, se = _is_identity(en, prior, stats['dof'])
    assert abs(mean - 1.0) < 4.0 * se, (mean, se)


@pytest.mark.fast
def test_two_centres_flip_on_independent_coins_and_the_mixture_is_their_density(prior):
    """NCCN: two amine N, each a SIBLING centre. Each centre has its own coin, so the four
    (flipped, flipped) cells each hold a quarter of the draws, and the density is the product
    of the two centres' mixtures: drawn from it, E_mix[q_one / q_mix] = 1 within 4 standard
    errors, q_one the density with every flip switched off. One coin shared by both centres
    puts half the draws in each diagonal cell, and a mixture about the wrong pivot or mean
    scores the flipped draws where they do not sit; either moves the expectation off 1."""
    en = ConformerTorsions(smiles='NCCN', **KW)
    inv = ic.invertible_centres(en)
    assert [c.kind for c in inv] == [ic.SIBLING, ic.SIBLING] and not any(c.planar for c in inv)
    assert [c.name[0] for c in inv] == ['N', 'N']
    m = 20000
    x, stats, _ = _draw(en, prior, m, 3)
    f = stats['reflected']
    cells = [np.mean(f[:, 0] & f[:, 1]), np.mean(f[:, 0] & ~f[:, 1]),
             np.mean(~f[:, 0] & f[:, 1]), np.mean(~f[:, 0] & ~f[:, 1])]
    bound = 5.0 * math.sqrt(0.25 * 0.75 / m)                  # binomial(m, 1/4), 5 sd
    assert all(abs(c - 0.25) < bound for c in cells), cells
    mean, se = _is_identity(en, prior, stats['dof'])
    assert abs(mean - 1.0) < 4.0 * se, (mean, se)


@pytest.mark.fast
def test_a_planar_centre_is_flipped_and_neither_required_nor_labelled(prior):
    """The carbonyl C of CC=O is planar: its flipped row's reference offset sits at pi, within
    `MIRROR_WIDTHS` of its flip. The prior flips it all the same (module docstring, the
    DECISION: no width bar on the prior's flip), on half its draws, and scores the mixture,
    whose two components nearly coincide: E_mix[q_one / q_mix] = 1 within 4 standard errors.
    It is its own flip, so the eval does not require two sides of it (`free_centres`) and
    the log Z check's label does not name it (`prior_held_parity_centres`); unlocked, both
    read the free methyl C."""
    import eval.conformer_logz_check as lzc
    en = ConformerTorsions(smiles='CC=O', **KW)
    (c,) = ic.invertible_centres(en)
    assert c.planar and c.name.startswith('C') and c.n_bonded == 3
    m = 4000
    x, stats, _ = _draw(en, prior, m, 3)
    flip = stats['reflected'][:, 0]
    assert stats['invertible_centres'] == [c.name]
    assert abs(int(flip.sum()) - m / 2) <= 5.0 * math.sqrt(m) / 2.0, 'binomial(m, 1/2), 5 sd'
    mean, se = _is_identity(en, prior, stats['dof'])
    assert abs(mean - 1.0) < 4.0 * se, (mean, se)
    free = [f.slot for f in ic.free_centres(en)]
    held = lzc.prior_held_parity_centres(en)
    assert c.slot not in free and c.slot not in held
    assert free == held and len(free) == 1, 'the free methyl C, which the prior does not flip'


# ------------------------------------------------------------------ one table, every reader


@pytest.mark.fast
def test_the_eval_the_log_z_label_and_the_prior_read_one_table(mols, prior):
    import eval.conformer_logz_check as lzc
    import eval.conformer_model_eval as me
    for k, en in mols.items():
        x = torch.zeros(4, en.ndim, dtype=torch.float64)
        par = me.parity_coverage(en, x)
        free = ic.free_centres(en)
        rows = par.get('centres', [])
        assert [(r['centre'], r['kind'], r['n_bonded']) for r in rows] == [
            (c.slot, c.kind, c.n_bonded) for c in free], k
        assert par.get('n_accessible', 0) == len(rows), 'every centre is required both ways'
        # the three-coordinate centres the eval requires on both sides are the non-planar ones
        # the prior flips (no member here has an unreadable lock table); the planar ones it
        # flips (H2CO's C) are neither required nor labelled
        assert [c.slot for c in ic.invertible_centres(en) if not c.planar] == [
            r['centre'] for r in rows if r['n_bonded'] == 3], k
        assert not any(c.planar for c in free), k
        assert lzc.prior_held_parity_centres(en) == sorted(
            c.slot for c in ic.centre_table(en) if not c.invertible and not c.planar), k
    assert [c.planar for c in ic.invertible_centres(mols['C=O/lock'])] == [True]
    # on the prior's own draws the eval sees both sides of the free N, and misses nothing
    en = mols['CN/lock']
    x, stats, _ = _draw(en, prior, 400, 8)
    par = me.parity_coverage(en, x)
    (row,) = par['centres']
    assert row['name'] == 'N4' and par['n_missed'] == 0
    assert row['ref_side_frac'] == pytest.approx(1.0 - stats['reflected_frac'][0])
    # the coverage label: nothing for NH3, the amine N or DABCO's cage N, which has no second
    # side; the free four-coordinate C of unlocked CH3NH2, which the prior does not flip, keeps it
    assert lzc.prior_coverage_bias(mols['N']) is None
    assert lzc.prior_coverage_bias(en) is None
    assert lzc.prior_coverage_bias(mols['C1CN2CCN1CC2/lock']) is None
    label = lzc.prior_coverage_bias(mols['CN'])
    assert label.startswith(lzc.KNOWN_BIASED)
    assert 'parity at C0 (4-coordinate' in label and 'N4' not in label
