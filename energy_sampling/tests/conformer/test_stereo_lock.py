"""The stereo lock (energies/stereo_lock.py): one labelled stereoisomer per condition.

Every energy term of the conformer force field is even under reflection, so without the lock
a condition's target is every labelled configuration its box holds. The lock is a flat-bottom
term, exactly zero inside the locked isomer, stiff outside. Its claims, each pinned here:

  * it is EXACTLY 0 at the reference, on fitted-prior draws and on thermal samples of the
    locked isomer, and the mirror image pays hundreds of kcal/mol;
  * the best-of-eight indicator stays clear of its band where a fixed neighbour tetrahedron
    changes sign inside the correct isomer's ensemble (a hand-built flattened centre, and two
    untagged QM9 bridgehead references), ties between candidates are broken by index, not
    roundoff, and `thermal_check` catches a reference the chosen indicator does fail on;
  * summed over every labelled sign assignment it reproduces the unlocked partition function
    up to the band mass and the finite wrong-sign leakage (the partition identity), and the
    locked log Z sits below the unlocked one by the symmetry count: ln 2 per centre on the
    converged grids;
  * it enters the potential BEFORE `energy_clip`, so the cap bounds the total, and a live row
    and a baked row score it identically at any temperature -- at one coefficient, which is
    fixed at construction and recorded on every condition graph, a row built under another
    being refused;
  * the one-pass multi-molecule energy reads it off the condition graph and equals the
    per-member oracle, with and without the clip, and keeps it inside the clip too; a row
    bound to the other stereoisomer's mol_id is refused;
  * with stereo_coeff > 0 a SMILES that does not pin one isomer, a reference that realises
    another isomer and an element too thin to lock are refused, and the perception it rests
    on does not move with RDKit's process-wide legacy flag;
  * the per-coordinate features carry E/Z (`row_ez_parity`) and the held double-bond rows;
  * off (stereo_coeff 0, the default) nothing changes.

Tier is stated per test. CPU, float64, MMFF, T = 1 kcal/mol unless noted.
"""
import itertools
import os
import pathlib
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from energies.conformer_torsions import ChartRefused, ConformerTorsions
from energies import stereo_lock as sl

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

KW = dict(force_field='mmff', device='cpu', dtype=torch.float64)
COEFF = 300.0
#: the fitted InternalPrior (not in git); GFN_CONFORMER_PRIOR points at it from a worktree, as
#: in test_conformer_resume_roundtrip.py
PRIOR_PATH = os.environ.get('GFN_CONFORMER_PRIOR', 'conformer_prior_v2.pt')
LN2 = float(np.log(2.0))

# tagged isomers, including the three strained fused three-ring QM9 molecules the stage-0
# critique found the design's fixed indicator firing on inside the correct isomer
PRIOR_CASES = [
    'C[C@@H](O)CC',                              # one stereocentre, plus CH2/CH3 parity
    'C[C@H](N)[C@H](C)O',                        # two stereocentres
    'C/C=N/O', 'C/C=N\\O', 'C/C=C/CO',           # E/Z: held double-bond rows
    'C[C@@H](C=O)N1[C@@H]2CC[C@@H]21',           # CC(C=O)N1C2CCC12, strained bridgehead
    'N#C[C@H]1[C@H](O)[C@@H]2N[C@H]12',          # OC1C2NC2C1C#N
    'COC[C@]1(O)[C@@H]2OC[C@@H]21',              # COCC1(O)C2COC12, design |V| 0.031
]


def _en(smi, level='full', coeff=COEFF, **kw):
    return ConformerTorsions(smiles=smi, level=level, stereo_coeff=coeff, **{**KW, **kw})


def _prior():
    # FAILS rather than skips: the prior-draw half of the zero claim is the half the design's
    # own indicator failed on, and a skipped check reads as a passed one
    if not pathlib.Path(PRIOR_PATH).exists():
        pytest.fail(f'fitted InternalPrior not found at {PRIOR_PATH}; set GFN_CONFORMER_PRIOR '
                    f'to the local conformer_prior_v2.pt')
    return torch.load(PRIOR_PATH, weights_only=False)


def _lock(en, x):
    pos = en.build_positions(x).reshape(x.shape[0], -1, 3)
    return en.stereo.lock_energy(pos, en.stereo_coeff)


def _thermal(en, steps=20000, seed=0):
    """Cartesian MMFF94 Metropolis at kT = 1 kcal/mol from the reference, RDKit's own force
    field -- an independent thermal sampler, not our chart. ``[S, N, 3]`` in slot order."""
    return sl.mmff_thermal_samples(en, steps, seed)


@contextmanager
def _legacy(flag):
    from rdkit import Chem
    prev = Chem.GetUseLegacyStereoPerception()
    Chem.SetUseLegacyStereoPerception(flag)
    try:
        yield
    finally:
        Chem.SetUseLegacyStereoPerception(prev)


# ------------------------------------------------------------------ zero inside, stiff outside


@pytest.mark.parametrize('smi', PRIOR_CASES)
def test_lock_is_exactly_zero_at_reference_on_prior_draws_and_thermal_samples(smi):
    """Tier `full`. P == 0.0 exactly -- not small -- on the locked isomer's own states.

    Prior: 2000 joint draws of the fitted InternalPrior, held double-bond rows included.
    Thermal: RDKit's MMFF94 Metropolis chain. The last three molecules are the realised
    isomers of the strained bridgeheads the stage-0 critique named; embedded from their TAGGED
    SMILES they sit at other references than the untagged ones it measured, where even a fixed
    neighbour tetrahedron is far from zero, so they are regression cases here and do not
    discriminate the indicator choice. `test_best_candidate_*` do.
    """
    en = _en(smi)
    assert en.stereo.n > 0
    assert float(_lock(en, torch.zeros(1, en.data_ndim))) == 0.0
    S = _thermal(en)
    P = en.stereo.lock_energy(S, COEFF)
    assert int((P > 0).sum()) == 0, f'{smi}: lock fired on {int((P > 0).sum())} of {len(P)} ' \
                                    f'thermal samples of its own isomer'
    x, stats = en.sample_prior_states(_prior(), 2000, np.random.default_rng(0), report=False)
    P = _lock(en, torch.as_tensor(x))
    assert int((P > 0).sum()) == 0, f'{smi}: lock fired on {int((P > 0).sum())} of 2000 ' \
                                    f'prior draws'
    if en.stereo.bonds():
        assert stats['n_held_bond'] > 0


def test_mirror_image_is_heavily_penalised_and_free_unlocked():
    """Tier `full`. The mirror of every conformer (phi -> -phi) flips every tetrahedral parity.

    Unlocked it scores the same to roundoff (the force field is even under reflection);
    locked it loses at least 100 nats. energy_clip off, so the full penalty is visible.
    """
    on, off = _en('C[C@@H](O)CC'), _en('C[C@@H](O)CC', coeff=0.0)
    x = torch.as_tensor(np.random.default_rng(1).uniform(-0.3, 0.3, (32, on.data_ndim)))
    r, th, ph = on.dof_from_state(x)
    xm = on.state_from_dof(r, th, -ph)
    assert bool((xm.abs() <= 1.0).all())
    d_off = (off.energy(xm) - off.energy(x)).abs().max()
    assert float(d_off) < 1e-6
    assert float((on.energy(xm) - on.energy(x)).min()) > 100.0
    assert float(_lock(on, x).max()) == 0.0


# ------------------------------------------------------------------ the indicator choice


def _neighbour_tetrahedron(en):
    """``(elements, quads, v_ref)``: the FIXED neighbour-tetrahedron quad (the four neighbours in
    slot order, the last as apex) of every tetrahedral element -- the stage-0 design's
    indicator, which `build_table`'s best-of-eight replaced."""
    nbr = {}
    for u, v in np.asarray(en.bond_index_slot).reshape(2, -1).T:
        nbr.setdefault(int(u), set()).add(int(v))
        nbr.setdefault(int(v), set()).add(int(u))
    els = np.flatnonzero(en.stereo.kind == sl.TETRAHEDRAL)
    quads = np.asarray([sorted(nbr[int(en.stereo.key[e])]) for e in els], dtype=np.int64)
    v = sl._values_np(en.ref_pos.numpy(), quads, np.full(len(quads), sl.TETRAHEDRAL))
    return els, quads, v


def test_best_candidate_keeps_the_band_clear_on_a_flattened_neighbour_tetrahedron():
    """Hand-built, no chart, no RDKit: a centre whose four neighbours are nearly coplanar and
    all on one side of it -- the shape of a fused three-ring bridgehead. The neighbour
    tetrahedron's |v| is small at the reference, and positional noise of 0.05 A flips its sign
    on a large share of draws; the chosen candidate is a centre-based triple, far from zero,
    and its lock stays at exactly 0 on every draw. Fails if the fixed quad is restored."""
    d = 0.02                                   # neighbour-tetrahedron |v| 0.029
    pos = np.array([[0.0, 0.0, 0.0],
                    [1.4, 0.0, -0.5 + d], [0.0, 1.4, -0.5 - d],
                    [-1.4, 0.0, -0.5 + d], [0.0, -1.4, -0.5 - d]])
    bonds = np.array([[0, 0, 0, 0], [1, 2, 3, 4]])
    tab = sl.build_table(pos, bonds, [])
    fixed = np.array([[1, 2, 3, 4]])
    v_fixed = float(sl._values_np(pos, fixed, np.array([sl.TETRAHEDRAL]))[0])
    assert abs(v_fixed) < 0.05
    assert int(tab.quad[0, 3]) == 0 and float(tab.margin[0]) > 0.5, (tab.quad, tab.margin)

    noise = np.random.default_rng(0).normal(0.0, 0.05, (500, 5, 3))
    noise[:, 0] = 0.0
    S = torch.as_tensor(pos[None] + noise)
    flips = (np.sign(v_fixed) * sl.values_from_points(
        S[:, torch.as_tensor(fixed)], torch.tensor([sl.TETRAHEDRAL]))[:, 0] < 0)
    assert float(flips.double().mean()) > 0.2
    assert float(tab.lock_energy(S, COEFF).max()) == 0.0


@pytest.mark.parametrize('smi', ['CC(C=O)N1C2CCC12', 'OC1C2NC2C1C#N'])
def test_best_candidate_stays_clear_where_the_neighbour_tetrahedron_flips_inside_the_isomer(smi):
    """Tier `full`, lock off (the table is built either way). The UNTAGGED molecules' ETKDG
    references, which the stage-0 critique measured: a bridgehead's neighbour tetrahedron has
    |v| below 0.2 there, and on RDKit's MMFF94 chain it changes sign on a large share of
    samples that RDKit re-perceives as the same isomer. The best-of-eight table never fires on
    them. Fails if the fixed quad is restored."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    en = _en(smi, coeff=0.0)
    els, quads, v_ref = _neighbour_tetrahedron(en)
    thin = np.abs(v_ref) < 0.2
    assert thin.any(), np.abs(v_ref).min()
    S = _thermal(en)
    v = sl.values_from_points(S[:, torch.as_tensor(quads[thin])],
                              torch.full((int(thin.sum()),), sl.TETRAHEDRAL))
    flipped = (torch.as_tensor(np.sign(v_ref[thin])) * v < 0).any(-1)
    assert float(flipped.double().mean()) > 0.2, float(flipped.double().mean())
    assert float(en.stereo.lock_energy(S, COEFF).max()) == 0.0
    # the flipped samples are the SAME isomer by RDKit's own reading of the 3D structure
    perm = np.asarray(en.spec.perm)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    m = Chem.Mol(en.mol)
    conf = m.GetConformer()
    for t in torch.nonzero(flipped).flatten()[:5].tolist():
        xyz = S[t].numpy()[inv]
        for a in range(m.GetNumAtoms()):
            conf.SetAtomPosition(a, Point3D(*map(float, xyz[a])))
        assert sl.realised_isomer(m) == en.stereo_isomer


def test_thermal_check_catches_a_reference_whose_chosen_indicator_fails():
    """The untagged COCC1(O)C2COC12's ETKDG reference sits in a metastable basin; the MMFF94
    chain leaves it for a lower one that RDKit calls the same isomer, and there the best
    candidate itself changes sign. No indicator choice is right at every strained centre, so
    the builder runs `thermal_check` per molecule and refuses one it fires on. A clean
    molecule passes."""
    bad = sl.thermal_check(_en('COCC1(O)C2COC12', coeff=0.0))
    assert bad['fired'] > 0 and bad['min_excess'] < 0.0, bad
    good = sl.thermal_check(_en('C[C@@H](O)CC'))
    assert good['fired'] == 0 and good['min_excess'] > 0.0, good


def test_ties_between_candidates_are_broken_by_index_not_roundoff():
    """Symmetric substituents make candidates tie exactly. At an ideal tetrahedral methane every
    candidate of a family ties, and at tert-butanol's MMFF reference the CH3 triples tie to
    roundoff; with the plain argmax, 1e-9 A of noise changes which quad methane stores and
    1e-6 A changes tert-butanol's. The stored quad is compared atom by atom in `_resolve_rows`,
    so a tie resolved by roundoff reads as another stereoisomer on another machine."""
    a = 1.09 / np.sqrt(3.0)
    ch4 = np.array([[0, 0, 0], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]], dtype=float)
    bonds = np.array([[0, 0, 0, 0], [1, 2, 3, 4]])
    quads = {tuple(sl.build_table(ch4 + np.random.default_rng(k).normal(0, 1e-9, ch4.shape),
                                  bonds, []).quad[0].tolist()) for k in range(20)}
    assert len(quads) == 1, quads
    en = _en('CC(C)(C)O', coeff=0.0)
    ref, bi = en.ref_pos.numpy(), np.asarray(en.bond_index_slot)
    base = sl.build_table(ref, bi, []).quad
    for k in range(20):
        noisy = ref + np.random.default_rng(k).normal(0.0, 1e-6, ref.shape)
        assert np.array_equal(sl.build_table(noisy, bi, []).quad, base), k


# ------------------------------------------------------------------ exact quadrature


def _partition(smi, level, grid):
    """``(log Z unlocked, {sign flips: log Z locked})`` over every labelled sign assignment."""
    en = _en(smi, level)
    base = en.stereo
    z0 = _en(smi, level, coeff=0.0).brute_force_log_z(grid=grid, chunk=65536)
    zs = {}
    for flips in itertools.product((1, -1), repeat=base.n):
        en.stereo = base.with_sign(base.sign * np.asarray(flips))
        zs[flips] = en.brute_force_log_z(grid=grid, chunk=65536)
    return z0, zs


@pytest.mark.parametrize('smi,grid,tol', [
    ('C', 128, 1e-9),                 # methane: the four H, labelled parity only
    ('CF', 128, 1e-9),                # CH3F
    ('F[C@H](Cl)Br', 128, 1e-9),      # a true stereocentre
    ('CO', 64, 1e-6),                 # methanol, k = 3
])
def test_symmetry_count_and_partition_identity(smi, grid, tol):
    """Tier `dihedral`, exact quadrature. Z(unlocked) = sum over sign assignments of Z(locked),
    and each centre's two assignments are equal, so log Z drops by exactly ln 2 per centre.

    The identity is not exact point by point. The lock is finite, so at a point where every
    element is outside its band the wrong-sign assignments still weigh exp(-k (lo + |v|)^2 / T)
    each -- exp(-12), about 6e-6, at |v| = lo = 0.1, k = 300, T = 1 -- and inside a band no
    assignment is free. Its error is that leakage plus the bands' Boltzmann mass, weighted by
    exp(-U); measured at or below 1e-11 here. The ln 2 needs convergence, which these grids
    have (unchanged from 128 to 256 on the k = 2 cases to 1e-10).
    """
    z0, zs = _partition(smi, 'dihedral', grid)
    tot = float(torch.logsumexp(torch.tensor(list(zs.values()), dtype=torch.float64), 0))
    assert abs(tot - z0) < 1e-10, (smi, tot - z0)
    locked = zs[(1,) * len(next(iter(zs)))]
    assert abs((z0 - locked) - LN2 * len(next(iter(zs)))) < tol, (smi, z0 - locked)


@pytest.mark.parametrize('smi,grid', [('CC', 12),          # two CH3 centres, four assignments
                                      ('F/C=C/F', 48)])     # one double bond: E and Z
def test_partition_identity_where_the_shift_is_not_the_symmetry_count(smi, grid):
    """Tier `dihedral`. Ethane's 5-D grid is not converged (log Z moves by ~2 nats from grid 12
    to 16), so only the identity -- exact on any grid -- is asserted; difluoroethene's two
    assignments are E and Z, which differ, so the shift is log1p(Z_Z / Z_E), not ln 2."""
    z0, zs = _partition(smi, 'dihedral', grid)
    tot = float(torch.logsumexp(torch.tensor(list(zs.values()), dtype=torch.float64), 0))
    assert abs(tot - z0) < 1e-10, (smi, tot - z0)
    if smi == 'F/C=C/F':
        e, z = zs[(1,)], zs[(-1,)]
        assert abs(e - z) > 0.1
        assert abs((z0 - e) - np.log1p(np.exp(z - e))) < 1e-10


def test_enantiomers_have_equal_log_z_by_converged_quadrature():
    """Tier `dihedral`, grid 256 (converged: 128 and 256 agree to 1e-10). R and S are built
    separately, each from its own embedding, so their frozen references differ slightly;
    the locked log Z agree to that level."""
    r = _en('F[C@H](Cl)Br', 'dihedral').brute_force_log_z(grid=256, chunk=65536)
    s = _en('F[C@@H](Cl)Br', 'dihedral').brute_force_log_z(grid=256, chunk=65536)
    assert abs(r - s) < 1e-5, r - s


def test_torsion_tier_is_unchanged_or_refused():
    """Tier `torsion`. Rigid bond rotations cannot invert a centre, so on a tetrahedral-only
    molecule the lock is identically 0 and log Z is bitwise the unlocked one. A molecule whose
    rotatable column turns a locked double bond is REFUSED rather than silently retargeted
    (the torsion prior draws that column into both E and Z)."""
    on, off = _en('C[C@@H](O)CC', 'torsion'), _en('C[C@@H](O)CC', 'torsion', coeff=0.0)
    x = torch.linspace(-1, 1, 4097, dtype=torch.float64)[:-1].reshape(-1, 1)
    assert float(_lock(on, x).max()) == 0.0
    assert on.brute_force_log_z(grid=4096) == off.brute_force_log_z(grid=4096)
    with pytest.raises(ChartRefused) as err:
        _en('CC/C=N/OC', 'torsion')
    assert err.value.code == 'stereo_torsion_double_bond'
    assert _en('CC/C=N/OC', 'torsion', coeff=0.0).data_ndim > 0


# ------------------------------------------------------------------ the cap and the currency


def test_lock_sits_inside_the_energy_clip():
    """With energy_clip 300 the stored potential is clip(U + P) (+ wall), so the owner's cap
    bounds the total; the lock is not added after the compression."""
    from mxtaltools.common.utils import log_rescale_positive
    from mxtaltools.conformers.energy import intramolecular_energy
    en = _en('C[C@@H](O)CC', energy_clip=300.0)
    x = torch.as_tensor(np.random.default_rng(2).uniform(-0.2, 0.2, (256, en.data_ndim)))
    r, th, ph = en.dof_from_state(x)
    xm = en.state_from_dof(r, th, -ph)                 # every parity flipped: P large
    one = torch.tensor(1.0, dtype=torch.float64)
    tree, ff = en._batch(len(xm))
    U = intramolecular_energy(tree, en.build_positions(xm), ff)
    P = _lock(en, xm)
    assert float(P.min()) > 300.0
    got = en.potential_energy(xm, one)
    want = log_rescale_positive(U + P, 300.0) + en.bounding_energy(xm, one)
    assert torch.allclose(got, want, rtol=0, atol=1e-9)
    assert float((got - en.bounding_energy(xm, one)).max()) < 300.0 + np.log1p(
        float((U + P).max()) - 300.0) + 1e-9


@pytest.mark.parametrize('log_t', [-0.3, 0.0, 0.3])
def test_live_and_baked_rows_score_the_lock_identically_at_any_temperature(log_t):
    """A live row (energy at T) and the same row baked at T = 1 and read back through
    prebuilt_sample_to_reward at T agree, on states where the lock is non-zero. States in
    the box, so the box wall -- which is pre-multiplied by T and so disagrees at T != 1 on
    its own -- is zero and cannot mask the lock."""
    from energies.conformer_data import attach_states, bake_energies, condition_from_energy
    en = _en('C[C@@H](O)CC')
    x = torch.as_tensor(np.random.default_rng(3).uniform(-0.5, 0.5, (64, en.data_ndim)))
    r, th, ph = en.dof_from_state(x)
    x = en.state_from_dof(r, th, -ph)
    assert float(_lock(en, x).min()) > 0.0
    batch = attach_states(condition_from_energy(en), x, bake_energies(en, x),
                          periodic=en.periodic_dims)
    # the log-temperature in float64: a float32 -0.3 is -0.30000001, which on these ~1e3-nat
    # rewards is a 1e-4 discrepancy that has nothing to do with the lock
    live = -en.energy(x, None, torch.tensor(log_t, dtype=torch.float64))
    baked = en.prebuilt_sample_to_reward(batch, 10.0 ** log_t)
    assert torch.allclose(live, baked, rtol=1e-12, atol=1e-9), (live - baked).abs().max()


# ------------------------------------------------------------------ indicator mechanics


def test_gradient_is_finite_when_substituents_overlap_and_at_the_crossing():
    """The normalisers are floored (NORM_FLOOR): two coincident quad atoms give a finite,
    bounded gradient instead of 0/0; along 2-butanol's inversion path the gradient is finite
    and non-zero where the lock turns on."""
    en = _en('C[C@@H](O)CC')
    pos = en.ref_pos.reshape(1, -1, 3).clone()
    q = en.stereo.quad[0]
    pos[0, q[0]] = pos[0, q[3]]                        # p1 on top of p4
    pos.requires_grad_(True)
    P = en.stereo.lock_energy(pos, COEFF).sum()
    (g,) = torch.autograd.grad(P, pos)
    assert torch.isfinite(g).all() and float(g.abs().max()) < 1e7

    # inversion path of the true stereocentre: its first neighbour pushed through the
    # plane of the other three
    k = int(np.flatnonzero(en.stereo.stereocentre)[0])
    c = int(en.stereo.key[k])
    nb = [int(a) for a in en.stereo.quad[k] if int(a) != c]
    ref = en.ref_pos.clone()
    plane = ref[nb].mean(0)
    for t in np.linspace(0.0, 2.0, 41):
        p = ref.clone()
        p[c] = ref[c] + float(t) * (plane - ref[c])
        p = p.reshape(1, -1, 3).requires_grad_(True)
        P = en.stereo.lock_energy(p, COEFF).sum()
        (g,) = torch.autograd.grad(P, p)
        assert torch.isfinite(g).all()
        if float(P) > 0.0:
            assert float(g.abs().max()) > 0.0


@pytest.mark.parametrize('smi', ['CN[C@@H]1C(C#N)[C@H]1NC', 'C1C[C@H]2C[C@H](C2)O1'])
def test_perception_is_pinned_against_the_process_flag(smi):
    """The lock set, the signs, the realised isomer and the verify step are the same under
    either value of RDKit's process-wide legacy-perception flag, and the flag is restored
    afterwards. Both molecules are ones where, UNPINNED under the non-legacy algorithm, the
    3D re-perception and the input's canonical SMILES disagree (a spurious centre on the
    cyclopropane carbon made non-stereogenic by the other two; the bicycle's canonical tags),
    so the lock-on build would be refused as stereo_verify_failed. Found in the QM9 census's
    first 400 molecules (5 of them). `cleanIt=True` is passed explicitly but has no measured
    effect: the same 400 molecules and nine hand-made spurious tags give identical
    FindPotentialStereo reports with it True or False."""
    from rdkit import Chem
    out = {}
    for flag in (True, False):
        with _legacy(flag):
            en = _en(smi)                                   # lock ON: the verify step runs
            assert Chem.GetUseLegacyStereoPerception() is flag
            out[flag] = (en.stereo.kind.tolist(), en.stereo.key.tolist(),
                         en.stereo.quad.tolist(), en.stereo.sign.tolist(),
                         en.stereo_isomer, [tuple(sorted(e.items())) for e in en.stereo_elements])
    assert out[True] == out[False]


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize('smi,code', [
    ('CC(O)CC', 'stereo_unspecified'),
    ('OC1CCC(O)CC1', 'stereo_unspecified'),       # ring cis/trans: hidden on the explicit-H graph
    ('CC=N', 'stereo_unspecified'),               # =NH imine: hidden on the heavy-atom graph
    ('C[C@@H]1C[N@]1C', 'stereo_unsupported'),    # a tagged invertible N
])
def test_lock_refuses_a_smiles_that_does_not_pin_one_isomer(smi, code):
    with pytest.raises(ChartRefused) as err:
        _en(smi)
    assert err.value.code == code
    _en(smi, coeff=0.0)                           # off: nothing is refused


def test_verify_step_refuses_a_reference_that_realises_another_isomer():
    """A tagged isomer whose ETKDG reference re-perceives as ANOTHER isomer (a QM9 census case,
    a cyclopropane fused to a seven-ring) is refused with the lock on: the signs are read off
    that reference, so the lock would pin the isomer it realised, not the one requested."""
    smi = 'C1COCO[C@@H]2C[C@H]2C1'
    with pytest.raises(ChartRefused) as err:
        _en(smi)
    assert err.value.code == 'stereo_verify_failed'
    off = _en(smi, coeff=0.0)
    assert off.stereo_isomer != off.stereo_input


def test_an_element_too_thin_to_lock_is_refused(monkeypatch):
    """`MIN_MARGIN` is set by what the WRONG configuration pays at its mirror point,
    k (lo + m)^2: at the floor it must still be a lock (tens of kcal/mol at the census's
    k = 300), and a molecule whose element falls below it is refused."""
    m = sl.MIN_MARGIN
    assert m > 0.0
    for kind in (sl.TETRAHEDRAL, sl.DOUBLE_BOND):
        assert COEFF * (min(sl.LO_MAX[kind], sl.LO_FRAC * m) + m) ** 2 > 50.0, kind
    en = _en('C[C@@H](O)CC')
    top = float(en.stereo.margin.max())
    monkeypatch.setattr(sl, 'MIN_MARGIN', top + 1e-6)
    with pytest.raises(ChartRefused) as err:
        _en('C[C@@H](O)CC')
    assert err.value.code == 'stereo_lock_in_band'
    _en('C[C@@H](O)CC', coeff=0.0)                  # off: nothing is refused


def test_an_allene_is_not_locked_and_is_refused_when_locking():
    """RDKit reports an allene's axis as two stereo double bonds sharing the centre atom, and
    the far substituent of each is collinear with it, so the E/Z indicator is identically 0.
    They are not elements: the condition graph still builds with the lock off, and a tagged
    allene is refused with it on. Level `dihedral` (`full` refuses a cumulated centre)."""
    from energies.conformer_data import condition_from_energy
    en = _en('CC=C=CC', 'dihedral', coeff=0.0)
    assert not en.stereo.bonds()
    condition_from_energy(en)
    with pytest.raises(ChartRefused) as err:
        _en('C/C=C=C/C', 'dihedral')
    assert err.value.code == 'stereo_unsupported'


@pytest.mark.parametrize('smi', [
    'CC(C)C12CC(C1)C2',            # symmetric cage bridgeheads: 'potential', never stereogenic
    'OC[C@@H]1C(O)[C@H]1CO',       # a centre made non-stereogenic by the other two
    'C[C@@H]1CN1C',                # untagged aziridine N: its invertomers stay together
    'C[N@H+](CCO)CC(=O)[O-]',      # a four-coordinate N+ is an ordinary locked centre
])
def test_lock_accepts_fully_pinned_isomers(smi):
    en = _en(smi)
    assert en.stereo_isomer == en.stereo_input


# ------------------------------------------------------------------ graph form, one pass


SET = ['C[C@@H](O)CC', 'C[C@H](O)CC',                     # an enantiomer pair
       'C[C@H](N)[C@H](C)O', 'C[C@H](N)[C@@H](C)O',       # a diastereomer pair
       'C/C=N/O']                                          # E/Z


def _set_batch(smis, sizes, coeff=COEFF, seed=0, scale=1.0, **kw):
    from energies.conformer_carrier import carrier_pad_condition
    from energies.conformer_data import collate_conditions, condition_from_energy
    from energies.multi_conformer import MultiConformerTorsions
    multi = MultiConformerTorsions(smis, identifiers=smis, level='full', stereo_coeff=coeff,
                                   **{**KW, **kw})
    lay = multi.carrier
    reg = {k: j for j, k in enumerate(sorted(smis, reverse=True))}
    multi.bind_identifier_registry(reg)
    rng = np.random.default_rng(seed)
    assign = np.repeat(np.arange(len(smis)), sizes)[rng.permutation(int(sum(sizes)))]
    conds = {}
    for ident, m in multi._members.items():
        c = condition_from_energy(m, identifier=ident)
        conds[ident] = carrier_pad_condition(c, lay, ident, m) if lay is not None else c
    rows, xs = [], []
    for j in assign:
        ident = smis[j]
        c = conds[ident].__copy__()
        c.mol_id = torch.tensor([reg[ident]])
        rows.append(c)
        k = multi._members[ident].data_ndim
        x = torch.as_tensor(rng.uniform(-scale, scale, (1, k)))
        xs.append(lay.to_carrier(ident, x) if lay is not None else x)
    return multi, collate_conditions(rows), torch.cat(xs), assign, reg


def test_graph_fields_are_written_and_survive_collation_and_a_buffer_draw():
    from energies.conformer_data import collate_conditions, condition_from_energy
    en = _en('C[C@H](N)[C@H](C)O')
    c = condition_from_energy(en)
    n = en.spec.n_atoms
    for f in sl.STEREO_FIELDS:
        assert getattr(c, f).shape[0] == n
    assert tuple(c.ctree_stereo_nbr.shape) == (n, 4)
    assert int((c.ctree_stereo_kind != 0).sum()) == en.stereo.n
    batch = collate_conditions([c] * 6).subsample_new_batch(np.array([1, 3, 4]))
    assert tuple(batch.ctree_stereo_nbr.shape) == (3 * n, 4)
    pos = en.ref_pos.repeat(3, 1)
    assert float(sl.batch_lock_energy(batch, pos, COEFF).abs().max()) == 0.0


@pytest.mark.parametrize('scale', [0.3, 1.0])
def test_one_pass_equals_the_member_oracle_with_stereoisomers_in_the_set(scale):
    """Enantiomers, diastereomers and an E/Z molecule in one mixed carrier batch; states
    uniform in the box, so the lock is live on many rows. The one-pass energy reads the lock
    off each row's graph; the oracle is each member's own `energy` at its own width."""
    multi, batch, X, assign, _ = _set_batch(SET, [7, 9, 5, 6, 5], scale=scale)
    got = multi.energy(X, batch)
    want = multi._energy_per_member(X, batch)
    assert torch.allclose(got, want, rtol=1e-10, atol=1e-8), (got - want).abs().max()
    # the baked potential carries the lock too
    _, scored = multi.energy(X, batch, return_exp=True)
    one = torch.tensor(1.0, dtype=torch.float64)
    for j, ident in enumerate(SET):
        rows = torch.as_tensor(np.flatnonzero(assign == j))
        m = multi._members[ident]
        xi = multi.carrier.from_carrier(ident, X.index_select(0, rows))
        assert torch.allclose(scored.conformer_energy.index_select(0, rows),
                              m.potential_energy(xi, one), rtol=1e-10, atol=1e-8)
    if scale == 1.0:
        pos_rows = sum(int((_lock(multi._members[i],
                                  multi.carrier.from_carrier(i, X[torch.as_tensor(
                                      np.flatnonzero(assign == j))])) > 0).sum())
                       for j, i in enumerate(SET))
        assert pos_rows > 0, 'no row had a live lock, so the comparison did not exercise it'


@pytest.mark.parametrize('log_t', [-0.3, 0.0, 0.3])
def test_one_pass_keeps_the_lock_inside_the_energy_clip(log_t):
    """The production conditional route: the one-pass energy on a mixed carrier batch with
    energy_clip at the owner's cap, 300 kcal/mol. Rows uniform in the box clash, and each row
    is put in whichever of its state and its mirror image the lock penalises more, so every
    row violates its lock and some carry U and P each above the cap. The baked potential is
    clip(U + P) row by row, so the cap's log tail bounds the total; the one-pass energy equals
    the member oracle; and a live row and its baked copy read back at T agree. The one-pass
    energy is tested apart from the member path because it reads the lock off the graph and
    places it itself: added after the clip it would bake clip(U) + P, hundreds of kcal/mol
    over that bound, and part from the oracle by as much. Tier `full`."""
    from mxtaltools.common.utils import log_rescale_positive
    from mxtaltools.conformers.energy import intramolecular_energy
    cap = 300.0
    multi, batch, X, assign, _ = _set_batch(SET, [7, 9, 5, 6, 5], energy_clip=cap)
    lay = multi.carrier
    assert lay is not None, 'SET must need a carrier: that is the production layout'
    assert multi.energy_clip == cap and all(m.energy_clip == cap
                                            for m in multi._members.values())
    X = X.clone()
    U = torch.zeros(len(X), dtype=torch.float64)
    P = torch.zeros(len(X), dtype=torch.float64)
    for j, ident in enumerate(SET):
        rows = torch.as_tensor(np.flatnonzero(assign == j))
        m = multi._members[ident]
        xi = lay.from_carrier(ident, X.index_select(0, rows))
        r, th, ph = m.dof_from_state(xi)
        xm = m.state_from_dof(r, th, -ph)             # every tetrahedral indicator flipped
        xi = torch.where((_lock(m, xm) > _lock(m, xi)).unsqueeze(-1), xm, xi)
        X[rows] = lay.to_carrier(ident, xi)
        tree, ff = m._batch(len(xi))
        U[rows] = intramolecular_energy(tree, m.build_positions(xi), ff)
        P[rows] = _lock(m, xi)
    # inside the box, so the wall is zero and the baked potential is the clipped term alone
    assert float(X.abs().max()) <= 1.0
    assert bool((P > 0).all()), 'a row satisfies its lock, so it cannot tell where it is added'
    assert bool(((U > cap) & (P > cap)).any()), 'no clashing, stereo-violating row'

    logT = torch.full((len(X),), log_t, dtype=torch.float64)
    got = multi.energy(X, batch, logT)
    want = multi._energy_per_member(X, batch, logT)
    assert torch.allclose(got, want, rtol=1e-10, atol=1e-8), (got - want).abs().max()

    _, scored = multi.energy(X, batch.clone(), torch.zeros_like(logT), return_exp=True)
    baked = scored.conformer_energy.flatten()
    capped = log_rescale_positive(U + P, cap)
    assert float(baked.max()) <= cap + float(np.log1p(float((U + P).max()) - cap)) + 1e-9
    assert torch.allclose(baked, capped, rtol=0, atol=1e-8), (baked - capped).abs().max()
    # the baked row, read back at T, scores the lock as the live row does: both divide it by T
    read = multi.prebuilt_sample_to_reward(scored, 10.0 ** log_t)
    assert torch.allclose(-got, read, rtol=1e-12, atol=1e-9), (-got - read).abs().max()


def test_a_mol_id_bound_to_the_other_stereoisomer_is_refused():
    """R- and S-2-butanol share z, atom count and every chart flag; only the per-atom lock
    code tells them apart. Rows carrying R's graph under S's mol_id are refused with the lock
    on, and scored (as nothing distinguishes them) with it off."""
    for coeff in (COEFF, 0.0):
        multi, batch, X, assign, reg = _set_batch(SET[:2], [4, 4], coeff=coeff)
        swapped = batch.mol_id.clone()
        a, b = reg[SET[0]], reg[SET[1]]
        swapped[batch.mol_id == a], swapped[batch.mol_id == b] = b, a
        batch.mol_id = swapped
        if coeff > 0:
            with pytest.raises(RuntimeError, match='STEREOISOMER'):
                multi.energy(X, batch)
        else:
            multi.energy(X, batch)


def test_the_coefficient_is_fixed_after_construction():
    """train.set_energy_coeffs sets any energy_config key a stage's coeff_schedule or anneal
    names. A baked row holds clip(U + P) at the coefficient it was scored under and is never
    re-scored, so after any change live and baked rows would score different locks: every
    change is refused, a same-sign one included. Setting the value already in force (a
    schedule's first step) is accepted."""
    multi, batch, X, _, _ = _set_batch(SET[:2], [4, 4])
    multi.stereo_coeff = COEFF
    for bad in (100.0, 0.0):
        with pytest.raises(ValueError, match='after construction'):
            multi.stereo_coeff = bad
    assert multi.stereo_coeff == COEFF
    assert all(m.stereo_coeff == COEFF for m in multi._members.values())
    off = _en('CC(O)CC', coeff=0.0)
    with pytest.raises(ValueError, match='after construction'):
        off.stereo_coeff = COEFF


@pytest.mark.parametrize('built,run', [(0.0, COEFF), (COEFF, 0.0), (COEFF, 100.0)])
def test_a_row_built_under_another_coefficient_is_refused(built, run):
    """Every condition graph records the coefficient it was built under
    (``ctree_stereo_coeff``). A row whose record differs from the scoring energy's is refused,
    live (`energy`) and baked (`prebuilt_sample_to_reward`, which never re-scores) alike, and
    the modeller refuses the FILE at load. Its baked energy carries the other lock, and its
    per-coordinate features mark held rows only when the builder locked."""
    from types import SimpleNamespace
    from conformer_modeller import ConformerModeller
    maker, batch, X, _, _ = _set_batch(SET, [3, 3, 2, 2, 2], coeff=built)
    assert torch.equal(batch.ctree_stereo_coeff,
                       torch.full((batch.num_graphs,), built, dtype=torch.float64))
    _, scored = maker.energy(X, batch, return_exp=True)
    user, _, _, _, _ = _set_batch(SET, [3, 3, 2, 2, 2], coeff=run)
    with pytest.raises(RuntimeError, match='built under stereo_coeff'):
        user.energy(X, batch)
    with pytest.raises(RuntimeError, match='built under stereo_coeff'):
        user.prebuilt_sample_to_reward(scored, 1.0)
    fake = SimpleNamespace(energy_function=SimpleNamespace(stereo_coeff=run))
    with pytest.raises(SystemExit, match='built under stereo_coeff'):
        ConformerModeller._refuse_stereo_mismatch(fake, batch, 'x.pt', 'conditions')
    fake.energy_function.stereo_coeff = built
    ConformerModeller._refuse_stereo_mismatch(fake, batch, 'x.pt', 'conditions')


def test_the_builder_records_the_coefficient(tmp_path, monkeypatch):
    """build_conformer_conditions.py --stereo-coeff builds every member with the lock on (so
    the refusals and the thermal check run at build time) and every graph records it; a run at
    that coefficient scores the file, one at another refuses it."""
    import sys
    import build_conformer_conditions as bcc
    from energies.multi_conformer import MultiConformerTorsions
    out = tmp_path / 'conds.pt'
    smis = ['C[C@@H](O)CC', 'C[C@H](O)CC']
    monkeypatch.setattr(sys, 'argv', ['build_conformer_conditions.py', '--smiles', *smis,
                                      '--level', 'full', '--force-field', 'mmff',
                                      '--stereo-coeff', str(COEFF), '--out', str(out)])
    dt, nt = torch.get_default_dtype(), torch.get_num_threads()
    try:
        bcc.main()
    finally:
        torch.set_default_dtype(dt)                  # main() sets both process-wide
        torch.set_num_threads(nt)
    batch = torch.load(out, weights_only=False)['prior']
    assert batch.ctree_stereo_coeff.reshape(-1).tolist() == [COEFF, COEFF]
    X = torch.zeros(2, int(batch.n_torsions[0]), dtype=torch.float64)
    for coeff, ok in ((COEFF, True), (0.0, False)):
        multi = MultiConformerTorsions(smis, identifiers=smis, level='full', stereo_coeff=coeff,
                                       **KW)
        if ok:
            assert torch.isfinite(multi.energy(X, batch)).all()
        else:
            with pytest.raises(RuntimeError, match='built under stereo_coeff'):
                multi.energy(X, batch)


# ------------------------------------------------------------------ features and prior rows


def test_features_carry_parity_and_e_z_and_the_held_rows():
    """atom_parity is non-zero exactly on the table's true stereocentres; row_ez_parity gives
    the two isomers of one double bond opposite signs on the same rows; is_held_bond marks
    the rows about a locked double bond only when the lock is on, and those rows leave
    torsion_groups for held_phi_rows."""
    from energies.dof_features import atom_parity, dof_features, feature_names, row_ez_parity
    for smi in ('C[C@H](N)[C@H](C)O', 'O[C@H]1CC[C@@H](O)CC1', 'C[C@@H](O)[C@@H](O)[C@H](C)O'):
        en = _en(smi)
        want = set(en.stereo.key[en.stereo.stereocentre].tolist())
        assert set(np.flatnonzero(atom_parity(en)).tolist()) == want, smi
    e, z = _en('C/C=N/O'), _en('C/C=N\\O')
    ee, zz = row_ez_parity(e), row_ez_parity(z)
    assert (ee != 0).any() and np.array_equal(ee != 0, zz != 0) and np.array_equal(ee, -zz)
    names = feature_names()
    held_c = names.index('is_held_bond')
    off = _en('C/C=N/O', coeff=0.0)
    assert off.held_phi_rows() == off.improper_phi_rows()
    assert dof_features(off)[:, held_c].sum() == 0
    held = sorted(set(e.held_phi_rows()) - set(e.improper_phi_rows()))
    assert held
    f = dof_features(e)
    assert np.flatnonzero(f[:, held_c]).tolist() == [e.n_r + e.n_th + j for j in held]
    assert not set(held) & {j for g in e.torsion_groups() for j in g}


def test_stereo_coeff_is_part_of_the_problem_identity():
    """Not exempt from the problem hash: a locked and an unlocked run are different targets.
    With the lock on the SMILES is fully tagged, so energy_config.smiles names the isomer."""
    from types import SimpleNamespace
    from utils import _NON_IDENTITY_ENERGY_CONFIG_KEYS, get_problem_definition, problem_hash
    assert 'stereo_coeff' not in _NON_IDENTITY_ENERGY_CONFIG_KEYS

    def ident(**ec):
        a = SimpleNamespace(energy_config=dict(ec), energy_function='conformer',
                            prior_path=None, space_groups=None, z_primes=None,
                            molecule_conditioning=False, temperature_conditioning=False)
        return problem_hash(get_problem_definition(a))
    base = dict(smiles='C[C@@H](O)CC', level='full')
    assert ident(**base, stereo_coeff=0.0) != ident(**base, stereo_coeff=COEFF)
    assert (ident(**base, stereo_coeff=COEFF)
            != ident(**{**base, 'smiles': 'C[C@H](O)CC'}, stereo_coeff=COEFF))
