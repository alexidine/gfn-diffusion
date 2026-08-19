"""
Step-1 gates for the internal-DoF ladder (docs/design/internal_dof_ladder.md section 10).

Four things are asserted, and three of them are written so that re-introducing the bug
produces a FAILURE rather than a quieter pass:

  1. `torsion` is BITWISE unchanged by the generalised scatter. Stated as three concrete
     assertions rather than an unscoped "bitwise", because the arithmetic identity only
     covers the phi block and would not notice a wrong block order or a wrong scale.
  2. The layout reaches GFN from the ENERGY. The old fallback silently unwrapped; that is
     pinned as a difference, not asserted away.
  3. r and theta stay on-domain for ANY latent, including far outside the wall.
  4. `level` cannot be swallowed.

    python test_conformer_levels.py
"""

import sys
import traceback

import numpy as np
import torch

from energies.conformer_torsions import ConformerTorsions
from models.gfn import GFN
from train_conformer import build_gfn

DEVICE = torch.device("cpu")
DTYPE = torch.float64
# butanol is the shipped dev molecule; propanol is the next target and is the smaller of
# the two, so it is the cheaper one to run every gate on
GATE_SMILES = "CCCCO"
PROPANOL = "CCCO"


def _energy(smiles=GATE_SMILES, level="torsion", **kw):
    """dtype PINNED to float64 here, while the shipped default is float32.

    These gates assert exactness -- bitwise geometry, and the measure term to 1e-12 -- and
    those tolerances are statements about the FORMULA, not about the working precision. Run
    at float32 they would have to be loosened to ~1e-6, which is a weaker test of the thing
    they exist to pin. test_float32_default_is_sound covers the shipped precision instead.
    """
    kw.setdefault("dtype", DTYPE)
    return ConformerTorsions(smiles=smiles, device="cpu", level=level, **kw)


# --------------------------------------------------------------------- gate 1: bitwise

def _legacy_build_positions(energy, x):
    """The pre-ladder formula, verbatim: phi = phi_ref + mask @ (pi * x), r/theta frozen.

    Kept here rather than in the module so the comparison is against what the code USED
    to do, not against a refactor of it.
    """
    from mxtaltools.conformers.builder import build
    x = x.to(energy.dtype) * np.pi
    b = x.shape[0]
    tree, _ = energy._batch(b)
    phi = (energy.ph0.unsqueeze(0) + x @ energy.mask.T).reshape(-1)
    return build(tree, energy.r0.repeat(b), energy.th0.repeat(b), phi)


def test_torsion_bitwise():
    e = _energy()
    torch.manual_seed(0)
    x = torch.rand(4096, e.data_ndim, dtype=DTYPE, device=DEVICE) * 2 - 1

    new = e.build_positions(x)
    old = _legacy_build_positions(e, x)
    assert torch.equal(new, old), (
        f"torsion build_positions is not bitwise identical: max |d| = "
        f"{(new - old).abs().max().item():.3e}")

    # ...and the comparison is capable of failing. A scale wrong by one ULP must be caught,
    # or this gate is measuring nothing.
    bumped = e._free_scale.clone()
    bumped[0] = np.nextafter(float(bumped[0]), float(bumped[0]) + 1.0)
    saved, e._free_scale = e._free_scale, bumped
    try:
        assert not torch.equal(e.build_positions(x), old), \
            "a one-ULP scale change went undetected; the bitwise gate is blind"
    finally:
        e._free_scale = saved

    # the POTENTIAL is still bitwise -- the wall must be SKIPPED, not added as zero
    assert e._lin_free_idx.numel() == 0
    lt = torch.tensor(0.0, dtype=DTYPE)
    t1 = 10 ** lt.to(DTYPE)
    from mxtaltools.conformers.energy import intramolecular_energy
    tree, ff = e._batch(x.shape[0])
    ref_pot = intramolecular_energy(tree, old, ff)
    assert torch.equal(e.potential_energy(x, t1), ref_pot), \
        "torsion potential_energy is not bitwise"

    # energy() is NOT bitwise any more, by design: step 2 added the change of measure.
    # At torsion it is a constant, so the difference must be EXACTLY that constant.
    assert e.log_jacobian_const is not None
    got = e.energy(x, None, lt)
    assert torch.allclose(got, ref_pot / t1 - e.log_jacobian_const,
                          atol=1e-12, rtol=1e-12), \
        "torsion energy() does not differ from the legacy value by exactly log J"
    print(f"PASS gate 1  torsion bitwise over 4096 states (d={e.data_ndim}); geometry and "
          f"potential exact, energy offset by exactly log J = {e.log_jacobian_const:.6f}")


def test_rotatable_axes_ordered():
    """Gate (ii): the free columns keep their ORDER.

    A permutation of the free columns is bitwise-invisible in build_positions (the same
    set of dihedrals is written, just via different state indices) and would silently
    reinterpret every stored state file. Order is the thing to pin.
    """
    e = _energy()
    axes = [(int(u), int(v)) for u, v in e.rotatable]
    # ascending child slot is the order _find_rotatable emits and the order every stored
    # state file was written against
    assert axes == sorted(axes, key=lambda uv: uv[1]), axes

    # A torsion state column is a COLLECTIVE coordinate: it drives every dihedral about
    # its bond, generally several. Assert that explicitly -- reading the mask as one-hot
    # is the bug the bitwise gate caught, and a molecule where every column happened to
    # drive one dihedral would hide it.
    mask = e.mask.detach().cpu().numpy()
    m = e._M.detach().cpu().numpy()
    driven = np.flatnonzero(e.free_mask)
    per_col = mask.sum(axis=0)
    assert (per_col >= 1).all(), per_col
    assert per_col.max() > 1, \
        (f"every rotatable bond of {GATE_SMILES} drives exactly one dihedral, so this "
         f"molecule cannot distinguish the collective map from an index map; pick another")
    # the map restricted to driven rows must BE the mask, row for row and column for column
    for j in range(m.shape[1]):
        rows_M = set(int(driven[i]) for i in np.flatnonzero(m[:, j]))
        rows_mask = set(int(r) + e.n_r + e.n_th for r in np.flatnonzero(mask[:, j]))
        assert rows_M == rows_mask, (j, sorted(rows_M), sorted(rows_mask))
    # and every column lands in the phi block
    assert all(b == 2 for b in e._free_block)
    print(f"PASS gate 2  axes ordered {axes}; columns are COLLECTIVE "
          f"({per_col.astype(int).tolist()} dihedrals each) and the map matches the mask")


# ------------------------------------------------------------- gate 3: layout from energy

def test_layout_reaches_gfn():
    class _Mdl:
        s_emb_dim = harmonics_dim = t_dim = 32
        policy_hidden_dim = flow_hidden_dim = 32
        policy_layers = flow_layers = 2
        t_scale, zero_init, clipping, gfn_clip = 0.05, True, True, 1e4

    for level, expect_all_wrapped in (("torsion", True), ("dihedral", True),
                                      ("flex", False), ("full", False)):
        e = _energy(PROPANOL, level=level)
        pd = e.periodic_dims
        assert len(pd) == e.data_ndim, (level, len(pd), e.data_ndim)
        assert all(pd) == expect_all_wrapped, (level, sum(pd), len(pd))

        gfn = build_gfn(e.data_ndim, _Mdl, DEVICE, pd)
        assert gfn.ang_dim == sum(pd), (level, gfn.ang_dim, sum(pd))
        assert gfn.lin_dim == len(pd) - sum(pd), (level, gfn.lin_dim)
        assert gfn.expanded_dim == gfn.lin_dim + 2 * gfn.ang_dim
        if expect_all_wrapped:
            assert gfn.ang_idx.tolist() == list(range(e.data_ndim)), level
        # the wrapped dims are exactly the phi block, in order
        assert gfn.ang_idx.tolist() == [i for i, p in enumerate(pd) if p], level
        print(f"     {level:9s} d={e.data_ndim:3d}  ang={gfn.ang_dim:3d} lin={gfn.lin_dim:3d}")

    # build_gfn must refuse a mask of the wrong width rather than build the wrong policy
    e = _energy(PROPANOL, level="full")
    try:
        build_gfn(e.data_ndim, _Mdl, DEVICE, e.periodic_dims[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("build_gfn accepted a short angular_mask")
    print("PASS gate 3  energy-declared layout reaches GFN at every level")


# --------------------------------------------------------------- gate 4: domain guarantee

def test_domain_guarantee():
    """r and theta stay physical for ANY latent, and log_reward stays finite.

    The wall alone does not give this: the forward kernel is Gaussian and puts mass
    outside any box, log_jacobian's `2 log r` is NaN at r <= 0, and `build` is
    non-injective off-domain. The clamp is the guarantee; this asserts it directly rather
    than asserting the wall is large.
    """
    from mxtaltools.conformers.builder import log_jacobian

    for level in ("flex", "full"):
        e = _energy(PROPANOL, level=level)
        assert e._lin_free_idx.numel() > 0, level
        lt = torch.tensor(0.0, dtype=DTYPE)

        for mag in (1.5, 5.0, 50.0, 1e3):
            for sign in (+1.0, -1.0):
                x = torch.full((8, e.data_ndim), sign * mag, dtype=DTYPE, device=DEVICE)
                r, th, ph = e.dof_from_state(x)
                assert torch.isfinite(r).all() and (r > 0).all(), (level, mag, sign)
                assert (th > 0).all() and (th < np.pi).all(), (level, mag, sign)

                lr = -e.energy(x, None, lt)
                assert torch.isfinite(lr).all(), \
                    f"log_reward non-finite at level={level} x={sign * mag}"

                # the term step 2 will add must also be finite on the same states
                tree, _ = e._batch(x.shape[0])
                lj = log_jacobian(tree, r.reshape(-1), th.reshape(-1))
                assert torch.isfinite(lj).all(), \
                    f"log_jacobian non-finite at level={level} x={sign * mag}"

        # the wall must actually bite outside the box and be exactly zero inside it
        t1 = torch.tensor(1.0, dtype=DTYPE)
        inside = torch.zeros(4, e.data_ndim, dtype=DTYPE, device=DEVICE)
        assert float(e.bounding_energy(inside, t1).abs().max()) == 0.0, level
        outside = torch.full((4, e.data_ndim), 3.0, dtype=DTYPE, device=DEVICE)
        assert float(e.bounding_energy(outside, t1).min()) > 0.0, level

        # temperature pre-multiplication: the wall's contribution to -log R is the SAME at
        # any T. If it were not pre-multiplied it would scale as 1/T.
        contrib = {}
        for log_t in (-0.5, 0.0, 0.5):
            lt_ = torch.tensor(log_t, dtype=DTYPE)
            t_ = 10 ** lt_.to(DTYPE)
            contrib[log_t] = float((e.bounding_energy(outside, t_) / t_).mean())
        vals = list(contrib.values())
        assert max(vals) - min(vals) < 1e-12, \
            f"wall is not temperature-invariant in -log R: {contrib}"
    print("PASS gate 4  clamp holds r/theta on-domain at |x| up to 1e3; wall zero inside, "
          "temperature-invariant outside")


# ------------------------------------------------------------------ config cannot lie

def test_level_cannot_be_swallowed():
    try:
        ConformerTorsions(smiles=PROPANOL, device="cpu")
    except TypeError:
        pass
    else:
        raise AssertionError("ConformerTorsions constructed with no level")

    try:
        ConformerTorsions(smiles=PROPANOL, device="cpu", level="fulll")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown level was accepted")

    # **kwargs is gone: an unknown key must raise rather than be absorbed. This is the
    # mechanism that would otherwise make a chirality_coeff in energy_config never become
    # an attribute, so set_energy_coeffs' hasattr guard skips the ramp in silence.
    try:
        ConformerTorsions(smiles=PROPANOL, device="cpu", level="torsion",
                          chirality_coeff=3.0)
    except TypeError:
        pass
    else:
        raise AssertionError("an unknown kwarg was swallowed")
    print("PASS         level is required, validated, and unknown kwargs raise")


def test_linearity_is_measured():
    """The flags must be a measurement, not a default.

    spec_from_graph is called with use_geometry=False (required: a geometry-steered tree
    is not reproducible at load), and _linear_mask returns all-False when pos is None. So
    spec.angle_is_linear has been all-False for every molecule ever run. The energy now
    measures its own, off the reference conformer, against the geometry-free tree.
    """
    e = _energy(PROPANOL, level="torsion")
    assert e.linearity_verified is True
    assert e.angle_is_linear.shape == (e.n_th,)
    assert e.torsion_frame_is_linear.shape == (e.n_ph,)
    assert not e.angle_is_linear.any(), "propanol should have no linear angle"

    # a molecule that DOES have one: the spec's own flags stay all-False (the bug) while
    # the measured ones fire (the fix). Without this the fix is untestable on any molecule
    # the force-field tables cover.
    lin = ConformerTorsions(smiles="C#CCO", device="cpu", level="dihedral")
    assert not np.asarray(lin.spec.angle_is_linear).any(), \
        "spec_from_graph(use_geometry=False) is supposed to report no linear angles; if " \
        "it now measures them, this fix is redundant and should be removed"
    assert lin.angle_is_linear.any(), \
        "the measured flags missed the alkyne in C#CCO"
    # and the flagged DoF are HELD, so they never reach log sin(theta) -> -inf
    held = lin.n_r + np.flatnonzero(lin.angle_is_linear)
    assert not lin.free_mask[held].any()
    print(f"PASS         linearity measured (C#CCO: {int(lin.angle_is_linear.sum())} "
          f"linear angle(s) found and held; spec reports 0)")


def test_propanol_widths():
    for level in ConformerTorsions.LEVELS:
        e = _energy(PROPANOL, level=level)
        n = e.spec.n_atoms
        assert e.n_r + e.n_th + e.n_ph == 3 * n - 6
        print(f"     {PROPANOL} {level:9s} N={n} 3N-6={3 * n - 6} d={e.data_ndim}")
    print("PASS         propanol widths")


# ------------------------------------------------------- step 2: the change of measure

def _bat_log_jacobian(e, x):
    """The BAT volume element, recomputed here from first principles.

    Deliberately NOT compared against the autograd determinant of `build`. That map is
    SE(3)-reduced and square, and this element relates the internal measure to the FULL
    3N Cartesian measure with the 6 external DoF integrated -- the two differ by the
    orbit volume log(r_1^2 * r_2 * sin theta_2), so that comparison fails on correct code
    by ~0.4-0.8 nats and would invite someone to "fix" the right answer into the wrong one.
    """
    r, th, _ = e.dof_from_state(x)
    return 2.0 * torch.log(r).sum(-1) + torch.log(torch.sin(th)).sum(-1)


def test_measure_term():
    for level in ("torsion", "flex", "full"):
        e = _energy(PROPANOL, level=level)
        torch.manual_seed(3)
        x = torch.rand(64, e.data_ndim, dtype=DTYPE, device=DEVICE) * 1.6 - 0.8

        # (a) wiring: jacobian_energy is -T * (the BAT element), not something else
        for log_t in (-0.4, 0.0, 0.7):
            t = 10 ** torch.tensor(log_t, dtype=DTYPE)
            assert torch.allclose(e.jacobian_energy(x, t), -t * _bat_log_jacobian(e, x),
                                  atol=1e-12, rtol=1e-12), (level, log_t)

        # (b) THE gate: the measure's contribution to log_reward is +log J, independent
        # of temperature. A single-temperature test passes on the un-compensated code,
        # which is the whole trap.
        contrib = {}
        for log_t in (-0.4, 0.0, 0.7):
            lt = torch.tensor(log_t, dtype=DTYPE)
            t = 10 ** lt.to(DTYPE)
            contrib[log_t] = -e.energy(x, None, lt) + e.potential_energy(x, t) / t
        ref = _bat_log_jacobian(e, x)
        for log_t, c in contrib.items():
            assert torch.allclose(c, ref, atol=1e-10, rtol=1e-10), (level, log_t)
        spread = max(float((contrib[a] - contrib[b]).abs().max())
                     for a in contrib for b in contrib)
        assert spread < 1e-10, (level, spread)

        # ...and re-introducing the omission must FAIL that. Drop the T pre-multiplication
        # and the contribution starts scaling as 1/T.
        saved = e.jacobian_energy
        e.jacobian_energy = lambda xx, tt: -_bat_log_jacobian(e, xx)
        try:
            broken = {}
            for log_t in (-0.4, 0.7):
                lt = torch.tensor(log_t, dtype=DTYPE)
                t = 10 ** lt.to(DTYPE)
                broken[log_t] = -e.energy(x, None, lt) + e.potential_energy(x, t) / t
            bad_spread = float((broken[-0.4] - broken[0.7]).abs().max())
            assert bad_spread > 1e-6, \
                "dropping the temperature pre-multiplication was NOT detected; this gate " \
                "is blind and a single-temperature version of it would be worse"
        finally:
            e.jacobian_energy = saved

        # (c) sign and assembly: energy == potential/T - log J. Flipping the sign must fail.
        lt = torch.tensor(0.0, dtype=DTYPE)
        t = 10 ** lt.to(DTYPE)
        assert torch.allclose(e.energy(x, None, lt), e.potential_energy(x, t) / t - ref,
                              atol=1e-10, rtol=1e-10), level
        assert not torch.allclose(e.energy(x, None, lt),
                                  e.potential_energy(x, t) / t + ref,
                                  atol=1e-10, rtol=1e-10), \
            f"{level}: the measure term is symmetric under a sign flip, so this cannot " \
            f"catch a sign error -- log J is probably identically zero here"

        # (d) constancy is exactly the freeze condition, and 'always on' does real work
        varies = float(ref.max() - ref.min())
        if e._lin_free_idx.numel() == 0:
            assert e.log_jacobian_const is not None and varies < 1e-12, (level, varies)
        else:
            assert e.log_jacobian_const is None and varies > 1e-3, (level, varies)
        note = (f"const {e.log_jacobian_const:.4f}" if e.log_jacobian_const is not None
                else f"varies over {varies:.3f} nats")
        print(f"     {level:9s} log J {note}")
    print("PASS         measure term: BAT element, T-invariant in log R, sign checked, "
          "and dropping the T factor is DETECTED")


def test_baked_energy_excludes_measure():
    """bake_energies must store the potential, and the read side adds the measure back.

    Baking `energy()` would make the measure scale as 1/T, since the stored value is
    divided by the sampling temperature on read -- and a change of measure is by
    definition temperature-independent.
    """
    from energies.conformer_data import bake_energies

    e = _energy(PROPANOL, level="torsion")
    torch.manual_seed(5)
    x = torch.rand(32, e.data_ndim, dtype=DTYPE, device=DEVICE) * 2 - 1
    one = torch.tensor(1.0, dtype=DTYPE)

    baked = bake_energies(e, x)
    assert torch.equal(baked, e.potential_energy(x, one)), \
        "bake_energies is not storing the bare potential"
    assert not torch.allclose(baked, e.energy(x, None, torch.tensor(0.0, dtype=DTYPE))), \
        "the baked value is indistinguishable from energy(); the measure leaked in"

    # the read side reconstructs the true log reward at ANY temperature
    class _Mols:
        pass
    for log_t in (-0.4, 0.0, 0.7):
        t = float(10 ** log_t)
        m = _Mols()
        m.conformer_energy = baked
        got = e.prebuilt_sample_to_reward(m, torch.tensor(t, dtype=DTYPE))
        want = -e.energy(x, None, torch.tensor(log_t, dtype=DTYPE))
        assert torch.allclose(got, want, atol=1e-10, rtol=1e-10), (log_t,
                                                                   float((got - want).abs().max()))

    # and at a level where the measure is state-dependent it must REFUSE, not return a
    # measure-free reward
    ef = _energy(PROPANOL, level="full")
    m = _Mols()
    m.conformer_energy = bake_energies(ef, torch.zeros(4, ef.data_ndim, dtype=DTYPE))
    try:
        ef.prebuilt_sample_to_reward(m, torch.tensor(1.0, dtype=DTYPE))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("prebuilt_sample_to_reward silently dropped a state-dependent "
                             "measure term at level 'full'")
    print("PASS         baked energy is measure-free; read side reconstructs log R at any T "
          "and refuses when the measure is state-dependent")


# ------------------------------------------------------------------- the prior

def test_state_dof_roundtrip():
    """state_from_dof must invert dof_from_state, and refuse where it cannot."""
    for level in ("dihedral", "flex", "full"):
        e = _energy(PROPANOL, level=level)
        assert not e.collective, level
        torch.manual_seed(11)
        x = torch.rand(128, e.data_ndim, dtype=DTYPE, device=DEVICE) * 1.8 - 0.9
        back = e.state_from_dof(*e.dof_from_state(x))
        assert torch.allclose(back, x, atol=1e-12, rtol=1e-12), \
            (level, float((back - x).abs().max()))

    # a collective level must REFUSE, not return a plausible-looking inverse
    e = _energy(GATE_SMILES, level="torsion")
    assert e.collective
    try:
        e.state_from_dof(*e.dof_from_state(torch.zeros(2, e.data_ndim, dtype=DTYPE)))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("state_from_dof inverted a collective map")
    print("PASS         state<->dof round-trips exactly on selection levels; "
          "torsion refuses")


def test_internal_prior_beats_uniform():
    """The whole point of the fitted prior: better terminals than uniform-on-box.

    Asserted on the ENERGY, not on the wiring, because a mis-ordered column map would
    still round-trip and still produce finite numbers -- it would just produce garbage
    geometry. A prior that does not beat uniform is a prior that is indexed wrong.
    """
    from pathlib import Path
    p = Path("conformer_prior.pt")
    if not p.exists():
        print("SKIP         no fitted prior at conformer_prior.pt")
        return
    fitted = torch.load(p, weights_only=False)
    lt = torch.tensor(0.0, dtype=DTYPE)
    for level in ("dihedral", "flex", "full"):
        e = _energy(PROPANOL, level=level)
        xp, stats = e.sample_prior_states(fitted, 512, np.random.default_rng(0),
                                          report=False)
        torch.manual_seed(0)
        xu = torch.rand(512, e.data_ndim, dtype=DTYPE, device=DEVICE) * 2 - 1
        ep, eu = e.energy(xp, None, lt), e.energy(xu, None, lt)
        assert torch.isfinite(ep).all(), level
        assert xp.abs().max() <= 1.0 + 1e-12, (level, float(xp.abs().max()))
        mp, mu = float(ep.median()), float(eu.median())
        u = stats['n_uniform']
        print(f"     {level:9s} median E  prior {mp:12.1f}   uniform {mu:12.1f}   "
              f"(fallback r{u['r']} th{u['theta']} ph{u['phi']}, "
              f"clip r {stats['clip_frac']['r']:.0%} th {stats['clip_frac']['theta']:.0%})")
        assert mp < mu, \
            (f"{level}: the fitted prior is no better than uniform (prior {mp:.1f} vs "
             f"uniform {mu:.1f}) -- the per-DoF column map is probably mis-indexed")
        assert stats['n_ring_marginal'] == 0, "propanol is acyclic"
    print("PASS         fitted InternalPrior beats uniform-on-box at every selection level")


RING_MOLS = [('cyclohexane', 'C1CCCCC1'), ('toluene', 'Cc1ccccc1'),
             ('naphthalene', 'c1ccc2ccccc2c1'), ('proline', 'OC(=O)C1CCCN1'),
             ('ibuprofen', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O')]


def test_ring_closure():
    """Ring systems must come out CLOSED. Asserted on the closure bond, not the energy.

    Ring closure is the second place a product of marginals cannot work, and unlike the
    sibling case there is no purely structural fix -- the ring block has to come from
    InternalPrior's joint bank/subspace, or be held near the reference. The
    no-ring-handling path is required to FAIL, so this cannot pass blind.

    BOTH PRIORS, AND THE BANKED PATH IS REQUIRED TO OCCUR. Run on conformer_prior.pt alone
    this gate exercised the HELD path only -- that prior predates the ring-signature fix,
    so no key resolves and all five molecules report 0 banked. The docstring said both
    paths were checked; the measurement said otherwise. conformer_prior_v2.pt is where a
    saturated ring actually reaches its fitted pucker subspace, and the assertion below
    requires at least one to, so the banked path cannot go untested again in silence.
    """
    from pathlib import Path
    from mxtaltools.conformers.builder import closure_length
    if not Path('conformer_prior.pt').exists():
        raise AssertionError("conformer_prior.pt missing -- this gate cannot run, and a "
                             "silent skip is how it would disappear on another machine")
    n = 256
    for tag, path in (('v1/held', 'conformer_prior.pt'), ('v2/banked', 'conformer_prior_v2.pt')):
        if not Path(path).exists():
            raise AssertionError(f'{path} missing -- this gate needs both priors: v1 '
                                 f'exercises the held path, v2 the banked one')
        prior = torch.load(path, weights_only=False)
        banked_total = 0
        for name, smi in RING_MOLS:
            e = _energy(smi, level='full')
            tree, ff = e._batch(n)
            assert ff.closure_index.numel() > 0, f'{name} has no ring closure bond'

            def cerr(**kw):
                x, st = e.sample_prior_states(prior, n, np.random.default_rng(0),
                                              report=False, **kw)
                cl = closure_length(tree, e.build_positions(x))
                return float((cl - ff.closure_r0).abs().reshape(n, -1).max(1).values.median()), st

            off, off_st = cerr(joint_rings=False)
            on, st = cerr()
            assert on < 0.25, f'{tag} {name}: closure error {on:.3f} A with rings wired'
            assert off > 4 * on, \
                (f'{tag} {name}: turning ring handling OFF barely changed closure '
                 f'({off:.3f} vs {on:.3f} A) -- this gate cannot detect the bug it '
                 f'exists for')
            # the OFF arm's own stats must SEE that breakage. The monitor used to be gated
            # on the ring-system count, which is 0 there, so it reported a perfect 0.000.
            assert abs(off_st['closure_err'] - off) < 1e-9, \
                (f'{tag} {name}: the sampler reported closure_err '
                 f'{off_st["closure_err"]:.3f} A with joint rings OFF while the true error '
                 f'is {off:.3f} A -- the monitor is gated on the ring path again')
            # every ring system is accounted for by exactly one path
            assert st['n_rings'] == st['n_ring_banked'] + st['n_ring_thermal'], (name, st)
            banked_total += st['n_ring_banked']
            print(f"     {tag:10s} {name:12s} closure {off:.2f} A -> {on:.3f} A   "
                  f"({st['n_ring_banked']} banked, {st['n_ring_thermal']} held, "
                  f"{st['n_ring_extra_held']} extra DoF held)")
        if tag.startswith('v2'):
            assert banked_total > 0, \
                ('no ring reached a bank or pucker subspace under the v2 prior, so this '
                 'gate is still testing the HELD path only and would pass with ring '
                 'banking entirely removed')

    print('PASS         ring closure holds on 5 ring types, under BOTH the held (v1) and '
          'banked (v2) paths')


def _ring_key(energy, prior):
    """The (signature, n_dof) key for a molecule's single ring system."""
    from energies.conformer_data import condition_from_energy
    m = condition_from_energy(energy, partial_charges=False)
    m.build_conformer_tree()
    _, blocks, sigs, _ = prior._layout(m)
    s = list(blocks)[0]
    cols = blocks[s]
    nd = len(cols['r']) + len(cols['theta']) + len(cols['phi'])
    return (sigs[s], nd), nd


def _ring_torsions(e, x, n):
    """The six ring torsions of a 6-ring, in degrees, [n, 6]."""
    from mxtaltools.conformers.geometry import dihedral
    z = np.asarray(e.spec.z)
    ring = [i for i in range(len(z)) if e.atom_in_ring[i] and z[i] > 1]
    adj = {i: [] for i in ring}
    for a, b in np.asarray(e.spec.graph_bond_index):
        a, b = int(a), int(b)
        if a in adj and b in adj:
            adj[a].append(b); adj[b].append(a)
    cyc, prev = [ring[0]], None
    while len(cyc) < 6:
        nxt = [y for y in adj[cyc[-1]] if y != prev][0]
        prev = cyc[-1]; cyc.append(nxt)
    pos = e.build_positions(x).reshape(n, -1, 3)
    return np.stack([np.degrees(dihedral(pos[:, cyc[i]], pos[:, cyc[(i + 1) % 6]],
                                         pos[:, cyc[(i + 2) % 6]], pos[:, cyc[(i + 3) % 6]]).numpy())
                     for i in range(6)], 1)


def test_ring_bank_rules():
    """Two rules decide whether a ring is banked; each is checked against its own
    counterfactual so neither can pass for the other's reason.

    AROMATIC rings are never banked -- rigid, so a bank buys nothing, and under signature
    version 1 it actively hurt: benzene shared cyclohexane's bank and drew chairs at a
    median |ring torsion| of 47 deg. SATURATED rings are banked only above a row count,
    since a single row is one observation on replay. The threshold is 2, not higher:
    with a v2 signature and a purpose-built bank, pyrrolidine's two envelope basins
    are COMPLETE rather than thin.
    """
    from pathlib import Path
    from dataclasses import replace
    from mxtaltools.conformers.prior import RingBank
    prior = torch.load(Path('conformer_prior.pt'), weights_only=False)

    # aromatic: refused, and NOT because of the row count
    for name, smi in (('benzene', 'c1ccccc1'), ('naphthalene', 'c1ccc2ccccc2c1')):
        e = _energy(smi, level='full', ring_min_bank_rows=1)
        assert all(b is None for _, b, _ in e.ring_blocks(prior)), \
            f'{name} was banked despite being aromatic, even at ring_min_bank_rows=1'
    e = _energy('c1ccccc1', level='full')
    n = 256
    x, _ = e.sample_prior_states(prior, n, np.random.default_rng(0), report=False)
    med = float(np.median(np.abs(_ring_torsions(e, x, n))))
    assert med < 5.0, f'benzene ring is not planar: median |ring torsion| {med:.1f} deg'

    # saturated: the row count decides. Tested against a SYNTHETIC bank so it depends on
    # neither what the shipped fit contains nor on the signature version.
    e = _energy('C1CCCCC1', level='full')
    key, nd = _ring_key(e, prior)
    row = np.concatenate([e.r0.numpy(), e.th0.numpy(), e.ph0.numpy()])[:nd]
    for n_rows, expect in ((1, False), (4, True)):
        fake = replace(prior, rings={key: RingBank(rows=np.tile(row, (n_rows, 1)))})
        got = any(b is not None for _, b, _ in e.ring_blocks(fake))
        assert got is expect, \
            (f'saturated ring with {n_rows} bank rows: banked={got}, expected {expect} '
             f'(ring_min_bank_rows={e.ring_min_bank_rows})')
    print(f'PASS         aromatic rings never banked and planar (benzene median '
          f'|ring torsion| {med:.1f} deg); saturated rings gated on bank row count')


def test_stale_ring_signature_detected():
    """A prior pickled before ``ring_sig_version`` existed must read as STALE.

    InternalPrior is a dataclass, so a defaulted field is also a CLASS attribute and
    getattr on an old pickle returns the current default -- reporting itself up to date
    while none of its ring keys can resolve. Every ring then silently falls through to the
    hold, which is indistinguishable from a molecule whose ring was never fitted.
    """
    from pathlib import Path
    from mxtaltools.conformers.prior import InternalPrior
    old = torch.load(Path('conformer_prior.pt'), weights_only=False)
    assert getattr(old, 'ring_sig_version', 1) == 2, \
        'the class default is no longer 2, so this test no longer exercises the trap'
    assert vars(old).get('ring_sig_version', 1) == 1, 'the shipped prior should read stale'
    assert vars(InternalPrior()).get('ring_sig_version', 1) == 2, 'a fresh fit should read 2'
    e = _energy('C1CCCCC1', level='full')
    e.ring_blocks(old)
    assert e.ring_sig_stale is True, 'a pre-fix prior was not flagged stale'
    print('PASS         stale ring signature detected via vars(), not getattr')


def test_float32_default_is_sound():
    """The SHIPPED precision must agree with float64 on everything this code reports.

    The exactness gates above pin the formula at float64 on purpose; this pins the working
    precision. Asserted on the reported quantities -- potential, closure, log J -- against
    the same DoF draw, because that is what a wrong dtype would corrupt. The bar is 1e-5
    relative: float32 epsilon is ~1.2e-7 and `build` walks ~8 topological rounds, so
    anything compounding through the NeRF chain would blow past this.
    """
    from mxtaltools.conformers.builder import closure_length, log_jacobian
    assert ConformerTorsions(smiles=PROPANOL, device="cpu", level="full").dtype         is torch.float32, "the shipped default is no longer float32"

    for smi in (PROPANOL, "CCC1CCCCC1"):
        e64 = ConformerTorsions(smiles=smi, device="cpu", level="full",
                                force_field="mmff", dtype=torch.float64)
        e32 = ConformerTorsions(smiles=smi, device="cpu", level="full",
                                force_field="mmff", dtype=torch.float32)
        torch.manual_seed(4)
        x64 = (torch.rand(256, e64.data_ndim, dtype=torch.float64) * 1.6 - 0.8)
        x32 = x64.to(torch.float32)
        t = float(e64.temperature)

        u64 = e64.potential_energy(x64, t).numpy()
        u32 = e32.potential_energy(x32, t).numpy().astype("float64")
        rel = float(np.median(np.abs(u32 - u64) / np.maximum(np.abs(u64), 1e-9)))
        assert rel < 1e-5, f"{smi}: float32 potential differs by {rel:.2e} relative"

        r64, th64, _ = e64.dof_from_state(x64)
        r32, th32, _ = e32.dof_from_state(x32)
        tr64, ff64 = e64._batch(256)
        tr32, ff32 = e32._batch(256)
        j64 = log_jacobian(tr64, r64.reshape(-1), th64.reshape(-1)).numpy()
        j32 = log_jacobian(tr32, r32.reshape(-1), th32.reshape(-1)).numpy().astype("float64")
        assert np.abs(j32 - j64).max() < 1e-4, "float32 log J drifted"

        if ff64.closure_index.numel():
            c64 = closure_length(tr64, e64.build_positions(x64))
            c32 = closure_length(tr32, e32.build_positions(x32)).double()
            dc = float((c32 - c64).abs().max())
            # the closure errors this code REPORTS are 0.02-0.09 A; noise must sit far below
            assert dc < 1e-4, f"{smi}: float32 closure differs by {dc:.2e} A"
        assert torch.isfinite(e32.energy(x32, None, torch.tensor(0.0))).all(), smi
    print("PASS         float32 default agrees with float64 on U, log J and closure")


TESTS = [test_torsion_bitwise, test_rotatable_axes_ordered, test_layout_reaches_gfn,
         test_domain_guarantee, test_level_cannot_be_swallowed, test_linearity_is_measured,
         test_propanol_widths, test_measure_term, test_baked_energy_excludes_measure,
         test_state_dof_roundtrip, test_internal_prior_beats_uniform,
         test_ring_closure, test_ring_bank_rules,
         test_stale_ring_signature_detected, test_float32_default_is_sound]

if __name__ == "__main__":
    torch.set_default_dtype(DTYPE)
    failed = 0
    for t in TESTS:
        try:
            t()
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
