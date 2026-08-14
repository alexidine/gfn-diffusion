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


TESTS = [test_torsion_bitwise, test_rotatable_axes_ordered, test_layout_reaches_gfn,
         test_domain_guarantee, test_level_cannot_be_swallowed, test_linearity_is_measured,
         test_propanol_widths, test_measure_term, test_baked_energy_excludes_measure,
         test_state_dof_roundtrip, test_internal_prior_beats_uniform]

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
