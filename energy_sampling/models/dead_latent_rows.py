"""
Which crystal latent rows are structurally dead, per space group.

The crystal latent is

    [0:3]  normed aunit lengths
    [3:6]  cell angles  (alpha, beta, gamma)
    [6 : 6+3*zp]        aunit centroids
    [6+3*zp : 6+6*zp]   aunit orientations (spherical rotvec)

`latent_to_cell_params` runs `enforce_crystal_system` after `inv_latent_transform`,
which OVERWRITES some cell angles with constants. Those latent components are read
and discarded: the crystal, and therefore the energy, does not depend on them.

Why this matters, and why it is not merely wasted capacity
----------------------------------------------------------
`latent_params()` recomputes the latent from the *built* cell, so a clobbered angle
round-trips to the canonical value (0.0) regardless of what went in. Every prior and
replay row therefore carries exactly 0.0 in those columns -- measured std 0.0e+00 on
the stored sg-14 prior, n=535 and n=3582 -- while the energy is flat across the whole
box. `bwd` starts from buffer states and so trains P_F toward a delta at 0; `fwd`
gets no gradient signal at all. Those dims are trained against inconsistent targets.
See docs/findings.md F-009 and docs/decisions.md D33.

The fix is to keep them out of the SDE and scatter the canonical value back in at the
energy boundary. This module answers only "which rows", and verifies its own answer.

The rule: a row is dead iff enforce_crystal_system writes a CONSTANT to it
-------------------------------------------------------------------------
Read off the live vectorized implementation (geometry_utils.py:1375). Angle rows
only; no length row is ever set to a constant.

    triclinic     no-op                                  -> ()
    monoclinic    alpha = gamma = pi/2                    -> (3, 5)
    orthorhombic  alpha = beta = gamma = pi/2             -> (3, 4, 5)
    tetragonal    angles = pi/2       (a = b)             -> (3, 4, 5)
    hexagonal     al = be = pi/2, ga = 2pi/3  (a = b)     -> (3, 4, 5)
    cubic         angles = pi/2       (a = b = c)         -> (3, 4, 5)
    rhombohedral  all means                               -> () and UNREACHABLE

The parenthesised length constraints are NOT dead rows. `a = b` is imposed as
`mean(a, b)`, so perturbing `a` alone does move the cell -- the degenerate direction
is the antisymmetric combination `a - b`, which is diagonal in this basis and needs
reparameterisation rather than row deletion. Same for the rhombohedral means. The
probe below classifies them correctly (as live) for exactly this reason.

Rhombohedral is unreachable in practice: LATTICE_TYPE assigns all 230 space groups
across triclinic (2), monoclinic (13), orthorhombic (59), tetragonal (68), hexagonal
(52) and cubic (36), and never rhombohedral. Kept in the table so a future
LATTICE_TYPE change surfaces here rather than silently.

Trigonal likewise has no code of its own -- those groups are folded into hexagonal or
rhombohedral upstream -- so there is no trigonal branch to mirror.
"""
from typing import Optional, Sequence, Tuple

from mxtaltools.constants.space_group_info import LATTICE_TYPE, CONTINUOUS_DIMS

# crystal system -> angle rows enforce_crystal_system overwrites with a constant
_CONSTANT_ANGLE_ROWS = {
    'triclinic': (),
    'monoclinic': (3, 5),
    'orthorhombic': (3, 4, 5),
    'tetragonal': (3, 4, 5),
    'hexagonal': (3, 4, 5),
    'cubic': (3, 4, 5),
    'rhombohedral': (),  # unreachable; means, not constants
}


def latent_ndim(max_z_prime: int) -> int:
    """Full width of the canonical crystal latent."""
    return 6 + 6 * int(max_z_prime)


def free_centroid_rows(sg_ind: int, max_z_prime: int = 1) -> Tuple[int, ...]:
    """
    Latent rows of the FIRST aunit centroid that lie along a free axis -- the shared +1
    eigenspace of G (findings.md F-008). Moving the centroid along one of these rigidly
    translates the whole crystal, so the energy and RDF are exactly invariant; the
    coordinate is pure origin choice. Canonicalised to the aunit box centre (latent 0) by
    `CrystalBatch.canonicalize_free_axes`, which is what makes them safe to hold.

    EMPTY at Z' > 1, matching that canonicaliser's own gate. There the free translation is
    ONE GLOBAL shift shared by all Z' units, so fixing it means a common offset rather than
    setting each unit to the centre -- the relative offsets along a free axis are physical.
    The common shift then pushes other units out of the box, and re-wrapping a single unit
    by auv_d is a symmetry only when auv_d == 1. Open question, so not claimed here.
    """
    if int(max_z_prime) > 1:
        return ()
    axes = sorted({axis
                   for vec in CONTINUOUS_DIMS.get(str(int(sg_ind)), [])
                   for axis, comp in enumerate(vec) if comp})
    return tuple(6 + a for a in axes)      # first centroid block only


def dead_latent_rows(sg_ind: int, max_z_prime: int = 1) -> Tuple[int, ...]:
    """
    Rows of the canonical crystal latent that carry no information for this space group:
    the cell angles enforce_crystal_system overwrites with a constant, plus the free
    centroid axes canonicalize_free_axes pins to the box centre.

    The angle block is Z'-independent (always rows 3:6). The free-axis rows are not --
    see free_centroid_rows.
    """
    try:
        system = str(LATTICE_TYPE[int(sg_ind)]).lower()
    except KeyError:
        raise ValueError(
            f"space group {sg_ind} has no crystal system in LATTICE_TYPE; valid space "
            f"groups are 1-230") from None
    if system not in _CONSTANT_ANGLE_ROWS:
        raise ValueError(
            f"SG{int(sg_ind)} has crystal system {system!r}, which has no entry in "
            f"_CONSTANT_ANGLE_ROWS. enforce_crystal_system may have gained a branch "
            f"this table does not mirror -- do not guess, read geometry_utils.py.")
    return tuple(sorted(set(_CONSTANT_ANGLE_ROWS[system])
                        | set(free_centroid_rows(sg_ind, max_z_prime))))


def resolve_dead_rows(sg_ind: int, is_crystal: bool, max_z_prime: int = 1) -> Tuple[int, ...]:
    """
    THE entry point. Everything that needs dead rows must come through here rather
    than calling dead_latent_rows directly, because of the toy gate below.

    Toy energies (latent_harmonic, latent_multiharmonic) carry `space_groups: [1]` as a
    PLACEHOLDER -- their state is not a crystal parameterization and no cell is ever
    built from it, so nothing is dead and every dim is real. Gating on the space group
    alone would be wrong in a way that hides: P1 is triclinic, so the crystal-system
    angle rows are empty and the toy would look fine today, but P1 has all THREE aunit
    centroid axes free (CONTINUOUS_DIMS['1'], see findings.md F-008). The moment free
    axes join this table, an ungated resolver would freeze 3 of a toy's 12 dims -- dims
    its energy genuinely depends on -- and present as unexplained loss of coverage
    rather than as an error.

    Same gate, and same reason, as `do_periodic_angles=self.energy_function.is_crystal`
    and the is_crystal check in _resolve_periodic_centroid_axes.
    """
    if not is_crystal:
        return ()
    return dead_latent_rows(sg_ind, max_z_prime)


def live_latent_rows(sg_ind: int, max_z_prime: int, is_crystal: bool = True) -> Tuple[int, ...]:
    """Complement of resolve_dead_rows over the full latent width."""
    dead = set(resolve_dead_rows(sg_ind, is_crystal, max_z_prime))
    return tuple(d for d in range(latent_ndim(max_z_prime)) if d not in dead)


def describe(sg_ind: int, max_z_prime: int, is_crystal: bool = True) -> str:
    n = latent_ndim(max_z_prime)
    if not is_crystal:
        return (f"non-crystal energy: no dead latent rows (space_groups is a placeholder "
                f"here); SDE flows the full {n} dims")
    dead = resolve_dead_rows(sg_ind, is_crystal, max_z_prime)
    system = str(LATTICE_TYPE[int(sg_ind)]).lower()
    if not dead:
        return (f"SG{int(sg_ind)} ({system}), Z'<={max_z_prime}: no dead latent rows; "
                f"SDE flows the full {n} dims")
    names = {3: 'alpha', 4: 'beta', 5: 'gamma', 6: 'cen_x(free)', 7: 'cen_y(free)', 8: 'cen_z(free)'}
    which = ', '.join(f"{d}={names.get(d, '?')}" for d in dead)
    return (f"SG{int(sg_ind)} ({system}), Z'<={max_z_prime}: dead latent rows [{which}] "
            f"held out of the SDE; flowing {n - len(dead)} of {n} dims")


def probe_dead_rows(crystal_batch,
                    max_z_prime: int,
                    values: Sequence[float] = (-0.9, -0.3, 0.3, 0.9),
                    atol: float = 0.0) -> Tuple[int, ...]:
    """
    Empirically discover which latent rows the crystal build ignores, by forcing each
    row across the box and checking whether any cell parameter moves.

    This is the guard, not the source of truth: `dead_latent_rows` is the tabulated
    answer and this asserts against it. The tabulated version is needed at model
    construction time, before a real batch exists; the probe is what catches the table
    drifting away from enforce_crystal_system -- which is precisely the failure that
    produced this bug (8dea6b56 moved reduction into a penalty that the projection
    preempts, and nothing noticed for weeks).

    `crystal_batch` must carry realistic, NON-DEGENERATE cell parameters. A batch that
    already satisfies a constraint cannot reveal it: with a == b going in, the
    tetragonal `mean(a, b)` is the identity and row 0 would look dead when it is not.

    atol defaults to 0.0 -- exact. A clobbered row produces a bit-identical cell, not
    an approximately identical one, so any tolerance here would only mask a real
    dependence.
    """
    import torch

    base_latents = crystal_batch.latent_params().clone()
    n = latent_ndim(max_z_prime)
    if base_latents.shape[-1] != n:
        raise ValueError(f"batch latent width {base_latents.shape[-1]} != expected {n} "
                         f"for max_z_prime={max_z_prime}")

    def canonical_params():
        """
        Cell parameters of the CANONICAL representative.

        Comparing canonical representatives rather than raw ones is what lets a single
        probe cover both kinds of dead row. A clobbered angle never reaches the cell at
        all, so it is dead under either comparison. A FREE centroid axis does reach the
        cell -- the crystal genuinely moves -- but the move is a rigid translation that
        canonicalize_free_axes undoes, so it is dead only under this comparison. Using
        raw params would report free axes LIVE and make the table disagree with itself.
        """
        crystal_batch.canonicalize_free_axes()
        return crystal_batch.full_cell_parameters()

    crystal_batch.latent_to_cell_params(base_latents.clone())
    reference = canonical_params().clone()

    dead = []
    for row in range(n):
        moved = False
        for v in values:
            trial = base_latents.clone()
            trial[:, row] = v
            crystal_batch.latent_to_cell_params(trial)
            delta = (canonical_params() - reference).abs().amax()
            if float(delta) > atol:
                moved = True
                break
        if not moved:
            dead.append(row)

    crystal_batch.latent_to_cell_params(base_latents)  # leave the batch as we found it
    return tuple(dead)


def verify_dead_rows(crystal_batch,
                     sg_ind: int,
                     max_z_prime: int,
                     expected: Optional[Sequence[int]] = None,
                     is_crystal: bool = True) -> Tuple[int, ...]:
    """
    Run the probe and assert it agrees with the tabulated answer. Returns the
    (agreed) dead rows.

    Crystal-only: the probe drives latent_to_cell_params/full_cell_parameters, which a
    toy state has no meaning for. Callers should skip it entirely when not is_crystal
    rather than relying on this guard.
    """
    if not is_crystal:
        raise ValueError(
            "verify_dead_rows is crystal-only -- it probes latent_to_cell_params, which "
            "a non-crystal (latent_harmonic/latent_multiharmonic) state does not feed. "
            "resolve_dead_rows already returns () for these; do not probe them.")
    expected = tuple(resolve_dead_rows(sg_ind, is_crystal, max_z_prime)) if expected is None else tuple(sorted(expected))
    found = probe_dead_rows(crystal_batch, max_z_prime)
    if found != expected:
        raise AssertionError(
            f"dead-row table disagrees with the crystal build for SG{int(sg_ind)}: "
            f"table says {expected}, probing latent_to_cell_params says {found}. "
            f"enforce_crystal_system (geometry_utils.py) and _CONSTANT_ANGLE_ROWS "
            f"(models/dead_latent_rows.py) have drifted -- fix the table, do not "
            f"suppress this check. If the probe batch has degenerate cell parameters "
            f"(e.g. a == b) it can also report a live row as dead.")
    return expected
