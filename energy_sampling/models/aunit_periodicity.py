"""
Which asymmetric-unit centroid axes are trivially periodic, per space group.

Crystals are parameterized with the aunit centroid in fractional cell coords, bounded
by the aunit box [0, auv] (auv = ASYM_UNITS[sg]). The GFN latent chain is

    latent in [-1, 1]  ->  u = latent/2 + 0.5 in [0, 1]  ->  frac = u * auv

so wrapping a latent dim with period 2 is exactly the shift frac: 0 -> auv.

The rule: auv_d == 1
--------------------
The aunit is NOT periodic in general -- crossing a face usually re-enters through a
symmetry operation rather than a translation (P21/c at y=1/4 being the standard
example), which is a genuine discontinuity. But where the aunit spans the full cell
width (auv_d == 1), the wrap is a shift by one whole cell along d, i.e. a plain
lattice translation, which is a symmetry of every space group by construction. So:

    auv_d == 1  =>  the coordinate is genuinely circular, for every space group.

There the sampler can flow straight through the face instead of being held off it by
the bounding-energy wall -- which is exactly where real probability mass tends to
bunch, precisely because the face is an artificial cut rather than a physical one.

Why not auv_d == 0.5
--------------------
A shift by 1/2 is also often a symmetry (it is iff every entry of (R_k[:,d] - e_d) is
even, over the ops (R_k | t_k)) -- true for the axis-preserving monoclinic/orthorhombic
groups, false where ops permute axes. Measured below: 138/219 such axes are wrap-safe.
It is deliberately NOT used here: a half-cell wrap identifies points within the
fundamental domain, which is a separate question that this code does not yet address.
Only the auv == 1 lattice-translation case is claimed.

Assumes standard symmetry settings, which is what we train on: reset_sg_info always
sets nonstandard_symmetry=False and takes the ops straight from SYM_OPS. A nonstandard
setting re-defines the axes and could behave differently -- don't reuse this there.

Validation (2026-07-16)
-----------------------
Direct experiment over all 123 space groups with defined aunit bounds: real Z'=1 CSD
molecules (new_prot_csd.pt), random valid cells, comparing elj energy AND crystal RDF
at u=0 vs u->1 against a delta-step noise floor, with a generic-offset sensitivity
control (so "the energies matched" cannot be a false positive from an energy that
ignores the coordinate). Cross-checked against the exact symmetry-operator derivation
above: the two agree on 369/369 (sg, axis) pairs. auv==1 was wrap-safe in 80/80 cases
-- no exceptions -- vs 138/219 for auv==0.5, 3/61 for auv==0.25 and 0/9 for auv==0.125.
"""
from typing import Tuple

from mxtaltools.constants.asymmetric_units import RAW_ASYM_UNITS


def sg_periodic_centroid_axes(sg_ind: int) -> Tuple[int, ...]:
    """
    Axes (0=x, 1=y, 2=z) whose aunit centroid coordinate is trivially periodic for this
    space group: those whose aunit spans the full cell width, where wrapping u from 1
    back to 0 is a whole-cell lattice translation and so reproduces the same crystal.

    Space groups without defined aunit bounds get none: ASYM_UNITS silently fills those
    in as [1,1,1] (i.e. treats them as P1), which is a placeholder rather than a real
    aunit, and trusting it would wrap axes whose true aunit is narrower. Reading
    RAW_ASYM_UNITS rather than ASYM_UNITS is what keeps those out.
    """
    auv = RAW_ASYM_UNITS.get(str(int(sg_ind)))
    if auv is None:
        return ()
    return tuple(d for d in range(3) if float(auv[d]) == 1.0)


def describe(sg_ind: int) -> str:
    axes = sg_periodic_centroid_axes(sg_ind)
    auv = RAW_ASYM_UNITS.get(str(int(sg_ind)))
    return (f"SG{int(sg_ind)} aunit={auv}: wrapping centroid axes "
            f"{''.join('xyz'[a] for a in axes) or '(none)'}")
