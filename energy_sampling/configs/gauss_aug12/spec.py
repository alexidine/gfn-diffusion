"""
gauss_aug12 -- the ONE source of truth for the latent_gaussian dead-row battery.

prep_prior.py, make.py and verify.py all import from here, so the prior file, the
YAML the run actually loads, and the number we compare the result against cannot
drift apart. That failure mode is the whole reason this module exists: a prior
drawn at width 0.1 scored by a config that says 0.15 trains perfectly well and
reports a wrong log Z, with nothing to see in either file on its own.

=============================================================================
THE TARGET
=============================================================================
`latent_gaussian` is a real crystal parameterization (is_crystal True: dead rows,
periodic angle dims, the box) scored by an analytic gaussian on the LATENT
(latent_energy True: no packing, no pressure, no mol_energy -- and, structurally,
no reduction and no jacobian). So

    E(x) = 0.5 * sum_d ((x_d - c_d) / w)^2  +  k * sum_d relu(|x_d| - 1)^2
    R(x) = exp(-E(x) / T)

with c_d = MODE on live rows and 0.0 on dead rows. Dead rows contribute exactly
((0 - 0)/w)^2 = 0, which is what makes the energy live-dims-only with no masking
code anywhere -- the crystal build already reads those rows back as 0.

MODE is 0.5, not 0. At 0 the target sits on the SDE's own origin and a
MIS-INDEXED dead row is invisible: correct and swapped both look "0-centred".
Off-centre, a swap is loud -- a live dim pinned at 0 instead of MODE, or a dead
dim chasing MODE while the crystal build clobbers it back, and the arm cannot
reach its analytic log Z.

=============================================================================
THE CLOSED FORM  (measured, not assumed -- see findings.md F-011)
=============================================================================
Rows HELD (hold_dead_latent_rows: true) -- dead rows are not coordinates:

    log Z = (n_live / 2) * log(2 * pi * T)  +  n_live * log w

Rows LIVE (false) -- each dead row is an ordinary SDE dim. The gaussian still
cannot see it (the crystal build clobbers it), but bounding_energy reads
raw_latents, so its marginal is exp(-k * relu(|x|-1)^2) and it contributes its
own normaliser:

    log Z = <above>  +  n_dead * log(2 + sqrt(pi / k))

That second term is the FICTITIOUS VOLUME D33 removes. Note it is NOT n_dead*log 2:
the wall is soft (quadratic, zero-slope onset), so the reachable volume exceeds the
[-1,1] box by sqrt(pi/k). At k=1 the leak is nearly as large as the box itself
(3.77 vs 2). The log-2 form was the first prediction and it is wrong by +0.63/dim
at k=1; the sweep in verify_latent_gaussian.py refutes it across a 20x range of k.
"""

# ------------------------------------------------------------------ constants
TAG = 'gauss0812'
T = 1.0             # sampling temperature; enters log Z as (n_live/2)*log(2*pi*T)
WIDTH = 0.1         # gaussian sigma at T=1. 5 sigma from the box wall at MODE 0.5,
                    # so live-dim leakage past the wall is ~1e-7 per dim: negligible
MODE = 0.5          # on LIVE rows only. See the note above -- this is load-bearing
BOUNDING_COEFF = 1.0    # k. Deliberately soft: it MAXIMISES the rows-live penalty,
                        # so the A/B signal is as large as this energy can make it
REDUCTION_COEFF = 1.0   # inert -- reduction_energy is structurally zero for a
                        # latent-scored problem (molecular_crystal.py). Stated so a
                        # reader does not have to guess whether it was forgotten.
DIM = 12                # 6 box + 6 aunit, at Z' = 1

PRIOR_DIR = r'D:\crystal_datasets\conditional\priors'
PRIOR_STEM = 'gauss_latent_sg{sg}_zp1_prior'
N_PRIOR = 20000     # prior rows. prior_buffer.min_size is 10000, so this gives the
                    # buffer room to churn without re-drawing from a thin dataset


def prior_path(sg):
    import os
    return os.path.join(PRIOR_DIR, PRIOR_STEM.format(sg=int(sg)) + '.pt')


# ----------------------------------------------------------------- the arms
# sg 2  triclinic          dead ()        -> THE CONTROL. The knob must be inert.
# sg 14 monoclinic         dead (3,5)     -> 2 clobbered angles. The primary pair.
# sg 19 orthorhombic       dead (3,4,5)   -> 3 clobbered angles. Largest angle signal,
#                                            and the most common Sohncke group.
# sg 4  monoclinic polar   dead (3,5,7)   -> 2 angles + 1 FREE AXIS. Must land on the
#                                            same log Z as sg 19 by a DIFFERENT path
#                                            (canonicalize_free_axes, not
#                                            enforce_crystal_system).
# sg 1  P1 triclinic       dead (6,7,8)   -> 3 free axes, no angle rows. Pure free-axis
#                                            arm; also the group whose free axes are why
#                                            the is_crystal gate exists.
SPACE_GROUPS = (2, 14, 19, 4, 1)

# sg 4 / sg 1 / sg 19 close gaps deadrow_aug12 states it CANNOT close: it has no
# orthorhombic prior on disk and no free-axis arm at all, because a physical prior
# must be a real crystal. A latent-scored toy has no such constraint -- nothing
# builds a cell -- so any space group can be synthesised here. That is the main
# reason this battery exists alongside that one.


def dead_rows(sg):
    """Delegates to the SHIPPING resolver -- never a second table."""
    from energy_sampling.models.dead_latent_rows import resolve_dead_rows
    return tuple(resolve_dead_rows(int(sg), is_crystal=True, max_z_prime=1))


def target_c(sg, dim=DIM, mode=MODE):
    """MODE on live rows, 0.0 (the canonical read-back value) on dead rows."""
    c = [float(mode)] * int(dim)
    for r in dead_rows(sg):
        c[r] = 0.0
    return c


def analytic_log_z(sg, hold, temperature=T, width=WIDTH, k=BOUNDING_COEFF, dim=DIM):
    """The number the run must reproduce. See the module docstring for the derivation."""
    import math
    n_dead = len(dead_rows(sg))
    n_live = int(dim) - n_dead
    z = (n_live / 2) * math.log(2 * math.pi * temperature) + n_live * math.log(width)
    if not hold:
        z += n_dead * math.log(2.0 + math.sqrt(math.pi / float(k)))
    return z


def predictions():
    """[(sg, dead, n_live, logZ_hold, logZ_live, delta), ...] for every arm."""
    out = []
    for sg in SPACE_GROUPS:
        d = dead_rows(sg)
        on = analytic_log_z(sg, hold=True)
        off = analytic_log_z(sg, hold=False)
        out.append((sg, d, DIM - len(d), on, off, off - on))
    return out


if __name__ == '__main__':
    import math
    print(f"T={T}  width={WIDTH}  mode={MODE}  bounding_coeff={BOUNDING_COEFF}")
    print(f"fictitious volume per live-but-dead row = log(2+sqrt(pi/k)) = "
          f"{math.log(2 + math.sqrt(math.pi / BOUNDING_COEFF)):+.4f}"
          f"   (the refuted log-2 model says {math.log(2):+.4f})\n")
    print(f"{'sg':>4} {'dead':>12} {'n_live':>7} {'logZ hold':>11} {'logZ live':>11} {'delta':>8}")
    for sg, d, n_live, on, off, delta in predictions():
        print(f"{sg:>4} {str(d):>12} {n_live:>7} {on:>11.4f} {off:>11.4f} {delta:>+8.4f}")
