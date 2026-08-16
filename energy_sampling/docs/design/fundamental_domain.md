# The fundamental domain, and flat latent directions

Argument doc. Why we would reduce the crystal latent space to a fundamental
domain of the Euclidean normalizer, why exactly-flat dimensions should be deleted
rather than pinned, and what is missing before either ships. Evidence lives in
[`findings.md`](../findings.md) F-007 / F-008 / F-009.

Two problems get conflated because both shrink the latent space. They are
independent and neither substitutes for the other:

| | **Discrete redundancy** | **Continuous redundancy** |
|---|---|---|
| Source | finite-index cosets of `N_E(G)/G` | `∩_k ker(R_k − I)`, the shared +1 eigenspace |
| Effect | *k* distinct parameter points describe one crystal | a continuum of points describes one crystal |
| Fix | fold to a canonical representative | delete the axis |
| Status | machinery works, not deployable (§2) | criterion derived, nothing implemented (§3) |
| Affects | 228 of 230 SGs | 68 of 230 SGs |

---

## 1. What exists

Built 7/2–7/4, all in `mxtaltools`:

- **The transform primitive.** `_transform_aunit_params`
  ([crystal_ops.py:798](../../../../mxtaltools/mxtaltools/dataset_utils/data_class_methods/crystal_ops.py))
  applies one fractional affine op to `(centroid, orientation, handedness)`.
  Validated 800/800 against literal atom-by-atom transformation to <1.2e-12.
- **`N_E(G)` coset representatives for all 230 SGs.** `NORMALIZER_OPS` plus
  `CONTINUOUS_DIMS` in `constants/space_group_info.py`, generated from cctbx by
  [`generate_normalizers.py`](../../../../mxtaltools/mxtaltools/constants/generate_normalizers.py).
- **The fold.** `get_fd_params` / `fold_to_fd` (`crystal_ops.py:843`). Works;
  reproduces the P21/c domain (F-007).
- **A G-box sanity check.**
  [`validate_asym_units.py`](../../../../mxtaltools/mxtaltools/dataset_utils/normalizer_reduction/validate_asym_units.py).

Nothing consumes any of it. `latent_transform` still scales by `ASYM_UNITS`,
`bounding_energy` still bounds to the aunit box, and `CONTINUOUS_DIMS` has zero
importers. The `normalizer_reduction/` directory is a superseded scratch copy —
its "STILL MISSING: nothing in mxtaltools has `N_E(G)`" docstring was resolved one
day later and is stale.

---

## 2. Why the fold is not deployable yet

Six gaps, in dependency order.

**No FD box table, and the obvious construction is wrong.** Using the FD as a
latent space needs explicit per-SG extents, so `latent_transform` can scale by
them and `bounding_energy` has something to bound to. Per-axis division of the
seminvariant moduli gives the right volume and the wrong shape (F-007). Only P21/c
has been measured against a prediction.

**The domain is per-molecule, not per-space-group.** 132 SGs have an improper
element in `N_E`; for those the chirality gate changes the achievable reduction
(P212121 z-extent 0.241 achiral vs 0.406 chiral). A conditional model over
molecules would need the latent scaling to vary with the conditioning input, which
the `asym_unit_lut` design has no room for.

**`is_chiral` is unplumbed**, and the `None` default is conservative only by
accident (F-007). `atom_chirality` is commented out at
`featurization_utils.py:209`.

**No bijectivity test.** The existing Monte Carlo checks G's box only. The
analogue — exactly one representative per `N_E` orbit inside the candidate FD — is
the gate for everything else and is the same loop with the ops replaced.

**Plumbing.** `get_fd_params` discards `h_folded` (fine at Z'=1 where handedness is
the molecule's own, wrong at Z'>1 where it is sampled); `fold_to_fd` leaves `.pos`
stale. Cost is `index ×` (`pose_aunit` + `build_unit_cell` + `reparameterize`) —
8× for P21/c, up to 40× — and the tie-break is a non-differentiable `argmax`. This
is an intake/canonicalisation operator, not something that can live inside the SDE.

**12 SGs have broken aunit boxes** (F-007), the classic ITA origin-choice-1-vs-2
groups. Folding on top of a broken box is meaningless.

### The easy space groups

Requiring a real aunit box, passing the coverage check, `N_E` pure translation (so
the fold never has to reduce in orientation space), and no free dimensions leaves
**39**:

```
2, 10-15, 47, 49, 51-58, 60-67, 69, 71-74, 91, 92, 95, 96, 124, 128, 131, 135, 142
```

P-1, P21/c, C2/c, Pbca, Pnma, Pccn, Pbcn and the centrosymmetric
orthorhombic/tetragonal set — roughly 70%+ of the CSD by frequency, and it
contains both SGs currently trained. All centrosymmetric except
P4122/P41212/P4322/P43212. Outside this set the fold acts on orientation too and
the FD stops being a product box, which is a different and harder problem, not a
bigger table.

---

## 3. Delete flat directions; do not pin them

Two kinds of exactly-flat latent dimension exist. They are **not** equally severe,
and the existing design already handles one of them.

**Constrained cell angles, monoclinic and above — orphaned by the rewrite.** The
intended mechanism is a quadratic pin, `(α − π/2)²` in `mono_reduction_penalty`
([sym_utils.py:110](../../../../mxtaltools/mxtaltools/common/sym_utils.py)), reaching the
energy through `reduction_energy`. In the GFN path it cannot fire:
`enforce_crystal_system` sets the angle to exactly π/2 before `reduction_en` is
evaluated, so the pin's argument is already at its target. It is live only in the
crystal-search optimiser, which skips the projection.

`8dea6b56` is the origin — "Niggli sampling ripped out in favor of general latent
space and an energy based penalty". Before it, the latent space *was* the reduced
domain by construction. After it, reduction is a penalty; that works for triclinic
but is preempted for monoclinic+ by a clobber predating the rewrite.

The damage is not just capacity. `latent_params()` collapses those latents to 0.0,
so **every prior and buffer row carries exactly 0.0** while the energy is flat across
the whole box. `bwd` trains P_F toward a delta at 0; `fwd` gets no gradient signal.
Two of twelve dims are trained against inconsistent targets (F-009).

**Triclinic is unaffected** — `enforce_crystal_system`'s "anything goes" branch means
all three angle latents reach the penalty, spanning `reduction_en` 0.46 → 16931. sg 1
and sg 2 are clean, which covers `mk_dev` and every current battery.

**Free aunit centroid axes — handled by nothing, in any crystal system.** No
projection, no pin. The coordinate round-trips faithfully, so identical crystals enter
the replay buffer at different coordinates: inflated apparent diversity, corrupted
`EffDim` and KDE-width, broken parameter-space de-duplication. RDF comparison would
catch it; coordinate metrics will not.

### Why deletion

| | log Z | physical marginal | cost |
|---|---|---|---|
| **Project, penalty preempted** (angles, GFN today) | physical | correct | dead dim **and** prior/energy disagreement |
| **Pin without projection** (angles, crystal-search) | shifts by a separable constant | unchanged | live dim, stiff timescale, off-SG cells reachable |
| **Nothing** (free axes, today) | physical | correct | flat dim **and** identical crystals scattered in coordinates |
| **Delete** | physical | correct | none |

A pin is *not wrong* — on an exactly-flat direction the energy factorises,
`Z = Z_phys · ∫exp(−pin)`, so the physical marginal is untouched and log Z shifts by a
known constant. But for the GFN it buys nothing the projection already gives, while
making off-space-group cells reachable during training.

Deletion is right because it matches the physics: a monoclinic Z'=1 crystal has 10
degrees of freedom, not 12. Restoring the penalty instead (skipping the projection)
would model a fiction and then penalise it back — self-consistent, and closer to the
rewrite's intent, but strictly more machinery for the same distribution.

Projection, not an energy term, is the established pattern for a discrete gauge
choice here — `canonicalize_zp_aunits` does the same thing for Z'-unit relabelling
([molecular_crystal.py:238](../../energies/molecular_crystal.py:238)), and that comment
makes the reasoning explicit: pin the gauge so every crystal that reaches a buffer is
gauge-independent by construction rather than only to within numerics. A free aunit
axis is the continuous member of that same family, and it is the one member with no
canonicaliser.

### Implementation shape

`data_ndim = 6 + 6·Z' − n_dead`, where `n_dead` = constrained angles + free axes.
Making `data_ndim` space-group-dependent is not a new coupling: `periodic_centroids`
already makes the model SG-specific and checkpoint-incompatible
([train.py:1027](../../train.py:1027)).

Three traps:

1. **At Z'>1 the free translation is one global shift** shared by all Z'
   molecules. Delete it from a single centroid block. Per-molecule zeroing
   destroys physical relative offsets (F-008).
2. **Angle deletion is axis-aligned; length averaging is not.** `a=b` in
   tetragonal/hexagonal and the rhombohedral means are diagonal degenerate
   directions — reparameterise (emit the mean directly), do not delete an axis.
3. **`sg_periodic_centroid_axes`**
   ([aunit_periodicity.py](../../models/aunit_periodicity.py)) **under-reports** for 16 of the 42 free-dim SGs
   with a real box (F-008). Deleting free axes makes this moot; leaving them in
   makes it a second bug.

---

## 4. Order of work

1. **Delete the orphaned monoclinic+ angle latents.** A real defect, not an
   efficiency question: the prior says delta-at-zero and the energy says flat, so
   `fwd` and `bwd` are trained against inconsistent targets in those dims (F-009).
   Only bites sg ≥ 3, so no current battery is affected and it can be fixed without
   invalidating anything — but it must be fixed before the next monoclinic or
   orthorhombic run. Reading the fwd/bwd gap across the change also settles F-009's
   conjecture.
2. **Gauge-fix the free aunit axes.** Same class of fix, and the only flat dimensions
   with no canonicaliser at all. Unlocks the polar/Sohncke groups needed for chiral
   molecules (P21, C2, Cc, Pca21, Pna21). Independent of all FD work. P212121, the
   most common Sohncke group, has no free dims, so this is not on the critical path
   for the commonest chiral case.
3. **Write the `N_E`-orbit coverage test.** Gates everything below it.
4. **Measure FD extents across the 39 clean SGs** on real CSD data, then decide
   whether an `FD_ASYM_UNITS` table is derivable or must be tabulated empirically.
5. **Plumb molecular chirality.** Gates everything outside the centrosymmetric set.

Steps 1–2 are worth doing regardless of whether the FD reduction ever ships. When
it does, this doc's §1–2 should be split out into a `module_*.md` State doc; until
then there is no shipped behaviour to describe.
