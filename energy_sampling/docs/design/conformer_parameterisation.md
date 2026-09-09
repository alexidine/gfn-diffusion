# The conformer chart: specification, instantiation, and build state

**Status: SPECIFICATION + BUILD STATE.** Written 2026-09-08 by reading the code rather than
the docs, because both surrounding design files are stale in known directions.

**What this file owns.** One statement of the conformer parameterisation and its
instantiation pipeline; which parts of it are implemented; the evidence for each; and the
objections the claim has to survive before it goes into print.

**What it points to rather than repeats.**
[`conformer_conditional_stack.md`](conformer_conditional_stack.md) §5 for the policy
architecture and the encoder's place in it;
[`internal_dof_ladder.md`](internal_dof_ladder.md) §2 for the long argument that the levels
are not approximations and §5 for the full chirality derivation;
[`conformer_encoder_architecture.md`](conformer_encoder_architecture.md) for the encoder
itself. Where this file and those disagree about build state, this one is newer.

Evidence grades per [`../EPISTEMIC_PROTOCOL.md`](../EPISTEMIC_PROTOCOL.md) §4.

---

## 1. The claim, and whether we can defend it today

| | Claim | Verdict |
|---|---|---|
| **C1** | The coordinate chart is a deterministic function of the labelled 2D molecular graph, so the condition and `log Z(c)` are well defined | **PARTIAL** — the chart's *structure* is, and is gated; its *origin* is not (§8 T1) |
| **C2** | Minimal and complete: exactly 3N−6 internal DoF, no redundancy, no external DoF | **HOLDS** (§2) |
| **C3** | Invariant to input atom numbering | **HOLDS**, gated (§2, §7) |
| **C4** | The reward is the Cartesian Boltzmann density read through the chart the sampler proposes in — both changes of measure carried, no missing constraint term | **HOLDS at `full`; a conditional slice at the frozen levels, and must be described as one** (§4) |
| **C5** | Every latent in R^d maps to a finite reward; the target is normalisable | **HOLDS** (§3.3) |
| **C6** | In a molecule set, each row is scored through its own molecule's chart | **HOLDS for the reward; FAILS for the prior draw and the forward conditions** (§6) |
| **C7** | The target is the Boltzmann distribution of *one stereoisomer* | **FAILS above `torsion`** (§3.5) |

C1, C6 and C7 are the three a referee will press. None of them invalidates a
single-molecule, `torsion`-level result; C7 invalidates any chiral molecule above it.

---

## 2. The tree and the reference atoms

`TreeSpec` (`mxtaltools/conformers/topology.py`) is one spanning tree per molecule, built
once on CPU and cached. Atoms are relabelled into **placement order**; the atom in slot `k`
carries `min(k, 3)` degrees of freedom:

```
k = 0   root, at the origin              0 DoF
k = 1   bond only                        1 DoF   (r)
k = 2   bond + angle                      2 DoF   (r, theta)
k >= 3  bond + angle + dihedral           3 DoF   (r, theta, phi)
```

`0 + 1 + 2 + 3(N−3) = 3N − 6` exactly. `TreeSpec.n_dof` asserts it and
`ConformerTorsions.__init__` re-asserts it against its own block counts. **C2 is structural,
not checked:** the count is a property of the construction, not of a filter applied
afterwards. The six external DoF are consumed by the seed convention — root at the origin,
slot 1 on +x (`place_seed_second`, `r` only), slot 2 in a canonical half-plane
(`place_seed_third`, `r` and `theta`) — so `build` emits positions that are *already*
SE(3)-reduced and nothing downstream aligns anything. Three of the six are spent at slot 2,
where the dihedral does not yet exist.

**Ordering: WL first, everywhere.** `canonical_rank` is Weisfeiler–Lehman colour
refinement — a pure graph function, no RDKit, no coordinates. Root choice
(`choose_root`: graph centre, tie-broken toward heavy high-degree atoms), BFS order
(`_canonical_bfs`) and every reference-atom choice break ties on that rank rather than on
atom index. Atoms still tied after refinement are genuinely automorphic (a methyl's three
hydrogens, the ortho carbons of a monosubstituted ring), so any choice among them yields an
isometric tree. This is what buys **C3**, and it is what makes the parameterisation a
property of the *molecule* rather than of how its SMILES happened to parse.

**Reference-atom selection, stated as the rule it is.** For the atom in slot `k` with
parent `c = parent(k)`:

| | choice | cascade |
|---|---|---|
| `ref_c` | `c` | the parent, unconditionally |
| `ref_b` (`k ≥ 2`) | apex of `theta = angle(b, c, k)` | `parent(c)` if placed and well-conditioned → else `_pick` over `c`'s other neighbours → else `_pick` again with the geometric guard dropped (a terminal linear centre; `theta ≈ pi`, flagged) |
| `ref_a` (`k ≥ 3`) | far atom of `phi = dihedral(a, b, c, k)` | `parent(b)` (the grandparent, so an ancestor chain) → else `_pick` over `b`'s other neighbours (a **proper** torsion) → else `_pick` over `c`'s other neighbours (an **improper** about the parent) → else the same two without the geometric guard → else any placed atom |

`_pick` returns the best *placed* candidate under `(degree, −WL rank, −index)`, so it
prefers branching atoms, and it is **strict** when a geometric predicate is supplied: it
returns −1 rather than an ill-conditioned reference, so the caller's cascade can fall
through to a different frame. `torsion_is_proper` records whether `a–b` is an actual bond,
which matters because impropers are the only rows that share a parent with a different
reference — see §5. `round_id[k] = 1 + max(round_id of its references)`, which is what lets
`build` place a whole round in one vectorised call.

**The design intent behind the `ref_a` cascade is that an improper about `c` beats a
collinear proper torsion.** That intent is currently not realised: the guard is
geometry-bound and the conformer route calls the tree builder with no geometry. See §8 T2 —
it is the single largest defect in the chart.

---

## 3. Which degrees of freedom stay free

### 3.1 The levels

`ConformerTorsions.LEVELS = ("torsion", "dihedral", "flex", "full")`, as freeze sets over
`(r, theta, phi)`:

| level | free | state dim `d` | invertible? |
|---|---|---|---|
| `torsion` | one **collective** angle per rotatable bridge bond | #rotatable axes | no — `state_from_dof` refuses |
| `dihedral` | all phi | N−3 | yes |
| `flex` | theta and phi | 2N−5 | yes |
| `full` | r, theta, phi | 3N−6 | yes |

`torsion` is the odd one out: a `torsion` column rotates *every* dihedral about its bond,
generally several. The state→DoF map is therefore a **linear map, not an index subset**, and
`ConformerTorsions.collective` records it; `state_from_dof` raises rather than returning a
plausible row-wise pseudo-inverse. This was a live bug — treating the mask column as an index
drove only the first dihedral of each bond and left the rest at their reference values.

**What counts as a rotatable axis** (`_find_rotatable`): a tree bond `(u, v)` qualifies when
it is a **bridge** of the full graph — a ring bond cannot be rotated independently, since the
ring constrains it — *and* the moving side (descendants of `v`) contains a heavy atom, or the
"rotation" is a terminal hydrogen wobble. Unless `include_trivial_rotations`, a moving side
of one heavy atom and ≤ 3 atoms total is also dropped: methyl, amine and hydroxyl spins.

### 3.2 Linear centres

A linear centre — alkyne, nitrile, azide, allene — is a **genuine singularity of any
atom-tree parameterisation**, not a bad reference choice: `log sin theta` diverges as
`theta → pi`, and the dependent dihedral frame is undefined there. Reference choice cannot
avoid it, because the angle is linear because the molecule is.

Two predicates exist. `angle_is_linear` marks `theta ≈ pi` rows; `torsion_frame_is_linear`
marks phi rows whose `angle(a, b, c)` is linear. Either may be computed two ways
(`ConformerTorsions.linearity_source`):

- `'mmff_typed'` — MMFF's **typed** equilibrium angle, `theta0 ≥ 179.99°`. A function of the
  graph alone, indexed by atom identity (never by row: `ff.angle_index` is the *graph* angle
  list and is longer than the tree's, so a positional lookup reads the wrong constant).
- `'measured'` — the angle in the reference conformer against `LINEAR_TOL_DEG = 175°`. Used
  when `force_field: 'reference'`, where there is no graph-determined constant to appeal to.

**The handling rule is now: change the CHART, not the dimension count.** A linear
equilibrium is not a rigid constraint, and the divergence is a property of the polar pair
`(theta, phi)` rather than of the molecule. With `rho = pi - theta` the bend magnitude and
`phi` the bending plane,

```
u = rho cos(phi)      v = rho sin(phi)
d = r cos(rho) bc  +  r u sinc(rho) m2  +  r v sinc(rho) n
sin(theta) dtheta dphi = sinc(rho) du dv
```

so the divergent `log sin(theta)` becomes `log sinc(rho)`, which is smooth and **zero** at
the linear geometry, and the coordinate count is unchanged: two in, two out. The singularity
is not removed but MOVED — `sinc` vanishes at `rho = pi`, i.e. `theta = 0`, the placed atom
folded onto the b–c axis on top of its own grandparent. That geometry is sterically
forbidden, so a vanishing measure there is correct; the linear one is a real equilibrium, so
a vanishing measure there was not.

`place_nerf_transverse` evaluates this **directly in `(u, v)`**. Recovering `(rho, phi)` by
`atan2(v, u)` and calling the ordinary placement would agree on values and reintroduce the
pole into the gradient — the failure mode that looks like it works. The same applies to
measurement: `measure_transverse` projects onto the placement frame and reads
`rho = atan2(hypot(p, q), w·bc)`, which is the *polar* `atan2` and regular at zero, not the
azimuthal one.

**What a transverse row is, mechanically.** The `theta` and `phi` slots belonging to the
same atom stop carrying `(theta, phi)` and carry `(u, v)`. Reusing the two existing slots
keeps every downstream shape untouched — the selection matrix, the per-atom chart blocks on
the graph, the policy's state width. The cost is that a slot's meaning is mask-dependent, so
the mask travels with the tensors through `build`, `measure` and `log_jacobian`; passing it
to one and not the others is a *silent geometry or density error*, not a shape error, which
is why `log_jacobian` refuses a transverse mask without `phi` and why `ConformerTorsions`
routes every Jacobian call through one `_log_jac` helper. Transverse columns get block code
3: not periodic (unlike phi) and **not clamped to `(0, pi)`** (unlike theta), because their
domain is a disc whose interior point is the linear geometry. The disc is enforced instead by
a construction-time check that `delta_theta_max` cannot reach `rho = pi`.

**Three conditions, and the last two are the limits.** A linear angle row is carried
transversely only if it (1) is linear, (2) has a torsion row to hold `v`, and (3) has a
non-collinear placement frame `a-b-c`. Failing (2) means the atom is a **frame seed**, whose
missing component is the sixth *external* DoF — and when atoms 0–1–2 are collinear the
"third atom in the xy half-plane" convention does not fix a frame at all. Failing (3) is
`torsion_frame_is_linear`: the normal defining `phi` is arbitrary, so there is no frame to
bend in. **Neither is fixed by this change**, and both still hold their rows. A smooth,
consistent frame construction is what (3) needs; it is not a convention evaluated at exactly
zero bend. Molecules in that state keep the CONSTRAINED report and are still refused entry to
a file labelled `full` (`build_conformer_conditions.py`).

**The acetonitrile case is resolved.** `CH3CN` at `full` now reports `d = 12 = 3N-6` with
zero held rows, against `d = 11` before: the `theta(C0–C4–N5)` bend that was held is driven,
and the `phi(H1–C0–C4–N5)` that was dead at `sin(theta) = 0` is now the second component of
the same bend rather than a coordinate that moved nothing. Propionitrile likewise reaches
`21 = 3N-6`. **Propyne does not** — its linear angle is at the frame seed and it carries four
collinear torsion frames — and it correctly still reports 9 of 15.

**Rank and measure were verified separately** (`mxtaltools/conformers/tests/
test_transverse_bending.py`), at the linear reference and at perturbations, because a chart
can have the right rank and the wrong density and the column count establishes neither:

- RANK — SVD of `d(free cartesian)/d(internal)` is `3N-6` for every molecule tested, in the
  polar chart, in the transverse chart, and with *every* eligible row made transverse.
- MEASURE — `log_jacobian` minus the Gram volume `log sqrt(det(J^T J))` of the same map
  equals `2 log r1 + log r2 + log sin(theta2)` to `< 1e-9`, with no free parameters. That
  term is the **gauge factor of the SE(3)-reduced seed convention**, not a discrepancy:
  `log_jacobian` is the BAT element with the rigid motions integrated out, while
  differentiating `build` gives the volume of one *section* of that quotient. The map is
  rectangular once a ring closure removes a tree DoF, so the Gram determinant is the check
  and a raw determinant is not.

**The domain, and how it is respected without clamping.** The chart is injective exactly on
the **open disc `rho < pi`** — verified numerically, not argued: `(rho, phi)` and
`(2pi - rho, phi + pi)` are the same point to 1e-16, the boundary circle `rho = pi` collapses
to a single point 0.30 Å from the grandparent atom, and past it the area element changes sign,
so the map is an orientation-reversing double cover. `log_sinc` returns `-inf` there, never
NaN, so an out-of-domain row is a visible zero rather than a poisoned batch.

Removing the singularity at `rho = 0` does **not** by itself bound anything, and the measure
does not do it either: `log sinc` diverges only **logarithmically**, costing 8.1 nats one
milliradian from the boundary against a box wall worth 279 kcal/mol at the same point. The
measure marks the boundary; it does not hold it. Three facts decide the treatment:

- **Measured reach.** Over 947,022 transverse rows of a 400-epoch run the largest `rho` was
  **0.682** — identical to the seed prior file's own maximum, so the policy never proposed a
  bend larger than what it was handed. Physical 99.99th percentile: 0.53. Extrapolated
  per-row crossing probability 1e-15 (pessimistic exponential fit) to 1e-127 (Gaussian).
- **The old wall had the wrong shape.** It is a square in `(x_u, x_v)` and knew nothing of the
  disc: at `|x_u| = |x_v| = 1`, where it is exactly zero, `rho` is already 0.71, and the
  diagonal route out is 42 kcal/mol *cheaper* than the axial one — the cheapest escape was the
  direction nothing checked.
- **A crossing is unattributable.** Nothing between `get_loss_reward` and `get_tb_loss` checks
  finiteness, so one crossed row takes the batch mean and every gradient to infinity. Worse,
  `potential_energy` stays **finite** (535 kcal/mol) where `energy()` is `+inf`, and
  `potential_energy` is the buffer currency — so such a row is admissible to a persisted
  sidecar and returns `-inf` on every replay thereafter.

So the domain is held by a **radial preference plus a counter**, not by a clamp:
`bounding_coeff * relu(rho - rho_wall)^2 * T` with `rho_wall = 1.0`, which is *identically
zero* on every state either run has ever produced, and `ConformerTorsions.transverse_crossings`,
which counts rows with `rho >= pi` so a crossing is a named event rather than a NaN.

A **radial clamp was rejected**, and not on the usual grounds: the region it would pin has
Boltzmann probability 1e-121 and is a 0.30 Å grandparent contact worth ~600 kcal/mol of
unclipped MMFF, so it is a genuinely different object from the `theta` clamp the docstrings
warn about (which would have fenced off `rho = 0`, the density's *maximum*). It is rejected
because it is **invisible**: `measure_transverse` reads a folded row back as an ordinary
in-domain geometry (built 4.0 measures back as 2.283), so a clamp trades a loud `inf` for a
silent non-injective plateau. The principled fix, held in reserve, is to make the domain a
property of the map — a radial squash `rho = R tanh(d|s| / R)` with `R < pi`, a smooth
bijection onto the open disc — which makes the `-inf` branch unreachable rather than
improbable. It waits because `log_chart_jacobian` stops being a constant under it. Revisit if
`delta_theta_max` rises above ~1.0, `bounding_coeff` falls, `T` rises, or a molecule appears
whose `u0` is not ~0.

**The refusal is structural.** `ConformerTorsions` itself raises when `level == 'full'` and
any row is held, with an explicit `allow_constrained=True` opt-out. Previously the only guard
was in `build_conformer_conditions.py` — one of eight paths that construct a chart — so a
config naming a constrained molecule at `full` trained silently while its run summary, its
checkpoint and its conditions file all said `full`. The run summary now also carries
`constrained_rows`, `uncovered_linear_angles`, `n_free_transverse` and `n_internal_dof`, the
last because `n_free_r + n_free_theta + n_free_phi` does **not** sum to `data_ndim` once a
transverse pair is free: those columns are block 3 and were counted nowhere.

**The policy features were an interface error, now fixed.** Geometry and density always read
`energy._ref_dof`; the policy read `th0`, so a transverse row was labelled `kind_theta` and
handed `+3.14` for a coordinate whose reference is `-0.002` — off by the whole dynamic range
of the coordinate, silently, because nothing downstream depends on that number being right.
Three further defects rode along and are fixed with it: the `v` row took the *torsion* branch's
jitter (`log 0.1`) instead of the bend width, it was flagged `is_group_member` (bond-rotation
semantics for an out-of-plane bend), and the `u` row's frame was the 3-atom angle frame, which
omits the very atom `a` that fixes which way `u` points. The kind block is now
`[r, theta, phi, bend_u, bend_v]`, both rows carry the partner angle's stiffness and the
four-atom placement frame, and a dedicated `ref_bend` column carries `u0` / `v0`. The earlier
note here claimed this "invalidates stored conditions files and the policy input width" — that
was wrong: `condition_from_energy` stores no feature matrix, and a set-policy resume is refused
outright, so no artefact carried the width. What *does* change values (not shape) is
`dof_atoms`, via the frame fix.

**Still open.** `prior_log_prob` refuses a molecule with transverse rows: it is a density over
the polar dof while the state is `(u, v)`. Verified: the missing factor is `1/rho` and spans
9.59 nats over a real draw, so it is not a constant that cancels in a self-normalised ESS —
and the correction is **not** just that Jacobian, because the polar fit folds 2:1 onto the disc
(49.8% of drawn `theta` exceed `pi`), so the pushforward carries two terms. Fitting the bend as
a 2-D distribution on the disc is the fix. Nothing depends on it: the density has three call
sites, all in `prior_diagnostics`, none reachable from the fwd/bwd/replay TB steps or from
`evaluation()` — confirmed by a traced 21/21-step run of the real config with a call count of
**0**. What is lost is the prior-proposal ESS family and the model-free IS log-Z estimate,
neither of which gates a controller. That the *draws* are correct (verified to 0.00e+00 Å
through the chart) says nothing about **coverage**: `coverage_report` enumerates rotamer basins
by dihedral and is blind to the bend block, so the prior's coverage of the disc is currently
unmeasured. A histogram of `(u, v)` against a long MMFF chain would close it without needing
the density at all.

### 3.3 State → DoF: one affine map, one clamp

The sampler proposes `x` on `[-1, 1]^d`. `ConformerTorsions.dof_from_state` writes

```
q = q_ref ;   q[driven] += (x * scale) @ M^T
scale_j = delta_r_max (r) | delta_theta_max (theta) | pi (phi)
```

one scatter, whatever the level.

**C5 rests on the clamp, not on the box wall.** `r` is clamped at `r_floor` and `theta` into
`(theta_floor, pi − theta_floor)` before the map. Without it: `2 log r` is NaN at `r ≤ 0`;
`log sin theta` is NaN or −inf outside `(0, pi)`; and worse, `build` is **non-injective
off-domain** — `d(r, 2pi−theta, phi+pi) == d(r, theta, phi)` — so an excursion *double-covers*
rather than being merely re-weighted, and `measure` cannot detect it because `bond_angle`
returns `[0, pi]` always. `phi` is not clamped: it wraps, which is a total map.
`bounding_energy` adds a quadratic wall on the non-periodic blocks, pre-multiplied by `T` so
it stays equally stiff at every sampling temperature — but the wall is a *preference* and the
clamp is the guarantee.

**Periodicity is declared, not inferred.** `periodic_dims` marks the phi block and only it.
`GFN`'s own inference keys on `is_crystal` and hands a non-crystal state zero wrapped dims —
for a torsion state that is not a degraded layout but an **unnormalisable target**, since the
reward is exactly 2-periodic in every phi dim. `ConformerModeller._build_gfn_config` passes
`angular_mask = energy.periodic_dims`; this is not optional, and it is what retired the old
`TorsionGFN` subclass.

### 3.4 Rings: closure, fused and bridged systems

A spanning tree cannot carry a ring-closure bond. Those edges go into
`broken_bond_index`, and their lengths become **determined functions** of the tree DoF;
`builder.closure_length` realises them.

**Closure is a penalty, never a constraint.** The force field runs its bond and angle terms
over the **whole graph**, not the tree, so a non-tree ring bond is simply another harmonic
bond. Two consequences, and the second is the one that matters for §4: ring closure comes out
for free rather than needing machinery, and an H–C–H angle at a methyl is not left
unconstrained (only the X–C–H angles are in the tree, and 1–3 pairs are excluded from LJ, so
the three hydrogens could otherwise collapse at zero cost).

**A ring *system* is a connected component of the non-bridge edges** (`ring_systems`), so
**fused and bridged rings correctly come out as one joint block** — naphthalene is one
system, not two, and a bicyclic bridgehead is not split. Each system carries a signature of
`(size, sorted multiset of (element, degree))`, which is substitution-invariant by
construction: benzene, toluene, phenol, styrene and ibuprofen's ring all give `((6,3),)*6`,
and cyclohexane, methylcyclohexane and cyclohexanol all give `((6,4),)*6`. Twelve test
molecules resolve to six buckets [OBSERVED]. Dropping `degree` from the signature is a
measured failure, not a hypothetical: benzene and cyclohexane then shared a bank, benzene
drew its ring from a bank half full of chairs and came out at a **median |ring torsion| of
47°, with 75% of draws past 20°** — and `ff_from_reference` does not object, because it
carries no torsion term and bond angles stay near 120° through a pucker [OBSERVED].

### 3.5 Stereochemistry

**The chart is SE(3)-reduced but not reflection-reduced, and that is the whole issue.** In
`place_nerf`, `phi` enters through both `cos(phi)·m2` and `sin(phi)·n̂`, and only the `n̂`
term is odd — so **negating every phi is exactly the global mirror**, verified numerically
orthogonal to 8.9e-16 with `det = −1.000000000` [MECHANISM]. The state space therefore
double-covers physical conformer space: every conformer appears as `(r, theta, phi)` and
`(r, theta, −phi)`.

For an achiral molecule that is a harmless factor of 2 in `Z`. For a chiral one, the two
halves are the two enantiomers — **and the force field cannot tell them apart**, because
MMFF94 and the reference field are both mirror-invariant. So nothing in the reward pins the
stereoisomer we asked for; the two enantiomers have identically equal energy at every point.

Where the configuration lives depends on the centre (derived in `internal_dof_ladder.md` §5):

- **Root centre, the common case.** `choose_root` takes the graph centre tie-broken toward
  heavy high-degree atoms, which *describes* an sp3 stereocentre. At the root two of four
  substituents carry no dihedral at all — their arrangement is fixed by the seed convention —
  so the first NeRF-placed child's `phi` decides which side of the plane it falls on.
  Measured: sweeping that `phi` through zero flips `chi` with an exactly odd signature, and
  sweeping any other `phi` leaves `chi` constant to all digits. **At the root, chirality is
  the sign of one state dimension** [OBSERVED].
- **Interior centre.** All of a centre's children share the frame `(V, W, C)`, so the
  configuration is the *cyclic ordering* of their three dihedrals — not any single state
  dimension.

**Current status by level.** At `torsion`, chirality is protected **for free**: `r` and
`theta` are frozen at a reference that already has a configuration, and rotating a rigid
fragment about a bridge cannot invert a centre. It becomes live at `dihedral` and is
unprotected at `flex` and `full`. Nor is it protected by a barrier: `soft_core_lj` caps at
`(24ε/k)(e^k − 1) = 10.735 kcal/mol` per pair at `ε = 0.1, k = 2.5` — bounded, not divergent
— and a GFN takes Gaussian jumps in latent space rather than following a continuous path,
while backward training seeds terminals straight from a buffer.

**The proposed mechanism, not built:** a wall on the **normalised** signed volume `chi`,
`k·relu(−s·chi)²` (normalised because raw volume scales as `r³`; pre-multiplied by `T` like
every other measure-preserving term). `relu` is exactly zero on the allowed side, so the wall
does no interior distortion and its only error is leakage. Two conditions the design has to
respect: the eval rejection predicate must be **the same set** as the wall's zero set (an
independent RDKit CIP re-perception is not), and recovering the partition function needs
`log Z_allowed = log Z_walled + log(acceptance rate)` — rejection alone corrects
expectations, not `log Z`. A half-period wrap is **not** a free exact constraint: `phi → phi+pi`
is a rotation, and folding by `phi → −phi` maps a *diastereomer* onto the target rather than
folding a symmetry.

---

## 4. The target measure and the Jacobian accounting

The sampler's target on the latent box is

```
p(x)  ∝  exp( −U(q(x)) / T )  ·  Π_i r_i(x)²  ·  Π_i sin theta_i(x)  ·  Π_j scale_j
             ^ potential          ^-------- BAT volume element -------^   ^ chart Jacobian
```

and `ConformerTorsions.energy` returns `E/T` such that `log_reward = −E`:

```
log_reward = −U/T + log J_BAT + log|dq/dx|
```

- **`log J_BAT`** = `Σ 2 log r_i + Σ log sin theta_i` (`builder.log_jacobian`) relates internal
  coordinates to Cartesian. Independent of `phi`, so it is a *constant* at `torsion` and
  `dihedral` and state-dependent at `flex`/`full`. `log_jacobian_const` records the constant,
  but the code deliberately does **not** short-circuit on it — one code path means the
  constant cannot silently drift from the computed value.
- **`log|dq/dx|`** = `Σ_j log scale_j` (`log_chart_jacobian`) relates the latent box to the
  internal coordinates. Constant in `x`, which is why it was invisible for so long: a
  constant cancels out of every TB residual, so no unconditional result depended on it and no
  gate fired.

Both terms are pre-multiplied by `T` before the division, so a change of measure contributes
the same amount to `log_reward` at every temperature — gated by
`test_chart_volume_element`.

**`log|dq/dx|` is what makes `log Z(c)` a physical quantity.** Its value depends on the
free-column count, so it differs per molecule: **9.0 nats over eight molecules at `full`**
(propanol −9.87 to ethylcyclohexane −18.90), 3.4 nats at `torsion` [OBSERVED]. Without it,
cross-condition `log Z` comparison is meaningless. Adding it **shifted `log Z`** relative to
the pre-2026-09 code; stored reference values from before the shift are not comparable.

**No constraint term is missing at `full`, and this is worth saying explicitly** because it
is the first thing a referee will look for. The chart is a 3N−6 parameterisation of the
*unconstrained* internal-coordinate space: nothing is held rigid, ring closure enters as a
harmonic penalty on a graph bond rather than as a holonomic constraint (§3.4), so no Fixman
factor and no constrained-metric determinant beyond the BAT element is required. `full` is
the level at which the measure is simply correct.

**At the frozen levels the accounting is different and must be described honestly.**
Freezing a DoF at a constant `c0` gives `p_full(free | frozen = c0)` — a **conditional
slice** — which differs from the *rigid-constraint* ensemble by a state-dependent Fixman
factor. So `torsion`, `dihedral` and `flex` are each *some* related distribution, useful for
staging and regression, and **must never be written up as approximations to `full`** or as
constrained ensembles. `internal_dof_ladder.md` §2 argues this at length.

---

## 5. How the prior represents coupled geometry

The fitted `InternalPrior` (`mxtaltools/conformers/prior.py`) is a deliberately dumb
one-body prior with two exceptions where a product of marginals provably cannot work. It is
a *sampling device for the trainer, not part of the reward*, and lives on the modeller rather
than on the energy.

**The design constraint throughout is that `sample` and `log_prob` describe the *same*
density.** That is most of why it is worth building rather than reusing ETKDG, whose draws
come without a density.

**Marginals.** `Histogram1D` per type, typed on `(element, degree)` tuples — no SMARTS, no
bond orders, no ring-aware refinement beyond the ring/acyclic split. Degree stands in for
hybridisation, which separates sp3/sp2/sp cheaply; anything finer is work the n-body heads
should be doing better. Bins 60/60/36 over `r ∈ (0.6, 2.6) Å`, `theta ∈ (0, pi)`,
`phi ∈ (−pi, pi]` periodic. Every marginal is **fattened toward uniform** (`fatten = 0.15`),
which buys full support — and support is all TB strictly needs, since TB is off-policy
consistent and only the support matters, not the weights.

**Coupling 1 — ring systems, drawn jointly.** Closure is a hard constraint that a product of
marginals is *guaranteed* to violate, and unlike the sibling case there is no structural fix,
so the whole DoF block of a ring system is drawn together. `ring_blocks` sorts each system
into four classes, **recorded rather than re-derived** because all four end in `bank = None`
and the tuple return cannot tell them apart: fitted pucker subspace / discrete bank, aromatic
and held planar by design, no key resolved, bank too thin. A system with no bank is *rattled*
at a **fraction** of thermal width (`ring_jitter_scale = 0.1`) rather than sampled: closure is
nonlinear, so independent per-DoF perturbations accumulate around the loop with a lever arm.
Measured on cyclohexane and naphthalene, closure error is linear in that scale and 0.1 puts
it at **0.025–0.043 Å**, at or under a bond's own thermal width of 0.041 Å [OBSERVED]. Ring
DoF outside the block, and substituents hanging off ring atoms, lock to what the ring chose.

**Coupling 2 — sibling torsions, drawn jointly.** The set of atoms placed onto one parent has
dihedral *differences* that fix a bond angle at that parent — an H–C–H angle at a methyl is
one such difference, and it is a graph angle the force field scores but the tree does not
expose as a coordinate. Drawn independently, even from perfect marginals, a substantial
fraction of sibling pairs land on the same rotamer mode and put two substituents in the same
place. `torsion_groups` partitions by parent and every member takes the leader's angular
displacement. **The load-bearing part is excluding the improper rows**, not the choice of
key: on a spanning tree every atom has exactly one parent, so for a proper row the reference
`b` is a function of `c` and keying on parent or on central bond gives the identical
partition. Impropers are the only rows sharing a parent with *different* references — that is
what an improper is — so leaving them in makes the displacement land about mismatched axes
and destroys the angle at the shared parent.

**The negative control is wired and is required to fail.** `joint_rings=False` gives every
ring DoF an independent marginal. On cyclohexane at `full`, closure error goes from **0.086 Å
(2.2 bond-σ) to 2.93 Å (75 bond-σ)** and the median potential rises by two orders of
magnitude [OBSERVED]. The draws remain valid *support*, which is all TB strictly needs, but
as a *proposal* they are broken — so a benchmark quoting that path is measuring the disabled
path, not the prior. `stats['closure_err']` is measured on both arms deliberately, and
`test_ring_closure` asserts both `on < 0.25 Å` and `off > 4·on`, so the gate cannot pass
blind.

**Where a product of marginals still fails, and the fix is descent not sampling.** A long
chain drawn from independent marginals walks through itself. At `full`/mmff on the glycine
series the LJ term is 36% of the median energy at Gly3, 59% at Gly4 and 91% at Gly6 (2397 of
2526 kcal/mol on Ala4), while every *bonded* term scales linearly and sanely — angle
20.7 → 52.9, bond 12.0 → 24.0, electrostatic 10.2 → 22.5 across the series. So the draw is
not wrong, it is **unaccepted**: the best decile is fine and the bulk self-intersects.
Oversampling barely helps (keeping the best 1/100 of 20,000 reaches only `T_eff/T` 3.0 on
Gly6); a few Rprop steps *in state space* fix it outright. `T_eff/T = 2.0` is the target for
a correctly thermal sample, because equipartition puts the median excess at `d/2`
[MECHANISM], and `frac_within_equipartition` agrees independently:

| molecule | `d` | 0 steps | 8 | 10 | 12 | 15 | 20 |
|---|---|---|---|---|---|---|---|
| phenyl-THP | 72 | 2.05 | 1.15 | 1.09 | 1.06 | 1.04 | 1.02 |
| Gly4 | 87 | 7.03 | 2.20 | 2.00 | 1.89 | 1.79 | 1.70 |
| Gly6 | 129 | 29.19 | 2.47 | 2.21 | 2.06 | 1.92 | 1.80 |
| Ala4 | 123 | 41.84 | 2.43 | 2.15 | 1.99 | 1.86 | 1.74 |

[OBSERVED, 4096 draws each.] 10–12 steps lands the peptides on 2.0; **phenyl-THP needs
none** — its raw draw is already thermal and relaxing it only over-cools, which is why the
default is 0. A molecule whose prior is already right must not be "repaired".

Overall the fitted prior benchmarks **32×–87,000× over uniform-on-box** on median energy
excess [OBSERVED], which is why the conformer protocol deliberately omits the crystal route's
`snapshot_prior`: phase 1 runs to broaden the policy, and must not displace a prior that is
already good.

---

## 6. Instantiation, and the multi-molecule boundary

| # | Stage | Code | Frequency | Function of the graph alone? |
|---|---|---|---|---|
| 1 | SMILES → mol, add H, ETKDG embed, MMFF-optimise | `ConformerTorsions.__init__` | once per molecule | **no** — seeded embedding |
| 2 | bond graph | `perception.infer_bond_index(z, ref_pos)` | once | **no** — perceived from geometry (§8 T6) |
| 3 | spanning tree | `topology.spec_from_graph(..., use_geometry=False)` | once | **yes** |
| 4 | reference DoF `r0, th0, ph0` | `builder.measure` on the embedded conformer | once | **no** (§8 T1) |
| 5 | force field | `_make_ff` → `ff_from_reference` \| `ff_from_mmff` | once | `mmff`: yes; `reference`: no |
| 6 | linearity flags | `_typed_linear` \| `_linear` | once | `mmff`: yes; `reference`: no |
| 7 | rotatable axes, state→DoF map `M`, `d` | `_find_rotatable`, the `keep` mask | once | yes, given 3 and 6 |
| 8 | Jacobian constants | `log_jacobian_const`, `log_chart_jacobian` | once | yes, given 4 and 7 |
| 9 | condition graph (`MolData` + `ctree_*`) | `conformer_data.condition_from_energy` | once per dataset build | yes, given 3–7 |
| 10 | frozen per-atom + pooled embeddings | `models/encoder_cache.py`, permuted by `spec.perm` | once per molecule, offline | yes |
| 11 | batched tree + FF for a batch size | `ConformerTorsions._batch` (cached, sample-budgeted) | once per distinct batch size | — |
| 12 | `x` → `(r, theta, phi)` | `dof_from_state` | every step | — |
| 13 | `(r, theta, phi)` → positions | `builder.build`, round-wise NeRF | every reward call | — |
| 14 | `U`, `log J`, reward | `potential_energy`, `jacobian_energy`, `energy` | every reward call | — |

`build` proceeds in **rounds** — every atom whose three references were placed earlier goes
down in one vectorised call across all molecules — so sequential kernel launches scale with
maximum tree depth (~10–25 for drug-like molecules), not atom count. `measure` is its exact
inverse and is fully parallel.

**Force-field choice is not cosmetic.** `ff_from_reference` measures `r0/theta0` off the
embedded conformer, so bonded terms are exactly zero there — but it carries **no torsion term
at all**, which makes every rotamer distribution nearly uniform (propanol's target entropy
7.601 of a maximum 7.625) and leaves amide omega degenerate between cis and trans; and its
parameters depend on the embedding seed, which is fatal conditionally. `ff_from_mmff` is
RDKit MMFF94 typing — graph-determined, a real 3-term torsion, full organic coverage — at the
cost that the reference conformer stops being the energy minimum. **`mmff` is the only
defensible choice for a conditional run.**

**Conditioning.** The policy sees the molecule through two static channels: a **pooled**
embedding appended to the condition vector (over which `GFN.init_flow_model`'s existing
`scalarMLP(condition_embedding_dim → 1)` *is* the `log Z(c)` head the design asks for), and
**per-atom** embeddings bound once per trajectory by `ConformerGFN.bind_molecular_conditioning`
and read by `SetPolicy` as the static per-coordinate identity `f_j`. Two silent traps, both
now guarded: the encoder's atom order (heavy-first, then H) differs from placement order and
is realigned once by `spec.perm` at cache-build time; and `dof_atoms` is stored in local tree
numbering while embeddings collate to `[N_total, E]`, so indices must be shifted by each
graph's `ptr`. **Precomputing is freezing** — no gradient reaches the encoder while the cache
is in use. Measured: butanol vs butylamine give condition `(2, 1)` and identical with the
flag off, `(2, 256)` and differing by 8.39 with it on [OBSERVED].

**The multi-molecule boundary.** `MultiConformerTorsions` holds one member chart per
identifier and dispatches by grouping rows; dispatch is cheap because
`ConformerTorsions.energy` never reads `mol_batch`. The molecule list is read from the
**conditions file**, not a config key, so it cannot disagree with the set being trained on.
It has to exist because at `torsion` the reward reduces to
`−U/T + log_jacobian_const + log_chart_jacobian` and both terms are per-molecule —

```
CCC(=O)CC    log_jacobian_const 4.508    log_chart_jacobian 2.289
CCCCO                           3.994                       2.289
OCc1cnco1                       3.472                       1.145
```

— so a mixed batch scored against one member is wrong by a different constant per row, which
is exactly a per-condition `log Z` error. Measured at **3.33 nats of per-row error across
three ordinary QM9 molecules** before the fix [OBSERVED]; the shapes agree whenever `d`
agrees, so it returned numbers rather than raising. One conditions file is **one `k`**, since
the GFN fixes its state dimension at construction.

Verified against code 2026-09-08:

| path | code | state |
|---|---|---|
| reward scoring | `MultiConformerTorsions.energy` | **per-row** |
| prebuilt reward off a buffer row | `MultiConformerTorsions.prebuilt_sample_to_reward` | **per-row** |
| prior dataset intake | `ConformerModeller.init_prior_dataset` (graph-form `prior_path`) | **multi-molecule, off disk** |
| prior *state generation* | `sample_prior_states` — **not overridden** in the set class | reference chart only |
| forward / eval conditions | `init_mol_dataset` — "one condition: the molecule itself" | **one molecule, no embedding** |
| prior-buffer growth | `Modeller.grow_prior_buffer`, not overridden | one molecule |
| prior churn | `sample_from_prior` — "exactly one condition on this route" | one molecule |

So a conditional run today is **backward-branch-only in the honest sense**: the backward
branch draws multi-molecule rows off disk and scores them through their own charts, while the
forward branch's condition source is a two-copy batch of `energy_config.smiles` carrying no
`embedding` at all — `condition_samples` raises on it when `embedding_conditioning` is on.
Plumbing, not physics, but it is the difference between a conditional result and a wiring
demonstration.

---

## 7. Evidence: reconstruction correctness vs. sampling correctness

These are two different claims and the doc has to keep them apart. **Reconstruction** is
"the chart is the map we say it is, and it round-trips". **Sampling** is "the trained policy
draws from the target". Reconstruction is comprehensively gated. Sampling is gated at
`torsion` and *unestablished* at every level above it.

### 7.1 Reconstruction — what is gated, and how tightly

| Claim | Gate | Bar |
|---|---|---|
| DoF count is exactly 3N−6 | `test_dof_count` | exact |
| `measure ∘ build = id` | `test_roundtrip_exact` | exact |
| dihedral sign convention matches RDKit | `test_dihedral_convention_matches_rdkit` | — |
| batched build == per-molecule loop | `test_batched_matches_individual`, `test_replicated_batch_pairs_match_the_per_molecule_loop` | — |
| the BAT volume element is what we claim | `test_log_jacobian`, `test_measure_term` | 1e-12 |
| the chart term is constant in `x` and scales correctly with `T` | `test_chart_volume_element` | 1e-8 |
| `state_from_dof` inverts `dof_from_state`, and **refuses** where it cannot | `test_state_dof_roundtrip` | 1e-12 |
| `torsion` is bitwise unchanged by the level machinery | `test_torsion_bitwise` | 1e-12 |
| the domain guarantee holds off-box | `test_domain_guarantee` | 1e-12 |
| rings close | `test_ring_closure`, `test_negative_control_worsens_closure` | `< 0.25 Å` **and** control `> 4×` |
| acyclic molecules have no closure bonds | `test_acyclic_has_no_broken_bonds` | exact |
| nonbonded exclusions are right | `test_nonbonded_exclusions` | exact |
| the tree is canonical / graph and mol paths agree | `test_tree_shape_is_canonical`, `test_graph_and_mol_paths_agree` | exact |
| MMFF matches RDKit term by term | `test_every_term_matches_rdkit`, `test_total_matches_rdkit` | per-term |
| torsion barriers match experiment | `test_torsion_barriers_match_experiment` | — |
| the chart *structure* does not move with the embedding seed | `test_chart_is_a_function_of_the_graph` | 4 seeds × 8 molecules, typed == measured, margin > 2° |
| float32 is sound against float64 | `test_float32_default_is_sound` | 1e-5 relative on `U`, 1e-4 on `log J`, 1e-4 Å on closure |
| the prebuilt reward equals `energy()` at every level | `test_prebuilt_reward_matches_energy_at_every_level` | 1e-6 |
| the baked buffer energy excludes the measure terms | `test_baked_energy_excludes_measure` | 1e-10 |
| a level cannot be swallowed by config | `test_level_cannot_be_swallowed` | raises |

The exactness gates run in **float64 deliberately** — at float32 they would have to be
loosened to ~1e-6, which is a weaker test of the thing being tested.

**The one reconstruction gate that does not work is the end-to-end one.**
`check_state_convention` rebuilds positions from a stored condition graph and compares
against the energy — and it is wrong at every level except `torsion` (§8 T3). So the *unit*
reconstruction claims are strong and the *integration* claim is unverified above `torsion`.

### 7.2 Sampling — what is established

| Claim | Evidence | Grade |
|---|---|---|
| the sampler reaches the exact partition function at `torsion` | propanol `log Z 5.935` against an exact `5.9365` | OBSERVED, one run |
| the prior is a far better proposal than uniform | 32×–87,000× on median energy excess | OBSERVED |
| the prior is thermal after relaxation | `T_eff/T → 2.0`, `frac_within_equipartition` 0.44/0.51 against 0.5 | OBSERVED, 4096 draws |
| the energy-marginal metric can tell a converged sampler from a broken one | `test_energy_marginal_overlap`: matched `< 4.0`, corrupted `> 10.0` | gated |
| **anything at `flex` or `full`** | **nothing** — see T5 | — |

`T_eff/T` and coverage replaced ESS deliberately; ESS on a `d ≈ 100` chart is dominated by a
few rows and reads as catastrophic whatever the sampler does.

**The honest summary for a methods section:** the chart is verified to reconstruct geometry
and carry its measure exactly, in float64, at every level. The *sampler* is verified only at
`torsion`, on one molecule, against one analytic answer.

---

## 8. Threats to the claim

**T1 — the chart's origin is not a function of the graph.** *Costs C1.* The chart's
*structure* is graph-determined and gated: `test_chart_is_a_function_of_the_graph` pins `d`
and both linearity counts across four embedding seeds on eight molecules under MMFF, requires
the typed and measured routes to agree exactly, and requires no angle within 2° of the
threshold. **The chart's *origin* is not gated and does move.** `r0, th0, ph0` are `measure`'d
off an ETKDG+MMFF conformer, and at `torsion`/`dihedral` `r` and `theta` are *frozen at those
values* — so the target distribution itself depends on the embedding seed. Measured drift
across seeds: **0.0086 Å / 0.20 rad / 3.14 rad**, carrying `e_ref` by **0.245–1.835 kT**
[OBSERVED]. The 3.14 rad on `ph0` is a frozen dihedral landing in a different well.
**This is the one we cannot argue our way out of in print.** Either the frozen values come
from graph-typed MMFF equilibria, or `full` is the only level we publish.

**T2 — the anti-collinear reference cascade is dead code.** *Costs C1 and C2 on ~3% of QM9.*
`spec_from_graph`'s `ref_a`/`ref_b` cascade (§2) is guarded by `_angle_ok(pos, ...)`, and
`pos is None` whenever `use_geometry=False` — which is exactly how the conformer route calls
it, deliberately, for load reproducibility. `_linear_mask` likewise returns all-False with no
geometry. So the cascade never runs, and `mask`/`rotatable` (no linearity predicate at all)
diverge from `_M`/`data_ndim` (which drop a column iff every row it drives is frame-linear).
Full QM9 [OBSERVED, 2026-09-08]: **3,863 molecules diverge = 2.89% of QM9, 5.64% of buildable;
3,861 contain a C#C, P(divergent | alkyne) = 35.9%.** A further **1,082 (0.81%) are refused
outright** with "no free degrees of freedom", including real flexible molecules like 3-hexyne.
At `full`, 13.5% of molecules have a linear phi frame and worst cases lose 14–17 of 30–45 DoF.
**Both layers are wrong, in measured proportion** (400 molecules, 616 dropped axes,
rigid-body-projected Jacobian): 13% of dropped axes move nothing and 46% duplicate a kept
axis, so `mask` over-counts — but **40% are a genuinely lost internal direction**, so `_M`
under-counts real physics. 20% of divergent molecules lose a direction reaching 1.35–1.87 Å
RMSD from anything the kept state can span, and 27.8% of those exceed kT at 300 K.
`_typed_linear` is already the graph-determined predicate the design asks for; only the wiring
into `ref_a` is missing. The acetonitrile inversion (§3.2) is the same code.

**T3 — the end-to-end reconstruction gate does not work above `torsion`.** *Costs our
ability to assert C4 at `full`.* `_state_columns` reads `energy.mask`, so on a divergent
molecule it emits a column index the state cannot hold and `check_state_convention` dies deep
in `state_to_phi` with a bare `IndexError` naming neither molecule nor mismatch. And it is
**wrong at every level except `torsion`**: `CCCCO` passes `condition_from_energy` at
dihedral/flex/full and then disagrees with the energy by **5.3 / 6.2 / 6.1 Å**, silently
[OBSERVED].
*Scoping, established while writing this file:* `ctree_state_col`, `state_to_phi` and
`states_to_positions` are reached **only** from `check_state_convention`, called **only** by
`build_conformer_conditions.py`. Nothing on the training path reads them. So this is not a
corrupted reward — it is a **build-time gate that does not work**, which is a different and
more embarrassing problem: every `full` dataset to date was built without a functioning
geometry cross-check.

**T4 — nothing pins the stereoisomer above `torsion`.** *Costs C7.* Derived in §3.5: the
chart double-covers physical conformer space under `phi → −phi`, and MMFF is mirror-invariant,
so both enantiomers carry identically equal reward. Protected for free at `torsion` by the
frozen reference; live at `dihedral`; unprotected at `flex`/`full`, and not protected by a
barrier either. The wall exists as a design (§3.5) and is not built; `atom_parity` exists in
`energies/dof_features.py` as a control feature and is not in the condition schema. A chiral
molecule above `torsion` is currently not a well-posed target.

**T5 — `full` has never converged, and TB at `full` is untested.** *Costs nothing in §1;
costs everything in a results section.* 14 runs, Gly4 only, all local, 2026-08-23/24. The
level is real — logs print `level 'full': 87 free (r 30/30, theta 29/29, phi 28/28)` — but 11
of 14 have no summary at all, and the 3 that finished read `E/emarg_w1_ratio`
nan / 898.1 / 1045.2 against the project's own bar of matched `< 4.0`. `log_z_emp` vs
`ema_logw` ≈ 3797 nats where a closed TB fixed point has zero. Every completed run is
`mle 1, tb 0, freeze_z 1` — pure MLE with the flow head frozen, so **TB at `full` is
entirely untested, not merely unconverged**. Phenyl-THP has `full` configs and was never run
[OBSERVED, audited against wandb summaries 2026-08-25].

**T6 — bonds are perceived from geometry when RDKit's bond table is right there.** *Costs C1
cosmetically; cheap to fix.* `ConformerTorsions.__init__` calls `infer_bond_index(z, ref_pos)`
— a covalent-radius distance rule at tolerance 1.2 — on a molecule it built from SMILES.
`perception.py`'s own docstring explains why the function exists (stored `MolData` carries no
bond topology), and that reason does not apply here. For any sane embedding this recovers the
RDKit graph, but it is a geometry dependence in the one pipeline whose headline claim is
graph-determinism, and a referee reading the code will find it.

---

## 9. What has to happen before print

In dependency order, each checkable:

1. **Wire `_typed_linear` into the `ref_a`/`ref_b` cascade** (T2). Simulated at +26% state
   dimensions on divergent molecules. In the same change, make
   `mask`/`rotatable`/`rotatable_cols` a **view of the surviving set**, expressed on the mask
   itself so it is level-independent: on 124 molecules that took `check_state_convention`
   from 123 failures to 0, max position discrepancy exactly 0 Å in float64. Do **not** derive
   `_state_columns` from `_M` — tried, and `_driven_idx` carries non-phi rows at other levels,
   breaking 47 passing tests.
2. **Fix `_state_columns` above `torsion`, or refuse it there** (T3), so
   `build_conformer_conditions.py` actually gates the geometry it writes. This is the gate
   that would have caught 1.
3. **Replace the embedded reference DoF with graph-typed equilibria** (T1), or restrict the
   published claim to `full`. Then re-measure the seed drift and show it is zero.
4. **Build the signed-volume wall and put parity in the condition schema** (T4), with leakage
   logged per centre and the `log Z` correction applied, or restrict the published claim to
   achiral molecules.
5. **Give the forward branch a multi-molecule condition source with embeddings** (§6), which
   also unblocks eval and prior-buffer growth.
6. **Dispatch `sample_prior_states` per chart** (§6), so the prior is the set's prior.
7. **Then, and only then, a converged `full` run with TB engaged** (T5), and a sampling-side
   evidence table that is not empty above `torsion`.

Items 1–4 change what we are allowed to say. 5–6 are plumbing. 7 is the experiment.
