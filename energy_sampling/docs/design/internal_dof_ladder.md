# Full internal-DoF conformer generation

Argument doc. Why the conformer GFN should move from a torsion-only state to the full
internal-coordinate set, what the intermediate levels are and are *not*, where chirality
enters, and what has to be built before any of it runs. Every claim is a code citation or
an argument, and the doc says which. No `decisions.md` entry yet: §10 lists what is still
open.

**Steps 1 and 2 of §10 are built** (`test_conformer_levels.py`, 9 gates). Steps 3–7 are
not; nothing above `dihedral` has been trained. Two things §3/§7 predicted were confirmed
in the doing: the state→DoF map is a LINEAR MAP and not an index subset, because a
`torsion` column drives every dihedral about its bond (§3), and the linearity flags were
all-False for every molecule ever run (§7 item 1). Git history is the log.

Revised 2026-08-13 after an adversarial review of the first draft. Four claims in that
draft were confirmed wrong and are corrected in place (§3, §4, §5, §9); the review is
Log and lives in git.

**The target is `full`.** A state carrying all 2-, 3- and 4-body internal coordinates for
a given molecule, scored by a force field determined by the molecular graph. sp3 geometry
stops being frozen input and becomes learned output. The lower levels in §2 are **not
approximations to it** — see §2.

---

## 1. What already exists

More than the current `train_conformer.py` route suggests. All in `mxtaltools/conformers`:

- **The builder already takes the full set.**
  [`build(tree, r, theta, phi)`](../../../../mxtaltools/mxtaltools/conformers/builder.py:163)
  consumes `r` [N−1], `theta` [N−2], `phi` [N−3] — 3N−6 = `TreeSpec.n_dof`
  ([topology.py:62](../../../../mxtaltools/mxtaltools/conformers/topology.py:62)).
  [`measure`](../../../../mxtaltools/mxtaltools/conformers/builder.py:195) is its exact
  inverse. Output is SE(3)-reduced by the seed convention, so nothing downstream needs
  alignment. `build_positions` ([conformer_torsions.py:204](../../energies/conformer_torsions.py:204))
  already calls it with all three and holds two at reference values.
- **The graph-determined force field exists.**
  [`ff_from_graph`](../../../../mxtaltools/mxtaltools/conformers/energy.py:289) emits the
  same `ForceField` as `ff_from_reference`, with r0/theta0/k from typed `(element, degree)`
  lookups. Bonds, the full redundant graph angle set, the full redundant proper-torsion
  set, soft-core LJ, ring closures.
  [`intramolecular_energy`](../../../../mxtaltools/mxtaltools/conformers/energy.py:392)
  scores from **Cartesians**, so one function scores an internal and a Cartesian
  parameterisation alike. Note its own docstring at
  [energy.py:305](../../../../mxtaltools/mxtaltools/conformers/energy.py:305) is stale —
  it claims "still no torsion term" while the body populates `tors_v`/`tors_n`/`tors_gamma`.
- **The volume element exists — but it is not what its docstring says.**
  [`log_jacobian`](../../../../mxtaltools/mxtaltools/conformers/builder.py:208) = `Σ 2 log r
  + Σ log sin θ`. Its docstring labels this `log|d(cartesian)/d(internal)|`, which is
  **wrong**: `build` is SE(3)-reduced, so its own determinant is square [3N−6 × 3N−6], and
  the formula exceeds it by exactly the SE(3) orbit volume `log(r₁²·r₂·sin θ₂)` — measured
  at 0.777 and 0.426 nats on a linear chain and a branched root. The formula is the **BAT
  volume element**, relating the internal measure to the full 3N Cartesian measure with the
  6 external DoF integrated at Haar × Lebesgue. That is the right object for a gas-phase
  conformer, and it is correct and complete for any tree topology including the seed atoms.
  See §4 for why the label matters.
- **The prior already covers all three classes.**
  [`InternalPrior`](../../../../mxtaltools/mxtaltools/conformers/prior.py:109) has
  `bond_key`/`angle_key`/`torsion_key` and a `RingBank` for ring-system DoF blocks.
- **The dataset layer was built for the full tree.** `ctree_r0`, `ctree_theta0`,
  `ctree_phi0` are already per-atom fields in
  [`conformer_data.py`](../../energies/conformer_data.py), because an atom at placement slot
  k owns exactly `min(k, 3)` DoF. Only the *state* was torsion-restricted.
- **The tree is graph-native.**
  [`spec_from_graph`](../../../../mxtaltools/mxtaltools/conformers/topology.py:161) runs with
  `pos=None` under `use_geometry=False`, which is also a hard persistence rule: anything
  stored must be built that way or the spec is not reproducible at load.
- **A relaxed-scan optimiser exists.**
  [`gradient_descent_optimization`](../../../../mxtaltools/mxtaltools/conformers/optimize.py:102),
  used by §8.

What does not exist: any level toggle, the Jacobian in any reward, impropers, chirality
handling anywhere, and — see §6 — a runtime that can run the conditional half of this.

---

## 2. The levels, and what they are not

Widths are for butanol, N=15, so 3N−6 = 39 (14 bonds, 13 angles, 12 torsions).

| level | free | d | FF source | 3D reference needed for |
|---|---|---|---|---|
| `torsion` | rotatable φ subset | 2 | either | frozen r0/θ0, linearity |
| `dihedral` | all φ | 12 | graph | linearity |
| `flex` | θ, φ | 25 | graph | linearity |
| `full` | r, θ, φ | 39 | graph | linearity |

**These are not a ladder of approximations.** Freezing a DoF at a constant gives
`p_full(free | frozen = c₀)` — a **conditional slice**, the flexible-constraint ensemble.
It is not the rigid-constraint ensemble, which differs by the Fixman `|H|^½` factor (the
Schur complement of the mass-metric tensor over the frozen block) — a *state-dependent*
factor, not a normalisation. Nor is it the stiff-spring limit of the full system, which
sits at the conditional minimum `s*(q)`, and `s*(q) ≠ (r0, θ0)` because LJ, ring closure
and the redundant graph angles all depend on r and θ.

So: **`full` is the target. `flex`, `dihedral` and `torsion` are each some related
distribution** — well-defined, useful for staging and regression, and carrying no limiting
relationship to `full`. The vocabulary borrowed from
[`InternalParams`](../../../../mxtaltools/mxtaltools/conformers/optimize.py:53)
(`freeze` over `CLASSES = ("r","theta","phi")`) names the mechanism, not an approximation
claim; and note that class-level freeze cannot express §4's requirement to freeze
*individual* θ rows, so the ladder's per-DoF `free_mask` strictly generalises it.

Two consequences the first draft got wrong:

- A φ-marginal measured at `dihedral` is not an approximation to the φ-marginal at `full`.
- Step 3 cannot "measure the FF change independently of the dimensionality change." Its
  real content is narrower and is restated in §9.

`dihedral` is retained because it is where chirality first becomes reachable (§5) and
because it exercises the mask without activating the Jacobian — not because it approximates
anything.

---

## 3. Parameterisation and the domain guarantee

### One mask, one builder call

Everything reduces to a `free_mask` over the concatenated `[r | θ | φ]` vector of width
3N−6. State `x [B, d]`, `d = free_mask.sum()`, scatters into the reference vector; the
complement holds. Then one `build`. `build_positions` is already this, specialised.

### The periodic mask — `TorsionGFN` is NOT simply deleted

**The first draft was wrong here.** `_finalize_dim_partition`
([models/gfn.py:320](../../models/gfn.py:320)) does not choose a layout — it *receives*
`angs` as an argument. The layout owner is `get_periodic_dimensions`
([models/gfn.py:250](../../models/gfn.py:250)), and that is what `TorsionGFN` overrides.
Deleting the override gives one of two failures:

- `do_periodic_angles=True` → raises `expected crystal state dim 6 + 6·max_z_prime`
  ([gfn.py:282](../../models/gfn.py:282)). At d=12 it **coincidentally passes** and assigns
  the crystal mask: 2 wrapped dims out of 12 torsions, 10 unbounded. Runnable and wrong.
- `do_periodic_angles=False` → `angs = [False] * self.dim`
  ([gfn.py:302](../../models/gfn.py:302)). This is the branch `train.py:1502` selects, since
  `do_periodic_angles = energy_function.is_crystal` and `ConformerTorsions.is_crystal` is
  False. **Zero wrapped dims.** `_wrap_ang` returns the state untouched,
  `expand_state_for_policy` emits no sin/cos, `gauss_logprob` loses its nearest-image
  residual. φ becomes an unbounded linear dim whose reward is exactly 2-periodic, so
  `∫R dx` diverges: **no finite log Z exists and the TB fixed point does not exist.**
  Nothing crashes.

**The fix is already in the codebase and inert.**
[`ConformerTorsions.periodic_dims`](../../energies/conformer_torsions.py:251) returns the
per-dim flag list and its docstring states it exists precisely so the model does not infer
from `is_crystal`. A grep of both repos returns **one hit: its own definition.** Nothing
reads it, while `train.py:1502` still infers. Wire it: add an explicit angular-mask argument
to `GFN`, make the crystal branch of `get_periodic_dimensions` conditional on its absence,
and assert at construction that `ang_dim == sum(energy.periodic_dims)` for non-crystal
energies.

### Per-block units

| block | latent | topology | scale |
|---|---|---|---|
| φ | delta from φ0, [−1,1] ≡ (−π,π] | periodic | π (existing) |
| θ | delta from typed θ0 | linear, bounded | Δθ_max |
| r | delta from typed r0 | linear, bounded | Δr_max |

Affine per-dim maps contribute only a constant to log Z — but the constant has counts affine
in N, so it is a **per-molecule offset**. Graph-determined, therefore benign, but it must be
included wherever log Z(c) is compared across molecules.

### The domain guarantee is a CLAMP, not the wall

The first draft argued that a bounded dim in this codebase is bounded by a wall. That is
half the crystal mechanism and the wrong half. The crystal is **hard-clamped before the map
is ever evaluated** — `crystal_ops.py:301` clamps the latents, with margins at exactly the
rows whose Jacobian has a log singularity, and `latent_params()` clips again on the way
back. The wall is a *preference* that discourages excursions; the clamp is what makes the
energy finite for every latent in ℝᵈ.

Without both, three things break:

1. `log_jacobian` has **no clamp** where the crystal's `compute_jacobian` has
   `.clamp_min(eps)` ([molecular_crystal.py:552](../../energies/molecular_crystal.py:552)).
   r ≤ 0 → `log` of a negative → NaN. θ ∉ (0, π) → `sin θ ≤ 0` → NaN or −inf.
2. `build` is **non-injective off-domain**: `d(r, 2π−θ, φ+π) = d(r, θ, φ)` and
   `d(−r, θ, φ) = d(r, π−θ, φ+π)`. So an excursion double-covers rather than merely being
   re-weighted, and the multiplicity is not uniform, so log Z cannot absorb it. `measure`
   cannot detect it either — `bond_angle` returns `atan2(|u×v|, u·v) ∈ [0, π]` always.
3. `train_conformer.py:605-611` counts non-finite gradients and **steps the optimizer
   anyway**, unlike `train.py:2856`. One out-of-domain sample NaNs every weight permanently.

So: clamp the latent before `build_positions`, clamp inside `log_jacobian`, **and** add a
`bounding_energy`-style wall on the raw r/θ latents pre-multiplied by temperature. The
physical ranges are narrow enough that the wall should rarely bind after early training —
which is exactly why the clamp is not optional, since "rare" times "unrecoverable" is still
fatal. Gate: a latent of ±5 on an r dim and a θ dim must give a finite `log_reward`.

---

## 4. The Jacobian

It enters the **energy**, pre-multiplied by temperature — not `log_reward`, and not `GFN`.

Three separate reasons, all confirmed against code:

1. **`log_reward` is not called on this route.** `BaseSet.log_reward` is `-self.energy(...)`,
   and `train_conformer.py:598` computes `log_reward = -energy.energy(final, None, log_temp)`
   directly. A term added to `log_reward` is **inert**: training, eval and the printed
   `dlogZ` all stay identical and the run converges cleanly to the wrong distribution.
2. **`energy()` divides by T.** A term added without compensation lands as `J^(1/T)`, not
   `J`. The crystal already does this correctly and for this reason:
   [`compute_jacobian`](../../energies/molecular_crystal.py:542) returns
   `- temperature * torch.log(...)` with a "CHANGE OF MEASURE" comment. The first draft cited
   `:446` (the bounding wall) and missed `:552` — the term that actually reshapes the target.
3. **The crystal precedent puts it in the energy**, in `generator_energy`, not in the model.

**Always add it; do not gate it on the freeze set.** `log_jacobian`'s own docstring says why:
with r and θ frozen the term becomes a *condition-dependent constant* which "must be added
back if partition functions are compared across molecules" — and §6/§8 compare Z(c) by
construction. Constant in x is not constant in c.

**Test discipline.** Two traps:

- The obvious test — autograd determinant of `build` versus `log_jacobian` — **fails on
  correct code** by the SE(3) orbit volume (§1). Compare against the BAT element, or against
  the autograd determinant *plus* `log(r₁²·r₂·sin θ₂)`. There is an existing numerical check
  at `mxtaltools/conformers/tests/test_conformers.py:104` to reuse.
- A single-temperature test passes on the un-compensated code, because T=1 is the
  `ConformerTorsions` default. **Run the gate at two temperatures.**

Four other consumers of `energy()` change meaning when the term lands and must be audited:
`brute_force_log_z`, `exact_references`/`boltzmann_reference`, `bake_energies`, and
`build_conformer_buffer`'s `local_optimize`/`harmonic_weights` — the last redefines what a
"mode" is.

**θ singularities.** `TreeSpec.angle_is_linear` / `torsion_frame_is_linear` flag the genuine
`log sin θ → −∞` cases (alkynes, nitriles, azides), and reference choice cannot avoid them.
Those DoF must be frozen or the molecule filtered — but see §7: both flags are all-False
today for every molecule, so the guard is currently a stored non-measurement.

---

## 5. Chirality

### Where it lives — two cases, not one

**Interior centre.** Reference-atom selection
([topology.py:238](../../../../mxtaltools/mxtaltools/conformers/topology.py:238)) prefers
`parent(b)`, so for an sp3 centre C with parent W and grandparent V, all of C's children
share the frame (V, W, C). The configuration is the *cyclic ordering* of their three
dihedrals — not any single state dimension.

**Root centre — the common case.** `choose_root`
([topology.py:99](../../../../mxtaltools/mxtaltools/conformers/topology.py:99)) takes the
graph centre tie-broken toward heavy atoms of high degree, which describes an sp3
stereocentre. At the root there is no (V, W, C): slot 1 is `place_seed_second` (r only,
pinned to +x), slot 2 is `place_seed_third` (r and θ, pinned to the xy half-plane), so two
of four substituents carry no dihedral and their arrangement is fixed by the seed
convention. The first NeRF-placed child's φ then decides which side of that plane it falls
on. Measured: sweeping that φ through zero flips χ with an exactly odd signature, and
sweeping any other φ leaves χ constant to all digits. **At the root, chirality is the sign
of one state dimension.**

`place_nerf` gives the global picture separately: φ enters through both `cos(φ)·m2` and
`sin(φ)·n̂`, and only the `n̂` term is **odd**, so negating every φ is exactly the mirror —
verified numerically, orthogonal to 8.9e-16 with `det = −1.000000000`.

A half-period wrap at the root is **not** a free exact constraint: φ → φ+π is a rotation,
not a reflection, and folding by φ → −φ maps a *diastereomer* onto the target rather than
folding a symmetry. The root case makes the constraint simpler — a sign on one latent rather
than a derived signed volume — not free.

### Why it is not protected here

`soft_core_lj` caps at `(24ε/k)(e^k − 1)` = **10.735 kcal/mol** per pair at ε=0.1, k=2.5 —
bounded, not divergent. A GFN takes Gaussian jumps in latent space rather than following a
continuous path, and backward training seeds terminals straight from a buffer.

At `torsion` the problem does not exist: r and θ are frozen at a reference that already has
a configuration, and rotating a rigid fragment about a bridge cannot invert a centre.
**Chirality is currently protected for free by the frozen reference.** It appears at
`dihedral`.

### The target and the mechanism

A molecule is graph + chiral labels, so the target is the Boltzmann distribution restricted
to one stereoisomer. The global mirror is therefore not a gauge to fold, and enforcing every
centre subsumes it.

Mechanism: a wall on the **normalised** signed volume χ — `k·relu(−s·χ)²`, matching
`generator_energy`'s idiom ([molecular_crystal.py:429](../../energies/molecular_crystal.py:429)).
Normalised because raw volume scales with r³. Pre-multiplied by temperature, same as §4.

`relu` is exactly zero on the allowed side, so the wall does **no interior distortion**; its
only error is leakage ε. Rejecting at eval recovers exact conditional statistics, subject to
two conditions the first draft understated: the rejection predicate must be **the same set**
as the wall's zero set (an independent RDKit CIP re-perception is not), and recovering log Z
needs `log Z_allowed = log Z_walled + log(acceptance rate)` — rejection alone corrects
expectations, not log Z.

Per-centre leakage is a first-class logged series.

### `s` must key on the condition — and that needs new plumbing

Read off the sample, `relu(−sign(χ)·χ)² ≡ 0` — a no-op. Walled on one direction only, P_F
and P_B score different rewards and TB cannot close. So `s = s(c)`, and then "sample all
stereo combinations forward" means the *condition sampler* draws the combination.

**But there is no channel today.** `get_condition_embedding`
([models/gfn.py:996](../../models/gfn.py:996)) returns only the scalar branch;
`VectorMoleculeGraphModel`'s scalar inputs are built from norms and inner products and there
is no cross product anywhere in the model stack — the code's own comment says "o(3)
invariant". So a 3D conditioner is mirror-blind exactly as a graph conditioner is. And
`ConformerTorsions.condition_samples` builds the condition vector from log-temperature or a
zeros column; `mol_id` reaches only `condition_id`, which the policy never reads.

Since the plan is graph-plus-descriptors rather than a 3D embedding (§6), the resolution is
direct: **emit one signed column per stereocentre into the condition vector**, padded to a
fixed width, and add it to `get_conditioning_dim`. That is genuine new plumbing, not free.

Note the tension this resolves: §8's mirror-pair Z check asserts the two conditions are
indistinguishable in Z, while the wall requires the policy to distinguish them. Both hold
only because the distinguishing bit lives in the condition vector, not in Z.

### The prior is the larger hole

`InternalPrior` samples per-DoF histograms independently, so it cannot express a joint
sibling ordering and lands in a random basin. Backward training draws terminals straight from
the prior file, bypassing the wall entirely.

Repair, do not reject (rejection is ~2⁻ⁿ). Measure χ per centre, then:

- **interior centre** — reflect its sibling torsions about their mean. This flips exactly
  that centre: the frame reflection and the local φ-negation cancel at every descendant,
  because each descendant's parent lies on its own frame's axis. Verified on a three-level
  tree — the target χ flips while the parent's and a descendant's are unchanged to all digits.
- **root centre** — negate the first NeRF-placed child's φ. The sibling recipe has no
  sibling set here.

Then **assert the invariant on load**, so an older prior file fails loudly.

### `scramble_conditions` — resolved by the repair

`train_prior` is `train_mode: bwd` with `bwd_sampling_mode: dataset`: no forward rollouts,
every reward evaluation on a dataset terminal, every terminal correct by construction. **The
wall is exactly zero on every sample that stage sees.** Phase 1 learns the target handedness
by MLE without seeing the condition. With k isomers in one dataset it learns a k-mode mixture
and the per-condition wall selects one mode per condition — a 1/k shock, not 2⁻ⁿ. Start at
k=1.

### The ramp is not free

`set_energy_coeffs` ([train.py:863](../../train.py:863)) is `if hasattr(...): setattr(...)`,
and `ConformerTorsions.__init__` ends in `**kwargs`, so a `chirality_coeff` passed through
`energy_config` is **swallowed**, never becomes an attribute, and the ramp is a silent no-op.
Two further blockers: `protocol.energy_coeffs()` requires a numeric key in the config schema,
and `anneal_coeffs` requires `balance.kind == 'lexicographic'` plus a live protocol —
`ConformerModeller.protocol` is `None`. Set the attribute explicitly in `__init__`, never via
`**kwargs`, and log the live value as a series so a stuck ramp is visible.

---

## 6. Runtime — the blocking question

**The conformer does not run on `train.py`.** `init_energy_function` hardcodes
`MolecularCrystal(**energy_config)` ([train.py:900](../../train.py:900)) with no branch; the
only occurrence of "conformer" in `train.py` is a comment. `ConformerBuffer`
([buffer.py:1062](../../buffer.py:1062)) is fully written and **instantiated nowhere**. The
entire Modeller energy-protocol block in
[conformer_torsions.py:237-338](../../energies/conformer_torsions.py:237) —
`condition_samples`, `set_n_molecules`, `prebuilt_sample_to_reward`, `periodic_dims` — is
unreachable code.

`train_conformer.ConformerModeller` is a duck-typed shim: `protocol = None`, no
`set_energy_coeffs`, no checkpointer, no buffers module, `conditional=False` and
`conditions_dim=0` at every call site. So everything in §5 that leans on conditions,
`anneal_coeffs`, `train_prior` or `scramble_conditions` sits behind a migration the first
draft never numbered.

The migration also changes one correctness property. On `train_conformer.py` both directions
share one reward line (`train_conformer.py:598`), so fwd and bwd agree by construction. On the
`train.py` route, `prebuilt_sample_to_reward` has signature `(mols, temperature)` — **no `x`**
— and is the reward source for bwd/dataset, bwd/prior, replay and bwd eval. It is structurally
incapable of computing a state-dependent Jacobian or wall, so on that route fwd and bwd would
score different rewards and TB could not close. The crystal avoids this by having its version
call `generator_energy` and recompute. The conformer's must do the same: read the stored
`torsion_state` off the graph and recompute J and the wall, pre-multiplied by T. Baking them
into `conformer_energy` is wrong — that field is baked at T=1 and divided by the sampling T,
so a change of measure would scale as 1/T.

**Plan of record.** Unconditional fixed-dimension MLP first, on `train_conformer.py`, through
`full` on a single molecule. Then migrate to `train.py`'s `Modeller` and swap the policy to a
graph model for the conditional route — which also disposes of the fixed-width problem, since
`d = 3N−6` is molecule-specific and `GFN` takes a scalar `dim`.

---

## 7. I/O, and the silent-failure inventory

**Always**: `z` + `edge_index` → `spec_from_graph(..., use_geometry=False)`.

**Per-atom `ctree_*`** already carries the tree. `ctree_state_col` — today a sparse `[N]`
index with −1 where frozen — becomes three columns or dense. Per-graph `torsion_state [k]`
becomes `state [d]`. The `size(0)` storage rule and the delta-encoding of atom references are
untouched.

Every item below is a present condition verified in code, not a hypothetical. Each is a place
where this change fails **silently**.

| # | what | consequence |
|---|---|---|
| 1 | `_linear_mask` returns all-False when `pos is None`, and `ConformerTorsions` builds its spec with `use_geometry=False` | `angle_is_linear` / `torsion_frame_is_linear` are all-False for **every molecule today** and are written into condition files where they read as measurements. §4's θ-singularity guard is a stored non-measurement. |
| 2 | `wrap_state` is applied unconditionally to every column in **four** independent copies (`train_conformer.py:219`, `conformer_data.py:135`, `build_conformer_buffer.py:32`, `build_prior_states.py:38`); `TerminalBuffers` wraps everything on intake | at `flex`/`full` a linear latent at 1.3 folds to −0.7 — the opposite corner of the box, with a plausible energy |
| 3 | `build_gfn` passes an explicit kwarg list off `args.model`, unlike `train.py`'s `**vars(...)` | a new `model:` key is ignored |
| 4 | `ConformerTorsions.__init__` ends in `**kwargs`; `preflight_config` is a retired-key blocklist only, with no positive schema | a yaml that says `level: full` on a run that is `torsion`, with a plausible loss curve |
| 5 | no artefact carries a force-field fingerprint; `build_terminal_buffers` loads `blob["modes"]`/`["states"]` with no check of smiles, k, epsilon or FF, and its missing-file path only **prints** before falling through to a uniform prior | step 3 silently invalidates every stored prior and buffer |
| 6 | `_state_columns`' 0/1 assertion does not fire at `dihedral` — each φ is still driven by one column | `state_to_phi`'s `π * state[graph, col]` would happily multiply an r-delta by π |
| 7 | `n_torsions` is set to `energy.data_ndim`; `state_dim`, `collate_conditions`' one-file-one-k rule and `rotatable_axes` all read it | at `full` it is 39, not a torsion count |
| 8 | `_batch` is a getter that mutates, and `energy()` reads `self._tree_cache[...]` directly, relying on `build_positions` having triggered it | reordering during §9 step 1 gives a stale tree or a `KeyError` |

Fixes, in order of cheapness: a `linearity_verified` flag that consumers must check (#1);
one shared mask-aware wrap helper plus a load-time assertion that linear columns lie inside
the box (#2); `level` as a required kwarg with no default and `**kwargs` deleted, with
`energy.level` and `energy.data_ndim` written to `wandb.run.summary` at startup (#3, #4);
an FF fingerprint written into every artefact and **raised on** at load, and the missing-file
print turned into a raise (#5).

**Conditionally**, the condition vector needs the stereo columns of §5. Enumerate
stereoisomers as distinct dataset entries — own identifier → own `mol_id` → own Z(c) — rather
than as a sampled sub-condition, since per-molecule centre counts break a uniform radix.

---

## 8. Benchmarks

`exact_references` and `boltzmann_reference` return `None` for k > 3; `brute_force_log_z`
refuses past `grid**k > 5e7`. At d=39 there is no quadrature ground truth, and **no full-DoF
problem will ever be small enough** — even 5 atoms is 9 DoF.

**Rotamer projection.** `measure()` full-DoF samples and compare the torsion marginal against
a k≤3 reference. Today's quadrature is `p(φ | r=r0, θ=θ0)`, a conditional slice, whereas the
sampler produces the marginal; build the reference as a **relaxed scan** with
`gradient_descent_optimization(..., freeze=('phi',))`.

| reference | cost | misses |
|---|---|---|
| rigid scan (today) | free | all (r,θ) relaxation — an O(1) term in the exponent |
| relaxed scan | k≤3 grid × opt | `|Hess|^{-½}`, and a φ-dependent `J(r*(φ), θ*(φ))` |
| relaxed + harmonic + `log_jacobian` at the optimum | + Hessian | anharmonicity |

The `J(r*(φ), θ*(φ))` factor is separate from vibrational entropy and free to add — evaluate
`log_jacobian` at the optimiser's output.

**The r/θ marginals are a companion check the projection is blind to**, but scope them
correctly: `p(r) ∝ exp(−k(r−r0)²/T)·r²` and `p(θ) ∝ exp(−k(θ−θ0)²/T)·sin θ` hold for the
**N−1 tree bonds and N−2 tree angles only**. The FF's angle term runs over the full redundant
graph set — a methyl carbon contributes 6 angles where the tree has 3 — and those extra terms
are a *bonded* coupling larger than the LJ coupling. Ring-closure bonds have no `r²` factor at
all, being in `ff.bond_index` but not `tree.bond_index`.

**Exact invariants.**

- **Enantiomeric pairs have exactly equal Z.** Exact only where the FF *and* the frozen
  centres are graph-typed — at `torsion`/`dihedral` each enantiomer carries its own measured
  r0/θ0 from its own embedding, so Z differs there for reasons unrelated to the model. It is
  also necessary but not diagnostic: a stereo-blind policy producing a racemate passes with a
  single wrong number for both. Pair it with a signed-χ statistic per condition.
- **build→measure round-trips exactly.** The first draft attributed this to
  `check_state_convention` ([conformer_data.py:615](../../energies/conformer_data.py:615)),
  which is wrong — that compares two *build* paths and never calls `measure`. The genuine
  round-trip test is on the mxtaltools side. **Nothing in `energy_sampling` calls `measure` on
  a sampler output**, which the whole projection harness needs. It is new code.

Also keep a k≤3 `torsion` problem as a regression harness, and replace the per-dim metric
surface: `torsion_latent_figure` builds one panel per state dim at `width=300*k` (39 panels,
~11,700 px), titles each "torsion j", and plots in degrees on [−180, 180] — so at `flex`/`full`
the bond-length panels are relabelled as degrees and clipped to a torsion range. `evaluate`
additionally emits `dist/torsion_{j}` per dim every 250 steps. Panel by **block** with correct
units, and emit one excursion scalar per block. `sde/sigma_over_halfperiod` divides by the φ
half-period and has no meaning for the linear blocks.

---

## 9. What gates the molecule ladder

- **FF table coverage.** 4 bond types, 6 angle types, 2 torsion centres — alkanes and
  alcohols. `_lookup` and the torsion typing **raise `KeyError`** on anything else, and
  `ConformerTorsions.__init__` evaluates the energy at construction, so after step 3 an
  untyped molecule fails at construction in the three data-prep scripts as well as in
  training. This is **not** independent of the other items, as the first draft claimed: after
  step 3 it gates step 3. `CCCCO` is fully covered, so `conformer_dev.yaml` survives.
- **No impropers.** sp2 planarity is held by propers plus sterics alone.
- **Rings.** Closure bonds ride the ordinary bond term evaluated from Cartesians, so closure
  is a stiff harmonic the sampler must learn to satisfy — probably the hardest sampling
  problem here. `RingBank` already exists on the prior side.

---

## 10. Order of work

1. **Free-mask plumbing, the level config, and the silent-failure fixes.** Generalise
   `build_positions`; wire `periodic_dims` into `GFN` (§3); clamp + wall for r/θ (§3); §7
   items 1–4. Gate, stated as three assertions rather than an unscoped "bitwise": (i)
   `build_positions(x)` bit-identical over 4096 random x at `torsion`; (ii) `rotatable_axes()`
   unchanged as an **ordered** list of (u,v) pairs; (iii) `ang_idx.tolist() == list(range(dim))`
   at `torsion`/`dihedral`. Plus: ±5 on an r and a θ latent gives a finite `log_reward`.
2. **Jacobian into the energy,** pre-multiplied by T, always on (§4). Gate: a synthetic-mask
   unit test against the BAT element **at two temperatures**, re-introducing the omission and
   requiring a failure. Note this gate is unobservable in a *run* until step 4, since the term
   is constant in x wherever r and θ are frozen.
3. **`ff_from_graph` swap, still at `torsion`.** Its real content, corrected: a rigid rotation
   about a bridge leaves every graph bond and graph angle invariant, so the bond+angle change
   is a pure additive constant — **the discriminating change is the proper-torsion term** plus,
   for rings, the closure term. It does **not** remove the embedding-seed dependence, since
   r0/θ0 still come from `measure()` on the ETKDG/MMFF conformer; and it *loses*
   `ff_from_reference`'s zero-at-reference property, so E gains a nonzero seed-dependent floor
   it does not have today. Swap the frozen r0/θ0 to typed values at the same time, and set φ0
   from the torsion phase γ (both `TORSION_PARAMS` phases are 0.0). Gate: two `seed` values give
   bit-identical `brute_force_log_z`. Requires §7 item 5 first.
4. **`flex` → `full`,** unconditional, single molecule, fixed-dimension MLP. Step 2's run-level
   gate lands here.
5. **Benchmarks:** relaxed scan, tree-scoped r/θ marginals, the `measure`-based projection
   harness, per-block metric surface.
6. **Runtime migration** (§6): `ConformerTorsions` through `train.py`'s `Modeller` — energy
   dispatch, `ConformerBuffer`, condition/prior loaders, and a `prebuilt_sample_to_reward` that
   recomputes from the stored state. Gate: the k≤3 harness reproduces `train_conformer`'s
   `eval/dlog_z` to within its own SE.
7. **Conditional, graph policy.** Stereo columns in the condition vector (§5); variable
   `d = 3N−6` handled by the graph model; chirality wall, prior repair, per-centre leakage
   series, k=1 handedness first.

Steps 1–4 are each verifiable against the current run. Chirality is deliberately **not** at
step 4: it first becomes reachable at `dihedral`, but its mechanism requires a condition, and
conditions arrive at step 6.

**Open:**

- Does `dihedral` warrant its own arm, given §2 — it is a related distribution, not a rung?
  Cheap if the mask lands cleanly at step 1.
- Does stereo ride scrambled or unscrambled? Only bites at k>1, and only after step 7.

Neither blocks steps 1–5.
