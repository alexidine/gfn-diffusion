# MLIP implementation validation — UMA and MACE

Status: **ACTIVE**
Scope: `mxtaltools/mlip_interfaces/` — our UMA (`uma_utils.py`) and MACE
(`AL_mace_utils.py`) energy routes, as production calls them.
Method: [`design/dependency_validation_protocol.md`](design/dependency_validation_protocol.md).
Evidence: F-053 (UMA), F-055 (MACE), F-054 (the coverage defect found alongside),
**F-056 (corrections to F-053's title and F-055's mechanism — read it alongside both)**,
F-057 (bias/noise/outlier decomposition on CPU).
Measured 2026-08-30, local RTX 5080 (GPU gates) and CPU (§5a).

---

## 1. Verdict

Our UMA route reproduces a stock fairchem ASE workflow to **2.3–3.8e-5 eV** on a
948 eV scale — *below* the same-stack nondeterminism control, replicated over
three runs — and our MACE route reproduces a stock `MACECalculator` to
**4.5–4.9e-3 eV** on a 598 eV scale, of which all but **3.7–4.3e-4 eV** is
attributable to a fractional round-trip in our own path perturbing positions by
1.9e-6 Å.

Stated per backend, because the two yardsticks are not the same: our UMA
construction is correct **below fairchem's own rerun nondeterminism**; our MACE
construction is correct to within **MACE's own scatter under a numerically-null
1.9e-6 Å position round-trip** (2.1-4.9e-3 eV), which exceeds our entire
disagreement with it. The MACE rerun control cannot serve as the referent because
MACE reruns itself almost exactly (control 0.0-7.6e-5 eV).

This is the first END-TO-END evidence of that kind for either backend. It is not the
first external evidence at all: `test_pbc_neighbours.py` already validated the
neighbour list itself against matscipy (via `mace.data.get_neighborhood`) on exact
edge sets. What was uncovered before these gates is the graph, batch construction,
shifts and model invocation *together*.

## 2. What was compared

| | our side | stock side |
|---|---|---|
| **UMA** | `compute_crystal_uma_on_mxt_batch` — vectorised fairchem batch, external neighbour list, `tf32=True`, `omc_forces`/`omc_stress` popped, `direct_forces/regress_forces/regress_stress=False` | `pretrained_mlip.load_predict_unit(ckpt, inference_settings='default')` → `FAIRChemCalculator(task_name='omc')` → `ase.Atoms`. No overrides, no task surgery. |
| **MACE** | `compute_crystal_mace_on_mxt_batch` — hoisted/device-built batch, batched neighbour list, `load_mace_model(...)` | `mace.calculators.MACECalculator(model_paths=<ckpt>, default_dtype='float32')` → `ase.Atoms`. Loads the checkpoint itself. |

Both stock sides are built **from the checkpoint path**, so the comparison
includes the load, not merely the forward. The `ase.Atoms` handed to them are
constructed from `unit_cell_pos` and `T_fc` directly — not through our fairchem
or MACE converters — so a converter bug cannot cancel.

Same checkpoints in both stacks: `esen_s.pt` (UMA),
`acr_112025_mh1_stagetwo.model` (MACE).

## 3. Why the comparison is fair

**UMA — the two stacks are handed different coordinates, deliberately.** Ours
passes *unwrapped* `unit_cell_pos` with an externally built neighbour list, which
is correct because the list is built for those coordinates.
`AtomicData.from_ase` calls `wrap_positions(pos, cell, pbc=pbc, eps=0)` before
building anything (fairchem-core 2.16.0), so the ASE route always sees atoms
inside the cell — the regime its internal `radius_graph_pbc_v2` documents itself
correct for. Wrapping by lattice vectors is a symmetry of the crystal, so the
physical energy is unchanged and the two must agree.

That difference is not incidental, it is the point. On this fixture
`unit_cell_pos` spans **1.5 cell widths** and wrapping displaces atoms by up to
**14.4 Å**. Had we wrapped by hand to make the sides match, the test would have
been unable to detect an F-047-class edge-dropping bug, which is worth 0.243 eV.

**MACE — cleaner, because no wrapping is involved.** `mace.data.get_neighborhood`
handles arbitrary unwrapped positions directly, which is why
`test_pbc_neighbours.py` can already demand exact edge-set equality against it on
real cells. Both stacks see the same coordinates and the comparison is direct.

**The cell convention is pinned separately, on CPU.** `T_fc` is a column-vector
operator, so `T_fc.T` rows are the lattice vectors ASE expects. Round-tripping
through ASE's own `cellpar` reproduces the stored `cell_lengths`/`cell_angles`
to ~1e-6. Without this, a convention error would read as a model disagreement —
or cancel on both sides and pass.

## 4. Results

**UMA** — 8 crystals, 948.4 eV mean per-crystal scale, three consecutive runs.

| condition | raw | per molecule | control (same stack, twice) |
|---|---|---|---|
| production (`tf32=True`) vs stock | 1.55–1.64e-2 eV | 0.119–0.141 mean / 0.389–0.396 max kJ/mol | — |
| matched precision (`tf32=False`) vs stock | **2.3–3.8e-5 eV** | ≤0.0009 kJ/mol | 4.6–7.6e-5 eV |

The matched-precision cross-stack delta sits **consistently below the control**:
two wholly independent stacks agree with each other better than one agrees with
itself across reruns. That is **4.0e-8 relative** (3.8e-5 / 948.4), and is the
ceiling of what this measurement can show.

> **Correction, 2026-08-30.** F-053's *title* and an earlier draft of this section
> said 1.25x and 8e-8. Those came from a single run taken BEFORE the bars were
> tightened (control 6.1e-5, cross-stack 7.6e-5) and are superseded by the three
> replicated runs tabulated above. F-053's body and table are correct; only its title
> carries the stale ordering. See F-056. The assertion value never moved:
> `bar = max(control*4, 1e-6*scale)` is pinned at 9.5e-4 eV by the scale term for any
> control below 2.4e-4, so both readings give an identical bar.

**MACE** — 4 crystals, 597.5 eV mean per-crystal scale.

| condition | raw | per molecule |
|---|---|---|
| ours vs stock, as production runs | 4.52–4.88e-3 eV | 0.22–0.24 kJ/mol |
| ours vs stock, positions matched | **3.7–4.3e-4 eV** | ≤0.02 kJ/mol |
| stock's own sensitivity to the same round-trip | 2.1–4.9e-3 eV | — |
| control (same stack, twice) | 0.0–7.6e-5 eV | — |

All three production-row measurements come from the test harness itself and cluster
tightly (4.52, 4.76, 4.88e-3). An earlier draft quoted 1.8e-3 at the low end; that
figure came from an ad-hoc diagnostic script that ran our side BEFORE constructing
`MACECalculator`, whose constructor calls `torch.set_default_dtype` — different
global state, not a different run of the same thing. Do not mix the two.

MACE repeats itself almost exactly, so its control is near zero; the bar is
therefore carried by the scale term and must not be written to depend on
`control` alone.

**Headroom, stated because this document's own method demands it.** Each gate
asserts `measured <= threshold`; how far the measurement sits toward that threshold is
the margin, and the three gates are not comparable without it:

| gate | threshold | measured | uses |
|---|---|---|---|
| UMA production (tf32) | 1.0 kJ/mol per molecule | 0.389-0.396 | **39-40%** |
| UMA matched precision (fp32) | 9.48e-4 eV | 2.3-3.8e-5 | **2.4-4.0%** |
| MACE | 5.98e-3 eV | 4.52-4.88e-3 | **76-82%** |

**The MACE gate sits within ~20% of firing.** A new fixture, checkpoint or upstream
version that moves its number by a quarter turns it red, and that failure would more
likely mean the bar was too tight than that a defect appeared. Compare the UMA
matched-precision gate, which would need its disagreement to grow 25x.

⚠ **The MACE bar has no defect-scale anchor.** `1e-5 * scale` was chosen by analogy
with the UMA gate before any MACE measurement existed. UMA's bar can be justified
against a known defect size (F-047, 0.243 eV); there is no MACE equivalent, so nothing
establishes that 5.98e-3 eV sits usefully below a real bug. Widening it to buy comfort
would be unjustified in the same way. Treat the margin as a known weakness of this
gate rather than a property of the code.

## 5. Residual attribution

**UMA: tf32, entirely.** The production-vs-stock gap is **>400×** the
matched-precision residual — a lower bound, since the denominator sits below the
control and is therefore unresolved — and closing the single `tf32` knob closes it.
The argument does not depend on the ratio.
`crystal_inference_settings` sets `tf32=True`; fairchem's default is `False`.
This is a deliberate speed choice, not a defect, and it independently
corroborates the ~0.1 kJ/mol tf32 reward-noise floor measured separately in
2026-08.

**MACE: our own fractional round-trip — the CAUSE is established, the MECHANISM is
not.** `AL_mace_utils.py` builds the neighbour list from the raw `unit_cell_pos`
(`batched_pbc_neighbour_list(pos_all, ...)`) and then, in
`compute_crystal_mace_on_mxt_batch`, overwrites `input_data['positions']` with a
fractional round trip (`T_cf` then `T_fc`) that moves atoms by max 1.9e-6 Å (mean
3.1e-7). So the model is *evaluated* at positions that differ slightly from the ones
its own graph was built for.

> **Correction, 2026-08-30.** An earlier version of this section, of F-055, and of the
> test docstring said the perturbation flips edges at MACE's neighbour cutoff. **That
> is false for our side by construction** — our edge set is fixed before the round
> trip is applied and cannot flip. How a 1.9e-6 Å shift produces a ~1e-3 eV change
> is therefore NOT established. See F-056.

The attribution is demonstrated, not asserted: feeding the **stock** calculator
the same round-tripped positions collapses the disagreement by 4–12×, and the
stock calculator's own sensitivity to that perturbation (2.1–4.9e-3 eV) is
**larger than our entire disagreement with it**. So the residual is a property of
MACE at its cutoff, not an error in our construction.
`test_the_residual_is_the_fractional_round_trip_not_our_code` pins both
directions of that argument.

**What remains is UNATTRIBUTED, and is recorded as such.** The attribution above
covers the *difference* between the production row and the position-matched row. The
position-matched residual itself — **3.7-4.3e-4 eV** — has no attribution. By this
document's own method (element 8: an attributed residual is a result, an unattributed
one is a tolerance), that remainder is currently a tolerance. Contributors not
separated: float32 accumulation order between a batched-4 evaluation and stock's
one-at-a-time, and the small edge-set difference between our raw-built graph and
stock's own build.

Whether the round-trip should exist at all is a separate question this
validation does not answer. It is not wrong — it is a no-op in exact arithmetic
— but it is not free either.

## 5a. Bias, noise and outliers, measured separately (CPU, n=1)

Every assertion in §4 bounds `max |delta|`, which detects outliers well and **cannot
distinguish a systematic bias from symmetric noise at all** — `.abs()` is applied
before any statistic. Those failure modes are not equally bad, so they are measured
apart. Full numbers in F-057.

CPU is the better instrument for this, not a fallback: reruns are bit-identical
(control exactly 0.0), there is no tf32, and there is no watchdog cap on n, so UMA
runs over all 95 buildable crystals instead of 8.

| | UMA (n=95) | MACE (n=21) |
|---|---|---|
| BIAS (signed mean) | **+0.00000** kJ/mol/molecule, bias/SE 0.03 | −0.039, bias/SE −0.91 (not significant) |
| NOISE (std) | 0.00028 | 0.020 excluding the top 3 |
| worst single point | 0.00092 | 0.036 on physical structures |

**No systematic bias in either route, and no landscape tilt** — every structural
correlation for UMA is under 0.12 over 95 crystals.

Why this ordering matters: a CONSTANT energy bias cancels exactly in a Boltzmann
target (`p ∝ exp(−E/T)` is invariant under `E → E + c`), so it costs only absolute
energies and log Z. A STRUCTURE-CORRELATED bias reweights basins and is the damaging
case. A LARGE SINGLE-POINT error invents or destroys one minimum. Symmetric noise well
under kT (2.5 kJ/mol) is a floor, not a defect.

**MACE's apparent tail is the energy-magnitude distribution, not an accuracy tail.**
Its worst raw figure (0.848 kJ/mol/molecule) comes from structures with *positive*
total energy — clashing configurations that cannot be sampled. On the 17 physical
crystals the median is **0.0015** and the worst **0.036**. The relative error is flat
at ~3e-7 (float32 territory), so a large absolute number appears exactly where the
energy is already absurd. ⚠ A Pearson `corr(delta, energy)` of −0.923 on this data was
entirely one leverage point: Spearman is −0.298 and dropping that crystal gives −0.006.


## 6. Scope and limits

- **Physical cells only.** fairchem's internal graph truncates at
  `max_neighbors=300`; physical CSD cells reach ~141 and never hit it, degenerate
  early-training cells reach ~2710 and would. Comparing on trash cells would
  measure fairchem's neighbour cap rather than our code.
- **MACE runs at 4 crystals, and this is not a tuning knob.** The acridine model
  OOMs above ~5, and the forward is ~425 ms at 2 graphs and near-fixed in batch
  size, which puts 16 graphs across the 2 s Windows WDDM watchdog. Two earlier
  sweeps BSOD'd the box that way with the card verified idle.
- **MACE's element table is restricted**, so the fixture keeps only crystals it
  covers.
- **This validates the CRYSTAL leg.** The two-leg lattice energy is not covered;
  on MACE specifically, `compute_lattice_mace` is known to differ from stored
  prior values by a constant per-molecule offset (the atomic-E0 sum counted in
  the crystal leg and omitted from the gas leg), which cancels in differences and
  therefore hides.
- **The symmetry expansion's ORDERING was a common-mode hole; it is now CLOSED
  separately.** Both stacks consume `batch.unit_cell_pos` and re-derive the same
  image-major tiling — production via `_tiled_gather_index`, the fixtures via
  `np.tile`. Two independent re-derivations of the same *assumption* are not
  independent evidence: a layout inversion in `build_unit_cell` would pair every
  position with the wrong element identically on both sides, cancel exactly, and leave
  both gates green on a chemically nonsensical structure. Neither L1 gate can see it,
  and it is the case element 10 of the method doc exists to name.
  `tests/test_unit_cell_layout.py` closes it geometrically rather than by assumption:
  every symmetry image is an isometry, so each contiguous block of `unit_cell_pos`
  must carry the asymmetric unit's intra-block distance matrix entry for entry.
  CPU-only, no model, with a negative control that re-indexes a cell atom-major and
  requires rejection. **4 passed — the layout is image-major as assumed.** Residual
  limit: elementwise distance equality pins the ordering only up to an automorphism of
  that matrix, so two chemically equivalent atoms could still swap undetected.
- **The MACE cell-convention pin runs on CPU only after a fixture fix.** It
  originally took the `crystals` fixture, which depends on the model and therefore on
  the GPU and checkpoint; it now uses `any_crystals`, which needs neither. Without
  that it would have skipped in CI while §3 claimed the convention was pinned on CPU.
- **This is not a check against DFT or any published number.** It establishes
  that we drive each backend the way its authors do. Whether the backend is right
  about chemistry is a different question. Level 2 is wired
  (`$UMA_REFERENCE_JSON`) and unpopulated by choice — a reference we generated
  ourselves would be Level 1 again.
- **One fixture, one GPU, one checkpoint per backend.** Nothing here is a
  cluster or A100 claim.

## 7. How to run it

```bash
cd C:/Users/mikem/Projects/mxt_gfn && PYTHONPATH="C:\Users\mikem\Projects\mxt_gfn\mxtaltools;C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion" MACE_CHECKPOINT='D:\crystal_datasets\acr_112025_mh1_stagetwo.model' UMA_CHECKPOINT='D:\crystal_datasets\esen_s.pt' python -m pytest mxtaltools/tests/test_uma_vs_stock_fairchem.py mxtaltools/tests/test_mace_vs_stock.py -q -rs -s --uma-checkpoint "D:/crystal_datasets/esen_s.pt"
```

Green is **12 passed, 1 skipped** — the skip is Level 2, which has no data.

**Read the skips, not the pass count.** These reasons mean the gate did *not*
run: `pass --uma-checkpoint ...` / `set MACE_CHECKPOINT ...` (no model),
`GPU pre-flight refused: no GPU` (no card, or a collected module disabled CUDA —
see F-054), `GPU is in use -- refusing to co-tenant` (busy card). Only the Level
2 skip is expected.

Four CPU tests in these files run without a GPU or checkpoint and are safe in CI:
three in the UMA file (density guard, cell convention, fixture-is-unwrapped) and
the MACE cell-convention pin. Verified as `4 passed, 9 skipped` with neither
checkpoint set and no GPU touched.

## 8. What would invalidate this

- A checkpoint change. Both results are per-checkpoint; `uma-s-1p1.pt` and
  `uma-m-1p1.pt` do not even load through our interface (`KeyError: 'oc20_forces'`).
- A fairchem or mace-torch version bump, particularly anything touching
  `AtomicData.from_ase`'s wrapping or `get_neighborhood`. The fairness argument in
  §3 is version-specific (fairchem-core 2.16.0, mace-torch 0.3.15).
- Removing or changing the fractional round-trip in `AL_mace_utils.py`, which
  would move the MACE numbers and should *improve* them.
- Any change to `crystal_inference_settings`, which would move the UMA
  production row while leaving the matched-precision row alone.
- Extending either gate to degenerate cells, which would silently start measuring
  neighbour caps.
