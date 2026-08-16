# Dead latent rows — session handoff

Written 2026-08-12 at the end of the D33 session. This is an **index**, not a new home
for facts: every claim below lives canonically somewhere else and is cited. Per
[`PROTOCOL.md`](../PROTOCOL.md) the four types have four homes, and a handoff note is
none of them — so this file holds only pointers, status and the to-do list.

Canonical homes:
- **Decision + argument** — [`decisions.md`](../decisions.md) **D33**, and
  [`fundamental_domain.md`](fundamental_domain.md) for the FD/free-axis reasoning.
- **Evidence** — [`findings.md`](../findings.md) **F-007, F-008, F-009, F-009b, F-010, F-010b, F-023**.
- **Tests** — `test_dead_latent_rows.py` (18 groups), `test_dead_latent_rows_deep.py`,
  `test_latent_gaussian.py` (51 checks, the analytic one).
- **Batteries** — `configs/gauss_aug12/` (analytic, 10 arms, the correctness one) and
  `configs/deadrow_aug12/` (physical energies, smoke).

---

## What the change is, in one paragraph

Latent rows that `enforce_crystal_system` overwrites with a constant carry no
information, but `latent_params()` round-trips them to a canonical value, so every
prior/buffer row sits at that constant while the energy is flat across the whole box.
`bwd` therefore trains `P_F` toward a delta while `fwd` gets no gradient — 2 of 12 dims
trained against inconsistent targets on monoclinic, 3 of 12 on orthorhombic. The fix
holds those rows out of the diffusion, entirely inside `models/gfn.py`, so states stay
full width and no consumer's index semantics move. Free aunit axes join the same
mechanism at Z'=1 via `canonicalize_free_axes`.

## Status

| item | state |
|---|---|
| D33 implementation (gfn.py, train.py, checkpointing.py, train_conformer.py) | **committed** |
| `models/dead_latent_rows.py` | **committed** |
| `canonicalize_free_axes` (mxtaltools) | **committed** |
| `angs = [False] * self.dim` latent-bug fix | uncommitted |
| `test_dead_latent_rows_deep.py` | uncommitted |
| `latent_gaussian` toy energy | uncommitted, **CPU-certified against closed form** (F-023) |
| jacobian + reduction structurally zero for latent energies | uncommitted |
| `test_latent_gaussian.py` (51 checks) | uncommitted |
| `configs/gauss_aug12/` (10 arms + 5 priors on disk) | uncommitted, **never run** |
| `configs/deadrow_aug12/` | uncommitted, **never run** |
| findings/decisions edits | uncommitted |

All four test suites pass. **No training run has ever exercised this change** — every
number below is CPU, analytic, or single-batch.

## The load-bearing evidence

- **Unbiased for log Z** — IS estimator (consistent regardless of training) on untrained
  policies, 3 seeds: recovers the closed-form constant to ≤0.018 nats. F-010.
- **Bitwise exact** — restricted log-prob equals an independent live-only
  reimplementation to `0.00e+00` on four dead sets, differing 0.29–1.02 nats from
  full-width. F-010.
- **`Var(log w)` 6.30 → 5.33 → 4.77** for `n_dead` 0 → 2 → 3 — the additive-variance
  argument that decided D33 against pinning, measured. F-010.
- **Energy-invariant free-axis canonicaliser** — ≤1.2e-06 relative on 40 real structures
  per space group, idempotent. F-010b.
- **Lands on a closed-form log Z, both arms, 5 space groups** — rows held to ≤0.0001
  nats at `Var(log w)` ~5e-12 (the proposal *is* the target); rows live to ≤0.005. And
  sg 4 / sg 1 reach sg 19's value through `canonicalize_free_axes` rather than
  `enforce_crystal_system`, which is the first evidence for the free-axis half. F-023.

## Three methodological lessons (all cost real time; all now in memory)

1. **Use the IS estimator for log-Z correctness, never a trained comparison.** A trained
   pair showed err −0.022 vs −0.213 with the gap *growing* (0.052 → 0.166 → 0.191) — a
   textbook bias signature. It collapsed to 0.005 by 12000 steps. `fwd_propagate` draws
   `batch × dim` noise per step, so a wider model consumes a different RNG stream and is
   effectively a different seed. **A growing gap over a short budget is not bias.**
   → `feedback_use_is_estimator_for_logz_correctness`
2. **Assert gauge invariance on the ENERGY, not the RDF.** An RDF comparison at fixed
   cutoff carries a boundary-counting term worth ~0.05 on a physically identical pair:
   one sg-4 structure in 40 showed 0.054 at `rdf_cutoff` 6, **exactly 0** at 8, 0.032 at
   10, with the energy invariant throughout. Cutoff and supercell must be lined up.
   F-010b.
3. **A validation assert written for one feature caught a latent bug in another.** The
   width check in `_finalize_dim_partition` exposed `angs = [False] * 12` hardcoded,
   which would have fed a `data_ndim`-18 toy only its first 12 dims to the policy,
   silently. F-009b.

## The `latent_gaussian` toy — design and the sharp prediction

`is_crystal=True` (crystal latent layout, dead rows, periodic angles, jacobian all live)
+ `latent_energy=True` (analytic reward, no packing/pressure/mol_energy). Reuses
`latent_harmonic_en` verbatim; registered `False` in `COMPUTES_REQUIRE_CLUSTER` and
absent from `COMPUTES_REQUIRE_UNIT_CELL`, so **no cluster and no unit cell are built** —
the molecules need not be real crystals. `reduction_energy` is *structurally* zero for
latent-scored problems, because P-1's reduced region is a thin set with no
zero-reduction ball wide enough for a gaussian (best of 4000 draws: 0 at centre, 0.105
at the edge of a ±0.15 ball).

Setting `c = 0` on dead rows makes the energy live-dims-only for free — their
contribution is exactly `((0−0)/w)² = 0`.

**MEASURED** (F-023; `spec.py` owns the arithmetic, `test_latent_gaussian.py` the check):

    rows HELD:  log Z = (n_live/2)·log(2πT) + n_live·log w
    rows LIVE:  log Z = ⟨above⟩ + n_dead·log(2 + √(π/k))

| sg | dead | n_live | HELD | LIVE | Δ |
|---|---|---|---|---|---|
| 2 | — | 12 | −16.6038 | −16.6038 | **0** (control) |
| 14 | (3,5) | 10 | −13.8365 | −11.1810 | **+2.6555** |
| 19 | (3,4,5) | 9 | −12.4528 | −8.4696 | **+3.9832** |
| 4 | (3,5,7) | 9 | −12.4528 | −8.4696 | **+3.9832** — via the *free-axis* path |
| 1 | (6,7,8) | 9 | −12.4528 | −8.4696 | **+3.9832** — free axes only |

Δ is the *fictitious volume*: unheld, those rows are still discarded by the crystal
build, so the gaussian is blind to them, but `bounding_energy` reads `raw_latents`, so
their marginal is `exp(−k·relu(|x|−1)²)`.

⚠ **`Δ = n_dead·log 2` was this session's first prediction and it is REFUTED**, not
merely imprecise. The wall is soft, so the volume per row is `2 + √(π/k)` — at k=1 that
is 3.77 against a box of 2, i.e. the leakage is nearly as large as the box. The log-2
model is wrong by +0.63/row at k=1, and the error grows as the wall softens (+2.43 at
k=0.5). Measured across a 20× sweep of `bounding_coeff`, which is a dial no space group
can reach. **Anyone reading an older copy of this table should discard it.**

## Verdict, 2026-08-12 — D33 IS CONFIRMED

**Rows held land on the closed-form log Z on four space groups, err ±0.002 nats**, and
two of them by a code path nothing else in either battery reaches:

| arm | dead rows | mechanism | err |
|---|---|---|---|
| c_sg19_on | (3,4,5) | `enforce_crystal_system` | +0.0004 |
| d_sg4_on | (3,5,**7**) | + `canonicalize_free_axes` | −0.0003 |
| e_sg1_on | (**6,7,8**) | free axes ONLY | +0.0019 |
| b_sg14_on | (3,5) | `enforce_crystal_system` | +0.0010 |

**Rows live also reach their own (larger) analytic value, from both sides** — so
`Δ = n_dead·log(2 + √(π/k))` holds in training as well as on CPU. What the live arm
lacks is *stability*, not accuracy. Evidence: F-023 (the closed form), F-026 (held vs
live), F-027 (the corrections).

**This line of investigation is CLOSED.** Correction ON has an ordinary
noise/finite-convergence floor with nothing to explain; correction OFF is the retired
method, whose floor is the documented cost D33 removes. Further characterisation of the
rows-live dynamics cannot change what we do, so the drafted LR sweep was dropped. Two
claims made along the way were **refuted by our own follow-up** and are corrected in
F-027: the floor is not a finite-T effect, and its variance is stationary, not growing.

**Physical energies, 2026-08-12.** elj / nehzor / SG 14 / Z'=1, rows (3,5) held: MLE
descends cleanly for 11000 steps (`bwd/mle` 8.6 -> -11.1), `Reduced Valid Fraction` 1.0
throughout, packing 0.74-0.76, and it transitions to fused TB at step 11250 with
`replay/loss` falling monotonically after. Not a convergence result (that is days away)
and it cannot certify a log Z, but D33 is not disturbing a physical run.

Two operational facts came out of it, both worth keeping:
- **Stage 1 will not exit on its own here.** Exit terms are ANDed and `gates/mle_flat`
  needs `mle_gate_rate_hi < 0.05`, which oscillates 0.09-0.19 on this problem because
  `bwd/mle` is still descending linearly at step 10000. `bwd/tbc` (0.87 vs 2.0) and
  `wass_debiased` (0.012 vs 0.015) were satisfied from ~step 8000. It cleared only after a
  resume gave it room, at 11070.
- **The LR servo cannot act during a post-transition warmup.** `on_calibration` returns
  early while the envelope is ramping, so the first two post-transition calibrations were
  discarded (`cal_status` = warmup). The third measured alpha* 2.83 against a target of 4
  and cut `peak_scale` to 0.841 (= (2.83/4)^0.5, `eta_down` 0.5). Read `raycal/*` with
  care: **it HOLDS the previous calibration's values between calibrations**, so a flat
  alpha* series is one measurement repeated, not a standing verdict — misreading that
  produced a wrong conclusion here before the cut arrived.

**Z'>1, 2026-08-12: now trained, and it immediately found F-028.** sg 9 Z'=2 elj, rows
(3,5) held, 16 of 18 dims flowing: it trains, `Reduced Valid Fraction` 0.9941, packing
0.785. It also revealed that the runtime probe had been silently absent at Z'>1 (F-028) —
now fixed and regression-tested.

⚠ **Z'=2 at batch 1000 with `eval_num_samples` 10000 OOMs on its FIRST eval, reproducibly,
on an idle card.** It asks for 2.68 GiB while already holding ~12 GiB reserved against its
own 14.33 GiB `cuda_memory_fraction` cap, so it trips its own ceiling — not a co-tenant.
`handle_train_epoch_error` halves the batch to 500 and the run completes, so it is
survivable, but **any Z'=2 arm written at batch 1000 silently becomes a batch-500 arm at
its first eval**, and with `grow_batch_size: false` it never climbs back. Budget Z'=2 at
batch 500, or cut `eval_num_samples`. Note 4.63 of the ~12 GiB reserved was
reserved-but-UNALLOCATED — fragmentation, permanent here because `expandable_segments` is
unsupported on this platform. First attributed to a collision with a concurrent GPU test
suite; the clear-card reproduction refutes that.

LIMITATION of this smoke test: `zp_ordering_energy` is folded into `bounding_energy`
rather than logged separately, so it shows the Z'>1 ordering term is FINITE (total 0.277),
not that it orders correctly.

## To-do, in priority order

The battery has been run (see the verdict above). What remains:

1. **Short elj run** as the second-order physical check — the only outstanding
   *experiment*. Physical energies have never converged perfectly, so a clean elj arm
   proves *less* than it looks like it does and a messy one does not implicate D33. It is
   a smoke test, never a certificate.
2. **Commit.** Nothing from this work is committed (see the status table).

Deliberately NOT doing: an LR sweep on the rows-live arm (see F-027 — it characterises a
retired configuration); the `a_sg2` GPU control (`test_dead_latent_rows` proves the
empty-dead-set case BITWISE on CPU, which is strictly stronger); a T=50 arm (~14 GB per
arm at batch 1000, so a pair cannot fit this card and cutting batch would confound it).

## Deferred, with reasons

- **`periodic_centroids` + free axes.** Pinned FALSE across `gauss_aug12` and NOT tested
  on. It is a real interaction, not an omission: a free axis that is also `auv == 1` (26
  space groups) becomes angular when wrapping is on, so its period is exactly 2 and its
  fictitious volume is `log 2` rather than `log(2 + √(π/k))`. Turning it on invalidates
  the arms D/E rows-live predictions and nothing in the battery would notice. Worth its
  own arm eventually, with the prediction rederived for a wrapped dim.
- **Z'>1 free axes** — needs centroid+delta reparameterisation. **Do not make aunit dims
  dead rows at Z'>1.** Already compliant: `free_centroid_rows` returns `()` above Z'=1.

  ⚠ **THIS GATE IS LOAD-BEARING, not just a deferral.** `compute_zp_order_penalty`
  indexes raw latents by ABSOLUTE position — `raw_latents[:, 6:6 + 3*k]`
  (`molecular_crystal.py:596`) — and it is the one term that does. Three facts
  currently interlock to keep it off every dead row:
    1. it is gated on `max_z_prime > 1`;
    2. at Z'>1 the only dead rows are crystal-system angles (3,4,5), all **below 6**;
    3. free axes, which live at **>=6**, are gated off above Z'=1.
  Ungating free axes at Z'>1 without also fixing this penalty would make it read a
  pinned constant as if it were an emitted centroid, silently. This is the concrete
  reason "states stay full width" was chosen over an energy-boundary scatter, and the
  reason that choice must survive the reparameterisation work.
- **`a=b` diagonal degeneracy** (tetragonal/hexagonal/cubic) — needs reparameterisation,
  not row deletion.
- **`dead_latent_values` unwired** — so any system whose canonical γ is 2π/3 asserts at
  startup by design. **MEASURED 2026-08-12: that is TRIGONAL *and* HEXAGONAL, not just
  hexagonal** as earlier notes said. γ=120° reads back as latent **0.5556**, not 0, on
  sg 147 (trigonal) as well as 168/176 (hexagonal); monoclinic and orthorhombic controls
  read back exactly 0.0. Verified by driving `latent_to_cell_params` -> `latent_params`,
  so the assert genuinely fires rather than being assumed to. Note sg 168 also carries a
  free axis (dead rows `(3,4,5,8)`), so wiring `dead_latent_values` is not the only thing
  those groups need.
- **Multi-SG conditional** — needs per-sample log-prob masking. Note `sg_conditioning`
  spanning two crystal systems is now a hard startup error, intentionally.
- **Ray probe interaction** — dropped. The probe re-calibrates in whatever configuration
  it runs in, and the correct configuration is rows-held; there is no "two answers".

## Known non-interactions (checked, not assumed)

The conditional pipeline does not interact: condition embeddings and `c` act on the
condition tensor not the latent; `r2` reads energies; `condition_block_m`'s grouped
VarGrad takes variance of `logw`, which excludes dead dims uniformly across conditions;
`Z(c)` becomes the live-dim log Z per condition with no relative shift.

## Pre-existing limitations confirmed, NOT caused here

Both reproduce with `dead=None`: DPLR + float64 raises in `fwd_propagate` (a Float `V`
meets a Double noise), and the states buffer is always float32 because
`init_traj_tensors` allocates at the default dtype.

## Review caveat

The adversarial review that found the `TorsionGFN` break was **partial** — 20 of 42
verifier agents died on a spend limit, so their findings were hand-triaged rather than
adjudicated. Five real defects were fixed (conformer break, silent stale-checkpoint
revert, `dead_latent_values` ordering, diagnostic dilution, two error types). The
remainder were judged cosmetic; that judgement is unverified.
