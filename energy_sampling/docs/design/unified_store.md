# One store for the NF route

*Status: plan, from the owner discussion of 2026-10-28. Not policy. Built 2026-10-28 behind `buffers.store.enabled`
(off in mk_dev): `row_store.py`, described as built in [row-store](../wiki/row-store.md), which supersedes this page
wherever they differ; not yet run at production parameters. Scope: `model.kind nf`.
The SDE route keeps its three buffers until it is retired (an SDE row's weight needs backward rollouts; at most a
k ≈ 5 refresh of protected and high-priority rows).*

## The idea

One store holds every row the run trains on or keeps: the pre-noised data (anchors and their children), the
policy's own draws, and any generated draws (noise kicks, condition swaps). Rows carry their reward and their
importance weight under the current model. Badly fit rows are drawn more for TB; MLE never sees the priority.
It replaces, on the NF route, the prior, replay and anchor buffers, harvest shards, prior top-ups, the anchor
surprise screen and confirm rollouts, and the anchor health gate.

## A row

Flat, about 100 B: condition id, chart state (FD row or latent), stored energy terms and log R, log q at the last
refresh and the step of that refresh, a tag (`anchor`, `child`, `onpolicy`, `generated`), a protected flag, and
the held-out flag (held-out rows are never drawn for training). A crystal is rebuilt from the condition's template
and the state when drawn. The flat format is the first thing to build.

## Draws

- **MLE** draws the data tags (`anchor`, `child`) uniformly. Priority never feeds MLE: forward KL learns whatever
  distribution it is fed, and a store of past draws drawn by priority would teach q its own history.
- **TB** draws all non-held-out rows by priority p ∝ |r|^κ, with a floor relative to the median and a uniform share,
  and importance-sampling weights 1/p in the loss (the prioritized-experience-replay recipe: Schaul et al. 2016,
  priority on |TD error| with IS correction under a Huber loss). The Huber β alone sets how much an outlier counts;
  the priority with IS correction only changes how often a row is seen. For TB the draw distribution reweights the
  objective and does not move its fixed point in the realizable case; the store's composition (set by admission) is
  itself part of that weighting.
- **The residual** is r = log R − log q − L(c), with L(c) the **training level**: whichever level the TB loss uses
  (fwd EMA, pooled, or the learned head). Priority then ranks rows by the residual the loss sees.
- **Fresh draws**: un-perturbed forward draws every 5 to 20 steps (empirical), for the level.

## J_B

J_B(c) is the mean log w over the condition's **reference rows** (anchors proper, fixed for the run, uniform per
row). It is a fit index on the reference set, not an estimate of log Z (anchors are not drawn in proportion to the
target). A pooled level uses it as a training lever, pulling mass toward the reference set to raise J_F;
Δ = J_B − J_F is logged whatever level trains. What counts as an anchor is open (minima, noised, Hessian).

## Refresh

Every row's log q is recomputed at every eval, and every row a training draw touches is refreshed for free.
Measured 2026-10-28 on the local 16 GB GPU at mk_dev's default flow size (one transformer block, 12 rows): about
250k rows/s, so 1M rows in about 4 s, unconditional or with a vector condition. Atom-token conditioning is
unmeasured (the encoder runs once per molecule).

## Admission and eviction

- An on-policy or eval draw is admitted when |r| is at or above a bar, or when it sets a new per-condition energy
  minimum. A record becomes protected.
- Each condition holds at most M unprotected rows. When full, the row evicted is drawn with probability
  ∝ 1/(|r| + ε), so well-fit rows have a half-life instead of certain eviction.
- Protected rows (data tags and records) are never evicted.
- Known risk: a mode found during the run (the common case on conditions with no reference rows, which is where
  conditional generalisation lands) can be evicted once well fit and then lost if the policy drifts. A retention rule
  on a calibrated FD distance is being worked on separately; until then, watch occupation (below).

## Monitoring (each eval, no offline jobs)

- Store size by tag; rows per condition (min, median, max; number of conditions at the cap).
- Admissions and evictions per eval; median |r| of admitted and of evicted rows.
- Age of evicted rows, in evals since admission.
- Per condition, protected rows with |r| above the bar: a rising count is the policy leaving a mode it held.

## Open

FD retention metric (other tab); anchor definition; values for M, the bar, κ and the floor; whether the IS
correction is annealed; saving the store across runs with the energy identity (as pre-noised builds are);
the first build on MIPCAS FD, compared with the current stage 2 at equal energy calls.

## FD distance in retention (proposal, 2026-10-28)

*Status: a proposal awaiting the FD noising calibration (other tab; nothing of it on disk at writing) and the owner.
Nothing here is built, and it changes no rule above. It uses per-row neighbour counts only, never clustering.
Distance: `GluedFDMetric` in `eval/fd_analysis/fd_distance.py` (uncommitted, 2026-10-28).*

### What the metric gives a store

- d(A, B), in Å, between two Z' = 1 crystals of one molecule in one space group. It is the minimum, over Γ+,
  lattice shifts in {−1, 0, 1}³ and (sg 1-15) near-minimal bases, of √(D_cell² + D_cent² + (r_g θ)²). There is one
  metric per condition. r_g is taken from the molecule; `molecule_info` reads it only from a .pt path, so a helper
  that takes the condition's molecule is needed.
- Inputs are the canonical (cell, t, R, h) only. No crystal is built.
  - FD-chart rows are canonical by construction (`FDCoordinates.to_description`).
  - Latent rows need `canon.canonicalize` on the CPU: 0.07 ms/row (NEHZOR) to 0.91 ms/row (MIPCAS), measured.
- d ignores h. In Sohncke groups, neighbours must also match on h.
- d ignores molecular point symmetry. Near-planar molecules need the twin variant (matched mirror RMSD: MIPCAS
  0.064 Å, NEHZOR 0.056 Å). Without it, mirror twins sit 2-7 Å apart.
- One-sided use. `pairs(A, ia, Balt, ib, cut=)` expands only B into its glued alternatives and only B needs
  Q_W R. So:
  - the query rows (few per eval) take the B side;
  - the store takes the A side and holds just (cell, t, R, h): 18 f32 + 1 int8, about 73 B/row.
  - The two one-sided values differ by up to about 0.03 Å on MIPCAS.
- Exact pruning, from the code: D_cell ≤ d, and each of |Δa|, |Δb|, |Δc| ≤ D_cell.

**Evidence, all on relaxed endpoints or prior rows, none on thermal rows:**
- Glued nearest-neighbour distances among 5000 distinct states within 2 kT per campaign
  (`fd_identity_oct28/summary.md`):
  - median 0.009 Å (MIPCAS) and 0.011 Å (NEHZOR);
  - same-registry-basin share ≥ 0.93 below 0.05 Å, falling to 0.37 (MIPCAS) and 0.72 (NEHZOR) in [0.2, 0.3) Å.
- The only COMPACK cuts are for the unglued metric: MIPCAS 0.915 Å, NEHZOR 0.695 Å.
- In three QM9 prior molecules, 159, 209 and 371 of about 1000 rows lie within 0.01 Å of another row.

### The radius is a mode's thermal width, not an identity cut

A store row of a mode is a thermal draw, not a relaxed endpoint. Take two independent draws from an isotropic
Gaussian mode in D free coordinates, with RMS radius ρ:

| radius d | P(pair within d), D = 10 | P(pair within d), D = 12 |
|---|---|---|
| ρ/2 | 5e-4 | 1e-4 |
| ρ | 0.11 | 0.08 |
| 2ρ | 0.97 | 0.98 |

*Probability that two independent draws from one isotropic Gaussian mode lie within d of each other, computed as
P(χ²_D ≤ D d² / (2ρ²)). D = 12 is a triclinic Z' = 1 crystal; D = 10 is monoclinic. Compare the rows against each
other: the share falls by about three orders of magnitude between d = ρ and d = ρ/2.*

Consequences:
- A radius at the identity scale of relaxed endpoints (0.01-0.1 Å) would call almost every thermal row isolated.
  Every rule below would then protect or admit noise: bounded, but useless.
- A radius r is usable only when ρ_T ≲ r < d_sep, where:
  - ρ_T is the mode's RMS FD radius at the run temperature;
  - d_sep is the distance below which two rows almost always share a basin.
- Where ρ_T ≥ d_sep, no fixed radius separates that system's modes, and the rules stay off for it.

### What the noising calibration must deliver

Per condition, or as a rule in r_g and T:

1. **Kick map.** The FD distance of a child from its parent as a function of the latent kick radius 10^h
   (`build_noised_prior`: log_lo to log_hi). This puts children, generated kicks and r on one scale. Needed by
   every use.
2. **ρ_T.** The thermal radius at the run temperature.
   - A random-direction kick of radius ρ costs about ½ρ² × the arithmetic-mean curvature.
   - Thermal spread is set by the soft directions (the harmonic mean).
   - So "the kick radius whose median excess is D/2 kT" underestimates ρ_T. The calibration should state which
     quantity it reports.
   - Needed by uses 2, 3 and 5.
3. **d_sep.** P(same basin | d) on thermal, unrelaxed rows, glued, with the twin where it applies. d_sep is the
   distance below which that probability is ≥ 0.95. Needed by uses 2, 3 and 5.
4. **d_id.** A COMPACK-calibrated identity cut for the glued metric (none exists yet). Needed by use 4.
5. **De-duplication radius.** Nothing beyond a numerical radius well below the smallest kick (10^log_lo mapped to
   Å). Needed by use 6.

### Uses, ranked by value per cost

1. **Monitor (build first).**
   - One neighbour query, no rule change.
   - It measures directly the known risk (a locality left empty) and the hole before it (a mode never admitted).
   - It needs no calibrated radius: it reports at three fixed radii.
   - It decides whether uses 2-3 are worth building.
2. **Last representative protected (needs the owner).**
   - Mechanism:
     - After `RowStore._evict`, walk this call's evicted rows from the least-deserving eviction (lowest Gumbel key)
       upward.
     - Restore row j, as protected flag `rep`, iff no surviving row of its condition lies within r and the condition
       holds fewer than K_rep reps.
     - At each admission, a rep with an admitted row within r loses `rep` and rejoins the pool.
   - Bounds:
     - Reps ≤ K_rep per condition.
     - Restored rows leave the pool, so the per-condition cap and max_rows still hold.
     - Restoration is one pass over the evicted list; release is one query of reps against admitted rows.
     - Nothing waits.
   - Invariant: every row evicted in a call has a surviving row within r, unless its condition's rep slots were full
     (counted).
   - Effect: a rep of an abandoned mode loses log q, so |r| and its TB priority rise and TB pulls q back. One row
     per locality is a weak signal; "≥ k survivors within r" is the extension.
3. **Novelty admission (needs the owner).**
   - Admit a candidate with no stored row within r, whatever its |r|, as unprotected (cap bounded).
   - Without it, a mode found and fitted below the bar in one step never enters the store, and use 2 has nothing
     to protect.
   - It queries every candidate, not only the admitted ones.
4. **Record identity (needs the owner; waits on records_kept).**
   - A new record within d_id of a protected record of its condition replaces it instead of adding a row.
   - Cost: at most one record per condition per call × records_kept pairs.
   - records_kept slots then hold distinct basins rather than successive refinements of one.
5. **Crowding-weighted eviction (not proposed now).** π_j ∝ n_j(r)/(|r_j| + ε). Against it:
   - it needs a count for every pool row: about n × budget pairs per eval, roughly 2e9 at 1e6 rows, or incremental
     counts that drift once subsampled;
   - it flattens composition toward uniform-in-FD, which reweights TB's objective;
   - it is close to the density-based retention the owner rejected.
   Use 2 gets its protective effect at count 0 without per-row counts.
6. **Admission de-duplication (not proposed).**
   - On-policy rows from a continuous sampler rarely coincide.
   - Data tags are never de-duplicated, since they are the MLE target.
   - The QM9 near-duplicates are a prior-build observation, not a store rule.

### The shared neighbour query (uses 1-5)

**Store side**
- Per-row columns (cell, t, R, h), written at admission:
  - FD chart: from x;
  - latent: `canonicalize` at admission;
  - seeds: once at init, or written by the prior build.
- Per condition, an integer key ⌊(a, b, c)/w⌋, with w the largest radius; rows are sorted by key after each
  admission.
- The pre-screen is exact: a store row within d of a query has every cell length within d, so it lies in one of the
  27 boxes around the query's alternative. The alternatives' a, b, c are what is keyed.

**Query side**
- The query's glued alternatives (identity only for sg ≥ 16), plus the twin's alternatives where the twin applies.
- Candidates are the rows in the 27 boxes, capped at a uniform subsample of B_q (for example 2048).
- count_r = candidates within r (equal h in Sohncke groups), via `pairs(..., cut=max r)`.

**Exactness and termination**
- The count is exact when the candidate list fits in B_q.
- Over budget it is a subsample, so it can be wrong only toward "isolated". In that case one extra row is
  retained or admitted, and the over-budget share is reported.
- Termination: chunked loops over Q queries × ≤ B_q candidates; nothing waits.

**Cost.** GPU rates are 0.2-3 µs per pair with a cut, inferred from wall times on a shared card. Alternatives per
query (K_alt) are unmeasured.

| store rows per condition | exact brute force, Q = 1000 queries (s per eval) | box index + B_q = 2048, worst case (s per eval, × K_alt) | store columns (MB) |
|---|---|---|---|
| 1e5 | 20-300 | 0.4-6 | 7 |
| 1e6 | 200-3000 | 0.4-6 | 73 |

*Estimated cost of one neighbour query pass per eval, against the measured 4 s per 1M-row log q refresh. The brute
force is Q × n pairs; the indexed form is Q × B_q pairs, and only queries in dense regions reach B_q. These are
estimates from the rates above, not measurements.*

Further costs:
- The key sort is milliseconds at 1e6 rows.
- Seed canonicalisation for latent rows: 70-910 s of CPU per 1e6 rows, once (estimated from the per-row rates).

### What the monitor reports, and what to read from it before committing

At init, once:
- `store/fd_child_parent_{p10,p50,p90}` (Å): d from a child to its parent anchor, over seed children whose parent
  is loaded. This is the run's own noise scale.
- A table by distance band (Å) for a sample of seed children: each child's nearest other seed row, and the share
  that share its parent. This is a free P(same parent | d) on unrelaxed rows. Same parent is not the same as same
  basin: two anchors can share a basin.

Each eval, at radii (0.1, 0.3, 1.0) Å, fixed in config until the calibration:
- `store/fd_evicted_isolated@r` (rows): evicted rows with no surviving row of their condition within r. This is
  the risk itself.
- `store/fd_rejected_novel@r` (rows): candidates refused admission with no stored row within r. This is the hole
  before it.
- `store/fd_admitted_novel@r` (rows).
- `store/fd_queried` (rows), `store/fd_over_budget` (share of queries), `store/fd_seconds` (s).

How to read it:
- `fd_evicted_isolated` at the radius nearest ρ_T, flat at about 0 while `protected_above_bar` stays flat: use 2
  is not needed on that system.
- `fd_evicted_isolated` positive and rising late in a run: build use 2.
- `fd_rejected_novel` positive: use 3.
- Reading it needs a short real run (AGENTS.md loop rule).

### Prerequisites and the first build

**Prerequisites:**
- `GluedFDMetric` importable outside `eval/` (MXT design item B5, `crystal_fd.metrics`), with no D:/ constants.
- MXT `crystal_fd` pushed, so the cluster can import it.
- An r_g helper that takes a molecule batch.
- The glued metric's exactness (an unchecked working assumption) checked on the rows that enter a store.

**First build:** use 1 on the MIPCAS FD-chart run that the design already names. There, x is canonical and
admission adds no canonicalisation. The latent route adds the per-row canonicalisation at admission.
