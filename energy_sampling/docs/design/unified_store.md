# One store for the NF route

*Status: plan, from the owner discussion of 2026-10-28. Not policy; nothing here is built. Scope: `model.kind nf`.
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
