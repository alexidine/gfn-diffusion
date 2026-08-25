# VarGrad convergence vs (huber beta, condition_block_m, repeats) — theory note

2026-08-25. Prompted by: qm9split (98 molecules) converged decently under
var_conditioning; the full qm9 problem "seemed rather noisy". Which of the three
knobs is the plausible cause, and what would each one's fingerprint be?
Grounded in the shipping estimator (`gflownet_losses.py`:
`condition_grouped_empirical_z` / `vg_lb`) and two small Monte Carlos
(numbers quoted below; scripts trivially rebuildable from the formulas).

## 0. The estimator, precisely

Per row i in condition-group g (size m):

    r_i    = log R + log P_B − log P_F          (the log-ratio)
    r̄_g    = mean_i r_i                          (Jensen centre; group empirical log Z)
    d_i    = r_i − r̄_g
    loss_i = β · huber_β(d_i)                    (= ½d_i² for |d|<β, else β|d|−½β²)

Groups are formed by SCATTER over condition_id, pooling K-repeats tiles with
any other same-condition rows in the batch. Singleton groups contribute exactly
zero loss and are masked out of emp_z. The per-row influence on the policy is
ψ_β(d_i) = clip(d_i, ±β). The fixed point (per condition, all r_i equal) is
independent of β and of the centre flavour.

**The single most important structural fact: m is EMERGENT, not configured.**
m ≈ (rows per batch) / (distinct conditions drawn) on the scatter path, floored
by `repeats` (fwd) or `condition_block_m` (bwd).

## 1. What changed between qm9split and the full problem: m collapsed

- qm9split: 98 conditions, fwd rows ~1000, repeats 2 → each condition drawn ~5
  times per batch → scatter pooling gives **m ≈ 10**.
- qm9c100k: 5265 conditions, same batch → ~500 distinct conditions per batch,
  collisions rare (~24 collided pairs/batch, P≈0.095 per condition) →
  **m = 2 almost everywhere** (the bare repeats tile).

Consequences of m = 2, each with a different visibility:

a) **Per-group estimator noise.** The group loss is a variance estimate with
   rel. sd √(2/(m−1)): 1.41 at m=2 vs 0.47 at m=10 (MC-exact). This is the
   3× per-group noise jump the library-size change bought.

b) **But the BATCH trace barely notices** (Gaussian case): at fixed row budget
   the batch-mean loss has rel. sd 0.063 (m=2) vs 0.047 (m=25) — the χ²
   degrees of freedom total ≈ B−G either way. So if the observed noise is in
   the pooled `fwd/vg_lb` trace, m alone is NOT a sufficient explanation in
   the Gaussian regime; the tails and the per-condition quantities are.

c) **Per-condition Z estimates are the quantity batch size cannot rescue —
   with the OWNER'S CORRECTION (2026-08-25) that Z(c) is a PURE SIDECAR on
   this route**: emp_z trains Z(c) FROM the group estimates and nothing feeds
   Z(c) back into the policy loss (fwd tb = 0 in var_conditioning, so
   tb_z_source is inert; VarGrad's centre replaces Z entirely). So the
   se = σ_c/√m target noise (0.71σ_c at m=2 vs 0.32σ_c at m=10) degrades the
   READOUT and the two indirect consumers — the weighted condition-sampling
   steering (which reads tracker fit-error/z-gap stats) and eval — not the
   training gradient itself. The part that DOES enter training is (f) below:
   the same √m error lives inside the VarGrad residuals as the group-centre
   jitter. The persistent tracker is the intended smoother across visits, but:

d) **Visit cadence also collapsed.** 500 conditions/batch over 5265 → each
   condition refreshes every ~10.5 batches, and `condition_log_z.min_visits
   20` is first reached ~210 batches in — so for the whole opening act of
   var_conditioning, `tb_z_source: persistent` is masked for most conditions
   and the estimates it eventually serves are built from few, stale, m=2
   observations. On qm9split the same numbers were ~5 visits/batch and
   min_visits inside ~20 batches.

e) **The centre-cancellation subtlety is MOOT at m=2** (documented in the
   code): ψ is odd, d_2 = −d_1, so Σψ_β(d_i) = 0 identically — the
   skew-weighted MLE leakage that detach_center exists to kill only exists on
   groups ≥3. The large-library regime is therefore *cleaner* in this one
   respect; the noise is honest estimator variance, not a hidden force.

f) **Per-group centre error is a per-batch fake-Z jitter**: r̄_g misses the
   true per-condition centre by ~σ_c/√m, and within the group this error acts
   coherently — a random per-condition push, new every visit. At revisit
   period ~10 batches this looks in the traces like slow per-condition
   wander, i.e. exactly "rather noisy convergence" rather than divergence.

## 2. Huber β: three separate roles, and their m=2 interactions

β appears in three places that pull different directions:

1. **Force ceiling / basin width** (the confirmed instability mechanism from
   qm9anchor: var(log w) IS the loss, and the restoring force saturates at β
   per row). Conditions with σ_c ≫ β live in the L1 regime: restoring force
   β·sign(d), independent of displacement — convergence of that condition
   proceeds at a fixed rate ∝ β and an excursion meets no growing resistance.
   With 5k diverse molecules the σ_c DISTRIBUTION is wide, so a fixed β
   splits the library: quadratic-regime conditions converge exponentially,
   saturated ones crawl linearly at rate β. A long noisy-looking tail of
   slow conditions is the predicted signature — and it presents as "the run
   is noisy" when the pooled metric mixes the two populations.

2. **Loss-scale readout**: in the saturated regime the logged loss is
   β·E|d| ≈ β·σ·√(2/π) — β linearly rescales the trace and its noise.
   Comparing runs across β values by loss level or wiggle amplitude is a
   trap; compare σ estimates (loss/β in the saturated regime) instead.

3. **Outlier robustness — largely VOID at m=2.** Huber's point is to stop one
   catastrophic row dragging the group consensus; in a pair there is no
   consensus — the outlier IS half the centre, both residuals are ±half the
   outlier, and clipping merely caps the pair's force. MC with 2% 30-nat
   contamination: huber cuts the trace noise 0.265→0.152 at m=2 — real but
   modest, and mostly the force cap, not consensus protection. Robustness in
   the m=2 regime has to come from somewhere else (the trimmed tracker mean,
   `trim_frac`, across visits).

Net on β for qm9c100k: RAISING β widens the quadratic basin and speeds the
high-σ_c tail linearly (until σ_c < β, then no further gain), at the cost of
letting single catastrophic pairs push harder (m=2 caps at β per row by
construction, so the exposure is bounded and linear in β). The qm9anchor
b005/b010/b020 wave was precisely a β ladder; its lesson (raise β) should
transfer, with the caveat that the loss floor will rise ∝β and must not be
misread as worse convergence.

## 3. condition_block_m (bwd)

Fixes m directly for the backward branch (the draw takes M distinct terminals
per sampled condition; K-repeats would share a terminal and give TBC instead).
Raising M at fixed bwd batch:

- per-group noise ↓ √(2/(M−1)): M 2→4 is a 1.7× cut, 2→8 is 2.4×;
- per-condition target se ↓ 1/√M — helps the SAME quantity the fwd side
  cannot fix cheaply, and bwd rows are buffer draws: NO new energy calls, so
  block_m is the CHEAPEST group-size lever on the board;
- coverage cost: conditions/batch = B/M — revisit period stretches ∝M
  (tracker staleness, retention risk on the unvisited tail);
- buffer support: needs ≥M distinct stored terminals per condition; qm9c100k
  seeds ~10/condition — M ≤ 4 comfortable, 8 marginal at seed time;
- off-policy caveat stands: the group centre is a biased Z estimate off-policy
  (why bwd emp_z is asserted off); block_m improves the policy-loss estimator,
  not a Z estimate.

## 4. repeats (fwd)

The fwd tile is the only group builder when collisions are rare — but each
extra tile costs a full rollout + energy evaluation. Two distinct sweeps that
must not be conflated (the "never sweep repeats alone" memory, now with
arithmetic):

- **repeats ↑ at fixed rows** (K 2→4, B constant): groups halve in count,
  double in size — per-group noise ↓1.7×, but G ↓2× takes back √2 of it on
  the batch gradient; per-condition target se ↓1.4×; coverage halves. Net:
  weak on the trace, moderate on targets, bad for coverage.
- **repeats ↑ at fixed conditions-per-batch** (K 2→4, B doubled): per-group
  ↓1.7×, G unchanged, coverage unchanged — clean win, paid in 2× rollout +
  energy compute.

So repeats is the expensive way to buy what block_m buys cheaply, with the one
advantage that it acts on the ON-POLICY branch (unbiased centre for the
POLICY loss; per the sidecar correction, Z(c) quality is a separate matter).

## 5. Suspects OUTSIDE the three knobs (do not skip these)

- **Measurement noise masquerading as training noise.** At eval_num_samples
  10000 over 5265 conditions, every per-condition eval statistic is a
  ~2-sample estimate; the `_worst`/`within`/fraction families inherit n_c
  bias (documented). Discriminator: if pooled traces are smooth while
  per-condition families jump, the run is converging and the METRICS are
  noisy.
- **Z(c) head capacity/conditioning**: 5k embeddings vs 98 — the flow network
  now interpolates a much denser condition manifold with noisier targets;
  its fit error adds condition-correlated noise the tracker can't remove.
- **weighted_condition_sampling / z_gap steering** driven by noisy
  per-condition stats recycles estimator noise into sampling churn.
- **level_gap tether saturation** (enters clamped — constant force on the way
  in; documented) adds a constant offset early, not noise per se.

## 5b. THE NON-AVERAGING NOISE: huber skew rectification (owner's suspicion,
confirmed by MC)

Any nonlinear influence function turns ASYMMETRIC zero-mean noise into a
persistent drift, and elj log-ratios are asymmetric by construction: rewards
are bounded above per condition while catastrophic energies produce a heavy
LOW tail in r. The huberized force E[ψ_β(r − c)] then balances not at the mean
(the quadratic fixed point) but at a pseudo-median displaced toward the short
tail — by an amount set by the tail mass beyond β, which no amount of batch or
time averaging removes.

MC (5% low-tail at 25-nat scale on a 2-nat core; mean −1.00, median −0.12):

    beta      huber balance point   offset vs mean
      2            −0.14                +0.86
      5            −0.24                +0.76
     10            −0.43                +0.57
     20            −0.71                +0.29
     50            −0.98                +0.02

So at the shipped β=10, a condition with this tail carries a ~0.6-nat
permanent displacement of its VarGrad centre; conditions differ in tail mass,
so the displacement differs per condition — a frozen, non-averaging,
condition-dependent distortion that presents as irreducible "noise" in every
per-condition family and as a persistent force on the policy along the tail
gradient. It compounds with small β exactly as suspected: the bias saturates
at the full mean-median gap once β falls under the core σ.

m=2 nuance: within a pair, ψ is odd so the group force still balances — the
rectification enters through WHICH pairs occur (a tail row drags its
partner's centre by up to β), i.e. as a visit-frequency-weighted drift rather
than a within-batch offset. Same direction, same β dependence.

Remedies in order of invasiveness: raise β toward the tail scale (removes the
bias, re-exposes the force ceiling to tails — the two-sided trade the
qm9anchor ladder was walking); trim/median the group centre (kills the bias at
any β; the tracker already trims across visits via trim_frac — the LIVE group
centre does not); treat the tail at the source (reward_range soft clip
placement relative to β).

## 6. Discriminating measurements (cheap, in preference order)

0. **Per-condition residual skewness** (from the same batch stats VarGrad
   already computes): the rectification story predicts bias ∝ per-condition
   tail mass beyond β; a skewness histogram is its direct witness.
1. **Log the emergent group-size histogram** per batch (fwd + bwd). One
   number decides whether the m-collapse story applies: expect mean m ≈ 2.05
   on qm9c100k vs ~10 on qm9split.
2. **Pooled vs per-condition trace comparison** on the same run (suspect 5a).
3. **β ladder re-read**: plot loss/β (σ readout) rather than loss; if the
   b005/b010/b020 conclusion was drawn on raw loss levels it under-credits
   large β.
4. **block_m 2→4 single-key arm** (cheapest intervention, targets the same
   noise as repeats without energy cost). Watch relative_under/retention for
   the coverage price.
5. **repeats 2→4 WITH batch doubled** (the clean version of the sweep) if the
   fwd side specifically is implicated.

## 7. Bottom line

The three knobs are not symmetric suspects. On the full qm9 problem the
emergent group size collapsed from ~10 to 2 (a structural consequence of
5265 conditions vs 98 — nobody set it), which (i) triples per-group estimator
noise, (ii) doubles per-condition Z-target noise that batch averaging cannot
remove, (iii) stretches the tracker's rescue cadence 10×, and (iv) voids
huber's consensus-robustness role. β then determines how the wide σ_c
distribution splits into fast (quadratic) and crawling (saturated) conditions
— a second, independent contributor that LOOKS like noise in pooled metrics.
block_m and repeats are the two levers over the same m; block_m is the cheap
one (no energy), repeats the on-policy one (pay compute, keep the centre
unbiased). And some fraction of the observed "noise" is plausibly the
per-condition metric families being 2-sample estimates at this library size —
worth ruling out before spending compute on any knob.
