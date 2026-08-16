# Sizing the prior buffer

Argument. Why `max_size` / `mean_lifetime` / `churn_batch_ref` / `init_fraction`
should be what they are. Verified against the working tree 2026-08-13.

The short version: three of the four numbers are choosable on ordinary grounds
(resolution, timescale, memory). `max_size` is not — it is a **switch between two
different learning problems**, because the prior buffer's only routine eviction
channel is residual-dependent and fires *only* at capacity. Retune the numbers
without fixing that and you are tuning the location of a discontinuity.

---

## 1. Two degrees of freedom written as four numbers

`mean_lifetime` (L) and `churn_batch_ref` (R) reach the dynamics only as their
ratio ([train.py:4691](../train.py:4691)):

    n_churn   = max(1000, (bwd_steps_since_last_call / L) * R)
    n_to_add  = min(eval_num_samples, n_churn)

so the admission rate is

    c = R / L = 1000 / 100 = 10 rows per bwd step

and everything else follows from `c` and `N = max_size`:

| quantity | formula | mipcas (N_c≈1) | qm9a198 (N_c=198) |
|---|---|---|---|
| admission rate `c` | `R/L` | 10 / step | 10 / step |
| residence / nominal turnover `τ` | `N/c` | 25,000 steps | 25,000 steps |
| fill time from `init_fraction f` | `(1−f)·N/c` | 18,750 steps | 18,750 steps |
| reuse (draws per row over its life) | `B/c` | 100 @ B=1000 | 100 @ B=1000 |
| rows per condition | `N/N_c` | 250,000 | **1,263** |
| turnovers per run | `epochs/τ` | 4 @ 100k, 40 @ 1M | 4 |

The fill-time formula is not a model, it is the measured behaviour: on
`9inim617` the rebuild fires at equilibration `on_enter` (~8.2k) and
`prior_buffer_length` reached 250k at step 27,000 = 8,250 + 18,750.

**`mean_lifetime` is not a lifetime and `churn_batch_ref` is not a batch size.**
The config comment already says so. One key — `churn_rows_per_step: 10` — carries
everything the pair carries. The pair is a fossil of the pre-`churn_batch_ref`
scheme, where `n_churn ∝ batch_size` made `L` *literally* the reuse count and
invariant to batch growth. Decoupling churn from batch (correctly — turnover
should not triple because the batch controller grew) moved that coupling into
reuse instead: reuse is now `B/c`, so under `prod0810`'s growing batch
(`max_batch_size: 50000`) it rides from 100 up to 5,000 while τ stays at 25,000.
§2 is why that does not matter.

---

## 2. Reuse is not a constraint on this buffer — `MECHANISM`

The replay buffer's reuse figure is load-bearing because `get_traj_replay`
replays *stored* trajectories: draw a row twice and the model sees the identical
computation twice. That is memorisation, and it is why replay sits at
`B/churn_rate` = 12.5.

The prior buffer stores **terminal states only** — no `traj` (§1 table,
`module_buffers.md`). `bwd_train_step` hands the latent to
`get_gfn_backward_loss` with `trajectories=None`, so every draw re-rolls a fresh
stochastic backward path to that endpoint. 100 draws of one row are 100 *distinct*
trajectories to the same terminal, which is precisely the averaging the backward
TB estimator wants.

So: **do not import replay's reuse target.** Reuse is not a reason to touch `c`,
and the batch-growth coupling noted above is not a defect to chase. What the
endpoint set does cap is the branch's floor (`bwd` bottoms out at `−H(buffer)`),
and that is a *resolution* question — §5, not a reuse question.

---

## 3. `max_size` is a regime switch — `MECHANISM`

[`manage_prior_buffer`](../train.py:4677) only ever calls `purge_lowest` inside
`if space_needed > 0` ([train.py:4724](../train.py:4724)). Below capacity there is
**no eviction at all**. At capacity, eviction switches on — and it is selective:

- [`get_elig_drop_count`](../buffer.py:809) marks eligible the rows whose
  `ema_loss` (stored as `|resid|`) sits **at or below the 25th percentile** of
  visited rows.
- [`purge_lowest`](../buffer.py:733) then draws from that set with
  `p ∝ softmax(−loss)` — sharply concentrated on the very lowest.

Eviction removes **the rows the policy already fits best.** That is hard-example
mining, and it exists only in the full regime. The two regimes are two different
targets:

| | below capacity | at capacity |
|---|---|---|
| composition set by | admission gate alone | gate **and** survival |
| survival depends on residual? | n/a (nothing leaves) | **yes** |
| target is | i.i.d. gated draws from the frozen prior model | gated draws *minus what the policy has learned* |
| `bwd/tb_err` can converge? | yes | **no — by construction** |

Three consequences, all derivable:

1. **The post-capacity plateau is a fixed point, not a stall.** The loop is: fit a
   row → it falls into the bottom quartile → it is evicted → it is replaced by
   something unfit. The only rest state is "the policy fits nothing in the
   buffer," and the level of that rest state is set by where the eviction quantile
   stops moving — i.e. by `_EVICT_QUANTILE`, not by the problem. Falsifiable:
   raise `_EVICT_QUANTILE` and the plateau should move. (`CONJECTURE` — argued,
   not measured.)
2. **Three quarters of the buffer has no eviction channel.** There is no age
   hazard and no random purge on the prior buffer. A row above the 25th residual
   percentile leaves only via the reach trigger's `purge_worst` (energy-based, ≤
   `reach_topup_size` = 1000, and only when `reach < 0.75`) or a stage flush.
   Membership rotates as the policy learns, but anything *persistently*
   unfittable — a numerical outlier, an early bad-energy admission — is permanent.
   The nominal τ = 25,000 describes nothing: at steady state the bottom quartile
   recycles about every 6,000 steps (5,000 evictions/window against a ~59k
   eligible pool) and the rest does not recycle.
3. **This is the exact `p_survive` the replay buffer retired.** `module_buffers.md`
   §3 / B10 killed `floor`/`stalled` because
   `μ_buf ∝ Q_admit · p_admit · p_survive` and the draw's IS weight divides by the
   draw only, so a residual-dependent `p_survive` re-enters the target uncorrected
   and stale. The prior buffer's draw is *uniform, with no IS weight at all*. The
   argument applies with more force here, not less.

---

## 4. What the drift measurement says — `OBSERVED`

**Scope:** `9inim617` (`prod0810_mipcas_elj`), T = 2.5 kJ/mol, elj, fused stage at
`Bwd Frac` 0.90, steps 27k–44k, n=1.

`prior_buffer_mean_energy` was flat at −290.95 for the 19k steps before the cap,
then fell from the step the buffer filled and did not stop: −312.7 by 44k, ≈0.9
kJ/mol per 1k steps = **0.36 nats/1k**. Over the same window the `bwd` TB level
offset went 3.05 → 3.56 nats (+0.03/1k) and `bwd/tb_err` sat flat at ~5.0 for 16k
steps, while the `fwd` offset — static target — fell 2.78 → 1.04 (−0.11/1k).

Read as a rate budget: the target moved ~5.5 nats, the branch tracked ~5.5 nats
and closed none of its own residual. **Drift consumed the entire measured
improvement rate of the branch carrying 90% of the loss weight.**

The flat-then-falling shape is the tell. Under admission alone the mean was
static; the fall starts at the cap because the surviving rows are the
high-residual ones, and on this problem high residual means the under-weighted
low-energy tail (that is what the anchor buffer's surprise criterion *is*). Same
step, both effects, one cause.

**This is why the answer is not a smaller `c`.** To get target motion to a third
of the `fwd` branch's demonstrated closure rate you would need τ ≈ 250,000 steps —
2.5 runs long. A self-reinforcing loop does not have a rate you can tune out of
it. The fix has to be structural.

---

## 5. Recommendation

**5.1 Make survival residual-independent, and move priority to the draw.**

`module_buffers.md` §1 already states the design principle — *intake, draw and
eviction are three separate axes* — and the replay buffer already obeys it. The
prior buffer has never been brought into line. Concretely: pass
`loss_floor`-style neutrality to survival (uniform-random among eligible, or an
age hazard mirroring replay's), and if hard-example emphasis is wanted, put it in
the **draw**, where `weighted_bwd_sampling` / `_loss_weights` already exists and
is IS-correctable.

Once survival is residual-independent the buffer relaxes to the gate's own
stationary distribution, drift becomes self-limiting (bounded by `Emin(c)`'s
ratchet rather than by feedback), `max_size` is a memory bound again, and
`bwd/tb_err` becomes readable as a convergence metric. Everything below assumes
this; without it, the numbers below just move the discontinuity.

**5.2 `init_fraction: 1.0`.**

0.25 is a leftover from the era when a full buffer *stalled* churn. That bug is
fixed — `loss_floor=+inf` ([train.py:4711](../train.py:4711)) — and the comment
there says so outright: *"This is what makes `prior_buffer.init_fraction: 1.0`
safe."* Keeping 0.25 now buys nothing but an 18,750-step unannounced regime
change in the middle of every run, which is a measurement hazard on top of a
dynamics one: every metric read before ~27k comes from a different system than
every metric read after.

Cost is bounded and safe: `rebuild_prior_by_churn` chunks at `min_size` (10,000)
and `fwd_eval_sampling` chunks *those* at `batch_size`
([train.py:5866](../train.py:5866)), so there is no oversized rollout and no
`on_enter` OOM exposure. The price is ~250 batch-1000 prior rollouts in one
blocking call at the transition — ~25 eval passes' worth. Cheap on elj, minutes
on UMA, once.

Order matters: do 5.1 first. With residual-dependent survival still in place,
`init_fraction: 1.0` puts the run in the hard-mining regime from step 0 — more
honest (one regime throughout) but not better.

**5.3 `max_size` from conditions, not from memory.**

`N/N_c` is the only thing `max_size` uniquely controls once 5.1 lands. The floor
is set by per-condition mode structure, and we have a measurement of that: the
QM9 anchor search found **~49.3 distinct minima from 50 random inits per
molecule, with 0 of 198 molecules collapsing** — the basin count is unsaturated
at 50, so per-condition populations in the high hundreds are a floor, not a
luxury.

- **qm9a198 (N_c = 198): keep 250,000.** 1,263 rows/condition is defensible and
  thin. Do not cut it. Note this is right for a completely different reason than
  the one it was picked for.
- **mipcas (N_c ≈ 1): 250,000 is ~5× larger than it needs to be.** The only cost
  is slower tracking of the gate and ~250 MB of GPU buffer (`buffer_device: cuda`,
  ~1 KB/graph). 50,000 gives τ = 5,000 and 20 turnovers in a 100k run. Weak
  recommendation — there is no correctness argument, only responsiveness.
- **General rule: scale `max_size` with `N_c`** at ~1,000–1,500 rows/condition,
  floor ~50,000. Read `z_cal/n_conditions` for the actual `N_c` — it is
  `n_molecules × n_sg × n_zp`, not the dataset row count.

**5.4 `c = 10 rows/step`: keep, and write it as one key.**

With drift self-limiting, τ = N/c is just the buffer's mixing time toward the
gate's stationary distribution, and it should be short relative to the run and
long relative to a bwd fit. τ = 25,000 in a 100k-step run (4 mixing times) is a
reasonable place to sit and there is no measurement that argues for moving it.
Collapse `mean_lifetime` + `churn_batch_ref` into `churn_rows_per_step: 10`.

---

## 6. Live traps

All `MECHANISM`, all verified in the working tree.

**`eval_num_samples` is the churn ceiling.** `n_to_add = min(eval_num_samples,
n_churn)` ([train.py:4695](../train.py:4695)). The churn *rate* is `c` only while
`eval_period · c ≤ eval_num_samples`, i.e. `eval_period ≤ 1000` at the current
10,000. mk_dev (250) and cluster (500) are safe with 2× margin; raise
`eval_period` past 1,000 and the buffer's turnover time silently doubles with no
buffer key changed. Wants an explicit `churn_max`.

**`loss_min` defaults to 1.0 nat and `manage_prior_buffer` never sets it.**
[`purge_lowest`](../buffer.py:766) force-purges every valid row with
`|resid| < loss_min`, and the forced set is **not capped by `num_to_purge`** —
`remaining = max(num_to_purge − forced, 0)` only shrinks the stochastic
remainder. On elj (residuals plateau 24–27) it never fires. On any problem or
stage where the backward branch converges below 1 nat, the buffer empties in one
call. This is the same absolute-nats-bar failure the `loss_floor=+inf` comment
was written about, still live in the sibling parameter.

**`purge_lowest`'s `temperature: 1.0` is also an absolute nats scale.** The
eviction draw is `softmax(−loss/T)` over the eligible quartile. On elj, spanning
~5 nats, that is a ~150× preference for the single best-fit row; on a toy
spanning 0.4 nats it is nearly uniform. Eviction selectivity is therefore
problem-scale-dependent by accident.

**`min_size` does two unrelated jobs and is not a minimum size.** Per-cycle chunk
bound in `rebuild_prior_by_churn` ([train.py:4880](../train.py:4880)) and draw
size in `grow_prior_buffer` ([train.py:5817](../train.py:5817)). Already
`module_buffers.md` B5 — but B5's companion claim that *"`init_fraction` is
inert"* is now **stale**: it is read at [train.py:4865](../train.py:4865).

**`purge_worst` ranks on RAW energy, not excess.** The reach trigger's
`top_up_prior_from_anchors(..., purge_worst=True)` sorts
`self.prior_buffer.y` descending ([train.py:4946](../train.py:4946)) — pooled
absolute energy. The trigger that *fires* it correctly uses excess above
`Emin(c)` ([train.py:4760](../train.py:4760)). On a multi-condition run absolute
elj scales with molecule size, so the purge systematically strips the
small-molecule conditions regardless of how good their structures are for their
own condition. Third instance of the pooled-statistic-on-a-conditional-problem
pattern (after the r2 gate and the eviction quantile). One-line fix: rank on
`y − Emin(c)`.

**The prior dataset does not size the buffer.** `init_prior_buffer_seed` takes
`min(len(prior_dataset), max_size)`. On qm9a198 that is 9,843 of 250,000 — the
buffer starts 3.9% full and the first 62,500 rows (or 250,000 under 5.2) come
from the frozen prior model through the gate, not from the dataset. Nothing is
wrong with that; it just means `max_size` is unbounded by any file on disk.

---

## 7. Gate-staleness eviction, and a per-condition anchor floor — **as built**

Neither of these is a sizing change. `max_size` and the churn rate are already
where they should be — large, slowly churned. These change **what leaves** and
**what is guaranteed to be in**. Both are residual-independent, so neither
depends on settling whether survival-based hard-mining is a good idea.

Two config keys, both read through `getattr` with a `0.0` default, so every
config that predates them is bit-identical to before:

| key | 0.0 means | mk_dev |
|---|---|---|
| `buffers.prior_buffer.expire_max_frac` | gate-staleness channel off | 0.1 |
| `buffers.prior_buffer.anchor_floor_frac` | pooled shortfall backfill only | 0.1 |

**Known soft edge:** the anchor floor can request more than `n_to_add`, since
`want_c = max(shortfall_c, quota_c)` sums above the budget when the prior's yield
is high. So the buffer can overshoot `max_size` by up to
`anchor_floor_frac · n_to_add` — ~500 rows on 250k at the settings above, 0.2%,
absorbed by the next call's headroom arithmetic. `max_size` was already a soft
ceiling (`purge_lowest` "may purge fewer than asked"); this widens the slack
slightly rather than introducing it.

### 7.1 Gate-staleness eviction

**Rule: membership is admission.** A row belongs in the buffer iff it would be
admitted *today* — `y < Emin(c) + ramp_floor`. `Emin(c)` ratchets down over the
run, so rows admitted under an older, looser gate stop clearing the current one.
That is the *only* sense in which a prior-buffer row is stale: the prior model is
a frozen snapshot, so an old draw is still a valid draw from it.

The quantity is already computed — the reach trigger measures
`excess = y − Emin(c)` against `ramp_floor` ([train.py:4760](../train.py:4760)).
Today it is used as an **alarm** (fires a 1000-row emergency top-up when the 90th
percentile of excess covers less than 75% of the margin). This promotes it to the
**membership rule**.

In `manage_prior_buffer`, before the headroom computation:

    prior_condition_id = self.prior_buffer.batch.condition_id.detach().cpu()
    energy_floor = self._condition_energy_floor(prior_condition_id)
    if energy_floor is not None:
        margin = self._ramp_params()[0]
        excess = self.prior_buffer.y.cpu() - energy_floor
        stale = torch.isfinite(energy_floor) & (excess >= margin)
        # cap, worst-excess-first -- see below
        ...
        self.prior_churn['expired'] += n_purged

Load-bearing details:

- **Runs before headroom**, so freed rows become headroom and the same call's
  churn cycle refills them. Gate-staleness becomes the primary eviction channel;
  the residual quantile degrades to an overflow handler.
- **`+inf` floors survive.** An unobserved condition gets `Emin(c) = +inf`
  ([train.py:2571](../train.py:2571)); `excess` is `−inf` and the row is kept.
  Same convention admission uses.
- **Must be capped.** A record-breaking anchor can drop `Emin(c)` by a lot in one
  step, making a large fraction of that condition's rows stale at once. Cap at
  `expire_max_frac` of the buffer per call (suggest 0.1), purging worst-excess
  first, and **log the uncapped count when it truncates** — the
  `rebuild_prior_by_churn` cycle cap is the precedent. This is exactly the
  failure mode `loss_min`'s uncapped force-purge still has (§6).
- **Telemetry:** `prior_buffer_expired` / `_expired_frac`, mirroring
  `replay_buffer_expired_delta`.
- **The reach trigger does not become dead.** With this in place `excess < margin`
  by construction, so `reach > 0` always — but it still fires below
  `reach_threshold: 0.75`, i.e. whenever the 90th-percentile excess exceeds
  0.25·`ramp_floor`. Its `purge_worst` bug (§6) therefore stays live and should be
  fixed in the same pass.

### 7.2 Per-condition anchor floor

**The gap is that the anchor mix is pooled, twice.** `_prior_churn_cycle`
computes one `shortfall = budget − admitted` over the whole draw, then
`top_up_prior_from_anchors` fills it with a priority-weighted draw over the
*whole* archive ([train.py:4952](../train.py:4952)). So a condition whose prior
yield is zero has no guarantee of any anchor coverage — the backfill goes
wherever archive-wide priority sends it. On a single-condition problem the two
coincide; on 198 conditions they do not.

    buffers.prior_buffer.anchor_floor_frac: 0.0   # new key, default 0 == today

Per churn cycle, with `N_c` conditions and a per-condition intake target of
`budget / N_c`:

    anchor_quota_c = ceil(anchor_floor_frac * budget / N_c)      # guaranteed
    shortfall_c    = max(0, budget/N_c - admitted_c)             # as today, but per-condition

and `top_up_prior_from_anchors` gains a per-condition count vector, stratifying
the draw by `anchor_buffer.batch.condition_id` while keeping the existing
priority weighting with its `replay_beta` random floor *within* each condition.

- Stratification is **free**: the top-up's cost is the energy re-score of the
  noised batch ([train.py:4962](../train.py:4962)), which scales with the number
  topped up, not with how it is partitioned.
- `N_c = 1` degenerates exactly to current behaviour, as does
  `anchor_floor_frac: 0.0`.
- A condition with no anchors leaves its quota unfilled — skip and log rather
  than borrowing from another condition.
- **`prior_buffer_anchor_fraction` cannot verify this.** It is a *flow* over one
  eval window, drained on read, and pooled. Report per-condition intake min /
  median alongside it. A true resident-composition metric needs a source tag per
  row, which is a `CrystalBuffer` schema change — worth it eventually, not
  required here.
