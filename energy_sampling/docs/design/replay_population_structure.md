# The replay population: how a row is born, used and evicted

Written 2026-09-08 for review. Companion to `rarer_rollouts.md` (the cadence design),
`replay_occupancy_and_cadence.md` (the levers) and `z_fill_process_variance.md` (the fill).
Line numbers are HEAD `d8c5da9` plus the uncommitted working tree; grep the symbol if
they have moved.

---

## 0. The shape in four lines

Store-all, full-batch draw every step, memoryless hazard:

```
admissions per step  ~  B / N_eff
draws per step       =  B                    (fracs are LOSS weights, every branch runs a full batch)
occupancy            O  ~  B * tau / N_eff           (long-run AVERAGE, Little's law)
reuse per TRAINING row ~ N_eff / (1 - v)             (v = val_frac; held-out rows take no training draws)
```

**These are approximations, not identities, and the note previously stated them as
exact.** Little's law relates a long-run *average* occupancy to the admission rate and
the *actual* mean residence. What the code does (train.py:8973) is different in ways
that matter:

- the hazard is a **per-call fractional budget** scaled by steps elapsed since the last
  call, with rounding — not continuous exponential death;
- a **backstop age cap** at `backstop_mult * tau` truncates the tail;
- **overflow** at `max_size` evicts irrespective of age;
- admissions arrive in **batches of B every N steps**, not as a smooth rate, so occupancy
  sawtooths rather than sitting at its equilibrium;
- the **eval-site call** admits outside the rollout cadence;
- `admit_reward_min` **rejects** rows, so admissions < B;
- `grow_batch_size` changes B mid-run, and occupancy takes ~tau steps to follow.

So `O/B ~ tau/N_eff` is a useful approximation under stated conditions, and
**`tau >= 2N` does not guarantee live occupancy >= 2 batches** — it sets the equilibrium
the buffer relaxes *toward*. That is still the right way to size it, but the guarantee
language in an earlier draft was wrong.

`N_eff` is the *effective* cadence, not `fwd_rollout_every` — that key is a **backstop**
and triggers only shorten the interval. Measured on `rr08_tol05`: configured 20,
effective **14.7** (991 on-cadence + 360 triggered rollouts over 19 820 steps).

Consequences worth holding:

- **`N` sets overfit and energy cost together.** Both scale with reuse. There is no way
  to buy the energy saving without buying the reuse, short of drawing less than a full
  batch.
- **`tau` moves staleness primarily.** It cancels out of *average* reuse, but it is not
  inert for overfitting: changing `tau` changes population size, diversity and the
  temporal correlation between successive draws. Equal average reuse does not imply
  equal overfitting.
- **Reuse is a population average, not any row's experience.** The draw is prioritised,
  so draw counts are unequal by design, and **validation rows receive zero training
  draws** — which is why the per-training-row figure carries the `1/(1-v)`.
  Corroborated: at N=7, `val_frac` 0.1, measured `expired_draws` = **7.53**, against
  N = 7.00 and N/(1-v) = **7.78**.
- **`O/B ~ tau/N`**, so the "occupancy in batches" trigger is close to reading back the
  *configured ratio* rather than an independent measurement. See §4.

---

## 1. Birth — admission

**Where:** `Modeller.manage_replay_buffer` (train.py:8823), reached only when
`_fwd_gates` (train.py:3826) returns `fwd_ran`.

- A rollout samples `B` trajectories from P_F and evaluates their energies. **The only
  energy call in a training step** — bwd and replay score pre-built rows.
- Store-all (`churn_rate = batch_size`) admits the **eligible** batch — `admit_reward_min` rejects rows upstream, so admissions <= B.
- Also called once per eval, outside the rollout cadence — this is the ~`eval_period`
  sawtooth in `replay/absorption_n`, not a bug.

**Columns set at birth** (`CrystalBuffer.add`, buffer.py:1082):

| column | set at | thereafter |
|---|---|---|
| terminal + full trajectory | admission | immutable |
| `log_r` | admission | **never recomputed** — this is what makes replay energy-free |
| `birth_log_pf` | admission | frozen; the drift reference |
| `birth_loss` | admission | frozen; the absorption denominator |
| `is_val` | admission, `val_frac` of rows | frozen |
| `ema_loss` | **admission** (`init_loss` seed, buffer.py:285) | EMA on each draw |
| `ema_logw` | **first draw**, seeded exactly (NaN branch) | EMA on each draw |

⚠ `log_pb` is **not** frozen. With `learn_pb` on, P_B is a trained network and the stored
trajectory is re-scored under the current P_B exactly as under the current P_F. So

```
d log w = d log p_B - d log p_F
```

and `birth_log_pf` gives one term of two. Measuring `d log w` would need a `birth_log_w`
column that does not exist — which is why **there is no log Z staleness sensor**.

---

## 2. Life — the draw

**Where:** `Modeller.draw_replay_sample` (train.py:6272) → `replay_train_step`
(train.py:5821). Runs **every step**, full batch.

**Priority** — `CrystalBuffer.prioritised_weights` (buffer.py:1449):

- `p ∝ |delta|` with `delta = log Z - ema_logw`, reconstructed each draw. Not the birth
  value — but **not the current step's log w either**.
- `ema_logw` is an EMA at **beta = 0.9** (`update_logw_stats`, buffer.py:1004, never
  overridden at the call site) and it **advances only when the row is drawn**. Clock is
  draws, not steps: half-life **6.6 draws** against a lifetime of `N` draws. At N=7 that
  is one half-life over the row's whole life; at N=20 it reaches ~88% converged.
- First draw seeds it *exactly* (the NaN branch writes raw `log w`), so a row starts
  correct and lags increasingly after.
- **The lag mainly costs variance.** `p` and the correcting weight `w` come from the
  *same* `ema_logw`, so the *unnormalised* estimator's identity `E_p[w f] =
  E_uniform[f]` is stale-invariant. But the loss **divides by the sampled weight sum**
  (gflownet_losses.py:688), i.e. it is *self-normalised* IS: consistent, with an
  O(1/n) finite-batch bias that does not vanish just because `p` and `w` agree.
  "Costs variance, not bias" was too strong — and the same overclaim is in the code
  comment there, including "the overall loss scale is unchanged", which likewise holds
  only asymptotically.

**Correction** — the weights are applied at the FINAL reduction
(gflownet_losses.py:688), so they cover every active loss term at once.

⚠ The code comment there says SNIS "belongs with a QUADRATIC branch loss". **That is a
gradient-allocation preference, not a mathematical requirement** — importance weighting is
valid with a Huber loss; what changes is that outside the knee the per-row push goes as
`beta/delta`, so the deepest rows push *least*, and prioritising toward them then buys
less than it appears to.

Whether the knee is active is **run-dependent**, and this note has previously stated both
answers without saying which run:

| run | replay residual scale | knee at beta = 80 |
|---|---|---|
| `rr07_rr_n7_v1` (bad warm start) | `logw_std_within` ~72-75 | **active** — roughly a quarter of rows outside |
| `rr07_rr_n7_v2` (good warm start) | `tb_err_worst` 45 -> 62 | mostly inside |
| `rr08_tol05` (cluster) | `tb_err_worst` 30 -> 25 | comfortably inside |

So it is quadratic across the operating range **on the good-warm-start runs**, and was
not on the arm the original observation came from.

⚠ Any `replay/*_mean` is a **draw statistic, not a buffer statistic**. The draw is
prioritised and the SNIS weights correct the *loss*, not the logged means. `replay/
logr_mean = -58.6` on rr07_rr_n7_v2 against a live forward batch at -8.5 is the draw
skew, not the buffer's contents.

---

## 3. Death — eviction

**Where:** `manage_replay_buffer`, keep-mask at buffer.py:1239.

- Approximately-memoryless hazard, mean residence `tau` **in steps** (converted from
  per-manage-call in v0; that silently reinterpreted `mean_residence_steps` in every
  prior config by a factor of N).
- Hard age cap at `backstop_mult * tau`; overflow at `max_size`.
- Implemented as a **per-call fractional budget scaled by elapsed steps** — memoryless
  in intent and in the long-run mean, but discretised by the call cadence and by
  rounding, and truncated by the backstop.
- ⚠ **Memoryless lifetimes alone do not give resident mean age `tau`.** That result needs
  **stationary arrivals** as well, and ours are not stationary: admissions arrive as a
  burst of `B` every `N` steps, and a fresh burst lowers the mean resident age even under
  exact continuous exponential death. So `tau` controls the age distribution, but "mean
  age = `tau` whatever the arrival pattern" is false here — an earlier draft said it, and
  a later one only hedged it to "the continuous version".
- `hazard_frac` reports the split; ~0.996 on the local arms, i.e. the hazard is doing
  essentially all the evicting.

⚠ **When `max_size` binds, `tau` stops being the mechanism.** On the running `rr08_tol05`,
`replay_buffer_length` = 12000 = `max_size` exactly with `mean_age` **21** against
`tau = 100` — rows leave by overflow long before the hazard would take them. The
regenerated arms derive `max_size` from `churn * tau/N * 1.25` so every one has 20–83%
spare, but the three arms in flight do not.

---

## 4. The cadence and its triggers

**Where:** `_fwd_gates` (train.py:3826), `_rollout_trigger_fires` (train.py:3917),
`_rollout_trigger_reading` (train.py:3901). Parsed in `Stage.__init__` (protocol.py:307),
audited by `fwd_rollout_cadence_is_well_formed` (config_invariants.py:1176).

```python
fwd_ran = ((step - stage_anchor) % rollout_every == 0)     # anchored to STAGE entry
if fwd_ran:      last_rollout = step
elif triggers_fire():   fwd_ran = True                     # elif -> disjoint paths
```

The cadence is anchored to **stage entry**, not the absolute counter, so the first fused
step of any stage always rolls out — that step is the Z bootstrap and the buffer's first
fill. `step_ind % N` only looked equivalent because the local acceptance runs skip phase 1.

A trigger fires only if ≥ 2 steps since the last rollout (a *placement* bound: readings
predate this step's admission, so without it a chronically tripped bar fires at step 1 of
every window). A `None` or non-finite reading **never** fires — an unmeasured quantity is
not evidence of a problem.

| bar | reads | shipped | status |
|---|---|---|---|
| `val_gap_max` | `replay/val_gap_nats` | **4.0** | **controlling** |
| `occupancy_min_batches` | `len(buffer)/batch_size` | **1.5** | anomaly detector |
| `drift_std_max` | `policy_drift_std` | off | — |
| ~~`ess_min`~~ | ~~`policy_drift_ess_frac`~~ | **retired** | see §5 |

**Why the occupancy bar sits below the design floor.** `O/B ~ tau/N`, so the reading is
close to the configured ratio rather than an independent signal. `TAU_OVER_N_MIN = 2` is
enforced at generation (rr_sep08/make.py raises rather than emitting a lower arm). Because
`fwd_rollout_every` is a **maximum** — triggers only shorten the interval, raising the
admission rate and hence occupancy — a shorter effective cadence pushes `O/B` *up*, so
`tau = 2N` sets an equilibrium of ~2 batches rather than a floor at 2.

⚠ **It is an equilibrium, not a guarantee.** Discrete batched admission, rounding, the
backstop, reward rejection, overflow and batch growth all let live occupancy sit below it
transiently. That is precisely what makes a bar *below* the design ratio meaningful: at
1.5 it fires when live occupancy has dropped materially under the configured equilibrium
(initial fill, a batch growth the buffer has not caught up with, a servo churn boost),
whereas a bar equal to the ratio would fire on ordinary discretisation noise and, on the
boundary arm, essentially forever.

### Current configuration vs the proposed experiment

These are **not the same** and the note must not blur them.

**Current (what the regenerated arms ship).** Two live bars: `val_gap_max` **and**
`occupancy_min_batches`. The second is a safety interlock on the draw pool, not a
quality signal — it exists so a cadence loosened for cost cannot starve the draw.

**Proposed experiment.** *The validation-training gap is the sole adaptive rollout
trigger; a deterministic maximum period bounds the time spent without fresh forward
samples.* Occupancy is held safe by construction instead of by a trigger — size `tau`
against the maximum period so the equilibrium pool clears the floor — leaving exactly
one adaptive input, so that any change in effective cadence is attributable to the gap
and to nothing else.

⚠ **A quiet gap is not the same as a healthy run.** The gap could stay flat while the
run deteriorates for reasons it cannot see. Two checks belong alongside it, read at the
**forced** (deterministic-maximum) rollouts, since those are the only samples not
selected by the trigger:

- **forward sample quality** — `Nonthermal Fraction`, mean/median energy, `fwd/tb_err`;
- **the Z correction at that rollout** — `z_fill/gap` and `z_fill/K`. A large accumulated
  gap at a forced rollout says log Z drifted materially while nothing was watching it,
  which is precisely the failure the removed Z bar cannot detect.

**There is deliberately no Z bar.** Nothing measures log Z's fixed point between rollouts,
because obtaining one *is* a rollout. The unpinned interval is bounded **open-loop** by
`fwd_rollout_every` alone. That is a real limit of the design, and the high-N arms are
where it gets tested.

---

## 5. The three instruments, and which one steers

Absorption and validation compare the residual `delta = log Z - log w` against different
references. **Drift does not compare `delta` at all** — it compares current `log p_F`
against birth `log p_F` on the same trajectory. An earlier draft called all three "the
same residual against a different reference"; that is wrong for drift, and the difference
matters because drift is therefore insensitive to log Z entirely, by construction.

| instrument | reference | code | verdict |
|---|---|---|---|
| **Validation** | other rows, right now | `_replay_val_stats` train.py:6044 | **controlling** |
| **Drift** | the policy that made the row | `_policy_drift_stats` train.py:5947 | diagnostic only |
| **Absorption** | the same row at admission | `absorption_stats` buffer.py:1392 | buffer_servo input |

**Why validation steers.** Rank the three by what happens if you ignore them:

- Drift is **plausibly self-correcting** — an abandoned row has a large residual ⇒ high
  priority ⇒ drawn more ⇒ `log p_F` pushed back up. **This is a hypothesis, not a
  guarantee**, and it has not been tested. Two reasons it can fail: the priority is
  computed from a *stale, draw-updated* `ema_logw`, so a row's current drift does not
  immediately raise its priority (a row must be drawn before the evidence that it should
  be drawn is recorded); and even once drawn, movement in the learned P_B, in shared
  trunk parameters, or from the competing bwd objective can prevent `log p_F` recovering.
  Separately, TB is off-policy-correct, so drift does not bias the loss — that part is
  structural rather than hypothetical.
- Absorption is **unactionable** — three causes in one number (genuine learning,
  memorisation, and log Z motion re-signing every residual at once).
  ⚠ Its **only live consumer is the `buffer_servo`** (`numerator: replay/ema_loss_mean`,
  `denominator: replay/birth_loss_mean`). It does **not** feed the prioritised draw —
  `prioritised_weights` reads `ema_logw` and the current `log Z`, and its own docstring
  says "`ema_loss` cannot serve here: it stores |resid|, and the sign is exactly what the
  one-sided priority needs." Earlier drafts of this note claimed absorption was
  irreplaceable *because* the draw depended on it; that was wrong in both directions.
  Retiring the servo would leave `resid_vs_intake` with no consumer at all.
- Overfit has **no explicit control targeting the generalisation gap.** Per-training-row
  reuse is ~`N_eff/(1-v)` and rises with the cadence; nothing in the system reads the gap
  and acts on it, the way the draw plausibly acts on drift.

Since per-training-row reuse scales with `N_eff` while energy cost scales as
`1/N_eff`, an overfit-driven cadence is close to a quality-versus-compute dial.

**The measurement** (`_replay_val_stats`, train.py:6044):

- `val_frac` of admissions flagged `is_val`, drawn by `sample_val_graphs` (buffer.py:900),
  **never trained on**, and deliberately routed around the draw counters so an untrained
  row never looks drawn.
- Scored with the identical functional, unweighted, Z tracker read-only, same step,
  before the optimiser moves — so both sides see the *same* Z and the *same* parameters.
- ⚠ **That is not exact cancellation.** A shared shift in `delta` cancels between signed
  means, but the statistic differenced here is a **median of |delta|**, and `|.|` is
  nonlinear: `med|delta + c| != med|delta| + c`. A Z error perturbs both cohorts in the
  same way *in distribution*; it does not cancel algebraically.
- ⚠ **Identical eviction rules make the cohorts comparable in expectation, not in
  realisation.** They do not guarantee matched realised ages or reward distributions in
  any given batch. The gap is strong evidence of **differential fit between rows that
  were trained on and rows that were not** — which is the thing we care about — but
  calling it a pure memorisation measurement overstates it.
- `val_gap_nats` = median |delta| val minus **IS-weighted** median train
  (`_weighted_median`, train.py:6008). Weighted because the draw is prioritised — a plain
  median there would be a median of the priority distribution, not of the buffer.
- Reports an uncertainty, `1.858 * MAD / sqrt(n)` — but ⚠ **this is the validation
  side's uncertainty alone**, from a normal-based MAD approximation. It omits the
  variance of the **weighted training median** and the covariance between the two.
  Calling it "the gap's standard error" overstates what is measured. It is an
  **incomplete uncertainty estimate and bounds the true one in neither direction**:
  `Var(gap) = Var(val) + Var(train) - 2 Cov`, so a large positive covariance could make
  the true variance *smaller* than the val side alone, and the MAD-normal approximation
  may misestimate either way. Any significance figure derived from it — including the
  "8.6 sigma" an earlier draft quoted — is unsupported.
- Probe size is `min(val_cap, batch_size, n_val)` (`_replay_val_size`, train.py:6257) —
  **a cap above the batch size is inert.**

⚠ `val_gap` (nats², the mean-Huber spelling) is kept for series continuity only. Its knee
sits *inside* the residual distribution, so its mean counts how many catastrophic rows
landed on each side: one row at |delta| = 200 moves a 200-row mean by **63**, larger than
any gap measured. The two spellings have already been observed disagreeing **in sign**.

**Why `ess_min` was retired.** `policy_drift_ess_frac` is Kish on `exp(d)`, therefore
scale-invariant: it reads **1.000** when *every* row's `log p_F` drops by the same 5.7
nats. It cannot see the policy walking away from the buffer, only the spread. It was also
the only bar firing on the rr08 arms (360 times, +36% rollouts), i.e. it was setting the
cadence off a quantity we do not want to steer on. `policy_drift_nats` (mean |d|) is the
natural reading — linear, in nats, not owned by the tail — and is **not yet wired as a
bar**.

---

## 6. Open

- **`val_gap_max = 4.0` is not calibrated.** The measured gap on `rr08_tol05` is
  **1.25 nats**, with a reported val-side uncertainty of 0.146 that (per §5) does not
  support a significance claim. Nor is it known what level of gap is *harmful*. The bar
  is a guess. Getting the harm curve means running `N` up until the gap is large and
  watching whether quality actually degrades — an experiment no arm currently performs.
- **The measured numbers quoted here need reconciling with their arms' actual configs
  and logging revisions** — `rr08_tol05` in particular is running HEAD `d8c5da9`, not the
  working tree this note describes.
- **Draw capture is unmeasured.** The drift concern that survives is not staleness but
  `p_F` collapsing on low-reward rows: a collapsed row has an enormous residual, therefore
  maximal priority, therefore it can capture the draw. The right metric is a Kish ESS on
  the **draw distribution `p`**, not on the IS weights — a different vector, currently
  unlogged.
- **`rollout/rate` is uncommitted**, so the effective cadence on the running arms was
  reconstructed from the trigger counters rather than read.
- **`replay_buffer_admitted` needs provenance before it means anything.** The current
  code *does* reset `replay_churn` immediately after logging (train.py:7818), so it is a
  **count per logging window** — a rate only once divided by the window duration. The series on `rr08_tol05` is 8 points reading
  5 000 → 242 000 → 251 000 and then flat — which fits neither a rate nor a clean
  counter. An earlier draft called this a defect; that was premature. Check the run's
  actual code revision and logging path before concluding anything.
- **Smoothness** is covered by an existing invariant rather than a sensor: fill event size
  `N/tau <= 0.2`, which `tau = 5N` enforces at any cadence. Under the new `tau >= 2N`
  floor the worst case is 0.5 — worth a look if long stale stretches followed by large
  injections are a concern.
