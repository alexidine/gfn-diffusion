# Module: buffers (`buffer.py` + management in `train.py`)

Pass 1 (audit + rationalize). Verified against the working tree, 2026-08-03;
**revised 2026-08-08** for the prioritised-replay package (`to_do_rebuild.md`
§B5/§B7b/§B7c) and the memorisation sensor (§B7d), which shipped 2026-08-07.
Unconditional route. `ConditionLogZTracker` also lives in `buffer.py` but is
covered in `module_metrics.md` — it is a statistics tracker, not a buffer.

> **Numbering.** `B1…B10` here are **module-local finding IDs**.
> `to_do_rebuild.md`'s `§B1…§B9` are a *different* series — the design argument
> for the replay rebuild. When a reference matters, this document always writes
> the latter as `to_do_rebuild.md §Bn`.

---

## 1. What it is

**One container class, three management policies.** `CrystalBuffer` holds a
resident PyG `Batch` plus per-row bookkeeping (`ema_loss`, `select_counts`,
`birth_step`, `birth_loss`, `ema_logw`, optionally full trajectories). It is
deliberately *policy-free*: [`add()`](buffer.py:571) has no admission gate at
all. Every decision about what enters and what leaves lives in `train.py` — with
one 2026-08-07 exception, [`prioritised_weights`](buffer.py:915) and
[`absorption_stats`](buffer.py:863), which are *pure functions of the resident
rows* and so belong on the container (B7, B8).

The three roles:

| | **prior** | **replay** | **anchor** |
|---|---|---|---|
| Class | `CrystalBuffer` | `CrystalBuffer` | `AnchorBuffer(CrystalBuffer)` |
| Holds | terminal states | terminal states **+ trajectories** | terminal states |
| Seed | full prebuilt `prior_dataset` | first eligible fwd batch | `prior_dataset` |
| Intake | prior-model draws within `ramp_floor` of Emin(c); anchor backfill | **two regimes** (B0): softmax over clipped \|resid\|, *or* uniform | surprise-gated screen → confirm |
| Draw | uniform (or `_loss_weights` under `weighted_bwd_sampling`) | **two regimes** (B7): uniform, *or* `p ∝ δ₊^κ` with IS weights | — |
| Eviction | relative quantile (bottom 25% of visited) | four causes, **two of them regime-gated** (below) | periodic thin; otherwise permanent |
| Size | 10k–250k | 4000 | ≤ 200k |
| Consumed by | `bwd` training | `replay` training | nothing directly — it refills `prior` |
| Manager | [`manage_prior_buffer`](train.py:4114) | [`manage_replay_buffer`](train.py:4859) | [`screen_and_admit_anchors`](train.py:5397) |

**The replay column now has two mutually exclusive configurations**, switched by a
single key (`buffers.replay_buffer.prioritise.enabled`). All three of its rows
change together, because they are one design — see B0 for the intake/purge side
and B7 for the draw. Nothing below that says "replay does X" is true
unconditionally; check which regime the config selects first.

Paper voice: *training draws from a large, slowly-churned prior buffer that
supplies distributional diversity, and from a small, fast-turnover replay buffer
that supplies the policy's own mis-weighted samples. A permanent archive of
high-reward, policy-surprising states backfills the prior buffer when
prior-model sampling comes up short.*

The answer to "are these three modules?" is **no** — they are one container with
three intake/eviction policies and one subclass that adds deduplication and
thinning. That is worth saying plainly in the paper: it is a simpler system than
the config surface implies.

## 2. Contract

**`CrystalBuffer` guarantees** — row-aligned side arrays (`x`, `y`, `ema_loss`,
`select_counts`, `birth_step`, `birth_loss`, `ema_logw*`, `traj`) stay consistent
across `add` / `purge_by_index`; `add` stages every allocation before committing
any of it, because train.py's OOM handler can catch a CUDA OOM mid-step and a
partial commit would leave a buffer that only detonates on a later draw.

**Persistence** — `state_dict` must `copy.copy` before `.cpu()`: PyG's
`Data.to()/.cpu()` mutate in place, so a bare `self.batch.cpu()` silently demotes
the live GPU-resident buffer at every save.

**Not guaranteed** — nothing about *what* is in the buffer. That is entirely the
manager's business.

## 3. The three eviction philosophies

This is where the real design content sits, and the three are genuinely
different — correctly so, but the unifying principle is written down nowhere.

**Prior — relative quantile, rate owned by the churn policy.**
[`get_elig_drop_count`](buffer.py:809) cuts at `min(loss_floor, quantile)`, and
`manage_prior_buffer` passes `loss_floor=+inf` deliberately. The reasoning is a
genuinely good catch: an absolute nats bar becomes a *rate gate* once the buffer
is full, because with zero headroom the eligible set is the only intake path — on
any problem whose backward residuals sit above the bar (elj plateaus at 24–27),
nothing is ever eligible, `n_to_add` collapses to 0, and churn stops **silently**,
since the buffer just looks full and quiet. Passing `+inf` keeps ~25% of visited
rows always evictable, so the *rate* stays owned by `n_churn` and loss only
decides *which* rows go.

**Replay — four causes, none of which shapes the bulk age distribution. Two of
the four are switched off under `prioritise`** *(revised 2026-08-08)*:

| Cause | Depends on the residual? | Under `prioritise` |
|---|---|---|
| `floor` — `ema_loss` fell below 0.1 (corrected, now boring) | **yes** | **disabled** |
| `stalled` — drawn ≥ `toxic_min_draws` and `ema_loss - birth_loss >= 0` | **yes** | **disabled** (`min_draws` forced to 0) |
| `hazard` — memoryless: evict `n/τ` rows per call, uniformly at random | no | live |
| `backstop` — hard ceiling at `5τ`, binding on ~exp(−5) of rows | no (age) | live |

The gate is the same `uniform_intake` flag as B0's, at
[train.py:5067](train.py:5067). **The reason is the force spectrum, not
convenience.** Every selective step multiplies into the buffer's density —
`μ_buf ∝ Q_admit · p_admit · p_survive` — and the draw's importance weight
divides by the **draw** only. A residual-dependent `p_survive` therefore re-enters
Φ uncorrected, counting the residual twice and once stale (`to_do_rebuild.md`
§B4/§B7b). `hazard` and `backstop` are independent of the residual, so
`p_survive` drops out of the weight and they survive.

Two further reasons `floor` had to go, both recorded at the call site:

- Under a `δ₊^κ` draw a corrected row already has `p ≈ 0`, so the floor was never
  buying gradient budget — only memory. The hazard reclaims memory *without* the
  bias.
- A row driven to `δ ≈ 0` **by repeated replay** is `δ ≈ 0` at that exact
  trajectory, i.e. memorised. That low value is the **evidence** B8's sensor
  reads; purging it destroys the signal.

The hazard is the interesting one and the warrant is unusually strong: a
memoryless TTL gives exponential residence (CV ≈ 1), which is a **wide** lag
distribution and therefore a strong lowpass on the policy → buffer → gradient
path. A hard age cap gives a uniform age profile with a sharp edge, which
concentrates phase lag at one frequency. Given that this codebase has a
documented phase-2 limit cycle, that is not an abstract concern.

It now carries a **second** load: making the hazard uniform is what removes
survivorship bias from B8's memorisation sensor. That is a dividend, and it is
also a coupling — reinstating a residual-conditioned purge silently invalidates
the sensor.

**Anchor — permanent, thinned.** Admission is the expensive one: a cheap energy
+ surprise screen, then a K-rollout IWAE confirmation on survivors only, both
behind a policy-health gate (`health_gate_floor`, `health_gate_ceiling`) so
novelty is never adjudicated against a miscalibrated ruler.

## 4. Findings

**B0 — uniform intake never became the default. `admit_temperature` is LIVE on
every unprioritised config, and that is nearly all of them.**
*(verified 2026-08-08 against the code, not the design docs)*

> **Update, later the same day: `mk_dev` now sets `prioritise.enabled: true`**,
> so the dev config is on the B7b path — but the statement below still holds for
> every generated battery config in the tree, and `admit_temperature` /
> `admit_cap_*` remain live on `mk_dev` too, because the **displacement purge**
> reads them regardless of the flag. The `lr_aug08` battery deliberately reverts
> to scored admission so its LR readings stay comparable to `local_aug08`.

The question "shouldn't `admit_temperature` be dead now that we do uniform
admission?" has an inverted premise: **we do not do uniform admission by
default.** B7b shipped as an **opt-in**, gated on one key:

```python
uniform_intake = self.replay_priority_config() is not None    # train.py:4968
if uniform_intake:
    elig = torch.argwhere(sane).flatten()
    clipped_score = torch.ones(elig.numel(), device=resid.device)
else:
    elig = torch.argwhere(sane & (resid.abs() > floor)).flatten()   # floor = 0.1
    clipped_score = resid[elig].abs().clamp(max=cap)
```

`replay_priority_config` ([train.py:3203](../train.py:3203)) returns `None` unless
`buffers.replay_buffer.prioritise` exists **and** `enabled: true`. `mk_dev.yaml`
has no such block. So the default path is the **else** branch: admission is a
softmax over clipped `|resid|` at `T = admit_temperature = 2.0`, over a pool
pre-filtered by a hardcoded `|resid| > 0.1`.

Exactly **13 configs** in the tree set `prioritise` — 9 in `local_aug07`, and
rb0808 arms **7, 16, 19, 20** only. Every other arm, including all 8 of
`local_aug08` and 22 of 26 rb0808 arms, runs scored admission.

| | admission | `admit_temperature` |
|---|---|---|
| no `prioritise` (default) | softmax over clipped \|resid\|, floor 0.1 | **live — sets the sharpness** |
| `prioritise.enabled` | uniform over the sane pool | inert *for admission* |

**It is never fully dead.** Even under `uniform_intake`, `cap` and `T` still
drive the **displacement purge** ([train.py:5155](../train.py:5155)):
`purge_score = ema[live].clamp(max=cap)` then `_softmax_draw(-purge_score, …, T)`.
That branch is not gated on `uniform_intake`, so composition is shaped on the way
*out* even when intake is uniform.

Three traps this exposes:

1. **The retirement is scheduled, not done.** `decisions.md` D5 and
   `to_do_rebuild.md` §C Phase 3 step 3 plan to retire these keys. Reading that
   as a statement about the code is wrong — and the four cap/temperature keys are
   read by **direct attribute access with no default**
   ([train.py:4947–4951](../train.py:4947)), so deleting them from a config
   raises `AttributeError` rather than falling back.
2. **The docstring is wrong.** [train.py:3204](../train.py:3204) says
   "None/<=0 disables it". `kappa: 0.0` returns `float(0.0)`, which `is not
   None`, so `{enabled: true, kappa: 0}` engages the **whole** B7b package. Three
   rb0808 arms (7, 19, 20) are set that way.
3. **rb0808's replay block is heterogeneous in admission policy.** `rep_b7b` (7)
   and `rep_kappa1` (16) run uniform intake; `rep_hi` (24), `rep_fixed_fracs`
   (23) and `rep_off` (8) run scored admission. A `rep_hi` vs `rep_b7b`
   comparison therefore varies frac dose **and** admission policy together.

`admit_reward_min` sits on a different footing: it is optional
(`getattr(..., None)`, [train.py:4936](../train.py:4936)) and it constrains
`sane` **upstream of the branch**, so it binds on both paths by design.

**B0a — the replay buffer's *hardness* came from ADMISSION, not eviction — and
B7b's uniform intake gives it up.** *(measured 2026-08-08, `local_aug09` v0/v3)*

Turning the B7b package on dropped `replay_buffer_mean_loss` from **16.9 to
7.2**, and the forward branch degraded with it (`fwd/tb_err` 21.5 → 33.9,
`over_coverage` 20.7 → 33.2) while the *drawn* batch's `replay/tb_err` barely
moved (15.8 → 16.2).

**The first hypothesis was wrong and the isolation arm killed it.** I proposed
that sizing `max_size` above equilibrium had made the displacement purge dormant,
and that the purge — which evicts preferentially on low `ema_loss` — had been
keeping the buffer hard as a side effect. `v3_size4000` reverts `max_size` to
4000 and changes nothing else. Result: buffer mean 7.35 vs v0's 7.30,
`fwd/tb_err` 33.84 vs 33.87. **Identical.** The purge is not the mechanism.

**The mechanism is admission, and `birth_loss` proves it.** `birth_loss` is
snapshotted once at admission ([buffer.py:93](../buffer.py:93), and
[:646](../buffer.py:646) on add) and **never updated** — so it is a pure
admission statistic that no amount of subsequent training can move.

| | `birth_loss_mean` | `ema_loss_mean` |
|---|---|---|
| scored admission (old) | **23.73** | 16.76 |
| uniform intake (B7b) | **10.86** | 7.49 |

Rows now *enter* the buffer with less than half the residual they used to. The
old softmax-over-clipped-`|resid|` admission was skimming the hard tail of each
forward batch; uniform intake admits a representative sample. That is exactly
what B7b intended — and the size of the effect was not anticipated.

**Restated in §B4's terms, this is a change in the force spectrum's shape:**

| | admission | draw | weight | Φ vs δ |
|---|---|---|---|---|
| old | `∝ softmax(\|δ\|/T)` | uniform | none | **superlinear** (and uncorrected) |
| B7b | uniform | `∝ δ₊^κ` | `(1/n_elig)/p` | **linear** (and unbiased) |

B7b moved the prioritisation from admission to the draw and made it *correct* —
the IS weight divides by the draw, so the estimator is unbiased for the
uniform-over-eligible mean. The cost is that the old, formally-wrong pipeline
delivered **more** force to high-δ regions, because its admission bias multiplied
into the density and was never divided out. The estimator got right and the
signal got weaker.

> So the open design question is not "is B7b correct" — it is. It is **whether
> uniform-over-eligible is the target we want.** Unbiasedness is a property
> relative to a target, and B7b silently changed the target from "the hard tail"
> to "the buffer". `is_elig_frac ≈ 0.40` says the estimator's actual support is
> the positive-δ 40%, which is a partial restoration of the old bias — but a much
> weaker one than `softmax(|δ|/2.0)`.

Second-order, consistent: `grad_norm_pre_clip` 536 → 397 and `lrprobe/
alpha_median` 1.66 → 3.17, i.e. the step probe independently reports the step is
~3× smaller than local curvature permits. With `is_ess_frac ≈ 0.39` the IS
estimator discards ~60% of its effective sample, so the replay gradient is both
smaller and noisier.

### B0b — settled at 3600 steps: a TRADE, not a regression, and κ cannot fix it

Five isolation arms plus two full-length runs. `fwd/tb_err` at matched steps
against the a_frz control (21.46):

| arm | κ | replay `beta` | `is_ess` | `w_max` | `fwd/tb_err` |
|---|---|---|---|---|---|
| **v4** | 1 | **10** | 0.363 | 7.3 | **27.29** ← best |
| v0 | 1 | 1e6 | 0.393 | 6.7 | 33.87 |
| v3 | 1 | 1e6, `max_size` 4000 | 0.396 | 6.7 | 33.84 |
| v6 | **2** | 10 | **0.073** | **58.4** | 36.29 |
| v5 | **0** | 1e6 | **1.000** | 1.0 | 38.60 ← worst |

Three hypotheses died to their own isolation arms: the displacement purge (v3 ≡
v0), IS-weight variance (v5 removes the weights entirely and is *worse*), and
κ-sharpening (v6 targeted harder rows exactly as designed and blew `is_ess_frac`
to 7% with one row at 58× the mean weight). **The variance bound bites before
κ = 2, so κ ≈ 1 is the practical ceiling and the admission gap cannot be bought
back through the draw.**

**What survives: κ = 1, keep Huber, and the residual gap is admission.**
De-huberisation costs ~6.6 nats — B5b requires quadratic so Φ ∝ δ holds under IS
correction, and on this route that correctness costs more than it buys. It
independently replicates the `local_aug07` β ladder.

**The verdict at 3600 steps** (v7 = κ 1 / β 10, final window vs a_frz):

| | a_frz | v7 | gap | seed floor |
|---|---|---|---|---|
| **`bwd/tb_err`** | 15.14 | **14.64** | **−0.50** | 0.04 ✅ |
| `fwd/tb_err` | 18.72 | 23.12 | +4.40 | 0.52 ❌ |
| `EffDim` | 5.80 | 5.90 | +0.11 | 0.10 — |

`bwd` draws from the prior buffer — a fixed, diverse population — while `fwd` is
fresh on-policy rollouts. **The new construction fits the typical population
better and leaves the forward tail uncorrected**, which is precisely what the old
hard-tail-skimming admission was buying. The fwd gap is *stable*, not closing
(per-window: 3.12, 3.97, 5.61, 5.09, 4.27, 4.09, 4.39).

> `replay/tb_err` rising 16.9 → 23.5 while `replay_buffer_mean_loss` falls to
> 5.75 is the draw **working**, not failing — a κ = 1 draw skimming the hard tail
> of a softening buffer. Read the two together or the draw looks broken.

**Two consequences needing action.** `bounds.replay: [0.05, 0.45]` **saturates**
in both long arms (pinned at 0.45 from ~step 3450 with `rt_rho` still above
setpoint), so the ratio controller loses authority under this construction. And
`prioritise.enabled` gates **intake and draw together**
([train.py:4968](../train.py:4968) and [:3218](../train.py:3218) read the same
flag), so "old admission + new draw" — the one configuration that would test the
admission hypothesis directly — is unreachable by config. Splitting that flag is
the prerequisite for settling B0a.

**B1 — `CrystalBuffer.purge()` was dead code. ✅ REMOVED 2026-08-03.** The
`max_count`/`loss_cutoff` variant had zero callers; `purge_lowest` and
`purge_by_index` are the live paths.

**B2 — `_loss_weights` min-max normalises, which makes prioritisation
scale-free in a way that may not be wanted.** *(confirmed, low severity)*

[Line 855](buffer.py:855) maps `[min, max] → [0, 1]` before applying the
temperature. Two consequences: (a) a converged buffer whose real loss spread has
collapsed still gets **full-strength** prioritisation, amplifying noise into the
sampling distribution; (b) one outlier compresses everyone else toward 0, making
the remaining rows near-uniform. Rank-based or robust-quantile normalisation
would be stable against both. Currently low-impact — `weighted_bwd_sampling` is
a per-stage flag and the `naive` stage does not set it, so this path is dormant
on the unconditional route.

**B3 — the replay occupancy equilibrium sits exactly at `max_size`.** *(confirmed, marginal)*

The design is explicit that occupancy should be *emergent* (Little's law,
`n = admit_rate × τ`) and that `max_size` is "a memory guard, not a target …
safety limits go where they do not bind." mk_dev runs `churn_rate: 80`,
`mean_residence_steps: 50`, `max_size: 4000` — and 80 × 50 = 4000 exactly.

It probably does not bind, because `churn_rate` budgets admission *attempts*
rather than successes, so realised intake is below 80. But the headroom is zero
at full admission efficiency, and if it ever binds, `max_size` — not the hazard
— starts shaping eviction at the margin, which is precisely what the design says
must not happen. Either raise `max_size` to ~2× the equilibrium or state that the
guard is intended to be tight.

**B4 — `max_residence_steps` retirement is enforced only at first
`manage_replay_buffer` call.** *(confirmed, operationally expensive)*

The guard at [train.py:5046](train.py:5046) raises `ValueError` on the retired
key — correctly. But it lives inside the manage path, which first runs at the
phase-1 → 2 transition. The aug02 battery lost **all 16 arms' entire phase 1**
(1.1–7.8 h each) to exactly this: every arm trained phase 1 to completion, then
died at the transition. A config preflight that walks every stage's key surface
at load time would have cost seconds.

**B5 — `init_fraction` is documented inert; `min_size` is doing two unrelated
jobs.** `min_size: 10000` is used as a per-cycle chunk bound in
`rebuild_prior_by_churn` ([train.py:4263](train.py:4263)) *and* as a sampling
count in `grow_prior_buffer` ([train.py:5607](train.py:5607)). It is not a
minimum size anywhere. Rename or split.

**B6 — the prior buffer's reach trigger, described (no defect).** *(user-confirmed fine)*

`reach = 1 − quantile_0.9(excess) / margin`, fires below `reach_threshold: 0.75`.
Pooling excess-above-own-condition-best is correct for the conditional case (one
condition's easier scale can't mask another); unconditionally there is one
condition, so the pooling is a no-op and this reduces to "is the buffer's 90th
percentile within 25% of `ramp_floor` of the best known energy." **This entry is
descriptive** — it is the only anchor→prior feedback path and is recorded so the
coupling is visible, not because anything is wrong with it.

**B7 — the prioritised draw: `p ∝ δ₊^κ` with row weights that undo it.**
*(built 2026-08-07; unbiasedness verified exactly, payoff not established)*

[`prioritised_weights`](buffer.py:915) returns `(p, w_of_row)` with
`w = (1/n_elig)/p`, so `E_p[w·f] = E_uniform[f]` **at every κ**. The estimator is
unbiased by construction; only its *variance* changes with κ. That is the whole
claim of the κ ladder, and it means any difference a ladder measures is estimator
variance and nothing else.

Four design points, each of which was changed by testing:

1. **δ is reconstructed, not stored.** `δ = log Z − log w`, and `ema_logw` already
   carried a per-row EMA of `log w` — checkpointed, resized on grow/purge, and
   **called by nothing** until now. `ema_loss` cannot serve: it stores `|resid|`,
   and the *sign* is exactly what a one-sided priority needs.
2. **One-sided by design.** `δ₊ = max(δ, 0)`. A row the policy has moved off has a
   fallen `log_pf` and therefore a strongly negative δ, so it takes priority ~0
   automatically — which is both the intended replay/backward split and most of
   what the §B8 drift term was introduced to do.
3. **Zero-priority rows are EXCLUDED, not floored.** The first cut mixed a uniform
   floor into `p` so every row stayed drawable, and measured `max(w) = 10⁴`: a row
   drawn at probability ~0 carries weight ~∞ and single-handedly owns a
   self-normalised batch. Now `δ₊ = 0` ⇒ `p = 0`, and the estimator targets the
   uniform mean over the **positive half** — which is what the replay branch is
   for.
4. **`floor_frac` is a RELATIVE floor on the survivors**, as a fraction of the
   median `δ₊`, so the weight range is bounded by `(median/floor)^κ` rather than by
   the smallest residual that happens to be resident. **The shipped 0.01 was far
   too permissive** and was re-measured against a live buffer on 2026-08-07:

   | `floor_frac` | ESS/n | max(w)/mean(w) |
   |---|---|---|
   | 0.01 | 0.11 | 73 |
   | 0.15 | 0.50 | 5.3 |
   | **0.25** | **0.63** | **3.3** |
   | 0.50 | 0.80 | 1.9 |

   At 0.01 the live run reported `is_ess_frac` 0.02–0.06 — a 1000-row batch doing
   the work of ~20–60 rows. **0.25 is the knee and is now the default.**

**The variance claim did NOT reproduce, and the reason is not a bug.**
`to_do_rebuild.md` §B5 predicted variance minimised at κ=1 by Cauchy–Schwarz;
measured over 300 draws of 1000 rows, ESS/n runs 1.00 / 0.85 / 0.65 / 0.34 at
κ = 0 / 0.5 / 1 / 2 and batch sd goes the **wrong way** (0.38 → 2.23). The optimal
draw for a *self-normalised* estimator is `p ∝ |f − μ|`, not `p ∝ |f|`; for a mean
of δ, δ is tightly clustered about its own mean, so prioritising by δ over-samples
where the integrand is least informative. **Correctness is established, the payoff
is not** — which makes the κ ladder diagnostic rather than confirmatory, and is why
`replay/is_ess_frac`, `is_w_max_ratio` and `is_elig_frac` all ship. If
`is_ess_frac` falls below ~0.3, lower κ.

**Watch `is_elig_frac`.** It drifted 0.74 → 0.33 over 1500 steps locally: the
buffer trending toward mostly-negative δ. If it approaches 0 the prioritised
branch has nothing to draw.

**B8 — the memorisation sensor, and why it needs no calibration.**
*(built 2026-08-07; `to_do_rebuild.md` §B7d)*

[`absorption_stats`](buffer.py:863) compares each resident row's **current**
residual against the one it was **admitted with** — both already stored, no new
field:

```
ratio       = mean(ema_loss) / mean(birth_loss)     in (0, 1]
absorbed    = 1 - ratio
lambda_tau  = -ln(ratio)
```

`ratio = 1` is a pure delay line: composition equals intake, nothing has been
fitted. Falling toward 0 means residents have been corrected **at their own
trajectories** while the intake distribution has not moved — memorisation by
definition.

**The setpoint is derived, not measured.** Under exponential relaxation at rate λ
and exponential residence with mean τ, `ratio ≈ exp(−λτ)`, so the `λτ = 1`
boundary lands at **`ratio = 1/e = 0.368`**. Nothing in it was measured, so it
transfers across problem, `T` and buffer size — which is the property every
previous buffer threshold lacked.

**No survivorship bias, and that is a dividend of the uniform hazard** (§3).
`birth_loss` exists only for resident rows, so it is the intake distribution *of
survivors*; under a residual-independent hazard survivors are an unbiased sample
of admits. This would **not** hold under the old floor/stalled eviction.

Undrawn rows have `ema_loss == birth_loss` exactly and contribute `ratio` 1.
Correct: a row nothing trained on cannot have been memorised, and a buffer
churning fast enough that most rows are never drawn genuinely is not memorising.

Returns `{}` below 8 valid rows, so pre-schema buffers lacking `birth_loss` make
the servo hold at cold start rather than act on garbage. Validated as a
*discriminator* on 33 historical runs: λτ > 1.0 on four arms (BASE32K 1.54,
local_aug02 1.44, neat_dev 1.10), 0.5–1.0 on five, < 0.5 on the rest.

A 1-D Wasserstein between the intake and resident loss histograms matched the
mean-shift statistic to three decimals on every arm — the distributions differ by
a translation, so the histogram machinery buys nothing. Recorded so it is not
re-proposed.

**B9 — two wiring bugs the prioritised draw shipped with. ✅ BOTH FIXED
2026-08-07.** *(the estimator was correct; the plumbing was not)*

**(a) The draw was never prioritised — `beta` is a uniform FRACTION, not a
temperature.** `_sample_indices` splits the batch as
`n_uniform = int(batch_size · beta)`, so the legacy `beta=1.0` inherited by
`draw_replay_sample` meant **100% uniform** and a supplied `p` was silently
ignored. Worse than a no-op: the IS weights `w ∝ 1/δ₊^κ` were *still applied to
the loss*, so a uniform draw carrying `1/p` weights targets a measure `∝ 1/δ^κ` —
**the inverse of the design**, up-weighting the lowest-residual rows. Fixed by
`draw_beta = 0.0` whenever `p` is computed ([train.py:3239](train.py:3239)).

*The tell was an identity that should have been impossible*: at κ=0 the draw and
the weights are both uniform, so `is_ess_frac` must be **exactly 1**. It read
0.40 — equal to `is_elig_frac`'s 0.39 — because the uniform draw kept pulling
ineligible `w = 0` rows. Post-fix κ=0 reads 1.000 with `is_w_max_ratio` 1.000.
**A unit test of the estimator cannot catch a mis-wired draw**; the degenerate
cell is what exposed it, which is an argument for always putting one in a ladder.

**(b) One-sided draw + without-replacement = crash.** `ValueError: Fewer non-zero
entries in p than size`, killing the κ=0 arm at step 119, because `δ₊ ≤ 0` rows
get `p = 0` by design and the eligible pool fell below the batch size. **Fixed at
the principle rather than with a guard**: a supplied `p` is a *design measure* and
IS correctness assumes iid draws from it, so it is drawn **with replacement**
([buffer.py:311](buffer.py:311)). Without-replacement was both theoretically wrong
and the proximate crash.

**B10 — `update_logw_stats` existed, was checkpointed, and was called by
nothing.** *(✅ wired 2026-08-07)* The field, its resize-on-grow/purge handling and
its checkpoint round-trip all predated any consumer. `replay_train_step` now
refreshes it from `log_r + log_pb − log_pf`, which is what makes B7's signed
residual available at all. Worth recording as a *class*: a checkpointed per-row
field with no reader is indistinguishable from a live one when reading the schema,
and this codebase had one for months.

**Anchor health gate — config-driven metrics, and the bars are now named after
their ROLE.** *(changed 2026-08-03; bar keys renamed 2026-08-08)*

The gate previously hardcoded `fwd/r2` and `fwd/tb_resid_clipped`. It reads
`anchor_buffer.health_gate_floor_metric` (lower bound) and
`health_gate_ceiling_metric` (upper bound on |value|), against
`health_gate_floor` / `health_gate_ceiling` — renamed from `health_gate_r2` /
`health_gate_zerr`, which encoded the *old metric names* into the bar keys, so
swapping the ruler left a key called `health_gate_r2` holding a number with
nothing to do with r2.

**The D9/N3 upgrade to `tb_err_worst` is deliberately NOT taken, and N3's own
rationale is why.** No bar transfers across a ruler swap — and the incumbent is
the better-warranted one:

| | `tb_resid_clipped @ 0.5` (shipped) | `tb_err_worst @ ?` |
|---|---|---|
| shape | signed, bounded by `beta` | unbounded RMS, floored at `std(log w)` |
| healthy value | ~0 | **18–21 on this route** |
| warrant for the bar | **derived** — it is the D29 Z-currency invariant, the same bar `z_calibration` actively holds | none yet; would need a battery |

So the swap trades a derived bar for one nobody can state. It stays on the
docket (`decisions.md` N3) as a question rather than a pending edit.

## 5. Warrants

| Choice | Warrant | Evidence |
|---|---|---|
| Memoryless hazard over hard TTL | **measured** | `expired_delta` ran −12…−28 nats across postfix_july30 arms — the TTL was culling the *improving* tail; `absorbed_frac` 0.000 meant age was doing 100% of eviction |
| `stalled` test replaces the TTL's job | **derived** — tests the thing the TTL proxied | docstring at [train.py:5010](train.py:5010) |
| `admit_temperature: 2.0` | **measured** — and still live on 22 of 26 rb0808 arms (B0) | T=2 beat T=5; T=20 diverged (replay_july26 / tsched_july24) |
| Residual-independent intake *and* purge under `prioritise` | **derived** — `μ_buf ∝ Q·p_admit·p_survive` and the IS weight divides by the draw only, so any other residual dependence enters Φ uncorrected | `to_do_rebuild.md` §B4/§B7b |
| Draw *with replacement* whenever `p` is supplied | **derived** — IS correctness assumes iid draws from the design measure; without-replacement also crashes once the eligible pool falls under the batch | B9(b), reproduced at 1500 / 900 / 300 eligible |
| `δ₊ = 0` rows excluded rather than floored | **measured** — a uniform floor put `max(w)` at 10⁴ and one row owned the batch | B7 |
| `floor_frac: 0.25` | **measured** — the ESS knee on a live buffer; the shipped 0.01 gave `is_ess_frac` 0.02–0.06 | B7 table, 2026-08-07 |
| `κ` as a ladder rather than a setting | **measured, against the design's own prediction** — variance rises with κ on a clustered integrand, so the ladder is diagnostic | B7; `to_do_rebuild.md` §B7c |
| `ratio = 1/e = 0.368` as the memorisation bar | **derived** — `ratio ≈ exp(−λτ)` puts `λτ = 1` exactly there, so it transfers across problem, T and buffer size | B8 |
| Mean-shift over a histogram distance for the sensor | **measured** — 1-D Wasserstein matched it to 3 dp on all 33 arms | B8 |
| Clip-then-divide (cap before T) | **derived** — keeps `cap` an absolute nats bound independent of T | docstring |
| Health-modulated cap off `fwd/scatter_err` | **derived** — an *external* signal cannot ratchet off the buffer's own contamination | docstring |
| `loss_floor=+inf` in prior eviction | **measured** — a finite bar silently stopped churn | docstring at [train.py:4146](train.py:4146) |
| `churn_batch_ref: 1000` | **measured** — decouples churn pacing from live batch size after a growth event | run gejezmjg |
| Hard exclusion for non-finite / sub-`admit_reward_min` | **derived** — admission scores badness, so without a floor the energy distribution is unbounded above | docstring |
| `mean_lifetime: 100`, `max_size: 250000` (prior) | **arbitrary** | — |
| `dup_cutoff: 0.05`, `surprise_cutoff: -5`, `confirm_cutoff: -5` | **arbitrary** | — |
| `reach_quantile: 0.9`, `reach_threshold: 0.75` | **arbitrary** | — |

The replay buffer is the best-warranted subsystem in the codebase — nearly every
constant traces to a measurement. The anchor buffer is the worst: its five gating
constants are all unmeasured, and it is the subsystem whose failure mode
(admitting garbage, or admitting nothing) is hardest to observe.

## 6. Failure signatures

| Symptom | First metric | Cause |
|---|---|---|
| Replay buffer drains to near-zero | `replay` occupancy | intake starved — check candidate pool size, not the hazard |
| Buffer full and quiet, no churn | `prior_churn/evicted` = 0 | absolute loss bar became a rate gate (B3 of the prior policy) |
| Replay degrades to FIFO | occupancy at `max_size` + flat admit rate | thin candidate pool; a small `fwd` frac starves admission |
| Anchors never admitted | `last_anchor_admitted` = 0 | health gate shut — the floor metric below `health_gate_floor`, or \|ceiling metric\| above `health_gate_ceiling` |
| Buffer silently demoted to CPU | step time collapses | missing `copy.copy` before `.cpu()` — regression guard |
| Whole battery dies at the phase transition | `ValueError` naming a retired key | B4 |
| Prioritised batch doing the work of a few rows | `replay/is_ess_frac` | weight tail — lower κ, or raise `floor_frac` (B7) |
| `is_ess_frac` ≠ **exactly** 1.000 at κ=0 | compare against `replay/is_elig_frac` | the draw is not coming from `p` — B9(a). Equality of the two is the specific signature |
| Prioritised arm dies with `Fewer non-zero entries in p than size` | `replay/is_elig_frac` trend | B9(b); if it is falling toward 0 the positive half is emptying, which is a real result, not a bug |
| Buffer being fitted at its own trajectories | `replay/lambda_tau` > 1 (`resid_vs_intake` < 0.368) | memorisation; the actuator is freshness, not loss weight — `module_modulators.md` §3 |
| Memorisation sensor reads a suspiciously flat 1.0 | `replay/absorption_n` | fewer than 8 valid rows, or a pre-`birth_loss` buffer — the sensor is abstaining, not reporting health |

## 7. Simplification candidates

**S1 — factor the three eviction policies behind one interface.** They share a
shape (score → eligibility → selection → `purge_by_index`) and differ in scoring
and rate. Right now they are three hand-written implementations in two files, and
the *principle* distinguishing them — prior evicts on relative rank because its
job is diversity, replay evicts on age because its job is freshness, anchor never
evicts because its job is memory — is stated nowhere.

**S2 — add a config preflight that walks every stage.** B4 cost a full battery's
phase 1. This is cheap and catches a whole class.

**S3 — `min_size` is misnamed and overloaded** (B5).

**S4 — the anchor buffer: KEEP on both routes, for different reasons.**
*(resolved 2026-08-03, user)*

Its primary purpose is **anti-forgetting for discovered modes**, which is a
conditional-route concern: modes are discovered per condition and can be lost.
Unconditionally, the canonical anchor set is either the target distribution
itself (toy problems) or locally gradient-optimised from exhaustive random
sampling (real crystal problems) — so the basins are pre-discovered and there is
little to forget.

But it still earns its place unconditionally: **thermal widening from the
forward policy can surface adjacent or shoulder density that qualifies as a
fresh anchor.** That is a genuine discovery channel even when the basin set is
complete, because a basin's *shoulder* is not part of an argmin-seeded archive.

So: not conditional-only, and not a deletion candidate. What remains true is
that its five gating constants are **arbitrary** in the strict sense
(user-confirmed: "not empirically tested, not really") — and the regime that
would most stress them (conditional, low mean prior quality) has not run yet.
The right time to calibrate them is when that route is live.

## 8. Open questions

1. **Answered.** Does the anchor buffer improve prior composition? *Depends
   strongly on the problem*: conditional routes have very low mean prior quality
   (so the archive matters), unconditional typically fairly high (so it matters
   less, and its value is the shoulder-discovery channel in S4).
2. **Open.** `admit_cap_max: 30 / admit_cap_min: 8 / h0: 10` — the *shape* of
   the health modulation is well argued, the three constants are unmeasured.
   User: "they seem reasonable. Would be better to derive even just
   qualitatively." A qualitative derivation is the deliverable here, not a
   battery.
3. **Open, and live — but the framing has changed.** `mean_lifetime: 100` (prior)
   vs `mean_residence_steps: 50` (replay). "Sufficient freshness" now has a
   *derived* target on the replay side: B8 says hold `λτ` under 1, i.e.
   `resid_vs_intake` above 0.368, and B7a proves **intake rate is the only lever
   on λτ — buffer size cancels out**. So this is no longer a number to pick but a
   setpoint to servo toward (`module_modulators.md` §3). The prior buffer has no
   equivalent sensor and its `mean_lifetime` remains arbitrary.
4. **Answered.** `weighted_bwd_sampling` is semi-obviated by the weighting
   already present in buffer churn itself — so B2's min-max normalisation is
   low-priority, but `weighted_bwd_beta` remains largely dead surface.
5. *(new, 2026-08-08)* **Does prioritisation pay on the real integrand?** B7
   establishes unbiasedness exactly and refutes the variance argument *on a
   clustered synthetic*. The case where it should pay is a gradient dominated by
   tail rows — `1 + CV²(|δ|·‖g‖)` — which that test does not exercise and which
   §0a bounds at only ≈1.7 for replay. `rep_kappa1` (rb0808 arm 16) is the first
   run long enough to say. **The honest prior is "modest or nil."**
6. *(new, 2026-08-08)* **Is the positive half stable enough to draw from?**
   `is_elig_frac` fell 0.74 → 0.33 in 1500 local steps. If that is the trend
   rather than a transient, the one-sided draw runs out of population and the
   whole scheme needs a two-sided variant or a different intake.

---

*Warrant classes: **derived** · **measured** · **inherited** · **arbitrary** · **contested**.*
