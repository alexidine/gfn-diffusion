# Module: buffers (`buffer.py` + management in `train.py`)

Pass 1 (audit + rationalize). Verified against the working tree, 2026-08-03;
revised 2026-08-08 for the prioritised-replay package (`to_do_rebuild.md`
§B5/§B7b/§B7c) and the memorisation sensor (§B7d), which shipped 2026-08-07;
**revised 2026-08-10** for D5 (admission's cap/temperature retired — see B0);
**revised again 2026-08-10** for the `floor`/`stalled` eviction retirement
(see B10) and the untested `prioritise.symmetric` draw option (see B11).
Unconditional route. `ConditionLogZTracker` also lives in `buffer.py` but is
covered in `module_metrics.md` — it is a statistics tracker, not a buffer.

> **Numbering.** `B1…B11` here are **module-local finding IDs**.
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
| Intake | prior-model draws within `ramp_floor` of Emin(c); anchor backfill | **unconditionally uniform** over the sane pool (D5, done) | surprise-gated screen → confirm |
| Draw | uniform (or `_loss_weights` under `weighted_bwd_sampling`) | **two regimes** (B7): uniform, *or* `p ∝ δ₊^κ` with IS weights | — |
| Eviction | relative quantile (bottom 25% of visited) | hazard + backstop, **both residual-independent** (below) | periodic thin; otherwise permanent |
| Size | 10k–250k | 4000 | ≤ 200k |
| Consumed by | `bwd` training | `replay` training | nothing directly — it refills `prior` |
| Manager | [`manage_prior_buffer`](train.py:3876) | [`manage_replay_buffer`](train.py:4387) | [`screen_and_admit_anchors`](train.py:4826) |

**Intake, draw and eviction are now three separate axes.** Admission is
always uniform (D5: `admit_cap_max/min/health_h0/admit_temperature` are
retired keys — `utils.py` `_RETIRED_KEYS` rejects them at load). The draw is
config-gated by `buffers.replay_buffer.prioritise.enabled` (B7): unset means
a uniform draw, set means `p ∝ δ₊^κ` with the IS correction (or `p ∝ |δ|^κ`
under the untested `prioritise.symmetric`, B11). Eviction no longer depends
on that flag at all — `floor`/`stalled` (the old residual-dependent purge
causes it used to gate) are retired outright, not merely switched off, so
eviction is unconditionally hazard + backstop regardless of `prioritise`
(B10).

**Replay management is stage-gated, and the gate is derived, not configured.**
`manage_replay_buffer` returns immediately — before the residual arithmetic and
before the `flow_states` D2H, which is the part that costs — unless the current
stage has a consumer for the buffer (`replay_in_play`, `train.py`). The three
consumers are the fused replay branch, the ray probe (it draws from replay), and
`z_calibration.mode: replay`. A VarGrad-only protocol has none of them, so it
never builds a replay buffer at all.

The branch test is the engine's own `mode_boostable('replay')` with one
correction: `Stage.active_modes` counts a pinned mode by the *presence* of its
key, so `pinned: {replay: 0.0}` reads as boostable. `replay_in_play` reads the
pin by value. There is no config switch — a protocol that trains replay cannot
be starved by a stale key, and one that never does pays nothing.

Being wrong in the OFF direction is bounded and self-healing: intake is
supply-paced, so a stage that does want replay fills it from its own first fwd
steps rather than inheriting a warm buffer from the stage before, and rows
surviving from an earlier managed stage are aged out by the backstop on the
first managed call. This is *not* the tsched_july24 failure — there the buffer
emptied **during** a stage that was drawing from it, because intake was paced
demand-side. Nothing about the gate changes intake while a stage draws.

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

**Replay — two causes, neither shapes the bulk age distribution, neither
depends on `prioritise`** *(revised again 2026-08-10 — B10: `floor`/`stalled`
retired outright, not gated)*:

| Cause | Depends on the residual? |
|---|---|
| `hazard` — memoryless: evict `n/τ` rows per call, uniformly at random | no |
| `backstop` — hard ceiling at `backstop_mult·τ` (default 5τ), binding on ~exp(−backstop_mult) of rows | no (age) |

There is no gate any more — hazard + backstop run unconditionally in
[`manage_replay_buffer`](train.py:4387), whether or not `prioritise.enabled`
is set. The old residual-dependent causes (`floor`: `ema_loss` fell below a
threshold; `stalled`: drawn ≥ `toxic_min_draws` and not improving) are gone —
code, config keys, and telemetry all removed, not merely disabled — B10.
**The reason is the force spectrum, not convenience.** Every selective step
multiplies into the buffer's density — `μ_buf ∝ Q_admit · p_admit · p_survive` —
and the draw's importance weight divides by the **draw** only. A
residual-dependent `p_survive` therefore re-enters Φ uncorrected, counting the
residual twice and once stale (`to_do_rebuild.md` §B4/§B7b). `hazard` and
`backstop` are independent of the residual, so `p_survive` drops out of the
weight and they survive. The displacement purge (headroom top-up beyond
hazard/backstop eviction) is uniform-random unconditionally too, for the same
reason admission is: there is no cap left to score it against.

**The same argument applied to `p_admit`, and that is what B7b changed, now
finished (D5):**

| | admission | draw | weight | Φ vs δ |
|---|---|---|---|---|
| pre-B7b (retired) | `∝ softmax(\|δ\|/T)` | uniform | none | **superlinear**, uncorrected |
| current | uniform, unconditionally | `∝ δ₊^κ` *(if `prioritise`)* | `(1/n_elig)/p` | **linear**, unbiased |

B7b moved the prioritisation from admission to the draw and made it *correct* —
the IS weight divides by the draw, so the estimator is unbiased for the
uniform-over-eligible mean. The cost is that the old, formally-wrong pipeline
delivered **more** force to high-δ regions, precisely because its admission bias
multiplied into the density and was never divided out. The estimator got right
and the signal got weaker (`F-003`). That trade is no longer a per-arm config
choice — the softmax-admission path is deleted, not merely unused.

So the open question is not whether B7b is correct — it is. It is **whether
uniform-over-eligible is the target wanted.** Unbiasedness is a property relative
to a target, and B7b silently moved the target from "the hard tail" to "the
buffer". `is_elig_frac ≈ 0.40` says the estimator's actual support is the
positive-δ 40% — a partial restoration of the old bias, but far weaker than
`softmax(|δ|/2.0)`.

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

**The anchor → prior feedback path** is the buffer system's only closed loop, and
it is one number: `reach = 1 − quantile_0.9(excess) / margin`, firing below
`reach_threshold: 0.75`. Unconditionally this reduces to *"is the buffer's 90th
percentile within 25% of `ramp_floor` of the best known energy."* Pooling
excess-above-own-condition-best is what makes it correct in the conditional case,
where one condition's easier scale must not mask another. Recorded so the
coupling is visible — nothing is wrong with it.

## 4. Findings

Current state only. Measurements live in [`findings.md`](findings.md); history in
git. `B`-ids are retained as stable anchors for existing cross-references.

### B0 — `admit_temperature` is retired (D5, done 2026-08-10)

Admission is now unconditionally uniform over the sane pool
(`manage_replay_buffer`, `train.py`) — the softmax-over-`|resid|` path is
deleted, not config-gated. `admit_cap_max`, `admit_cap_min`,
`admit_cap_health_h0`, and `admit_temperature` are retired keys
(`utils.py` `_RETIRED_KEYS`); a config that still sets any of them fails at
load with a message naming the replacement. The displacement purge (headroom
top-up beyond toxic eviction) is uniform-random too now, for the same reason —
there is no cap left to score it against.

**Historical note, for reading old runs.** Every arm before this date that did
**not** set `buffers.replay_buffer.prioritise.enabled` ran the softmax
admission path at whatever `admit_temperature`/`admit_cap_*` its config
specified — that includes all 8 `local_aug08` arms and 22 of 26 rb0808 arms.
`admit_reward_min` is unaffected (optional, `train.py`, constrains `sane`
upstream of admission on every path, retired or not).

Two traps that are now moot but explain old rb0808 results:

1. **The docstring at old `train.py:3204` was wrong** for that era: it said
   "None/<=0 disables it", but `kappa: 0.0` returns `float(0.0)`, which `is not
   None` — so `{enabled: true, kappa: 0}` engaged the **whole** B7b package.
   rb0808 arms 7, 19, 20 were set that way.
2. **rb0808's replay block was heterogeneous in admission policy.** A `rep_hi`
   (24) vs `rep_b7b` (7) comparison varied frac dose **and** admission policy
   together — irrelevant to any run started after this retirement, since there
   is now only one admission policy.

### B0a/B0b — uniform intake was a trade, not an upgrade — `F-003`

Buffer hardness comes from **admission, not eviction**. Uniform intake fits the
typical population better (`bwd/tb_err` −0.50) and leaves the forward tail
uncorrected (`fwd/tb_err` +4.40, stable). Standing constraints, now that uniform
admission is the only mode (B0):

- **κ ≈ 1 is the practical ceiling** — the IS variance bound bites before κ = 2.
- **`max_size` is not a lever here** — the displacement purge is not the mechanism.
- **Keep the IS weights** — removing them is worse than any κ setting.
- **Keep Huber** — de-huberising costs ~6.6 nats.

**The discriminating arm is retired, not merely unreachable.** Before D5,
`prioritise.enabled` gated intake and draw together, so "old (softmax)
admission + new (κ) draw" — the one configuration that would have isolated
admission's contribution from the draw's — could not be expressed by config.
D5 didn't split the flag to make that arm reachable; it deleted the old
admission side outright, so the comparison can no longer be run going
forward. `F-003`'s measurement stands as a historical A/B (old admission vs
new), not as a currently-selectable config choice.

**`bounds.replay: [0.05, 0.45]` saturates** under this construction — pinned at
0.45 from ~step 3450 with `rt_rho` still above setpoint — so the ratio controller
loses authority.

### B2 — `_loss_weights` min-max normalises

[buffer.py:855](../buffer.py:855) maps `[min, max] → [0, 1]` before applying the
temperature, so (a) a converged buffer whose loss spread has collapsed still gets
full-strength prioritisation, and (b) one outlier compresses everyone else toward
uniform. Rank-based or robust-quantile normalisation is stable against both.
**Dormant on the unconditional route** — `weighted_bwd_sampling` is per-stage and
`naive` does not set it.

### B3 — replay occupancy equilibrium sits exactly at `max_size`

Occupancy is designed to be emergent (`n = admit_rate × τ`), with `max_size` a
memory guard that should not bind. mk_dev runs `churn_rate: 80`,
`mean_residence_steps: 50`, `max_size: 4000` — and 80 × 50 = 4000 exactly.
Headroom is zero at full admission efficiency. It probably does not bind, because
`churn_rate` budgets admission *attempts*, but if it ever does, `max_size` rather
than the hazard starts shaping eviction at the margin. Raise to ~2× equilibrium
or state that the guard is meant to be tight.

### B4 — retired-key guards fired hours into a run — **RESOLVED**

Both retirement guards used to live in the manage path, which first runs at a
stage transition, so a stale config was rejected hours in. The aug02 battery
lost all 16 arms' entire phase 1 (1.1–7.8 h each) to exactly this.

`max_residence_steps`, `toxic_min_draws` and `toxic_delta_threshold` are now in
`utils._RETIRED_KEYS`, so `preflight_config` rejects them at LOAD, and the
runtime guards are deleted rather than kept alongside. The gate is also stricter
than the guards were: it fires on key PRESENCE, where the old code tested
truthiness, so `toxic_min_draws: 0` no longer slips through.

Recorded as project state 3 (`config_state.CHANGES`), which also gives the
affected configs a repair path they never had — see B10.

### B5 — `min_size` does two unrelated jobs

`init_fraction` is inert. `min_size: 10000` is a per-cycle chunk bound in
`rebuild_prior_by_churn` ([train.py:4025](../train.py:4025)) *and* a sampling
count in `grow_prior_buffer` ([train.py:5009](../train.py:5009)). It is not a
minimum size anywhere. Rename or split.

### B7 — the prioritised draw, as built — `F-005`

Unbiasedness is exact at every κ and the variance payoff does not exist; see the
finding. Four design points that are load-bearing and easy to get wrong:

1. **δ is reconstructed, not stored** — `δ = log Z − log w`, from `ema_logw`.
   `ema_loss` cannot serve: it stores `|resid|`, and the *sign* is what a
   one-sided priority needs.
2. **One-sided by design** — `δ₊ = max(δ, 0)`. A row the policy has moved off has
   a fallen `log_pf` and takes priority ~0 automatically, which is both the
   intended replay/backward split and most of what the §B8 drift term was for.
3. **Zero-priority rows are EXCLUDED, not floored.** A row drawn at probability
   ~0 carries weight ~∞ and owns a self-normalised batch. `δ₊ = 0 ⇒ p = 0`, and
   the estimator targets the uniform mean over the **positive half**.
4. **A supplied `p` is drawn with replacement** ([buffer.py:311](../buffer.py:311)).
   IS correctness assumes iid draws from the design measure; without-replacement
   is both wrong and a crash source once the eligible pool falls below the batch.

`floor_frac` default is **0.25**, the measured knee. Point 2's one-sidedness
has an **untested** symmetric alternative — see B11.

### B10 — `floor`/`stalled` eviction is retired, not gated (2026-08-10)

[`manage_replay_buffer`](../train.py:4387) no longer has a residual-dependent
purge path at all. `floor` (`ema_loss` fell below a threshold) and `stalled`
(drawn ≥ `toxic_min_draws` and `ema_loss − birth_loss ≥ 0`) are deleted — the
code, the `toxic_min_draws`/`toxic_delta_threshold` config keys, and the
`self.replay_cohort` `'absorbed'`/`'stalled'` telemetry (and the
`replay_buffer_absorbed_frac`/`replay_buffer_stalled_frac` wandb metrics they
fed) are all gone. Eviction is unconditionally hazard + backstop (§3);
`replay_buffer_backstop_frac`, `replay_buffer_hazard_frac`, and the
`expired_*` hazard-cohort metrics still exist and are unaffected.

Setting `buffers.replay_buffer.toxic_min_draws` or `toxic_delta_threshold` in a
config is refused at LOAD by `utils._RETIRED_KEYS`, alongside the
`max_residence_steps` retirement (B4). 103 tracked configs still set
`toxic_min_draws` and 106 set `max_residence_steps`; they were never patched,
because they are closed-out historical battery runs and this protocol treats git
as their log, not the configs.

They are no longer a dead end, though. Project state 3 carries a migration:
`toxic_min_draws`/`toxic_delta_threshold` are dropped mechanically, since the
purge they fed is gone and dropping them changes nothing.

`max_residence_steps` is deliberately **not** dropped — it is `manual`, so
`migrate` reports it and refuses `--write` until a human resolves it. It was
REPLACED by `mean_residence_steps`, and the overlap between the two across
tracked configs is **zero**: dropping it would leave the buffer with no residence
setting, `tau` reads 0, and the `if tau > 0` branch that arms both the hazard and
the backstop never runs — a config that loads clean, trains, and still reports a
healthy `replay_buffer_age_cv` ≈ 1, because the surviving displacement purge is
itself memoryless. A hard cap and the mean of an exponential are not the same
number, so the value cannot carry across mechanically.

Two reasons `floor`/`stalled` had to go, both recorded at the call site
(`manage_replay_buffer`'s docstring) and already covered in §3: under a
prioritised draw a corrected row already has `p ≈ 0`, so a residual-dependent
purge was only reclaiming memory, not gradient budget — hazard reclaims the
same memory without reintroducing a residual-dependent survival probability.
And a row driven to `δ ≈ 0` by repeated replay **is** B8's memorisation
sensor's positive evidence (`birth_loss` vs `ema_loss`); purging it early
destroyed the signal rather than merely wasting variance budget.

### B11 — `prioritise.symmetric`: an untested draw-eligibility alternative

[`prioritised_weights`](../buffer.py:915) gained a `symmetric: bool = False`
parameter. Default (`symmetric=False`) is exactly B7's existing one-sided
draw, unchanged: `score = delta_plus = max(δ, 0)`. `symmetric=True` swaps the
score for `score = |δ|`, which admits **under-weighted** rows (`δ < 0`) into
the eligible set instead of zeroing them — everything downstream (nan fill,
relative floor, `κ` power, IS weight) is unchanged; only which rows have
`score > 0` differs.

Two motivations, both from the code:

- **F-003** measured that the one-sided draw leaves the forward tail
  uncorrected (`fwd/tb_err` +4.40, stable) while fitting the typical
  population better. `symmetric` is the untested alternative that widens the
  eligible set to include that tail rather than excluding it.
- **Absorbing starvation** (one-sided mode only): `ema_logw` is written only
  at draw time, so a row that falls to `δ ≤ 0` has its estimate frozen and can
  only re-cross zero via `log_Z` drifting back up past the stale value — an
  escape that closes once `log_Z` converges, which is the expected end state.
  This is a plausible mechanism behind `is_elig_frac`'s monotonic drift (F-003
  measured 0.74 → 0.33 over 1500 steps): a one-way exclusion ratchets, it does
  not equilibrate. `symmetric` closes the trap as a side effect — `elig =
  |δ| > 0` is false only at exact float equality, so virtually no row is ever
  hard-excluded.

**Status: config-reachable, untested.** `mk_dev.yaml` sets
`prioritise.symmetric: true` (`configs/mk_dev.yaml:418`) and so do
`configs/prod0810/{0..6}.yaml`; no isolation arm has run with it yet, so
neither motivation above is confirmed on-policy — both are read off the
docstring's reasoning, not a measurement. Read `replay/is_elig_frac` and
`replay/is_ess_frac` on the first run that carries it: `symmetric` trades a
narrower, harder-prioritised eligible set for a wider, softer one, so `κ ≈ 1`
being the practical ceiling (B0a/B0b) is not guaranteed to transfer.

### B8 — the memorisation sensor — `F-006`

[`absorption_stats`](../buffer.py:863) compares each row's current residual
against the one it was admitted with — both already stored, no new field:

```
ratio       = mean(ema_loss) / mean(birth_loss)     in (0, 1]
absorbed    = 1 - ratio
lambda_tau  = -ln(ratio)
```

`ratio = 1` is a pure delay line. Falling toward 0 means residents were corrected
at their own trajectories while intake did not move — memorisation by definition.
**Setpoint `ratio = 1/e = 0.368`, derived, so it transfers across problem, `T`
and buffer size.**

Undrawn rows have `ema_loss == birth_loss` and contribute `ratio` 1 — correct, a
row nothing trained on cannot have been memorised. Returns `{}` below 8 valid
rows, so pre-schema buffers make the servo hold at cold start rather than act on
garbage.

### Anchor health gate — bars are named after their role

Reads `anchor_buffer.health_gate_floor_metric` (lower bound) and
`health_gate_ceiling_metric` (upper bound on |value|), against
`health_gate_floor` / `health_gate_ceiling`. Renamed from `health_gate_r2` /
`health_gate_zerr`, which encoded the *old metric names* into the bar keys.

**The D9/N3 upgrade to `tb_err_worst` is deliberately not taken.** No bar
transfers across a ruler swap, and the incumbent is better warranted:

| | `tb_resid_clipped @ 0.5` (shipped) | `tb_err_worst @ ?` |
|---|---|---|
| shape | signed, bounded by `beta` | unbounded RMS, floored at `std(log w)` |
| healthy value | ~0 | **18–21 on this route** |
| warrant | **derived** — the D29 Z-currency invariant | none yet; needs a battery |

Stays on the docket as a question, not a pending edit.

## 5. Warrants

| Choice | Warrant | Evidence |
|---|---|---|
| Memoryless hazard over hard TTL | **measured** | `expired_delta` ran −12…−28 nats across postfix_july30 arms — the TTL was culling the *improving* tail; `absorbed_frac` 0.000 meant age was doing 100% of eviction |
| `floor`/`stalled` retired outright, not gated (B10, retired D5-era) | **derived** — a corrected row already draws at `p ≈ 0` under `prioritise`, so the residual-dependent purge was reclaiming memory, not gradient budget, and it was destroying B8's memorisation evidence in the process | manage_replay_buffer docstring |
| `admit_temperature: 2.0` (retired D5, historical) | **measured**, on the softmax-admission path before it was deleted (B0) | T=2 beat T=5; T=20 diverged (replay_july26 / tsched_july24) |
| Residual-independent intake *and* purge, unconditionally (not just under `prioritise`) | **derived** — `μ_buf ∝ Q·p_admit·p_survive` and the IS weight divides by the draw only, so any other residual dependence enters Φ uncorrected | `to_do_rebuild.md` §B4/§B7b; B10 |
| Draw *with replacement* whenever `p` is supplied | **derived** — IS correctness assumes iid draws from the design measure; without-replacement also crashes once the eligible pool falls under the batch | B9(b), reproduced at 1500 / 900 / 300 eligible |
| `δ₊ = 0` rows excluded rather than floored | **measured** — a uniform floor put `max(w)` at 10⁴ and one row owned the batch | B7 |
| `floor_frac: 0.25` | **measured** — the ESS knee on a live buffer; the shipped 0.01 gave `is_ess_frac` 0.02–0.06 | B7 table, 2026-08-07 |
| `κ` as a ladder rather than a setting | **measured, against the design's own prediction** — variance rises with κ on a clustered integrand, so the ladder is diagnostic | B7; `to_do_rebuild.md` §B7c |
| `ratio = 1/e = 0.368` as the memorisation bar | **derived** — `ratio ≈ exp(−λτ)` puts `λτ = 1` exactly there, so it transfers across problem, T and buffer size | B8 |
| Mean-shift over a histogram distance for the sensor | **measured** — 1-D Wasserstein matched it to 3 dp on all 33 arms | B8 |
| Clip-then-divide (cap before T) | **derived** — keeps `cap` an absolute nats bound independent of T | docstring |
| Health-modulated cap off `fwd/scatter_err` | **derived** — an *external* signal cannot ratchet off the buffer's own contamination | docstring |
| `loss_floor=+inf` in prior eviction | **measured** — a finite bar silently stopped churn | docstring at [train.py:3898](train.py:3898) |
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
2. **Closed by deletion (D5, 2026-08-10).** `admit_cap_max/min/h0`'s health
   modulation no longer exists — admission is unconditionally uniform, so
   there is no cap left to shape. The unmeasured-constants question is moot
   rather than answered.
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
