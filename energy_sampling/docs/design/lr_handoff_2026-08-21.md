# LR control — session record and implementation plan, 2026-08-21

**Read this first if you are picking up learning-rate control.** It is the
single source: what was tried today, what it measured, why the direction
reversed, and what to build. Sections 1–5 are the narrative and the evidence;
6–8 are the work.

Companions, not prerequisites: `docs/design/lr_probe_protocol.md` (the spec for
the approach that was retired today — read only for the reasoning, not as
instruction), `docs/hypergradient_review.md` (standing sensor assessment),
`docs/lr_control_session_2026-08-20.md` (the measurements sections 1 and 5 rest
on).

---

## 1. What the day set out to do

The 2026-08-20 review left three things on the table. Within a stage the optimal
learning rate is constant — variogram flat from 500 to 40,000 steps, the mean
pinned to ±7% — so no within-stage servo is warranted. The non-stationarity that
could matter is between stages and across loss-composition changes. And neither
shipped sensor measures it: `ray` has never fired outside phase 2, and `hyper`'s
sign is uninformative once a surface has equilibrated.

The plan adopted was a discrete, checkpointed probe. At calibration points, fork
short trials at candidate rates, run them on data harvested verbatim from the
preceding live steps, score on a held-out slice, restore, and resume at whichever
rate won. Training on recorded batches meant no rollouts and no energy calls,
which is what made it look affordable.

## 2. What was built

A decision layer with no torch dependency (`lr_race.py`), a measurement-level
simulator for its gates (`bench/race_sim.py`), the trainer-side probe
(`lr_race_probe.py`), and wiring into `train.py` behind an opt-in config block.
67 unit tests for the new code, 250 in `bench/` overall, all passing.

Three results are worth keeping whatever happens to the method:

- **Snapshot/restore is bitwise.** A duplicate arm re-run from the snapshot
  reproduced the incumbent exactly (`duplicate_spread 0.000e+00`) on CPU *and*
  on the GPU. The GPU nondeterminism floor was the single largest risk to the
  design — a floor rivalling the effect size would make the validity gate veto
  every reading — and on the elj route it is zero.
- **Isolation holds.** The probe consumes no RNG the run needs, asserted at
  runtime rather than assumed.
- **The harvest works.** Trajectories already leave every live step as
  `loss_dict['flow_states']`, so recording them costs a read. The records park in
  host memory and are replay-scoreable. **This is the piece that carries
  forward** — see section 6A.

## 3. What it measured, and where it broke

Seeded-rate tests on elj phase 1: set the rate deliberately 8× cold or 4× hot,
and ask what the probe recommends.

| seed | W=10 (raced step 300) | W=30 (step 300) | W=100 (step 608) |
|---|---|---|---|
| baseline 1.25e-4 | hold | move ×4 ✗ | hold |
| 8× cold | move ×4 | move ×16 | move ×16 |
| 4× hot | expand up ✗ | move ×0.25 | expand up, then ×2 ✗ |

✗ marks a verdict wrong in *direction*, not merely in size.

**The window sweep is confounded.** The larder needs `replicates × W + holdout`
batches, so the window length sets how late the first calibration can fire —
step 300 at W=10 and W=30, step 608 at W=100. Only the first two columns are a
controlled comparison. Any future sweep must hold the calibration step fixed and
report it beside every verdict.

**The recommendation is wrong in closed loop.** At W=30 the probe asks for a 4×
raise from the baseline rate; a run actually seeded at that rate has its own
reference arm diverging by step 300. So it recommends a rate that destabilises
the route — and it oscillates: up from baseline, down from where it just put you.

## 4. The structural problem

A trial trains on a fixed set of recorded trajectories with fixed rewards. That
is supervised regression toward frozen targets, and its optimal step size is set
by the curvature and noise of that finite-sample problem. Live training is not
that problem: the policy generates its own data, so the learning rate governs how
fast the target distribution moves, and in a GFlowNet that coupling is most of
the dynamics rather than a correction.

So the probe optimises a *different function*, whose optimum sits elsewhere. On
recorded data there is no penalty for drifting off-distribution, and a faster
rate fits the stored targets better right up until numerical divergence — left
alone, the method converges toward the stability edge rather than the operating
point.

Window length does not fix this; it trades between two wrong answers. Short
windows are nearly pure "fit the targets faster" and favour high rates. Long
windows let the policy drift from the recorded distribution, and since the
held-out slice is drawn from *that same recorded distribution*, drift is
penalised — but that penalty punishes moving away from the old distribution,
which is exactly what training is supposed to do. The crossover has no principled
setting, which is what the non-monotonic table in section 3 is.

**The cheapness and the invalidity were the same property.** Never regenerating
data is what removed the rollout and energy cost, and it is what removed the
on-policy feedback.

## 5. Why the conclusion is to go back to `ray`

The 2026-08-20 review reads easily as "both existing sensors are untrustworthy".
What it established is narrower.

For `hyper`: the *sign* of the gradient correlation is uninformative once the
iterate has equilibrated, so using it as an error signal for an integrator rails.
The *magnitude* is a real reading — `|cos| ≈ ηλ/2` estimates distance to the
stability edge — and that is a legitimate guard.

For `ray`: the sensor passed everything it was tested against (149 of 152
readings resolved; its slope test came out indistinguishable from a perfectly
tracked fixed optimum). What was broken was the *response* around it — acting on
each reading when the variogram says successive readings are white noise around a
constant, plus an up/down gain asymmetry that parks the loop ~15% hot.

Ray's blindness is one-step: it cannot see noise accumulating over future steps,
which is what `alpha_target` covers as a deliberate margin. That is a bounded
defect on the right quantity. Measuring a different function is not.

Ray's real gap was coverage — it draws from replay and scores replay
coefficients, so it skips every stage that does not train replay TB, which is why
phase 1 has never been measured. **The harvest built today closes exactly that
gap.**

---

## 6. Cruise control: generalise `ray`, replace the servo

Four changes. A and B are the substance; C and D are cleanup. This is what the
calibration ramp of section 7 hands over to.

> **BUILD STATUS, 2026-08-22.** A and B are **built and verified on real runs**;
> C needs nothing here but section 11 raises a measurement about it; D is
> **blocked** on `lr_race{,_probe}.py` reaching version control (section 10).
> Section 7's ladder is **built and unit-tested**, wired behind an opt-in
> `adaptive_lr.ramp` block, with its own validation runs reported in section 11.
> New modules: `lr_larder.py` (harvest + scorer), `lr_pool.py` (the estimator,
> no torch), `lr_ramp.py` (the ladder, no torch), `lr_ramp_probe.py` (its
> trainer side). New tests: `test_lr_larder.py`, `test_lr_pool.py`,
> `test_lr_ramp.py`, `test_lr_ramp_driver.py`, plus additions to
> `test_ray_probe_gate.py`, `test_warmup_ramp_freeze.py` and
> `test_config_invariants.py`. See section 11.

### A. Generalise what `ray` scores to the loss the stage actually trains

Today `_draw_probe_batch` (`train.py`) is hardwired to a replay draw and returns
`None` otherwise, and `RayCalibration` skips when it gets nothing. Replace the
source with a larder draw per **active** branch and score each branch's own
coefficient bank on its harvested trajectories at each alpha.

Per stage that yields: `train_prior` → the bwd/MLE bank on harvested bwd
trajectories; `var_conditioning` → fwd and bwd VarGrad (records carry
`condition_id`, so the grouping survives); `equilibration` → all three fused
losses.

- **Actuate on the composite, weighted by the live frac weights.** The optimizer
  takes one step along the fused gradient, so alpha* must be measured against the
  objective that step descends. A per-branch alpha* is the optimum for a
  direction nobody took.
- **Use the CURRENT weights at each calibration** (decided this session), not
  weights frozen at the last one. A composition change resets the pool (B), so
  pooling stays coherent without freezing the reading.
- **Log per-branch alpha* as a diagnostic — it is free.** Each branch is already
  evaluated at each alpha to form the sum. Branch disagreement (fwd wanting 4×
  while replay wants 0.25×) is worth surfacing.
- **The gradient-path problem does not apply here.** `RayCalibration` runs under
  `no_grad` and needs loss *values* only, so the dropped reparameterised P_B path
  gradient that affects *training* on stored trajectories is irrelevant. Value
  agreement is what `test_periodic_scoring.py`'s roundtrip tests assert.
- **Assert the fwd-bank caveat at arm time.** Scoring the fwd branch goes through
  `get_gfn_backward_loss`, which has no counterpart for `z_level`, the
  condition-grouped `emp_z` branch, `reward_grads`, or `traj_grads`. All are 0 in
  the canonical banks; assert rather than assume, and refuse the branch if not.
- Cost: roughly 3× the forward passes on a fused stage, still zero rollouts and
  zero energy calls.

### B. Replace the servo with a pooled estimator

Pool `log(live_lr × alpha*)` — the implied one-step optimum, and the quantity
invariant to what the controller did — within a regime, with exponential
forgetting. Set the operating rate to the pooled optimum divided by
`alpha_target`. Move only when the pooled estimate and the live rate differ by
more than a threshold tied to the pooled standard error. Symmetric in both
directions; otherwise hold.

This addresses both recorded defects directly: the ±17% hold sawtooth was the
loop chasing white noise, and the persistent hot offset was `eta_up 0.25` against
`eta_down 0.5` rectifying that noise into drift. An estimator over a window still
tracks genuine drift — it just requires evidence to move.

Reuse from this session: regime keying (stage × loss composition), the
composition-change trigger (L1 distance ≥ 0.2 on the normalised frac vector), the
settling gate (`z_cal/p < 0.1` sustained over consecutive observations, with a
step floor for stages publishing no such signal), and the discipline of logging
per-reading evidence rather than verdicts.

### C. `alpha_target`

Leave at 4. It is a deliberate conservative undershoot, not an unmeasured
constant. Worth measuring per route in sensor-only mode later — pin the rates as
explicit floats at a known-good hand-tuned rate and read off the alpha* reported
— mainly to know how much margin is being bought. **See 8(a): the ramp's accept
criterion in section 7 is expressed in different units and currently disagrees
with this by about 4×.**

### D. Keep, and delete

**Keep:** the harvest and larder (`RaceLarder`, the three branch tees in
`train.py`, host offload via `copy.copy(batch).cpu()` — PyG's `.cpu()` mutates in
place); the regime and settling gates; the warmup-freeze bar change in
`controller.py` (`warmup_freeze_cos_bar`, default −0.25, replacing a threshold at
exactly 0.0 that fired on noise and pinned stages at a fraction of their
configured rate); `hyper` as a magnitude-only edge guard.

**Delete:** the frozen training race — trial windows, arms, screen/confirm, the
step-down fallback, the near-tie collapse, and the snapshot/restore machinery
(`RayCalibration` has its own param clone/restore). That is most of `lr_race.py`
and much of `lr_race_probe.py`. Salvage the larder and the gates out of the
latter first.

---

## 7. Initial calibration: the checkpointed ramp

Status: approved implementation direction. Scope: general GFN LR control across
routes and stages. Supersedes earlier ramp-until-explosion and
protected-catapult language.

### Objective

Select a high but non-brittle initial learning rate without waiting for numerical
divergence. For TB training, define the operational boundary as the first loss of
local update margin or on-policy coherence, not merely the first NaN or exploding
loss.

**The ramp is real on-policy training. Do not substitute frozen multi-step
candidate trials.**

### Ramp protocol

1. Save a complete restartable checkpoint: model, optimizer/controller state,
   replay state where applicable, and RNG state.
2. Begin from a conservatively safe LR.
3. Increase LR in geometric rungs, initially `×1.5`.
4. Do not reset optimizer moments between rungs.
5. Remain at each rung for a configurable minimum residence period sufficient for
   sampler and replay feedback to appear.
6. Collect several ray readings and live diagnostics during the assessment
   window.
7. Pool readings only within the current LR, stage, loss composition, and
   optimizer branch. Reset the pool whenever any of these changes.
8. Classify the rung as `clean`, `pending`, `boundary`, or `hard_failure`.
9. After every clean rung, save it as the latest clean checkpoint.
10. At the first boundary or hard failure, restore the latest clean checkpoint and
    begin cruise one geometric rung below its LR.

With a `×1.5` ramp this normally places cruise about `2.25×` below the first
rejected rung — not the arbitrary `2–5×` cut previously applied after an
explosion.

If the configured maximum LR is reached cleanly, report `lower_bound_only`. Do
not claim the stability boundary was found: either extend the search explicitly,
or cruise one rung below the maximum and request later recalibration.

If the initial LR is already rejected, descend geometrically until a clean rung
is established.

### Ray margin

Ray evaluates the current stage's actual composite loss on paired data, with
enough multipliers to resolve the region around the live step, initially:

    0, 0.5, 1, 1.5, 2, 4

`alpha = 0` is the pre-update model; `alpha = 1` is the realized update.

Prefer a directly observed bracket for the no-improvement crossing
`L(alpha_edge) = L(0)`. A quadratic estimate may supplement this only when its
fit and bracketing gates pass. Treat above-range and below-range results as
censored bounds, not point estimates.

Initial ray-margin requirement:

- the pooled paired result at `alpha = 1` must improve over `alpha = 0`; and
- the estimated or bounded no-improvement crossing should remain at least
  `1.5–2×` beyond the live step.

Use robust pooling and persistence, not independence assumptions or nominal
p-values. A single adverse or unresolved reading must not reject a rung.

### TB coherence

A finite TB loss is not sufficient evidence of healthy training. Assess
persistent changes across available metric families:

- TB-residual median and upper-tail quantiles
- reward or energy quantiles
- validity, terminal diversity, or effective sample size
- replay-versus-live discrepancy
- `logZ` and other major loss-component behavior
- per-branch loss, gradient, update, and clipping behavior

Distribution movement is not itself failure. Reject a rung only when movement
represents persistent adverse collapse, or when multiple metric families agree
that training has lost coherence. Route-specific metrics may be supplied through
adapters, but the core controller must remain route-general.

### Classification

- `clean` — ray margin passes, no hard failure, on-policy coherence acceptable.
- `pending` — evidence insufficient or ray unresolved; extend the dwell without
  increasing LR.
- `boundary` — ray margin persistently below the required headroom, or persistent
  on-policy coherence deterioration.
- `hard_failure` — nonfinite values, runaway parameters or optimizer state, or
  another explicit emergency condition.

Initial persistence rules and metric thresholds are working assumptions and must
remain configurable and logged.

### After calibration

Hand control to the pooled sample-and-hold ray controller of section 6B. Periodic
ray readings maintain local LR headroom; stage changes, material loss-composition
changes, or persistent sampler drift request a new checkpointed ramp.

Hard failures always roll back immediately. Hypergradient signals remain
telemetry-only unless separately validated.

Do not deliberately continue beyond a detected boundary in search of catapult
recovery. One rejected rung is sufficient to bracket the operational boundary.

### Required implementation properties

- General across crystal and conformer GFN workflows; not a conformer-only
  facility.
- Separate controller and pooling state by optimizer branch.
- Preserve the actual loss composition and on-policy sampling behavior of the
  active stage.
- Restore the complete checkpoint after a rejected rung.
- Log rung LR, residence time, ray bounds, classification reasons, coherence
  diagnostics, and selected cruise LR.
- Unit-test state transitions, pool resets, censored ray readings, full rollback,
  initial-hot recovery, and maximum-LR-without-boundary behavior.
- Validate on at least one cold-start and one hot-start training run before giving
  the ramp automatic authority on expensive TB training.

---

## 8. Open reconciliation items

Not objections to the direction — keeping the measurement on real on-policy
training is what removes the structural fault of section 4. These are places
where sections 6 and 7 have to be reconciled with each other and with the
existing system. **(a) changes behaviour if left alone.**

**a. The ramp's accept margin and the cruise setpoint are in different units,
and disagree by roughly 4×.** The margin asks that the no-improvement crossing
sit `1.5–2×` beyond the live step. Under the quadratic model that crossing is
`2 × alpha_star`, so the requirement is `alpha_star ≥ 0.75–1.0` — cruise at about
the one-step optimum. The pooled ray controller that takes over targets
`alpha_target = 4`, a deliberate undershoot, so it will want to cut ~4×
immediately after handover. Either express the ramp's accept criterion in
`alpha_star` against the same target, or state that the handoff is deliberately
hot and cruise is expected to settle down from it. As written the two phases pull
against each other, and it will look like controller instability rather than a
units mismatch.

**b. Per-rung checkpoints must not use the run's own tag namespace.** `best` is a
hardlink to whatever `running` last wrote, and the emergency rewind path reads
`best` and `stage_start` to choose its target — so a ramp writing either edits
the run's own rollback target. Use a dedicated ramp namespace or hold the clean
rung in host RAM, and respect `checkpoint_read_only`.

**c. Decide whether the ramp runs through the stage-entry transient or after
it.** The ramp is the natural instrument for traversing that transient, but ray
readings taken during it describe the transition rather than the stage — which is
why the settling gate (`z_cal/p < 0.1`) exists. If the ramp runs through the
transient, rung classification is reading a moving target and the residence rule
in step 5 is carrying more weight than it appears to.

**d. "Do not reset optimizer moments" interacts with residence.** Adam's second
moment is an EMA with a horizon near `1/(1-beta2)`. If residence per rung is
shorter than that, each rung is read partly through the previous rung's moments,
so the reading lags the rate. Residence should account for the moment timescale
alongside sampler and replay feedback.

**e. Budget the ramp before building it.** Pooling resets per rung (step 7), and
ray's measured per-reading noise is 0.20–0.25 dex, so resolving a margin takes
several readings per rung. Cost is residence × rungs, and unlike the frozen-trial
design a rejected rung's steps are genuinely discarded on rollback. Better to
state a target ramp cost up front than to discover it.

**f. The alpha grid changes ray's response geometry.** The shipped grid is
log-spaced by design, so the controller's response is proportional to
log-distance and saturates at the grid edge rather than extrapolating. The
proposed `0, 0.5, 1, 1.5, 2, 4` is deliberately finer near the live step, which
suits a margin measurement — just note that any code assuming a doubling grid
(bracket construction, `sqrt(lo*hi)` midpoints) needs revisiting with it.

---

## 9. Pitfalls recorded today (do not rediscover)

- `report_losses=True` alone is not enough to replay a step: loss dicts carry
  `flow_states` and `log_r` but not `condition`, `mol_batch` or `log_T_tensor`, so
  harvesting must tee inside the branch step functions, not around them.
- `get_gfn_forward_loss` has no `trajectories` parameter; every branch replays
  through `get_gfn_backward_loss`, which reads `mle`/`pf_boost` without a
  `getattr` guard and so raises on the fwd bank.
- Stage transitions only fire from `evaluation()`, so a config with a large
  `eval_period` never reaches the phase-1→2 boundary.
- `device: cpu` does not keep a run off the card — `buffer_device` is a separate
  key defaulting to cuda (~10 GB reserved). For genuinely GPU-free work set both,
  plus `CUDA_VISIBLE_DEVICES=-1`; the empty-string form leaves
  `torch.cuda.is_available()` True with `device_count()` 0, which crashes the VRAM
  ledger in `train.py`.
- The GPU preflight guard refuses a second training process on the card. Respect
  it; overriding it is the co-tenancy that preceded two blue screens on this
  machine.

## 10. Status

- **6D IS DONE.** `lr_race.py` and `lr_race_probe.py` reached version control in
  commit `2833bd5`, so the salvage has history and the delete could proceed.
  Removed: `lr_race.py`, `lr_race_probe.py`, `bench/race_sim.py`,
  `bench/test_lr_race.py`, `bench/test_race_probe.py`, and 18
  `configs/race_*.yaml` -- 23 files. `train.py` loses `_build_race_probe`, the
  `race_probe` attribute and its report merge.
- **A stray `lr_probe` block fails AT LOAD**, via a new config invariant
  (`lr_probe_is_retired`) rather than `utils._RETIRED_KEYS`. Nothing reads the
  key any more, so leaving it unguarded would make it a silent no-op -- the
  failure mode this whole file keeps recording. It is an INVARIANT and not a
  retirement because the retired-key gate requires a matching migration, which
  moves `project_state_version` and forces every config in the tree to be
  restamped -- for a key no tracked config outside the deleted
  `configs/race_*.yaml` ever carried. State 9 made the same call for
  `batch_util_target` and its record says why: a load-time gate is "the honest
  shape" when there is nothing to transform.
- **One thing 6D said to delete was KEPT, deliberately**: the snapshot/restore
  machinery. 6D reasoned that `RayCalibration` has its own param clone/restore,
  which is true for CRUISE -- but section 7's rollback needs the model AND the
  stepping optimizers' moments AND the RNG, which that clone does not carry. It
  is salvaged into `lr_ramp_probe.RampDriver`, which is also how section 8b's
  "hold the clean rung in host RAM" is implemented.
- The canonical config carries no `lr_probe` block and loads clean; the probe is
  off by omission and has never actuated.
- `bench/` is 250 tests green as of the end of the 2026-08-21 session.

---

## 11. What the 6A/6B build measured, 2026-08-22

### Verified

- **`ray` measures phase 1.** `raycal_phase1_smoke` (latent_gaussian, CPU,
  `train_prior` declaring `lr_sensor: {kind: ray}`): **11 calibrations, 0
  skipped / 0 deferred / 0 refused**, on a bwd/MLE stage the sensor had never
  produced a single reading on across 19 prior runs.
- **The composite and its per-branch diagnostics work on a fused stage.**
  `raycal_fused_smoke`, 9 calibrations, larder 2201 records over three branches.
  One reading: composite alpha* 8 (below_range), **bwd 8, fwd 4, replay 2.83
  (bracketed)** -- the branch disagreement 6A predicted would be worth
  surfacing, and it is free.
- **The pooled estimator holds where the servo sawtooths, and the two disagree
  by 7.6x.** One key apart (`adaptive_lr.calibration.mode`), same seed, same
  data, 620 steps of `train_prior`, 31 calibrations each:

  | | pooled | servo (retired v8) |
  |---|---|---|
  | moves applied | **2** | **25** |
  | readings pooled | 17 | n/a -- acts on each |
  | final `peak_scale` | 0.154 | 0.0203 |

  The servo's trace is the recorded pathology, live: it moves on nearly every
  reading and oscillates in a +-20% band about a slowly falling centre
  (0.0221 -> 0.0263 -> 0.0313 -> 0.0221 -> ... -> 0.0287 -> 0.0203 -> 0.0241 ->
  0.0203). The pooled arm sees the SAME alternation -- alpha* 8, 2, 2, 4, 4, 2,
  4, 2 -- and holds every one of them, then moves twice when the pooled estimate
  clears its own standard error.
- **The settling gate holds out the transient, and the transient is biased.**
  The pooled arm pooled NOTHING from its first 14 calibrations (steps 20-300),
  which read alpha* 4, 1, 2, 1.4, 1, 1, 1, 1.4, 2, 1, 2, 2, 1, 1 -- systematically
  low, i.e. "too hot". The servo acted on all of them and had cut 45x before the
  stage had settled. That is section 8c's concern, demonstrated rather than
  argued.
- **The probe now perturbs nothing.** Two runs identical but for the phase-1
  `lr_sensor`, with `lr_servo_managed` empty so no rate moved: **334 of 349
  summary metrics bit-identical**. The 15 that differ are wall-clock timings
  plus `lr_ctrl/peak_scale`, which the control arm does not apply. This is
  F-039's own test, and it could not have passed under the replay draw -- that
  draw consumed NumPy RNG nothing restored. `Larder.take` is deterministic.

### Found during the build, not anticipated by this plan

- **The alpha grid tests one entry FEWER than `ray_calibration.py` claimed.** A
  grid point is tested only when its double is also on the grid, so
  `{0,1,2,4,8}` tests alpha* against `{1,2,4}` -- not `{0.5,1,2,4}` as the
  module docstring said. Corrected there. It matters for 8(f): the lowest
  TESTED alpha is the grid's second entry, so a rate hotter than that reads as
  a censored bound and never as a point.
- **`var_conditioning` cannot be ray-measured as 6A assumed.** Its fwd bank runs
  `emp_z: 1.0` through the condition-grouped path, and `get_gfn_backward_loss`
  ASSERTS against `emp_z` under `vg_by_condition` (and reaches an undefined
  `log_Z_emp` without it). 6A says "assert rather than assume, and refuse the
  branch if not", so the calibration refuses with `branch_refused` -- but 6A
  also expects that stage to yield "fwd and bwd VarGrad". **Open decision:**
  score fwd there with `emp_z` zeroed and log the omission, or leave the stage
  on `hyper`. Scoring it zeroed is not obviously wrong -- `emp_z` trains the
  flow head, which the ray holds fixed anyway -- but its value still varies
  along the ray through `log_Z_emp`, so it is not a free omission either.
- **6B attributes the parking offset to the eta asymmetry with the sign read one
  way; the arithmetic reads it the other.** `eta_down 0.5 > eta_up 0.25` makes
  an up-down pair a net CUT (0.917x on the measured 2.83/5.66 alternation), so
  peak_scale walks DOWN and alpha* equilibrates ABOVE target. Solving
  `0.25(mu+s) + 0.5(mu-s) = 0` at `s = log10(5.66/4)` gives **mu = +0.050 dex of
  alpha***, inside `raydrift`'s predicted +0.043..+0.059 band and consistent
  with every run holding alpha* 4.57-5.47 against target 4. So the offset is
  real and the mechanism is right; "hot" in 6B means hot IN ALPHA*, which is a
  ~12% COLDER learning rate than `alpha_target` names. Pinned in
  `test_lr_pool.py`.
- **Censored readings needed handling 6B does not specify.** v8 used a bound as
  a point estimate; for `below_range` that overstates alpha* and licenses a
  hotter rate than the evidence supports. The pool takes bounds as one-sided
  hinges instead -- a bound pulls only when the estimate violates it. With only
  bounds in the window the minimiser is an interval and the estimate is the
  point of it nearest the incumbent, i.e. the smallest move the evidence
  requires.

### The ramp, end to end on a cold start (`ramp_cold`)

Every stage of section 7 fired in order, on real on-policy training:

| step | what happened |
|---|---|
| 20-280 | 14 calibrations inside the stage transient. Ramp not started, nothing pooled. |
| 300 | Settling gate opens. `ramp: armed by stage_change at peak_scale 1; residence 100 steps (config); budget to the ceiling ~20 rungs / 2000 steps, of which 100 are discarded on the rejected rung` |
| 300-380 | Rung 0 dwells its residence. Coherence families resolve: `bwd_ess, bwd_loss, bwd_tb_residual, bwd_tb_upper_tail` -- all four live, so the gate is not inert on this route. |
| 400 | Rung 0 `clean` -> host-RAM snapshot -> climb to peak 1.5. |
| 520 | Rung 1 `boundary` (`ray_margin_-0.301dex_x2`) -> ROLLBACK, model restored, `cruise_scale 1.0`. |
| 540+ | Cruise takes over from an EMPTY pool -- correct, the ramp's readings belonged to its rungs -- and refills. |
| 780 | The pooled estimator's first cruise move, peak 1.0 -> 1.431 on 14 readings. |

The budget line and the residence attribution (`config`, because the smoke sets
`adam_beta2: null`) are section 8e and 8d reporting themselves.

**⚠ FINDING: the ramp's rung classification is noisier than the controller it
hands to, and by construction.** Cruise raised the rate to 1.43 -- ABOVE the 1.5
rung the ramp had just rejected, and within one geometric rung of it. Look at
what rejected that rung: readings 8, 8, 4, 4 followed by two `below_range` bounds
at alpha* < 2. With per-reading noise measured at 0.233 dex on this route, TWO
readings is a much thinner sample than the ~14 cruise pools, so `persistence: 2`
lets noise reject a rung that the same sensor, averaged, would have accepted.

The rejection is not a bug -- the pool did exactly what it should, and the two
recent upper bounds are genuine evidence -- but the SETTING is wrong.
**`persistence` (or the per-rung `min_readings`) has to be chosen from the
route's measured per-reading noise, not left at its default**, or the ramp will
stop early and cruise will spend its first window undoing that. Both are
configurable and logged, which section 7 requires; neither has been calibrated.

A second thing that showed up in the same rung, which the plan does not cover:
**contradictory censored bounds**. `above_range` at alpha* > 8 and, four
readings later, `below_range` at alpha* < 2 cannot both hold. The convex
objective has no zero-violation point and settles where the squared violations
balance, which under exponential forgetting is nearer the recent bound. That is
the behaviour to want from a tracker, and it is now pinned in `test_lr_pool.py`
rather than left emergent.

### The real crystal route: elj/mipcas phase 1 on the GPU (`elj_raycal_phase1`)

The toy validated the mechanism; this is the route `mk_dev` runs.

- **16 calibrations on `train_prior`, 0 skipped / 0 deferred / 0 refused.** `ray`
  measuring phase 1 on elj, which is what 19 prior runs never did.
- **The pooled estimator converged and then held.** Held through the transient
  (calibrations 1-2), then cut `peak_scale` 1.0 -> 0.25 -> 0.0625 -> 0.031 ->
  0.021 -> 0.031 over calibrations 3-12, then **held for the last 5** while
  alpha* read 8, 4, 4, 2, 2. Final gap -0.111 dex inside a 0.124 dex bar.
- **⚑ The per-reading noise replicates across routes.** `lrpool/sd` **0.22 dex**
  on elj, 0.22-0.23 on the latent_gaussian toy, against `raydrift`'s published
  0.20-0.25 measured on 19 elj runs by a different method. Three independent
  estimates agreeing is the strongest evidence yet that the noise floor is a
  property of the sensor rather than of a route.

**⚠ AND THE COST IS 4-5x WHAT THE FUSED-STAGE FIGURE SUGGESTS.** Measured here:
a calibration costs **~28 training steps** on `train_prior` (median step 0.158 s;
calibration step 4.5 s), which is **5.6% at period 500** against the **1.2%**
recorded for the same probe on the fused stage.

The absolute cost is similar -- `n_sub x len(alphas)` = 64 forward passes over
one batch. What changed is the DENOMINATOR: a bwd/dataset step runs no rollout
and no energy call, so it is ~20x cheaper than a fused one, and the same probe is
a much larger fraction of it. (The bwd bank's `repeats: 2` also doubles the rows
scored relative to replay's `repeats: 1`.)

**Fixed, because one global period cannot serve both stages:**
`lr_sensor: {kind: ray, period: N, n_sub: M}` now overrides
`adaptive_lr.ray_calibration` FOR THAT STAGE. Absent keys keep the global value,
so nothing changes for existing configs. To run `ray` on elj phase 1 under 2%,
`period: 1500` there (or `n_sub: 4` at period 750).

### The ramp on elj, with the REAL residence floor (`elj_ramp_phase1`)

Same route, `adam_beta2: 0.999` this time, so section 8d's floor binds and says
so: `residence 1000 steps (adam_moments)`, budget `~20 rungs / 20000 steps, of
which 1000 are discarded on the rejected rung`. All four coherence families
resolve here too.

**⚠ AND IT FOUND A REAL INEFFICIENCY IN SECTION 7'S DESCENT RULE.** Rungs 0 and 1
both returned a margin of **exactly -0.602 dex** -- that is `log10(4)`, the
signature of a `below_range` reading at the BOTTOM of the alpha grid. The censored
statement is "alpha* < 1", i.e. "the optimum is at most a quarter of this rate".

Section 7 says "descend geometrically", and taken literally the ramp moved 1.5x
per rung -- 1000 steps of real training each -- against evidence that already said
at least 4x. Getting from `peak_scale` 1.0 to the ~0.031 the pooled estimator
independently found would have taken ~9 rungs and **~9,000 steps** for a cut the
estimator made in one move.

**Changed: the descent now takes the LARGER of one geometric rung and what the
margin licenses.** One rung remains the floor, so a marginal rejection behaves
exactly as before; a censored bound is evidence the run has already paid for, and
down is the safe direction to be wrong in. Measured immediately after the change,
same config, same seed:

| | as specified | evidence-driven |
|---|---|---|
| rung 0 (1.0) rejected -0.602 dex | -> 0.667 | **-> 0.25** |
| rung 1 rejected -0.602 dex | -> 0.444 | (already past it) |
| steps to cover a 4x cut | ~3,000 | **1,000** |

Pinned in `test_lr_ramp.py`, both the jump and the one-rung floor.

**⚑ AND THE SAME RUN CAUGHT A BUG THE UNIT TESTS COULD NOT SEE.** The descent
factor was written as a local named `step`, which SHADOWED `_reject`'s `step`
parameter -- so `_enter(nxt, step)` stamped the rung's entry at 0.25 instead of
at step 1000. Every rung after the first then read `resident = step_ind - 0`,
the residence gate never bound, and the ramp terminated on a spurious
`no_evidence` timeout 400 steps into a 1000-step rung.

The unit tests were blind to it by construction: they use short residences and
`persistence: 1`, so the BOUNDARY short-circuit fires before residence is ever
consulted. **This is exactly what section 7's "validate on at least one
cold-start and one hot-start training run before giving the ramp automatic
authority" is for**, and it earned its place on the first real-route run.

Fixed, and pinned by two tests that were checked to FAIL with the bug
re-introduced and pass without it. A second, smaller thing fell out of the same
investigation: `outcome` was stamped on every rejection, so a ladder still
descending reported `boundary` while it was mid-descent. It is now set only on
the paths that finish.

After the stage transition a SECOND ramp armed itself, `armed by
composition_change` -- section 7's "stage changes, material loss-composition
changes... request a new checkpointed ramp", firing on its own.

**⚠ AND IT THEN ARMED AGAIN 30 STEPS LATER, and would have kept doing so.** On
the fused stage the balance controller nudges the branch fracs every tick, so the
L1 composition trigger fires repeatedly -- and each firing DISCARDS the ramp and
restarts it. The ramp could never have completed a rung, and every rung it did
start would have cost real training for nothing.

The trigger was inherited from the POOL, where a reset costs nothing. For the
ramp it does not. Section 7 asks for a new ramp on stage changes, MATERIAL
composition changes, or PERSISTENT drift -- so the re-arm is now rate-limited: a
stage change always re-arms (unambiguous), a composition change may not restart a
ramp younger than one residence, and refusals are counted as
`ramp/rearms_suppressed` so a starved ramp and a quietly working one are
distinguishable. Pinned in `test_lr_ramp_driver.py`.

### Hot start (`ramp_hot`): the recovery path, and an independent cross-check

Same config, seeded ~10x hot. There is no clean checkpoint to roll back to, so
section 7's descent path runs instead -- and the ray margin closes monotonically
all the way down:

| rung | peak_scale | verdict |
|---|---|---|
| 0 | 1.0 | `boundary`, margin **-0.602 dex** -> descend |
| 1 | 0.667 | `boundary`, **-0.490** -> descend |
| 2 | 0.444 | `boundary`, **-0.398** -> descend |
| 3 | 0.296 | `boundary`, **-0.249** -> descend |
| 4 | 0.198 | dwelt 200 steps, not 100 -- `pending` extended the dwell |
| 5 | 0.132 | `boundary`, **-0.147** -> descend |
| 6 | 0.132 | **`clean`** -> climb to 0.198 |

Two things worth keeping:

- **The margin closes at ~0.11 dex per 0.176 dex rung, i.e. a slope near 0.63
  rather than 1.** A fixed optimum would give exactly 1. The shortfall is the
  optimum MOVING as the model trains, which is what a ramp on a fresh model
  should expect and what section 8c's residence rule is really up against.
- **⚑ TWO INDEPENDENT CONTROLLERS CONVERGED ON THE SAME RATE.** The pooled
  cruise estimator, run separately at the same `seed_lr` with no ramp at all,
  settled at `peak_scale` **0.154**. The ramp -- a different mechanism, on a
  different run, from the opposite direction -- established its first clean rung
  at **0.132** and climbed to **0.198**, bracketing it within one geometric
  rung. Nothing in the two paths shares state, so this is a real cross-check on
  the sensor rather than on either controller.

### ⚑ THE RESULT THAT COMPLICATES ALL OF THIS: the settling gate has a cost

**On this toy the servo trained better than either pooled arm**, and the reason
is not the one it first looks like.

| arm | `alpha_target` | final `peak_scale` | when it first moved | `eval/wass_debiased` |
|---|---|---|---|---|
| servo (retired v8) | 4 | 0.0203 | **step 40** | **0.0025** |
| pooled | 4 | 0.154 | step 340 | 0.0378 |
| pooled | 32 | 0.0147 | step 340 | 0.0267 |

The obvious reading is "alpha_target 4 is too small here", and the third arm was
run to test exactly that -- a single-key change with a predicted direction. The
direction held: `alpha_target: 32` landed 10.5x colder, close to the 8x the
setpoint change implies. **But the quality did not follow.** Pooled@32 ends
COLDER than the servo (0.0147 vs 0.0203) and still trains 10x worse. A 1.4x rate
difference cannot produce a 10x difference in wass, so the rate is not what
separates them.

**It is the first 300 steps.** Both pooled arms held `peak_scale` at 1.0 for the
whole stage transient, because that is exactly what the settling gate is for --
`lrpool/n` is 0 until step 300 in both. The servo acted from step 40 and was at
0.088 by step 120. On this route peak 1.0 is ~50-70x hot, so the pooled arms
spent 300 steps being cooked before their first move, and never recovered inside
620 steps.

**What this does and does not say:**

- It does NOT vindicate the per-reading servo. The transient reading is
  systematically biased toward "too hot" (alpha* 4, 1, 2, 1.4, 1, 1, 1, ... on
  these arms), so acting on it CUTS -- which is right whenever the seed is hot
  and wrong whenever it is cold. The servo was not correct here; it was biased in
  the direction that happened to help.
- It DOES say section 8c is a live tension rather than a settled call, and that
  the resolution taken in `lr_ramp.py` (start the ramp AFTER the transient) does
  not cover a badly-set seed. The ramp is the instrument for traversing a bad
  seed, and it is currently gated behind the very transient that makes a bad seed
  expensive.
- The gate's cost is BOUNDED and VISIBLE: `MIN_STAGE_STEPS` (300) plus the
  `z_cal/p` run, and `lrpool/holding_on_transient` publishes it.

**Owner decision, and it changes behaviour if left alone.** Three options:

1. **Let the ramp start during the transient**, with rung classification that
   expects a moving optimum. Most work; addresses the actual problem.
2. **Allow ONE conservative cut during the transient** -- act only on a resolved
   reading below target, only downward, at most once per stage. Cheap, keeps the
   "no servo" property, covers the hot-seed case (which is the damaging one) and
   not the cold-seed case (which is merely slow).
3. **Accept it** and require a sane seed. Defensible on a route whose seed is
   already hand-tuned; indefensible on a new one.

Recommended: (2), with (1) as the durable answer. Nothing here is implemented --
the pooled estimator holds through the transient as built.

Note also that this whole comparison is ONE SEED on a toy at batch 64, evaluated
at a single point. The section 6B claims that ARE established are about the rule:
2 moves against 25, the sawtooth held, the transient excluded, and a 7.6x
difference in where the two rules park.

### Section 8, item by item, as of this build

| | resolution |
|---|---|
| **a** units mismatch | RESOLVED. The ramp's accept criterion is now `alpha* >= alpha_target`, the same quantity cruise steers to. `lr_ramp.py` DECISION 8a. |
| **a(ii)** cruise backoff | DECIDED. Cruise starts AT the latest clean rung; the old "one rung below" double-counted the margin. `cruise_backoff_rungs: 1` restores it. |
| **b** checkpoint namespace | RESOLVED by its own second option: the clean rung lives in HOST RAM. Nothing is written to `best`/`stage_start`, so the emergency rewind's target is untouched. |
| **c** transient | DECIDED. The ramp starts AFTER the transient, on the settling gate the pool already uses. Verified: `ramp_cold` armed at step 300, not before. |
| **d** Adam moments | IMPLEMENTED. `residence = max(configured, 1/(1-beta2))`, and `residence_bound_by` records which bound bound. |
| **e** budget | IMPLEMENTED. `projected_cost` is printed at arm time -- rungs, steps, and the steps discarded on the rejected rung. |
| **f** alpha grid | NO CODE CHANGE NEEDED, and the reason is the docstring correction above. `_bracket` iterates the grid and tests a point only when its double is on it, and `sqrt(lo*hi)` is defined for any pair -- so nothing assumes DOUBLING, and the proposed `0, 0.5, 1, 1.5, 2, 4` simply tests alpha* against {0.5, 1, 2}. It is usable as written. |

### Two things the 6D delete surfaced

`bench/old/test_lr_controller.py` is SLOW-tier (it imports torch), so `pytest -m
fast` never ran it -- and it pins twelve behaviours of the shipping controller
that nothing else pins. Running the whole bench after the delete turned up two
real questions, not just stale expectations:

- **A saturated sensor now climbs 8x per period, where the servo climbed 1.68x.**
  When the truth is above the grid every reading returns `above_range` pinned at
  the largest testable alpha (32 on the shipping grid). That says "alpha* > 32"
  and nothing more. The servo applied `(32/4)^0.25 = 1.68` and UNDER-used the
  bound; the pooled estimator reads it for what it is and licenses 8x. More
  faithful, and more dangerous -- the binding constraint on this route is the
  EXPOSURE WINDOW between calibrations, and a 12x-hot excursion was measured
  unrecoverable in ~30 steps (`mvwsu5d5`), with the divergence bars never firing.
  The servo's slowness was, accidentally, a safety property.

  **Added: `max_raise_dex`, default log10(4).** A single RAISE is capped; a CUT
  is not. This is not the v8 asymmetry returning -- that was an asymmetric GAIN,
  which biases the FIXED POINT because symmetric noise rectifies into drift. A
  cap on the STEP SIZE leaves the fixed point exactly where it was; it binds only
  while the estimate is far from the incumbent and vanishes as the loop
  approaches. Both properties are pinned.

- **A censored reading missing `lo`/`hi` was SILENTLY REJECTED.**
  `RayCalibration` sets `alpha_star` TO the bound for `above_range`/`below_range`
  and never extrapolates past it, so the two agree by construction -- but a
  caller filling only `alpha_star` had every censored reading dropped, and the
  pool looked merely quiet rather than starved. It now falls back to
  `alpha_star`, and still refuses a reading with no usable bound at all.

### Still owed

- **`hyper` as a magnitude-only edge guard (6D "keep").** NOT built. It is
  currently a sign-driven integrator (`peak_scale *= exp(beta*cos)`), and
  section 5's argument is that the sign is uninformative once equilibrated while
  `|cos| ~ eta*lambda/2` is a real distance-to-edge reading. Converting it needs
  an actuation rule section 6 does not give: what a large `|cos|` should DO, and
  by how much. Owner call.
- **`persistence` and per-rung `min_readings` for the ramp** have to be set from
  the route's measured per-reading noise. See the `ramp_cold` finding above.
- **`alpha_target` per route.** See below.
- **`max_raise_dex` has not been measured**, only reasoned. log10(4) is one
  grid doubling either side of the target; the right value is a property of the
  exposure window (`period` x step cost) and should be measured per route.

### Behaviour changes a reader should know about

- `adaptive_lr.calibration.mode` defaults to **`pooled`**. `servo` restores the
  retired per-reading rule for comparison. Absent key = pooled.
- `replay_in_play` no longer returns True merely because a stage declares
  `ray`: the sensor no longer draws from replay, so it no longer forces a
  buffer the stage may not train.
- `configs/mk_dev.yaml` still carries comments saying `ray` "draws from replay
  and scores replay_loss_coeffs, so it is incoherent in a bwd stage". That is
  no longer true and the file is owner-controlled, so it is flagged rather than
  edited. The same sentence is copied through ~20 generated configs under
  `configs/`, which are non-authoritative and were left alone.
