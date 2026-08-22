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
> C needs nothing; D is **blocked** on `lr_race{,_probe}.py` reaching version
> control (section 10). New modules: `lr_larder.py` (harvest + scorer),
> `lr_pool.py` (the estimator, no torch). New tests: `test_lr_larder.py`,
> `test_lr_pool.py`, plus additions to `test_ray_probe_gate.py` and
> `test_warmup_ramp_freeze.py`. See section 11 for what was measured and what
> the build found that this plan did not anticipate.

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

- `lr_race.py` and `lr_race_probe.py` are **untracked**, while their call sites in
  `train.py` and `controller.py` were captured by checkpoint commits. **This now
  blocks 6D**: the harvest they carried has moved to `lr_larder.py`, so the tees
  no longer feed `RaceProbe.larder` and a race would defer forever on an empty
  one. Rather than leave that silent, `_build_race_probe` RAISES on an
  `lr_probe` block. Commit the two files and the delete can proceed.
- Probe configs written 2026-08-21 are `configs/race_*.yaml`; not user-owned,
  and they can go with the delete.
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
