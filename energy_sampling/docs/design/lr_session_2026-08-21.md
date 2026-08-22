# LR control — session record and handoff, 2026-08-21

Companion to `docs/design/lr_probe_protocol.md` (the v2 spec, rev d) and
`docs/hypergradient_review.md` (the standing sensor assessment). This file is
the narrative: what was built, what it measured, why the conclusion reversed,
and what the next agent should build instead. The technical plan is section 6.

---

## 1. What the day set out to do

The 2026-08-20 review left three things on the table. Within a stage the optimal
learning rate is constant (variogram flat from 500 to 40,000 steps, mean pinned
to ±7%), so no within-stage servo is warranted; the non-stationarity that could
matter is between stages and across loss-composition changes; and neither
shipped sensor measures it, because `ray` has never fired outside phase 2 and
`hyper`'s sign is uninformative once a surface has equilibrated.

The plan adopted was a discrete, checkpointed probe: at calibration points, fork
short trials at candidate rates, run them on data harvested verbatim from the
preceding live steps, score on a held-out slice, restore, and resume at whatever
won. Training on recorded batches meant no rollouts and no energy calls, which
is what made it look affordable. That design is specified in
`docs/design/lr_probe_protocol.md`; it was reviewed adversarially over several
rounds and the spec is sound as a description of the thing it describes.

## 2. What was built

A decision layer with no torch dependency (`lr_race.py`), a measurement-level
simulator for its gates (`bench/race_sim.py`), the trainer-side probe
(`lr_race_probe.py`), and wiring into `train.py` behind an opt-in config block.
67 unit tests, 250 in `bench/` overall, all passing.

The machinery works and several parts are worth keeping regardless of the
verdict on the method:

- **Snapshot/restore is bitwise.** A duplicate arm re-run from the snapshot
  reproduced the incumbent exactly — `duplicate_spread 0.000e+00` — on CPU and,
  more importantly, on the GPU too. The GPU nondeterminism floor was the single
  largest risk to the design (a floor rivalling the effect would make the
  validity gate veto every reading), and on the elj route it is zero.
- **Isolation holds.** The probe consumes no RNG the run needs, asserted at
  runtime rather than assumed.
- **The harvest works.** Trajectories already leave every live step as
  `loss_dict['flow_states']`, so recording them costs a read; the records park
  in host memory and are replay-scoreable.

## 3. What it measured, and where it broke

Seeded-rate tests on elj phase 1 — deliberately setting the rate 8× cold or 4×
hot and asking what the probe recommends:

| seed | W=10 (raced step 300) | W=30 (step 300) | W=100 (step 608) |
|---|---|---|---|
| baseline 1.25e-4 | hold | move ×4 ✗ | hold |
| 8× cold | move ×4 | move ×16 | move ×16 |
| 4× hot | expand up ✗ | move ×0.25 | expand up, then ×2 ✗ |

✗ marks a verdict wrong in *direction*. Two things came out of this.

**The window sweep is confounded.** The larder needs `replicates × W + holdout`
batches, so the window length sets how late the first calibration can fire —
step 300 at W=10 and W=30, step 608 at W=100. Only the first two columns are a
controlled comparison. Any future sweep must hold the calibration step fixed and
report it beside every verdict.

**The recommendation is wrong in closed loop.** At W=30 the probe asks for a 4×
raise from the baseline rate; a run actually seeded at that rate has its own
reference arm diverging by step 300. So it recommends a rate that destabilises
the route, and it also oscillates — up from baseline, down from where it just
put you.

## 4. The structural problem

A trial trains on a fixed set of recorded trajectories with fixed rewards. That
is supervised regression toward frozen targets, and its optimal step size is set
by the curvature and noise of that finite-sample problem. Live training is not
that problem: the policy generates its own data, so the learning rate governs
how fast the target distribution moves, and in a GFlowNet that coupling is most
of the dynamics rather than a correction.

So the probe optimises a *different function*, whose optimum sits somewhere
else. On recorded data there is no penalty for drifting off-distribution, and a
faster rate fits the stored targets better right up until numerical divergence —
left alone, the method converges toward the stability edge rather than the
operating point.

The window length does not fix this; it trades between two wrong answers. Short
windows are nearly pure "fit the targets faster" and favour high rates. Long
windows let the policy drift from the recorded distribution, and since the
held-out slice is drawn from *that same recorded distribution*, drift is
penalised — but that penalty punishes moving away from the old distribution,
which is exactly what training is supposed to do. The crossover has no
principled setting, which is what the non-monotonic table in section 3 is.

**The cheapness and the invalidity were the same property.** Never regenerating
data is what removed the rollout and energy cost, and it is what removed the
on-policy feedback.

## 5. Why the conclusion is to go back to `ray`

The 2026-08-20 review is easy to read as "both existing sensors are
untrustworthy". What it actually established is narrower.

For `hyper`: the *sign* of the gradient correlation is uninformative once the
iterate has equilibrated, so using it as an error signal for an integrator
rails. The *magnitude* is a real reading — `|cos| ≈ ηλ/2` estimates the distance
to the stability edge — and that is a legitimate guard.

For `ray`: the sensor passed everything it was tested against (149 of 152
readings resolved; its slope test came out indistinguishable from a perfectly
tracked fixed optimum). What was broken was the *response* around it — acting on
each reading when the variogram says successive readings are white noise around
a constant, plus an up/down gain asymmetry that parks the loop ~15% hot.

And ray's blindness is one-step: it cannot see noise accumulating over future
steps, which is what `alpha_target` covers as a deliberate margin. That is a
bounded defect on the right quantity. Measuring a different function is not.

Ray's real gap was coverage — it draws from replay and scores replay
coefficients, so it skips every stage that does not train replay TB, which is
why phase 1 has never been measured. **The harvest built today closes exactly
that gap**, and that is the piece of this work worth carrying forward.

---

## 6. Technical plan for the next agent

Four changes. A and B are the substance; C and D are cleanup.

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
  takes one step along the fused gradient, so alpha\* must be measured against
  the objective that step descends. A per-branch alpha\* is the optimum for a
  direction nobody took.
- **Use the CURRENT weights at each calibration** (decided this session), not
  weights frozen at the last one. A composition change resets the pool (B), so
  pooling stays coherent without freezing the reading.
- **Log per-branch alpha\* as a diagnostic — it is free.** Each branch is
  already evaluated at each alpha to form the sum. Branch disagreement (fwd
  wanting 4× while replay wants 0.25×) is worth surfacing.
- **The gradient-path problem does not apply here.** `RayCalibration` runs under
  `no_grad` and needs loss *values* only, so the dropped reparameterised P_B
  path gradient that affects training on stored trajectories is irrelevant.
  Value agreement is what `test_periodic_scoring.py`'s roundtrip tests assert.
- **Assert the fwd-bank caveat at arm time.** Scoring the fwd branch goes
  through `get_gfn_backward_loss`, which has no counterpart for `z_level`, the
  condition-grouped `emp_z` branch, `reward_grads`, or `traj_grads`. All are 0
  in the canonical banks; assert rather than assume, and refuse the branch if
  not.
- Cost: roughly 3× the forward passes on a fused stage, still zero rollouts and
  zero energy calls.

### B. Replace the servo with a pooled estimator

Pool `log(live_lr × alpha*)` — the implied one-step optimum, and the quantity
that is invariant to what the controller did — within a regime, with exponential
forgetting. Set the operating rate to the pooled optimum divided by
`alpha_target`. Move only when the pooled estimate and the live rate differ by
more than a threshold tied to the pooled standard error. Symmetric in both
directions; otherwise hold.

This addresses both recorded defects directly: the ±17% hold sawtooth was the
loop chasing white noise, and the persistent hot offset was `eta_up 0.25`
against `eta_down 0.5` rectifying that noise into drift. An estimator over a
window still tracks genuine drift — it just requires evidence to move.

Reuse from this session: regime keying (stage × loss composition), the
composition-change trigger (L1 distance ≥ 0.2 on the normalised frac vector),
the settling gate (`z_cal/p < 0.1` sustained over consecutive observations, with
a step floor for stages that publish no such signal), and the discipline of
logging per-reading evidence rather than verdicts.

### C. `alpha_target`

Leave at 4. It is a deliberate conservative undershoot, not an unmeasured
constant. Worth measuring per route in sensor-only mode later — pin the rates as
explicit floats at a known-good hand-tuned rate and read off the alpha\* it
reports — mainly to know how much margin is being bought.

### D. Keep, and delete

**Keep:** the harvest and larder (`RaceLarder`, the three branch tees in
`train.py`, host offload via `copy.copy(batch).cpu()` — PyG's `.cpu()` mutates
in place); the regime and settling gates; the warmup-freeze bar change in
`controller.py` (`warmup_freeze_cos_bar`, default −0.25, replacing a threshold
at exactly 0.0 that fired on noise and pinned stages at a fraction of their
configured rate); `hyper` as a magnitude-only edge guard.

**Delete:** the frozen training race — trial windows, arms, screen/confirm, the
step-down fallback, the near-tie collapse, and the snapshot/restore machinery
(`RayCalibration` has its own param clone/restore). That is most of `lr_race.py`
and much of `lr_race_probe.py`. Salvage the larder and the gates out of the
latter before removing it.

### Pitfalls recorded this session (do not rediscover)

- `report_losses=True` alone is not enough to replay a step: loss dicts carry
  `flow_states` and `log_r` but not `condition`, `mol_batch` or `log_T_tensor`,
  so harvesting must tee inside the branch step functions, not around them.
- `get_gfn_forward_loss` has no `trajectories` parameter; every branch replays
  through `get_gfn_backward_loss`, which reads `mle`/`pf_boost` without a
  `getattr` guard and so raises on the fwd bank.
- Stage transitions only fire from `evaluation()`, so a config with a large
  `eval_period` never reaches the phase-1→2 boundary.
- `device: cpu` does not keep a run off the card — `buffer_device` is a separate
  key defaulting to cuda (~10 GB reserved). For genuinely GPU-free work set
  both, plus `CUDA_VISIBLE_DEVICES=-1`; the empty-string form leaves
  `torch.cuda.is_available()` True with `device_count()` 0, which crashes the
  VRAM ledger in `train.py`.
- The GPU preflight guard refuses a second training process on the card. It is
  correct to respect it; overriding it is the co-tenancy that preceded two
  blue screens on this machine.

---

## 7. Status

- `lr_race.py` and `lr_race_probe.py` are **untracked**, while their call sites
  in `train.py` and `controller.py` were captured by checkpoint commits. Nothing
  is broken (the import is lazy inside `_build_race_probe` and only a config
  carrying an `lr_probe` block reaches it), but they need adding to version
  control before the delete in D, so the salvage has history.
- Probe configs written this session are `configs/race_*.yaml`; they are mine,
  not user-owned, and can go with the delete.
- The canonical config carries no `lr_probe` block and loads clean; the probe is
  off by omission and has never actuated.
