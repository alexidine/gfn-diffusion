# Reading runs: the interpretive method

How a GFN run is read here. State: present tense, overwritten in place.

Scope: this is the *interpretive* layer — what to look at, in what order, and what
a shape means. §8 covers the run that is still going; everything else assumes it
has stopped. Metric **definitions** live in `module_metrics.md`; the mechanics of
pulling data out of wandb live in the `local-wandb-reading` memory. Cross-reference
those, do not restate them.

Derived by synthesis over the session transcripts, so it is a model of the practice,
not a spec. Where it contradicts the practice, the practice wins — say so and
overwrite.

---

## 1. Read order

Nine passes. Stop early when something obviously broke; the later passes exist to
tell you *what* broke, not *whether*.

1. **Stage structure.** Did it advance, and when? Phase 1 (MLE) at a few thousand
   steps is normal; >20k is a flag. But a slow MLE whose loss is still descending
   at a decent rate is LR-starved, not broken — check the rate before calling it.
2. **Fit quality.** `tb_err_worst`, `scatter_err`, `r2`. These are *lagging*
   indicators, not leading ones: highly sensitive to policy quality, so when they
   turn they turn fast.
3. **Partition function.** `log_Z_learned` and `fwd/jensen_z` — the bellwethers of
   policy trend and health. Smooth and generally rising is right, provided the
   rise is absorption and not collapse.
4. **Allocation vs. response.** The fwd/bwd/replay frac traces and the LR trace,
   read *against* the metrics they are supposed to be steering. This is the
   controller-did-its-job check and it belongs before any theory about the model.
5. **Variances.** `logw_std_within`, `vg_lb`, the Jensen-vs-logmeanexp gap, and the
   fwd–bwd Jensen gap (`zmatch/delta_worst`).
6. **Sample statistics.** Is it making anything good — mean energy, non-thermal /
   reasonable sample fraction, effective dimension, bounding and reduction energies.
7. **SDE health.** `step_var`, mean drift, variance, rho. Exploding drift or
   variance means the policy is sick; rapid swings usually do too.
8. **Buffers.** Size, mean/median loss and energy, lifetimes, composition and
   absorption fractions.
9. **Cost.** GPU utilisation, batch size, step time. First-class: cluster jobs are
   killed for low utilisation, and a slower run is slower regardless of the
   mechanism story attached to it.

Figures carry texture the scalars miss: TB parity plots, latent terminal and
trajectory distributions, condition-tracker histograms (plotly histograms are
often illegible, hence the separate tracker figures). Then, depending on what is
stuck, the specific mode's regression and its attendant buffer.

## 2. The tiers

**Topline** — the six read first:

- `[fwd|bwd|replay]/tb_err_worst`
- `fwd/over_coverage`
- `bwd/under_coverage_wcen`
- `abs(fwd/tb_resid_clipped)`
- `log_Z_learned`
- `zmatch/delta_worst`

**Level 2:** `[fwd|bwd|replay]/scatter_err` · `[fwd|bwd]/step_var` ·
`[fwd|bwd]/logw_std_within` · `[fwd|bwd|replay]/z_gap`.

**Context:** sample stats, buffer stats, SDE stats, usage stats — all read
conditionally on what the topline showed.

**Low-trust:** `tracker/logw_std_rms` and `tracker/z_bias_rms` are too noisy to
act on. `wass_*` is interpretively useful only on toys where the anchors *are* the
target; on crystal targets it is uncorrelated with the truth.

**Physical sanity (real crystals):** packing coefficient 0.6–0.8; energy tens to
hundreds of kJ/mol below zero; bounding, reduction and density energies at or near
zero. Toy problems have entirely different ranges — there, the profile is the
signal, not the level.

## 3. Standing principles

**R1 — Read the allocation before the metric.** Most "the model got worse" is
"the controller sent the weight somewhere." Frac trace and LR trace first.

**R2 — Confirm the thing ever fired.** A frac below its deactivation threshold, a
gate that never tripped, a knob retired upstream, a servo silent because its
sensor is structurally zero. An inert mechanism is the most common explanation for
a null result, and it invalidates the arm rather than answering it.

**R3 — Separate level from spread.** Nearly every metric confusion is this.
`tb_err` carries both; the `relative_*` and `*_wcen` forms exist to divide out a
log Z offset or a batch-composition bias. When a number moves, ask which half moved.

**R4 — Z is the ruler.** Anything measured against a mis-levelled Z is displacement,
not coverage. An under-coverage spike concurrent with a log Z crash is the crash,
mechanically.

**R5 — Read the gradient, not the loss, at a fixed point.** `tb_resid_clipped` is
the winsorised quantity the optimiser actually sees; raw `tb_resid` sits several
nats positive at a converged Z purely from Huber. Sensor and actuator must share a
fixed point or the loop latches.

**R6 — A smooth curve is often the EMA.** Check smoothing and reporting cadence
before reading a shape. Irregular cadence also breaks histogram legibility.

**R7 — Symptom, driver, cascade.** Variance growth, wall contact and log Z crashes
are usually downstream of something else. Test with temporal precedence: does the
candidate move *first*? Treating a symptom is still legitimate when the causal
story is convincing and treatment relieves the rest.

**R8 — Match a periodicity to a process with the same period.** Clean repeating
oscillation is a mechanism, not noise: eval cadence, replay churn and step counts,
buffer purge, LR cycle. Ordinary noise looks different and is distinguishable by eye.

**R9 — Never read one channel alone.** Too-hot LR is a *correlated* deterioration
across several metrics while others continue fine. That is why single-loss
"too hot" detectors keep failing on the equilibration problem, and why the loss
alone misses most of the richness.

**R10 — Trading off is normal; a hard trade-off is a warning.** fwd/bwd/replay are
Pareto against each other until terminal convergence, where they overlap. But two
metrics that move strictly against each other should on average be mutually
reinforcing; when they are not, something structural is wrong.

**R11 — Replay error below forward error is overfitting, not success.** Replay is
drawn from higher-residual trajectories by construction. Healthy is roughly 2×
fwd; below 1× means rows are being corrected faster than they are replaced.

**R12 — Slopes, not plateaus.** There is never a truly flat plateau here; the slope
decays smoothly forever. Judge convergence on slope across several channels, and
treat a secondary channel still moving linearly (mean sample energy is the usual
one) as proof the distribution is still in motion.

**R13 — Know which floors are known.** `relative_under`'s irreducible floor is under
~2 nats for almost any problem. `fwd/tb_err_worst` and `replay/over_coverage` floors
are strictly unknown. Never ratchet a threshold below a floor you have not measured —
that is how a controller chokes a branch off on noise.

**R14 — A pinned metric is a dead sensor.** Zero spread, a value bound at its clip,
a threshold annealed below its own noise floor, a censored estimator reported at its
censoring bound. None of these are readings.

**R15 — Handoffs are read on log Z.** A deep log Z dive is the hallmark of a failed
handoff; a shallower dive is the primary transition metric. Separately: log Z must
track the policy models, and lagging Z is dangerous in its own right.

**R16 — Compare against a named baseline.** A new arm beats a specific trusted run
or it has shown nothing. Guard the identity of the traces — which colour is which
run — before drawing a conclusion from a comparison.

**R17 — Conditional runs: held-out first.** Read `eval_test` before `eval_fwd`.
Train r2, tb_err and scatter_err can all improve on the same evaluation where the
held-out set blows up.

**R18 — Per-condition metrics need care.** A thresholded per-condition fraction is
biased by n_c and does not compare across streams. Gate on a quantile rather than
the worst condition: early conditional manifolds are legitimately non-smooth, and a
worst-case gate holds the model hostage to a handful of pathological conditions.

**R19 — Cost is a run metric.** Utilisation, batch size, step time, samples/sec.
A mechanism story does not overturn a headline timing regression.

## 4. Confounds named routinely

Before a comparison means anything: code version drift between arms · checkpoint
chaining, where arms silently resume from each other rather than a pinned start ·
different problem, different T (T dominates; keep `eval_T` = train `T`) · runs
barely started, where the read is really a phase-2 injection point · arms that
differ by omission and are therefore duplicates · a knob that was retired or inert
in that tree · the LR sitting in a different part of its cycle at read time · a
fresh optimiser or LR ramp at a stage boundary, confounding the boundary itself ·
another process on the GPU · an arm that was not running the code it was written
to test, which voids its hypotheses outright.

## 5. What earns the name "finding"

- Every finding carries a **mechanistic hypothesis**. A correlation with no
  mechanism is a run-table row.
- **Calibrations are earned.** A finding is a principle or a mechanism, derived or
  reliably observed such that it generalises. Intensely local numbers promoted to
  "X is Y" statements distract more than they inform, and specific operating-point
  values are to be forgotten, not memorised.
- **A run that confirmed the expected is not written up.** Neither is one that died
  operationally.
- Not everything interesting goes to memory. Most of it does not.

## 6. Moves after a read

- **Test on the checkpoint, not from scratch.** Reload and intervene; almost every
  hypothesis here is testable that way.
- **Force the extreme to isolate a mechanism.** Set log Z to ±100 and see which
  direction converges; pin every condition's level at 9× dose; force a frac to 1.0.
  A mechanism that survives an overkill dose is real; one that does not is dead.
- **Prefer experiments that pay information inside 30 minutes**, and cap arm length
  rather than letting a run wander.
- **Imagine the downside before running it.** Name the failure mode the design
  admits, then check whether the run took it.
- **Necessary-condition tests.** If one mode cannot be fitted, the distribution
  cannot be fitted, so test the cheap necessary condition first.
- **One variable per arm**, or name the confound explicitly in the write-up.

## 7. Division of labour

- **Refer to runs by name, tag, or a distinguishing config feature** — never by
  wandb id.
- **Online runs only**, except smoke tests and pure functionality checks.
- **Python beats the model on raw data processing.** Extract running means,
  oscillations, noise levels, trends and correlations with scripts; do not ingest
  hundreds of scalars and narrate them.
- **Concise and mechanical.** No narrative essays. Invented jargon and unexplained
  variable names are a failure mode, not a shorthand.
- **Reading is context-dependent and intuitive.** The order in §1 is a floor for
  someone with no priors, not an algorithm — the real read jumps to whatever the
  phase and the symptom implicate.
- **A measurement that contradicts a mechanism is a reason to re-check the
  measurement.** "This feels mechanistically impossible" is a valid challenge and
  has repeatedly been right.
- **Complexity that grows while results get worse is the tell.** Patch-on-patch is
  the failure mode to watch for; build in simple concrete steps instead.

## 8. Watching a run in flight

A finished run is the expensive way to learn something the first few hundred steps
already said. Most of what a run has to teach arrives early, and the discipline is
knowing what to look at and when to stop — not watching harder.

- **Name the question, the metric, and both answers before launching.** "Does the
  controller cut when the rate jumps at the transition" → `lr_ctrl/peak_scale`
  falls within N calibrations of the boundary, or it does not. A run without a
  pre-named falsifier is a demonstration, and you will watch it to the end because
  nothing tells you when to stop.
- **Compute when the metric can FIRST move, and check just after that** — not on a
  clock, and never by polling for completion. The answer is usually a sum of
  configured waits: a settling gate, plus `min_readings × period`, plus a
  residence, plus `eval_period` for anything that only publishes at eval. If that
  sum is longer than the run, the run cannot answer the question and should be
  reconfigured rather than started.
- **Stop when the question is answered, including when the answer is no.** A
  diverging run has already told you; the remaining hours are GPU time, not
  evidence. Keep the log — it is the finding.
- **Watch the ACTUATOR against the thing it is supposed to be controlling**, on the
  same axis: `lr_ctrl/peak_scale` and `lr_fused` beside the loss, `Bwd/Replay Frac`
  beside the metrics their balance rule reads. A controller that should have acted
  and has not is visible in seconds this way and in nothing else.
- **Read the "why am I silent" channels first when a controller looks inert.**
  `lrpool/n`, `lrpool/holding_on_transient`, `raycal/skipped|deferred|refused`,
  `ramp/rearms_suppressed`. These exist precisely so that "held because the
  evidence says hold" and "never got to look" are distinguishable, and they turn a
  multi-hour diagnosis into a single glance.

### Live tells that mean stop now

- **A controller that should have acted and has not.** Confirm against its own
  silence channels before concluding it is broken — it is usually gated, and the
  gate is the finding.
- **A loop moving faster than its configured gain allows.** Check whether a fade
  or converge term has come unpinned rather than assuming the gain was edited:
  `converge_floor` fades toward zero near convergence and sits at its MAXIMUM
  while the metrics are far from it, so a diverging run gets full gain
  continuously.
- **A window that keeps resetting.** Any counter that returns to zero repeatedly
  (`lrpool/n`, an exit streak, a pool) is being cleared by something upstream, and
  no amount of waiting will fill it.
- **Two clocks that can disagree.** Where one mechanism advances on steps and
  another on a measured signal, they will eventually disagree without bound. That
  is a design fault, not a tuning one.
