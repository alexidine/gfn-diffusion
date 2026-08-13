# LR control: what the bench has established

*2026-08-12. Standalone summary of the learning-rate investigation. Not a
`findings.md` entry — this is the working state of one question, and it will keep
moving. All results are from `bench/`, on synthetic surfaces driving the REAL
`LRController` and `RayCalibration`.*

---

## 0. THE ANSWER -- 2026-08-13, after adversarial review

**Use Baydin et al.'s published rule, unmodified. There is no evidence any
modification helps.**

```
lr *= exp(0.02 * cos(g_t, g_{t-1}))
```

One dot product per step. No L*, no probe batch, no extra forward pass.

| arm | % over a 2x budget | worst cell |
|---|---|---|
| **`hyper` symmetric** | **0.0%** | 0% |
| `hyper` 2:1 asymmetric | **0.0%** | 0% |
| `hyper gated` | **0.0%** | 0% |
| `ray+ray` (shipping) | 1.2% | 8% |
| `ramp+plateau` | 2.3% | 15% |
| **NULL (no sensor at all)** | 25.4% | 28% |

**THE THREE HYPERGRADIENT VARIANTS ARE INDISTINGUISHABLE.** Every earlier
ordering among them (0.1/0.5/1.8%, and before that 0.0/2.1/4.7%) was an artifact.
Recommend the published rule because nothing beats it, not because it wins.

### Two defects an adversarial review found, both mine, both invalidating

**(1) THE BUDGET WAS UNREACHABLE.** `steps_to_target` returns at most `steps`, so
with the denominator at `DEEP_FRAC=0.60` of the run the largest FINITE ratio is
1.63 -- below the 2.0 budget. Every "% over budget" reported before this was
identically "% never converged", and BUDGET could have been set to anything in
[1.64, inf) without changing a number. This is the SAME defect section 7 claims
to have fixed, reintroduced worse, and it went unchecked precisely because it had
been "fixed" once. Now `frac=0.25` (ceiling 4x) with a per-cell assertion in
`_oracle_task` so it cannot regress silently again.

**(2) ONLY ONE OF FOUR SCENARIOS TESTS A CONTROLLER.** The NULL arm -- servo live,
no sensor -- scores 25.4%, which is 1 of 4 scenarios: it fails 100% of COLD
STARTS and **0% of drift, regime_change and hot_90pct**. Those three are
pass-through columns; the tripwire and rewind handle them for everybody. So
`hyper gated`'s 95% on regime_change, on which the whole 2026-08-12 reversal
rested, was self-inflicted damage relative to STANDING STILL, not a failure to
track. The NULL arm is now permanent in `ARMS`.

Also corrected: `0.0104` was claimed as "the blind ramp's own rate" but is
`ln(1.682)/50` using the BENCH's period override; `MK_DEV_RAYCAL` ships period
**500**, so the shipping rate is 0.00104 and the constant was 10x its stated
source. The claim "every constant has a source, not a sweep" was false.

### What is NOT established

- **The margin over the incumbents is small and only marginally significant.**
  0.0% vs 1.2% is 0/1040 against ~12/1040, but seeds within a cell share the
  surface, oracle and target, so the effective n is closer to the 13 cells than
  to 1040 runs. `ray+ray` fails 3 cells, hyper 0 -- a paired sign test over cells
  gives p ~ 0.12, not significance.
- **The held-out set has been used more than once** and is burned for any further
  selection.
- **The noise axis is still uncalibrated** (see (e) below). Unchanged and still
  the highest-value open action.

---

## 0a. Superseded answer (2026-08-13 morning)

**The published rule, unmodified, wins. Everything I invented was compensating
for a scenario that does not happen.**

```
lr *= exp(0.02 * cos(g_t, g_{t-1}))        # Baydin et al., symmetric, nothing added
```

**0.1% over a 2x budget -- ONE failure in 1040 held-out runs** (13 cells x 4
scenarios x 20 seeds), against the shipping probe's 10.1%.

| arm | % over budget | worst cell |
|---|---|---|
| **`hyper` symmetric (published)** | **0.1%** | 1% |
| `hyper` 2:1 asymmetric | 0.5% | 6% |
| `hyper gated` (my design) | 1.8% | 24% |
| `ray+ray` (shipping) | 10.1% | 100% |
| `ramp+plateau` | 14.5% | 89% |

**WHAT CHANGED: the `regime_change` scenario.** It softened curvature 8x IN ONE
STEP. MK: real regime changes outside phase transitions are ~1.5x in LR over
~5000 steps -- two orders of magnitude gentler in rate-of-change. Corrected to a
gradual drift, the entire ranking INVERTS. `hyper gated` goes 0.0% -> 1.8% and is
now WORSE than plain hyper, with 24% at cond=30 (95% on regime_change): its bias
is a headwind on a gradual drift.

Every mechanism in sections 1-2 below -- asymmetric gain, ramp bias, confidence
gate -- existed to survive a violent step change. Given a realistic one they are
all net harmful. **Sections 1-2 are retained as a record and should not be read
as recommendations.**

Six further variants (`hyperz`, `hypern`, `hyperm`, `hyperp`, the rho branch, the
||g|| gate) are dead for the reason in (b) below and were also chasing this
artifact.

**STILL OPEN, and it gates everything above:** the noise axis has never been
calibrated (see (e)). Median cos is 0.29 at our WORST cell; real training may be
0.01-0.1.

---

## 0b. Earlier status (2026-08-12 late), superseded above

**Section 1 below overstates `hyper gated` as a DESIGN.** It is the empirical
leader (0% over budget on 13 held-out cells) and its justification is wrong in
both halves. Later rounds established:

**(a) THE TWO MECHANISMS CANCEL EACH OTHER.** The asymmetric brake
(beta_down > beta_up) was justified on consequences -- overshoot costs a sticky
ceiling -- and that argument ignored what it does to the DRIFT of a noisy
statistic. Under heavy noise cos is symmetric about a small positive mean, so a
larger beta_down drives the rate DOWNWARD and it can never climb out: `beta
.02/.08` scores 50% at dim2048/noise2 with cold_start at 100%, while the PURE
SYMMETRIC published rule (`.02/.02`) scores **0%** there. The bias then cancels
that cold drift, and the brake cancels the bias's upward drift. Two wrongs making
a right, not a design.

**(b) SIX VARIANTS, ONE FAILURE MODE.** Every attempt to make hypergradient
faster works by AMPLIFYING cos, and every amplification saturates the response
into a switch; bang-bang plus noise then fails. Dead: `hyperz` (divide by the
theoretical null -- x51 at d=2048), `hypern` (divide by a running mean of |cos|),
`hyperm` (momentum on raw beta*cos -- compounds same-sign noise runs), `hyperp`
(smooth THEN amplify -- k=8 scores identically to k=4, proving the clip binds
always), plus the rho branch and the ||g|| gate via thresholds. The additive bias
is the ONLY thing that ever bought speed without amplifying, because addition
offsets the response rather than scaling it, so proportionality survives.

**(c) RAISING THE GAIN IS NOT THE ANSWER.** beta_up 0.02 -> 0.04 -> 0.08 gives
3.8% -> 7.8% -> 11.8%. Scaling beta leaves the signal-to-noise of each decision
unchanged but scales the random-walk excursion ~beta*sigma*sqrt(T), and the
surface has an absorbing boundary above and none below.

**(d) THE `regime_change` SCENARIO IS UNREALISTIC.** It softens curvature 8x in
ONE step. MK: real regime changes outside phase transitions are ~1.5x in LR over
~5000 steps. Every arm that lost only on that column -- plain `hyper`'s entire
2.1%, and `persist`'s headline failure -- must be re-scored before it means
anything.

**(e) THE NOISE AXIS HAS NEVER BEEN CALIBRATED.** Median cos(g_t, g_{t-1}) is
0.9997 at noise 0.01 and 0.29 at our WORST cell. Real deep-net training with
modest batches is often 0.01-0.1, i.e. an order of magnitude noisier than
anything tested here. A checkpoint (`nehzor elj`) is available to measure it. This
is the highest-value open action: one dot product per step calibrates every
ranking in this document.

**Honest current default: `hyper` SYMMETRIC (beta .02/.02)** -- Baydin's rule
with nothing added, 4.7% over budget, and 0% at the hardest noise cell.

---

## 1. The candidate

> **SUPERSEDED by §0 and §0a — record only, not a recommendation.** Every number
> in this section was scored at `DEEP_FRAC=0.60`, where the largest finite ratio
> is 1.63 and the 2.0 budget is unreachable, so each "% over" below is really
> "% never converged". Two labels in it are also wrong; both are corrected
> in place below.

Three lines on top of published hypergradient descent. Free -- one dot product
per step, no probe batch, no extra forward pass, no `L*`:

```
cos  = <g_t, g_{t-1}> / (||g_t|| ||g_{t-1}||)
beta = 0.02 if cos > 0 else 0.08
lr  *= exp(beta*cos + 0.0104*(1 - |cos|))
```

**0 of 800 runs over a 2x budget on TEN HELD-OUT CELLS**, 0% in every one --
cells chosen after the constants were fixed and never looked at while choosing
them.

| arm | held-out % over | worst cell |
|---|---|---|
| **`hyper gated`** | **0.0%** | 0% |
| `hyper` **2:1 asymmetric — MISLABELLED "(published)"** | 2.8% | 16% @ `quartic=0.1` |
| `ray+ray` (shipping) | 5.1% | 25% @ `cond=1000` |
| `hyper+ramp d8` (constant bias) | 5.1% | 38% @ `quartic=0.1` |
| `ramp+plateau` | 18.0% | 89% @ `eq w_rep=0.3` |

**LABEL CORRECTION.** That second row was not Baydin's rule. It was `hyperx`
declared with no standard-override, and `BenchRun.STANDARD` ships
`hyper_beta_down=0.04` against `hyper_beta=0.02` — i.e. the 2:1 asymmetric
variant. The published symmetric rule was never on this board, so "the candidate
beats the published rule by 2.8 points" was never measured here. §0's board,
where all three variants are declared explicitly, is the one that compares them.

**"Every constant has a source, not a sweep" — FALSE, see §0.** `0.02` is
Baydin et al.'s. `0.0104` is `ln(1.682)/50`, and only the `ln(1.682)` half is
sourced: the servo really does apply `(grid_top/alpha_target)^eta_up =
(32/4)^0.25 = 1.682` on an unresolved reading, but the `/50` is the BENCH's
`ray_calibration.period` override (`bench/oracle.py:77`), not the servo's clock.
`MK_DEV_RAYCAL` ships period **500**, so the shipping per-step rate is
`ln(1.682)/500 = 0.00104` and this constant is 10x its stated source — a chosen
number. `0.08` is the house 2:1 asymmetry
(`eta_up 0.25 / eta_down 0.5`) doubled, and the doubling is FORCED: a constant
upward bias against a brake that attenuates under noise barely descends, which is
why the same arm at 0.04 scored WORSE than no bias at all (28.5% vs 15.2%).

### Why each piece is there

- **Plain hyper degrades to the IDENTITY under noise.** `E[cos] =
  ||gbar||^2/(||gbar||^2 + tr Sigma)` is the signal-to-noise ratio, so noise
  attenuates the statistic and `exp(0) = 1` is "do nothing". Safe -- and wrong
  when the rate is 35x too cold. Measured: median cos 0.9997 -> 0.2901 as noise
  goes 0.01 -> 2, so the climb rate falls 0.0202 -> 0.0058/step and closing a 35x
  gap goes from 178 steps to **613**, of a 2000-step run. Its crucible failures
  are ONLY cold_start and hot_90pct -- never drift, never regime change, i.e.
  only where the rate has far to travel.
- **A CONSTANT bias fixes the climb but is a headwind where the signal is
  clean.** Its one remaining failure was `mle q1e-2` (29%): quartic at noise
  0.01, cos ~ 1, no help needed.
- **The GATE `(1 - |cos|)` is the fraction of the reading that is noise**, so the
  arm ramps exactly to the extent the measurement has nothing to say. This is the
  ray probe's abstain-and-apply-the-constant with a CONTINUOUS gate rather than a
  significance threshold -- continuous deliberately, since a threshold on a noisy
  statistic is what killed four arms (§3).

**Not yet an engineering recommendation.** See §6.

### Method note: hold cells back

`hyper gated` scored 0.2% on the cells used to design it and **0.0%** held out,
while `hyper+ramp d8` went 4.8% -> 5.1% with its worst cell RISING 29% -> 38%.
The constant was partly fitted; the gate generalises. That distinction was
invisible until cells were held back, and the earlier headline here -- "0 of 840,
the only arm with no tail" -- was an artifact of scoring against a target at 25%
of the oracle's run instead of 60% (§7). At the deep target that same arm is
15.2%.

## 2. Why it works — and it is not the reason I first gave

Hypergradient descent is gradient descent on the learning rate:
`dL/d(alpha) = -g_t . g_{t-1}`. Gradients agreeing means the step was too small;
disagreeing means it overshot.

The property that actually matters here is that **its statistic is bounded**.
Under batch noise `E[cos] = ||gbar||^2 / (||gbar||^2 + tr(Sigma))`, so noise
*attenuates* the signal toward zero, and `exp(beta * 0) = 1` is "do nothing". Its
noise limit is the identity, so it degrades toward inaction rather than toward a
confident wrong answer.

It does carry a bias. Expanding with `g_t ~ gbar + eps_t - alpha*H*P*(gbar + eps_{t-1})`,
the `eps_{t-1}` in the step correlates with the `eps_{t-1}` in `g_{t-1}`:

```
E[g_t . g_{t-1}]  =  ||gbar||^2  -  alpha * gbar' H P gbar  -  alpha * tr(H P Sigma)
```

That last term is noise-driven and negative, so hypergradient cools under noise.
**But the noise-optimal step size genuinely IS lower** (the standard noise-ball
tradeoff), so the bias points at the right answer. Contrast `bb` in §4, whose
algebraically identical bias is pure corruption because it is estimating a
curvature, a quantity noise has no business entering.

## 3. THE recurring failure mode

Four separate arms died the same death this session:

| arm | the noisy statistic | the threshold |
|---|---|---|
| `armijo` on a floored loss | `L(theta+d) - L(theta)` in float32 | sufficient-decrease bar |
| `armijo` at high noise | same, gradient noise | same |
| `hyperx`'s rho branch | `\|\|g_t\|\|/\|\|g_{t-1}\|\|` | rho > 1 |
| `slope_seek` | windowed progress rate | dead zone |

**A threshold test on a noisy statistic, combined with an asymmetric
multiplicative response, does not wander — it collapses.** With backtrack ×0.5
against a climb of ×1.014, the MEASURED 61.2% accept rate gives
`E[log step] = 0.612*ln(1.014) + 0.388*ln(0.5) = -0.26` per step and the rate
slams into its floor. (`-0.26` was previously attributed to a coin flip; a true
coin flip is worse, `0.5*ln(0.5) + 0.5*ln(1.014) = -0.3396`.)

The worst case is putting the threshold at the statistic's own noise centre,
which is what `rho > 1` does: `hyperx` went from 0% over budget to **100% at
noise 2**, having been the best arm on the board at `quartic=0.01`.

**The mechanism it needs is abstention.** The ray probe survives noise precisely
because its significance test returns `unresolved` and the servo falls back to a
constant. F-020 read "72–82% of readings come back saturated" as the probe not
paying for itself; that reading is backwards — **the fallback IS the noise
robustness**, and the probe is a ramp that measures when it can prove something.

## 4. The through-line: common random numbers

Every finite-difference sensor that survived pays for paired samples. Every one
that failed does not.

- **ray probe** — `n_sub` paired sub-batches + significance test. Survives.
- **`bb`** — `y = g_t - g_{t-1}` across *different* batches. `E[s.y] = s'Hs +
  lr*tr(P*Sigma)`, so it reads **4× too cold at noise 2**. Measured against
  ground truth: median `alpha_bb/alpha_true` 0.998 → 0.680 → 0.381 → 0.242 as
  noise goes 0.01 → 0.1 → 0.5 → 2.
- **`armijo`** — losses on different batches. Collapses (§3).

**Averaging does not fix `bb`.** A 200× window buys a 1.57× spread reduction
where a well-behaved statistic gives sqrt(200) ≈ 14×, because the estimator is a
*ratio* whose denominator crosses zero. And `frac < 0.5x` RISES with the window:
averaging works fine, it just converges onto the wrong number. The fix is
pairing, which costs `bb` the free status that made it interesting.

## 5. Method inventory

> **PARTLY SUPERSEDED by §0.** The costs and the structural verdicts stand — they
> are properties of the methods, not of the scoring. The `hyper` row's pointer
> does not: it sends you to §1 and §6, both superseded. Read §0 for the ranking.

| method | cost | verdict |
|---|---|---|
| **`hyper`** | free | Best guarantee measured, and the SYMMETRIC published form is the one to use — see §0, not §1/§6. |
| `ray+ray` (shipping) | ~2% | Sound architecture (abstention + fallback), never tuned as one object. |
| `ramp+plateau` | free | Noise-immune (no sensor to corrupt) but cannot track an optimum that moves UP — the divergence ceiling is cleared only at a stage transition. |
| `bb` | free | Structurally biased, §4. Best on the moving target where noise is low. |
| `armijo` | ~50%/step | Fastest median anywhere; worst thing on the board at high noise. §3. |
| `dog` | free | Cold-start bootstrap failure. **Wrong family member tested** — DoG is SGD-derived; Prodigy is the Adam variant. |
| `sps` | free | Fails on `mle` because `L*` is unknown there. **TB's `L*` is genuinely 0**, so this is untested where it applies. |
| `slope_seek` | free | Retracted — premise was an invented `a/b=4`. |
| `plateau` alone | free | Cannot climb. Bit-identical to no sensor on a cold start. |

## 6. What is NOT established

> **SUPERSEDED by §0 — corrections in place below.** Everything here was measured
> at `DEEP_FRAC=0.60`, where the 2.0 budget is unreachable, so "0% over budget"
> in this section means "always converged", not "always within budget". Two
> bullets are wrong for reasons independent of that and are struck through.

- **`mle` has one loss and one batch stream**, so the standing objection to
  hypergradient — that `g_t` and `g_{t-1}` are gradients of *different objective
  realizations* (changing replay batches, branch mixtures, loss weights) — cannot
  manifest there. It remains untested.
- ~~**`equilibration` did not discriminate** because the best *fixed* LR is a weak
  baseline on a 3-player game.~~ **WRONG DIAGNOSIS — it was MIS-NORMALISED.**
  `find_oracle` minimises last-50 median distance while the crucible scores TIME,
  and on flat-above-optimum surfaces those pick different rates: on `eq base` the
  fastest fixed rate reaches the deep target in ~190 steps against the
  distance-optimal rate's ~1823, a denominator 10.9× too large, which turns a "2×
  budget" into a 0.18× budget that nothing can fail. `crucible._time_oracle`
  documents the fix. **It is not wired in** — nothing calls it — so the
  equilibration numbers in §0 still carry this.
- **`var_cond` is currently unmeasurable** — its oracle converges 8.7× at 1500
  steps, 17.4× at 4000 and 40× at 8000, failing the 100× convergence guard at
  every length tried. It only looked measurable under terminal regret, which does
  not require the run to converge at all. It is no longer in `crucible.CELLS`.
- ~~**Every over-budget run on `mle` is a `cold_start`.**~~ **FALSE.** `hyper
  gated` scores 24% at `cond=30` — an `mle` cell — of which **95% is
  `regime_change`** (§0a). The true statement is the weaker one in §0(2): the
  three non-cold-start scenarios are pass-through for an arm that STANDS STILL
  (the NULL arm fails 100% of cold starts and 0% of the other three), so a
  failure there is self-inflicted damage rather than a failure to track. That is
  a reason to read those columns carefully, not a reason to assume they are
  always empty.
- **The asymmetric-gain experiment did not run its intended test.** All three
  ratios tie at 0%; the asymmetry shows up only in the medians (baseline 1.03 →
  0.75). It buys speed, not robustness.

## 7. Metric design, which cost three re-runs

> **THE FIX THIS SECTION DESCRIBES DID NOT HOLD — see §0(1).** `DEEP_FRAC` was
> moved back to 0.60 afterwards, on the argument that 0.25 made the target too
> shallow, and that reintroduced the same censoring defect worse: the finite
> ceiling `~1/frac` fell to 1.63, below the 2.0 budget, so every "% over budget"
> printed between then and 2026-08-13 was "% never converged". It went unchecked
> precisely because it had been "fixed" once. It is back at 0.25 and
> `crucible._oracle_task` now asserts the ceiling per cell, which is the part
> that was missing here: a stated fix with no guard is one edit from being undone.
> Read this as the derivation of 0.25, not as evidence that 0.25 is what ran.

`steps_to_target(run) / steps_to_target(oracle)`, target = the distance the
oracle passed at 25% of its run.

Two earlier choices were wrong for the same reason — **the measurable range of a
ratio is set by how much run remains after the denominator finishes**:

- oracle's FINAL distance → reachable only on the last step, censored 8 of 11 arms
- oracle's MID-RUN distance → oracle scores steps/2, so any over-budget run is
  necessarily censored. "Over budget" and "never converged" became the same
  event, verifiable in the numbers (88% of 60 = 53, against a never-count of 53).

Also added: a **cell guard**. `find_oracle` accepts a cell if a best LR *exists*,
which is not the same as the run converging. At `cond=3000` the oracle improves
only 9× in 2000 steps and its trace is 5× flatter than baseline, so a 10%
shortfall costs ~100 steps to recover and every arm fails — a run-length problem
reported as a controller problem. Cells failing a 100× drop are now skipped.

## 8. Next

1. **Combined hardness.** The axes select opposite winners — noise favours the
   blind arm, a moving target favours the measuring arm. One-at-a-time sweeps
   cannot find an arm that fails on the *interaction*, which is where nothing
   should work.
2. ~~**Fix the hard surfaces**: `var_cond` needs a longer run; `equilibration`
   needs a stronger baseline than the best fixed LR.~~ **BOTH PRESCRIPTIONS ARE
   WRONG, see §6.** A longer run does not rescue `var_cond` (8000 steps still
   only reaches 40× against a 100× guard) and `equilibration` does not need a
   stronger baseline — it needs the baseline it already has, selected on TIME
   instead of on final distance. The live action is to wire in
   `crucible._time_oracle`.
3. **Log, do not control** (per MK): record `alpha_t`, `alpha_Polyak`, `h_t`
   (hypergradient) and `alpha*_probe` on one real run. If `sign(h_t)` predicts
   `sign(alpha*_probe - alpha_t)`, the expensive probe has a free online
   surrogate. Testable on the bench first, where `alpha_star_true` is exact.
4. **Directional Polyak** `alpha_P = (F - F*)/(-g'd)`: denominator strictly
   positive for any PSD preconditioner, so no heavy tail, and its bias is bounded
   and absorbable by a slow calibration factor. The one cheap estimator these
   results do not rule out.
