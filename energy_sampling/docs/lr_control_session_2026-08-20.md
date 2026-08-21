# LR control review — session record, 2026-08-20

Companion to `docs/hypergradient_review.md`, which is the standing assessment. This
file records what was asked, what was found, what was measured, what was corrected,
and what is still owed.

---

## 1. Scope

Assess the `hyper` LR sensor (`controller.py::on_hypergradient`, `train.py::_hyper_apply`)
against the literature, audit the implementation, and decide whether the method is
salvageable. Prompted by a failure on the conformer ring MLE warm start
(`configs/conformer_ring_mle.yaml`, stage `train_prior`): `cos` sat at -0.006 to -0.07
for 20,000 steps, the live rate fell to ~1/800 of configured, and the model appeared
unable to fit a target it fits fine at a pinned 1e-4.

Deliverable was an assessment, not an implementation. No trainer code was changed.

---

## 2. The mechanism of the failure

### 2.1 The statistic means two different things

For constant-step SGD on a quadratic with additive gradient noise, in one
eigendirection with curvature `lambda` and noise variance `s`, at stationarity:

    stationary variance   v = eta*s / (lambda*(2 - eta*lambda))
    E<g_{t+1}, g_t>         = -eta*lambda*s / (2 - eta*lambda)   < 0 for all 0 < eta < 2/lambda
    E|g|^2                  = 2s / (2 - eta*lambda)
    =>  cos                ~ -eta*lambda / 2

Consequences:

- Once the iterate has equilibrated in its noise ball, `cos` is **negative at every
  stable step size**. There is no zero crossing to find. The sign encodes overshoot
  only in the *transient* phase.
- Only the magnitude carries rate information: `|cos| = eta*lambda_eff/2`, so
  `eta_stability / eta_current = 1/|cos|`. The observed -0.006..-0.07 was saying the
  stability edge sat **14x-170x above** the rate being run.
- A warm-started MLE stage begins near-stationary, so the negative reading appears
  from step 1.

This also reconciles two opposite failures already on record: `cos` positive at every
rate on the equilibration surface (`peak_scale` ran to its upper bound) and negative at
every rate here (ran to its lower bound). Same defect — no fixed point — with the sign
set by transient vs equilibrated.

### 2.2 Two arithmetic defects, independent of the interpretation

- **Asymmetric gain rectifies noise into drift.** `beta_down = beta * hyper_down_gain`,
  default 2.0 (`controller.py:598`). For symmetric zero-mean error of s.d. `sigma`,
  `E[delta log p] = -beta*(gamma-1)*E|err|/2 ~ -beta*(gamma-1)*0.4*sigma`. At
  `beta=0.1`, `gamma=2`, measured `sd(cos)` 0.062-0.168: **-0.0024 to -0.0068 per
  firing**, i.e. the floor is reached in 700-1900 firings from pure noise.
- **No leak.** `peak_leak` defaults to 0.0 (`controller.py:663`), so the loop is a pure
  integrator in log space: pole on the unit circle, infinite DC gain. `bounds` is
  saturation, not a restoring force. The fix is implemented and off.

### 2.3 The failure compounded

Two authorities key on the sign of the same statistic: the warmup-envelope freeze
(`controller.py:616-633`) and the `peak_scale` integrator. The envelope froze at
**0.1194** (`controller.py:763-768`, run `mmnxotsr`) and `peak_scale` railed at **0.01**.

    0.1194 x 0.01 = 1.19e-3  =  1/838

which is the reported 1/800. Live rate ~1.5e-7 against a working pinned rate of 1e-4.
Time to rail at the observed mean `cos ~ -0.03`: `ln(0.01)/(0.2*0.03) = 767` firings.

---

## 3. Literature

### 3.1 Provenance

Kesten 1958 (sign-change step shrinking) -> Pflug 1983/1990 (`<g_t,g_{t-1}>` as a
transient/stationary **diagnostic**) -> Jacobs 1988 delta-bar-delta, Riedmiller & Braun
1993 RPROP (per-weight sign agreement; both documented as noise-sensitive at small
batch) -> Sutton 1992 IDBD -> Almeida et al. 1998 (multiplicative correlation form) ->
Schraudolph 1999 SMD -> Baydin et al. ICLR 2018.

Two points that matter:

- **Schraudolph's SMD exists because gradient correlation is a poor curvature proxy
  under noise.** It replaces the correlation with an exact curvature-vector product.
  The historical record contains an explicit move *away* from this statistic.
- **Baydin et al. proves essentially nothing** about convergence in the stochastic
  deep-learning setting. Its claims are empirical, on well-conditioned vision problems
  with i.i.d. minibatches from a fixed dataset. Its update is additive and
  unnormalised; the cosine form here is a defensible modification with no inherited
  guarantees.

### 3.2 The two proven negative results

- **Wu, Ren, Liao & Grosse, ICLR 2018** — short-horizon meta-objectives "cause a serious
  bias towards small step sizes". Proven on a noisy quadratic; even 100-step unrolls
  pick rates smaller "by multiple orders of magnitude". A one-step hypergradient is
  horizon 1. **This predicts the direction of the observed failure.**
- **Pesme, Dieuleveut & Flammarion, ICML 2020** — "Even in the simple case where the
  objective function is quadratic we show that this test cannot lead to an adequate
  convergence diagnostic." About precisely this statistic, in the easiest setting for
  it. They replace it with a distance-based test.

Also relevant: Chu, Gao, Ye & Udell (arXiv:2502.11229, ICML 2025) give the first
rigorous convergence analysis of hypergradient descent and state that it "explains the
instability of HDM reported in the literature" — headline experiments on deterministic
convex problems.

### 3.3 Where the literature does not reach

No source analyses the cosine-against-realised-Adam-displacement form that ships here,
nor the statistic under a self-generated (on-policy) sampling distribution. The
`s > 0`-style stationarity claim is derived for SGD on a quadratic; its Adam analogue is
not proved anywhere I could find.

---

## 4. Implementation audit

### What is correct

- **The sign convention.** `train.py:3639` uses `d = -self._hyper_prev_step`, so
  `cos > 0` means agreement with the previous descent direction. No sign error.
- Scale-freeness: the cosine bounds one move by `exp(beta)`.
- `_clip_saturated` is a real regime guard.
- `configs/hyperslope_aug17/make.py` already specifies the decisive measurement.

### Defects, in order of consequence

1. No interior fixed point established on this surface.
2. Two authorities on one statistic (section 2.3).
3. Asymmetric gain rectifies noise into a cut.
4. No leak by default.
5. Asymmetric bounds: `[0.01, 2000]` is -4.6 vs +7.6 nats, so downward failures rail
   1.7x sooner and look like a model that cannot fit rather than a controller fault.
6. **`_hyper_prev_step` is a single slot, not keyed by branch.** `_hyper_sensor_cfg`
   admits `fused`/`fwd`/`bwd`/`replay` and each has its own Adam at its own LR
   (`train.py:2169`), so an unfused multi-branch stage differences a gradient from one
   branch against another branch's displacement. Did not cause this failure
   (`train_prior` is single-branch) but is live for any multi-branch stage.
7. `lr_ctrl/hyper_cos` publishes the raw cosine while the actuator uses
   `cos - cos_target`; with a nonzero target the dashboard shows the wrong series.
8. Every regime gate is one-way down. Nothing detects "pinned at the floor for N steps
   and not improving" — which is what would have caught this in hours.

---

## 5. Correction made during the session

I first framed the stationarity regime as "the slope has the wrong sign (`s > 0`)".
That is wrong. At stationarity `cos = -eta*lambda/2`, so

    s = d(cos)/d(log lr) = eta * d(cos)/d(eta) = -eta*lambda/2 = cos  <  0

The slope is *negative* and the loop is locally stable in the textbook sense. What is
missing is an **interior zero crossing**: the only solution of `cos = 0` is `eta = 0`,
so the controller's attracting fixed point is the floor it railed into.

The conclusion was unaffected but the ladder's decision rule was wrong, and it is the
part that would have been acted on. Corrected in the review at sections 3.1, 6.2, 7.2.
**The discriminator is where the zero crossing sits, not the sign of the slope.**

---

## 6. Measurement: does the optimal rate actually move?

The premise behind wanting a servo at all — a non-stationary optimum — had never been
separated from measurement noise. It can be, from data already on disk.

### 6.1 Two traps

- **`raycal/*` is a sticky summary**, republished every reporting row. The raw series
  counts DWELL TIME, not calibration events, and weights each reading by how long it
  persisted. `5t7ny5lw` reads as 1281 "calibrations" until deduplicated on the
  `lr_ctrl/calibrations` counter; it is then the documented 26.
- **`alpha_star` is the wrong variable.** Under a live servo it is a closed-loop
  residual. The servo-invariant quantity is `live_lr * alpha_star` — the implied
  one-step-optimal LR.

### 6.2 Within a stage, the optimum is constant

19 runs with `raycal/alpha_star` (14 local, 5 cluster):

| run | bracketed n | span | implied-opt sd | trend |
|---|---|---|---|---|
| `prod0810_mipcas_elj` | 67 | 33,500 | 0.254 dex | -0.0049 dex/1k |
| `prod0810_nehzor_elj` | 58 | 29,000 | 0.201 dex | -0.0027 dex/1k |
| `prod0810_nehzor_uma` | 14 | 6,500 | 0.224 dex | +0.0016 dex/1k |
| `d33elj_..._t10_r2` | 10 | 5,000 | 0.140 dex | +0.0666 dex/1k |
| `5t7ny5lw` | 24 | 12,000 | 0.121 dex | -0.0001 dex/1k |

The two long runs trend by 0.16 and 0.08 dex across their **entire** stage.

The spread is white noise, not drift. Empirical variogram (semivariance / total
variance; 1.00 = no temporal structure):

    prod0810_mipcas_elj   lag 0-1k  1-2.5k  2.5-5k  5-10k  10-20k  20-40k
                              1.02    1.02    1.03   1.01    1.03    0.98
    prod0810_nehzor_elj       1.18    1.03    1.03   0.96    1.04    1.02

Two calibrations 500 steps apart disagree by as much as two 40,000 apart. Lag-1
autocorrelation -0.057 (se 0.12) and -0.169 (se 0.13); `var(diff)/(2*var)` 1.04 and
1.17 against 1.00 for white noise.

Per-reading noise 0.20-0.25 dex (+-60-75%) against a grid quantisation floor of 0.087
dex. But the **mean of 67 readings has se 0.031 dex: the optimum is pinned to +-7% and
constant over 33,500 steps.**

### 6.3 The "1.6x per 1000 steps drift" was the servo

On `5t7ny5lw`, `alpha_star` alone trends -0.045 dex/1k (47% of its variance); the
implied optimum on the same readings trends -0.0001 dex/1k (0.0%). The run was seeded
6x low and climbed. The open question in the v8 record is retired.

### 6.4 Ray passes its slope test where the test is valid

`5t7ny5lw`'s climb is quasi-open-loop (seeded 6x low, LR moved 8.7x):

    d log(alpha*) / d log(lr) = -0.891 +- 0.114     t vs 0 = -7.8     t vs -1 = +1.0

-1 is a perfectly tracked **fixed** optimum; 0 is a sensor with no rate information. On
runs holding at setpoint the regression is circular and reads ~0 by construction, so
`prod0810_mipcas_elj`'s +0.10 +- 0.22 is **not** evidence of sensor failure.

### 6.5 The same rectification, measured in production

`ray` ships `eta_up 0.25` against `eta_down 0.5`, so it must sit hot of setpoint to
stand still:

| run | sigma | predicted offset | observed | held alpha* |
|---|---|---|---|---|
| `prod0810_mipcas_elj` | 0.214 | +0.059 dex | **+0.058** | 4.57 |
| `prod0810_nehzor_elj` | 0.186 | +0.051 dex | +0.075 | 4.75 |
| `prod0810_nehzor_uma` | 0.211 | +0.058 dex | +0.075 | 4.75 |
| `d33elj_..._t10_r2` | 0.157 | +0.043 dex | +0.136 | 5.47 |

All hold above target 4, in the predicted direction; the best-sampled run matches to
0.001 dex. Same mechanism that railed `hyper` — bounded here only because `ray` has a
real interior fixed point. It also explains the +-17% hold sawtooth: the servo is
reacting to a statistic the variogram says is white noise.

### 6.6 Between stages there is no data at all

`ray` produced **zero calibrations outside phase 2** across all 19 runs:

    prod0810_nehzor_elj   phase 1 spans      0-7,430    first calibration  7,500
    prod0810_nehzor_uma   phase 1 spans      0-15,660   first calibration 16,000
    d33elj_..._t10_r2     phase 1 spans 10,000-11,070   first calibration 11,500
    prod0810_mipcas_elj   phase 1 ends at 8,720         first calibration  9,000

Over 24,000 steps of phase-1 training with the sensor configured it never fired.
Structural: phase 1 trains `bwd` while the probe scores replay, and it holds through
the re-armed warmup. The between-stage component is **unmeasured and unmeasurable at
the sensor's current placement.**

### 6.7 Between-route spread (context for pinning)

Per-run mean implied one-step optimum across the five runs: sd **0.126 dex (+-34%)**,
full range **2.19x**, spanning 3.13e-4 to 6.87e-4 (operating rate 7.8e-5 to 1.7e-4 at
`alpha_target = 4`). All crystal-route phase 2; the conformer phase-1 rate is not in
this set because `ray` has never measured it.

### 6.8 Second correction: the ray status distribution

An earlier chat figure of "3 bracketed of 11" for `7tjno8m6` was an artefact of keying
on rows with a finite `alpha_star`, which drops exactly the readings that failed.
Tabulated from the status series instead:

| run | events | status |
|---|---|---|
| `prod0810_mipcas_elj` | 68 | 67 bracketed, 1 below_range |
| `prod0810_nehzor_elj` | 59 | 58 bracketed, 1 below_range |
| `prod0810_nehzor_uma` | 14 | 14 bracketed |
| `d33elj_..._t10_r2` | 11 | 10 bracketed, 1 below_range |
| `7tjno8m6` (var_conditioning) | 17 | 3 bracketed, **6 inconsistent**, 5 above, 3 below |

35% inconsistent, matching the docstring. "Inconsistent" means the tests contradict —
a variance problem. On that stage replay is pinned to zero, so the loss path re-samples
a trajectory at every alpha and the differences are trajectory noise. On the four
crystal routes where pairing holds, `ray` resolves **149 of 152**.

---

## 7. Conclusions

1. **`hyper` as specified is a dead end as a general-purpose controller.** Not because
   of tuning: because the statistic's meaning flips between transient and equilibrated
   regimes, and the regime is not observable from the statistic itself. The two
   opposite railings on record are the same defect with opposite signs.
2. **Within a stage, no servo is needed.** A constant rate sits within ~7% of the
   measured optimum for 30,000+ steps and the variogram says there is nothing to track.
   Every within-stage move the servo makes responds to white noise.
3. **The non-stationarity that could matter is between stages — and neither sensor
   measures it.** `ray` never fires in phase 1; `hyper` fires there and reads a
   statistic whose sign is uninformative on an equilibrated surface.
4. **This reframes the `train_prior` failure.** The choice was never "servo vs no
   measurement" — it was "a biased measurement acted on every step vs a per-stage
   constant". `conformer_ring_mle_fixedlr` is the demonstration that the constant wins.
5. **`ray` is the better sensor**, for structural reasons: it scores the objective
   rather than a proxy, it is a test with an abstain state (which is exactly the
   probabilistic-accuracy condition the noisy-line-search theory requires — Paquette &
   Scheinberg 2020; Berahas, Cao & Scheinberg 2021), and its paired design is common
   random numbers, worth the measured 30x in `t`. But its response should become an
   **estimator** (pool readings, act once, hold), not a per-reading servo.
6. **Nothing in the literature beats a tuned baseline consistently.** Schmidt, Schneider
   & Hennig (ICML 2021, >50,000 runs): "we cannot discern an optimization method clearly
   dominating across all tested tasks". The burden of proof is on the controller.

---

## 8. The three jobs, and the best option for each

**Calibration (find the rate).** `ray`, pooled, with a front-loaded cadence: se ~
`0.25/sqrt(k)` dex, so 10 readings give +-20% and 30 give +-13%. Burst after a stage
entry, then back off. Rejected alternatives: an LR range test needs a throwaway run and
cannot reach a mid-protocol stage's state; the parameter-free family (D-Adaptation,
Prodigy, DoG) keeps monotone internal estimates that cannot come down across a stage
change.

**Re-calibration (detect that it must change).** Trigger on events, not drift. Re-burst
at stage entry and t-test the new pooled estimate against the held one. If a
within-stage detector is wanted anyway, use Pesme's distance-based test or the
Yaida/SASA fluctuation-dissipation test — never the gradient-correlation form. Note
what those detect is *equilibration*, which is a cue for terminal decay, not for
tracking a moving optimum; and terminal decay is a schedule decision, removed entirely
by Schedule-Free (Defazio et al., NeurIPS 2024, AlgoPerf self-tuning winner).

**Fast monitoring (catch a blow-up between calibrations).** A smoke alarm, not a
thermostat: a low-false-positive tripwire plus rewind, never a continuous actuator.
Order: (1) grad-clip fire rate per branch — already computed, a *rate* not a *sign*, so
no regime dependence; (2) the relative loss bar `divergence_loss_rel`; (3) gated
backtracking (the line-search sufficient-decrease test, run only when (1) fires);
(4) prevention via a trust-ratio update (LARS/LAMB) or Adafactor-style relative step,
which bounds the damage rather than detecting it.

---

## 9. Open items

1. **`alpha_target` has never been re-measured per route.** It is a direct multiplier on
   every rate `ray` sets, measured once on `elj-mipcas`. Cheapest measurement: run
   `ray` in sensor-only mode (pin the rates as explicit floats so `lr_servo_managed` is
   empty) on a run at a known-good hand-tuned rate; the alpha* it reports at that rate
   **is** `alpha_target` for that route. Highest-value open item.
   Note this is not the same as short-horizon bias and the two point opposite ways:
   ray's probe is a deterministic paired line search at frozen theta, blind to the noise
   future steps inject, so it reads too hot; Wu et al.'s stochastic multi-step unroll
   sees the noise ball and reads too cold.
2. **Extend `ray` coverage to phase 1**, so the between-stage question can be measured
   rather than argued. Condition: the pairing must come with it — a loss that re-draws
   its stochastic path per alpha reproduces `7tjno8m6`'s 35% inconsistent.
3. **Retrospective check on `mvwsu5d5`** (the 12x-hot unrecoverable run): would
   `divergence_loss_rel` at 100x have tripped? The 1e9 absolute bars did not. If it
   would have, job C may already be solved. Data is on disk; not yet run.
4. **The `cos` slope ladder on the ring MLE surface** — still the decisive test for
   `hyper` itself, with the corrected decision rule (interior zero crossing, not slope
   sign). Plus the frozen-theta control: two independent batches at the same
   parameters, to separate anti-correlated *sampling* from anti-correlated *dynamics*.
5. **Structural fixes worth making regardless of any verdict**: decouple the
   warmup-freeze rule from the `peak_scale` integrator; make the cut/raise gains
   symmetric; key `_hyper_prev_step` by branch.
6. `docs/hypergradient_review.md` section 11.4's rectification prediction matches well
   on the best-sampled run and runs high on the small-n runs. Not investigated.

---

## 10. Artifacts

- `docs/hypergradient_review.md` — the standing assessment (12 sections: provenance,
  mechanism, failure modes, modifications, alternatives, implementation audit,
  recommendation, literature gaps, repair ranking, set-and-forget options under
  non-stationarity, the measured decomposition, citation verification status).
- This file.
- `raydrift.py` / `raydrift_cloud.py` in the session scratchpad — ephemeral, rebuildable
  from review section 11.
- Memory: `project_hyper_cos_is_a_stationarity_statistic`,
  `project_lr_optimum_is_stationary_within_stage`; the stale drift claim in
  `project_lr_controller_v8_ray_calibration` is marked resolved.

## 11. Verification status

Citations verified against publisher/arXiv pages this session: Baydin et al. ICLR 2018;
Wu et al. ICLR 2018; Chu, Gao, Ye & Udell ICML 2025; Pesme et al. ICML 2020; Chee &
Toulis AISTATS 2018; Lang, Xiao & Zhang NeurIPS 2019; Schraudolph ICANN 1999; Almeida et
al. 1998; Defazio & Mishchenko ICML 2023; Mishchenko & Defazio ICML 2024; Ivgi, Hinder &
Carmon ICML 2023; Cutkosky, Defazio & Mehta NeurIPS 2023; Vaswani et al. NeurIPS 2019;
Tan et al. NIPS 2016; Paquette & Scheinberg SIOPT 2020; Berahas, Cao & Scheinberg SIOPT
2021; Schaul, Zhang & LeCun ICML 2013; Schmidt, Schneider & Hennig ICML 2021; Defazio et
al. NeurIPS 2024; Smith, Kindermans, Ying & Le ICLR 2018.

Cited from prior knowledge, not re-verified: Kesten 1958; Pflug 1983/1990; Jacobs 1988;
Riedmiller & Braun 1993; Sutton 1992; Plagianakos et al. 2001; Chandra et al. NeurIPS
2022; Donini et al. IJCAI 2020; Balles & Hennig ICML 2018; Simsekli et al. ICML 2019;
Zhang et al. ICLR 2020; Cohen et al. ICLR 2021; Gur-Ari et al. 2018; Shazeer & Stern ICML
2018; Smith 2017/2018; Loshchilov & Hutter ICLR 2017; Liu et al. ICLR 2020; You et al.
2017/2020; McCandlish et al. 2018; Yaida ICLR 2019; Martinez Rubio 2017 (thesis existence
asserted by Baydin et al., not inspected).

All numbers in section 6 were computed this session from `analysis.pull` over the runs
named. No trainer code was modified and no training runs were launched.
