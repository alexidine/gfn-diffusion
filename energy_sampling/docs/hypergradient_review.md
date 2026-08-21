# Hypergradient LR adaptation: literature assessment and implementation audit

Scope: the `hyper` LR sensor (`controller.py::on_hypergradient`, `train.py::_hyper_apply`).
Written against the observed failure on the conformer ring MLE warm start
(`configs/conformer_ring_mle.yaml`, stage `train_prior`), where `cos` sat at
-0.006 .. -0.07 for the whole run and the live rate ended at ~1/800 of configured.

This is an assessment, not a change proposal. No repo code is modified by it.

---

## 0. Verdict

**The statistic is a stationarity detector, not a step-size-optimality detector, and the
controller reads it as the latter.**

For constant-step SGD on a quadratic with additive gradient noise, the expected inner
product of consecutive stochastic gradients is *negative for every stable step size* once
the iterate has equilibrated in its noise ball. Its magnitude is proportional to the step
size, so shrinking the rate makes `cos` smaller in magnitude but **never changes its
sign**. The derivation is three lines and is given in section 2.1. This is exactly the
reported signature: small, negative, never crossing zero, all the way to the floor.

Under that account the observed number, read correctly, says the opposite of what the
controller did with it: `|cos| = eta*lambda_eff/2` implies the stability limit is a factor
`1/|cos|` = 14x-170x *above* the rate being run.

Two further defects are arithmetic rather than interpretive, and either alone is
sufficient to produce a monotone ratchet:

1. **Asymmetric gain rectifies zero-mean noise into a systematic cut.** `beta_down`
   defaults to `2 * beta` (`controller.py:598`). For a symmetric zero-mean error of s.d.
   `sigma`, the expected log-move per firing is `-beta*(gamma-1)*E|err|/2`. At
   `beta=0.1, gamma=2` and the measured `sd(cos)` of 0.062-0.168, that is -0.0024 to
   -0.0068 per firing: the floor at `peak_scale = 0.01` (-4.6 nats) is reached in
   700-1900 firings **with no signal at all**.
2. **The loop is a pure integrator in log space** (`peak_leak` defaults to 0.0,
   `controller.py:663`), so any constant bias in the error integrates without limit and
   the `bounds` clip is saturation, not a restoring force. The code comment at
   `controller.py:646-674` states this correctly; the remedy is implemented and off.

And the failure compounded because **two independent authorities are keyed on the sign of
the same statistic**: the warmup-envelope freeze (`controller.py:616-633`) and the
`peak_scale` integrator. The envelope froze at 0.1194 on this route (recorded in
`controller.py:763-768`, run `mmnxotsr`, 2026-08-20) and `peak_scale` railed at 0.01;
0.1194 x 0.01 = 1/838, which is the reported 1/800.

---

## 1. Provenance

What each source actually establishes, and under what assumptions.

### 1.1 The rule is ~40 years old and has always been a heuristic

- **Kesten, "Accelerated stochastic approximation", *Annals of Mathematical Statistics*
  29(1):41-59, 1958.** The ancestor: shrink the step only when the sign of successive
  increments *changes*. Proven for 1-D stochastic approximation with decreasing gains;
  a convergence-rate result under Robbins-Monro conditions, not a statement about a
  constant-step regime.
- **Pflug, "On the determination of the step size in stochastic quasigradient methods",
  IIASA CP-83-25, 1983**, and **Pflug, "Non-asymptotic confidence bounds for stochastic
  approximation with constant step size", *Numerische Mathematik* 56:385-406, 1990.**
  The origin of `<g_t, g_{t-1}>` as a *diagnostic*: it detects the transition from the
  transient phase into the stationary (noise-ball) phase, and thereby decides *when to
  drop* the step size. Note the framing — it answers "have I equilibrated at this rate",
  not "is this rate optimal". I cite these through Pesme et al. and Chee & Toulis
  (section 1.4); I did not read the 1983 report directly.
- **Jacobs, "Increased rates of convergence through learning rate adaptation",
  *Neural Networks* 1(4):295-307, 1988** (delta-bar-delta) and **Riedmiller & Braun,
  "A direct adaptive method for faster backpropagation learning: the RPROP algorithm",
  IEEE ICNN 1993.** Per-weight rules keyed on *sign agreement* of consecutive partial
  derivatives: agree -> grow, disagree -> shrink. Both are heuristics validated on
  full-batch or near-full-batch learning. RPROP is documented as unsuitable for small
  minibatches precisely because gradient noise destroys sign agreement.
- **Sutton, "Adapting bias by gradient descent: an incremental version of
  delta-bar-delta", AAAI-92** (IDBD). First derivation of a sign/correlation rule as
  *gradient descent on the meta-parameter* rather than a heuristic. Proven for linear LMS.
- **Almeida, Langlois, Amaral & Plakhov, "Parameter adaptation in stochastic
  optimization", in Saad (ed.), *On-Line Learning in Neural Networks*, Cambridge
  University Press, 1998, pp. 111-134.** The multiplicative form closest to the rule under
  review: the gain is multiplied by a function of the correlation between successive
  gradients. Its analysis is for a local quadratic model with stationary noise.
- **Schraudolph, "Local gain adaptation in stochastic gradient descent", ICANN 1999,
  pp. 569-574** (stochastic meta-descent). Directly relevant and usually skipped: SMD
  exists **because gradient correlation is a poor curvature proxy under noise.** It
  replaces the correlation with an exact curvature-vector product `Hv` from a second AD
  pass (see also Schraudolph, "Fast curvature matrix-vector products", ICANN 2002;
  Schraudolph, Yu & Guenter, JMLR 7, 2006). The historical record contains an explicit
  move *away* from the statistic this controller uses.
- **Plagianakos, Magoulas & Vrahatis, "Learning rate adaptation in stochastic gradient
  descent", in *Advances in Convex Analysis and Global Optimization*, Springer, 2001.**
  Sign-based adaptation with convergence results under deterministic-gradient assumptions.

### 1.2 Baydin et al. — the modern reference, and what it does not prove

**Baydin, Cornish, Martinez Rubio, Schmidt & Wood, "Online Learning Rate Adaptation with
Hypergradient Descent", ICLR 2018** (arXiv:1703.04782). The contribution is that
`dL(theta_t)/d(alpha) = <g_t, -g_{t-1}>` for SGD, so the learning rate can be updated by
gradient descent on the same objective at the cost of one extra copy of the gradient.
Variants SGD-HD, SGDN-HD, **Adam-HD**.

What it proves: essentially nothing about convergence in the stochastic deep-learning
setting. The claims are empirical — faster early convergence and reduced sensitivity to the
initial learning rate on logistic regression, an MLP on MNIST, and VGG on CIFAR-10. Two
properties matter here:

- The update is **additive** (`alpha <- alpha - beta * dL/d(alpha)`) and **unnormalised**:
  the hypergradient carries the units of `|g|^2`. The cosine form used in this repo removes
  that scale dependence, which is a defensible modification (as `bench/arms.py:32-37`
  already argues) — but it is a *different rule with no inherited guarantees*.
- The reported robustness is with respect to `alpha_0`, over a limited range, on
  well-conditioned vision problems with i.i.d. minibatches from a fixed dataset. None of
  those conditions hold for a GFlowNet warm start on self-generated trajectories.

A convergence analysis for the convex/quadratic case is attributed to D. Martinez Rubio's
2017 Oxford MSc thesis, "Convergence Analysis of an Adaptive Method of Gradient Descent".
I did not verify that document in this session; treat it as unverified.

### 1.3 Later hypergradient work

- **Chandra, Xie, Ragan-Kelley & Meijer, "Gradient Descent: The Ultimate Optimizer",
  NeurIPS 2022.** Stacks hyperoptimizers recursively; the empirical claim is that stacking
  reduces sensitivity to the top-level hyper-hyperparameter. It does not remove it.
- **Donini, Franceschi, Pontil, Majumder & Frasconi, "MARTHE: Scheduling the Learning Rate
  via Online Hypergradients", IJCAI 2020.** Interpolates between the greedy hypergradient
  and a longer-horizon estimate, precisely to attack the greedy bias. Empirical.
- **Chu, Gao, Ye & Udell, "Provable and Practical Online Learning Rate Adaptation with
  Hypergradient Descent", arXiv:2502.11229 (ICML 2025).** The most relevant modern
  citation. Verbatim from the abstract: "We provide the first rigorous convergence analysis
  of HDM using the online learning framework ... Our analysis explains the instability of
  HDM reported in the literature and proposes efficient strategies to address it." The
  guarantee comes from treating step-size selection as an online learning problem with
  regret bounds; the headline experiments are on **deterministic convex problems**. So the
  instability is now analysed rather than folklore — but the analysis does not cover a
  stochastic, non-convex, self-sampling objective, and I could not verify from the abstract
  which specific stabilisation carries the result.

### 1.4 The negative results that bear directly on this failure

- **Wu, Ren, Liao & Grosse, "Understanding Short-Horizon Bias in Stochastic
  Meta-Optimization", ICLR 2018** (arXiv:1803.02021). Verbatim: "We show that such
  short-horizon meta-objectives cause a serious bias towards small step sizes, an effect we
  term short-horizon bias." Proven on a noisy quadratic and confirmed empirically: even
  with 100-step unrolls, meta-optimisation picks learning rates smaller "by multiple orders
  of magnitude" than those that train the network. A one-step hypergradient is the extreme
  end of this spectrum. **This is a proven result and it predicts the direction of the
  observed failure.**
- **Pesme, Dieuleveut & Flammarion, "On Convergence-Diagnostic based Step Sizes for
  Stochastic Gradient Descent", ICML 2020** (PMLR 119:7641-7651). Verbatim: "We analyse the
  classical statistical test proposed by Pflug (1983), based on the inner product between
  consecutive stochastic gradients. Even in the simple case where the objective function is
  quadratic we show that this test cannot lead to an adequate convergence diagnostic." They
  replace it with a distance-based statistic (distance from the iterate at which the step
  size was last decreased). **This is a proven negative result about precisely this
  statistic, in the easiest setting for it.**
- **Chee & Toulis, "Convergence diagnostics for stochastic gradient descent with constant
  learning rate", AISTATS 2018** (PMLR 84:1476-1485). Uses the same statistic as a *phase*
  detector and halves the LR when it fires. Note the contrast in usage: a one-shot trigger
  for a discrete cut, never a continuous error signal to integrate.

Caveat on sign conventions: the literature is not uniform about whether the diagnostic
"fires" on a positive or a negative running sum, and I could not extract the convention
verbatim from either PDF. The derivation in section 2.1 is self-contained and does not
depend on resolving that.

---

## 2. Why `cos` can be persistently negative at a small step size

### 2.1 The dominant explanation: it is a stationarity statistic, and stationarity is negative at every stable rate

Take the noisy quadratic every paper above uses. `theta` measured from the optimum,
`g_t = H theta_t + xi_t` with `E xi = 0`, `Cov(xi) = S`, `xi` independent across steps, SGD
with constant `eta`. In one eigendirection with curvature `lambda` and noise variance `s`:

    theta_{t+1} = (1 - eta*lambda) theta_t - eta*xi_t
    stationary variance   v = eta*s / (lambda*(2 - eta*lambda))

    E<g_{t+1}, g_t> = lambda^2 (1 - eta*lambda) v  -  eta*lambda*s
                    = -eta*lambda*s / (2 - eta*lambda)     < 0 for all 0 < eta < 2/lambda

    E|g|^2 = lambda^2 v + s = 2s / (2 - eta*lambda)

    =>   cos  ~  -eta*lambda / 2

Three consequences, and they are exactly the reported observations:

1. **The sign is negative for every stable step size.** There is no zero crossing to find
   at stationarity. The diagnostic intuition that motivated this review — "if the signal
   were meaningful it should cross zero once the step size is far below optimal" — is
   correct about a *transient* iterate and false about a *stationary* one. The statistic is
   bimodal in meaning: in the transient phase the first term dominates and the sign does
   encode overshoot; after equilibration it does not.
2. **Magnitude, not sign, carries the rate information**: `|cos| = eta*lambda_eff/2`, so
   `eta_stability / eta_current = 1/|cos|`. The observed -0.006 .. -0.07 reads as "the
   stability limit is 14x-170x above the current rate".
3. In `d` dimensions with per-direction noise `s_i`, `cos = -(eta/2) * mean_w(lambda)` with
   weights `w_i` proportional to `s_i/(2 - eta*lambda_i)` — a *noise-weighted* mean
   curvature. The normalisation hides anisotropy rather than exposing it (section 2.4).

(The last step replaces `E[cos]` with a ratio of expectations; in high dimension the
concentration makes that a good approximation, but it is an approximation.)

This also reconciles the two opposite failures already on record in this project: under
Adam on the equilibration surface the bench measured `cos` **positive at every rate** and
`peak_scale` ran to its upper bound; here on an MLE surface it is **negative at every rate**
and `peak_scale` ran to its lower bound. Same structural defect — no fixed point — with the
sign set by whether the surface is still in transient or has equilibrated. A warm-started
MLE stage against a *fixed* target starts near-stationary, which is why the negative reading
appears from step 1 and freezes the warmup envelope inside 100 steps.

**Measurement that confirms or refutes it.** Two, both cheap:

- **LR ladder (slope test).** Pin the rate at 4-5 values spanning ~30x,
  `lr_warmup_ratio: 1`, sensor live but not actuating, and regress median `cos` (second
  half of each run) on `log lr`. Prediction: `cos` negative at every rung, `|cos|` scaling
  roughly linearly with `eta`. `configs/hyperslope_aug17/make.py` is exactly this design,
  built for the conditional route; the finding here is that it should be re-pointed at the
  ring MLE surface, where the failure actually occurred.
- **Frozen-theta control.** On the same step, compute `cos` between two *independent*
  minibatch gradients evaluated at the *same* parameters, with no optimizer step between
  them. This is the null: it isolates sampling-induced correlation from dynamics.

### 2.2 Adam's per-parameter normalisation: the operand is not the gradient

`train.py:3639` sets `d = -self._hyper_prev_step`, i.e. minus the **realised displacement**,
which under Adam is `~ eta * mhat/(sqrt(vhat)+eps)`, not `eta * g`. The measured quantity is
therefore a *preconditioned* inner product `<g_t, P g_{t-1}>/(|g_t| |P g_{t-1}|)` with
`P = diag(1/(sqrt(vhat)+eps))`. This is deliberate and documented (`train.py:3755-3758`),
and it is the right choice for an actuator that scales the applied step. But it breaks the
identification with the hypergradient of section 1.2 and with the analysis of section 2.1:
`eta*lambda` becomes `eta*lambda_i/sigma_i`, and normalising by `|d|` rather than
`|g_{t-1}|` changes the constant. Adam is close to a sign method in the small-noise limit
(**Balles & Hennig, "Dissecting Adam: The Sign, Magnitude and Variance of Stochastic
Gradients", ICML 2018**), so the cosine is closer to a *sign-agreement* statistic than to a
curvature probe — which pushes it back toward RPROP, documented as failing at small batch.

**Measurement.** Compute both cosines on the same steps: `cos(g_t, -dtheta_{t-1})` (what
ships) and `cos(g_t, g_{t-1})` (Baydin's operand). One extra stored vector, one dot product.
If they differ in sign, or if the raw-gradient version crosses zero across the LR ladder
while the displacement version does not, the preconditioner is the mechanism.

### 2.3 Momentum: the operand is a ~10-step average, not the last gradient

With `beta1 = 0.9`, `mhat` is an EMA with an effective horizon of ~10 steps, so `cos`
compares a fresh gradient against a smoothed history. Two effects: it *raises* the
correlation under persistent drift, and it makes the one-step response of section 2.1
ill-defined — the negative stationary term accumulates across the EMA horizon rather than a
single step.

**Measurement.** Report the lag profile `cos(g_t, g_{t-k})` for `k = 1..20` at fixed LR. A
single negative lag-1 value with rapid decay is a one-step response effect; negativity
persisting across many lags points to a systematic drift or a moving target (section 2.6).
Equivalently, re-run one arm with `beta1 = 0` and see whether the sign changes.

### 2.4 Curvature anisotropy

A small number of stiff directions can dominate both the inner product and the norms
(**Gur-Ari, Roberts & Dyer, "Gradient Descent Happens in a Tiny Subspace",
arXiv:1812.04888, 2018**), and gradient descent on neural nets characteristically drives the
top curvature up until `eta*lambda_max ~ 2` (**Cohen, Kaur, Li, Kolter & Talwalkar,
"Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability", ICLR
2021**). A single global cosine averages a stiff subspace sitting at the stability edge with
a bulk far below it — and per section 2.1(3) that average is *noise-weighted*, not
curvature-weighted.

**Measurement.** Recompute `cos` per parameter group (per-submodel gradient norms,
`gradnorm/*`, already exist) and report the spread. If one block carries the whole negative
number while the rest sit near zero, the global scalar is an artefact of pooling and no
single global rate is correct.

### 2.5 Heavy tails and clipping

Clipping is a uniform rescale, so it cannot change a cosine directly on the step it fires;
it changes the *dynamics*. The controller already has a regime gate for the saturated case
(`controller.py::_clip_saturated`) — which, note, only ever cuts. Heavy-tailed gradient
noise (**Simsekli, Sagun & Gurbuzbalaban, "A Tail-Index Analysis of Stochastic Gradient
Noise in Deep Neural Networks", ICML 2019**; **Zhang, He, Sra & Jadbabaie, "Why gradient
clipping accelerates training", ICLR 2020**) makes a windowed *mean* cosine a poor summary:
one outlier dominates the EMA operand for ~10 steps.

**Measurement.** Report median and IQR of `cos` alongside the mean over each window, plus
the clip fire rate for the failing run. A mean far from the median is the tell; the
controller integrates the mean.

### 2.6 An LR-independent bias: the target moves, or the batches are anti-correlated

This is the class that survives if the LR ladder shows `|cos|` **not** shrinking with `eta`.
Candidates specific to this trainer:

- **Sampling without replacement from a finite pool.** For a finite population whose
  gradients sum to (near) zero, distinct minibatch gradients have *negative* expected
  covariance, of order `-Var/(N-1)`. That gives a small, persistent, LR-independent negative
  cosine of order `-1/(N-1)` — numerically in the observed range for `N ~ 15-170`. Whether
  it applies depends on whether the `bwd_sampling_mode: dataset` draw is a fresh i.i.d.
  draw from the fitted prior each step or a pass over a finite buffer.
- **A self-generated target.** The MLE branch fits trajectories whose terminals come from a
  distribution the training also perturbs. Any coupling between "where the last step moved
  the policy" and "what the next batch looks like" enters `cos` directly and does not vanish
  as `eta -> 0`.
- **Cross-branch differencing** — an implementation hazard rather than a statistical one;
  see section 6.2 item 6. It does not apply to the failing stage, which trains one branch.

**Measurement.** The frozen-theta control of section 2.1 discriminates all of these from the
dynamics in one shot. Under i.i.d. sampling from a fixed distribution its expectation is
`+|grad f|^2 / E|g|^2 >= 0`; a negative value is direct evidence of anti-correlated
sampling.

### 2.7 What the null is

At 6.16M policy parameters, the null for a cosine between independent isotropic vectors is
`sqrt(2/(pi*d)) ~ 3e-4`. The observed -0.006 .. -0.07 is 20x-230x the null, so the effect is
a genuine systematic, not sampling noise in the cosine itself. Worth stating plainly:
**the sensor is measuring something real; the error is in what it is taken to mean.**

---

## 3. Known failure modes and stability

### 3.1 The multiplicative-exponential form: when is it stable?

Write `p = log(peak_scale)`, `err = cos - cos_target`, and let `s = d(cos)/d(log lr)` be the
local slope of the statistic in the rate. Linearising around a candidate fixed point:

    p_{t+1} = p_t + b*(c0 + s*p_t)

- A fixed point exists **only if `s != 0`**, and it attracts only if `s < 0`.
- Local stability then requires `-2 < b*s < 0`, i.e. `beta < 2/|s|`; overshoot begins at
  `b*|s| > 1`.
- If `s ~ 0` the loop is **open**: `p` ramps at `b*c0` per step without limit and **no value
  of `beta` helps.** Saturation at `bounds` is the only thing that stops it — which is why
  the failure looks like "railed at 0.01" rather than "settled somewhere unhelpful".

**The sign of `s` is not the discriminator; the LOCATION of the zero crossing is.** Under
the stationarity law of section 2.1, `cos = -eta*lambda/2`, so

    s = d(cos)/d(log lr) = eta * d(cos)/d(eta) = -eta*lambda/2 = cos < 0

The slope is *negative* — the loop is locally stable in the textbook sense — but the only
solution of `cos = 0` is `eta = 0`. The fixed point is at the floor. So a negative measured
slope does **not** license the controller: it is equally consistent with a healthy interior
equilibrium and with a monotone glide into the floor, and the two are told apart only by
where the zero crossing sits.

The two regimes give distinguishable ladder signatures:

- **Transient / curvature regime (method viable):** `cos` runs from near `+1` at small `eta`,
  crosses zero at `eta ~ 1/lambda`, and reaches `-1` at `eta ~ 2/lambda`. The crossing is
  interior and is what the SGD bench cell measured.
- **Stationary regime (method rails):** `cos <= 0` at every rung with `|cos|` roughly
  *proportional to* `eta` — halve the rate, halve the magnitude — and no crossing anywhere in
  the band. The apparent "fixed point" is `eta = 0`.

`configs/hyperslope_aug17` measures the right thing; the decision rule attached to it has to
be about the crossing, not the slope. **This is still the single missing number**, and it
remains unmeasured on the MLE route.

### 3.2 Noise rectification by the asymmetric gain

`beta_down = beta * hyper_down_gain`, default `2.0` (`controller.py:598`). For an error
with symmetric zero-mean noise of s.d. `sigma`:

    E[delta log p] = beta*E[err+] - gamma*beta*E[err-] = -beta*(gamma-1)*E|err|/2
                   ~ -beta*(gamma-1)*0.4*sigma

At `beta = 0.1`, `gamma = 2` and the measured `sd(cos)` of 0.062-0.168 (recorded in
`configs/hyperslope_aug17/make.py`), that is **-0.0024 to -0.0068 per firing** — the floor
reached in 700-1900 firings from a pure-noise input. The asymmetry is defended in the code
as "a cut is the recoverable direction", which is sound as a *risk* argument and unsound as
an *estimator*: it converts variance into drift. With the observed mean of -0.03 the signal
term is `0.2*0.03 = 0.006` per firing (rail in ~770 firings) — the same order as the
rectification term. Both are present; neither alone is small.

### 3.3 Integrator without a leak

Documented correctly in the code (`controller.py:646-674`): pole on the unit circle,
infinite DC gain, unbounded random walk under zero-mean noise, unbounded drift under bias.
`peak_leak` implements the fix and defaults to 0.0. The stated reason for the default — that
`lam` encodes a timescale measured on only one route — is a defensible epistemic position,
but its consequence is that **the shipped configuration is the one with no restoring force
at all.**

### 3.4 Short-horizon bias

Proven (Wu et al., ICLR 2018): greedy meta-optimisation of the step size is biased toward
step sizes that are too small, by orders of magnitude, and the bias *increases* as the
horizon shortens. A one-step hypergradient is horizon 1. The `ray` sensor already carries an
explicit margin for the analogous effect (`alpha_target = 4.0`, i.e. run at a quarter of the
one-step optimum); `hyper`'s `cos_target` is the same idea and ships at 0.0 on every arm.

### 3.5 Documented instability of HDM

Chu, Gao, Ye & Udell (2025) state that instability of hypergradient descent is "reported in
the literature" and that their analysis explains it. The ratcheting seen here is not a local
implementation curiosity — it is the known failure mode of the family, now with an analysis
attached, albeit in the deterministic convex setting.

### 3.6 Interaction with warmup

Specific to this implementation and, on the evidence, decisive. `hyper` deliberately runs
*through* the warmup ramp (unlike `ray` and `plateau`, which hold) so that it can terminate
the ramp early. The termination test is the sign of a 25-step EMA of the same error
(`controller.py:616-633`). A statistic that is negative at stationarity therefore freezes the
ramp almost immediately — here at envelope 0.1194, roughly step 77 of a 1000-step ramp — and
the same statistic then rails `peak_scale`. **The two authorities are not independent, so
their errors multiply**: 0.1194 x 0.01 = 1/838.

---

## 4. Canonical modifications

For each: what it fixes, what it costs, what evidence exists.

| Modification | Fixes | Cost / risk | Evidence |
|---|---|---|---|
| **Deadband on `err`** (no move while `\|err\| < tau`) | Noise-driven random walk; with an asymmetric gain, most of the rectification bias too | A dead zone also suppresses genuine small signals; `tau` is a new knob that must be set from the measured `sd(cos)` | Standard control practice; no ML paper I can point to that fixes a value. Folklore. |
| **Normalise `err` by its running s.d.** | Makes `beta` scale-free across surfaces — the stated reason `beta` has no universal value | **Does not fix a bias.** It rescales a biased signal to unit variance and leaves the sign; if the mean is systematically negative it rails just as surely, and faster when the signal is quiet | Analogous to Adam's own normalisation; no direct evidence for the hypergradient case |
| **Leak** (`p <- (1-lam)p + b*err`) | Turns unbounded drift into a bounded offset `b*err_bar/lam`; a random walk into a stationary spread | Caps total authority, so a genuinely needed 10x move may be unreachable; `lam` is a per-route timescale | Already implemented and reasoned out at `controller.py:646-674`; the principle is textbook control, not an ML result |
| **Symmetric bounds** | Current `[0.01, 2000]` is -4.6 vs +7.6 nats: the floor is 1.7x closer in log space, so a downward ratchet rails sooner and the run "dies quietly" | Loses headroom | Arithmetic |
| **Symmetric gain** (`hyper_down_gain = 1.0`) | Removes the rectification term of section 3.2 entirely | Loses the deliberate safety asymmetry; a hot excursion is corrected more slowly | Arithmetic (3.2). The code records that `beta_down` was plumbed but unset for most of the sensor's life, so historical runs were symmetric and the present default is a behaviour change |
| **Measure `cos` on the raw gradient** | Restores the identification with Baydin's hypergradient and with the section 2.1 analysis; removes preconditioner and EMA from the operand | The actuator scales the *applied* step, which is preconditioned, so sensor and actuator would then measure different objects | Baydin et al. 2018 derive the raw form; the displacement form is this repo's extension |
| **Longer horizon** (average `cos` over K steps before acting; or an RTHO-style multi-step hypergradient) | Short-horizon bias — the single proven pathology | Cost grows with horizon; multi-step estimates need extra memory/compute | Wu et al. ICLR 2018 (the bias); Donini et al. IJCAI 2020 (MARTHE, empirically better than greedy HD) |
| **Multiplicative vs additive** | Multiplicative is right for a quantity on a log scale and bounds one move by `exp(beta)` | Makes the loop an integrator in log space, so 3.3 applies | Baydin's original is additive; the multiplicative/cosine form is this repo's defensible modification |
| **Per-parameter-group rather than global** | Anisotropy (2.4): one global scalar averages a stiff subspace with a flat bulk | N independent integrators, N times the ratchet risk, and per-group rates interact | Delta-bar-delta and RPROP are per-weight versions of exactly this; both are documented as noise-sensitive at small batch |
| **Gate on a stationarity test instead of actuating on the sign** | The category error itself: use the statistic for what it was invented for | Gives a *cut* trigger only, not a continuous servo | Chee & Toulis AISTATS 2018; Lang, Xiao & Zhang NeurIPS 2019 (SASA) — but see Pesme et al. ICML 2020 for the proof that the Pflug form is inadequate, and their distance-based replacement |

---

## 5. Alternatives for the same job

- **LR range test / one-cycle** — Smith, "Cyclical Learning Rates for Training Neural
  Networks", WACV 2017; Smith & Topin, "Super-Convergence", arXiv:1708.07120; Smith,
  "A disciplined approach to neural network hyper-parameters", arXiv:1803.09820. Cheap,
  empirical, well-attested, and it produces *a number a human owns*. Costs one short
  throwaway run per surface; gives no online adaptation. For a warm-start stage on a fixed
  target this is close to the best value-for-effort available.
- **D-Adaptation** — Defazio & Mishchenko, "Learning-Rate-Free Learning by D-Adaptation",
  ICML 2023. Estimates distance-to-solution `D` online with a proven bound for convex
  Lipschitz problems, matching the optimally-tuned rate up to constants. Convexity does not
  hold here; the empirical record on deep nets is nonetheless strong. Monotone `D` estimates
  make it hard to *lower* the rate after a regime change.
- **Prodigy** — Mishchenko & Defazio, "Prodigy: An Expeditiously Adaptive Parameter-Free
  Learner", ICML 2024. Improves D-Adaptation's rate by `O(sqrt(log(D/d0)))`; a drop-in Adam
  variant exists and is widely used. Same convexity caveat.
- **DoG** — Ivgi, Hinder & Carmon, "DoG is SGD's Best Friend: A Parameter-Free Dynamic Step
  Size Schedule", ICML 2023. Step size from distance-from-init over accumulated gradient
  norms; no learning rate at all. Proven for convex; empirically needs the
  polynomial-averaging variant (L-DoG) to be robust.
- **Mechanic** — Cutkosky, Defazio & Mehta, "Mechanic: A Learning Rate Tuner", NeurIPS 2023.
  Tunes a *scale factor* on top of any base optimizer and schedule, from an online convex
  optimisation reduction. Architecturally the closest match to `peak_scale`: same actuator,
  regret-based update instead of a sign heuristic, and the theory addresses exactly the
  ratchet problem.
- **Adafactor-style relative step sizes** — Shazeer & Stern, "Adafactor: Adaptive Learning
  Rates with Sublinear Memory Cost", ICML 2018. Scales the update relative to the parameter
  norm, removing the absolute LR scale. Not a servo; a reparameterisation.
- **Plain cosine decay with a hand-tuned peak** — Loshchilov & Hutter, "SGDR", ICLR 2017.
  The baseline every method above is measured against, and the one that already works on
  this surface (`configs/conformer_ring_mle_fixedlr.yaml`, 1e-4 pinned). Costs a tuning
  sweep per route and adapts to nothing.
- **Gradient-noise-scale sizing** — McCandlish, Kaplan, Amodei et al., "An Empirical Model of
  Large-Batch Training", arXiv:1812.06162, 2018. Estimates `B_simple = tr(H S)/(g' H g)` and
  predicts where batch-size returns diminish. It answers "how big should the batch be", and
  thereby bounds how much of the observed cosine is noise — a *diagnostic* for section 2,
  not a rate servo. Directly relevant: the term a one-step probe cannot see, `tr(H S)`, is
  exactly what `alpha_target = 4.0` exists to cover.
- **Stochastic line search** — Vaswani, Mishkin, Laradji, Schmidt, Gidel & Lacoste-Julien,
  "Painless Stochastic Gradient: Interpolation, Line-Search, and Convergence Rates", NeurIPS
  2019; Paquette & Scheinberg, SIAM J. Optim. 30(1), 2020. Proven rates under interpolation.
  **The repo's own `ray_calibration.py` is a paired-sample stochastic line search and is the
  strongest asset here** — it measures the loss response along the real step, which is the
  quantity `cos` only proxies. Its documented limitation is that it needs a replay draw and
  a loss the stage actually trains, which is why `hyper` was reached for on this stage.
- **Statistical stationarity tests** — Lang, Xiao & Zhang, "Using Statistics to Automate
  Stochastic Optimization", NeurIPS 2019 (SASA), built on Yaida, "Fluctuation-Dissipation
  Relations for Stochastic Gradient Descent", ICLR 2019; and Zhang, Lang, Liu & Xiao,
  "Statistical Adaptive Stochastic Gradient Methods", arXiv:2002.10597 (SALSA, which pairs an
  LR-range warmup with the SASA test). This family does what `cos` is *actually* capable of —
  detect equilibration, then cut — with a principled test.
- **Barzilai-Borwein step sizes** — Tan, Ma, Dai & Qian, "Barzilai-Borwein Step Size for
  Stochastic Gradient Descent", NIPS 2016. Uses the secant pair `(dtheta, dg)` rather than a
  cosine, so it estimates a curvature *magnitude* instead of only a sign. Proven for SVRG-BB
  under strong convexity.

---

## 6. Implementation audit

### 6.1 What is correct

- **The sign convention is right.** `train.py:3639` uses `d = -self._hyper_prev_step`, i.e.
  minus the realised displacement, so `cos > 0` means the current gradient agrees with the
  previous *descent* direction, and the docstring identity `dL/d(lr) = -<g_t, d_{t-1}>` is
  consistent. There is no sign error. (Worth checking explicitly: had `d` been the
  displacement itself, the correct rule would invert, and a persistent small negative `cos`
  would have meant "step too small" in the textbook Wolfe-curvature sense.)
- **Scale-freeness.** The cosine removes the `|g|^2` units of Baydin's raw hypergradient and
  bounds one move by `exp(beta)`. A genuine improvement over the additive original for a
  multiplicative actuator.
- **Clip-saturation gate.** `_clip_saturated` is a real regime guard: when the clip binds
  every step, the update magnitude is set by the LR alone and `cos` stops carrying curvature
  information.
- **Warmup awareness exists.** The reasoning at `controller.py:560-572` about not actuating
  through a deliberate suppression is correct.
- **The right measurement was already specified.** `configs/hyperslope_aug17/make.py` states
  the decisive question — `d(cos)/d(log lr)` — and designs a clean ladder for it, including
  the crucial detail that the LR must be pinned so the regression is not circular.

### 6.2 Defects and hazards, in order of consequence

1. **No interior fixed point is established on this surface** (3.1). The zero crossing of
   `cos` in `lr` has not been located on the MLE route, and section 2.1 predicts there is
   none: the only solution of `cos = 0` at stationarity is `eta = 0`, i.e. the controller's
   equilibrium is the floor it railed into. Everything else is secondary.
2. **Two authorities on one statistic** (3.6, `controller.py:616-633`). The warmup freeze
   and the `peak_scale` integrator both key on the sign of the same error, so a biased
   statistic produces a multiplicative, not additive, error: 0.1194 x 0.01 = 1/838. The
   envelope freeze is also latched and, per the comment at `controller.py:761-768`, survived
   stage transitions until that was fixed.
3. **Asymmetric gain rectifies noise into drift** (3.2, `controller.py:598`). Default
   `hyper_down_gain = 2.0` rails the floor from pure noise in 700-1900 firings at the
   measured `sd(cos)`.
4. **No leak by default** (3.3, `controller.py:663`). The remedy is implemented and off.
5. **Asymmetric bounds.** `[0.01, 2000]` is -4.6 vs +7.6 nats. Downward failures rail 1.7x
   sooner than upward ones and are far less visible: a run at 1/800 rate looks like a model
   that cannot fit, not like a controller fault.
6. **`_hyper_prev_step` is a single slot, not keyed by branch or optimizer**
   (`train.py:3735-3758`). `_hyper_sensor_cfg` admits `fused`, `fwd`, `bwd` and `replay`,
   and each has its **own Adam instance with its own moments and its own LR**
   (`train.py:2169-2171`). On an unfused, turn-taking stage the sensor differences a gradient
   from one branch against a displacement produced by a *different* branch's optimizer, so
   the cosine mixes two loss families at two rates. This does **not** explain the reported
   failure — `train_prior` is `train_mode: bwd`, a single branch — but it is a live hazard
   for any multi-branch stage declaring `kind: hyper`.
7. **The published statistic and the actuated one differ when `cos_target != 0`.**
   `lr_ctrl/hyper_cos` reports the raw cosine while the actuator uses `cos - cos_target`
   (already documented in `configs/hyperslope_aug17/make.py:58-63`). Anyone reading the
   dashboard to check the controller is reading the wrong series.
8. **Every regime gate is one-way down.** `_clip_saturated` cuts; divergence cuts; the
   asymmetric gain cuts faster than it raises. There is no symmetric mechanism that detects
   "the rate has been pinned at the floor for N steps and nothing is improving" — which is
   what would have caught this in hours rather than 20,000 steps.

### 6.3 Numbers, for the record

- Live config on the failing stage: `lr_sensor: {kind: hyper, beta: 0.1}`, so
  `beta_down = 0.2`; `peak_leak` unset (0.0); `bounds [0.01, 2000]`; `warmup_steps 1000`,
  `lr_warmup_ratio 10`, `warmup_freeze_cos_window 25`, `envelope_freeze true`;
  `seed_lr 1.25e-4`; `use_weight_decay: false`, so decoupled-decay explanations for a
  persistent bias are ruled out; plain Adam with `weight_decay = 0`.
- Time to rail at the observed mean `cos ~ -0.03`: `ln(0.01)/(0.2*0.03) = 767` firings.
- Envelope frozen at 0.1194 gives `0.1194 * 0.01 = 1.19e-3` of seed = **1/838**, matching the
  reported 1/800. Live rate ~1.5e-7 against a working pinned rate of 1e-4.

---

## 7. Recommendation

**7.1 Do not let this sensor own the rate on MLE stages until `s = d(cos)/d(log lr)` has been
measured on an MLE surface.** The pinned-rate config already shows the surface trains at
1e-4. That is the correct interim state; it is not a defeat.

**7.2 The one measurement that decides the method's fate.** Re-point the
`configs/hyperslope_aug17` design at the ring MLE surface: 4-5 arms, LRs spanning ~30x around
1e-4, `lr_warmup_ratio: 1`, rates pinned as explicit floats so the sensor reads and logs but
does not actuate, same seed, ~2000 steps. Regress median `cos` over the second half on
`log lr`. Decision rule, stated in advance:

- **No zero crossing in the band: `cos <= 0` at every rung with `|cos|` roughly proportional
  to `lr`** — confirms section 2.1. The only solution of `cos = 0` is `eta = 0`, so the
  controller's fixed point *is* the floor, and no `beta`, deadband, target or leak repairs
  it. Retire `hyper` as an *actuator* for MLE stages; keep it as a *reported diagnostic*,
  where `2|cos|/eta` estimates the noise-weighted curvature and `1/|cos|` the distance to
  the stability edge.
- **An interior zero crossing that coincides with the best fit metric**: the method is sound
  here and the failure was an aiming error. Then add, in this order: symmetric gain, a
  deadband at ~1 s.d. of `cos`, a leak sized so total authority is ~3x, and a `cos_target`
  placed from the ladder rather than at 0.
- **`cos` flat in `lr`**: the loop is open. The rate wants a schedule or the `ray` line
  search, not a servo.

Note that a *negative slope* is present in all three cases and is therefore not the
discriminator (section 3.1); only the crossing location is.

**7.2a A cheaper retrospective check, on data already logged.** The failing run published
`lr_ctrl/hyper_cos` and the live rate throughout, and the rate moved ~800x during it. Plot
one against the other. The stationarity law predicts `|cos|` falls roughly in proportion, so
if the reported spread is only the ~10x implied by -0.006 .. -0.07 while the rate moved 800x,
part of the negative reading is **LR-independent** and section 2.6 (anti-correlated sampling
or a moving target) is contributing a floor that the dynamics cannot explain. That
distinction changes which fix is worth attempting and costs nothing to obtain.

**7.3 Run the frozen-theta control alongside it** (2.1). Two independent batches at the same
parameters, cosine between their gradients, reported once per N steps. It costs one extra
backward pass on a sub-batch and separates "the dynamics are anti-correlated" from "the
sampling is anti-correlated". If the frozen-theta cosine is itself negative, no step-size
controller built on this statistic can work on this route at any setting.

**7.4 Two things are worth fixing regardless of the verdict**, because they are defects
rather than open questions: the warmup-freeze rule and the `peak_scale` integrator should not
both key on the sign of the same statistic (3.6), and the default asymmetric gain converts
noise into a monotone cut (3.2). Both are arithmetic claims, not empirical ones.

**7.5 If a servo is genuinely wanted on stages where `ray` cannot run**, the ordered shortlist
is: (a) extend `ray_calibration` to score the MLE loss so the existing paired line search
covers this stage — it measures the loss response along the real step, which is what `cos`
only proxies; (b) Mechanic (Cutkosky et al., NeurIPS 2023), which tunes exactly this actuator
— a scale factor on a base optimizer — with a regret bound instead of a sign heuristic;
(c) Prodigy for the warm-start stage specifically, since it fits a fixed target and is the
closest thing here to the setting where parameter-free methods are attested.

---

## 8. Where the literature does not cover this regime

Stated explicitly so these are not read as gaps in the review:

- **No source I found analyses the cosine-against-realised-Adam-displacement form.** The
  theory is for `<g_t, g_{t-1}>` under SGD. The preconditioned, momentum-smoothed,
  norm-normalised variant that ships here is this project's own construction.
- **No source analyses the statistic under a self-generated (on-policy) sampling
  distribution**, the GFlowNet case, where the objective between two consecutive steps is not
  literally the same function.
- **The `s > 0` claim of section 2.1 is derived for SGD on a quadratic at stationarity.** Its
  Adam analogue is not proved here or, as far as I can tell, anywhere. It is offered as a
  hypothesis with a falsifiable prediction, not as a theorem about this trainer.
- **Chu et al. (2025) is the only rigorous convergence analysis of the family**, and its
  strong empirical results are on deterministic convex problems. Nothing in it licenses use
  on this workload.
- The claim that `beta = 0.1` is the best worst-case over 12 bench cells is this project's own
  measurement; the literature offers no default, and Baydin et al.'s reported values are for
  the additive, unnormalised form and do not transfer.

---

## 9. Which repair is best supported, and why `ray` is the better default

### 9.1 Ranking the three repairs by evidence, not by appeal

1. **Replace the loop, keep the actuator (Mechanic) — strongest formal support.** Cutkosky,
   Defazio & Mehta (NeurIPS 2023) tune exactly this object — a scale factor on a base
   optimizer — via a reduction from online convex optimisation with regret guarantees. The
   decisive property for this failure: the guarantee does **not** rest on the sign of a
   curvature proxy carrying information, which is the assumption that broke. Behind it, the
   D-Adaptation / Prodigy / DoG family carries proven convex bounds and strong empirical
   records. Caveat: the guarantees are convex/OCO; the deep-learning evidence is empirical.
2. **Demote to a cut trigger — strongest empirical support for the *architecture*, but not
   for this statistic.** "Hold a constant rate, test for stationarity, cut" is well attested
   (Chee & Toulis AISTATS 2018; Lang, Xiao & Zhang NeurIPS 2019; SALSA, arXiv:2002.10597,
   which pairs an LR-range warmup with the same test). But SASA uses Yaida's
   fluctuation-dissipation test and Pesme et al. use a distance-based statistic —
   *specifically because* the consecutive-gradient form is proved inadequate (ICML 2020).
   Reusing `cos` as the trigger inherits that negative result. **Keep the architecture,
   replace the statistic.**
3. **Magnitude-based deadbeat solve (`eta_new = c*eta/|cos|`) — weakest.** I found no paper
   that sets a rate from the stationary gradient autocorrelation this way. Its two nearest
   relatives are real but do not validate it: Schaul, Zhang & LeCun, "No More Pesky Learning
   Rates", ICML 2013 (vSGD) has exactly this *shape* — compute the rate in closed form from
   gradient mean, variance and a curvature estimate rather than nudge it — and Yaida,
   ICLR 2019, whose fluctuation-dissipation relations are the same type of identity as
   `cos = -eta*lambda/2`. vSGD did not see wide adoption, which is itself weak evidence.

**The overlay that governs all three.** Schmidt, Schneider & Hennig, "Descending through a
Crowded Valley: Benchmarking Deep Learning Optimizers", ICML 2021, ran >50,000 runs over
fifteen optimizers. Verbatim: "we cannot discern an optimization method clearly dominating
across all tested tasks", and "Adam remains a strong contender, with newer methods failing to
significantly and consistently outperform it". The burden of proof therefore sits on any
controller, not on the pinned rate — and `configs/conformer_ring_mle_fixedlr.yaml` training
this surface at 1e-4 is a local instance of that finding.

### 9.2 Why `ray` is the better general-purpose controller

Four reasons, three of which the literature makes precise.

1. **It scores the objective; `hyper` scores a proxy.** A line search's decision variable is
   the actual loss change along the actual step. There is no regime in which its meaning
   inverts. `cos` inverts between transient and stationary (section 2.1), and the regime is
   not observable from the statistic. This is the whole argument in one line.
2. **It is a test with an abstain state, not an unconditional update.** This is exactly the
   structure the modern noisy-line-search theory requires: Paquette & Scheinberg, "A
   Stochastic Line Search Method with Expected Complexity Analysis", SIAM J. Optim.
   30(1):349-376, 2020, and Berahas, Cao & Scheinberg, "Global Convergence Rate Analysis of
   a Generic Line Search Algorithm with Noise", SIAM J. Optim. 31:1489-1518, 2021, prove
   complexity bounds for line search on noisy values and inexact gradients **under the
   condition that the estimates are sufficiently accurate with sufficiently high probability
   at each iteration**. The `|t| > 2` gate and the `inconsistent -> no move` rule are that
   condition, implemented. `hyper` has no such condition: every firing moves the actuator
   whatever the evidence. This is the largest structural difference between the two sensors.
3. **Pairing is what makes the test resolvable.** Scoring `L(0)` and `L(2a)` on the same
   sub-batch is common random numbers, a classical variance-reduction technique in
   simulation optimisation. The measured 30x in `t` (-31.7 paired vs +0.7 unpaired on the
   same data) is that effect, and without it the accuracy condition in (2) is unattainable
   at this noise level.
4. **Bounded proportional response vs unbounded integrator.** `peak_scale *=
   (alpha_hat/alpha_target)^eta` on a log-spaced grid saturates at the grid edge, so a biased
   reading buys a bounded offset; `hyper` is a pure integrator in log space, where bias buys
   unbounded drift. The observed behaviours match: `ray` held a +-17% sawtooth for 8800
   steps; `hyper` railed.

**Where `ray` is not better, stated plainly.**

- Vaswani et al. (NeurIPS 2019) prove rates for stochastic line search **under
  interpolation** — the model fits every sample exactly. That does not hold for GFN MLE with
  irreducible trajectory noise, so those rates do not transfer. Paquette-Scheinberg and
  Berahas-Cao-Scheinberg are the right framework, and they give complexity bounds under a
  probabilistic-accuracy assumption, not superiority over a tuned constant rate.
- **Short-horizon bias applies to `ray` too** (Wu et al., ICLR 2018): a one-step probe at
  frozen `theta` cannot see `tr(H*Sigma)`. `alpha_target = 4.0` is an empirical margin for
  exactly that, not a derived one, and it must be re-measured per route. `ray` is not immune;
  it carries the margin explicitly, which is better than not carrying one.
- **It is periodic, not continuous**, so the exposure window between calibrations is the
  binding constraint — the 12x-hot test was unrecoverable within one period. A fast guard
  between calibrations is a separate mechanism it still needs.
- **Coverage is its real limitation**: it needs a re-drawable batch and a loss the stage
  actually trains. That gap is why `hyper` exists, and closing it is worth more than
  repairing `hyper`.

So `ray` is better in the sense that matters operationally — it measures the thing it is
controlling, and it abstains when it cannot resolve — not in the sense of having a
guarantee that transfers to this workload. Neither sensor has that.

---

## 10. Set-and-forget options when the optimal rate is non-stationary

The three properties in "safe, general, set-and-forget" are not the same and they trade
against each other. **Safe** means the failure mode is bounded: an open-loop schedule is safe
because it has no feedback path and therefore cannot ratchet; a servo is safe only if it can
abstain, leaks, and has symmetric bounds. **General** means it works across stages without a
per-stage number. **Set-and-forget** means no per-route number at all. Schmidt, Schneider &
Hennig (ICML 2021) is the standing evidence that nothing yet is all three.

### 10.1 First: separate the two non-stationarities

- **Known, event-driven** — stage transitions, optimizer rebuilds, a changed loss mixture.
  These are events the trainer already knows about. The correct response is not a servo but a
  re-armed warmup at the event, which `rearm_warmup` already does. Supported by Liu, Jiang,
  He, Chen, Liu, Gao & Han, "On the Variance of the Adaptive Learning Rate and Beyond",
  ICLR 2020 (warmup as variance control for adaptive methods) and by Loshchilov & Hutter,
  "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017.
- **Unknown, continuous drift within a stage.** This is the only case that needs online
  control, and it is the case that has not been separated from measurement noise here. See
  10.4.

### 10.2 What is safe under drift

1. **Cyclical or restarting schedules (open-loop).** Smith, "Cyclical Learning Rates for
   Training Neural Networks", WACV 2017; Loshchilov & Hutter, ICLR 2017. The safety property
   is structural: with no feedback path, a schedule **cannot rail**, and because it re-visits
   the whole usable band every cycle it catches a drifting optimum within one period. Cost:
   time is spent at wrong rates by construction. Unfashionable, and the strongest safety
   guarantee on this list.
2. **Schedule-Free (Defazio, Yang, Khaled, Mishchenko, Mehta & Cutkosky, "The Road Less
   Scheduled", NeurIPS 2024).** Removes the schedule entirely — iterate averaging does the
   work decay would — so no stopping time `T` need be specified, which matches a staged
   protocol whose stage lengths are not known in advance. It introduces no extra
   hyperparameters over momentum SGD/AdamW. **Schedule-Free AdamW won the MLCommons 2024
   AlgoPerf Self-Tuning track**, which is the closest thing the field has to a competition for
   exactly "safe, general, set-and-forget": one configuration, many workloads, no per-workload
   tuning. It still requires a base learning rate, so it is set-and-forget for the schedule's
   *shape*, not its *scale*.
3. **Mechanic (Cutkosky, Defazio & Mehta, NeurIPS 2023)** for the scale, on top of either of
   the above. Its update is derived in the online learning setting, which is the setting that
   admits a moving comparator, and it tunes exactly the `peak_scale` actuator.
4. **Move the batch, not the rate.** Smith, Kindermans, Ying & Le, "Don't Decay the Learning
   Rate, Increase the Batch Size", ICLR 2018, show that decaying the rate and increasing the
   batch produce equivalent learning curves for SGD, momentum, Nesterov and Adam. The
   asymmetry that matters here is a safety one: **increasing the batch cannot destabilise
   training; raising the rate can.** If the optimal rate falls because the run has entered a
   noise-dominated regime — which is what section 2.1 predicts — enlarging the batch is the
   strictly safer actuator with the same effect, and `train.select_batch_size` already exists.
   Sizing rule: McCandlish et al., arXiv:1812.06162, 2018.
5. **Bound the update rather than the rate.** LARS (You, Gitman & Ginsburg, arXiv:1708.03888,
   2017), LAMB (You et al., ICLR 2020) and Adafactor's relative step size (Shazeer & Stern,
   ICML 2018) all set the step as a fraction of the parameter norm, so a stale rate has a
   bounded consequence when the gradient scale shifts. With the existing gradient-clip guard
   this is a safety layer rather than a controller — and a safety layer is what makes any
   set-and-forget choice tolerable.

### 10.3 What is specifically wrong for a non-stationary optimum

**The parameter-free distance family is the wrong tool here, despite being the best-proved.**
D-Adaptation maintains a lower bound `d_k` on the distance to the solution that is
**monotonically non-decreasing by construction**; DoG's `r_bar` is a running maximum; AdaGrad's
denominator is a growing sum. All three are built around a single fixed target and can only
move their estimate one way. Under a staged protocol whose target changes, the estimate
carries stale information across the change and cannot come back down. (The monotonicity is
stated in the D-Adaptation and DoG papers; the consequence for staged non-stationarity is my
inference, not a published result.) Prodigy inherits the same structure.

### 10.4 The check that decides whether a servo is needed at all

The premise "the optimal LR is non-stationary" has not been separated from measurement noise
on this trainer, and the evidence to do it already exists in the `ray` calibration history.
Run `5t7ny5lw` held a median 1.375e-4 within a +-17% band for 8800 steps — that is a *stable*
within-stage optimum, and the residual sawtooth was explicitly not separated from sub-batch
overlap inflating `t`. Against that, `alpha*` drift of 1.6x per 1000 steps at constant rate was
measured on the same route.

So: pull `alpha_hat` against step for every stage that has run `ray`, and split the variance
into within-stage drift and between-stage jumps. If the drift is dominated by stage
transitions, **no servo is required** — a per-route peak from a range test plus the existing
re-armed warmup covers it, and every failure mode in this document disappears with the
controller. If within-stage drift is real and large, the ranking in 10.2 applies. This is a
query over data already on disk, and it should precede any further controller work.

---

## 11. MEASURED: alpha* drift decomposition from the ray logs (2026-08-20)

Section 10.4 asked for this and it is now done. Nineteen runs carrying
`raycal/alpha_star` were pulled — 14 local, 5 from the cluster (`analysis.pull`) —
the largest being `prod0810_mipcas_elj` (67 bracketed calibrations over 33,500
steps) and `prod0810_nehzor_elj` (58 over 29,000).

**Two methodological points, both load-bearing.**

- `raycal/*` is a **sticky summary**: it is republished on every reporting row, so
  the raw series counts DWELL TIME, not calibration events, and weights each
  reading by how long it happened to persist. `5t7ny5lw` reads as 1281
  "calibrations" until deduplicated on the `lr_ctrl/calibrations` counter, at which
  point it is the documented 26. Every number below is event-deduplicated.
- **`alpha_star` is the wrong variable.** Under a live servo it is a closed-loop
  residual: the controller moves the rate back toward `alpha_target` after every
  reading, so a drifting `alpha_star` can mean the servo is converging rather than
  the optimum moving. The servo-invariant quantity is

      implied one-step-optimal LR  =  (live LR at the probe) x alpha_star

  and that is what is decomposed.

### 11.1 Within a stage the optimum does not drift

| run | bracketed n | span (steps) | implied-opt sd | trend |
|---|---|---|---|---|
| `prod0810_mipcas_elj` | 67 | 33,500 | 0.254 dex | -0.0049 dex/1k |
| `prod0810_nehzor_elj` | 58 | 29,000 | 0.201 dex | -0.0027 dex/1k |
| `prod0810_nehzor_uma` | 14 | 6,500 | 0.224 dex | +0.0016 dex/1k |
| `d33elj_..._sg14_t10_r2` | 10 | 5,000 | 0.140 dex | +0.0666 dex/1k |
| `5t7ny5lw` | 24 | 12,000 | 0.121 dex | -0.0001 dex/1k |

The two longest runs trend by 0.16 and 0.08 dex **across their entire stage**. No
linear drift worth acting on exists anywhere.

**The spread is white noise, not a wandering optimum.** The empirical variogram of
the implied optimum is flat at every separation tested (semivariance divided by
total variance; 1.00 = no temporal structure at all):

    prod0810_mipcas_elj (n=67, total var 0.0645)
      lag   0-1k   1-2.5k   2.5-5k   5-10k   10-20k   20-40k
            1.02   1.02     1.03     1.01    1.03     0.98

    prod0810_nehzor_elj (n=58, total var 0.0405)
            1.18   1.03     1.03     0.96    1.04     1.02

Two calibrations 500 steps apart disagree by exactly as much as two 40,000 steps
apart. A wandering optimum would make near pairs agree better and the variogram
rise; it does not, at any separation. Lag-1 autocorrelation of consecutive
calibrations is -0.057 (se 0.12) and -0.169 (se 0.13); `var(diff)/(2*var)` is 1.04
and 1.17 against 1.00 for pure white noise.

Per-reading noise is 0.20-0.25 dex (+-60-75%), against a grid quantisation floor of
0.087 dex, so single readings are genuinely noisy. But the **mean** over 67 readings
has a standard error of 0.031 dex: **the optimum is pinned to about +-7% and is
constant over 33,500 steps.**

### 11.2 The "1.6x per 1000 steps drift" was the servo, not the optimum

On `5t7ny5lw`, `alpha_star` alone trends -0.045 dex/1k and its trend carries 47% of
its variance. The implied optimum on the same readings trends **-0.0001 dex/1k, 0.0%
of variance**. The run was seeded 6x low and climbed; the trend was in the
closed-loop residual. This resolves the open question left in the v8 record.

### 11.3 Where the ray probe can be validated, it passes decisively

`5t7ny5lw`'s climb is a quasi-open-loop sweep (seeded 6x low, LR moved 8.7x), so
`d log(alpha*) / d log(lr)` is interpretable there:

    -0.891 +- 0.114     t vs 0 = -7.8      t vs -1 = +1.0

-1 is the signature of a perfectly tracked **fixed** optimum; 0 is a sensor carrying
no rate information. The probe is indistinguishable from the former and decisively
not the latter. **This is the same slope test section 7.2 asks for on `cos`, run on
`ray` where the data exists.** On runs holding at setpoint the regression is
circular — the LR is a function of `alpha_star` — and reads near zero by
construction, so `prod0810_mipcas_elj`'s +0.10 +- 0.22 is not evidence of sensor
failure.

### 11.4 The asymmetric gain's noise rectification, measured in production

`ray` ships `eta_up 0.25` against `eta_down 0.5`, so the rectification analysed for
`hyper` in section 3.2 applies here too: for `x = log10(alpha_hat/target)` the
expected applied move `0.25*E[x+] - 0.5*E[x-]` is zero only at a **positive** mean,
i.e. the loop must sit slightly HOT of setpoint to stand still.

| run | measured sigma | predicted offset | observed offset | held alpha* |
|---|---|---|---|---|
| `prod0810_mipcas_elj` | 0.214 | +0.059 dex | **+0.058 dex** | 4.57 |
| `prod0810_nehzor_elj` | 0.186 | +0.051 dex | +0.075 dex | 4.75 |
| `prod0810_nehzor_uma` | 0.211 | +0.058 dex | +0.075 dex | 4.75 |
| `d33elj_..._sg14_t10_r2` | 0.157 | +0.043 dex | +0.136 dex | 5.47 |

Every run holds above target, in the predicted direction, and the best-sampled one
matches to 0.001 dex; the smaller-n runs read high. **This is the same mechanism
that railed `hyper`** — the difference is that `ray` has a genuine interior fixed
point, so rectification buys a bounded ~15% offset instead of an unbounded ratchet.
It also explains the documented +-17% hold sawtooth: the servo is responding to a
statistic the variogram says is white noise.

### 11.5 Between stages there is NO DATA, and that is structural

Across all nineteen runs, **`ray` produced zero calibrations outside phase 2**:

    prod0810_nehzor_elj   phase 1 spans steps      0-7,430   first calibration  7,500
    prod0810_nehzor_uma   phase 1 spans steps      0-15,660  first calibration 16,000
    d33elj_..._t10_r2     phase 1 spans steps 10,000-11,070  first calibration 11,500
    prod0810_mipcas_elj   phase 1 ends at 8,720              first calibration  9,000

Over 24,000 steps of phase-1 training with the sensor configured, it never once
fired. The cause is structural rather than incidental: phase 1 trains `bwd`, where
the ray probe is incoherent (it scores replay), and the probe additionally holds
through the re-armed warmup at every transition. So the between-stage component of
the variance is not small — **it is unmeasured, and unmeasurable at the sensor's
current placement.**

### 11.6 What this settles

1. **Within a stage, no servo is needed.** A constant rate sits within ~7% of the
   measured optimum for 30,000+ steps, and the variogram says there is nothing for a
   controller to track. Every within-stage move the servo makes is a response to
   white noise.
2. **The non-stationarity that matters is between stages, and it is exactly what
   neither sensor measures.** `ray` never fires in phase 1; `hyper` fires there and,
   per sections 0-3, reads a statistic whose sign is uninformative on an equilibrated
   surface.
3. **This reframes the `train_prior` failure.** On phase 1 the choice was never
   "servo vs no measurement" — it was "a biased measurement acted on every step vs a
   per-stage constant". The constant was strictly better, and
   `conformer_ring_mle_fixedlr` is the demonstration.
4. Priority order that follows: (a) a per-stage rate from a range test, plus the
   existing re-armed warmup; (b) `ray` coverage extended to phase 1 so the
   between-stage question can be *measured* rather than argued; (c) only then, if the
   between-stage spread turns out to be large and predictable, a controller.

Reproduce with `raydrift.py` / `raydrift_cloud.py` (session scratchpad; ephemeral,
rebuildable from this section).

---

## 12. Citation verification status

Verified this session against publisher/arXiv pages: Baydin et al. ICLR 2018; Wu et al. ICLR
2018 (abstract quoted); Chu, Gao, Ye & Udell arXiv:2502.11229 / ICML 2025 (abstract quoted);
Pesme, Dieuleveut & Flammarion ICML 2020 (abstract quoted); Chee & Toulis AISTATS 2018;
Lang, Xiao & Zhang NeurIPS 2019; Schraudolph ICANN 1999; Almeida et al. 1998 (CUP chapter,
pp. 111-134); Defazio & Mishchenko ICML 2023; Mishchenko & Defazio ICML 2024; Ivgi, Hinder &
Carmon ICML 2023; Cutkosky, Defazio & Mehta NeurIPS 2023; Vaswani et al. NeurIPS 2019;
Tan, Ma, Dai & Qian NIPS 2016; Paquette & Scheinberg SIAM J. Optim. 30(1):349-376, 2020;
Berahas, Cao & Scheinberg SIAM J. Optim. 31:1489-1518, 2021; Schaul, Zhang & LeCun ICML 2013;
Schmidt, Schneider & Hennig ICML 2021 (abstract quoted); Defazio, Yang, Khaled, Mishchenko,
Mehta & Cutkosky NeurIPS 2024 (Schedule-Free; AlgoPerf 2024 self-tuning win);
Smith, Kindermans, Ying & Le ICLR 2018; D-Adaptation `d_k` monotonicity and DoG's running
maximum (from the papers' algorithm statements, via search summaries rather than a full read).

Cited from prior knowledge and **not** re-verified this session: Kesten 1958; Pflug 1983 and
1990; Jacobs 1988; Riedmiller & Braun 1993; Sutton 1992; Plagianakos et al. 2001; Chandra et
al. NeurIPS 2022; Donini et al. IJCAI 2020; Balles & Hennig ICML 2018; Simsekli et al. ICML
2019; Zhang et al. ICLR 2020; Cohen et al. ICLR 2021; Gur-Ari et al. 2018; Shazeer & Stern
ICML 2018; Smith 2017/2018; Loshchilov & Hutter ICLR 2017; Liu et al. ICLR 2020 (RAdam);
You, Gitman & Ginsburg 2017 (LARS); You et al. ICLR 2020 (LAMB); McCandlish et al. 2018; Yaida ICLR
2019; Paquette & Scheinberg 2020; Martinez Rubio 2017 (thesis — existence asserted by Baydin
et al., not inspected).
