# LR control: what the bench has established

*State doc for one open question, per [`PROTOCOL.md`](PROTOCOL.md): rewritten in
place, no supersession chain, git holds the history. Measurements live in
`findings.md` and are cited by ID, never restated. All results are from `bench/`,
on synthetic surfaces driving the REAL `LRController` and `RayCalibration`,
except the real-gradient numbers in §3.*

---

## 0. The answer

**Use Baydin et al.'s published rule, unmodified.**

```
lr *= exp(0.02 * cos(g_t, g_{t-1}))
```

One dot product per step. No `L*`, no probe batch, no extra forward pass.

13 held-out cells × 5 scenarios × 20 seeds = 1300 runs per arm:

| arm | % over a 2× budget | **passable only** | worst cell |
|---|---|---|---|
| **`hyper` symmetric (published)** | 5.3% | **0.7%** | 29% @ `h eq base` |
| `hyper` 2:1 asymmetric | 5.3% | **0.7%** | 29% |
| `hyper gated` | 5.3% | **0.7%** | 29% |
| `ray+ray` (shipping) | 6.5% | 1.9% | 27% |
| `ramp+plateau` | 8.5% | 4.1% | 32% @ `h eq w_rep.3` |
| **NULL (no sensor)** | 21.0% | 17.2% | 29% |

Nothing beats it, and nothing invented here beats it either. The three
hypergradient variants — symmetric, 2:1 asymmetric, and the gated-bias design —
are **indistinguishable**; every earlier ordering among them was an artifact of a
scoring defect. The recommendation is the published rule because it is the
simplest member of a tied set, not because it won.

**Read the second column, not the first.** The first includes cold starts that
the shipping controller's own `peak_scale` cap and maximum climb rate make
unreachable for *any* arm, on 3 of 13 cells — F-032. Those columns add the same
~4.6 points to every arm and bury the differences that matter.

**The same order holds on a second, unrelated surface family.** `EQ_HARD`, 8
three-player equilibration cells (2 refused — their best fixed rate sat at the
edge of the searched range), 20 seeds:

| arm | % over | passable only | vs. standing still |
|---|---|---|---|
| **`hyper` symmetric** | 23.7% | **8.4%** | −4.0 |
| `hyper` 2:1 / gated | 24.2% | 9.0% | −3.4 |
| `ray+ray` | 24.7% | 9.6% | −2.8 |
| **NULL (no sensor)** | 27.0% | 12.4% | — |
| `ramp+plateau` | 29.3% | **15.2%** | **+2.8 — worse than nothing** |

Two things this adds. **The blind ramp does active damage on a multi-player
surface** — on `eq kappa.002` (a nearly frozen buffer, the one genuinely
discriminating cell here) it fails 45% of drifts, 45% of regime changes and 85%
of mixture drifts against hyper's 5% overall. And **the sensors are worth much
less here**: 4 points over the control, against 16.5 on the held-out battery.
Most of these cells are saturated, not discriminating.

### The one place the shipping probe genuinely wins

`h eq base`, `hot_90pct`: `ray+ray` fails 35% where all three hyper variants fail
**45% — identical to the NULL arm to the seed**. Hypergradient is not merely
worse there, it is *inert*: it does exactly what standing still does.

The mechanism is the same rate limit as the cold-start floor, pointed downward.
Cos saturates at −1.0 above the stability boundary (§3), so the fastest possible
cooling is `exp(−0.02)` per step — 2%. From 90% of the way to the cliff that is
~47 steps of pure cooling against a 100-step budget, and hyper spends most of the
budget getting there. The ray probe applies `(alpha/target)^0.5` in one move, so a
single resolved reading crosses the same distance.

**This is the architectural trade, stated honestly:** hypergradient is a bounded,
proportional controller that cannot make large corrections quickly, and that is
exactly the property that makes it safe under noise (§4). Where the budget is
short relative to the correction needed, a sensor that can jump wins. Nothing in
the bench tests the obvious combination — a bounded climber with an abstaining
probe allowed to make one large correction — and that is the live design
question, not another hypergradient variant.

## 1. What the bench can and cannot say

**It can say:** which arm converges within a time budget across noise, curvature,
conditioning, width and a moving optimum, on 13 held-out cells × 5 scenarios × 20
seeds, driving the real controller objects.

**It cannot say:**

- **That the margin over the incumbents is significant.** Seeds within a cell
  share the surface, oracle and target, so the effective n is closer to 13 than
  to 1300. A paired sign test over cells gives p ≈ 0.12.
- **Anything further selected on the held-out set.** It has now been scored more
  than once and is burned for selection.
- **That the noise axis represents the real system.** It does not — F-033. No
  cell reproduces the real fused branch's cos distribution, and the mismatch is
  in the direction that *understates* hypergradient rather than flattering it.
- **Anything about `var_cond`.** Its oracle converges 8.7× at 1500 steps, 17.4×
  at 4000 and 40× at 8000, failing the 100× cell guard at every length tried. It
  is not in `CELLS`, and a longer run does not rescue it.

**Two structural limits of hypergradient, both derived and both permanent:**

- Cold-start recovery has a floor of `ln(lr*/lr_seed)/beta` steps — 177 for a 35×
  gap at β=0.02, 408 for a 3464× gap.
- `peak_scale` is capped at 2000× the seed rate (`mk_dev.yaml:48`), which no
  amount of time fixes. Raising β moves the first floor and not the second.

## 2. Why it works — and it is not the reason first given

Hypergradient descent is gradient descent on the learning rate:
`dL/d(alpha) = -g_t · g_{t-1}`. Gradients agreeing means the step was too small;
disagreeing means it overshot.

The property that actually matters is that **its statistic is bounded**. Under
batch noise `E[cos] = ||gbar||² / (||gbar||² + tr Σ)`, so noise *attenuates* the
signal toward zero and `exp(β·0) = 1` is "do nothing". Its noise limit is the
identity: it degrades toward inaction rather than toward a confident wrong
answer.

It does carry a bias. Expanding
`g_t ≈ gbar + ε_t − α·H·P·(gbar + ε_{t-1})`, the `ε_{t-1}` in the step correlates
with the `ε_{t-1}` in `g_{t-1}`:

```
E[g_t · g_{t-1}]  =  ||gbar||²  −  α·gbar'HP·gbar  −  α·tr(H P Σ)
```

That last term is noise-driven and negative, so hypergradient cools under noise.
**But the noise-optimal step size genuinely IS lower**, so the bias points at the
right answer. Contrast `bb` in §5, whose algebraically identical bias is pure
corruption because it is estimating a curvature — a quantity noise has no
business entering.

## 3. What `cos` actually measures

**The learning rate, not the noise** — F-033. A 200× noise sweep moves median cos
about as far as a 16× rate sweep, and the zero crossing sits at ~1× the optimal
rate at every noise level tested. At 2× optimal, cos is −1.0000 in all four
quartiles of the run: exact period-2 oscillation, so the statistic saturates hard
on the hot side and is an unambiguous too-hot detector.

This retires the standing "calibrate the noise axis" action. The axis it wanted
to calibrate is not the axis `cos` responds to, and one cos value cannot identify
a noise level at all.

**On the real system** (T=10, elj nehzor sg14, n=1 per regime): fused cos is
**0.29 with an IQR of 0.24–0.35**, which is 903× the `sqrt(2/πd)` null at
6.16M params — a nearly noiseless reading, and cleaner than any bench cell. Read
as a rate statistic, it says the production rate sits somewhat **below** its
optimum.

Two facts that change what a controller can be built on:

- **Phase 1 has no `fused` steps.** `train_prior` is `bwd`-only. Its cos has a
  fused-like median (0.344) with a **13× wider IQR** (1.42 vs 0.11), so a
  controller crossing the transition changes statistic, not just noise level.
- **`bwd` is unusable outside a fresh model** — 0.015 mid-run and −0.045 at
  convergence, while `fused` in the SAME window reads 0.289. The branch, not the
  noise, is what separates them. Drive any controller off `fused`.

## 4. THE recurring failure mode

Four separate arms died the same death:

| arm | the noisy statistic | the threshold |
|---|---|---|
| `armijo` on a floored loss | `L(θ+d) − L(θ)` in float32 | sufficient-decrease bar |
| `armijo` at high noise | same, gradient noise | same |
| `hyperx`'s rho branch | `‖g_t‖/‖g_{t-1}‖` | rho > 1 |
| `slope_seek` | windowed progress rate | dead zone |

**A threshold test on a noisy statistic, combined with an asymmetric
multiplicative response, does not wander — it collapses.** With backtrack ×0.5
against a climb of ×1.014, the measured 61.2% accept rate gives
`E[log step] = 0.612·ln(1.014) + 0.388·ln(0.5) = −0.26` per step and the rate
slams into its floor. The worst case is putting the threshold at the statistic's
own noise centre, which is what `rho > 1` does: `hyperx` went from 0% over budget
to **100% at noise 2**, having been the best arm on the board at `quartic=0.01`.

**The mechanism it needs is abstention.** The ray probe survives noise precisely
because its significance test returns `unresolved` and the servo falls back to a
constant. "72–82% of readings come back saturated" reads as the probe not paying
for itself; that reading is backwards — **the fallback IS the noise robustness**.

**Every amplification of `cos` saturates into a switch.** Six variants died this
way: `hyperz` (divide by the theoretical null — ×51 at d=2048), `hypern` (divide
by a running mean of |cos|), `hyperm` (momentum on raw β·cos — compounds same-sign
noise runs), `hyperp` (smooth THEN amplify — k=8 scores identically to k=4,
proving the clip binds always), the rho branch, and the ‖g‖ gate. An additive
bias was the only thing that ever bought speed without amplifying, because
addition offsets the response rather than scaling it — and it is not needed.

**Raising the gain is not the answer either.** β 0.02 → 0.04 → 0.08 scored
3.8% → 7.8% → 11.8% — *those three percentages were measured under the censored
metric of §7 and are not comparable to any number in §0*; what survives is the
monotone ordering and the reason for it. Scaling β leaves the signal-to-noise of
each decision unchanged while scaling the random-walk excursion ~β·σ·√T, and the
surface has an absorbing boundary above and none below. β does buy one real
thing — F-032's cold-start floor is `ln(R)/β` steps — so it trades tail risk
against recovery time, and the cap at 2000× the seed rate is untouched either
way.

## 5. The through-line: common random numbers

Every finite-difference sensor that survived pays for paired samples. Every one
that failed does not.

- **ray probe** — `n_sub` paired sub-batches + significance test. Survives.
- **`bb`** — `y = g_t − g_{t-1}` across *different* batches.
  `E[s·y] = s'Hs + lr·tr(PΣ)`, so it reads **4× too cold at noise 2**. Median
  `alpha_bb/alpha_true` 0.998 → 0.680 → 0.381 → 0.242 as noise goes 0.01 → 2.
- **`armijo`** — losses on different batches. Collapses (§4).

**Averaging does not fix `bb`.** A 200× window buys a 1.57× spread reduction
where a well-behaved statistic gives √200 ≈ 14×, because the estimator is a
*ratio* whose denominator crosses zero — and `frac < 0.5×` RISES with the window:
averaging converges onto the wrong number. The fix is pairing, which costs `bb`
the free status that made it interesting.

## 6. Method inventory

| method | cost | verdict |
|---|---|---|
| **`hyper`** | free | Best guarantee measured; the SYMMETRIC published form is the one to use. |
| `ray+ray` (shipping) | ~2% | Sound architecture (abstention + fallback), never tuned as one object. |
| `ramp+plateau` | free | Noise-immune (no sensor to corrupt) but cannot track an optimum that moves UP — and on a MULTI-PLAYER surface it is **worse than having no sensor at all** (15.2% vs 12.4%). |
| `bb` | free | Structurally biased, §5. Best on the moving target where noise is low. |
| `armijo` | ~50%/step | Fastest median anywhere; worst thing on the board at high noise. §4. |
| `dog` | free | Cold-start bootstrap failure. **Wrong family member tested** — DoG is SGD-derived; Prodigy is the Adam variant. |
| `sps` | free | Fails on `mle` because `L*` is unknown there. **TB's `L*` is genuinely 0**, so this is untested where it applies. |
| `slope_seek` | free | Retracted — premise was an invented `a/b=4`. |
| `plateau` alone | free | Cannot climb. Bit-identical to no sensor on a cold start. |
| **NULL (no sensor)** | free | The control arm, permanent in `ARMS`. Fails 100% of cold starts and little else. |

## 7. Metric design, which cost four re-runs

`steps_to_target(run) / steps_to_target(oracle)`, target = the distance the
oracle passed at `DEEP_FRAC=0.25` of its run, denominator selected on TIME
(`_time_oracle`) rather than on final distance.

**The measurable range of a ratio is set by how much run remains after the
denominator finishes**, and this was got wrong at both ends:

- **Too loose.** `steps_to_target` returns at most `steps`, so the largest finite
  ratio is ~`1/frac`. At `frac=0.60` the ceiling is 1.63, *below* the 2.0 budget,
  and every "% over budget" printed was identically "% never converged". This
  defect was fixed once, reintroduced worse on the argument that 0.25 made the
  target too shallow, and went unchecked precisely because it had been fixed
  once. `_oracle_task` now asserts the ceiling per cell.
- **Too tight.** Selecting the denominator on time — correct in itself — collapsed
  the equilibration denominators to as little as 50 steps of a 3000-step run,
  making cold_start unreachable for any arm on 3 cells (F-032). `_oracle_task`
  now checks that end too, and `main` prints a `passable only` aggregate.

Also live: a **cell guard** — `find_oracle` accepts a cell if a best LR *exists*,
which is not the same as the run converging, so cells failing a 100× drop are
skipped. `EQ_HARD` exercised a second form of it: two cells were refused because
the best fixed rate sat at the *edge of the searched range*, which is the best of
a badly chosen set rather than an oracle.

**One saturated column is still unguarded.** On `eq n3` every arm fails 100% of
hot starts, including the no-sensor control — the same signature the cold-start
guard catches, for the same reason: the deep target is at 56 steps of a
3000-step run, so the budget is 112 steps and cooling from 90% of the way to the
cliff at 2%/step does not fit either. `_cold_start_feasible` should grow a hot
counterpart; the arithmetic is the same with the sign flipped.

**A stated fix with no guard is one edit from being undone.** Both ends are now
assertions, not prose.

## 8. Next

1. **A bounded climber plus a jumping brake.** The only scenario where the
   shipping probe beats hypergradient is the one needing a large correction in a
   short budget (§0), and the two mechanisms fail in opposite directions:
   hyper cannot move fast, the probe cannot move without a resolved reading.
   Pairing them has never been run, and it is a cheaper question than any
   remaining hyper variant. Note F-021's constraint — the brake must run on the
   climber's clock; split-clock arms were the worst in the factorial.
2. **The `bwd` statistic in phase 1.** F-033 shows it is a different statistic,
   not a noisier one (IQR 1.42 vs 0.11 at a similar median), and that `bwd` is
   near-useless outside a fresh model (0.015 mid-run, −0.045 converged). Whether
   hypergradient is safe to run through phase 1 is untested — the bench has no
   `bwd`-analogue surface.
3. **Combined hardness.** The axes select opposite winners — noise favours the
   blind arm, a moving target favours the measuring arm — so a one-at-a-time
   sweep cannot find an arm that fails on the *interaction*.
4. **Log, do not control.** Record `alpha_t`, `alpha_Polyak`, `h_t` and
   `alpha*_probe` on one real run. If `sign(h_t)` predicts
   `sign(alpha*_probe − alpha_t)`, the expensive probe has a free online
   surrogate. Testable on the bench first, where `alpha_star_true` is exact.
5. **Directional Polyak** `alpha_P = (F − F*)/(−g'd)`: denominator strictly
   positive for any PSD preconditioner, so no heavy tail, and its bias is bounded
   and absorbable by a slow calibration factor.
