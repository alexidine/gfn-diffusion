# Predictions committed BEFORE each run

Written before the run, never after. A wrong prediction is the cheapest bug
detector in this project -- more real problems have come from a prediction
missing than from reading the numbers.

---

## 2026-08-13 -- `python -m bench.crucible 20 8 heldout`

First heldout run that includes **`_time_oracle` wired into `_oracle_task`**
(crucible.py:425) and the **`mixture_drift` scenario** (5 scenarios, not 4). The
headline table in `docs/lr_control_summary.md` section 0 predates both.

**What should move, and where:**

1. **The two `eq` cells (`h eq base`, `h eq w_rep.3`) are where the change
   lands.** `_time_oracle` re-selects the denominator rate on TIME instead of on
   final distance; section 6 measured that gap at 10.9x on `eq base` (~190 steps
   vs ~1823). So `denom` on those cells should FALL by roughly an order of
   magnitude, and every arm's `%over` there should RISE off 0%. If the eq cells
   still read 0% for all six arms, the fix did not take effect and I should
   check the printed `deep target at N of 3000 steps` line, not the ranking.
2. **`NULL (no sensor)` should rise the most on the eq cells**, since a real
   denominator is the thing that makes standing still cost something.
3. **`mle` cells should be nearly unchanged.** On a convex bowl the
   distance-optimal and time-optimal rates are close, so `_time_oracle` should
   pick nearly the same lr. Large moves on `mle` cells would mean the two
   selections disagree on a quadratic, which I do not believe and would want to
   see explained.
4. **`mixture_drift` is a NO-OP on all 11 `mle` cells** (`_sc_mixture` returns a
   plain run when the game has no `w_rep`), so that column should be ~0% there
   for every arm including NULL. Non-zero on an `mle` cell = the scenario is
   doing something it should not.
5. **Headline ranking: I expect the three `hyper` variants to stay tied at or
   near 0.0%** and `ray+ray` / `ramp+plateau` to rise, because the newly-live eq
   cells add failures for arms that were previously getting a free pass. I do
   NOT expect hyper to stay at exactly 0.0% -- a real eq denominator is a new
   test it has never had to pass.
6. **A cell may now be SKIPPED for the opposite reason than before.** With a
   ~10x smaller `denom`, `BUDGET*denom >= steps` gets easier to satisfy, so the
   eq cells should stop being at risk from that guard. Any NEW skip is a
   finding.

**What would falsify the whole exercise:** all 13 cells reporting identical
numbers to section 0's table. Two runs of different code agreeing exactly is a
bug, not a replication.

---

## 2026-08-13 -- `python -m bench.calibrate_noise 400` (all four regimes)

First run after `None` was made to force a fresh model. The four earlier JSONs
are on disk, so three of these predictions are checkable against a control.

1. **`mle_fresh` starts at step ~0, not 10001**, prints `FRESH --
   checkpoint_name=None` and `VERIFIED loaded=nothing`. The old file covers
   steps 10001-10399.
2. **`mle_fresh` will be `bwd`, not `fused`.** Phase 1 is MLE. If that holds, a
   hypergradient controller has no fused signal to run on during phase 1 at all,
   which is a design constraint and not a measurement detail.
3. **`mle_fresh` median cos will NOT be -0.0448 and should be clearly positive**
   -- a fresh model is far from stationary, which is the entire premise of
   measuring during active descent. I expect somewhere in 0.05-0.4. A near-zero
   value would say the bwd branch carries no usable signal even when descending.
4. **`eq_phase1exit` and `eq_descent` REPRODUCE** their earlier medians (0.2871
   and 0.2901) to within seed noise -- those regimes already loaded the right
   checkpoints, so they are the control. A large move means something other than
   the reload changed.
5. **THE IDENTITY CHECK. `mle_converged` should come back at cos = -0.0448 on
   `bwd` over steps 10001-10399** -- i.e. it should reproduce the OLD
   `mle_fresh` file almost exactly, because the diagnosis says those two regimes
   were always the same run. This is the one prediction that can prove the
   diagnosis rather than merely be consistent with it. If `mle_converged` comes
   back DIFFERENT from the old `mle_fresh`, the reload was not the whole story
   and I should stop and find the rest of it.

Seed/dataloader order is not pinned, so "reproduce" means close, not identical
-- and an EXACTLY identical median across two separate processes would itself be
suspicious.

---

## 2026-08-13 -- crucible re-run with the cold-start feasibility guard

Nothing that affects a measurement changed: same seeds, same surfaces, same
arms. Only the REPORTING moved (a per-cell feasibility note and a second
aggregate column). So:

1. **The first column must come back BIT-IDENTICAL** -- hyper sym / 2:1 / gated
   5.3%, ray+ray 6.5%, ramp+plateau 8.5%, NULL 21.0%, and every per-cell row
   unchanged. This is the one place in this project where an identical number is
   the CORRECT outcome rather than a bug, because it is the same deterministic
   computation; a difference here means the bench is not reproducible across
   processes and that is a bigger finding than anything about learning rates.
2. **Exactly three cells print the UNREACHABLE note**: `h cond=30` (deadline),
   `h eq base` and `h eq w_rep.3` (peak_scale cap). No others.
3. **The `passable only` column, by hand from the per-cell tables:** hyper
   0.73%, ray+ray 1.94%, ramp+plateau 4.11%, NULL 17.2%. If the code disagrees
   with these, my arithmetic in the write-up is wrong and the write-up must
   follow the code.
4. The ORDER is unchanged either way. The point of the second column is not a
   new winner, it is that the gaps stop being buried under ~4.6 points of budget
   artifact that every arm pays equally.

### Outcome

**All four confirmed.** First column bit-identical (5.3 / 5.3 / 5.3 / 6.5 / 8.5 /
21.0); exactly three UNREACHABLE notes on exactly the three predicted cells, with
the two wall types correctly separated; `passable only` came back 0.7 / 1.9 / 4.1
/ 17.2 against a hand computation of 0.73 / 1.94 / 4.11 / 17.2.

**The calibration predictions scored 3 confirmed, 1 partial, 1 pending:**
(1) fresh really starts at step 1, (2) it is `bwd`-only in `train_prior`, (3)
median 0.3441, inside the predicted 0.05-0.4 and nothing like the −0.0448 the
broken regime reported. (4) `eq_phase1exit` came back 0.3037 against this
morning's 0.2871 over the IDENTICAL step range — close, but not the clean control
I claimed it would be, because the checkpoint itself had been rewritten in
between by the clobbering bug. A cleaner reproduction is not available; that file
is gone.

**Unpredicted, and the most useful thing the run produced:** `[calib] BLOCKED 14
checkpoint writes` in a single 400-step regime — 8 × `running`, plus
`phase1_exit`, `prior`, `stage_start` and three buffer saves. I had reasoned the
diagnostic was writing checkpoints; I had not guessed the rate.

---

## 2026-08-13 -- `python -m bench.beta_ladder 10`

Can hypergradient run faster than beta 0.02? First run scored on the off-target
metric as well as on time-to-target.

1. **The two metrics should DISAGREE about beta, and that is the point.**
   `%over` measures how fast you arrive; `off-target` measures whether you sat
   at a bad rate. Low beta's failure is being stuck COLD, which `%over` only
   sees when it costs a deadline. So I predict **`%over` is minimised at or near
   0.02 while `off-target` is minimised higher, around 0.05-0.08.** If both are
   minimised at the same beta, the new metric is not adding anything and I
   should say so rather than keep it.
2. **`too cold` falls monotonically with beta; `too hot` rises monotonically.**
   These are the two mechanisms, and if either is non-monotone I have the model
   wrong.
3. **Divergences per run rise sharply somewhere in 0.15-0.3.** That is the
   absorbing boundary, and it is why the ladder goes past the estimate rather
   than stopping at it.
4. **`eq base` and `eq n1` break at a lower beta than the `mle` cells** -- a
   second player amplifies an over-fast climber. If the equilibration cells
   tolerate MORE beta than the bowls, my reasoning about why this is risky is
   wrong.
5. The estimate under test is beta ~ 0.08 (4x, from the 4x tighter real
   signal). Since the bench statistic is ~4x noisier than reality, **whatever
   beta survives here is a lower bound** on what is safe in production.

**What would make me drop the whole idea:** beta 0.05 already showing elevated
divergences on the mle cells. That would mean the bench's absorbing boundary
binds well before the noise argument does, and the gain is not the free lunch it
looks like.
