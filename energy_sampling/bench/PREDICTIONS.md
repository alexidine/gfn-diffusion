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
