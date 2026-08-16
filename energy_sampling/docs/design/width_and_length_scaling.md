# Scaling with model width W and rollout length T

Why some config values move with `(W, T)` and most do not. Argument: revised when
the reasoning changes, not when a number does.

`W` is the model hidden width — the `model.*_hidden_dim` fields, which are set
equal to each other. `T` is `integrator.T`, and `eval_T` must equal it (enforced
at load: the policy learns a drift per step at one dt, so scoring at another
integrates a different SDE).

## The one distinction everything follows from

The TB residual

    delta = log Z + sum_t (log P_F - log P_B) - log R

is **intensive**. `T` discretizes a fixed `[0, 1]` horizon, so `delta` converges to
its continuous (Girsanov) limit as `T` grows rather than accumulating with it. The
heads are bounded (tanh log-variance, `gfn_clip`), so `delta` is width-independent
too. **Anything measured in loss space therefore does not scale.**

The raw gradient

    d(delta)/d(theta) = sum_t d(...)/d(theta)

flows through **shared** weights — the same `theta` is re-used at every one of the
`T` steps — so it is **extensive**: it grows with `T` (the sum over reuses) and
with `W` (~`sqrt(W)`, more parameters). **Anything measured in gradient space
does scale.**

That is the whole argument. Loss-space bars (Huber `beta`, fracs,
`deactivate_threshold`, the divergence loss bar) are `(W, T)`-flat. Gradient-space
bars are not.

## What is actually derived, and by what

One value: `gradient_norm_clip`. Written `auto`, it resolves at load
(`utils.resolve_derived_config`) as

    250 * grad_median(T) / grad_median(25) * sqrt(W / 512)

with `grad_median` log-log interpolated over measured pre-clip medians
(`utils._GRAD_MEDIAN`). It scales with the gradient's own measured magnitude,
which is a property of the rollout length rather than of a tuning choice, and
nothing servos it afterwards.

`utils.py` is the executable source of truth for this. This document explains
*why* the formula has the shape it has; it does not restate the numbers, which
would only give them somewhere to rot.

## What is NOT derived, and used to be

**The learning rates.** There was a rule — anchor × 25/T — and it is gone. It
promoted one battery's measurement (one energy, one T, one W, one clip) to a law,
and the problem shifts constantly enough that there is no stable "here" for an
anchor to be anchored to. Every run is a transfer.

`lr_*: auto` now means **servo-managed**: seeded at `adaptive_lr.seed_lr` and
owned from there by whichever adaptive sensor the stage declares. So the answer to
"what LR should I use at this (W, T)" is no longer a formula — it is *measure it
during the run*, which is what the sensor does. An explicit float is a fixed peak
that takes the warmup envelope and divergence handling but no servo.

**The tripwire bars.** `cut_grad_abs`, `reset_grad_abs`, `cut_loss_abs`,
`reset_loss_abs` and the ratios that tied them to the clip (30x, 10x) are retired
along with the graduated cut tier. What survives is a pair of absolute divergence
bars near 1e9 whose only job is catching numerical explosion, and which are
refused at construction if set low enough to fire on ordinary training.

**Convergence timings.** `hold_steps` and `decay_halflife_steps` scaled inversely
with the LR cut to hold total weight-travel constant. Both are retired with the
decay leg; the envelope holds at 1.0 and there is no step budget to schedule
against.

## Rescaling in practice

Edit the primitives only:

- **W** — the `model.*_hidden_dim` fields, set equal. `t_dim`, `harmonics_dim`,
  `condition_embedding_dim` and the `*_layers` counts stay fixed.
- **T** — `integrator.T` and `eval_T` together.

`gradient_norm_clip` recomputes at load and prints what it resolved to. The LRs
re-seed and the sensor re-measures. Nothing else on this axis needs touching, and
a key not named above is `(W, T)`-invariant.

Beyond about 2x width the `sqrt(W)` term on any LR is a standard-parametrization
artifact rather than a real ceiling: per-layer LR groups (muP — a 1/W group on the
hidden matrices) make LRs width-flat, and only the clip keeps its `sqrt(W)`
because it reads the raw pre-Adam gradient. A global LR that vanishes with width
means std-param is being fought with one scalar.
