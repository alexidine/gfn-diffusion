# Module: training modes (`train.py` step dispatch)

> **Status: MODULE SNAPSHOT.** The verification dates below are historical.
> Use this document for explanation and navigation; verify material claims
> against current code, canonical config, and focused tests. See
> [`EPISTEMIC_PROTOCOL.md`](EPISTEMIC_PROTOCOL.md).

Pass 1 (audit + rationalize). Verified against the working tree, 2026-08-03.
User pass 2026-08-06 — rulings recorded inline. **Revised 2026-08-08**: the replay
draw is now regime-dependent (§4), plus M7 (the mis-wired prioritised draw) and M8
(the step probe's batch). Unconditional route.

> **This doc is scheduled for rewrite.** The replay redesign
> ([`to_do_rebuild.md`](to_do_rebuild.md) Part B) changes what `replay` and `bwd`
> *are*, so §4, §6 and §7 will not survive it intact. Rulings below are recorded
> so they are not re-litigated, not because the surrounding text is settled.
>
> **The framing question, user 2026-08-06:** replay has to **re-earn its place**
> against a plain fwd/bwd balance. The expectation is optimistic — if the
> redesign works it should be *strictly* better — and in that case `fwd` reduces
> to log Z calibration only. Possibly with policy grads, but if replay is
> effective, probably not. That is what `P8` arm (i) tests.

---

## 1. What it is

Four `step_type` values — `fwd`, `bwd`, `replay`, `fused` — dispatched from
[`train_step`](train.py:1974). The first three are **samplers**, not objectives:
each draws a batch from a different distribution and hands it to one of the two
loss assemblers. `fused` is not a fourth sampler; it runs the other three in one
step and combines their losses.

| Mode | Sampler | Loss entry point | Trajectory |
|---|---|---|---|
| `fwd` | on-policy rollout from the prior latent | `get_gfn_forward_loss` | generated forward |
| `bwd` | draw from `prior_buffer` (or `prior_dataset` in phase 1) | `get_gfn_backward_loss` | **resampled backward** |
| `replay` | draw from `replay_buffer` | `get_gfn_backward_loss` | **stored, replayed exactly** |

Paper voice: *the same trajectory-balance objective is evaluated on three
trajectory distributions — on-policy forward rollouts, backward rollouts from a
diverse buffer, and exact replays of previously generated forward trajectories —
combined as a weighted sum in a single optimizer step.*

## 2. The single most important fact

**In `fused` mode, `fwd_frac` / `bwd_frac` / `replay_frac` are LOSS WEIGHTS, not
throughput shares.** [Lines 2161–2163](train.py:2161):

```python
total_weight = sum(weights.values())
fused_loss = sum((weights[k] / total_weight) * sub_losses[k][0] for k in ...)
```

Every active branch runs **every step**, regardless of its frac. Consequences:

- Step cost is roughly the sum of all three branches and is **independent of the
  fracs**. Moving a frac changes the gradient mixture, not the compute budget.
- The only frac-driven compute saving is `deactivate_threshold`, which skips a
  branch entirely below the bar — deliberately unreachable, see M5.
- Any reasoning of the form "raise bwd_frac to spend more time on bwd" is wrong.
  The right reading is "raise bwd's weight in the gradient."

## 3. Contract

- Exactly one optimizer per `step_type`. On a fused step, the flow (Z-head) group
  lives *inside* `optimizers['fused']`, so the standalone `flow` optimizer is not
  zeroed — the right owner zeroes it either way ([lines 1991–1997](train.py:1991)).
- `replay` must run `repeats: 1`. K copies of a *stored* trajectory are
  identical, so tiling buys nothing and costs K×. (Arithmetic, not a tuning
  choice — distinct from the `bwd repeats` question in §6.)
- The three fracs sum to 1, so with `deactivate_threshold < 1/3` at least one
  branch always survives (asserted at [line 2159](train.py:2159)).
- `*_step_count` means "times **trained** on" — it paces buffer churn via
  `bwd_step_delta` / `replay_step_delta`, so a force-refresh-only run must not
  advance it.

## 4. What each mode is *for* (the phase-2 design)

The `naive` stage assigns one trainer to each parameter group and one sampler to
each trainer:

- **fwd → Z** (`freeze_policy: 1`), on-policy. Pinned at 0.2 — not for Z's sake
  (Adam cancels a single source's scale) but because replay admission draws its
  candidates from the fwd batch, and a thin candidate pool degrades the replay
  buffer from prioritized to FIFO. **The pin is unwarranted in general** (user,
  §6).
- **bwd → policy only** (`freeze_z: 1`), from the churned prior buffer. Spread
  and mode retention; the long-horizon restoring force.
- **replay → policy only**, from the replay buffer. The over-weighting corrector.

Two corrections to the earlier draft of this section:

**`fwd` is not Z's sole source.** *(user, 2026-08-06)* Off-cycle **z_calibration**
steps also produce Z gradients — a second, independently-paced source. The
"exactly one trainer per parameter group" bijection is therefore a statement
about the *fused step*, not about the run. Anything reasoning from "Z sees only
fwd" needs to account for the calibration cadence, including
[[zcal-replay-intake-contamination]].

**Replay's prioritization is entirely at admission; the draw is uniform.**
*(confirmed 2026-08-06; **now regime-dependent — see below**)*
`draw_replay_sample` passes `weighted=False`
([train.py:3212](train.py:3212)), so `_loss_weights` is never consulted on this
path. Calling replay "a `|resid|`-prioritized resample" is right about the
*buffer* and wrong about the *batch*. `_loss_weights` is live only on the `bwd`
path, gated by `weighted_bwd_sampling`, at `temperature: 0.5` and `beta: 0.9` —
i.e. **10% of the bwd batch loss-weighted, 90% uniform**.

That sharpens Part B's diagnosis rather than changing it: admission is not merely
doing *some* of the draw's job, it is doing **all** of it.

> ⚠ **Revised 2026-08-08 — this now holds only on the DEFAULT path, and the two
> regimes are exact opposites.** With `buffers.replay_buffer.prioritise.enabled`,
> `draw_replay_sample` computes `p ∝ δ₊^κ` and draws entirely from it
> (`draw_beta = 0.0`), while admission goes **uniform**. So prioritisation moves
> from the admission side to the draw side wholesale; it is never in both places
> and never in neither. `weighted=False` stays true in both regimes — it disables
> `_loss_weights`, which is a *different* prioritisation mechanism from `p` and is
> not the one that got wired up. See `module_buffers.md` B0 (intake) and B7 (draw).
>
> Note which is the default: **`mk_dev` and 22 of 26 rb0808 arms run the original
> admission-side scheme.** The paragraph above is still the majority description
> of the codebase.

The rationale for taking the policy off the on-policy branch remains the
strongest single argument in the codebase: **on-policy TB is mode-seeking**,
because it only ever sees the policy's own support — now with a rate attached
(`to_do_rebuild.md` §B1c: `Q(δ < −m) ≤ e^{−m}`, exponential blindness).

## 5. Findings

**M1 — "modes" are samplers; the naming hides that.** *(structural)*

Two loss entry points, three samplers. `bwd` and `replay` call the *same*
function, differing only in whether a stored trajectory is supplied
([line 398](gflownet_losses.py:398)). Documenting the axis as "sampling
distribution" rather than "training mode" would remove most of the conceptual
overhead.

**M2 — the replay share silently folds into bwd when the buffer is empty.**
*(confirmed — **user: "that can't be good"**, so this is a defect, not a fallback)*

[Line 2157](train.py:2157): if `replay_available` is false,
`weights['bwd'] += self.replay_frac`. At stage entry — before the replay buffer
has bootstrapped — `bwd` runs at `0.6 + 0.2 = 0.8` rather than the configured
0.6, so **the controller's stated entry operating point is not the one it starts
from** and the first balance ticks read metrics produced under a different
mixture than the config describes.

**Ruled a defect.** Two fixes, not exclusive: (i) log when the fold-in engages,
so an arm's actual entry mixture is readable; (ii) **pre-fill the replay buffer**
so the condition never arises — see §9 Q2 and `to_do_rebuild.md` §C.

**M3 — `scramble_conditions: true` is inert on the unconditional route.**
*(confirmed; **drop it**, §8 S4)*

[`scramble_applicable`](train.py:2981) requires the model to be conditional *and*
on vector conditions. With every `*_conditioning` flag false, `gfn.conditional`
is False, so the guard returns False and the scramble never fires — while the
`train_prior` stage declares the flag with the comment "unconditional prior by
construction."

**M4 — replay-buffer intake continues when fwd is deactivated as a trainer.**
*(confirmed; **accepted in principle, with a named caveat**)*

`manage_replay_buffer` fires on `fwd_ran` ([line 2165](train.py:2165)), which
includes force-refresh-only runs whose loss was detached, and also fires from
eval and z_calibration forward passes. So intake runs on an uneven, externally
paced cadence.

**User ruling 2026-08-06: acceptable *if the buffer represents a distribution*.**
On that reading rows are exchangeable and an uneven fill rate is harmless. **If
instead the particular states matter, off-cycle churn will degrade it** — an
eval or z_calibration pass injects a burst of intake that a distribution-valued
buffer absorbs and a state-valued buffer does not.

This is the same axis as the prior buffer's exchangeability argument, and it is
load-bearing for the redesign: `to_do_rebuild.md` §B7's uniform-admission
proposal is exactly the "buffer is a distribution" position, and it makes this
ruling self-consistent. A selective-admission buffer is state-valued and would
have to care.

**M5 — force-refresh cost is bounded, real, and deliberately paid.**
*(**user ruling: eat the cost**)*

Every `refresh_every: 10` fused steps, each non-dormant branch below
`deactivate_threshold` runs a full rollout purely to refresh rolling stats, then
has its loss detached.

The earlier framing — "the compute-saving path is unreachable as configured,
that may be deliberate" — is answered: **it is deliberate.** Fresh grads of this
type are wanted, real weights do sometimes sit below 0.1, and the most expensive
branch (`fwd`) is going to be on essentially always regardless. The floor /
threshold relationship is intentional (§8 S3), not an inconsistency.

**M6 — gradient accumulation engages only below a sample target.**
`fused_grad_accum_min_samples` triggers accumulation when `batch_size` is *under*
target, so a large batch is never loss-scaled by `batch/target > 1`. Clean, and
the `fused_accum_count` reset on batch-size jumps ([line 1986](train.py:1986))
correctly drops a partial cycle. No issues found.

**M7 — the prioritised draw shipped mis-wired, and the failure was
sign-inverting rather than merely ineffective. ✅ FIXED 2026-08-07.**
*(the single most consequential defect of the local shakedown)*

`_sample_indices`' `beta` is **the fraction of the batch drawn uniformly**, not a
temperature. `draw_replay_sample` inherited `beta=1.0` from the legacy uniform
call, so a supplied `p` was silently ignored — and because the IS weights
`w ∝ 1/δ₊^κ` were *still applied to the loss*, a uniform draw carrying `1/p`
weights targets a measure `∝ 1/δ^κ`: **the inverse of the design**, up-weighting
the lowest-residual rows. Details and the κ=0 identity that exposed it are in
[`findings.md`](findings.md) `F-004`.

**Why it belongs in this module and not only in buffers:** the draw and the
weighting are wired in *two different places* — `draw_replay_sample` picks the
rows, `replay_train_step` passes `sample_weights` to the loss — and each was
individually defensible. A prioritisation scheme spread across a sampler flag, a
loader kwarg, and a loss kwarg has no single place where "is this consistent?" can
be read. That is the structural lesson; the fix was one line.

**M8 — the step probe draws from `replay`, and that choice is load-bearing.**
*(added 2026-08-07; sensor only)*

[`_draw_probe_batch`](train.py:2482) supplies the frozen batch the LR step probe
re-scores at α ∈ {0, ½, 1} (`module_lr_controller.md` F0). Four properties, each
deliberate:

- **Source is `replay`, i.e. on-policy rollouts with stored energies.** Replay rows
  are fed by the fwd branch, so a replay draw *is* the on-policy distribution — and
  it carries the highest loss variance in the system, which is where a step-size
  sensor should be read for stability. Falls back to the `bwd` draw only when
  replay is unavailable (phase 1, or an empty buffer).
- **`repeats` must come from the branch whose coeffs score it.** A coeff bank is
  only valid at its own branch's K: `bwd_loss_coeffs` carries `tbc`, which asserts
  `K > 1`, so scoring a bwd draw at replay's K crashes in phase 1 — exactly where
  the fallback is the only path.
- **Fresh every probe.** The invariant that matters is *identical data across the
  three α within one probe*, not across probes. Re-drawing averages the particular
  draw out for free, since a buffer draw costs no energy calls.
- **It must not count as a training visit.** `_probe_loss` runs under `no_grad`
  with `update_log_z=False`, and deliberately does **not** call
  `buffer.update_losses` — a probe draw must not affect churn, priority, or
  residence. This is the one place in the codebase that reads the buffer without
  touching its bookkeeping.

## 6. Warrants

*Revised 2026-08-06. Several rows were carrying more warrant than they had
earned; the general note from the user is that **pin values, slopes and
thresholds in this module are tentative/TBD and have not been re-earned
recently.***

| Choice | Warrant | Note |
|---|---|---|
| fracs as loss weights, not throughput | **derived** | one optimizer step needs one loss |
| `replay repeats: 1` | **derived** | arithmetic — K copies of a stored trajectory are identical |
| Policy off the on-policy branch | **derived + measured** | on-policy TB is mode-seeking; rate in `to_do_rebuild.md` §B1c |
| Dormant modes skip force-refresh | **measured** | a rollout every 10 steps for unread stats was the dominant waste |
| `fwd` pinned at 0.2 | **~~derived~~ → arbitrary** | **unwarranted in general** (user). The candidate-pool argument justifies *some* fwd, not 0.2. Pins have been varied before and will be again (§9 Q3) |
| `bwd repeats: 2` in phase 1 | **~~measured~~ → open empirical question** | (user). Not re-established recently; belongs in a ladder, not a warrant table |
| `tbc` on in phase 1 | **~~measured~~ → open empirical question** | (user). Same status as above |
| `_bwd_retention_priority` centred on tracker mean | **derived** | see the note below — the derivation holds, but the whole quantity is scheduled for replacement |
| `deactivate_threshold: 0.01` | **binary intent, not a tuned number** | (user). It encodes *"this branch may be deactivated"* vs *"this branch may never be"*. **`fwd` must never deactivate** — keep a live Z gradient always, despite the energy cost |
| `floor: 0.03` | **empirical** | (user). Set by observation, unlike the threshold above |
| Entry fracs 0.2 / 0.6 / 0.2 | **arbitrary** | (user). To be redesigned with the replay work |

### What `_bwd_retention_priority` is

*(added 2026-08-06 — user: "I don't even know what this is")*

It is **the prior buffer's `ema_loss` feed** — the bwd-side analogue of the
replay buffer's admission score. [train.py:2925](train.py:2925) computes a
per-sample priority and [line 2917](train.py:2917) writes it into
`prior_buffer.update_losses` (or `prior_dataset`, in phase 1). That one number
then drives **two** things:

1. **Retention** — `purge_lowest` keeps high-priority rows, so it decides what
   stays in the prior buffer.
2. **Draw** — when `weighted_bwd_sampling` is on, `_loss_weights` reads it for
   the 10% weighted portion of the bwd batch.

The value is `|resid|`, **centred on the tracker's per-condition `ema_logw`** —
the buffer's own normalizer. The centring exists because whenever learned Z lags
the buffer-implied level, `|log_Z − log_w|` is dominated by that collective
offset, so ranking on it degenerates to one-sided ranking by `log_w` and skews
retention toward off-policy/blowup tails instead of the samples defining each
condition's `Var(log w)` spread. It coincides with the raw ranking exactly when Z
has caught up.

**This is precisely the quantity `to_do_rebuild.md` §B5a replaces** — with a
signed, terminal-averaged `δ₋` priority. Worth knowing before the rewrite, since
changing it moves both retention and the draw at once.

## 7. Failure signatures

*Most of these are expected to be **superseded** by the replay rework (user,
2026-08-06) — they describe pathologies of the current admission/draw split.*

| Symptom | First metric | Cause |
|---|---|---|
| Step time unchanged after a big frac move | step time | expected — fracs are weights, not shares (§2) |
| bwd dominating early in phase 2 | `bwd_frac` vs config | replay buffer not yet populated; share folded in (M2) — pre-fill removes this |
| Replay degrades to FIFO | replay admit rate | fwd frac too small → thin candidate pool. Becomes moot under uniform admission |
| A branch's rolling stats frozen | `{mode}_step_count` flat | branch deactivated *and* dormant — no force refresh |
| Controller reads stale metrics at stage entry | any `{mode}/*` at transition | force-refresh hasn't run yet; stats carry over from the previous stage |

## 8. Simplification candidates — ruled

**S1 — rename `*_frac` to `*_weight`. ⚠️ DOWNGRADED, not rejected.** *(user
2026-08-06: "it's a frac of a weight so I actually like it. You're the only one
who keeps getting confused.")* The name is defensible — the value genuinely is a
fraction of the total weight. Recorded so this stops being proposed. **Carried
into `decisions.md` — `R2` is closed as declined, and slot 3 of the priority
order is now `M2` / replay pre-fill.**

*(Numbering note: this section's `S1`/`S2` are module-local and collide with the
register's `S1` = delete the LR middle layer and `S2` = drive-liveness reporting.
In `decisions.md` these two are `R2` and `D18` respectively.)*

**S2 — collapse `bwd` and `replay` into one sampler. ❌ REJECTED** unless code
simplification is valued for its own sake. *(user: "They do different things
(rollout vs static trajs)")* — the resample-vs-replay distinction is real and
the current split names it.

**S3 — reconcile `deactivate_threshold` with `floor`. ❌ NOT A DEFECT.** *(user:
"this is intended to be set intentionally")* — see M5 and the §6 warrant row.
The threshold is a binary declaration of deactivatability, not a number to
reconcile against the floor.

**S4 — drop `scramble_conditions` from the unconditional protocol. ✅ ACCEPTED.**
*(user: "if it's not a no-op, it should indeed not come into it on the
unconditional version")* — it is a no-op today, so dropping it is
behaviour-preserving and removes a false claim from the config.

## 9. Open questions — answered

**1. Is the deactivation path meant to be reachable?**
**A: In principle yes, but not on the current default — and that is fine.**
*(user)* See M5: the cost is deliberately paid.

**2. How long before the replay buffer bootstraps, and does the controller settle
differently because of it?**
**A: Fills very fast; not a concern in practice — but pre-filling is a good
move.** *(user)* Same pattern as the prior buffer's pre-fill. Added to
[`to_do_rebuild.md`](to_do_rebuild.md) §C. It also removes M2's fold-in entirely
rather than merely logging it.

**3. Has the `fwd` pin ever been varied against replay admission rate directly?**
**A: Many pins and controller functions have been tried, and will be again.**
*(user)* The 0.2 is not defended; §6 marks it arbitrary.

---

*Warrant classes: **derived** · **measured** · **inherited** · **arbitrary** · **contested**.*
