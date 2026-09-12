# Work queue

Opened 2026-09-10 from a pass over the `configs/mk_dev.yaml` comments. Items are
things the owner deferred, not a plan. Delete an item when it lands.

---

## 1. Rip out the LR bracket

**Owner: deferred to "when I have spare tokens" (2026-09-10).**

`lr_control.mode` is now `fixed` on the live route and the bracket is not
expected to come back. Ripping it out means the mechanism, its config surface,
its docs and its comments.

**Scope, measured 2026-09-10:** 732 occurrences of `bracket` across ~40 files.

| where | what |
|---|---|
| `lr_bracket.py`, `lr_bracket_probe.py` | the mechanism itself |
| `controller.py` | `_managed_keys`, `_check_rails`, `BracketDriver` wiring, controller state `ver`/`scale`/`bracket` |
| `configs/mk_dev.yaml` | `mode`, `candidate_scales`, `trial_steps`, `safety_rungs`, `repeat_every`, `boundary_confirm_repeats`, `boundary_densify`, `trial_settle_steps`, `logz_detour_nats`, and section 2 of the `lr_control` essay |
| `analysis/keys.py`, `analysis/checks.py` | `lr_bracket/*` metric keys and the checks that read them |
| `checkpointing.py` | bracket state in the checkpoint |
| `bench/` | `runner.py`, `arms.py`, `band.py`, `ladder.py`, `surfaces.py`, `test_fidelity.py`, `test_tracking.py`, `test_ray_calibration.py` |
| `configs/*/make.py` | every generator that writes `mode: bracket` |
| `tests/config/test_config_invariants.py` | 5 tests currently FAILING because mk_dev is already `mode: fixed` (see §4 below) |

**What survives the rip:** `seed_lr`, `fixed_scale`, `burn_in_*`, `min_lr`,
`max_lr`, the whole `hard_failure` block (it is the live tripwire, not just the
trial judge), and the divergence-response/rewind path.

**What has to be decided, not just deleted:**

- `hard_failure`'s bars are fitted at `burn_in_scale` and refitted after
  promotion (`cruise_rederive`). With no bracket there is no promotion event —
  what triggers the redraw?
- `min_root_bias_correction` is a go/no-go on *measuring at all*. Meaningless
  without trials; delete or repurpose as a burn-in adequacy check?
- `lr_servo_managed` is written by `resolve_derived_config` (`utils.py:687`) from
  which `lr_*` keys are spelled `auto`. If `auto` goes, so does that whole path.

---

## 2. Re-key `z_calibration`

**Owner: "let's first lay out what the functions actually are capable of doing,
and then decide how to key / document them" (2026-09-10).**

The inventory is written: **[`design/z_calibration_capabilities.md`](design/z_calibration_capabilities.md)**.

Waiting on the owner's naming decision. The inventory's §9 states the question:
the servo and the fill are two mechanisms sharing one block name, one split
enable path (`flags.z_calibration` gates the servo, `fill_threshold > 0` gates
the fill), and a `mode` key that means the servo's mode while `fill_mode` means
the fill's.

---

## 3. Rework `conditional_vargrad`

**Owner: "I'll do it when we get to it" (2026-09-10).**

The protocol is parsed and validated but not selected. Its stage numbers are one
battery's operating point, not settled defaults.

---

## 4. Corpus arms are stale against mk_dev

Not an owner item — a consequence of the uncommitted mk_dev edits, recorded so it
is not rediscovered.

`tests/config/test_generate_corpus.py` fails on `bsz_b1000`, `bsz_b500`,
`qm9_conditional` because the historical arms carry the pre-`gated_ramp` shape:

    churn_rate            80      -> 0
    mean_residence_steps  50      -> 80
    max_size              12000   -> 50000
    balance.kind          ratio   -> gated_ramp
    balance.bounds        [.02,.93] -> bwd [.25,.9], replay [.1,.75]
    fracs.fwd             0.05    -> 0.0
    on_enter[1]           bootstrap_z:train_conditioner -> bootstrap_z:rollout:4000

`tests/config/test_config_invariants.py` fails 5 bracket tests + 2 others for the
same reason (mk_dev is `mode: fixed`, `fracs.fwd: 0.0`). Those 5 go away with §1.

---

## 5. Replay cap binds at the current settings

`churn_rate: 0` means "admit the whole live draw" (`train.py:9358`), so at
`batch_size: 1000` with growth off, the Little's-law equilibrium is
`1000 * 80 = 80,000` against `max_size: 50000`.

**The cap binds.** Per the block's own reasoning that turns eviction into
displacement, which is residual-dependent and breaks `birth_loss` as an unbiased
intake baseline. Either `max_size` wants ~240k, or `mean_residence_steps` comes
down.

---

## 6. mk_dev completeness — the standing rule

**Owner, 2026-09-10, non-negotiable:** *"Any and all live keys should exist in
mk_dev either with default values or (in the case of the training protocol
phases) in comments. mk_dev represents the codebase in miniature in its entirety,
period."*

Audit run 2026-09-10 over every `getattr(<config object>, 'key', default)` in the
package. 20 live keys were absent and have been added. Two classes were **not**
added and need a ruling:

**(a) The `lr_sensor` stage-block keys.** `kind: hyper` reads `beta`,
`beta_down`, `cos_target`, `every` (`train.py:5082`, `:5137`). `kind: ray` reads
`alphas`, `n_sub`, `period`, `dual_score`, `log_grid`, `larder_depth`
(`train.py:2938`). Both are stage-declared, so under the rule they belong in
mk_dev *as comments*.

The `ray` block conflicts with a recorded decision: it was **deleted rather than
flagged off** (owner 2026-08-26, "obvious option to turn this off" = the block's
absence), on the grounds that an enable flag beside the stage declaration is a
second switch able to disagree with it. The new rule says it should be present as
a comment. A commented block is not a second switch, so these may not actually
conflict — but it needs saying out loud once.

**(b) Conformer-route keys.** `energy_clip`, `prior_dataset_path`,
`prior_relax_steps`, `seed_relax_steps` (all `energy_config`), plus `num_threads`
and `wandb_mode` (`train_conformer.py`) are absent from **`configs/conformer_mk.yaml`**.
`internal_prior_path` and `prior_sample_size` are present. Does the "in its
entirety" rule make `conformer_mk.yaml` the conformer route's own mk_dev, or does
mk_dev carry these too?

**Deliberately not added:** `ramp_floor_range` / `ramp_knee_range` (explicit
legacy aliases for the live `ramp_floor` / `ramp_width`, `train.py:4481`), and
`lr_servo_managed` (derived at load, not hand-written).

**To re-run the audit:** the scan is a regex over `getattr(X, 'name', default)`
where `X` matches `\b(args|cfg|config|conf)\b`, diffed against every key name in
mk_dev. Note the two failure modes it has: it misses keys read without a default
(those crash on absence, so they are safe), and it misses stage-block keys read
off `stage`/`st` dicts.
