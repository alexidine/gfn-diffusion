# Infrastructure stabilization

The plan for reaching a feature-stable production foundation. Argument, in the
`docs/PROTOCOL.md` sense: it records *why* the work is sequenced this way, and is
revised when the reasoning changes rather than appended to.

Completion is defined at the bottom. Until every box there is checked this is the
active plan; after that, foundational change requires a demonstrated need.

**Current position**, by phase. The detail is below; this is the ledger.

| Phase | State | What is left |
|---|---|---|
| **0** baseline + version primitive | **DONE** | — |
| **1** canonical config | **~95%** | the comment rewrite (1.3, tiers S2-S5) and the runtime half of the mode-safety audit (1.1). Optimizer-block nesting DROPPED |
| **2** config generation | **DONE** | 2.1 `configs/generate.py` + 2.2 corpus shipped. 2.3 stays a deliberate stub |
| **3** executable invariants | **DONE**, folded into Phase 0 | extend as new rules earn it |
| **3b** analysis package | Tiers 0/1/2 **SHIPPED** | Tier 3 (figures), specified and not built |
| **4** profiling + benchmark spec | spec + `benchmarks/` **SHIPPED** | the profiling runs — cluster, user-run. Gated on `a100_stab_aug16` launching |
| **5** MLIP | local optimisations **DONE** | A100 validation |
| **6** batch sizer | **DESIGNED**, not built | gated on Phase 4 measurement |

The config schema has moved through **seven states** since this document was
written; `docs/change_history.md` is the record and `config_state.CHANGES` is its
source. States 2–7 are all Phase 1 work: they are what "consolidate" turned out to
mean once each key was moved next to the thing that consumes it -- and state 7 is
the reminder that moving a key is only half of it, the other half being that the
old spelling has to start failing.

**Phase 0 is done and the first slice of Phase 3 with it:**

- `.gitignore` extended — untracked tree 1.1 GB → 67 MB. *The commit is the
  user's to make, and Phase 1 is gated on it.*
- `project_state_version` threaded through config load, self-arming severity
  (`utils._check_state_version`).
- `config_state.py` — history and migrations as one artifact; the v0→v1 record
  covers all 43 retired keys. `docs/state_history.md` is generated from it, with
  a test guarding drift. PROTOCOL gains the *Transition* type.
- `config_invariants.py` — seven rules that were previously only config comments.
- Six invariant tests that **never ran** now run: `test_batch_invariance.py` and
  `test_periodic_scoring.py` were script-style files whose positional arguments
  read as missing fixtures, so pytest reported collection errors and collected
  nothing. `test_batch_invariance.py` was doubly blind — its `check()` recorded
  failures without raising, so a failed invariance check would have reported as a
  pass. Both now fail on a mutation and keep their `__main__` drivers.
- **Phase 3b Tiers 0, 1 and 2 shipped** — `analysis/` now carries `checks.py`
  (R2 liveness, R14 dead sensors, §4 confounds, R11) and `compare.py`, tested
  against fixtures captured from eight real runs. State in
  `docs/module_analysis.md`.
- **Phase 4's benchmark specification shipped** — `docs/design/benchmarks.md`
  plus the `benchmarks/` registry.
- **Test suite tiered** — one test was 73% of a 19m37s run and was never meant to
  run at that budget (the file's own driver used a quarter of it; pytest supplied
  the default by collecting on name). Now `check_`-prefixed and uncollected; the
  suite is 5m11s, with `pytest -m fast` as the dev loop.

Measured over the 2,244-config corpus. With back-compat dropped, the first two
rows are context rather than a work queue — nothing needs those configs to load:

| Finding | Count | Consequence |
|---|---:|---|
| retired keys, mechanically repairable | 103 | `migrate` fixes them; nothing requires it to |
| retired keys needing judgment | 303 | 285 are the `health_gate_zerr` ruler swap — left alone |
| `figs_period` not a multiple of `eval_period` | 47 | **logged no figures at all**, silently |
| DPLR unmasked with angular dims | 119 | dies at model construction under current code |
| effective batch below the ≥1000 baseline | 1,714 | reported, not failed — §9 revisits it |

The last three are the ones that still matter, because they describe faults a
*new* config can reproduce. All three are now asserted in `config_invariants.py`,
which is what stops them recurring.

**Phase 1 in progress.** The tier-C results below were measured against
`7625d09`; the schema has since moved to state 6, so re-run them against the
current baseline before quoting them as current.

- **1.0 comparator — DONE.** `config_snapshot.py` + 16 tests. Snapshots the
  *resolved* config (so `auto` is the number it will train at), the parsed
  protocol, and each stage's effective loss coefficients computed by the
  trainer's own `StageProtocol.coeffs` rather than a copy of the overlay rule.
  `compare()` splits CHANGED / ADDED / REMOVED, because consolidation adds keys
  by design and a text diff drowns in that. Validated on the case that matters: a
  single loss coefficient altered inside a full file reordering is caught, while
  the reordering alone passes clean. The loop is two commands, baseline from git:

      git show HEAD:energy_sampling/configs/mk_dev.yaml > /tmp/base.yaml
      python -m config_snapshot /tmp/base.yaml configs/mk_dev.yaml

- **1.1 mode safety — static half DONE.** `test_mode_safety.py`, 13 tests. Eleven
  keys the canonical config documents as inert are now checked to have no
  *derived* effect, so those comments are executable. Two mutations prove the
  check can see a live key. The **runtime half** needs a run, and the harness it
  needs now exists (below); the per-key runtime proofs are not written yet.

- **1.5 tier-C smoke harness — DONE.** `tierc_smoke.py` + 55 tests. Runs
  `train.py` for a fixed number of steps on `latent_gaussian`, captures a
  per-step trace of losses, learning rates, fused sub-losses and every logged
  metric, and diffs two traces exactly. It reuses `benchmarks/registry.yaml`'s
  `defaults.overrides` and the `epochs`-is-an-absolute-index arithmetic rather
  than restating either, and then **verifies the executed step count** against
  the budget, because a computed budget is a belief until something counts.

      python -m tierc_smoke --null configs/mk_dev.yaml
      python -m tierc_smoke --negative-control configs/mk_dev.yaml
      python -m tierc_smoke /tmp/base.yaml configs/mk_dev.yaml

  **The same-config spread on this target is exactly zero**, so §1's tier C is
  an exact test here as predicted — 0 of 243 values at 30 steps across two
  seeds, 0 of 14,313 at 600 steps. Two conditions are needed and both are
  measured rather than chosen: `torch.use_deterministic_algorithms` (without it
  7 of 243 differ, all grouped reductions at float32 rounding, and strict mode
  raises nothing — torch simply does not select those kernels by default), and
  excluding wall-clock keys, 21 of 522. The negative control resolves a **1e-6**
  relative change to one loss coefficient.

  Three things it found on the way, each of which would have made a passing
  result meaningless:

  - **The registry's `defaults.overrides` no longer loads.** It sets
    `ray_calibration.enabled: false`, and both that key and its parent are now
    retired and hard-fail at preflight. Applied verbatim the harness dies;
    applied quietly the block is gone and nothing says so. The harness drops
    them and reports each drop. **`benchmarks/registry.yaml` needs updating for
    its own purposes** — `registry.py::_validate_defaults` still *requires* the
    retired spelling, so this is a two-file change and is not done here.
  - **`configs/problems.yaml` cannot run `latent_gaussian` as it stands.**
    `prior_path` is null and `init_prior_dataset` `torch.load`s it
    unconditionally; `analyze_kwargs` is empty, so the analytic target has no
    centre or width. The harness fills both from `configs/gauss_aug12/spec.py`
    and reports the gap rather than closing it silently.
  - **The pre-consolidation config does not load under current code**, for two
    reasons of different kind — five retired keys, which `config_state.migrate`
    repairs mechanically, and `auto` LR rates with no stage sensor, which
    migration correctly refuses because choosing a sensor is judgment. See the
    tier-C result below for how that is resolved.

- **Tier C, run.** `configs/mk_dev.yaml` at `7625d09` against the current file,
  both under current code, 600 steps. **Steps 0–500 are bit-identical**,
  including the phase-1→2 stage transition, which fires at step 381 in both.
  From step 501 every loss differs.

  The baseline was made loadable by pinning its four `auto` rates to
  `adaptive_lr.seed_lr` — a translation, not a guess: with no sensor, `auto`
  trains at the seed with `peak_scale` fixed at 1.0, which is exactly what an
  explicit float does. That is stated on every run rather than applied quietly.

  **The divergence is one feature, and it is the documented one** — the
  `equilibration` stage's `lr_sensor: {kind: ray}`. Tier A/B over the same pair
  reports 8 changed values and all 8 are that feature or follow from it; no loss
  coefficient, batch, clip, schedule or buffer setting moved. So the
  consolidation did not violate *schema may change freely, behavior may not*.

  What the run adds is worth its own entry, and has one — **`findings.md`
  F-039, fixed in F-040**. The sensor changes **no learning rate at all** (every LR bit-identical
  across 600 steps, `cal_status: warmup`, `cal_applied: 0.0`), yet the run
  diverges completely, because the probe *samples* eight replay draws before the
  controller refuses the reading, and that shifts the RNG stream. `in_warmup()`
  exists for exactly this and the plateau sensor uses it; the ray path does not.

  **What tier C has NOT covered:** any sensor actually actuating.
  `warmup_steps` is 1000 and `rearm_warmup` restarts it at every stage
  transition, so the transition at 381 pushes the first permitted action to step
  1381 — no sensor acts anywhere inside a 1200-step window. Covering actuation
  needs a run past that, and is not the same measurement as the one above.

- **1.4 `problems.yaml` — DONE.** `mode_presets.yaml` is retired. It had declared
  itself "Reference only — never loaded by train.py" and drifted exactly as an
  unexecuted reference does: it prescribed **seven keys that are now retired and
  hard-fail at load**, taught the deleted `anchor × 25/T` learning-rate rule, and
  labelled its worked example the "current mk_dev state" at W256/T40 when the file
  was W512/T10. Replaced by `configs/problems.yaml` — problem-intrinsic settings
  only, with 18 tests including one that rejects any retired key and one that
  rejects a tuning knob leaking in. Its durable half (the intensive/extensive
  derivation) moved to `docs/design/width_and_length_scaling.md`.

- **LR sensors — DONE.** Both canonical stages now declare one (`train_prior`
  hyper β=0.1, `equilibration` ray); previously neither did, so all four `auto`
  rates sat at the seed for entire runs. `auto` now *requires* an adaptive sensor
  at load. The old gate tested `ray_calibration.enabled`, which passed mk_dev
  while nothing asked and would have rejected a legitimate hyper-only config.

**1.2 — the five mode-varying key groups, classified, then resolved.** Each was
classified against the code and then adversarially checked; four of the five
recommendations did not survive, two of them because they would have changed the
canonical unconditional route. What replaced them was not coexistence machinery
but **stage scoping**: four of the five now travel with the protocol, and only
`dplr_*` — which was never mode-varying in the first place — needed nothing.

| Key group | Verdict | Action |
|---|---|---|
| `dplr_rank` / `dplr_rho_max` | **TUNED**, check held | **none** — already one value (6 / 0.5). The 10/0.9 pair has zero recorded support, entered in a bulk mode-switch commit, and exists nowhere in the tree today |
| `lr_flow` | **not derivable** | derivation *refuted*: 22 live configs carry `1.0`, falsifying the 0.1/1e-4 two-branch story. Left explicit, then **resolved structurally** — a stage sets its own rate via `on_enter: [set_lr_flow:<x>]` |
| `protocol.stages` | **STRUCTURAL** | **DONE** (state 4) — `protocol:` names one, `protocols:` holds them all |
| `z_calibration.*` | recommendation refuted | the *proposed* sensor change was refused for flipping the canonical trigger; **resolved structurally instead** — enablement is a stage flag, the block keeps only *how* |
| `tb_z_source` family | recommendation refuted | the *proposed* hard gate would have aborted the canonical run at its first stage; **resolved structurally instead** — moved into `{mode}_loss_coeffs`, which a stage can override |

**Why four "refuted" verdicts still ended in a move, and why that is not a
reversal.** Each refuted recommendation was a *gate* — make the wrong combination
fail at load — and each was refused for the same reason: the canonical
unconditional route is a legitimate config, so any gate strict enough to catch the
conditional mistake also rejected the route we actually run. Stage-scoping
dissolves that. It does not ask which combination is wrong; it makes the route
that needs a value the thing that declares it, so the wrong combination is not
reachable rather than merely forbidden. The classification was right about the
mechanism it evaluated and wrong only about that being the last mechanism
available.

### 1.2's structural remainder

Five keys sat parked as `# todo` in the canonical config. They are not a separate
refactor — they are the unfinished half of §1's "activation derives from the
problem block", and Phase 1 cannot be called done while they stand. **Four are
now landed; one remains.**

| Key | Status |
|---|---|
| `ray_calibration` merge/move | **DONE** — moved under `adaptive_lr`, `enabled` deleted (state 2) |
| `conditional_worst_quantile` | **DONE** — re-homed beside the stages that consume it |
| `protocol:` | **DONE** — `protocol:` names one; `protocols:` holds them all (state 4) |
| `mle_slope_*` → phase-1 exit | **DONE** — now `mle_gate: {slope_t, min_rate, window}` on the declaring stage (state 5) |
| optimizer-block nesting | **DROPPED** 2026-08-17 by the user — see below |

They are one change in kind, not five: every one is *move a key nearer the thing
that consumes it*. And 1.2's own verdict stands for the hard one — **name a
protocol per problem**, which is what `configs/problems.yaml` exists to hold.

`protocol:` was parked here pending the experiment synthesis, and unparking it
first was what made `mle_slope_*` cheap: the prediction below was that the MLE
gate should move *with* `protocol:` rather than before it, and that is what
happened one state later.

**THE CONSTRAINT, and why `CHANGED: 0` does not certify this class of change.**
Four validators read literal dotted paths, and
`config_invariants.auto_lr_requires_an_adaptive_sensor` returns `[]` on absence
while `utils` makes it a *raising* load gate. A validator that goes quiet
produces an **identical resolved config** — so the comparator, which compares
what *this* config resolves to rather than what a future bad config would be
allowed to do, cannot see the characteristic failure. Validators move in the same
commit or the restructure does not happen.

The procedure that does certify it, used for the two landed moves: write a
negative control per validator, demonstrate it FAILS before the move,
move keys and validator together, demonstrate the same control still fails. A
control that passes at step two is not a control, it is a bug.

**The optimizer-block nesting is DROPPED, not deferred.** The user scratched it
on 2026-08-17. The case for it was consistency -- the four learning rates sit at
top level while everything else that steers them lives under `adaptive_lr` -- and
consistency is the weakest reason on offer for a change with this shape: 64 read
sites, several reached through an object alias a name-based sweep cannot see, and
`getattr(args, 'lr_flow', None)` returning None rather than raising at every one
of them. It buys no new capability and closes no failure that has cost anything.
The cost sat entirely on the side of the risk. Recorded here rather than deleted
so the next reader does not re-propose it; the sizing below is why.

**The original sizing, kept as the reason.** Sized at **64 attribute-read
sites across 16 files** — `train.py`, `controller.py`, `utils.py`,
`checkpointing.py`, `train_conformer.py`, nine `bench/` files — and several read
through an object alias (`a = m.args; a.min_lr`), which a name-based sweep does
not see. Worse, the silent-failure sites are real: `getattr(args, 'lr_flow',
None)` and friends return `None` rather than raising. It is a single deliberate
change with its own negative controls, not a rider on a comment pass.

**The mode-key migration, which this table understates.** Landing `protocol:`
turned the stage list into the place a route declares itself, and three keys that
had been hand-edited at every mode switch followed it there. They are listed
together because they are one argument, not three:

| was | is | mechanism |
|---|---|---|
| `condition_log_z.{fwd,bwd,replay}_tb_z_source` | `{fwd,bwd,replay}_loss_coeffs.tb_z_source` | a property of what a branch's loss does, and `loss_coeffs` is the one block a stage can override per branch |
| `z_calibration.enabled` (global) | a stage flag, `flags: {z_calibration: true}` | off by omission, which is what the conditional protocol wants; the block below keeps only *how* |
| `lr_flow` (hand-edited 0.1 → 1e-4) | stage `on_enter: [set_lr_flow:<x>]` | two new stage actions, `set_lr_flow` and `set_lr_policy` |

This is the payoff the whole phase was aimed at. **Selecting a protocol now
carries the route's Z regime with it**, so the F-042 trio — the three settings
whose inheritance detonates `var_conditioning` within ~30 steps — is no longer
something a mode switch has to remember. Exactly one key still needs a hand-edit,
`condition_log_z.half_life_visits` (7.0 unconditional / 28.0 conditional), because
it is global rather than stage-scoped; the mode-switch table in
`configs/mk_dev.yaml` is where that is recorded.

**It is not finished, and the unfinished half is the load gate.** Measured
2026-08-17 on the real load path, with a known-retired key as a positive control:
all four old spellings **load clean and are silently ignored**. None was added to
`utils._RETIRED_KEYS`, the migration carries no `config_state.CHANGES` record, and
`project_state_version` did not move — so by PROTOCOL's own rule (a renamed or
reinterpreted key gets a record *and* the integer) this migration is unrecorded.

The cost is not hypothetical. Every arm of `configs/a100_stab_aug16` writes
`tb_z_source` into the dead home, where it resolves to `None` and falls back to
`'learned'` at `train.py:1105` — the F-042 detonation value — while the file reads
`persistent` in three places and stamps state 6. A version stamp that says
*migrated* over contents that are not is worse than no stamp. Closing this is
**state 7**, and it is the first thing Phase 1 owes.

### State 7's scope, held open deliberately

It is held rather than landed because it makes `configs/a100_stab_aug16/make.py`'s
`CURRENT_STATE_VERSION` assertion fail until `base_uncond.yaml` is re-snapshotted,
and that battery is mid-regeneration. Two changes ride together, so the
re-snapshot is paid once:

| key | disposition |
|---|---|
| `condition_log_z.{fwd,bwd,replay}_tb_z_source` | retire — moved to `{mode}_loss_coeffs.tb_z_source` |
| `z_calibration.enabled` | retire — moved to the stage flag `flags.z_calibration` |
| `adaptive_lr.envelope_freeze_drop` | retire — replaced by a boolean freeze switch |

**Why `envelope_freeze_drop` goes with them, since "unused knob" is the wrong
reason.** It was a threshold on how far `peak_scale` had fallen from its
high-water mark, and it existed to separate *"the sensor is pulling against the
ramp"* from noise in that signal. Once the ramp-exit detector stopped being the
actuator, there was no noise left in the signal to threshold: nothing moves
`peak_scale` during a ramp except `on_divergence`, whose cut is unambiguous by
construction. **The knob did not rot — the fix made it unnecessary**, which is a
better reason to delete something than disuse.

Measured before proposing it: no sensor moves `peak_scale` during warmup (all
three read exactly 1.0 when driven), and across `hyper`/`ray`/`plateau` every
threshold from 0.0 to 0.4 gives the identical verdict on a divergence, because
`divergence_cut: 0.5` is a 50% fall and the largest per-sensor default is 5%. The
`_FREEZE_DROP_DEFAULT` lookup cannot change an outcome. No git-tracked config sets
the key, so its migration is a pure drop needing no judgment — the cheapest entry
in the record. What survives is the one bit that was always load-bearing: freeze
on, or off.

**A live bug found and fixed on the way:** `lr_flow: auto` resolved to the
literal string `'auto'`. It is deliberately absent from `_LR_KEYS` (alpha* is
measured over policy parameters only, so the flow groups are exempt from both the
envelope and `peak_scale`), so nothing filled it in and the string was assigned
straight to `param_group['lr']`. Now refused at load with the reason.

**1.3 (comment discipline) — the audit landed and S1 with it.** It was held so the
rewrite would be driven by the §4 work list rather than duplicating it;
`docs/design/comment_audit.md` is that list, and it is the authority on what
remains. S1 — the ten comments *contradicted by the code they sit on* — was
applied ahead of 1.2 rather than after, on the reasoning that a comment wrong
about behaviour does its damage during consolidation, which is the operation most
likely to trust it. S2–S5 are open. S4 was deferred behind 1.2 and 1.2's config
prose has now been rewritten, so S4 is unblocked.

The audit's own ranking has been vindicated twice over, and both times in the
canonical config. `mk_dev.yaml`'s mode-switch table is now S1 in the audit's exact
sense: it lists five keys as needing a hand-edit at every mode switch, and three
of them stopped needing one when the mode-key migration moved them. Two of those
three name keys that **no longer exist**, and a reader who follows the table edits
a key that loads clean and does nothing. That is the same defect class as the
`increment_batch_size` docstring, in the one file §1 says a reader inspects to
learn current state.

---

## 1. Framing

**`configs/mk_dev.yaml` is the canonical master.** Production configs are spawned
from it. It is a microcosm of intended project state: a reader inspects that one
file and knows the current defaults, modes, and production settings.

**Mode selection is a problem choice, not a config rewrite.** Switching between
conditional/unconditional or molecule/toy changes only *problem-intrinsic*
settings — energy function, data paths, the conditioning flags, space groups,
temperature. Every other section of the canonical config is written so both
alternatives coexist and only the selected one activates. `mode_presets.yaml`
demoted accordingly and is now retired: a 322-line mode-overlay matrix that
rewrote whole config sections became `configs/problems.yaml`, a thin problem
registry that is *loaded* rather than reference-only.

**Where "only the selected one activates" ended up living.** Not in a coexistence
rule applied to every section, which is what this paragraph originally implied,
but in the stage: `protocol:` names a route, `protocols:` holds them all, and each
stage declares the settings its route needs. Switching route is that one word plus
the problem keys. The one key that still resists is
`condition_log_z.half_life_visits`, which is global rather than stage-scoped.

This makes the central §1 requirement and the central §5 invariant the same
statement:

> **Inactive config modes must not influence execution.**

which is why it is written as a test before the consolidation, not after.

**Ownership constraint.** The canonical config is user-controlled; the standing
rule is *schema may change freely, behavior may not*. Consolidation therefore has
a hard acceptance criterion — stated at three tiers, because "bit-identical" is
the wrong bar on a GPU/MLIP path and this document elsewhere says exactly that:

| Tier | What | Bar |
|---|---|---|
| **A. Parsed config** | the loaded, preflighted, derived-resolved config object | **exact equality** |
| **B. Deterministic pre-runtime state** | seeded model init, protocol stage parse, buffer seeding, resolved LR/clip values, batch plan | **exact equality** |
| **C. Execution** | losses, energies, metrics over N steps | **within a measured floor** |

A and B carry most of the risk and are cheap to test exactly: they are pure
functions of the YAML and the seed, with no kernel nondeterminism in them. A
consolidation that changes nothing must change nothing there, full stop.

C's floor is **measured, not chosen**. The reference is the spread of the *same*
config run twice — that is the null distribution, and the consolidated config has
to be indistinguishable from it. Picking a tolerance by eye instead yields a test
that passes because it is loose, which is a failure mode this project has already
paid for. Two consequences:

- On a **deterministic** target the same-config spread is zero, so tier C
  collapses to an exact test and should be run that way. `latent_gaussian` is the
  natural choice, being the analytic toy. Do this one first — it is the sharpest
  instrument available, and it costs seconds.
- On the **MLIP** route the spread is not zero: UMA is not bit-reproducible on
  GPU and the reward noise floor is ~0.1 kJ/mol. `torch.equal` is the wrong bar
  there; the comparison is against the measured repeat-run spread.

Any change that alters what an experiment does is a separate, explicit ask.

---

## 2. What already existed

Roughly a third of the requested machinery was built before this plan started.
Building it again is the main avoidable cost in this project, which is why the
inventory is recorded rather than the gaps alone.

**This table is the STARTING position and is not maintained** — it is the evidence
for how the phases were sequenced, not a status board. Current status is the
ledger at the top.

| Workstream | Existing | Gap |
|---|---|---|
| §1 canonical config | `configs/mode_presets.yaml` — the mode matrix, already written down | header says *"Reference only — never loaded by train.py"*; nothing consumes it |
| §2 semantic history | `utils._RETIRED_KEYS` — 43 entries, each carrying its reason (`utils.py:157`) | no version stamps, no migrations, not readable as history |
| §3 migration | `utils.preflight_config` **rejects** retired keys at load (`utils.py:306`); checkpoint `problem_def` carries `schema_version` | rejection is not migration; the config itself has no version |
| §6B generation | 21 `configs/*/generate_configs.py`, each forked from the last | no canonical source, no provenance, conventions live only in the fork lineage |
| §8A/D MLIP preprocessing | `batch_to_fairchem_batch` (vectorized, live) with `batch_to_fairchem_atomicdata` as reference and `verify_fairchem_batch_equivalence` between them; `batched_pbc_neighbour_list` (GPU, wired into the MACE route); equivalence tests for all of it | **substantially done** — see §3 below |
| §5 invariants | `bench/` sandbox (fake modeller, 12 tests), `grad_clip_guard`, `gpu_guard` | the *config* rules are not asserted anywhere |
| §7 profiling | `gpu/util_recent` + `gpu/util_policy` logged on wall-clock cadence; `energy/mace_host_frac` | no canonical workload spec, no rerunnable suite |

---

## 3. Two corrections to the requested sequence

**§8 gets re-measured before it gets resequenced — and the standing prior just
got weaker, which strengthens the case rather than weakening it.**

This section used to reason from a preprocessing pass measuring **1.38x**
end-to-end against a forward that was ">99% of the call". The MLIP tab has since
corrected both numbers: the end-to-end gain is **~1.08x**, and the UMA graph
build, once broken out of the forward rather than lumped into it, is
**2.6–6.1%** — small, but not the invisible remainder the old figure implied.

Follow the argument through, because the conclusion inverts. At 1.38x the prior
was "preprocessing has already been harvested, so look at the forward" — an
argument for *skipping* A and D. At ~1.08x that reading collapses: a 1.08x
end-to-end gain is close to the noise of the thing it was measured on, so it no
longer licenses a claim about where the time is at all. **A weaker prior is not
permission to guess in the other direction; it is a smaller budget of belief to
spend before measuring.** Whatever headroom exists, we now know less about where
it sits than the original number suggested we did.

So §8's order changes only in that **measurement comes first** — Phase 5.0
re-profiles the whole pathway, neighbour-list construction included and broken
out explicitly, and the sequence follows what the split actually shows. What
survives as expectation is thin and should be held that way: if the forward does
dominate, `crystal_inference_settings` (activation checkpointing, precision), the
`always_use_pbc` path and §8B's built-in execution modes are where to look, and
`energy/mace_host_frac` exists to say whether MACE splits the same way. The
2.6–6.1% graph build is now a named, separable term rather than a rounding error,
which is exactly the kind of thing a re-profile should size properly.

**§2 conflicts with `docs/PROTOCOL.md` as written.** PROTOCOL declares four doc
types and puts *Log* — "what happened when" — in git history only, never in a
file. A chronological record of functional changes reads as Log.

*Resolution:* it is not Log. A state-transition record answers "how do I get a
config from state N to state N+1", which is State about the migration path, and
its correctness is checkable. It is stored as **data next to the migration that
implements it**, so history and executable transform are one artifact and cannot
drift. The human-readable chronology is generated from that data, never
hand-edited. PROTOCOL gains one line naming this exception.

---

## 4. Phases

Ordered by dependency. Cost class: **L** = local, no cluster, no measurement;
**M** = local measurement; **H** = cluster time, user-run.

---

### Phase 0 — Baseline and the version primitive · **L**

Everything downstream stamps or migrates against a version, so the version comes
first. Nothing here changes runtime behavior.

**0.1 Clean tree.** 110 untracked paths, no `.gitignore`, ~1.1 GB of untracked
checkpoint binaries under `checkpoints/`. Every later phase rewrites docs and
configs, and PROTOCOL forbids overwriting uncommitted docs. Write `.gitignore`
first (checkpoints, `wandb/`, `SCRATCH/`, `logs_*.txt`, `*.pt`), then commit.
*The commit is the user's.*

**0.2 `project_state_version`.** A monotone integer in the canonical config.
Threaded through `preflight_config`; recorded in run provenance and alongside the
checkpoint `problem_def` (reusing that precedent, not colliding with its
`schema_version`). A config lacking the key fails at load with a message naming
the migration command — the same hard-failure discipline `_RETIRED_KEYS` already
uses, and for the same reason.

**0.3 `config_state.py` — history and migration in one module.** One record per
transition, carrying the §2 fields: version, functional behavior changed,
principal components, config keys added/removed/renamed/reinterpreted, migration
callable (or `None` + the reason judgment is required), invariants, validation
performed, commit hash. `migrate(cfg, from_version) -> cfg` composes the sequence.
`v1` is the current state; the 28 `_RETIRED_KEYS` entries fold in as its history,
gaining the version stamps and migrations they lack.

**Acceptance.** An unstamped config fails preflight with an actionable message. A
`v1` config round-trips through `migrate` unchanged. The generated Markdown
chronology matches the records.

---

### Phase 1 — Canonical config · **L**, gated on Phase 0

**1.1 Mode-safety audit, written as a test first.** Enumerate every
mode-dependent key in the canonical config and prove the inactive branch is inert
— *at runtime, on physical configs*, because this codebase fails silently:
inert-looking flags and mutating getters are a known failure mode here. The
`mode_presets.yaml` header already asserts a set of keys are "mode-safe as of
2026-07-21"; that assertion becomes executable or it gets corrected.

**1.2 Consolidate.** Fold conditional settings in alongside unconditional so both
are present and activation derives from the problem block. Remove stale,
duplicated, and conflicting defaults and pathways. Resolve the existing
`# todo` comments.

*Refined by what it took to execute:* activation derives from the **stage**, and
the problem block's job is to select the protocol. A global key cannot be
mode-safe by inspection, because "inactive" is a claim about a route that the key
itself does not name — which is why the mode-safety audit (1.1) kept finding
assertions it could not make executable. A key declared on the stage that consumes
it needs no such claim: it is present exactly on the route that reads it. That is
the generalisation of §1's requirement, not a departure from it.

**1.3 Comment discipline (§4).** The canonical config currently narrates
experimental history in-line — run ids (`tuphwfkm`, `cazwlyy1`, `gejezmjg`,
`rpvez6ep`), battery names, dated observations, evolution stories. Rewrite to:
what the setting controls, how it interacts with others, a brief justification of
a non-obvious default, an invariant, or a pointer to `docs/design/` where a real
derivation exists. Derivations that are worth keeping move to `design/`; the rest
is already in git.

**1.4 `mode_presets.yaml` → `configs/problems.yaml`.** Problem-intrinsic settings
only, and *loaded*, not reference-only. Its previous job — recording which
sections to rewrite per mode — is obsoleted by 1.2.

**Acceptance.** The three-tier equivalence in §1: exact on parsed config and
deterministic pre-runtime state, exact on a `latent_gaussian` run, and within the
measured same-config repeat spread on the MLIP route. Mode-safety tests pass, and
each fails when its invariant is deliberately broken — a test that cannot fail is
not evidence.

The tier-A/B comparator is worth building before 1.2 rather than after: a
harness that dumps the resolved config and pre-runtime state for a given YAML
turns every subsequent consolidation edit into a one-command diff, which is what
makes it safe to do the rewrite in small steps instead of one leap.

---

### Phase 2 — Config generation · **L**, gated on Phase 1

> **Backward compatibility is not a requirement** — for configs or for
> checkpoints. That removes most of what §3/§6A asked for. The migration runner
> drops to a stub (2.3), the 303 judgment-needing configs are left alone, and the
> retired-key gate keeps doing the one job that still matters: refusing a stale
> config loudly instead of ignoring a key the author believes is live.
>
> The *semantic history* half of §2 is unaffected and still worth its keep. It
> exists so a future reader can tell what a change meant, which is a different
> need from rescuing an old file and does not go away when back-compat does.

**2.1 Production config generation (§6B).** `configs/generate.py`:
`problem + mode + run-specific overrides → config`. Starts from canonical, never
from a nearby exemplar. Knows output locations and naming conventions. Accepts
only genuinely run-specific overrides. Stamps `project_state_version` and
provenance. Validates paths, mutually dependent settings, and inactive-mode
behavior. Emits a compact deviation-from-canonical summary.

**2.2 Regression corpus.** Reproduce a handful of representative historical
production configs from explicit inputs. The 21 existing generators are the
corpus and the source of current conventions. Once reproduction holds, canonical
+ generator are authoritative and the historical configs are reference only.

**2.3 Migration runner (§6A) — STUB, not built.** `config_state.migrate` already
does the mechanical part and reports what needs judgment; that is enough for the
occasional one-off. The workflow around it (identify state → migrate → diff
against canonical → validate → report) is not worth building while nothing
depends on old configs loading. Revisit only if a specific old run family has to
be revived.

**Acceptance.** `generate.py` reproduces the corpus from explicit inputs. The
workflow is documented as a runnable command.

---

### Phase 3 — Executable invariants · **L**, partly independent

The §5 rules as assertions. Independent of Phase 1 except where noted:

- inactive config modes do not influence execution *(Phase 1)*;
- deprecated keys migrate or fail clearly, never silently vanish *(Phase 0)*;
- representation/boundary invariants hold before scoring *(independent)*;
- effective training batch meets the configured minimum *(independent)*;
- optimized MLIP paths agree with reference within tolerance *(exists; extend)*.

Each test must be shown to fail when its invariant is broken.

---

### Phase 3b — Run-analysis package · **L**, independent

Specified in full at `docs/analysis_package_spec.md`; that file is the build
document and is deleted once executed. Summarised here only for sequencing.

This is the first workflow to graduate from §5's stub list, and it meets §6's bar
for automation — inputs, outputs, sequence, failure behavior and acceptance
criteria are all pinned. It also has the strongest independent justification of
anything in this plan: the feature-extraction script has been rewritten **six
times** in six session scratchpads, diverging each time, with none of it in the
repo. Every session pays that cost again.

**Runs in parallel with Phases 1–2.** It reads wandb output rather than code, so
it is insulated from the config rewrite everywhere except `keys.py`, which owns
the metric taxonomy. That one coupling is real: Phase 1 renames config keys, and
any metric name that moves with them lands in `keys.py`. Keeping every
metric-name literal in that single file is what makes the coupling a one-file
change rather than a sweep.

Build Tier 0 and stop for review. Three open questions go to the user before
`TOPLINE[VARGRAD]` can be coded — the spec lists them.

Two requirements from the spec deserve repeating because they are the ones most
likely to be got wrong, and both are failure modes this project has already paid
for:

- **A metric has three states, not two:** live / absent / not-meaningful-on-this-
  route. On the conditional VarGrad route the log Z and TB keys exist and carry
  numbers that must not be read as they would be on a TB run. Collapsing that
  third state into "absent", or rendering it as zero, is worse than crashing.
- **The package emits no verdicts.** Mechanical principles become checks; the
  rest surface their inputs and stop. A tool that concludes "the run is healthy"
  has failed its spec.

---

### Phase 4 — Profiling and the benchmark spec · **H**, user-run

Specification before measurement: §11's benchmark definitions come first, so the
sweep produces a rerunnable suite rather than a one-off. Per benchmark: workload,
training mode, hardware class, fixed work quantity, metrics, correctness
reference, comparison criteria.

Characterize per §7 across representative modes — fwd/bwd/replay/fused,
conditional/unconditional, toy/MLIP — since their profiles differ materially.
Local profiling for development; A100 for anything where production hardware
behavior matters.

**The utilization question, and what counts as answering it.** Phase 6 needs to
know what the scheduler judges. It does *not* need the scheduler's source code,
and must not be blocked on getting it — the enforcement mechanism is someone
else's implementation detail and may simply be unavailable.

The deliverable is a **conservative observable proxy, demonstrated to track the
real thing well enough**, not the real thing. Concretely, an acceptable answer is
any of:

1. the actual statistic and window, if documentation or an admin can supply it;
2. a proxy whose agreement with cluster-visible evidence is *shown* — job
   cancellations against the logged trace, and `sacct`/`scontrol` fields — over
   enough jobs to mean something;
3. failing both, the most conservative available reading plus a stated margin,
   with the margin's cost in throughput measured so the price of the ignorance is
   known rather than hidden.

**`nvidia-smi` carries none of case (2)'s weight, and it was wrong to list it
there.** It is an independent *sampler* but not an independent *instrument*:
`torch.cuda.utilization()` wraps `nvmlDeviceGetUtilizationRates().gpu` and
`nvidia-smi --query-gpu=utilization.gpu` reports **the same NVML counter**. So it
can control for cadence, phase and eval blindness — genuinely useful — but it
cannot corroborate anything about that counter's own semantics, because it *is*
that counter. Only cluster-visible outcomes are outside the instrument. See
`docs/design/phase6_measurement_request.md`.

**The error has a DIRECTION, and that is what makes "conservative" definable.**
The in-process sampler sits in the training portion of the loop body, so eval,
figure logging and archiving — later in the same iteration — contribute no
samples. A 300 s eval contributes zero. The run's *least occupied* minutes are
therefore omitted from the series while the scheduler counts them, so
**`gpu/util_policy` overstates what the scheduler sees.**

That asymmetry is not decoration; it decides which way to err. A proxy that
reads low costs throughput. A proxy that reads high gets the job **killed while
the dashboard looks clean** — the failure with no warning attached. Conservative
here means *biased toward the cheap failure*, and case (3)'s margin is sized
against this direction, not around it.

`gpu/util_policy` (already logged, 7200 s window) is the standing candidate for
(2) and (3). Whichever holds, **write down which case applies** — a proxy adopted
under case 3 and later remembered as case 1 is how a margin quietly becomes a
law. Phase 6 proceeds on the proxy either way.

The subsidiary question is measurable regardless of how the first resolves, but
it answers something narrower than it looks: `gpu/util_policy`, `gpu/util_recent`
and external `nvidia-smi` all read one counter, so a disagreement between them
isolates a **sampling or windowing artifact** — cadence, window length, phase,
eval blindness — and never a discrepancy between two measurements of the world.
That is still worth having: those artifacts are exactly the terms the direction
argument above turns on.

**The benchmark specification is `docs/design/benchmarks.md`**, with the machine
-readable workload registry under `benchmarks/`. That document is the authority
on what a named benchmark means; this section is only its place in the sequence.

**Acceptance.** A small named set of canonical workloads, rerunnable by name,
with current baseline numbers recorded as graded findings; plus a named
utilization proxy with its case (1/2/3) stated.

---

### Phase 5 — MLIP optimization · **M/H**, gated on Phase 4

Numerical equivalence is a hard gate, and the existing `verify_*_equivalence`
harnesses are the pattern — extended to whatever diversity of structures the
change touches. Note the measured reward noise floor: UMA is not bit-reproducible
on GPU, so `torch.equal` is the wrong bar and the tolerance must be stated
against that floor.

> **Before any local measurement is trusted: the policy rollout is DISPATCH-BOUND
> and `compile_policy` is OFF on this dev box.** Measured 2026-08-16, eager, batch
> 256, T=12, toy target: widths 64 / 256 / 512 (52x the parameters) cost
> 61.1 / 62.7 / 58.7 ms/step — identical — and CUDA beats CPU by only 1.9x. Cause:
> ~937 `nn.Module` calls per training step, so the cost is dispatch and launch
> overhead, not arithmetic. `compile_policy: auto` resolves to Linux+CUDA only, so
> **a local ms/step under-represents the A100 by an unknown factor, and width
> looks free when it is not**. Re-measure with compile on before transferring any
> number. Not yet measured: production width with compile actually on, batch
> 1000+, or a crystal route with a live MLIP.

**5.0 Re-profile the whole pathway first.** Not a formality, and now with less
prior to lean on than when this was written — the end-to-end figure fell from
1.38x to ~1.08x and the graph build turned out to be a separable 2.6–6.1% rather
than part of the forward. Break the energy call into preprocessing, neighbour-list
construction, forward, and host↔device transfer, separately for MACE
(representative acridine structures) and UMA (broader crystals). Until that split
is current, every later step in this phase is guesswork.

**5.1 Neighbour lists, explicitly (§8D).** Called out on its own because it is
the item most likely to be dismissed on stale evidence, and because the current
implementation has a specific way of being fast on paper and slow in practice:

- `pbc_neighbours.batched_pbc_neighbour_list` has a fast path
  (`_pairs_by_radius_search`, ghost expansion + one radius search, ~92x less work)
  that **returns `None` and silently falls back** to the O(Σn²·K) all-pairs
  kernel when `torch_cluster` is not importable. *Verify the fast path is
  actually taken on both the dev box and the A100* — a silent fallback here looks
  like a slow GPU rather than a missing dependency.
- It is wired into the MACE route (`AL_mace_utils`). UMA delegates graph
  construction to fairchem internally, so establish by measurement whether
  anything there is replaceable at all before assuming it is.
- The module's own docstring records matscipy/ASE at 64.9% of the AtomicData
  build at 128 graphs. Re-measure it; that is the number the whole approach rests
  on.
- The shift-grid range is the part that fails silently — too small a range drops
  long edges and nothing raises, the energy just moves. Any change here is gated
  on the exact edge-set comparison the existing tests already make.

**5.2** Built-in execution modes (§8B), then **5.3** act on the re-measured
split, then **5.4** end-to-end validation on training throughput, not isolated
kernel timing.

---

### Phase 6 — Batch sizing and utilization · **H**, gated on Phases 4–5

Replace, do not patch further. Two objectives in strict priority: satisfy the
cluster utilization requirement, then maximize throughput subject to it.

Measurement-driven: probe candidate operating points and pick the fastest that
safely satisfies the constraint. Startup calibration first; periodic re-probing
only if measurements show the optimum moves materially during training.

`bench/` is the pattern for the controller sandbox — fixed rates as arms, metrics
as pure functions of a trace, no oracle. Two traps that sank the previous
generation and must be designed against: an occupancy rule that was measured
false and still outranked the throughput gate, and a knee walk with no floor that
descends forever under flat throughput.

---

## 5. Deferred

Stubs only, until repeated concrete use justifies design:

- documentation generation/refresh;
- standard experiment launch;
- regression/failure triage.

*(Canonical W&B retrieval and analysis has left this list — it is specified and
scheduled as Phase 3b.)*

---

## 6. Completion criterion

Unchecked means unchecked. Where a box has real partial progress it is annotated,
because a half-built thing recorded as done is how this list would stop working.

- [x] **one clear canonical production-config state** — `mk_dev.yaml` is
      canonical and stamped at state 7, and the stamp means what it says: the
      migrated keys hard-fail at load, verified by injection. The last open item,
      the optimizer-block nesting, was dropped as consistency-only churn.
- [x] **production-config generation is a scripted workflow, not an agent search**
      — `configs/generate.py`: canonical + problem + overrides → validate → stamp
      → emit, with a deviation-from-canonical summary. 20 tests, and a regression
      corpus (2.2) reproducing three historical arms from explicit inputs.
- [x] **generated configs carry reliable provenance** — every arm records the
      canonical file's SHA and the state it was built against, so "was this built
      from the config I am looking at" is a string comparison. The bar was set by
      the failure it answers: `a100_stab_aug16` arms once carried a state stamp
      that was *wrong about their own contents*, which misleads rather than
      merely missing.
- [x] **substantive code/config changes have semantic history** — the machinery
      works (`config_state.CHANGES` → `docs/change_history.md`, drift guarded by a
      test) and the counterexample is closed: the mode-key migration's missing
      retirements shipped as state 7, with the measurements that justified each
      disposition in the record. The discipline now has teeth, because an
      unretired rename is a key that loads clean and that is what state 7 was.
- ~~historical configs migrate systematically~~ — dropped: back-compat is not a
      requirement. A stale config fails loudly at load; that is sufficient, and
      the gate it depends on now fires (state 7).
- [ ] **comments and docstrings follow current-state discipline** — audit landed
      (`docs/design/comment_audit.md`), S1 applied, S2–S5 open.
- [ ] the initial recurring workflows are scripted and documented
      (update-old-run · production-config generation · functional-change ·
      performance-investigation · run analysis) — **two of five done**: run
      analysis (Phase 3b, Tiers 0–2) and production-config generation (Phase 2.1).
      The other three are not.
- [ ] representative training modes have current end-to-end profiles — gated on
      `a100_stab_aug16` launching.
- [ ] MLIP bottlenecks addressed, numerical equivalence demonstrated — local
      optimisations done, A100 validation outstanding.
- [ ] A100 utilization behavior empirically understood — measurement request
      written (`docs/design/phase6_measurement_request.md`), not yet run.
- [ ] batch sizing satisfies the utilization constraint at near-best throughput —
      designed (`docs/design/phase6_batch_sizer.md`), not built.
- [ ] **canonical performance and regression benchmarks exist** — the
      *specification* exists (`docs/design/benchmarks.md` + `benchmarks/`); the
      baseline numbers it is supposed to hold do not.
