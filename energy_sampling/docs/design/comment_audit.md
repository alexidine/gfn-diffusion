# Comment and docstring audit

Ranked work list for §1.3 / the "comments and docstrings follow current-state
discipline" completion box of `docs/design/infrastructure_stabilization.md`.
Read-only audit: nothing below has been changed.

**Scope.** `train.py`, `controller.py`, `protocol.py`, `buffer.py`, `utils.py`,
`checkpointing.py`, `models/**`, `energies/**`, `eval/**`, `configs/mk_dev.yaml`.
`bench/old` and `.claude/worktrees` excluded.

**Method.** Every docstring and every comment run of ≥3 lines was extracted and
ranked by stale-narrative signal; the candidates were then read against the code
they sit on. Every claim recorded below was checked against the current source —
line references in the *Evidence* column are the check, not a citation of the
claim. The suite collects 449 tests and was used only as a sanity check; the
audit itself is a reading of the code, which is the sharper instrument here.

**Line numbers drift.** The tree moved while this audit was being written:
`configs/mode_presets.yaml` was deleted and `configs/problems.yaml` created (1.4),
and `train.py` grew ~100 lines. Line references below were correct when filed and
were re-verified at the moment each S1 fix was applied; anything still open should
be re-located by its quoted text, not by its line number. Two S2 findings are
partly overtaken already — every `mode_presets.yaml` reference in scope has been
repointed, which removes a dangling pointer the audit had not yet filed.

**Ranking.** By harm, not by verbosity. A comment that is *wrong about current
behaviour* costs the next reader a bad edit or a failed run; a run-id narrative
only costs them scrolling. The tiers are:

| Tier | Class | Count | Status |
|---|---|---:|---|
| **S1** | Contradicted by the code it sits on — acting on it produces the wrong change | 10 | **APPLIED** |
| **S2** | Names something retired or deleted as if live — the reader cannot resolve it | 7 | **APPLIED** |
| **S3** | Pinned numbers presented as current fact | 6 | **APPLIED** |
| **S4** | Experimental history / run-id narrative | 11 | **APPLIED**, except `controller.py` |
| **S5** | Housekeeping — dead files, `# todo` markers, dormant blocks | 3 | **APPLIED** |

**S2–S5 applied 2026-08-17.** Three judgment calls worth recording, because each
is a case where the tier's rule said "trim" and the right answer was not to:

- **`utils.py`'s `_SERVO_SEED_LR` citations KEPT.** The claim is that the seed
  sits below *every measured optimum*, and that is only checkable if the
  measurements are named. Trimming the run names would leave a bare assertion —
  the same test the audit applies to `controller.py`'s alpha_target calibration,
  which it also marks keep.
- **`_GRAD_MEDIAN` given a scope and a re-measure trigger rather than a trim.**
  Three constants set the gradient clip for every `auto` config, and they were
  measured on one molecule with no route caveat. The fix a pinned-number finding
  usually wants — delete the provenance — would have made it worse.
- **`controller.py` left alone**, as the user was editing it. Its S4 entries (the
  "WHAT V8 DELETED" autopsy and the tuphwfkm calibration) are theirs to keep or
  move to `docs/lr_control_summary.md`.

One correctness question was SHARPENED, not resolved: `calibrate_prior_noise`
carried `# todo confirm right latents / dists`, which named no checkable thing.
It now states the two: the magnitude sweep is a deterministic ramp under a name
that says otherwise, and the unit-box clip truncates the large end so realised
displacement falls below nominal exactly where the sweep is widest.

**S1 was applied on 2026-08-16**, ahead of 1.2 rather than after it: a comment
that is wrong about behaviour does its damage during consolidation, which is the
operation most likely to trust it. Every S1 edit is comment/docstring text only —
no config value and no code path was touched, confirmed by diff (`lr_warmup_ratio:
10` and `symmetric: true` are byte-identical across the change). 79 config,
mode-safety and snapshot tests pass. The S4 trim stays deferred: 1.2 will rewrite
that prose anyway.

The S1 group is the whole priority. It is the same class as the
`increment_batch_size` docstring already found and fixed — and **finding #1 below
is that same defect, uncorrected, in the canonical config**.

---

## S1 — Contradicted by the code it sits on — **APPLIED**

Highest harm. Each of these told a reader the opposite of what the code does, or
omitted the branch the canonical config actually takes. All ten are fixed; the
table is kept as the record of what was wrong and why, since the next reader of
these comments has no other way to know they were rewritten.

Two were widened on application:

- **#5** also reached `buffer.py:946` and `:952-966`, whose `prioritised_weights`
  docstring presented the one-sided draw as the operating mode and the
  `is_elig_frac` ratchet as a "plausible mechanism". The canonical config ships
  `symmetric: true`, and the ratchet is settled — `is_elig_frac` holds at 1.0.
  Both now say so, and both flag that the forward-tail question is still open.
- **#1** picked up a closing clause noting the whole growth walk is gated on
  `grow_batch_size: false` and does not run as shipped (was S5).

| Site | Claim | Why it is wrong | Evidence |
|---|---|---|---|
| `configs/mk_dev.yaml:170` | "The growth walk scores every jump by how much it **slowed the STEP down**, not by raw samples/sec" | Exactly inverted. The shipped gate is throughput saturation: a jump is kept iff it buys `batch_growth_min_throughput_gain` more **samples/sec**, and step time does not enter at all. This is the retired `batch_growth_max_step_regression` rule described as if live — and it contradicts its own file six lines below. | Gate at `train.py:698-718` ("This gate is therefore a SATURATION DETECTOR"); `train.py:455-460`; the rule is retired at `utils.py:170-178`; `configs/mk_dev.yaml:176` states the correct rule |
| `controller.py:6-7`, `controller.py:12-13` | "a peak set by PERIODIC RAY CALIBRATION, plus one coarse divergence bar. **Nothing else.**" / "`peak_scale` is moved **only** by `on_calibration()`" | Three sensors move `peak_scale`, not one. Worse for the canonical config: `mk_dev`'s first stage runs `lr_sensor: {kind: hyper}`, so the sensor the class docstring says is the only one is *not* the one driving `train_prior`. | `on_calibration` `controller.py:205`, `on_plateau` `:229`, `on_hypergradient` `:292`; all three wired at `train.py:2535`, `:4760`, `:3094`; kinds `protocol.py:121`; stage sensors `configs/mk_dev.yaml:341` (`hyper`) and `:377` (`ray`) |
| `configs/mk_dev.yaml:37` | "`auto` on an lr_* key = SERVO-MANAGED: **seeded at 1e-5**" | The file seeds at `1.25e-4`, 12.5x higher, nineteen lines below. `1e-5` is only the fallback used when `adaptive_lr.seed_lr` is absent, and this file sets it. | `configs/mk_dev.yaml:56` (`seed_lr: 1.25e-4`); resolution order `utils.py:487-488` — config value wins, `_SERVO_SEED_LR` is the `or` branch |
| `configs/mk_dev.yaml:49` | `lr_warmup_ratio: 10` — "10 rather than 100 because the servo seed already sits **~10x under any measured optimum**, and **1e-5/100 is below min_lr**" | Both halves of the justification fail at the shipped seed. `1.25e-4/100 = 1.25e-6`, which is *above* `min_lr: 1e-6`, so the ramp would not be clipped flat. And `1.25e-4` is not 10x under a measured optimum — it *is* the `local_aug08` T=10 optimum this codebase records. | `min_lr` `configs/mk_dev.yaml:48`; the measured optima the claim leans on are listed at `utils.py:142-147` (`local_aug08 ~1.25e-4 at T=10`) |
| `configs/mk_dev.yaml:485` | `symmetric: true` — "untested alternative, **default off preserves current behavior**" | The comment describes the *code* default while the canonical config ships the flag ON. A reader of the master config concludes the one-sided `delta_plus` draw is live; the symmetric `\|delta\|` draw is. The two have different eligible sets and therefore different estimator targets. | Code default `buffer.py:921`, `train.py:3825`; the branch `buffer.py:1015`; the docstring that assumes one-sided operation `buffer.py:946`, `:952` |
| `protocol.py:38-52` | Balance rules documented as three kinds — `lexicographic`, `proportional`, `constraint` | `ratio` is a fourth kind and is the **only one the canonical config uses**. The module docstring does not mention it. | `elif kind == 'ratio'` `protocol.py:695`, `_ratio_tick` `:1661`; `configs/mk_dev.yaml:393` (`kind: ratio`) |
| `protocol.py:28-29` | `on_exit/on_enter` actions listed as "snapshot:`<tag>`, snapshot_prior, bootstrap_z, seed_prior_from_anchors" | Two of the six actions are missing, and one of the missing ones is used by the canonical config's only `on_enter`. | Full tuple `protocol.py:115-116` adds `reseed_prior_from_dataset` and `rebuild_prior_by_churn`; `configs/mk_dev.yaml:381` uses `rebuild_prior_by_churn` |
| `protocol.py:13-16` | Stage `flags` listed as five: `update_log_z, scramble_conditions, weighted_condition_sampling, buffers_active, mle_gate` | Six exist. `weighted_bwd_sampling` is omitted, and the canonical config's prior-buffer commentary refers to "stages with `weighted_bwd_sampling`" — a flag the module docstring says is not a flag. | `STAGE_FLAGS` `protocol.py:113-114`; `configs/mk_dev.yaml:446` |
| `controller.py:417-419` | `step()` — "otherwise **only** `on_calibration`, `on_plateau` and `on_divergence`" move `peak_scale` | Omits `on_hypergradient`, which moves `peak_scale` every step on any `kind: hyper` stage — i.e. throughout `train_prior` on the canonical config. Same defect as the class docstring, in the method that documents the invariant. | `controller.py:292`; caller `train.py:3094` |
| `controller.py:347-351` | `rearm_warmup()` — resetting `peak_scale` is safe because "the **plateau rule only ever cuts**, so there is nothing to re-climb" | The reasoning is stale even though the behaviour is right. Two of the three sensors climb: ray calibration raises by `eta_up` when `alpha_hat > alpha_target`, and the hypergradient raises whenever `cos > 0`. A stage reset does discard climb, so the stated justification no longer supports the decision it defends. | `controller.py:197-199` (`eta_up` branch), `:292` (`exp(beta*cos)`) |

---

## S2 — Retired or deleted names presented as live

The reader cannot resolve these. #11 is the one that costs a run: it names a
config key that hard-fails preflight.

| Site | Claim | Why it is wrong | Evidence |
|---|---|---|---|
| `controller.py:132` | "train.py's **`max_reloads`** cap is what stops a rewind loop" | `max_reloads` is a retired key. A reader who follows this comment into a config gets a hard `preflight_config` failure. The live key is `max_reloads_per_1k_steps`, and it is a *rate*, not a count — the rename carried a semantic change this comment erases. | Retirement record `utils.py:259-262`; live read `train.py:2347`; config `configs/mk_dev.yaml:108` |
| `protocol.py:4-5` | The stage engine "replac[ed] the tangled phase (1/2/3) + forward-first (A/B) frameworks (`phases.PhaseController`, `controller.ModeBalanceController`, `controller.ForwardFirstController`)" | None of the three classes exists anywhere in the tree, and there is no `phases.py`. `controller.py` today contains exactly one class, `LRController`, so a reader sent to `controller.ModeBalanceController` lands on an unrelated module. | `grep` over the tree returns only `protocol.py:4,5,41,1357`; `controller.py:4` is the sole class; no `phases.py` |
| `protocol.py:41` | "the same EMA-toward-target nudge `ModeBalanceController` and `ForwardFirstController` both used" | Same dangling classes — here used as the *definition* of the nudge, so the behaviour has no readable specification. | as above |
| `protocol.py:1357` | `_lookahead` is a "verbatim port from `ModeBalanceController._log_ema_lookahead`" | Same. "Verbatim port from X" is unfalsifiable once X is gone. | as above |
| `utils.py:1657`, `utils.py:1685` | "**the phase-3 controller** keys backward allocation on this" (`relative_under` / `relative_under_wcen`) | `phase` is now just `protocol.stage.index + 1`. The canonical config declares two stages, so phase 3 does not exist on it. The actual consumer is the `kind: ratio` balance loop on `equilibration`, which does read `bwd/relative_under_wcen`. | `train.py:286-290`; stages `configs/mk_dev.yaml:322`, `:369`; consumer `configs/mk_dev.yaml:395` |
| `train.py` — phase 1/2/3 vocabulary, ~20 sites | Comments throughout speak of "phase 1", "phase 2", "phase 3" as a fixed framework | Same root cause as above: `phase` is a 1-based index into whatever stage list a config declares, so these names denote different stages in different configs and denote nothing at all past the end of the list. Worth one sweep rather than 20 separate edits. | `train.py:1088, 1153, 1409, 1426, 1819, 2567, 2981, 2992, 3102, 3644, 3647, 3747, 3750, 4071, 6209, 6221, 6392, 6483, 6591, 6724-6725`; also `protocol.py:49, 859, 985` |
| ~~`train.py:6076-6091`~~ | Three keys were retired *in prose only*: `max_residence_steps`, `toxic_min_draws`, `toxic_delta_threshold` | They were absent from `utils._RETIRED_KEYS`, so they raised at **first use inside `manage_replay_buffer`** rather than at load — precisely the failure `utils.py` records above that dict as having cost the aug02 battery all 16 arms' phase 1. **FIXED** — see below. | `_RETIRED_KEYS` `utils.py`; `config_state.CHANGES` state 3 |

---

### S2 #17, resolved — the three replay keys now fail at load

Applied as **project state 3** (`config_state.CHANGES`), not as a comment edit,
because the honest fix was mechanical:

- `utils._RETIRED_KEYS` gains all three keys, each with its reason. Rejection now
  happens in `preflight_config`, before a single energy call.
- A `Transition` gives them a repair path. This is the part that was actually
  missing: **106 tracked configs carry `max_residence_steps` and 103 carry
  `toxic_min_draws`** (counted over git-tracked yaml — a naive grep also picks up
  git-ignored wandb artifacts and overcounts), and until now none had a
  mechanical route forward, because the keys were rejected by code no migration
  knew about.
- **Only two of the three are dropped mechanically.** `max_residence_steps` is
  `manual`: it was REPLACED by `mean_residence_steps`, and the overlap between
  the two across tracked configs is **zero**. Dropping it would leave no residence
  setting at all, `tau` reads 0, and the `if tau > 0` branch that arms both the
  hazard and the age backstop never runs — a config that loads clean, trains, and
  still reports a healthy `replay_buffer_age_cv` ≈ 1 because the surviving
  displacement purge is itself memoryless. `manual` is also the only category
  that blocks `migrate --write`, which is the affordance this needs.
- The two runtime guards in `manage_replay_buffer` are **deleted** rather than
  kept as defence in depth. Every load path runs `preflight_config`, and holding
  the same reason text in two places is how the two come to disagree.
- Three comments that named these keys as live were reworded
  (`train.py` age-CV metric, the TTL race note, and `manage_replay_buffer`'s own
  docstring, which pointed at "the toxic_min_draws ValueError below").

**The gate is stricter than the guard it replaced**, which is intended rather than
incidental. The old guard tested truthiness (`toxic_min_draws`) or non-None
(`max_residence_steps`), so `toxic_min_draws: 0` passed it and ran with the key
silently ignored. The gate fires on PRESENCE. Verified end-to-end on a real
`configs/aug02` config: load-time rejection, the present-but-falsy widening, and
`migrate` dropping all three.

---

## S3 — Pinned numbers presented as current fact

These rot into the next stale claim. #18 is already arithmetically wrong.

| Site | Claim | Why it is wrong | Evidence |
|---|---|---|---|
| `configs/mk_dev.yaml:474` | `max_size: 12000` — "**1.5x** the 80 x 50 equilibrium (was 4000 = exactly it, zero headroom)" | The arithmetic does not hold: `80 × 50 = 4000`, and `12000` is **3x** that, not 1.5x. The file confirms the equilibrium is 4000 fifty lines earlier. Either the value moved and the comment did not, or the comment was wrong when written. | Little's law stated `configs/mk_dev.yaml:465-470`; `churn_rate: 80` `:472`, `mean_residence_steps: 50` `:473`; "occupancy still 4000" `:418` |
| `buffer.py:976-990` | "MEASURED 2026-08-07 against r2_wiring's live buffer" + ESS table + "the shipped 0.01 was far too permissive" | The table is a designed calibration whose result *does* define the present `floor_frac: 0.25`, so §4 permits it. The diary framing around it does not: "the shipped 0.01" names a value that has not been shipped since, and the dated attribution invites the reader to treat a nine-day-old measurement as current. Trim to the knee and the invariant; drop the date and the superseded value. Also carries an author `# todo shorten comment`. | `configs/mk_dev.yaml:484` ships `0.25`; `# todo` `buffer.py:976` |
| `utils.py:151` | `_GRAD_MEDIAN = {10: 1.0e3, 25: 6.6e3, 100: 1.7e4}` — "empirical pre-clip grad medians (mipcas)" | Three pinned constants that silently set the gradient clip bar for every `auto` config, attributed to one molecule with no route caveat and no re-measure trigger. The derivation formula itself is correct and matches its config comment — it is the table underneath that is a measurement frozen into a law. | Formula `utils.py:516-521` matches `configs/mk_dev.yaml:98` exactly (`250 * grad_median(T)/6.6e3 * sqrt(W/512)`); anchors `utils.py:140-150` |
| `train.py:443` | "rpvez6ep: batch 50k ran the same 13.3k samples/s as batch 1.6k at 31x the step time" | A single run's numbers embedded in an otherwise-current docstring. The *claim* it supports (throughput saturates past the knee) is durable; the run id and the three figures are not. | `train.py:440-453` |
| `train.py:496-505` | The `umaperf0812 c_controller` batch/util/samples table | Five-column measurement table inside a block comment justifying the absence of an occupancy rule. The conclusion is durable and belongs; the table is a log. Note the same measurement is already recorded once, in `utils.py:158-169`, so this is a duplicate copy that can drift from it. | `train.py:488-510`; duplicate at `utils.py:158-169` |
| `configs/mk_dev.yaml:31` | `archive_period` — "~910MB of disk per archive" | Sized against one model/buffer configuration; moves with `s_emb_dim`, buffer `max_size` and `archive_buffers`, none of which are pinned. | `configs/mk_dev.yaml:32`, `:112-128`, `:449`, `:474` |

---

## S4 — Experimental history and run-id narrative

Lowest priority, per the brief. Recorded so the §1.3 rewrite has a checklist; all
of it is in git. Several carry durable rationale that should survive the trim —
noted where so.

| Site | Narrative | Note |
|---|---|---|
| `controller.py:15-25` | "WHAT V8 DELETED, AND WHY" — v7's parabola/censoring/quorum machinery, `sd/mean ~3.3`, "a 55x LR ramp ... (lrdisc v1, 2026-08-10)" | Keep the one-line reason the ratio estimator was abandoned; the autopsy belongs in `docs/lr_control_summary.md` |
| `controller.py:34-38` | `tuphwfkm`: "stable, improving training sat at alpha* 3.6-5.0, and an excursion to ~2.8 degraded it" | **Keep** — a deliberately designed calibration whose result defines `alpha_target: 4.0`. Duplicated verbatim at `configs/mk_dev.yaml:62-66`; keep one |
| `controller.py:442-448` | `qm9anchor_aug14` — "detonated five of six arms from a healthy state" | Keep the invariant ("peak_scale at the floor means the floor is too high"), drop the arm count |
| `buffer.py:952-966` | "is_elig_frac drift F-003 measured (0.74 -> 0.33 over 1500 steps)" | The absorbing-starvation *mechanism* is durable; the measurement is a log |
| `buffer.py:310` | "That killed the kappa=0 arm at step 119 on 2026-08-07" | |
| `utils.py:1667` | `cazwlyy1`: pooled `jensen_z` vs per-condition `log_Z_learned` | Mechanism is durable; run id is not |
| `utils.py:402-406` | `stab_july21` elj battery ran `eval_T = 2T` and floored `wass_debiased` on a dt artifact | Inside `preflight_config`. **Keep the invariant** (`eval_T` must equal `T`) — it is enforced code at `utils.py:407-414`; drop the battery |
| `utils.py:134`, `:143`, `:154`, `:162`, `:184`, `:198`, `:597` | `aug02`, `ty4xdlzo`, `local_aug08`, `umaperf0812`, `prod0810`, `lrdisc` | Most sit inside `_RETIRED_KEYS`, where a "why it was deleted" record is the point of the structure — treat that dict as exempt |
| `train.py:332`, `:470`, `:600-613`, `:2167`, `:3969` | `prod0810` — "Three further guards, each from an observed prod0810 failure"; the acridine/mace OOM story; "All three died" | The three guards are current behaviour and must stay; the failure stories are the trim |
| `protocol.py:47` (`b9ze0p5c`), `:76-79` (`s706frkh`), `:343`, `:1226`, `:1693` | run ids in rule/floor rationale | `s706frkh` (bwd dark for ~1700 steps) is the reason floors are explicit per stage — keep the rule, drop the id |
| `configs/mk_dev.yaml:65`, `:294`, `:447`, `:541`, `:546`; `checkpointing.py:163`; `models/gfn.py:311`; `energies/molecular_crystal.py:83`; `energies/base_set.py:41` | `tuphwfkm`, "retired to hygiene duty after the 9x mean-pinned-Z falsification", `gejezmjg`, `prod0810`, "the old soft-start + terminal anneal was removed" | §1.3 names the config ones explicitly |

---

## S5 — Housekeeping

| Site | Issue | Note |
|---|---|---|
| `eval/offline_evaluation.py` | **418 of 419 lines are commented-out code** under a bare `# DEPRECATED` header | Not a comment-discipline problem — a dead file. Delete it rather than audit it; git holds it |
| 14 `# todo` markers | `configs/mk_dev.yaml:60`; `buffer.py:976`, `:1018`, `:2715`; `utils.py:1111`, `:1364`; `train.py:4019`; `models/gfn.py:18`; `energies/molecular_crystal.py:761`; `eval/evaluations.py:242`, `:655`, `:803`, `:804`, `:866` | §1.2 asks for these to be resolved. Three are self-directed comment-length complaints (`mk_dev:60`, `buffer.py:976`, `:1018`) and fall out of the S1/S3 rewrites above. `utils.py:1111` ("confirm right latents / dists") is the only one that reads as an unresolved correctness question |
| `configs/mk_dev.yaml:161-184` | `grow_batch_size: false` makes the whole 22-line batch-growth commentary dormant on the canonical config, with nothing saying so | Not wrong, but it is the largest block of prose in the file describing machinery that never runs as shipped. One line noting the master switch would fix it. Interacts with S1 #1 — fix that first |

---

## Checked and found accurate

Recorded so the next pass does not re-verify them.

- `configs/mk_dev.yaml:98` — the `gradient_norm_clip: auto` formula matches
  `utils.py:516-521` exactly, including the `6.6e3` denominator being
  `_GRAD_MEDIAN[_SCALING_T_REF]`.
- `configs/mk_dev.yaml:19` — `continue_from_checkpoint` is genuinely inert while
  `checkpoint_name` is set: `train.py:1679` takes the `if` branch, `:1691` is the
  `elif`.
- `train.py:1713-1742` — the `maybe_compile_policy` docstring is current on every
  point checked (`auto` = Linux+CUDA, default mode not `reduce-overhead`,
  conditioner not compiled, recompile-per-shape).
- `train.py:199-218` (`_ray_probe_armed`) and `train.py:5962-6007`
  (`manage_replay_buffer`) — both accurate against the code, including the
  admission/purge/draw split the config comments echo.
- `configs/mk_dev.yaml:440-445` — the prior-buffer residence arithmetic checks
  out: `100 × 250000 / 1000 = 25,000` steps, and `25000 / 50 = 500x` separation
  from the replay buffer.
- `configs/mk_dev.yaml:412-413` — `bar: 0.368` and `release: 0.60` are the stated
  `1/e` and `exp(-0.5)`.
- `configs/mk_dev.yaml:176` — `0.05 < batch_growth_factor - 1 = 0.65` holds, and
  the invariant is asserted in `config_invariants.growth_gain_below_growth_factor`.
- `configs/mk_dev.yaml:190` — `figs_period: 500` is a multiple of
  `eval_period: 250`.
- Every `docs/*.md` path referenced from a comment in scope resolves to a file
  that exists.

---

## Suggested order of work

1. **S1 #1** (`mk_dev:170`) — same defect class as the `increment_batch_size`
   docstring already fixed, and it is in the canonical config. One paragraph.
2. **S1 #2, #8, #9** (`controller.py`) — one pass over the class docstring,
   `step()` and `rearm_warmup()`; they are three views of the same stale
   single-sensor assumption.
3. **S1 #3, #4, #5** (`mk_dev` LR seed and `symmetric`) — three self-contradictions
   inside the canonical config, each fixable in a line.
4. **S1 #6, #7, #10** (`protocol.py` module docstring) — three incomplete
   enumerations, all three of which omit exactly what `mk_dev` uses. One edit.
5. **S2 #11** (`controller.py:132`) — one retired key name, one word.
6. **S2 #12-#15** — the dangling-class and phase-vocabulary sweep. Mechanical but
   broad; do it as one commit.
7. ~~**S2 #17** — move the three replay keys into `_RETIRED_KEYS`.~~ **DONE** —
   shipped as project state 3; see the resolution note in the S2 section. (Filed
   as #16 here originally; it is the seventh S2 row, #17.)
8. **S3** — the arithmetic error at `mk_dev:474` first, then the pinned tables.
9. **S5** — delete `eval/offline_evaluation.py`, resolve the `# todo`s.
10. **S4** — the §1.3 narrative trim, last.
