# Spec: `analysis/` — run-reading toolkit

Forward-looking build spec, for a downstream agent. Delete this file once
executed; the resulting package documents itself and its State belongs in a
`module_analysis.md`.

---

> **STATUS: Tiers 0, 1 and 2 are BUILT** (`analysis/`, 351 tests, State in
> `docs/module_analysis.md`). Tier 3 remains, and this file stays until it
> is done. Open questions 1–3 are answered: the VarGrad topline carries all four
> candidate families; `GFN Energy` is the priority with `GFN Conformers` riding
> along where easy; the cache lives in the system temp dir.
>
> **Three of this spec's assumptions were wrong**, each failing silently — they
> are corrected in `module_analysis.md`:
>
> 1. Route detection cannot read `protocol` from the config. wandb stores it as a
>    repr **string**; the usable form is the flattened `protocol_stages_N_*` keys.
> 2. The stage metric is `phase`, not `protocol/stage_index`, and it is
>    **one-based**.
> 3. Classification must filter by the stage's `train_mode`. Counting all three
>    modes reads the canonical base `replay_loss_coeffs_tb: 1.0` as live during
>    `train_prior` and classifies the MLE warm-start as the TB route.
>
> The instruction to verify by inspection rather than trust the prose is what
> caught all three. **Four more were caught the same way in Tier 1**, and the
> pattern held — each was a silent wrong answer, not an error:
>
> 4. `loss_coeffs/*` — the coefficients the trainer is actually holding, and the
>    strongest liveness evidence in a run — is a **change-only** channel. Its
>    history series is one or two points and the datastore reader drops anything
>    shorter than three, so it must be read from the summary.
> 5. `detect_route` defaults an unknown stage to the **last declared** one.
>    Reading a run that way asserts a route it never reached, and since NA_ROUTE
>    marking is driven entirely by the route, it switches NA_ROUTE **off**.
> 6. `protocol/thr_*` is published by the lexicographic balance controller, not
>    by the stage exit block. R13 derived from it alone is dark on nearly every
>    run while still returning a full sensor table.
> 7. `pull` raises only on empty **history**, so a run arrives fully parsed with
>    `config == {}` — and every cross-arm comparison then reads `<missing>` on
>    one side and flags confidently about an arm nothing is known about.
>
> Tier 1 was also adversarially verified against the ~110-run local corpus rather
> than against its fixtures alone, which is what surfaced 4, 6 and 7.

## Why this exists

The feature-extraction script for wandb runs (`wa.py`) has been written **six
times**, in six session scratchpads, 2.4 KB–8.7 KB, 2026-07-29 → 2026-08-13. The
copies diverge and none is in the repo, so every session re-derives it and loses
the previous session's fixes. It is also local-only and cannot see a cluster run.

Goal: one versioned package, so the tool accumulates instead of resetting.

## What you need to know

- This project trains GFlowNet samplers; runs log to wandb projects
  `mkilgour/GFN Energy` and `mkilgour/GFN Conformers`.
- **`docs/reading_runs.md` is the requirements document.** The package exists to
  serve that method. `R*` references below are its principles.
- **You do not need to understand the training code.** This package reads wandb
  output. Read `train.py` only to resolve a metric's definition, never to
  reimplement one.
- Use the project venv: `C:\Users\mikem\venvs\csd_mxt_gfn\Scripts\python.exe`.
  System python has no wandb.

## Non-goals

- Not a replacement for the wandb web UI. The user reads figures there.
- **Does not emit verdicts.** `reading_runs.md` §3 splits principles into
  mechanical and judgment. Implement the mechanical ones as checks; surface the
  inputs for the rest and stop. A tool that concludes "the run is healthy" is a
  failure of this spec.
- Does not touch `train.py`, write checkpoints, or run training.

## Layout

```
analysis/
  keys.py      route + stage detection, key resolution      <- the critical file
  pull.py      resolve run(s), fetch history, disk cache
  features.py  trend / oscillation / escape extraction      <- rescue wa.py
  checks.py    R2 / R14 / §4 assertions
  compare.py   multi-run arm table
  cli.py       entry point
  tests/
```

---

## Hard requirements

These are measured traps, not style preferences. Each has already produced a
wrong or empty read.

**H1 — `scan_history(keys=[...])` returns zero rows SILENTLY if any single
requested key is absent.** No error, no warning. Measured: requesting seven keys
of which two were absent returned 0 rows in 0.4 s and looked like a run with no
data. Therefore: resolve every key against `run.summary` first, request only
resolved keys, and assert the result is non-empty. An empty pull must raise.
This is an `R14` failure inside the tooling and it is check number one.

**H2 — key names are route- and stage-dependent, and so is their MEANING.**
Do not hardcode a topline list. Two failure modes, and the second is worse:

- *Absent*: `bwd/under_coverage_wcen` does not exist on a `var_conditioning`
  run (it is `bwd/under_coverage`; the `_wcen` form appears in the `naive`
  stage's balance config). `log_Z_learned` is namespaced `bwd/log_Z_learned`,
  not bare.
- *Present but not meaningful*: **on the conditional VarGrad route the log Z and
  TB reporting does not track** — the keys exist and carry numbers, and reading
  them as you would on a TB run is wrong. This is the user's explicit statement
  and is the single most important thing in this spec.

So each metric resolves to one of three states, and the report must distinguish
all three: **live** / **absent** / **not-meaningful-on-this-route**. Never
collapse the third into the second, and never render it as zero.

**H3 — local run ordering.** Directory mtime *and* `.wandb` file mtime are both
unusable: the wandb sync service sweeps old runs, touching and even growing their
files. Order by the launch timestamp in the directory NAME
(`run-YYYYMMDD_HHMMSS-id` sorts chronologically). Ghost filter: a stub is small
**and** old; a freshly launched run is small but active, so exempt recent writes
or the filter races launches.

**H4 — local binary parse.** `item.key` is empty in this wandb version — use
`item.key or '.'.join(item.nested_key)`. Not every history row carries `_step`;
carry the last seen value forward. Wrap `scan_data()`/`ParseFromString` in
try/except and `break` — a live run has a partial final record. A just-restarted
run's `.wandb` can be 0 bytes (header race).

**H5 — EMA-derived series.** `tracker/*` are EMA outputs. Show the trend;
suppress significance testing (no trend test is valid on a smoothed series).
`tracker/logw_std_rms` and `tracker/z_bias_rms` are additionally flagged
low-trust in `reading_runs.md` §2 — carry them, do not rank on them.

**H6 — count catastrophes, never average them.** See `bench/metrics.py`'s
docstring, which also explains why no metric here may depend on a reference rate.
Reuse that reasoning; the five metrics there are a good model.

---

## Tier 0 — stop rebuilding it

The minimum that ends the six-rewrites problem. **Build this first and stop for
review.**

### `keys.py`
- `detect_route(config) -> Route`. Distinguish at minimum: unconditional TB,
  conditional VarGrad, MLE/prior. Derive from `protocol.stages[*].loss_coeffs`
  (`vg_lb`/`vg_by_condition` ⇒ VarGrad leg; `tb` ⇒ TB leg; `mle` ⇒ prior leg)
  and the conditional flag. **Verify by inspection against a live run; do not
  trust this sentence.**
- `current_stage(run) -> str`. Determine the stage key by inspecting a live
  run's history and assert it exists — do not assume a name.
- `resolve(run, wanted) -> {name: LIVE|ABSENT|NA_ROUTE}`. Fuzzy-match against
  `run.summary` keys (there are ~490) so a renamed metric is reported as a
  rename, not a hole.
- `TOPLINE[route]`. Seed the TB/unconditional set from `reading_runs.md` §2.
  **The VarGrad topline is an open item — see Open questions.**

### `pull.py`
- One entry point covering both sources: cloud (`wandb.Api`, works for cluster
  runs — they sync) and local (`.wandb` datastore, per H3/H4).
- Resolve by name, id, tag, or "newest". Accept a list.
- **Disk cache** keyed by `(run_id, last_step)`, so a completed run is pulled
  once. Measured cost: full history for a 14.3k-step run is 1,435 rows in 2.1 s;
  keyed pull 912 rows in 0.8 s; listing 20 runs 1.2 s. Cheap enough that a full
  unkeyed pull is the sane default, and caching is for repeat analysis, not
  latency.
- Pull `config` and `summary` alongside history; both are needed for §4.

### `features.py`
Rescue from `wa.py` — the newest good copy is
`AppData/Local/Temp/claude/C--Users-mikem-Projects-mxt-gfn-gfn-diffusion-energy-sampling/a4e6dc28-007b-4136-b8b4-5a876dd7095d/scratchpad/wa.py`
(8.7 KB). **Read it before writing anything**; it is mostly correct and its
comments record real traps. Port, do not reinvent:

- Theil–Sen slope with significance vs. detrended noise (`R12`)
- ACF oscillation: dominant period, rms amplitude, growing/steady/damping
  (`R8`)
- exponential-escape flag via trailing log-slope doubling time
- the EMA guard (H5)

Retarget its `GROUPS` taxonomy — it is stale (uses `fwd/r2`, `fwd/tb_err`,
`bwd/relative_under`) and must come from `keys.TOPLINE[route]` instead.

### `cli.py`
`python -m analysis <run-spec>` → resolved-key table (live/absent/NA-route),
then the feature report grouped in `reading_runs.md` §1 read order.

---

## Tier 1 — checks — **DONE**

`checks.py`. Each returns structured findings; each **fails loudly**, never
silently passes.

- **`R2` liveness.** For every mechanism configured active (nonzero frac, a
  threshold, a servo, a gate), assert a nonzero activation trace. Emit
  `mechanism → fired? → n_steps_active`. This is the highest-value check in the
  package: an inert mechanism voids an arm rather than answering it, and it has
  repeatedly made whole batteries meaningless.
- **`R14` dead sensor.** For each series used as a controller input: zero
  variance over a window, pinned at a clip or censoring bound for >X% of ticks,
  all-NaN, or a threshold annealed below the series' own measured noise floor
  (`R13`).
- **§4 confounds.** Code version stamp; `checkpoint_name` non-null across a
  comparison battery; `T == eval_T`; arms differing from a sibling only by
  omission; differing start conditions across arms.
- **`R11`.** `replay/scatter_err ÷ fwd/scatter_err` — healthy ≈ 2, flag < 1 as
  replay overfitting. TB route only; N/A on VarGrad.

## Tier 2 — compare — **DONE**

`compare.py`. Config diff across arms → the sweep table (which knobs actually
differ, decoded to arm names). Aligned feature table. Flag arms that are not
comparable per §4 before showing any metric. Worked example, verified
2026-08-15: nine `qm9a13` arms reduce to exactly two differing knobs —
`bwd_loss_coeffs.beta` (10 vs 80) and `protocol.stages[1].lr_sensor.beta`
(0.05 / 0.1 / 0.2) — and the three currently running arms have
`checkpoint_name: None` while the nine crashed ones carried an explicit
checkpoint, so the two batches are **not** comparable.

## Tier 3 — figures

Pull `files/media/plotly/*.json` and `files/media/images/*.png`. Lowest
priority: the user reads figures in the wandb UI and this only helps a headless
agent.

---

## Acceptance tests

`tests/`, network-free — cache one real run's history and config as a fixture.

**Every check must be mutation-tested: re-introduce the failure it detects and
require the test to FAIL.** A check that has never fired has not been tested.

Minimum:
1. Request an absent key → raises. (H1: must not return empty.)
2. Fixture with a frac pinned at 0 → `R2` fires.
3. Fixture with a series pinned at its clip → `R14` fires.
4. A VarGrad-route run → log Z / TB metrics report **NA_ROUTE**, not absent, not
   zero. (H2 — the one most likely to be got wrong.)
5. Ghost/0-byte local run dir → skipped, no crash. (H3/H4.)
6. A run whose `_step` is missing from some rows → parses without gaps. (H4.)

## Coupling to the rewrite

This package reads **wandb output, not code**, so it is insulated from the
rewrite almost everywhere. The one coupling point is `keys.py`, which tracks
metric names and their route semantics. Keep the taxonomy in that single file so
a rename in the rewrite is a one-file change. Do not scatter metric-name string
literals across the other modules — if you find yourself typing `'fwd/'`
anywhere outside `keys.py`, stop.

## Open questions — ask the user, do not guess

1. **The VarGrad topline set.** `reading_runs.md` §2 gives the TB/unconditional
   set. The conditional VarGrad route needs its own, and the user has stated log
   Z and TB reporting are *less meaningful* there. Candidates to put to him:
   `[fwd|bwd]/logw_std_within`, `vg_lb`, `zmatch/delta_worst`, the per-condition
   fractions, and the held-out `eval_test/*` series (`R17`). Get his call before
   coding `TOPLINE[VARGRAD]`.
2. Which wandb projects to support beyond `GFN Energy` (`GFN Conformers` is
   live).
3. Whether the cache lives in the repo, the scratchpad, or a user directory.
