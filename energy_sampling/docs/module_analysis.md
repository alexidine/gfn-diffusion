# module: analysis

> **Status: MODULE SNAPSHOT.** Use this document for explanation and navigation,
> not as proof of current behavior. Verify material claims against the current
> implementation and focused tests. See
> [`EPISTEMIC_PROTOCOL.md`](EPISTEMIC_PROTOCOL.md).

The run-reading toolkit. `docs/reading_runs.md` is its requirements document;
this file records what exists.

Tiers 0, 1 and 2. Tier 3 (figures) is specified in
`docs/analysis_package_spec.md` and not built.

## What it is for

One versioned package, so the feature-extraction tool accumulates instead of
resetting. It had been rewritten six times in six session scratchpads, diverging
each time, with none of it in the repo.

It reads wandb output, not code, so it is insulated from changes to the trainer
everywhere except `keys.py`.

## Layout

| File | Role |
|---|---|
| `keys.py` | route + stage detection, key resolution, the mechanism registry. **The only file that may contain a metric-name or config-key literal.** |
| `pull.py` | run resolution, local `.wandb` and cloud history, disk cache |
| `features.py` | trend / oscillation / runaway-growth extraction |
| `checks.py` | R2 / R14 / §4 / R11 assertions |
| `compare.py` | cross-arm sweep and aligned feature tables |
| `cli.py` | `python -m analysis <run-spec> [<run-spec> ...]` |

## It emits no verdicts

Mechanical principles are checks; everything else surfaces its inputs and stops.
The ported `wa.py` had a `bellwether_verdict` printing "healthy climb" and
"policy losing ground"; that did not come across, deliberately. Reading here is
context-dependent and jumps to whatever the phase and symptom implicate, and
encoding that jump produces confident wrong answers.

The checks hold the same line one level down. A check reports STATE and NUMBERS:
`FIRED` is not "working" and `INERT` is not "broken" — a frac pinned at its
declared value fires on every tick, and a servo whose actuator correctly never
left its rest point is inert. R11 names which band its ratio is in and stops,
because R11 is a mechanism and a mechanism does not survive being compressed
into a verdict.

## Three states, never two

A metric resolves to **LIVE**, **ABSENT**, or **NA_ROUTE** — logged, populated,
and not meaningful on this route. On the conditional VarGrad route the log Z and
TB series carry numbers that must not be read as they would be on a TB run.
NA_ROUTE is checked *before* presence (its defining property is that the key is
there), it is never requested from history, and it is named in the report where
it would have appeared rather than filtered out silently.

A fourth outcome is reported as ABSENT with a reason: **ambiguous**, when an
unnamespaced name matches several real keys. `log_Z_learned` matches three
(`fwd/`, `bwd/`, `replay/`), which are different quantities; the tool names all
three and picks none.

## Routes, and the stage they are read from

`TB_UNCONDITIONAL` · `VARGRAD_CONDITIONAL` · `MLE_PRIOR` · `UNKNOWN`, derived
from the effective loss coefficients of the stage the run is **currently in**.

Two rules the classification depends on:

- **Effective coefficients are base overlaid with the stage's overrides.** The
  base blocks are structural; a stage turns things on.
- **Only the stage's `train_mode` branches count.** `train_prior` is
  `train_mode: bwd`, and the canonical base `replay_loss_coeffs_tb` is 1.0 — so
  counting all three modes classifies the MLE warm-start as the TB route.

VarGrad is tested before TB, because a stage can carry both and the VarGrad leg
is what makes the TB reporting stop tracking. VarGrad with no conditioning flag
returns `UNKNOWN` rather than a guess.

**The route is never inferred from a stage the run did not reach.**
`keys.detect_route` defaults a None stage index to the last declared stage — right
for a caller asking about the terminal stage, wrong for reading a run. The single
resolution point is `checks.context()`, which returns `UNKNOWN` when the run's own
record does not say which stage it is in. The single-stage case is exempt, and
only that case. This is not cosmetic: NA_ROUTE marking is driven entirely by the
route, so a route inferred from the wrong stage silently switches NA_ROUTE **off**
and hands back TB numbers on a VarGrad run. `cli.py` reads the same `context()`,
so the feature report and the check blocks cannot state different routes for one
run.

## Toplines

TB/unconditional is `reading_runs.md` §2 verbatim, including the two entries that
do not resolve on real runs — they surface as a rename and an ambiguity rather
than being silently corrected here, so the discrepancy with the doc stays visible.

The VarGrad topline carries four families: the dispersion the objective
minimises (`logw_std_within`, `vg_lb`), the worst-case per-condition Z mismatch
(`zmatch/delta_worst`), per-condition fractions in Cond × Spread form only, and
the held-out `eval_test/*` series. Per-parameter subtleties within these are
deliberately not encoded — they are judgment.

## The checks

Each returns a `CheckResult` carrying every subject examined and the numbers
behind each. `findings` is the subset a reader must look at; `rows` is
everything, because a report showing only findings cannot be distinguished from
one produced by a check that never ran.

**A check that cannot run says so.** `CheckResult.not_run(reason)` renders
before any finding, under a heading that states it is not a pass. An empty result
would render identically to "looked, found nothing wrong", and a swallowed
diagnostic does not fail as silence — it fails as reassurance.

**NA_ROUTE rows always render, and are counted separately in the heading.** They
are not findings — nothing is wrong — but they are not silence either. Withheld
at default verbosity, a conditional VarGrad run's R11 rendered *byte-identically*
to a clean TB run: the collapse H2 forbids, arrived at through the renderer
rather than through the state.

Every block names its run and states `route`, `stage`, `window` and `last_step`.
Without those a battery renders one indistinguishable block per arm, and a reader
cannot audit a withheld metric without knowing which route's rules were applied.
**Runs are named by something that means something.** `reading_runs.md` §7:
refer to a run by NAME, TAG, or A DISTINGUISHING CONFIG FEATURE — never by
wandb id, which is a hash that carries nothing and makes the reader look every
arm up. Display names collide, though — nine are shared by two or more runs in
the local corpus, `mk_dev` alone by eleven, and two arms of a real cluster
battery are both `prod0810_mipcas_elj` — so `battery_labels` breaks a tie with
**the config knob that actually differs**: `prod0810_mipcas_elj[alpha_target=6]`
against `[alpha_target=4]`. That is the thing the reader wanted anyway, which is
why §7 lists it as an alternative to the name rather than as a fallback.
Candidates are ranked on how readable the VALUE is, not the key: wandb stores
each config section a second time as a repr string, and those keys are the
shortest while their values are the longest. The id appears only when nothing
else separates two arms, where it is the honest answer.

### R2 — confirm the thing ever fired

The highest-value check. An inert mechanism voids an arm rather than answering
it. Emits `mechanism → fired? → n_steps_active` over three families of subject:

1. **`keys.MECHANISMS`** — a declaring config key and the trace that proves the
   mechanism ran.
2. **The loss coefficients the trainer is actually holding** (`loss_coeffs/*`)
   against the config's effective values for the current stage. The strongest
   liveness evidence in a run: it is what the optimiser saw, after the stage's
   overrides. It is a change-only channel, so it is read from the **summary** —
   its history series is one or two points and the datastore reader drops
   anything shorter than three.
3. **The current stage's exit conditions** — a gate cannot trip if its streak
   never reached 1, and cannot trip at all if the metric it is defined on is not
   logged.

`UNDECLARED_ACTIVE` requires the declaring key to be **present and falsy**. An
absent key makes no claim — stage-scoped keys live on the stage that uses them,
and config eras differ — and treating absent as "off" manufactured the state on
most of the corpus.

### R14 — a pinned metric is a dead sensor

Subjects come **from the config**, which names its own sensors: the balance
controller's metrics, the LR sensor's, the stage's exit metrics, the buffer
servo's numerator and denominator, the anchor gate's ceiling and floor. That is
what makes the check general rather than a list that rots — and what keeps it
honest in the other direction, since a constant *set point* is config being
echoed, not a reading.

Conditions: not logged · ambiguous name · no finite values · zero spread ·
pinned at an extremum · at a censoring bound · **R13**, a bar below the metric's
own detrended sigma · a bar with no measurable floor.

Three orderings are load-bearing:

- **Presence is resolved route-blind; meaning is not.** `keys.resolve` tests the
  NA pattern *before* presence — correct there, since NA_ROUTE's defining
  property is that the key is present. Asking it route-first here made a
  genuinely absent sensor on a VarGrad run report NA_ROUTE, i.e. "logged,
  populated, and not this route's to read", about a key that is not logged at
  all. The row asserted a falsehood and swallowed every dead-sensor condition
  behind it.
- **Censoring is tested before pinning.** A clamped value is zero-spread too, and
  calling it flat loses the fact that explains it.
- **Sigma of zero is not a floor of zero.** `bar < 0` is false for every bar, so a
  gated metric that is itself pinned turned R13 from a flag into an OK — killing
  a controller input made the check *quieter*. That is now its own finding, and
  the more serious of the two.

R13 reads both the published annealed bar (`protocol/thr_*`) and the
**statically configured** exit bar — only the lexicographic balance controller
publishes the former, so the published bar alone left the check dark on nearly
every run while still returning a full sensor table.

### §4 — the confounds named routinely

The only check that spans arms, and `run_all` calls it first: a comparison across
arms that are not comparable is not a weaker result, it is not a result.

Per run: T vs eval_T · code version stamp · start condition · stage residence ·
whether the read window straddles a stage boundary · whether the config loaded at
all. Across arms: stage · route · code version · `checkpoint_name` null-ness ·
**checkpoint source** · start conditions · duplicate arms · the sweep table.

Checkpoint source is separate from null-ness because null-ness cannot see §4's
second-named confound. A battery in which every arm carries a checkpoint passes
the null test while one arm resumed from a phase-1 exit and the rest resumed from
that arm's own rolling checkpoint — "arms silently resume from each other",
observed on a real 16-arm battery.

Three confounds §4 names are deliberately absent. A retired or inert knob is
R2's subject, and two checks detecting one condition can disagree about it.
Another process on the GPU is not readable from wandb output — the record carries
this process's utilisation, not the machine's tenancy. The LR's position in its
cycle needs a cycle model, which is `features.py`'s oscillation extraction.

### R11 — replay error below forward error

`replay/scatter_err ÷ fwd/scatter_err`, **pointwise** on aligned steps, median
over the window. A ratio of medians is not the median of ratios and one excursion
moves the former.

**Two sensors, and only one of them has a bar worth flagging on.** Both ratios
are computed **pointwise** on aligned steps, median over the window — a ratio of
medians is not the median of ratios, and one excursion moves the former.

- **Sensor A — `replay/scatter_err ÷ fwd/scatter_err`.** The one the build spec
  named. **Reported, never flagged.** `module_metrics.md` says why: a ratio below
  1 "is equally the signature of memorisation ... and of a coverage gap. The
  statistic does not distinguish them, and reading it as either one alone is
  unwarranted." `module_modulators.md` adds that its thresholds are
  "uncalibrated". Measured, that is not theoretical: **45 of 60 TB-route runs sit
  below 1, and 58 of 60 below the "~2 healthy" figure.** A state that fires on
  three quarters of a corpus is not a finding. The number is surfaced with its
  ambiguity named; the ~2 figure is printed as a stated reference, never a bar.
- **Sensor B — `replay/ema_loss_mean ÷ replay/birth_loss_mean`.** The preferred
  memorisation sensor and the one that **flags**, because its bar is **derived**:
  0.368 is λτ = 1, rows corrected exactly as fast as they are replaced. It
  behaves as a bar should — 0 of 44 runs below it.

The **release** (0.60) is *not* derived; it comes from the config generators, and
most runs declare no `buffer_servo` at all. So the row reads the stage's own
`buffer_servo` and says which source the figure came from — naming [bar, release)
"the servo's hold band" unread asserted a controller the run may not have, and
misnamed it on runs whose servo steers a different pair.

**TB route only**, per the spec. The gate is a property of the **route**, not of
the keys: `keys.resolve` does not mark either series NA on any route, so asking
the key would hand back a number to be read as if it were on a TB run. Off the TB
route the check runs and its rows are NA_ROUTE. An unclassified route is
`not_run` — undetermined applicability is a hole, not a table row. The check
refuses only when **neither** sensor can be read.

An unreadable Sensor B is reported as such, and the *reason* is not overwritten
by a story: a sensor that is absent is abstaining or predates its plumbing; a
sensor logged on every tick whose values are all unusable is a dead sensor. The
same sentence on both told the reader to stop looking at the case worth looking
at.

## Comparing arms — `compare.py`

`compare(runs)` returns a `Comparison`: the §4 result, the sweep table, and the
aligned feature table. `records()` flattens it to one dict per (block, metric,
arm) — that flat form is the primary deliverable, because the thing that reads it
is usually a script or an agent synthesising a write-up, not a person reading a
terminal. `format_comparison` renders the same data for the terminal.

**Comparability travels with the data, it is not a sibling section.** `blockers`
is a field of the feature table and every flat record carries `comparable` and
`n_blockers`, so a consumer cannot tabulate the numbers without also holding the
reason they may not be tabulated.

A blocker is a cross-arm §4 finding — defined by the check, not re-decided here —
**plus one per-run subject: an arm whose config did not load.** That is a fact
about one run and it still voids the comparison, because every cross-arm subject
reads a missing config as `<missing>` and therefore agrees with itself. Two
unparseable configs produced a clean bill, a "no knob differs" sweep, and a full
aligned table putting their numbers in the same rows.

**`span='matched'`** reads every arm over the steps they all cover, ending at the
earliest last step. A trailing window is measured from each arm's *own* last
step, so arms that stopped at different points are read at different training
ages — and every topline metric improves with age. `check_confounds` flags that
(`battery/training_age`); this is the half that makes the comparison answerable.
On two real cluster arms it reversed the result: at their own trailing windows
one led on every topline metric, over the shared span the other led on every one.
The confounds deliberately run on the UNTRIMMED runs, since an arm's stopping
point is a fact about the run and trimming first would hide it.

**Arms on different routes do not share a topline**, so the feature table splits
into per-route blocks rather than forcing a union. ABSENT and NA_ROUTE render
distinctly and never as blank or zero.

The sweep table drops `K.CFG_IDENTITY` — every arm differs in its name, and
listing that is listing the index the sweep is keyed by. Knobs differing by
PRESENCE count, since an absent key takes its default. When the column cap would
render two genuinely different values as the same string, the cap loses and the
row prints one value per line: a sweep row exists *because* those values differ,
so a rendering that hides it asserts the opposite of the fact that created it.

## Facts about the data, established by inspection

Each was an assumption that was wrong in a way that produced silence rather than
an error.

| Fact | Consequence if assumed otherwise |
|---|---|
| wandb stores `protocol` as a repr **string**; the usable form is the flattened `protocol_stages_N_*` keys | every route classifies as UNKNOWN |
| the stage metric is `phase`, and it is **one-based** (`stage.index + 1`) | every run reported one stage behind, or as "unknown" |
| local `config.yaml` wraps values as `{'value': x}`; the cloud API returns them bare | the cloud path sees an empty config |
| `bwd/under_coverage_wcen` is not logged; the run has `bwd/under_coverage` | a topline entry silently missing |
| `log_Z_learned` is namespaced three ways | resolving it at all would be a guess between different quantities |
| `loss_coeffs/*` is emitted **only when a stage transition moves it** | the most direct evidence of which loss terms are live is a sub-3-point series the parser drops |
| the git commit lives in the `_wandb` config blob under a per-machine hash key, and **the cloud API strips that key from `run.config`** — cluster runs carry it on the run object instead (`run.commit` / `run.metadata`), which is why `Run` has a `metadata` field | §4's first confound goes dark on exactly the runs a battery is made of: three real cluster arms all reported "no commit stamp" while all three had one, and were on three DIFFERENT commits |
| `gates/mle_flat` is published to the protocol and never logged as a metric | its exit streak is its only observable trace |
| `pull` raises only on empty **history**, so a run can arrive fully parsed with `config == {}` | every cross-arm comparison reads `<missing>` on one side and flags confidently |

## The mechanism registry

`keys.MECHANISMS` pairs a declaring config key with the trace that proves the
mechanism ran. **Every entry was verified against the local corpus**, because an
unverified pair is worse than a missing one: it manufactures findings on runs
that are fine, and a check that cries wolf gets switched off.

Two traps it encodes, both of which produced a wrong reading during construction:

- `protocol/bs_boost` is `exp(log_boost)` and reads 1.0 while the servo is idle.
  A trace that is nonzero at rest is not a trace; the actuator is
  `protocol/bs_log_boost`.
- `ray_calibration.enabled` and a stage's `lr_sensor.kind: ray` are different
  declarations — the trainer's own startup check warns when they disagree — so
  both are registered and they answer different questions.

`adaptive_lr` carries two entries for one mechanism because the declaring key
changed and the eras are disjoint in the corpus. Registering only the newer key
left R2 asserting nothing about the LR controller on most runs, in language that
read as "not configured".

**Known coverage gaps.** These config switches are active on real runs and have
no registry entry, because no declaration→trace pair for them has been verified:
`step_probe_enabled`, `adaptive_lr_servo_enabled`, `adaptive_lr_damage_enabled`,
`adaptive_lr_discovery_enabled`, `grad_clip_guard_enabled`,
`buffers_anchor_buffer_health_gate_*`, and the `*_gain` knobs. Some may have no
logged trace at all — that is a finding to record, not a silence to keep.

## Traps encoded in the code

`H1` an unresolved key makes `scan_history` return zero rows **silently**, so it
costs the whole pull, not that key — resolve first, and an empty pull raises
`EmptyPull` rather than returning nothing. `H3` local runs order by the launch
timestamp in the directory *name*, never mtime, because the sync service touches
old runs; the ghost filter requires small **and** old, or it races launches.
`H4` `item.key` is empty for nested metrics (use `nested_key`), not every row
carries `_step` (carry it forward), parsing breaks rather than raises on a
partial final record, and a just-restarted run's file can be 0 bytes.
`H5` `tracker/*` are EMA outputs: trend shown, significance and oscillation
suppressed, since smoothing manufactures the autocorrelation a significance test
reads as signal — and lowers variance, so an EMA sensor can read as pinned. R14
says so in the row rather than suppressing the finding.

## Cache

`<tempdir>/gfn_analysis_cache`, keyed by `(run_id, last_step)`. Outside the repo
deliberately. A cold pull is ~2 s, so the cache is for repeat analysis, not
latency; a corrupt entry is a miss, never an error.

## Tests

`analysis/tests/`, network-free and `wandb/`-free. 354 tests.

The fixtures are eight **real captured runs** (`tests/fixtures/`, built by
`_capture.py`, which records why each one is there), not mocks. They span all
three routes and carry the counter-cases: a TB run whose ray calibration fired,
a VarGrad run that declares the same mechanism and never fired it, a run that
died in phase 1, the only run publishing annealed exit bars, a resumed-from-
checkpoint arm, and a two-arm battery that turns out to be one arm written twice.

**Every check is mutation-tested**: each condition has a test that re-introduces
it into a real fixture and requires the check to fire, *and* the companion that
the unmutated run does not fire for that subject. A check that has never fired
has not been tested; a check that fires on everything is not a check. The
datastore fixture is a real `.wandb` file written by wandb offline, because the
empty-`item.key` trap exists only in the real encoding.
