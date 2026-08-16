# module: analysis

The run-reading toolkit. `docs/reading_runs.md` is its requirements document;
this file records what exists.

Tier 0 only. Tiers 1–3 (`checks.py`, `compare.py`, figures) are specified in
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
| `keys.py` | route + stage detection, key resolution. **The only file that may contain a metric-name literal.** |
| `pull.py` | run resolution, local `.wandb` and cloud history, disk cache |
| `features.py` | trend / oscillation / runaway-growth extraction |
| `cli.py` | `python -m analysis <run-spec>` |

## It emits no verdicts

Mechanical principles are checks; everything else surfaces its inputs and stops.
The ported `wa.py` had a `bellwether_verdict` printing "healthy climb" and
"policy losing ground"; that did not come across, deliberately. Reading here is
context-dependent and jumps to whatever the phase and symptom implicate, and
encoding that jump produces confident wrong answers.

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

## Routes

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

## Toplines

TB/unconditional is `reading_runs.md` §2 verbatim, including the two entries that
do not resolve on real runs — they surface as a rename and an ambiguity rather
than being silently corrected here, so the discrepancy with the doc stays visible.

The VarGrad topline carries four families: the dispersion the objective
minimises (`logw_std_within`, `vg_lb`), the worst-case per-condition Z mismatch
(`zmatch/delta_worst`), per-condition fractions in Cond × Spread form only, and
the held-out `eval_test/*` series. Per-parameter subtleties within these are
deliberately not encoded — they are judgment.

## Facts about the data, established by inspection

These were assumptions in the spec and are now verified. Each was wrong in a way
that produced silence rather than an error.

| Fact | Consequence if assumed otherwise |
|---|---|
| wandb stores `protocol` as a repr **string** (`"Namespace(stages=[...])"`); the usable form is the flattened `protocol_stages_N_*` keys | every route classifies as UNKNOWN |
| the stage metric is `phase`, and it is **one-based** (`stage.index + 1`, `train.py:289`) | every run reported one stage behind, or as "unknown" |
| local `config.yaml` wraps values as `{'value': x}`; the cloud API returns them bare | the cloud path sees an empty config |
| `bwd/under_coverage_wcen` is not logged; the run has `bwd/under_coverage` | a topline entry silently missing |
| `log_Z_learned` is namespaced three ways | resolving it at all would be a guess between different quantities |

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
reads as signal.

## Cache

`<tempdir>/gfn_analysis_cache`, keyed by `(run_id, last_step)`. Outside the repo
deliberately. A cold pull is ~2 s, so the cache is for repeat analysis, not
latency; a corrupt entry is a miss, never an error.

## Tests

`analysis/tests/`, network-free, 59 tests. The datastore fixture is a real
`.wandb` file written by wandb offline rather than a mock, because the
empty-`item.key` trap exists only in the real encoding.
