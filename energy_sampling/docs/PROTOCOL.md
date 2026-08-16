# Documentation protocol

How these docs are written. Adopted 2026-08-10. This file is State: if a rule
changes, overwrite it.

## Four types, four homes

Docs failed previously by interleaving all four in one narrative voice, which
made State unreadable without Log and made Evidence impossible to grade.

| Type | Home | Voice | Lifecycle |
|---|---|---|---|
| **State** — what is true now, what the code does | `synthesis.md`, `module_*.md` | present tense, no dates | **overwritten in place** |
| **Evidence** — a measurement and its conditions | `findings.md` | dated, graded | append-only, never edited |
| **Argument** — why we chose X | `design/` | timeless | revised rarely |
| **Log** — what happened when | **git history only** | — | never written to a file |

`audit_since_ty4xdlzo.md` is frozen Evidence; the `.html`/`.tex` derivations are
Argument. Neither gets edited.

### Transition — the one exception to "Log lives in git"

A **state transition** record says how a config at project state N reaches state
N+1. It reads like Log and is not: Log narrates what happened, a Transition
answers a question a future reader must be able to answer mechanically, and
unlike Log its correctness is checkable — the migration either produces a
loadable config or it does not.

It gets a home because git cannot serve that question. Reconstructing a migration
from a diff means reading every commit between two states and inferring intent.

**Transitions are data, not prose.** They live in `config_state.STATE_HISTORY`
beside the migration that implements them, so a description and its transform
cannot drift. `docs/state_history.md` is generated from those records and is
never hand-edited; a test asserts the committed copy matches. Append one record
per transition; never edit a shipped one.

## Grades

Every finding carries one. Prose cannot be trusted to carry confidence, so it is
a required field. **Default is `OBSERVED`.** Promotion requires naming the thing
that earned it — the replicate, the measured noise floor, the derivation.

| Grade | Bar | Generalizes? |
|---|---|---|
| `MECHANISM` | derived, or verified against code | yes — flat claims allowed |
| `REPLICATED` | ≥2 seeds or ≥2 conditions, effect > measured noise floor | within stated scope only |
| `OBSERVED` | measured once | **no** — a fact about that run |
| `CONJECTURE` | argued, not measured | no |

There is no `DEAD` grade. A refuted claim is either restated **positively as
State** ("unfreezing is not an LR effect") or it is Log and belongs in git.

## Scope line

Mandatory on everything below `MECHANISM`: **T, problem, stage, steps, seeds.**
T dominates outcomes here, so a claim without T is unreadable. Most
over-generalization is just a missing scope line.

## Write gate

Write a finding only if it:

- changes what we would do next, **or**
- kills a hypothesis someone would otherwise re-propose, **or**
- is a mechanism.

Otherwise: one row in a run table, no prose. **A run that confirmed the expected
is not written up.** Neither is a run that died for an operational reason, nor an
intermediate step a later entry supersedes within the same session.

## Hard overwrite

State docs are rewritten in place, silently. No supersession chains, no "update,
later the same day", no preserved wrong hypothesis. Git holds the history.

Corollary: **never overwrite uncommitted docs.** Commit before a rewrite pass.

## One fact, one home

Cross-reference by ID, never by restatement. If a fact's status changes, exactly
one file changes.

- `F-<n>` — findings (`findings.md`)
- `D<n>` — decisions (`decisions.md`)
- `train.py:1234` — code citations, preferred over paraphrase

A module doc's Findings section holds the **current one-line state plus the
`F-` id**. The evidence lives in `findings.md` and is not repeated.

## Caps

Caps force triage; they are not style preferences.

- Finding entry: **~150 words + one table.** Overflow is either a mechanism
  (promote to the module doc) or Log (drop it).
- Instrumentation facts — how to read a metric, what its scatter is — are
  **not** findings about the thing being measured. Promote them to
  `module_metrics.md`, where they apply to every future reading.
