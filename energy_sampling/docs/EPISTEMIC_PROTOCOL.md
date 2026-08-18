# GFN epistemic protocol

Status: **ACTIVE**  
Scope: institutional knowledge under `gfn_diffusion/`  
Authority: implements the repository-root `AGENTS.md`; if they conflict, `AGENTS.md` wins

This is the one operating procedure for deciding what repository claims mean, where they belong, what supports them, and how they stay current. It supersedes the documentation rules in `PROTOCOL.md` while retaining their useful evidence discipline.

## 1. Principles

1. Prefer executable proof for executable claims.
2. Code proves current behavior, not necessarily correctness, intent, or desired policy.
3. Put one material claim in one authoritative home; other documents point to it.
4. Separate current state, decisions, working assumptions, observations, arguments, and history.
5. Surface conflicts. Do not resolve them by recency, repetition, or apparent Git history.
6. Update directly affected context when behavior changes; audit high-exposure context periodically.
7. Keep institutional work bounded. Do not turn a local change into a repository-wide documentation sweep.

## 2. Knowledge types, homes, and proof

| Type | Primary home | Required support | Lifecycle |
|---|---|---|---|
| **Invariant / interface** | implementation, validation, focused tests | executable check where practical | changes with implementation |
| **Default / operational state** | canonical config or explicit code default | loader/preflight and contract or smoke check | overwritten as the live control surface changes |
| **Workflow** | named entry point plus concise routing instructions | smoke or integration check | updated with the workflow |
| **Decision** | executable source when self-evident; otherwise a concise owner decision | explicit acceptance and scope | revised or retired when the decision changes |
| **Working assumption** | canonical config comment or active scoped design note | label, scope, and review trigger | removed, promoted, or revised |
| **Observation / measurement** | `findings.md` or a named benchmark artifact | inputs, conditions, result, confidence grade | retained as scoped evidence; may be superseded |
| **Argument / rationale** | `design/` or a focused explanation near the implementation | links to the decision and supporting evidence | never policy by itself; revise or retire |
| **History** | explicitly labelled legacy artifact | date and reason it remains useful | no current authority |

Do not infer current design from historical configs, experiment prevalence, old scripts, module prose, comments, or run chronology. A source is authoritative only for the type and scope it owns.

## 3. Code as proof

Documentation should point to the strongest practical proof rather than duplicate detailed executable state.

- Behavioral claims point to implementation and, where practical, a focused test or validation path.
- Config claims point to the canonical config and its loader/preflight behavior.
- Workflow claims point to the entry point and a smoke check.
- Performance claims point to a named benchmark with reproducible inputs and environment.
- Decisions and rationale may have no code proof; they require explicit owner acceptance and scope.
- Working assumptions are never written in the voice of settled fact.

If code and prose disagree, report the conflict. Do not force code to match prose, and do not bless accidental code behavior as intended policy.

Prefer stable symbolic references—function, class, key, test, or section—over bare line numbers. Line numbers may be included as navigation but are not durable proof.

## 4. Evidence grades

Every material finding carries a confidence grade. The default is `OBSERVED`.

| Grade | Bar | Generalization |
|---|---|---|
| `MECHANISM` | derived or verified against implementation, with assumptions stated | within the mechanism's assumptions |
| `REPLICATED` | at least two seeds or conditions and effect exceeds measured noise | within the stated scope |
| `OBSERVED` | measured once | that run and stated conditions only |
| `CONJECTURE` | argued but not measured | none |

Below `MECHANISM`, include the conditions that materially control interpretation: problem/route, temperature, stage, steps, seeds, relevant hardware, and configuration deviations. Omit irrelevant fields rather than filling a ceremonial template.

### Finding write gate

Write a finding only if it:

- changes what should be done next;
- prevents a plausible failed hypothesis from being repeated; or
- establishes a reusable mechanism or measurement method.

A routine run that confirms expectations belongs in a run table or external run record, not narrative repository context.

`findings.md` remains an append-only evidence ledger. Correct an old entry by adding a new entry that names what it supersedes. Active docs and indexes must point to the newest applicable evidence; agents should not load the entire ledger to discover current policy.

## 5. Decisions, assumptions, and arguments

A decision record contains only:

- the accepted choice;
- its scope;
- the owner or acceptance event;
- the reason only to the extent it is not evident from executable sources;
- links to supporting or contradicting evidence;
- a review trigger when the choice is intentionally provisional.

Open questions, work queues, experiments, documentation-sync logs, and closed historical dockets are not current decisions.

The existing `decisions.md` predates this protocol and is a mixed legacy ledger. Do not append new decisions there or treat the file as a unit of current authority. A specific ruling is evidence of owner intent in its stated scope, but must be checked against current executable state and any later explicit ruling. Create `current_decisions.md` only when a new accepted decision cannot live more clearly in code, config, tests, or root instructions.

Design documents explain options and reasoning. They are proposals or rationale unless an active decision explicitly adopts them. “Argument” does not mean timeless or authoritative.

## 6. Change and migration records

A semantic change record is required only when an active config or checkpoint written before the change could be read incorrectly afterward.

- Renamed, removed, or reinterpreted config keys and defaults may require a checked transition.
- A transition moves project/config state only when executable migration is genuinely needed.
- Bug fixes, performance work, metrics, and refactors do **not** automatically require a Change record.
- Historical configs do not need migration. They may fail loudly.
- Preserve compatibility only for an explicitly active run or checkpoint.

Generated `change_history.md` is a view of executable migration data, not a general development log or design authority.

## 7. Current-state and module documentation

A narrative document does not become current merely by calling itself State.

High-exposure narrative docs should begin with a short status statement:

- `ACTIVE` — routes to current proof and has been checked for its stated scope;
- `WORKING ASSUMPTION` — provisional, with a review trigger;
- `SNAPSHOT` — explanation tied to a past working state;
- `LEGACY` — retained evidence/history, no current authority.

A review date routes attention but does not prove correctness. `ACTIVE` claims still point to code, config, tests, decisions, or scoped evidence.

When a directly relevant change invalidates prose, update it, mark it `SNAPSHOT`/`LEGACY`, or remove it from active routing in the same change. Do not sweep unrelated docs.

## 8. Periodic context audit

At a project milestone or dedicated maintenance pass, inspect only high-exposure context:

1. root `AGENTS.md` and `docs/README.md`;
2. this protocol;
3. README and active workflow instructions;
4. canonical-config commentary;
5. current decision records;
6. module/synthesis docs actively routed to agents.

For each material claim:

1. identify its knowledge type;
2. locate its appropriate proof;
3. classify it as current, assumption, observation, or history;
4. correct, demote, or remove anything that sounds more authoritative than its proof permits.

The procedure in `design/comment_audit.md` is a useful model: rank by harm, quote the claim, and check it against the source. Its specific findings are a completed historical audit, not continuing authority.

## 9. Git and history

Do not rely on commit discipline. Commit recency, messages, branches, tags, and apparent chronology do not establish authority or intent.

If historical context is important enough to affect future work, preserve it explicitly as dated, scoped evidence. Otherwise remove obsolete claims from active context. Do not keep poisoned prose merely because Git might contain an earlier version.

## 10. Documentation routing

`docs/README.md` is the active routing surface for this directory. It classifies document families and prevents broad ingestion of stale material. Update the router when a document is promoted, demoted, replaced, or added to the active reading path.

No task should load the entire docs tree by default.
