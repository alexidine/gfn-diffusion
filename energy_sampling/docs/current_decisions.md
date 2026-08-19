# Current GFN decisions

- **Status:** ACTIVE DECISION RECORD
- **Scope:** accepted owner choices that cannot live more clearly in code, canonical config, validation, or focused tests
- **Authority:** subordinate to repository-root `AGENTS.md`; each entry is authoritative only for its stated scope

This file is intentionally small. It is not a work queue, experiment log, findings ledger, or replacement for executable proof. Add an entry only after explicit owner acceptance, and retire it when executable structure makes the intent self-evident or the owner supersedes it.

## D-001 — Do not introduce optimizer-block nesting

- **Class:** decision
- **Status:** accepted
- **Accepted:** 2026-08-17 by the repository owner
- **Scope:** GFN optimizer/configuration organization
- **Decision:** Keep the existing optimizer learning-rate fields flat. The proposed optimizer-block nesting is dropped, not deferred.
- **Reason:** Nesting would add consistency but no capability or demonstrated failure prevention, while touching many direct and aliased read sites with silent-fallback risk.
- **Executable consequence:** None required. The current flat schema remains the intended structure; absence of the proposed nesting is deliberate rather than unfinished work.
- **Supporting argument:** `design/infrastructure_stabilization.md`, “1.2's structural remainder.”
- **Review trigger:** Reconsider only if a concrete capability or recurring failure cannot be addressed safely with the flat structure.

## D-002 — Infrastructure stabilization remains the active plan

- **Class:** decision
- **Status:** accepted
- **Accepted:** 2026-08-18 by the repository owner
- **Scope:** `design/infrastructure_stabilization.md` and the infrastructure work it sequences
- **Decision:** Treat the document as the active operating plan until every remaining completion criterion is either completed with appropriate proof or explicitly retired/obviated. Its to-dos remain live by default.
- **Authority boundary:** The plan owns work sequencing and remaining-work status. Code, canonical config, validation, and focused tests remain authoritative for current behavior; completed-phase narration is historical rationale unless reverified.
- **Maintenance:** Reformatting, evidence-backed status updates, and corrections for superseded procedure are allowed. Removing or materially redefining a live outcome requires owner acceptance.
- **Completion trigger:** Retire or replace the plan when all live completion criteria are resolved, or earlier if the owner explicitly supersedes it.

## D-003 — The recurring-workflow criterion names performance-investigation

- **Class:** decision
- **Status:** accepted
- **Accepted:** 2026-08-18 by the repository owner
- **Scope:** the recurring-workflow completion criterion in `design/infrastructure_stabilization.md`
- **Decision:** Of the five workflows the plan originally enumerated, four are retired and one remains. The criterion now reads: **the performance-investigation workflow is scripted and documented.** Retired: `update-old-run`, obviated by dropping backward compatibility; `functional-change`, covered by root change discipline, proof selection, and bounded test routing rather than a script; documentation generation/refresh and standard experiment launch, both already deferred in the plan's own §5 and never live outcomes. Production-config generation and run analysis shipped.
- **Reason:** The retirements are individually sound, and a `functional-change` script in particular would be prose in a file rather than a mechanism -- it fails root `AGENTS.md`'s own widening check, "would a new abstraction or document solve a demonstrated recurring problem?". What required owner acceptance was not the retirements but the restatement of the bar: an intermediate wording, "the remaining recurring workflow **with demonstrated automation value**", made the criterion self-adjusting. Anything unbuilt could be reclassified as lacking demonstrated value, so the box could not fail to be satisfied. A completion criterion that cannot come out wrong is not a criterion. Naming the one surviving workflow restores a bar that can still be failed, while keeping every judgment the retirement rested on.
- **Executable consequence:** None. This fixes what the plan claims, not what the code does.
- **Supporting argument:** `design/infrastructure_stabilization.md` §6 completion criterion; root `AGENTS.md` change discipline and verification sections.
- **Review trigger:** Reconsider if a second workflow demonstrates recurring manual cost, or if performance-investigation is itself obviated by the Phase 4 measurement it exists to consume.
