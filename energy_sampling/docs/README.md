# GFN documentation router

Status: **ACTIVE ROUTER**  
Scope: `energy_sampling/docs/`

Read the repository-root `AGENTS.md` first. This page routes documentation; it does not override executable sources or the root instructions.

Do not load this directory wholesale. Most files are evidence, snapshots, proposals, or history and will degrade context if treated as a single current specification.

## Active epistemic system

- [`EPISTEMIC_PROTOCOL.md`](EPISTEMIC_PROTOCOL.md) — the current operating procedure for authority, proof, evidence, decisions, assumptions, and freshness.
- [`current_decisions.md`](current_decisions.md) — accepted owner choices that cannot live more clearly in executable sources; read only when the task implicates one.
- [`design/infrastructure_stabilization.md`](design/infrastructure_stabilization.md) — the active infrastructure plan adopted by decision D-002; use for sequencing and remaining-work status, not as proof of executable behavior.
- Root `AGENTS.md` — the repository constitution and higher authority.
- `configs/mk_dev.yaml` — the integrated operational control surface for current crystal runs; outside this docs directory.

## Document-family status

| Family | Status | Use |
|---|---|---|
| `EPISTEMIC_PROTOCOL.md` | **ACTIVE** | Follow for institutional/epistemic work |
| `current_decisions.md` | **ACTIVE DECISION RECORD** | Accepted non-executable choices; not a work queue or general log |
| `design/infrastructure_stabilization.md` | **ACTIVE PLAN** | Remaining infrastructure sequence and completion criteria; subordinate to executable proof |
| `PROTOCOL.md` | **LEGACY** | Historical predecessor; do not follow |
| `findings.md` | **EVIDENCE LEDGER** | Read cited entries only; scoped observations/mechanisms, not policy |
| `decisions.md` | **LEGACY MIXED LEDGER** | Investigate specific historical rulings; do not append or treat the file as current policy |
| `change_history.md` | **GENERATED MIGRATION VIEW** | Config/checkpoint transition evidence only; not a development log |
| `synthesis.md`, `module_*.md`, `lr_control_summary.md` | **SNAPSHOTS** until individually reverified | Explanatory navigation; verify every material claim against current proof |
| other `design/*.md`, `to_do_rebuild.md`, `analysis_package_spec.md` | **ARGUMENT / PROPOSAL / PLAN by default** | Rationale and candidate work; not accepted policy unless individually promoted here by an active decision |
| `reading_runs.md`, `audit_*.md`, completed handoffs/audits | **EVIDENCE / HISTORY** | Read only for the named investigation |

The method in `design/comment_audit.md` is reusable for proof-based freshness audits. Its particular findings describe a completed audit and are not a permanent current-state document.

## Reading order for a task

1. Identify the live code, canonical config branch, interface, or test that owns the question.
2. Read only the scoped explanatory document, decision, or finding needed to understand intent/evidence.
3. Check material prose claims against the proof type required by `EPISTEMIC_PROTOCOL.md`.
4. If sources disagree, report the conflict; do not infer authority from date, volume, repetition, or Git history.

## Maintaining this router

Update this page when a document becomes part of the active reading path, is demoted to a snapshot/legacy artifact, or is replaced. This is routing maintenance, not an invitation to synchronize the entire docs tree.

At milestones, audit the high-exposure families listed here. Correct, demote, or remove stale claims that still sound current. Preserve important history explicitly; do not leave it interleaved with current guidance.
