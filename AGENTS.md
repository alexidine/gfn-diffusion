# GFN repository instructions

## Purpose and priorities

Make the research codebase trustworthy to change while preserving rapid iteration. Prefer simple, composable mechanisms and introduce only as much structure as current work requires.

Work priorities are:

1. finish and stabilize unconditional GFN crystal training;
2. develop and test conditional GFN crystal training;
3. advance the conformer GFN refactor as an authorized parallel track, for later integration with the crystal machinery;
4. maintain crystal search and dataset preparation as supporting dataset-generation workflows.

Conformer GFN is midway through a total refactor, which is an authorized workstream in its own right and not general cleanup. The restriction binds *other* work: do not modify conformer code, configuration schema, or institutional contracts as part of repository cleanup, crystal-side changes, or schema alignment sweeps. Changes made as part of the refactor itself are exempt and do not require that authorization to be re-established. Align conformer code with the settled schema after the refactor, not during it.

The refactor's training target is the **`full` internal-coordinate level** — every bond length, angle and dihedral free. `torsion`, `dihedral` and `flex` are helper tiers: comparison arms, ground-truth anchors, and utilities. They are deliberately retained and are not approximations of the target — freezing a degree of freedom yields a different distribution, so a figure measured at one tier does not describe another. Label the tier on every conformer measurement.

`energy_sampling/` is the current code location, not an independent subsystem or authority boundary. These instructions apply to the whole repository. Do not add a nested instruction file there.

GFN depends on the sibling MXtalTools repository. The production dependency is one-way: GFN may use MXtalTools; MXtalTools must remain independently usable.

## Authority by knowledge type

- Invariants: current runtime validation and focused contract tests.
- Interfaces: current callable/data contracts and their boundary tests.
- Defaults and operational state: explicitly named canonical config and explicit code defaults.
- Workflows: named entry points, one current launch recipe, and smoke tests.
- Decisions: concise accepted decisions not already expressed above.
- Working assumptions: explicitly labelled, scoped, and revisable.
- Observations: findings, measurements, and run results; evidence, not policy.
- History: explicitly labelled historical notes, old configs, prior experiments, run sheets, and archived scripts; no current authority unless a canonical source explicitly incorporates them.

Agent memories, prior-chat summaries, handoffs, and tool-specific state are navigation leads, not repository authority; recheck their material claims against the sources above before acting. Do not include `.claude/worktrees/` or similar generated/cache directories in repository-wide searches unless the task explicitly targets that worktree; a nested copy is never evidence about the main working tree.

If appropriate authoritative sources disagree, surface the conflict. Do not infer project policy from repeated historical usage, a comment, or the newest-looking document.

## Proof model

Documentation should carry or point to the strongest practical proof for each material claim:

- Current behavior, invariants, and interfaces: implementation plus focused validation or tests where practical. Code proves what currently happens; it does not by itself prove that the behavior is correct or intended.
- Operational defaults and workflows: canonical config, named entry point, and a smoke or contract check.
- Decisions, rationale, priorities, and intended direction: an explicit owner decision with clear scope. These claims may have no code proof.
- Working assumptions: an explicit label, scope, and condition for revisiting them.
- Observations and performance claims: reproducible inputs, measurements, or a named benchmark.

Prefer docs that point to proof over prose that duplicates detailed executable state. If proof and prose disagree, surface the conflict rather than forcing code to match the document or treating accidental behavior as policy. Label material claims that do not yet have adequate proof.

Do not use commit recency, commit messages, branch names, or apparent Git history to infer authority or intent. If historical context matters, preserve it explicitly as dated evidence; otherwise remove obsolete claims from active context.

## Owner adjudication and knowledge capture

Ask the owner only when missing intent would materially change the result. Present one direct question, lead with the recommended answer, give two or three mutually exclusive choices with their consequences, and state the default when proceeding without an answer would be safe. Keep it short. Do not turn status updates, facts discoverable from current sources, reversible implementation details, or low-consequence preferences into owner adjudications. Once answered, treat the decision as settled within its stated scope unless new evidence creates a real conflict.

Before proposing to store a claim as repository knowledge, name its class—**invariant, interface, default, workflow, decision, working assumption, observation, or history**—and its proposed authoritative home. Explain why persistence is warranted. Do not ask the owner to classify trivia or log routine implementation details. If code, config, or a focused test can express the claim self-evidently, encode it or point to it there instead of creating duplicative prose.

## Current workflows

- Crystal training: `energy_sampling/configs/mk_dev.yaml`. This is the integrated operational control surface and spawn point for runs. It intentionally contains unconditional and conditional logic because separate versions de-synchronized. Do not split it merely for neatness.
- Conformer training: under active refactor; `energy_sampling/configs/conformer_dev.yaml` is not a stable schema contract yet.
- MXtalTools training: no GFN-owned canonical config.
- Crystal search: no canonical global config; current runs are task-specific.

All other YAMLs are non-authoritative by default. Future work does not need to preserve compatibility with previous config versions unless an active run or checkpoint explicitly requires it.

A value in `mk_dev.yaml` may be durable structure, current operational state, or a working assumption for a selected branch. Do not automatically promote every value into repository-wide policy. Preserve fused unconditional/conditional configuration; prevent drift through shared structure and validation rather than duplicated configs.

From `energy_sampling/`, the proved crystal commands are:

- Validate the fused operational config without runtime services: `python -m config_snapshot configs/mk_dev.yaml --check`.
- Launch crystal training: `python -u train.py --config configs/mk_dev.yaml`.
- Exercise the live CPU/synthetic GFN-to-MXtalTools ELJ boundary: `python -m pytest -q tests/crystal/test_mxtaltools_crystal_boundary.py`.
- Run the broader CPU development lane when the change warrants it: `python -m pytest -m fast -q`.

The focused contract check is the normal verification for config-loader or launch-surface edits. The broader fast lane is not required for unrelated localized changes.

`MolCrystalData.analyze(...)` is the shared analysis/energy dispatch surface used by GFN; ELJ, MACE, UMA, and other `crystal_analysis.py` computations are selectable backends, not competing definitions of the whole workflow. ELJ is the cheap backend selected by the current `mk_dev.yaml`; MACE is used specifically for acridine; UMA is the more expensive general MLIP backend. The two MLIP interfaces are live optimization targets and require their own focused correctness/performance evidence.

For the current ELJ route, the proved MXtalTools chain is `MolecularCrystal.analyze_crystal_batch` -> `MolCrystalData.latent_to_cell_params` -> `MolCrystalData.analyze(['reduction_en', 'elj'])` -> `mol2cluster` -> `construct_radial_graph` -> eLJ analysis. MXtalTools' on-device PBC neighbour-list implementation is currently used by the MACE adapter, not this ELJ route. UMA has its own interface and must not be assigned MACE's indexing contract by inference. Dataset-construction indexing is a separate migration surface. Do not silently substitute or generalize among these paths.

## Change discipline

- Prefer the smallest change that advances the named live workflow.
- Do not reconcile historical configs or update unrelated narrative documents.
- Do not turn a local observation into a repository-wide invariant without owner confirmation or executable evidence.
- Every wait needs a termination proof. For any gate, quorum, streak, or settling condition, answer: what happens if the thing it waits on never arrives? If there is no answer, bound it. This repository has shipped that fault more than once — a quorum whose denominator could never enter its numerator, and a settling gate waiting on a signal that the unsettled learning rate was itself keeping high.
- When changing GFN use of MXtalTools, identify the consumed interface and test that boundary where practical.
- Do not add reverse MXtalTools-to-GFN runtime dependencies.

Before widening a task, check:

1. Does it directly advance the primary crystal workflow or protect an interface that workflow currently uses?
2. Is the knowledge an invariant, interface, default, workflow, decision, working assumption, observation, or history—and is it going into the right representation?
3. What is the smallest verification that covers the changed contract?
4. Would it enter the conformer refactor, historical compatibility, general MXtalTools modernization, or broad documentation cleanup without explicit authorization?
5. Would a new abstraction or document solve a demonstrated recurring problem?

If the task fails these checks, stop and surface the scope expansion instead of proceeding.

## Verification

Run the smallest test tier that covers the changed contract:

1. stable canonical-config and import contracts (crystal now; conformer after its refactor);
2. owning CPU unit tests;
3. small synthetic integration tests for cross-component changes;
4. GPU, real-data, checkpoint, MLIP, and benchmark tests only when the change reaches those paths.
5. Anything that runs in a LOOP with the trainer — an LR controller, a balance rule, a buffer servo — is verified by a short real run, not by unit tests alone. Its defects live in the interaction, and tests written from the same mental model as the code inherit that model's blind spots. Run first, at production-shaped parameters, then write tests to pin what the run showed. Where a test shrinks a parameter for speed, something else must cover the unshrunk value — the shrink usually disables the path that breaks. See `energy_sampling/docs/reading_runs.md` §8 for how to watch one.

Old, WIP, diagnostic, and experiment tests are opt-in unless a current contract explicitly depends on them. Do not run the full repository test estate for every localized change.

## Reporting

When ending a work session or completing a delegated task, report in this shape:

1. **Outcome** — one or two sentences: what is now true that was not before, written to parse without the transcript.
2. **Changes** — one line per changed file or contract: the meaning of the change first, the identifier in parentheses. Omit files touched only mechanically.
3. **Verified by** — what was actually run and what it showed. "Not verified" is a valid entry; silence is not.
4. **Open items** — anything deferred, discovered, or still owed, each as an actionable line. Say "none" rather than omitting the section.

Size the report to the task: a one-file fix warrants roughly four lines; a typical task at most ~15. Do not narrate process, restate the request, or use shorthand coined mid-session — a name invented during the work must not appear in the report. Before sending, apply the test: would the owner, having read none of the transcript, know what changed, how it was checked, and what they must do next?

### Tables

A table is a claim, not a grid of numbers. Every table an agent presents — in a report, in chat, or printed to stdout by a diagnostic or analysis script — carries labels and a caption:

- **Labels.** Each column names the quantity and its unit or scale (`lattice energy (kJ/mol)`, `log Z (nats)`, `wall time (s/step)`); each row names its case — the run name, config key, arm, or molecule, not an index. Never present a bare number whose meaning lives only in the surrounding conversation.
- **Caption.** One or two sentences above or below the table: what was measured, on what (run, config, dataset, step range), how many samples or seeds each cell rests on, and what the reader is meant to compare. Where a number is only meaningful against a reference, give the reference — the baseline arm, the prior value, or the resolution below which differences are noise.

If a column cannot be given a quantity and a unit, it does not belong in the table. If the caption cannot say what the comparison shows, the table is not yet a result.

## Keeping context current

Stale prose is harmful context, not harmless history. Maintain active knowledge in two bounded ways:

1. **Event-driven:** when a change invalidates directly relevant prose, update it or explicitly demote it in the same change. Do not sweep unrelated documentation.
2. **Milestone-triggered:** at an owner-declared project milestone or during an explicitly requested dedicated audit, inspect only high-exposure context: `AGENTS.md`, README/routing material, active workflow documents, canonical-config comments, and accepted decisions.

Classify reviewed material as current policy, working assumption, observation, or history. A stale or unresolved document must not continue to sound current: correct it, mark its status prominently, or remove it from active context. Do not assume Git can recover missing intent; preserve important history explicitly and discard obsolete claims from places agents are expected to consult.

Freshness dates may help route attention but do not establish correctness. Verification against current code, config, and tests does.

## Documentation

For work involving repository knowledge, read `energy_sampling/docs/README.md` for routing and `energy_sampling/docs/EPISTEMIC_PROTOCOL.md` for the operating procedure. They are subordinate to this file.

For infrastructure-stabilization work, `energy_sampling/docs/design/infrastructure_stabilization.md` is the active plan adopted by `current_decisions.md` D-002. It controls sequencing and remaining-work status, not current executable behavior; read only the sections implicated by the task.

When available, use `$audit-active-context` at an owner-declared project milestone or when active guidance may be stale or conflicting; do not run calendar audits or invoke it for every ordinary change by default. When unavailable, follow this file's bounded context-maintenance rules and `energy_sampling/docs/EPISTEMIC_PROTOCOL.md` §8 directly. Directly invalidated active guidance is still corrected event-by-event.

When available, use `$orchestrate-repository-work` only after the user authorizes delegation and the project has genuinely separable workstreams. When unavailable, keep one agent as the default; give any authorized agents narrow, non-overlapping deliverables and require evidence-backed handoffs to one integration owner. Collapse the organization when only sequential work remains. Skill names are optional environment capabilities, not repository dependencies; this file and scoped repository sources retain authority.

Do not load `energy_sampling/docs/` wholesale. `current_decisions.md` contains accepted non-executable decisions and should be read only when the task implicates one. `findings.md` is scoped evidence; `decisions.md` is a legacy mixed ledger; `synthesis.md` and `module_*.md` are explanatory snapshots until individually reverified; except for the individually adopted infrastructure plan above, `design/` contains arguments, proposals, and plans rather than policy by default.

Comments and module documents explain intent but are not automatically current. Verify them against code, canonical config, and tests. Record a decision only when it cannot be made self-evident there. Record a finding as evidence, never as policy.
