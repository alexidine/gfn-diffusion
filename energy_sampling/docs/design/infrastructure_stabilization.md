# Infrastructure stabilization

The plan for reaching a feature-stable production foundation. Argument, in the
`docs/PROTOCOL.md` sense: it records *why* the work is sequenced this way, and is
revised when the reasoning changes rather than appended to.

Completion is defined at the bottom. Until every box there is checked this is the
active plan; after that, foundational change requires a demonstrated need.

**Current position.** Phase 0 is done and the first slice of Phase 3 with it:

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
- **Phase 3b Tier 0 shipped** — `analysis/`, 59 tests, State in
  `docs/module_analysis.md`.

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

Next: Phase 1, gated on the commit. Phase 3b Tier 1 and Phase 5.0/5.1 are
independent of it.

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
demotes accordingly: from a 322-line mode-overlay matrix that rewrites whole
config sections, to a thin problem registry.

This makes the central §1 requirement and the central §5 invariant the same
statement:

> **Inactive config modes must not influence execution.**

which is why it is written as a test before the consolidation, not after.

**Ownership constraint.** The canonical config is user-controlled; the standing
rule is *schema may change freely, behavior may not*. Consolidation therefore has
a hard acceptance criterion: **a run of `mk_dev.yaml` is bit-identical before and
after.** Any change that alters what an experiment does is a separate, explicit
ask.

---

## 2. What already exists

Roughly a third of the requested machinery is built. Building it again is the
main avoidable cost in this project.

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

**§8 gets re-measured before it gets resequenced.** A preprocessing pass has
already run and measured **1.38x** end-to-end, on the reading that the UMA
forward is >99% of the call. That is a prior, not a licence to skip A or D: it is
one measurement, on one route, and the code has moved since.

So the only change to §8's order is that **measurement comes first** — Phase 5.0
re-profiles the whole pathway, neighbour-list construction included and broken
out explicitly, and the sequence after that follows what the split actually
shows. The 1.38x figure is a hypothesis to re-test, not a conclusion to plan
around. Where it does still bear is on expectations: if the forward really is
>99%, then `crystal_inference_settings` (activation checkpointing, precision),
the `always_use_pbc` path and §8B's built-in execution modes are where the
headroom lives, and `energy/mace_host_frac` exists precisely to say whether MACE
splits the same way.

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

**Acceptance.** A short `mk_dev.yaml` run is **bit-identical** before and after
(same seed, same trace). Mode-safety tests pass, and each fails when its
invariant is deliberately broken — a test that cannot fail is not evidence.

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

One question is prerequisite to Phase 6 and should be answered here: **what
statistic does the cluster actually enforce on**, and how does it relate to
`gpu/util_policy`, `gpu/util_recent`, and nvidia-smi sampling? These are not
assumed interchangeable.

**Acceptance.** A small named set of canonical workloads, rerunnable by name,
with current baseline numbers recorded as graded findings.

---

### Phase 5 — MLIP optimization · **M/H**, gated on Phase 4

Numerical equivalence is a hard gate, and the existing `verify_*_equivalence`
harnesses are the pattern — extended to whatever diversity of structures the
change touches. Note the measured reward noise floor: UMA is not bit-reproducible
on GPU, so `torch.equal` is the wrong bar and the tolerance must be stated
against that floor.

**5.0 Re-profile the whole pathway first.** Not a formality and not gated on the
existing 1.38x reading. Break the energy call into preprocessing, neighbour-list
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

- [ ] one clear canonical production-config state
- [ ] production-config generation is a scripted workflow, not an agent search
- [ ] generated configs carry reliable provenance
- [ ] substantive code/config changes have semantic history
- ~~historical configs migrate systematically~~ — dropped: back-compat is not a
      requirement. A stale config fails loudly at load; that is sufficient.
- [ ] comments and docstrings follow current-state discipline
- [ ] the initial recurring workflows are scripted and documented
      (update-old-run · production-config generation · functional-change ·
      performance-investigation · run analysis)
- [ ] representative training modes have current end-to-end profiles
- [ ] MLIP bottlenecks addressed, numerical equivalence demonstrated
- [ ] A100 utilization behavior empirically understood
- [ ] batch sizing satisfies the utilization constraint at near-best throughput
- [ ] canonical performance and regression benchmarks exist
