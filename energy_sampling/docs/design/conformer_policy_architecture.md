# Conformer policy architecture — build plan

Status: **BUILD PLAN**, subordinate to [`conformer_conditional_stack.md`](conformer_conditional_stack.md)
§5, which is the architecture of record.

**Revised 2026-08-27 against code.** The architecture claims are unchanged and still hold; what
had drifted was build state and runtime. The drift has a precise cause worth recording: §4's
table was last written 2026-08-19 17:51, and `models/graph_encoder.py` landed at 22:43 the same
evening, so the row reading "GNN encoder — not started" was wrong about five hours after it was
typed. `conformer_modeller.py` then grew to 49 KB by 2026-08-24 while nothing in `docs/design/`
was touched again. §1 and §4 are rewritten; §2, §6 and §8 are new.

**What §5 owns:** the static/dynamic split, the GNN encoder and its selection criteria, n-body
correlators for `f_j`, augmented-softmax aggregation, the log Z head, capacity escalation, and
the chirality gates.

**What this file owns:** the constraint that the policy network is shared with crystal, the
wiring path into it, current build state, and owner decisions.

---

## 1. Where the code is

**CORRECTED 2026-08-27 — the previous §1 described a superseded route.** It said
`train_conformer.build_gfn` constructs the policy. It no longer does.

**The route is `conformer_modeller.py::ConformerModeller(Modeller)`** (`:62`) — train.py's own
protocol, buffers, stage machinery and checkpointer. `train_conformer.py` is the stripped
parallel loop it replaced, and the successor's docstring (`conformer_modeller.py:3`) says so.
This is §5 Sequencing step 1 still, but reached on the shared runtime rather than beside it.

**The policy is built through the shared config path.** `ConformerModeller._build_gfn_config`
(`:439`) calls `super()._build_gfn_config()` and adds one key: `angular_mask =
self.energy_function.periodic_dims`. The policy is still a flat `scalarMLP` over
`[lin | sin | cos]` with `dim = data_ndim`, so the three consequences the earlier draft named
still stand — width tied to one molecule, graph never reaching the policy, arbitrary
spanning-tree storage order treated as meaningful.

**`TorsionGFN` is retired, and the mechanism that replaced it is load-bearing.** `GFN` now takes
an explicit `angular_mask` (`models/gfn.py:43`, consumed at `:291`). The non-crystal branch of
`get_periodic_dimensions` writes `[False] * dim`, which on a conformer is not a degraded layout
but a *silently unnormalizable target*: the phi block is 2-periodic, and a reward with no wrap
has no finite log Z at all. `_build_gfn_config`'s docstring records this, and omitting the key
would reintroduce the bug through the back door.

**Verified — the constraint §5 does not cover.** `GFN` is shared. `train.py::_build_gfn_config`
(crystal) and the conformer override construct the same class, and `checkpointing.py` rebuilds
it from a stored `gfn_config`. Crystal is priority 1 in `AGENTS.md`. So the conformer policy
must be an **additive path**, and nothing may change `_fwd_step`, the variance schedule, DPLR
covariance, dead latent rows or periodic centroids without an explicit decision.

**Verified.** The trajectory hot loop is still geometry-free — `_fwd_step` →
`predict_next_state(s_emb, t_emb)` (`models/gfn.py:622`) is pure MLP. §5's cost argument
describes a property the code already has; the replacement must preserve rather than achieve it.

**STALE CLAIM RETIRED.** Earlier drafts called `ConformerTorsions`' Modeller-protocol block
unreachable code. It is reachable: `condition_samples` is called from `conformer_modeller.py:499`
and `:613`.

---

## 2. The blocker — molecular identity has no channel into the policy

New section. This sits upstream of every piece in §4 and is why none of them can be read yet.

`ConformerTorsions.condition_samples` (`energies/conformer_torsions.py:1531`) builds the
condition vector from **log-temperature, or a single zeros column** when
`temperature_conditioning` is false. `mol_id` reaches only `condition_id`, which the policy
never reads. Every other piece could land and the policy would remain molecule-blind.

Both live conformer configs are `temperature_conditioning: false` under
`protocol: conformer_unconditional`, so **no conditional axis is exercised today, on any
molecule.**

A second constraint sits in the same place. `build_conformer_conditions.py` writes a genuine
multi-molecule condition set, but its own docstring states that every molecule in one file must
have the same number of rotatable torsions, because the GFN's state dimension is fixed at
construction — **one file is one `k`**. `SetPolicy` is what lifts that, and `SetPolicy` is
unwired (§4, §5).

---

## 3. Owner decisions — settled 2026-08-19, still standing

1. **Go straight from the unconditional MLP to a learned GNN.** No handcrafted-features stage on
   the critical path. The GNN's expressiveness is to be confirmed first on toy self-supervised
   problems — leave-one-out node identification, simple molecular-graph property prediction — to
   establish the architecture is functional before it is asked to carry a policy. Full
   pretraining by that route is a live possibility, not a commitment.
2. **Parity is the bar** for the architecture swap. No reason to expect a set model to beat a
   flat MLP on one molecule.
3. **No changes to shared trajectory machinery** without an explicit decision, per §1.
4. **Evaluation ladder:** unconditional runs on a series of challenging molecules first, then a
   large conditional set over growing subsets — 1, 2, 100, N→∞.
   *Owner note, and it re-frames the ring work:* the policy outputs the full internal parameters
   directly, so ring closure is not a constraint the model must satisfy structurally — it either
   learns the distribution or it does not. The prior still has to help with search. So the ring
   machinery matters for the **proposal**, not for the policy's expressiveness.

---

## 4. Build state

Status of each piece of §5's architecture as of 2026-08-27, established by reading the files in
the evidence column rather than by running anything. "Built" means the module exists and imports;
it does not mean any training run reaches it. The passthrough shipped 2026-08-27, so the set
policy CAN now be reached — but no config selects it yet, so **no row has been exercised in a
training run**, and the test counts pin construction, shape and contracts rather than behaviour
under training.

| Piece | Status | Evidence and note |
|---|---|---|
| `energies/dof_features.py` | built, 10 tests | `tests/conformer/test_dof_features.py`. Handcrafted `f_j`; **demoted by decision 1** to a control. Carries `atom_parity` (`:188`), the chirality pseudoscalar — so A3's feature is not from scratch. |
| `models/set_policy.py` | built, 8 tests | `tests/models/test_set_policy.py`. Parameter count independent of `d`; `set_policy_for` builds against `energy.periodic_dims`. **Not wired into `GFN`.** |
| `models/graph_encoder.py` | built, 13 tests | `MPNNEncoder`: per-atom `h` plus augmented-softmax global `g`. **Attention half added 2026-08-27** (`SPDAttention`, Graphormer-style learned bias per shortest-path bucket, GPS-shaped residual per layer) so `attention=False`/`True` are the battery's two arms. `tests/models/test_graph_encoder.py`. |
| `models/graph_encodings.py` | **new 2026-08-27**, 20 tests | RWSE and LapPE, plus every observable the battery scores against — one module on purpose, so a probe and its feature cannot drift apart. `tests/models/test_graph_encodings.py`. |
| `models/encoder_ssl.py` | **new 2026-08-27**, runnable | The battery itself: 4 arms × 12 probes in 4 blocks, size-only baselines, orbit ceiling, determinism gate. `python -m models.encoder_ssl`. |
| n-body correlators `F_τ` | not started | Referenced only in docstrings. The last piece between the encoder and `{f_j}`. |
| log Z(c) head | not started | No `Z_MLP` or equivalent anywhere in `models/`. §5 requires an **extensive** aggregation; a mean pool discards the size dependence log Z needs. |
| Toy self-supervised validation | not started | Decision 1's gate on the encoder, before any policy training. |
| Wiring into `GFN` | **SHIPPED 2026-08-27**, 5 tests | `predict_next_state` now takes the raw state — §5. `tests/models/test_policy_raw_state_passthrough.py`. |
| Molecular identity channel | not started | **The blocker, §2.** Upstream of every row above. |

---

## 5. Wiring path — SHIPPED 2026-08-27

`predict_next_state(s_emb, t_emb, state=None)` now takes the raw pre-expansion latent, and
`_forward_kernel` passes it. Duck-typed on the policy declaring `wants_raw_state`, so the
crystal construction path is untouched and every existing two-argument caller still works.

Two refusals ride with it, both deliberate:

- **`wants_raw_state` with no `state` raises.** Falling back to `s_emb` would train a
  wrong-but-plausible policy rather than fail.
- **`wants_raw_state` with `dplr_rank > 0` raises.** The layouts genuinely disagree:
  `SetPolicy._to_blocks` emits the low-rank factor as rank-major blocks while `split_params`
  reads it `.view(-1, dim, rank)`, i.e. dim-major, so `u_raw` would be **silently
  transposed**. Both live conformer configs currently set `dplr_rank: 6`, so this is a live
  constraint, not hypothetical — either run the swap at `dplr_rank: 0` or teach the policy
  the DPLR block order first. **That is the next piece of work on this path.**

**Verified by:** `tests/models/test_policy_raw_state_passthrough.py` (5 gates, including a
bitwise-identity guard that the argument is inert on the flat path); the bug it exists to
catch was injected and 2 of the 5 fired; `tests/crystal/` + `tests/models/` +
`tests/losses/test_z_gradient_isolation.py` — **148 passed**, no regressions.

What remains before the swap can run: a config key selecting the set policy over
`scalarMLP`, which is the next diff and does not touch shared code.

---

## 6. Which encoder is which

New section. The two are easy to confuse, and confusing them costs the chirality argument.

**`E_GNN` — the scoped one — is `models/graph_encoder.py`.** A 2D bond graph with `edge_attr`
and no geometry at all. Its docstring cites §5 and states outright that no equivariant machinery
is required.

**`VectorMoleculeGraphModel` is NOT the scoped one.** It is the crystal conditioner, imported
from MXtalTools at `models/gfn.py:11` and constructed at `:187` behind
`conditions_type == 'molecule'`: a radial graph over positions, `cutoff: 6.0`, `num_radial: 32`,
`max_num_neighbors: 100`, vector norms, `concat_pos_to_node_dim=True`. Two structural reasons it
cannot be reused here:

1. **It reads `pos`, and for a conformer `pos` is the sample.** Feed it the sampled geometry and
   the policy conditions on the answer; feed it the reference conformer and it inherits the
   chart defect in §7, so the condition stops being a function of the labelled graph.
2. **It is O(3)-invariant, not SO(3).** Reflection is in the group, so it is mirror-blind
   regardless of how much geometry it sees — the one axis where a 3D model would otherwise be
   expected to win. The construction site's own comment says "o(3) invariant".

The 2D route recovers that cheaply: internal coordinates are already invariants and the outputs
are scalars, so the policy is SE(3)-invariant by construction, leaving chirality as the only
asymmetry and `atom_parity` as its carrier.

---

## 7. Dependencies outside this file

Three items in §6 Known gaps bear on the conditional stages and are not this file's to fix. All
three are still open as of 2026-08-27.

- **`log |dq/dx|` is missing from the reward.** Constant in `x`, so unconditional results are
  unaffected, but it varies by 9.8 nats across the measured molecules, which makes `log Z(c)`
  non-physical and cross-condition comparison invalid. **Blocks conditioning.** The fix is a
  per-condition constant recorded as a reporting attribute, not a term added to `energy()` —
  see the sprint plan's Track A1 for why the latter does not typecheck at the shipped level.
- **The chart is not yet a function of the graph alone.** `r0`/`θ0`/`φ0` are serialised into the
  condition and move 0.0086 Å / 0.20 rad / 3.14 rad across embedding seeds, carrying the
  reward's own `e_ref` by 0.245–1.835 kT. The *condition itself* is not reproducible.
- **Parity must enter the condition schema**, not merely `dof_features`. This is the hard
  dependency on the encoder path: a 2D graph plus atom types is identical for enantiomers, and
  nothing currently fails if parity is absent.

---

## 8. Sequencing

New section — §5's ladder restated with the current position marked, because the ordering is
what makes a regression attributable to one cause.

1. **Unconditional, fixed-dim MLP, one molecule.** ← where the stack is.
2. **Swap the policy for the set architecture, still one molecule at a time**, so any regression
   is attributable to the architecture rather than the conditioning. Bar is **parity**
   (decision 2).
3. **Condition over a molecule set**, watching held-out evaluation, which catches what training
   metrics hide.

The self-supervision battery that gates the encoder is specified in
[`encoder_ssl_battery.md`](encoder_ssl_battery.md) and implemented in
`models/encoder_ssl.py`.

Step 2 before step 3 is deliberate, and it is the cheaper of the two: `set_policy.py` is built
and tested, and the swap needs only §5's optional-argument passthrough. The §2 blocker is step
3's prerequisite, not step 2's.

---

## 9. Out of scope

Conditional training dynamics, replay, the flow head, the variance schedule, and the stage
protocol. Folding any of them in makes every result ambiguous about which change caused it.
