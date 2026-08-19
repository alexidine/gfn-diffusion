# Conformer policy architecture — build plan

Status: **BUILD PLAN**, subordinate to [`conformer_conditional_stack.md`](conformer_conditional_stack.md)
§5, which is the architecture of record. Revised 2026-08-19 after reading that section
properly; the first draft of this file restated it more coarsely and diverged from it in two
places, both now corrected in code.

**What §5 owns:** the static/dynamic split, the GNN encoder and its selection criteria,
n-body correlators for `f_j`, augmented-softmax aggregation, the log Z head, capacity
escalation, and the chirality gates.

**What this file owns:** the constraint that the policy network is shared with crystal, the
wiring path into it, current build state, and owner decisions.

---

## 1. Where the code is

**Verified this session.** `train_conformer.build_gfn` constructs the shared
`models/gfn.py::GFN` with `conditions_dim=0`, `conditional=False`, `dim = data_ndim`, and a
flat `scalarMLP` policy over `[lin | sin | cos]`. Three consequences: the policy width is
tied to one molecule, the molecular graph never reaches it, and the arbitrary spanning-tree
storage order is treated as meaningful. This is §5 Sequencing step 1, which that document
already identifies as where the stack sits.

**Verified — the constraint §5 does not cover.** `GFN` is shared. `train.py::_build_gfn_config`
(crystal) and `train_conformer.build_gfn` construct the same class, and `checkpointing.py`
rebuilds it from a stored `gfn_config`. Crystal is priority 1 in `AGENTS.md`. So the
conformer policy must be an **additive path**, and nothing may change `_fwd_step`, the
variance schedule, DPLR covariance, dead latent rows or periodic centroids without an
explicit decision.

**Verified.** The trajectory hot loop is already geometry-free — `_fwd_step` →
`predict_next_state(s_emb, t_emb)` is pure MLP. §5's cost argument therefore describes a
property the current code already has, and the replacement must preserve rather than achieve
it.

---

## 2. Owner decisions — settled 2026-08-19

1. **Go straight from the unconditional MLP to a learned GNN.** No handcrafted-features
   stage on the critical path. The GNN's expressiveness is to be confirmed first on toy
   self-supervised problems — leave-one-out node identification, simple molecular-graph
   property prediction — to establish the architecture is functional before it is asked to
   carry a policy. Full pretraining by that route is a live possibility, not a commitment.
   *This partially resolves §5's "Pretraining the encoder — OPEN": the self-supervised tasks
   enter first as a **diagnostic**, and only then as a candidate pretraining target.*
2. **Parity is the bar** for the architecture swap. No reason to expect a set model to beat
   a flat MLP on one molecule.
3. **No changes to shared trajectory machinery** without an explicit decision, per §1.
4. **Evaluation ladder:** unconditional runs on a series of challenging molecules first,
   then a large conditional set over growing subsets — 1, 2, 100, N→∞.
   *Owner note, and it re-frames the ring work:* the policy outputs the full internal
   parameters directly, so ring closure is not a constraint the model must satisfy
   structurally — it either learns the distribution or it does not. The prior still has to
   help with search. So the ring machinery matters for the **proposal**, not for the
   policy's expressiveness.

---

## 3. Build state

| Piece | Status | Note |
|---|---|---|
| `energies/dof_features.py` | built, 7 tests | Handcrafted `f_j`. **Demoted by decision 1** to a baseline/control, not the path. |
| `models/set_policy.py` | built, 11 tests | The dynamic head. Parameter count independent of `d`. Not wired into `GFN`. |
| GNN encoder + n-body correlators | not started | §5 "Static molecular encoding" / "Static internal-coordinate embeddings". |
| Toy self-supervised validation | not started | Decision 1. Gates the encoder before any policy training. |
| Wiring into `GFN` | not started | Needs `predict_next_state` to receive the raw state. |
| log Z(c) head | not started | §5 "The log Z head". Blocked on a known gap — see §5 below. |

### Two divergences from §5, found and corrected

- **Aggregation was mean-and-sum; §5 specifies augmented softmax.** The extensive half was
  right; the selective half was a plain average, which cannot emphasise the one strained
  coordinate that should dominate a step. Corrected in `models/set_policy.py`. The failure
  was invisible on a fixed molecule — both variants train.
- **`f_j` was handcrafted.** §5 derives it as `f_j = F_τ(g̃_{i1}, …, g̃_{in}, e_j)` — n-body
  correlators over per-atom GNN embeddings, one correlator per DoF class. Superseded by
  decision 1; the handcrafted version survives only as a control.

### A gap in what is built

`dof_features` carries **no chirality feature**. §5 is explicit that the 2D graph plus atom
types is identical for enantiomers, so any encoder over it is enantiomer-blind unless parity
enters as an atom feature — and §6 records that nothing currently fails if it is absent. The
handcrafted featurizer inherits that blindness. It matters for the GNN path, not the
demoted one, but the gates in §6 should exist before the encoder is trained rather than
after.

---

## 4. Wiring path

`_forward_kernel` computes `expanded_state`, `s_emb`, `t_emb`, then calls
`predict_next_state(s_emb, t_emb)`. A set policy needs the **raw state**, which is gone by
then. The minimal additive change is to pass it through as an optional argument, leaving the
flat path untouched when it is absent.

That is the one shared-code edit currently anticipated, and per decision 3 it wants its own
reviewable diff rather than arriving inside a feature branch.

---

## 5. Dependencies outside this file

Two items in §6 Known gaps bear directly on the conditional stages and are not this file's
to fix:

- **`log |dq/dx|` is missing from the reward.** Constant in `x`, so unconditional results
  are unaffected, but it varies by 9.8 nats across the measured molecules, which makes
  `log Z(c)` non-physical and cross-condition comparison invalid. **Blocks stage D**, and
  the fix is a per-condition constant.
- **The chart is not yet a function of the graph alone.** Linearity is measured off a
  reference conformer and changes `d`, so the same molecule can get different charts from
  different embedding seeds. This is the same class of defect found independently in the
  featurizer work — the reference dihedral differs by up to 2.08 rad between two SMILES
  orderings of one molecule, which is why it was dropped from `f_j`. §6 names the fix:
  MMFF's typed θ₀ ≥ 179.99, graph-determined and already computed.

---

## 6. Out of scope

Conditional training dynamics, replay, the flow head, the variance schedule, and the stage
protocol. Folding any of them in makes every result ambiguous about which change caused it.
