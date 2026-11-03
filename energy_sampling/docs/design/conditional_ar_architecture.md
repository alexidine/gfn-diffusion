# Conditional AR flow: conditioner, space-group step and prefix features (proposal, 2026-11-02)

Status: proposal. Part 1 (the split encoder and the condition pass-through) is built and tested. Parts 2-4 are
designs. The owner's choices in this document were given in the architecture thread on 2026-11-02 and 2026-11-03
(§4). The diagram of what is built and planned is `conditional_ar_architecture.html` beside this file. Everything else
here is a recommendation, and claims carry their evidence tag: **[built]** code and tests in the tree, **[run]** measured,
**[derived]**, **[proposed]**.

Sources: the review reports of 2026-11-02 (`trace.md`, `critic_conditioner.md`, `critic_process.md`, `sg/discrete_sg_design.md`,
`sg/refutation.md`, `zc/zc_map.md`, in the hub session's scratchpad), the owner's slide `architecture.pptx`, and
`eval/nf_fit.py` (the fit/overfit harness, deliverable 1).

## 0. What the owner decided, and what follows

- **One shared condition embedding, computed once per rollout.** It is pooled first, with per-atom tokens kept as an
  option.
- **Two kinds of molecule feature.**
  - An invariant `s`, from distances and elements only. `log Z` and the SG choice read `s` (plus the SG, detached).
  - A point embedding `p`, which depends on the standard frame. The policy reads `s`, `p` and the SG.
  - Rule: `s` may feed `p`; `p` never feeds `s`.
  - In a strict sense only the orientation rows need `p`. The cell and centroid rows (a, b, c, α, β, γ, u, v, w)
    should depend on the molecule only through invariants. The policy is not split by row for now: one shared
    embedding feeds every row, and this is kept in mind.
- **The SG is a discrete first step drawn by the model**, `log Z_g(c) = log Z(c) + log p(g | c)`, with an option to
  force or skip it. The SG head reads `s`.
- **Layout: option C.** One AR sequence of variable length, with a coordinate-type embedding instead of a slot index,
  and the SG as the first token.
- **SG weights are a declared fixed-symmetry convention, not physical free energies.**
- **SG weights should be estimated during training, with care** (§2.3).

## 1. The condition encoder [built]

`models/spline_flow.py::SplitAtomEncoder` is selected by `model.nf.atom_tokens: split` or `split_pooled`, in molecule
conditioning.

- **Trunk.** mxtaltools `ScalarGNN` over atomic numbers alone (`input_node_dim=1`), on the radius graph within the
  molecule. Geometry enters only through interatomic distances, through the radial basis. Its node features `h_i`
  are therefore invariant to rotation and reflection, and so is their attention pool `s`.
- **Point head.** An MLP over `[h_i, MLP(x_i)]`, with `x_i` the standard-frame coordinates. It gives per-atom point
  tokens and their attention pool `p`. The head reads `h_i`; nothing reads the point head back into `h_i` or `s`.
- **Readers.**
  - The level head (`log Z`) reads `[s, condition columns]`, detached, as before.
  - The context MLP reads `[s, p, condition columns]`. Its output is added to every slot's token.
  - `split` also cross-attends to the point tokens in every causal layer. `split_pooled` does not: the conditioner
    is built without cross-attention, and the embedding is the one pooled vector, computed once.
- **The condition columns no longer drop in molecule mode.** `NFPolicy._context` used to return the row ids alone
  (`cond_dim = 1`), so on the atom route the SG, temperature and Z' columns never reached the flow (review trace
  §0.4). It now returns `[id, standardised condition columns]`.
  - The columns are standardised by the flow's own `c_mu`/`c_sd`, fitted by `fit_data_scalers` on the conditions
    files. The train.py hook now passes conditions on every conditional route.
  - The id column is never standardised. The shared `_ContextFlow` still takes those columns as given, so
    `inn_modeller.py`'s atom route is unchanged.
  - A molecule-mode call without its condition columns now raises.
- **Tests** (`tests/models/test_nf_split_encoder.py`, fast; 66 passed together with `test_nf_policy.py`,
  `test_nf_fit_rows.py`, `test_inn_atom_tokens.py` and `test_nf_xcond.py`):
  - `s` and the level head's input are unchanged under a proper or improper rotation of the molecule, to 1e-5;
    `p` and `log q` change;
  - swapping the condition columns moves `log q` and the level input in all three atom modes;
  - a draw and its re-score agree;
  - the per-row terms of `eval/nf_fit.py` sum to `log q`;
  - an unknown mode is refused.
- **Checkpoints.** An atom-token checkpoint with no condition columns loads unchanged. One with condition columns now
  has a wider context input; none such exists on the train.py route.
- **Not yet built.** Making the cell and centroid rows read `s` alone (owner: keep in mind). A pretrained trunk:
  Mo3ENet's role is now this trunk plus point head, trained our own way (deliverable 4).

The harness gives the test this encoder must pass. On a 600-step smoke run with the frozen Mo3ENet vector route
(1,105 QM9 molecules, one seed, not a finding) **[run]**:

- the transfer ratio of the conditioning gain was ≈0.9 on held-out basins and ≈0 on held-out molecules;
- a free context per molecule recovered 1.9 nats on held-out molecules against 0.1 on training molecules;
- the held-out excess sat in the cell-length rows.

So the encoder's placement of new molecules is the failure to beat. The comparison run is the same harness on
`atom_tokens: split_pooled` against `true` against the vector route, at matched steps.

## 2. The space-group step

### 2.1 Layout: option C, and what it takes beyond an embedding [proposed]

The owner reads option C as "just a type of positional embedding". For the conditioner that is right, and causal
attention makes the variable length cheaper than it looks:

- **Coordinate-type embedding.** Replace the per-index `slot_emb`/`dim_emb` (`energies/ar_transformer.py`) with
  embeddings keyed by the FD coordinate's name: about 23 names across the 26 Day groups (`log_c`, `t_x`, `theta`, ...).
  The same name means the same thing in every group, so the orientation tokens share weights across 24 groups.
- **SG token.** Slot 0 is a learned embedding of `g`, in place of today's `start` vector. Every later slot reads it.
- **Variable length, padded at the end.** Under a causal mask a real token never attends to a later position, so a
  sequence padded after its last coordinate needs no extra attention mask. Its padded outputs are dropped. The KV-cached
  decode runs to the batch's longest chart, and short rows ignore their tail.

Three things outside the embedding must also change, all elementwise:

- **Per-token spline kind.** Circle or interval, bound, and standardiser are buffers per row index today
  (`_ContextBlock`/`_ContextFlow.is_wrap`, `is_bounded`, `unit_base`, `mu`, `sd`). They become per-token tensors
  looked up from the coordinate type. `_rq_spline` is elementwise already.
- **Input features.** `feat_w` maps feature index to width; a circle contributes sin and cos. It becomes a
  per-coordinate-type projection.
- **log q** sums over the real tokens only, with padded tokens masked out of the log-det and base density.

Estimate: a day for the flow plus its tests, of which the decode and the masked log-det need the most care. One
exactness test per layout pins it: a triangular Jacobian, autograd log-det against the flow's own, and a round trip,
as the trace probe did per SG.

Dependency outside this thread: the FD energy route accepts one SG only (`molecular_crystal.py`). The search/prior
session owns that fix, and until it lands a mixed-SG batch cannot be scored. The flow can be built and tested on
synthetic targets first.

### 2.2 The SG head [proposed]

`p_θ(g | s)` is an `_ResMLP` from `s` to logits over the allowed groups G_m. Logits outside G_m are masked; G_m
depends on the molecule's chirality class. Its input is `s`, so it is invariant, as `log Z` is. The draw is g first,
then the flow given the SG token. `log P_F = log p(g | s) + log q(x | g, m)`, still a length-1 trajectory, so the
losses need no change.

### 2.3 Getting Z(g | m) right during training [owner chose TB, 2026-11-03; the alternative below is in reserve]

The owner chose to train the SG head by TB. The head then learns `Z_g · e^{−KL_g}`, so reported SG weights come from
per-group importance sampling, never from the head. The rest of this section is the alternative that was not chosen,
kept in case the TB head starves a group.

**The problem.** If the head is trained by TB, its optimum is `p(g) ∝ Z_g · e^{−KL_g}` on-policy, where `KL_g` is
how badly the flow fits group g. On stored rows the bias reverses, and under MLE the head copies the search's per-SG
counts. The review's toy, one seed:

| head trained by | head p(g=0) | true p(g=0) |
|---|---|---|
| on-policy TB | 0.862 | 0.833 |
| TB on data rows | 0.281 | 0.833 |
| MLE on a 30/70 dataset | 0.300 | 0.800 |

Per-group importance sampling gave 0.834 in both TB cases. The head is a fine proposal and a biased readout.

**Alternative (not chosen): train the head to the IS estimate, not by TB.** Four pieces, all inside the training
loop:

1. **Draws.** g is drawn from `(1 − ε) p_θ(g | s) + ε · Uniform(G_m)`, so every group keeps getting draws whatever
   the head believes. Then x is drawn from `q(x | g, m)`. Each draw gives `log w = log R(x; g, m) − log q(x | g, m)`.
   It does not involve `p(g)`, so how g was drawn does not bias it.
2. **An amortised per-(molecule, SG) normaliser** `F(s, g)`, a head reading `s` and the SG embedding, trained on those
   `log w` with the loss `exp(log w − F) + F`. Its minimiser per condition is `F = log E_q[w] = log Z_g(m)`, the
   importance-sampling estimate itself, not the Jensen bound that a regression of `log w` gives
   (`d/dF: −E[w] e^{−F} + 1 = 0`) **[derived]**. Because it is amortised over molecules, it needs no per-condition
   tracker. At about 10⁵ molecules a per-condition tracker cannot work (`zc_map.md` §0.2).
   - `log w` is heavy-tailed, so the loss needs clipping or a defensive mixture (point 4).
3. **The head is distilled from F.** Cross-entropy of `p_θ(· | s)` against `softmax_g F(s, g)`, detached. The head
   never sees TB, so it learns `Z_g`, not `Z_g · e^{−KL_g}`. The flow's TB residual per row uses `F(s, g)`, detached,
   as its level. That is `log Z_g(c) = log Z(c) + log p(g | c)` with both terms coming from F.
4. **Coverage guard.** `E_q[w]` misses basins `q` drops: synthetic −0.33 nats at an ESS fraction above 0.84
   (`zc_map.md` §0.6). A fraction α of draws therefore comes from a broad proposal `r` with a density, such as kernels
   around the basin set, and `log w` uses the mixture density `α r + (1 − α) q`. That removed the bias in the synthetic
   check (−0.002 nats).

The reported SG weights at the end are a separate, larger per-group IS run with ESS and bootstrap intervals, read
against F and the head. Agreement is the check that the training-time estimate worked.

Bookkeeping the refutation fixed, to apply once:

- `κ_g` sits inside `log R`, so the weights are `Ẑ_g / Σ Ẑ`, not `κ_g Ẑ_g / Σ`.
- The flat-molecule twin factor is 2 in non-Sohncke groups and 1 in Sohncke groups. It does not cancel.
- `energy_reference: seed_min` must key its zero per molecule, not per (molecule, SG). Otherwise each group carries
  its own `E_ref / T` offset.

### 2.4 Order of build, smallest first

The owner's plan (2026-11-03) is to build the whole architecture before serious testing, since conditional
training needs time and data that local smoke runs do not give.

1. The option C flow with the SG given, checked on a synthetic two-layout target for exact density and round trip.
2. The SG head, trained by TB, with force and skip.
3. The prefix-feature hook (§3), which unconditional runs can exercise early because they converge faster.
4. Serious testing on the cluster, once the FD energy route scores mixed-SG batches (search/prior session).

## 3. Prefix physics features (deliverable 5) [proposed]

**Idea.** Give later AR steps a physical function of the rows already drawn. The owner's cheapest test is the
packing coefficient after the cell rows.

**Why it is legal.** The conditional of row j may depend on any function of rows `< j`. A feature computed from the
cell rows only, and added to the tokens of the slots after the cell block, keeps the Jacobian triangular and the
density exact. It does not need to be differentiable in x; it is an input, not a transform.

**How the packing coefficient is computed.** `c_pack = Z · V_mol / V_cell`. `V_cell` is a function of the cell rows
alone in both charts: in FD the cell coordinates come first and their bounds depend only on earlier cell
coordinates. One function therefore serves both passes:

- **density pass (teacher forced):** decode the full row x with the trainer's own decode (`fd_rows_to_params` or
  `latent_to_cell_params`), take `V_cell`, and compute `log c_pack` per row. Molecule volume comes from the batch
  (`mol_volume`), `Z` from the SG.
- **sampling (decode):** after step `n_cell`, fill the pose rows with any in-box value (they do not enter `V_cell`)
  and call the same function.

A unit test asserts that the feature is unchanged when the pose rows are randomised, which guards the
"cell rows only" claim on each chart.

**Plumbing.**

- `CausalTransformerConditioner` takes an optional `[B, k]` prefix feature and its start slot. A zero-initialised
  `Linear(k → W)` is added to slots `≥ n_cell`, so a new feature starts as the identity.
- `invert_block_cached` takes a callback that the decode calls at step `n_cell`.
- The feature provider is installed on `NFPolicy` by the trainer, as the drift force is, because it needs the
  energy function's decode and the batch.
- About 150 lines with tests.

**What it can show, and the test.**

- The flow already sees the log-length rows linearly. What `c_pack` adds is the molecule-dependent normalisation by
  `V_mol` and the angle factor. A positive result says the cell-to-pose conditional benefits from an explicit density.
- Test: unconditional MIPCAS, then conditional QM9 at the harness's scale, with and without the feature, 2 seeds,
  matched steps. Read:
  - held-out NLL of the pose rows, from the per-row table;
  - excess energy of draws;
  - transfer ratio.
- Retire it if the pose-row NLL and the excess energy do not move beyond seed spread.

**Later candidates**, each gated on the first showing anything:

- after the pose rows: centroid distances of symmetry images, and contacts of the molecule's inertia ellipsoids;
- SG symmetry operations as tokens, which fits option C.

## 4. Owner decisions, 2026-11-03

1. **The SG head is trained by TB** (§2.3).
2. **Build the full architecture before serious testing** (§2.4). Mid-AR features such as the packing coefficient may
   be tried earlier on unconditional runs.
3. **Old checkpoints are not used again**, so nothing keeps them loadable.
