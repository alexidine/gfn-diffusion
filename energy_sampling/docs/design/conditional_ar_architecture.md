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
4. **Option C, with consistent semantic ordering.** Build the SG step now, but do not test it yet. Converge on a
   single SG first.
5. **Give every row both s and p for now.** The encoder is a new model, not Mo3ENet.
6. **The conditional spline defaults equal the unconditional ones** unless there is a compelling reason. A small spline
   battery is advisable.
7. **The placing flow gets the symmetry operations directly.** The SG head may read them too. Standard settings and
   one consistent chart always apply.
8. **The op-set autoencoder pretraining is in.**
9. **Molecule scalars come from the featurization workflow.**

## 5. Agreed wiring, 2026-11-03 [proposed; supersedes §1 to §3 where they differ]

This is the build specification the owner agreed after three rounds. The sections above stay as the record of how it
was reached.

### 5.1 Encoder

- **Trunk (invariant).** mxtaltools `ScalarGNN` on atomic numbers and the radius graph, as built. Per-atom `h_i`. Its
  pool `s` is an `AugSoftmaxAggregation` (mxtaltools), replacing the learned-query attention pool.
- **Point stream.** A second `ScalarGNN` on the same graph. Node input is `[Z_i, h_i, MLP(x_i)]`, with `x_i` in the
  standard frame (`EmbeddingBlock` concatenates the continuous columns after column 0). Toggle `point_convs`: 0 gives
  today's per-atom MLP. The trunk runs first and never reads this stream, so `s` is invariant by construction. Its
  pool `p` is AugSoftmax. Per-atom point tokens stay available to cross-attention (`atom_tokens: split`).
- **Molecule scalars** go into the context as explicit invariants:
  - `V_mol` and `r_mol` come from the featurization workflow (`MolData.mol_volume` and `MolData.radius`, filled by
    `mol_analysis`);
  - the principal moments come from mxtaltools `batch_molecule_principal_axes_torch`, computed under `no_grad`
    (it adds noise to coordinates that require grad).
- **Chirality to the SG head.** `s` is mirror-blind, so the SG head also reads the allowed-group mask G_m and one
  pseudoscalar, a chirality index that changes sign under reflection.

### 5.2 Symmetry operations

- **Op features.** Per operation (M, m) in the standard setting, centring included:
  - M, as 9 integers;
  - m, as sin and cos of 2πm;
  - whether the operation is proper or improper (det M);
  - its rotation order;
  - its intrinsic screw or glide component;
  - a pure-translation flag.

  The largest Day group has 18 operations (R-3, hexagonal setting).
- **e_g.** MLP per operation, then an AugSoftmax pool. It feeds three readers:
  - the policy context `c`, so every slot sees it;
  - the SG head;
  - the level head. These read the static form only, so they stay frame-free.
- **Op tokens.** These join the atom tokens in the conditioner's cross-attention memory. They are refreshed when a
  block completes:
  - static, while the cell block is drawn;
  - after the cell: the Cartesian rotation `T M T⁻¹`, and the screw/glide length over `r_mol`;
  - after the position: the minimum-image vector to the image centroid, and its length over `r_mol`.

  Every refresh is a function of g and of completed blocks only, so the Jacobian stays triangular.
- **Toggles.** `sym_ops: none | pooled | tokens | tokens_dynamic`; and the pretrained encoder, `frozen | finetune | off`.
- **Pretraining.** A minimal set autoencoder over all 230 groups. The vocabulary is the distinct (M, m) operations
  across the standard settings. The decoder gives per-operation membership logits, trained with BCE. No energy data is
  involved.
- **The overfit test,** once training uses more than one SG: leave-one-group-out NLL for three arms, a pretrained e_g,
  an e_g learned from scratch, and a one-hot group label as the control.

### 5.3 Option C flow

- **Typed tokens.** About 14 quantity types: `L_a`, `L_b`, `L_c`, `cos α`, `cos β`, `cos γ`, `X`, `Y`, `Z`, the
  `X±Y` diagonals, `θ`, `φ`, `r` and `r_signed`. Each carries its kind (circle or interval) from the chart's wrap mask.
  `r` is an interval in the hemisphere chart; only `φ` is a circle in the orientation block.
- **Slot input.** Each slot adds together:
  - the type embedding of the coordinate it predicts;
  - that coordinate's allowed range given the prefix (`chart.bounds`), decoded into physical units;
  - the previous row's raw value and its decoded value;
  - `c`;
  - the prefix features of completed blocks.

  The decoded units are: lengths as `log(L / r_mol)`, angles as cosines, fractional coordinates with sin and cos for
  circles, and orientation angles.
- **g is slot 0.** Sequences are padded at the end, and log q sums over real tokens only.
- **Per-token spline settings** are looked up from the type. The spline defaults are `UNCONDITIONAL_DEFAULTS` on both
  routes: tied derivatives with F = 2, a floor of 1e-7, and bounded rows linear.
- **Block order is a toggle,** keyed by block name. The prefix-feature hook and the op-token stages are keyed by
  which blocks are complete, never by slot index.
- **Comparisons across orders use total NLL only.** Per-row NLL measures different conditionals under different
  orders.
- **Tests:** exactness (a triangular Jacobian, autograd log-det) and a round trip, for every layout and every order.

### 5.4 Prefix features

All are closed-form, about 15 numbers in total. None is tested.

- **After the cell block:**
  - `log(V_cell / V_mol)`;
  - `log(V_asu / V_mol)`, with `V_asu = V_cell / Z`; this is `−log c_pack`;
  - the sorted principal lattice lengths over `r_mol`;
  - `d₁₀₀`, `d₀₁₀` and `d₀₀₁` over the molecule's thinnest extent;
  - the shortest reduced-cell vector over its longest extent.
- **After the position block:**
  - the nearest image-centroid distance over `r_mol`;
  - the number of images within `2 r_mol`;
  - the distance to the nearest inversion centre, 2-fold axis and mirror/glide plane.
- **Per-group FD measure.** It is computed exactly from the chart's polytope (`crystal_fd/certify.py`) and checked
  against the group-theoretic count. It is offered to the SG and level heads only, never to the flow, where it is
  redundant with the cell rows. In polar groups it is not `V_cell` × a constant: the position block lacks the free
  axis.

### 5.5 Open: block order [observation that conflicts with §5.3's default]

The oct19 MIPCAS run (unconditional, latent chart, one seed) is recorded in agent memory, not in a frozen result. It
found that drawing the cell rows first was **worse** than `Z1_ORDER` (`8.11.10.9.7.6.5.4.3.2.1.0`: centroid z,
orientation, centroid y and x, then the cell last). The figures:

| order | ELBO (nats) | draws above 30 kT (of 40k) | P99.9 excess (kT) |
|---|---|---|---|
| cell first (`5.4.3.2.1.0.8.11.10.9.7.6`) | 34.59 | 144 | 95 |
| `Z1_ORDER` (pose first) | 35.34 | 99 | 44 |

Caption: unconditional MIPCAS on the latent chart, 3M energy calls, one seed each, 40k evaluation draws. Higher ELBO is
better.

`Z1_ORDER` is what `resolve_flow_settings` gives an unconditional one-block flow today. The cell-stage prefix features
and the cell-stage op tokens of §5.2 and §5.4 assume that the cell comes first. Under a pose-first order, the cell
slots come last and those features have nothing to inform. They would have to become pose-stage features read by the
cell slots. The default order is therefore an owner decision, not something to settle in the build.

Two further points bear on that decision:

- **`Z1_ORDER` also reverses the rows within each block.** It draws r before θ and φ, so the run above confounds block
  order with in-block order. On the FD chart only whole-block permutations keep `chart.bounds` causal: the bounds are
  autoregressive within a block (`charts.py:1290`), and the K-cut bound on r reads θ and φ (`charts.py:685`).
- **Under a pose-first order, every position-stage feature in §5.4 is inert too.** All of them need the cell metric.

### 5.6 Adversarial check, 2026-11-03: amendments to §5.1 to §5.4

An independent review checked §5 against the code. Its material findings were spot-checked in the code
(`spline_flow.py:572`, `ar_transformer.py:54-84, 170`, `eval/day_bench/protocol_random.py:24`). These amendments
supersede the text above:

1. **The level head reads g only when g is forced.**
   - Today the level input already includes the condition columns, SG among them (`spline_flow.py:574`). That is
     right while g is forced: the level is then log Z_g(c).
   - When g is drawn, a level that reads g absorbs log p(g | c) for each (c, g). The SG head's TB gradient then has
     zero mean, so the head learns nothing.
   - So in drawn mode the level reads s and the molecule scalars only, and e_g goes to the SG head only.
   - g is drawn from (1 − ε)·p_θ + ε·Uniform(G_m), so no group can starve without end; TB tolerates the off-policy
     draws.
   - The per-condition trackers key per molecule, not per (molecule, SG) (`buffer.py::ConditionLogZTracker`).
   - The FD measure is dropped: it is a function of (g, r_mol) only, so it adds nothing to either head.
2. **Stage-indexed memory.** The parallel pass holds every stage version of the op tokens in the memory at once, and
   masks each query to its stage with a `[b, 1, n, m]` mask. Today the mask is `[b, 1, 1, m]`
   (`ar_transformer.py:81`). The decode appends a stage's keys and values at the step that completes its block; today
   it projects the memory once (`ar_transformer.py:170`). Replacing the memory in place would leak later rows.
3. **The exactness test randomises the zero-initialised weights first.** These are `proj_x` and the feature
   projection; at their zero init a leak has no effect. The test then perturbs rows ≥ j and asserts that row j's
   spline parameters are bit-identical, in float64. A Jacobian check alone misses detached or non-differentiable
   features.
4. **Features are computed identically in both passes.**
   - They are computed in float64 from the float32-rounded, clamped prefix in both passes.
   - Counts are soft, and nearest-element choices are smooth.
   - The reason: draws are re-scored by the teacher-forced pass (`nf_policy.py:359`), so a feature that switches on
     rounding differences samples from a different q than the one scored.
5. **Prefix features only on the block that reads data-space x.** That is `blocks[0]`, or the hook refuses
   `n_blocks > 1`. The decode inverts the later blocks before data-space x exists (`spline_flow.py:645`).
6. **No after-cell rotation refresh.**
   - The FD charts fix the Cartesian frame of the cell, so T M T⁻¹ depends on g alone. Monoclinic b is along y
     whatever β is, orthorhombic and higher have fixed angles, and triclinic has only ±I.
   - The after-cell op tokens keep the translation parts in Å over r_mol: screw/glide lengths and the lattice
     translations.
   - If the orientation is drawn before anything that reads it, add an after-orientation refresh with RᵀQR, the
     image's orientation in the molecule's frame.
7. **The allowed-range input uses the physical range.**
   - `PolytopeARChart.bounds` returns [0, 1] (`charts.py:288`). The physical centroid range comes from `t_bounds`.
   - θ and φ have constant bounds, so the input carries information only on the cell rows, the centroid rows and the
     K-cut r.
8. **The type names follow the chart's coordinates.** These are `log_c`, `r2`, `r1`, `s_beta`, `s_gamma`, `v`, the
   monoclinic `s`, `log_a`, `log_b`, `t_x`, `t_y`, `t_z`, `t_x±y`, `θ`, `φ`, `r` and `r_signed`, keyed by
   quantity, not slot. The decoded physical value goes in beside the raw one; it is not a renaming.
9. **Molecule scalars need plumbing.**
   - The flow receives only z and pos (`nf_policy.py:271`), so `V_mol` and `r_mol` go through the atom table, on a
     log scale with their own scaler.
   - The extents come from `eigvalsh` of the heavy-atom gyration tensor, with the thinnest and longest extents
     included.
   - There are two radius conventions: `mol_analysis` uses heavy atoms; QM9 and `random_conformers` use all atoms. The
     FD length box reads one of them.
10. **Redundant features are dropped.**
    - `log(V_asu / V_mol)` differs from `log(V_cell / V_mol)` by a constant per group, so only one is kept.
    - The sorted lattice lengths and the shortest reduced-cell vector are already in the reduced FD prefix.
11. **Mixed-SG batches run as per-SG sub-batches.** Today NFPolicy takes one geometry from one probe
    (`nf_policy.py:140`). Wrap masks, dead-row pinning, stage boundaries and the chart all differ per group, and
    `fd_coordinates_rows` rebuilds the chart on CPU in float64 whenever radii differ. Splitting a mixed batch by SG
    keeps every per-chart path unchanged.
12. **Spline defaults.** No technical blocker. The test that pins conditional parity with `inn_modeller`
    (`tests/models/test_nf_policy.py:244`) and the run-settings section of `ar-spline-flow.md` change in the same
    commit. The `mk_dev.yaml` comments that describe the old behaviour are the owner's to edit. The per-row floor
    buffer becomes per-token.

### 5.7 Owner decisions and clarifications after the check, 2026-11-03

These supersede amendment 1 of §5.6 and the SG-head design of §2.2 and §2.3.

- **Option A: the level learns log Z(m, g) directly** (owner, 2026-11-03).
  - The level reads s, the molecule scalars, the static e_g, and a chirality input. Chirality is needed for the
    enantiomorphic groups, where Z(X, P4₁) = Z(X′, P4₃).
  - p(g | m) = softmax_g L(m, g), detached, is used for sampling.
  - There is no separately trained SG head, so there is no ε mixing.
  - In training, g is forced. The allocation is uniform over G_m, or scheduled. Each (m, g) is then an ordinary TB
    problem with x drawn on-policy from q(x | g, m). The allocation does not bias L, and every group is drawn by
    construction.
  - Drawing g from softmax(L) is for evaluation and late training only. A level that reads g together with a separate
    TB-trained head is the combination to avoid: TB would fix only their sum.
  - The caveat as before: while the flow is unconverged, TB's level sits near log Z(m, g) − KL_g. Reported SG weights
    come from importance sampling within each group.
- **Op tokens are embedded once, after g.** They are static. The stage-indexed memory of §5.6 item 2 is therefore not
  built. Geometric per-operation content is a later toggle.
- **New pathways use the default initialisation** (owner): every model trains from scratch. The no-peeking test sets
  random weights regardless.
- **Features are torch operations on the prefix in float64,** so autograd follows them:
  - polytope bounds as linear maps of the earlier rows;
  - a soft minimum over images;
  - counts as sums of sigmoids.

  The no-peeking test is the backstop for anything left non-differentiable.
- **The centroid allowed range comes from `PolytopeARChart.t_bounds`** (§5.6 item 7).

**For the owner** (not settled here):

- **The default block order** (§5.5).
- **Enantiomorphic closure of the Day groups.** The set holds 76, 96, 145, 154 and 169 but not their partners 78, 92,
  144, 152 and 170. So Σ_g Z_g differs between a chiral molecule and its mirror image, and a mirror-blind level head
  is misspecified. Options:
  - close the allowed set G_m under the partner map;
  - fix one enantiomer per molecule;
  - accept the asymmetry as part of the benchmark's declared convention.
- **The radius convention** that the length box and r_mol share: heavy atoms or all atoms. This is a data-thread
  interface, not this thread's call.
