# Pretrained molecule conditioner: a SplitAtomEncoder autoencoder (scope, 2026-11-08)

**Status: proposal, not policy.** Nothing here is built or trained. It scopes an autoencoder whose encoder is the
current conditioner, `models/spline_flow.py::SplitAtomEncoder`. The trained encoder would replace the frozen Mo3ENet
as a pretrained molecule conditioner for the conditional AR flow policy (`models/nf_policy.py`). Code facts cite
file:line at GFN `feecbf66` (working tree) and MXtalTools `8f557b3b`. Claims carry tags: **[code]** read in the
source; **[check]** a CPU check of a few seconds run for this document (random weights, toy molecules; scripts in the
session scratchpad, not in the repo); **[derived]**; **[proposed]**; **[lead]** a navigation lead that was not
re-verified.

**Amended 2026-11-08 after an adversarial review** (notes: `conditioner_ae_review_2026-11-08.md` in the session-family
notes folder, outside the repo; check scripts in the session scratchpad `review/`). Corrected in place:
- the proper-frame formula (§3.1);
- the symmetry obstruction and augmentation rule (§2.3);
- the SVD gradient hazard (§3.1);
- the consistency term's claims and distance (§3.3);
- a decoder initialisation rule (§3.2);
- the chirality bit, now dropped from the first experiment (§2.2, §7.3);
- the eval-mode wiring of the learned frame (§7.1);
- the first experiment's controls and seeds (§7.2).

Checks quoted from the review are tagged **[review check]**.

Why now (owner, 2026-11-08). On the norbornane P-1 set (117 training molecules, 35 held out), the molecule
conditioner learns molecule identity when it is co-trained with the flow. At matched training fit (train mean
−log q = −2 nats) the held-out-molecule gap is 0.66 nats with about 7 molecule scalars as the condition, and 0.80–0.85
with graph encoders. Co-trained `rotate_frames` made the gap 1.5–2.5× larger. The owner concluded that a pretrained,
frozen conditioner separates the embedding from the policy's memorisation.

The proposal in one paragraph **[proposed]**:
- **Keep SplitAtomEncoder's invariant trunk as the scalar channel `s`.** It is exactly invariant by construction.
- **Give the encoder an exact polar-plus-pseudovector readout and reduce it to one molecular frame F in SO(3)** by the
  special Procrustes projection. That is three rotational degrees of freedom.
- **Decode as F · D(s).** D(s) is a canonical-pose point cloud built from `s` alone.
- **Train with random rotations and a decoder-free consistency term** over rotated copies.
- **Downstream, the flow reads frozen `s`.** It models orientation relative to the learned frame, so the policy needs
  no vector input at all. The frame is precomputed per molecule and applied in every mode (§7.1).

The sections below argue each step.

---

## 1. Code facts

### 1.1 Mo3ENet (MXtalTools, read only)

**Architecture [code].**
- **Encoder.** `Mo3ENetEncoder` (`mxtaltools/models/task_models/autoencoder_models.py:599`) wraps
  `VectorMoleculeGraphModel`. Its node input is Z plus |x_i| (`concat_pos_to_node_dim=True`, :624), and each node
  starts with a single vector channel, its centred position.
  - Scalar and vector tracks run side by side through a `VectorGNN`. Vector messages are bias-free channel mixes of
    the two endpoint vectors, scaled by radial invariants (`graph_convolution.py:143`, `v_MConv`).
  - Scalars and vectors mix only in the per-node `vectorMLP` blocks (`components.py:527`). Scalars receive a
    `Scalarizer` of the vectors. Vectors are gated by scalars and pass through a VN-style `VectorActivation`
    (`components.py:93`).
  - Both tracks are pooled (`molecule_graph_model.py:517-531`). The encoder returns `(x_out [n, 64], v_out [n, 3, 64])`.
  - Every vector operation is a channel mix or a scaling by norms and dot products. There are no cross products. The
    encoder is therefore exactly O(3)-equivariant: reflections act on v like rotations, as polar vectors.
- **Production checkpoint.** `D:\crystal_datasets\model_checkpoints\_best_autoencoder_experiments_dev_26-09-13-48-15`,
  the default in `build_qm9_conditions.py:90`. It has 9.26 M parameters, bottleneck 64, encoder node width 512, 2
  convolutions, a 3 Å cutoff and `norm: graph layer`. Its decoder is the `gnn` type with 64 nodes. Its `dataDims`
  record elements {1, 6, 7, 8, 9} and 6–29 atoms, which is QM9-shaped. **[check]**: the stored config was read on
  CPU; the top-level dataset keys are not stored in the checkpoint.
- **Decoder.** `Mo3ENetGraphDecoder` (:377). It places 64 nodes at positions linear in the vector latent
  (`v_to_pos`, :429, :471), gives them scalar features from the scalar latent and vector features from the vector
  latent, and runs an O(3)-equivariant `VectorGNN` over the complete graph on those nodes. Each node outputs a position,
  type logits and a weight (`output_depth = types + 4`, :73).

**Training [code].**
- **Loss.** `ae_reconstruction_loss` (`autoencoder_utils.py:475`) compares a type-and-position Gaussian overlap of
  the decoded swarm against the molecule's self-overlap. The type channels are scaled by `type_distance_scaling` 2. To
  it are added:
  - a radius constraint;
  - a node-weight constraint;
  - a nearest-component term (`modeller.py:1753`).
- **Overlap width.** σ anneals from 1.05 Å by ×0.99 whenever the recent reconstruction loss is below 0.15
  (`ae_annealing`, `modeller.py:1635`). It is evaluated at 0.35 Å.
- **Augmentation.** Each batch gets positional noise (`positional_noise.autoencoder`: 0.01 Å in
  `configs/experiments/dev.yaml:57`, 1 in `base/autoencoder.yaml`) and a random roto-inversion
  (`preprocess_ae_inputs`, `orient_molecule(mode='random', include_inversion=True)`, `modeller.py:1650-1669`).
  - The augmentation is redundant for an exactly equivariant model, apart from numerical error.
  - Equivariance is enforced by construction. `ae_equivariance_loss` (`modeller.py:1317`) only reports the encoder and
    decoder equivariance errors; nothing trains on it.
- **Data.** The checkpoint is QM9-shaped. On-the-fly SMILES-to-conformer synthesis exists (`modeller.py:1376`,
  `otf_synthesize_molecules`). §6 covers both.

**What the "scalar embedding" is (the owner's question) [code]. The suspicion is right, and on the GFN side the facts
are stronger than suspected.**
- **Mo3ENet throws its own scalar track away.** `Mo3ENet.encode` keeps only the vector output:
  `_, encoding = self.encoder(mol_batch)` (`autoencoder_models.py:185`). The genuine pooled scalars `x_out` are
  computed and discarded.
- **The decoder's scalar input is a function of the vector latent.** It is `self.scalarizer(encoding)` (:214). A
  `Scalarizer` (`components.py:21-90`) mixes the 64 vectors down to 16, then takes their norms and their dot products
  with 3 learned directions, and applies a linear map.
  - That output is O(3)-invariant, so it is a true invariant.
  - But it is derived from the vector latent. It carries no information the vector latent does not.
  - The "scalar embedding" logged in MXtalTools (`modeller.py:601`, :1209) is this Scalarizer output.
- **GFN consumes neither scalar.** `build_qm9_conditions.py::embed` (:433) stores `Mo3ENet.encode`'s
  `[n, 3, 64]` vector latent. The trainer flattens it to 192 numbers
  (`energies/molecular_crystal.py:1948`, `emb.reshape(num_graphs, -1)`). So the condition was the vector latent's raw
  Cartesian components in the pinned standard frame.
  - It is neither invariant nor scalarised.
  - It is well defined only because the frame is pinned, as the comment at :1940-1945 says.
- **[lead]** The Mo3ENet condition was measured fragile (memory `reference_mo3enet_condition_embedding_fragility`,
  2026-10-05, not re-verified):
  - it depends on batch composition (a molecule embedded alone moved a median 6.8%);
  - 0.002 Å of coordinate noise moved it 14%;
  - the pinned frame flipped or mirrored for 13–63% of perturbed geometries.

  A plausible source of the batch dependence is the `graph layer` norm. It was not traced here.

### 1.2 SplitAtomEncoder (GFN)

`models/spline_flow.py:526-572` **[code]**:
- **Trunk.** `self.gnn` is a mxtaltools `ScalarGNN` on Z alone (`input_node_dim = 1 + chirality`, :551), on the
  radius graph. Geometry enters only through the radial basis of interatomic distances. Its output is per-atom `h_i`,
  pooled by `AugSoftmaxAggregation` to `s` (:569). With `chirality > 0`, `ChiralChannels` pseudoscalars (:494) join
  the node input, so `s` stays rotation-invariant but tells a chiral molecule from its mirror image.
- **Point stream.** `self.point` is a second `ScalarGNN` on the same graph, with node input
  `[Z_i, h_i, MLP(x_i)]` and x_i in the presented frame (:553, :570). It is pooled to `p` (:571). The per-atom outputs,
  projected, are the point tokens. `s` feeds `p`; nothing feeds back.
- **Readers** (`_ContextFlow._context_input`, :829-840):
  - the level head reads `[s, condition columns]`, detached (`ConditionalLatentFlow.level`, :986);
  - the context MLP reads `[s, p, columns]`;
  - `split` also cross-attends to the point tokens.
- **Norms are per node** (`norm='layer'`). The pools are per graph.

**[check] (random initialisation, float64):**
- `s` is unchanged under a proper or improper rotation: relative change 3.7e-16 and 3.0e-16.
- `p` changes by 9.1% under a proper rotation and 6.3% under an improper one.
- A molecule's `s` and `p` are identical (4e-16) whether it is encoded alone or in a batch of 8. The encoder is
  batch-independent, unlike the Mo3ENet lead above.

**`p` is not a vector channel [code + check].**
- It is `pooled_dim` frame-dependent numbers (128 by default) with no Cartesian index and no group action. "Equivariance of `p`" has no meaning
  until a representation is chosen.
- A least-squares linear map from p(x) to p(Rx), fitted on 150 random molecules for one fixed R, leaves 18% held-out
  residual. That is evidence, though not proof, that no linear representation is hidden in it.
- To ask for equivariance we must first give the point stream an output with a 3-index (§3.1).

**Frames on the policy side [code].**
- The FD orientation rows (θ, φ, r) chart R ∈ SO(3), which acts on the molecule's coordinates in the frame it is
  presented in.
- Under `rotate_frames` (`nf_policy.py:81`, `NF_PENDING_DEFAULTS`), each training step draws R0 uniform on SO(3)
  (`_frames`, :444):
  - the encoder reads the atoms rotated by R0 (`spline_flow.py:811-815`);
  - the flow models the orientation rows re-expressed as R → R R0ᵀ (`energies/fd_coordinates.py:220`,
    `rotate_orientation`), with the chart's Haar Jacobian (:238).

---

## 2. Theory: what each channel must carry

### 2.1 How the orientation transforms [derived]

The crystal places atom i at R x_i (+ t), with x_i in the presented frame. Present the molecule as x′ = R0 x. The same
crystal then needs R′ x′ = R x, so R′ = R R0ᵀ, which is what `rotate_orientation` implements. That is a right action.
An equivariant vector transforms by the left action, v(R0 x) = R0 v(x). Transposing:

  R′ᵀ = R0 Rᵀ.

So the columns of Rᵀ transform as equivariant vectors. They are the crystal's lab axes written in the molecule's frame.
R itself is "inverse-equivariant".

The consequence for the flow is direct. Let F(x) ∈ SO(3) be any equivariant molecular frame, F(R0 x) = R0 F(x), whose
columns are the molecule's own axes in the presented coordinates. Then

  Q = R F(x)  ↦  R R0ᵀ R0 F(x) = Q,

so Q is invariant. Three results follow:
- A flow that models Q, conditioned on invariants only, is exactly frame-independent. Its density pushes forward to R
  through the fixed map R = Q F(x)ᵀ.
- Right-multiplication preserves Haar measure, so only the chart's own density ratio enters log q. That is the term
  `rotate_orientation` already returns.
- Equivalently, present the molecule in its own frame, x̃ = F(x)ᵀ x, and keep the flow as it is. This is a *learned
  standard frame* replacing `orient_molecule(mode='standard')`, with a deterministic R0 = F(x_std)ᵀ per molecule. It
  must be applied in every mode, so it cannot reuse the `rotate_frames` path, which is training-mode only (§7.1).

The alternative is to hand the flow equivariant vectors V and let it learn the action. The AR spline flow's conditional
would then have to satisfy q(R R0ᵀ | R0 V) = q(R | V) for every R0. The flow has no structure that represents this
symmetry, so it must learn it from data, which is the task co-trained `rotate_frames` currently sets. The policy should
not be asked to do that.

### 2.2 What is left for each channel [derived]

A molecular geometry is its shape (an O(3) orbit), a pose in SO(3) and a handedness when the shape is chiral. The
shape is everything a distance-only GNN can see. So:

- **Scalar channel `s`: all intramolecular information.**
  - This includes composition, topology as seen through distances, conformation and size.
  - It is read by the level head and the flow context.
  - It is mirror-blind if built from distances alone.
- **Vector channel: the pose F ∈ SO(3).** Anything else in it duplicates `s`. Its information content is three
  rotational degrees of freedom.
  - **Downstream, F is consumed by canonicalising, never as a context input.**
  - **Chirality is not a bit of the frame.** The first draft took χ = sign det M as a handedness bit. That is wrong:
    det M ≈ 0 for every planar molecule and every mirror-symmetric one, so χ is the sign of rounding noise there (§3.1).
  - With the pseudovector readout of §3.1, F is SO(3)-equivariant only. An enantiomer pair then gets unrelated frames
    under the same mirror-blind `s`, so chirality must enter the condition through `s`, via `ChiralChannels` (:494),
    a continuous per-atom pseudoscalar field. The choice is an owner decision (§7.3).
  - **[review check]** In GFN's stored frame (`orient_molecule(mode='std')`, handedness +1) the stored enantiomer is
    chosen by the principal axes, so a frame-derived handedness bit would report a PCA artefact. No χ input in the
    first P-1 experiment.
- **"Global" information has nothing beyond F.** Under the pinned standard frame (no `rotate_frames`), the pose
  is fixed by convention. The vector channel then carries no information beyond `s` *in principle*. What it supplies is
  computability: the principal-axes frame is a discontinuous function of shape that the flow would otherwise have to
  learn from `s`. Today's `p` (an MLP of standard-frame coordinates) mostly supplies this.

### 2.3 The obstruction, and why it is benign here [derived]

No continuous SO(3)-equivariant frame exists on all shapes. A molecule with a proper rotational stabiliser
G_m ⊂ SO(3) has g x = x as a point set. A permutation-invariant encoder therefore gives E(gx) = E(x), while
equivariance demands F(gx) = g F(x). Only proper symmetry elements obstruct an SO(3) frame: the C_n axes, and linear
molecules. Mirror planes, inversion centres and S_n axes do not.

**[check]** An exactly equivariant invariant-weighted *polar* readout V = Σ_i a(h_i) x_i shows the obstruction, and
also an artefact of using polar vectors only:
- it is equivariant to 5e-16 under proper and improper rotations;
- it lies exactly on the axis of a C2 toy molecule (off-axis fraction 5e-17), which is the genuine obstruction;
- it is exactly zero (1e-15) on a centrosymmetric toy molecule. This is not an obstruction to an SO(3) frame. A polar
  vector must vanish under inversion, but a pseudovector need not.

**[review check]** Adding pseudovectors P = Σ_ij b(h_i, h_j, d_ij)(x_i − x̄) × (x_j − x̄) gives exact, noise-stable
SO(3) frames (special Procrustes, §3.1) on C1, Cs, Ci and planar toys: equivariance error ≤ 3e-14, change under 1e-6 Å
noise ≤ 7e-5. Only the C2 toy still fails (error 1.88). The weights must depend on d_ij; with b(h_i, h_j) alone the
Ci terms cancel exactly.

Near-symmetric molecules add the boundary problem: F is ill-conditioned and jumps under small changes. Dym, Lawrence &
Siegel (ICML 2024, "Equivariant frames and the impossibility of continuous canonicalization") prove that no continuous
canonicaliser avoids this. The PCA standard frame has the same defect: degenerate moments and sign flips, the 13–63%
flips of the lead in §1.1.

**It is benign for the crystal target, if equivariance is defined up to the molecule's symmetry.**
- If F jumps to F g with g ∈ G_m, then Q → Q g and the canonical cloud c = Fᵀ x → gᵀ c. The crystal
  Q g · gᵀ c = Q c is unchanged.
- The target density over Q is therefore exactly right-G_m-invariant, so any frame choice among the |G_m| equivalent
  ones is correct.
- For a near-symmetric molecule, a seam crossing permutes near-equivalent modes of Q. The cost is of order the target's
  asymmetry under the near-symmetry.

Design rules:
- **Measure equivariance up to G_m.** A loss ‖F(R0x) − R0F(x)‖ is wrong for symmetric molecules. It penalises
  harmless frame choices, and a vector-valued version ‖V(R0x) − R0V(x)‖ is minimised by V → 0.
- **The augmentation group is a free choice, not a correctness rule.** The first draft required SO(3)-only
  augmentation because "O(3) augmentation forces polar vectors to zero on centrosymmetric molecules". That was
  misattributed. The proposed losses (factorised reconstruction, §3.2(d); consistency, §3.3) compare frames up to the
  stabiliser, and under them O(3) augmentation of a centrosymmetric molecule demands nothing new, since −x = x as a set.
  Zero-forcing happens only under the vector-valued loss above, which this design does not use.
- **Mask proper-symmetric molecules (C_n, linear) out of the frame losses.** Their frame is arbitrary by symmetry, and
  the target is invariant to the choice. The singular values of M detect them (§3.1).

---

## 3. Approximate equivariance

### 3.1 The vector readout [proposed]

Give the point stream a 3-index output. In increasing order of how much structure it imposes:

| readout | exact equivariance | frames it fails on | cost |
|---|---|---|---|
| V2: a plain MLP of `p`, reshaped to [3, k] | none; the loss must create it from scratch | — (no guarantee anywhere) | trivial |
| V1 polar: V_k = Σ_i a_ik(h_i) (x_i − x̄) | O(3) | C_n, linear; also Ci (V = 0), Cs and planar (rank ≤ 2) | one linear layer |
| V1 polar + pseudovector: V1 plus P_k = Σ_ij b_k(h_i, h_j, d_ij)(x_i − x̄) × (x_j − x̄) | SO(3) | C_n, linear only | one pair MLP on the radius graph's edges |
| L1 equivariant GNN (Mo3ENet) | O(3) | as V1 polar | large |

Caption: readouts compared by construction; the failure column for the two V1 rows is the [review check] of §2.3
(toy molecules, random weights). h_i are the invariant trunk features, d_ij interatomic distances, x̄ the centroid
convention the FD chart uses.

**Recommended: V1 polar + pseudovector, with k = 3 or 4 vectors of each kind.**
- Exact SO(3) equivariance everywhere except at proper symmetries, which no continuous frame can handle (§2.3).
- No L1 machinery and no tensor products. The vectors live in the molecule's span and scale with its size.
- The first draft added a frame-dependent pt_i term to the weights, to break symmetry. With pseudovectors it is needed
  only for C_n molecules, whose frame is arbitrary anyway. Drop it from the first build. If it is kept, initialise it at
  zero (§3.2(d)).
- The pseudovectors give up the enantiomer relation a polar-only frame has. Chirality then enters through `s` (§2.2).

**F from V.**
- Stack the readout into the 3×3 matrix M = [V P] A, with A a learned mix, and take M = U Σ Wᵀ.
- **F = U diag(1, 1, det(U Wᵀ)) Wᵀ**, the special orthogonal Procrustes projection (SVD+ in Levinson et al., NeurIPS
  2020, "An analysis of SVD for deep rotation estimation"). It is continuous wherever σ2 > 0.
- **Corrected.** The first draft set F = U Wᵀ diag(1, 1, sign det M), which flips the output basis instead. As det M
  passes through zero it jumps by a rotation that is not a symmetry. **[review check]** On polar-only Cs and planar
  toys it gave equivariance error 1.86–1.88 and a change of 1.9–2.0 under 1e-6 Å noise, against 1e-14 and 3e-5 to
  2e-4 for the special projection.
- **Gradients diverge near degeneracy.** The polar factor's derivative carries 1/(σ_i + σ_j).
  - **[review check]** On a C2 toy displaced δ off the symmetry, ‖dF/dx‖ grew as 1/δ: 3.4e2 at 0.1 Å, 1.9e7 at 1e-6 Å.
  - Torch's SVD backward also fails where the polar factor is smooth: NaN in float32 for two singular values within
    1e-7, and a wrong directional derivative in float64 at an exact tie (0.944 against a finite difference of 0.619).
    A learned M will produce near-ties without any molecular symmetry.
  - So: use a polar factor with a stable custom backward (the 1/(σ_i + σ_j) form), or Björck / Newton–Schulz
    orthogonalisation. Gate the frame losses by σ2 + σ3, and mask molecules below a floor (§2.3).
- The singular values of M are invariants that report how well defined the frame is. Feed them to `s`'s readers as a
  frame-confidence input.

### 3.2 Does reconstruction alone induce equivariance? It depends on the decoder [derived]

The reconstruction is scored in the presented frame, with augmentation by random R0. The setup:
- the latent is (s, F), and `s` is exactly invariant by construction;
- the loss is a permutation-invariant overlap of x̂ with R0 x;
- the decoder is one of four types, compared below.

**(a) L1 equivariant GNN decoder (Mo3ENet-style).**
- Since D(s, R V) = R D(s, V), perfect reconstruction of every R0 x forces V(R0 x) = R0 V(x), up to D's null space.
  Reconstruction alone induces equivariance.
- **Bias.** Positions come out of the vector track (`v_to_pos`). The vector latent must therefore hold the whole shape
  as well as the pose. Mo3ENet's own scalars are a function of it (§1.1).
- This is the entanglement the owner wants to remove. Shape information sits in the frame-dependent channel, which
  is the channel the flow memorises through.

**(b) Plain MLP decoder.**
- There is no equivariance pressure. Any code for (shape, R0) reconstructs, so the encoder may represent rotation
  arbitrarily and nonlinearly.
- The decoder must learn the SO(3) action from data. Shape and pose entangle freely.
- Equivariance must come entirely from an explicit loss, which the decoder does not exploit. **Not recommended.**

**(c) Equivariant MLP decoder (vectorMLP, Mo3ENet's `mlp` option, `autoencoder_models.py:337`).** As (a), with the
same bias toward shape in the vector track, at lower cost.

**(d) Factorised decoder (recommended): x̂ = F̃ · D(s).**
- D is any network of `s` alone. It emits a canonical-pose swarm: positions, type logits and weights, as Mo3ENet's
  decoder does. F rotates it.
- **Equivariance is induced, up to symmetry.** With `s` invariant, D(s) is the same for every presentation. Perfect
  reconstruction gives F̃(R0x) D(s) = R0 F̃(x) D(s), so F̃(R0 x) = R0 F̃(x) g with g in the stabiliser of D(s). That
  is exactly the symmetry-aware equivariance of §2.3. The loss is indifferent among the |G_m| frames, so the symmetry
  obstruction never enters the objective.
- **It also holds at imperfect reconstruction [review, derived].** argmin_F overlap(F · D, R0 x) =
  R0 · argmin_F overlap(F · D, x), so the optimal aligner is equivariant whatever the quality of D.
- **Initialisation rule.** If F starts inconsistent across poses, the best D(s) for random presentations is the
  rotation-averaged blob, and for a blob every F is optimal: a stationary bad point. Start F exactly equivariant: the
  readout of §3.1 is, and any frame-dependent term is initialised at zero.
- **Intramolecular information must flow through `s`.** F̃ is a 3×3 orthogonal matrix and cannot carry shape.
- **Its cost: D(s)'s canonical pose must be a function of shape.** By §2.3 that function is discontinuous somewhere.
  An MLP is continuous, so reconstructions will blur near the seams. Expect higher reconstruction error on
  near-symmetric shapes; the readouts in §7 measure it. The other decoders carry the same impossibility, only hidden
  inside the latent.

**Is (a)'s bias good or bad?**
- **For reconstruction quality: good.** An equivariant decoder with a rich vector latent is the most expressive and
  sample-efficient choice, and Mo3ENet reconstructs well.
- **For this downstream use: bad, for three reasons:**
  1. The flow needs shape only through invariants and pose only through a frame. (a) delivers both mixed in a
     frame-dependent latent.
  2. A latent that mixes shape and pose cannot be canonicalised: no explicit frame comes out of it.
  3. The memorisation evidence points at high-capacity frame-dependent inputs. The gap widened 1.5–2.5× when the
     frame-dependent stream was trained under rotation.
- **(d) keeps (a)'s good part**: equivariance induced by reconstruction, with no L1 encoder. It drops the
  entanglement.

### 3.3 The explicit consistency term [proposed]

Put K rotated copies of each molecule in a batch, x_a = R_a x with R_a uniform on SO(3). K = 2 is enough, and 4
tightens the estimate. Then

  L_cons = Σ_{a<b} d(F(x_a)ᵀ x_a, F(x_b)ᵀ x_b),

a distance between the *canonicalised input* point clouds.
- **Decoder-free.** No D pass.
- **Symmetry-aware.** Two canonical clouds that differ by g ∈ G_m are the same set, so the term is zero.
- **Cannot collapse.** F is orthogonal, unlike a raw ‖V(R0x) − R0V(x)‖² term, which is minimised by V → 0.
- **The distance d.** The first draft named MXtalTools' overlap distance. That is unsuitable: `ae_reconstruction_loss`
  (`autoencoder_utils.py:475`) is asymmetric (decoded density at each true atom against the self-overlap), and its
  resolution moves with the σ schedule. Use a symmetric typed-kernel MMD at a fixed small σ, which is zero exactly when
  the typed sets coincide. For molecules with trivial G_m (every norbornane, §7.2), a matched-index RMSD is exact,
  cheaper and needs no σ.

**What it does not do [review, derived].** The first draft overclaimed it:
- With an exactly equivariant readout (§3.1), L_cons is zero up to float error and gives no gradient. It polices only
  a frame-dependent term, under noise and conformer change (§5 item 3), or a non-exact readout such as V2.
- Any equivariant frame satisfies it, the PCA frame included away from its flips. It does not select a frame that
  makes D(s) smooth. Only reconstruction does that.

So it is a cheap guard and diagnostic, not the mechanism. It is the owner's "rotated copies should agree" term, stated
so that it is correct for symmetric molecules.

---

## 4. Scalar and vector heads [proposed]

The two options:
- **(i)** Keep any decoder and add an invariant head on `s`: a distance-matrix or distance-histogram regression,
  bond-graph reconstruction, or similar.
- **(ii)** Restructure so that intramolecular information can only reach the reconstruction through `s`, and pose and
  handedness only through F̃. This is the factorised decoder of §3.2(d).

**Recommend (ii).**
- **(i) does not separate the channels.** An auxiliary head teaches `s` the shape, but nothing stops the vector
  channel from carrying shape too. A shared decoder will use whichever channel is easier, and with (a) or (c) that is
  the vector one.
- **(ii) separates them by an information bottleneck.** After the SVD projection the vector channel has three
  continuous degrees of freedom. Shape cannot pass through it.
- **(ii) also removes the one remaining frame-dependent input from the policy.** The flow conditions on invariants
  only (§2.1).

An invariant distance head on `s` stays a cheap optional extra: a warm-start signal while D is weak, and a diagnostic.
It is not the mechanism.

---

## 5. Upgrades and speedups to check once the theory is settled [proposed]

1. **Reuse the trunk across rotated copies.** `s` and h_i are exactly invariant. Compute them once per molecule and run
   only the point stream (and V) per copy. With trunk and point stream of equal size, K = 4 copies cost 5 units
   instead of 8.
2. **Reconstruction loss.**
   - Start with MXtalTools' `ae_reconstruction_loss` (`autoencoder_utils.py:475`). GFN importing MXtalTools is the
     allowed direction. It has a fixed σ schedule rather than Mo3ENet's adaptive one (§1.1), so runs are comparable.
   - Then check a matching loss. The decoder can be told the composition: emit exactly n_Z atoms of each element and
     score with a per-element Hungarian or Sinkhorn assignment. QM9 and norbornane sizes are at most 31 atoms, so the
     cost is small.
   - The matching loss removes σ, the weight and clumping constraints, and the empty-node machinery. It also gives a
     direct RMSD readout.
   - Owner choice: giving D the composition means `s` need not encode it. That weakens `s` as an identity key, which may
     help generalisation.
3. **Noise and conformer consistency.**
   - Make `s` and F stable under small noise and under conformer changes: s(x + ε) ≈ s(x), and L_cons across noised
     copies.
   - The lead in §1.1 measured Mo3ENet moving 14% at 0.002 Å. A conditioner that moves that much is an identity key on
     exact coordinates.
   - Report the same jitter curve for the new encoder.
4. **Latent regularisation.**
   - The policy already has `cond_bottleneck` and `cond_noise` (`nf_policy.py`, `NF_PENDING_DEFAULTS`).
   - On the autoencoder side, a VICReg-style variance floor or a small KL on `s` keeps the latent smooth and avoids
     dimensional collapse.
   - Check by the downstream gap, not by reconstruction.
5. **Keep norms per node.** SplitAtomEncoder is batch-independent ([check], §1.2); keep it that way, with no graph
   norms in the new heads.
6. **Cache the graphs.** Precomputed radius graphs per conformer, `_atom_graph`'s cache (`spline_flow.py:429`)
   generalised to the dataset. Mixed precision (bf16) for the trunk.

---

## 6. Data plan

### 6.1 What exists

Sizes are from `ls`, manifests, logs or docs. No multi-GB `.pt` file was loaded. Conformer code was read, not run or
modified.

| set | molecules | conformers per molecule | elements, size | geometry source | path |
|---|---|---|---|---|---|
| QM9 processed | 133,728, every SMILES distinct | 1 | H C N O F, ≤ 9 heavy atoms | DFT | `D:\crystal_datasets\qm9_dataset.pt` (360 MB) |
| QM9 basin database | 133,641 constitutions, 218,473 stereo conditions | 3,400,546 MMFF basins in total; mean 15.7, median 4 per condition | as QM9 | MMFF multi-start descent within 10 kT | `D:\crystal_datasets\qm9_db_sep29\` (400 shards, 2.9 GB) |
| rc_oct22 random-conformer priors | about 681k: qm9rc 121,469, csdrc 52,645, zincrc 177,256, gdbrc 192,004, pubrc 175,678 | 1 | H C N O F; many are capped fragments of larger parents | ETKDG with random torsions (`data_processing/random_conformers.py`) | `D:\crystal_datasets\conditional\priors\rc_oct22\` (about 50 GB, crystal rows) |
| training pool v1 | 3,678,339 (train 3,319,676) | none (SMILES only) | H C N O F, MW < 230, ≤ 1 rotatable bond | — | `D:\crystal_datasets\paper_review_nov02\pool\pool_v1.csv.gz` |
| GEOM, ZINC22 | — | — | — | — | not on D: (no GEOM builder; ZINC22 only at a cluster path) |

Caption: molecule sets on the local disk that could feed pretraining. Counts are from `qm9_db_sep29/summary.txt`,
`pool_v1_counts.json`, `docs/results/day-landscape-condition-sets-2026-11-02.md` (family table) and
`docs/design/conformer_encoder_architecture.md` (QM9 count). Not re-derived. Compare rows by molecules × conformers.
The norbornane test molecules (C N O F, 8–11 heavy atoms, median 11; [check] from `test_families/norbornane.csv`)
come from pool v1.

**Tooling and cost [code; survey].**
- **MXtalTools on-the-fly synthesis** (`mxtaltools/dataset_utils/construction/parallel_synthesis.py:24`, driven from
  `modeller.py:1376`).
  - It samples SMILES from a ZINC22-layout directory and embeds one conformer each with RDKit `EmbedMolecule`. It
    runs no MMFF (`minimize=False`).
  - Filters: atom count, heavy-atom count, element set and radius. It refreshes the training set in the background
    during Mo3ENet training.
  - Its worker `outputs` list is never appended (`parallel_synthesis.py:43,65`), so worker errors go unseen. That is an
    MXtalTools finding, not fixed here.
- **GFN conformer tooling** (under the conformer refactor; read only). `build_conformer_set.py` and
  `build_conformer_database.py` built the QM9 basin database.
- **Recorded costs.**
  - QM9 basin database: **20.0 s per constitution on one thread**, 18.2 s of it relaxation. That is 742 CPU-hours for
    QM9 (`qm9_db_sep29/summary.txt`, table 4).
  - The 2,000-molecule `qm9_db_big` set: about 0.64 s per molecule (1,503 s over 2,339 attempted, from the manifest).
  - One-conformer ETKDG plus MMFF has **no recorded throughput** anywhere found.

### 6.2 Plan [proposed]

**The constraint: QM9 alone does not cover the test molecules.** Norbornanes have 8–11 heavy atoms (training median
11, held-out median 10; **[review check]** on the 117/35 split); QM9 stops at 9. That puts most of the 35 held-out
molecules outside the pretraining size range. The owner also
recalls that earlier work overfit QM9. So the first round goes beyond QM9 from the start.

1. **Measure conformer cost first.** Embed 1,000 pool-v1 molecules (8–14 heavy atoms) with ETKDGv3 plus MMFF on one
   core, and record seconds per molecule and the embed failure rate.
   - It takes about a minute or two. That is longer than this scope's few-second limit, so it was not run; the owner's
     go is needed.
   - Pool v1 molecules have at most one rotatable bond, so one relaxed conformer is close to complete. A second
     conformer is needed only for the rotatable ones.
   - Expected: tens of milliseconds per molecule. Unmeasured; this is an assumption.
2. **First-round set.**
   - QM9 (133,728 molecules, DFT geometries).
   - About 500k pool-v1 molecules with 8–14 heavy atoms, one ETKDG plus MMFF conformer each.
   - Up to 2 MMFF basins per QM9 condition from `qm9_db_sep29`, for the conformer-consistency readout and term (§5
     item 3).
   - Exclude every norbornane constitution (the 54 `ik14` blocks of `norbornane.csv`, both splits) by constitution, not
     by molecule key. A stereoisomer of a held-out molecule would otherwise leak (`norbornane_p1_split.json` splits by
     constitution for the same reason).
   - If step 1 measures 30 ms per molecule, the 500k cost about 4 CPU-hours, under 20 minutes on 16 cores.
3. **Mixed geometry sources.** DFT (QM9) and MMFF (pool, database) geometries differ systematically. Without 0.01 Å
   noise and the consistency term, the encoder could learn the source as a feature. The lead in §1.1 found that noisy
   geometries sit off Mo3ENet's DFT manifold.
   - Readout: the `s` distance between a QM9 molecule's DFT geometry and its MMFF basin, against the nearest-neighbour
     distance between distinct molecules.
4. **Later rounds.**
   - The rest of pool v1's 3.3M training molecules.
   - The rc_oct22 family molecules.
   - A second conformer where there is a rotatable bond.
   - Elements beyond H C N O F need a new SMILES source; pool v1 is H C N O F only. The encoder's atom embedding
     already covers any Z.
   - Overfitting is checked by held-out reconstruction and by the downstream gap, not by training loss.

---

## 7. Minimal first experiment, and owner decisions

### 7.1 Build (about half a day) [proposed]

1. **`models/conditioner_ae.py`.**
   - The encoder is `SplitAtomEncoder` unchanged, plus the polar + pseudovector readout and the special Procrustes
     frame with a stable backward (§3.1).
   - The decoder is factorised (§3.2(d)), with D an MLP of `s` emitting a 64-node swarm.
   - The loss is MXtalTools' reconstruction loss plus L_cons with K = 2, its distance as in §3.3, with proper-symmetric
     molecules masked out of both frame terms.
2. **A pretraining script.** It runs on the first-round set of §6, with random rotations and 0.01 Å noise. It saves an
   encoder file that carries its own widths, as `sym_ops_init` does (`nf_policy.py:68`).
3. **Policy wiring, settings in `NF_PENDING_DEFAULTS`, so `mk_dev.yaml` is untouched.**
   - `atom_encoder_init`: a pretrained encoder file, frozen by default.
   - `atom_frame`: `standard` (today), `learned` or `core` (the hand-built frame of §7.2's `core_frame` arm).
   - **Corrected wiring.** The first draft reused `rotate_frames` with a deterministic R0. That path is training-mode
     only: `_frames` returns None unless `self.training` (`nf_policy.py:444-450`; comment at :86-87), and
     `log_density` presents the standard frame when `frames is None` (:705-718). A flow trained on Q would be scored and
     sampled as if its rows were R in every eval readout, including the primary gap. Frames also switch off the keyed
     per-molecule atom table (`nf_policy.py:559`) and the cached atom graph (`spline_flow.py:831`).
   - **Instead:** precompute `s` and F once per molecule from the frozen encoder. Apply the fixed map R → R F, with
     `rotate_orientation`'s Jacobian, at the policy boundary in **every** mode: training, `log_density` and sampling,
     where draws map back by R = Q Fᵀ. The flow reads `s` through the existing per-molecule vector route
     (`set_mol_scalars` with encoder `'none'` is nearly this). The encoder never runs in the policy unless point tokens
     are on.
4. **Tests.**
   - `s` invariance.
   - F: SO(3) equivariance to float precision on C1, Cs, Ci and planar toys; continuity under 1e-6 Å noise; a finite
     gradient in float32 at a near-tie of singular values.
   - A frozen encoder receives no gradient.
   - The learned-frame round trip: a row re-expressed and back is bit-identical, and log q includes the Jacobian.
   - The frame is applied in eval: `log_density` of a row in eval mode equals its training-mode value with the same
     frame.
   - A `rotate_frames`-equivalence check replaces the old `ae_s_rot` arm: Q is unchanged when the molecule is presented
     rotated and F is recomputed. **[review check]** through `rotate_orientation` on 2,000 random sg-2 rows: max row
     difference 5.2e-15, max log-Jacobian difference 8.9e-15.

### 7.2 Runs [proposed]

- **Pretraining.** One run, a few GPU-hours. Not measured; this is an estimate to confirm on the first epoch.
- **Downstream.** The norbornane P-1 conditional MLE recipe of the `norb_mle_nov05` battery: base prior, batch 500,
  14–20k steps. At most one data-on-GPU run at a time locally (memory lead), or as a cluster battery.

**What the test set does and does not exercise [review check].** All 152 norbornanes are C1. Each is at least 1.06 Å
from any principal-axis C2, mirror or inversion, and none forms an exact enantiomer pair. The symmetry machinery of
§2.3 never binds here. What a frame can change on this set is the alignment *across* molecules. In the stored frame,
the shared bicyclo[2.2.1] core sits in 22 distinct poses across the 117 training molecules (45° clustering threshold;
50 at 20°). Each held-out molecule's nearest training core pose is 0–33° away, median about 14°.

**A note on the key.** A frozen 128-d `s` is still an exact molecule key: it separates the 117 training molecules, and
the context MLP and transformer can memorise through it. The frozen 7-scalar arm already shows a 0.66 gap. Freezing is
therefore not the mechanism; the geometry and information content of the key are. The controls below separate these.

**Stage A: no pretraining needed** (can run before the autoencoder exists).

| arm | condition read by the flow | frame | what it isolates |
|---|---|---|---|
| `base` | co-trained split encoder s, p | PCA standard | reference: about 0.80–0.85 nats at matched fit −2 (single seed) |
| `scalars` | about 7 molecule scalars | PCA standard | reference: 0.66 (single seed) |
| `s_only` | co-trained split encoder s, no p | PCA standard | the p channel against the rest of the encoder |
| `rand_s` | frozen randomly initialised SplitAtomEncoder s | PCA standard | freezing alone, without pretraining |
| `core_frame` | co-trained s, p | hand-built core frame | does cross-molecule core alignment lower the gap, at zero pretraining cost? |
| `trunk_s` | frozen IntraTrunk pool (`pretrain_intra_trunk.py`, MMFF energy and force pretraining over QM9) | PCA standard | a ready pretrained invariant; energy/force pretraining against reconstruction |

**Stage B: after pretraining.**

| arm | condition read by the flow | frame | what it isolates |
|---|---|---|---|
| `ae_s` | frozen pretrained s | PCA standard | reconstruction pretraining, against `rand_s` and `trunk_s` |
| `ae_s_b8` | frozen pretrained s through `cond_bottleneck` about 8 wide | PCA standard | information-matched to `scalars` |
| `ae_s_frame` | frozen s, frame confidence | learned F | the learned frame, against `ae_s` and `core_frame` |

Caption: proposed arms, nothing run. Each row is a norbornane P-1 conditional MLE run on the recipe above. `base` and
`scalars` are rerun with seeds on the exact recipe; the owner-quoted single-seed readings are only a prior. The first
draft's `ae_s_rot` arm is dropped: with F recomputed on the rotated presentation it equals `ae_s_frame` to float error,
so it is a unit test (§7.1 item 4); with F fixed and a random R0 applied after it, the flow cannot see R0 and the
orientation target collapses to the Haar marginal.

- `core_frame` needs a deterministic core frame: u toward the one-carbon bridge (C7), v along the bridgehead axis. The
  core's own C2 leaves a two-fold choice; the substituents break it, so a fixed substituent rule must pick one. It runs
  through the §7.1 wiring with `atom_frame: core`.
- `trunk_s`: the script exists (commit `92b8d730`); whether a usable checkpoint exists was not checked **[lead]**.

**Readouts.**
1. **Primary: the held-out-molecule gap at matched training fit.** Held-out `eval_test/mle` minus training-row
   `eval_bwd/mle`, interpolated at training levels {−1, −1.5, −2, −2.5, −3}. Use only each run's path up to its first
   arrival at the level, as `matched_fit.py` does (another session's scratchpad, `norb/rot_nov08/`; copy it into
   `eval/` when this runs).
2. **Steps to each level.** A frozen encoder may reach a given fit more slowly; the matched-fit gap controls for that.
3. **Autoencoder side.**
   - Held-out reconstruction RMSD.
   - Frame consistency (canonical-cloud overlap) against rotation, noise and conformer change.
   - Jitter sensitivity of `s` at 0.0005–0.02 Å, against Mo3ENet's 2.6–123% lead.
   - The `s`-space distance of each held-out norbornane to its nearest training norbornane and to its nearest
     pretraining molecule.
   - The core-pose dispersion of the learned frame across the 117 training molecules, against the 22 clusters of the
     stored frame. If reconstruction does not align cores, `ae_s_frame` cannot be expected to beat `core_frame`.

**Resolution.**
- Seed spread on this test is about 0.07 nats (two seeds, smoothed).
- Held-out n is effectively 11 constitutions (`norbornane_p1_split.json`: 35 molecules in 11 constitutions held out;
  117 molecules in 43 constitutions for training). The `test_split` column of `norbornane.csv` is an older 119/33 split;
  do not use it.
- Differences of 0.1–0.25 nats are unresolved (sibling review, `docs/design/conditional_learning_review_2026-11-08.md`
  §6). The two reference readings differ from each other by 0.15–0.19 nats, about that floor.
- The SE of a difference between two single-seed arms is about √2 × 0.07 ≈ 0.10 nats. So every arm needs at least 3
  seeds, or the comparison is a per-molecule paired analysis with a bootstrap over held-out constitutions.
- A result worth acting on would be the gap at −2 falling from about 0.8 toward or below the scalars' 0.66, by more
  than twice the paired SE.

### 7.3 Open owner decisions (recommended default first)

1. **Vector channel shape: a frame bottleneck with the factorised decoder (default), or a free k-vector latent with an
   equivariant decoder (Mo3ENet-like).** The bottleneck enforces the channel separation and makes the flow's input
   invariant. The free latent reconstructs better but entangles shape with pose (§3.2).
2. **Chirality: none for the first P-1 experiment (default); `ChiralChannels` in `s` for the Sohncke groups.**
   - The first draft's default, a bit χ = sign det M in the vector channel, is withdrawn: it is noise on planar and
     mirror-symmetric molecules, and a PCA artefact in GFN's stored frame (§2.2, §3.1).
   - On P-1, which contains inversion, enantiomers have the same landscape, and the norbornane set has no exact
     enantiomer pairs.
   - With an SO(3)-only frame, enantiomers get unrelated targets under a mirror-blind `s`, so any chiral space group
     needs `ChiralChannels` (`chirality > 0`).
   - Alternative: a polar-only frame keeps the enantiomer relation exactly, at the cost of failing on Ci, Cs and planar
     molecules (§3.1).
3. **Frozen or slow fine-tune.** Frozen for the first experiment (default). A fine-tune arm at about 0.01× the policy
   LR comes second, only if frozen is not worse at matched fit.
4. **Pretraining set: exclude all 152 norbornanes (default), or include the training ones.** Excluding all keeps
   "unseen" strict. Including the training 117 is a later arm that measures whether familiarity matters.
5. **Composition given to the decoder (matching loss, §5 item 2): not for the first experiment (default).** Later, as
   an arm.
6. **Point tokens and cross-attention (`atom_tokens: split`): off (default, pooled `s` only).** With a learned frame,
   tokens could be canonical-frame atom coordinates. That is a later arm.
7. **Where the code lives: GFN `models/` (default).** The encoder is GFN's. MXtalTools is used only as a library (its
   loss and `ScalarGNN`), so no reverse dependency is added.
8. **Run order: Stage A first (default), or Stage A and B together.** Stage A needs no autoencoder and tests the
   frame hypothesis (`core_frame`) and the freezing and p-channel controls directly. If `core_frame` and `rand_s` move
   nothing, the autoencoder's case rests on reconstruction pretraining alone, and Stage B can be resized before the
   build.
9. **Seeds: 3 per arm (default), or 1 seed per arm with a constitution-bootstrapped paired analysis.** Three seeds
   resolve about 0.1 nats; the paired analysis is cheaper but rests on 11 constitutions.

Settings proposed here go in `NF_PENDING_DEFAULTS` until the owner adds them to `mk_dev.yaml`.
