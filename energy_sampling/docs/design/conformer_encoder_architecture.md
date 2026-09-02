# Conformer encoder — specification, evidence, and reproduction

**Status:** current as of 2026-09-01. Self-contained; supersedes and subsumes both the
2026-08-27 version of this file and `encoder_ssl_battery.md`. Every number in those came from
runs contaminated by a training collapse and a leaking train/test split, and none survived.

Surrounding stack: [`conformer_conditional_stack.md`](conformer_conditional_stack.md) §5.

---

## 1. Desiderata

The encoder reads a molecule as a 2D labelled graph and hands the conformer policy a per-atom
embedding `g_i` and a pooled `g`. Six requirements follow from that job. They are numbered
because the rest of this document refers back to them.

| | Requirement | Why it is load-bearing | Met by |
|---|---|---|---|
| **D1** | Injective on the molecule | Two molecules the policy must treat differently need different embeddings. Information absent here cannot be recovered downstream, however good the policy. | Tested — §4, §5 |
| **D2** | Atom-resolved | The policy acts per atom — torsions, placements. A molecule-level summary cannot steer an individual atom. | Construction |
| **D3** | Chirality-aware | Enantiomers have *identical adjacency*. Every structural encoding is a function of the adjacency, so without an explicit parity channel the encoder is enantiomer-blind and generates the wrong hand. | Design + tripwire |
| **D4** | Long-range reach | Gly4 has diameter 14, Gly6 has 20. A k-layer message-passing network reaches k hops; beyond that, distant atoms cannot influence each other. | Provided, **UNTESTED** |
| **D5** | Permutation-equivariant | Atom indexing is arbitrary. Relabelling the input must permute the output identically, or the policy inherits an index-dependent bias. | Construction\* |
| **D6** | SE(3)-invariant | Conformer state is internal coordinates, themselves invariants. | Construction |

- **D5, D6 hold by construction.** The input is a 2D graph with no coordinates, the outputs
  are scalars, and message passing plus attention are permutation-equivariant. Nothing is
  learned and nothing can drift. \*See the `is_root` caveat in §8.
- **D2 holds by construction, and deliberately.** Every probe head is a single `nn.Linear`, so
  100% means the answer is explicitly present per atom, not merely recoverable by a head with
  hidden layers that could compute it itself.
- **D3 is design plus tripwire.** Parity is an atom feature and appears nowhere else. The
  `cip_code` probe is a near-copy of that input by construction — anything below 100% means
  the chirality channel is not surviving the encoder, and every stereochemical claim is void.
- **D1 is the one the battery tests**, and tests as a proxy: *n* probes establish that *n*
  functions are computable. See §10 for what would prove it outright.

---

## 2. Architecture

### 2.1 Inputs, per molecule

| tensor | shape | contents |
|---|---|---|
| atom features | `[N, 15]` | 7 one-hot element (H, C, N, O, F, S, Cl) · 6 one-hot degree (0–5, saturating) · 1 parity ∈ {−1, 0, +1} · 1 `is_root` marker |
| structural encoding | `[N, 16]` | RWSE — `diag((A D⁻¹)ˢ)`, s = 1..16 |
| edge features | `[2E, 4]` | one-hot bond order: single / aromatic / double / triple. Both directions |
| `edge_index` | `[2, 2E]` | both directions |
| `spd` | `[G, L, L]` | shortest-path distances bucketed `0..max_spd`, plus one bucket for unreachable. Padding is filled with −1, i.e. *unreachable* — a padded key must never read as an atom at distance zero, which would be the maximally attractive bias |

Node input is the concatenation, `[N, 31]`.

Two properties that are easy to lose and load-bearing:

- **Parity is the only chirality signal.** Every structural encoding is a function of the
  adjacency, and the adjacency is identical for enantiomers.
- **Ring membership is deliberately NOT an input.** `IsInRing` was an edge feature until
  2026-08-27; it reproduced the battery's own ring label on 511/511 atoms in one hop.

### 2.2 Encoder — L identical layers

GPS-shaped: local message passing then a global step, **interleaved per layer**, each as
residual + LayerNorm.

```
per layer i:
  # --- local ---
  m    = MLP_msg([ h_src ‖ h_dst ‖ edge_attr ])     Linear(2H+4→H) · SiLU · Linear(H→H)
  w    = softmax_dst( Linear(m, 1) )                 augmented softmax over each atom's
  agg  = [ Σ_dst w·m  ‖  Σ_dst m ]                   OWN incoming edges  → [N, 2H]
  h   ← LayerNorm( h + MLP_upd([ h ‖ agg ]) )        Linear(3H→H) · SiLU · Linear(H→H)

  # --- global, one of two ---
  attention=True   (SPDAttention, one per layer):
     q,k,v   = Linear(H → 3H) → n_heads × (H/n_heads)
     logits  = QKᵀ/√d_h + spd_bias[bucket(spd)]      Embedding(max_spd+2, n_heads), ZERO-init
     logits  = masked_fill(padded keys, −inf) → softmax → nan_to_num
     routed  = attn @ V
     summed  = Σ_valid V, broadcast to every position
     out     = proj(routed) + proj_sum(summed)       proj_sum ZERO-init
     h      ← LayerNorm( h + scatter_back(out) )

  attention=False  (rank-one broadcast):
     w    = softmax_graph( Linear(h, 1) )
     sel  = Σ_graph w·h  [broadcast]      ext = Σ_graph h  [broadcast]
     h   ← bcast_norm( h + bcast(sel) + bcast_sum(ext) )    bcast_sum ZERO-init
```

**Readout.** Per-atom `g_i = h [N, H]`. Pooled
`g = [ Σ softmax(score(h))·h ‖ Σ h ] → [G, 2H]` — unused by the probes, and the vector the
n-body correlators consume in the real stack.

No equivariant machinery anywhere: the input is a 2D graph, the outputs are scalars, and
internal coordinates are themselves invariants, so D6 is satisfied by construction.

### 2.3 Design choices, and why

- **Every pool is augmented softmax** `[selective ‖ extensive]` — neighbour aggregation, the
  broadcast, the attention block, the final readout. The form strictly contains sum
  (unnormalised half), mean (uniform logits) and max (large logit scale), so the right
  behaviour is *learned* rather than assumed. Guessing wrong in the intensive direction is a
  silent bug this codebase has hit twice.
- **The extensive branches are ZERO-INITIALISED** (`proj_sum`, `bcast_sum`). Not cosmetic: the
  unnormalised sum is **L× the magnitude** of the convex combination — 9× at L=9, **31× at
  Gly4's 31 atoms**, 63× at L=63. Projected jointly it swamps the routed signal at init and
  the routed path gets a vanishing share of the gradient. Zero-init makes the block start as
  pure attention and learn the extensive channel in.
- **`spd_bias` is zero-initialised too**, so attention begins unbiased and the spatial prior is
  learned. It is 40 parameters: one scalar per (bucket, head).
- **Unreachable pairs get their own bucket**, distinct from "saturated distance" — a
  disconnected fragment is a different statement from a long chain.
- **The attention block strictly contains the broadcast.** Its unnormalised half over the
  values *is* a rank-one broadcast, so attention can only win by using routing. This is what
  makes the arm comparison meaningful rather than a comparison of two unrelated designs.
- **Passing `spd` to a non-attention encoder is refused, not ignored** — silently dropping the
  one input that distinguishes the arms is how a comparison comes back null for the wrong
  reason.

### 2.4 Parameter budget (H = 128, 4 layers, 4 heads, max_spd = 8)

| component | params | share |
|---|---|---|
| `attn` (4 × SPDAttention) | 330,400 | 41% |
| `upd` | 263,168 | 33% |
| `msg` | 199,680 | 25% |
| `embed` | 4,096 | |
| norms (`norm`, `attn_norm`) | 2,048 | |
| gate scores (`mscore`, `score`) | 645 | |
| 9 linear probe heads | 2,580 | |
| **total** | **802,626** | |

One attention block: `qkv` 49,536 · `proj` 16,512 · `proj_sum` 16,512 · `spd_bias` 40.
A vestigial `log_sigma` (9 params) is retained but unused — see §7.

---

## 3. The test principle

Every probe target is a **deterministic function of the graph** — ring size, eccentricity,
orbit size, CIP code. There is no label noise and no irreducible error, so the achievable score
is exactly 100 and any shortfall is the model's. That converts a noisy ranking into a pass/fail
plus a learning curve, and separates two obstructions a single score conflates:

```
cannot fit TRAIN         -> an EXPRESSIVENESS bound. No amount of data fixes it.
fits train, fails TEST   -> SAMPLE COMPLEXITY. Read the rate off the size ladder.
```

It also self-diagnoses leakage: a probe whose answer is already in the input saturates at 100%
for every arm at trivial dataset size.

---

## 4. Probes

Nine live probes. `models/encoder_probe.py`, `PROBES`.

| block | probe | dim | target |
|---|---|---|---|
| **A** floor | `formula` | 7 | element counts for the molecule |
| **A** floor | `degree_hist` | 6 | degree histogram for the molecule |
| **A** floor | `pi_degree` | 1 | π-bond count at the atom |
| **B** cycles | `smallest_ring` | 1 | size of the smallest ring containing the atom, 0 if acyclic |
| **C** distance | `eccentricity` | 1 | max graph distance from the atom to any other |
| **C** distance | `spd_to_marked` | 1 | graph distance from the atom to the marked root |
| **D** symmetry | `orbit_size` | 1 | automorphism orbit size on the **(element, parity, is_root)**-labelled graph |
| **E** stereo | `cip_code` | 1 | CIP R/S at the atom as ±1, 0 elsewhere |
| **E** stereo | `chiral_moment` | 1 | `Σ_atoms cip_code · (1 + distance to marked root)` |

Block A are **floor checks, not discriminators** — see §5. Block C is where attention would
earn its place, if the dataset could test it (§8). `cip_code` is the chirality tripwire (§1).

**Seven probes were retired 2026-08-31 after a task-design audit**, each for a stated reason:
`n_atoms` (exactly `formula.sum()`), `cycle_rank` (an exact linear function of `degree_hist`,
R²=1.0000), `diameter` (exactly `max(eccentricity)`), `spectral_moments` (R² 0.979 from size
features; tolerance 0.071 SD over 4 components), `wiener` (R² 0.964 from size features;
tolerance 0.028 SD), `ring_member` (leaked — linear 1-hop 95.6% against a 73.2% base rate),
`parity_sum` (ill-posed at 55.3% re-serialisation survival, and blind to (R,S) vs (S,R)).

---

## 5. Results

`mp+attn+spd`, 20,000 molecules, 3,000 held out, 60,000 steps. Exact-match accuracy,
**train / held out**. The split is grouped on parent skeleton with a runtime assertion that no
skeleton crosses it.

| block | probe | train | held out | trivial floor |
|---|---|---|---|---|
| A | `formula` | **100.0** | **100.0** | 13.1% |
| A | `degree_hist` | **100.0** | **100.0** | 3.0% |
| A | `pi_degree` | **100.0** | **100.0** | 83.8% |
| B | `smallest_ring` | **100.0** | **100.0** | 70.5% |
| C | `eccentricity` | **100.0** | **100.0** | 34.8% |
| C | `spd_to_marked` | **100.0** | **100.0** | 22.5% |
| D | `orbit_size` | **100.0** | 99.6 | 64.3% |
| E | `cip_code` | **100.0** | **100.0** | 0.0% |
| E | `chiral_moment` | **100.0** | **100.0** | 31.4% |

**Read the trivial floor before the accuracy.** It is the better of a size-regression and a
majority-constant predictor. `pi_degree` at 83.8% and `smallest_ring` at 70.5% are weakly
discriminating probes, and four — `formula`, `degree_hist`, `pi_degree`, `cip_code` — are
reproduced at exactly 100% by a closed-form linear readout of the inputs with no network
trained at all. They function as floor checks.

`orbit_size`'s residual 0.4% is sample complexity, not a ceiling: train is 100.0, and held-out
loss falls with data at roughly `n^-0.3` across a 2k/8k/20k ladder. We used 20,000 of 133,728
available molecules. Its 1-WL ceiling was measured directly and is *not* binding: 4-round WL
leaves 21 of 54,047 held-out atoms ambiguous (0.0389%) against a 0.975% observed error, and
adding shortest-path information drives the ambiguity to exactly zero.

**Checkpoint:** `models/results/encoder_ckpt/mp+attn+spd_n20000_s0.pt`, best held-out at step
51,000 of 60,000.

---

## 6. Files

| file | lines | role |
|---|---|---|
| `models/graph_encoder.py` | 292 | the encoder — `MPNNEncoder`, `SPDAttention`, `dense_spd_batch` |
| `models/graph_encodings.py` | 537 | label and encoding functions — RWSE, shortest paths, orbits, WL, CIP |
| `models/encoder_probe.py` | 1010 | the battery — loading, probe definitions, training, scoring, report |
| `models/curve_report.py` | 223 | curve diagnosis — under-trained / memorising / over-fit / collapsed |
| `models/lr_sweep.py` | 185 | the learning-rate bracket that produced the recipe |
| `models/encoder_ssl.py` | — | **superseded** by `encoder_probe.py`; kept only to reproduce pre-2026-09-01 runs |
| `tests/models/test_graph_encoder.py` | — | plus `tests/models/test_graph_encodings.py` |

**Data:** `D:\crystal_datasets\qm9_dataset.pt` — 133,728 molecules, every SMILES distinct,
343 MB. **Not** `conditional/anchors/qm9c100k_chunk*.pt`, which is a crystal-anchor subset of
~5,850 unique molecules — about 4% of QM9, at ~8.2 duplicated rows each.

**Outputs:** `models/results/*.json` (per-step curves included),
`models/results/curves/*.png`, `models/results/encoder_ckpt/*.pt`.

---

## 7. Reproduce

```bash
# the recipe: phase 1 brackets, 2 tests schedules, 3 confirms on a fresh seed
python -m models.lr_sweep --phase 1 --arm mp --n-mol 3000 --n-test 600 \
  --steps 6000 --lrs 3e-5 1e-4 3e-4 1e-3 3e-3 --seeds 0 --device cuda

# the headline run (§5), ~35 min on one GPU
python -m models.encoder_probe --arms mp+attn+spd --sizes 20000 --n-test 3000 \
  --steps 60000 --seeds 0 --device cuda --out models/results/encoder_final2.json

# the sample-complexity ladder
python -m models.encoder_probe --arms mp+attn+spd --sizes 2000 8000 20000 \
  --n-test 3000 --steps 40000 --seeds 0 --device cuda \
  --out models/results/pretrain_fullqm9.json

# ALWAYS read the curves -- a final score cannot tell the failure shapes apart
python -m models.curve_report models/results/encoder_final2.json
```

Arms available: `mp`, `mp+rwse`, `mp+attn`, `mp+attn+spd`. Parameter budgets are matched
across arms by default (`--match-params`); `mp+attn` is routing *without* being handed the
distances, which separates "can route" from "was given the answer".

**Recipe, settled.** Flat **lr 3e-4**, plain MSE on standardised targets, Adam, grad clip 5.0,
**no schedule**, batch 128. 1e-3 collapses on roughly one seed in three. Schedules do not help:
flat 0.005, warmup 0.006, cosine 0.009, warmup+cosine 0.009.

Kendall uncertainty weighting is available behind `--loss-weighting uncertainty` **only to
reproduce pre-2026-09-01 runs**. Its effective weight on a task is `1/mse`, so fitting better
raises the gradient — positive feedback, and the accelerant for the collapse below. The
`log_sigma` parameters in §2.4 are its vestige.

---

## 8. Known limits

- **D4 IS UNTESTED, and this is the largest gap.** QM9's diameter is median 6, max 9, and four
  message-passing layers already span it — global attention has nothing to win, so *none* of §5
  is evidence that the attention is necessary. That question needs the large-diameter ladder
  (`size_ladder` in `encoder_probe.py`), which has seven known faults including an automorphism
  cap that silently drops 80 of 216 molecules, rungs sized against a molecule count that no
  longer survives the build, `max_spd=8` saturating against diameters up to 38, and a
  stereochemistry block that is entirely vacuous there (0 of 9,392 atoms carry a CIP code).
- **One seed.** Given that collapse at the wrong learning rate is roughly a 1-in-3 coin flip,
  single-seed results deserve the caveat.
- **`is_root` breaks D5 deliberately.** `atom_features` marks one atom so `spd_to_marked` is
  answerable. Marking collapses the automorphism group to that atom's stabiliser, so the
  encoder is equivariant with respect to the *marked* graph, not the molecule. **This marker is
  a probe artefact and should probably not ship in the production encoder** — the policy does
  not need it, and it cost one full defect cycle (§9, defect 6).
- **Four probes are floor checks, not discriminators** (§5).

---

## 9. Method note — six defects, all in task design

Every failure found in this battery was a defect in the *question*, not the architecture. This
is recorded because the pattern is the reusable part.

| | defect | what it faked |
|---|---|---|
| 1 | learning rate 1e-3, collapses ~1 seed in 3 | "message passing cannot do distances, 46%" |
| 2 | crystal-anchor subset read as QM9 | a data ceiling that did not exist |
| 3 | split on stereo-enumerated rows | 67.9% of held-out shared a skeleton with train |
| 4 | QM9 carries 0.0% chiral tags | a dead parity channel |
| 5 | target weighted by `hash(structure) mod 7` | unlearnable by construction, read as model failure |
| 6 | `orbit_size` label ignored the input's root mark | the last 1% |

Two instruments came out of it and should be used by default:

- **`curve_report.py`.** A summary statistic cannot distinguish under-training from
  memorisation from over-fitting from collapse — and all four are silent in a final score.
  Render the curves and look at them. The distinguishing signature of a **hash-like,
  unlearnable target** is a held-out loss that is *flat and identical at every dataset size*;
  more data is the right diagnosis only when the held-out curve **moves**.
- **The trivial-floor column.** A probe whose constant-predictor baseline is 70% was never
  testing much. Before the 2026-09-01 fix this column reported only a size-regression line,
  which on `orbit_size` scored 25.8 points *below* a constant predictor — a floor beneath the
  real floor is worse than no floor at all.

**Chirality on QM9.** QM9 as stored carries 0.0% chiral tags in the full set exactly as in the
subset, so `load_qm9_stereo` **assigns chirality arbitrarily** at every genuine tetrahedral
centre (`FindPotentialStereo`, which accounts for substituent symmetry), seeded per molecule
from a hash of its SMILES so a given molecule gets the same assignment at every dataset size.
72.5% of molecules carry at least one centre; 19.9% of heavy atoms end up CIP-labelled. This
replaced stereoisomer *enumeration*, which emitted two rows per parent that were the same graph
differing only in parity — near-duplicates that leaked 67.9% of the held-out set.

---

## 10. Natural successor

A **reconstruction objective** — pooled embedding to canonical SMILES via a small
autoregressive decoder — would replace this battery and prove information completeness rather
than sampling it. *n* probes establish that *n* functions are computable; reconstruction is a
single test that dominates all of them, and it has no target to design, which is where all six
defects above lived.

Canonical ordering is what makes it tractable: decoding a graph from one vector requires
solving node correspondence (GraphVAE-style matching is O(n⁴) and fiddly), whereas a canonical
SMILES *fixes* the order and collapses the problem to sequence decoding. Prior art: Kipf &
Welling 2016 for the inner-product adjacency decoder from per-node embeddings; Gómez-Bombarelli
et al. 2018 for the SMILES VAE.

Three traps worth knowing before building it:

1. **String syntax gaming** — SMILES→SMILES is partly solvable by learning grammar. Feed a
   randomised atom order and decode the canonical form.
2. **Memorisation of a small molecule space** — ~10⁵ molecules is ~17 bits to index, and a
   256-d pooled vector could hold a lookup table. Held-out reconstruction on the
   skeleton-grouped split is the only meaningful score.
3. **Sufficiency is not accessibility** — reconstruction with a nonlinear decoder proves the
   information survives, not that a linear head can read it. The linear probes here test the
   opposite property, so the two are complementary; for a policy consuming `g_i` through MLPs,
   sufficiency is the one that matters.
