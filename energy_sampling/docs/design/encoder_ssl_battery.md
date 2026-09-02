# Encoder self-supervision battery

Argument doc. What the encoder has to be able to compute, which probes separate the
candidate architectures, and what each probe's floor and ceiling are. Written 2026-08-27.

**Why this exists.** Owner decision 1 makes toy self-supervised validation the gate on the
encoder before it carries a policy. Decisions taken 2026-08-27 promote it further: it is now
the *instrument* that settles three questions that would otherwise be argued rather than
measured —

| open question | the probe that answers it |
|---|---|
| structural encoding: RWSE, LapPE+SignNet, or none | the ring block, plus the determinism gate |
| broadcast vs global attention (§5's own discriminator) | the distance block |
| dynamic capacity level 1 vs higher | deferred; follows from the above |

**And it is now on the critical path.** Molecular identity enters the condition as the
pooled encoder output (owner decision, 2026-08-27), so the encoder is a prerequisite for
Sequencing step 3, not an optional upgrade. It is *not* a prerequisite for step 2 — the set
policy swap runs on `dof_features`.

---

## 1. What g_i has to contain

The requirement, stated exactly: **`g_i` should be a complete invariant of the
chirality-labelled pointed graph (G, i), up to automorphism of that labelled graph.**

Two clauses, and they are not the same clause:

- *the graph* — global observables must be decodable from `g_i` alone;
- *where I am in it* — per-node observables that depend on position must be decodable too.

"Up to automorphism" is not a concession. Two automorphic atoms *should* collide: a method
that separated them would be encoding spanning-tree storage order, which is the defect the
flat MLP already has. Note the limit of what the parity labels buy: they separate
enantiomers, but valine's two methyls stay automorphic even with parity, because the swap
fixes every label. That residual blindness is **correct while the force field is
graph-typed**, since E(σ·r) = E(r) for every graph automorphism σ when every parameter is
graph-determined — the encoder is exactly as blind as the target. It stops being correct the
day the potential stops being graph-typed (`ff_from_graph` upgrade, an MLIP, QM). Record the
expiry; do not design as if it were a principle.

---

## 2. Probes, sorted by what they discriminate

Sorting is the point. The blocks are ordered by whether an aggregate-and-broadcast encoder
can reach them, because that ordering *is* §5's discriminator.

### Block A — broadcast-reachable. Floor checks, not discriminators.

Every one is a sum over nodes, so a rank-one global vector suffices. A candidate failing
these is broken; a candidate passing them has said nothing about routing.

| probe | target | why it is here |
|---|---|---|
| heavy-atom count | scalar | the extensive channel reached the node at all |
| molecular formula | count per element | same, per-species |
| degree histogram | counts by degree | pure node sum |
| cycle rank | `|E| - |V| + C` | exact, and the cheapest ring signal |
| Laplacian spectral moments | `tr(L^k)`, k = 2..5 | closed-walk counts; the global form of what RWSE holds locally |

### Block B — the 1-WL breaker. Separates "has a structural encoding" from "has none".

A plain MPNN is bounded by 1-WL and cannot count cycles. RWSE makes these near-trivial.

| probe | target |
|---|---|
| ring membership | per-node bool |
| smallest ring size containing i | per-node int, 0 if acyclic |
| aromatic-ring membership | per-node bool |

### Block C — distance. Not broadcast-reachable. **This is where attention earns its place.**

Every one depends on a *specific* far node, not on any sum, so a rank-one broadcast cannot
produce it.

| probe | target | note |
|---|---|---|
| **eccentricity of i** | per-node int | the sharpest single probe: "where am I" as a scalar |
| graph diameter | scalar, per-node prediction | requires both ends |
| Wiener index | scalar, per-node prediction | needs the whole distance matrix |
| SPD from i to the designated root | per-node int | pairwise, the most direct form |

### Block D — symmetry. Probes the degeneracy behaviour directly.

| probe | target | note |
|---|---|---|
| orbit size \|orbit(i)\| | per-node int | ceiling-bearing — see §4 |
| \|Aut(G)\| | scalar | graph-level |

### Determinism — a GATE, not a score

Same molecule through ≥5 SMILES orderings and ≥5 embedding seeds must give **identical
`g_i`** up to the known orbit permutation. RWSE passes by construction (a matrix power).
Naive LapPE **fails**: eigenvectors are defined up to sign, and up to a full O(m) rotation
inside a degenerate eigenspace, which molecular symmetry produces constantly. SignNet is what
makes LapPE pass.

Treat a failure here as **disqualifying regardless of accuracy**. `{f_j}` is cached per
molecule, so a non-deterministic encoding makes the cached condition a function of the
solver rather than of the graph — the same defect as the A2 chart problem already on record.

---

## 3. Arms

Cross the encoder against the encoding. Capacity stays at the §5 default until this reports.

| arm | message passing | structural encoding | global attention |
|---|---|---|---|
| `mp` | 4-layer MPNN | none | no |
| `mp+rwse` | 4-layer MPNN | RWSE, k = 16 | no |
| `mp+lap` | 4-layer MPNN | LapPE + SignNet | no |
| `mp+rwse+attn` | 4-layer MPNN | RWSE, k = 16 | yes, SPD-biased |

Read Block A across all four as a floor. Read Block B as `mp` against the rest. Read Block C
as `mp+rwse` against `mp+rwse+attn` — that comparison, on that block, **is** the answer to
whether attention is purchased or unpurchased complexity.

---

## 4. Baselines, ceilings, and controls — the part that makes it falsifiable

Four ways this battery could report a number that means nothing. Each has a required guard.

- **Size confounding.** Diameter, Wiener index, ring count and eccentricity all correlate
  hard with atom count. A model that learned only "how big am I" scores well on all of them.
  **Guard:** regress every observable on N alone first, and report the encoder against that
  baseline, not against zero. An arm that fails to beat size-only has not demonstrated
  anything.
- **Orbit ceiling.** Leave-one-out node identification — decision 1's named task — is the
  orbit probe in disguise: it can only succeed up to automorphism, so its ceiling is set by
  the orbit structure and is **below 1.0 on any symmetric molecule**. **Guard:** compute the
  per-molecule ceiling first and score against it. Without this, correct behaviour reads as
  failure.
- **The negative control.** Ethanol, per §5. It has no rings, no long-range structure and
  trivial symmetry, so no arm should beat any other on it. **Guard:** an arm showing gains on
  ethanol means the comparison is measuring noise, and the whole table is void until that is
  explained.
- **Capacity confounding — found the hard way, 2026-08-27.** At equal width the attention arm
  carries qkv, projection and SPD-bias weights per layer, giving it ~43% more parameters
  (222k vs 155k at hidden 64). In the first functional run it then won on **block A**, which
  is broadcast-reachable and should not separate the arms at all — the signature of capacity
  leaking into the comparison. **Guard:** run with `--match-params`, which sizes every arm to
  a common budget and prints the achieved counts and their deviation. A block A separation
  after matching means something else is wrong; a block A separation without matching means
  the table is not yet readable.
- **Checks that cannot fail.** Block A on a working broadcast is near-tautological. **Guard:**
  keep it, but label it a floor in the output, and never quote a Block A win as evidence for
  an architecture.

**Injection requirement.** Every probe ships with a deliberate perturbation that must make it
fail: zero the structural encoding (Block B must collapse), truncate message passing to 1
layer (Block C must collapse), and randomise the sign of the LapPE at read time (the
determinism gate must fire). A probe with no demonstrated failure mode is not a probe.

---

## 5. Done when

1. All four arms run on one committed molecule set with a **matched parameter budget**,
   fixed before looking, with the achieved counts printed.
2. Every cell carries a seed count and spread, and its size-only baseline.
3. The determinism gate is reported as pass/fail per arm, ahead of any accuracy number.
4. Each of the three injections above is demonstrated to fire.
5. The Block C comparison is reported **including a null result** — no measurable difference
   means the broadcast was sufficient and the attention is unpurchased, which is a valid and
   useful outcome that closes question 3.

---

## 6. Explicitly out of scope

Pretraining as a *target* — this suite enters as a diagnostic first, per decision 1, and
whether the encoder is later frozen and pretrained by this route is a separate decision that
the results inform rather than settle. Also out: capacity escalation in the dynamic branch,
which follows from these results rather than being decided alongside them.
