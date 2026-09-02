# Review: chirality-augmented 2D molecular graph encoder

Status: **REVIEW / ARGUMENT — NOT AN ACCEPTED ARCHITECTURE DECISION**  
Reviewed: 2026-08-31  
Scope: `MPNNEncoder`, its stereochemical inputs, and the encoder probe battery  
Primary design under review: [`design/conformer_encoder_architecture.md`](design/conformer_encoder_architecture.md)

This document records a technical review. It does not establish repository policy, promote
the encoder to architecture of record, or supersede executable code, focused tests, or an
accepted owner decision.

---

## Verdict

The local-message-passing plus global-attention design is a strong experimental candidate.
Its static, once-per-molecule execution makes dense attention affordable, and its per-atom
outputs are well matched to the downstream n-body coordinate correlators.

It is not ready to be locked as the conformer encoder. Two issues are blocking:

1. the current parity feature is not yet demonstrated to be a well-defined, ordering-stable
   stereochemical label; and
2. the reported probe run does not yet establish either an expressivity bound for message
   passing or a capacity-controlled attention win.

The attention result is promising evidence that query-dependent global communication is
useful. It should be treated as an observed lead, not yet as a settled architecture result.

---

## What is strong in the design

### Static local/global hybrid

Running the graph encoder once and caching its outputs changes the normal cost trade-off.
For molecules with roughly 10–100 atoms, quadratic attention is paid once and amortised over
hundreds of trajectory steps. Local message passing can handle bond-local chemistry while
attention supplies a short global communication path.

### Per-atom and molecular outputs

The encoder returns both per-atom embeddings and a pooled molecular representation. This is
the right interface for coordinate identities assembled from the atoms participating in a
bond, angle, or dihedral, while retaining a separate molecular signal for quantities such as
`log Z(c)`.

### Selective and extensive aggregation

Concatenating a learned softmax-weighted reduction with an unnormalised sum preserves two
different capabilities: selecting a chemically important environment and retaining
size-dependent information. Zero-initialising the extensive projection in the attention and
broadcast blocks is a reasonable optimisation safeguard against the sum dominating the
routed signal at initialisation.

### Explicit stereochemical input

An adjacency-only encoder cannot distinguish enantiomers. Carrying stereochemical labels as
part of the condition is therefore necessary. The remaining question is whether the present
label definition is stable and whether the downstream model obeys the required physical
symmetry.

---

## Blocking findings

### 1. The implemented parity convention does not match the design convention

`graph_from_smiles` currently reduces RDKit's `CHI_TETRAHEDRAL_CW` and
`CHI_TETRAHEDRAL_CCW` tags directly to `+1` and `-1`:
[`models/graph_encodings.py`](../models/graph_encodings.py), `graph_from_smiles`.

Those tags describe winding relative to a local neighbour ordering. The surrounding design
instead calls for the sign of an improper dihedral in TreeSpec's canonical placement order:
[`design/conformer_conditional_stack.md`](design/conformer_conditional_stack.md),
“Choosing the encoder.” The implementation itself notes that these conventions may disagree
in sign.

Consequences:

- equivalent SMILES or atom orderings may not produce the same labelled graph after the
  resulting atoms are aligned;
- cached embeddings could become a function of serialization rather than molecule identity;
- mixing the RDKit and TreeSpec conventions across preprocessing and policy construction
  could silently invert stereochemical meaning.

Required gate: generate multiple atom orderings and isomeric SMILES for the same molecule,
align atoms by an explicit map, and require identical encoder outputs up to that permutation.
The parity feature should be computed in one canonical convention used by graph construction,
caching, coordinate construction, and downstream conditioning.

The current probe targets are all functions of adjacency and atom types. None requires the
model to retain parity, so the entire battery can succeed while the trained encoder assigns
zero effective weight to the stereochemical channel. Separate gates are needed for:

- **diastereomer sensitivity:** identical connectivity with different local configurations
  must produce distinguishable conditions;
- **enantiomer distinguishability:** globally inverted stereocentres must produce
  distinguishable labelled conditions; and
- **physical mirror symmetry:** mirrored conformers in an achiral environment must retain
  equal energy and partition function, while signed coordinate outputs transform under the
  appropriate reflection action.

A signed scalar permits these behaviours but does not enforce them. If E/Z, axial, allene,
or other non-tetrahedral stereochemistry is in scope, the input contract also needs explicit
bond or higher-order stereo labels. Otherwise the architecture should state that it supports
tetrahedral stereochemistry only.

### 2. `spd_to_root` asks for information the encoder is not given

The probe target is `s[0]`, the shortest-path distance from each atom to atom index zero:
[`models/encoder_probe.py`](../models/encoder_probe.py), `PROBES`.

The encoder receives no root marker or atom index. It is permutation-equivariant and cannot,
in general, know which atom an external serialization happened to place first. High accuracy
can therefore arise from correlations between the QM9 SMILES convention and the root atom's
chemical environment. It does not establish generic pairwise routing.

Two valid replacements are:

1. add a root-indicator input and select a root independently of atom order; or
2. evaluate a pairwise decoder on `(g_i, g_j)` for explicitly selected atom pairs.

The first asks whether information from a marked atom can be routed to every other atom. The
second asks whether the representation exposes pairwise graph distance. Either question is
well-defined; the current unmarked-root version is not.

### 3. Failure to reach 100% training accuracy is not by itself an expressivity proof

Deterministic labels establish that the Bayes error is zero. They do not establish that a
particular finite optimisation run has found the best function representable by the model.
Training shortfall can also result from:

- finite optimisation time;
- interference among the twelve jointly trained probe heads;
- learned uncertainty weights suppressing the gradients of difficult tasks;
- insufficient width or an unfavourable parameterisation; or
- the requirement that the selected quantity be linearly accessible, rather than merely
  computable by the encoder family.

The saved run scores only the first 1,000 members of the training set even when trained on
4,000 molecules: [`models/encoder_probe.py`](../models/encoder_probe.py), `run`.

Before calling a result an expressivity ceiling, overfit the disputed task in isolation,
score the complete training set, remove learned task reweighting, and sweep optimisation
time and width. A constructive collision—two labelled graphs the architecture must map
identically but whose targets differ—would be stronger evidence still.

### 4. The attention comparison is not capacity-controlled

At the reported width, the saved results contain 605,487 trainable parameters for `mp` and
`mp+rwse`, versus 803,275 for `mp+attn` and `mp+attn+spd`. The attention arms are therefore
about 32.7% larger.

This conflicts with the probe specification's own completion criterion requiring a matched
parameter budget: [`design/encoder_ssl_battery.md`](design/encoder_ssl_battery.md), “Done
when.” The current command-line implementation has no parameter-matching option.

The controlled architectural contrast in the existing run is `mp+rwse` versus `mp+attn`,
because those arms differ by attention while sharing RWSE. At 4,000 training molecules,
their recorded `spd_to_root` training accuracies are 57.6% and 94.7%, respectively. That is a
large and interesting separation, but the unmatched parameter budget, invalid root target,
and single seed prevent it from settling the design.

### 5. Several required controls remain absent

The probe design requires more than the cells in the current result file:

- a size-only baseline for size-correlated observables;
- multiple seeds with spread;
- an ethanol negative control;
- deliberate failure injections;
- a determinism gate; and
- a successful large-diameter evaluation.

The large-diameter ladder failed, and the QM9 run is precisely the regime in which four
message-passing layers already span much of each graph. Calling the QM9 attention advantage
a “lower bound” is plausible as a conjecture, but it is not established until the intended
large-diameter experiment runs successfully.

---

## Interpretation issues in the probe battery

### Graph-level distance statistics do not uniquely test routing

Diameter and Wiener index are graph-level invariants repeated at every node. A rank-one
aggregate-and-broadcast architecture is topologically suited to broadcasting such a scalar
once it can compute it. Their difficulty may expose inadequate structural information, but
it does not specifically demonstrate the need for query-dependent routing.

Eccentricity is the cleanest current routing probe because its answer varies with the query
atom. A corrected marked-root or pairwise-distance task would provide the complementary
controlled probe.

### Ring membership remains partly exposed by aromatic bond types

Removing `IsInRing` fixed the direct leak, but aromatic edge labels still imply ring
membership for aromatic atoms. A cleaner structural-encoding test would emphasize
non-aromatic rings and matched acyclic neighbourhoods, or use graph pairs chosen to be
indistinguishable to the baseline message-passing depth while differing in the target cycle
property.

### Exact-match tolerances should not define the apparent task ordering

The architecture note correctly identifies that the relative tolerances for Wiener index
and spectral moments are much tighter than for integer-valued probes. Reporting R² or
standardised residual error alongside an equalised tolerance is appropriate. Conclusions
should compare encoder arms within the same probe, not use the raw exact-match percentages
to order difficulty across differently calibrated probes.

---

## Architectural gaps to test before acceptance

### Through-bond path chemistry

The attention bias includes shortest-path length but not the bond sequence along that path.
Two atom pairs at the same graph distance can be connected through saturated single bonds,
an aromatic system, a conjugated chain, or an amide. Once the relevant path exceeds the
local message-passing radius, endpoint embeddings plus distance alone may not expose this
difference.

The surrounding design already identifies shortest-path edge encoding as chemically
important. A compact encoding of bond types along the path should be implemented as an
ablation, not assumed necessary in advance. The useful question is whether it improves
long-range coordinate identity or downstream rotamer weighting beyond distance-only
attention.

### Physical parity symmetry

Adding signed parity makes enantiomers distinguishable, but an unconstrained network may
assign unrelated `log Z` values or policies to globally mirrored labels. If both enantiomers
can appear in the conditioning set, the expected transformation should be either enforced by
the architecture or strongly pinned by mirrored data and contract tests.

The relevant symmetry is not simply “the output is scalar.” Bond lengths and angles remain
unchanged under reflection, while signed dihedrals change sign according to the coordinate
chart. The policy means, variances, and correlations therefore have different transformation
rules that should be stated and tested explicitly.

### Completeness claim

“Complete invariant of the chirality-labelled pointed graph” is too strong for the current
evidence. Finite RWSE features and SPD-biased attention are not known to be injective over
all labelled molecular graphs, and a finite probe suite cannot establish injectivity.

A defensible target is:

> The representation is sufficient for the graph and stereochemical distinctions exercised
> by the declared molecule domain and downstream conformer task, up to labelled-graph
> automorphism.

That claim can be supported incrementally by probe results, collision searches on enumerated
small molecular graphs, and downstream held-out behaviour.

---

## Recommended acceptance gate

Keep the present hybrid as the leading experimental candidate, but do not call it settled
until all of the following hold:

1. **Canonical stereo contract** — one parity convention across graph preprocessing,
   TreeSpec, caching, and policy conditioning; invariant under atom and SMILES reordering.
2. **Stereo behaviour gates** — diastereomer sensitivity plus enantiomer distinguishability
   and physical mirror symmetry.
3. **Valid routing probes** — eccentricity plus a marked-root or pairwise-distance task;
   graph-level distance statistics reported separately.
4. **Controlled comparison** — matched parameter or compute budget, at least three seeds,
   full-training-set scoring, size-only baselines, and demonstrated failure injections.
5. **Long-diameter evidence** — a repaired ladder covering peptide- and alkane-scale graph
   diameters, with `max_spd` chosen from the evaluated diameter range.
6. **Downstream confirmation** — broadcast versus attention on a molecule with genuine
   long-range rotamer interactions, accompanied by an ethanol null control.
7. **Path-feature ablation** — distance-only attention versus attention augmented with
   through-bond path chemistry.

If attention retains a clear advantage on the query-dependent probes and downstream
rotamer task under those controls, it is purchased. If its advantage disappears after root,
capacity, or size confounds are removed, the cheaper broadcast encoder remains adequate.

---

## Evidence basis and limitations of this review

Reviewed sources:

- [`models/graph_encoder.py`](../models/graph_encoder.py)
- [`models/graph_encodings.py`](../models/graph_encodings.py)
- [`models/encoder_probe.py`](../models/encoder_probe.py)
- [`models/results/encoder_probe_qm9b.json`](../models/results/encoder_probe_qm9b.json)
- [`tests/models/test_graph_encoder.py`](../tests/models/test_graph_encoder.py)
- [`tests/models/test_graph_encodings.py`](../tests/models/test_graph_encodings.py)
- [`design/conformer_encoder_architecture.md`](design/conformer_encoder_architecture.md)
- [`design/encoder_ssl_battery.md`](design/encoder_ssl_battery.md)
- [`design/conformer_conditional_stack.md`](design/conformer_conditional_stack.md)

The numerical claims above are **OBSERVED** from the single saved QM9 run (`seed=0`, 4,000
training molecules, 3,000 steps) and do not generalise beyond that run without replication.
The interface and probe findings are **MECHANISM** claims derived from the inspected
implementation under its current inputs.

No runtime stereochemical permutation test was executed during this review because the
active Python environment did not provide RDKit. The ordering-stability concern therefore
remains a mechanism-based review finding that must be closed by an executable test.

---

## Documentation status conflict

The encoder architecture note describes `conformer_conditional_stack.md` section 5 as the
architecture of record, while that section currently labels encoder selection “STUB, still
under discussion.” No accepted encoder decision appears in `current_decisions.md`.

Until the owner resolves that conflict, the conservative reading is that the implementation
and results are an active conformer-refactor experiment, while the final encoder choice
remains open.
