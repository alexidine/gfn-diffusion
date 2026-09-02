"""Encoder probing battery, v2 -- learning curves against an achievable 100%.

Replaces the R^2-on-a-tiny-holdout framing of models/encoder_ssl.py, which could not resolve
the differences it was being asked about. Spec: docs/design/encoder_ssl_battery.md.

THE REFRAME, AND IT IS THE WHOLE POINT. Every target here is a DETERMINISTIC FUNCTION OF THE
GRAPH -- eccentricity, orbit size, ring size, Wiener index. There is no label noise and no
irreducible error, so the achievable score is exactly 100% and anything less is the model's
fault. That converts a noisy ranking into a pass/fail plus a learning curve, and it splits
the two obstructions that were previously conflated:

    cannot fit TRAIN            -> an EXPRESSIVENESS bound. No test set required, and no
                                   amount of data fixes it. (1-WL cannot count cycles.)
    fits train, fails TEST      -> SAMPLE COMPLEXITY. Read it off the curve.

It also self-diagnoses leakage: a probe whose answer is already in the input saturates at
100% for every arm at trivial dataset size. That is how the previous battery's `in_ring` leak
should have been caught, instead of by reading the feature list afterwards.

THE HEADS ARE LINEAR, ON PURPOSE. A head with a hidden layer can COMPUTE what the
representation does not contain, which defeats the question being asked. One `nn.Linear` per
task is a linear probe, so "100%" means the answer is explicitly present in g_i rather than
merely recoverable from it.

LOSS IS EQUAL-WEIGHT MSE ON STANDARDISED TARGETS, and that is deliberate. Kendall
uncertainty weighting was the default until 2026-09-01 and CAUSED A TRAINING COLLAPSE: its
effective weight on a task is 1/mse, so fitting better raises the weight, raises the
gradient, and runs away. Standardisation already balances the scales -- every task starts at
MSE ~ 1 -- so the weighting bought nothing and cost stability. Available behind
`--loss-weighting uncertainty` only to reproduce pre-2026-09-01 runs, which are all
contaminated by it. See project_encoder_training_collapse.

    python -m models.encoder_probe --n-train 4000 --steps 3000 --device cuda
    python -m models.encoder_probe --smoke
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from models.graph_encoder import MPNNEncoder, dense_spd_batch
from models.graph_encodings import (
    bond_features_from_smiles, cycle_rank, degree_histogram, diameter, eccentricity,
    graph_from_smiles, lap_pe, orbit_sizes, ring_membership, rwse, shortest_paths,
    smallest_ring_size, spectral_moments, to_dense_adjacency, wiener_index,
    cip_codes, wl_colours, pi_degree, mol_for_labels, canonical_root,
)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
QM9_FULL = r'D:\crystal_datasets\qm9_dataset.pt'
QM9_DIR = r'D:\crystal_datasets\conditional\anchors'

ELEMENTS = (1, 6, 7, 8, 9, 16, 17)
MAX_DEGREE = 5


# --------------------------------------------------------------------------- molecules

def load_qm9(n: int, chunks: Sequence[int] = tuple(range(40))) -> List[str]:
    """SMILES from the FULL QM9 dataset -- 133,728 molecules, every SMILES distinct.

    USE THE FULL SET, NOT THE ANCHOR CHUNKS. Until 2026-09-01 this read
    `conditional/anchors/qm9c100k_chunk*.pt`, which is a CRYSTAL-ANCHOR SUBSET holding about
    5,850 unique molecules across 30 chunks -- roughly 4% of QM9 -- multiplied by ~8.2 crystal
    rows each, so a request for 5,564 rows returned 585 unique molecules before the dedup went
    in. Sampling 20,000 rows of the full set gives 19,996 unique canonical SMILES and 19,996
    unique parent skeletons: duplication is effectively nil.

    That subset was the binding constraint on the battery's hardest probes. `chiral_moment`
    showed a FLAT held-out loss against a train loss falling two orders of magnitude --
    memorisation, whose fix is more molecules, and there were none left to give.

    QM9 STILL CANNOT TEST THE RADIUS QUESTION. Diameter median 6, p95 8, max 9; four
    message-passing layers reach 4 hops from each end, so essentially every QM9 molecule is
    fully covered by message passing alone and the oversquashing argument -- the entire reason
    attention is a candidate -- does not arise. Gly4's diameter is 14, Gly6's is 20. QM9 is the
    right set for expressiveness and sample complexity, the WRONG set for the distance block.
    Pair it with :func:`size_ladder`.

    AND NO STEREOCHEMISTRY: 0.0% of molecules carry a chiral tag, in the full set exactly as in
    the subset. Anything touching chirality must go through :func:`load_qm9_stereo`.
    """
    import torch as _t
    if os.path.exists(QM9_FULL):
        out, seen = [], set()
        for s in _t.load(QM9_FULL, weights_only=False, map_location='cpu'):
            smi = getattr(s, 'smiles', None)
            if smi and smi not in seen:
                seen.add(smi)
                out.append(smi)
                if len(out) >= n:
                    return out
        if out:
            if len(out) < n:
                raise RuntimeError(
                    'asked for %d unique QM9 molecules, the full set yielded %d.' % (n, len(out)))
            return out
    # fallback: the crystal-anchor chunks. Deduplicated, because the ROWS ARE NOT MOLECULES.
    out, seen = [], set()
    for c in chunks:
        path = os.path.join(QM9_DIR, 'qm9c100k_chunk%d.pt' % c)
        if not os.path.exists(path):
            continue
        for s in _t.load(path, weights_only=False, map_location='cpu'):
            smi = getattr(s, 'smiles', None)
            if smi and smi not in seen:
                seen.add(smi)
                out.append(smi)
            if len(out) >= n:
                return out
    if len(out) < n:
        raise RuntimeError(
            'asked for %d UNIQUE QM9 molecules, found %d. The anchor chunks hold only ~5,850 '
            'unique molecules; the full dataset holds 133,728.' % (n, len(out)))
    return out


def load_qm9_stereo(n: int, chunks: Sequence[int] = tuple(range(40))) -> List[str]:
    """QM9 with chirality ASSIGNED ARBITRARILY at every genuine tetrahedral centre.

    QM9 AS STORED HAS NO STEREOCHEMISTRY -- 0.0% of molecules carry a chiral tag, in the full
    133k set exactly as in the anchor subset -- so `graph_from_smiles`' parity column is
    identically zero and the encoder's only stereochemical channel is DEAD on raw QM9. Every
    result produced on it is compatible with the parity feature carrying no weight at all.

    THE FIX IS TO MANUFACTURE THE LABEL, NOT TO FIND IT. R-versus-S at a given centre is a
    free choice; what requires analysis is WHICH atoms are stereogenic. So: ask RDKit for the
    genuine tetrahedral centres (`FindPotentialStereo`, which accounts for substituent
    symmetry rather than merely counting four distinct neighbours), flip a coin at each, then
    re-derive the CIP codes from the result.

    WHY THIS REPLACED STEREOISOMER ENUMERATION. `EnumerateStereoisomers(maxIsomers=2)` emitted
    TWO rows per parent that are the SAME GRAPH differing only in parity -- near-duplicates
    that leaked 67.9% of the held-out set across a row-wise split, and that halve the
    structural diversity of any fixed row budget. One arbitrary assignment per parent removes
    the sibling mechanism at its source instead of grouping around it.

    Measured over 4,000 QM9 molecules: 72.5% carry at least one tetrahedral centre (mean 2.58,
    max 8), 97.1% of those yield a CIP code once assigned, and 19.9% of heavy atoms end up
    CIP-labelled -- against 10.7% under enumeration.

    THE ASSIGNMENT IS SEEDED PER MOLECULE, from a hash of its SMILES, so a given molecule gets
    the SAME chirality in every run and at every dataset size. Seeding from position instead
    would relabel the same molecule between the n=2000 and n=20000 rungs and quietly turn a
    sample-complexity curve into a comparison of different problems.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    import hashlib

    out = []
    for smi in load_qm9(n, chunks):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        try:
            centres = [e.centeredOn for e in Chem.FindPotentialStereo(m)
                       if str(e.type) == 'Atom_Tetrahedral']
            if centres:
                h = hashlib.blake2b(smi.encode(), digest_size=8).digest()
                bits = int.from_bytes(h, 'big')
                for j, idx in enumerate(centres):
                    m.GetAtomWithIdx(idx).SetChiralTag(
                        Chem.ChiralType.CHI_TETRAHEDRAL_CW if (bits >> (j % 64)) & 1
                        else Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                Chem.AssignStereochemistry(m, cleanIt=True, force=True)
            out.append(Chem.MolToSmiles(m))
        except Exception:
            continue
        if len(out) >= n:
            break
    return out[:n]


def parent_skeleton(smi: str) -> str:
    """The molecule with all stereochemistry stripped -- its structural equivalence class.

    THE SPLIT MUST BE GROUPED ON THIS, not on rows. `load_qm9_stereo` enumerates
    stereoisomers, and two stereoisomers of one parent are the SAME GRAPH: identical
    adjacency, identical formula, degrees, rings, distances and orbits. They differ only in
    the parity column. So a row-wise split puts near-copies on both sides and the held-out
    column reads as generalisation when it is memorisation. Measured on the shipped split:
    67.9% of held-out rows had a same-skeleton sibling in training at n_train=4000, and it
    GROWS with n (14.0% -> 50.5% -> 67.9%) -- the very axis a sample-complexity curve is read
    along, so the curve bent for the wrong reason.

    Third instance of this failure in this battery: raw anchor rows (8.2x duplication), then
    SMILES-level duplicates, now stereo siblings. The lesson that generalises is to split on
    the STRUCTURAL EQUIVALENCE CLASS rather than on whatever the loader happened to return.
    """
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def mirror_smiles(smi: str) -> str:
    """Invert every tetrahedral centre -- the mirror image, as a labelled graph."""
    return smi.replace('@@', chr(0)).replace('@', '@@').replace(chr(0), '@')


def size_ladder() -> List[str]:
    """A large-diameter set, which QM9 cannot supply (median diameter 6 vs Gly4's 14).

    Sized to stand alone as a dataset rather than as a garnish: a few hundred molecules
    spanning diameter ~4-30. Branched and ring-terminated variants are included at each
    length ON PURPOSE -- on a pure linear ladder, eccentricity is a function of atom count
    and the probe degenerates into the size baseline it is supposed to beat.
    """
    out = []
    for k in range(1, 13):                                   # Gly1 .. Gly12
        out.append('NCC(=O)' * (k - 1) + 'NCC(=O)O')
        out.append('N' + 'C(C)C(=O)' * k + 'O')              # poly-alanine, side chains
    for n in range(2, 26):
        out.append('C' * n)                                  # n-alkane
        out.append('OC' + 'C' * max(n - 2, 0) + 'O')         # alpha,omega-diol
        out.append('NC' + 'C' * max(n - 2, 0) + 'N')         # alpha,omega-diamine
    for n in range(2, 20):                                   # branched at one end
        out.append('CC(C)' + 'C' * n)
        out.append('CC(C)(C)' + 'C' * n)
    for n in range(1, 18):                                   # ring at one or both ends
        out.append('C' * n + 'c1ccccc1')
        out.append('c1ccccc1' + 'C' * n + 'c1ccccc1')
        out.append('C1CCCCC1' + 'C' * n)
    for n in range(1, 12):                                   # heteroatom-punctuated chains
        out.append('C' + 'OC' * n)
        out.append('C' + 'NC' * n)
        out.append('C' + 'C(=O)NC' * n)
    return sorted(set(out))


# ------------------------------------------------------------------------------ probes

@dataclass(frozen=True)
class Probe:
    name: str
    block: str            # 'A' floor | 'B' cycles | 'C' distance | 'D' symmetry
    dim: int
    fn: Callable          # ctx dict -> labels
    per_node: bool
    integral: bool = True   # exact-match after rounding is meaningful
    #: score only where the label is non-zero. For `cip_code` the all-atom majority class is
    #: 89.8% (most atoms are not stereocentres), so an unmasked score is dominated by
    #: predicting "not a stereocentre" and says almost nothing about R/S discrimination.
    nonzero_only: bool = False


PROBES: List[Probe] = [
    # ---- block A: floor. Sums over atoms; a broadcast suffices.
    Probe('formula', 'A', len(ELEMENTS),
          lambda ctx: np.array([float((ctx['z'] == el).sum()) for el in ELEMENTS]), False),
    Probe('degree_hist', 'A', MAX_DEGREE + 1,
          lambda ctx: degree_histogram(ctx['e'], ctx['n'], MAX_DEGREE).astype(float), False),
    #: THE ONLY LABEL THAT DEPENDS ON edge_attr. Without it the encoder can score 100% on
    #: everything while discarding all four bond-order columns, and bond order is what a
    #: torsion profile is made of. Trivial on purpose -- it is a tripwire, not a challenge.
    Probe('pi_degree', 'A', 1,
          lambda ctx: pi_degree(ctx['mol']).astype(float)[:, None], True),
    # ---- block B: cycles. 1-WL cannot count them.
    #: subsumes the retired `ring_member` (its > 0 test matches on 100.0% of atoms), which was
    #: leaked -- and NOT through aromaticity as first thought: only 15.4% of ring atoms here
    #: are aromatic, and it leaked through degree plus incident bond order.
    Probe('smallest_ring', 'B', 1,
          lambda ctx: smallest_ring_size(ctx['e'], ctx['n']).astype(float)[:, None], True),
    # ---- block C: distance. Depends on a SPECIFIC far atom, not on any sum.
    Probe('eccentricity', 'C', 1,
          lambda ctx: eccentricity(ctx['spd']).astype(float)[:, None], True),
    #: root is RESAMPLED at collation time and marked in the input, so the probe tests
    #: routing from an ARBITRARY marked atom. Two earlier versions were ill-posed: the
    #: original asked for distance to atom index 0 (a serialisation property, 20.7% label
    #: survival), and the first fix seeded the root from the SMILES text, which marked a
    #: different atom on 88.7% of re-serialisations.
    Probe('spd_to_marked', 'C', 1,
          lambda ctx: np.where(ctx['spd'][ctx['root']] < 0, -1.0,
                               ctx['spd'][ctx['root']]).astype(float)[:, None], True),
    # ---- block D: symmetry. Requires comparison against every other atom.
    #: THE LABEL INCLUDES is_root, BECAUSE THE INPUT DOES. `atom_features` marks one atom so
    #: that `spd_to_marked` is answerable, and MARKING AN ATOM DESTROYS THE SYMMETRY: the input
    #: graph's automorphism group is the STABILISER Aut(G)_root, whose orbits are strictly
    #: FINER than those of the unmarked graph. Scoring against unmarked orbits asked the model
    #: for a property of a graph it is never shown, and it answered correctly about the graph
    #: it WAS shown. Measured on the 2026-09-01 checkpoint at n=20,000: 5.05% of held-out atoms
    #: have their orbit size changed by the mark, they carry 77.8% of ALL squared error at an
    #: 11.5% error rate against 0.42% elsewhere, and on those misses the prediction equals the
    #: ROOT-MARKED orbit size 67.2% of the time. 61.7% of all misses were EXACTLY HALF the
    #: unmarked orbit size -- the mark breaking a 2-fold symmetry -- and 83.1% were
    #: under-estimates, the direction a finer partition forces. Controlled for symmetry level
    #: the effect is still 9.5x.
    #: Also on the CHIRALITY-labelled graph: orbits of the bare z-labelled graph differ on
    #: 0.55% of atoms and 1.0% of molecules.
    Probe('orbit_size', 'D', 1,
          lambda ctx: orbit_sizes(ctx['e'], ctx['n'],
                                  labels=list(zip(ctx['z'].tolist(),
                                                  ctx['parity'].tolist(),
                                                  [int(i == ctx['root']) for i in
                                                   range(ctx['n'])])))
          .astype(float)[:, None], True),
    # ---- block E: stereochemistry. Nothing else in the battery needs the parity channel.
    #: TRIPWIRE, not a challenge -- it is now a COPY of the parity input by construction, and
    #: that is the point. Its job is to fail loudly if the chirality channel is dropped, the
    #: same role `pi_degree` plays for bond features. Below ~100% means the input is not
    #: surviving the encoder, and every chirality claim is void.
    #:
    #: The previous version referenced the INPUT to canonical rank and the LABEL to CIP. Same
    #: physical handedness, different bookkeeping, agreeing on 59.4% of centres -- so the model
    #: was being asked to re-derive RDKit's canonicalisation. It scored 0% (chance) while
    #: merely copying its own input would have scored 59.4%. Manufacturing difficulty by
    #: mismatching conventions is not the same as testing a capability.
    #:
    #: ROUNDING TRAP on any +/-1 target: under MSE the optimal prediction is the conditional
    #: mean 2p-1 at sign accuracy p, so exact-match-after-rounding reports EXACTLY 0% until
    #: p > 0.75. Harmless for a tripwire that should sit at ~100%; do NOT reuse this scoring
    #: for a hard +/-1 target -- use classification.
    Probe('cip_code', 'E', 1,
          lambda ctx: cip_codes(ctx['mol']).astype(float)[:, None], True,
          nonzero_only=True),
    #: THE REAL CHIRALITY TEST now that `cip_code` is a tripwire: the sign is an input column
    #: but the WL weight is not, so no copy of a single input reproduces it.
    #: forces chirality to be combined with graph POSITION rather than merely summed. The
    #: retired `parity_sum` was a plain graph sum of an input column (linear-on-graph-sums
    #: reproduced it at 100%) and collided on 22% of distinct stereoisomer pairs -- (R,S)
    #: against (S,R). Weighting by a WL colour separates 99.0% of those pairs.
    #: MODULUS 7, not 97. Measured over 121 distinct stereoisomer pairs: modulus 97 separates
    #: 95.9% of them but its SD is 81.7, so exact match demands 0.006 of one SD -- tighter
    #: than the `wiener` probe retired for exactly that. Modulus 7 separates 93.4% at 0.071
    #: SD, a 12x looser metric for 2.5 points of separation.
    #: WEIGHT IS THE DISTANCE TO THE MARKED ATOM, not a WL hash. The hash version was
    #: UNLEARNABLE BY CONSTRUCTION and its flat held-out loss was identical at 2,000 / 8,000
    #: and 20,000 molecules -- ten times the data moved it by nothing. Measured: 3-round WL
    #: gives 18,933 distinct classes over 54,027 train atoms, 73.0% of held-out classes are
    #: never seen in training, and 29.0% of held-out ATOMS sit in an unseen class. The mod-7
    #: value carries essentially no structural signal -- predicting it from (degree, element)
    #: scores 19.5% against a 14.3% chance rate -- so for those atoms the model must invent a
    #: weight it cannot infer, and the target SUMS the errors over the molecule.
    #: The weight exists to stop the statistic collapsing to a plain sum, which cannot tell
    #: (R,S) from (S,R). Distance-to-root does that just as well and IS learnable: separation
    #: over 600 distinct stereoisomer pairs is 90.7% against the hash's 91.2%, and the model
    #: already computes this exact quantity at 100/100 on `spd_to_marked`.
    Probe('chiral_moment', 'E', 1,
          lambda ctx: np.array([float((cip_codes(ctx['mol']) *
                                       (1.0 + np.where(ctx['spd'][ctx['root']] < 0, 0,
                                                       ctx['spd'][ctx['root']]))).sum())]),
          False),
]

#: RETIRED 2026-08-31 after a task-design audit, with reasons:
#:   n_atoms       exactly formula.sum() and degree_hist.sum() (100% agreement)
#:   cycle_rank    an exact linear function of degree_hist, R^2 = 1.0000
#:   diameter      exactly max(eccentricity) (100%), and 100/100 for every arm already
#:   spectral_moments  R^2 0.979 from (formula, degree_hist, n); tolerance 0.071 SD over 4
#:                     components, so it measured the metric
#:   wiener        R^2 0.964 from size features; tolerance 0.028 SD, 7x tighter than n_atoms
#:   ring_member   leaked (linear 1-hop 95.6% against a 73.2% base rate); subsumed by
#:                 smallest_ring
#:   parity_sum    ill-posed (55.3% survival), leaked (linear-on-graph-sums 100%), and blind
#:                 to (R,S) vs (S,R)
RETIRED = ('n_atoms', 'cycle_rank', 'diameter', 'spectral_moments', 'wiener',
           'ring_member', 'parity_sum')

BLOCK_NAMES = {'A': 'floor (broadcast-reachable)', 'B': 'cycles (1-WL breaker)',
               'C': 'distance (routing)', 'D': 'symmetry',
               'E': 'stereochemistry (needs the parity channel)'}
BLOCKS = ('A', 'B', 'C', 'D', 'E')   # block E was trained but never printed


# ----------------------------------------------------------------------------- samples

@dataclass
class Sample:
    smiles: str
    n: int
    x: np.ndarray
    struct: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    spd: np.ndarray
    labels: Dict[str, np.ndarray] = field(default_factory=dict)


def atom_features(z, edge_index, parity, n, root: int = 0) -> np.ndarray:
    """One-hot element, one-hot degree, parity. Parity is here and nowhere else -- every
    structural encoding is a function of the adjacency and therefore enantiomer-blind."""
    unknown = set(np.unique(z).tolist()) - set(ELEMENTS)
    if unknown:
        raise ValueError(f'elements {sorted(unknown)} outside ELEMENTS')
    el = np.zeros((n, len(ELEMENTS)))
    for j, e in enumerate(ELEMENTS):
        el[:, j] = (z == e)
    deg = np.minimum(to_dense_adjacency(edge_index, n).sum(1).astype(int), MAX_DEGREE)
    dg = np.zeros((n, MAX_DEGREE + 1))
    dg[np.arange(n), deg] = 1.0
    #: the marked root, one column. Without it `spd_to_marked` would be asking for something
    #: the encoder is never given -- exactly what invalidated its predecessor.
    is_root = np.zeros((n, 1))
    is_root[root] = 1.0
    return np.concatenate([el, dg, parity.astype(float)[:, None], is_root], axis=1)


def build_sample(smiles: str, encoding: str, k: int, root_seed: int = 0) -> Sample:
    z, e1, parity = graph_from_smiles(smiles)
    n = len(z)
    # CANONICAL RANK, so the root is a function of the molecule and is REPRODUCIBLE. Atom
    # index 0 was a serialisation property (20.7% label survival); resampling per call fixed
    # that but made the training set non-reproducible, so "train accuracy" stopped being a
    # memorisation measure at all.
    mol = mol_for_labels(smiles)
    root = canonical_root(mol)
    bf = bond_features_from_smiles(smiles)
    edge_index = np.concatenate([e1, e1[::-1]], axis=1)
    edge_attr = np.concatenate([bf, bf], axis=0)
    spd = shortest_paths(e1, n)
    struct = {'none': lambda: np.zeros((n, k)),
              'rwse': lambda: rwse(e1, n, k=k),
              'lap': lambda: lap_pe(e1, n, k=k)}[encoding]()
    s = Sample(smiles, n, atom_features(z, e1, parity, n, root), struct,
               edge_index, edge_attr, spd)
    ctx = {'z': z, 'e': e1, 'n': n, 'spd': spd, 'root': root, 'parity': parity,
           'mol': mol}
    for p in PROBES:
        v = np.asarray(p.fn(ctx), dtype=np.float64)
        s.labels[p.name] = v.reshape(n, p.dim) if p.per_node else v.reshape(1, p.dim)
    return s


def collate(samples: Sequence[Sample], device, want_spd: bool):
    offs, xs, sts, eis, eas, batch = 0, [], [], [], [], []
    for gi, s in enumerate(samples):
        xs.append(s.x); sts.append(s.struct); eis.append(s.edge_index + offs)
        eas.append(s.edge_attr); batch.append(np.full(s.n, gi)); offs += s.n
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=device)
    b = {'x': t(np.concatenate(xs)), 'struct': t(np.concatenate(sts)),
         'edge_index': t(np.concatenate(eis, axis=1), torch.long),
         'edge_attr': t(np.concatenate(eas)),
         'batch': t(np.concatenate(batch), torch.long),
         'n_graphs': len(samples)}
    if want_spd:
        length = max(s.n for s in samples)
        b['spd'] = dense_spd_batch([s.spd for s in samples], len(samples), length, device)
    for p in PROBES:
        lab = (np.concatenate([s.labels[p.name] for s in samples]) if p.per_node else
               np.concatenate([np.repeat(s.labels[p.name], s.n, axis=0) for s in samples]))
        b['y_' + p.name] = t(lab)
    return b


# ------------------------------------------------------------------------------- model

class ProbeModel(nn.Module):
    """Shared encoder, one LINEAR head per task, learned per-task loss weights."""

    def __init__(self, node_dim, struct_dim, edge_dim, hidden=128, layers=4,
                 attention=False, n_heads=4, max_spd=8):
        super().__init__()
        self.encoder = MPNNEncoder(node_dim + struct_dim, edge_dim, hidden=hidden,
                                   layers=layers, attention=attention, n_heads=n_heads,
                                   max_spd=max_spd)
        #: LINEAR, deliberately -- see the module docstring. A hidden layer here would let the
        #: head compute what the representation does not contain.
        self.heads = nn.ModuleDict({p.name: nn.Linear(hidden, p.dim) for p in PROBES})
        #: Kept as a DIAGNOSTIC readout and for `--loss-weighting uncertainty`, but NOT used
        #: by default -- see `loss`.
        self.log_sigma = nn.ParameterDict(
            {p.name: nn.Parameter(torch.zeros(())) for p in PROBES})

    def forward(self, b):
        x = torch.cat([b['x'], b['struct']], dim=-1)
        h, _ = self.encoder(x, b['edge_index'], b['edge_attr'], b['batch'],
                            b['n_graphs'], spd=b.get('spd'))
        return {p.name: self.heads[p.name](h) for p in PROBES}

    #: 'none' = equal-weight MSE on standardised targets. THE DEFAULT, and the fix for a
    #: training collapse that contaminated every run before 2026-09-01.
    weighting = 'none'

    def loss(self, pred, b, mu, sd):
        """Equal-weight MSE on standardised targets.

        WHY NOT KENDALL UNCERTAINTY WEIGHTING, WHICH THIS REPLACED. Its objective is
        ``mse * exp(-2 s) / 2 + s``, minimised at ``s = 0.5 ln(mse)``, so the effective weight
        on a task is ``1 / mse``. Fitting better RAISES the weight, which raises the gradient,
        which is positive feedback with no bound. Measured on the chirality tripwire, single
        task, identical seed and data: sign accuracy climbs to 75.4% by step 500, then the run
        COLLAPSES to a constant predictor by step 1500 and never recovers -- final exact-match
        0.0%. Plain MSE on the same setup reaches 100.0% and stays there, at lr 1e-3 and 3e-4,
        with or without warmup.

        It was solving a problem that no longer existed: the targets are already standardised,
        so every task starts at MSE ~ 1 and the scales are balanced before any weighting is
        applied. It bought nothing and cost stability, and it was active in every run of this
        battery before 2026-09-01 -- the seed-to-seed spread, the one-bad-seed counting
        failures and the loss curves whose final value was worse than their minimum are all
        consistent with the same collapse.
        """
        total = 0.0
        parts = {}
        for p in PROBES:
            y = (b['y_' + p.name] - mu[p.name]) / sd[p.name]
            if p.nonzero_only:
                # MASK THE LOSS THE SAME WAY THE SCORE IS MASKED. cip_code is scored on
                # stereocentres (10.7% of atoms) but was OPTIMISED over all of them, where
                # the other 89.3% want 0. Predicting ~0 everywhere is near-optimal for that
                # loss and scores EXACTLY 0% on the metric, because 0 never rounds to +/-1.
                # The tell was log_sigma sitting at its init value on every seed: the loss
                # never left the predict-the-mean level. Copying the parity input alone
                # scores 59.4%, so 0% was a bug, not difficulty.
                keep = (b['y_' + p.name] != 0).any(dim=-1)
                pr, y = pred[p.name][keep], y[keep]
                mse = ((pr - y) ** 2).mean() if y.numel() else pred[p.name].sum() * 0.0
            else:
                mse = ((pred[p.name] - y) ** 2).mean()
            if self.weighting == 'uncertainty':
                ls = self.log_sigma[p.name]
                total = total + mse * torch.exp(-2 * ls) * 0.5 + ls
            else:
                total = total + mse
            parts[p.name] = float(mse.detach())
        return total, parts


# ----------------------------------------------------------------------------- scoring

def target_stats(samples, device):
    mu, sd = {}, {}
    for p in PROBES:
        v = np.concatenate([s.labels[p.name] for s in samples])
        m, s_ = v.mean(0), v.std(0)
        mu[p.name] = torch.as_tensor(m, dtype=torch.float32, device=device)
        sd[p.name] = torch.as_tensor(np.where(s_ < 1e-8, 1.0, s_), dtype=torch.float32,
                                     device=device)
    return mu, sd


@torch.no_grad()
def score(model, batches, mu, sd):
    """Exact-match accuracy (the 100% target) plus R^2 as the fallback signal.

    A prediction counts as EXACT when every component rounds to the right integer. That is
    the metric the reframe is built on: these targets are deterministic, so exact is
    achievable, and anything under 100% is the model's fault rather than the label's.
    """
    acc = {p.name: [0, 0] for p in PROBES}
    near = {}
    sse = {p.name: 0.0 for p in PROBES}
    ys = {p.name: [] for p in PROBES}
    preds = {p.name: [] for p in PROBES}
    for b in batches:
        out = model(b)
        for p in PROBES:
            y = b['y_' + p.name]
            q = out[p.name] * sd[p.name] + mu[p.name]
            if p.integral:
                hit = (q.round() == y).all(dim=-1)
                # NEAR-MISS at twice the tolerance. On six of nine probes the median miss is
                # 0.58-0.60 -- just past the 0.5 rounding boundary -- so the gap between
                # exact and near IS the story, and quoting only exact makes a well-fitted
                # model look several points short.
                nr = ((q - y).abs() <= 1.0).all(dim=-1)
            else:
                rel = (q - y).abs() / y.abs().clamp(min=1e-6)
                hit = (rel < 0.01).all(dim=-1)
                nr = (rel < 0.02).all(dim=-1)
            if p.nonzero_only:
                keep = (y != 0).any(dim=-1)
                hit, nr = hit[keep], nr[keep]
            acc[p.name][0] += int(hit.sum()); acc[p.name][1] += int(hit.numel())
            near[p.name] = near.get(p.name, 0) + int(nr.sum())
            ys[p.name].append(y.cpu().numpy()); preds[p.name].append(q.cpu().numpy())
    out = {}
    for p in PROBES:
        y = np.concatenate(ys[p.name]); q = np.concatenate(preds[p.name])
        sst = ((y - y.mean(0)) ** 2).sum()
        r2 = float(1 - ((y - q) ** 2).sum() / sst) if sst > 0 else float('nan')
        out[p.name] = {'exact': acc[p.name][0] / max(acc[p.name][1], 1),
                       'near': near.get(p.name, 0) / max(acc[p.name][1], 1), 'r2': r2}
    return out


# ------------------------------------------------------------------------------- arms

ARMS = {
    'mp':          dict(encoding='none', attention=False, spd=False),
    'mp+rwse':     dict(encoding='rwse', attention=False, spd=False),
    #: routing WITHOUT being handed the distances -- the arm the previous battery lacked, and
    #: the one that separates "can route" from "was given the answer".
    'mp+attn':     dict(encoding='rwse', attention=True,  spd=False),
    'mp+attn+spd': dict(encoding='rwse', attention=True,  spd=True),
}


def arm_params(arm, hidden, layers, k, node_dim, edge_dim) -> int:
    m = ProbeModel(node_dim, k, edge_dim, hidden=hidden, layers=layers,
                   attention=ARMS[arm]['attention'])
    return int(sum(q.numel() for q in m.parameters()))


def fit_hidden(arm, target, layers, k, node_dim, edge_dim, lo=16, hi=512) -> int:
    """Width whose parameter count is closest to ``target``.

    RESTORED. `models/encoder_ssl.py` had this; the rewrite dropped it, and the resulting run
    compared 605,487-parameter broadcast arms against 803,275-parameter attention arms --
    **+32.7%** -- while the battery spec's own completion criterion demands a matched budget.
    An unmatched win is a capacity result wearing an architecture result's clothes.
    """
    best, gap = lo, None
    for h in range(lo, hi + 1, 4):
        d = abs(arm_params(arm, h, layers, k, node_dim, edge_dim) - target)
        if gap is None or d < gap:
            best, gap = h, d
    return best


def size_only_baseline(train, test):
    """What a predictor seeing ONLY the atom count achieves, per probe.

    RESTORED, and it is the reference every size-correlated probe must beat -- diameter,
    eccentricity, cycle_rank and the counting probes all scale with N, so an encoder that
    learned nothing but molecule size still posts a respectable number against zero.

    TAKES THE BEST OF TWO TRIVIAL PREDICTORS, not just the regression line. A least-squares
    line in atom count, rounded, is NOT an upper bound on "knows nothing but size" -- on
    `orbit_size` it lands on {1,2} and scores 35.3% where simply predicting the TRAINING
    MAJORITY VALUE everywhere scores 61.1%. Reporting 35.3 claimed a floor 25.8 points BELOW
    the real one, which is worse than reporting no baseline at all: it makes a probe look
    discriminating when a constant beats it. Affects every probe, so the fix is generic.
    """
    out = {}
    for pr in PROBES:
        xs = np.concatenate([np.full(x.n, x.n) for x in train]).astype(float)
        ys = np.concatenate([x.labels[pr.name] if pr.per_node
                             else np.repeat(x.labels[pr.name], x.n, axis=0) for x in train])
        a = np.stack([xs, np.ones_like(xs)], axis=1)
        coef, *_ = np.linalg.lstsq(a, ys, rcond=None)
        xt = np.concatenate([np.full(x.n, x.n) for x in test]).astype(float)
        yt = np.concatenate([x.labels[pr.name] if pr.per_node
                             else np.repeat(x.labels[pr.name], x.n, axis=0) for x in test])
        q = np.stack([xt, np.ones_like(xt)], axis=1) @ coef
        if pr.integral:
            hit = (np.round(q) == yt).all(axis=-1)
        else:
            hit = (np.abs(q - yt) / np.maximum(np.abs(yt), 1e-6) < 0.01).all(axis=-1)
        # the OTHER trivial predictor: the single most common training value, everywhere.
        vals, counts = np.unique(ys, axis=0, return_counts=True)
        const = np.repeat(vals[counts.argmax()][None, :], len(yt), axis=0)
        if pr.integral:
            hit_c = (np.round(const) == yt).all(axis=-1)
        else:
            hit_c = (np.abs(const - yt) / np.maximum(np.abs(yt), 1e-6) < 0.01).all(axis=-1)
        if pr.nonzero_only:
            keep = (yt != 0).any(axis=-1)
            hit, hit_c = hit[keep], hit_c[keep]
        out[pr.name] = (max(float(hit.mean()), float(hit_c.mean()))
                        if hit.size else float('nan'))
    return out


def determinism_gate(smiles, encoding, k, n_repeats=5) -> bool:
    """Is the structural encoding a function of the GRAPH, or of the eigensolver?

    RESTORED. A hard gate reported ahead of any accuracy number: `{f_j}` is cached per
    molecule, so a non-deterministic encoding makes the cached condition depend on the solver.
    RWSE passes by construction (a matrix power); naive LapPE fails on any molecule whose
    Laplacian is degenerate, which symmetry produces constantly.
    """
    ref = build_sample(smiles, encoding, k).struct
    for _ in range(n_repeats - 1):
        if not np.allclose(ref, build_sample(smiles, encoding, k).struct, atol=1e-10):
            return False
    return True


def probe_tolerances(samples):
    """Each probe's tolerance as a FRACTION OF ONE SD, and how many components must pass.

    Printed with the results because exact-match is only comparable ACROSS probes when this
    number is comparable, and measured on QM9 it spans 7x (0.026 for wiener against 0.180 for
    n_atoms). Difficulty orderings drawn from raw exact-match percentages are largely an
    artefact of this column.
    """
    out = {}
    for pr in PROBES:
        v = np.concatenate([x.labels[pr.name] for x in samples])
        if pr.nonzero_only:
            v = v[(v != 0).any(axis=-1)]
        sd = v.std(0)
        tol = np.full_like(sd, 0.5) if pr.integral else 0.01 * np.abs(v).mean(0)
        # DEGENERATE components (F, S, Cl counts are near-constant on QM9) carry SD ~ 0 and
        # are trivially predictable; averaging over them reported 2e8 for . Report
        # the MINIMUM over live components -- the binding constraint -- and how many
        # components are live, since the metric is a conjunction over all of them.
        live = sd > 1e-6
        n_live = int(live.sum())
        ratio = float((tol[live] / sd[live]).min()) if n_live else float("inf")
        out[pr.name] = (ratio, n_live)
    return out


def run(arm, train, test, steps, hidden, layers, k, batch_mols, lr, seed, device,
        save_to: Optional[str] = None, weighting: str = 'none'):
    cfg = ARMS[arm]
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = target_stats(train, device)
    model = ProbeModel(train[0].x.shape[1], k, train[0].edge_attr.shape[1], hidden=hidden,
                       layers=layers, attention=cfg['attention']).to(device)
    model.weighting = weighting
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    model.train()
    curve = []
    best_ho, best_step, best_state = float('inf'), -1, None
    # HELD-OUT LOSS ALONG THE RUN, not just at the end. A train-only curve shows collapse and
    # under-training but is BLIND TO OVER-FITTING -- the one failure mode that looks perfectly
    # healthy from the training side, because train loss keeps falling while the model is
    # memorising. ~60 extra forward passes over a 6,000-step run, on a fixed subset so the
    # points are comparable to each other.
    _eval_pool = test[:min(len(test), 512)]
    _eval_batches = [collate(_eval_pool[i:i + batch_mols], device, cfg['spd'])
                     for i in range(0, len(_eval_pool), batch_mols)]

    def _heldout():
        model.eval()
        tot, acc, nb = 0.0, {}, 0
        with torch.no_grad():
            for eb in _eval_batches:
                t, pr = model.loss(model(eb), eb, mu, sd)
                tot += float(t)
                nb += 1
                for kk, vv in pr.items():
                    acc[kk] = acc.get(kk, 0.0) + vv
        model.train()
        n = max(nb, 1)
        return tot / n, {kk: vv / n for kk, vv in acc.items()}

    for step in range(steps):
        idx = rng.choice(len(train), size=min(batch_mols, len(train)), replace=False)
        b = collate([train[i] for i in idx], device, cfg['spd'])
        opt.zero_grad()
        total, parts = model.loss(model(b), b, mu, sd)
        total.backward()
        gn = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        opt.step()
        # LOSS CURVE, recorded because a plateau and a divergence look identical in a final
        # score. Seeds differing by 12x in residual with no crash could be either. Read it
        # with models/curve_report.py -- a summary statistic cannot tell you WHICH failure
        # you have, and this project has twice shipped a conclusion built on one that could.
        if step % max(1, steps // 60) == 0 or step == steps - 1:
            ho_total, ho_parts = _heldout()
            curve.append({'step': step, 'total': float(total.detach()),
                          'grad_norm': gn, 'parts': parts,
                          'test_total': ho_total, 'test_parts': ho_parts})
            # KEEP THE BEST HELD-OUT STATE, NOT THE LAST. Saving the last state means a
            # collapse or an over-fit at any point after the minimum is what gets written to
            # disk and reused downstream -- and both are silent in a final score. Selecting on
            # HELD-OUT rather than train also makes this an early stop, not just a safety net.
            if ho_total < best_ho:
                best_ho = ho_total
                best_step = step
                best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}

    # restore the best held-out state before SCORING, so the reported numbers and the saved
    # weights describe the same model. Scoring the last state and saving the best would be a
    # quiet lie about which model produced the table.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    mk = lambda pool: [collate(pool[i:i + batch_mols], device, cfg['spd'])
                       for i in range(0, len(pool), batch_mols)]
    # the FULL training set. Scoring a 1,000-molecule prefix of a 4,000-molecule train set
    # was the basis of an "expressiveness bound" claim, which a subset cannot support.
    fit = score(model, mk(train), mu, sd)
    gen = score(model, mk(test), mu, sd)
    if save_to:
        # nothing was ever saved before, so no result could be re-examined without retraining
        # -- which blocks the collision search entirely.
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        torch.save({'state_dict': model.state_dict(), 'arm': arm, 'seed': seed,
                    'n_train': len(train), 'hidden': hidden, 'layers': layers, 'k': k,
                    'attention': cfg['attention'], 'best_step': best_step,
                    'best_heldout': best_ho, 'steps_run': steps}, save_to)
    return {'arm': arm, 'seed': seed, 'n_train': len(train), 'steps': steps, 'curve': curve,
            'best_step': best_step, 'best_heldout': best_ho,
            'n_params': int(sum(p.numel() for p in model.parameters())),
            'train': fit, 'test': gen,
            'log_sigma': {p.name: float(model.log_sigma[p.name].detach()) for p in PROBES}}


def report(agg, sizes, arms, baseline, tols, meta) -> str:
    w = max(len(pr.name) for pr in PROBES) + 2
    L = ['ENCODER PROBE BATTERY -- exact-match accuracy (%), train / test. TARGET IS 100.',
         'Targets are deterministic functions of the graph: no label noise, no irreducible',
         'error, so anything under 100 is the model rather than the data.',
         '  train < 100              -> EXPRESSIVENESS bound, IF isolated (see caveat below)',
         '  train = 100, test < 100  -> sample complexity',
         '  every arm at 100 on little data -> the probe is LEAKED',
         '',
         'CAVEAT ON "expressiveness": heads here are LINEAR and the tasks are trained',
         'JOINTLY, so a plateau confounds representability with',
         'linear accessibility, task interference and optimisation. A plateau is a verdict',
         'only after the task is overfit IN ISOLATION with no task reweighting.',
         '',
         f"molecules {meta['n_train_pool']} pool / {meta['n_test']} held out   "
         f"steps {meta['steps']}   seeds {meta['seeds']}   "
         f"budgets {'MATCHED' if meta['matched'] else 'UNMATCHED'}   "
         f"stereo {'ON' if meta['stereo'] else 'OFF'}",
         '',
         'MODELS',
         f"    {'arm':<14}{'hidden':>8}{'params':>10}{'determinism':>13}"]
    for a in arms:
        m = agg[a]['meta']
        L.append(f"    {a:<14}{m['hidden']:>8}{m['n_params']:>10,}"
                 f"{('PASS' if m['determinism'] else 'FAIL'):>13}")
    L += ['', 'PROBE CALIBRATION (tolerance as a fraction of one SD; exact-match is only',
          'comparable ACROSS probes where this is comparable)',
          f"    {'probe':<{w}}{'tol/SD':>9}{'components':>12}{'size-only':>11}"]
    for pr in PROBES:
        t, d = tols[pr.name]
        L.append(f"    {pr.name:<{w}}{t:>9.3f}{d:>12}{100 * baseline[pr.name]:>10.1f}%")
    for blk in BLOCKS:
        rows = [pr for pr in PROBES if pr.block == blk]
        if not rows:
            continue
        L += ['', f'-- block {blk}: {BLOCK_NAMES[blk]}',
              f"{'probe':<{w}}{'n':>7}" + ''.join(f'{a:>26}' for a in arms),
              f"{'':<{w}}{'':>7}" + ''.join(f"{'train/ test(near)':>26}" for _ in arms)]
        for pr in rows:
            for i, n in enumerate(sizes):
                row = f"{pr.name if i == 0 else '':<{w}}{n:>7}"
                for a in arms:
                    c = agg[a]['cells'].get((n, pr.name))
                    row += ('-'.rjust(26) if c is None else
                            f"{100 * c[0]:>8.1f}/{100 * c[1]:>6.1f}"
                            f"({100 * c[3]:>5.1f})")
                L.append(row)
    return chr(10).join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    ap.add_argument('--sizes', type=int, nargs='+', default=[250, 1000, 4000])
    ap.add_argument('--n-test', type=int, default=1000)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch-mols', type=int, default=128)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--k', type=int, default=16)
    #: 3e-4, from the full-horizon bracket in models/lr_sweep.py. The detonation boundary is
    #: between 3e-4 and 1e-3, and 1e-3 -- the previous default -- is ABOVE it: best loss 8.3
    #: and a 56% regression from its own minimum, against 0.005 and 0.0% at 3e-4. Every battery
    #: run before 2026-09-01 used the collapsing rate.
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--stereo', action='store_true', default=True,
                    help='stereo-enumerated QM9. ON by default: raw QM9 has 0.0%% chiral '
                         'tags, so the parity channel is dead and chirality untestable')
    ap.add_argument('--no-stereo', dest='stereo', action='store_false')
    ap.add_argument('--match-params', action='store_true', default=True,
                    help='size each arm to a common parameter budget')
    ap.add_argument('--no-match-params', dest='match_params', action='store_false')
    ap.add_argument('--ladder-only', action='store_true')
    ap.add_argument('--save-dir', default=os.path.join(RESULTS, 'encoder_ckpt'))
    ap.add_argument('--probes', nargs='+', default=None,
                    help='restrict to these probes. The 2026-08-31 audit found only '
                         'eccentricity, spd_to_marked and cip_code have outcomes that are '
                         'not predictable a priori; the rest either score 100 for every arm '
                         'or have per-seed ranges that overlap between arms.')
    ap.add_argument('--loss-weighting', default='none',
                    choices=['none', 'uncertainty'],
                    help="'none' (default) = equal-weight MSE on standardised "
                         'targets. uncertainty weighting is retained only to '
                         'reproduce pre-2026-09-01 runs: its weight is 1/mse, which '
                         'is positive feedback, and it collapses training.')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=os.path.join(RESULTS, 'encoder_probe.json'))
    a = ap.parse_args(argv)
    if a.smoke:
        a.sizes, a.n_test, a.steps = [60], 60, 40
        a.hidden, a.layers, a.k, a.seeds = 32, 2, 6, [0]

    if a.probes:
        global PROBES
        unknown = set(a.probes) - {p.name for p in PROBES}
        if unknown:
            raise SystemExit(f'unknown probes {sorted(unknown)}; '
                             f'available: {[p.name for p in PROBES]}')
        PROBES = [p for p in PROBES if p.name in set(a.probes)]
        print(f'probes restricted to: {[p.name for p in PROBES]}')

    need = int((max(a.sizes) + a.n_test) * 1.1) + 64
    if a.ladder_only:
        smis = size_ladder()
    elif a.stereo:
        smis = load_qm9_stereo(need)
    else:
        smis = load_qm9(need)
    print(f'{len(smis)} molecules; building samples per encoding ...')

    cache, group_ids = {}, {}
    for enc in sorted({ARMS[arm]['encoding'] for arm in a.arms}):
        built, skels = [], []
        for smi in smis:
            try:
                sample = build_sample(smi, enc, a.k)
            except Exception:
                continue
            built.append(sample)
            skels.append(parent_skeleton(smi))
        # GROUP-CONTIGUOUS ORDER: shuffle the SKELETON GROUPS, then emit each group's members
        # together, so a prefix split can never put two stereoisomers of one parent on
        # opposite sides. Shuffling rows instead is what leaked 67.9% of the held-out set.
        groups = {}
        for i, k_ in enumerate(skels):
            groups.setdefault(k_, []).append(i)
        keys = sorted(groups)
        key_id = {k_: j for j, k_ in enumerate(keys)}
        perm = np.random.default_rng(12345).permutation(len(keys))
        order = [i for j in perm for i in groups[keys[j]]]
        cache[enc] = [built[i] for i in order]
        group_ids[enc] = [key_id[skels[i]] for i in order]
        n_chiral = sum(1 for b in cache[enc] if (b.labels['cip_code'] != 0).any())
        print(f'  {enc}: {len(built)} usable, {n_chiral} with a stereocentre')

    ref = cache[ARMS[a.arms[0]]['encoding']]
    # SNAP THE HELD-OUT BOUNDARY TO A GROUP EDGE. Groups are contiguous, so advancing to the
    # next group start guarantees no skeleton straddles the split. Costs a few molecules.
    gid = group_ids[ARMS[a.arms[0]]['encoding']]
    n_te = min(a.n_test, len(ref))
    while n_te < len(gid) and gid[n_te] == gid[n_te - 1]:
        n_te += 1
    if n_te != a.n_test:
        print(f'  held-out snapped to a skeleton-group edge: {a.n_test} -> {n_te}')
    a.n_test = n_te
    leak = len(set(gid[:n_te]) & set(gid[n_te:]))
    print(f'  SPLIT CHECK: {len(set(gid))} skeleton groups; '
          f'{leak} shared between train and held out (must be 0)')
    assert leak == 0, 'skeleton leaked across the split'
    node_dim, edge_dim = ref[0].x.shape[1], ref[0].edge_attr.shape[1]
    hidden = {arm: a.hidden for arm in a.arms}
    if a.match_params:
        tgt = max(arm_params(arm, a.hidden, a.layers, a.k, node_dim, edge_dim)
                  for arm in a.arms)
        hidden = {arm: fit_hidden(arm, tgt, a.layers, a.k, node_dim, edge_dim)
                  for arm in a.arms}
        print(f'matching parameter budgets to {tgt:,}:')
        for arm in a.arms:
            g = arm_params(arm, hidden[arm], a.layers, a.k, node_dim, edge_dim)
            print(f'    {arm:<14} hidden {hidden[arm]:>4}  params {g:>9,} '
                  f'({100.0 * g / tgt - 100:+.1f}%)')

    baseline = size_only_baseline(ref[a.n_test:], ref[:a.n_test])
    tols = probe_tolerances(ref)

    rows, agg = [], {}
    for arm in a.arms:
        pool = cache[ARMS[arm]['encoding']]
        test, rest = pool[:a.n_test], pool[a.n_test:]
        cells, params = {}, None
        for n in a.sizes:
            if n > len(rest):
                print(f'  skip {arm} n={n}: only {len(rest)} available')
                continue
            per_seed = []
            for seed in a.seeds:
                ck = os.path.join(a.save_dir, f'{arm}_n{n}_s{seed}.pt')
                r = run(arm, rest[:n], test, a.steps, hidden[arm], a.layers, a.k,
                        a.batch_mols, a.lr, seed, a.device, save_to=ck,
                        weighting=a.loss_weighting)
                rows.append(r)
                per_seed.append(r)
                params = r['n_params']
                shown = ' '.join(
                    f"{pr.name[:4]} {100*r['train'][pr.name]['exact']:.0f}"
                    f"/{100*r['test'][pr.name]['exact']:.0f}" for pr in PROBES[:3])
                print(f'  {arm} n={n} s={seed}: {shown}')
            for pr in PROBES:
                tr = np.array([x['train'][pr.name]['exact'] for x in per_seed])
                te = np.array([x['test'][pr.name]['exact'] for x in per_seed])
                nr = np.array([x['test'][pr.name]['near'] for x in per_seed])
                cells[(n, pr.name)] = (tr.mean(), te.mean(),
                                       (te.max() - te.min()) / 2, nr.mean())
        agg[arm] = {'cells': cells,
                    'meta': {'hidden': hidden[arm], 'n_params': params,
                             'determinism': determinism_gate(smis[0],
                                                             ARMS[arm]['encoding'], a.k)}}

    meta = {'steps': a.steps, 'seeds': len(a.seeds), 'matched': a.match_params,
            'stereo': a.stereo, 'n_train_pool': len(ref) - a.n_test, 'n_test': a.n_test}
    text = report(agg, a.sizes, a.arms, baseline, tols, meta)
    print(chr(10) + text)
    os.makedirs(RESULTS, exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump({'rows': rows, 'sizes': a.sizes, 'arms': a.arms, 'meta': meta,
                   'size_only': baseline,
                   'tolerances': {k: v[0] for k, v in tols.items()}}, f, indent=2)
    print(f'wrote {a.out}')
    return rows


if __name__ == '__main__':
    main()
