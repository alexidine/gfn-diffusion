"""The encoder self-supervision battery -- runnable.

Spec and argument: docs/design/encoder_ssl_battery.md. This file is the executable half; read
that first, because the probe ORDERING is the whole design and is not recoverable from the
code.

WHAT IT DECIDES. Four arms crossed against probes sorted by whether an
aggregate-and-broadcast encoder can reach them:

    block A  broadcast-reachable  -- sums over atoms. FLOOR CHECKS, never evidence for an
                                    architecture.
    block B  the 1-WL breaker     -- cycles. Separates "has a structural encoding" from
                                    "has none".
    block C  distance             -- depends on a SPECIFIC far atom, not on any sum. THIS is
                                    where attention earns its place or does not.
    block D  symmetry             -- orbit structure; ceiling-bearing.

EVERY GRAPH-LEVEL OBSERVABLE IS PREDICTED PER NODE, not from the pooled vector. Pooling would
let atoms specialise and cover for each other; per-node prediction forces every atom to carry
the whole graph, which is the actual requirement on ``g_i``.

THE BASELINE IS NOT ZERO. Diameter, Wiener index, ring count and eccentricity all correlate
hard with atom count, so a model that learned only "how big am I" scores well on all of them.
Every regression cell is therefore reported against a SIZE-ONLY baseline fitted on atom count
alone. An arm that fails to beat size-only has demonstrated nothing, and the table says so.

    python -m models.encoder_ssl --arms mp mp+rwse mp+rwse+attn --steps 3000
    python -m models.encoder_ssl --smoke          # 60 s sanity pass, not a result
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

from models.graph_encoder import MPNNEncoder, dense_spd_batch, to_dense_batch
from models.graph_encodings import (
    bond_features_from_smiles, cycle_rank, degree_histogram, diameter, eccentricity,
    graph_from_smiles, lap_pe, orbit_sizes, ring_membership, rwse, shortest_paths,
    smallest_ring_size, spectral_moments, wiener_index,
)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

#: Fixed element set. An unseen element RAISES rather than folding into a catch-all bin: a
#: silent "other" column is how a molecule set quietly stops being what the table claims.
ELEMENTS = (1, 6, 7, 8, 9, 16, 17)
MAX_DEGREE = 5

#: The committed molecule set. Deliberately mixed: acyclic chains that exercise radius,
#: rings and fused rings for the cycle probes, symmetric molecules that give the orbit
#: probes something to find, and peptides because they are the real target. ETHANOL IS THE
#: NEGATIVE CONTROL and must never be dropped -- no arm should beat any other on it, and an
#: arm that does is measuring noise.
MOLECULES = (
    'CCO',                                   # ethanol -- NEGATIVE CONTROL
    'CCCC', 'CCCCCC', 'CCCCCCCC', 'CCCCCCCCCC',
    'OCCCCO', 'OCCCCCCCCCCO',                # long, floppy: radius probes
    'CC(C)C', 'CC(C)(C)C', 'CC(C)CC(C)C',    # symmetric substituents: orbit probes
    'c1ccccc1', 'Cc1ccccc1', 'c1ccc2ccccc2c1',
    'C1CCCCC1', 'C1CCCC1', 'C1CCCCCC1',
    'C1CCC(CO1)c1ccccc1',                    # phenyl-THP: a stack molecule
    'CC(=O)NC', 'NCC(=O)NCC(=O)O',
    'NCC(=O)NCC(=O)NCC(=O)O',
    'NCC(=O)NCC(=O)NCC(=O)NCC(=O)O',         # Gly4: the molecule the stack trains on
    'C[C@H](N)C(=O)O', 'C[C@@H](N)C(=O)O',   # an enantiomer pair
    'OCC(O)CO', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
)
#: Held out, and ETHANOL IS AMONG THEM DELIBERATELY. It was previously in the training set,
#: which meant the negative control the spec requires was never actually evaluated -- an arm
#: could have been beating the others on it with nobody looking.
HOLDOUT = ('NCC(=O)NCC(=O)NCC(=O)NCC(=O)O', 'c1ccc2ccccc2c1', 'CC(C)CC(C)C',
           'OCCCCCCCCCCO', 'CCO')
CONTROL = 'CCO'


# ------------------------------------------------------------------------------ probes

@dataclass(frozen=True)
class Probe:
    name: str
    block: str            # 'A' floor | 'B' 1-WL | 'C' distance | 'D' symmetry
    dim: int
    fn: Callable          # (z, edge_index, n, spd) -> np.ndarray [n, dim] or [dim]
    per_node: bool        # False -> a graph observable, broadcast to every atom
    binary: bool = False


def _formula(z, e, n, spd):
    return np.array([float((z == el).sum()) for el in ELEMENTS])


PROBES: List[Probe] = [
    # ---- block A: broadcast-reachable. Floor checks.
    Probe('n_atoms', 'A', 1, lambda z, e, n, s: np.array([float(n)]), False),
    Probe('formula', 'A', len(ELEMENTS), _formula, False),
    Probe('degree_hist', 'A', MAX_DEGREE + 1,
          lambda z, e, n, s: degree_histogram(e, n, MAX_DEGREE).astype(float), False),
    Probe('cycle_rank', 'A', 1,
          lambda z, e, n, s: np.array([float(cycle_rank(e, n))]), False),
    Probe('spectral_moments', 'A', 4,
          lambda z, e, n, s: spectral_moments(e, n, ks=(2, 3, 4, 5)), False),
    # ---- block B: the 1-WL breaker.
    Probe('ring_member', 'B', 1,
          lambda z, e, n, s: ring_membership(e, n).astype(float)[:, None], True,
          binary=True),
    Probe('smallest_ring', 'B', 1,
          lambda z, e, n, s: smallest_ring_size(e, n).astype(float)[:, None], True),
    # ---- block C: distance. The discriminator.
    Probe('eccentricity', 'C', 1,
          lambda z, e, n, s: eccentricity(s).astype(float)[:, None], True),
    Probe('spd_to_root', 'C', 1,
          lambda z, e, n, s: np.where(s[0] < 0, -1.0, s[0]).astype(float)[:, None], True),
    Probe('diameter', 'C', 1, lambda z, e, n, s: np.array([float(diameter(s))]), False),
    Probe('wiener', 'C', 1, lambda z, e, n, s: np.array([float(wiener_index(s))]), False),
    # ---- block D: symmetry.
    Probe('orbit_size', 'D', 1,
          lambda z, e, n, s: orbit_sizes(e, n, labels=z.tolist()).astype(float)[:, None],
          True),
]
BLOCK_NAMES = {'A': 'floor (broadcast-reachable)', 'B': '1-WL breaker (cycles)',
               'C': 'distance (routing)', 'D': 'symmetry'}


# ----------------------------------------------------------------------------- samples

@dataclass
class Sample:
    smiles: str
    n: int
    x: np.ndarray                 # [n, F] atom features WITHOUT the structural encoding
    struct: np.ndarray            # [n, K] structural encoding (zeros for the `mp` arm)
    edge_index: np.ndarray        # [2, 2E] both directions
    edge_attr: np.ndarray         # [2E, 5]
    spd: np.ndarray               # [n, n]
    labels: Dict[str, np.ndarray] = field(default_factory=dict)


def atom_features(z: np.ndarray, edge_index: np.ndarray, parity: np.ndarray,
                  n: int) -> np.ndarray:
    """One-hot element, one-hot degree, and the parity pseudoscalar.

    Parity is here and nowhere else: every structural encoding below is a function of the
    adjacency, which is identical for enantiomers, so if parity does not enter as an atom
    feature the encoder is enantiomer-blind no matter what else it is given.
    """
    unknown = set(np.unique(z).tolist()) - set(ELEMENTS)
    if unknown:
        raise ValueError(f'elements {sorted(unknown)} are outside ELEMENTS; add them '
                         f'deliberately rather than letting a catch-all bin hide them')
    el = np.zeros((n, len(ELEMENTS)))
    for j, e in enumerate(ELEMENTS):
        el[:, j] = (z == e)
    from models.graph_encodings import to_dense_adjacency
    deg = np.minimum(to_dense_adjacency(edge_index, n).sum(1).astype(int), MAX_DEGREE)
    dg = np.zeros((n, MAX_DEGREE + 1))
    dg[np.arange(n), deg] = 1.0
    return np.concatenate([el, dg, parity.astype(float)[:, None]], axis=1)


def build_sample(smiles: str, encoding: str, k: int) -> Sample:
    z, e_single, parity = graph_from_smiles(smiles)
    n = len(z)
    bf = bond_features_from_smiles(smiles)
    # both directions, edge features duplicated to match
    edge_index = np.concatenate([e_single, e_single[::-1]], axis=1)
    edge_attr = np.concatenate([bf, bf], axis=0)
    spd = shortest_paths(e_single, n)

    if encoding == 'none':
        struct = np.zeros((n, k))
    elif encoding == 'rwse':
        struct = rwse(e_single, n, k=k)
    elif encoding == 'lap':
        struct = lap_pe(e_single, n, k=k)
    else:
        raise ValueError(f'unknown encoding {encoding!r}')

    s = Sample(smiles, n, atom_features(z, e_single, parity, n), struct,
               edge_index, edge_attr, spd)
    for p in PROBES:
        v = np.asarray(p.fn(z, e_single, n, spd), dtype=np.float64)
        s.labels[p.name] = v.reshape(n, p.dim) if p.per_node else v.reshape(1, p.dim)
    return s


def collate(samples: Sequence[Sample], device):
    """Offset the edge indices, stack the nodes, and densify the distance matrices."""
    n_graphs = len(samples)
    offs, xs, sts, eis, eas, batch = 0, [], [], [], [], []
    for gi, s in enumerate(samples):
        xs.append(s.x)
        sts.append(s.struct)
        eis.append(s.edge_index + offs)
        eas.append(s.edge_attr)
        batch.append(np.full(s.n, gi))
        offs += s.n
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=device)
    out = {
        'x': t(np.concatenate(xs)),
        'struct': t(np.concatenate(sts)),
        'edge_index': t(np.concatenate(eis, axis=1), torch.long),
        'edge_attr': t(np.concatenate(eas)),
        'batch': t(np.concatenate(batch), torch.long),
        'n_graphs': n_graphs,
        'length': max(s.n for s in samples),
        'spd_list': [s.spd for s in samples],
        'n_nodes': torch.as_tensor([s.n for s in samples], device=device),
    }
    for p in PROBES:
        lab = np.concatenate([s.labels[p.name] for s in samples]) if p.per_node else \
            np.concatenate([np.repeat(s.labels[p.name], s.n, axis=0) for s in samples])
        out['y_' + p.name] = t(lab)
    return out


# ------------------------------------------------------------------------------- model

class SSLModel(nn.Module):
    """Encoder plus one linear head per probe, reading per-atom ``g_i``."""

    def __init__(self, node_dim, struct_dim, edge_dim, hidden=64, layers=4,
                 attention=False, n_heads=4, max_spd=8):
        super().__init__()
        self.encoder = MPNNEncoder(node_dim + struct_dim, edge_dim, hidden=hidden,
                                   layers=layers, attention=attention, n_heads=n_heads,
                                   max_spd=max_spd)
        self.heads = nn.ModuleDict({
            p.name: nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, p.dim))
            for p in PROBES})

    def forward(self, b, spd=None):
        x = torch.cat([b['x'], b['struct']], dim=-1)
        h, _ = self.encoder(x, b['edge_index'], b['edge_attr'], b['batch'],
                            b['n_graphs'], spd=spd)
        return {p.name: self.heads[p.name](h) for p in PROBES}


# --------------------------------------------------------------------------- statistics

def target_stats(samples: Sequence[Sample]) -> Dict[str, tuple]:
    """Per-probe mean/std over the TRAINING set only, for standardised regression."""
    out = {}
    for p in PROBES:
        v = np.concatenate([s.labels[p.name] for s in samples])
        mu, sd = v.mean(0), v.std(0)
        out[p.name] = (mu, np.where(sd < 1e-8, 1.0, sd))
    return out


def size_only_baseline(train: Sequence[Sample], test: Sequence[Sample]) -> Dict[str, float]:
    """R^2 of the best predictor that sees ONLY the atom count.

    This is the number every regression cell has to beat. Diameter, Wiener index and ring
    count all scale with N, so an encoder that learned nothing but size still posts a
    respectable R^2 against zero -- and would look like a result.
    """
    out = {}
    for p in PROBES:
        if p.binary:
            continue
        xs = np.concatenate([np.full(s.n, s.n) for s in train]).astype(float)
        ys = np.concatenate([s.labels[p.name] if p.per_node
                             else np.repeat(s.labels[p.name], s.n, axis=0) for s in train])
        a = np.stack([xs, np.ones_like(xs)], axis=1)
        coef, *_ = np.linalg.lstsq(a, ys, rcond=None)
        xt = np.concatenate([np.full(s.n, s.n) for s in test]).astype(float)
        yt = np.concatenate([s.labels[p.name] if p.per_node
                             else np.repeat(s.labels[p.name], s.n, axis=0) for s in test])
        pred = np.stack([xt, np.ones_like(xt)], axis=1) @ coef
        out[p.name] = _r2(yt, pred)
    return out


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    sse = ((y - pred) ** 2).sum()
    sst = ((y - y.mean(0)) ** 2).sum()
    return float(1.0 - sse / sst) if sst > 0 else float('nan')


def orbit_ceiling(samples: Sequence[Sample]) -> float:
    """Accuracy ceiling for any node-identification task, set by the orbit structure.

    Leave-one-out node identification can only succeed up to automorphism, so on a symmetric
    molecule the ceiling is below 1.0. Reported so that correct behaviour is not read as
    failure -- see the battery doc, section 4.
    """
    per = [1.0 / s.labels['orbit_size'] for s in samples]
    return float(np.concatenate(per).mean())


# -------------------------------------------------------------------------- determinism

def determinism_gate(smiles: str, encoding: str, k: int, n_repeats: int = 5) -> bool:
    """Same molecule, repeated construction: is the encoding a function of the graph?

    A hard gate, reported ahead of any accuracy number. RWSE passes by construction; naive
    LapPE fails on any molecule whose Laplacian is degenerate, which symmetry produces
    constantly. Because ``{f_j}`` is cached per molecule, a failure here means the cached
    condition is a function of the eigensolver rather than of the molecule.
    """
    ref = build_sample(smiles, encoding, k).struct
    for _ in range(n_repeats - 1):
        if not np.allclose(ref, build_sample(smiles, encoding, k).struct, atol=1e-10):
            return False
    # and under relabelling, which is where the O(m) freedom actually bites
    z, e, parity = graph_from_smiles(smiles)
    n = len(z)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    inv = np.argsort(perm)
    fn = {'none': lambda *a: np.zeros((n, k)), 'rwse': rwse, 'lap': lap_pe}[encoding]
    a = fn(e, n, k) if encoding != 'none' else np.zeros((n, k))
    b = fn(inv[e], n, k) if encoding != 'none' else np.zeros((n, k))
    return bool(np.allclose(a, b[inv], atol=1e-8))


# ------------------------------------------------------------------------------- train

ARMS = {
    'mp':            dict(encoding='none', attention=False),
    'mp+rwse':       dict(encoding='rwse', attention=False),
    'mp+lap':        dict(encoding='lap',  attention=False),
    'mp+rwse+attn':  dict(encoding='rwse', attention=True),
}


def arm_params(arm: str, hidden: int, layers: int, k: int, node_dim: int,
               edge_dim: int) -> int:
    cfg = ARMS[arm]
    m = SSLModel(node_dim, k, edge_dim, hidden=hidden, layers=layers,
                 attention=cfg['attention'])
    return int(sum(p.numel() for p in m.parameters()))


def fit_hidden(arm: str, target: int, layers: int, k: int, node_dim: int, edge_dim: int,
               lo: int = 8, hi: int = 512) -> int:
    """Smallest ``hidden`` (a multiple of 4, so the attention heads still divide) whose
    parameter count is closest to ``target``.

    WHY THIS EXISTS. The attention arm carries qkv, projection and bias weights per layer, so
    at equal width it has ~43% more parameters than the message-passing arms -- and in the
    first functional run it won on block A, which is broadcast-reachable and should not
    separate the arms at all. That is capacity leaking into the comparison. Matching budgets
    is what turns "attention wins" into a claim about ROUTING rather than about size.
    """
    best, best_gap = lo, None
    for h in range(lo, hi + 1, 4):
        gap = abs(arm_params(arm, h, layers, k, node_dim, edge_dim) - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = h, gap
    return best


def run_arm(arm: str, steps: int, k: int = 16, hidden: int = 64, layers: int = 4,
            lr: float = 3e-4, seed: int = 0, device: str = 'cpu',
            molecules: Sequence[str] = MOLECULES,
            holdout: Sequence[str] = HOLDOUT) -> dict:
    cfg = ARMS[arm]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_smi = [m for m in molecules if m not in holdout]
    train = [build_sample(m, cfg['encoding'], k) for m in train_smi]
    test = [build_sample(m, cfg['encoding'], k) for m in holdout]
    stats = target_stats(train)

    model = SSLModel(train[0].x.shape[1], k, train[0].edge_attr.shape[1],
                     hidden=hidden, layers=layers, attention=cfg['attention']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def batch_of(samples):
        b = collate(samples, device)
        spd = (dense_spd_batch(b['spd_list'], b['n_graphs'], b['length'], device)
               if cfg['attention'] else None)
        return b, spd

    tb, tspd = batch_of(train)
    eb, espd = batch_of(test)
    mu = {p.name: torch.as_tensor(stats[p.name][0], dtype=torch.float32, device=device)
          for p in PROBES}
    sd = {p.name: torch.as_tensor(stats[p.name][1], dtype=torch.float32, device=device)
          for p in PROBES}

    bce = nn.BCEWithLogitsLoss()
    model.train()
    for step in range(steps):
        opt.zero_grad()
        pred = model(tb, tspd)
        loss = 0.0
        for p in PROBES:
            y = tb['y_' + p.name]
            if p.binary:
                loss = loss + bce(pred[p.name], y)
            else:
                loss = loss + ((pred[p.name] - (y - mu[p.name]) / sd[p.name]) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(eb, espd)
    scores = {}
    for p in PROBES:
        y = eb['y_' + p.name].cpu().numpy()
        q = pred[p.name].cpu().numpy()
        if p.binary:
            scores[p.name] = float(((q > 0) == (y > 0.5)).mean())
        else:
            scores[p.name] = _r2(y, q * stats[p.name][1] + stats[p.name][0])
    return {
        'arm': arm, 'seed': seed, 'steps': steps, 'encoding': cfg['encoding'],
        'attention': cfg['attention'], 'hidden': hidden, 'layers': layers,
        'n_params': int(sum(p.numel() for p in model.parameters())),
        'train_loss': float(loss.item()),
        'determinism_gate': determinism_gate('c1ccccc1', cfg['encoding'], k),
        'scores': scores,
    }


def report(agg, baseline, ceiling, meta) -> str:
    arms = [a for a in agg]
    w = max(len(pr.name) for pr in PROBES) + 2
    lines = [
        'ENCODER SSL BATTERY',
        f"molecules {meta['n_train']} train / {meta['n_test']} held out   "
        f"steps {meta['steps']}   seeds {meta['seeds']}   k {meta['k']}   "
        f"budgets {'MATCHED' if meta['matched'] else 'UNMATCHED'}",
        '',
        'Held-out R^2 (accuracy for ring_member), mean +/- half-range over seeds. Every',
        'regression cell is read against SIZE-ONLY -- the R^2 of a predictor seeing nothing',
        'but atom count. An arm that does not beat size-only has demonstrated nothing.',
        'Block A is a FLOOR: passing it is not evidence for an architecture, and an arm that',
        'SEPARATES on block A is showing capacity, not routing.',
        f"Orbit ceiling for node identification on the held-out set: {ceiling:.3f}",
        '',
        'MODELS',
        f"    {'arm':<16}{'encoding':>10}{'attn':>6}{'hidden':>8}{'params':>10}"
        f"{'determinism':>13}",
    ]
    for a in arms:
        m = agg[a]['meta']
        lines.append(f"    {a:<16}{m['encoding']:>10}{str(m['attention']):>6}"
                     f"{m['hidden']:>8}{m['n_params']:>10,}"
                     f"{('PASS' if m['determinism'] else 'FAIL'):>13}")
    head = f"{'probe':<{w}}{'size-only':>11}" + ''.join(f'{a:>18}' for a in arms)
    for blk in ('A', 'B', 'C', 'D'):
        lines += ['', f'-- block {blk}: {BLOCK_NAMES[blk]}', head]
        for pr in PROBES:
            if pr.block != blk:
                continue
            base = baseline.get(pr.name)
            row = f'{pr.name:<{w}}' + (f'{base:>11.3f}' if base is not None
                                       else f'{"n/a":>11}')
            for a in arms:
                mu, half = agg[a]['scores'][pr.name]
                row += f'{mu:>12.3f}{half:>6.3f}'
            lines.append(row)
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--match-params', action='store_true',
                    help='size each arm so parameter counts match the largest arm, so a '
                         'win is about routing rather than capacity')
    ap.add_argument('--smoke', action='store_true',
                    help='fast sanity pass. NOT a result -- too few steps to separate arms.')
    ap.add_argument('--out', default=os.path.join(RESULTS, 'encoder_ssl.json'))
    a = ap.parse_args(argv)
    if a.smoke:
        a.steps, a.hidden, a.layers, a.k = 60, 16, 2, 6

    train = [build_sample(m, 'none', a.k) for m in MOLECULES if m not in HOLDOUT]
    test = [build_sample(m, 'none', a.k) for m in HOLDOUT]
    baseline = size_only_baseline(train, test)
    ceiling = orbit_ceiling(test)
    node_dim, edge_dim = train[0].x.shape[1], train[0].edge_attr.shape[1]

    hidden = {arm: a.hidden for arm in a.arms}
    if a.match_params:
        target = max(arm_params(arm, a.hidden, a.layers, a.k, node_dim, edge_dim)
                     for arm in a.arms)
        hidden = {arm: fit_hidden(arm, target, a.layers, a.k, node_dim, edge_dim)
                  for arm in a.arms}
        print(f'matching parameter budgets to {target:,}:')
        for arm in a.arms:
            got = arm_params(arm, hidden[arm], a.layers, a.k, node_dim, edge_dim)
            print(f'    {arm:<16} hidden {hidden[arm]:>4}  params {got:>9,}  '
                  f'({100.0 * got / target - 100:+.1f}%)')

    results, agg = [], {}
    for arm in a.arms:
        per_seed = []
        for seed in a.seeds:
            r = run_arm(arm, a.steps, k=a.k, hidden=hidden[arm], layers=a.layers,
                        seed=seed, device=a.device)
            results.append(r)
            per_seed.append(r)
            print(f"  {arm} seed {seed}: train loss {r['train_loss']:.4f}")
        scores = {}
        for pr in PROBES:
            v = np.array([r['scores'][pr.name] for r in per_seed])
            scores[pr.name] = (float(v.mean()), float((v.max() - v.min()) / 2))
        agg[arm] = {'scores': scores,
                    'meta': {'encoding': per_seed[0]['encoding'],
                             'attention': per_seed[0]['attention'],
                             'hidden': hidden[arm],
                             'n_params': per_seed[0]['n_params'],
                             'determinism': per_seed[0]['determinism_gate']}}

    meta = {'steps': a.steps, 'seeds': len(a.seeds), 'k': a.k,
            'matched': a.match_params, 'n_train': len(train), 'n_test': len(test)}
    text = report(agg, baseline, ceiling, meta)
    print('\n' + text)
    if a.smoke:
        print('\nSMOKE RUN -- these numbers are not a result.')
    os.makedirs(RESULTS, exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump({'results': results, 'aggregate': agg, 'size_only': baseline,
                   'orbit_ceiling': ceiling, 'meta': meta,
                   'molecules': list(MOLECULES), 'holdout': list(HOLDOUT)}, f, indent=2)
    print(f'wrote {a.out}')
    return agg


if __name__ == '__main__':
    main()
