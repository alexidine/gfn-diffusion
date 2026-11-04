"""Space groups as operation sets (models/sym_ops.py; design docs/design/conditional_ar_architecture.md §5.2).

Contracts: one operation's features carry what it is (a 2_1 screw's intrinsic translation, a glide's, inversion's
determinant and order, a centring translation's flag); the vocabulary covers every group's operations; e_g does not
depend on the order the operations are listed in, and different groups get different e_g. The full autoencoder fit
(4000 steps, ~3 min on CPU: every one of the 230 groups reconstructed exactly at 2026-11-04) is not run here."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.sym_ops import (OP_DIM, OP_FEATS, SymOpAutoencoder, SymOpEncoder, group_ops, group_tokens,  # noqa: E402
                            membership, op_features, vocabulary)

pytestmark = pytest.mark.fast
F = {n: i for i, n in enumerate(OP_FEATS)}


def _tau(f):
    return np.mod(np.arctan2(f[[F['sin_tau0'], F['sin_tau1'], F['sin_tau2']]],
                             f[[F['cos_tau0'], F['cos_tau1'], F['cos_tau2']]]) / (2 * np.pi), 1.0)


def test_a_screw_a_glide_inversion_and_centring_are_told_apart():
    screw = op_features(np.diag([-1.0, 1.0, -1.0]), [0.0, 0.5, 0.5])        # P2_1/c's 2_1 along b
    assert np.allclose(_tau(screw), [0.0, 0.5, 0.0]) and screw[F['order']] == 2 and screw[F['det']] == 1
    glide = op_features(np.diag([1.0, -1.0, 1.0]), [0.0, 0.5, 0.5])         # P2_1/c's c glide
    assert np.allclose(_tau(glide), [0.0, 0.0, 0.5]) and glide[F['det']] == -1
    inv = op_features(-np.eye(3), [0.0, 0.0, 0.0])
    assert inv[F['det']] == -1 and inv[F['order']] == 2 and np.allclose(_tau(inv), 0)
    centring = op_features(np.eye(3), [0.5, 0.5, 0.0])                       # C centring
    assert centring[F['translation']] == 1 and op_features(np.eye(3), [0, 0, 0])[F['translation']] == 0
    assert group_tokens(14).shape == (4, OP_DIM)


def test_the_vocabulary_covers_every_group():
    keys, _ = vocabulary()
    assert len(keys) == 786                 # distinct operations of the 230 standard-setting groups, 2026-11-04
    for g in (1, 2, 14, 19, 61, 88, 148, 225, 230):
        assert int(membership(g).sum()) == len(group_ops(g))


def test_e_g_ignores_the_order_of_the_operations_and_separates_groups():
    torch.manual_seed(0)
    enc = SymOpEncoder(out_dim=16, hidden=32)
    tok = group_tokens(61)
    idx = torch.zeros(tok.shape[0], dtype=torch.long)
    perm = torch.randperm(tok.shape[0])
    a, b = enc(tok, idx, 1), enc(tok[perm], idx, 1)
    assert torch.allclose(a, b, atol=1e-6)
    e = enc.embed([3, 4, 14, 19])           # P2 vs P2_1: same rotations, one screw
    assert torch.cdist(e, e).add(torch.eye(4) * 1e9).min() > 1e-4


def test_the_autoencoder_reads_and_writes_the_vocabulary():
    torch.manual_seed(0)
    model = SymOpAutoencoder(out_dim=16, hidden=32)
    assert model([1, 14]).shape == (2, len(vocabulary()[0]))
