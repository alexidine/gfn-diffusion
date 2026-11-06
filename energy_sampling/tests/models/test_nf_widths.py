"""nf tf_mlp_ratio and level_hidden (models/nf_policy.py NF_PENDING_DEFAULTS): the transformer layers' MLP width and the
level head's width, settable like the other widths. Their defaults are the widths that were hard-wired before (4 x
tf_width; 256), so a config that does not name them builds the same model.

    python -m pytest -q tests/models/test_nf_widths.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nf_typed_rows import _typed_policy  # noqa: E402

pytestmark = pytest.mark.fast
BASE = dict(atom_tokens=True, atom_node_dim=128, atom_convs=3, tf_width=256, tf_heads=8, tf_layers=4,
            context_dim=128, context_hidden=512, row_tokens='prefix', n_bins=32)


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_the_defaults_build_the_model_they_replaced():
    assert _n(_typed_policy(2, scale=0.0, **BASE)) == 6_037_282   # counted before the keys existed (2026-11-06)


def test_tf_mlp_ratio_sets_every_layers_mlp():
    pol = _typed_policy(2, scale=0.0, **BASE, tf_mlp_ratio=1)
    layer = pol.sampler.flow.blocks[0].made.layers[0]
    assert layer.mlp[0].out_features == 256 and layer.mlp[2].in_features == 256
    assert _n(pol) < _n(_typed_policy(2, scale=0.0, **BASE))


def test_level_hidden_sets_the_level_heads_width():
    pol = _typed_policy(2, scale=0.0, **BASE, level_hidden=64)
    assert pol.sampler.flow.level.inp.out_features == 64
