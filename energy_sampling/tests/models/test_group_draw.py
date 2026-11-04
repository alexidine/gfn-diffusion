"""The space-group step under option A (models/group_draw.py; docs/design/conditional_ar_architecture.md §5.7).

Contracts:
- draw_groups draws from (1 - eps) softmax(L) + eps Uniform over the allowed groups, returns the head's log p(g | m),
  never a disallowed group, and refuses a row with none;
- level_group_logits is the policy's level with each row's space group set to each candidate, so it moves with g only
  through what the level reads (e_g here);
- group_weights_is recovers each group's log Z from importance weights, with its ESS;
- condition_log_z.rollout_group_draw and its uniform share are validated, a dead share refused."""
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.group_draw import draw_groups, group_weights_is, level_group_logits  # noqa: E402

pytestmark = pytest.mark.fast


def test_draw_groups_follows_the_mixture_and_the_allowed_set():
    g = torch.Generator().manual_seed(0)
    groups = [2, 14, 61]
    logits = torch.tensor([[2.0, 0.0, -1.0]]).expand(40000, -1)
    eps = 0.3
    drawn, logp, p = draw_groups(logits, groups, uniform_frac=eps, generator=g)
    want = (1 - eps) * torch.softmax(logits[0].double(), 0) + eps / 3
    freq = torch.stack([(drawn == sg).double().mean() for sg in groups])
    assert torch.allclose(freq, want, atol=0.01), (freq, want)
    k = torch.tensor([groups.index(int(x)) for x in drawn[:100]])
    assert torch.allclose(logp[:100], torch.log_softmax(logits[0].double(), 0)[k])
    allowed = torch.tensor([[True, False, True]]).expand(40000, -1)
    drawn, logp, p = draw_groups(logits, groups, uniform_frac=0.5, allowed=allowed, generator=g)
    assert not bool((drawn == 14).any()) and float(p[0, 1]) == 0.0
    assert torch.allclose(p[0, [0, 2]].sum(), torch.tensor(1.0, dtype=torch.float64))
    with pytest.raises(ValueError, match='no allowed'):
        draw_groups(logits[:2], groups, allowed=torch.zeros(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match='uniform_frac'):
        draw_groups(logits[:2], groups, uniform_frac=1.5)


def test_group_weights_is_recovers_each_groups_log_z():
    g = torch.Generator().manual_seed(1)
    # q(x | g) = N(0, 1); R_g(x) = Z_g N(x; 0, 1.1^2), so E_q[R/q] = Z_g
    log_z_true = torch.tensor([0.0, math.log(3.0)], dtype=torch.float64)
    n = 200000
    x = torch.randn(2, n, generator=g, dtype=torch.float64)
    log_q = -0.5 * x ** 2 - 0.5 * math.log(2 * math.pi)
    log_r = log_z_true[:, None] - 0.5 * (x / 1.1) ** 2 - 0.5 * math.log(2 * math.pi * 1.1 ** 2)
    log_w = (log_r - log_q).reshape(-1)
    gi = torch.arange(2).repeat_interleave(n)
    log_z, w, ess = group_weights_is(log_w, gi, 3)
    assert torch.allclose(log_z[:2], log_z_true, atol=0.01), log_z
    assert torch.allclose(w[:2], torch.tensor([0.25, 0.75], dtype=torch.float64), atol=0.005)
    assert log_z[2] == -math.inf and w[2] == 0 and ess[2] == 0
    assert 0.9 * n < float(ess[0]) <= n


def test_level_group_logits_is_the_level_under_each_group():
    from test_nf_policy import SMALL, gfn_config
    from test_nf_split_encoder import TF, _mols, _with_fields
    from models.nf_policy import NFPolicy
    cfg = gfn_config(True, cond_dim=4)
    cfg['conditions_type'] = 'molecule'
    torch.manual_seed(0)
    pol = NFPolicy(cfg, {**SMALL, **TF, 'atom_tokens': 'split', 'sym_ops': 'pooled'}, seed=0)
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in pol.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=gen))
    pol.eval()
    cond = torch.randn(3, 4)

    def condition_of(sg):
        mb = _with_fields(_mols())
        mb.sg_ind = torch.full((3,), sg)
        return mb, cond
    logits = level_group_logits(pol, condition_of, None, [2, 14])
    assert logits.shape == (3, 2)
    for k, sg in enumerate((2, 14)):
        mb, c = condition_of(sg)
        with torch.no_grad():
            want = pol.flow_model(pol.get_condition_embedding(c, mb))
        assert torch.allclose(logits[:, k], want.float())
    assert not torch.allclose(logits[:, 0], logits[:, 1])


def test_rollout_group_draw_keys_are_validated():
    from config_invariants import rollout_condition_draw_problems

    def glob_of(d):
        return lambda dotted: d.get(dotted.split('.')[-1])
    assert rollout_condition_draw_problems(glob_of({'rollout_group_draw': 'level',
                                                    'rollout_group_draw_uniform_frac': 0.2})) == []
    assert rollout_condition_draw_problems(glob_of({'rollout_group_draw': 'uniform',
                                                    'rollout_group_draw_uniform_frac': 0.1})) == []
    bad = rollout_condition_draw_problems(glob_of({'rollout_group_draw': 'softmax'}))
    assert bad and 'rollout_group_draw' in bad[0]
    dead = rollout_condition_draw_problems(glob_of({'rollout_group_draw': 'uniform',
                                                    'rollout_group_draw_uniform_frac': 0.3}))
    assert dead and 'dead key' in dead[0]
    out = rollout_condition_draw_problems(glob_of({'rollout_group_draw': 'level',
                                                   'rollout_group_draw_uniform_frac': 1.5}))
    assert out and '[0, 1]' in out[0]
