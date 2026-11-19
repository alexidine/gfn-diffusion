"""Per-space-group readouts (eval/sg_readouts.py) and the Z(sg) panels (train.py::Modeller.log_zsg_panels). What each
test would catch:

  * a group's train-cadence tb_err / logw_std_within / mle / log_Z_learned taken over another group's rows, the
    group decoded from the wrong digit of condition_id, the within-condition spread seeing a per-condition offset,
    the loss's own centre (log_z_target) ignored, the K-tiled MLE rows mis-aligned, or any of it written on a
    single-group run;
  * a buffer share missing a group that holds no rows (it must read 0);
  * the head panel's p(g | m) not a softmax of the level per molecule, or its entropy / level means wrong, on a small
    mixed-group flow policy (GROUPS = (2, 14), sym_ops pooled, so the level reads the group);
  * the IS panel not recovering each (molecule, group) log Z, its normalised weights, ESS, or head - IS;
  * the panel redrawn between evals, the IS half firing off its every_n_evals cadence, the panel present when the
    block is off or the run has one group, an OOM retried, a non-OOM error swallowed, the test panel missing.

CPU, synthetic, no energy computed.

    python -m pytest -q tests/crystal/test_sg_readouts.py
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests' / 'models'))

from eval.sg_readouts import (branch_group_stats, buffer_group_share, condition_groups,  # noqa: E402
                              head_panel_stats, is_panel_stats, panel_line, ZSG_CAVEAT)

pytestmark = pytest.mark.fast

GROUPS, N_ZP = [2, 14], 2


def _cid(mol, sg_local, zp_local=0):
    return torch.as_tensor(mol) * (len(GROUPS) * N_ZP) + torch.as_tensor(sg_local) * N_ZP + torch.as_tensor(zp_local)


def test_condition_groups_decodes_the_group_digit():
    cid = _cid([0, 0, 3, 3, 7], [0, 1, 0, 1, 1], [1, 0, 1, 1, 0])
    assert condition_groups(cid, GROUPS, N_ZP).tolist() == [2, 14, 2, 14, 14]


def _tb_dict(seed=0):
    """Two groups, two conditions each (4 rows per condition); group 14 fits worse; each condition's log w carries
    its own offset, which logw_std_within must not see."""
    g = torch.Generator().manual_seed(seed)
    mol = torch.tensor([0] * 4 + [1] * 4 + [0] * 4 + [1] * 4)
    sgl = torch.tensor([0] * 8 + [1] * 8)
    cid = _cid(mol, sgl)
    log_r = torch.randn(16, generator=g, dtype=torch.float64)
    spread = torch.where(sgl == 0, 0.5, 2.0).double()
    noise = torch.randn(16, generator=g, dtype=torch.float64) * spread
    offset = (mol * 30.0 + sgl * 100.0).double()
    log_pb = torch.zeros(16, dtype=torch.float64)
    log_pf = log_r - offset - noise            # log w = offset + noise
    log_z = torch.full((16,), 7.0, dtype=torch.float64)
    return dict(log_pf=log_pf, log_pb=log_pb, log_r=log_r, log_Z=log_z, condition_id=cid), noise, offset, sgl


def test_branch_stats_split_tb_err_and_within_spread_by_group():
    d, noise, offset, sgl = _tb_dict()
    out = branch_group_stats(d, 'fwd', GROUPS, N_ZP)
    assert set(out) == {'sg2', 'sg14'}
    resid = d['log_pf'] + d['log_Z'] - d['log_pb'] - d['log_r']
    for k, g in enumerate(GROUPS):
        sel = sgl == k
        assert math.isclose(out[f'sg{g}']['fwd/tb_err'], float(resid[sel].pow(2).mean().sqrt()), rel_tol=1e-9)
        n = noise[sel].reshape(2, 4)
        want = float((n - n.mean(1, keepdim=True)).pow(2).mean().sqrt())   # offsets drop out
        assert math.isclose(out[f'sg{g}']['fwd/logw_std_within'], want, rel_tol=1e-9)
        assert math.isclose(out[f'sg{g}']['fwd/log_Z_learned'], 7.0)
    assert out['sg14']['fwd/logw_std_within'] > out['sg2']['fwd/logw_std_within']
    # bwd: no within spread (fwd and replay only); the loss's own centre replaces the head
    d['log_z_target'] = d['log_Z'] + 1.0
    out = branch_group_stats(d, 'bwd', GROUPS, N_ZP)
    assert 'bwd/logw_std_within' not in out['sg2']
    sel = sgl == 0
    want = float((resid[sel] + 1.0).pow(2).mean().sqrt())
    assert math.isclose(out['sg2']['bwd/tb_err'], want, rel_tol=1e-9)


def test_branch_stats_split_the_mle_rows_on_a_k_tiled_draw():
    # 3 terminals x K = 2 rollouts: condition_id is [6], mle_rows [3] (one per terminal)
    cid = _cid([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 1, 1])
    d = dict(condition_id=cid, mle_rows=torch.tensor([1.0, 4.0, 6.0]), log_Z=torch.zeros(6))
    out = branch_group_stats(d, 'bwd', GROUPS, N_ZP)
    assert out['sg2']['bwd/mle'] == 1.0 and out['sg14']['bwd/mle'] == 5.0
    assert 'bwd/tb_err' not in out['sg2']                     # a reward-free draw has no residual


def test_branch_stats_are_inert_on_one_group_or_without_condition_ids():
    d, *_ = _tb_dict()
    assert branch_group_stats(d, 'fwd', [2], 1) == {}
    d.pop('condition_id')
    assert branch_group_stats(d, 'fwd', GROUPS, N_ZP) == {}


def test_buffer_share_reads_zero_for_an_empty_group():
    out = buffer_group_share(torch.tensor([2, 2, 2, 14]), [2, 14, 61], 'replay_buffer_share')
    assert out == {'sg2/replay_buffer_share': 0.75, 'sg14/replay_buffer_share': 0.25, 'sg61/replay_buffer_share': 0.0}
    assert buffer_group_share(torch.tensor([2, 2]), [2], 'x') == {}


def _mixed_policy():
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
    return pol.eval(), _mols, _with_fields


def test_head_panel_on_a_mixed_group_policy():
    from models.group_draw import level_group_logits
    pol, _mols, _with_fields = _mixed_policy()
    cond = torch.randn(3, 4, generator=torch.Generator().manual_seed(1))

    def condition_of(sg):
        mb = _with_fields(_mols())
        mb.sg_ind = torch.full((3,), sg)
        return mb, cond
    L = level_group_logits(pol, condition_of, None, GROUPS).double()
    out = head_panel_stats(L, GROUPS)
    p = torch.softmax(L, -1)
    assert not torch.allclose(L[:, 0], L[:, 1])               # the level reads the group
    assert math.isclose(out['zsg/head_p_sg2'] + out['zsg/head_p_sg14'], 1.0, rel_tol=1e-9)
    assert math.isclose(out['zsg/head_p_sg14'], float(p[:, 1].mean()), rel_tol=1e-9)
    assert math.isclose(out['zsg/head_level_sg2'], float(L[:, 0].mean()), rel_tol=1e-9)
    ent = float((-(p * p.log()).sum(-1)).mean())
    assert math.isclose(out['zsg/head_entropy_nats'], ent, rel_tol=1e-9) and 0 < ent <= math.log(2)
    assert head_panel_stats(L[:, :1], [2]) == {}


def test_is_panel_recovers_each_molecules_group_log_z():
    g = torch.Generator().manual_seed(2)
    a = torch.tensor([[0.0, math.log(3.0)], [1.0, 1.0]], dtype=torch.float64)   # true log Z(m, g)
    n = 100000
    lw, mi, gi = [], [], []
    for m in range(2):
        for k in range(2):
            x = torch.randn(n, generator=g, dtype=torch.float64)
            log_q = -0.5 * x ** 2 - 0.5 * math.log(2 * math.pi)
            log_r = a[m, k] - 0.5 * (x / 1.1) ** 2 - 0.5 * math.log(2 * math.pi * 1.1 ** 2)
            lw.append(log_r - log_q)
            mi.append(torch.full((n,), m))
            gi.append(torch.full((n,), k))
    lw = torch.cat(lw)
    lw[5] = math.nan                                           # a crashed row is dropped, not contagious
    level = a + torch.tensor([[0.5, -0.5], [0.0, 2.0]], dtype=torch.float64)
    out = is_panel_stats(lw, torch.cat(mi), torch.cat(gi), 2, GROUPS, level=level)
    assert math.isclose(out['zsg/is_log_z_sg2'], float(a[:, 0].mean()), abs_tol=0.02)
    assert math.isclose(out['zsg/is_log_z_sg14'], float(a[:, 1].mean()), abs_tol=0.02)
    w = torch.softmax(a, -1)
    assert math.isclose(out['zsg/is_p_sg14'], float(w[:, 1].mean()), abs_tol=0.01)
    ent = float((-(w * w.log()).sum(-1)).mean())
    assert math.isclose(out['zsg/is_entropy_nats'], ent, abs_tol=0.01)
    assert math.isclose(out['zsg/head_minus_is_sg2'], 0.25, abs_tol=0.02)
    assert math.isclose(out['zsg/head_minus_is_sg14'], 0.75, abs_tol=0.02)
    assert 0.9 * n < out['zsg/is_ess_sg2'] <= n
    line = panel_line(out, GROUPS, 'zsg/', 2, n)
    assert ZSG_CAVEAT in line and 'sg14' in line and 'IS logZ' in line


# --- Modeller.log_zsg_panels on a Modeller built without __init__ ------------------------------------------------

A = torch.tensor([[0.0, 1.0], [2.0, 0.5], [-1.0, 3.0], [0.3, 0.3], [1.5, -0.5]], dtype=torch.float64)  # log Z(m, g)
B = A + torch.tensor([1.0, -2.0], dtype=torch.float64)                                                 # level L(m, g)


class _Mols:
    """Panel molecules: mol_id per row, indexing A and B."""
    device = torch.device('cpu')

    def __init__(self, mol_id):
        self.mol_id = torch.as_tensor(mol_id)
        self.num_graphs = len(self.mol_id)
        self.ptr = torch.arange(self.num_graphs + 1)

    def subsample_new_batch(self, idx):
        return _Mols(self.mol_id[idx])

    def to(self, device):
        return self

    def clone(self):
        return _Mols(self.mol_id.clone())

    def orient_molecule(self, mode):
        self.oriented = mode


class _Ef:
    space_groups, n_zp, z_primes = GROUPS, 1, [1]
    temperature, temperature_conditioning, data_ndim = 1.0, False, 3

    def condition_samples(self, mb, temperature=None, sg_inds=None, z_primes=None, repeats=1):
        mb.sg_ind = sg_inds
        local = (sg_inds == 14).long()
        cond = torch.stack([mb.mol_id.double(), local.double()], -1)
        return mb, torch.zeros(mb.num_graphs), cond, mb.mol_id * 2 + local

    def log_reward(self, x, mol_batch, log_temperature, return_exp=True):
        # log R = A[m, g] + log N(x; 0, 1), so log w = log R - log q = A[m, g] exactly under q = N(0, 1)
        local = (mol_batch.sg_ind == 14).long()
        log_n = -0.5 * x.pow(2).sum(-1) - 0.5 * x.shape[-1] * math.log(2 * math.pi)
        return A[mol_batch.mol_id, local] + log_n, SimpleNamespace(cpu=lambda: SimpleNamespace(detach=lambda: 0))


class _Policy:
    device = torch.device('cpu')
    conditional = True

    def __init__(self, fail=None):
        self.fail, self.calls = fail, []

    def get_condition_embedding(self, cond, mb):
        return cond

    def flow_model(self, emb):
        return B[emb[:, 0].long(), emb[:, 1].long()]

    def get_traj_fwd(self, init_state, disc, _, condition, mol_batch, return_gauss_params=False):
        if self.fail is not None:
            raise self.fail
        n = init_state.shape[0]
        self.calls.append(n)
        x = torch.randn(n, 3, dtype=torch.float64)
        log_q = -0.5 * x.pow(2).sum(-1) - 1.5 * math.log(2 * math.pi)
        states = torch.stack([torch.zeros_like(x), x], 1)
        return states, log_q[:, None], torch.zeros(n, 1), torch.zeros(n, 2), {}


def _modeller(enabled=True, groups=GROUPS, every=2, step=10, fail=None, test=False):
    from energy_sampling.train import Modeller
    m = Modeller.__new__(Modeller)
    m.args = SimpleNamespace(seed=0, eval_period=5, space_groups=list(groups), z_primes=[1],
                             zsg_panel=SimpleNamespace(enabled=enabled, molecules=4, draws=8, every_n_evals=every),
                             oom_batch_shrink_factor=0.5)
    ef = _Ef()
    ef.space_groups = list(groups)
    m.energy_function, m.gfn_model = ef, _Policy(fail)
    m.device = torch.device('cpu')
    m.batch_size, m.step_ind, m.optimizers = 20, step, {}
    m.mol_dataset = SimpleNamespace(batch=_Mols(torch.arange(10) // 2))  # 5 molecules, 2 rows each
    m.test_mol_dataset = SimpleNamespace(batch=_Mols(torch.arange(10) // 2)) if test else None
    return m


def test_panels_read_head_and_is_on_their_cadences():
    m = _modeller(test=True)
    out = m.log_zsg_panels(None)                       # step 10 = 2 evals x 5: the IS half fires
    rows = m._zsg_panel_cache['mol_dataset']['rows']
    assert rows.numel() == 4 and bool((rows % 2 == 0).all())              # one row per distinct molecule
    mols = (rows // 2).long()
    for k, g in enumerate(GROUPS):
        assert math.isclose(out[f'zsg/head_level_sg{g}'], float(B[mols, k].mean()), abs_tol=1e-6)   # level: fp32
        assert math.isclose(out[f'zsg/is_log_z_sg{g}'], float(A[mols, k].mean()), rel_tol=1e-9)
        assert math.isclose(out[f'zsg/head_minus_is_sg{g}'], float((B - A)[0, k]), abs_tol=1e-6)
        assert math.isclose(out[f'zsg/is_ess_sg{g}'], 8.0, rel_tol=1e-9)                 # log w constant per cell
        assert math.isclose(out[f'zsg/is_p_sg{g}'], float(torch.softmax(A[mols], -1)[:, k].mean()), rel_tol=1e-9)
    assert out['zsg/is_skipped'] == 0.0
    assert any(k.startswith('zsg_test/is_log_z') for k in out)
    assert max(m.gfn_model.calls) <= 20                                  # chunked at the eval draw
    assert sum(m.gfn_model.calls) == 2 * (4 * 2 * 8)                     # 2 panels x 4 mols x 2 groups x 8 draws
    m.step_ind = 15                                                      # off the IS cadence: head only
    out2 = m.log_zsg_panels(None)
    assert not any('/is_' in k for k in out2) and 'zsg/head_p_sg2' in out2
    assert torch.equal(m._zsg_panel_cache['mol_dataset']['rows'], rows)  # the panel is fixed


def test_panels_are_inert_when_off_or_on_one_group():
    assert _modeller(enabled=False).log_zsg_panels(None) == {}
    assert _modeller(groups=[2]).log_zsg_panels(None) == {}


def test_an_oom_drops_the_is_panel_and_other_errors_raise():
    m = _modeller(fail=RuntimeError('CUDA out of memory. Tried to allocate'), step=0)   # step 0: no batch cut
    out = m.log_zsg_panels(None)
    assert out['zsg/is_skipped'] == 1.0 and 'zsg/head_p_sg2' in out and len(m.gfn_model.calls) == 0
    with pytest.raises(ValueError, match='boom'):
        _modeller(fail=ValueError('boom')).log_zsg_panels(None)


def test_zsg_panel_block_is_validated():
    from config_invariants import parse_zsg_panel, zsg_panel_is_well_formed
    assert parse_zsg_panel(None) == {'enabled': False, 'molecules': 64, 'draws': 32, 'every_n_evals': 4}
    with pytest.raises(ValueError, match='unknown'):
        parse_zsg_panel({'molecule': 3})
    with pytest.raises(ValueError, match='draws'):
        parse_zsg_panel({'draws': 0})
    assert parse_zsg_panel({'every_n_evals': None})['every_n_evals'] == 0
    assert zsg_panel_is_well_formed({'zsg_panel': {'enabled': True}, 'space_groups': [2, 14]}) == []
    v = zsg_panel_is_well_formed({'zsg_panel': {'enabled': True}, 'space_groups': [2]})
    assert v and v[0].severity == 'BASELINE'
    assert zsg_panel_is_well_formed({'zsg_panel': {'enabled': 'yes'}})[0].severity == 'ERROR'
