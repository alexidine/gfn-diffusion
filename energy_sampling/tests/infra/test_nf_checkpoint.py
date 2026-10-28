"""A flow policy (model.kind nf, models/nf_policy.py::NFPolicy) through the REAL Checkpointer.save / load_full /
load_weights_only, with the gfn_config built by the real train.py::Modeller._build_gfn_config and the optimizers by
the real Modeller.init_schedulers_optimizers, on a stub modeller carrying only what those paths read.

A full load gives back the same exact density (bitwise), the fitted standardisers, the level head and every Adam
step count; a weights-only load gives back the density and the standardisers and leaves the optimizers fresh. The
policy entry is stored resolved (models/nf_policy.py::resolve_policy_entry), a checkpoint whose entry was stored as
configured loads when its resolution matches, and the precision-only settings follow the run's config.

    python -m pytest -q tests/infra/test_nf_checkpoint.py
"""
import copy
import types
from types import SimpleNamespace

import pytest
import torch

from energy_sampling.checkpointing import MODELLER_STATE_DEFAULTS, Checkpointer
from energy_sampling.train import Modeller
from models.nf_policy import NF_DEFAULTS, build_policy, resolve_flow_settings

# small flows on the CPU; nothing reads the data drive
pytestmark = pytest.mark.fast

COND_DIM = 4
SMALL = dict(conditioner='made', hidden=32, hidden_layers=1, n_bins=8, context_dim=16, context_hidden=32,
             tied_derivatives=False, tf_fp32=False)


def _base_config(conditional):
    """The latent gfn_config a crystal run builds, at a 12-row latent with no dead rows."""
    return dict(dim=12, s_emb_dim=32, conditions_dim=COND_DIM if conditional else 0, harmonics_dim=16, t_dim=8,
                t_hidden_dim=32, s_hidden_dim=32, s_layers=2, policy_hidden_dim=32, policy_layers=2,
                flow_hidden_dim=32, flow_layers=2, cond_hidden_dim=32, cond_layers=2, log_var_range=6.0, t_scale=0.05,
                learned_variance=True, condition_embedding_dim=16 if conditional else 0, conditions_type='vector',
                clipping=True, gfn_clip=200.0, pb_drift_range=0.4, pb_var_range=6.0, conditional=conditional,
                learn_pb=True, dropout=0, norm='layer', zero_init=False, device='cpu', max_z_prime=1,
                full_flow=False, do_periodic_angles=True, periodic_centroids=True, periodic_centroid_axes=(1, 2),
                dead_latent_rows=None, dead_latent_values=None, dplr_rank=0, pb_exact_reversal=True)


def _modeller(ckdir, conditional, nf=None):
    """A stub Modeller: the real _policy_entry, _build_gfn_config and init_schedulers_optimizers bound to it."""
    m = SimpleNamespace()
    m.args = SimpleNamespace(
        checkpoints_dir=str(ckdir), archive_period=0, archive_buffers=False, checkpoint_read_only=False,
        grow_batch_size=False, batch_size=8, max_batch_size=8, override_learning_rates=False,
        model=SimpleNamespace(), buffers=SimpleNamespace(), seed=3,
        policy=SimpleNamespace(kind='nf', nf=SimpleNamespace(**(SMALL if nf is None else nf))),
        lr_policy=1e-3, lr_back=1e-3, lr_replay=1e-3, lr_fused=1e-3, lr_flow=1e-2,
        weight_decay=0.0, use_weight_decay=False, lr_control=None)
    m.run_name, m.problem_slug = 'nf_ck', 'probe-abc123'
    m.problem_def, m.problem_hash = {'energy_function': 'probe'}, 'abc123'
    m.device = m.buffer_device = 'cpu'
    for k, v in MODELLER_STATE_DEFAULTS.items():
        setattr(m, k, copy.deepcopy(v))
    m.metric_tracker = SimpleNamespace(state_dict=dict, load_state_dict=lambda s: None)
    m.grad_guard = SimpleNamespace(state_dict=lambda: None, load_state_dict=lambda s: None)
    m.energy_function = SimpleNamespace(fd_layout=None)
    m.protocol = SimpleNamespace(stage=SimpleNamespace(lr_sensor=None))
    m.lr_controller = SimpleNamespace(announce=lambda: None)
    m._ray_askers = lambda: []
    m._build_latent_gfn_config = lambda periodic_centroids=True: _base_config(conditional) | m._policy_entry()
    for name in ('_policy_entry', '_build_gfn_config', 'init_schedulers_optimizers'):
        setattr(m, name, types.MethodType(getattr(Modeller, name), m))
    m.gfn_from_config = build_policy
    m.optimizers = {}
    m.checkpointer = Checkpointer(m)
    return m


def _fresh_run(ckdir, conditional):
    """A fresh build: gfn_config, the policy and its EMA, standardisers fitted, two optimizer steps taken."""
    m = _modeller(ckdir, conditional)
    m.gfn_config = m._build_gfn_config()
    torch.manual_seed(0)
    m.gfn_model = build_policy(m.gfn_config)
    torch.manual_seed(1)
    m.gfn_model.fit_data_scalers(torch.rand(200, 12) * 1.6 - 0.8,
                                 torch.randn(50, COND_DIM) * 2 + 1 if conditional else None)
    m.ema_model = copy.deepcopy(m.gfn_model)
    m.init_schedulers_optimizers()
    x, c = _rows(conditional, seed=2)
    for _ in range(2):
        for opt in m.optimizers.values():
            opt.zero_grad(set_to_none=True)
        _, log_pfs, _, log_flow = m.gfn_model.get_traj_bwd(x, None, c, None)
        (-log_pfs.mean() + (log_flow[:, 0] - 1.0).pow(2).mean()).backward()
        m.optimizers['bwd'].step()
        m.optimizers['flow'].step()
    m.step_ind = 2
    return m


def _rows(conditional, seed, n=16):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 12, generator=g) * 2 - 1
    c = torch.randn(n, COND_DIM, generator=g) if conditional else None
    return x, c


def _readout(m, conditional):
    """log q of fixed rows under the train and EMA policies, and the level head on those rows."""
    x, c = _rows(conditional, seed=5)
    with torch.no_grad():
        level = m.gfn_model.get_traj_bwd(x, None, c, None)[3][:, 0]
    return m.gfn_model.log_density(x, c, None), m.ema_model.log_density(x, c, None), level


def _adam_steps(m):
    return {k: sorted(float(s['step']) for s in opt.state.values()) for k, opt in m.optimizers.items()}


@pytest.mark.parametrize('conditional', [False, True])
def test_load_full_restores_the_flow_its_scalers_level_and_adam_state(tmp_path, conditional):
    saved = _fresh_run(tmp_path, conditional)
    ref, steps = _readout(saved, conditional), _adam_steps(saved)
    assert steps['bwd'] and set(steps['bwd']) == {2.0} and set(steps['flow']) == {2.0}
    saved.checkpointer.save('running')

    m = _modeller(tmp_path, conditional)
    m.checkpointer.load_full(m.checkpointer.path_for('running'))
    assert bool(m.gfn_model.scalers_fitted) and bool(m.ema_model.scalers_fitted)
    got = _readout(m, conditional)
    for a, b in zip(got, ref):
        assert torch.equal(a, b)
    assert _adam_steps(m) == steps
    assert m.step_ind == 2 and m.gfn_config['policy'] == saved.gfn_config['policy']


@pytest.mark.parametrize('conditional', [False, True])
def test_load_weights_only_keeps_the_flow_and_scalers_and_starts_the_optimizers_fresh(tmp_path, conditional):
    saved = _fresh_run(tmp_path, conditional)
    ref = _readout(saved, conditional)
    saved.checkpointer.save('running')

    m = _modeller(tmp_path, conditional)
    m.checkpointer.load_weights_only(saved.checkpointer.path_for('running'))
    assert m.weights_only_loaded and bool(m.gfn_model.scalers_fitted)
    for a, b in zip(_readout(m, conditional), ref):
        assert torch.equal(a, b)
    m.init_schedulers_optimizers()   # train.py::Modeller.init_gfn's next call on this path
    assert all(not opt.state for opt in m.optimizers.values())
    assert m.step_ind == 0


@pytest.mark.parametrize('conditional', [False, True])
def test_gfn_config_stores_the_settings_resolved(tmp_path, conditional):
    m = _modeller(tmp_path, conditional)
    entry = m._build_gfn_config()['policy']
    assert entry['kind'] == 'nf' and entry['seed'] == 3
    assert entry['nf'] == resolve_flow_settings({**NF_DEFAULTS, **SMALL}, conditional, list(range(12)))
    assert 'auto' not in entry['nf'].values()
    assert m._policy_entry()['policy']['nf'] == SMALL   # the configured block, before resolution


def test_a_checkpoint_storing_the_settings_as_configured_loads_when_they_resolve_alike(tmp_path):
    """A checkpoint written before gfn_config stored the entry resolved holds the configured block ('auto' and
    unset values included); it loads, and its gfn_config is the resolved entry."""
    saved = _fresh_run(tmp_path, False)
    ref = _readout(saved, False)
    saved.checkpointer.save('running')
    path = saved.checkpointer.path_for('running')
    ck = torch.load(path, weights_only=False)
    ck['gfn_config']['policy'] = {'kind': 'nf', 'nf': dict(SMALL), 'seed': 3}
    torch.save(ck, path)

    m = _modeller(tmp_path, False)
    m.checkpointer.load_full(path)
    assert m.gfn_config['policy'] == saved.gfn_config['policy']
    for a, b in zip(_readout(m, False), ref):
        assert torch.equal(a, b)


def test_precision_settings_follow_the_run_and_other_settings_refuse(tmp_path, capsys):
    saved = _fresh_run(tmp_path, True)
    saved.checkpointer.save('running')
    path = saved.checkpointer.path_for('running')

    m = _modeller(tmp_path, True, nf={**SMALL, 'tf_fp32': True})
    m.checkpointer.load_full(path)
    assert m.gfn_config['policy']['nf']['tf_fp32'] is True
    assert "['tf_fp32']: checkpoint False -> config True" in capsys.readouterr().out

    m = _modeller(tmp_path, True, nf={**SMALL, 'n_bins': 16})
    with pytest.raises(ValueError, match="policy in this run's config"):
        m.checkpointer.load_full(path)


def test_resolution_reads_the_live_rows_as_the_gfn_does():
    """Dead rows leave the live rows (block_orders 'auto' is Z1_ORDER only on live rows 0-11); with
    hold_dead_latent_rows false the GFN holds none, and neither does the resolution."""
    from models.nf_policy import Z1_ORDER, resolve_policy_entry
    entry = {'kind': 'nf', 'nf': dict(SMALL), 'seed': 3}
    cfg = dict(dim=13, conditional=False, dead_latent_rows=[12])
    assert resolve_policy_entry(entry, cfg)['nf']['block_orders'] == Z1_ORDER
    assert resolve_policy_entry(entry, dict(cfg, hold_dead_latent_rows=False))['nf']['block_orders'] is None
    assert resolve_policy_entry(None, cfg) is None and resolve_policy_entry({'kind': 'sde'}, cfg) == {'kind': 'sde'}
