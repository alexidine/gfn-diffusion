"""A carrier `policy_kind: set` run resumed from its checkpoint continues EXACTLY: same state,
same next fused loss, same gradients.

The fast unit test (test_set_policy_resume.py) pins the model half on a stub. This one runs
the REAL trainer: ConformerModeller built from a config through get_train_args, the train()
init sequence (energy, init_gfn, datasets, identifiers, buffer seeds, tracker, protocol.begin),
a few bwd steps in train_prior, the stage transition into a fused TB stage with `freeze_pb` on
its entry, a few fused steps, and Checkpointer.save with its buffer sidecar. A second modeller
then loads that file through the same init sequence, and the two are compared.

WHAT IS COMPARED, AND WHEN. Model, EMA, P_B snapshot, every optimizer's state (Adam's step
counter included), modeller_state, metric tracker and condition_log_z are read straight after
the load. The buffers are read after the WHOLE init sequence: init_prior_buffer_seed,
grow_prior_buffer and init_anchor_buffer_seed all run after init_gfn, so a mutation any of
them makes to a restored buffer is a resume defect this test is meant to catch, not noise.
Then both modellers compute the fused loss under identical seeds, without stepping, and the
loss and every gradient must agree bitwise.

THREE THINGS ARE DELIBERATELY NOT EQUAL, and each is asserted as what it is:
  * `batch_size_last_grow` -- with grow_batch_size false the config pins the batch, and
    Checkpointer.reconcile_batch_size clears the sizer bookkeeping on purpose;
  * the prior DATASET -- rebuilt on every leg from the fitted InternalPrior, now drawn from
    the RESTORED prior rng, so it continues the stream instead of replaying it (RES-7);
  * the global torch RNG -- nothing checkpoints it (model construction reseeds it), so the
    fused-loss comparison seeds torch, numpy and random explicitly on both sides.

The replay branch is off (fracs replay 0): on this route `_finish_replay_draw` passes
`raw_latents` to a conformer energy that does not take it (train.py, owner lane).

DATA. The conditions file is built here, carrier-padded, with random frozen embeddings -- the
values are irrelevant to a resume test. The fitted InternalPrior is the local
`conformer_prior_v2.pt` (not in git); point GFN_CONFORMER_PRIOR at it from a worktree.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/conformer/test_conformer_resume_roundtrip.py
"""
import copy
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

pytestmark = pytest.mark.slow

HERE = Path(__file__).resolve().parents[2]            # energy_sampling/
BASE_CONFIG = HERE / 'configs' / 'conformer_mk_multi.yaml'
SMIS = ['C', 'CO', 'N']                                # full/mmff: carrier K = 12 (5|4|3)
MOL_DIM, ENC = 16, 8
BWD_STEPS, FUSED_STEPS = 3, 4

#: reconciled by design on a pinned-batch config (Checkpointer.reconcile_batch_size)
RECONCILED = {'batch_size_last_grow': 0, 'batch_sizer': None, 'batch_size_cooldown_until': -1}


def _prior_path():
    p = Path(os.environ.get('GFN_CONFORMER_PRIOR', HERE / 'conformer_prior_v2.pt'))
    if not p.exists():
        pytest.fail(f'fitted InternalPrior not found at {p}: this test cannot run without it. '
                    f'Set GFN_CONFORMER_PRIOR to the local conformer_prior_v2.pt.')
    return str(p).replace('\\', '/')


def _write_conditions(path):
    """Carrier-padded condition graphs, as build_conformer_conditions.py --carrier writes them
    (float64, like the builder), with random frozen embeddings in place of the encoder's."""
    from energies.conformer_carrier import carrier_pad_condition
    from energies.conformer_data import (collate_conditions, condition_from_energy,
                                         save_condition_file)
    from energies.dof_features import free_dof_atom_index
    from energies.multi_conformer import MultiConformerTorsions

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        en = MultiConformerTorsions(SMIS, identifiers=SMIS, device='cpu', level='full',
                                    force_field='mmff')
        g = torch.Generator().manual_seed(0)
        rows = []
        for ident, mem in en._members.items():
            c = condition_from_energy(mem, identifier=ident)
            a, msk = free_dof_atom_index(mem)
            cc = carrier_pad_condition(c, en.carrier, ident, mem, atoms=a, mask=msk,
                                       R=int(a.shape[1]))
            cc.atom_embedding = torch.randn(int(cc.num_nodes), ENC, generator=g)
            cc.embedding = torch.randn(1, MOL_DIM, generator=g)
            rows.append(cc)
        save_condition_file(collate_conditions(rows), str(path))
    finally:
        torch.set_default_dtype(old)


def _config(tmp, prior):
    """The multi-molecule conformer config, shrunk to CPU size. Only sizes, paths and the
    protocol change; every other key is the file's."""
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding='utf-8'))
    cfg.update(run_name='rt', device='cpu', buffer_device='cpu',
               checkpoints_dir=str(tmp).replace('\\', '/'),
               molecules_path=str(tmp / 'cond.pt').replace('\\', '/'),
               test_molecules_path=None, checkpoint_name=None, load_weights_only=False,
               compile_policy=False, embedding_conditioning_dim=MOL_DIM, batch_size=12,
               max_batch_size=12, grow_batch_size=False, fused_grad_accum_min_samples=0,
               # a decayed EMA, so model_eval differs from model_train and its equality
               # after the load is a real check (null aliases them; the fast test pins that)
               ema_decay=0.5, archive_period=0, epochs=1000, eval_T=6)
    cfg['integrator']['T'] = 6
    for k in ('s_emb_dim', 't_hidden_dim', 's_hidden_dim', 'policy_hidden_dim',
              'flow_hidden_dim', 'cond_hidden_dim', 't_dim', 'harmonics_dim'):
        cfg['model'][k] = 32
    for k in ('s_layers', 'policy_layers', 'flow_layers', 'cond_layers'):
        cfg['model'][k] = 2
    cfg['model'].update(policy_kind='set', set_policy_hidden=32, set_policy_layers=2,
                        set_policy_corr_dim=8, dplr_rank=0)
    cfg['energy_config'].update(internal_prior_path=prior, prior_sample_size=60)
    cfg['buffers']['prior_buffer'].update(source='anchors', min_size=10, max_size=60,
                                          churn_batch_ref=12)
    cfg['buffers']['anchor_buffer'].update(max_size=200)
    cfg['buffers']['replay_buffer'].update(max_size=200)
    cfg['protocol'] = 'roundtrip'
    cfg['protocols']['roundtrip'] = {'stages': [
        {'name': 'train_prior', 'train_mode': 'bwd', 'bwd_sampling_mode': 'dataset',
         'flags': {'update_log_z': True},
         'loss_coeffs': {'bwd': {'mle': 1.0, 'tbc': 0.0, 'repeats': 1.0}},
         'exit': [{'metric': 'gates/progress_done', 'above': 0.5, 'patience': 1}]},
        {'name': 'tb', 'train_mode': 'fused', 'bwd_sampling_mode': 'prior',
         'flags': {'update_log_z': True, 'buffers_active': True, 'z_calibration': False},
         'on_enter': ['freeze_pb'], 'fracs': {'fwd': 0.5, 'bwd': 0.5, 'replay': 0.0},
         'deactivate_threshold': 0.01,
         'loss_coeffs': {'fwd': {'tb': 1.0, 'freeze_policy': 0.0, 'freeze_z': 0.0},
                         'bwd': {'tb': 1.0, 'freeze_z': 0.0}}},
    ]}
    return cfg


def _modeller(tmp, cfg, **overrides):
    from conformer_modeller import ConformerModeller
    from utils import get_train_args

    c = copy.deepcopy(cfg)
    c.update(overrides)
    path = tmp / f"{c['run_name']}.yaml"
    path.write_text(yaml.safe_dump(c), encoding='utf-8')
    return ConformerModeller(args=get_train_args(['--config', str(path)]))


def _init_rest(m):
    """train()'s init sequence after init_gfn, in train()'s order."""
    m.init_mol_dataset()
    m.init_prior_dataset()
    m.init_identifiers()
    m.init_prior_buffer_seed()
    if m._has_prior_sampler() and m.args.buffers.prior_buffer.source != 'anchors':
        m.grow_prior_buffer()
    m.init_condition_log_z()
    m.init_anchor_buffer_seed()
    m.checkpointer.assert_buffer_currency('buffers seeded')
    m.protocol.begin()
    m.gfn_model.train()


def _run(m, n):
    """The train loop's per-step order (coefficients, step, z fill), minus the LR controller,
    evals and saves -- none of which this comparison is about."""
    start = int(m.step_ind)
    for step in range(start, start + n):
        m.step_ind = step
        if step % 10 == 0 or step == start:
            m.set_loss_coeffs()
            m.set_energy_coeffs()
        kind = m.train_logic(step)
        if m.train_step(kind) is not None:
            m.z_level_fill()
            m.z_calibration_tick(kind)
    m.step_ind = start + n


def _diff(x, y, path='root'):
    """Every path at which x and y are not bitwise equal (NaN == NaN)."""
    if torch.is_tensor(x) or torch.is_tensor(y):
        ok = (torch.is_tensor(x) and torch.is_tensor(y) and x.shape == y.shape
              and x.dtype == y.dtype)
        if ok and x.is_floating_point():
            nx, ny = torch.isnan(x), torch.isnan(y)
            ok = torch.equal(nx, ny) and torch.equal(x[~nx].cpu(), y[~ny].cpu())
        elif ok:
            ok = torch.equal(x.cpu(), y.cpu())
        return [] if ok else [path]
    if hasattr(x, '_store') and hasattr(y, '_store'):          # a PyG batch
        return _diff(dict(x._store), dict(y._store), path + '.batch')
    if isinstance(x, dict) and isinstance(y, dict):
        out = []
        for k in set(x) | set(y):
            out += ([f'{path}.{k} (missing)'] if k not in x or k not in y
                    else _diff(x[k], y[k], f'{path}.{k}'))
        return out
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            return [f'{path} (len {len(x)} vs {len(y)})']
        return [d for i, (u, v) in enumerate(zip(x, y)) for d in _diff(u, v, f'{path}[{i}]')]
    if isinstance(x, np.random.Generator) and isinstance(y, np.random.Generator):
        return [] if x.bit_generator.state == y.bit_generator.state else [path]
    if isinstance(x, np.ndarray):
        return [] if np.array_equal(x, y, equal_nan=True) else [path]
    if isinstance(x, float) and isinstance(y, float) and x != x and y != y:
        return []
    return [] if x == y else [f'{path} ({x!r} vs {y!r})']


@pytest.fixture(scope='module')
def pair(tmp_path_factory):
    """(A, B, snapshot of B right after its load). A trains and saves; B loads."""
    old_dtype, old_wandb = torch.get_default_dtype(), os.environ.get('WANDB_MODE')
    torch.set_default_dtype(torch.float32)           # the route's dtype (conformer_modeller main)
    os.environ['WANDB_MODE'] = 'disabled'
    try:
        tmp = tmp_path_factory.mktemp('roundtrip')
        _write_conditions(tmp / 'cond.pt')
        cfg = _config(tmp, _prior_path())

        a = _modeller(tmp, cfg)
        a.init_energy_function()
        a.init_gfn()
        _init_rest(a)
        _run(a, BWD_STEPS)
        a.protocol.advance(None)                      # train_prior -> tb, fires freeze_pb
        _run(a, FUSED_STEPS)
        a.checkpointer.save('probe', with_buffers=True)

        b = _modeller(tmp, cfg, run_name='rt_b', checkpoint_read_only=True,
                      checkpoint_name=Path(a.checkpointer.path_for('probe')).name)
        b.init_energy_function()
        b.init_gfn()
        after_load = {
            'modeller_state': copy.deepcopy(b.checkpointer.get_state_dict()),
            'metrics': copy.deepcopy(b.metric_tracker.state_dict()),
            'tracker': copy.deepcopy(b.condition_log_z.state_dict()),
        }
        _init_rest(b)
        yield a, b, after_load
    finally:
        torch.set_default_dtype(old_dtype)
        if old_wandb is None:
            os.environ.pop('WANDB_MODE', None)
        else:
            os.environ['WANDB_MODE'] = old_wandb


def test_the_run_is_where_it_was(pair):
    a, b, _ = pair
    from models.conformer_gfn import ConformerGFN
    from models.ragged_set_policy import RaggedConditionalSetPolicy
    assert (b.stage, b.step_ind) == ('tb', BWD_STEPS + FUSED_STEPS) == (a.stage, a.step_ind)
    for model in (b.gfn_model, b.ema_model):
        assert type(model) is ConformerGFN and model._carrier is True
        assert isinstance(model.forward_policy, RaggedConditionalSetPolicy)
    assert b.gfn_config['conformer'] == a.gfn_config['conformer']


def test_model_ema_pb_and_optimizers_are_restored(pair):
    a, b, _ = pair
    assert _diff(a.gfn_model.state_dict(), b.gfn_model.state_dict()) == []
    assert _diff(a.ema_model.state_dict(), b.ema_model.state_dict()) == []
    assert _diff(a.gfn_model.state_dict(), a.ema_model.state_dict()) != [], \
        'separator: the decayed EMA must differ from the live model'
    assert a.gfn_model.pb_frozen and b.gfn_model.pb_frozen
    assert b.ema_model._pb_frozen is b.gfn_model._pb_frozen
    assert _diff(a.gfn_model.pb_snapshot_state(), b.gfn_model.pb_snapshot_state()) == []
    assert _diff({k: o.state_dict() for k, o in a.optimizers.items()},
                 {k: o.state_dict() for k, o in b.optimizers.items()}) == []
    steps = {float(s['step']) for s in b.optimizers['fused'].state_dict()['state'].values()}
    assert steps == {float(FUSED_STEPS)}, 'Adam restarted at the stage entry and ran FUSED_STEPS'


def test_modeller_state_tracker_and_metrics_are_restored(pair):
    a, _, after_load = pair
    diffs = _diff(a.checkpointer.get_state_dict(), after_load['modeller_state'])
    unexpected = [d for d in diffs if not any(d.startswith(f'root.{k} ') for k in RECONCILED)]
    assert unexpected == [], unexpected
    for k, v in RECONCILED.items():
        assert after_load['modeller_state'][k] == v, k
    assert after_load['modeller_state']['_prior_rng_state'] is not None
    assert _diff(a.metric_tracker.state_dict(), after_load['metrics']) == []
    assert _diff(a.condition_log_z.state_dict(), after_load['tracker']) == []


def test_buffers_survive_the_whole_init_sequence(pair):
    a, b, _ = pair
    for name in ('prior_buffer', 'replay_buffer', 'anchor_buffer'):
        ba, bb = getattr(a, name), getattr(b, name)
        assert len(ba) > 0, f'{name} is empty -- the comparison would be vacuous'
        # the conditions file makes every store compact (buffer.py::ConformerCompactRows),
        # so this also round-trips compact rows through the sidecar and the bind on load
        assert ba.is_compact and bb.is_compact, name
        assert _diff(ba.state_dict(), bb.state_dict()) == [], name
        for key in ('torsion_state', 'state_mask', 'dof_static', 'mol_id'):
            assert torch.equal(getattr(ba.batch, key), getattr(bb.batch, key)), (name, key)


def test_the_prior_dataset_continues_the_stream_rather_than_replaying_it(pair):
    a, b, _ = pair
    assert not torch.equal(a.prior_dataset.batch.torsion_state,
                           b.prior_dataset.batch.torsion_state)


def test_the_next_fused_loss_and_gradients_are_bitwise_equal(pair):
    a, b, _ = pair
    import train

    def fused(m):
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)
        for opt in m.optimizers.values():
            opt.zero_grad(set_to_none=True)
        m.set_loss_coeffs()
        loss, _ = m.fused_train_step(train.get_discretizer(m.args.integrator),
                                     report_losses=True)
        loss.backward()
        return loss.detach(), {n: None if p.grad is None else p.grad.detach().clone()
                               for n, p in m.gfn_model.named_parameters()}

    la, ga = fused(a)
    lb, gb = fused(b)
    assert torch.isfinite(la) and torch.equal(la, lb), (float(la), float(lb))
    assert _diff(ga, gb) == []
    assert any(g is not None and bool(g.abs().sum() > 0) for g in ga.values()), \
        'no gradient reached any parameter -- the equality above would be vacuous'
