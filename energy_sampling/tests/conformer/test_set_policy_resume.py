"""A `policy_kind: set` model survives a checkpoint round trip -- bitwise, EMA and Adam included.

WHAT WAS WRONG. The set head is swapped in AFTER `GFN(**gfn_config)` is built
(ConformerModeller._install_set_policy), and its keys were popped out of gfn_config, so a
checkpoint recorded nothing about it. Both load paths rebuilt a FLAT GFN from the stored
config and their strict load_state_dict failed on every `forward_policy.*` tensor. The only
guard was a refusal that fired on the wrong path: a FRESH launch with
`continue_from_checkpoint: true` and no running file was refused, while a real reload never
reached it.

WHAT IS PINNED HERE, through the REAL paths -- ConformerModeller.init_gfn, Modeller.init_gfn,
Checkpointer.save / load_full / load_weights_only -- on a stub modeller carrying only what
those paths read (no wandb, no buffers, no training loop):

  * a strict load succeeds and rebuilds class ConformerGFN with `_carrier` and the ragged head;
  * the EMA is a distinct model that shares ONE P_B snapshot object with the live one -- right
    after load; under `ema_decay: null` the trainer then aliases the two, also pinned here;
  * forward, backward and replay rollouts are bitwise equal, live and EMA, under a fixed seed;
  * after the optimizer state loads, one Adam step gives bitwise-equal parameters;
  * a second hop (save -> load -> save -> load) still carries gfn_config['conformer'];
  * the config is checked against the file field by field, and a width or layout mismatch is
    refused on BOTH load paths, naming both widths.

THE PERTURBATION IS NOT DECORATION. GFN construction reseeds torch's global RNG at 0
(scalarMLP), so a freshly built model equals any other fresh build of the same config. Every
weight is moved off its init before the save, live and EMA differently, and P_B's live trunk
is moved again after the freeze -- otherwise "the reload equals the original" would pass with
no load at all.

    python -m pytest -q tests/conformer/test_set_policy_resume.py
"""
import copy
from types import SimpleNamespace

import pytest
import torch

from energy_sampling.checkpointing import MODELLER_STATE_DEFAULTS, Checkpointer
from conformer_modeller import ConformerModeller
from energies.conformer_carrier import CarrierLayout, carrier_pad_condition
from energies.conformer_data import collate_conditions, condition_from_energy
from energies.dof_features import free_dof_atom_index
from energies.multi_conformer import MultiConformerTorsions
from models.conformer_gfn import ConformerGFN
from models.gfn import GFN
from models.ragged_set_policy import RaggedConditionalSetPolicy
from utils import MetricTracker

# torch is imported and small models are built, but nothing reads the data drive and the
# whole file runs in seconds on CPU: the fast lane is where a resume regression must show
pytestmark = pytest.mark.fast

SMIS = ['C', 'CO', 'N']            # full/mmff: k = 9, 12, 6 -> carrier K = 12, r|theta|phi 5|4|3
MOL_DIM, ENC = 16, 8
T = 5
KW = dict(device='cpu', level='full', force_field='mmff')


@pytest.fixture(scope='module', autouse=True)
def _float64():
    """float64 for the module, RESTORED after: a module-scope default leaks into every test
    collected after this one."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.fixture(scope='module')
def carrier():
    return MultiConformerTorsions(SMIS, identifiers=SMIS, **KW)


def _args(ckdir, **kw):
    model = SimpleNamespace(
        s_emb_dim=32, harmonics_dim=8, t_dim=16, t_hidden_dim=32, condition_embedding_dim=16,
        learn_pb=True, norm='layer', s_hidden_dim=32, policy_hidden_dim=32,
        flow_hidden_dim=32, cond_hidden_dim=32, s_layers=2, policy_layers=2, flow_layers=2,
        cond_layers=2, dplr_rank=0,
        policy_kind='set', set_policy_hidden=32, set_policy_layers=2, set_policy_corr_dim=8)
    for k in list(kw):
        if k.startswith('model__'):
            setattr(model, k[len('model__'):], kw.pop(k))
    args = SimpleNamespace(
        model=model, checkpoints_dir=str(ckdir), checkpoint_name=None,
        continue_from_checkpoint=False, load_weights_only=False, prior_model_name=None,
        compile_policy=False, embedding_conditioning=True, embedding_conditioning_dim=MOL_DIM,
        temperature_conditioning=False, molecule_conditioning=False, sg_conditioning=False,
        zp_conditioning=False, vector_conditioning=False, z_primes=[1], molecules_path=None,
        lr_policy=1e-3, lr_back=1e-3, lr_replay=1e-3, lr_fused=1e-3, lr_flow=1e-2,
        weight_decay=0.0, use_weight_decay=False, lr_control=None,
        freeze_backward_policy=False, traj_checkpoint=False, traj_checkpoint_modes=None,
        integrator=SimpleNamespace(T=T), grow_batch_size=False, batch_size=8,
        max_batch_size=8, ema_decay=None, seed=0, checkpoint_read_only=False,
        override_learning_rates=False, condition_log_z=None)
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def _modeller(energy, args, run_name='probe'):
    """A ConformerModeller carrying exactly what init_gfn and the checkpointer read."""
    m = ConformerModeller.__new__(ConformerModeller)
    m.args = args
    m.device = 'cpu'
    m.energy_function = energy
    m.run_name = run_name
    m.problem_def = {'energy_function': 'conformer', 'energy_config': {'level': 'full'}}
    m.problem_hash = 'abc123'
    m.problem_slug = 'conformer-test'
    for k, v in MODELLER_STATE_DEFAULTS.items():
        setattr(m, k, copy.deepcopy(v))
    m.metric_tracker = MetricTracker(period=25.0)
    m.grad_guard = SimpleNamespace(state_dict=lambda: None, load_state_dict=lambda s: None)
    m.protocol = SimpleNamespace(stage=SimpleNamespace(lr_sensor=None))
    m._ray_askers = lambda: []
    m.lr_controller = SimpleNamespace(announce=lambda: None)
    m.checkpointer = Checkpointer(m)
    return m


def _condition_batch(energy, n_rep=2, seed=0):
    """Carrier-padded condition graphs with random frozen embeddings, n_rep rows per member."""
    lay = energy.carrier if energy.is_carrier else CarrierLayout(energy._members)
    g = torch.Generator().manual_seed(seed)
    rows = []
    for ident, mem in energy._members.items():
        c = condition_from_energy(mem, identifier=ident)
        a, msk = free_dof_atom_index(mem)
        cc = carrier_pad_condition(c, lay, ident, mem, atoms=a, mask=msk, R=int(a.shape[1]))
        cc.atom_embedding = torch.randn(int(cc.num_nodes), ENC, generator=g)
        cc.embedding = torch.randn(1, MOL_DIM, generator=g)
        rows += [cc.__copy__() for _ in range(n_rep)]
    return collate_conditions(rows)


def _disc(b):
    return torch.linspace(0, 1, T + 1).repeat(b, 1)


def _rollouts(model, batch):
    """(fwd, bwd, replay) outputs under fixed seeds. The backward and replay legs run from
    the model's OWN forward states, so a mismatch anywhere upstream propagates."""
    n = batch.num_graphs
    cond = batch.embedding.reshape(n, -1)
    with torch.no_grad():
        torch.manual_seed(7)
        fwd = model.get_traj_fwd(torch.zeros(n, model.dim), _disc, None, cond, batch)
        torch.manual_seed(8)
        bwd = model.get_traj_bwd(fwd[0][:, -1], _disc, cond, batch)
        torch.manual_seed(9)
        rep = model.get_traj_replay(fwd[0], _disc, cond, batch)
    return fwd, bwd, rep


def _flat(outs):
    return torch.cat([t.detach().reshape(-1) for leg in outs for t in leg])


def _synthetic_step(m, seed):
    """One fused Adam step with seeded gradients on every trainable parameter."""
    g = torch.Generator().manual_seed(seed)
    opt = m.optimizers['fused']
    opt.zero_grad(set_to_none=True)
    for p in m.gfn_model.parameters():
        if p.requires_grad:
            p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype)
    opt.step()


def _perturb(model, seed, scale):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))


@pytest.fixture(scope='module')
def trained(carrier, tmp_path_factory):
    """Modeller A, built fresh through the real init_gfn, moved off its init, P_B frozen,
    two Adam steps taken, and saved through the real Checkpointer.save."""
    ckdir = tmp_path_factory.mktemp('ck')
    a = _modeller(carrier, _args(ckdir))
    a.init_gfn()
    _perturb(a.gfn_model, 1, 0.02)
    _perturb(a.ema_model, 2, 0.01)
    a.set_pb_freeze('full')
    # the live trunk moves AFTER the freeze, so P_B's snapshot and the live t/s/backward
    # modules differ -- a reload that rebuilt P_B from the live weights would then show
    _perturb(a.gfn_model.backward_policy, 3, 0.05)
    _synthetic_step(a, 11)
    _synthetic_step(a, 12)
    a.checkpointer.save('probe')
    return a, ckdir, a.checkpointer.path_for('probe')


def _load(energy, ckdir, path, run_name='reload', **kw):
    b = _modeller(energy, _args(ckdir, checkpoint_name=path.split('/')[-1], **kw),
                  run_name=run_name)
    b.init_gfn()
    return b


# ------------------------------------------------------------------ the round trip


def test_the_old_path_fails_the_strict_load(trained):
    """The SEPARATOR for everything below: without the builder seam the stored config
    rebuilds a flat GFN and the strict load refuses the set head's tensors."""
    _, _, path = trained
    ck = torch.load(path, weights_only=False)
    cfg = dict(ck['gfn_config'])
    cfg.pop('conformer')
    with pytest.raises(RuntimeError, match='forward_policy'):
        GFN(**cfg).load_state_dict(ck['model_train'])


def test_full_load_rebuilds_the_architecture(trained, carrier):
    a, ckdir, path = trained
    b = _load(carrier, ckdir, path)
    for model in (b.gfn_model, b.ema_model):
        assert type(model) is ConformerGFN
        assert model._carrier is True
        assert isinstance(model.forward_policy, RaggedConditionalSetPolicy)
    assert b.gfn_config['conformer'] == a.gfn_config['conformer']
    for mine, theirs in ((a.gfn_model, b.gfn_model), (a.ema_model, b.ema_model)):
        sa, sb = mine.state_dict(), theirs.state_dict()
        assert list(sa) == list(sb)
        assert all(torch.equal(sa[k], sb[k]) for k in sa)
    pa, pb = a.gfn_model.pb_snapshot_state(), b.gfn_model.pb_snapshot_state()
    assert pa is not None and all(torch.equal(pa[k], pb[k]) for k in pa)


def test_ema_is_distinct_but_shares_the_pb_snapshot(trained, carrier):
    """Distinct RIGHT AFTER LOAD. Under ema_decay: null the trainer's update_ema_model then
    ALIASES the two (ema_model = gfn_model) after the first step -- the canonical conformer
    configs run that way, so the aliasing is pinned too, and a decayed EMA stays distinct."""
    _, ckdir, path = trained
    b = _load(carrier, ckdir, path)
    assert b.ema_model is not b.gfn_model
    live = {p.data_ptr() for p in b.gfn_model.parameters()}
    assert not any(p.data_ptr() in live for p in b.ema_model.parameters())
    assert b.gfn_model.pb_frozen and b.ema_model._pb_frozen is b.gfn_model._pb_frozen

    b.args.ema_decay = 0.99
    b.update_ema_model()
    assert b.ema_model is not b.gfn_model
    b.args.ema_decay = None
    b.update_ema_model()
    assert b.ema_model is b.gfn_model, 'ema_decay: null aliases the EMA to the live model'


def test_rollouts_are_bitwise_equal_live_and_ema(trained, carrier):
    a, ckdir, path = trained
    b = _load(carrier, ckdir, path)
    batch = _condition_batch(carrier)
    for name, mine, theirs in (('live', a.gfn_model, b.gfn_model),
                               ('ema', a.ema_model, b.ema_model)):
        x, y = _flat(_rollouts(mine, batch)), _flat(_rollouts(theirs, batch))
        assert torch.equal(x, y), f'{name}: max|diff| {float((x - y).abs().max()):.3g}'
    # separator: live and EMA were perturbed differently, so their rollouts must differ --
    # or the equality above could be two copies of one function
    assert not torch.equal(_flat(_rollouts(b.gfn_model, batch)),
                           _flat(_rollouts(b.ema_model, batch)))


def test_adam_state_survives_and_one_step_agrees(trained, carrier):
    a, ckdir, path = trained
    b = _load(carrier, ckdir, path)
    oa, ob = a.optimizers['fused'], b.optimizers['fused']
    sa, sb = oa.state_dict(), ob.state_dict()
    assert sa['state'].keys() == sb['state'].keys() and len(sa['state']) > 0
    for k in sa['state']:
        for field, v in sa['state'][k].items():
            assert torch.equal(torch.as_tensor(v), torch.as_tensor(sb['state'][k][field])), \
                (k, field)
    steps = {float(torch.as_tensor(s['step'])) for s in sb['state'].values()}
    assert steps == {2.0}, f'Adam step counters {steps}, expected the two pre-save steps'

    a2 = copy.deepcopy(a.gfn_model.state_dict())
    _synthetic_step(a, 21)
    _synthetic_step(b, 21)
    for k, v in a.gfn_model.state_dict().items():
        assert torch.equal(v, b.gfn_model.state_dict()[k]), k
    assert any(not torch.equal(a2[k], v) for k, v in a.gfn_model.state_dict().items()), \
        'the step moved nothing -- the comparison above would be vacuous'
    # restore A for the other tests that read it (module fixture)
    a.gfn_model.load_state_dict(a2)


def test_install_after_a_reload_is_a_checked_no_op(trained, carrier):
    """RES-2: the swap must NOT run again over a loaded model -- it would reset the head,
    the EMA and the Adam moments load_full just restored."""
    _, ckdir, path = trained
    b = _load(carrier, ckdir, path)
    ids = (id(b.gfn_model), id(b.ema_model), id(b.gfn_model.forward_policy),
           id(b.optimizers['fused']), id(b.gfn_model._pb_frozen))
    before = copy.deepcopy(b.optimizers['fused'].state_dict())
    b._install_set_policy()
    assert ids == (id(b.gfn_model), id(b.ema_model), id(b.gfn_model.forward_policy),
                   id(b.optimizers['fused']), id(b.gfn_model._pb_frozen))
    after = b.optimizers['fused'].state_dict()
    for k in before['state']:
        assert torch.equal(torch.as_tensor(before['state'][k]['step']),
                           torch.as_tensor(after['state'][k]['step']))


def test_two_hops_keep_the_stamp(trained, carrier):
    """m.gfn_config must KEEP the block through a load, or the next save drops it and the
    leg after that rebuilds flat."""
    a, ckdir, path = trained
    b = _load(carrier, ckdir, path, run_name='hop1')
    b.checkpointer.save('hop')
    c = _load(carrier, ckdir, b.checkpointer.path_for('hop'), run_name='hop2')
    assert c.gfn_config['conformer'] == a.gfn_config['conformer']
    assert type(c.gfn_model) is ConformerGFN
    batch = _condition_batch(carrier)
    assert torch.equal(_flat(_rollouts(a.gfn_model, batch)),
                       _flat(_rollouts(c.gfn_model, batch)))


def test_weights_only_load_rebuilds_and_flags_the_skip(trained, carrier):
    a, ckdir, path = trained
    w = _load(carrier, ckdir, path, run_name='warm', load_weights_only=True)
    assert type(w.gfn_model) is ConformerGFN and w.gfn_model._carrier is True
    assert w.weights_only_loaded is True
    sa, sw = a.gfn_model.state_dict(), w.gfn_model.state_dict()
    assert all(torch.equal(sa[k], sw[k]) for k in sa)
    assert len(w.optimizers['fused'].state_dict()['state']) == 0, \
        'a weights-only start builds its optimizers fresh'


def test_a_first_launch_with_continue_from_checkpoint_builds_fresh(carrier, tmp_path):
    """The case the old refusal blocked: a requeue-safe config (continue_from_checkpoint
    true) on its FIRST leg, with no running file yet, must simply build the set head."""
    m = _modeller(carrier, _args(tmp_path, continue_from_checkpoint=True))
    m.init_gfn()
    assert isinstance(m.gfn_model.forward_policy, RaggedConditionalSetPolicy)
    assert m.gfn_config['conformer']['policy_kind'] == 'set'
    assert m._gfn_reloaded is False


# ------------------------------------------------------------------ refusals


def test_a_changed_set_policy_hidden_is_refused_by_name(trained, carrier):
    _, ckdir, path = trained
    with pytest.raises(ValueError, match='set_policy_hidden'):
        _load(carrier, ckdir, path, model__set_policy_hidden=48)


def test_a_flat_config_on_a_set_checkpoint_is_refused(trained, carrier):
    _, ckdir, path = trained
    with pytest.raises(ValueError, match='policy_kind'):
        _load(carrier, ckdir, path, model__policy_kind='flat')


@pytest.mark.parametrize('weights_only', [False, True])
def test_a_different_carrier_width_is_refused_naming_both(trained, weights_only):
    """A rung trained at one K handed to a run at another: both P_F and P_B are K-shaped."""
    _, ckdir, path = trained
    narrow = MultiConformerTorsions(['C', 'N'], identifiers=['C', 'N'], **KW)
    assert narrow.data_ndim == 9
    with pytest.raises(ValueError, match=r'width 12 .* width 9'):
        _load(narrow, ckdir, path, load_weights_only=weights_only)


def test_an_equal_width_different_layout_is_refused(trained, carrier, tmp_path):
    """Equal K is not the same layout: the r|theta split decides where member columns land.
    Tampered stamp, same K and same periodic columns, so only the layout check can see it."""
    _, _, path = trained
    ck = torch.load(path, weights_only=False)
    ck['gfn_config'] = dict(ck['gfn_config'])
    ck['gfn_config']['conformer'] = dict(ck['gfn_config']['conformer'], block_width=[4, 5, 3])
    bad = tmp_path / 'bad_conformer-test_probe.pt'
    torch.save(ck, bad)
    with pytest.raises(ValueError, match=r'r\|theta\|phi \[4, 5, 3\]'):
        _load(carrier, tmp_path, str(bad).replace('\\', '/'))


def test_a_flat_checkpoint_without_the_stamp_under_a_set_config_is_refused(carrier, tmp_path):
    """A flat checkpoint written before the stamp existed carries no block. Read as 'fresh',
    a set config would swap a new head over the loaded trunk; it is refused instead."""
    f = _modeller(carrier, _args(tmp_path, model__policy_kind='flat'), run_name='flat')
    f.init_gfn()
    assert f.gfn_config['conformer'] == {'policy_kind': 'flat', 'carrier': True,
                                         'block_width': [5, 4, 3]}
    f.checkpointer.save('probe')
    path = f.checkpointer.path_for('probe')
    ck = torch.load(path, weights_only=False)
    ck['gfn_config'] = {k: v for k, v in ck['gfn_config'].items() if k != 'conformer'}
    torch.save(ck, path)
    with pytest.raises(ValueError, match='no conformer block'):
        _load(carrier, tmp_path, path)
    # and the same file under a FLAT config loads, re-classed onto the carrier
    g = _load(carrier, tmp_path, path, model__policy_kind='flat')
    assert type(g.gfn_model) is ConformerGFN and g.gfn_model._carrier is True


# 'off' and 'false' are STRINGS here: train.py maybe_compile_policy reads them as
# bool(setting), i.e. True, and compiles -- so they must be refused like 'auto'.
@pytest.mark.parametrize('setting', ['auto', 'step', True, 'off', 'false'])
def test_a_compiled_set_policy_is_refused(carrier, tmp_path, setting):
    m = _modeller(carrier, _args(tmp_path, compile_policy=setting))
    with pytest.raises(ValueError, match='compile_policy'):
        m.init_gfn()


@pytest.mark.parametrize('setting', [False, None, 0])
def test_an_uncompiled_set_policy_builds(carrier, tmp_path, setting):
    _modeller(carrier, _args(tmp_path, compile_policy=setting))._refuse_compiled_set_policy()


def test_prior_model_name_is_refused_on_this_route(carrier, tmp_path):
    m = _modeller(carrier, _args(tmp_path, prior_model_name='x_prior.pt'))
    with pytest.raises(ValueError, match='prior_model_name'):
        m.init_gfn()


# ------------------------------------------------------------------ the rest of the resume


def test_scramble_is_never_applicable_on_the_set_head(trained):
    a, _, _ = trained
    assert a.gfn_model.conditional and a.gfn_model.conditions_type == 'vector', \
        'the base guard alone would say yes -- which is the hole'
    assert a.scramble_applicable() is False


def _restored_tracker(library_size):
    from energy_sampling.buffer import ConditionLogZTracker
    saved = ConditionLogZTracker(library_size=library_size, min_visits=20,
                                 half_life_visits=200.0, trim_frac=0.1,
                                 max_batch_weight=200.0, clip_beta=10.0)
    return ConditionLogZTracker.from_state_dict(saved.state_dict())


def test_a_full_resume_adopts_the_configs_tracker_settings(tmp_path, capsys):
    """TRK-7 / RES-11: from_state_dict restores the CHECKPOINT's knobs and the base method
    returns early, so a retuned half-life read as applied while the old one ran. clip_beta
    is kept: z_grad_ema is denominated in it. An unset key keeps the restored value."""
    en = MultiConformerTorsions(SMIS, identifiers=SMIS, **KW)
    en.set_n_molecules(3)                                     # what init_identifiers does
    cz = SimpleNamespace(min_visits=5, half_life_visits=50.0, trim_frac=None,
                         max_batch_weight=100.0)
    m = _modeller(en, _args(tmp_path, condition_log_z=cz))
    m.condition_log_z = _restored_tracker(3)
    m.init_condition_log_z()
    t = m.condition_log_z
    assert (t.min_visits, t.half_life_visits, t.max_batch_weight) == (5, 50.0, 100.0)
    assert t.trim_frac == 0.1 and t.clip_beta == 10.0
    assert 'half_life_visits: checkpoint 200.0 -> config 50.0' in capsys.readouterr().out


def test_a_restored_tracker_of_another_size_is_refused(tmp_path):
    en = MultiConformerTorsions(SMIS, identifiers=SMIS, **KW)
    en.set_n_molecules(3)
    m = _modeller(en, _args(tmp_path))
    m.condition_log_z = _restored_tracker(5)
    with pytest.raises(ValueError, match='5 conditions'):
        m.init_condition_log_z()


def test_the_prior_rng_resumes_mid_stream(carrier, tmp_path):
    """RES-7: the next prior draw after a resume is the one the saving process would have
    taken, not the seed's first draw again."""
    assert MODELLER_STATE_DEFAULTS['_prior_rng_state'] is None
    a = _modeller(carrier, _args(tmp_path, seed=5))
    a._prior_rng.random(17)                     # the leg's draws so far
    path = tmp_path / 'ms.pt'
    torch.save(a.checkpointer.get_state_dict(), path)
    b = _modeller(carrier, _args(tmp_path, seed=5))
    b.checkpointer.set_state_dict(torch.load(path, weights_only=False))
    fresh = _modeller(carrier, _args(tmp_path, seed=5))
    want = a._prior_rng.random(5)
    assert (b._prior_rng.random(5) == want).all()
    assert not (fresh._prior_rng.random(5) == want).all(), 'separator: a fresh leg replays'
