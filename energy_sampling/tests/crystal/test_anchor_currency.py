"""
The ANCHOR CURRENCY: every anchor decision compares E_anchor (the training total
at lambda=1) against the per-condition minimum kept in the same currency
(ConditionLogZTracker.best_energy_phys), and every update_best_energy call
supplies it.

THE BUG THIS PINS (B3). AnchorBuffer.energy was frozen at each row's
admission-time lambda while the minimum it was trimmed against was the live
MIXED best_energy -- a running minimum over every lambda seen. On a flow run the
two are different quantities, so thin() evicted structures that were good in the
only currency that is monotone. And the physical stream it should have read was
never written: the five update_best_energy sites passed no energy_phys, so a run
carrying a prior_flow died at its first 2-argument call (the anchor seed).

WHAT MUST HOLD:
  lambda-free    E_anchor IS the site's own energy tensor and energy_phys is None,
                 so every anchor site is bit-identical to before (the round trip
                 -log_r * T is not bit-identical to the leg sum, and on the
                 energy_clip branch the leg sum double-counts bounding).
  flow           E_anchor = physical_energy + bounding_energy * bounding_coeff,
                 bitwise the lambda=1 total at any lambda.

    pytest tests/crystal/test_anchor_currency.py
"""
import ast
import importlib.util
import inspect
import math
import os
import sys
import textwrap
import types
from types import SimpleNamespace

import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling.buffer import (  # noqa: E402
    ANCHOR_ENERGY_CURRENCY, AnchorBuffer, BufferCurrencyError, ConditionLogZTracker)
from energy_sampling.train import Modeller  # noqa: E402

CPU = torch.device('cpu')
FLOW = object()          # any non-None prior_flow: only its presence is read


def _load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(__file__), filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bind(obj, *names):
    """Attach real Modeller methods to a stub, so the method under test reaches
    the helpers it calls through `self`."""
    for name in names:
        setattr(obj, name, types.MethodType(getattr(Modeller, name), obj))
    return obj


class FakeBatch:
    """Row-aligned columns with the four batch operations the anchor sites use."""
    _store = {}                      # AnchorBuffer._drop_keys reads it
    device = CPU

    def __init__(self, **cols):
        self.__dict__.update(cols)

    def clone(self):
        return FakeBatch(**{k: v.clone() for k, v in self.__dict__.items()})

    def to(self, device):
        return self

    def cpu(self):
        return self

    def subsample_new_batch(self, idx):
        return FakeBatch(**{k: v[idx] for k, v in self.__dict__.items()})


def tracker(n=4, **kw):
    return ConditionLogZTracker(library_size=n, **kw)


def stub(flow=True, coeff=2.0):
    m = SimpleNamespace(energy_function=SimpleNamespace(
        prior_flow=FLOW if flow else None, bounding_coeff=coeff))
    return bind(m, '_anchor_energy_phys', '_anchor_energy')


PHYS = torch.tensor([10.0, -4.0, 7.5])
BND = torch.tensor([0.0, 0.5, 2.0])


# ------------------------------------------------------------ the helper itself

def test_flow_run_reads_the_lambda1_total_off_the_legs():
    m = stub(flow=True, coeff=2.0)
    b = SimpleNamespace(physical_energy=PHYS.clone(), bounding_energy=BND.clone())
    energy = torch.tensor([1.0, 2.0, 3.0])            # the mixture: never used
    e_anchor, energy_phys = m._anchor_energy(b, energy)
    assert torch.equal(energy_phys, PHYS + BND * 2.0)
    assert torch.equal(e_anchor, PHYS + BND * 2.0)


def test_lambda_free_run_hands_back_the_sites_own_tensor():
    """Bit-identity by construction: the SAME object, not an equal one, and no
    physical leg, so the tracker keeps its alias."""
    m = stub(flow=False)
    energy = torch.tensor([1.0, 2.0, 3.0])
    e_anchor, energy_phys = m._anchor_energy(SimpleNamespace(), energy)
    assert e_anchor is energy
    assert energy_phys is None


def test_rows_without_legs_raise_on_a_flow_run():
    m = stub(flow=True)
    with pytest.raises(AttributeError, match='physical_energy'):
        m._anchor_energy(SimpleNamespace(physical_energy=PHYS), torch.zeros(3))
    with pytest.raises(ValueError, match='scored batch'):
        m._anchor_energy_phys(None)


def test_a_row_count_mismatch_raises():
    m = stub(flow=True)
    b = SimpleNamespace(physical_energy=PHYS, bounding_energy=BND)
    with pytest.raises(ValueError, match='same rows'):
        m._anchor_energy(b, torch.zeros(5))


# ------------------------------------------- against the real energy function

L = _load_sibling('lambda_mix_helpers', 'test_lambda_mix.py')
BC, T = 10.0, 2.5


def _ef(path=None, lam=1.0):
    kw = dict(lambda_mix=lam)
    if path is not None:
        kw['prior_flow_path'] = path
    from energy_sampling.energies.molecular_crystal import MolecularCrystal
    return MolecularCrystal(
        device=CPU, energy_function='latent_gaussian', space_groups=[L.SG], z_primes=(1,),
        temperature=T, bounding_coeff=BC, reduction_coeff=1.0, reward_range=None,
        internal_oom_recovery=False, host_gas_phase_reference=False,
        analyze_kwargs={'c': [0.5] * L.DIM, 'width': 0.4}, **kw)


def _outside_the_box(n, seed=3):
    g = torch.Generator().manual_seed(seed)
    return 1.6 * (2.0 * torch.rand(n, L.DIM, generator=g) - 1.0)   # bounding LIVE


@pytest.fixture(scope='module')
def flow_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp('flow') / 'flow.pt')
    L.fitted_flow(path)
    return path


def test_e_anchor_is_bitwise_the_lambda1_total_at_any_lambda(flow_path):
    n = 32
    x = _outside_the_box(n)
    temps = torch.full((n,), T)
    mixed_total, mixed_batch = _ef(flow_path, lam=0.3).analyze_crystal_batch(
        x, L.mol_batch(n), temperature=temps, return_batch=True)
    one_total, _ = _ef(flow_path, lam=1.0).analyze_crystal_batch(
        x, L.mol_batch(n), temperature=temps, return_batch=True)
    assert int((mixed_batch.bounding_energy != 0).sum()) > 0, 'precondition: bounding live'
    m = bind(SimpleNamespace(energy_function=_ef(flow_path, lam=0.3)),
             '_anchor_energy_phys', '_anchor_energy')
    e_anchor, _ = m._anchor_energy(mixed_batch, mixed_total.detach().cpu())
    assert torch.equal(e_anchor, one_total.detach().cpu()), \
        'E_anchor off a lambda=0.3 batch is not the lambda=1 total'
    assert not torch.allclose(e_anchor, mixed_total.detach().cpu(), atol=1e-3), \
        'precondition: the mixture differs, or this test proves nothing'


def test_the_leg_formula_would_double_count_bounding_on_the_clip_branch():
    """Why a lambda-free run does NOT use the formula: the clip seals bounding
    inside physical_energy and still publishes bounding_energy."""
    n = 32
    x = _outside_the_box(n)
    ef = _ef()
    ef.reward_range = 5.0
    ef.set_reward_clip([0.0, 1.0, 2.0])
    total, cb = ef.analyze_crystal_batch(x, L.mol_batch(n), temperature=torch.full((n,), T),
                                         return_batch=True)
    naive = cb.physical_energy + cb.bounding_energy * BC
    assert not torch.allclose(naive, total.detach().cpu(), atol=1e-3)
    m = bind(SimpleNamespace(energy_function=ef), '_anchor_energy_phys', '_anchor_energy')
    energy = total.detach().cpu()
    assert m._anchor_energy(cb, energy)[0] is energy


# --------------------------------------------------------- tracker arming

def _arm_stub(t, flow):
    return bind(SimpleNamespace(condition_log_z=t, energy_function=SimpleNamespace(
        prior_flow=FLOW if flow else None)), '_arm_phys_energy_guard')


def test_arming_follows_the_flow():
    for flow in (True, False):
        m = _arm_stub(tracker(), flow)
        m._arm_phys_energy_guard()
        assert m.condition_log_z.requires_phys_energy is flow


def test_a_full_resume_re_arms_the_guard():
    """load_full restores the tracker through from_state_dict (guard OFF), then
    init_condition_log_z used to return early before arming -- so a resumed flow
    run could write mixed energies into best_energy_phys with no refusal."""
    t = tracker()
    t.update_best_energy(torch.tensor([0]), torch.tensor([1.0]), energy_phys=torch.tensor([2.0]))
    restored = ConditionLogZTracker.from_state_dict(t.state_dict())
    assert restored.requires_phys_energy is False
    m = _arm_stub(restored, flow=True)
    Modeller.init_condition_log_z(m)
    assert m.condition_log_z.requires_phys_energy is True


def test_an_unvouched_alias_is_refused_on_a_flow_run():
    """Finite minima under phys_is_alias=True cannot have come from a flow run
    whose updates all pass energy_phys -- they may be MIXED."""
    t = tracker()
    t.update_best_energy(torch.tensor([0]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match='weights-only'):
        _arm_stub(t, flow=True)._arm_phys_energy_guard()
    _arm_stub(t, flow=False)._arm_phys_energy_guard()      # lambda-free: unchanged
    assert t.requires_phys_energy is False


def _legacy_alias():
    """A pre-currency flow tracker as from_state_dict restores it: finite minima,
    phys_is_alias True, guard off."""
    t = tracker()
    t.update_best_energy(torch.tensor([0]), torch.tensor([1.0]))
    return ConditionLogZTracker.from_state_dict(t.state_dict())


def test_one_physical_update_opens_the_window_the_early_arm_closes():
    """Why the guard runs straight after load_full. train() calls
    grow_prior_buffer before init_condition_log_z, and its update_best_energy
    passes energy_phys, which clears the alias -- after that the refusal passes
    and the legacy minima stay in best_energy_phys."""
    t = _legacy_alias()
    t.update_best_energy(torch.tensor([1]), torch.tensor([5.0]), energy_phys=torch.tensor([5.0]))
    _arm_stub(t, flow=True)._arm_phys_energy_guard()       # no longer refused
    assert torch.isfinite(t.best_energy_phys[0])


def _resume_stub(t, flow, auto):
    ckpt = SimpleNamespace(find_matching=lambda kind: 'running.pt')
    m = SimpleNamespace(
        args=SimpleNamespace(checkpoint_name=None if auto else 'x.pt', checkpoints_dir='d',
                             load_weights_only=False, continue_from_checkpoint=auto),
        checkpointer=ckpt, energy_function=SimpleNamespace(prior_flow=FLOW if flow else None))
    ckpt.load_full = lambda path: setattr(m, 'condition_log_z', t)
    return bind(m, 'init_gfn', '_arm_restored_tracker', '_arm_phys_energy_guard')


@pytest.mark.parametrize('auto', [False, True], ids=['checkpoint_name', 'auto_resume'])
def test_a_full_resume_is_judged_before_anything_writes_the_tracker(auto):
    """Both load_full paths in init_gfn refuse the legacy alias at once -- the
    refusal fires inside init_gfn, before any stub for the rest of it is reached."""
    with pytest.raises(ValueError, match='weights-only'):
        _resume_stub(_legacy_alias(), flow=True, auto=auto).init_gfn()


def test_the_restored_tracker_arm_needs_a_tracker():
    m = bind(SimpleNamespace(energy_function=SimpleNamespace(prior_flow=FLOW)),
             '_arm_restored_tracker', '_arm_phys_energy_guard')
    m._arm_restored_tracker()                                # checkpoint without one: no-op
    m.condition_log_z = tracker()
    m._arm_restored_tracker()
    assert m.condition_log_z.requires_phys_energy is True


# -------------------------------------------------- AnchorBuffer currency stamp

H = _load_sibling('lj_stamp_helpers', 'test_lj_stamp_buffer_restore.py')


def _anchors(energy, cid):
    _, built = H.built_batch(H.MIPCAS)
    n = built.num_graphs
    energy = torch.as_tensor(energy, dtype=torch.float32)[:n]
    built.add_graph_attr(torch.as_tensor(cid, dtype=torch.long)[:n], 'condition_id')
    return AnchorBuffer(built, device=CPU, reward=torch.zeros(n), energy=energy)


def test_a_new_store_is_stamped_and_the_stamp_survives_disk():
    a = _anchors([1.0, 2.0], [0, 0])
    assert a.energy_currency == ANCHOR_ENERGY_CURRENCY
    back = AnchorBuffer.from_state_dict(H.pickle_round_trip(a.state_dict()), device=CPU)
    assert back.energy_currency == ANCHOR_ENERGY_CURRENCY


def test_an_unstamped_store_stays_unstamped():
    state = _anchors([1.0, 2.0], [0, 0]).state_dict()
    del state['energy_currency']
    back = AnchorBuffer.from_state_dict(H.pickle_round_trip(state), device=CPU)
    assert back.energy_currency is None
    assert back.state_dict()['energy_currency'] is None, \
        'a re-save must not launder rows nothing vouches for'


def _policy_stub(buf, flow):
    return SimpleNamespace(anchor_buffer=buf,
                           energy_function=SimpleNamespace(prior_flow=FLOW if flow else None),
                           args=SimpleNamespace(buffers=SimpleNamespace(anchor_buffer=SimpleNamespace(
                               frozen=False, refresh_every_n_evals=3))))


class _Buf:
    def __init__(self, currency):
        self.energy_currency = currency
        self.ema_loss = torch.ones(2)
        self.birth_loss = torch.ones(2)

    def __len__(self):
        return 2


def test_a_flow_run_refuses_an_unstamped_store():
    buf = _Buf(None)
    with pytest.raises(BufferCurrencyError, match='energy_currency'):
        Modeller.apply_anchor_buffer_policy(_policy_stub(buf, flow=True), 'sidecar restore')
    Modeller.apply_anchor_buffer_policy(_policy_stub(buf, flow=False), 'sidecar restore')
    buf.energy_currency = ANCHOR_ENERGY_CURRENCY
    Modeller.apply_anchor_buffer_policy(_policy_stub(buf, flow=True), 'sidecar restore')


def test_thin_keeps_anchors_that_are_good_in_the_anchor_currency():
    """B3 in miniature. Rows sit within the window of the PHYSICAL minimum; the
    mixed minimum (lower, a running min over every lambda) would evict them all."""
    t = tracker()
    t.update_best_energy(torch.tensor([0]), torch.tensor([-500.0]),
                         energy_phys=torch.tensor([10.0]))
    a = _anchors([15.0, 30.0], [0, 0])
    n = len(a)
    a.thin(t.best_energy_phys, energy_window=50.0)
    assert len(a) == n
    a.thin(t.best_energy, energy_window=50.0)
    assert len(a) == 0, 'precondition: the mixed minimum evicts them'


# ------------------------------------------------------------- the call sites

def _modeller_calls(attr):
    tree = ast.parse(textwrap.dedent(inspect.getsource(Modeller)))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == attr]


def test_every_update_best_energy_call_supplies_the_physical_leg():
    calls = _modeller_calls('update_best_energy')
    assert len(calls) == 4, f'expected the four trainer sites, found {len(calls)}'
    bare = [c.lineno for c in calls if 'energy_phys' not in {k.arg for k in c.keywords}]
    assert not bare, f'update_best_energy without energy_phys at Modeller lines {bare}'


def test_every_thin_call_reads_the_physical_minimum():
    calls = _modeller_calls('thin')
    assert len(calls) == 3, f'expected three thin sites, found {len(calls)}'
    wrong = [c.lineno for c in calls
             if not (isinstance(c.args[0], ast.Attribute) and c.args[0].attr == 'best_energy_phys')]
    assert not wrong, f'thin() against a non-anchor-currency minimum at Modeller lines {wrong}'


def test_the_forward_step_hands_the_loss_the_anchor_reader():
    tree = ast.parse(textwrap.dedent(inspect.getsource(Modeller.fwd_train_step)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, 'id', None) == 'get_gfn_forward_loss']
    assert calls and all('anchor_energy_fn' in {k.arg for k in c.keywords} for c in calls)


# ------------------------------------------- behaviour at the anchor sites

def test_the_anchor_seed_writes_and_stores_the_anchor_currency():
    """The startup crash: on a flow run the seed's 2-argument update raised."""
    n = 3
    seed = FakeBatch(mol_id=torch.zeros(n, dtype=torch.long),
                     physical_energy=torch.full((n,), 999.0),     # stale, admission-time
                     bounding_energy=torch.zeros(n))
    cid = torch.tensor([0, 0, 1])
    live_phys, live_bnd = torch.tensor([5.0, 3.0, 8.0]), torch.tensor([0.0, 1.0, 0.0])
    reward = torch.tensor([-1.0, -2.0, -3.0])           # T = 1: mixture = 1, 2, 3
    made = {}

    def prebuilt(batch, temperature, return_ens_dict=False):
        ens = dict(physical_energy=live_phys, bounding_energy=live_bnd,
                   flow_energy=torch.zeros(n))
        return (reward, ens) if return_ens_dict else reward

    def anchor_cls(batch, device, reward, energy, exclude_keys=None, **kw):
        made['energy'] = energy
        return SimpleNamespace()

    t = tracker()
    t.requires_phys_energy = True
    m = SimpleNamespace(
        condition_log_z=t, device=CPU, buffer_device=CPU,
        args=SimpleNamespace(buffers=SimpleNamespace(anchor_buffer=SimpleNamespace(
            seed_source='prior_dataset'))),
        prior_dataset=SimpleNamespace(batch=seed),
        energy_function=SimpleNamespace(
            prior_flow=FLOW, bounding_coeff=2.0, prebuilt_sample_to_reward=prebuilt,
            condition_samples=lambda b, **kw: (b, torch.zeros(n), None, cid)),
        anchor_buffer_cls=anchor_cls, _buffer_kwargs=lambda: {},
        apply_anchor_buffer_policy=lambda source: None)
    bind(m, '_anchor_energy_phys', '_anchor_energy')
    Modeller.init_anchor_buffer_seed(m)
    want = live_phys + live_bnd * 2.0                    # 5, 5, 8 -- not the stale 999
    assert torch.equal(made['energy'], want)
    assert not t.phys_is_alias
    assert t.best_energy_phys[:2].tolist() == [5.0, 8.0]
    assert t.best_energy[:2].tolist() == [1.0, 3.0]


def _screen_stub(t, sample, flow):
    cfg = SimpleNamespace(health_gate_floor=None, health_gate_ceiling=None,
                          screen_energy_window=1.0, surprise_cutoff=0.0, confirm_cutoff=0.0,
                          confirm_k=2, dup_cutoff=0.0, max_size=10 ** 6)
    made = {}

    def anchor_cls(batch, device, reward, energy, original_surprise=None,
                   exclude_keys=None, **kw):
        made['energy'] = energy
        return [0] * len(energy)

    def bwd(latents, disc, cond, batch):
        z = torch.zeros(latents.shape[0], 3)
        return None, z, z, None

    m = SimpleNamespace(
        condition_log_z=t, device=CPU, buffer_device=CPU,
        args=SimpleNamespace(buffers=SimpleNamespace(anchor_buffer=cfg), eval_T=10),
        metric_tracker=SimpleNamespace(get=lambda *a: None),
        energy_function=SimpleNamespace(prior_flow=FLOW if flow else None, bounding_coeff=1.0),
        ema_model=SimpleNamespace(get_traj_bwd=bwd),
        anchor_buffer_cls=anchor_cls, _buffer_kwargs=lambda: {},
        _batch_latents=lambda b: torch.zeros(b.condition_id.shape[0], 4),
        apply_anchor_buffer_policy=lambda source: None)
    bind(m, '_anchor_energy_phys', '_anchor_energy')
    return m, made


def _screen_rows():
    # row A: mixed 0.5 (inside the MIXED window), E_anchor 150 (outside the physical one)
    # row B: mixed 50  (outside the mixed window), E_anchor 100.5 (inside the physical one)
    return FakeBatch(physical_energy=torch.tensor([150.0, 100.5]),
                     bounding_energy=torch.zeros(2),
                     condition_id=torch.tensor([0, 0]), conditions=torch.zeros(2, 3))


def _warm(t):
    t.update(torch.tensor([0]), torch.tensor([0.0]), step=0)       # log Z(c) axis warm


def test_the_screen_window_is_judged_in_the_anchor_currency():
    t = tracker(min_visits=1)
    _warm(t)
    t.update_best_energy(torch.tensor([0]), torch.tensor([0.0]), energy_phys=torch.tensor([100.0]))
    m, made = _screen_stub(t, _screen_rows(), flow=True)
    Modeller.screen_and_admit_anchors(m, _screen_rows(), torch.full((2,), 100.0),
                                      torch.tensor([0.5, 50.0]), torch.zeros(2))
    assert made['energy'].tolist() == [100.5], 'admitted on the mixture, or stored it'


def test_the_screen_is_unchanged_on_a_lambda_free_run():
    t = tracker(min_visits=1)
    _warm(t)
    t.update_best_energy(torch.tensor([0]), torch.tensor([0.0]))
    m, made = _screen_stub(t, _screen_rows(), flow=False)
    Modeller.screen_and_admit_anchors(m, _screen_rows(), torch.full((2,), 100.0),
                                      torch.tensor([0.5, 50.0]), torch.zeros(2))
    assert made['energy'].tolist() == [0.5]


class _Anchors:
    def __init__(self, batch):
        self.batch, self.admitted = batch, []
        self.original_surprise = torch.zeros(len(batch.condition_id))

    def __len__(self):
        return len(self.batch.condition_id)

    def sample_graphs(self, n, **kw):
        return self.batch, torch.arange(n), None

    def admit(self, batch, reward, energy, **kw):
        self.admitted.append(energy)
        return len(energy)


def test_top_up_record_breakers_are_judged_in_the_anchor_currency():
    # child 0: mixed -50 (NOT below the mixed min -100), E_anchor 40 (below the physical 50)
    # child 1: mixed -150 (below the mixed min), E_anchor 60 (NOT below the physical 50)
    children = FakeBatch(physical_energy=torch.tensor([40.0, 60.0]), bounding_energy=torch.zeros(2),
                         condition_id=torch.tensor([0, 0]))
    t = tracker()
    t.update_best_energy(torch.tensor([0]), torch.tensor([-100.0]), energy_phys=torch.tensor([50.0]))
    t.requires_phys_energy = True
    anchors = _Anchors(children)
    cfg = SimpleNamespace(replay_beta=0.0, noise_log_range=(0.0, 0.0), dup_cutoff=0.0,
                          topup_admit_record_breakers=True, max_size=10 ** 6)
    m = SimpleNamespace(
        condition_log_z=t, device=CPU, anchor_buffer=anchors, prior_churn={'from_anchors': 0},
        args=SimpleNamespace(buffers=SimpleNamespace(
            anchor_buffer=cfg, prior_buffer=SimpleNamespace(reward_min=math.inf))),
        energy_function=SimpleNamespace(
            prior_flow=FLOW, bounding_coeff=1.0,
            log_reward=lambda x, b, logT, **kw: (torch.tensor([50.0, 150.0]), b)),
        _noise_and_condition=lambda b, r: (b, torch.zeros(2), None, b.condition_id),
        _batch_latents=lambda b: torch.zeros(2, 4),
        _condition_energy_floor=lambda cid: None)
    bind(m, '_anchor_energy_phys', '_anchor_energy')
    Modeller.top_up_prior_from_anchors(m, 2)
    assert len(anchors.admitted) == 1 and anchors.admitted[0].tolist() == [40.0]
    assert float(t.best_energy_phys[0]) == 40.0 and float(t.best_energy[0]) == -150.0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
