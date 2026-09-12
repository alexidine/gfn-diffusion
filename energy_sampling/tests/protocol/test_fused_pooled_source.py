"""fused_train_step: the pooled term's first side (fwd_loss_coeffs.pooled_source),
the live-slot clears, and the forward Z sidecar (stage.fwd_z_sidecar).

Driven on a stub Modeller with every branch step faked, so what is pinned is
the step's own wiring: which slots reach pooled_condition_vargrad, which losses
reach fused_loss and at what weight, and that nothing survives the step.
pooled_condition_vargrad is patched through the method's own globals, which is
the module the method actually reads (the energy_sampling.train / train twin
would otherwise make a module-level patch a silent no-op).
"""
from types import MethodType, SimpleNamespace

import pytest
import torch

from energy_sampling.train import Modeller


def _m(src='fwd', pooled_vg=1.0, sidecar=False, gates=(True, False, False),
       fwd_frac=0.0, replay_rows=5, drop_src=False):
    gfn = SimpleNamespace(_live_fwd='stale', _live_bwd='stale', _live_replay='stale')
    fwd_coeffs = SimpleNamespace(pooled_vg=pooled_vg, pooled_source=src, pooled_beta=40.0,
                                 pooled_ratio=0.5, pooled_bridge_only=0.0)
    if drop_src:
        del fwd_coeffs.pooled_source
    stage = SimpleNamespace(name='var_conditioning', deactivate_threshold=0.01,
                            fwd_z_sidecar=sidecar)
    m = SimpleNamespace(
        args=SimpleNamespace(controller=SimpleNamespace(deactivate_threshold=0.01,
                                                        refresh_every=10),
                             fwd_loss_coeffs=fwd_coeffs),
        protocol=SimpleNamespace(stage=stage, mode_boostable=lambda mode: True,
                                 mode_dormant=lambda mode: False),
        replay_buffer=[0] * replay_rows, gfn_model=gfn, fwd_frac=fwd_frac,
        bwd_frac=0.5, replay_frac=0.5, fused_step_count=0, seen={})
    m.w = torch.tensor(1.0, requires_grad=True)        # the forward loss's leaf

    def fwd_step(discretizer, return_exp, repeats, report_losses):
        gfn._live_fwd = {'tag': 'fwd', 'condition_id': torch.tensor([3, 4])}
        return 5.0 * m.w, 'crystal_batch', {}

    def bwd_step(discretizer, repeats, report_losses, target_cids):
        m.seen['target_cids'] = target_cids
        gfn._live_bwd = {'tag': 'bwd'}
        return torch.tensor(1.0), {}

    def replay_step(discretizer, repeats, report_losses):
        gfn._live_replay = {'tag': 'replay'}
        return torch.tensor(3.0), {}

    m._fwd_gates = lambda deact, force_refresh: gates
    m.fwd_train_step, m.bwd_train_step, m.replay_train_step = fwd_step, bwd_step, replay_step
    m.mode_repeats = lambda mode: 1
    m._stash_z_fill_logw = lambda d: None
    m._fused_grad_diag_armed = lambda: False
    m.manage_replay_buffer = lambda d, b: m.seen.setdefault('admitted', True)
    m.fused_train_step = MethodType(Modeller.fused_train_step, m)
    return m


@pytest.fixture
def pooled(monkeypatch):
    calls = []

    def fake(first, bwd, **kw):
        calls.append((first, bwd))
        if not first or not bwd:
            return None, {}
        return torch.tensor([2.0]), {'pooled_mixed_frac': torch.tensor(1.0)}

    monkeypatch.setitem(Modeller.fused_train_step.__globals__,
                        'pooled_condition_vargrad', fake)
    return calls


def _tag(slot):
    return slot['tag'] if isinstance(slot, dict) else slot


# ------------------------------------------------------------ pooled_source

def test_the_replay_source_pairs_replay_with_bwd(pooled):
    m = _m(src='replay')
    loss, _ = m.fused_train_step(None)
    assert [(_tag(a), _tag(b)) for a, b in pooled] == [('replay', 'bwd')]
    assert m._pooled_stats['pooled_source_replay'] == 1.0
    assert float(loss) == pytest.approx(2.0 + 2.0)        # mix (1+3)/2 + pooled 2


def test_the_fwd_source_is_today_s_pairing(pooled):
    m = _m(src='fwd')
    m.fused_train_step(None)
    assert [(_tag(a), _tag(b)) for a, b in pooled] == [('fwd', 'bwd')]
    assert m._pooled_stats['pooled_source_replay'] == 0.0


def test_an_absent_key_reads_as_fwd(pooled):
    """Configs without the key resolve to the code default, today's pairing."""
    m = _m(drop_src=True)
    m.fused_train_step(None)
    assert [(_tag(a), _tag(b)) for a, b in pooled] == [('fwd', 'bwd')]
    assert list(m.seen['target_cids']) == [3, 4]


def test_only_the_fwd_source_aligns_the_backward_draw(pooled):
    """Replay runs after bwd, so bwd cannot be aligned to it in this step."""
    m = _m(src='replay')
    m.fused_train_step(None)
    assert m.seen['target_cids'] is None


def test_the_source_is_published_even_when_the_term_is_inert(pooled):
    m = _m(src='replay', replay_rows=0)          # replay unavailable this step
    m.fused_train_step(None)
    assert m._pooled_stats == {'pooled_source_replay': 1.0}


def test_nothing_is_published_with_the_term_off(pooled):
    m = _m(src='replay', pooled_vg=0.0)
    m.fused_train_step(None)
    assert pooled == [] and m._pooled_stats == {}


def test_an_unknown_source_is_refused(pooled):
    with pytest.raises(ValueError, match='pooled_source'):
        _m(src='bwd').fused_train_step(None)


# ------------------------------------------------------------ the live slots

def test_a_slot_no_branch_refilled_reads_missing_not_stale(pooled):
    """Writers run between steps (z_calibration tick, ray probe). A replay branch
    that did not run this step must hand the term None, not another step's rows."""
    m = _m(src='replay', replay_rows=0)
    m.fused_train_step(None)
    assert pooled[0][0] is None


def test_every_slot_is_empty_after_the_step(pooled):
    m = _m(src='replay')
    m.fused_train_step(None)
    g = m.gfn_model
    assert g._live_fwd is None and g._live_bwd is None and g._live_replay is None


# ------------------------------------------------------------ the Z sidecar

def test_the_sidecar_adds_the_forward_loss_at_weight_one(pooled):
    m = _m(src='replay', sidecar=True, gates=(True, False, False))
    loss, subs = m.fused_train_step(None)
    assert float(loss) == pytest.approx(2.0 + 2.0 + 5.0)
    loss.backward()
    assert float(m.w.grad) == pytest.approx(5.0), 'undetached, and at weight 1'
    assert subs['fwd'][2] is False, 'fwd is not a trained branch of the mix'
    assert 'fwd' not in m._probe_weights, 'and not part of the ray composite'
    assert m._sidecar_count == 1
    assert m.seen.get('admitted'), 'the rollout still feeds replay'


def test_a_z_pin_rollout_carries_no_sidecar(pooled):
    m = _m(src='replay', sidecar=True, gates=(True, False, True))
    loss, subs = m.fused_train_step(None)
    assert float(loss) == pytest.approx(4.0)
    assert not subs['fwd'][0].requires_grad
    assert getattr(m, '_sidecar_count', 0) == 0


def test_off_cadence_steps_are_untouched(pooled):
    m = _m(src='replay', sidecar=True, gates=(False, False, False))
    loss, subs = m.fused_train_step(None)
    assert 'fwd' not in subs and float(loss) == pytest.approx(4.0)


def test_without_the_key_a_zero_weight_rollout_stays_detached(pooled):
    m = _m(src='replay', sidecar=False, gates=(True, False, False))
    loss, subs = m.fused_train_step(None)
    assert float(loss) == pytest.approx(4.0) and not subs['fwd'][0].requires_grad


def test_a_weighted_forward_branch_with_the_sidecar_is_refused(pooled):
    """The same loss would enter the frac mix and the sidecar both."""
    m = _m(src='replay', sidecar=True, gates=(True, True, False), fwd_frac=0.2)
    with pytest.raises(RuntimeError, match='fwd_z_sidecar'):
        m.fused_train_step(None)
