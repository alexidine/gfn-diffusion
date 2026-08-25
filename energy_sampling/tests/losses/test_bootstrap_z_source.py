"""`bootstrap_z` must anchor log_Z to the FORWARD Jensen, not the tracker's ema_logw.

log_Z's fixed point in the TB stages this action opens is E[log w] under FORWARD
samples -- `fwd` is the only branch without freeze_z there -- so the forward
reading is the fixed point itself and any other anchor only sets how far the
scalar has to travel to reach it.

`ema_logw` is the same estimand ONLY when the same sampler feeds it. The tracker
is updated by whichever branches ran, so after a bwd-only MLE phase 1 it reports
the ANCHOR level, and anchoring there hands the next stage the whole
anchor-to-policy gap as an opening transient. It stays as the fallback for a
transition that fires without an eval stream.
"""
import types

import pytest
import torch

from protocol import StageProtocol


def _stub(counts, logws, min_visits=1, conditional=False, full_flow=False):
    flow = types.SimpleNamespace(scalar=torch.zeros(1))
    ema_flow = types.SimpleNamespace(scalar=torch.zeros(1))
    m = types.SimpleNamespace(
        gfn_model=types.SimpleNamespace(full_flow=full_flow, conditional=conditional,
                                        flow_model=flow),
        ema_model=types.SimpleNamespace(flow_model=ema_flow),
        condition_log_z=types.SimpleNamespace(
            count=torch.tensor(counts, dtype=torch.long),
            ema_logw=torch.tensor(logws, dtype=torch.float64),
            min_visits=min_visits),
    )
    return types.SimpleNamespace(m=m), m


def test_prefers_forward_jensen_over_a_visited_tracker():
    """The whole point: eval_fwd/jensen_z WINS over a fully visited tracker.

    The two levels are far apart on purpose -- that separation is exactly the
    bwd-only-phase-1 case, where the tracker reports the anchor level and the
    forward reading reports where TB is actually going to take the scalar.
    """
    proto, m = _stub([50], [46.1])
    StageProtocol._bootstrap_z(proto, {'eval_fwd/jensen_z': -29.3})
    assert m.gfn_model.flow_model.scalar.item() == pytest.approx(-29.3, abs=1e-3)
    assert m.ema_model.flow_model.scalar.item() == pytest.approx(-29.3, abs=1e-3)


def test_falls_back_to_the_tracker_when_there_is_no_eval_stream():
    """A transition that did not fire at an eval must still bootstrap."""
    proto, m = _stub([50], [-239.4])
    StageProtocol._bootstrap_z(proto, {'eval/wass_debiased': 0.01})
    assert m.gfn_model.flow_model.scalar.item() == pytest.approx(-239.4, abs=1e-3)


def test_unvisited_slots_are_excluded_from_the_fallback_mean():
    """NaN slots must be dropped, not averaged in -- a NaN anchor is unrecoverable."""
    proto, m = _stub([50, 0, 50], [-200.0, float('nan'), -300.0], min_visits=1)
    StageProtocol._bootstrap_z(proto, None)
    got = m.gfn_model.flow_model.scalar.item()
    assert got == pytest.approx(-250.0, abs=1e-3), got
    assert got == got, 'anchored on NaN'


def test_a_cold_tracker_does_not_block_the_forward_anchor():
    """The tracker is only the fallback, so its being cold is not an error."""
    proto, m = _stub([0], [float('nan')], min_visits=5)
    StageProtocol._bootstrap_z(proto, {'eval_fwd/jensen_z': -8202.7})
    assert m.gfn_model.flow_model.scalar.item() == pytest.approx(-8202.7, abs=1e-2)


def test_raises_only_when_neither_source_exists():
    proto, m = _stub([0], [float('nan')], min_visits=5)
    with pytest.raises(RuntimeError):
        StageProtocol._bootstrap_z(proto, None)


def test_conditional_route_is_untouched():
    """The conditional branch must still delegate to bootstrap_log_z."""
    proto, m = _stub([50], [-239.4], conditional=True)
    called = {}
    m.bootstrap_log_z = lambda train_conditioner=False: called.setdefault('hit', train_conditioner)
    StageProtocol._bootstrap_z(proto, {'eval_fwd/jensen_z': -8202.7})
    assert called == {'hit': False}
    assert m.gfn_model.flow_model.scalar.item() == 0.0, 'scalar path ran on a conditional model'
