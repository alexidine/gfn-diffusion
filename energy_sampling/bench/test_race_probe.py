"""
Unit tests for the TRAINER-SIDE half of Replay Racing (`lr_race_probe.py`).

`bench/test_lr_race.py` gates the decision layer. This file gates the parts that
decide WHAT EVIDENCE that layer is handed -- the larder's partition and the
triggers -- because a rule with perfect error properties still lies if the
records it is fed are not what they claim to be.

Each test names the failure it exists to catch, not the function it calls.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lr_race_probe import Harvested, RaceLarder, RaceProbe  # noqa: E402


def _rec(branch, i):
    """A larder row identified by `i`, so a partition can be checked by identity."""
    return Harvested(branch=branch, condition=None, condition_id=None,
                     log_r=i, mol_batch=None, traj=i, repeats=1)


def _fill(larder, branch, n, start=0):
    for i in range(start, start + n):
        larder.record(branch, _rec(branch, i))


# ------------------------------------------------------------------- larder

def test_replicates_train_on_disjoint_batches():
    """The sign test counts replicates as independent. Two replicates sharing
    training batches would share those batches' luck, so `n` would be a fiction
    and the test would report confidence it has not earned."""
    lar = RaceLarder(depth=100)
    _fill(lar, 'bwd', 40)
    deal = lar.deal(('bwd',), n_train_sets=5, window=4, n_hold=6)
    sets = deal['bwd']['sets']
    assert len(sets) == 5 and all(len(s) == 4 for s in sets)
    ids = [{r.traj for r in s} for s in sets]
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            assert not (ids[a] & ids[b]), f'replicates {a},{b} share batches'


def test_holdout_is_never_trained_on():
    """The held-out slice is the ruler. A batch in both the ruler and an arm's
    training set measures that arm on its own homework, and the hot arm --
    which memorises fastest -- gains the most from that."""
    lar = RaceLarder(depth=100)
    _fill(lar, 'bwd', 40)
    deal = lar.deal(('bwd',), n_train_sets=5, window=4, n_hold=6)
    hold = {r.traj for r in deal['bwd']['hold']}
    assert len(hold) == 6
    for s in deal['bwd']['sets']:
        assert not (hold & {r.traj for r in s})


def test_a_short_larder_refuses_to_deal():
    """Refuse rather than deal short: a silently truncated partition runs the
    race on fewer replicates than the decision layer is told it has."""
    lar = RaceLarder(depth=100)
    _fill(lar, 'bwd', 25)
    assert lar.deal(('bwd',), n_train_sets=5, window=4, n_hold=6) is None
    _fill(lar, 'bwd', 1, start=25)
    assert lar.deal(('bwd',), n_train_sets=5, window=4, n_hold=6) is not None


def test_every_branch_must_be_dealable_not_just_one():
    """A fused stage composes branches into ONE loss. If bwd could deal and
    replay could not, the arms would train on a different objective from the
    one the stage actually runs."""
    lar = RaceLarder(depth=100)
    _fill(lar, 'bwd', 40)
    _fill(lar, 'replay', 10)
    assert lar.deal(('bwd', 'replay'), n_train_sets=5, window=4, n_hold=6) is None


def test_ring_forgets_oldest_first():
    """The larder is a ring so a race trains on RECENT data. Unbounded growth
    would leak, and would let a race score a rate against a policy long gone."""
    lar = RaceLarder(depth=8)
    _fill(lar, 'bwd', 20)
    assert [r.traj for r in lar.rings['bwd']] == list(range(12, 20))
    assert lar.count('bwd') == 8


def test_branches_reports_only_populated_rings():
    lar = RaceLarder(depth=8)
    _fill(lar, 'bwd', 3)
    assert lar.branches() == ('bwd',)
    assert not lar.ready(('bwd',), 4) and lar.ready(('bwd',), 3)


def test_ready_is_false_when_nothing_was_harvested():
    """`all()` over an empty branch tuple is True, so without the explicit
    guard an un-harvested stage would report ready and race on nothing."""
    assert not RaceLarder(depth=8).ready((), 1)


# -------------------------------------------------------------- host offload

class _FakeBatch:
    """Stands in for a PyG Data: `.cpu()` MUTATES and returns self."""

    def __init__(self, where='cuda'):
        self.where = where

    def cpu(self):
        self.where = 'cpu'          # in place, exactly like PyG
        return self


def test_harvest_does_not_drag_the_live_batch_off_the_device():
    """PyG's `.cpu()`/`.to()` rewrite the store in place, so a bare
    `mol_batch.cpu()` would move the batch the LIVE step is still training on.
    Delete the `copy.copy` in `_to_host` and this test must fail."""
    live = _FakeBatch('cuda')
    parked = RaceProbe._to_host(live)
    assert live.where == 'cuda', 'harvest mutated the live batch'
    assert parked.where == 'cpu' and parked is not live


def test_host_offload_passes_none_through():
    assert RaceProbe._to_host(None) is None


# ----------------------------------------------------------------- triggers

class _FakeStage:
    def __init__(self, name, train_mode='bwd'):
        self.name, self.train_mode = name, train_mode


class _FakeModeller:
    def __init__(self, stage='train_prior'):
        self.protocol = types.SimpleNamespace(stage=_FakeStage(stage))
        self.step_ind = 0
        self.fwd_frac, self.bwd_frac, self.replay_frac = 0.0, 1.0, 0.0
        self.lr_ctrl = {'peak_scale': 1.0}
        self.lr_controller = None       # -> _ramping() False


def _probe(**kw):
    p = RaceProbe(_FakeModeller(), verbose=False, **kw)
    p._guarded = lambda entry, why: {'entry': entry, 'why': why}   # do not race
    return p


def test_a_stage_change_arms_the_entry_race_but_does_not_fire_it():
    """Firing AT the transition would measure the most hostile instant in the
    run -- optimizers just rebuilt, clip guard deliberately uncalibrated -- and
    at maximum cost, since the z-cal transient is ~12x there."""
    p = _probe()
    assert p.tick() is None and p._armed_entry
    p.m.step_ind = 1
    assert p.tick() == {'entry': True, 'why': 'stage_entry'}


def test_the_entry_race_defers_while_the_ramp_is_moving():
    """A rate measured through a deliberately-suppressed envelope is not the
    rate the run will keep. Deferred, not dropped: it fires at ramp end."""
    p = _probe()
    p._ramping = lambda: True
    p.tick()
    for step in (1, 2, 3):
        p.m.step_ind = step
        assert p.tick() is None
    assert p._armed_entry, 'the ramp dropped the entry race instead of deferring it'
    p._ramping = lambda: False
    p.m.step_ind = 4
    assert p.tick() == {'entry': True, 'why': 'stage_entry'}


def test_each_clock_milestone_fires_at_most_once():
    p = _probe()
    p.tick()
    p.m.step_ind = 1
    p.tick()                                        # consume the entry race
    p.m.step_ind = 600
    assert p.tick() == {'entry': False, 'why': 'clock'}
    p.m.step_ind = 601
    assert p.tick() is None, 'the same milestone fired twice'
    p.m.step_ind = 1600
    assert p.tick() == {'entry': False, 'why': 'clock'}


def test_a_composition_change_triggers_a_race():
    """The loss mixture is the one non-stationarity the trainer KNOWS about, so
    it is a trigger rather than something a detector has to infer."""
    p = _probe()
    p.tick()
    p.m.step_ind = 1
    p.tick()
    p.m.step_ind = 2
    assert p.tick() is None
    p.m.bwd_frac, p.m.replay_frac = 0.5, 0.5        # composition moved
    p.m.step_ind = 3
    assert p.tick() == {'entry': False, 'why': 'composition'}


def test_a_transition_resets_the_clock_to_stage_relative():
    """The clock is stage-RELATIVE. Keyed on the absolute step it would fire
    immediately in every stage after the first."""
    p = _probe()
    p.tick()
    p.m.step_ind = 1
    p.tick()
    p.m.step_ind = 600
    p.tick()                                        # 500 milestone consumed
    p.m.protocol.stage = _FakeStage('equilibration', 'fused')
    p.m.step_ind = 700
    assert p.tick() is None and p._clock_fired == set()
    p.m.step_ind = 701
    assert p.tick() == {'entry': True, 'why': 'stage_entry'}
    p.m.step_ind = 900
    assert p.tick() is None, 'clock fired on absolute rather than stage-relative steps'


def test_the_entry_race_defers_when_the_larder_is_still_filling():
    """MEASURED FAILURE, run race_L1_phase1 (elj, 2026-08-21): the ramp froze at
    step 50 with 51 of 68 batches harvested, the entry race fired, found the
    larder short, and was CONSUMED -- so the run's cold-start escape was thrown
    away silently and no race ran until the clock. A stage entry is precisely
    when the harvest is thinnest, so this was not an edge case."""
    p = _probe()
    ran = {'n': 0}

    def _guard(entry, why):
        ran['n'] += 1
        return None if ran['n'] == 1 else {'entry': entry, 'why': why}

    p._guarded = _guard
    p.tick()                                        # stage change arms it
    p.m.step_ind = 1
    assert p.tick() is None                         # larder short -> deferred
    assert p._armed_entry, 'a short larder consumed the entry race'
    p.m.step_ind = 2
    assert p.tick() == {'entry': True, 'why': 'stage_entry'}
    assert not p._armed_entry, 'the entry race fired but stayed armed'


def test_a_transition_drops_the_previous_stage_harvest():
    """Batches drawn under the outgoing stage carry its branches and its loss
    mixture. Racing on them would score candidate rates against an objective the
    run has already left -- the one comparison the design forbids."""
    p = _probe()
    p.tick()                                        # first sight registers the stage
    _fill(p.larder, 'bwd', 30)                      # then the stage harvests
    assert p.larder.count('bwd') == 30
    p.m.protocol.stage = _FakeStage('equilibration', 'fused')
    p.m.step_ind = 100
    p.tick()                                        # transition
    assert p.larder.count('bwd') == 0, 'stale harvest survived the transition'


def test_the_probe_never_takes_the_run_down_with_it():
    """A measurement device that can kill training is worse than no device."""
    p = RaceProbe(_FakeModeller(), verbose=False)
    p.larder.ready = lambda *a, **k: True

    def _boom(entry):
        raise RuntimeError('boom')

    p.run_event = _boom
    assert p._guarded(entry=False, why='clock') is None


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
