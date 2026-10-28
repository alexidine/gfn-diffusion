"""
`controller.py::LRController` under `lr_control.mode: fixed`, the only mode: the
per-stage schedule (`lr_control.schedule`, a stage's `lr_schedule`), its
multiplier for each segment kind, its stage-entry anchor and resume, the cut on a
fire, the hard-failure bars fitted on the first segment and refitted after the
last rate change, and the refusal of the deleted bracket mode, the bracket keys
and the keys `schedule` replaced.

THE FAKE IS A REAL TRAINING LOOP, not a mock: real `torch.optim.Adam`, real
parameters and gradients. Its loss floor grows like lr^2 (Adam's step is about
`lr` whatever the curvature), so a hot held rate parks the loss above a bar
fitted at the cold first-segment rate -- the shape the bar refit exists for.

    python -m pytest -q tests/lr/test_lr_fixed_schedule.py
"""

import math
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml

from config_invariants import legacy_lr_schedule, normalize_lr_schedule
from controller import (BURN_IN, CRUISE, LRController, lr_schedule_ends,
                        lr_schedule_value)
from grad_clip_guard import GradClipGuard
from hard_failure_bars import HardFailureBars
from protocol import Stage

MK_DEV = Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'


class _Store:
    """A value that rises by 1 per step; enters the loss at 1e-3."""

    def __init__(self, value=0.0):
        self.value = torch.tensor([float(value)])

    def churn(self):
        self.value.add_(1.0)


def _stage_ns(name='s0', lr_schedule=None, max_steps=None):
    return Namespace(name=name, lr_sensor=None, train_mode='bwd',
                     lr_schedule=lr_schedule, max_steps=max_steps)


class FakeTrainer:
    """loss = ||w + sigma * eps||^2 + store_value * 1e-3, stepped by Adam."""

    def __init__(self, base_lr=0.1, sigma=0.01, dim=8, lr_control=None,
                 energy_function=None, stage=None, stage_ctrl=True):
        torch.manual_seed(1234)
        self.gfn_model = torch.nn.Linear(dim, 1, bias=False)
        with torch.no_grad():
            self.gfn_model.weight.fill_(0.02)
        self.flow = torch.nn.Parameter(torch.zeros(1))
        # train.py builds all five optimizers; a stage steps one of them.
        self.train_key = 'bwd'
        self.optimizers = {
            'bwd': torch.optim.Adam(self.gfn_model.parameters(), lr=base_lr),
            'flow': torch.optim.Adam([self.flow], lr=0.1),
        }
        for spectator in ('fwd', 'replay', 'fused'):
            self.optimizers[spectator] = torch.optim.Adam(
                [torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
        self.sigma = float(sigma)
        self.step_ind = 0
        self.phase = 0
        self.lr_ctrl = None
        if stage_ctrl:
            # the stage entered at step 0, as protocol.begin + the first tick
            # stamp it on a fresh run
            self.stage_ctrl = {'lr_sched_entry': 0}
        self.grad_guard = GradClipGuard(static_clip=1e6, enabled=True,
                                        warmup_steps=30)
        self.protocol = Namespace(stage=stage if stage is not None else _stage_ns())
        self.store = _Store(0.0)
        self.last_grad_norm_pre_clip = None
        self.losses = []
        self.args = Namespace(
            lr_policy=base_lr, lr_back=base_lr, lr_replay=base_lr,
            lr_fused=base_lr, lr_flow=0.1, min_lr=1e-12, max_lr=None,
            lr_servo_managed=('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'),
            lr_control=lr_control if lr_control is not None else _control())
        if energy_function is not None:
            self.args.energy_function = energy_function
        self.lr_controller = LRController(self)

    def train_step(self, step_type):
        opt = self.optimizers[self.train_key]
        opt.zero_grad(set_to_none=True)
        w = self.gfn_model.weight
        eps = torch.randn_like(w) * self.sigma
        loss = ((w + eps) ** 2).sum() + self.store.value.squeeze() * 1e-3
        loss.backward()
        pre_clip = torch.nn.utils.clip_grad_norm_(
            self.gfn_model.parameters(), self.grad_guard.threshold(step_type)).item()
        self.grad_guard.observe(step_type, pre_clip)
        self.last_grad_norm_pre_clip = pre_clip
        opt.step()
        self.store.churn()
        value = float(loss.detach())
        self.losses.append(value)
        return value


def _sched(*segs):
    return [dict(to=to, steps=steps, kind=kind) for to, steps, kind in segs]


#: hold 0.05 for 60 steps, then a geometric move to 0.2 over 40.
SCHED = _sched((0.05, 60, 'hold'), (0.2, 40, 'geometric'))


def _control(**kw):
    """A fixed-mode block at test scale: a 25-step root window and k = 3."""
    cfg = dict(mode='fixed', seed_lr=0.1, control_flow_lr=False,
               schedule=SCHED, verbose=False)
    hf = dict(loss_excursion_k=3.0, grad_excursion_x=100.0,
              loss_abs=1.0e6, grad_abs=1.0e6, root_window=25,
              min_observations=10)
    hf.update(kw.pop('hard_failure', {}))
    cfg.update(kw)
    cfg = {k: v for k, v in cfg.items() if v is not _DROP}
    cfg['hard_failure'] = Namespace(**hf)
    return Namespace(**cfg)


_DROP = object()


def _host_step(m):
    """One host-loop iteration: train, observe, tick."""
    c = m.lr_controller
    m.step_ind += 1
    loss = m.train_step(m.train_key)
    verdict = c.observe(m.train_key, loss, m.last_grad_norm_pre_clip)
    c.tick()
    return loss, verdict


def _advance(m, c, n):
    for _ in range(n):
        m.step_ind += 1
        c.tick()


# ================================================================ construction

def test_bracket_mode_is_refused_with_its_replacement_named():
    with pytest.raises(ValueError, match="mode 'bracket' was deleted") as e:
        FakeTrainer(lr_control=_control(mode='bracket'))
    assert 'schedule' in str(e.value)


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="must be 'fixed'"):
        FakeTrainer(lr_control=_control(mode='servo'))


@pytest.mark.parametrize('key,value', [
    ('candidate_scales', [0.05, 0.2, 0.8]), ('trial_steps', 150),
    ('safety_rungs', 1), ('repeat_every', 0), ('trial_settle_steps', 10),
    ('logz_detour_nats', 2.0), ('boundary_confirm_repeats', 1),
    ('boundary_densify', True), ('min_root_bias_correction', 0.9)])
def test_a_retired_bracket_key_raises_on_the_crystal_route(key, value):
    with pytest.raises(ValueError, match=key):
        FakeTrainer(lr_control=_control(**{key: value}))


def test_retired_bracket_keys_are_ignored_on_the_conformer_route():
    m = FakeTrainer(lr_control=_control(candidate_scales=[0.05, 0.2],
                                        min_root_bias_correction=0.9,
                                        logz_detour_nats=2.0),
                    energy_function='conformer_torsions')
    assert m.lr_controller.mode == 'fixed'


@pytest.mark.parametrize('key,value', [
    ('burn_in_steps', 3000), ('burn_in_scale', 0.05),
    ('promotion_ramp_steps', 1000), ('fixed_scale', 0.2)])
def test_a_replaced_burn_in_key_raises_on_the_crystal_route(key, value):
    with pytest.raises(ValueError, match=f"{key}.*replaced 2026-10-28") as e:
        FakeTrainer(lr_control=_control(**{key: value}))
    assert 'schedule' in str(e.value)


def test_the_conformer_route_reads_the_replaced_keys_as_the_old_schedule():
    """configs/conformer_mk*.yaml still carry the four keys and no schedule."""
    m = FakeTrainer(lr_control=_control(schedule=_DROP, burn_in_steps=3000,
                                        burn_in_scale=0.05, promotion_ramp_steps=1000,
                                        fixed_scale=0.2),
                    energy_function='conformer_torsions')
    assert m.lr_controller._run_schedule == legacy_lr_schedule(3000, 0.05, 1000, 0.2)


def test_the_conformer_route_refuses_a_schedule_beside_the_replaced_keys():
    with pytest.raises(ValueError, match='both `schedule`'):
        FakeTrainer(lr_control=_control(fixed_scale=0.2),
                    energy_function='conformer_torsions')


def test_a_block_with_no_schedule_is_refused():
    with pytest.raises(ValueError, match='needs an explicit lr_control.schedule'):
        FakeTrainer(lr_control=_control(schedule=_DROP))


def test_an_unknown_key_still_raises():
    with pytest.raises(ValueError, match='unknown key'):
        FakeTrainer(lr_control=_control(warmup_steps=100))


@pytest.mark.parametrize('schedule,match', [
    (_sched((0.05, 10, 'hold'), (0.2, 10, 'ramp')), r"\.kind is 'ramp'"),
    (_sched((0.0, 10, 'hold')), r'\.to is 0\.0'),
    (_sched((0.05, 10, 'hold'), (-1.0, 10, 'linear')), r'\.to is -1\.0'),
    (_sched((0.05, 10, 'linear')), 'must be a hold'),
    (_sched((0.05, None, 'hold'), (0.2, 10, 'linear')), 'segments follow it'),
    (_sched((0.05, -1, 'hold')), r'\.steps is -1'),
    ([], 'non-empty list'),
    ([{'to': 0.05, 'steps': 10, 'kind': 'hold', 'gamma': 0.9}], 'unknown key'),
])
def test_a_malformed_schedule_is_refused(schedule, match):
    with pytest.raises(ValueError, match=match):
        FakeTrainer(lr_control=_control(schedule=schedule))


def test_an_open_ended_move_needs_the_stage_cap():
    sched = _sched((0.05, 10, 'hold'), (0.01, None, 'cosine'))
    with pytest.raises(ValueError, match='no max_steps'):
        FakeTrainer(lr_control=_control(schedule=sched))
    c = FakeTrainer(lr_control=_control(schedule=sched),
                    stage=_stage_ns(max_steps=110)).lr_controller
    assert c._schedule()[1] == [10, 110]
    # a hold may run open-ended on an uncapped stage: it holds anyway
    FakeTrainer(lr_control=_control(schedule=_sched((0.05, 10, 'hold'),
                                                    (0.2, None, 'hold'))))


def test_no_lr_control_block_runs_the_float_base_rates():
    """train_conformer.py: float rates, no lr_control, no managed keys, and the
    controller only ever stepped -- every policy group runs its base rate."""
    m = FakeTrainer()
    del m.args.lr_control
    m.args.lr_servo_managed = ()
    m.lr_ctrl = None
    c = LRController(m)
    assert c.mode == 'fixed' and c._run_schedule == legacy_lr_schedule()
    c.step()
    for key in ('fwd', 'bwd', 'replay'):
        assert m.optimizers[key].param_groups[0]['lr'] == pytest.approx(0.1)
    assert m.optimizers['flow'].param_groups[0]['lr'] == pytest.approx(0.1)


# ============================================================ the multiplier

@pytest.mark.parametrize('kind,expect', [
    ('hold', lambda a, b, p: b),
    ('linear', lambda a, b, p: a + (b - a) * p),
    ('geometric', lambda a, b, p: a * (b / a) ** p),
    ('cosine', lambda a, b, p: b + (a - b) * 0.5 * (1 + math.cos(math.pi * p))),
])
def test_each_kind_moves_from_the_previous_to_its_own_to(kind, expect):
    """Segment 0 holds 0.4 for 100 steps; segment 1 moves 0.4 -> 0.1 over 200."""
    segs = normalize_lr_schedule(_sched((0.4, 100, 'hold'), (0.1, 200, kind)))
    for elapsed in (0, 50, 99):
        assert lr_schedule_value(segs, elapsed) == (pytest.approx(0.4), 0)
    for elapsed, p in ((100, 0.0), (150, 0.25), (200, 0.5), (299, 0.995)):
        v, i = lr_schedule_value(segs, elapsed)
        assert i == 1 and v == pytest.approx(expect(0.4, 0.1, p)), (kind, elapsed)
    for elapsed in (300, 10_000):              # past the last segment it holds
        assert lr_schedule_value(segs, elapsed) == (pytest.approx(0.1), 1)


def test_the_cosine_and_geometric_midpoints_differ():
    """The kinds are distinct curves: the cosine midpoint is the arithmetic
    mean, the geometric one the geometric mean."""
    cos = normalize_lr_schedule(_sched((0.4, 1, 'hold'), (0.1, 100, 'cosine')))
    geo = normalize_lr_schedule(_sched((0.4, 1, 'hold'), (0.1, 100, 'geometric')))
    assert lr_schedule_value(cos, 51)[0] == pytest.approx(0.25)
    assert lr_schedule_value(geo, 51)[0] == pytest.approx(0.2)


def test_a_zero_step_segment_is_a_jump_and_segments_compose():
    segs = normalize_lr_schedule(_sched((0.05, 10, 'hold'), (1.0, 0, 'linear'),
                                        (1.0, 20, 'hold'), (0.1, 10, 'linear')))
    assert lr_schedule_ends(segs) == [10, 10, 30, 40]
    assert lr_schedule_value(segs, 9) == (pytest.approx(0.05), 0)
    assert lr_schedule_value(segs, 10) == (pytest.approx(1.0), 2)
    assert lr_schedule_value(segs, 35) == (pytest.approx(0.55), 3)
    assert lr_schedule_value(segs, 40) == (pytest.approx(0.1), 3)


def test_a_null_step_segment_runs_to_the_stage_cap():
    segs = normalize_lr_schedule(_sched((1.0, 100, 'hold'), (0.01, None, 'linear')))
    assert lr_schedule_ends(segs, max_steps=1100) == [100, 1100]
    assert lr_schedule_value(segs, 600, max_steps=1100)[0] == pytest.approx(0.505)
    assert lr_schedule_value(segs, 1100, max_steps=1100)[0] == pytest.approx(0.01)


def _old_burn_in_and_ramp(elapsed, burn_steps=3000, burn=0.05, ramp=1000, fixed=0.2):
    """The deleted controller's scale on a fresh stage with a full window at the
    end of burn-in: burn-in at `burn`, then burn * (fixed/burn)^(done/ramp)."""
    if elapsed < burn_steps:
        return burn
    done = elapsed - burn_steps
    return fixed if done >= ramp else burn * (fixed / burn) ** (done / ramp)


def test_the_mk_dev_schedule_reproduces_the_old_burn_in_and_ramp():
    with open(MK_DEV) as f:
        node = yaml.safe_load(f)['lr_control']['schedule']
    segs = normalize_lr_schedule(node)
    assert segs == legacy_lr_schedule(3000, 0.05, 1000, 0.2)
    for elapsed in list(range(0, 6000, 7)) + [2999, 3000, 3001, 3999, 4000, 4001]:
        assert lr_schedule_value(segs, elapsed)[0] == pytest.approx(
            _old_burn_in_and_ramp(elapsed), rel=1e-12), elapsed
    # ...and through the controller's own seat, with the stage anchored at 0
    m = FakeTrainer(lr_control=_control(schedule=node))
    c = m.lr_controller
    seen = {}
    for _ in range(4500):
        c.tick()
        seen[m.step_ind] = c.scale
        m.step_ind += 1
    for elapsed in (0, 2999, 3000, 3500, 3999, 4000, 4499):
        assert seen[elapsed] == pytest.approx(_old_burn_in_and_ramp(elapsed)), elapsed
    assert c.live_rates()[m.train_key] == pytest.approx(0.1 * 0.2)


# ====================================================== anchoring and resume

def test_the_schedule_is_anchored_at_the_stage_entry_step():
    m = FakeTrainer()
    c = m.lr_controller
    m.step_ind = 500
    assert c.on_stage_change() == 60
    assert m.stage_ctrl['lr_sched_entry'] == 500
    assert c.scale == pytest.approx(0.05)
    _advance(m, c, 79)                        # elapsed 79: 19 steps into the move
    assert c.scale == pytest.approx(0.05 * (0.2 / 0.05) ** (19 / 40))
    assert c.phase == CRUISE


def test_a_fresh_run_stamps_the_entry_at_its_first_tick():
    m = FakeTrainer()
    m.stage_ctrl = {}
    c = m.lr_controller
    c.tick()
    assert m.stage_ctrl['lr_sched_entry'] == 0 and c.phase == BURN_IN


def test_without_a_stage_ctrl_the_entry_rides_lr_ctrl():
    m = FakeTrainer(stage_ctrl=False)
    m.step_ind = 7
    m.lr_controller.on_stage_change()
    assert m.lr_ctrl['lr_sched_entry'] == 7


def test_a_resume_continues_mid_segment_and_rereads_the_config():
    """The new process gets the checkpointed stage_ctrl and lr_ctrl and a config
    whose schedule moved: it continues from the stamped entry, on the config's
    schedule, and fits no warm-up bar for a first segment it did not see end."""
    m = FakeTrainer()
    m.lr_controller.tick()
    _advance(m, m.lr_controller, 70)
    saved_stage, saved_lr = dict(m.stage_ctrl), dict(m.lr_ctrl)

    sched2 = _sched((0.05, 60, 'hold'), (0.8, 40, 'geometric'))
    m2 = FakeTrainer(lr_control=_control(schedule=sched2))
    m2.stage_ctrl, m2.lr_ctrl, m2.step_ind = saved_stage, saved_lr, 80
    c2 = m2.lr_controller
    c2.tick()
    assert c2.scale == pytest.approx(0.05 * (0.8 / 0.05) ** (20 / 40))
    assert c2.phase == CRUISE and not c2.bars.loss_bar
    _advance(m2, c2, 20)
    assert c2.scale == pytest.approx(0.8)


def test_a_stale_lr_ctrl_layout_is_discarded():
    m = FakeTrainer()
    m.lr_ctrl = {'ver': 10, 'scale': 0.7, 'stage_entry_step': 0, 'bracket': None}
    c = m.lr_controller
    assert c._state() == {'ver': LRController._STATE_VER, 'scale': 0.05, 'cut': 1.0}


def test_a_stage_lr_schedule_replaces_the_run_level_one():
    own = normalize_lr_schedule(_sched((1.0, 5, 'hold'), (0.5, 10, 'linear')))
    m = FakeTrainer(stage=_stage_ns(lr_schedule=own))
    c = m.lr_controller
    assert c.entry_scale() == 1.0
    c.tick()
    _advance(m, c, 10)
    assert c.scale == pytest.approx(0.75)


def test_the_stage_parser_normalizes_and_judges_an_lr_schedule():
    spec = {'name': 's', 'train_mode': 'fused',
            'fracs': {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}}
    assert Stage(dict(spec), 0).lr_schedule is None
    st = Stage(dict(spec, lr_schedule=[{'to': 1, 'steps': 5, 'kind': 'hold'}]), 0)
    assert st.lr_schedule == [{'to': 1.0, 'steps': 5, 'kind': 'hold'}]
    open_ended = [{'to': 1, 'steps': 5, 'kind': 'hold'},
                  {'to': 0.1, 'steps': None, 'kind': 'geometric'}]
    with pytest.raises(ValueError, match='no max_steps'):
        Stage(dict(spec, lr_schedule=open_ended), 0)
    assert Stage(dict(spec, lr_schedule=open_ended, max_steps=50), 0).lr_schedule[1]['steps'] is None


def test_a_scale_set_by_another_caller_stands_until_the_schedule_moves():
    """bench/arms.py actuates through set_scale past the first segment."""
    m = FakeTrainer(lr_control=_control(schedule=_sched((0.05, 10, 'hold'),
                                                        (0.2, 0, 'geometric'))))
    c = m.lr_controller
    c.tick()
    _advance(m, c, 15)
    c.set_scale(0.9, why='bench_arm')
    _advance(m, c, 5)
    assert c.scale == pytest.approx(0.9)


# ================================================================ the cut

def test_a_cut_multiplies_the_rest_of_the_stage_and_a_stage_change_resets_it():
    m = FakeTrainer(lr_control=_control(fire_cut_factor=0.5))
    c = m.lr_controller
    c.tick()
    _advance(m, c, 70)                         # 10 steps into the move
    c.on_divergence()
    assert c.scale == pytest.approx(0.5 * 0.05 * 4 ** (10 / 40))
    _advance(m, c, 1)
    assert c.scale == pytest.approx(0.5 * 0.05 * 4 ** (11 / 40)), 'the cut was undone'
    _advance(m, c, 50)
    assert c.scale == pytest.approx(0.5 * 0.2)
    assert m.lr_ctrl['cut'] == pytest.approx(0.5)
    c.on_stage_change()
    assert m.lr_ctrl['cut'] == 1.0 and c.scale == pytest.approx(0.05)


# ================================================================== the bars

def test_the_bars_are_fitted_on_the_last_root_window_steps_of_the_first_segment():
    m = FakeTrainer()
    c = m.lr_controller
    for _ in range(59):
        _host_step(m)
    assert c.phase == BURN_IN and not c.bars.loss_bar, 'a bar was fitted inside the warm-up'
    _host_step(m)                              # step 60: the warm-up ends
    assert c.phase == CRUISE
    ref = HardFailureBars(loss_excursion_k=3.0, grad_excursion_x=100.0,
                          root_window=25, min_observations=10)
    assert ref.derive({'bwd': m.losses[:60][-25:]}, []) is None
    assert c.bars.loss_bar[m.train_key] == pytest.approx(ref.loss_bar['bwd'])
    first = dict(c.bars.loss_bar)
    for _ in range(30):                        # inside the move: no refit yet
        _host_step(m)
    assert c.bars.loss_bar == first and c._cruise_bar is None


def test_the_refit_is_queued_when_the_last_rate_change_ends_not_a_later_hold():
    sched = _sched((0.05, 30, 'hold'), (0.2, 20, 'linear'), (0.2, 100, 'hold'))
    m = FakeTrainer(lr_control=_control(schedule=sched))
    c = m.lr_controller
    for _ in range(49):
        _host_step(m)
    assert c._cruise_bar is None
    _host_step(m)                              # step 50: the linear move ends
    assert c._cruise_bar is not None


def test_no_refit_is_queued_when_no_segment_changes_the_rate():
    sched = _sched((0.05, 30, 'hold'), (0.05, 20, 'geometric'))
    m = FakeTrainer(lr_control=_control(schedule=sched))
    for _ in range(80):
        _host_step(m)
    c = m.lr_controller
    assert c.bars.loss_bar and c._cruise_bar is None


def test_a_short_window_at_the_end_of_the_warm_up_is_fitted_once_full():
    """A first segment shorter than min_observations: the bars wait for a full
    window instead of fitting nothing."""
    m = FakeTrainer(lr_control=_control(schedule=_sched((0.05, 5, 'hold'),
                                                        (0.2, 0, 'geometric'))))
    c = m.lr_controller
    for _ in range(10):
        _host_step(m)
    assert not c.bars.loss_bar
    for _ in range(30):
        _host_step(m)
    assert m.train_key in c.bars.loss_bar


def test_the_span_bar_is_well_defined_on_a_signed_channel():
    bars = HardFailureBars(loss_excursion_k=10.0, min_observations=3)
    assert bars.derive({'bwd': [-25.0, -24.0, -26.0, -25.5]}, []) is None
    assert bars.loss_bar['bwd'] == pytest.approx(-24.0 + 10.0 * 2.0)
    assert bars.judge('bwd', -20.0, None) is None
    assert bars.judge('bwd', 318.0, None) is not None


def test_a_window_too_short_fits_no_loss_bar():
    bars = HardFailureBars(min_observations=20)
    why = bars.derive({'fwd': [1.0, 2.0]}, [])
    assert why is not None and 'no loss bar' in why
    assert bars.loss_bar == {}


def test_the_backstops_and_non_finite_checks_fire_without_a_fitted_bar():
    bars = HardFailureBars(loss_abs=100.0, grad_abs=10.0)
    assert bars.judge('bwd', float('nan'), None) == 'nonfinite_loss_bwd'
    assert bars.judge('bwd', 150.0, None).startswith('loss_abs_bwd')
    assert bars.judge('bwd', 1.0, float('inf')) == 'nonfinite_grad'
    assert bars.judge('bwd', 1.0, 20.0).startswith('grad_abs')
    assert bars.judge('bwd', 1.0, 1.0) is None


def _hot_seat(held=3.2, **kw):
    """The held rate 64x the warm-up rate, reached by a jump at step 60."""
    hf = dict(root_window=25, min_observations=10, cruise_settle_steps=10)
    hf.update(kw.pop('hard_failure', {}))
    sched = _sched((0.05, 60, 'hold'), (held, 0, 'geometric'))
    return FakeTrainer(lr_control=_control(schedule=sched, hard_failure=hf, **kw))


def _to_cruise(m, cap=400):
    c = m.lr_controller
    for i in range(cap):
        _host_step(m)
        if c.phase == CRUISE:
            return i + 1
    raise AssertionError('the first segment never ended')


def _cruise_for(m, n):
    """n cruise steps through the host seat. Returns the losses a bar fired on."""
    c, fired = m.lr_controller, []
    for _ in range(n):
        m.step_ind += 1
        loss = m.train_step(m.train_key)
        before = c._moderate_fires
        if c.observe(m.train_key, loss, m.last_grad_norm_pre_clip) == 'diverged' \
                or c._moderate_fires > before:
            fired.append(loss)
        c.tick()
    return fired


def test_the_cold_bar_is_crossed_by_ordinary_training_at_the_held_rate():
    """The premise of the refit, measured on the fixture."""
    m = _hot_seat()
    c = m.lr_controller
    _to_cruise(m)
    root_bar = (c._cruise_bar or {}).get('prev_loss', {}).get(m.train_key)
    assert root_bar is not None, 'no root bar was ever derived'
    assert c.scale > 0.05 * 4
    healthy = [m.train_step(m.train_key) for _ in range(120)]
    assert [v for v in healthy if v >= root_bar], (
        f'ordinary training at the held rate peaks at {max(healthy):.4g}, under the '
        f'root bar {root_bar:.4g} -- the fixture cannot test the refit')


def test_the_cold_bar_stays_live_until_the_refit():
    m = _hot_seat()
    c = m.lr_controller
    _to_cruise(m)
    assert c._cruise_bar is not None
    assert m.train_key in c.bars.loss_bar, 'the warm-up bar was dropped before the refit'
    assert c.bars.loss_bar[m.train_key] == c._cruise_bar['prev_loss'][m.train_key]


def test_the_bars_are_refitted_at_the_held_rate():
    m = _hot_seat()
    c = m.lr_controller
    _to_cruise(m)
    root_bar = c._cruise_bar['prev_loss'][m.train_key]
    assert not c._bars_redrawn
    _cruise_for(m, 10 + 25 + 5)
    assert c._cruise_bar is None, 'the refit never completed'
    assert c._bars_redrawn
    assert c.bars.loss_bar[m.train_key] > root_bar


def test_keeping_the_cold_bar_turns_healthy_training_into_fires():
    """`cruise_rederive: false` must fire more after the refit window. Counted
    over 60 steps: the fixture's store term raises the loss 1e-3 per step, so a
    longer horizon drifts past the refitted bar too."""
    def firings(rederive):
        m = _hot_seat(hard_failure=dict(cruise_rederive=rederive),
                      fire_cut_factor=1.0, fire_cooldown_steps=0)
        _to_cruise(m)
        _cruise_for(m, 10 + 25 + 5)
        return len(_cruise_for(m, 60))

    with_fix, without_fix = firings(True), firings(False)
    assert without_fix > 0, 'the cold bar never fired, so this fixture cannot see it'
    assert with_fix * 4 < without_fix, (with_fix, without_fix)


def test_the_settle_window_is_discarded_rather_than_fitted():
    m = _hot_seat(hard_failure=dict(cruise_settle_steps=30))
    c = m.lr_controller
    _to_cruise(m)
    _cruise_for(m, 28)
    assert c._cruise_bar is not None and len(c._loss_history[m.train_key]) > 0
    _cruise_for(m, 2)
    assert len(c._loss_history[m.train_key]) <= 1


def test_the_refit_clock_counts_ticks_not_step_ind():
    m = _hot_seat(hard_failure=dict(cruise_settle_steps=40))
    c = m.lr_controller
    _to_cruise(m)
    m.step_ind += 10_000
    _cruise_for(m, 38)
    assert c._cruise_bar is not None
    assert len(c._loss_history[m.train_key]) > 0


def test_a_refit_that_cannot_be_derived_keeps_the_previous_bars():
    """The deadline path with nothing to fit restores the warm-up bars."""
    m = _hot_seat(hard_failure=dict(cruise_settle_steps=2))
    c = m.lr_controller
    _to_cruise(m)
    root = dict(c._cruise_bar['prev_loss'])
    assert root
    for _ in range(2 + 4 * 25 + 2):
        m.step_ind += 1
        c.tick()
    assert c._cruise_bar is None, 'the refit never hit its deadline'
    assert c.bars.loss_bar == root


def test_a_stage_change_clears_the_bars_and_not_only_the_window():
    m = _hot_seat()
    c = m.lr_controller
    _to_cruise(m)
    _cruise_for(m, 10 + 25 + 5)
    assert c.bars.loss_bar
    c.on_stage_change()
    assert c.bars.loss_bar == {} and c.bars.grad_bar is None
    assert c._cruise_bar is None and c._bars_redrawn is False
    assert c.phase == BURN_IN


# ============================================================ fire response

def _cruising(**kw):
    m = _hot_seat(**kw)
    _to_cruise(m)
    return m, m.lr_controller


def test_an_excursion_fire_returns_diverged_and_the_cut_follows_the_restore():
    m, c = _cruising(hard_failure=dict(cruise_rederive=False),
                     fire_cut_factor=0.5, fire_cooldown_steps=50)
    before = c.scale
    fired = None
    for _ in range(120):
        m.step_ind += 1
        fired = c.observe(m.train_key, m.train_step(m.train_key),
                          m.last_grad_norm_pre_clip)
        if c._moderate_fires:
            break
    assert c._moderate_fires == 1, 'the cold bar never fired on this fixture'
    assert fired == 'diverged'
    assert c.scale == pytest.approx(before), 'observe() must not cut'
    c.on_divergence()
    assert c._divergences == 1
    assert c.scale == pytest.approx(before * 0.5)
    for _ in range(10):                      # inside the cooldown
        m.step_ind += 1
        assert c.observe(m.train_key, m.train_step(m.train_key),
                         m.last_grad_norm_pre_clip) is None
    assert c._moderate_fires == 1


def test_a_disaster_returns_diverged_and_the_rewind_cut_follows():
    m, c = _cruising(fire_cut_factor=0.5)
    assert c.observe(m.train_key, float('nan'), 10.0) == 'diverged'
    before = c.scale
    c.on_divergence()
    assert c._divergences == 1
    assert c.scale == pytest.approx(before * 0.5)
    c.tick()
    assert c.scale == pytest.approx(before * 0.5), 'the next tick undid the cut'


def test_the_report_carries_the_phase_segment_and_cut():
    m, c = _cruising()
    c.step()
    r = c.report()
    assert r['lr_bracket/phase'] == 2.0
    assert r['lr_ctrl/segment'] == 1.0 and r['lr_ctrl/cut'] == 1.0
    assert r['lr_ctrl/scale'] == pytest.approx(3.2)
    for gone in ('lr_bracket/brackets', 'lr_bracket/discarded_steps',
                 'lr_bracket/refused', 'lr_bracket/fixed_mode', 'lr_bracket/status',
                 'lr_bracket/promoted_scale'):
        assert gone not in r
