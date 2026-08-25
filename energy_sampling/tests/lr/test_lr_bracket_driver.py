"""
The bracket's TRAINER side: does a candidate trial actually measure the rate it
claims to?

WHY THIS FILE IS THE LOAD-BEARING ONE. `test_lr_bracket.py` proves the decision
layer picks the right rung given a set of outcomes. It cannot see whether those
outcomes mean anything -- and every way this mechanism can lie lives on THIS
side, because each of them still produces a trial that runs, reports, and gets
selected:

  * a candidate starting from the previous candidate's end state;
  * a snapshot that aliases the live trainer, so the first restore rewrites it;
  * an optimizer restored with a fresh Adam counter, running the trial at 15-30%
    of its nominal rate;
  * a candidate that cuts its own rate and survives a rate it never held;
  * discarded trial compute advancing the run's logical clock.

THE TEST THAT CATCHES MOST OF THEM IS ONE LINE OF INTENT: run two candidates at
the SAME scale and require bitwise-identical losses. Anything weaker -- an
allclose, a spot check on the weights, an assertion inside the driver -- passes
on a snapshot that leaks. So the fake trainer below is built to make that test
sharp rather than convenient: its loss depends on the RNG stream, on a mutable
buffer, and on the optimizer's accumulated moments, so ANY of the three failing
to round-trip moves the number.

THE FAKE IS A REAL TRAINING LOOP, not a mock. Real `torch.optim.Adam`, real
parameters, real gradients, the real `Checkpointer.load_optimizer_state`, the
real `MetricTracker` and `GradClipGuard`. What it stands in for is the energy
function and the rollout, which are the parts that need a GPU and which the
driver never looks at.
"""

import copy
import math
from argparse import Namespace

import pytest
import torch

from checkpointing import Checkpointer
from controller import LRController
from grad_clip_guard import GradClipGuard
from lr_bracket import ALL_FAILED, BRACKETED, CRUISE, Trial, SCREEN
from lr_bracket_probe import BracketDriver, HardFailureBars, bias_correction
from utils import MetricTracker


# --------------------------------------------------------------- the trap ---

class AliasingStore:
    """A buffer-shaped store that hands out its LIVE tensors and keeps whatever
    it is restored from.

    THIS IS NOT A CONTRIVANCE, it is the shipping hazard reduced to ten lines.
    `CrystalBuffer.state_dict()` moves its tensors with `.cpu()`, and
    `Tensor.cpu()` on a CPU-resident tensor RETURNS ITSELF -- so on
    `buffer_device: cpu` the state dict is the live store. `from_state_dict`
    then calls `.to(device)` on a PyG `Data`, which mutates in place in this
    codebase and returns self. Chain those and a snapshot taken at the root is
    the same storage the first trial then writes into, so candidate 2 starts
    from candidate 1's buffer while every seam reports success.
    """

    def __init__(self, value=0.0):
        self.value = torch.tensor([float(value)])

    def state_dict(self):
        return {'value': self.value.cpu()}          # on CPU: the SAME object

    @classmethod
    def from_state_dict(cls, state, device):
        obj = cls.__new__(cls)
        obj.value = state['value']                  # no copy: aliases its input
        return obj

    def churn(self):
        self.value.add_(1.0)                        # in place, like a real buffer


# ------------------------------------------------------------- the trainer ---

class FakeTrainer:
    """A minimal but genuine trainer the driver can drive.

    The loss is deliberately sensitive to three separate pieces of restored
    state, so the bitwise-identity test convicts any of them leaking:

        loss = ||w + sigma * eps||^2  +  buffer_value * 1e-3

    `eps` comes from the global RNG stream, `w` from the parameters and the
    optimizer's moments, and `buffer_value` from a store that mutates in place.

    IT ALSO FAILS AT HOT RATES BY THE MECHANISM A REAL RUN DOES, rather than by
    a scripted rule. Adam's step magnitude is about `lr` per step whatever the
    curvature, so the iterate cannot settle closer to the optimum than the rate
    allows: the loss FLOOR grows like lr^2. Burn-in at a cold rate converges to a
    small floor and establishes a narrow span; a hot rung then parks the loss far
    above it and trips the derived bar. That is the same shape as the excursion
    this route actually produced (-25 -> +318) -- a rate whose reachable loss is
    nowhere near the one the root established -- without needing an energy
    function to produce it.

    The weights start NEAR the optimum for that reason: from far away a hot rate
    is simply faster, the loss falls rather than rises, and the fake would test
    the bar in the one regime where a bracket is not being asked anything.
    """

    def __init__(self, base_lr=0.1, sigma=0.01, dim=8, lr_control=None,
                 with_buffer=True):
        torch.manual_seed(1234)
        self.gfn_model = torch.nn.Linear(dim, 1, bias=False)
        with torch.no_grad():
            self.gfn_model.weight.fill_(0.02)
        self.ema_model = copy.deepcopy(self.gfn_model)
        self.flow = torch.nn.Parameter(torch.zeros(1))
        # THE REAL OPTIMIZER SHAPE, and it is not decoration. train.py builds all
        # FIVE keys unconditionally while a stage runs ONE train_mode, so on any
        # canonical config three of the four managed optimizers never take a step
        # and report an Adam counter of None. A fake with a single
        # always-stepping optimizer cannot see that, and did not: it let a bug
        # ship that made the bracket refuse on every stage of every config while
        # reporting the refusal as caution.
        #
        # `train_mode` here is 'bwd' because protocol.TRAIN_MODES is ('bwd',
        # 'fused') -- there is no 'fwd' stage -- so 'bwd' is the optimizer that
        # steps and 'fwd'/'replay'/'fused' are the spectators.
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
        self.buffer_device = 'cpu'
        self.metric_tracker = MetricTracker(period=25.0)
        self.grad_guard = GradClipGuard(static_clip=1e6, enabled=True,
                                        warmup_steps=30)
        self.checkpointer = Checkpointer(self)
        self.protocol = Namespace(stage=Namespace(name='s0', lr_sensor=None,
                                                  train_mode='bwd'))
        self.prior_buffer = AliasingStore(0.0) if with_buffer else None
        self.last_grad_norm_pre_clip = None
        self._nonfinite_pending = False
        self._grad_nonfinite_streak = 0
        self.losses = []                 # every loss this trainer has produced
        self.lr_seen = []                # the LR in force at each step
        self.oom_handled = []            # (exception, step_type) the shared path saw
        self.raise_at = None             # (scale, exception) to throw mid-trial
        self.raise_once = False          # clear raise_at after the first throw

        self.args = Namespace(
            lr_policy=base_lr, lr_back=base_lr, lr_replay=base_lr,
            lr_fused=base_lr, lr_flow=0.1, min_lr=1e-12, max_lr=None,
            # ALL FOUR managed, as every canonical config has them: mk_dev
            # writes lr_policy/lr_back/lr_replay/lr_fused as `auto`.
            lr_servo_managed=('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'),
            lr_control=lr_control if lr_control is not None else _control())
        self.lr_controller = LRController(self)

    # -- the surface the driver drives -------------------------------------

    def train_logic(self, step):
        return self.train_key

    def z_level_fill(self):
        pass

    def z_calibration_tick(self, step_type):
        pass

    def handle_train_epoch_error(self, e, step_type):
        """The shared OOM recovery path, recorded rather than performed."""
        self.oom_handled.append((type(e).__name__, step_type))

    def train_step(self, step_type):
        if self.raise_at is not None:
            want, exc = self.raise_at
            if abs(self.lr_controller.scale - want) < 1e-12:
                if self.raise_once:
                    self.raise_at = None
                raise exc
        opt = self.optimizers[self.train_key]
        self.lr_seen.append(opt.param_groups[0]['lr'])
        opt.zero_grad(set_to_none=True)
        w = self.gfn_model.weight
        eps = torch.randn_like(w) * self.sigma      # consumes the RNG stream
        loss = ((w + eps) ** 2).sum()
        if self.prior_buffer is not None:
            loss = loss + self.prior_buffer.value.squeeze() * 1e-3
        loss.backward()
        pre_clip = torch.nn.utils.clip_grad_norm_(
            self.gfn_model.parameters(), self.grad_guard.threshold(step_type)).item()
        self.grad_guard.observe(step_type, pre_clip)
        self.last_grad_norm_pre_clip = pre_clip
        opt.step()
        if self.prior_buffer is not None:
            self.prior_buffer.churn()
        value = float(loss.detach())
        self.losses.append(value)
        self.metric_tracker.update(self.train_key, {'loss': value}, self.step_ind)
        return value

    # -- helpers ------------------------------------------------------------

    def burn_in(self, n=60, scale=None):
        """Run the trainer at the burn-in scale and feed the controller, the way
        the host loop does. Returns the losses produced."""
        c = self.lr_controller
        c.set_scale(scale if scale is not None else c.bracket.burn_in_scale,
                    why='burn_in')
        out = []
        for _ in range(n):
            self.step_ind += 1
            out.append(self.train_step(self.train_key))
            c.observe(self.train_key, out[-1], self.last_grad_norm_pre_clip)
        return out


#: Rungs this fake survives, and the one it does not. MEASURED on the fixture
#: rather than assumed: at a 12-step horizon the loss floor Adam can reach runs
#: 0.073 / 0.073 / 0.080 / 0.104 across COLD against a derived bar of 0.131, and
#: HOT trips it within two steps. A grid whose top rung merely looked hot would
#: make every test below pass for the wrong reason.
COLD = (0.05, 0.2, 0.8, 3.2)
HOT = 12.8


def _control(**kw):
    cfg = dict(mode='bracket', seed_lr=0.1, control_flow_lr=False,
               burn_in_steps=60, burn_in_scale=0.05,
               min_root_bias_correction=0.1,
               candidate_scales=COLD + (HOT,),
               trial_steps=12, safety_rungs=1, repeat_every=0,
               boundary_confirm_repeats=0, boundary_densify=False,
               fixed_scale=0.2, verbose=False)
    # THE WINDOW IS THE TAIL, and at this scale that has to be said explicitly:
    # the shipping 200 is shorter than a 3000-step burn-in but LONGER than this
    # fixture's 60, so leaving it would fold the fake's opening descent into the
    # span and inflate the bar past every rung. `k` is 3 rather than 10 for the
    # same reason -- a 12-step horizon leaves less room to travel than a real
    # trial does.
    hf = dict(loss_excursion_k=3.0, grad_excursion_x=100.0,
              loss_abs=1.0e6, grad_abs=1.0e6, root_window=25,
              min_observations=10)
    hf.update(kw.pop('hard_failure', {}))
    cfg.update(kw)
    cfg['hard_failure'] = Namespace(**hf)
    return Namespace(**cfg)


def _armed(**kw):
    """A trainer that has burned in, with a driver holding a fresh root."""
    m = FakeTrainer(lr_control=_control(**kw))
    m.burn_in(m.lr_controller.bracket.burn_in_steps)
    c = m.lr_controller
    refusal = c.bars.derive(c._loss_history, c._grad_history)
    assert refusal is None, refusal
    d = BracketDriver(m, c.bracket, c.bars, verbose=False)
    assert d.take_root(m.step_ind) is None
    c.bracket.begin_bracket(m.step_ind, d.root_bias_correction()[0])
    return m, d


def _trial(scale, kind=SCREEN, seed=None, label='t'):
    return Trial(scale, kind, seed=seed, label=label)


# ============================================================ isolation ======

def test_two_candidates_at_the_same_scale_produce_bitwise_identical_losses():
    """THE TEST. Everything about candidate isolation reduces to this, and
    nothing weaker catches a leak: an allclose passes on a snapshot that aliases
    one tensor, and a spot check on the weights misses the RNG stream and the
    optimizer moments entirely.

    The fake's loss depends on all three plus a buffer that mutates in place, so
    a single one failing to round-trip moves the number."""
    m, d = _armed()
    d.run_trial(_trial(0.2, label='a'))
    first = list(m.losses[-d.bracket.trial_steps:])
    d.run_trial(_trial(0.2, label='b'))
    second = list(m.losses[-d.bracket.trial_steps:])

    assert len(first) == d.bracket.trial_steps
    assert first == second, (
        'two candidates at the same scale diverged, so something is leaking '
        'between trials: ' + repr([(a, b) for a, b in zip(first, second) if a != b][:3]))


def test_the_buffer_restore_does_not_alias_the_root():
    """THE PyG TRAP, reproduced. `state_dict()` hands back live tensors on a
    CPU-resident store and `from_state_dict` keeps whatever it is given, so a
    snapshot that copied neither would be rewritten by the first trial's in-place
    churn -- and candidate 2 would start from candidate 1's buffer."""
    m, d = _armed()
    root_value = float(d.root.buffers['prior_buffer']['value'].item())
    d.run_trial(_trial(0.2, label='a'))
    after = float(d.root.buffers['prior_buffer']['value'].item())
    assert after == root_value, (
        f'the trial rewrote the ROOT snapshot in place ({root_value} -> {after})')
    # ...and the live store really is being restored, not merely left alone
    d.root.restore()
    assert float(m.prior_buffer.value.item()) == root_value
    assert m.prior_buffer.value is not d.root.buffers['prior_buffer']['value']


def test_the_root_snapshot_does_not_share_storage_with_the_live_buffer():
    """The capture-side half of the same hazard, asserted STRUCTURALLY because
    that is the only way it can be asserted.

    `state_dict()` really does hand back the live store on a CPU-resident
    buffer. With the restore-side deep copy in place nothing currently mutates
    the captured tensors, so no behavioural test convicts this -- measured by
    mutation: dropping the capture-side copy leaves all the behavioural tests
    green. It is worth keeping and worth pinning: the two copies guard the same
    hazard from opposite ends, and a refactor is far likelier to touch the
    restore path than this one."""
    m, d = _armed()
    captured = d.root.buffers['prior_buffer']['value']
    assert captured is not m.prior_buffer.value
    assert captured.data_ptr() != m.prior_buffer.value.data_ptr(), (
        'the root snapshot shares storage with the live buffer')


def test_a_candidate_cannot_contaminate_a_later_one():
    """A -> B -> A. If B left anything behind, A's second run differs from its
    first. Ordering matters here in a way the same-scale test cannot see: it
    catches state that only a DIFFERENT rate disturbs."""
    m, d = _armed()
    d.run_trial(_trial(0.05, label='a1'))
    a1 = list(m.losses[-d.bracket.trial_steps:])
    d.run_trial(_trial(HOT, label='hot'))
    d.run_trial(_trial(0.05, label='a2'))
    a2 = list(m.losses[-d.bracket.trial_steps:])
    assert a1 == a2, 'a hot candidate contaminated the rung that followed it'


def test_the_metric_tracker_freshness_stamps_round_trip():
    """`MetricTracker.written_at` and `changed_keys` are NOT in its state_dict --
    they are within-process facts -- and the protocol's exit streaks read the
    first of them to tell a new measurement from the same one read again. A
    snapshot built on `state_dict()` alone would silently reset them."""
    m, d = _armed()
    before = dict(m.metric_tracker.written_at)
    d.run_trial(_trial(0.2, label='a'))
    d.root.restore()
    assert m.metric_tracker.written_at == before


def test_the_grad_guard_drained_counters_round_trip():
    """`GradClipGuard.report()` DRAINS n_obs/n_fired/n_saturated, and
    `state_dict()` omits them -- so a snapshot taken through the state dict loses
    exactly the counters that decide whether the bar is firing."""
    m, d = _armed()
    branch = m.grad_guard._branches[m.train_key]
    branch.n_obs, branch.n_fired = 41, 7
    root = d.root.__class__(m, 'probe')
    branch.n_obs, branch.n_fired = 0, 0
    root.restore()
    assert (m.grad_guard._branches[m.train_key].n_obs,
            m.grad_guard._branches[m.train_key].n_fired) == (41, 7)


# =============================================== the rate under test =========

def test_the_candidate_lr_is_fixed_for_the_whole_trial():
    """Set once, held for the horizon. A rate that moved mid-trial would make
    the rung's label a fiction."""
    m, d = _armed()
    m.lr_seen.clear()
    d.run_trial(_trial(0.8, label='a'))
    seen = m.lr_seen[-d.bracket.trial_steps:]
    assert len(set(seen)) == 1, f'the rate moved during the trial: {sorted(set(seen))}'
    assert seen[0] == pytest.approx(m.args.lr_policy * 0.8)


def test_a_hard_failing_candidate_ends_its_trial_rather_than_cutting_its_rate():
    """A candidate that lowered its own rate and survived would have measured a
    rate the bracket never tested, and would then be selected as though it had
    held the one it was given."""
    m, d = _armed()
    m.lr_seen.clear()
    ok = d.run_trial(_trial(HOT, label='hot'))
    assert not ok, 'the hot rung was expected to trip the derived bar'
    seen = m.lr_seen[-len(m.lr_seen):]
    assert len(set(seen)) == 1, 'the candidate changed its own rate'
    out = d.bracket.results()[-1]
    assert out.steps_completed < d.bracket.trial_steps
    assert out.steps_to_failure == out.steps_completed


def test_successful_candidates_run_the_full_horizon():
    m, d = _armed()
    assert d.run_trial(_trial(0.05, label='cold'))
    out = d.bracket.results()[-1]
    assert out.steps_completed == d.bracket.trial_steps
    assert out.steps_to_failure is None


# ================================================ the optimizer counter ======

def test_the_restored_adam_step_counter_must_match_the_saved_value():
    """Adam's update carries sqrt(1-beta2^t)/(1-beta1^t), so a trial from a reset
    counter runs at 15-30% of its nominal rate for its first hundred steps -- a
    too-hot rung then survives for a reason that has nothing to do with the rate.
    The restore reads the counter BACK rather than trusting that it was carried."""
    m, d = _armed()
    corrupted = copy.deepcopy(d.root.adam_t)
    d.root.adam_t = {k: (v or 0) + 5 for k, v in corrupted.items()}
    with pytest.raises(RuntimeError, match='step counter'):
        d.root.restore()


def test_a_missing_optimizer_state_raises_rather_than_printing():
    """Outside a bracket, `load_optimizer_state`'s two fallbacks are legitimate
    recovery. Inside a trial they are a corrupted measurement that looks normal,
    so the bracket passes strict=True."""
    m, d = _armed()
    d.root.optimizers.pop(m.train_key)
    with pytest.raises(KeyError, match="no saved optimizer state"):
        d.root.restore()


def test_an_unrestorable_optimizer_state_raises_rather_than_printing():
    m, d = _armed()
    d.root.optimizers[m.train_key] = {'state': {}, 'param_groups': []}
    with pytest.raises(RuntimeError, match='could not restore optimizer state'):
        d.root.restore()


def test_the_non_strict_path_still_recovers():
    """Mutation guard for the two above: `strict` has to be what changed, not the
    method. A resume outside a bracket must still limp on rather than refuse to
    start."""
    m, _ = _armed()
    m.checkpointer.load_optimizer_state({'optimizers': {}}, strict=False)   # no raise


def test_bias_correction_is_the_adam_formula():
    """The table the burn-in length is chosen from, pinned so a refactor cannot
    quietly change what `min_root_bias_correction` means."""
    for t, want in ((10, 0.153), (100, 0.309), (500, 0.627),
                    (1000, 0.795), (3000, 0.975)):
        assert bias_correction(t, 0.9, 0.999) == pytest.approx(want, abs=5e-3)
    assert bias_correction(None, 0.9, 0.999) == 0.0
    assert bias_correction(0, 0.9, 0.999) == 0.0


def test_bracketing_is_refused_when_the_root_is_not_at_steady_state():
    """Computed from the optimizer's REAL step counter, not from the configured
    burn-in length -- a stage that did not rebuild its optimizers enters with t
    already large, and that passing trivially is correct rather than a loophole."""
    m = FakeTrainer(lr_control=_control(burn_in_steps=8,
                                        min_root_bias_correction=0.9))
    m.burn_in(8)
    c = m.lr_controller
    c.bars.derive(c._loss_history, c._grad_history)
    d = BracketDriver(m, c.bracket, c.bars, verbose=False)
    refusal = d.take_root(m.step_ind)
    assert refusal is not None and 'bias correction' in refusal
    assert 'burn_in_steps' in refusal, 'the refusal must name the knob that fixes it'


def test_a_mature_root_passes_the_check():
    """Mutation guard: a refusal that fires on every root is not a check."""
    m = FakeTrainer(lr_control=_control(burn_in_steps=60,
                                        min_root_bias_correction=0.2))
    m.burn_in(60)
    c = m.lr_controller
    c.bars.derive(c._loss_history, c._grad_history)
    d = BracketDriver(m, c.bracket, c.bars, verbose=False)
    assert d.take_root(m.step_ind) is None


# ================================================== the derived bars =========

def test_the_bars_are_derived_from_the_root_and_can_fire():
    """The load-bearing bar. Absolute bars at 1e6 catch nothing on a loss that
    lives at O(1); the span bar sits just above the root's own range."""
    m, d = _armed()
    bar = d.bars.loss_bar[m.train_key]
    root_losses = m.losses[:m.lr_controller.bracket.burn_in_steps]
    assert bar > max(root_losses)
    assert d.bars.judge(m.train_key, max(root_losses), None) is None
    assert d.bars.judge(m.train_key, bar * 1.01, None) is not None


def test_the_span_bar_is_well_defined_on_a_signed_channel():
    """The reason it is a SPAN and not a ratio: on the MLE channel the loss
    passes through zero and goes negative, so a running minimum has no positive
    scale to take a ratio against and the ratio rule correctly declines. A run
    with no bar is a bracket that cannot fail a candidate."""
    bars = HardFailureBars(loss_excursion_k=10.0, min_observations=3)
    assert bars.derive({'bwd': [-25.0, -24.0, -26.0, -25.5]}, []) is None
    bar = bars.loss_bar['bwd']
    assert bar == pytest.approx(-24.0 + 10.0 * 2.0)
    assert bars.judge('bwd', -20.0, None) is None
    assert bars.judge('bwd', 318.0, None) is not None, (
        'the -25 -> +318 excursion this route actually produced must fail')


def test_a_root_too_short_to_derive_a_bar_refuses_rather_than_bracketing():
    """A bracket whose bars cannot fire finds no boundary, reports
    unbracketed_high every cycle, and returns the same answer forever while
    every seam fires correctly."""
    bars = HardFailureBars(min_observations=20)
    why = bars.derive({'fwd': [1.0, 2.0]}, [])
    assert why is not None and 'not a bracket' in why


# ============================================ promotion and the clock ========

def test_the_promoted_continuation_restores_the_winners_end_state():
    m, d = _armed()
    root_step = m.step_ind
    promoted = d.run(root_step)

    v = d.bracket._verdict
    assert v['status'] == BRACKETED, v
    assert promoted == d.bracket.trial_steps
    assert m.step_ind == root_step + d.bracket.trial_steps, (
        'the promoted clock advanced by one horizon, not by the sum of all trials')
    assert m.lr_controller.scale == v['scale']
    assert d.bracket.phase == CRUISE


def test_discarded_trial_compute_does_not_advance_the_promoted_clock():
    """Four rungs at 12 steps is 48 steps of compute. The run keeps 12 of them --
    the winner's -- and the clock has to say so, or the eval grid, the checkpoint
    cadence and the repeat clock all drift by the bracket's own cost."""
    m, d = _armed()
    root_step = m.step_ind
    spent_before = len(m.losses)
    d.run(root_step)
    spent = len(m.losses) - spent_before
    assert spent > d.bracket.trial_steps, 'the bracket did not run several trials'
    assert m.step_ind - root_step == d.bracket.trial_steps


def test_promotion_falls_back_to_the_root_when_everything_fails():
    m, d = _armed(candidate_scales=(HOT, 4 * HOT, 16 * HOT))
    root_step = m.step_ind
    promoted = d.run(root_step)
    v = d.bracket._verdict
    assert v['status'] == ALL_FAILED
    assert promoted == 0 and m.step_ind == root_step
    assert m.lr_controller.scale == d.bracket.burn_in_scale


def test_the_driver_releases_its_held_state_after_promoting():
    """A six-rung bracket on the crystal route holds a couple of gigabytes of
    host memory describing rates the run is not going to use."""
    m, d = _armed()
    d.run(m.step_ind)
    assert d.root is None and not d.trial_states


# ============================================== the confirmation seed ========

def test_a_confirmation_rerun_is_not_a_bit_identical_replay():
    """Screen trials restore an identical RNG state so they are comparable,
    which makes a same-seed re-run a deterministic replay: it reproduces the
    original outcome by construction, confirms nothing, and reads as a passing
    check. The seed is the ONLY thing that differs."""
    m, d = _armed()
    d.run_trial(_trial(0.2, label='screen'))
    screen = list(m.losses[-d.bracket.trial_steps:])

    d.run_trial(_trial(0.2, label='replay'))
    replay = list(m.losses[-d.bracket.trial_steps:])
    assert replay == screen, 'fixture assumption: same seed IS a replay'

    seed = d.bracket.confirm_seed(1)
    d.run_trial(_trial(0.2, seed=seed, label='confirm'))
    confirmed = list(m.losses[-d.bracket.trial_steps:])
    assert confirmed != screen, (
        'the confirmation reproduced the screen exactly, so it is a replay and '
        'confirms nothing')
    # ...and it is reproducible: the same derived seed gives the same run again.
    d.run_trial(_trial(0.2, seed=seed, label='confirm2'))
    assert list(m.losses[-d.bracket.trial_steps:]) == confirmed


# ================================================== the control arm ==========

def test_no_managed_rate_means_no_trials_are_run():
    """The documented control arm: every lr_* key is an explicit float, so the
    scale reaches no optimizer. Spending four rungs of discarded training to
    find a boundary on a multiplier nothing applies would be pure waste."""
    m = FakeTrainer(lr_control=_control())
    m.args.lr_servo_managed = ()
    m.lr_controller = LRController(m)
    m.burn_in(m.lr_controller.bracket.burn_in_steps)
    spent = len(m.losses)
    m.step_ind += 1
    assert m.lr_controller.tick() == 0
    assert len(m.losses) == spent, 'a control arm ran trials'
    assert m.lr_controller.bracket.refusal is not None
    assert m.lr_controller.bracket.phase == CRUISE


# ================================================== the whole loop ===========

def test_the_host_seat_burns_in_then_brackets_exactly_once():
    """End to end through the seat the host loop actually calls: burn in for
    exactly the configured number of steps at the burn-in scale, then bracket,
    then hold. `tick` returns the promoted horizon for the caller's clock."""
    m = FakeTrainer(lr_control=_control(burn_in_steps=40))
    c = m.lr_controller
    skips = []
    for _ in range(80):
        m.step_ind += 1
        loss = m.train_step(m.train_key)
        c.observe(m.train_key, loss, m.last_grad_norm_pre_clip)
        skips.append(c.tick())
        if c.bracket.phase == CRUISE:
            break

    assert sum(1 for s in skips if s) == 1, 'the bracket ran more than once'
    fired_at = next(i for i, s in enumerate(skips) if s)
    assert fired_at + 1 == 40, (
        f'burn-in ran {fired_at + 1} steps, not the configured 40')
    assert c.bracket.phase == CRUISE and c.bracket.promoted_scale is not None
    assert c.scale == c.bracket.promoted_scale


def test_a_repeat_takes_the_current_state_as_a_new_root():
    """No second burn-in: the run has been training at a promoted rate, so the
    optimizers are at steady state by construction."""
    m = FakeTrainer(lr_control=_control(burn_in_steps=40, repeat_every=20))
    c = m.lr_controller
    brackets = 0
    for _ in range(200):
        m.step_ind += 1
        loss = m.train_step(m.train_key)
        c.observe(m.train_key, loss, m.last_grad_norm_pre_clip)
        if c.tick():
            brackets += 1
        if brackets == 2:
            break
    assert brackets == 2, 'the repeat clock never fired a second bracket'
    assert c.bracket._brackets == 2


# ================================================== the abort path ===========

def test_a_single_oom_repeats_the_rung_and_the_race_completes():
    """AN OOM IS NOT A MEASUREMENT, AND NOT A BATCH CUT EITHER (owner,
    2026-08-25). Trials run at whatever batch the preceding cruise grew to, so
    the first OOM in a race says "the race's own overhead tipped a full card",
    not anything about the rate OR about the cruise batch. The rung is repeated
    once from the root at the SAME batch -- a rung measured at a smaller batch
    would not be comparable to the rungs already on the board -- and the shared
    recovery (batch cut + OOM ceiling) must NOT run: qm9c aug25 lost a whole
    23-minute six-rung ladder and its batch to a single transient OOM on c5."""
    m = FakeTrainer(lr_control=_control())
    m.burn_in(m.lr_controller.bracket.burn_in_steps)
    c = m.lr_controller
    root_step = m.step_ind
    m.raise_at = (COLD[1], RuntimeError('CUDA out of memory. Tried to allocate 2.00 GiB'))
    m.raise_once = True

    skip = c._open_bracket(root_step)

    assert m.oom_handled == [], (
        'a single race OOM reached the shared recovery path -- that cuts the '
        'batch and installs an OOM ceiling from race memory, not cruise memory')
    # the race decided: every rung measured, the OOMed one on its repeat
    assert c.bracket.phase == CRUISE
    assert c.bracket.refusal is None, c.bracket.refusal
    outcomes = c.bracket.results()
    assert [o for o in outcomes if abs(o.trial.scale - COLD[1]) < 1e-12], (
        'the OOMed rung never produced a measurement')
    assert skip == c.bracket.trial_steps
    assert math.isfinite(m.train_step(m.train_key))


def test_a_second_oom_in_one_race_aborts_the_bracket_without_killing_the_run():
    """THE SEAT IS OUTSIDE THE HOST LOOP'S RECOVERY. `train.py` wraps
    `train_step` in `try/except (RuntimeError, ValueError)`; the controller's
    tick runs after it, so anything raised inside a trial leaves the training
    loop entirely and the job dies.

    Two OOMs in one race exhaust the retry budget: that is a VRAM squeeze or a
    leak, not rate evidence, and racing on at a reduced batch would put
    incomparable rungs on one board. The second OOM has to reach the shared
    recovery path (the run must cruise on afterwards), put the trainer back on
    the root, and leave the run training at the kept rate."""
    m = FakeTrainer(lr_control=_control())
    m.burn_in(m.lr_controller.bracket.burn_in_steps)
    c = m.lr_controller
    root_step = m.step_ind
    # persistent: the retry of the same rung OOMs again -> budget exhausted
    m.raise_at = (COLD[1], RuntimeError('CUDA out of memory. Tried to allocate 2.00 GiB'))

    skip = c._open_bracket(root_step)

    from protocol import TRAIN_MODES
    assert len(m.oom_handled) == 1, 'the aborting OOM did not reach the shared recovery path'
    kind, step_type = m.oom_handled[0]
    assert kind == 'RuntimeError'
    # THE STEP TYPE MUST BE A REAL TRAIN MODE, not a descriptive label.
    # `handle_train_epoch_error` gates BOTH halves of its recovery -- the batch
    # cut and the OOM ceiling -- on `step_type in TRAIN_MODES`, because an eval
    # OOM has a different memory profile and must not install a train ceiling.
    # Passing 'lr_bracket' skipped both and left the batch untouched, so the very
    # next attempt would OOM again.
    assert step_type in TRAIN_MODES, (
        f'{step_type!r} is not in protocol.TRAIN_MODES, so neither the batch cut '
        f'nor the OOM ceiling fires and the recovery is a no-op')
    assert step_type == m.protocol.stage.train_mode
    assert skip == 0
    assert m.step_ind == root_step, 'the trainer was left mid-trial'
    assert c.bracket.refusal and 'aborted' in c.bracket.refusal
    # first cycle: nothing promoted yet, so the kept rate IS the burn-in scale
    assert c.scale == c.bracket.burn_in_scale
    assert c.driver is None, 'the driver reference outlived the abort'
    # ...and the run can keep training rather than having died.
    assert math.isfinite(m.train_step(m.train_key))


def test_an_abort_on_a_repeat_cycle_keeps_the_promoted_rate():
    """THE FALLBACK IS THE KEPT RATE, NOT THE BURN-IN SCALE. On a repeat cycle
    the run has a rate a previous bracket measured and has been training at
    ever since; an abort that dropped to the burn-in scale would cost the run
    its operating point for no evidential reason. (The abort PRINT also said
    "burn-in scale" while the code held the promotion -- qm9c aug25 was
    misread as cruising 16x cold off exactly that message.)"""
    m = FakeTrainer(lr_control=_control(repeat_every=20))
    m.burn_in(m.lr_controller.bracket.burn_in_steps)
    c = m.lr_controller
    assert c._open_bracket(m.step_ind) > 0, 'the first race should promote'
    promoted = c.bracket.promoted_scale
    assert promoted is not None and promoted != c.bracket.burn_in_scale
    # second race: two OOMs -> abort; the promotion must survive it
    m.raise_at = (COLD[1], RuntimeError('CUDA out of memory. Tried to allocate 2.00 GiB'))
    c._open_bracket(m.step_ind)
    assert c.bracket.refusal and 'aborted' in c.bracket.refusal
    assert c.scale == promoted, (
        f'the abort dropped the rate to {c.scale} instead of keeping the '
        f'promoted {promoted}')


def test_a_non_oom_exception_fails_the_candidate_rather_than_the_bracket():
    """An exception that makes ONE continuation unusable is a hard failure of
    that candidate -- which is what the failure contract says -- not a reason to
    abandon the measurement."""
    m, d = _armed()
    m.raise_at = (COLD[1], ValueError('the policy produced an invalid sample'))
    ok = d.run_trial(_trial(COLD[1], label='bad'))
    assert not ok
    assert d.bracket.results()[-1].reason == 'exception_ValueError'
    assert m.oom_handled == []


def test_fixed_mode_still_derives_a_hard_failure_bar():
    """Fixed mode chooses the rate from outside; it does not make the rate SAFE.
    Without a derived bar the whole stage's only guard is the absolute backstop,
    which is what caught nothing on this route when the loss ran -25 -> +318."""
    m = FakeTrainer(lr_control=_control(mode='fixed', fixed_scale=0.2,
                                        burn_in_steps=40))
    c = m.lr_controller
    for _ in range(40):
        m.step_ind += 1
        loss = m.train_step(m.train_key)
        c.observe(m.train_key, loss, m.last_grad_norm_pre_clip)
        c.tick()
    assert c.bracket.phase == CRUISE and c.scale == 0.2
    assert m.train_key in c.bars.loss_bar, 'fixed mode ran the stage with no derived bar'
    assert c.bars.judge(m.train_key, c.bars.loss_bar[m.train_key] * 1.01, None) is not None


# ============================ the review findings, pinned by name ============

def test_the_bracket_runs_when_only_one_managed_optimizer_steps():
    """FINDING A, and it was fatal: the bracket refused on EVERY stage of EVERY
    canonical config while reporting the refusal as caution.

    `mk_dev` writes all four policy rates `auto`, so the managed set is
    {fwd, bwd, replay, fused}. A stage runs ONE train_mode, so three of those
    four never take a step and `_adam_t` returns None -- which the check
    correctly maps to a bias correction of 0.0. Taking the WORST over the managed
    set therefore gave 0.0 always, under any `min_root_bias_correction`.

    The check must range over the optimizers this stage actually STEPS."""
    m, d = _armed()
    c = m.lr_controller
    assert c.managed_optimizer_keys() == {'fwd', 'bwd', 'replay', 'fused'}
    assert c.stepping_optimizer_keys() == {'bwd'}, (
        'only the stage train_mode optimizer steps')

    factor, t, key = d.root_bias_correction()
    assert key == 'bwd', f'the check read optimizer {key!r}, which this stage never steps'
    assert t is not None and t > 0, 'the reading came from an optimizer with no state'
    assert factor > 0.0, (
        f'bias correction is {factor} -- 0.0 is what an unstepped optimizer gives, '
        f'and it refuses every bracket under any min_root_bias_correction')
    assert d.take_root(m.step_ind) is None, 'the bracket refused a mature root'

    # ...and the spectators really are unstepped, so this is the bug's exact shape.
    from lr_bracket_probe import _adam_t
    for spectator in ('fwd', 'replay', 'fused'):
        assert _adam_t(m.optimizers[spectator]) is None


def test_a_resume_into_cruise_re_derives_the_hard_failure_bars():
    """FINDING B. `bars.derive` runs once, inside `_open_bracket`. A run resumed
    mid-cruise is past burn-in and past the bracket, so it never called it -- and
    the derived excursion bar is the only tripwire that can fire on this route.
    The whole remaining stage would train on the absolute backstops alone, which
    is the 1e9 situation this design replaced."""
    m = FakeTrainer(lr_control=_control(burn_in_steps=30))
    c = m.lr_controller
    # a resume: the bracket state says CRUISE, and this process has derived nothing
    c.bracket.phase = CRUISE
    c.bracket.promoted_scale = 0.2
    c.bracket.promoted_at = 0
    assert not c.bars.loss_bar, 'fixture assumption: no bars yet'

    for _ in range(40):
        m.step_ind += 1
        loss = m.train_step(m.train_key)
        c.observe(m.train_key, loss, m.last_grad_norm_pre_clip)
        c.tick()

    assert m.train_key in c.bars.loss_bar, (
        'a resumed run reached cruise with no derived hard-failure bar')
    assert c.bars.judge(m.train_key, c.bars.loss_bar[m.train_key] * 1.01, None)


def test_an_impossible_observation_window_is_refused_at_construction():
    """FINDING F. The window is a ring of `root_window` entries, so
    `min_observations` above it can never be met: burn-in never ends, no bracket
    ever runs, and nothing says so. An unbounded wait is exactly what this design
    does not have anywhere else."""
    with pytest.raises(ValueError, match='can never be met'):
        FakeTrainer(lr_control=_control(
            hard_failure={'root_window': 20, 'min_observations': 50}))


def test_the_bracket_state_survives_a_trial_restore():
    """FINDING G. `lr_ctrl` is in `TrainerSnapshot.FIELDS`, so every trial
    restore REPLACES `modeller.lr_ctrl`. `tick` captured that dict before running
    the bracket, so its verdict was written into an orphan -- and a checkpoint
    taken on that iteration recorded a bracket that never happened."""
    m = FakeTrainer(lr_control=_control(burn_in_steps=40))
    c = m.lr_controller
    for _ in range(40):
        m.step_ind += 1
        loss = m.train_step(m.train_key)
        c.observe(m.train_key, loss, m.last_grad_norm_pre_clip)
        c.tick()

    assert c.bracket.phase == CRUISE
    persisted = m.lr_ctrl.get('bracket')
    assert persisted is not None, 'nothing was persisted'
    assert persisted['phase'] == CRUISE, (
        f"the LIVE lr_ctrl records phase {persisted['phase']!r}; the verdict went "
        f'into a dict the trial restore had already orphaned')
    assert persisted['promoted_scale'] == c.bracket.promoted_scale
    assert m.lr_ctrl['scale'] == c.scale


# ================================ the bars are drawn twice ==================
#
# The root bars are fitted to burn-in, at burn_in_scale. That is the right scale
# for judging a TRIAL -- every trial restores that same root, so it is comparable
# to it by construction. It is the WRONG scale for the live tripwire, which then
# holds for the rest of the stage at a promoted rate up to the top of the grid: a
# hotter rate moves the loss more for ordinary reasons, and the response to a
# crossing is a rewind charged to max_reloads_per_1k_steps.


def _to_cruise(m, cap=400):
    """Drive the host seat until a rate is promoted. Returns the steps taken."""
    c = m.lr_controller
    for i in range(cap):
        m.step_ind += 1
        loss = m.train_step(m.train_key)
        c.observe(m.train_key, loss, m.last_grad_norm_pre_clip)
        c.tick()
        if c.bracket.phase == CRUISE:
            return i + 1
    raise AssertionError('never promoted')


def _cruise_for(m, n):
    """n ordinary cruise steps through the host seat. Returns the losses the
    live bars fired on -- EITHER tier: a finite excursion is a moderate fire
    (handled inside observe, which then returns None) and non-finite/absolute
    hits return 'diverged', so counting the return value alone went blind to
    the moderate tier the day the two-tier response landed."""
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


def _hot_seat(**kw):
    """A trainer whose promoted rate is far above its burn-in rate, so the two
    genuinely have different loss scales."""
    hf = dict(root_window=25, min_observations=10, cruise_settle_steps=10)
    hf.update(kw.pop('hard_failure', {}))
    return FakeTrainer(lr_control=_control(burn_in_steps=60, hard_failure=hf, **kw))


def test_the_cold_bar_is_the_one_the_promoted_rate_would_be_judged_by():
    """THE PREMISE, MEASURED ON THE FIXTURE rather than asserted in prose.

    If the loss at the promoted rate sat comfortably under a bar fitted at the
    burn-in rate, none of the machinery below would be worth its 400 steps. This
    pins that it does not: ordinary post-promotion losses exceed the root bar, so
    keeping that bar live is a stream of false rewinds.
    """
    m = _hot_seat()
    c = m.lr_controller
    _to_cruise(m)
    root_bar = (c._cruise_bar or {}).get('prev_loss', {}).get(m.train_key)
    assert root_bar is not None, 'no root bar was ever derived'
    assert c.scale > c.bracket.burn_in_scale * 4, (
        f'the fixture promoted {c.scale} against a burn-in scale of '
        f'{c.bracket.burn_in_scale} -- not a rate change worth refitting for')
    # The same horizon the mutation test below counts firings over, and stepped
    # WITHOUT the controller, so nothing here is refitting or suspending: this
    # asks only what the trainer does at the promoted rate.
    healthy = [m.train_step(m.train_key) for _ in range(120)]
    over = [v for v in healthy if v >= root_bar]
    assert over, (
        f'ordinary training at the promoted rate peaks at {max(healthy):.4g}, '
        f'under the root bar {root_bar:.4g} -- the premise does not hold on this '
        f'fixture and every test below would pass for the wrong reason')


def test_the_bars_are_refitted_at_the_promoted_rate():
    m = _hot_seat()
    c = m.lr_controller
    _to_cruise(m)
    root_bar = c._cruise_bar['prev_loss'][m.train_key]
    assert not c._bars_redrawn

    _cruise_for(m, 10 + 25 + 5)
    assert c._cruise_bar is None, 'the refit never completed'
    assert c._bars_redrawn
    new_bar = c.bars.loss_bar[m.train_key]
    assert new_bar > root_bar, (
        f'refitted bar {new_bar:.4g} is not above the cold one {root_bar:.4g} -- '
        f'the refit did not use post-promotion observations')


def test_keeping_the_cold_bar_turns_healthy_training_into_rewinds():
    """THE MUTATION TEST. `cruise_rederive: false` is the shipped behaviour this
    change replaces, and it must FAIL here -- otherwise the fix is unobservable
    and every test above passes on a fixture that cannot tell the two apart."""
    def firings(rederive):
        # cut factor 1.0 + cooldown 0 NEUTRALIZE the fire response, so this
        # measures the bar-refit question alone: an actual cut would change the
        # rate mid-count and the fixture's loss floor with it.
        m = _hot_seat(hard_failure=dict(cruise_rederive=rederive),
                      fire_cut_factor=1.0, fire_cooldown_steps=0)
        _to_cruise(m)
        return len(_cruise_for(m, 120))

    with_fix, without_fix = firings(True), firings(False)
    assert without_fix > 0, (
        'the cold bar never fired even once, so this fixture cannot see the bug')
    assert with_fix * 4 < without_fix, (
        f'refitting the bars cut false divergences only from {without_fix} to '
        f'{with_fix} -- not a difference worth the machinery')


def test_the_settle_window_is_discarded_rather_than_fitted():
    """The steps right after a rate change are the transient the change causes.
    Folding them into the window would fit the bar to the very movement that
    must not count as normal."""
    m = _hot_seat(hard_failure=dict(cruise_settle_steps=30))
    c = m.lr_controller
    _to_cruise(m)
    _cruise_for(m, 28)
    assert c._cruise_bar is not None and len(c._loss_history[m.train_key]) > 0, (
        'the window was cleared before the settle elapsed')
    _cruise_for(m, 2)
    assert len(c._loss_history[m.train_key]) <= 1, (
        'the burn-in observations were not dropped at the end of the settle')


def test_the_refit_clock_counts_cruise_steps_not_step_ind():
    """`_open_bracket` returns the winner's horizon and the host loop SKIPS that
    many steps, so `step_ind` jumps forward the instant a rate is promoted. A
    clock written as `step + settle` is in the past before the first cruise step
    runs, and the settle it enforces is skipped entirely."""
    m = _hot_seat(hard_failure=dict(cruise_settle_steps=40))
    c = m.lr_controller
    _to_cruise(m)
    m.step_ind += 10_000                       # the horizon skip, exaggerated
    _cruise_for(m, 38)
    assert c._cruise_bar is not None, 'no refit was pending to time'
    assert len(c._loss_history[m.train_key]) > 0, (
        'the settle was skipped -- the clock read step_ind, not cruise steps')


def test_a_tested_rate_suspends_the_cold_bar_and_an_untested_one_does_not():
    """Bracket mode promotes a rate that survived a full trial horizon, so the
    cold bar is the likelier source of a wrong answer and comes down. Fixed mode
    asserts its rate from outside with nothing testing it, so a too-tight bar
    beats no bar."""
    bracketed = _hot_seat()
    _to_cruise(bracketed)
    assert bracketed.lr_controller._cruise_bar['suspended'] is True
    assert not bracketed.lr_controller.bars.loss_bar, (
        'a trial-validated rate should not be judged by the burn-in bar')

    fixed = _hot_seat(mode='fixed', fixed_scale=3.2)
    _to_cruise(fixed)
    assert fixed.lr_controller._cruise_bar['suspended'] is False
    assert fixed.train_key in fixed.lr_controller.bars.loss_bar, (
        'fixed mode dropped its only derived guard on an untested rate')


def test_a_stage_change_clears_the_bars_and_not_only_the_window():
    """Clearing the observations but keeping the bars fitted to them left the
    outgoing stage's bars deciding rewinds through the incoming stage's burn-in
    -- a different train_mode, a different composite, and a loss scale that can
    differ by orders of magnitude."""
    m = _hot_seat()
    c = m.lr_controller
    _to_cruise(m)
    _cruise_for(m, 10 + 25 + 5)
    assert c.bars.loss_bar, 'nothing to clear -- the fixture never refitted'

    c.on_stage_change()
    assert c.bars.loss_bar == {} and c.bars.grad_bar is None, (
        'the previous stage bars survived into the next stage')
    assert c._cruise_bar is None and c._bars_redrawn is False


def test_a_refit_that_cannot_be_derived_restores_the_suspended_bars():
    """The deadline path with nothing to fit: the bars must come BACK, not stay
    suspended for the rest of the stage."""
    m = _hot_seat(hard_failure=dict(cruise_settle_steps=2))
    c = m.lr_controller
    _to_cruise(m)
    root = dict(c._cruise_bar['prev_loss'])
    assert root, 'no root bars were captured'
    # Tick the seat with NO observations -- the stage stopped producing this
    # channel -- until the deadline passes.
    for _ in range(2 + 4 * 25 + 2):
        m.step_ind += 1
        c.tick()
    assert c._cruise_bar is None, 'the refit never hit its deadline'
    assert c.bars.loss_bar == root, (
        'a refit with nothing to fit left the tripwire suspended')


def test_a_repeat_bracket_mid_refit_does_not_lose_the_suspended_bars():
    """`repeat_every` and the post-promotion refit are independent clocks. At the
    shipped 20000 against a ~400-step refit they cannot overlap, but nothing
    enforces that -- and a re-bracket that drops the stash while the bars are
    SUSPENDED leaves the stage on the absolute backstops alone if its own derive
    then refuses."""
    m = _hot_seat(hard_failure=dict(cruise_settle_steps=200), repeat_every=5)
    c = m.lr_controller
    _to_cruise(m)
    assert c._cruise_bar is not None and c._cruise_bar['suspended']
    assert not c.bars.loss_bar, 'the fixture is not in the suspended state'
    stashed = dict(c._cruise_bar['prev_loss'])

    c._cancel_pending_refit()
    assert c._cruise_bar is None
    assert c.bars.loss_bar == stashed, (
        'a cancelled refit left the tripwire suspended with nothing pending')
