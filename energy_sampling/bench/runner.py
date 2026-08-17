"""
ONE RUN: a game, an arm, a seed. Nothing else.

Rebuilt 2026-08-13 after an adversarial review found the previous harness had
accumulated defects faster than results. What is different, and why:

  * NO ORACLE. The old stack computed a "best fixed rate" and scored everything
    against it. That selection was wrong in three separate ways (a rate 187x off
    on one surface family, a grid-edge winner that bypassed its own guard, and a
    denominator pinned at the metric's floor), and every one of them silently
    rescaled results rather than failing. Fixed rates are now ARMS. A leaderboard
    needs no reference.

  * ADAM BY DEFAULT. Hypergradient's rule is derived from the SGD update
    (`d(theta)/d(lr) = -g`), production runs Adam on every optimizer
    (`train.py:1664`), and every cell of the old bench ran SGD. The derivation
    does not carry over -- under Adam the step direction is `mhat/(sqrt(vhat)+eps)`
    -- so measuring on SGD and recommending for Adam was never valid. SGD stays
    available as a control.

  * PAIRED SEEDS, WITH ONE BOUNDARY THEY DO NOT CROSS. A seed fixes the game's
    noise stream, so the same seed across arms is common random numbers and arm
    differences are not seed luck -- for every arm that only trains.

    IT IS FALSE FOR ANY ARM THAT PROBES. `RayCalibration.measure` draws through
    `run.game.draw`, which advances the game's generator, so a probing arm
    consumes extra randomness the non-probing arms never see. Measured: an arm
    identical to `HyperStep` except for a probe that never acts diverges from it
    at step 101 with a period-100 probe -- 480 extra draws over a 6000-step run.
    So `ray+ray` is NOT paired against `hyper`/`null`/`fixed`; it is paired only
    against other probing arms at the same cadence. Comparisons across that
    boundary carry seed noise the design was supposed to remove, which is worth
    remembering whenever a ray-vs-hyper gap is small.

  * THE REAL CONTROLLER, ON ALL THREE SENSORS. `ray` and `plateau` verdicts go
    through the shipping `LRController.on_calibration` / `on_plateau`, including
    the `ratio**eta` damping, bounds, ceiling and warmup hold. So does `hyper`:
    `LRController.on_hypergradient` ships (controller.py:237), `train.py:3089`
    calls it when a stage selects it, and `protocol.py:121` lists 'hyper' beside
    'ray' and 'plateau' as a configurable `lr_sensor` kind.

    THIS PARAGRAPH USED TO SAY THE OPPOSITE -- "there is no hypergradient in
    `controller.py`" -- and it was false when written; the sensor and the bench
    rebuild both landed on 2026-08-13. It was load-bearing, because it is what
    made the hyper rows read as bench code rather than as statements about a
    shipping, user-configurable sensor -- and hyper is FIRST on the SGD board.

    Still bench-only, and each for its own reason: `Hyper`'s OPERAND (previous
    gradient rather than realised displacement, which is the right statistic
    under SGD only, and is in no production path), `HyperStep`'s `target`
    setpoint, and `HyperSNR` entirely.
"""
import copy
import math

import torch


def _clone_opt_state(opt):
    """A DEEP copy of the optimizer state.

    `state_dict()` hands back live tensors -- Adam's exp_avg among them -- so a
    shallow save is a view of the state it is meant to protect, and the "restore"
    would write back the detonated moments it captured. The bug is invisible on
    SGD, which keeps no state at all."""
    return copy.deepcopy(opt.state_dict())

from bench.fake_modeller import FakeModeller, FakeStage, make_args
from bench.surfaces import SurfaceCannot, _Game

#: The unbound stubs, captured once. `type(game).expected_loss is _Game.<stub>`
#: is the only reliable "did this subclass override it" test -- `hasattr` sees
#: the stub and says yes.
_Game_expected_loss = _Game.expected_loss
_Game_distance_to_opt = _Game.distance_to_opt
from energy_sampling.controller import LRController
from energy_sampling.ray_calibration import RayCalibration

#: train.py's always-on tripwire bar (controller.py::check_spike). A run that
#: crosses it is a catastrophe and is COUNTED, never averaged into anything.
DIVERGENCE_BAR = 1.0e9

#: train.py:2155. The reload budget is a RATE, not a count, with a floor of 3 so
#: an early detonation still aborts. Exceeding it is the UNRECOVERABLE signal --
#: rewinding restores healthy weights but not a survivable rate, and a run that
#: keeps re-detonating must die rather than hold the GPU forever.
MAX_RELOADS_PER_1K = 0.2
RELOAD_FLOOR = 3.0

#: How often a healthy state is banked. Production rewinds to the last `running`
#: or `best` checkpoint, written on the eval cadence; this is the same idea on
#: the bench's clock.
SNAPSHOT_EVERY = 100


class Run:
    """
    One arm on one game, stepped to completion, recording a trace.

    The trace is the only output. Every metric in `bench/metrics.py` is a pure
    function of it, so a metric can be added or corrected without re-running
    anything -- which is what the old stack could not do.
    """

    def __init__(self, game, arm, seed=0, steps=2000, batch=64, rewind=True):
        self.game = game
        self.arm = arm
        self.seed = int(seed)
        self.steps = int(steps)
        self.batch = int(batch)
        # REWIND ON, because production's divergence response is a rewind AND a
        # peak cut and doing only the cut tests half of it -- the half that does
        # not recover. Off is available so a cell can measure what the rewind is
        # worth rather than assuming.
        self.rewind = bool(rewind)
        self.reloads = 0
        self.rewinds_refused = 0
        self._snap = None

        # make_args already layers the MK_DEV_* blocks; only the deltas go here.
        # `min_lr` is dropped to 1e-12 so the production floor cannot truncate a
        # fixed-rate arm at the bottom of the ladder and silently make it a
        # different rate than its own name.
        args = make_args(**{'min_lr': 1e-12, **arm.args_overrides()})
        # THE ARM DECLARES ITS SENSOR ON THE STAGE, exactly as a config stage
        # does, because that declaration is what arms the probe.
        self.m = FakeModeller(args, game.optimizers,
                              stage=FakeStage(lr_sensor=arm.lr_sensor()))
        rc = args.adaptive_lr.ray_calibration
        # ENABLED IS DERIVED, not configured -- train.py:1871 passes exactly
        # `bool(self._ray_askers())`. There is no `ray_calibration.enabled` to
        # set any more (utils._RETIRED_KEYS deletes both spellings), so the arm's
        # declaration and the probe's state cannot disagree. They could before,
        # and the failure was silent in the dangerous direction: an unarmed probe
        # returns no readings, the arm scores bit-identical to `null`, and the
        # board prints a plausible row. bench/test_arms.py counts armings and
        # resolutions for that reason.
        self.m.ray_cal = RayCalibration(
            game.policy_params, alphas=tuple(float(a) for a in rc.alphas),
            n_sub=rc.n_sub, period=rc.period,
            enabled=bool(self.m._ray_askers()))
        self.m.lr_controller = LRController(self.m)

        self.trace = []
        self.divergences = 0
        self.aborted = None

        # CAPABILITY IS DECIDED ONCE, HERE, AND LOUDLY.
        #
        # This used to be `hasattr(game, 'expected_loss')` per step, which is
        # True for every game alive -- `_Game` defines the stub. So a game
        # without a real one did not fall back, it RAISED on step 1, and
        # `VarCondGame` did exactly that for the whole life of the rebuild while
        # its tests passed (they never construct a `Run`).
        #
        # Scoring must not silently fall back either. With `eloss` absent for
        # every step, `metrics._series` reverts to the NOISY training loss --
        # the ranking-on-a-coin-flip trap `expected_loss` exists to prevent --
        # and nothing in the output says which loss produced the table. A
        # surface that cannot be scored is a bug in the surface, not a run to
        # score differently.
        cls = type(game)
        if cls.expected_loss is _Game_expected_loss:
            raise SurfaceCannot(
                f'{cls.__name__} does not implement expected_loss, so this run '
                f'could only be scored on the noisy training loss. Implement it '
                f'(the noise-free loss at the current parameters) rather than '
                f'letting the scoring fall back.')
        self._has_dist = cls.distance_to_opt is not _Game_distance_to_opt
        arm.reset(self)

    # ------------------------------------------------------------------ step

    def step(self):
        m, game = self.m, self.game
        # move any non-stationary part of the surface FIRST, so this step's
        # gradient is measured against the target as it now stands
        game.advance()
        batch = game.draw(self.batch)

        # BEFORE the step: the ray probe snapshots parameters here. train.py
        # arms at 2304 and measures at 2324, either side of the step, for the
        # same reason -- the probe rates the step that just happened.
        self.arm.pre_step(self)

        loss = game.train_step(batch)
        g_before = None
        if self.arm.needs_gradient:
            g_before = torch.cat([p.grad.detach().reshape(-1)
                                  for p in game.policy_params
                                  if p.grad is not None]).float()
        grad_norm = float(g_before.norm()) if g_before is not None else None

        # --- train.py's step body ORDER, which the old harness inverted:
        #     1992 batch sizer | 1996 lr_controller.step() | 1998 monitor_losses
        #     (check_spike at 2082) | 2007 on_plateau. The batch sizer is not exercised here.
        if m.step_ind % 10 == 0:
            m.lr_controller.step()

        # train.py calls check_spike from monitor_losses, inside `% 10 == 0`.
        # The old harness called it EVERY step, which let the reload budget be
        # consumed 10x faster than production allows.
        if m.step_ind % 10 == 0:
            if self._diverged(loss, grad_norm):
                self.divergences += 1
                self._fire_loss_spike()
            elif self.rewind and m.step_ind % SNAPSHOT_EVERY == 0:
                # bank a HEALTHY state only. Snapshotting unconditionally would
                # let the blow-up itself become the rewind target, which is the
                # one thing the mechanism must never do.
                self._snapshot()

        self.arm.tick(self, loss=loss, g_before=g_before, batch=batch)

        # TWO LOSSES, DELIBERATELY. `loss` is the noisy training loss -- what the
        # controller sees and acts on, as in production. `eloss` is the same loss
        # with the minibatch noise term zeroed: the quality of the parameters,
        # which is what the SCORING uses. Near the optimum the noise term is all
        # that is left of `loss` and its sign is random, so ranking arms on it
        # ranks them on a coin flip.
        self.trace.append({
            'step': m.step_ind,
            # THE OPTIMIZER THIS GAME ACTUALLY STEPS, not a hardcoded 'fwd'.
            # MLEGame trains 'fwd' so the two coincided; EquilibrationGame trains
            # 'fused', and recording 'fwd' there reports a SPECTATOR optimizer --
            # a rate nothing is training with. train.py has the same five-key
            # dict and the same trap, which is why `train_key` exists.
            'lr': m.lr_of(game.train_key),
            'loss': loss,
            'eloss': game.expected_loss(),
            'grad_norm': grad_norm,
            'dist': game.distance_to_opt() if self._has_dist else None,
        })
        m.step_ind += 1

    # ---------------------------------------------------------- divergence

    def _all_params(self):
        """Every trainable tensor, not just `policy_params`.

        The ray probe snapshots policy only (decision D26b), but a REWIND that
        restored only the policy would leave the level head at its detonated
        value and re-explode on the next step -- the two are coupled, which is
        the entire premise of the equilibration game."""
        seen, out = set(), []
        for opt in self.game.optimizers.values():
            for grp in opt.param_groups:
                for p in grp['params']:
                    if id(p) not in seen:
                        seen.add(id(p))
                        out.append(p)
        return out

    def _snapshot(self):
        self._snap = {
            'params': [p.detach().clone() for p in self._all_params()],
            # load_model_only(..., load_optimizers=True) at train.py:2194 -- the
            # optimizer state RIDES ALONG. Leaving Adam's moments at their
            # detonated values while restoring the weights is its own documented
            # failure mode, so this follows production rather than guessing.
            'opt': {k: _clone_opt_state(o)
                    for k, o in self.game.optimizers.items()},
            'extra': self.game.extra_state(),
            'step': self.m.step_ind,
        }

    def _fire_loss_spike(self):
        """
        train.py::fire_loss_spike, on the bench's clock: rewind to the last
        healthy state AND cut the peak, with the same rate-based budget and the
        same abort when the budget is gone.

        The pairing is the point. A reload without a cut re-enters the same
        state at the same rate and explodes again; a cut without a reload keeps
        the damaged weights. A fixed rate cannot be cut at all -- nothing manages
        it -- so on this bench that arm is exactly the case the budget exists for
        and it will abort, which is the correct outcome rather than a gap.
        """
        m = self.m
        if not self.rewind:
            m.lr_controller.on_divergence()
            return
        self.reloads += 1
        cap = max(RELOAD_FLOOR, MAX_RELOADS_PER_1K * m.step_ind / 1000.0)
        if self.reloads > cap:
            self.aborted = (f'unrecoverable: {self.reloads} rewinds at step '
                            f'{m.step_ind} (budget {cap:.1f})')
            return
        if self._snap is None:
            # Reachable and NOT a bug: production hits it whenever a run
            # detonates before the first checkpoint exists, and used to fall
            # through in silence. Take the half of the response available.
            self.rewinds_refused += 1
            m.lr_controller.on_divergence()
            return
        for p, saved in zip(self._all_params(), self._snap['params']):
            with torch.no_grad():
                p.copy_(saved)
            p.grad = None
        for k, state in self._snap['opt'].items():
            self.game.optimizers[k].load_state_dict(state)
        self.game.load_extra_state(self._snap['extra'])
        # step_ind is NOT rewound: train.py restores the checkpoint and then
        # puts the live step back (2197-2200). Time does not run backwards, and
        # a rewound clock would reset the reload budget it is measured against.
        m.lr_controller.on_divergence()
        self.arm.on_rewind(self)

    def _diverged(self, loss, grad_norm):
        for v in (loss, grad_norm):
            if v is None:
                continue
            if not math.isfinite(v) or abs(v) >= DIVERGENCE_BAR:
                return True
        return False

    def run(self):
        for _ in range(self.steps):
            self.step()
            if self.aborted:
                break
        return self
