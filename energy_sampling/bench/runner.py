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
    (`train.py:1647`), and every cell of the old bench ran SGD. The derivation
    does not carry over -- under Adam the step direction is `mhat/(sqrt(vhat)+eps)`
    -- so measuring on SGD and recommending for Adam was never valid. SGD stays
    available as a control.

  * PAIRED SEEDS. A seed fixes the game's noise stream, so the same seed across
    arms is common random numbers: arm differences are not seed luck. This is
    the one thing from the old harness worth keeping unchanged.

  * THE REAL CONTROLLER, and an honest account of where that stops. `ray` and
    `plateau` verdicts go through the shipping `LRController.on_calibration` /
    `on_plateau`, including its `ratio**eta` damping, bounds, ceiling and warmup
    hold. `hyper` has NO production counterpart -- there is no hypergradient in
    `controller.py` -- so it writes `peak_scale` through the same actuator
    (`_apply_lrs`) and the same bounds, but its update law is bench code. The old
    harness header claimed all arms "drive the SAME actuator ... only the source
    of the verdict changes"; that was false, and pretending the laws are
    interchangeable is what made the comparison look tighter than it was.
"""
import math

import torch

from bench.fake_modeller import FakeModeller, FakeStage, make_args
from energy_sampling.controller import LRController
from energy_sampling.ray_calibration import RayCalibration

#: train.py's always-on tripwire bar (controller.py::check_spike). A run that
#: crosses it is a catastrophe and is COUNTED, never averaged into anything.
DIVERGENCE_BAR = 1.0e9


class Run:
    """
    One arm on one game, stepped to completion, recording a trace.

    The trace is the only output. Every metric in `bench/metrics.py` is a pure
    function of it, so a metric can be added or corrected without re-running
    anything -- which is what the old stack could not do.
    """

    def __init__(self, game, arm, seed=0, steps=2000, batch=64):
        self.game = game
        self.arm = arm
        self.seed = int(seed)
        self.steps = int(steps)
        self.batch = int(batch)

        # make_args already layers the MK_DEV_* blocks; only the deltas go here.
        # `min_lr` is dropped to 1e-12 so the production floor cannot truncate a
        # fixed-rate arm at the bottom of the ladder and silently make it a
        # different rate than its own name.
        args = make_args(**{'min_lr': 1e-12, **arm.args_overrides()})
        self.m = FakeModeller(args, game.optimizers, stage=FakeStage())
        rc = args.ray_calibration
        self.m.ray_cal = RayCalibration(
            game.policy_params, alphas=tuple(float(a) for a in rc.alphas),
            n_sub=rc.n_sub, period=rc.period, enabled=rc.enabled)
        self.m.lr_controller = LRController(self.m)

        self.trace = []
        self.divergences = 0
        self.aborted = None
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
        #     1966 batch sizer | 1971 lr_controller.step() | 1973 check_spike
        #     | 1982 on_plateau. The batch sizer is not exercised here.
        if m.step_ind % 10 == 0:
            m.lr_controller.step()

        # train.py calls check_spike from monitor_losses, inside `% 10 == 0`.
        # The old harness called it EVERY step, which let the reload budget be
        # consumed 10x faster than production allows.
        if m.step_ind % 10 == 0:
            if self._diverged(loss, grad_norm):
                self.divergences += 1
                m.lr_controller.on_divergence()

        self.arm.tick(self, loss=loss, g_before=g_before, batch=batch)

        # TWO LOSSES, DELIBERATELY. `loss` is the noisy training loss -- what the
        # controller sees and acts on, as in production. `eloss` is the same loss
        # with the minibatch noise term zeroed: the quality of the parameters,
        # which is what the SCORING uses. Near the optimum the noise term is all
        # that is left of `loss` and its sign is random, so ranking arms on it
        # ranks them on a coin flip.
        self.trace.append({
            'step': m.step_ind,
            'lr': m.lr_of('fwd'),
            'loss': loss,
            'eloss': (game.expected_loss() if hasattr(game, 'expected_loss')
                      else None),
            'grad_norm': grad_norm,
            'dist': game.distance_to_opt() if hasattr(game, 'distance_to_opt')
                    else None,
        })
        m.step_ind += 1

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
