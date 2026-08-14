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
from energy_sampling.controller import LRController
from energy_sampling.ray_calibration import RayCalibration

#: train.py's always-on tripwire bar (controller.py::check_spike). A run that
#: crosses it is a catastrophe and is COUNTED, never averaged into anything.
DIVERGENCE_BAR = 1.0e9

#: train.py:2130. The reload budget is a RATE, not a count, with a floor of 3 so
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
            'eloss': (game.expected_loss() if hasattr(game, 'expected_loss')
                      else None),
            'grad_norm': grad_norm,
            'dist': game.distance_to_opt() if hasattr(game, 'distance_to_opt')
                    else None,
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
            # load_model_only(..., load_optimizers=True) at train.py:2169 -- the
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
        # puts the live step back (2172-2175). Time does not run backwards, and
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
