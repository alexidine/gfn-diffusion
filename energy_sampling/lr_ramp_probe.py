"""
Trainer side of the checkpointed LR ramp -- section 7 of
`docs/design/lr_handoff_2026-08-21.md`. The decision half is `lr_ramp.py` and has
no torch; this module executes the actions it emits.

THE CLEAN RUNG LIVES IN HOST RAM, NOT IN THE RUN'S TAG NAMESPACE. That is
section 8b, taken at its own second option and for its own reason: `best` is a
hardlink to whatever `running` last wrote, and the emergency rewind path reads
`best` and `stage_start` to choose its target -- so a ramp writing either would
edit the run's own rollback target while the ramp is the thing most likely to
need one. `checkpoint_read_only` is likewise never a question, because nothing is
written.

The snapshot/restore is the machinery the retired race probe proved: a duplicate
arm re-run from a snapshot reproduced the incumbent BITWISE, `duplicate_spread
0.000e+00`, on CPU and on the GPU (handoff section 2). That measurement is what
licenses rolling a rejected rung back rather than merely lowering the rate and
carrying its damage forward.

WHAT A ROLLBACK COSTS, stated rather than discovered (section 8e): the rejected
rung's steps are genuinely discarded -- one rung's residence, once, since one
rejection is enough to bracket the boundary and the design forbids continuing
past it in search of catapult recovery.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch

from energy_sampling.lr_ramp import (CLIMB, DESCEND, DWELL, FINISH, ROLLBACK,
                                     SAVE_CLEAN, RampLadder)


class RampDriver:
    """Executes one `RampLadder` against a live trainer."""

    def __init__(self, modeller, ladder: RampLadder, verbose: bool = True):
        self.m = modeller
        self.ladder = ladder
        self.verbose = bool(verbose)
        self.clean_snapshot = None
        self.n_rollbacks = 0
        self.n_snapshots = 0
        self._baseline = None          # coherence reference for the current rung

    # ------------------------------------------------------------- snapshots

    def _stepping_optimizers(self):
        """Exactly the optimizers this stage's train_mode drives.

        Mirrors `step_loss`: a bwd stage steps 'bwd' then 'flow'; a fused stage
        steps 'fused' alone (its param groups already carry the flow head at
        lr_flow). Getting this list wrong is the way a rollback silently leaves
        one optimizer's moments from the rejected rung in place.
        """
        mode = self.m.protocol.stage.train_mode
        keys = ['fused'] if mode == 'fused' else [mode, 'flow']
        return [(k, self.m.optimizers[k]) for k in keys if k in self.m.optimizers]

    def snapshot(self):
        """Everything a rung can move, held on the host."""
        params = [p.detach().to('cpu', copy=True)
                  for p in self.m.gfn_model.parameters()]
        # state_dict hands back LIVE tensors; without the deepcopy the snapshot
        # aliases the optimizer state it is supposed to preserve.
        opts = {k: copy.deepcopy(o.state_dict())
                for k, o in self._stepping_optimizers()}
        rng = {'torch': torch.get_rng_state(),
               'cuda': (torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else None),
               'numpy': np.random.get_state(),
               'python': random.getstate()}
        self.n_snapshots += 1
        return {'params': params, 'opts': opts, 'rng': rng,
                'peak_scale': float(
                    self.m.lr_controller._state().get('peak_scale', 1.0))}

    @torch.no_grad()
    def restore(self, snap):
        for p, s in zip(self.m.gfn_model.parameters(), snap['params']):
            p.copy_(s.to(p.device))
        for k, o in self._stepping_optimizers():
            if k in snap['opts']:
                o.load_state_dict(copy.deepcopy(snap['opts'][k]))
        r = snap['rng']
        torch.set_rng_state(r['torch'])
        if r['cuda'] is not None:
            torch.cuda.set_rng_state_all(r['cuda'])
        np.random.set_state(r['numpy'])
        random.setstate(r['python'])
        # The hypergradient sensor's operand describes a step in a trajectory we
        # have just thrown away, and its only guard is a numel check that cannot
        # see that.
        self.m._hyper_prev_step = None
        self.n_rollbacks += 1

    # ------------------------------------------------------------- coherence

    #: The metric families section 7 names, per ACTIVE BRANCH, with the
    #: direction each one fails in. `rise` families are magnitudes that grow as
    #: training degrades; `fall` families are yields that shrink. Getting a
    #: direction wrong does not make a family noisy -- it inverts it, so a
    #: collapsing ESS would read as health.
    FAMILY_METRICS = (
        ('tb_residual', 'tb_err', 'rise'),      # TB-residual median
        ('tb_upper_tail', 'resid_p95', 'rise'),  # ...and its upper tail
        ('loss', 'loss', 'rise'),               # per-branch loss
        ('ess', 'ess_frac', 'fall'),            # effective sample size
    )

    def _read(self, branch, key):
        tracker = getattr(self.m, 'metric_tracker', None)
        if tracker is None:
            return None
        return tracker.get(branch, key)

    def _families(self):
        """(name -> (branch, metric, sense)) over the branches this stage trains.

        Derived from `_probe_weights` -- the composite the optimizer step
        actually descends -- so the coherence check covers what the rung is
        training rather than a fixed list that happens to fit one route.
        """
        branches = getattr(self.m, '_probe_weights', None) or {}
        return {f'{b}_{name}': (b, metric, sense)
                for b in branches
                for name, metric, sense in self.FAMILY_METRICS}

    def coherence(self, ratio: float = 3.0, families=None):
        """One coherence sample: {family: 'ok' | 'adverse' | 'unknown'}.

        DISTRIBUTION MOVEMENT IS NOT FAILURE -- section 7 is explicit, and
        training is supposed to move the distribution. So a family is `adverse`
        only once it has moved `ratio`-fold in its failing direction from the
        BASELINE THIS RUNG ENTERED AT: a persistent-collapse test, not a drift
        detector. The baseline is per rung because the rate is what changed.

        `unknown` is neither verdict. A route that does not publish a signal must
        not read as healthy on it -- and an all-unknown family set is announced
        at the first sample rather than passed silently, because a coherence gate
        that cannot see anything and one that sees nothing wrong are the same
        from their output alone.
        """
        fams = dict(families or self._families())
        now = {name: self._read(branch, metric)
               for name, (branch, metric, _s) in fams.items()}
        if self._baseline is None:
            self._baseline = {k: v for k, v in now.items() if v is not None}
            if self.verbose:
                live = sorted(self._baseline)
                said = ', '.join(live) if live else (
                    'NONE -- the coherence gate is inert on this route and can '
                    'only ever report unknown')
                print(f'ramp: coherence families resolved at rung '
                      f'{self.ladder.rung}: {said}')
            return {k: 'unknown' for k in fams}
        out = {}
        for name, value in now.items():
            base = self._baseline.get(name)
            sense = fams[name][2]
            if value is None or base is None or not (abs(base) > 0):
                out[name] = 'unknown'
            elif sense == 'fall':
                out[name] = 'adverse' if value < abs(base) / ratio else 'ok'
            else:
                out[name] = 'adverse' if abs(value) > ratio * abs(base) else 'ok'
        return out

    # ------------------------------------------------------------------ drive

    def apply(self, action: dict) -> None:
        """Execute one ladder action against the live run."""
        what = action.get('action')
        if what in (CLIMB, DESCEND):
            if what == CLIMB and action.get('then') == SAVE_CLEAN:
                self.clean_snapshot = self.snapshot()
            self._set_peak(action['scale'])
            self._baseline = None                 # new rung, new reference
            if self.verbose:
                print(f"ramp: {what} -> peak_scale {action['scale']:.4g} "
                      f"({action['reason']})")
        elif what == ROLLBACK:
            if self.clean_snapshot is not None:
                self.restore(self.clean_snapshot)
            self._set_peak(action['scale'])
            if self.verbose:
                print(f"ramp: ROLLBACK to peak_scale {action['scale']:.4g} "
                      f"({action['reason']}); cruise starts here")
        elif what == FINISH:
            self._set_peak(action['scale'])
            if self.verbose:
                print(f"ramp: finished, outcome {self.ladder.outcome}, "
                      f"cruise peak_scale {action['scale']:.4g}")

    def _set_peak(self, scale):
        """Move the rung, through the controller's own state and its own writer.

        `_state()` rather than `m.lr_ctrl` directly: that attribute can still be
        a stale dict from a previous state version until the controller has
        materialised it, and writing `peak_scale` into one the controller is
        about to DISCARD would set a rung nothing ever applies.

        The clamps stay where they are -- `_apply_lrs` owns max_lr/min_lr and the
        flow-head pin, and the ladder's own max_scale/min_scale bound the rung.
        A ramp that reached past either would be a second actuator on the same
        rate.
        """
        ctrl = self.m.lr_controller
        st = ctrl._state()
        st['peak_scale'] = float(scale)
        ctrl._apply_lrs(st)

    def report(self) -> dict:
        out = self.ladder.report()
        out['ramp/rollbacks'] = float(self.n_rollbacks)
        out['ramp/snapshots'] = float(self.n_snapshots)
        return out
