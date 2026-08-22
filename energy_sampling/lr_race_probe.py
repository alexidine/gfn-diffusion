"""
Replay Racing -- the trainer-side probe: harvest, trial windows, fork/restore.

Design: `docs/design/lr_probe_protocol.md` (rev d). The DECISION half lives in
`lr_race.py` (pure, no torch, gated by `bench/test_lr_race.py`); this module is
the half that touches the trainer, and its whole job is to produce an honest
`RaceRecord` for that decision layer without perturbing the run it measures.

THE IDEA. Candidate learning rates race each other on REPLAYED data: batches
the run has already trained on, recorded verbatim as they went past. Every arm
trains on the identical frozen sequence and is scored on the identical held-out
slice, so the arms are exactly paired and the probe costs no rollouts and no
energy calls. Then the snapshot is restored and training resumes at whichever
rate won -- the winning arm's weights are DISCARDED (winner's curse).

WHY REPLAY MAKES THIS AFFORDABLE. `get_gfn_backward_loss(trajectories=...)`
routes to `gfn.get_traj_replay`, which scores a stored path instead of sampling
one: no rollout, no energy call, and -- measured in the map, and asserted here
-- no RNG consumption at all. `_probe_loss` (train.py) already relies on this
contract for the ray sensor; a trial step is that same evaluation with a
backward pass and an optimizer step attached.

FOUR THINGS THIS MODULE MUST NOT DO, each of which is a recorded scar:

  * Draw from a buffer. Draws consume NumPy RNG that nothing restores (F-039)
    and bump `select_counts`, so a probe that drew would move every subsequent
    training step. The larder is fed by the LIVE steps instead, which is why
    harvesting is a tee and not a fetch.
  * Route through `step_loss`. It owns the run's non-finite streaks, the reload
    budget, `last_grad_norm_pre_clip` and `_hyper_prev_step`; a trial's failures
    are not the parent's.
  * Let the clip guard learn. Its bar self-updates multiplicatively per step, so
    a live bar would clip a hot arm's larger steps back toward unchanged and the
    race would read "2x is free" -- the ratchet, manufactured mechanically. We
    call `threshold()` and never `observe()`.
  * Write a checkpoint. The tag namespace has no probe slot and `best` is a
    hardlink to `running`, so a probe save would edit the run's own rollback
    target. Snapshots live in host RAM.

STATUS. Actuation is gated on the validation of design section 6: with
`lr_probe.enabled: true` but `actuate: false` (the default) the probe runs,
logs a complete record, and CHANGES NOTHING.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import replace
from collections import deque, namedtuple

import numpy as np
import torch

from gflownet_losses import get_gfn_backward_loss
from lr_race import (ENTRY_ARMS, FINE_ARMS, INCUMBENT, ArmScores, RaceConfig,
                     RaceRecord, decide, shifted_bracket)
from lr_race import _replicate_advantage as _rep_adv
from lr_race import _unit_advantage as _unit_adv


def _mean_adv(challenger, incumbent):
    a = _unit_adv(challenger, incumbent)
    return (sum(a) / len(a)) if a else 0.0

#: One replay-scoreable training batch, recorded as a live step went past.
#: `traj` is the trajectory that step ACTUALLY took -- for bwd/dataset draws the
#: draw itself returns None and the sampled path is only visible as
#: `loss_dict['flow_states']`, which is precisely why harvesting has to happen
#: inside the branch step functions rather than around them.
Harvested = namedtuple('Harvested',
                       'branch condition condition_id log_r mol_batch traj repeats')


class RaceLarder:
    """A rolling, per-branch ring of harvested batches.

    ALWAYS ON while the probe is enabled, because triggers are not predictable:
    a composition change or a ramp completion has to find a full larder already
    there. The cost is the tee itself plus the memory of `depth` batches per
    branch, and both are part of what the cost gate measures -- not assumed free.
    """

    def __init__(self, depth: int = 48):
        self.depth = int(depth)
        self.rings: dict[str, deque] = {}
        self.n_seen = 0

    def record(self, branch: str, rec: Harvested) -> None:
        ring = self.rings.get(branch)
        if ring is None:
            ring = self.rings[branch] = deque(maxlen=self.depth)
        ring.append(rec)
        self.n_seen += 1

    def branches(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.rings.items() if v)

    def count(self, branch: str) -> int:
        return len(self.rings.get(branch, ()))

    def ready(self, branches, need: int) -> bool:
        return all(self.count(b) >= need for b in branches) and bool(branches)

    def deal(self, branches, n_train_sets: int, window: int, n_hold: int):
        """Partition the larder into disjoint training sub-larders plus a
        held-out slice shared by every arm.

        DISJOINT is the point. Replicates exist to resample the training path,
        so two replicates sharing batches would share the luck of those batches
        and the sign test's sample size would be a fiction. The held-out slice
        is common to all arms (it is the ruler) and is never trained on.
        """
        per = {}
        for b in branches:
            ring = list(self.rings.get(b, ()))
            need = n_train_sets * window + n_hold
            if len(ring) < need:
                return None
            hold = ring[:n_hold]
            rest = ring[n_hold:n_hold + n_train_sets * window]
            sets = [rest[i * window:(i + 1) * window] for i in range(n_train_sets)]
            per[b] = {'hold': hold, 'sets': sets}
        return per


class RaceProbe:
    """Runs one race and returns a `RaceRecord` for `lr_race.decide`."""

    def __init__(self, modeller, cfg: RaceConfig = RaceConfig(),
                 window: int = 10, n_hold: int = 10, depth: int = 0,
                 actuate: bool = False, verbose: bool = True):
        self.m = modeller
        self.cfg = cfg
        self.window = max(2, int(window))
        self.n_hold = max(2, int(n_hold))
        self.actuate = bool(actuate)
        self.verbose = bool(verbose)
        # One larder must serve the widest race: r training sets plus the
        # held-out slice, with the entry screen's r as the floor.
        need = cfg.replicates * self.window + self.n_hold
        self.larder = RaceLarder(depth=max(int(depth or 0), need))
        self.n_races = 0
        self.last = {}
        self.history = []
        self.enabled = True
        self._stage = None
        self._stage_entry = 0
        self._armed_entry = False
        self._clock_fired = set()
        self._last_composition = None
        self._bank_cache = {}
        self._skip_logged = False
        self._zp = deque(maxlen=self.Z_CAL_SETTLED_OBS)

    # ------------------------------------------------------------- harvest

    @staticmethod
    def _to_host(v):
        """Detach and park on the host WITHOUT touching the live object.

        `copy.copy` first is not optional: PyG's `Data.cpu()`/`.to()` MUTATE IN
        PLACE (apply rewrites the store and returns self), so a bare
        `mol_batch.cpu()` would drag the batch the live step is still training
        on off the device. A shallow copy gives `.cpu()` its own store to
        rewrite, and tensor-level `.cpu()` is non-mutating, so the live tensors
        are untouched. This is buffer.py's idiom, for the same reason.
        """
        if v is None:
            return None
        if torch.is_tensor(v):
            return v.detach().to('cpu', copy=True)
        if hasattr(v, 'cpu'):                      # a PyG Data/Batch
            return copy.copy(v).cpu()
        return v

    def _to_device(self, v):
        if v is None:
            return None
        if torch.is_tensor(v):
            return v.to(self.m.device)
        if hasattr(v, 'to'):
            return copy.copy(v).to(self.m.device)  # same in-place hazard, back
        return v

    def harvest(self, branch, condition, condition_id, log_r, mol_batch,
                traj, repeats):
        """Tee from a live branch step, parking the record in HOST memory.

        Host, not device: a full larder is `replicates * window + holdout`
        batches PER BRANCH, and on a fused crystal stage that is three rings of
        trajectories and crystal batches. Left on the card they would compete
        with training for exactly the VRAM the batch sizer is trying to fill.
        The cost is one H2D copy per trial step, which the cost gate measures.
        """
        if traj is None:
            return
        self.larder.record(branch, Harvested(
            branch=branch,
            condition=self._to_host(condition),
            condition_id=self._to_host(condition_id),
            log_r=self._to_host(log_r),
            mol_batch=self._to_host(mol_batch),
            traj=self._to_host(traj),
            repeats=int(repeats)))

    # ------------------------------------------------------------ snapshot

    def _stepping_optimizers(self):
        """Exactly the optimizers this stage's train_mode drives.

        Mirrors `step_loss`: a bwd stage steps 'bwd' and then 'flow'; a fused
        stage steps 'fused' alone (its param groups already carry the flow head
        at `lr_flow`). Snapshotting the others as well would be harmless but
        slower, and getting this list WRONG is the way a trial silently trains
        something the live step does not.
        """
        mode = self.m.protocol.stage.train_mode
        keys = ['fused'] if mode == 'fused' else [mode, 'flow']
        return [(k, self.m.optimizers[k]) for k in keys if k in self.m.optimizers]

    def _snapshot(self):
        """Everything a trial can move, held in host RAM."""
        params = [p.detach().clone() for p in self.m.gfn_model.parameters()]
        opts = {k: copy.deepcopy(o.state_dict())      # state_dict hands back LIVE
                for k, o in self._stepping_optimizers()}  # tensors; deepcopy or the
        rng = {                                        # snapshot aliases the state
            'torch': torch.get_rng_state(),
            'cuda': (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
        return {'params': params, 'opts': opts, 'rng': rng}

    @torch.no_grad()
    def _restore(self, snap):
        for p, s in zip(self.m.gfn_model.parameters(), snap['params']):
            p.copy_(s)
        for k, o in self._stepping_optimizers():
            if k in snap['opts']:
                o.load_state_dict(copy.deepcopy(snap['opts'][k]))
        r = snap['rng']
        torch.set_rng_state(r['torch'])
        if r['cuda'] is not None:
            torch.cuda.set_rng_state_all(r['cuda'])
        np.random.set_state(r['numpy'])
        random.setstate(r['python'])
        # The sensor's operand describes a step in a trajectory we just threw
        # away, and its only guard is a numel check that cannot see that.
        self.m._hyper_prev_step = None

    def _rng_fingerprint(self):
        t = torch.get_rng_state()
        return (int(t.sum()), int(np.random.get_state()[2]))

    # --------------------------------------------------------------- scoring

    #: Coefficient keys `get_gfn_backward_loss` reads WITHOUT a getattr guard,
    #: so an absent one raises AttributeError instead of meaning "off". Every
    #: branch is replayed through that evaluator (there is no
    #: `get_gfn_forward_loss(trajectories=...)`), and the FWD bank legitimately
    #: has no `mle`/`pf_boost` because the fwd branch does not train them.
    #: For a replayed score that means coefficient ZERO, which is what we fill.
    #: `coeff_matrix` is deliberately NOT in this list: it is structural rather
    #: than a coefficient and is only read when subtb > 0, so filling it would
    #: mask a real misconfiguration instead of expressing an absent term.
    _REQUIRED_COEFFS = ('db', 'mle', 'pf_boost', 'subtb', 'tb', 'traj_grads',
                        'vg_lb', 'vg_lme')

    def _bank(self, branch):
        """The branch's coefficient bank, padded for the replay evaluator.

        Found the hard way: the first fused-stage race died with
        `'Namespace' object has no attribute 'pf_boost'`. The design note had
        anticipated fwd-ONLY terms being dropped by the backward evaluator; it
        missed the mirror case, where the evaluator REQUIRES bwd-only keys the
        fwd bank never had. Padding is announced once per branch rather than
        applied silently -- a zero that nobody chose is exactly the kind of
        default that later reads as a measurement.
        """
        a = self.m.args
        bank = {'fwd': a.fwd_loss_coeffs, 'bwd': a.bwd_loss_coeffs,
                'replay': a.replay_loss_coeffs}[branch]
        missing = [k for k in self._REQUIRED_COEFFS if not hasattr(bank, k)]
        if not missing:
            return bank
        cached = self._bank_cache.get(branch)
        if cached is None:
            cached = copy.copy(bank)
            for k in missing:
                setattr(cached, k, 0.0)
            self._bank_cache[branch] = cached
            if self.verbose:
                print(f"race: {branch} bank padded with {missing} = 0 for the "
                      f"replay evaluator (terms this branch does not train)")
        return cached

    def _loss_on(self, rec, discretizer, grad: bool):
        """Score one harvested batch under the CURRENT parameters.

        Stored trajectory and stored reward, so `get_gfn_backward_loss` routes
        to `get_traj_replay`: no rollout, no energy call. `update_log_z=False`
        and `mode_level_stream=None` are the gates that keep this out of the
        run's trackers -- the same contract `_probe_loss` maintains, with a
        backward pass allowed when `grad` is set.
        """
        traj = self._to_device(rec.traj)
        mol_batch = self._to_device(rec.mol_batch)
        condition = self._to_device(rec.condition)
        condition_id = self._to_device(rec.condition_id)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            loss, _ = get_gfn_backward_loss(
                self._bank(rec.branch),
                traj[:, -1] if traj.dim() == 3 else traj,
                self.m.gfn_model,
                self._to_device(rec.log_r),
                discretizer,
                mol_batch,
                condition=condition,
                repeats=rec.repeats,
                report_losses=False,
                trajectories=traj,
                condition_log_z=self.m.condition_log_z,
                condition_id=condition_id,
                tb_z_source=self.m.tb_z_source(rec.branch),
                update_log_z=False,
                step=self.m.step_ind,
                mode_level_stream=None)
        return loss

    def _weights(self):
        """The stage's live frac weights, frozen for the whole race."""
        w = {'fwd': float(getattr(self.m, 'fwd_frac', 0.0)),
             'bwd': float(getattr(self.m, 'bwd_frac', 0.0)),
             'replay': float(getattr(self.m, 'replay_frac', 0.0))}
        avail = {b: w.get(b, 0.0) for b in self.larder.branches()}
        tot = sum(avail.values())
        if tot <= 0:                       # a single-branch stage: weight it 1
            return {b: 1.0 / max(len(avail), 1) for b in avail}
        return {b: v / tot for b, v in avail.items()}

    @torch.no_grad()
    def _score_holdout(self, deal, weights, discretizer):
        """One number per held-out batch index: the frac-weighted loss."""
        n = min(len(deal[b]['hold']) for b in deal)
        out = []
        for i in range(n):
            tot = 0.0
            for b, w in weights.items():
                if w <= 0 or b not in deal:
                    continue
                tot += w * float(self._loss_on(deal[b]['hold'][i], discretizer,
                                               grad=False).detach())
            out.append(tot)
        return out

    # ------------------------------------------------------------- one window

    def _set_arm_lr(self, mult):
        """Scale exactly the groups the live actuator scales, for the trial."""
        saved = []
        for _k, o in self._stepping_optimizers():
            for g in o.param_groups:
                saved.append((g, g['lr']))
                g['lr'] = g['lr'] * mult
        return saved

    @staticmethod
    def _reset_lr(saved):
        for g, lr in saved:
            g['lr'] = lr

    def _window(self, mult, train_set, deal, weights, discretizer, snap):
        """One arm, one replicate: W trial steps, scored at half and at end."""
        self._restore(snap)
        saved = self._set_arm_lr(mult)
        opts = self._stepping_optimizers()
        half = max(1, len(train_set[next(iter(train_set))]) // 2)
        mid = end = None
        died = False
        try:
            n = min(len(v) for v in train_set.values())
            for t in range(n):
                for _k, o in opts:
                    o.zero_grad(set_to_none=True)
                total = None
                for b, w in weights.items():
                    if w <= 0 or b not in train_set:
                        continue
                    li = w * self._loss_on(train_set[b][t], discretizer, grad=True)
                    total = li if total is None else total + li
                if total is None:
                    died = True
                    break
                total.backward()
                # The guard's bar, NOT the guard's learning: threshold() reads,
                # observe() would move the bar under the arm.
                gn = torch.nn.utils.clip_grad_norm_(
                    self.m.gfn_model.parameters(),
                    self.m.grad_guard.threshold(self.m.protocol.stage.train_mode))
                if not torch.isfinite(gn):
                    died = True
                    break
                for _k, o in opts:
                    o.step()
                if t + 1 == half:
                    mid = self._score_holdout(deal, weights, discretizer)
            if not died:
                end = self._score_holdout(deal, weights, discretizer)
                if mid is None:
                    mid = end
                if not all(math.isfinite(v) for v in end + mid):
                    died = True
        except RuntimeError as e:                       # OOM or a shape fault
            died = True
            if self.verbose:
                print(f'  race: window at {mult}x died: {type(e).__name__}: {e}')
        finally:
            self._reset_lr(saved)
            for _k, o in opts:
                o.zero_grad(set_to_none=True)
        if died:
            z = [float('nan')] * self.n_hold
            return z, z, True
        return mid, end, False

    # ------------------------------------------------------------- one race

    def race(self, arms, kind='fine', expansions_used=0):
        """Run every arm of one race and return the record for `decide`."""
        branches = self.larder.branches()
        reps = self.cfg.screen_replicates if kind == 'screen' else self.cfg.replicates
        # One extra training set for the same-order duplicate of the incumbent.
        deal = self.larder.deal(branches, reps, self.window, self.n_hold)
        if deal is None:
            return None
        weights = self._weights()
        discretizer = self.m.get_discretizer(self.m.args.integrator) \
            if hasattr(self.m, 'get_discretizer') else _discretizer(self.m)

        snap = self._snapshot()
        fp_before = self._rng_fingerprint()
        scored = []
        try:
            for mult in arms:
                mids, ends, died = [], [], False
                for j in range(reps):
                    train_set = {b: deal[b]['sets'][j] for b in deal}
                    mid, end, d = self._window(mult, train_set, deal, weights,
                                               discretizer, snap)
                    mids.append(tuple(mid))
                    ends.append(tuple(end))
                    died = died or d
                scored.append(ArmScores(multiplier=float(mult), mid=tuple(mids),
                                        end=tuple(ends), died=died))
            # The restore certificate: the incumbent re-run at the SAME order.
            # On a deterministic route it must reproduce bitwise; anything else
            # is state leaking between arms, and a leak is indistinguishable
            # from a real effect from the outputs alone.
            train_set = {b: deal[b]['sets'][0] for b in deal}
            dup_mid, dup_end, dup_died = self._window(INCUMBENT, train_set, deal,
                                                      weights, discretizer, snap)
            base = scored[[a.multiplier for a in scored].index(INCUMBENT)]
            spread = (float('inf') if dup_died else
                      max(abs(a - b) for a, b in zip(dup_end, base.end[0])))
        finally:
            self._restore(snap)

        fp_after = self._rng_fingerprint()
        rec = RaceRecord(
            arms=tuple(scored), kind=kind, expansions_used=expansions_used,
            isolation_ok=(fp_before == fp_after),
            duplicate_spread=spread,
            expect_bitwise=not torch.cuda.is_available(),
            note=f'step {self.m.step_ind} branches {",".join(branches)}')
        self.n_races += 1
        return rec

    # ---------------------------------------------------------- the event

    def run_event(self, entry: bool):
        """A full calibration: fine race, or screen -> expand -> confirm."""
        if not entry:
            rec = self.race(FINE_ARMS, kind='fine')
            if rec is None:
                return self._report('no_larder', None, None)
            return self._report('fine', rec, decide(rec, self.cfg))

        arms, used = ENTRY_ARMS, 0
        while True:
            rec = self.race(arms, kind='screen', expansions_used=used)
            if rec is None:
                return self._report('no_larder', None, None)
            d = decide(rec, self.cfg)
            # An entry event can run several screens before it reports, and only
            # the LAST one used to reach the log -- so a verdict arrived with no
            # visible account of how it got there. `race/races` said 3 while the
            # console showed 1.
            self._trace(f'screen@{used}', arms, rec, d)
            if d.action in ('expand_up', 'expand_down') and used < self.cfg.max_expansions:
                used += 1
                shifted = shifted_bracket(arms, 'up' if d.action == 'expand_up' else 'down')
                arms = tuple(sorted(set(shifted) | {INCUMBENT}))
                continue
            if d.action != 'candidate':
                return self._report('screen', rec, d)
            break
        # ONE PRE-DECLARED STEP DOWN. A half-window rejection does not mean
        # "do not move" -- it means "not THAT far". The check fails when an arm
        # improves in the first half of the window and stops in the second,
        # which is the signature of a step too large to sustain, and the honest
        # response to that is a smaller step rather than none at all.
        #
        # MEASURED, run race_L1_cold8x (elj, seeded 8x cold): the screen read
        # 4x +4.68, 16x +20.28, 64x +12.70, selected 16x, and the confirm passed
        # its sign test 6 of 6 -- then held on halves=False. The run stayed 8x
        # cold with the evidence for escaping it sitting in the log.
        #
        # The fallback is fixed in advance (always the next rung down, at most
        # once), so it is a 2-contrast family rather than a search: alpha is
        # split across both looks and the second is NOT free.
        ladder = [d.multiplier]
        lower = [m for m in arms if INCUMBENT < m < d.multiplier] if d.multiplier > INCUMBENT             else [m for m in arms if d.multiplier < m < INCUMBENT]
        if lower:
            ladder.append(max(lower) if d.multiplier > INCUMBENT else min(lower))
        split = replace(self.cfg, alpha=self.cfg.alpha / len(ladder))

        last = None
        for mult in ladder:
            conf = self.race((INCUMBENT, mult), kind='confirm')
            if conf is None:
                return self._report('no_larder', None, None)
            cd = decide(conf, split)
            self._trace('confirm', (INCUMBENT, mult), conf, cd)
            last = (conf, cd)
            if cd.action == 'move':
                return self._report('confirm', conf, cd)
            # Only a "too far" rejection earns the step down. A candidate that
            # simply lost, died, or produced an invalid race gets no second bite.
            if cd.reason != 'candidate_not_confirmed':
                break
            det = (cd.detail or {}).get('result') or {}
            # Step down only when the SIGN TEST passed and the halves check was
            # the sole objection -- i.e. the evidence says "better, but not this
            # far". An arm that simply lost gets no second bite.
            test_ok = (det.get('need') is not None
                       and det.get('favoring', 0) >= det.get('need', 1))
            if det.get('half_window_ok', True) or not test_ok:
                break
        return self._report('confirm', last[0], last[1])

    def _trace(self, phase, arms, rec, d):
        """One line of EVIDENCE per internal race.

        The design says log the evidence, not the verdict: without the per-arm
        advantages and the replicate tally, a `hold` is indistinguishable from
        a test that had no power to do anything else.
        """
        if not self.verbose:
            return
        bits = []
        for a in rec.arms:
            if a.died:
                bits.append(f'{a.multiplier:g}x=died')
                continue
            if a.multiplier == INCUMBENT:
                continue
            adv = _mean_adv(a, rec.arm(INCUMBENT))
            fav = sum(1 for r in _rep_adv(a, rec.arm(INCUMBENT)) if r > 0)
            bits.append(f'{a.multiplier:g}x adv{adv:+.4f} fav{fav}/{len(a.end)}')
        det = (d.detail or {}).get('result') if d else None
        tail = ''
        if det:
            tail = (f" | need {det['need']} favoring {det['favoring']}"
                    f"{' UNDERPOWERED' if det['underpowered'] else ''}"
                    f" halves={det['half_window_ok']}")
        print(f'  race.{phase}: {d.action}'
              + (f" x{d.multiplier:g}" if d and d.multiplier else '')
              + f' ({d.reason}) [' + ', '.join(bits) + ']' + tail)

    def _report(self, phase, rec, d):
        out = {'phase': phase,
               'action': (d.action if d else 'skipped'),
               'multiplier': (d.multiplier if d else 1.0),
               'reason': (d.reason if d else 'larder_not_ready'),
               'step': self.m.step_ind,
               'stage': self.m.protocol.stage.name,
               'actuated': False}
        if rec is not None:
            out['isolation_ok'] = rec.isolation_ok
            out['duplicate_spread'] = rec.duplicate_spread
            out['arms'] = {a.multiplier: ('died' if a.died else
                                          round(float(np.mean(a.end)), 5))
                           for a in rec.arms}
        if d is not None and d.action == 'move' and self.actuate:
            out['actuated'] = self._apply(d.multiplier)
        self.last = out
        self.history.append(out)
        if self.verbose:
            print(f"race[{phase}] step {out['step']} stage {out['stage']}: "
                  f"{out['action']} x{out['multiplier']:g} ({out['reason']})"
                  + ('' if rec is None else
                     f" iso={out['isolation_ok']} dup={out['duplicate_spread']:.3e}"))
        return out

    def _apply(self, mult):
        """Move `peak_scale` -- the one global the live actuator owns."""
        st = getattr(self.m, 'lr_ctrl', None)
        if not isinstance(st, dict) or 'peak_scale' not in st:
            return False
        st['peak_scale'] = float(st['peak_scale']) * float(mult)
        return True

    # ---------------------------------------------------------- triggers

    #: Stage-relative steps at which the fallback clock fires. Dense across the
    #: fast-equilibration window, sparse after: under the measured within-stage
    #: stationarity the k-th mid-stage race buys ~k^-1.5, so a fixed 1k clock is
    #: mostly paying to re-measure a constant.
    CLOCK = (500, 1500, 3500, 7500, 15000, 30000)

    #: L1 distance on the normalised branch-weight vector that counts as a new
    #: loss regime. Branch dormancy flips and coefficient ramps are special
    #: cases of the same move, so one number covers all three.
    COMPOSITION_L1 = 0.2

    #: `z_cal/p` below this counts as "the log-Z level shift is done". User-set
    #: for the equilibration entry (2026-08-21): the other candidate settling
    #: signals -- z_bias_rms, step-time decay -- were judged less reliable.
    Z_CAL_SETTLED = 0.1

    #: Consecutive OBSERVATIONS required, not one. `z_calibration_tick` pre-sets
    #: `z_cal/p` to 0.0 and then returns early on several paths (mid-grad-accum,
    #: scrambled conditions), so a single 0 can mean "did not run" rather than
    #: "settled". Requiring a run of them makes those spurious zeros harmless.
    Z_CAL_SETTLED_OBS = 5

    #: Floor for stages that publish no `z_cal/p` at all, so "no signal" cannot
    #: mean "no wait". Stage entry is transient for hundreds to a couple of
    #: thousand steps whether or not a sidecar happens to measure it.
    MIN_STAGE_STEPS = 300

    def _note_transient(self):
        """Sample the z-calibration actuator, which opens wide at a transition
        and closes as the level shift resolves. Sampled every tick because the
        report dict is cleared each time it is published."""
        rep = getattr(self.m, '_z_cal_report', None) or {}
        if 'z_cal/p' in rep:
            self._zp.append(float(rep['z_cal/p']))

    def _transient_settled(self, rel):
        """Is the stage past its opening transient?

        A rate measured while log Z is still making a large level shift
        describes the transient, not the stage, and does not extrapolate to it.
        """
        if rel < self.MIN_STAGE_STEPS:
            return False
        if not self._zp:            # no z sidecar here -> the floor is the gate
            return True
        return (len(self._zp) >= self.Z_CAL_SETTLED_OBS
                and all(v < self.Z_CAL_SETTLED for v in self._zp))

    def _ramping(self):
        ctrl = getattr(self.m, 'lr_controller', None)
        st = getattr(self.m, 'lr_ctrl', None)
        if ctrl is None or not isinstance(st, dict):
            return False
        try:
            return bool(ctrl._ramping(st))
        except Exception:
            return False

    def _composition(self):
        """The STAGE's configured loss mixture, normalised.

        Deliberately NOT `_weights()`: that one is keyed on the branches the
        larder happens to hold, because those are the only ones a trial can
        replay. The trigger has to answer a different question -- "has the
        objective the run is training changed?" -- and that is true whether or
        not a branch has been harvested yet. Keying the signal on larder
        contents made the detector blind exactly while a newly-woken branch
        was filling, which is precisely when the mixture had just moved.
        """
        fr = {'fwd': float(getattr(self.m, 'fwd_frac', 0.0) or 0.0),
              'bwd': float(getattr(self.m, 'bwd_frac', 0.0) or 0.0),
              'replay': float(getattr(self.m, 'replay_frac', 0.0) or 0.0)}
        tot = sum(fr.values())
        if tot > 0:
            fr = {k: v / tot for k, v in fr.items()}
        return tuple(sorted(fr.items()))

    def _composition_moved(self):
        if self._last_composition is None:
            return False
        now = dict(self._composition())
        was = dict(self._last_composition)
        keys = set(now) | set(was)
        return sum(abs(now.get(k, 0.0) - was.get(k, 0.0)) for k in keys) >= self.COMPOSITION_L1

    def tick(self):
        """Called once per host-loop iteration, AFTER the step-timing window has
        closed and the sizer deques have been appended.

        Placement is not cosmetic: the `z_calibration_tick` seat sits INSIDE the
        timing window on purpose, so a race there would post its own multi-second
        wall time as this iteration's `step_dt` and hand it to the batch sizer's
        rung median, the runaway guard and the throughput meters. Racing after
        the appends keeps the probe out of every one of those estimators.
        """
        if not self.enabled:
            return None
        stage = self.m.protocol.stage.name
        if stage != self._stage:                    # a transition just happened
            self._stage, self._stage_entry = stage, int(self.m.step_ind)
            self._armed_entry = True                # ARM here, FIRE at ramp end
            self._clock_fired = set()
            self._last_composition = None
            self._skip_logged = False
            self._bank_cache = {}
            self._zp.clear()
            # DROP the harvest. Those batches were drawn under the OUTGOING
            # stage's branches and loss mixture, so racing on them would score
            # candidate rates against an objective the run has already left --
            # the one comparison the design says must never be made across a
            # composition change.
            self.larder.rings.clear()
            return None

        self._note_transient()

        # Never race through a moving ramp: the envelope is a scheduled fraction
        # of the operating rate, so every arm would be measured at a rate the
        # run is not going to keep. Deferred, not dropped.
        if self._ramping():
            return None

        rel = int(self.m.step_ind) - self._stage_entry
        # Nor through the stage's opening transient. Measured on race_L2
        # (elj, equilibration entry): step times were still swinging 1.8-2.7 s
        # 160 steps in, and the race fired at ~150 -- so its verdict described
        # the transition, not the stage it was meant to set a rate for.
        if not self._transient_settled(rel):
            return None
        if self._armed_entry:
            # STAY ARMED until a race actually runs. Consuming the flag on the
            # attempt silently dropped the entry race whenever the larder was
            # still filling -- which is ALWAYS at a stage entry, because the
            # transition just cleared it. Measured on race_L1_phase1: the ramp
            # froze at step 50 with 51 of 68 batches and the most important
            # race of the run -- the cold-start escape -- was thrown away.
            out = self._guarded(entry=True, why='stage_entry')
            if out is not None:
                self._armed_entry = False
                self._skip_logged = False
                self._last_composition = self._composition()
            return out

        if self._composition_moved():
            self._last_composition = self._composition()
            return self._guarded(entry=False, why='composition')

        for c in self.CLOCK:
            if rel >= c and c not in self._clock_fired:
                self._clock_fired.add(c)
                return self._guarded(entry=False, why='clock')
        return None

    def _guarded(self, entry, why):
        """Race, but never let the probe take the run down with it."""
        need = self.cfg.replicates * self.window + self.n_hold
        if not self.larder.ready(self.larder.branches(), need):
            if self.verbose and not self._skip_logged:
                have = {b: self.larder.count(b) for b in self.larder.branches()}
                print(f'race: deferred ({why}) at step {self.m.step_ind}: '
                      f'larder {have} < {need} -- still filling')
                self._skip_logged = True     # once per armed period, not per step
            return None
        try:
            out = self.run_event(entry=entry)
            if out is not None:
                out['trigger'] = why
            return out
        except Exception as e:                  # a probe must never kill a run
            print(f'race: ABORTED ({why}) at step {self.m.step_ind}: '
                  f'{type(e).__name__}: {e}')
            return None

    def report(self) -> dict:
        """Loggable view. Every race, whatever the verdict -- a verdict-only log
        would make the offline analysis in appendix A impossible."""
        if not self.last:
            return {'race/races': float(self.n_races)}
        r = self.last
        return {
            'race/races': float(self.n_races),
            'race/action': float({'hold': 0, 'move': 1, 'invalid': 2,
                                  'candidate': 3, 'expand_up': 4,
                                  'expand_down': 5, 'skipped': 6}.get(r['action'], -1)),
            'race/multiplier': float(r['multiplier']),
            'race/actuated': float(bool(r['actuated'])),
            'race/isolation_ok': float(bool(r.get('isolation_ok', True))),
            'race/duplicate_spread': float(r.get('duplicate_spread') or 0.0),
            'race/larder': float(self.larder.n_seen),
        }


def _discretizer(m):
    from utils import get_discretizer
    return get_discretizer(m.args.integrator)
