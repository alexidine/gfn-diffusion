"""
THE DIVERGENCE REWIND, THE NEW CELL MACHINERY, AND THE HYBRID ARM.

Written because all three are the kind of thing that fails by DOING NOTHING:
a rewind that never fires, a schedule that never triggers, a cap that never
binds. Each one would leave a battery that runs to completion, prints a full
table, and is quietly measuring the arm without it. Every test here therefore
checks that the mechanism RAN, separately from whether its effect was good.
"""
import math

import pytest
import torch

from bench.arms import Fixed, HyperStep, Null
from bench.metrics import score_run
from bench.runner import Run
from bench.surfaces import EquilibrationGame

KW = dict(dim=8, a=2.0, b=1.0, w_rep=0.7, w_bwd=0.3, kappa=0.02, noise=0.1,
          init_scale=1.0, cond_rep=100.0)


def _game(lr, optimizer='sgd', **over):
    return EquilibrationGame(lr=lr, optimizer=optimizer, seed=0,
                             **{**KW, **over})


# ------------------------------------------------------------------ rewind

def test_the_rewind_actually_fires_and_recovers_the_run():
    """
    THE MECHANISM MUST RUN, AND THEN IT MUST HELP -- two claims, checked apart.

    A shock kicks theta hard enough to cross the tripwire. With the rewind the
    run should return to a healthy state; without it, the same seed and the same
    shock should end up materially worse. Comparing only "final loss with rewind
    is small" would pass on a run where the shock never fired at all.
    """
    shock = (1500, 5e6)
    with_rw = Run(_game(0.01, shock=shock), Null(0.01), seed=0, steps=2500,
                  batch=64, rewind=True).run()
    without = Run(_game(0.01, shock=shock), Null(0.01), seed=0, steps=2500,
                  batch=64, rewind=False).run()

    assert with_rw.divergences > 0, (
        'the shock never tripped the divergence bar, so this test is not '
        'exercising the rewind at all')
    assert with_rw.reloads > 0, 'divergence fired but no rewind was performed'

    a = with_rw.game.expected_loss()
    b = without.game.expected_loss()
    assert math.isfinite(a), f'the rewind did not recover the run ({a})'
    assert (not math.isfinite(b)) or a < b, (
        f'rewind ended at {a:.4g} and no-rewind at {b:.4g} -- the rewind bought '
        f'nothing, so either the snapshot or the restore is not working')


def test_the_reload_budget_aborts_instead_of_rewinding_forever():
    """
    A FIXED RATE CANNOT BE CUT -- nothing manages it -- so a rate above the
    cliff re-detonates after every restore. Production's answer is a rate-based
    budget and then an abort (train.py:2130), because N rewinds without recovery
    IS the unrecoverable signal. Without the budget this run would rewind for
    the whole battery and report a healthy final loss it never held.
    """
    run = Run(_game(0.3), Fixed(0.3), seed=0, steps=4000, batch=64,
              rewind=True).run()
    assert run.aborted, (
        f'a rate ~10x over the cliff ran to completion with {run.reloads} '
        f'rewinds and never aborted -- the budget is not binding')
    assert run.reloads > 0
    assert len(run.trace) < 4000, 'aborted but the run kept stepping'


def test_the_snapshot_is_a_deep_copy_of_the_optimizer_state():
    """
    `state_dict()` returns LIVE tensors. A shallow save is a view of the very
    state it is meant to protect, so the restore writes back the detonated
    moments it captured -- and the bug is invisible under SGD, which keeps no
    state, i.e. invisible on every cell of this battery except the Adam one.
    """
    g = _game(0.01, optimizer='adam')
    run = Run(g, Null(0.01), seed=0, steps=400, batch=64, rewind=True)
    for _ in range(150):                      # snapshots banked at step 0, 100
        run.step()
    assert run._snap is not None, 'no snapshot was ever banked'
    saved = run._snap['opt']['fused']['state']
    before = {k: v['exp_avg'].clone() for k, v in saved.items()
              if isinstance(v, dict) and 'exp_avg' in v}
    assert before, 'adam kept no exp_avg -- this test is not testing anything'

    # THE ALIASING CHECK, and the whole point of the test: `saved` is the dict
    # banked at step 100 and must have stopped tracking the live tensors the
    # moment it was taken. Stepping on and re-reading it is what shows that --
    # if it aliased, these clones would have moved with the live state.
    for _ in range(100):
        run.step()
    assert all(torch.equal(before[k], saved[k]['exp_avg']) for k in before), (
        'the banked snapshot CHANGED while training continued -- state_dict() '
        'handed back live tensors and the "restore" would write back whatever '
        'the run had drifted to, including a detonation')

    live = run.game.optimizers['fused'].state
    moved = any(not torch.equal(before[k], st['exp_avg'])
                for k, st in zip(before, live.values()))
    assert moved, ('the live optimizer state did not move in 100 steps, so this '
                   'test cannot distinguish a deep copy from a shallow one')


def test_the_arm_drops_cross_step_state_across_a_rewind():
    """
    Every arm differences against the previous step. After a rewind that step is
    on an abandoned trajectory, so the next difference spans the discontinuity.

    WHAT THAT DOES AND DOES NOT COST, corrected after a mutation test: because
    `hyper` normalises to a COSINE, a stale displacement cannot produce a large
    move -- the response is capped at exp(beta) however wrong the operand is. So
    the damage is a verdict pointing the WRONG WAY, not a wild jump, and an
    earlier version of this test asserting a bound on the largest log-move was
    vacuous: it passed with the hook removed entirely.

    Tested here as what it is -- a wiring claim plus a state claim. Both are
    checkable exactly; neither needs an appeal to magnitude.
    """
    fired = []

    class Watched(HyperStep):
        def on_rewind(self, run):
            super().on_rewind(run)
            fired.append((run.m.step_ind, self._last_step, self._theta_before))

    arm = Watched(1e-4, beta=0.2)
    run = Run(_game(0.01, shock=(800, 5e6)), arm, seed=0, steps=1200, batch=64,
              rewind=True).run()
    assert run.reloads > 0, 'no rewind happened; nothing was tested'
    assert len(fired) == run.reloads, (
        f'{run.reloads} rewinds but the arm was notified {len(fired)} times -- '
        f'the hook is not wired, so every arm keeps differencing across the '
        f'discontinuity')
    assert all(a is None and b is None for _, a, b in fired), (
        'on_rewind ran but left the cross-step operands in place')


def test_the_rewind_restores_every_player_not_just_the_policy():
    """
    The ray probe snapshots POLICY ONLY (decision D26b) and it is easy to carry
    that convention into the rewind, where it is wrong: the level head would
    stay at its detonated value and re-explode against the restored policy.

    Asserted directly on the parameters rather than through the final loss. A
    mutation restoring only the first parameter still passed an outcome-based
    version of this test, because zeta's own objective drags it back within
    ~1/lr_flow steps -- the surface repairs the omission faster than the score
    can notice it, which is exactly when a state assertion is the honest one.
    """
    run = Run(_game(0.01, shock=(600, 5e6)), Null(0.01), seed=0, steps=1200,
              batch=64, rewind=True)
    seen = 0
    for _ in range(1200):
        run.step()
        if run.reloads > seen:
            seen = run.reloads
            break
        if run.aborted:
            break
    assert seen > 0, 'no rewind fired; nothing was tested'
    assert len(run._snap['params']) > 1, (
        'the game exposes a single parameter tensor, so this test cannot '
        'distinguish restoring one player from restoring all of them')
    for i, (p, saved) in enumerate(zip(run._all_params(), run._snap['params'])):
        assert torch.equal(p.detach(), saved), (
            f'parameter {i} was not restored by the rewind -- a player was left '
            f'at its detonated value')


# ------------------------------------------------------------- new cells

def test_the_schedule_moves_the_cliff_mid_run():
    """
    The tracking cell. Every other cell holds one boundary for the whole run, so
    an arm whose natural shape is ramp-then-settle fits them for free; this is
    the only cell that can tell tracking from a lucky shape. If the schedule
    does not fire, the cell silently duplicates `base`.
    """
    g = _game(0.01, schedule=((300, {'cond_rep': 1000.0}),))
    before = g.stability_lr(lr_level=0.1)
    run = Run(g, Null(0.01), seed=0, steps=400, batch=64)
    run.run()
    after = g.stability_lr(lr_level=0.1)
    assert after < 0.5 * before, (
        f'the scheduled regime change did not move the boundary '
        f'({before:.4g} -> {after:.4g})')


def test_the_shock_is_survivable_by_a_cold_rate_and_fatal_to_a_hot_one():
    """
    A blow-up cell only measures the arms if the blow-up is SURVIVABLE. If every
    arm dies the column ranks nothing; if none can die it is not a blow-up. This
    pins both ends.
    """
    cold = Run(_game(0.001, shock=(600, 1e4)), Fixed(0.001), seed=0, steps=1200,
               batch=64).run()
    assert math.isfinite(cold.game.expected_loss()), (
        'even a cold fixed rate cannot survive the shock -- nothing to measure')
    hot = Run(_game(0.1, shock=(600, 1e4)), Fixed(0.1), seed=0, steps=1200,
              batch=64).run()
    assert hot.divergences > 0 or not math.isfinite(hot.game.expected_loss()), (
        'a rate well over the cliff sailed through the shock -- it is not a '
        'blow-up cell')


def test_an_aborted_run_is_not_scored_as_a_finish():
    """
    THE TRAP THIS BATTERY WALKED INTO. `final_loss` is a trailing-window MEDIAN,
    and the rewind restores a healthy state after every divergence -- so a run
    that detonates repeatedly and then exhausts its reload budget has a tail full
    of restored, healthy losses and scores EXCELLENTLY at the moment it dies.

    Measured on the `regime shift` cell before the fix: `ramp+plateau` aborted on
    5 seeds of 5 and ranked FIRST in the cell; `fixed@0.01` aborted on 5 of 5 and
    ranked second. The `died in k/n` column read '-' for every arm while 30 runs
    had aborted, because each one had a perfectly finite loss.
    """
    run = Run(_game(0.3), Fixed(0.3), seed=0, steps=4000, batch=64).run()
    assert run.aborted, 'this rate was supposed to abort; test is not testing'
    row = score_run(run)
    assert not math.isfinite(row['final_loss']), (
        f"an aborted run scored {row['final_loss']:.4g} -- a run that never "
        f"reached the end is being ranked against runs that did")
    # and the pre-abort value is kept, because it is what made the trap subtle
    assert 'final_loss_at_abort' in row
