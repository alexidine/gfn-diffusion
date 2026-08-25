"""
The race must be REPORTABLE, not just printable (owner ask, 2026-08-24): per-rung
results as plain data for the wandb table + run-summary stamp, so battery arms
can be compared through the API instead of grepping N SLURM .out files.

`pytest tests/lr/test_lr_bracket_race_report.py -q`
"""

import pytest

from lr_bracket import LRBracket, SCREEN

pytestmark = pytest.mark.fast

GRID = [0.05, 0.1, 0.2, 0.4]


def _cycle(b, failing=()):
    b.begin_bracket(1000, bias_correction=0.99)
    while True:
        t = b.next_trial()
        if t is None:
            break
        fails = t.scale in failing
        b.record(t, ok=not fails, reason='loss_excursion' if fails else None,
                 steps_completed=10 if fails else b.trial_steps,
                 steps_to_failure=10 if fails else None)
    return b.select()


def test_race_rows_mirror_the_summary():
    b = LRBracket(candidate_scales=GRID, burn_in_steps=3000, burn_in_scale=0.05,
                  trial_steps=150, boundary_densify=False)
    _cycle(b, failing=(0.4,))
    rows = b.race_rows()
    assert len(rows) == len(b._results)
    by_scale = {r['scale']: r for r in rows if r['kind'] == SCREEN}
    assert by_scale[0.05]['survived'] and by_scale[0.05]['steps_to_failure'] is None
    assert not by_scale[0.4]['survived']
    assert by_scale[0.4]['steps_to_failure'] == 10
    assert by_scale[0.4]['reason'] == 'loss_excursion'
    # every row is JSON-plain: the table publisher must never meet an object
    for r in rows:
        for v in r.values():
            assert v is None or isinstance(v, (str, int, float, bool))
    assert b.cycle_index == 1


def test_cycle_index_advances_per_bracket():
    b = LRBracket(candidate_scales=GRID, burn_in_steps=3000, burn_in_scale=0.05,
                  trial_steps=150, boundary_densify=False, repeat_every=100)
    _cycle(b, failing=(0.4,))
    b.promote(b.select()['scale'], 1150)
    _cycle(b, failing=(0.4,))
    assert b.cycle_index == 2


def test_publisher_is_inert_without_wandb_run():
    """The CPU driver tests exercise run() with no wandb.init; the publisher
    must be a silent no-op there, never a raise."""
    from lr_bracket_probe import BracketDriver
    import wandb
    assert wandb.run is None, 'this test assumes no live wandb run'
    fake = type('D', (), {})()
    fake.bracket = LRBracket(candidate_scales=GRID, burn_in_steps=3000,
                             burn_in_scale=0.05, trial_steps=150)
    fake.last_summary = 'no cycle ran'
    BracketDriver._publish_race(fake)   # must simply return
