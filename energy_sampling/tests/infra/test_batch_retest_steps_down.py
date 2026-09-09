"""The retest re-walk must start BELOW the size it is re-measuring.

`batch_sizer_retest_steps` exists because `target_met` is a verdict about one
moment and the occupancy moves underneath it. But clearing the verdict without
moving the batch makes the ladder a ONE-WAY RATCHET:

  * the fresh-walk branch restores the base only UPWARD (`if self.batch_size <
    base`), so a grown batch is never re-measured against a smaller one;
  * the walk therefore starts at the previous walk's outcome, and the configured
    `batch_size` is never re-measured after the first walk;
  * `grew = len(s['table']) > 1`, so a rung that clears immediately leaves a
    ONE-ROW table, `audit_at` is never armed, and the S2 stand-down -- the only
    slow downward path -- never runs.

That leaves OOM and a wallclock overrun as the sole ways down, on a knob that is
the denominator of the memorisation dose and the numerator of replay reuse.

Stepping one rung down forces the question either way, for one rung of
measurement rather than a full walk from base.
"""
import pytest


def _retest_start(batch, base, growth_factor):
    """The arithmetic of the retest step-down, in isolation."""
    f = float(growth_factor) or 2.0
    return max(int(base), int(round(batch / f)))


def _walk_rows(start, clears_at, base, growth_factor, ceiling=None):
    """Rows a ladder walk appends: ascend from `start` until a rung clears."""
    rows, b = [], start
    while True:
        rows.append(b)
        if b >= clears_at:
            return rows
        nxt = int(round(b * growth_factor))
        if ceiling is not None and nxt > ceiling:
            return rows
        if nxt <= b:
            return rows
        b = nxt


def test_it_steps_exactly_one_rung_down():
    assert _retest_start(4096, base=1000, growth_factor=1.6) == 2560


def test_it_never_steps_below_the_configured_base():
    """This is a re-measurement, not a cut -- the base-restore branch owns
    everything below the base and has its own OOM-ceiling and cooldown rules."""
    assert _retest_start(1200, base=1000, growth_factor=1.6) == 1000
    assert _retest_start(1000, base=1000, growth_factor=1.6) == 1000


def test_at_the_base_rung_it_is_a_no_op():
    b = _retest_start(1000, base=1000, growth_factor=1.6)
    assert b == 1000, 'nothing to step down to; the walk just re-measures'


# --------------------------------------------------------------- the consequence

def test_the_old_behaviour_leaves_a_one_row_table_and_never_arms_the_audit():
    """Re-walking from the CURRENT size, when that size still clears, measures
    exactly one rung -- so `grew` is False and audit_at stays None."""
    rows = _walk_rows(start=2560, clears_at=2560, base=1000, growth_factor=1.6)
    assert rows == [2560]
    assert (len(rows) > 1) is False, 'this is the defect: no audit is armed'


def test_stepping_down_arms_the_audit_when_the_grown_size_still_wins():
    """One rung down, the lower rung fails, the walk climbs back -- two rows, so
    `grew` is True and the S2 audit is scheduled."""
    start = _retest_start(2560, base=1000, growth_factor=1.6)
    rows = _walk_rows(start=start, clears_at=2560, base=1000, growth_factor=1.6)
    assert rows == [1600, 2560]
    assert len(rows) > 1, 'grew -> audit_at is armed'


def test_stepping_down_descends_when_conditions_have_changed():
    """If the lower rung now clears, the walk stops there: a genuine downward
    path that neither OOM nor a wallclock overrun had to provide."""
    start = _retest_start(2560, base=1000, growth_factor=1.6)
    rows = _walk_rows(start=start, clears_at=1600, base=1000, growth_factor=1.6)
    assert rows == [1600] and rows[-1] < 2560


@pytest.mark.parametrize('retests', [1, 2, 3, 6])
def test_repeated_retests_can_walk_all_the_way_back_down(retests):
    """Bounded descent: one rung per retest, so a badly oversized ladder takes
    several retests to unwind rather than being stuck until an OOM."""
    b = 16384
    for _ in range(retests):
        b = _retest_start(b, base=1000, growth_factor=1.6)
    assert b < 16384
    assert b >= 1000, 'never below base'
