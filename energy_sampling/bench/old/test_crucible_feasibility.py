"""
Does `_cold_start_feasible` reject exactly the cells that were MEASURED to be
impossible?

The predicate is arithmetic -- a peak_scale cap and a maximum climb rate -- so it
could be self-consistent and still describe nothing. The table below is the
observed cold_start column from the 13-cell held-out run of 2026-08-13 (20 seeds,
`hyper sym`), which is ground truth: 100% means every one of 20 seeds failed, 0%
means every one passed. There is nothing in between on any cell, which is itself
the tell that the column is structural rather than a controller property.

If the predicate and the measurement ever disagree, one of them is wrong and the
`passable only` aggregate is not to be trusted until that is resolved.
"""
import pytest

from bench.old.crucible import _cold_start_feasible

#: (label, oracle lr, denom, cold_start % for `hyper sym` over 20 seeds)
#: Read off the run's per-cell tables; see docs/lr_control_summary.md section 0.
MEASURED = [
    ('h baseline',    0.00433, 526, 0.0),
    ('h noise=0.1',   0.00433, 526, 0.0),
    ('h noise=0.5',   0.00433, 526, 0.0),
    ('h cond=30',     0.0351,  128, 1.0),
    ('h cond=1000',   0.00152, 776, 0.0),
    ('h quartic1e-4', 0.00433, 526, 0.0),
    ('h quartic=0.1', 0.00433, 526, 0.0),
    ('h n.1 q1e-1',   0.00433, 526, 0.0),
    ('h eq base',     0.433,    50, 1.0),
    ('h eq w_rep.3',  1.23,    119, 1.0),
    ('h dim=256',     0.00433, 526, 0.0),
    ('h dim=2048',    0.00433, 526, 0.0),
    ('h dim2048 n2',  0.00433, 526, 0.0),
]


@pytest.mark.parametrize('label,lr,denom,observed', MEASURED,
                         ids=[m[0] for m in MEASURED])
def test_predicate_matches_the_measured_cold_start_column(label, lr, denom,
                                                          observed):
    feasible, why = _cold_start_feasible(lr, denom)
    if observed == 1.0:
        assert not feasible, (
            f'{label}: every seed failed cold_start, but the predicate calls it '
            f'passable -- so those 20 failures are being scored as a controller '
            f'defect')
        assert 'UNREACHABLE' in why
    else:
        assert feasible, (
            f'{label}: every seed PASSED cold_start, but the predicate calls it '
            f'unreachable ({why}) -- the predicate is over-rejecting and the '
            f'`passable only` column is discarding real results')


def test_the_two_walls_are_distinguishable():
    """
    The two cells rejected for DIFFERENT reasons, kept apart on purpose: one is
    a hard cap (no speed helps), the other is a deadline (a faster climber would
    pass). Collapsing them would hide that raising `hyper_beta` fixes one and
    not the other.
    """
    _, why_cap = _cold_start_feasible(1.23, 119)        # 9840x > 2000
    _, why_time = _cold_start_feasible(0.0351, 128)     # 281x, but 282 steps
    assert 'cap' in why_cap
    assert 'steps to climb' in why_time


def test_a_generous_budget_makes_the_deadline_cell_passable():
    """The time wall is a budget property, not a surface property: the same cell
    with a longer denominator passes."""
    assert not _cold_start_feasible(0.0351, 128)[0]
    assert _cold_start_feasible(0.0351, 526)[0]


def test_the_cap_wall_ignores_the_budget():
    """...and the cap wall does not, at any budget."""
    assert not _cold_start_feasible(1.23, 119)[0]
    assert not _cold_start_feasible(1.23, 100_000)[0]
