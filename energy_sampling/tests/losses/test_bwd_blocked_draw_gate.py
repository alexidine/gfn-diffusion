"""The bwd blocked-draw gate in train.py routes through the shared predicate.

Moved unchanged from tests/lr/test_lr_bracket_driver.py when that file was
deleted with the LR bracket (2026-10-28).

    python -m pytest -q tests/losses/test_bwd_blocked_draw_gate.py
"""
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[2] / 'train.py'


def test_bwd_blocked_draw_gate_does_not_reopen_the_coefficient_list():
    """The bwd blocked-draw gate must ASK, not re-list.

    History: the gate tested `vg_lb` alone until 2026-08-26, when `vg_lme`
    silently disabled condition-blocked draws; it was fixed by adding one
    disjunct. On 2026-08-28 the same omission recurred for `pooled_vg` -- a
    pooled-only arm ran with blocked draws off, backward groups collapsed to
    1.30 rows/condition against the control's 1.99, and three quarters of
    backward rows contributed exactly zero. Adding a disjunct did not prevent
    the recurrence; removing the list from the call site does.

    This test therefore asserts the INVERSE of its predecessor: the gate must
    name no coefficients at all.
    """
    src = TRAIN.read_text(encoding='utf-8')
    i = src.index("blc = self.args.bwd_loss_coeffs")
    gate = src[i:i + 900]
    assert '_runs_grouped_vargrad' in gate, (
        'the bwd blocked-draw gate no longer routes through the shared '
        'predicate; it will drift out of sync with config_invariants again')
    for name in ("'vg_lb'", "'vg_lme'", "'pooled_vg'"):
        assert name not in gate.split('block_m =')[1][:300], (
            f'the gate re-opens the coefficient list ({name}); add it to '
            'config_invariants.BRANCH_VARGRAD_COEFFS instead')
