"""The progress gate's evidence must survive a resume.

progress_gate needs SIX evals per metric before `_series` returns anything, and
the production phase-1 stage exits on `gates/progress_done`:

    exit    = [{'metric': 'gates/progress_done', 'above': 0.5, 'patience': 1}]
    on_exit = ['snapshot:phase1_exit', 'snapshot_prior']

so on_exit is the ONLY writer of phase1_exit.pt and the prior. With the history
unpersisted a resumed leg started empty and could not conclude for ~6 evals --
~3000 steps at eval_period 500, ~5.8 h on the MLIP route.

`min_history` does not save it. That guard is keyed on the ABSOLUTE step, which a
resume inherits at ~19000, so it passes instantly and the blackout is invisible.

The failure that matters is not wasted steps: a leg that ENDS inside the blackout
never fires on_exit, so it writes no phase1_exit and no prior -- and a _best.pt
without a phase1_exit is the known-fatal case. On a leg shorter than the rebuild,
phase 1 could never exit at all.
"""
import pytest

from energy_sampling.checkpointing import MODELLER_STATE_DEFAULTS
from energy_sampling import progress_metrics


def test_the_history_rides_in_the_checkpoint():
    assert '_progress_history' in MODELLER_STATE_DEFAULTS


def test_its_default_is_what_the_writer_expects():
    """train.py does `hist = getattr(self, '_progress_history', None)` and creates
    {} when it is None, so None is the only safe default -- and the __init__ loop
    over MODELLER_STATE_DEFAULTS means the attribute always exists for save()."""
    assert MODELLER_STATE_DEFAULTS['_progress_history'] is None


def test_min_history_is_keyed_on_the_absolute_step_which_a_resume_inherits():
    """Why the guard could not catch this: it passes instantly on a resumed leg."""
    spec = {'min_history': 2000, 'metrics': [], 'veto_metrics': []}
    out = progress_metrics.progress_gate({}, spec, step=19000.0)
    assert out['gates/progress_done'] == 0.0, 'still not done -- but not because of min_history'
    blocked = progress_metrics.progress_gate({}, spec, step=100.0)
    assert blocked['gates/progress_done'] == 0.0


@pytest.mark.parametrize('n', [0, 1, 5])
def test_a_short_history_cannot_conclude(n):
    """_series needs 6 entries; below that the gate cannot say done, whatever the
    step is. This is the blackout, and it is why the evidence has to persist."""
    spec = {'min_history': 2000,
            'metrics': [{'key': 'bwd/mle', 'target': 0.0}],
            'veto_metrics': []}
    hist = {'bwd/mle': [(float(i * 500), 1.0) for i in range(n)]}
    out = progress_metrics.progress_gate(hist, spec, step=19000.0)
    assert out['gates/progress_done'] == 0.0
