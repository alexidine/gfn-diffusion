"""
What happens to a batch when the MLIP forward fails, end to end.

THE DEFECT THIS PINS. A non-OOM RuntimeError inside `safe_predict_uma` used to
substitute an all-ZEROS energy and tell nobody. Zero is finite and of plausible
magnitude for these quantities, so it passed every `isfinite` guard in the pipeline
and flowed into the TB target, the log-Z EMA, the replay buffer and the per-condition
running minimum, indistinguishable from a measurement. `uma_pot/(sym_mult*z_prime)`
is around -209 eV/molecule, so a zeroed GAS leg puts the lattice energy ~20,000
kJ/mol BELOW anything physical -- which reads as a spectacular discovery, exactly the
direction a sampler chases.

THE CONTRACT NOW, and each clause has a test below:

  1. a crashed batch returns NaN, never zero -- unusable by construction rather than
     plausibly wrong;
  2. the persistent sinks refuse it: ConditionLogZTracker.update_best_energy drops
     non-finite rows, because its scatter-min would otherwise pin a condition's
     minimum at NaN for the rest of the run;
  3. it is COUNTED and reported, so "has this ever fired?" is answerable from wandb;
  4. the gas-phase reference cache RAISES instead of caching a bad value, because
     that one is computed once per molecule and reused forever.

WHY THE NEGATIVE CONTROL MATTERS. `test_the_old_zeros_substitute_would_slip_through`
asserts that the previous behaviour passes the very filter that now catches NaN. A
guard whose failure mode has never been demonstrated is a guard of unknown power, and
this one is cheap to demonstrate.

CPU-only: no GPU, no checkpoint, no MLIP. The crash path is exercised through the
substitution helper and the consumers directly, which is the whole point -- the
failure is unreachable from a passing forward.
"""
import numpy as np
import pytest
import torch

from buffer import ConditionLogZTracker
from mxtaltools.mlip_interfaces import uma_utils as U


class _FakeBatch:
    """Only what `_crashed_energy` reads."""

    def __init__(self, n):
        self.num_graphs = n
        self.device = 'cpu'


# ------------------------------------------------------- 1. the substituted value

def test_crashed_energy_is_nan_not_zero():
    """The whole fix in one assertion. Zero was the old value and is the one thing
    this must never be, because zero survives every downstream finiteness filter."""
    energy = U._crashed_energy(_FakeBatch(6))
    assert energy.shape == (6,)
    assert torch.isnan(energy).all(), f'expected all-NaN, got {energy}'
    assert not (energy == 0).any(), 'a zero here is the original defect'


def test_crashed_energy_counts_calls_and_rows():
    """Counted per ROW as well as per call: one crash on a batch of 1000 is a very
    different event from one on a batch of 8, and a call counter alone cannot say
    how much reward was fabricated."""
    U._CRASH_CALLS = 0
    U._CRASH_ROWS = 0
    U._crashed_energy(_FakeBatch(4))
    U._crashed_energy(_FakeBatch(10))
    assert U._CRASH_ROWS == 14, f'expected 14 rows counted, got {U._CRASH_ROWS}'


def test_drain_reports_the_crash_counters_even_when_zero():
    """Absent-when-clean is indistinguishable in wandb from never-wired-up, which is
    the failure mode this counter exists to escape. So it reports 0, not nothing."""
    U._CRASH_CALLS = 0
    U._CRASH_ROWS = 0
    U._PHASE_CALLS = 1                     # drain returns {} when nothing was timed
    out = U.drain_uma_phase_timing()
    assert out['energy/uma_crash_calls'] == 0
    assert out['energy/uma_crash_rows'] == 0


def test_drain_resets_the_crash_counters():
    """Drained counters are per-window; a cumulative one would make every later
    window look like it crashed."""
    U._CRASH_CALLS, U._CRASH_ROWS, U._PHASE_CALLS = 3, 30, 1
    first = U.drain_uma_phase_timing()
    U._PHASE_CALLS = 1
    second = U.drain_uma_phase_timing()
    assert first['energy/uma_crash_rows'] == 30
    assert second['energy/uma_crash_rows'] == 0


def test_mace_reports_its_crash_counters_too():
    """
    BOTH MLIP routes or neither. The zeros substitution existed identically in
    AL_mace_utils, and 67 configs run the mace route -- fixing only uma would leave
    the same silent failure live on a production path while the metrics implied
    every MLIP was covered.
    """
    from mxtaltools.mlip_interfaces import AL_mace_utils as M
    M._CRASH_CALLS, M._CRASH_ROWS, M._PHASE_CALLS = 2, 20, 1
    out = M.drain_mace_phase_timing()
    assert out['energy/mace_crash_calls'] == 2
    assert out['energy/mace_crash_rows'] == 20
    M._PHASE_CALLS = 1
    assert M.drain_mace_phase_timing()['energy/mace_crash_rows'] == 0, 'not reset'


def test_neither_route_substitutes_zero():
    """
    The one-line statement of the whole contract, asserted against the SOURCE rather
    than a helper, so a future edit that reintroduces `torch.zeros` on either path
    fails here even if it never calls the helper.
    """
    import inspect
    from mxtaltools.mlip_interfaces import AL_mace_utils as M
    for mod in (U, M):
        src = inspect.getsource(mod)
        assert 'torch.zeros(batch.num_graphs' not in src, (
            f'{mod.__name__} substitutes a ZERO energy for a crashed batch again -- '
            f'zero is finite and plausible, so nothing downstream can reject it')


# ------------------------------------------- 3. the streak tripwire

class _AlwaysFails:
    """A predictor whose forward raises a non-OOM RuntimeError every time -- the
    deterministic case the retry cannot help with."""

    def predict(self, batch):
        raise RuntimeError('simulated non-OOM UMA failure')


def test_isolated_crashes_are_absorbed(monkeypatch):
    """A blip must NOT kill the run: below the bound the call reports failure and
    lets the caller substitute NaN."""
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda *a, **k: None)
    U._CONSECUTIVE_CRASHES = 0
    out, crashed = U.safe_predict_uma(_AlwaysFails(), None, retries=0)
    assert out is None and crashed is True


def test_a_persistent_failure_raises_instead_of_stalling(monkeypatch):
    """
    THE POINT OF THE TRIPWIRE. Substituting NaN indefinitely is a stalled run that
    still looks alive -- every batch dropped, the loop turning, gates never firing
    because NaN fails every comparison. After MAX_CONSECUTIVE_CRASHES the failure
    becomes an exception the caller's handler can actually act on.
    """
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda *a, **k: None)
    U._CONSECUTIVE_CRASHES = 0
    for _ in range(U.MAX_CONSECUTIVE_CRASHES - 1):
        U.safe_predict_uma(_AlwaysFails(), None, retries=0)
    with pytest.raises(RuntimeError, match='times in a row'):
        U.safe_predict_uma(_AlwaysFails(), None, retries=0)


def test_a_success_clears_the_streak(monkeypatch):
    """Otherwise crashes spread across an entire run eventually trip the wire for no
    reason -- the counter has to mean 'persistently broken now', not 'ever failed'."""
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda *a, **k: None)

    class _Works:
        def predict(self, batch):
            return {'energy': torch.zeros(1)}

    U._CONSECUTIVE_CRASHES = 0
    U.safe_predict_uma(_AlwaysFails(), None, retries=0)
    U.safe_predict_uma(_Works(), None, retries=0)
    assert U._CONSECUTIVE_CRASHES == 0, 'a success must reset the streak'


def test_recovery_failure_does_not_mask_the_original_error(monkeypatch):
    """On a sticky CUDA fault the synchronize/empty_cache recovery raises too. That
    secondary error must not escape -- it would bury the error worth reading."""
    def _sticky(*a, **k):
        raise RuntimeError('CUDA context is corrupt')

    monkeypatch.setattr(torch.cuda, 'synchronize', _sticky)
    monkeypatch.setattr(torch.cuda, 'empty_cache', _sticky)
    U._CONSECUTIVE_CRASHES = 0
    out, crashed = U.safe_predict_uma(_AlwaysFails(), None, retries=1)
    assert out is None and crashed is True


# ------------------------------------------- 4. eval aggregates survive one bad row

def test_nan_is_contagious_through_a_reduction():
    """
    The premise behind the eval fix, asserted rather than assumed. Unlike a wrong
    number, ONE NaN row takes out the whole pooled statistic -- which is why eval
    filters before reducing instead of relying on the substitution being visible.
    """
    from utils import logmeanexp
    lw = torch.tensor([1.0, 2.0, float('nan'), 3.0])
    assert torch.isnan(logmeanexp(lw)), 'premise: logmeanexp propagates NaN'
    assert torch.isnan(lw.mean())
    # and a NaN metric fails a threshold test in BOTH directions, so a gate reading
    # it never fires -- a stall that looks like a healthy run
    assert not (float('nan') < 0.5) and not (float('nan') > 0.5)


def test_eval_pooled_z_survives_one_crashed_row():
    """
    The fix itself, at the level of the arithmetic it protects: excluding the bad row
    must reproduce the estimate computed from the good rows alone, not merely be
    finite.
    """
    from utils import logmeanexp
    good = torch.tensor([1.0, 2.0, 3.0])
    with_bad = torch.tensor([1.0, 2.0, float('nan'), 3.0])

    finite = torch.isfinite(with_bad)
    filtered = logmeanexp(with_bad[finite])
    assert torch.isfinite(filtered)
    assert torch.allclose(filtered, logmeanexp(good)), (
        'filtering changed the estimate -- it must drop only the crashed row')


# ------------------------------------------- 2. the persistent sink refuses NaN

def _tracker(library_size=4):
    return ConditionLogZTracker(library_size=library_size)


def test_best_energy_rejects_nan():
    """
    THE LOAD-BEARING ONE. best_energy is a persistent running minimum reduced with
    scatter_reduce_(amin), and amin PROPAGATES NaN -- so without the filter a single
    crashed row pins that condition's minimum at NaN permanently, since every later
    comparison against NaN is False and no real energy can ever improve on it.
    """
    t = _tracker()
    ids = torch.tensor([0, 1, 2])
    t.update_best_energy(ids, torch.tensor([-50.0, float('nan'), -30.0]))

    assert torch.isfinite(t.best_energy[0]) and float(t.best_energy[0]) == -50.0
    assert torch.isfinite(t.best_energy[2]) and float(t.best_energy[2]) == -30.0
    assert not torch.isnan(t.best_energy[1]), (
        'a NaN energy reached the persistent minimum -- condition 1 can now never '
        'record a real best energy again')
    assert torch.isinf(t.best_energy[1]), (
        'condition 1 should be untouched (+inf, unvisited), not silently updated')


def test_best_energy_still_improves_after_a_crashed_row():
    """Recovery, not merely survival: the condition whose row was dropped must still
    accept a real measurement afterwards. A guard that left the slot unusable would
    pass the test above and still lose the condition."""
    t = _tracker()
    ids = torch.tensor([1])
    t.update_best_energy(ids, torch.tensor([float('nan')]))
    t.update_best_energy(ids, torch.tensor([-42.0]))
    assert float(t.best_energy[1]) == -42.0, (
        f'after a dropped NaN the condition recorded {t.best_energy[1]}, not the '
        f'real energy that followed')


def test_dropped_rows_are_counted():
    """Silently dropping is how the original defect worked; the count is what makes
    this a report rather than another swallow."""
    t = _tracker()
    t.update_best_energy(torch.tensor([0, 1]),
                         torch.tensor([float('nan'), float('-inf')]))
    assert t.nonfinite_energies_seen == 2, (
        f'expected 2 non-finite rows recorded, got {t.nonfinite_energies_seen}')


def test_an_all_nan_call_is_a_no_op():
    """The whole-batch crash case. It must not raise, and must not touch state --
    a crash should cost the batch, not the run."""
    t = _tracker()
    t.update_best_energy(torch.tensor([0, 1]),
                         torch.tensor([float('nan'), float('nan')]))
    assert torch.isinf(t.best_energy).all(), 'state was modified by an all-NaN call'


# ------------------------------------------------------- the negative control

def test_the_old_zeros_substitute_would_slip_through():
    """
    Demonstrates the defect rather than asserting the fix, which is the only way to
    show the new guard has any power at all.

    Zero passes `isfinite`, so under the old substitution the tracker accepted a
    fabricated energy as a real one -- and because it is a MINIMUM, 0.0 beats every
    genuine positive energy and sticks.
    """
    t = _tracker()
    ids = torch.tensor([0])
    old_style = torch.zeros(1)                       # what the code used to return
    assert torch.isfinite(old_style).all(), (
        'premise of this control: zeros pass the finiteness filters')

    t.update_best_energy(ids, torch.tensor([250.0]))  # a real, poor crystal
    t.update_best_energy(ids, old_style)              # the fabricated one wins
    assert float(t.best_energy[0]) == 0.0
    assert t.nonfinite_energies_seen == 0, (
        'the old value is invisible to the guard by construction -- which is why '
        'the substitute had to change, not just the filter')


# ---------------------------------------- 4. the cache refuses a bad gas reference

def test_gas_reference_cache_refuses_a_non_finite_value():
    """
    The one place a crash must STOP the run. The gas leg is computed once per
    molecule and reused for every future sample of it, so a cached NaN (or the old
    cached zero) offsets that molecule's lattice energy by the whole gas leg until
    the process exits. There is no per-row salvage and no later chance to notice.
    """
    from energies.molecular_crystal import MolecularCrystal

    obj = MolecularCrystal.__new__(MolecularCrystal)   # no __init__: no GPU, no MLIP
    obj.energy_function = 'uma'
    obj.host_gas_phase_reference = True
    obj._gas_pot_cache = {}
    obj.predictor = None

    class _Sub:
        def compute_lattice_gas_phase_uma(self, predictor):
            return torch.tensor([float('nan')])

    class _Batch:
        mol_id = torch.tensor([7])
        device = 'cpu'

        def subsample_new_batch(self, rows):
            return _Sub()

    with pytest.raises(RuntimeError, match='non-finite'):
        obj.attach_gas_phase_reference(_Batch())

    assert obj._gas_pot_cache == {}, (
        'the bad value was cached anyway -- every future sample of molecule 7 is '
        'now silently offset by the entire gas leg')
