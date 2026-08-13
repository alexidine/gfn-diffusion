"""
Does the noise calibration measure the model it SAYS it measures?

`calibrate_noise.py` is a measurement instrument, and the failure that matters
for an instrument is not crashing -- it is running cleanly on the wrong subject.
That already happened: `REGIMES['mle_fresh'] = None` meant "do not OVERRIDE the
checkpoint", the config's own `checkpoint_name` therefore applied, and the
`mle_fresh` column reported a CONVERGED model. It is the same file
`mle_converged` names explicitly, so two of four regimes were one measurement
under two names and nothing in the output said so.

These tests are cheap (no torch, no GPU, no `train.py`) because they test the
guard rather than the run: `_assert_regime` is the whole fix, so it is the thing
that has to fail on the historical inputs. `test_the_historical_bug_now_raises`
replays them exactly -- a fresh regime, the converged checkpoint loaded, step
10001 -- and REQUIRES a failure. A guard nobody has watched fail is a guard
nobody has tested.
"""
import os

import pytest

from bench import calibrate_noise as cn

CKPT_A = 'checkpoints/a_running.pt'
CKPT_B = 'checkpoints/b_final.pt'


# ------------------------------------------------------------------ the bug

def test_the_historical_bug_now_raises():
    """
    The exact inputs of the run that produced `noise_calib_mle_fresh.json`:
    regime `mle_fresh` (ckpt None), the config's `..._final.pt` loaded anyway,
    and a first step of 10001. That run reported a median cos of -0.0448 as a
    FRESH MLE model. It must now be impossible to get that number back without
    an exception.
    """
    with pytest.raises(RuntimeError, match='FRESH regime LOADED A CHECKPOINT'):
        cn._assert_regime([cn.CK_OLD + 'final.pt'], None, 10001, 'equilibration')


def test_mle_fresh_and_mle_converged_are_not_the_same_file():
    """
    WHY `None` HAD TO CHANGE MEANING, asserted rather than remembered.

    The trap is a property of the config: `elj_nehzor_sg14_t10_r2.yaml` ships a
    `checkpoint_name`, so "no override" resolves to a real checkpoint -- and that
    checkpoint is byte-for-byte the one `mle_converged` asks for. If this ever
    stops holding, the fix is still correct but this file's rationale has moved,
    and the next reader should be told rather than left to rediscover it.
    """
    cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       cn.CONFIG)
    if not os.path.exists(cfg):
        pytest.skip(f'config not on disk: {cfg}')
    with open(cfg) as f:
        named = [ln.split(':', 1)[1].strip() for ln in f
                 if ln.startswith('checkpoint_name:')]
    assert named, 'config no longer sets checkpoint_name -- see docstring'
    assert os.path.basename(cn.REGIMES['mle_converged']) == named[0], (
        'the config no longer names the same checkpoint as the mle_converged '
        'regime; `None` is now a different trap than the one documented')
    assert cn.REGIMES['mle_fresh'] is None


# ------------------------------------------------------------------ fresh

def test_fresh_from_step_zero_passes():
    cn._assert_regime([], None, 0, 'mle')


def test_fresh_that_resumed_without_a_recorded_load_raises():
    """
    The second, independent mechanism. Even if the loader spy is bypassed --
    some other path restores training state -- a fresh run cannot start at step
    10001. One check on the args, one on the state; the args are what lied last
    time.
    """
    with pytest.raises(RuntimeError, match='resumed at step'):
        cn._assert_regime([], None, 10001, 'mle')


# ----------------------------------------------------------------- loaded

def test_loaded_matching_checkpoint_passes():
    cn._assert_regime([CKPT_A], CKPT_A, 10642, 'equilibration')


def test_loaded_a_different_checkpoint_raises():
    with pytest.raises(RuntimeError, match='the loader opened'):
        cn._assert_regime([CKPT_B], CKPT_A, 10642, 'equilibration')


def test_asked_for_a_checkpoint_and_loaded_nothing_raises():
    with pytest.raises(RuntimeError, match='the loader opened'):
        cn._assert_regime([], CKPT_A, 10642, 'equilibration')


def test_a_second_load_mid_run_raises():
    """Samples spanning two models pool two regimes into one median."""
    with pytest.raises(RuntimeError, match='the loader opened'):
        cn._assert_regime([CKPT_A, CKPT_A], CKPT_A, 10642, 'equilibration')


def test_loaded_but_state_did_not_restore_raises():
    """
    A checkpoint that loads WEIGHTS ONLY leaves the step counter at zero, so the
    run is a fresh model wearing the regime's name -- the same defect as the
    headline bug with the sign flipped.
    """
    with pytest.raises(RuntimeError, match='did not restore'):
        cn._assert_regime([CKPT_A], CKPT_A, 0, 'mle')


def test_relative_and_absolute_paths_compare_equal():
    """The spy records whatever `init_gfn` built; the regime holds a relative
    path. Comparing those as strings would fail every real run."""
    cn._assert_regime([os.path.abspath(CKPT_A)], CKPT_A, 10642, 'eq')
