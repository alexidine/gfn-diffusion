"""Warm-starting a rebuilt prior: same target, different filename.

A prior topped up with more noised rows around the SAME anchors at the SAME
calibrated noise range describes the same distribution -- but `prior_path` is
hashed into the problem definition by FILENAME, so the identity moves when the
target has not, and assert_problem_match refuses the old checkpoints.

`warm_start_ignore_problem_keys` lets a config assert the two priors are one
target. It is honoured ONLY on the weights-only path, which restores no buffers,
no optimizer and no step count -- buffer corruption being the guard's own stated
reason for existing.
"""
import pytest

from energy_sampling.checkpointing import Checkpointer


def _mgr(**argkw):
    mgr = Checkpointer.__new__(Checkpointer)
    mgr.modeller = type('M', (), {'args': type('A', (), argkw)()})()
    return mgr


def test_no_key_is_exempt_by_default():
    """Absent config key means the CODE DEFAULT, and the default is to exempt
    nothing -- otherwise the guard weakens for every run that never asked."""
    assert _mgr()._warm_start_ignore_keys() == ()
    assert _mgr(warm_start_ignore_problem_keys=None)._warm_start_ignore_keys() == ()
    assert _mgr(warm_start_ignore_problem_keys=[])._warm_start_ignore_keys() == ()


def test_prior_path_is_exemptible():
    m = _mgr(warm_start_ignore_problem_keys=['prior_path'])
    assert m._warm_start_ignore_keys() == ('prior_path',)


@pytest.mark.parametrize('key', ['energy_function', 'space_groups', 'z_primes',
                                 'mol_cond', 'vec_cond', 'energy_config'])
def test_the_keys_that_change_the_TARGET_cannot_be_exempted(key):
    """Weights trained under a different energy function or crystal system do not
    transfer, and the guard is the only thing that says so. A config must not be
    able to switch it off wholesale."""
    with pytest.raises(ValueError, match='may not be exempted'):
        _mgr(warm_start_ignore_problem_keys=[key])._warm_start_ignore_keys()


def test_one_bad_key_rejects_the_whole_list():
    with pytest.raises(ValueError, match='energy_function'):
        _mgr(warm_start_ignore_problem_keys=['prior_path',
                                             'energy_function'])._warm_start_ignore_keys()


def test_the_exemption_actually_lets_a_rebuilt_prior_through():
    """End to end on the comparison itself: two problem defs differing ONLY in
    prior_path pass with the exemption and fail without it."""
    old = {'energy_function': 'uma', 'prior_path': '/p/mipcas_uma_f047.pt',
           'space_groups': [2], 'z_primes': [1]}
    new = dict(old, prior_path='/p/mipcas_uma_f047_200k.pt')

    mgr = _mgr(warm_start_ignore_problem_keys=['prior_path'])
    mgr.modeller.problem_def = new

    ck = {'problem_def': old}
    with pytest.raises(ValueError, match='different problem'):
        mgr.assert_problem_match(ck, 'x.pt', 'checkpoint_name')
    mgr.assert_problem_match(ck, 'x.pt', 'checkpoint_name',
                             ignore_keys=mgr._warm_start_ignore_keys())


def test_a_genuinely_different_problem_still_fails_WITH_the_exemption():
    """The exemption must not become a blanket pass: change the energy function
    too and the guard has to fire even though prior_path is exempt."""
    old = {'energy_function': 'uma', 'prior_path': '/p/a.pt', 'space_groups': [2]}
    new = {'energy_function': 'elj', 'prior_path': '/p/b.pt', 'space_groups': [2]}
    mgr = _mgr(warm_start_ignore_problem_keys=['prior_path'])
    mgr.modeller.problem_def = new
    with pytest.raises(ValueError, match='different problem'):
        mgr.assert_problem_match({'problem_def': old}, 'x.pt', 'checkpoint_name',
                                 ignore_keys=mgr._warm_start_ignore_keys())
