"""Which energy_config keys constitute a PROBLEM's identity.

get_problem_definition folds all of energy_config into the identity minus
_NON_IDENTITY_ENERGY_CONFIG_KEYS, and assert_problem_match raises on the
checkpoint_name path -- which is exactly how a phase-1 exit is consumed, and how
prior_model_name loads too. So adding a key to energy_config without exempting it
silently REFUSES every checkpoint written before that key existed.

That happened: 504277b added the three MLIP execution knobs and every phase-1
exit stopped loading. These pin the rule so the next one is caught here instead
of at startup on the cluster.
"""
import glob
import yaml
import pytest

import utils


EXECUTION_KNOBS = ('mlip_compile', 'mlip_edge_chunk_size', 'mlip_activation_checkpointing')


@pytest.mark.parametrize('key', EXECUTION_KNOBS)
def test_the_mlip_execution_knobs_are_not_part_of_problem_identity(key):
    """They choose HOW the energy is computed -- compiled or eager, edge dim
    bucketed or not, activations stored or recomputed -- never WHAT it is. Same
    exemption internal_oom_recovery already carries."""
    assert key in utils._NON_IDENTITY_ENERGY_CONFIG_KEYS


def test_internal_oom_recovery_is_the_precedent():
    """Named so the rule above is anchored to something, not just asserted."""
    assert 'internal_oom_recovery' in utils._NON_IDENTITY_ENERGY_CONFIG_KEYS


def _energy_config(path):
    return dict((yaml.safe_load(open(path, encoding='utf-8')) or {}).get('energy_config') or {})


def test_a_pre_mlip_exit_config_still_matches_a_post_mlip_run_config():
    """The regression itself: an old mipu config and one carrying the new keys
    must differ ONLY in exempt keys, or the exit is refused at load."""
    old = [f for f in glob.glob('configs/prod_t100*/*.yaml') if 'mipu' in f]
    new = glob.glob('configs/mlipc_sep09/*.yaml')
    if not old or not new:
        pytest.skip('needs both a pre- and post-mlip_* config on disk')
    a, b = _energy_config(old[0]), _energy_config(new[0])
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    in_identity = differing - set(utils._NON_IDENTITY_ENERGY_CONFIG_KEYS)
    assert not in_identity, (
        'these energy_config keys differ AND are part of problem identity, so every '
        'phase-1 exit predating them is refused at load: %s' % sorted(in_identity))


# A broader guard was tried here and removed: "no identity key may appear in only a
# few configs" flags every ROUTE-SPECIFIC key as a false positive -- force_field,
# delta_r_max, theta_floor, scale_14 and ten more sit in 23 conformer configs and
# belong in the conformer problem's identity, because a conformer checkpoint is only
# ever loaded by a config that has them. Distinguishing "route-specific" from "added
# later" needs the route, not a count, and a guard that cries wolf gets ignored.
