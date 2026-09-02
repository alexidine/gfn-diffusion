"""Gates for the `model.policy_kind: set` config key (conformer_modeller.py).

The first test is the one that matters and it caught a real bug during authoring. The key
names are popped out of `gfn_config` before it reaches `GFN(**cfg)`, because that constructor
has an explicit signature and would `TypeError` on an unknown key. But `policy_layers` and
`policy_hidden_dim` ALREADY exist in the model block as the FLAT policy's own GFN arguments --
so a set-policy key sharing either name would pop a live argument and unbuild the flat path,
with no error and a differently-shaped network.

    python -m pytest -q tests/conformer/test_set_policy_config.py
"""
import inspect
import os
import sys

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _root in (os.path.dirname(_here),
              os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import pytest

from energy_sampling.conformer_modeller import ConformerModeller
from energy_sampling.models.gfn import GFN


def test_set_policy_keys_cannot_shadow_a_gfn_argument():
    """Generic, so it keeps holding as either side grows a key."""
    keys = set(ConformerModeller._SET_POLICY_KEYS)
    gfn_args = set(inspect.signature(GFN.__init__).parameters)
    clash = keys & gfn_args
    assert not clash, (
        f'set-policy keys {sorted(clash)} shadow GFN constructor arguments. Popping one '
        f'removes a live argument from gfn_config and silently rebuilds the flat policy at '
        f'a different shape.')


def test_the_flat_policy_keys_really_are_gfn_arguments():
    """The separator for the test above -- without this, the clash test could pass simply
    because nothing is a GFN argument, and it would be blind."""
    gfn_args = set(inspect.signature(GFN.__init__).parameters)
    for name in ('policy_layers', 'policy_hidden_dim'):
        assert name in gfn_args, (
            f'{name} is no longer a GFN argument; the shadowing hazard the sibling test '
            f'guards has changed shape and both tests need revisiting')


def test_gfn_has_no_kwargs_sink():
    """Why popping is required at all. If GFN ever grows **kwargs, an unrecognised model key
    would be swallowed in silence instead of raising, and this whole mechanism would need a
    different guard."""
    params = inspect.signature(GFN.__init__).parameters.values()
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params), \
        'GFN grew a **kwargs sink; unknown model: keys would now fail silently'


def test_install_is_a_no_op_without_the_key():
    """Default must be exactly the flat path: absent key -> return before touching anything.

    Called on a bare instance with no runtime attached; reaching any further would raise
    AttributeError, so completing the call IS the assertion.
    """
    m = ConformerModeller.__new__(ConformerModeller)
    m._policy_spec = {}
    ConformerModeller._install_set_policy(m)          # no exception == no-op
    m._policy_spec = {'policy_kind': 'flat'}
    ConformerModeller._install_set_policy(m)


def test_unknown_policy_kind_is_refused():
    m = ConformerModeller.__new__(ConformerModeller)
    m._policy_spec = {'policy_kind': 'transformer'}
    with pytest.raises(ValueError, match='policy_kind'):
        ConformerModeller._install_set_policy(m)


class _Args:
    checkpoint_name = None
    continue_from_checkpoint = False


def test_dplr_is_refused_at_construction_not_mid_rollout():
    """The transpose is silent, so this must fail at startup rather than on the first step."""
    m = ConformerModeller.__new__(ConformerModeller)
    m._policy_spec = {'policy_kind': 'set'}
    m.args = _Args()
    m.gfn_config = {'dplr_rank': 6, 't_dim': 64}
    with pytest.raises(NotImplementedError, match='dplr_rank'):
        ConformerModeller._install_set_policy(m)


def test_resume_is_refused_loudly():
    """The policy choice is not in gfn_config, so the checkpointer would rebuild a FLAT GFN
    and its strict load would fail on the set head's weights. Refuse with the reason."""
    m = ConformerModeller.__new__(ConformerModeller)
    m._policy_spec = {'policy_kind': 'set'}
    m.args = _Args()
    m.args.checkpoint_name = 'something.pt'
    m.gfn_config = {'dplr_rank': 0, 't_dim': 64}
    with pytest.raises(NotImplementedError, match='cannot be resumed'):
        ConformerModeller._install_set_policy(m)
