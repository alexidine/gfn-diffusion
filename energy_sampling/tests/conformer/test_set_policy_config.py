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
    """Default must be exactly the flat path on a single chart: absent key -> return before
    touching anything.

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


# `test_resume_is_refused_loudly` pinned a refusal that fired on the WRONG path -- a fresh
# launch with continue_from_checkpoint and no running file -- while no real reload reached
# it. The set policy now resumes; tests/conformer/test_set_policy_resume.py is the round trip
# that replaces it, including the fresh-launch case the refusal used to block.


def _install_stub(smiles, conditional=True, kind='set', dplr_rank=0):
    """A ConformerModeller holding a real energy and a real GFN, and nothing else
    _install_set_policy reads. The optimizer rebuild is stubbed: what is under test is
    WHICH head gets built, not the optimizer."""
    from copy import deepcopy

    from energies.multi_conformer import MultiConformerTorsions
    from models.gfn import GFN as BareGFN      # the class ConformerGFN subclasses

    en = MultiConformerTorsions(smiles, identifiers=smiles, device='cpu', level='full',
                                force_field='mmff')
    m = ConformerModeller.__new__(ConformerModeller)
    m.energy_function = en
    m.device = 'cpu'
    m.args = _Args()
    m.args.embedding_conditioning = conditional
    m.args.embedding_conditioning_dim = 16
    m._policy_spec = {'policy_kind': kind, 'set_policy_hidden': 16, 'set_policy_layers': 2,
                      'set_policy_corr_dim': 8}
    m.gfn_config = dict(dim=en.data_ndim, s_emb_dim=16, conditions_dim=16, harmonics_dim=4,
                        t_dim=8, t_hidden_dim=16, condition_embedding_dim=8,
                        conditional=True, learn_pb=True, device='cpu',
                        do_periodic_angles=False, angular_mask=en.periodic_dims,
                        s_hidden_dim=16, policy_hidden_dim=16, flow_hidden_dim=16,
                        cond_hidden_dim=16, s_layers=2, policy_layers=2, flow_layers=2,
                        cond_layers=2, dplr_rank=dplr_rank)
    m.gfn_model = BareGFN(**m.gfn_config)
    m.ema_model = deepcopy(m.gfn_model)
    m.init_schedulers_optimizers = lambda: None
    return m


def test_an_identity_layout_set_gets_the_ragged_head():
    """NH3 and H2CO both have 4 atoms, so at `full` their block counts agree (3|2|1): the
    layout is the IDENTITY and the energy is not a carrier. The dense conditional head would
    bake the REFERENCE member's static features into every row -- wrong for the other
    molecule, silently. More than one distinct molecule must get the ragged head."""
    from models.conformer_gfn import ConformerGFN
    from models.ragged_set_policy import RaggedConditionalSetPolicy

    m = _install_stub(['N', 'C=O'])
    assert not m.energy_function.is_carrier and m.energy_function.distinct_smiles == 2, \
        'separator: this set must be an identity layout, or the test proves nothing'
    ConformerModeller._install_set_policy(m)
    for model in (m.gfn_model, m.ema_model):
        assert isinstance(model.forward_policy, RaggedConditionalSetPolicy)
        assert type(model) is ConformerGFN and model._carrier is True
    stamp = m.gfn_config['conformer']
    assert stamp['carrier'] is True and stamp['block_width'] == [3, 2, 1]


def test_an_identity_layout_set_without_embeddings_is_refused():
    m = _install_stub(['N', 'C=O'], conditional=False)
    with pytest.raises(NotImplementedError, match='embedding_conditioning'):
        ConformerModeller._install_set_policy(m)


def test_a_flat_policy_with_dplr_on_a_carrier_is_refused():
    """DPLR's Woodbury forward density never calls ConformerGFN.gauss_logprob, the only
    masked path, so pad columns would enter the forward log-prob."""
    m = _install_stub(['C', 'CO', 'N'], kind='flat', dplr_rank=2)
    assert m.energy_function.is_carrier
    with pytest.raises(NotImplementedError, match='dplr_rank 2'):
        ConformerModeller._install_set_policy(m)
    ok = _install_stub(['C', 'CO', 'N'], kind='flat', dplr_rank=0)
    ConformerModeller._install_set_policy(ok)
    # the condition set rides in the same block (tests/conformer/test_condition_set_identity.py)
    stamp = dict(ok.gfn_config['conformer'])
    assert stamp.pop('condition_set')['identifiers'] == ['C', 'CO', 'N']
    assert stamp == {'policy_kind': 'flat', 'carrier': True, 'block_width': [5, 4, 3]}
