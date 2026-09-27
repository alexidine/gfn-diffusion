"""`model.zero_init` reaches the set head (conformer_modeller.py::ConformerModeller._install_set_policy).

The set head is swapped in after the GFN is built, and the swap used to build it without
`zero_init`, so the key was inert under `policy_kind: set`: an untrained set head was a random
function of its per-coordinate features. Any change in the feature width re-drew it -- adding
`row_ez_parity` and `is_held_bond` to energies/dof_features.py moved the untrained-policy IS
proposal of tests/conformer/test_logz_check.py from ESS 365 to 38 on NH3 at level full
(N = 150000, seed 0) with the chart, energy and prior bitwise unchanged. With the key honoured
the untrained head emits exactly 0: its output layer has no bias.

All three heads the swap can build are covered -- ragged (a multi-molecule set), dense
conditional and dense unconditional (one molecule) -- each with a `zero_init: false` separator,
so the test cannot pass because every head starts at zero anyway.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/conformer/test_set_policy_zero_init.py
"""
from copy import deepcopy

import pytest
import torch

from conformer_modeller import ConformerModeller


class _Args:
    checkpoint_name = None
    continue_from_checkpoint = False


def _stub(smiles, conditional, zero_init):
    """A ConformerModeller holding a real energy and a real GFN, and nothing else
    `_install_set_policy` reads; the optimizer rebuild is stubbed."""
    from energies.multi_conformer import MultiConformerTorsions
    from models.gfn import GFN

    en = MultiConformerTorsions(smiles, identifiers=smiles, device='cpu', level='full',
                                force_field='mmff')
    m = ConformerModeller.__new__(ConformerModeller)
    m.energy_function = en
    m.device = 'cpu'
    m.args = _Args()
    m.args.embedding_conditioning = conditional
    m.args.embedding_conditioning_dim = 16
    m._policy_spec = {'policy_kind': 'set', 'set_policy_hidden': 16, 'set_policy_layers': 2,
                      'set_policy_corr_dim': 8}
    m.gfn_config = dict(dim=en.data_ndim, s_emb_dim=16, conditions_dim=16, harmonics_dim=4,
                        t_dim=8, t_hidden_dim=16, condition_embedding_dim=8,
                        conditional=True, learn_pb=True, device='cpu',
                        do_periodic_angles=False, angular_mask=en.periodic_dims,
                        s_hidden_dim=16, policy_hidden_dim=16, flow_hidden_dim=16,
                        cond_hidden_dim=16, s_layers=2, policy_layers=2, flow_layers=2,
                        cond_layers=2, dplr_rank=0, zero_init=zero_init)
    m.gfn_model = GFN(**m.gfn_config)
    m.ema_model = deepcopy(m.gfn_model)
    m.init_schedulers_optimizers = lambda: None
    return m


def _heads():
    from models.ragged_set_policy import RaggedConditionalSetPolicy
    from models.set_policy import ConditionalSetPolicy, SetPolicy
    return [(['N', 'C=O'], True, RaggedConditionalSetPolicy),
            (['N'], True, ConditionalSetPolicy),
            (['N'], False, SetPolicy)]


@pytest.mark.parametrize('case', range(3), ids=['ragged', 'conditional', 'dense'])
def test_zero_init_zeroes_every_set_heads_output_layer(case):
    smiles, conditional, cls = _heads()[case]
    for zero_init in (True, False):
        m = _stub(smiles, conditional, zero_init)
        ConformerModeller._install_set_policy(m)
        for model in (m.gfn_model, m.ema_model):
            head = model.forward_policy
            assert type(head) is cls, (type(head), cls)
            out = head.rho.output_layer
            assert out.bias is None, 'a bias would make the zeroed head emit a constant, not 0'
            zeroed = bool((out.weight == 0).all())
            assert zeroed is zero_init, (cls.__name__, zero_init, float(out.weight.abs().max()))


def test_an_untrained_zero_init_dense_head_emits_exactly_zero():
    """End to end on the head whose forward needs no batch: whatever the state and time, the
    untrained output is 0 -- the zero-drift proposal the logz test's untrained policy relies on."""
    m = _stub(['N'], False, True)
    ConformerModeller._install_set_policy(m)
    head = m.gfn_model.forward_policy
    g = torch.Generator().manual_seed(0)
    dtype = head.static.dtype
    state = torch.rand(5, head.dim, generator=g, dtype=dtype) * 2 - 1
    t_emb = torch.randn(5, int(m.gfn_config['t_dim']), generator=g, dtype=dtype)
    out = head(state, t_emb)
    assert out.shape == (5, 2 * head.dim) and bool((out == 0).all())
