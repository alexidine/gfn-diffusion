"""
THE INVARIANT: a Z-side gradient reaches flow_model's own parameters and nothing else.

WHY IT IS A TEST AND NOT A CONVENTION. The previous containment was
`freeze_z` / `freeze_policy` -- per-branch loss coefficients that a stage's
`loss_coeffs` had to remember to set. `var_conditioning` set neither, so on the
QM9 conditional route the `emp_z` term backpropagated through
`flow_model(condition_embedding)` into conditions_embedding_model every forward
step, and the SAME tensor feeds `s_model(state, condition_embedding)`, which is
upstream of both policies. conditions_embedding_model also sits in the POLICY
param groups (utils.py:93), so the leaked Z gradient was applied at the
servo-managed policy rate -- raising the policy LR amplified a Z-side gradient.

MEASURED, 2026-08-17, battery hyperslope_aug17, QM9 conditional, rate pinned at
5e-4: conditions_embedding_model and flow_model sit at exactly 0 until the
train_prior -> var_conditioning transition switches them on at step 510, then
grow 187x and 70x within 40 steps while forward_policy grows 1.1x and
backward_policy 1.4x. The Z side detonates on its own and the NaN propagates
into the policy.

SCOPE, stated so this file is not read as more than it is: cutting this path is
a CONTRIBUTOR, not a cure. The arm that cut the same path via `freeze_z` moved
the detonation from step 560 to 976 rather than preventing it. What this test
pins is the wiring invariant, not run survival.

The full_flow case is covered too, because the leak there is a different one:
that head reads [s_emb, t_emb] from s_model and t_model, so detaching the
conditioner does nothing for it. `_step_flow`'s own docstring used to carry a
note saying "revisit here if full_flow + freeze_policy is ever used"; these
tests are that revisit, and the detach is unconditional rather than tied to
freeze_policy, because the invariant is about WHERE Z gradient may go, not about
which branch happens to be training.

THE SECOND ASSERTION IS THE ONE THAT KEEPS THE FIX HONEST: the conditioner must
still train from the POLICY path. A detach that silently froze the conditioner
would pass every "no Z gradient" assertion in this file while deleting
conditioning entirely.
"""

import pytest
import torch

from models.gfn import GFN

pytestmark = pytest.mark.fast

DIM = 4
BATCH = 5


def _gfn(full_flow=False):
    """A small conditional GFN on CPU. Dims are tiny on purpose -- this is a
    wiring test, and nothing here depends on capacity."""
    return GFN(dim=DIM, s_emb_dim=16, harmonics_dim=8, t_dim=8,
               t_hidden_dim=16, s_hidden_dim=16, s_layers=2,
               policy_hidden_dim=16, policy_layers=2,
               flow_hidden_dim=16, flow_layers=2,
               cond_hidden_dim=16, cond_layers=2,
               conditions_dim=3, condition_embedding_dim=8,
               conditions_type='vector', conditional=True,
               full_flow=full_flow, device=torch.device('cpu'),
               do_periodic_angles=False, hold_dead_latent_rows=False)


def _grad_norm(module):
    """Total gradient magnitude over a submodule. 0.0 means nothing reached it."""
    grads = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
    return sum(grads) if grads else 0.0


# ------------------------------------------------ constant flow (the live route)

def test_condition_flow_trains_only_the_flow_head():
    """Backward on log Z(c) alone. flow_model must move; nothing else may."""
    gfn = _gfn(full_flow=False)
    gfn.zero_grad(set_to_none=True)

    embedding = gfn.get_condition_embedding(torch.randn(BATCH, 3), None)
    gfn._condition_flow(embedding).sum().backward()

    assert _grad_norm(gfn.flow_model) > 0, 'the flow head did not train at all'
    for name in ('conditions_embedding_model', 's_model',
                 'forward_policy', 'backward_policy'):
        reached = _grad_norm(getattr(gfn, name))
        assert reached == 0.0, f'Z gradient reached {name} (|grad| = {reached:.4g})'


def test_the_conditioner_still_trains_from_the_policy_path():
    """The fix must not be "detach the conditioner", which would pass every
    assertion above while deleting conditioning."""
    gfn = _gfn(full_flow=False)
    gfn.zero_grad(set_to_none=True)

    embedding = gfn.get_condition_embedding(torch.randn(BATCH, 3), None)
    state = gfn.expand_state_for_policy(torch.randn(BATCH, DIM))
    gfn.s_model(state, embedding).sum().backward()

    assert _grad_norm(gfn.conditions_embedding_model) > 0, \
        'the conditioner is frozen -- conditioning has been deleted, not isolated'


# ---------------------------------------------------------------- full flow

def test_step_flow_does_not_train_the_shared_trunk():
    """Under full_flow the head reads [s_emb, t_emb], both SHARED with the
    policy, so the conditioner detach does nothing here and this needs its own."""
    gfn = _gfn(full_flow=True)
    gfn.zero_grad(set_to_none=True)

    # built exactly as _forward_kernel builds them (models/gfn.py:643-645)
    embedding = gfn.get_condition_embedding(torch.randn(BATCH, 3), None)
    s_emb = gfn.s_model(gfn.expand_state_for_policy(torch.randn(BATCH, DIM)), embedding)
    t_emb = gfn.t_model(torch.rand(BATCH))

    gfn._step_flow(s_emb, t_emb).sum().backward()

    assert _grad_norm(gfn.flow_model) > 0, 'the flow head did not train at all'
    for name in ('s_model', 't_model', 'conditions_embedding_model'):
        reached = _grad_norm(getattr(gfn, name))
        assert reached == 0.0, f'Z gradient reached {name} (|grad| = {reached:.4g})'


def test_step_flow_is_none_when_full_flow_is_off():
    """The constant-flow route must not silently acquire a per-step head."""
    assert _gfn(full_flow=False)._step_flow(None, None) is None
