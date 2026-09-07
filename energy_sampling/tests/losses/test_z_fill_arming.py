"""Modeller._z_fill_head_is_fillable -- when a forward batch may set log Z.

The predicate used to be four inline lines inside _stash_z_fill_logw. It is
extracted because it is the single point at which the Z pin can be turned off
by a config change made for another reason entirely: with it refusing, the
stash stays None forever, z_level_fill returns on its first line, log Z is
unpinned, and NOTHING errors (docs/design/rarer_rollouts.md, invariant 1).

The rule is that the forward branch must not be training the policy against the
same batch it levels from -- otherwise the fill snaps to a target that is still
moving. `freeze_policy > 0` is today's way of satisfying that. `tb_z_source:
batch_root` under a rollout cadence is the second, added ahead of the item that
introduces the coefficient: inert now (nothing sets it), so today's behaviour is
unchanged, but a build that turns freeze_policy off without it silently unpins Z.
"""
from types import MethodType, SimpleNamespace

import pytest

from energy_sampling.train import Modeller


def _m(freeze_policy=1.0, tb_z_source=None, every=0, conditional=False, full_flow=False):
    coeffs = SimpleNamespace(freeze_policy=freeze_policy)
    if tb_z_source is not None:
        coeffs.tb_z_source = tb_z_source
    m = SimpleNamespace(
        args=SimpleNamespace(fwd_loss_coeffs=coeffs),
        gfn_model=SimpleNamespace(conditional=conditional, full_flow=full_flow),
        protocol=SimpleNamespace(stage=SimpleNamespace(fwd_rollout_every=every)))
    m.tb_z_source = MethodType(Modeller.tb_z_source, m)
    m._z_fill_head_is_fillable = MethodType(Modeller._z_fill_head_is_fillable, m)
    return m


def test_todays_equilibration_stage_is_fillable():
    assert _m(freeze_policy=1.0, every=7)._z_fill_head_is_fillable() is True
    assert _m(freeze_policy=1.0, every=0)._z_fill_head_is_fillable() is True


def test_a_training_forward_branch_is_not_fillable():
    assert _m(freeze_policy=0.0)._z_fill_head_is_fillable() is False


@pytest.mark.parametrize('head', ['conditional', 'full_flow'])
def test_a_field_valued_head_is_never_fillable(head):
    """There is no single scalar to fill -- the level is a field."""
    assert _m(freeze_policy=1.0, every=7, **{head: True})._z_fill_head_is_fillable() is False
    assert _m(freeze_policy=0.0, tb_z_source='batch_root', every=7,
              **{head: True})._z_fill_head_is_fillable() is False


def test_batch_root_under_a_cadence_is_fillable():
    assert _m(freeze_policy=0.0, tb_z_source='batch_root',
              every=7)._z_fill_head_is_fillable() is True


def test_batch_root_without_a_cadence_is_refused():
    """The cadence conjunct is load-bearing: without it a stage could take its
    level from the batch root while training the policy at full weight on every
    step, which is the moving target the predicate exists to exclude."""
    assert _m(freeze_policy=0.0, tb_z_source='batch_root',
              every=0)._z_fill_head_is_fillable() is False


def test_the_default_source_is_inert():
    """tb_z_source falls back to 'learned', so the new clause changes nothing
    for any config that does not name it."""
    assert _m(freeze_policy=0.0, every=7)._z_fill_head_is_fillable() is False
    assert _m(freeze_policy=0.0, tb_z_source='learned',
              every=7)._z_fill_head_is_fillable() is False
