"""Trajectory checkpointing is per BRANCH, because the branches do not share a
memory profile.

It trades ~33x trajectory activation memory for recompute (33.6x measured at
T=100) and was applied identically to fwd, bwd and replay. But only fwd holds the
energy function's footprint at the same time, and fwd runs 1 step in N -- so at
N=200 the 99.5% of steps that are bwd+replay pay the recompute while sitting on
headroom nothing is using.

What these pin is the GATING, not the win: the win is a wall-clock measurement at
production T and batch, and the risk it does not remove is allocator
fragmentation across a bwd step and the next rollout.
"""
import pytest
import torch

from energy_sampling.models.gfn import GFN

MODES = ('fwd', 'bwd', 'replay')


def _g(enabled=True, modes=None):
    g = GFN.__new__(GFN)
    g.traj_checkpoint = enabled
    g.traj_checkpoint_modes = modes
    return g


def test_none_means_every_branch_the_historical_behaviour():
    g = _g(modes=None)
    with torch.enable_grad():
        assert all(g._use_traj_checkpoint(m) for m in MODES)


def test_an_empty_collection_also_means_every_branch():
    """Falsy is 'unset', not 'none of them' -- a config writing [] must not
    silently disable checkpointing everywhere and blow the memory estimate."""
    for empty in ([], (), None):
        g = _g(modes=empty)
        with torch.enable_grad():
            assert all(g._use_traj_checkpoint(m) for m in MODES), empty


def test_the_intended_production_setting_checkpoints_only_fwd():
    g = _g(modes=('fwd',))
    with torch.enable_grad():
        assert g._use_traj_checkpoint('fwd') is True
        assert g._use_traj_checkpoint('bwd') is False
        assert g._use_traj_checkpoint('replay') is False


@pytest.mark.parametrize('mode', MODES)
def test_the_master_switch_still_outranks_the_per_branch_list(mode):
    g = _g(enabled=False, modes=('fwd', 'bwd', 'replay'))
    with torch.enable_grad():
        assert g._use_traj_checkpoint(mode) is False


@pytest.mark.parametrize('mode', MODES)
def test_no_grad_disables_it_everywhere(mode):
    """Why listing fwd is usually moot: with fracs.fwd = 0 the forward branch
    carries no gradient, and checkpointing a graph nobody backprops is already a
    no-op through this same guard."""
    g = _g(modes=None)
    with torch.no_grad():
        assert g._use_traj_checkpoint(mode) is False


def test_an_unknown_branch_name_does_not_silently_checkpoint():
    g = _g(modes=('fwd',))
    with torch.enable_grad():
        assert g._use_traj_checkpoint('typo') is False


# ---------------------------------------------------------------------------
# The two places a per-branch restriction re-opens a hole someone already closed
# ---------------------------------------------------------------------------

def test_the_peak_cache_key_separates_fwd_only_from_all_branches():
    """gpu_guard's key already carries traj_checkpoint because reusing an ON peak
    for an OFF run under-estimates, the direction that crashes the box. Modes
    re-open that hole one level down: fwd-only peaks like OFF on the 99.5% of
    steps that are bwd+replay."""
    from energy_sampling.gpu_guard import _traj_checkpoint_key as k

    assert k({'traj_checkpoint': True, 'traj_checkpoint_modes': ['fwd']}) \
        != k({'traj_checkpoint': True})
    assert k({'traj_checkpoint': True, 'traj_checkpoint_modes': ['fwd']}) \
        != k({'traj_checkpoint': False})


def test_an_unset_restriction_hashes_to_the_HISTORICAL_value():
    """Otherwise every peak already cached on every box silently re-measures."""
    from energy_sampling.gpu_guard import _traj_checkpoint_key as k

    assert k({'traj_checkpoint': True}) == '1'
    assert k({'traj_checkpoint': True, 'traj_checkpoint_modes': []}) == '1'
    assert k({'traj_checkpoint': True, 'traj_checkpoint_modes': None}) == '1'
    assert k({'traj_checkpoint': False}) == '0'
    assert k({}) == '0'


def test_the_key_does_not_depend_on_the_order_modes_were_written():
    from energy_sampling.gpu_guard import _traj_checkpoint_key as k

    assert k({'traj_checkpoint': True, 'traj_checkpoint_modes': ['bwd', 'fwd']}) \
        == k({'traj_checkpoint': True, 'traj_checkpoint_modes': ['fwd', 'bwd']})


def test_the_protocol_panic_lever_drops_the_per_branch_restriction():
    """set_traj_checkpoint:1 is what a stage that OOMs reaches for. Re-arming
    fwd alone -- the branch that runs 1 step in N and is usually under no_grad --
    would print a lever that fired while the peak did not move."""
    from energy_sampling.protocol import StageProtocol

    class _Model:
        traj_checkpoint = False
        traj_checkpoint_modes = ('fwd',)

    class _Args:
        traj_checkpoint = False
        traj_checkpoint_modes = ['fwd']

    class _M:
        pass

    class _Proto(StageProtocol):          # `stage` is a read-only property
        stage = type('S', (), {'name': 'equilibration'})()

    m, gfn, ema = _M(), _Model(), _Model()
    m.args, m.gfn_model, m.ema_model, m.batch_sizer = _Args(), gfn, ema, object()
    proto = object.__new__(_Proto)
    proto.m = m

    _Proto._run_action(proto, 'set_traj_checkpoint', '1', {})

    assert m.args.traj_checkpoint is True
    assert not m.args.traj_checkpoint_modes, 'args kept a restriction the panic lever must drop'
    for mdl in (gfn, ema):
        assert mdl.traj_checkpoint is True
        assert not mdl.traj_checkpoint_modes, 'model still checkpoints one branch only'
    assert m.batch_sizer is None, 'ladder must re-arm: per-sample memory changed'
