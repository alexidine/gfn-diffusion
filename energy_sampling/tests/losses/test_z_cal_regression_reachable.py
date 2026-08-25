"""
`z_calibration mode: regression` must be REACHABLE.

The regression branch of `z_calibration_tick` consumes exactly one thing:
`self._z_cal_cache`, filled by `_stash_z_cal_cache` after every fwd rollout.
That stash used to gate on `cfg.enabled` -- a key `utils._RETIRED_KEYS` rejects
at load, so it could never be truthy. The cache stayed None, the tick's
`mode == 'regression'` guard returned every time, and the mode was dead while
reading exactly like a calibration with nothing to do (`z_cal/p = 0.0`, no
`z_cal/train_rms`).

These tests drive the stash directly against a stub. They are about REACHABILITY
-- does the cache get filled when a config asks for regression -- not about
whether the regression fit is any good. The general form of the defect is pinned
separately in tests/config/test_no_gating_on_retired_keys.py.
"""

from types import SimpleNamespace

import pytest
import torch

from energy_sampling.train import Modeller


def _stub(mode='regression', flag=True, n_rows=6, n_cond=3):
    """The narrowest object `_stash_z_cal_cache` reads: a z_calibration config,
    a protocol flag, and a model carrying the detached embedding."""
    emb = torch.arange(n_rows * 4, dtype=torch.float32).reshape(n_rows, 4)
    ids = torch.tensor([i % n_cond for i in range(n_rows)])
    return SimpleNamespace(
        args=SimpleNamespace(z_calibration=SimpleNamespace(mode=mode)),
        protocol=SimpleNamespace(flag=lambda name: flag),
        gfn_model=SimpleNamespace(_z_cal_embedding=emb),
        _z_cal_cache=None,
    ), ids, n_cond


def test_the_cache_fills_under_regression_mode():
    """The fix. Before it, this asserted None on every input."""
    s, ids, n_cond = _stub()
    Modeller._stash_z_cal_cache(s, ids)
    assert s._z_cal_cache is not None, (
        'regression mode is unreachable: the stash refused a config that asks '
        'for it with the stage flag set')
    emb, uniq = s._z_cal_cache
    # one row per UNIQUE condition -- repeats broadcast identical embedding rows
    assert emb.shape[0] == n_cond
    assert uniq.tolist() == sorted(set(ids.tolist()))


def test_the_rollout_route_does_not_pay_for_the_stash():
    """Not merely an optimisation: on the rollout route nothing reads this
    cache, so filling it is an embedding slice and a unique() per fwd step for
    no consumer."""
    s, ids, _ = _stub(mode='rollout')
    Modeller._stash_z_cal_cache(s, ids)
    assert s._z_cal_cache is None


def test_a_stage_not_flagging_z_calibration_does_not_fill():
    """The flag is the switch `z_calibration.enabled` was relocated INTO, so a
    stage that omits it must behave as off."""
    s, ids, _ = _stub(flag=False)
    Modeller._stash_z_cal_cache(s, ids)
    assert s._z_cal_cache is None


def test_an_unconditional_rollout_fills_nothing():
    """No conditioner, so no embedding. Regression is a conditional-route
    feature and must decline here rather than raise."""
    s, ids, _ = _stub()
    s.gfn_model._z_cal_embedding = None
    Modeller._stash_z_cal_cache(s, ids)
    assert s._z_cal_cache is None


def test_a_mispaired_batch_is_dropped_not_mispaired():
    """Guarding the guard: if ids and embedding rows disagree the stash must
    drop the batch. Pairing them anyway would regress log_Z(c) onto the wrong
    conditions, which is worse than not calibrating."""
    s, ids, _ = _stub()
    Modeller._stash_z_cal_cache(s, ids[:-2])
    assert s._z_cal_cache is None


def test_reintroducing_the_retired_gate_would_fail_these():
    """Mutation. Restore the old `cfg.enabled` gate and require the reachability
    test above to stop passing -- otherwise it could pass for reasons unrelated
    to the gate."""
    s, ids, _ = _stub()
    cfg = s.args.z_calibration
    assert not getattr(cfg, 'enabled', False), (
        'a config object carrying `enabled` means the retirement was undone; '
        'the old gate would then pass and this suite would stop proving anything')
