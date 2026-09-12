"""The three per-row buffer columns: `birth_log_pf`, `is_val` and `origin`.

All are per-row side arrays, which is a shape with exactly one interesting
failure mode: a column that survives the pickle but not the REINDEX. It passes
every count check and then silently reports another row's value -- the lesson
the lj stamp taught on 2026-09-02. So the tests here check alignment BY MARKER
across add and purge, not by length.

`origin` is a LABEL -- it changes no admission, eviction or draw decision -- so
its whole value is that the code a row was admitted under is still the code it
reads back with. Its tests are therefore the alignment and persistence ones, plus
the call-site tagging in tests/protocol/test_eval_admission_gate.py.

`is_val` additionally makes a row structurally undrawable, and there are four
draw paths through CrystalBuffer plus three degenerate fallbacks inside
prioritised_weights that each return a uniform measure. A naive exclusion that
covers the prioritised path only leaks the entire held-out set into training
with no error and no visible symptom, so every path gets a test.
"""
import copy
import importlib.util
import io
import os
import sys

import numpy as np
import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling.buffer import (  # noqa: E402
    ORIGIN_BOOTSTRAP, ORIGIN_EVAL, ORIGIN_NAMES, ORIGIN_ROLLOUT,
    BufferColumnError, CrystalBuffer)

_spec = importlib.util.spec_from_file_location(
    'lj_carried_helpers', os.path.join(os.path.dirname(__file__), 'test_lj_coeff_carried.py'))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)

CPU = torch.device('cpu')
MIPCAS = 0.3635836825
TRAJ_LEN, TRAJ_DIM = 3, 12


def _built(n, seed=0):
    ef = _helpers.energy_fn(MIPCAS)
    b = _helpers.mol_batch(n)
    g = torch.Generator().manual_seed(seed)
    x = 0.4 * (2.0 * torch.rand(b.num_graphs, 12, generator=g) - 1.0)
    x[:, 3:6] = 0.0
    with torch.no_grad():
        _, built = ef.analyze_crystal_batch(x, b, torch.ones(b.num_graphs), return_batch=True)
    return built


def _traj(n, offset=0):
    """A trajectory tensor whose every entry names its own row."""
    return (torch.arange(offset, offset + n, dtype=torch.float32)[:, None, None]
            .expand(n, TRAJ_LEN, TRAJ_DIM).contiguous())


def _buffer(n, birth_log_pf=None, is_val=None, origin=None, with_traj=True, seed=0):
    return CrystalBuffer(_built(n, seed), device=CPU, y_fn='elj',
                         traj=_traj(n) if with_traj else None,
                         init_loss=torch.arange(1, n + 1, dtype=torch.float32),
                         birth_step=0,
                         birth_log_pf=birth_log_pf, is_val=is_val, origin=origin)


def _pickle_round_trip(state):
    bio = io.BytesIO()
    torch.save(state, bio)
    bio.seek(0)
    return torch.load(bio, map_location='cpu', weights_only=False)


# ------------------------------------------------------------------- defaults

def test_absent_columns_default_to_unknown_and_trainable():
    buf = _buffer(4)
    assert torch.isnan(buf.birth_log_pf).all(), 'NaN is "no policy scored this row"'
    assert not bool(buf.is_val.any())
    assert buf.birth_log_pf.dtype == torch.float32 and buf.is_val.dtype == torch.bool


def test_origin_defaults_to_rollout_as_int8():
    """The ordinary admission is the fused step's own forward rollout, so that
    is what an unstated origin means; eval/bootstrap say so at their call site."""
    buf = _buffer(4)
    assert buf.origin.dtype == torch.int8
    assert torch.equal(buf.origin, torch.full((4,), ORIGIN_ROLLOUT, dtype=torch.int8))


def test_the_origin_codes_are_distinct_and_all_named():
    """A code with no name emits no metric; two codes sharing one name merge two
    populations into one number. Both are silent."""
    codes = (ORIGIN_ROLLOUT, ORIGIN_EVAL, ORIGIN_BOOTSTRAP)
    assert len(set(codes)) == 3
    assert set(ORIGIN_NAMES) == set(codes)
    assert len(set(ORIGIN_NAMES.values())) == 3
    assert all(isinstance(v, str) and v for v in ORIGIN_NAMES.values())


# ----------------------------------------------------------------- round trip

def test_columns_round_trip_through_a_real_pickle():
    pf = torch.tensor([-1.5, float('nan'), 3.25, -8.0])
    val = torch.tensor([False, True, False, True])
    org = torch.tensor([ORIGIN_ROLLOUT, ORIGIN_EVAL, ORIGIN_BOOTSTRAP, ORIGIN_EVAL])
    buf = _buffer(4, birth_log_pf=pf, is_val=val, origin=org)
    back = CrystalBuffer.from_state_dict(_pickle_round_trip(buf.state_dict()), device=CPU)
    assert torch.equal(back.is_val, val)
    assert back.origin.dtype == torch.int8
    assert torch.equal(back.origin, org.to(torch.int8))
    ok = ~torch.isnan(pf)
    assert torch.allclose(back.birth_log_pf[ok], pf[ok])
    assert torch.isnan(back.birth_log_pf[~ok]).all(), 'a NaN row must stay NaN, not become 0'


@pytest.mark.parametrize('column', ['birth_log_pf', 'is_val', 'origin'])
def test_a_traj_carrying_store_without_a_column_is_refused(column):
    """The load-bearing refusal: a replay sidecar written before the column
    existed must fail at RESTORE, not fill a default and detonate at a draw."""
    state = _pickle_round_trip(_buffer(4).state_dict())
    state.pop(column)
    with pytest.raises(BufferColumnError, match=column):
        CrystalBuffer.from_state_dict(state, device=CPU)


@pytest.mark.parametrize('column', ['birth_log_pf', 'is_val', 'origin'])
def test_a_traj_free_store_without_the_columns_still_loads(column):
    """Prior/anchor/dataset stores never had a generating policy or a held-out
    split, so the defaults there are the true values -- refusing them would
    brick every warm start for no gain."""
    state = _pickle_round_trip(_buffer(4, with_traj=False).state_dict())
    state.pop(column)
    back = CrystalBuffer.from_state_dict(state, device=CPU)
    assert torch.isnan(back.birth_log_pf).all() and not bool(back.is_val.any())
    assert torch.equal(back.origin, torch.full((4,), ORIGIN_ROLLOUT, dtype=torch.int8))


# ------------------------------------------------------------------ alignment

def test_add_and_purge_keep_the_columns_aligned():
    buf = _buffer(4, birth_log_pf=torch.tensor([0.0, 1.0, 2.0, 3.0]),
                  is_val=torch.tensor([False, True, False, True]))
    buf.add(_built(2, seed=1), traj=_traj(2, offset=4),
            init_loss=torch.ones(2),
            birth_log_pf=torch.tensor([4.0, 5.0]),
            is_val=torch.tensor([True, False]))
    assert torch.allclose(buf.birth_log_pf, torch.arange(6, dtype=torch.float32))
    assert torch.equal(buf.is_val, torch.tensor([False, True, False, True, True, False]))

    buf.purge_by_index([0, 2, 5])
    # BY MARKER, not by count: a column carried but not reindexed passes a
    # length check and fails here
    assert torch.allclose(buf.birth_log_pf, torch.tensor([1.0, 3.0, 4.0]))
    assert torch.equal(buf.is_val, torch.tensor([True, True, True]))
    assert torch.allclose(buf.traj[:, 0, 0], torch.tensor([1.0, 3.0, 4.0]))


def test_add_defaults_the_columns_for_the_new_rows_only():
    buf = _buffer(3, birth_log_pf=torch.tensor([0.0, 1.0, 2.0]),
                  is_val=torch.tensor([True, True, True]),
                  origin=torch.full((3,), ORIGIN_EVAL))
    buf.add(_built(2, seed=1), traj=_traj(2, offset=3), init_loss=torch.ones(2))
    assert torch.isnan(buf.birth_log_pf[3:]).all()
    assert torch.equal(buf.is_val, torch.tensor([True, True, True, False, False]))
    assert torch.equal(buf.origin, torch.tensor(
        [ORIGIN_EVAL] * 3 + [ORIGIN_ROLLOUT] * 2, dtype=torch.int8))


def test_origin_survives_add_and_purge_by_marker():
    """The eviction the whole column exists to survive: a cohort admitted as
    eval rows must still read as eval rows after arbitrary rows around them are
    purged. A column carried but not reindexed passes every length check here."""
    buf = _buffer(4, origin=torch.tensor(
        [ORIGIN_ROLLOUT, ORIGIN_EVAL, ORIGIN_BOOTSTRAP, ORIGIN_ROLLOUT]))
    buf.add(_built(2, seed=1), traj=_traj(2, offset=4), init_loss=torch.ones(2),
            origin=ORIGIN_EVAL)     # a scalar seeds the whole admission batch
    assert torch.equal(buf.origin, torch.tensor(
        [ORIGIN_ROLLOUT, ORIGIN_EVAL, ORIGIN_BOOTSTRAP, ORIGIN_ROLLOUT,
         ORIGIN_EVAL, ORIGIN_EVAL], dtype=torch.int8))

    buf.purge_by_index([0, 3, 4])
    assert torch.equal(buf.origin, torch.tensor(
        [ORIGIN_EVAL, ORIGIN_BOOTSTRAP, ORIGIN_EVAL], dtype=torch.int8))
    # the trajectory names its own row, so this pins origin to the SAME rows
    assert torch.allclose(buf.traj[:, 0, 0], torch.tensor([1.0, 2.0, 5.0]))


# ----------------------------------------------------------------- the draw

VAL16 = torch.tensor([i % 4 == 0 for i in range(16)])       # 4 of 16 held out


def _val_buffer(**kw):
    return _buffer(16, is_val=VAL16.clone(), **kw)


def test_val_rows_are_never_drawn_by_the_uniform_path():
    buf = _val_buffer()
    held = set(np.flatnonzero(VAL16.numpy()).tolist())
    for _ in range(200):
        inds = buf._sample_indices(8)
        assert not (set(np.asarray(inds).tolist()) & held)


def test_the_uniform_draw_stays_without_replacement():
    """The reason the exclusion is a row POOL and not a substituted `p`: a
    measure would flip this draw to with-replacement (replace = True whenever
    p is not None) and silently duplicate rows."""
    buf = _val_buffer()
    for _ in range(50):
        inds = np.asarray(buf._sample_indices(8))
        assert len(set(inds.tolist())) == 8


def test_val_rows_are_never_drawn_by_the_prioritised_path():
    buf = _val_buffer()
    buf.ema_logw = torch.linspace(-5.0, 5.0, 16)
    p, w = buf.prioritised_weights(0.0, kappa=1.0, symmetric=True, exclude=buf.is_val)
    assert p[VAL16.numpy()].sum() == 0.0
    assert p.sum() == pytest.approx(1.0)
    held = set(np.flatnonzero(VAL16.numpy()).tolist())
    for _ in range(50):
        inds = buf._sample_indices(8, p=p, beta=0.25)
        assert not (set(np.asarray(inds).tolist()) & held)


def test_the_exclusion_changes_the_population_not_the_prioritisation():
    buf = _val_buffer()
    buf.ema_logw = torch.linspace(-5.0, 5.0, 16)
    p_ex, _ = buf.prioritised_weights(0.0, kappa=1.0, symmetric=True, exclude=buf.is_val)
    p_all, _ = buf.prioritised_weights(0.0, kappa=1.0, symmetric=True)
    keep = ~VAL16.numpy()
    assert np.allclose(p_ex[keep] / p_ex[keep].sum(), p_all[keep] / p_all[keep].sum())


@pytest.mark.parametrize('degenerate', ['all_nan_logw', 'no_eligible_row'])
def test_the_degenerate_fallbacks_still_exclude_val_rows(degenerate):
    """THE LOAD-BEARING EXCLUSION TEST. Each fallback returns a uniform
    measure; a uniform measure over ALL rows puts the whole held-out set back
    into training with no error and no symptom."""
    buf = _val_buffer()
    if degenerate == 'all_nan_logw':
        pass                                    # ema_logw is NaN from construction
    else:
        buf.ema_logw = torch.zeros(16)          # one-sided: delta == 0 everywhere
    p, _ = buf.prioritised_weights(
        0.0, kappa=1.0, symmetric=(degenerate == 'all_nan_logw'), exclude=buf.is_val)
    assert p[VAL16.numpy()].sum() == 0.0
    assert p.sum() == pytest.approx(1.0)


def test_a_blocked_draw_is_refused_while_rows_are_held_out():
    """_sample_condition_blocked_indices selects whole CONDITIONS and returns
    before any row mask is read, so it must refuse rather than leak."""
    buf = _val_buffer()
    with pytest.raises(ValueError, match='condition_block_m'):
        buf._sample_indices(8, condition_block_m=2)


def test_sample_val_graphs_returns_only_held_out_rows_and_does_not_bump_counts():
    buf = _val_buffer()
    graphs, traj, inds = buf.sample_val_graphs(3)
    assert graphs.num_graphs == 3 and len(set(inds.tolist())) == 3
    assert set(inds.tolist()) <= set(np.flatnonzero(VAL16.numpy()).tolist())
    # the stored trajectories, for exactly those rows
    assert torch.allclose(traj[:, 0, 0], torch.as_tensor(inds, dtype=torch.float32))
    # a draw count on a row that is never trained on would make it look drawn
    # to prioritised_weights' NaN policy and to every live-delta consumer
    assert int(buf.select_counts.sum()) == 0


def test_sample_val_graphs_is_empty_when_nothing_is_held_out():
    graphs, traj, inds = _buffer(4).sample_val_graphs(3)
    assert graphs is None and traj is None and inds.size == 0


# ---------------------------------------------------------- absorption sensor

def test_absorption_stats_ignores_held_out_rows():
    """Held-out rows have ema_loss == birth_loss forever (update_losses only
    touches drawn rows), so each contributes ratio 1.0 and pulls BOTH means --
    the live buffer servo's numerator and denominator -- toward the intake."""
    buf = _val_buffer()
    buf.ema_loss = buf.birth_loss * 0.4
    buf.ema_loss[VAL16] = buf.birth_loss[VAL16]
    st = buf.absorption_stats()
    keep = ~VAL16
    assert st['replay/ema_loss_mean'] == pytest.approx(
        float(buf.ema_loss[keep].double().mean()))
    assert st['replay/birth_loss_mean'] == pytest.approx(
        float(buf.birth_loss[keep].double().mean()))
    assert st['replay/resid_vs_intake'] == pytest.approx(0.4, abs=1e-6)
    assert st['replay/absorption_n'] == float(int(keep.sum()))
