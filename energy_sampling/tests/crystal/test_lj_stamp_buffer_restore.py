"""The buffer sidecar records its energy currency, and refuses rows without one.

WHY THIS FILE EXISTS. 2026-09-02, cluster arm p02_mipu_lr0p015625: leg 1 restored
the phase-1 exit's PRE-RELOCATION sidecar (rows with a raw `.elj` and no
`lj_coeff`), the launch-time eval wrote those rows back out as the arm's own
rolling sidecar before the stage transition rebuilt the prior buffer, and the
resubmission restored them into a prior-mode stage: first backward draw,
`AttributeError: crystal_batch carries no lj_coeff`. Reproduced locally on
ELJ against lp02 (frame-for-frame the same traceback) before the fix.

The stamp itself SURVIVES the pickle -- lp02's own sidecar carries it on all
three buffers, which is why lp02_resume passed. What was missing was any
statement, on disk or at restore, of whether rows are supposed to carry one.
Each test here fails against the pre-fix build:

  ROUND TRIP    state_dict -> pickle -> from_state_dict keeps the per-row stamp
                at a NON-UNIT coefficient and records the format version.
  REFUSED       a legacy dict (no version, no stamp) is refused at restore,
                never defaulted -- the default is the 2.62x silent error.
  PRODUCER BUG  a version-2 dict with unstamped rows is refused with a
                different diagnosis (do not migrate; fix the producer).
  COMPAT        a post-relocation, pre-version dict (stamped rows, no version
                key: the lp02 kind, and the running paper arms' kind) loads
                and is re-saved as version 2.
  MIGRATION     migrate_legacy_lj_coeff on a raw legacy dict stamps AND
                rescales, restoring the calibrated `.elj`, `y` and the
                composite energy exactly; it refuses a version-2 dict, a
                differently calibrated stamp, and a non-unit coefficient off
                the elj route; on uma it relabels without rescaling.
  PARITY        add() refuses to merge stamped rows into an unstamped store
                (and the reverse): append_batch(validate=False) would drop the
                key and blend two currencies in memory.
  VALUE CHECK   Checkpointer.assert_buffer_currency refuses a store stamped at
                a different coefficient than the run's.

Every test runs at mipcas's 0.3635836825, where a wrong stamp is numerically
visible; on UMA (coefficient 1) a relabel is a no-op and hides everything.
"""
import copy
import importlib.util
import io
import os
import sys
from types import SimpleNamespace

import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling.buffer import (  # noqa: E402
    AnchorBuffer, BUFFER_FORMAT_VERSION, BufferCurrencyError, CrystalBuffer,
    migrate_legacy_lj_coeff)
from energy_sampling.checkpointing import Checkpointer  # noqa: E402

# The synthetic-crystal builders live beside this file; imported by path so
# this test does not depend on pytest's import mode for sibling test modules.
_spec = importlib.util.spec_from_file_location(
    'lj_carried_helpers', os.path.join(os.path.dirname(__file__), 'test_lj_coeff_carried.py'))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)

CPU = torch.device('cpu')
MIPCAS = 0.3635836825


def built_batch(coeff=MIPCAS, seed=0):
    """A stamped, analyzed 2-crystal batch at `coeff`, plus its energy function."""
    ef = _helpers.energy_fn(coeff)
    b = _helpers.mol_batch()
    g = torch.Generator().manual_seed(seed)
    x = 0.4 * (2.0 * torch.rand(b.num_graphs, 12, generator=g) - 1.0)
    x[:, 3:6] = 0.0
    with torch.no_grad():
        _, built = ef.analyze_crystal_batch(x, b, torch.ones(b.num_graphs), return_batch=True)
    return ef, built


def buffer_at(coeff=MIPCAS, seed=0):
    ef, built = built_batch(coeff, seed)
    return ef, CrystalBuffer(built, device=CPU, y_fn='elj')


def pickle_round_trip(state):
    """A REAL pickle, not a dict copy: the claim is about what survives disk."""
    bio = io.BytesIO()
    torch.save(state, bio)
    bio.seek(0)
    return torch.load(bio, map_location='cpu', weights_only=False)


def as_legacy(state, coeff):
    """Rewrite a version-2 dict into the exact pre-relocation shape: no version,
    no stamp, RAW `.elj`, `y` mirroring it. This is what the pt100 phase-1 exit
    sidecars on the cluster look like."""
    legacy = dict(state)
    legacy.pop('format_version', None)
    legacy.pop('lj_coeff', None)
    batch = copy.copy(state['batch'])
    del batch.lj_coeff
    batch.elj = batch.elj / coeff
    legacy['batch'] = batch
    if legacy.get('y') is not None:
        legacy['y'] = legacy['y'] / coeff
    return legacy


# ------------------------------------------------------------------ ROUND TRIP

def test_round_trip_keeps_the_stamp_and_records_the_version():
    ef, buf = buffer_at(MIPCAS)
    state = buf.state_dict()
    assert state['format_version'] == BUFFER_FORMAT_VERSION
    assert state['lj_coeff'] == pytest.approx(MIPCAS)
    back = CrystalBuffer.from_state_dict(pickle_round_trip(state), device=CPU)
    assert back.batch.lj_coeff.shape == (len(buf),)
    assert torch.allclose(back.batch.lj_coeff, torch.full((len(buf),), MIPCAS))
    ef.assert_lj_coeff_stamped(back.batch)          # the draw-time check passes
    assert torch.allclose(back.y, buf.y) and torch.allclose(back.batch.elj, buf.batch.elj)


def test_anchor_buffer_round_trip_keeps_the_stamp():
    ef, built = built_batch(MIPCAS)
    n = built.num_graphs
    anchors = AnchorBuffer(built, device=CPU, reward=torch.zeros(n), energy=torch.zeros(n))
    state = anchors.state_dict()
    assert state['format_version'] == BUFFER_FORMAT_VERSION
    back = AnchorBuffer.from_state_dict(pickle_round_trip(state), device=CPU)
    ef.assert_lj_coeff_stamped(back.batch)


# -------------------------------------------------------------------- REFUSED

def test_legacy_unstamped_dict_is_refused_not_defaulted():
    ef, buf = buffer_at(MIPCAS)
    legacy = pickle_round_trip(as_legacy(buf.state_dict(), MIPCAS))
    with pytest.raises(BufferCurrencyError, match='migrate_buffer_sidecar'):
        CrystalBuffer.from_state_dict(legacy, device=CPU)


def test_legacy_unstamped_anchor_dict_is_refused():
    """Anchors too: they were the unstamped store in every post-transition
    rolling sidecar the prod_sep02 arms wrote."""
    ef, built = built_batch(MIPCAS)
    n = built.num_graphs
    anchors = AnchorBuffer(built, device=CPU, reward=torch.zeros(n), energy=torch.zeros(n))
    legacy = as_legacy(anchors.state_dict(), MIPCAS)
    with pytest.raises(BufferCurrencyError, match='carry no'):
        AnchorBuffer.from_state_dict(pickle_round_trip(legacy), device=CPU)


def test_versioned_dict_without_stamp_is_a_producer_bug_not_a_migration():
    ef, buf = buffer_at(MIPCAS)
    state = buf.state_dict()
    batch = copy.copy(state['batch'])
    del batch.lj_coeff
    state['batch'] = batch                            # version 2, no stamp
    with pytest.raises(BufferCurrencyError, match='NOT a legacy'):
        CrystalBuffer.from_state_dict(pickle_round_trip(state), device=CPU)


def test_mixed_stamps_in_one_dict_are_refused():
    ef, buf = buffer_at(MIPCAS)
    state = buf.state_dict()
    batch = copy.copy(state['batch'])
    batch.lj_coeff = torch.tensor([MIPCAS, 1.0])
    state['batch'] = batch
    with pytest.raises(BufferCurrencyError, match='more than one'):
        CrystalBuffer.from_state_dict(pickle_round_trip(state), device=CPU)


# --------------------------------------------------------------------- COMPAT

def test_stamped_but_unversioned_dict_loads_and_resaves_as_version_2():
    """The lp02 kind: written after the relocation, before the version key."""
    ef, buf = buffer_at(MIPCAS)
    state = buf.state_dict()
    state.pop('format_version')
    state.pop('lj_coeff')
    back = CrystalBuffer.from_state_dict(pickle_round_trip(state), device=CPU)
    ef.assert_lj_coeff_stamped(back.batch)
    resaved = back.state_dict()
    assert resaved['format_version'] == BUFFER_FORMAT_VERSION
    assert resaved['lj_coeff'] == pytest.approx(MIPCAS)


# ------------------------------------------------------------------ MIGRATION

def test_migration_restores_the_calibrated_currency_exactly():
    ef, buf = buffer_at(MIPCAS)
    state = buf.state_dict()
    legacy = pickle_round_trip(as_legacy(state, MIPCAS))
    # sanity: the forged legacy really is raw and unstamped
    assert not hasattr(legacy['batch'], 'lj_coeff')
    assert not torch.allclose(legacy['batch'].elj, state['batch'].elj)

    migrated = migrate_legacy_lj_coeff(legacy, MIPCAS, 'elj')
    assert migrated['format_version'] == BUFFER_FORMAT_VERSION
    assert migrated['lj_coeff'] == pytest.approx(MIPCAS)
    assert not hasattr(legacy['batch'], 'lj_coeff'), 'input must not be mutated'
    back = CrystalBuffer.from_state_dict(pickle_round_trip(migrated), device=CPU)
    ef.assert_lj_coeff_stamped(back.batch)
    assert torch.allclose(back.batch.elj, state['batch'].elj, rtol=1e-6)
    assert torch.allclose(back.y, state['y'], rtol=1e-6)
    # the composite energy every draw scores is byte-identical to the original's
    T = torch.ones(len(buf))
    e_orig, _ = ef.generator_energy(buf.batch, T)
    e_back, _ = ef.generator_energy(back.batch, T)
    assert torch.allclose(e_back, e_orig, rtol=1e-6)


def test_migration_stamping_without_rescale_would_be_wrong_and_is_not_what_happens():
    """The trap the refusal exists for, made explicit: a raw row stamped as
    calibrated scores 1/lj_coeff too large on its elj term."""
    ef, buf = buffer_at(MIPCAS)
    legacy = as_legacy(buf.state_dict(), MIPCAS)
    wrong = copy.copy(legacy['batch'])
    wrong.lj_coeff = torch.full((len(buf),), MIPCAS)  # stamp only, no rescale
    T = torch.ones(len(buf))
    e_orig, ens_orig = ef.generator_energy(buf.batch, T)
    e_wrong, ens_wrong = ef.generator_energy(wrong, T)
    ratio = ens_wrong['mol_energy'] / ens_orig['mol_energy']
    assert torch.allclose(ratio, torch.full_like(ratio, 1.0 / MIPCAS), rtol=1e-5), (
        'a stamp-only "migration" must inflate mol_energy by exactly 1/lj_coeff -- '
        'this is the silent 2.75x the refusal guards against')


def test_migration_refuses_a_versioned_dict():
    ef, buf = buffer_at(MIPCAS)
    with pytest.raises(BufferCurrencyError, match='already format version'):
        migrate_legacy_lj_coeff(buf.state_dict(), MIPCAS, 'elj')


def test_migration_verifies_an_existing_stamp_and_only_marks():
    ef, buf = buffer_at(MIPCAS)
    state = buf.state_dict()
    state.pop('format_version')
    state.pop('lj_coeff')
    marked = migrate_legacy_lj_coeff(state, MIPCAS, 'elj')
    assert marked['format_version'] == BUFFER_FORMAT_VERSION
    assert torch.equal(marked['batch'].elj, state['batch'].elj), 'stamped rows are never rescaled'
    with pytest.raises(BufferCurrencyError, match='differently calibrated'):
        migrate_legacy_lj_coeff(state, 1.0, 'elj')


def test_migration_off_the_elj_route_relabels_at_one_and_refuses_other_values():
    ef, buf = buffer_at(1.0)
    legacy = as_legacy(buf.state_dict(), 1.0)
    with pytest.raises(ValueError, match='non-unit'):
        migrate_legacy_lj_coeff(legacy, MIPCAS, 'uma')
    relabelled = migrate_legacy_lj_coeff(legacy, 1.0, 'uma')
    assert torch.all(relabelled['batch'].lj_coeff == 1.0)
    assert torch.equal(relabelled['batch'].elj, legacy['batch'].elj)
    assert torch.equal(relabelled['y'], legacy['y'])


def test_sidecar_level_migration_walks_every_buffer():
    from migrate_buffer_sidecar import migrate_sidecar_state
    ef, buf = buffer_at(MIPCAS)
    ef2, built = built_batch(MIPCAS, seed=1)
    n = built.num_graphs
    anchors = AnchorBuffer(built, device=CPU, reward=torch.zeros(n), energy=torch.zeros(n))
    sidecar = {
        'problem_def': {'energy_function': 'elj', 'prior_path': 'does/not/exist.pt'},
        'step_ind': 5010,
        'prior_buffer': as_legacy(buf.state_dict(), MIPCAS),
        'replay_buffer': None,
        'anchor_buffer': as_legacy(anchors.state_dict(), MIPCAS),
    }
    with pytest.raises(ValueError, match='cannot derive lj_coeff'):
        migrate_sidecar_state(pickle_round_trip(sidecar))     # no prior to read from
    out, report = migrate_sidecar_state(pickle_round_trip(sidecar), lj_coeff=MIPCAS)
    assert report['replay_buffer']['action'] == 'absent'
    assert report['prior_buffer']['action'].startswith('stamped + rescaled')
    assert report['anchor_buffer']['action'].startswith('stamped + rescaled')
    CrystalBuffer.from_state_dict(out['prior_buffer'], device=CPU)
    back = AnchorBuffer.from_state_dict(out['anchor_buffer'], device=CPU)
    ef.assert_lj_coeff_stamped(back.batch)
    assert torch.allclose(back.batch.elj, built.elj, rtol=1e-6)
    assert torch.equal(back.energy, anchors.energy), 'composite anchor energy is never rescaled'
    # a second pass finds nothing to do
    out2, report2 = migrate_sidecar_state(out, lj_coeff=MIPCAS)
    assert all(report2[k]['action'].startswith(('skipped', 'absent')) for k in report2)


# --------------------------------------------------------------------- PARITY

def test_add_refuses_stamped_rows_into_an_unstamped_store():
    ef, unstamped_built = built_batch(MIPCAS, seed=2)
    del unstamped_built.lj_coeff
    store = CrystalBuffer(unstamped_built, device=CPU, y_fn='elj')   # constructor does not judge
    _, incoming = built_batch(MIPCAS, seed=3)
    with pytest.raises(BufferCurrencyError, match='UNSTAMPED'):
        store.add(incoming)


def test_add_refuses_unstamped_rows_into_a_stamped_store():
    ef, store = buffer_at(MIPCAS)
    _, incoming = built_batch(MIPCAS, seed=3)
    del incoming.lj_coeff
    with pytest.raises(BufferCurrencyError, match='UNSTAMPED'):
        store.add(incoming)


def test_add_of_matching_stamps_still_works():
    ef, store = buffer_at(MIPCAS)
    _, incoming = built_batch(MIPCAS, seed=3)
    n0 = len(store)
    store.add(incoming)
    assert len(store) == n0 + incoming.num_graphs
    ef.assert_lj_coeff_stamped(store.batch)


# ---------------------------------------------------------------- VALUE CHECK

def _checkpointer_for(ef, **buffers):
    m = SimpleNamespace(energy_function=ef, args=SimpleNamespace(), **buffers)
    return Checkpointer(m)


def test_checkpointer_refuses_a_store_in_another_runs_currency():
    ef_run, _ = built_batch(MIPCAS)
    _, store_at_one = buffer_at(1.0)          # e.g. an anchor set built at 1.0
    ck = _checkpointer_for(ef_run, prior_buffer=store_at_one)
    with pytest.raises(BufferCurrencyError, match='two energy currencies'):
        ck.assert_buffer_currency('test')


def test_checkpointer_accepts_matching_currency_and_skips_latent_routes():
    ef_run, store = buffer_at(MIPCAS)
    _checkpointer_for(ef_run, prior_buffer=store).assert_buffer_currency('test')
    latent_ef = SimpleNamespace(is_crystal=True, latent_energy=True, lj_coeff=1.0,
                                assert_lj_coeff_stamped=lambda b: (_ for _ in ()).throw(
                                    AssertionError('must not be called')))
    _checkpointer_for(latent_ef, prior_buffer=store).assert_buffer_currency('test')
