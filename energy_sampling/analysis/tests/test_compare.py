"""
Tests for Tier 2 -- `analysis/compare.py`.

MUTATION-TESTED THROUGHOUT. Every behaviour gets two tests: the real, unmutated
runs do NOT produce the state, and a real run with the condition re-introduced
DOES. A check that has never fired has not been tested, and this repo has
shipped tests that passed while blind more than once.

Mutations are edits to REAL captured runs (`fixtures.mutate` deep-copies, so the
module-scoped fixtures survive), never hand-built configs. A hand-built config
agrees with whatever the author assumed, which is the failure the package exists
to stop.

Real-run facts this file pins, because they are evidence and not accidents:

  * `ring_probe` and `ring_cal` are ONE ARM WRITTEN TWICE. Their configs differ
    in `run_name` and in nothing else, so the sweep table must say "no knob
    differs" rather than invent a dimension. Verified 2026-08-16 on the captured
    pair and on the `wandb/` originals.
  * `vg_normal` / `vg_blowup` are a real conditional VarGrad pair: ten differing
    flattened knobs, and two repr-string sections (`adaptive_lr`, `protocol`)
    that differ ONLY because those flattened knobs do.
  * `mle_only` (MLE/prior, T=10, resumed from a named checkpoint) beside
    `vg_normal` (conditional VarGrad, T=40, `checkpoint_name: None`) is a
    genuinely incomparable pair -- different route, different stage, different
    T, different start.
  * `tb_ramp` logs `fwd/tb_err_worst`; `vg_normal` logs it too and it does NOT
    MEAN THE SAME THING there. That pair is the NA_ROUTE case, on real data.

FIRING RATES, measured over the 300 same-tag arm PAIRS in the local corpus
(`wandb/`, 85 configured runs) rather than over the fixtures, because a state
that fires on most of a corpus is not a state worth reporting:

    cells, per-route toplines   LIVE 80.5%  ABSENT 19.5%
    cells, TB topline forced
      across both arms          LIVE 78.9%  ABSENT 17.6%  NO_SERIES 1.9%
                                NA_ROUTE 1.7%
    pairs with no knob differing              3.3%
    pairs split by route                     29.7%
    pairs with a cross-arm §4 blocker        46.0%
    pairs with a BLOB_ONLY knob               0.0%   <- mutation-only, by design

Run: python -m pytest analysis/tests -q
"""

import io
import tokenize

import numpy as np
import pytest

from analysis import compare as P
from analysis import checks as C
from analysis import keys as K
from analysis.tests import fixtures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(sweep, key):
    """The single sweep row for a knob, or a loud failure. Deliberately not a
    filter that shrugs on an empty list -- that turns every later assertion into
    a vacuous pass."""
    hits = [r for r in sweep.rows if r.key == key]
    assert len(hits) == 1, (
        f'{key!r}: expected one sweep row, got {len(hits)}. '
        f'rows={[r.key for r in sweep.rows]}')
    return hits[0]


def _keys(sweep):
    return [r.key for r in sweep.rows]


def _cell(table, metric, arm_label):
    for block in table.blocks:
        for row in block.rows:
            if row.metric == metric and arm_label in row.cells:
                return row.cells[arm_label]
    raise AssertionError(
        f'no cell for {metric!r} on {arm_label!r}; blocks='
        f'{[(b.label, b.metrics) for b in table.blocks]}')


def _label(run):
    return C.run_label(run)


def _strip(run, metric, cap=8):
    """Drop every key `metric` can resolve to, until it is genuinely ABSENT.

    `K.resolve` substitutes a neighbouring name (`fwd/tb_err_worst` ->
    `fwd/tb_err`), so removing one key produces a RENAME, not a hole. Returns
    `(mutated run, dropped keys)`."""
    ctx = C.context(run)
    out, dropped = run, []
    for _ in range(cap):
        res, = K.resolve(out.available_keys(), [metric], ctx.route)
        if res.key is None:
            break
        dropped.append(res.key)
        out = fixtures.mutate(out, drop=[res.key])
    else:
        raise AssertionError(f'{metric} still resolves after {cap} drops')
    return out, dropped


def _first_live_topline(run):
    """A topline metric this run actually has a usable series for."""
    ctx = C.context(run)
    for m in K.TOPLINE[ctx.route]:
        res, = K.resolve(run.available_keys(), [m], ctx.route)
        if res.state is K.KeyState.LIVE and res.key in run.history:
            s, _ = run.history[res.key]
            if len(s) >= 8:
                return m, res.key
    raise AssertionError(f'{run.name}: no topline metric with a usable series')


# ===========================================================================
# The comparability gate
# ===========================================================================
# §4 is not reimplemented here -- `checks.check_confounds` owns it. What is
# tested is that its cross-arm findings reach the caller ATTACHED TO THE
# NUMBERS, and that no code path renders the metric table without them.

def test_the_incomparable_pair_blocks(mle_only, vg_normal):
    """Different route, different stage, different T, different start. This is
    the pair §4 exists for, and every one of those subjects must arrive."""
    cmp = P.compare([mle_only, vg_normal])
    assert not cmp.comparable
    subjects = {f.subject for f in cmp.blockers}
    assert 'battery/route' in subjects
    assert 'battery/stage' in subjects
    assert f'battery/start/{K.CFG_TRAIN_T}' in subjects
    assert f'battery/start/{K.CFG_EVAL_T}' in subjects
    assert 'battery/checkpoint_name' in subjects


def test_a_real_pair_with_no_cross_arm_finding_does_not_block(vg_normal,
                                                              vg_blowup):
    """THE COMPANION. Without a pair that produces zero blockers, `blockers`
    could be returning every §4 row and every test above would still pass."""
    cmp = P.compare([vg_normal, vg_blowup])
    assert cmp.comparable, [f.subject for f in cmp.blockers]
    assert cmp.blockers == ()


def test_introducing_one_confound_produces_one_blocker(vg_normal, vg_blowup):
    """MUTATION. The unmutated pair is comparable (above); change the training
    integrator on one arm and exactly that subject appears."""
    mutated = fixtures.mutate(vg_blowup, config={K.CFG_TRAIN_T: 999})
    cmp = P.compare([vg_normal, mutated])
    subjects = [f.subject for f in cmp.blockers]
    assert f'battery/start/{K.CFG_TRAIN_T}' in subjects
    assert not cmp.comparable


def test_per_run_findings_are_not_blockers(vg_normal, vg_blowup):
    """MUTATION. A confound of ONE arm qualifies that arm; it does not void the
    comparison. Merging the two would make the gate fire on facts about a single
    run, and then it fires on nearly everything and gets switched off.

    `eval_T != integrator_T` produces BOTH kinds at once -- a per-run `{arm}/T`
    row and a cross-arm `battery/start/eval_T` row -- so one mutation shows the
    partition rather than only one side of it."""
    mutated = fixtures.mutate(vg_blowup, config={K.CFG_EVAL_T: 999})
    cmp = P.compare([vg_normal, mutated])
    per_run = [f for f in cmp.confounds.findings
               if f.subject.startswith(f'{_label(mutated)}/')]
    assert per_run, [f.subject for f in cmp.confounds.findings]
    blocked = {f.subject for f in cmp.blockers}
    assert not {f.subject for f in per_run} & blocked
    assert all(f.subject.startswith('battery/') for f in cmp.blockers)
    assert f'battery/start/{K.CFG_EVAL_T}' in {f.subject for f in cmp.blockers}


def test_the_gate_is_rendered_before_a_single_number(mle_only, vg_normal):
    """`format_feature_table` emits the banner itself, so there is no code path
    that produces a bare metric table."""
    cmp = P.compare([mle_only, vg_normal])
    text = P.format_feature_table(cmp.features)
    gate = text.index('COMPARABILITY')
    first_metric = min(text.index(m) for m in K.TOPLINE[C.context(mle_only).route]
                       if m in text)
    assert gate < first_metric
    assert 'battery/route' in text


def test_the_gate_also_renders_when_it_is_clean(vg_normal, vg_blowup):
    """A clean gate is not silence. Omitting the line when nothing fired makes
    'checked, nothing across the arms' indistinguishable from 'never looked'."""
    cmp = P.compare([vg_normal, vg_blowup])
    assert 'COMPARABILITY' in P.format_feature_table(cmp.features)


def test_the_blockers_travel_on_the_table_itself(mle_only, vg_normal):
    """The enforcement: a caller holding the numbers is holding the gate."""
    table = P.compare([mle_only, vg_normal]).features
    assert table.blockers
    assert not table.comparable


def test_every_flat_record_carries_the_gate(mle_only, vg_normal):
    """A flattened row is the form most likely to be pulled out of context, so
    the gate is inside the row, not in a header a consumer can drop."""
    cmp = P.compare([mle_only, vg_normal])
    recs = cmp.records()
    assert recs
    assert all(r['comparable'] is False for r in recs)
    assert all(r['n_blockers'] == len(cmp.blockers) for r in recs)


def test_a_comparable_pairs_records_say_so(vg_normal, vg_blowup):
    """COMPANION to the above -- otherwise `comparable` could be hardcoded."""
    recs = P.compare([vg_normal, vg_blowup]).records()
    assert recs and all(r['comparable'] is True for r in recs)


# ===========================================================================
# The sweep table
# ===========================================================================

def test_one_arm_written_twice_reports_no_knob(ring_probe, ring_cal):
    """The real case: `ring_probe` and `ring_cal` differ in `run_name` and in
    nothing else. The table must say so rather than invent a dimension."""
    sweep = P.compare([ring_probe, ring_cal]).sweep
    assert sweep.rows == (), _keys(sweep)
    assert not sweep.differs
    assert sweep.n_knobs_compared > 100, 'expected a real config, not an empty one'
    assert 'NO KNOB DIFFERS' in P.format_sweep(sweep)


def test_a_real_multi_arm_sweep_finds_its_dimensions(vg_normal, vg_blowup):
    """COMPANION. Without an arm pair that DOES differ, 'no knob differs' could
    be the only answer the table ever gives."""
    sweep = P.compare([vg_normal, vg_blowup]).sweep
    assert sweep.differs
    assert len(sweep.rows) >= 4, _keys(sweep)


def test_changing_one_value_adds_exactly_one_knob(ring_probe, ring_cal):
    """MUTATION on the pair that differs in nothing."""
    mutated = fixtures.mutate(ring_cal, config={K.CFG_SEED: 777})
    sweep = P.compare([ring_probe, mutated]).sweep
    assert _keys(sweep) == [K.CFG_SEED]
    row = _row(sweep, K.CFG_SEED)
    assert row.kind is P.KnobKind.VALUE
    assert row.values[_label(ring_probe)] != row.values[_label(mutated)]
    assert row.n_distinct == 2


def test_a_knob_that_differs_by_presence_counts(ring_probe, ring_cal):
    """MUTATION. An absent key takes its default, and a default that differs
    from a sibling's explicit value is a swept dimension whether or not anyone
    meant to sweep it -- §4's 'arms that differ by omission', read from the
    other side."""
    mutated = fixtures.mutate(ring_cal)
    del mutated.config[K.CFG_SEED]
    sweep = P.compare([ring_probe, mutated]).sweep
    row = _row(sweep, K.CFG_SEED)
    assert row.kind is P.KnobKind.PRESENCE
    assert row.present[_label(ring_probe)] is True
    assert row.present[_label(mutated)] is False
    assert row.values[_label(mutated)] == C._CONF_MISSING


def test_a_null_value_is_not_a_missing_key(ring_probe, ring_cal):
    """`<null>` and `<missing>` are different findings and render differently:
    the first is a knob explicitly set to nothing, the second is a config from
    another tree."""
    mutated = fixtures.mutate(ring_cal, config={K.CFG_SEED: None})
    row = _row(P.compare([ring_probe, mutated]).sweep, K.CFG_SEED)
    assert row.values[_label(mutated)] == C._CONF_NULL
    assert row.values[_label(mutated)] != C._CONF_MISSING


def test_identity_keys_are_never_swept(ring_probe, ring_cal):
    """MUTATION on every identity key at once. Every arm differs in its name;
    listing that is listing the index the sweep is keyed BY."""
    edits = {k: f'changed_{k}' for k in K.CFG_IDENTITY if k != K.CFG_WANDB_BLOB}
    mutated = fixtures.mutate(ring_cal, config=edits)
    sweep = P.compare([ring_probe, mutated]).sweep
    assert sweep.rows == (), _keys(sweep)


def test_repr_blobs_are_collapsed_to_their_flattened_children(vg_normal,
                                                              vg_blowup):
    """wandb stores each config section twice -- flattened scalars AND the
    section's `repr()`. Kept as knobs, the reprs report every swept dimension a
    second time as a few thousand unreadable characters."""
    sweep = P.compare([vg_normal, vg_blowup]).sweep
    blob_keys = {b[0] for b in sweep.blobs}
    assert blob_keys, 'expected repr-string sections on this real pair'
    assert not (blob_keys & set(_keys(sweep))), 'a blob leaked into the knob rows'
    for key, children in sweep.blobs:
        assert children, f'{key} collapsed with nothing to collapse onto'
        assert all(c.startswith(key + '_') for c in children)
        assert all(c in _keys(sweep) for c in children)


def test_a_blob_differing_with_no_flattened_child_is_its_own_row(ring_probe,
                                                                 ring_cal):
    """MUTATION, and the one that matters: a section whose repr differs while
    every flattened child agrees. The difference is real and invisible in every
    scalar key, so silence here would DELETE a swept dimension from the table.

    Measured on 300 real arm pairs this never happens, which is exactly why it
    needs a mutation test -- it is a safety net, and an untriggered safety net
    is indistinguishable from a missing one.
    """
    blob = sorted(P._blob_keys(ring_cal.config))[0]
    original = K._value(ring_cal.config, blob)
    mutated = fixtures.mutate(ring_cal, config={blob: original + ' # edited'})
    sweep = P.compare([ring_probe, mutated]).sweep
    row = _row(sweep, blob)
    assert row.kind is P.KnobKind.BLOB_ONLY
    assert row.note and 'repr' in row.note
    assert blob not in {b[0] for b in sweep.blobs}
    assert '!' in P.format_sweep(sweep)


def test_the_unmutated_pair_produces_no_blob_only_row(vg_normal, vg_blowup):
    """COMPANION. BLOB_ONLY must not be what a normal battery gets."""
    sweep = P.compare([vg_normal, vg_blowup]).sweep
    assert all(r.kind is not P.KnobKind.BLOB_ONLY for r in sweep.rows)


def test_a_plain_string_knob_with_a_child_key_is_not_a_blob(ring_probe,
                                                            ring_cal):
    """THE FALSE POSITIVE THAT WAS REAL. `z_calibration_sensor` holds `pooled`
    or `rms` and has an unrelated `z_calibration_sensor_quantile` beside it. On
    a sibling test alone it was classified as a repr blob, and its row then
    claimed the difference was 'inside the repr string' -- which was false about
    a knob that is genuinely swept. Detection requires the value to LOOK like a
    repr as well."""
    sensor = [k for k in ring_cal.config
              if k.endswith('_sensor') and isinstance(K._value(ring_cal.config, k), str)]
    assert sensor, 'expected a plain string knob with a child key in this config'
    key = sensor[0]
    assert any(o.startswith(key + '_') for o in ring_cal.config), \
        f'{key} has no child key, so it does not exercise the false positive'
    mutated = fixtures.mutate(ring_cal, config={key: 'something_else'})
    sweep = P.compare([ring_probe, mutated]).sweep
    row = _row(sweep, key)
    assert row.kind is P.KnobKind.VALUE
    assert key not in {b[0] for b in sweep.blobs}


def test_a_capped_render_says_how_much_it_hid(mle_only, vg_normal):
    """The structured table is always complete; only the render is trimmed, and
    a truncation that does not state its own size is the report deciding what
    the reader may see."""
    sweep = P.compare([mle_only, vg_normal]).sweep
    assert len(sweep.rows) > 5, len(sweep.rows)
    text = P.format_sweep(sweep, limit=5)
    assert f'{len(sweep.rows) - 5} further differing knob(s)' in text
    assert f'{len(sweep.rows)} knob(s) differ' in text


def test_a_capped_render_never_eats_a_blob_only_row(ring_probe, ring_cal):
    """BLOB_ONLY is the row whose swept value is readable NOWHERE else, so it is
    the one row a cap must not drop."""
    blob = sorted(P._blob_keys(ring_cal.config))[0]
    original = K._value(ring_cal.config, blob)
    mutated = fixtures.mutate(ring_cal, config={blob: original + ' # edited'})
    for i in range(30):        # bury it under ordinary differing knobs
        mutated.config[f'zzz_filler_{i}'] = {'value': i}
    sweep = P.compare([ring_probe, mutated]).sweep
    assert len(sweep.rows) > 5
    assert blob in P.format_sweep(sweep, limit=1)


def test_sweep_records_are_one_row_per_knob_and_arm(vg_normal, vg_blowup):
    cmp = P.compare([vg_normal, vg_blowup])
    recs = cmp.sweep_records()
    assert len(recs) == len(cmp.sweep.rows) * 2
    assert {r['arm'] for r in recs} == {a.label for a in cmp.arms}


def test_the_sweep_renderer_does_not_elide_a_knob_into_its_neighbour(
        ring_probe, ring_cal):
    """A ROW LABEL IS THE ROW'S IDENTITY. Measured on the real four-arm batt0807
    battery, a fixed 34-column cap rendered
    `buffers_replay_buffer_prioritise_enabled` and
    `buffers_replay_buffer_prioritise_kappa` as the SAME string -- two rows that
    could not be told apart, in the table whose whole job is telling them apart.

    Those two keys are the mutation, verbatim, because they are the pair that
    actually collided."""
    a = 'buffers_replay_buffer_prioritise_enabled'
    b = 'buffers_replay_buffer_prioritise_kappa'
    mutated = fixtures.mutate(ring_cal, config={a: True, b: 1.0})
    text = P.format_sweep(P.compare([ring_probe, mutated]).sweep)
    assert a in text, text
    assert b in text, text
    body = [l for l in text.splitlines() if a in l or b in l]
    assert len(body) == 2 and body[0] != body[1]


def test_a_knob_name_too_long_for_the_column_keeps_its_tail(ring_probe,
                                                            ring_cal):
    """Elide the MIDDLE, never the tail. These names are prefix-heavy
    (`protocol_stages_1_lr_sensor_*`), so cutting the tail deletes the only
    distinguishing part."""
    stem = 'protocol_stages_9_lr_sensor_' + 'x' * P._NAME_COL_MAX
    a, b = stem + '_alpha', stem + '_omega'
    mutated = fixtures.mutate(ring_cal, config={a: 1, b: 2})
    text = P.format_sweep(P.compare([ring_probe, mutated]).sweep)
    body = [l for l in text.splitlines() if l.strip().startswith(stem[:20])]
    assert len(body) == 2, body
    assert body[0] != body[1]
    assert 'alpha' in text and 'omega' in text


# ===========================================================================
# The aligned feature table -- three states, never a blank, never a zero
# ===========================================================================

def test_na_route_and_live_share_a_row_on_real_data(tb_ramp, vg_normal):
    """THE CENTRAL CASE. `fwd/tb_err_worst` is logged and populated on BOTH
    arms. On the TB arm it is a reading; on the conditional VarGrad arm it is
    not, and the two must not line up as numbers."""
    metric = K.TOPLINE_TB[0]
    cmp = P.compare([tb_ramp, vg_normal], metrics=[metric], window=6000)
    tb = _cell(cmp.features, metric, _label(tb_ramp))
    vg = _cell(cmp.features, metric, _label(vg_normal))
    assert tb.state is P.CellState.LIVE
    assert vg.state is P.CellState.NA_ROUTE
    assert metric in vg_normal.summary or metric in vg_normal.history, \
        'the point of NA_ROUTE is that the key IS there'


def test_na_route_never_renders_as_zero_or_blank(tb_ramp, vg_normal):
    metric = K.TOPLINE_TB[0]
    cmp = P.compare([tb_ramp, vg_normal], metrics=[metric], window=6000)
    vg = _cell(cmp.features, metric, _label(vg_normal))
    assert vg.value('last') is None
    assert vg.value('slope/1k') is None
    text = P.format_feature_table(cmp.features)
    assert 'NA_ROUTE' in text
    cell_text = P._cell_text(vg, 'last')
    assert cell_text.strip() == 'NA_ROUTE'
    assert cell_text.strip() not in ('', '0', '0.0')


def test_na_route_is_driven_by_the_route_not_by_the_key(tb_ramp, vg_normal):
    """MUTATION COMPANION, and the one that proves the marking is real. Same
    keys, same numbers; take the conditioning flag away so the run no longer
    classifies as conditional VarGrad and the withholding must STOP."""
    metric = K.TOPLINE_TB[0]
    before = _cell(P.compare([tb_ramp, vg_normal], metrics=[metric],
                             window=6000).features, metric, _label(vg_normal))
    assert before.state is P.CellState.NA_ROUTE

    unconditional = fixtures.mutate(vg_normal, config={'vector_conditioning': False,
                                                       'molecule_conditioning': False})
    assert C.context(unconditional).route is not K.Route.VARGRAD_CONDITIONAL
    after = _cell(P.compare([tb_ramp, unconditional], metrics=[metric],
                            window=6000).features, metric, _label(unconditional))
    assert after.state is P.CellState.LIVE
    assert after.value('last') is not None


def test_absent_and_na_route_are_different_words(tb_ramp, vg_normal):
    """Both are blank cells in a naive table, and they mean opposite things: one
    sends the reader after a logging bug, the other after the route's
    semantics."""
    cmp = P.compare([tb_ramp, vg_normal], metrics=K.TOPLINE_TB, window=6000)
    states = {c.state for b in cmp.features.blocks for r in b.rows
              for c in r.cells.values()}
    assert P.CellState.NA_ROUTE in states
    assert P.CellState.ABSENT in states
    assert (P._CELL_TOKEN[P.CellState.NA_ROUTE]
            != P._CELL_TOKEN[P.CellState.ABSENT])


def test_dropping_the_series_and_its_fuzzy_sibling_turns_a_cell_absent(tb_ramp):
    """MUTATION, with its companion in the same test: the same metric is LIVE
    before the drop and ABSENT after.

    THE FIRST DROP IS NOT ENOUGH, and finding that out is the point. Removing
    `fwd/tb_err_worst` leaves `fwd/tb_err`, which `K.resolve`'s suffix-family
    rule substitutes -- so a test that dropped one key and asserted ABSENT was
    asserting something that is not true of this data. Every candidate has to
    go before the cell is genuinely a hole."""
    metric, key = _first_live_topline(tb_ramp)
    before = _cell(P.compare([tb_ramp], window=6000).features, metric,
                   _label(tb_ramp))
    assert before.state is P.CellState.LIVE

    stripped, dropped = _strip(tb_ramp, metric)
    assert dropped, 'nothing was dropped, so nothing was mutated'
    after = _cell(P.compare([stripped], window=6000).features, metric,
                  _label(stripped))
    assert after.state is P.CellState.ABSENT
    assert after.note, 'an ABSENT cell without a reason is a blank with a name'
    assert after.value('last') is None


def test_dropping_only_the_primary_name_is_a_rename_not_a_hole(tb_ramp):
    """COMPANION to the above, and the reason it needs three drops. A metric
    logged under a neighbouring name is a RENAME -- the substitution is made,
    and it is made visible in the cell."""
    metric, key = _first_live_topline(tb_ramp)
    stripped = fixtures.mutate(tb_ramp, drop=[key])
    cell = _cell(P.compare([stripped], window=6000).features, metric,
                 _label(stripped))
    if cell.state is P.CellState.ABSENT:
        pytest.skip(f'{metric} has no fuzzy sibling on this run')
    assert cell.state is P.CellState.LIVE
    assert cell.resolved_to and cell.resolved_to != key


def test_a_short_series_is_thin_and_not_absent(tb_ramp):
    """MUTATION. The series IS there; the window does not hold enough of it for
    a trend. Reporting that as a hole sends the reader after a logging bug that
    is not there."""
    metric, key = _first_live_topline(tb_ramp)
    s, v = tb_ramp.history[key]
    short = fixtures.mutate(tb_ramp, history={key: (s[:2], v[:2])})
    cell = _cell(P.compare([short], window=6000).features, metric, _label(short))
    assert cell.state is P.CellState.THIN
    assert cell.value('last') is None
    assert P._CELL_TOKEN[P.CellState.THIN] not in (
        P._CELL_TOKEN[P.CellState.ABSENT],
        P._CELL_TOKEN[P.CellState.NA_ROUTE])


def test_a_summary_only_metric_is_thin_on_a_real_run(ring_probe):
    """NOT A MUTATION -- this is what the config-only captures actually look
    like, and it is what a run reads like before its first eval. The key is in
    the record with a value and there is no series behind it."""
    cmp = P.compare([ring_probe])
    states = {r.metric: c.state for b in cmp.features.blocks for r in b.rows
              for c in r.cells.values()}
    assert P.CellState.THIN in states.values(), states


def test_a_non_numeric_summary_entry_is_no_series_not_absent(tb_ramp):
    """MUTATION. The key resolves LIVE -- it is in `available_keys` -- and holds
    nothing a feature can be computed from. Calling that ABSENT would be a
    statement about the run record that is false."""
    metric, key = _first_live_topline(tb_ramp)
    mutated = fixtures.mutate(tb_ramp, drop=[key])
    mutated.summary[key] = 'not a number'
    cell = _cell(P.compare([mutated], window=6000).features, metric,
                 _label(mutated))
    assert cell.state is P.CellState.NO_SERIES
    assert cell.value('last') is None


def test_a_renamed_key_is_reported_as_a_rename(tb_ramp):
    """`bwd/under_coverage_wcen` is the doc's name and the runs log
    `bwd/under_coverage`. The substitution is made and made VISIBLE."""
    cmp = P.compare([tb_ramp], window=6000)
    renamed = [c for b in cmp.features.blocks for r in b.rows
               for c in r.cells.values() if c.resolved_to]
    assert renamed, 'expected at least one topline rename on a real TB run'
    assert 'renamed ->' in P.format_feature_table(cmp.features)


def test_no_non_live_cell_ever_carries_a_number(tb_ramp, vg_normal, mle_only,
                                                buildout):
    """The invariant behind 'never render it as zero': if a state is not LIVE,
    every statistic is None, in the dataclass and in the flat records."""
    cmp = P.compare([tb_ramp, vg_normal, mle_only, buildout],
                    metrics=K.TOPLINE_TB + K.TOPLINE_VARGRAD, window=6000)
    for block in cmp.features.blocks:
        for row in block.rows:
            for cell in row.cells.values():
                if cell.state is not P.CellState.LIVE:
                    assert cell.feature is None
                    assert all(cell.value(s) is None for s in P.STATS)
    for rec in cmp.records():
        if rec['state'] != P.CellState.LIVE.value:
            assert all(rec[s] is None for s in P.STATS)


def test_no_cell_renders_blank(tb_ramp, vg_normal):
    """A blank cell is unreadable as either of the two things it could be."""
    cmp = P.compare([tb_ramp, vg_normal], metrics=K.TOPLINE_TB, window=6000)
    for block in cmp.features.blocks:
        for row in block.rows:
            for cell in row.cells.values():
                for stat in P.DEFAULT_STATS:
                    assert P._cell_text(cell, stat).strip() != ''


def test_the_reason_a_cell_is_not_a_number_reaches_the_render(tb_ramp):
    """`log_Z_learned` is ABSENT because it is logged under three namespaces
    that are DIFFERENT QUANTITIES. Shown as a bare ABSENT, a reader goes hunting
    a logging bug that is not there."""
    cmp = P.compare([tb_ramp], window=6000)
    text = P.format_feature_table(cmp.features)
    notes = [c.note for b in cmp.features.blocks for r in b.rows
             for c in r.cells.values() if c.state is P.CellState.ABSENT]
    assert notes
    assert any(n in text for n in notes)


# ===========================================================================
# Routes do not share a topline
# ===========================================================================

def test_arms_on_different_routes_get_one_block_each(mle_only, vg_normal):
    cmp = P.compare([mle_only, vg_normal])
    table = cmp.features
    assert table.split_by_route
    assert len(table.blocks) == 2
    for block in table.blocks:
        assert len(block.arm_labels) == 1
        assert block.route is not None
        assert block.metrics == K.TOPLINE[block.route]


def test_no_row_spans_two_routes_by_default(mle_only, vg_normal):
    """The union is what 'handled honestly' rules out: a row holding one number
    from each route asserts the two are commensurable."""
    table = P.compare([mle_only, vg_normal]).features
    for block in table.blocks:
        for row in block.rows:
            assert set(row.cells) == set(block.arm_labels)


def test_same_route_arms_get_one_block(vg_normal, vg_blowup):
    """COMPANION -- otherwise the split could be unconditional."""
    table = P.compare([vg_normal, vg_blowup]).features
    assert not table.split_by_route
    assert len(table.blocks) == 1
    assert table.blocks[0].route is K.Route.VARGRAD_CONDITIONAL


def test_naming_the_metrics_is_the_deliberate_cross_route_read(mle_only,
                                                               vg_normal):
    """One block, both arms, each cell still resolved against ITS OWN route."""
    cmp = P.compare([mle_only, vg_normal], metrics=K.TOPLINE_TB, window=6000)
    assert len(cmp.features.blocks) == 1
    block = cmp.features.blocks[0]
    assert block.route is None
    assert set(block.arm_labels) == {_label(mle_only), _label(vg_normal)}
    assert cmp.features.split_by_route


def test_the_split_note_says_which_of_the_two_it_is(mle_only, vg_normal):
    """Two different situations, and saying the wrong one is a lie about the
    table underneath it."""
    default = P.format_feature_table(P.compare([mle_only, vg_normal]).features)
    requested = P.format_feature_table(
        P.compare([mle_only, vg_normal], metrics=K.TOPLINE_TB).features)
    assert 'no row spans two of them' in default
    assert 'no row spans two of them' not in requested
    assert 'NA_ROUTE' in requested


def test_the_route_a_block_was_read_on_is_stated(tb_ramp):
    """A reader cannot audit a withheld metric without knowing which route's
    rules were applied."""
    text = P.format_feature_table(P.compare([tb_ramp], window=6000).features)
    assert C.context(tb_ramp).route.value in text


# ===========================================================================
# Arms, labels, and refusals
# ===========================================================================

def test_arms_carry_the_route_checks_resolved(vg_normal, tb_ramp):
    """One resolution point. Two modules disagreeing about a run's route is how
    a withheld metric becomes unauditable."""
    for run in (vg_normal, tb_ramp):
        arm, = P.arms([run])
        ctx = C.context(run)
        assert arm.route is ctx.route
        assert arm.stage_name == ctx.stage_name
        assert arm.stage_index == ctx.stage_index


def test_two_arms_sharing_a_display_name_keep_separate_columns(ring_probe,
                                                               ring_cal):
    """MUTATION. Nine display names in the local corpus are shared by two or
    more runs and `mk_dev` alone by eleven, so a battery of two `mk_dev` arms
    gets two identical headings unless the id goes back on."""
    twin = fixtures.mutate(ring_cal)
    twin.name = ring_probe.name
    cmp = P.compare([ring_probe, twin])
    labels = [a.label for a in cmp.arms]
    shorts = [a.short for a in cmp.arms]
    assert len(set(labels)) == 2
    assert len(set(shorts)) == 2, shorts


def test_a_unique_name_is_used_bare(ring_probe, ring_cal):
    """COMPANION -- otherwise the heading could always be the long form."""
    cmp = P.compare([ring_probe, ring_cal])
    assert {a.short for a in cmp.arms} == {ring_probe.name, ring_cal.name}


def test_the_same_run_passed_twice_does_not_lose_a_column(ring_probe):
    """Every table is keyed by label. A silent collision loses a whole arm."""
    cmp = P.compare([ring_probe, ring_probe])
    assert len({a.label for a in cmp.arms}) == 2
    block = cmp.features.blocks[0]
    assert len(block.arm_labels) == 2
    assert all(len(r.cells) == 2 for r in block.rows)


def test_an_empty_battery_raises(ring_probe):
    """An empty `Comparison` renders identically to 'compared them, nothing
    differs', which is the opposite of what is true."""
    with pytest.raises(ValueError):
        P.compare([])


def test_an_empty_metric_list_raises(ring_probe):
    """Same rule one level down: a table with no rows reads as a table whose
    every row came back clean."""
    with pytest.raises(ValueError):
        P.compare([ring_probe], metrics=[])


def test_one_arm_compares_honestly(tb_ramp):
    """Not an error and not a silent pass: the §4 check says its cross-arm
    subjects were skipped, and the sweep has nothing to compare."""
    cmp = P.compare([tb_ramp], window=6000)
    assert cmp.confounds.ran
    assert not cmp.sweep.differs
    assert 'one arm' in P.format_sweep(cmp.sweep)
    assert len(cmp.features.blocks) == 1


def test_an_unconfigured_arm_is_not_silently_compared(ring_probe, ring_cal):
    """MUTATION. `pull` raises only on empty HISTORY, so a run arrives fully
    parsed with `config == {}`. Every knob then reads `<missing>` on one side
    and the sweep table looks like a 300-dimension sweep."""
    blank = fixtures.mutate(ring_cal)
    blank.config = {}
    cmp = P.compare([ring_probe, blank])
    subjects = [f.subject for f in cmp.confounds.findings]
    assert any(s.endswith('/config') for s in subjects), subjects
    assert all(r.kind is P.KnobKind.PRESENCE for r in cmp.sweep.rows)


# ===========================================================================
# Corpus invariants
# ===========================================================================

def test_every_captured_pair_compares_without_a_traceback(all_runs):
    names = sorted(all_runs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = all_runs[names[i]], all_runs[names[j]]
            cmp = P.compare([a, b], window=6000)
            P.format_comparison(cmp)
            P.format_comparison(P.compare([a, b], metrics=K.TOPLINE_TB,
                                          window=6000))


def test_the_whole_corpus_at_once_compares(all_runs):
    runs = [all_runs[n] for n in sorted(all_runs)]
    cmp = P.compare(runs, window=6000)
    assert len(cmp.arms) == len(runs)
    assert {a.label for a in cmp.arms} == {_label(r) for r in runs}
    text = P.format_comparison(cmp)
    assert 'COMPARABILITY' in text and 'SWEEP' in text and 'FEATURES' in text


def test_no_state_fires_on_the_whole_corpus(all_runs):
    """A finding state that fires on everything is not a finding. Measured on
    the 300 real same-tag pairs in `wandb/` the rates are 3.3% (no knob
    differs), 29.7% (route split) and 46.0% (a cross-arm blocker); this pins the
    weaker property the fixtures can carry -- that none of them is universal."""
    runs = [all_runs[n] for n in sorted(all_runs)]
    pairs = [(runs[i], runs[j]) for i in range(len(runs))
             for j in range(i + 1, len(runs))]
    blocked = [bool(P.compare(list(p)).blockers) for p in pairs]
    split = [P.compare(list(p)).features.split_by_route for p in pairs]
    no_knob = [not P.compare(list(p)).sweep.differs for p in pairs]
    for name, flags in (('blocked', blocked), ('split', split),
                        ('no_knob', no_knob)):
        assert any(flags), f'{name} never fires -- it has not been tested'
        assert not all(flags), f'{name} fires on every pair -- it is not a state'


def test_the_report_emits_no_verdict(all_runs):
    """`healthy`, `broken`, `working` are not outputs of this package."""
    banned = ('healthy', 'unhealthy', 'broken', 'is working', 'looks good',
              'looks fine', 'all good', 'no problem')
    runs = [all_runs[n] for n in sorted(all_runs)]
    text = P.format_comparison(P.compare(runs, window=6000)).lower()
    for word in banned:
        assert word not in text, word


def test_compare_holds_no_metric_or_config_literal():
    """`keys.py` is the only file that may carry one, so a rename upstream stays
    a one-file change. Checked by tokenising this module's source rather than by
    reading it, because a literal added in six months will not be read."""
    with open(P.__file__, encoding='utf-8') as f:
        src = f.read()
    literals = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type != tokenize.STRING:
            continue
        try:
            val = eval(tok.string)          # noqa: S307 - our own source
        except Exception:
            continue
        # Prose is excluded by the space test: a metric name has none.
        if isinstance(val, str) and val and ' ' not in val and '\n' not in val:
            literals.add(val)

    metrics = set()
    for topline in K.TOPLINE.values():
        metrics |= set(topline)
    for _, group in K.READ_ORDER:
        metrics |= set(group)
    assert not (literals & metrics), sorted(literals & metrics)

    config_keys = set()
    for name in fixtures.names():
        config_keys |= set(fixtures.load(name).config)
    assert not (literals & config_keys), sorted(literals & config_keys)


def test_arms_whose_configs_did_not_load_are_not_comparable(ring_probe, ring_cal):
    """Two unparseable configs produced a CLEAN comparability bill, a "no knob
    differs" sweep, and a full aligned metric table putting their numbers in the
    same rows -- a confident comparison of two things nothing is known about.
    Every cross-arm subject reads a missing config as `<missing>`, so they all
    agree with each other and nothing flags."""
    a, b = fixtures.mutate(ring_probe), fixtures.mutate(ring_cal)
    a.config.clear()
    b.config.clear()
    c = P.compare([a, b])
    assert not c.comparable
    assert {f.subject for f in c.blockers} == {f'{a.name}/config', f'{b.name}/config'}
    assert all(not rec['comparable'] for rec in c.records())


def test_a_sweep_row_never_renders_two_different_values_the_same(ring_probe, ring_cal):
    """A sweep row exists BECAUSE these values differ. If the column cap renders
    them identically the row asserts the opposite of the fact that put it in the
    table, so the cap loses."""
    shared = 'x' * 40
    a = fixtures.mutate(ring_probe, config={K.CFG_PRIOR_PATH: shared + 'AAAA' + shared})
    b = fixtures.mutate(ring_cal, config={K.CFG_PRIOR_PATH: shared + 'BBBB' + shared})
    text = P.format_sweep(P.compare([a, b]).sweep)
    assert 'AAAA' in text and 'BBBB' in text
