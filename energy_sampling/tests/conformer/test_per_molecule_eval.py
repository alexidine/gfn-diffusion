"""Per-molecule eval on a molecule SET: member dispatch, bounded aggregation, panel, w1r.

WHY THESE GATES. On a carrier every single-chart statistic is refused by the dispatcher, and
a pooled carrier statistic is a mixture reading that one broken molecule of many cannot move.
The per-molecule functions exist to see that molecule, so each gate below constructs the
case the function is for and requires the number to move -- or, for the bookkeeping, to NOT
move when it must not:

  * dispatch is exact: every copied scalar equals the single-molecule call on that member's
    own rows, read out of the carrier;
  * the aggregated key set does not grow with the number of molecules;
  * an unavailable molecule is counted, never averaged in; a thin one is excluded, counted;
  * the panel is a function of (seed, identifier set, features) only;
  * per-molecule w1r sees a collapsed molecule that pooled w1r reports as null;
  * T_eff/T and the non-thermal bar use each molecule's OWN k, not the carrier width;
  * the numbers a gate reads have their defined VALUES, not just their keys: each
    headline's badness transform and its median / worst / max, the keys the block derives
    itself, and the two-sided spread ratio, which is circular on a wrapped column;
  * rows made against a labelled population cannot be read without that label.

Level `full`, force field `mmff` throughout: the conformer training target.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

warnings.filterwarnings('ignore')
try:
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')
except Exception:
    pass

import energies.conformer_eval_metrics as cm
from energies.conformer_carrier import CarrierLayout, carrier_pad_condition
from energies.conformer_data import collate_conditions, condition_from_energy
from energies.conformer_torsions import ConformerTorsions
from energies.dof_features import free_dof_atom_index
from energies.multi_conformer import MultiConformerTorsions
from energies.prior_diagnostics import basin_reference
from progress_metrics import _PROGRESS_GATE, _column_w1_ratio

KW = dict(device='cpu', level='full', force_field='mmff')
#: k = 30, 33, 12 -- three different widths, so the carrier is not the identity. Propanol
#: carries three rotor groups (coverage, coupling and the basin tail go live), THF a ring
#: (ring closure and ring torsions go live), methanol neither.
SMIS = ['CCCO', 'C1CCOC1', 'CO']
N_ROWS = 48
N_REF = 200          # >= 4 x N_ROWS, so per-molecule w1r is available
S = 4.0              # nonthermal_entropy_per_dim, the canonical conformer value
WQ = 0.25            # conditional_worst_quantile, as the canonical configs set it
NS = 'panel_train'


def _draw(member, n, seed):
    """Member-width states: r/theta a small displacement, phi spread over the circle, so the
    rotamer basins are populated and coverage/coupling have something to read."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, member.data_ndim, generator=g, dtype=torch.float64) * 2 - 1
    lin = torch.as_tensor(np.asarray(member._free_block) != 2)
    x[:, lin] *= 0.15
    return x


@pytest.fixture(scope='module')
def carrier():
    """A 3-member carrier batch with its rows SHUFFLED, so dispatch has to regroup.

    float64 for the module's duration, restored at teardown: the default dtype is process
    global, and leaving it set would change every test module collected after this one.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield _build_carrier()
    finally:
        torch.set_default_dtype(old_dtype)


def _build_carrier():
    en = MultiConformerTorsions(SMIS, identifiers=SMIS, **KW)
    lay = en.carrier
    assert en.is_carrier and lay.K == 33
    one = torch.tensor(1.0, dtype=torch.float64)
    conds, xs, es, refs, ref_x = [], [], [], {}, {}
    for j, (ident, m) in enumerate(en._members.items()):
        a, msk = free_dof_atom_index(m)
        cc = carrier_pad_condition(condition_from_energy(m, identifier=ident), lay, ident, m,
                                   atoms=a, mask=msk, R=1)
        x = _draw(m, N_ROWS, seed=j)
        e = m.potential_energy(x, one).detach()
        conds += [cc.__copy__() for _ in range(N_ROWS)]
        xs.append(lay.to_carrier(ident, x))
        es.append(e)
        refs[ident] = {'e_min': float(e.min()) - 1.0, 'basin_ref': basin_reference(m)}
        ref_x[ident] = _draw(m, N_REF, seed=100 + j)
    perm = np.random.default_rng(0).permutation(len(conds))
    batch = collate_conditions([conds[i] for i in perm])
    X = torch.cat(xs)[torch.as_tensor(perm)]
    E = torch.cat(es)[torch.as_tensor(perm)]
    return SimpleNamespace(en=en, lay=lay, batch=batch, X=X, E=E, refs=refs, ref_x=ref_x)


@pytest.fixture(scope='module')
def split(carrier):
    return cm.split_by_member(carrier.en, carrier.X, carrier.batch, carrier.E)


@pytest.fixture(scope='module')
def block(carrier, split):
    members, states, energies, _ = split
    return cm.per_molecule_block(members, states, energies, carrier.refs, n_min=16,
                                 reference_x=carrier.ref_x, reference_label='prior',
                                 nonthermal_entropy_per_dim=S)


def _is_scalar(v):
    return isinstance(v, (bool, int, float, np.integer, np.floating, np.bool_))


# ----------------------------------------------------------------------- dispatch


def test_split_regroups_shuffled_rows_through_each_members_own_columns(carrier, split):
    members, states, energies, rows = split
    assert list(states) and set(states) == set(SMIS)
    for ident, x in states.items():
        k = carrier.lay.k(ident)
        assert tuple(x.shape) == (N_ROWS, k) and members[ident].ndim == k
        # back into the carrier, the member rows ARE the batch rows they came from
        assert torch.equal(carrier.lay.to_carrier(ident, x), carrier.X[rows[ident]])
        assert torch.equal(energies[ident], carrier.E[rows[ident]])


def test_split_refuses_a_nonzero_pad_and_a_foreign_mask(carrier):
    X2 = carrier.X.clone()
    row0 = carrier.batch.identifier[0]
    X2[0, int(carrier.lay.pad_cols(row0)[0])] = 1e-3
    with pytest.raises(RuntimeError, match='PAD'):
        cm.split_by_member(carrier.en, X2, carrier.batch)
    b2 = carrier.batch.clone()
    b2.state_mask = torch.ones_like(b2.state_mask)
    with pytest.raises(RuntimeError, match='state_mask'):
        cm.split_by_member(carrier.en, carrier.X, b2)


def test_block_equals_the_single_molecule_calls(carrier, split, block):
    """THE DISPATCH CLAIM. Every scalar a single-molecule cm.* call produces on a member's
    own rows appears in that member's row with exactly that value -- including the ones the
    block only relabels (population-referenced, under vs_prior/) or reduces (ring torsions,
    max over cycles)."""
    _, _, _, rows = split
    lay = carrier.lay
    live = set()
    for ident, m in carrier.en._members.items():
        row = block[ident]
        assert all(_is_scalar(v) for v in row.values()), 'a per-molecule row carried an array'
        x = lay.from_carrier(ident, carrier.X[rows[ident]])
        e = carrier.E[rows[ident]]
        ref, rx = carrier.refs[ident], carrier.ref_x[ident]
        br, e_min, k = ref['basin_ref'], ref['e_min'], int(m.ndim)

        direct = {}
        for d in (cm.energy_component_stats(m, x), cm.geometry_stats(m, x),
                  cm.dof_class_stats(m, x), cm.dof_element_stats(m, x), cm.ring_stats(m, x),
                  cm.basin_coverage(m, x, br), cm.basin_coupling(m, x, br),
                  cm.thermal_stats(m, e, e_min),
                  cm.basin_nonthermal(m, x, e, e_min, br, S * k)):
            direct.update({key: v for key, v in d.items() if _is_scalar(v)})
        for key, v in direct.items():
            assert row[key] == v, f'{ident} {key}: block {row.get(key)} != direct {v}'

        # population-referenced: the keys the reference ADDS, under the label
        for fn in (cm.dof_class_stats, cm.dof_element_stats):
            free, ref_d = fn(m, x), fn(m, x, reference=rx)
            added = {key: v for key, v in ref_d.items() if key not in free and _is_scalar(v)}
            assert added, f'{ident}: {fn.__name__} added nothing with a reference'
            for key, v in added.items():
                assert row[f'vs_prior/{key}'] == v, (ident, key)
        w = _column_w1_ratio(x.numpy(), rx.numpy(), np.asarray(m.periodic_dims, dtype=bool))
        assert w, f'{ident}: w1r should be available at n={N_ROWS}, n_ref={N_REF}'
        for key, v in w.items():
            assert row[f'vs_prior/{key}'] == v, (ident, key)

        # ring torsions, reduced within the molecule
        rt = cm.ring_torsion_stats(m, x, reference=rx)
        if rt['ringtor/available']:
            tags = [f'ringtor/c{c}' for c in range(rt['ringtor/n_cycles'])]
            assert row['ringtor/sd_max_deg'] == max(rt[f'{t}_sd_max_deg'] for t in tags)
            want = max(max(abs(np.log(rt[f'{t}_sd_ratio_max'])),
                           abs(np.log(rt[f'{t}_sd_ratio_min']))) for t in tags)
            assert row['vs_prior/ringtor/abs_log_sd_ratio_max'] == pytest.approx(want, rel=1e-12)
            assert row['vs_prior/ringtor/corr_dist_max'] == max(rt[f'{t}_corr_dist'] for t in tags)
            live.add('ringtor')
        else:
            assert row['ringtor/available'] == 0
        live |= {g for g, key in (('coupling', 'cover/coupling_tc_debiased'),
                                  ('basin_tail', 'cover/nonthermal_worst_basin_frac'),
                                  ('closure', 'ring/closure_err_a')) if key in row}
    # every reading this set was chosen to exercise actually ran on at least one member --
    # otherwise the equality above could be passing on abstentions
    assert live >= {'ringtor', 'coupling', 'basin_tail', 'closure'}, live


def test_ring_torsions_reduce_to_the_worst_cycle_in_either_direction():
    """The within-molecule reduction over SEVERAL cycles (the fixture's THF has one, where
    min and max coincide). One ring too wide, one collapsed, one shifted: each reduction
    picks the worst cycle, and |log sd ratio| sees the collapse as readily as the widening."""
    rt = {'ringtor/available': 1, 'ringtor/n_cycles': 2,
          'ringtor/c0_sd_max_deg': 30.0, 'ringtor/c0_sd_ratio_max': 1.5,
          'ringtor/c0_sd_ratio_min': 1.1, 'ringtor/c0_corr_dist': 0.1,
          'ringtor/c0_mean_shift_max_deg': 40.0,
          'ringtor/c1_sd_max_deg': 12.0, 'ringtor/c1_sd_ratio_max': 0.9,
          'ringtor/c1_sd_ratio_min': 0.2, 'ringtor/c1_corr_dist': 0.7,
          'ringtor/c1_mean_shift_max_deg': 5.0}
    out = cm._reduce_ringtor(rt, 'vs_prior/')
    assert out['ringtor/sd_max_deg'] == 30.0
    assert out['vs_prior/ringtor/abs_log_sd_ratio_max'] == pytest.approx(abs(np.log(0.2)))
    assert out['vs_prior/ringtor/corr_dist_max'] == 0.7
    assert out['vs_prior/ringtor/mean_shift_max_deg'] == 40.0
    # without a reference only the sampler-side reduction is there
    assert set(cm._reduce_ringtor(rt, None)) == {'ringtor/available', 'ringtor/n_cycles',
                                                 'ringtor/sd_max_deg'}


def test_T_eff_reads_two_on_a_harmonic_excess_of_half_d(carrier, split):
    """Each molecule's median excess set to exactly d/2 with d = ITS k: T_eff/T = 2.0 on
    every member. With the carrier width K in place of k, two of these three would not."""
    members, states, _, _ = split
    T = {i: float(m.temperature) for i, m in members.items()}
    energies = {i: torch.full((N_ROWS,), 0.5 * int(m.ndim) * T[i], dtype=torch.float64)
                for i, m in members.items()}
    refs = {i: {'e_min': 0.0} for i in members}
    per = cm.per_molecule_block(members, states, energies, refs, n_min=16)
    assert sum(int(m.ndim) != carrier.lay.K for m in members.values()) == 2
    for ident in members:
        assert per[ident]['E/T_eff_over_T'] == pytest.approx(2.0, abs=1e-12), ident
    agg = cm.aggregate_per_condition(per, None, WQ, ns=NS)
    assert agg[f'{NS}/thermal/T_eff_over_T_dev/max'] == pytest.approx(0.0, abs=1e-12)


def test_nonthermal_bar_is_s_times_the_members_own_k(carrier, split):
    """u* = s * k per member. Excess straddles s*k by +-0.5 nats on each member, so the
    fraction reads exactly 0.5 everywhere; a bar built on K would read 0 on the two
    narrower molecules."""
    members, states, _, _ = split
    energies, refs = {}, {}
    for ident, m in members.items():
        us = S * int(m.ndim) * float(m.temperature)
        e = torch.full((N_ROWS,), us - 0.5, dtype=torch.float64)
        e[N_ROWS // 2:] = us + 0.5
        energies[ident], refs[ident] = e, {'e_min': 0.0, 'basin_ref': carrier.refs[ident]['basin_ref']}
    per = cm.per_molecule_block(members, states, energies, refs, n_min=16,
                                nonthermal_entropy_per_dim=S)
    for ident, m in members.items():
        k = int(m.ndim)
        assert per[ident]['thermal/nonthermal_u_star'] == S * k
        assert per[ident]['thermal/nonthermal_frac'] == 0.5, ident
        # and the SAME bar reached the basin-grouped tail
        want = cm.basin_nonthermal(m, states[ident], energies[ident], 0.0,
                                   refs[ident]['basin_ref'], S * k)
        for key, v in want.items():
            assert per[ident][key] == v, (ident, key)


def test_block_derived_keys_have_their_defined_values(split):
    """The keys the block COMPUTES rather than copies, at their defined values: missed
    basins over the ACCESSIBLE count, the finite fraction, and the non-thermal fraction
    over FINITE rows with the excess in units of the member's own temperature. T = 2
    here, so an excess left undivided would read every row as tail; two non-finite rows
    on the cool side, so a denominator over all rows would read 0.5."""
    members, states, _, _ = split
    ident = 'CCCO'
    m, x = members[ident], states[ident]
    k = int(m.ndim)
    # a TIGHTER accessible window than basin_reference's default, so that the accessible
    # count and the mode count differ and the two denominators can be told apart
    br = basin_reference(m, accessible_kt=0.5)
    acc = np.asarray(br['accessible'], dtype=bool)
    old = m.log_temperature
    m.log_temperature = float(np.log10(2.0))
    try:
        T = float(m.temperature)
        e = torch.full((N_ROWS,), T * (S * k - 0.5), dtype=torch.float64)
        e[N_ROWS // 2:] = T * (S * k + 0.5)
        e[0], e[1] = float('nan'), float('inf')
        per = cm.per_molecule_block({ident: m}, {ident: x}, {ident: e},
                                    {ident: {'e_min': 0.0, 'basin_ref': br}}, n_min=16,
                                    nonthermal_entropy_per_dim=S)[ident]
        assert per['phys/finite_frac'] == (N_ROWS - 2) / N_ROWS
        assert per['thermal/nonthermal_u_star'] == S * k
        assert per['thermal/nonthermal_frac'] == (N_ROWS // 2) / (N_ROWS - 2)
        # the copies are still the single-molecule calls, at this temperature
        for d in (cm.thermal_stats(m, e, 0.0), cm.basin_nonthermal(m, x, e, 0.0, br, S * k)):
            for key, v in d.items():
                if _is_scalar(v):
                    assert per[key] == v, key
    finally:
        m.log_temperature = old
    assert 0 < acc.sum() < len(br['combos']) and per['cover/n_missed'] > 0
    assert per['cover/n_modes'] == len(br['combos'])
    assert per['cover/n_accessible'] == int(acc.sum())
    assert per['cover/missed_frac'] == per['cover/n_missed'] / int(acc.sum())


def test_abs_log_sd_ratio_is_two_sided_and_circular_on_wrapped_columns(carrier, split, block):
    """``dof/abs_log_sd_ratio_max``: a collapsed column scores as readily as a widened one,
    and a wrapped column's spread is CIRCULAR. The wrapped case is a tight rotamer 180
    degrees from the reference dihedral, i.e. straddling x = +-1, where a linear sd reads
    the wrap instead of the width."""
    members, states, _, _ = split
    for ident, m in members.items():
        assert block[ident]['vs_prior/dof/abs_log_sd_ratio_max'] == cm._abs_log_sd_ratio_max(
            states[ident], carrier.ref_x[ident], m.periodic_dims), ident
    # samples = the reference itself with ONE linear column rescaled: every other ratio is
    # exactly 1, so the reading is exactly |log scale| on either side of 1
    m, rx = members['CCCO'], carrier.ref_x['CCCO']
    j = int(np.flatnonzero(~np.asarray(m.periodic_dims, dtype=bool))[0])
    for scale in (0.3, 2.0):
        xs = rx.clone()
        xs[:, j] *= scale
        assert cm._abs_log_sd_ratio_max(xs, rx, m.periodic_dims) == pytest.approx(
            abs(np.log(scale)), rel=1e-9), scale

    z = np.random.default_rng(0).standard_normal((256, 2))
    w = 8.0 / 180.0                                   # 8 degrees, in units of pi

    def wrap(a):
        return (a + 1.0) % 2.0 - 1.0

    ref = np.stack([0.1 * z[:, 0], wrap(1.0 + w * z[:, 1])], axis=1)
    assert ref[:, 1].std() > 0.5                      # the linear sd sees the wrap: ~1
    periodic = [False, True]
    collapsed = np.stack([ref[:, 0], wrap(1.0 + 0.3 * w * z[:, 1])], axis=1)
    shifted = np.stack([ref[:, 0], wrap(1.0 + 15.0 / 180.0 + w * z[:, 1])], axis=1)
    assert cm._abs_log_sd_ratio_max(collapsed, ref, periodic) == pytest.approx(
        abs(np.log(0.3)), abs=0.02)
    assert cm._abs_log_sd_ratio_max(shifted, ref, periodic) == pytest.approx(0.0, abs=1e-9)


def test_reference_free_readings_need_no_reference(split, carrier):
    """No population and no target reference: the reference-free half is still there, and
    nothing labelled vs_* appears."""
    members, states, energies, _ = split
    per = cm.per_molecule_block(members, states, energies, None, n_min=16)
    for ident, row in per.items():
        assert 'phys/finite_frac' in row and 'geom/all_in_range' in row and 'E/total_p50' in row
        assert row['E/excess_available'] == 0 and row['cover/available'] == 0
        assert not any(key.startswith('vs_') for key in row), ident
    agg = cm.aggregate_per_condition(per, None, WQ, ns=NS)
    assert not any('/vs_' in key for key in agg)
    assert agg[f'{NS}/phys/nonfinite_frac/n'] == len(SMIS)


def test_a_population_reference_must_be_labelled(split, carrier):
    members, states, energies, _ = split
    for bad in (None, 'Prior!', ''):
        with pytest.raises(ValueError, match='label'):
            cm.per_molecule_block(members, states, energies, carrier.refs, n_min=16,
                                  reference_x=carrier.ref_x, reference_label=bad)
    with pytest.raises(ValueError, match='without reference_x'):
        cm.per_molecule_block(members, states, energies, carrier.refs, n_min=16,
                              reference_label='prior')


def test_temperature_conditioning_is_refused(split):
    members, states, energies, _ = split
    m = members['CO']
    m.temperature_conditioning = True
    try:
        with pytest.raises(ValueError, match='temperature'):
            cm.per_molecule_block({'CO': m}, {'CO': states['CO']}, {'CO': energies['CO']},
                                  None, n_min=16)
    finally:
        m.temperature_conditioning = False


# -------------------------------------------------------------------- aggregation


def _jittered_copies(per, n_copies):
    """M -> M x n_copies rows, each copy under a new identifier with its floats scaled."""
    out = {}
    for r in range(n_copies):
        for ident, row in per.items():
            out[f'{ident}#{r}'] = {key: (v * (1.0 + 0.01 * r) if isinstance(v, float) else v)
                                   for key, v in row.items()}
    return out


def test_aggregated_key_set_does_not_grow_with_M(block):
    a3 = cm.aggregate_per_condition(block, None, WQ, ns=NS, reference_label='prior')
    per30 = _jittered_copies(block, 10)
    a30 = cm.aggregate_per_condition(per30, None, WQ, ns=NS, reference_label='prior')
    assert a3[f'{NS}/n_molecules'] == 3 and a30[f'{NS}/n_molecules'] == 30
    assert set(a3) == set(a30)
    assert len(a3) <= 2 + 5 * len(cm.PANEL_HEADLINES)
    assert not any(ident.split('#')[0] in key for key in a30 for ident in per30)
    # every headline of this set is live on at least one molecule, so the bound above is
    # met with values rather than with a column of abstentions
    names = [f"vs_prior/{h['name']}" if h['reference'] == 'population' else h['name']
             for h in cm.PANEL_HEADLINES]
    assert all(f'{NS}/{n}/median' in a3 for n in names), \
        [n for n in names if f'{NS}/{n}/median' not in a3]


def test_an_unavailable_molecule_is_counted_never_averaged(block):
    """A molecule whose every reading abstained: n_unavailable +1 on every headline, and
    median / worst / max do not move."""
    a0 = cm.aggregate_per_condition(block, None, WQ, ns=NS, reference_label='prior')
    ghost = {'phys/n_rows': N_ROWS, 'phys/k': 9, 'phys/energy_available': 0,
             'E/excess_available': 0, 'cover/available': 0, 'cover/coupling_available': 0,
             'ring/available': 0, 'ringtor/available': 0, 'vs_prior/w1r_available': 0,
             'thermal/nonthermal_available': 0, 'geom/all_in_range_frozen': 1}
    a1 = cm.aggregate_per_condition({**block, 'ghost': ghost}, None, WQ, ns=NS,
                                    reference_label='prior')
    assert a1[f'{NS}/n_molecules'] == a0[f'{NS}/n_molecules'] + 1
    heads = [key[:-len('/n_unavailable')] for key in a0 if key.endswith('/n_unavailable')]
    assert len(heads) == len(cm.PANEL_HEADLINES)
    for h in heads:
        assert a1[f'{h}/n_unavailable'] == a0[f'{h}/n_unavailable'] + 1, h
        assert a1[f'{h}/n'] == a0[f'{h}/n'], h
        for stat in ('median', 'worst', 'max'):
            assert a1.get(f'{h}/{stat}') == a0.get(f'{h}/{stat}'), (h, stat)


def test_molecules_below_n_min_are_excluded_and_counted(carrier, split):
    members, states, energies, _ = split
    short = dict(states)
    short['CO'] = states['CO'][:10]
    e_short = dict(energies)
    e_short['CO'] = energies['CO'][:10]
    per = cm.per_molecule_block(members, short, e_short, carrier.refs, n_min=16,
                                reference_x=carrier.ref_x, reference_label='prior',
                                nonthermal_entropy_per_dim=S)
    assert per['CO'] == {'phys/n_rows': 10, 'phys/k': 12, 'phys/below_n_min': 1}
    agg = cm.aggregate_per_condition(per, None, WQ, ns=NS, reference_label='prior')
    assert agg[f'{NS}/n_below_n_min'] == 1 and agg[f'{NS}/n_molecules'] == 2
    for key in agg:
        if key.endswith('/n_unavailable'):
            h = key[:-len('/n_unavailable')]
            assert agg[f'{h}/n'] + agg[key] == 2, h


def test_worst_is_the_per_condition_fraction_tail():
    """'worst' is utils.per_condition_fraction's convention, checked against that function:
    ten molecules whose missed fractions are exactly the per-condition fractions of a 0/1
    indicator over ten conditions."""
    from utils import per_condition_fraction

    p = np.arange(10) / 10.0
    per = {f'm{i}': {'phys/n_rows': 10, 'cover/missed_frac': float(v)} for i, v in enumerate(p)}
    agg = cm.aggregate_per_condition(per, None, WQ, ns=NS)
    ind = np.concatenate([(np.arange(10) < round(10 * v)).astype(float) for v in p])
    cid = np.repeat(np.arange(10), 10)
    ref = per_condition_fraction(torch.as_tensor(ind), torch.as_tensor(cid), bar=0.5,
                                 worst_quantile=WQ, higher_is_worse=True)
    assert agg[f'{NS}/cover/missed_frac/worst'] == pytest.approx(ref['worst'], abs=1e-6)
    assert agg[f'{NS}/cover/missed_frac/max'] == pytest.approx(0.9)
    assert agg[f'{NS}/cover/missed_frac/median'] == pytest.approx(0.45)
    # the other twelve headlines have no value on these rows and say so
    assert agg[f'{NS}/geom/out_of_box_frac/n_unavailable'] == 10


def test_headline_transforms_and_statistics_on_asymmetric_rows():
    """Each transform's badness, and median / worst / max over it, against hand-computed
    values. The rows are asymmetric on purpose: every list's mean differs from its median,
    the worst T_eff/T and the worst equipartition reading come from a molecule BELOW its
    ideal (too cold), and the shortfall fractions sit below their ideal of 1."""
    raw = {'E/T_eff_over_T': [0.5, 2.0, 2.5, 2.2, 2.1],               # abs_dev, ideal 2
           'E/frac_within_equipartition': [0.1, 0.5, 0.6, 0.45, 0.5],  # abs_dev, ideal 0.5
           'geom/all_in_range': [1.0, 0.9, 0.5, 1.0, 0.98],            # shortfall, ideal 1
           'phys/finite_frac': [1.0, 0.75, 1.0, 1.0, 0.5],             # shortfall, ideal 1
           'cover/missed_frac': [0.0, 0.0, 0.1, 0.2, 0.9]}             # identity
    per = {f'm{i}': {'phys/n_rows': 64, **{key: v[i] for key, v in raw.items()}}
           for i in range(5)}
    agg = cm.aggregate_per_condition(per, None, WQ, ns=NS)
    badness = {'thermal/T_eff_over_T_dev': [1.5, 0.0, 0.5, 0.2, 0.1],
               'thermal/equipartition_dev': [0.4, 0.0, 0.1, 0.05, 0.0],
               'geom/out_of_box_frac': [0.0, 0.1, 0.5, 0.0, 0.02],
               'phys/nonfinite_frac': [0.0, 0.25, 0.0, 0.0, 0.5],
               'cover/missed_frac': [0.0, 0.0, 0.1, 0.2, 0.9]}
    for name, b in badness.items():
        s = sorted(b)
        assert np.mean(b) != pytest.approx(s[2]), name        # median is not the mean here
        assert agg[f'{NS}/{name}/n'] == 5 and agg[f'{NS}/{name}/n_unavailable'] == 0
        assert agg[f'{NS}/{name}/median'] == pytest.approx(s[2]), name
        # worst = the (1 - WQ) = 0.75 quantile: over five values, exactly the 4th smallest
        assert agg[f'{NS}/{name}/worst'] == pytest.approx(s[3]), name
        assert agg[f'{NS}/{name}/max'] == pytest.approx(s[4]), name
    assert cm.worst_molecules(per, 'thermal/T_eff_over_T_dev', k=1) == [('m0', 1.5)]


def test_rows_made_against_a_population_must_be_read_with_its_label(block):
    """Read with no label, or another one, vs_prior/ rows would silently lose every
    population headline -- the per-molecule w1r among them. All three readers refuse."""
    for bad in (None, 'target'):
        with pytest.raises(ValueError, match='labelled'):
            cm.aggregate_per_condition(block, None, WQ, ns=NS, reference_label=bad)
        with pytest.raises(ValueError, match='labelled'):
            cm.worst_molecules(block, 'geom/out_of_box_frac', reference_label=bad)
        with pytest.raises(ValueError, match='labelled'):
            cm.per_molecule_correlations(block, {}, ns=NS, reference_label=bad)
    # a label over rows with no population keys is not an error: the population headlines
    # are there, counted unavailable on every molecule
    plain = {i: {key: v for key, v in r.items() if not key.startswith('vs_')}
             for i, r in block.items()}
    agg = cm.aggregate_per_condition(plain, None, WQ, ns=NS, reference_label='prior')
    assert agg[f'{NS}/vs_prior/w1r/median/n_unavailable'] == len(plain)


def test_worst_molecules_names_the_tail(block):
    top = cm.worst_molecules(block, 'vs_prior/w1r/worst', k=2, reference_label='prior')
    assert len(top) == 2 and top[0][1] >= top[1][1]
    assert top[0][1] == max(r['vs_prior/w1r/worst'] for r in block.values())


# -------------------------------------------------------------------- correlations


def test_feature_correlations_over_molecules(block, split):
    """Live where a feature takes >= 3 distinct values, *_available = 0 below that."""
    members = split[0]
    feats = {i: cm.molecule_features(m) for i, m in members.items()}
    assert [feats[i]['n_rings'] for i in SMIS] == [0, 1, 0]
    out = cm.per_molecule_correlations(block, feats, ns=NS, reference_label='prior')
    base = f'{NS}/corr/vs_prior/dof/abs_log_sd_ratio/'
    assert out[f'{base}k_available'] == 1 and np.isfinite(out[f'{base}k_pearson'])
    assert out[f'{base}n_rings_available'] == 0 and out[f'{base}n_rings_n_distinct'] == 2
    assert out[f'{base}n_modes_available'] == 0       # not passed: all None
    # and on a constructed library the correlation is actually measured
    per = {f'm{i}': {'phys/n_rows': 64, 'cover/missed_frac': 0.1 * i} for i in range(6)}
    fs = {f'm{i}': {'k': 10 + 3 * i, 'n_rings': i % 3, 'n_rotors': i, 'n_modes': 2 + i}
          for i in range(6)}
    live = cm.per_molecule_correlations(per, fs, ns=NS)
    assert live[f'{NS}/corr/cover/missed_frac/k_pearson'] == pytest.approx(1.0)


def test_n_rotors_counts_acyclic_single_bond_groups(split):
    """Rotors, not central bonds: THF's three ring-bond groups are not rotors, and neither
    are toluene's five aromatic-bond groups or 2-butene's C=C. Methyl and hydroxyl groups
    are -- free torsions with rotamer modes at explicit hydrogens."""
    members = split[0]
    assert {i: cm.molecule_features(m)['n_rotors'] for i, m in members.items()} == \
        {'CCCO': 3, 'C1CCOC1': 0, 'CO': 1}
    for smi, n_groups, n_rotors in (('Cc1ccccc1', 6, 1), ('C/C=C/C', 3, 2)):
        m = ConformerTorsions(smi, **KW)
        assert len(m.torsion_groups()) == n_groups, smi
        assert cm.molecule_features(m)['n_rotors'] == n_rotors, smi


# --------------------------------------------------------------------------- panel


def _library(M, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f'mol{j:04d}' for j in range(M)]
    feats = {i: {'k': int(rng.integers(6, 76)), 'n_rings': int(rng.integers(0, 3)),
                 'n_modes': [None, 1, 3, 9, 27][int(rng.integers(0, 5))]} for i in ids}
    return ids, feats


def test_select_panel_is_deterministic_and_stratified():
    ids, feats = _library(300)
    p = cm.select_panel(ids, feats, 32, seed=7)
    assert len(p) == 32 == len(set(p)) and p == sorted(p)
    # input order and dict order do not matter
    rng = np.random.default_rng(1)
    shuffled = [ids[j] for j in rng.permutation(len(ids))]
    feats_shuffled = {i: dict(feats[i]) for i in reversed(shuffled)}
    assert cm.select_panel(shuffled, feats_shuffled, 32, seed=7) == p
    assert cm.select_panel(ids, feats, 32, seed=8) != p
    # every stratum axis is represented at both ends
    assert {feats[i]['n_rings'] > 0 for i in p} == {True, False}
    assert {(feats[i]['n_modes'] or 0) >= 2 for i in p} == {True, False}
    ks = np.array([feats[i]['k'] for i in ids])
    lo, hi = np.quantile(ks, [1 / 3, 2 / 3])
    pk = np.array([feats[i]['k'] for i in p])
    assert (pk <= lo).any() and (pk > hi).any()


def test_select_panel_with_fewer_slots_than_strata():
    """size below the stratum count: no stratum gets a guaranteed slot, and the panel is
    still exactly `size` molecules, deterministic under permuted input."""
    ids, feats = _library(300)
    edges = np.quantile([feats[i]['k'] for i in ids], [1 / 3, 2 / 3])
    strata = {(int(np.searchsorted(edges, feats[i]['k'], side='right')),
               int(feats[i]['n_rings'] > 0), int((feats[i]['n_modes'] or 0) >= 2))
              for i in ids}
    assert len(strata) == 12
    for size in (1, 5, 11):
        p = cm.select_panel(ids, feats, size, seed=3)
        assert len(p) == size == len(set(p)) and p == sorted(p), size
        assert cm.select_panel(list(reversed(ids)), feats, size, seed=3) == p


def test_select_panel_takes_everything_when_M_fits():
    ids, feats = _library(20)
    assert cm.select_panel(list(reversed(ids)), feats, 32, seed=0) == sorted(ids)
    assert cm.select_panel(ids, {}, 20, seed=0) == sorted(ids)   # no stratification needed


def test_select_panel_refuses_duplicates_and_missing_features():
    ids, feats = _library(50)
    with pytest.raises(ValueError, match='duplicate'):
        cm.select_panel(ids + ids[:1], feats, 8, seed=0)
    feats.pop(ids[3])
    with pytest.raises(KeyError, match='no features'):
        cm.select_panel(ids, feats, 8, seed=0)


# ----------------------------------------------------------------- w1r dilution


def test_per_molecule_w1r_hosts_the_reference_before_caching(carrier, split):
    """The cached path fingerprints the reference, and must host it first like every other
    input: on a CUDA buffer np.asarray raises. CUDA is not exercised on CPU, so a reference
    that still requires grad stands in -- np.asarray refuses that one too."""
    members, states, _, _ = split
    rx = {i: carrier.ref_x[i].clone().requires_grad_(True) for i in members}
    with pytest.raises((RuntimeError, TypeError)):
        np.asarray(rx['CO'])
    cached = cm.per_molecule_w1r(members, states, rx, 'prior', cache={})
    assert cached == cm.per_molecule_w1r(members, states, carrier.ref_x, 'prior')


def _fake_member(n_atoms):
    """The only parts of a member w1r and the carrier read: block layout and wrap mask.
    Block counts (N-1, N-2, N-3) are what every admissible molecule has at `full`."""
    fb = np.repeat([0, 1, 2], [n_atoms - 1, n_atoms - 2, n_atoms - 3])
    return SimpleNamespace(_free_block=fb, periodic_dims=[bool(b == 2) for b in fb])


def _molecule_sampler(k_lin, k_phi, rng):
    """A fixed per-molecule distribution: Gaussian linear columns, 1-3 wrapped rotamer
    modes per torsion column. Returns draw(n, generator) -> [n, k_lin + k_phi]."""
    mu, sd = rng.uniform(-0.3, 0.3, k_lin), rng.uniform(0.05, 0.25, k_lin)
    modes = [rng.uniform(-1, 1, rng.integers(1, 4)) for _ in range(k_phi)]
    width = rng.uniform(0.05, 0.2, k_phi)

    def draw(n, g):
        lin = np.clip(mu + sd * g.standard_normal((n, k_lin)), -1, 1)
        phi = np.stack([modes[j][g.integers(0, len(modes[j]), n)] + width[j] * g.standard_normal(n)
                        for j in range(k_phi)], axis=1)
        return np.concatenate([lin, ((phi + 1) % 2) - 1], axis=1)
    return draw


def test_per_molecule_w1r_flags_a_collapsed_molecule_that_pooled_w1r_misses():
    """eval_audit/w1r_dilution.py, on a synthetic 40-molecule library (k 18..57, K 57): one
    molecule's samples shrunk to 0.3x, every other molecule an exact draw of its own
    distribution. Pooled carrier w1r stays under the progress-gate bars and barely moves;
    the worst-molecule per-molecule reading is that molecule, far past the control.

    Margins, measured over library seeds 0-3 with this molecule (k = 33) collapsed: pooled
    median moves by at most 0.17; per-molecule median/max goes from 1.1-1.2 to 4.3-4.6.
    The same run on the real prior (w1r_dilution.py's 40 QM9-like molecules): median/max
    1.29 -> 3.2-3.8, pooled median 0.88 -> 0.90-1.11.
    Collapsing the WIDEST molecule (k = K) is not diluted the same way -- the carrier
    columns only it occupies are its alone, and pooled median moved 0.72 -> 1.29 there -- so
    the dilution claim is about a molecule inside the carrier, which is most of them."""
    rng = np.random.default_rng(0)
    atoms = [8 + (j % 14) for j in range(40)]                     # N 8..21 -> k 18..57
    idents = [f'mol{j:02d}' for j in range(40)]
    members = {i: _fake_member(a) for i, a in zip(idents, atoms)}
    lay = CarrierLayout(members)
    assert lay.K == 57 and min(lay.k(i) for i in idents) == 18
    samplers = {i: _molecule_sampler(2 * a - 3, a - 3, rng) for i, a in zip(idents, atoms)}
    g = np.random.default_rng(1)
    ref = {i: samplers[i](250, g) for i in idents}
    sam = {i: samplers[i](62, g) for i in idents}
    per_col = np.asarray(lay.free_block == 2)
    pooled_ref = np.concatenate([lay.to_carrier(i, torch.as_tensor(ref[i])).numpy()
                                 for i in idents])
    bad = 'mol05'

    def readings(collapse):
        s = {i: (x * 0.3 if collapse and i == bad else x) for i, x in sam.items()}
        pooled_s = np.concatenate([lay.to_carrier(i, torch.as_tensor(s[i])).numpy()
                                   for i in idents])
        pooled = _column_w1_ratio(pooled_s, pooled_ref, per_col, cache={})
        per = cm.per_molecule_w1r(members, s, ref, 'prior')
        agg = cm.aggregate_per_condition(per, None, WQ, ns=NS, reference_label='prior')
        top = cm.worst_molecules(per, 'vs_prior/w1r/median', k=1, reference_label='prior')
        return pooled, agg, top

    p0, a0, _ = readings(collapse=False)
    p1, a1, top = readings(collapse=True)
    bars = {m['key']: m['bar'] for m in _PROGRESS_GATE['metrics']}
    for key in ('w1r/median', 'w1r/worst'):
        assert p1[key] < bars[key], (key, p1[key])            # pooled: under the gate bar
    assert abs(p1['w1r/median'] - p0['w1r/median']) < 0.25, (p0['w1r/median'], p1['w1r/median'])
    assert top[0][0] == bad                                     # per-molecule: THIS molecule
    # the gate reading is median/max, not worst/max: each molecule's worst column is an
    # extreme value over k noisy columns at n = 62 (see per_molecule_w1r)
    key = f'{NS}/vs_prior/w1r/median/max'
    assert a1[key] > 2.0 * a0[key], (a0[key], a1[key])
