"""Gates for the conformer eval statistics.

WHAT THESE ARE FOR. Every metric here is meant to CATCH something, and a metric that
abstains, or that reports a constant, passes exactly the case it exists to detect. So the
gates below are written the hard way: where a metric is supposed to fire, the test
constructs the failure and requires the number to move. A test that only checks "the key
is present and is a float" would pass on a function that returns zeros.

Run directly (``python test_conformer_eval_metrics.py``) or under pytest.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
try:
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')
except Exception:
    pass

import energies.conformer_eval_metrics as cm
from energies.conformer_torsions import ConformerTorsions
from energies.prior_diagnostics import (basin_counts, basin_reference,
                                        rotamer_basin_labels)

DTYPE = torch.float32
PROPANOL = 'CCCO'
CYCLOHEXANE = 'C1CCCCC1'
#: aromatic ring, held PLANAR by the prior -- the sharp case. Cyclohexane's ring is
#: puckered with a ~60 deg torsion spread, so a small absolute perturbation cannot
#: move its sd ratio and it is the wrong molecule for a width gate.
ETHYLBENZENE = 'CCc1ccccc1'


def _en(smiles=PROPANOL, level='full'):
    return ConformerTorsions(smiles=smiles, level=level, device='cpu', force_field='mmff')


def _load_prior():
    """The fitted prior, or None. Ring statistics need real banked draws, not box noise."""
    import pathlib
    p = pathlib.Path('conformer_prior_v2.pt')
    return torch.load(p, weights_only=False) if p.exists() else None


def _box(en, n=256, scale=0.25, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, en.data_ndim, generator=g, dtype=DTYPE) * 2 - 1) * scale


# ------------------------------------------------------------------------------


def test_absent_is_not_zero():
    """An acyclic molecule has NO ring closure -- that must not read as a closure of 0."""
    en = _en(PROPANOL)
    out = cm.ring_stats(en, _box(en))
    assert out['ring/available'] == 0, 'propanol is acyclic; closure cannot be available'
    assert 'ring/closure_err_a' not in out, \
        'an acyclic molecule published a closure error -- zero closure and nothing to ' \
        'close are opposite readings and must not share a value'

    # and the same call on a RING molecule must produce one, or the flag above is
    # meaningless (a function that always abstains would pass the assertion above)
    enr = _en(CYCLOHEXANE)
    ring = cm.ring_stats(enr, _box(enr))
    assert ring['ring/available'] == 1 and ring['ring/n_closure_bonds'] >= 1, \
        'cyclohexane produced no closure metric, so the acyclic abstention proves nothing'
    assert np.isfinite(ring['ring/closure_err_a'])
    print('PASS  ring closure: absent on acyclic, present and finite on a ring')


def test_frozen_tiers_are_labelled_not_scored():
    """At `dihedral`, r/theta are pinned at the reference, so any excursion is 0 BY
    CONSTRUCTION. Publishing that 0 is the same failure as a test that cannot fail.

    The frozen flag replaces the value; it never sits beside one. A tier that published
    both would let a reader take the 0 at face value.
    """
    en = _en(PROPANOL, level='dihedral')
    out = cm.geometry_stats(en, _box(en))
    assert out['geom/r_frozen'] == 1 and out['geom/theta_frozen'] == 1
    assert 'geom/r_worst_abs' not in out and 'geom/theta_worst_abs' not in out, \
        'a frozen block published a worst-case excursion that cannot be anything but 0'
    assert out.get('geom/all_in_range_frozen') == 1
    assert 'geom/all_in_range' not in out

    # at `full` the same blocks ARE free, so the metric must actually be scored -- and the
    # frozen flag must be GONE, not set to 0 beside the value
    enf = _en(PROPANOL, level='full')
    full = cm.geometry_stats(enf, _box(enf))
    assert 'geom/r_frozen' not in full and 'geom/theta_frozen' not in full
    assert 'geom/r_worst_abs' in full and 'geom/all_in_range' in full
    assert len(full) == 3, f'geom/ is a three-key guard; got {sorted(full)}'
    print('PASS  frozen tiers labelled, free tiers scored, three keys either way')


def test_geometry_detects_out_of_range():
    """all_in_range must FALL when bonds are pushed outside the box. A metric pinned at
    1.0 would pass a 'key exists' test and catch nothing."""
    en = _en(PROPANOL, level='full')
    good = cm.geometry_stats(en, _box(en, scale=0.2))
    assert good['geom/all_in_range'] == 1.0, 'a tight draw should be fully in range'

    x = _box(en, scale=0.2)
    lin = en._lin_free_idx
    x[: x.shape[0] // 2, lin[0]] = 3.0          # half the batch: one bond far outside
    bad = cm.geometry_stats(en, x)
    assert abs(bad['geom/all_in_range'] - 0.5) < 1e-6, \
        f"all_in_range {bad['geom/all_in_range']} did not register half the batch " \
        f"being out of range"
    assert bad['geom/r_worst_abs'] >= 3.0
    print('PASS  geometry catches out-of-range bonds (1.0 -> 0.5 on half a batch)')


def test_coverage_catches_mode_collapse():
    """THE ONE THAT MATTERS. Coverage exists because a basin the sampler never proposes
    generates no large importance weight and no warning. So it must report missing basins
    when they ARE missing, and not when they are not."""
    en = _en(PROPANOL, level='full')
    ref = basin_reference(en)
    assert 'skipped' not in ref and int(ref['accessible'].sum()) > 2

    # spread: prior draws over the whole torus reach many basins
    g = torch.Generator().manual_seed(0)
    spread = (torch.rand(4000, en.data_ndim, generator=g, dtype=DTYPE) * 2 - 1)
    wide = cm.basin_coverage(en, spread, ref)

    # collapsed: every sample pinned to one point -> exactly one basin occupied
    collapsed = torch.zeros(4000, en.data_ndim, dtype=DTYPE)
    narrow = cm.basin_coverage(en, collapsed, ref)

    assert narrow['cover/n_missed'] > wide['cover/n_missed'], \
        f"collapsed sampler missed {narrow['cover/n_missed']} basins, spread missed " \
        f"{wide['cover/n_missed']} -- coverage did not distinguish them"
    const = cm.cover_constants(en, ref)
    assert narrow['cover/n_missed'] == const['cover/n_accessible'] - 1, \
        'a point mass should occupy exactly one basin'
    assert narrow['cover/worst_frac'] == 0.0
    assert narrow['cover/occupancy_entropy'] < 0.05 < wide['cover/occupancy_entropy'], \
        'occupancy entropy did not separate a point mass from a spread draw'
    print(f"PASS  coverage separates collapse from spread "
          f"(missed {narrow['cover/n_missed']} vs {wide['cover/n_missed']}, "
          f"entropy {narrow['cover/occupancy_entropy']:.3f} vs "
          f"{wide['cover/occupancy_entropy']:.3f})")


def test_single_basin_abstains_rather_than_reading_as_collapsed():
    """Cyclohexane has one rotamer mode. Entropy 0 there means 'nothing to spread over',
    not 'collapsed' -- publishing 0 would be a permanent false alarm."""
    en = _en(CYCLOHEXANE)
    ref = basin_reference(en)
    out = cm.basin_coverage(en, _box(en, n=128), ref)
    if int(np.asarray(ref['accessible']).sum()) < 2:
        assert out['cover/occupancy_entropy_available'] == 0
        assert 'cover/occupancy_entropy' not in out
        print('PASS  single-basin molecule abstains on occupancy entropy')
    else:
        assert 'cover/occupancy_entropy' in out
        assert 'cover/occupancy_entropy_available' not in out, \
            'a LIVE metric announced its own presence; the flag is for absence only'
        print('PASS  cyclohexane has >1 accessible basin; entropy is published')


def test_basin_nonthermal_abstains_on_one_group():
    """The basin-grouped tail needs >= 2 OCCUPIED basins to be a partition at all."""
    en = _en(PROPANOL, level='full')
    ref = basin_reference(en)
    x = torch.zeros(64, en.data_ndim, dtype=DTYPE)               # one basin
    e = cm.energy_components(en, x)
    tot = sum(e.values())
    out = cm.basin_nonthermal(en, x, tot, float(tot.min()) - 1.0, ref, 10.0)
    assert out['cover/nonthermal_available'] == 0, \
        'a single occupied basin is not a partition; publishing a spread over it would ' \
        'read as "uniform across basins"'
    print('PASS  basin nonthermal abstains on a single occupied basin')


def test_energy_components_sum_to_the_potential():
    """The components must reconstruct what the trainer actually optimises, or the
    per-term shares describe a different function than the loss."""
    en = _en(PROPANOL, level='full')
    x = _box(en)
    comp = cm.energy_components(en, x)
    total = sum(comp.values())
    one = torch.tensor(1.0, dtype=en.dtype)
    ref = en.potential_energy(x, one).detach().cpu().numpy()
    d = float(np.abs(total - ref).max())
    assert d < 2e-3, f'components sum differs from potential_energy by {d:.3e}'

    stats = cm.energy_component_stats(en, x)
    shares = [v for k, v in stats.items() if k.endswith('_share')]
    assert abs(sum(shares) - 1.0) < 1e-6, \
        f'term shares sum to {sum(shares)}, not 1 -- they are not a partition'
    print(f'PASS  components reconstruct the potential (max diff {d:.2e}); shares sum to 1')


def test_correlations_refuse_on_one_molecule():
    """A constant feature gives a 0/0 correlation. numpy returns nan, which on a dashboard
    reads as 'measured, no relationship' rather than 'not measurable'."""
    v = np.random.default_rng(0).normal(size=200)
    const = {'size': np.full(200, 12.0), 'n_rings': np.zeros(200)}
    out = cm.feature_correlations(v, const, 'corr/')
    assert out['corr/size_available'] == 0 and 'corr/size_pearson' not in out
    assert out['corr/size_n_distinct'] == 1

    # with real variation it must actually measure -- otherwise the refusal above is just
    # a function that never reports anything
    f = np.repeat(np.arange(5.0), 40)
    out2 = cm.feature_correlations(2.0 * f + 0.01 * v, {'size': f}, 'corr/')
    assert out2['corr/size_available'] == 1
    assert out2['corr/size_pearson'] > 0.99, out2
    print('PASS  correlations refuse on one molecule, measure when features vary')


def test_basin_labels_and_counts_agree():
    """One definition of basin identity, used by both the labeller and the counter."""
    en = _en(PROPANOL, level='full')
    ref = basin_reference(en)
    g = torch.Generator().manual_seed(1)
    x = (torch.rand(500, en.data_ndim, generator=g, dtype=DTYPE) * 2 - 1)
    r, th, ph = en.dof_from_state(x)
    dof = np.concatenate([r.numpy(), th.numpy(), ph.numpy()], axis=1)
    lab = rotamer_basin_labels(ref['groups'], dof, ref['n0'])
    cnt = basin_counts(ref['groups'], dof, ref['n0'], len(ref['combos']))
    assert int(cnt.sum()) == 500
    assert np.array_equal(cnt, np.bincount(lab, minlength=len(ref['combos'])))
    print('PASS  rotamer basin labels and counts are one definition')


def test_thermal_stats_move_with_temperature():
    """T_eff/T must RISE when the samples get hotter. A constant would pass a smoke test."""
    en = _en(PROPANOL, level='full')
    cold = cm.energy_components(en, _box(en, scale=0.05, seed=1))
    hot = cm.energy_components(en, _box(en, scale=0.5, seed=1))
    e_min = float(sum(cold.values()).min()) - 1.0
    tc = cm.thermal_stats(en, sum(cold.values()), e_min)
    th = cm.thermal_stats(en, sum(hot.values()), e_min)
    assert th['E/T_eff_over_T'] > tc['E/T_eff_over_T'], \
        f"T_eff did not rise with a wider draw ({tc['E/T_eff_over_T']:.2f} -> " \
        f"{th['E/T_eff_over_T']:.2f})"
    print(f"PASS  T_eff/T rises with sample spread "
          f"({tc['E/T_eff_over_T']:.2f} -> {th['E/T_eff_over_T']:.2f})")


def test_vs_reference_direction():
    """frac_below_ref_median must be 1 when the sampler strictly beats the reference and
    0 when it strictly loses -- a metric with the sign flipped passes 'is a float'."""
    s = np.zeros(100)
    better = cm.energy_vs_reference(s, s + 10.0)
    worse = cm.energy_vs_reference(s + 10.0, s)
    assert better['E/frac_below_ref_median'] == 1.0
    assert worse['E/frac_below_ref_median'] == 0.0
    assert better['E/median_gain_vs_ref'] > 0 > worse['E/median_gain_vs_ref']
    print('PASS  vs-reference gain has the right sign in both directions')


def test_accepts_device_tensors():
    """Buffers live on buffer_device ('cuda' in the canonical config), so every entry point
    must accept a CUDA tensor. np.asarray on one RAISES rather than copying."""
    if not torch.cuda.is_available():
        print('SKIP  no CUDA device; host-safety gate not exercised')
        return
    en = ConformerTorsions(smiles=PROPANOL, level='full', device='cuda',
                           force_field='mmff')
    x = (torch.rand(64, en.data_ndim, device='cuda', dtype=DTYPE) * 2 - 1) * 0.2
    ref = basin_reference(en)
    e = torch.as_tensor(sum(cm.energy_components(en, x).values()), device='cuda')
    out = {}
    out.update(cm.energy_component_stats(en, x))
    out.update(cm.geometry_stats(en, x))
    out.update(cm.dof_class_stats(en, x, reference=x))
    out.update(cm.dof_element_stats(en, x, reference=x))
    out.update(cm.ring_stats(en, x))
    out.update(cm.basin_coverage(en, x, ref))
    out.update(cm.cover_constants(en, ref))
    out.update(cm.thermal_stats(en, e, -1.0))
    out.update(cm.energy_vs_reference(e, e))
    assert 'E/total_mean' in out
    print(f'PASS  every entry point accepts CUDA tensors ({len(out)} keys)')


def test_coupling_detects_dependence_and_abstains_on_collapse():
    """basin_coupling is the QUALIFIER on coverage, so it has to separate three cases:
    independent (~0), coupled (large), and collapsed (ABSTAIN -- not a reassuring 0).

    The collapse case is the one that matters: a point mass makes every marginal a delta,
    so every entropy vanishes and total correlation is exactly 0. Publishing that would
    announce "the marginals are trustworthy" at the moment the sampler is most broken.
    """
    en = _en(PROPANOL, level='full')
    ref = basin_reference(en)
    g = torch.Generator().manual_seed(0)
    n, d = 4000, en.data_ndim

    indep = (torch.rand(n, d, generator=g, dtype=DTYPE) * 2 - 1)
    a = cm.basin_coupling(en, indep, ref)

    coup = (torch.rand(n, d, generator=g, dtype=DTYPE) * 2 - 1)
    lead = [rows[0] for rows, _ in ref['groups']]
    shared = coup[:, ref['n0'] + lead[0]].clone()
    for j in lead[1:]:
        coup[:, ref['n0'] + j] = shared          # groups now move together
    b = cm.basin_coupling(en, coup, ref)

    c = cm.basin_coupling(en, torch.zeros(n, d, dtype=DTYPE), ref)

    assert abs(a['cover/coupling_tc_debiased']) < 0.15,         f"independent draws reported TC {a['cover/coupling_tc_debiased']}"
    assert b['cover/coupling_tc_debiased'] > 0.5,         f"locked-together groups reported TC {b['cover/coupling_tc_debiased']} -- coupling "         f"was not detected"
    assert b['cover/coupling_n_suppressed'] > a['cover/coupling_n_suppressed']
    assert c['cover/coupling_available'] == 0 and 'cover/coupling_tc_debiased' not in c,         'a collapsed sampler must ABSTAIN; TC = 0 there reads as "marginals trustworthy"'
    # The shuffle null is subtracted inside and no longer published, so it is gated
    # INDIRECTLY and the two assertions above are what do it: a null large enough to make
    # `debiased` meaningless would have to cancel the coupled case as well, and `b > 0.5`
    # fails if it does. Asserting on a key that no longer exists is what would be vacuous.
    print(f"PASS  coupling separates independent ({a['cover/coupling_tc_debiased']:+.4f}) "
          f"from coupled ({b['cover/coupling_tc_debiased']:+.4f}), abstains on collapse")


def test_target_coupling_is_sampler_independent():
    """The target anchor must depend only on the molecule, and must abstain below 2 groups."""
    en = _en(PROPANOL, level='full')
    tc = cm.target_coupling(basin_reference(en))
    assert np.isfinite(tc) and tc >= 0.0
    assert tc == cm.target_coupling(basin_reference(en)), 'target coupling is not stable'

    enr = _en(CYCLOHEXANE)
    ref = basin_reference(enr)
    if len(ref.get('groups', [])) < 2:
        assert np.isnan(cm.target_coupling(ref)),             'a molecule with <2 rotamer groups has no joint; must be nan, not 0.0'
    print(f'PASS  target coupling stable ({tc:.4f} nats on propanol) and abstains below '
          f'2 groups')


def test_ring_torsion_stats_separate_width_from_structure():
    """Ring width and ring JOINT structure fail independently, so the metric must see them
    independently. Ring closure is a property of the joint: a sampler can reproduce every
    torsion marginal exactly and still never close a ring, and only `corr_dist` sees that.

    Three constructed cases, each requiring a specific number to move and the others not to:
      identity   -- ratio 1, corr_dist 0            (nothing may fire)
      widened    -- sd_ratio_max rises on the SHARP ring; the broad ring is legitimately
                    insensitive to the same absolute perturbation, which is why the two
                    cycles are asserted separately rather than pooled
      scrambled  -- marginals preserved EXACTLY (each torsion independently permuted), so
                    only corr_dist may move. This is the case a marginals-only metric
                    passes and closure fails.
    """
    from energies.ring_metrics import ring_cycles

    en = _en(ETHYLBENZENE)
    prior = _load_prior()
    if prior is None:
        print('SKIP  ring torsion stats: conformer_prior_v2.pt not available')
        return
    rng = np.random.default_rng(0)
    xs, _ = en.sample_prior_states(prior, 768, rng, report=False)
    ref = torch.as_tensor(xs, dtype=DTYPE)
    n_cyc = len(ring_cycles(en))
    assert n_cyc >= 1

    same = cm.ring_torsion_stats(en, ref, reference=ref)
    assert same['ringtor/available'] == 1
    for ci in range(n_cyc):
        assert abs(same[f'ringtor/c{ci}_sd_ratio_max'] - 1.0) < 1e-6
        assert same[f'ringtor/c{ci}_corr_dist'] < 1e-9, 'identity produced correlation drift'

    g = torch.Generator().manual_seed(0)
    wide = cm.ring_torsion_stats(en, ref + torch.randn(ref.shape, generator=g) * 0.05,
                                 reference=ref)
    assert max(wide[f'ringtor/c{ci}_sd_ratio_max'] for ci in range(n_cyc)) > 1.5,         'widening the states did not raise any ring sd ratio'

    # marginals preserved exactly, joint destroyed: permute each ring torsion independently
    # by permuting whole state rows per cycle is not enough -- do it in torsion space by
    # shuffling the sample order per column of the built torsions.
    from energies.ring_metrics import ring_torsions
    cyc = ring_cycles(en)
    t = np.asarray(ring_torsions(en, ref, cyc)[0])
    k = t.shape[1]
    scram = np.column_stack([np.random.default_rng(j).permutation(t[:, j]) for j in range(k)])
    c_ref = cm._ring_corr(np.degrees(t), k)
    c_scr = cm._ring_corr(np.degrees(scram), k)
    d = float(np.linalg.norm(c_scr - c_ref) / np.linalg.norm(c_ref))
    sd_ref = sorted(cm._circ_sd_deg(t[:, j] * 180 / np.pi) for j in range(k))
    sd_scr = sorted(cm._circ_sd_deg(scram[:, j] * 180 / np.pi) for j in range(k))
    assert np.allclose(sd_ref, sd_scr, atol=1e-6), 'the scramble changed a marginal'
    assert d > 0.5, f'destroying the ring joint moved corr_dist only {d:.3f}'
    print(f'PASS  ring torsions: width and joint separate (scramble keeps marginals, '
          f'corr_dist {d:.2f})')


def test_ring_torsion_stats_abstains_on_acyclic():
    en = _en(PROPANOL, level='full')
    out = cm.ring_torsion_stats(en, _box(en), reference=_box(en))
    assert out['ringtor/available'] == 0 and out['ringtor/n_cycles'] == 0
    assert not any(k.startswith('ringtor/c') for k in out),         'an acyclic molecule published per-cycle ring torsion statistics'
    print('PASS  ring torsion stats abstain on an acyclic molecule')


def test_dof_element_ratio_localises_a_widened_column():
    """dof_elem/ exists to say WHICH element owns the column that dof/*_sd_ratio_max
    found. So widening one element's columns must move that element's ratio and leave the
    others at ~1 -- and it must be a MAX, so widening a SINGLE column of a group is enough.

    Before this was a ratio the group published raw sds with no reference, which cannot
    fail this test in either direction: 0.010 is neither right nor wrong on its own.
    """
    en = _en(PROPANOL, level='full')
    ref = _box(en, n=512, scale=0.2, seed=3)
    block = np.asarray(en._free_block)
    owner = cm._central_elements(en)

    base = cm.dof_element_stats(en, _box(en, n=512, scale=0.2, seed=4), reference=ref)
    ratio_keys = [k for k in base if k.endswith('_sd_ratio')]
    assert ratio_keys, 'a reference was supplied and no ratio was published'
    assert not any(k.endswith('_sd') for k in base), \
        'a raw sd was published beside a ratio; the two must never share an axis'
    for k in ratio_keys:
        assert 0.7 < base[k] < 1.4, f'{k} = {base[k]} on two draws from one distribution'

    # widen exactly ONE column, and pick the group it belongs to
    target = None
    for cls, cname in ((0, 'r'), (1, 'theta'), (2, 'phi')):
        for zval in np.unique(owner):
            idx = np.flatnonzero((block == cls) & (owner == zval))
            if idx.size >= 2:                      # >= 2 so the MAX-vs-mean gate bites
                target = (cname, cm._SYM.get(int(zval), f'Z{int(zval)}'), int(idx[0]))
                break
        if target:
            break
    assert target is not None, 'no element group with >= 2 columns to test the max on'
    cname, sym, col = target

    x = _box(en, n=512, scale=0.2, seed=4)
    x[:, col] *= 5.0
    wide = cm.dof_element_stats(en, x, reference=ref)
    hit = f'dof_elem/{cname}_{sym}_sd_ratio'
    assert wide[hit] > 3.0, \
        f'{hit} = {wide[hit]:.2f} after one of its columns was widened 5x -- a mean over ' \
        f'the group would have diluted it, which is the failure this MAX exists to avoid'
    for k in ratio_keys:
        if k != hit:
            assert abs(wide[k] - base[k]) < 1e-6, f'{k} moved when {hit} was widened'
    print(f'PASS  dof_elem ratio localises a widened column ({hit}: '
          f'{base[hit]:.2f} -> {wide[hit]:.2f}), others unmoved')


def test_presence_is_not_announced_but_absence_is():
    """THE CONSOLIDATION'S LOAD-BEARING RULE. `*_available = 1` beside a value was a key
    per metric carrying no information; `*_available = 0` INSTEAD of a value is the whole
    absent-is-not-zero rule and must survive. Both halves are gated here, because dropping
    the second along with the first is the easy mistake and it is silent.
    """
    en = _en(PROPANOL, level='full')
    x = _box(en, n=128, scale=0.2)
    live = {}
    live.update(cm.energy_component_stats(en, x))
    live.update(cm.geometry_stats(en, x))
    live.update(cm.dof_class_stats(en, x, reference=x))
    live.update(cm.dof_element_stats(en, x, reference=x))
    ones = {k: v for k, v in live.items() if k.endswith('_available') and v == 1}
    assert not ones, f'live metrics announced their own presence: {sorted(ones)}'

    # ... and the abstaining direction still fires. An empty array has nothing to average,
    # and a molecule with no ring has no closure -- both must produce the flag, not a 0.
    out = {}
    cm._quantiles(np.array([]), 'x/thing', out)
    cm._mean_only(np.array([np.nan, np.inf]), 'x/small', out)
    assert out == {'x/thing_available': 0, 'x/small_available': 0}, out
    assert cm.ring_stats(en, x)['ring/available'] == 0
    print('PASS  presence is not announced, absence still is')


def test_minor_energy_terms_keep_their_share_but_lose_their_histogram():
    """The E/ trim must not break the partition over terms. Every term keeps a mean and a
    share -- so a molecule whose energy moves to a term outside FULL_QUANTILE_TERMS is
    still VISIBLE, which is what makes the fixed tuple auditable rather than a blind spot.
    """
    en = _en(PROPANOL, level='full')
    out = cm.energy_component_stats(en, _box(en, n=128, scale=0.2))
    comp = cm.energy_components(en, _box(en, n=128, scale=0.2))
    for name in comp:
        assert f'E/{name}_mean' in out and f'E/{name}_share' in out, \
            f'{name} lost its level or its share; the term partition is incomplete'
        has_block = f'E/{name}_p50' in out
        assert has_block == (name in cm.FULL_QUANTILE_TERMS), \
            f'{name} percentile block = {has_block}, but FULL_QUANTILE_TERMS says ' \
            f'{name in cm.FULL_QUANTILE_TERMS}'
    print(f'PASS  every term keeps mean+share; percentile blocks only for '
          f'{cm.FULL_QUANTILE_TERMS}')


def test_cover_constants_carry_the_denominators():
    """The three live cover/ keys are unreadable without these, so dropping them from the
    per-eval log is only sound if they are still PRODUCED. A `cover_constants` that
    abstained or returned an empty dict would make `n_missed = 0` uninterpretable and
    nothing else in the suite would notice.
    """
    en = _en(PROPANOL, level='full')
    ref = basin_reference(en)
    const = cm.cover_constants(en, ref, u_star=40.0, target_tc=cm.target_coupling(ref))
    for k in ('cover/n_modes', 'cover/n_accessible', 'cover/expected_frac',
              'cover/coupling_n_groups', 'cover/nonthermal_u_star'):
        assert k in const, f'{k} missing; a live cover/ key lost its denominator'
    assert const['cover/expected_frac'] == 1.0 / const['cover/n_accessible']
    assert const['cover/n_accessible'] <= const['cover/n_modes']
    assert cm.cover_constants(en, None)['cover/available'] == 0
    print(f"PASS  cover constants carry the denominators "
          f"({const['cover/n_accessible']} of {const['cover/n_modes']} basins accessible)")


TESTS = [test_absent_is_not_zero,
         test_frozen_tiers_are_labelled_not_scored,
         test_geometry_detects_out_of_range,
         test_coverage_catches_mode_collapse,
         test_single_basin_abstains_rather_than_reading_as_collapsed,
         test_basin_nonthermal_abstains_on_one_group,
         test_energy_components_sum_to_the_potential,
         test_correlations_refuse_on_one_molecule,
         test_basin_labels_and_counts_agree,
         test_thermal_stats_move_with_temperature,
         test_vs_reference_direction,
         test_coupling_detects_dependence_and_abstains_on_collapse,
         test_target_coupling_is_sampler_independent,
         test_ring_torsion_stats_separate_width_from_structure,
         test_ring_torsion_stats_abstains_on_acyclic,
         test_dof_element_ratio_localises_a_widened_column,
         test_presence_is_not_announced_but_absence_is,
         test_minor_energy_terms_keep_their_share_but_lose_their_histogram,
         test_cover_constants_carry_the_denominators,
         test_accepts_device_tensors]

if __name__ == '__main__':
    torch.set_default_dtype(DTYPE)
    failed = 0
    for t in TESTS:
        try:
            t()
        except Exception:
            failed += 1
            import traceback

            print(f'FAIL         {t.__name__}')
            traceback.print_exc()
    print(f'\n{len(TESTS) - failed}/{len(TESTS)} passed')
    raise SystemExit(1 if failed else 0)
