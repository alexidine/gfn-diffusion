"""eval/conformer_model_eval.py: the offline per-condition eval of a trained conformer GFN.

FAST (numbers and single member charts, no model):
  * the Zhang-Stephens GPD fit recovers a known shape, as scipy's maximum likelihood does,
    and equals pyro's PSIS estimator (prior included) to rounding, pinned and live;
  * PSIS k-hat separates heavy-tailed importance weights from light-tailed ones, and reads
    equal weights as having no tail;
  * the exact bar FAILS a far estimate through `judge`, and applies the floors and the
    convergence rule first; the ESS floor fails a row without an exact value too, and a row
    whose IS status is not ok has its gaps blanked;
  * excess is (E - e_min)/T; the summary names a tie instead of one condition;
  * a COLLAPSED sampler (every draw in the reference's rotamer basin) reports missed basins
    and fails the coverage bar; draws spread over the phi columns miss none and pass; the
    same holds on the marginal fallback where the product enumeration was skipped, and a
    skipped table without draws is UNAVAILABLE;
  * draws on one side of a free non-planar centre (NH3's pyramid, which the lock leaves free)
    miss it and fail the parity bar, half of them reflected pass; a free centre whose mirror
    is out of reach (DABCO's cage N) is not missed; a locked centre and a planar one are not
    free centres;
  * a stereo-lock-violating draw (the mirror image of a locked centre) is counted, active and
    inverted, and a molecule without a lock is n/a, not a pass;
  * the clash count reads the pair overlap against its threshold;
  * conditions of one constitution are classified mirror / same isomer / diastereomer.

SLOW (the real modeller path, on the width-9 carrier set {NH3, H2CO, CH4} at level `full`
that tests/conformer/test_logz_check.py uses, with its zero-output untrained proposal):
  * the eval's rollouts are conformer_logz_check.policy_rollouts' draws, row for row;
  * IS log Z of an untrained policy agrees with the two-grid quadrature on NH3 and H2CO;
  * the loader refuses a conditions file whose stereo coefficient, carrier layout or
    condition count does not match, and a config whose problem definition does not; it
    builds a checkpoint stamped for 'cuda' on the CPU, draws the prior dataset at its
    minimum with the trainer's registry, and names the identity-exempt energy keys;
  * the CLI on a saved checkpoint writes labelled tables and a JSON, and its exit status
    follows the verdicts;
  * a subset eval builds reference entries for that subset only, and reuses them.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/conformer/test_conformer_model_eval.py
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import eval.conformer_logz_check as lzc
import eval.conformer_model_eval as me

HERE = Path(__file__).resolve().parents[2]            # energy_sampling/
BASE_CONFIG = HERE / 'configs' / 'conformer_mk_multi.yaml'
SMIS = ['N', 'C=O', 'C']                              # full/mmff: carrier K = 9 (4|3|2)
MOL_DIM, ENC = 16, 8


# ------------------------------------------------------------------ fast: statistics

@pytest.mark.fast
@pytest.mark.parametrize('c', [0.2, 0.5, 0.9])
def test_gpd_fit_recovers_a_known_shape(c):
    from scipy.stats import genpareto
    x = genpareto(c=c, scale=1.0).rvs(size=4000, random_state=np.random.default_rng(1))
    k, sigma = me.gpd_fit(x)
    mle = genpareto.fit(x, floc=0)[0]
    assert abs(k - c) < 0.08, (k, c)
    assert abs(k - mle) < 0.03, (k, mle)
    assert sigma == pytest.approx(1.0, rel=0.1)


#: (k0, n) -> (k, sigma) from pyro-ppl 1.9.1's pyro.ops.stats.fit_generalized_pareto on the
#: GPD(k0, 1) quantiles at (i - 0.5)/n, i = 1..n (computed 2026-09-27; `_gpd_quantiles`)
PYRO_GPD = {(0.3, 200): (0.3150712424607639, 0.9919964876956558),
            (0.8, 136): (0.775564838475384, 0.9995810403676033),
            (-0.2, 60): (-0.06132078263794294, 0.9538327165867492)}


def _gpd_quantiles(k0, n):
    p = (np.arange(1, n + 1) - 0.5) / n
    return ((1.0 - p) ** (-k0) - 1.0) / k0


@pytest.mark.fast
@pytest.mark.parametrize('k0,n', sorted(PYRO_GPD))
def test_gpd_fit_is_the_psis_estimator_to_rounding(k0, n):
    """The same numbers as an independent implementation, prior included: pyro's
    fit_generalized_pareto (Zhang-Stephens, then (n k + 5)/(n + 10)), pinned so the check
    holds without pyro, and re-run live where pyro is installed."""
    k, sigma = me.gpd_fit(_gpd_quantiles(k0, n))
    assert k == pytest.approx(PYRO_GPD[(k0, n)][0], abs=1e-10)
    assert sigma == pytest.approx(PYRO_GPD[(k0, n)][1], abs=1e-10)


@pytest.mark.fast
def test_pareto_khat_matches_pyro_on_the_psis_tail():
    stats = pytest.importorskip('pyro.ops.stats')
    rng = np.random.default_rng(11)
    for lw in (rng.normal(0.0, 2.0, 2048), -1.5 * np.log(rng.random(5000))):
        got = me.pareto_khat(lw)
        m = math.ceil(min(0.2 * lw.size, 3 * math.sqrt(lw.size)))
        w = np.sort(np.exp(lw - lw.max()))
        k, _ = stats.fit_generalized_pareto(torch.as_tensor(w[-m:] - w[-m - 1]))
        assert got['k'] == pytest.approx(k, abs=1e-10), (got, k)


@pytest.mark.fast
def test_pareto_k_separates_heavy_from_light_tailed_weights():
    """w = U^(-1) has tail index 1 (infinite variance, infinite mean at the edge); log w
    Gaussian with sd 0.2 has an essentially light tail."""
    rng = np.random.default_rng(0)
    heavy = me.pareto_khat(-np.log(rng.random(20000)))
    light = me.pareto_khat(rng.normal(0.0, 0.2, 20000))
    assert heavy['m_tail'] == light['m_tail'] == math.ceil(3 * math.sqrt(20000))
    assert heavy['k'] > 0.8, heavy
    assert light['k'] < 0.3, light
    assert me.judge({'tb': {'pareto_k': heavy['k']}}, me.Bars())['pareto_k']['status'] == me.FAIL
    assert me.judge({'tb': {'pareto_k': light['k']}}, me.Bars())['pareto_k']['status'] == me.PASS
    flat = me.pareto_khat(np.full(500, 3.0))
    assert flat['k'] == float('-inf') and 'zero' in flat['note']
    assert me.judge({'tb': {'pareto_k': flat['k']}}, me.Bars())['pareto_k']['status'] == me.PASS


@pytest.mark.fast
def test_pareto_k_fits_the_largest_weights_over_the_next_one():
    """The tail is the M largest weights as exceedances over the (M+1)-th largest, M =
    ceil(min(0.2 N, 3 sqrt N)): a light bulk with a heavy top reads heavy, and the value is
    gpd_fit on exactly those exceedances."""
    rng = np.random.default_rng(4)
    bulk = rng.normal(0.0, 0.2, 19000)
    top = 1.0 - np.log(rng.random(1000))                 # log of a tail-index-1 Pareto, above
    lw = rng.permutation(np.concatenate([bulk, top]))
    got = me.pareto_khat(lw)
    m = math.ceil(3 * math.sqrt(lw.size))
    w = np.sort(np.exp(lw - lw.max()))
    assert got['k'] == pytest.approx(me.gpd_fit(w[-m:] - w[-m - 1])[0], abs=1e-12)
    assert got['k'] > 0.8, got
    assert me.pareto_khat(bulk)['k'] < 0.3


@pytest.mark.fast
def test_the_log_weight_block_is_is_summary_plus_a_bootstrap_se():
    rng = np.random.default_rng(3)
    lw = torch.as_tensor(rng.normal(-4.0, 0.5, 4000))
    b = me.logw_block(lw, n_boot=400, seed=0)
    s = lzc.is_summary(lw)
    assert b['log_z'] == pytest.approx(s['log_z']) and b['se_delta'] == pytest.approx(s['se'])
    assert b['se'] == b['se_boot'] and b['se_boot'] == pytest.approx(s['se'], rel=0.2)
    assert b['kl_gap'] == pytest.approx(s['log_z'] - s['mean_log_w'])
    assert b['kl_gap'] == pytest.approx(0.5 ** 2 / 2, abs=0.03)     # lognormal: sigma^2 / 2
    assert b['std_log_w'] == pytest.approx(0.5, abs=0.02)


def _tb(log_z=-10.0, se=0.02, ess=500.0, ess_frac=0.25, k=0.3, nonfinite=0):
    return dict(log_z=log_z, se=se, ess=ess, ess_frac=ess_frac, n_nonfinite=nonfinite,
                pareto_k=k, kl_gap=0.4)


@pytest.mark.fast
def test_the_exact_bar_fails_a_far_estimate_and_applies_the_floors_first():
    """judge runs conformer_logz_check's exact_verdict: FAIL beyond max(3 SE + |A-B|, 0.1),
    PASS inside it, and a floor or an unconverged quadrature fails whatever the distance."""
    bars = me.Bars()
    near = {'log_z': -10.05, 'diff': 0.001, 'converged': True}
    far = {'log_z': -10.7, 'diff': 0.001, 'converged': True}
    v = me.judge({'tb': _tb(), 'exact': near}, bars)['exact_logz']
    assert v['status'] == me.PASS and v['bar_value'] == pytest.approx(0.1)
    v = me.judge({'tb': _tb(), 'exact': far}, bars)['exact_logz']
    assert v['status'] == me.FAIL and v['note'] == 'FAIL' and v['value'] == pytest.approx(0.7)
    v = me.judge({'tb': _tb(ess=50.0), 'exact': near}, bars)['exact_logz']
    assert v['status'] == me.FAIL and v['note'] == 'BELOW_ESS_FLOOR'
    v = me.judge({'tb': _tb(), 'exact': dict(near, converged=False)}, bars)['exact_logz']
    assert v['status'] == me.FAIL and v['note'] == 'UNCONVERGED'
    assert me.judge({'tb': _tb(), 'exact': None, 'exact_na': 'k = 9 > 6'},
                    bars)['exact_logz']['status'] == me.N_A


@pytest.mark.fast
def test_the_ess_floor_holds_without_an_exact_value_and_blanks_the_gaps():
    """A row with no quadrature is still refused below the floors (conformer_logz_check's
    rule), and its IS status then blanks every difference taken from the estimate; a k-hat
    above the bar does the same past the floors."""
    bars = me.Bars()
    ok, low, heavy = _tb(), _tb(ess=14.3, ess_frac=0.007), _tb(k=1.19)
    assert me.judge({'tb': ok}, bars)['ess_floor']['status'] == me.PASS
    v = me.judge({'tb': low}, bars)['ess_floor']
    assert v['status'] == me.FAIL and v['note'] == 'BELOW_ESS_FLOOR'
    assert me.judge({'tb': _tb(nonfinite=3)}, bars)['ess_floor']['note'] == 'NONFINITE_ROWS'
    assert me.is_status(ok, bars) == 'ok'
    assert me.is_status(low, bars) == 'BELOW_ESS_FLOOR'
    assert me.is_status(heavy, bars) == me.PARETO_K_ABOVE_BAR
    assert me.judge({'tb': heavy}, bars)['ess_floor']['status'] == me.PASS
    assert me._gap({'is_status': 'ok'}, 1.5) == 1.5
    for status in ('BELOW_ESS_FLOOR', me.PARETO_K_ABOVE_BAR):
        assert math.isnan(me._gap({'is_status': status}, 1.5))


@pytest.mark.fast
def test_excess_is_energy_over_the_floor_in_kt():
    e = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, float('nan'), 0.5])
    s = me.excess_stats(e, e_min=1.0, temperature=2.0)
    u = (np.array([2.0, 3.0, 4.0, 5.0, 6.0, 0.5]) - 1.0) / 2.0
    assert s['p50'] == pytest.approx(np.percentile(u, 50)) and s['max'] == pytest.approx(2.5)
    assert s['p50'] == pytest.approx(1.25) and s['p90'] == pytest.approx(np.percentile(u, 90))
    assert s['p10'] == pytest.approx(0.125)
    assert s['n_below_floor'] == 1 and s['e_min'] == 1.0
    assert me.excess_stats(e, None, 1.0)['na'] == 'no floor'


@pytest.mark.fast
def test_the_summary_names_a_tie_rather_than_one_condition():
    rows = [dict(condition=c, v=v) for c, v in (('A', 0.0), ('B', 0.0), ('C', 0.0))]
    assert me._median_worst(rows, lambda r: r['v'], 'max') == (0.0, 0.0, 'all tied (3)')
    rows[2]['v'] = -1.0
    assert me._median_worst(rows, lambda r: r['v'], 'max')[2] == 'tie: A, B'
    assert me._median_worst(rows, lambda r: r['v'], 'min')[1:] == (-1.0, 'C')


# ------------------------------------------------------------------ fast: coverage

@pytest.fixture(scope='module')
def ethanol():
    from energies.conformer_torsions import ConformerTorsions
    from energies.prior_diagnostics import basin_reference
    en = ConformerTorsions(smiles='CCO', device='cpu', level='full', force_field='mmff',
                           dtype=torch.float64)
    return en, basin_reference(en)


def _coverage_verdict(en, br, x):
    from energies.conformer_eval_metrics import per_molecule_block
    row = per_molecule_block({'CCO': en}, {'CCO': x}, None, {'CCO': {'basin_ref': br}},
                             n_min=32)['CCO']
    cov = me.coverage_block(row, {'basin_ref': br})
    return cov, me.judge({'coverage': cov}, me.Bars())['coverage']


@pytest.mark.fast
def test_a_collapsed_sampler_misses_basins_and_fails_the_coverage_bar(ethanol):
    en, br = ethanol
    n_acc = int(np.asarray(br['accessible']).sum())
    assert n_acc >= 2, 'the test needs more than one accessible basin'
    g = torch.Generator().manual_seed(0)
    collapsed = 0.01 * torch.randn(512, en.ndim, generator=g, dtype=torch.float64)
    cov, v = _coverage_verdict(en, br, collapsed)
    assert cov['n_missed'] == n_acc - 1 and cov['n_visited'] == 1
    assert v['status'] == me.FAIL and v['value'] == n_acc - 1
    # the control: the same small r/theta noise, every phi column uniform on its circle
    spread = collapsed.clone()
    phi = torch.as_tensor(np.asarray(en.periodic_dims, dtype=bool))
    spread[:, phi] = torch.rand(512, int(phi.sum()), generator=g, dtype=torch.float64) * 2 - 1
    cov, v = _coverage_verdict(en, br, spread)
    assert cov['n_missed'] == 0 and cov['n_visited'] == n_acc and v['status'] == me.PASS


@pytest.mark.fast
def test_coverage_without_a_reference_or_a_skipped_table_without_draws_is_unavailable():
    v = me.judge({'coverage': me.coverage_block({}, {})}, me.Bars())['coverage']
    assert v['status'] == me.UNAVAILABLE
    skipped = {'basin_ref': {'skipped': '1000 modes exceeds max_modes=512'}}
    v = me.judge({'coverage': me.coverage_block({}, skipped)}, me.Bars())['coverage']
    assert v['status'] == me.UNAVAILABLE and 'skipped' in v['note']
    assert me.overall({'coverage': dict(status=me.UNAVAILABLE)}).startswith(me.FAIL)


@pytest.mark.fast
def test_a_skipped_enumeration_reads_marginal_coverage(ethanol):
    """The table skipped the product enumeration: the bar reads each group's rotamer centres.
    Collapsed draws miss every centre but the reference's in each group and fail; draws
    uniform on every phi column visit them all and pass. The centres are basin_reference's
    own, so a group's centre count is the product table's factor for that group."""
    en, br = ethanol
    skipped = {'basin_ref': {'skipped': '9 modes exceeds max_modes=4'}}
    sizes = [len(c) for _, c in br['groups']]
    g = torch.Generator().manual_seed(0)
    collapsed = 0.01 * torch.randn(512, en.ndim, generator=g, dtype=torch.float64)
    cov = me.coverage_block({}, skipped, en, collapsed)
    assert cov['kind'] == 'marginal' and cov['n_accessible'] == sum(sizes)
    assert cov['n_missed'] == sum(sizes) - len(sizes) and cov['worst_over_uniform'] == 0.0
    v = me.judge({'coverage': cov}, me.Bars())['coverage']
    assert v['status'] == me.FAIL and 'marginal' in v['bar'] and 'skipped' in v['note']
    spread = collapsed.clone()
    phi = torch.as_tensor(np.asarray(en.periodic_dims, dtype=bool))
    spread[:, phi] = torch.rand(512, int(phi.sum()), generator=g, dtype=torch.float64) * 2 - 1
    cov = me.coverage_block({}, skipped, en, spread)
    assert cov['n_missed'] == 0 and cov['n_visited'] == sum(sizes)
    assert me.judge({'coverage': cov}, me.Bars())['coverage']['status'] == me.PASS


def _mirror(en, x):
    r, th, ph = en.dof_from_state(x)
    return en.state_from_dof(r, th, -ph)


@pytest.mark.fast
def test_draws_on_one_side_of_a_free_centre_miss_it():
    """NH3 at full: the lock leaves the three-coordinate N free, and the inverted pyramid is
    the reference's mirror image at equal energy. Draws on the reference pyramid alone miss
    it; half of them reflected do not. NH3 has no rotor, so its rotamer coverage is one
    basin whatever the draws."""
    from energies.conformer_torsions import ConformerTorsions
    nh3 = ConformerTorsions(smiles='N', device='cpu', level='full', force_field='mmff',
                            dtype=torch.float64, stereo_coeff=300.0)
    g = torch.Generator().manual_seed(0)
    x = 0.01 * torch.randn(256, nh3.ndim, generator=g, dtype=torch.float64)
    one = me.parity_coverage(nh3, x)
    assert one['n_centres'] == one['n_accessible'] == one['n_missed'] == 1
    assert one['centres'][0]['name'] == 'N0' and one['minority_frac'] == 0.0
    assert one['centres'][0]['mirror_de_kt'] == pytest.approx(0.0, abs=1e-6)
    assert me.judge({'parity': one}, me.Bars())['parity']['status'] == me.FAIL
    both = me.parity_coverage(nh3, torch.cat([x[:128], _mirror(nh3, x[128:])]))
    assert both['n_missed'] == 0 and both['minority_frac'] == pytest.approx(0.5)
    assert me.judge({'parity': both}, me.Bars())['parity']['status'] == me.PASS


@pytest.mark.fast
def test_a_free_centre_whose_mirror_is_out_of_reach_is_not_missed():
    """DABCO: the lock leaves the three-coordinate cage N free, but reflecting it at the
    reference geometry breaks the cage (about 1e5 kT), so the inverted side is not
    accessible and draws on one side miss nothing. The same draws are missed once the
    accessibility cut is raised past that energy."""
    from energies.conformer_torsions import ConformerTorsions
    dabco = ConformerTorsions(smiles='C1CN2CCN1CC2', device='cpu', level='full',
                              force_field='mmff', dtype=torch.float64, stereo_coeff=300.0)
    x = torch.zeros(16, dabco.ndim, dtype=torch.float64)
    p = me.parity_coverage(dabco, x)
    assert p['n_centres'] == 1 and p['n_accessible'] == 0 and p['n_missed'] == 0
    assert p['centres'][0]['mirror_de_kt'] > 1e3 and p['minority_frac'] is None
    assert me.judge({'parity': p}, me.Bars())['parity']['status'] == me.PASS
    wide = me.parity_coverage(dabco, x, accessible_kt=1e9)
    assert wide['n_accessible'] == wide['n_missed'] == 1


@pytest.mark.fast
def test_a_centre_the_lock_names_is_not_free_and_planar_is_not_a_centre(methane_locked):
    from energies.conformer_torsions import ConformerTorsions
    x = torch.zeros(4, methane_locked.ndim, dtype=torch.float64)
    locked = me.parity_coverage(methane_locked, x)
    assert locked['n_centres'] == 0 and 'lock' in locked['na']
    assert me.judge({'parity': locked}, me.Bars())['parity']['status'] == me.N_A
    free = ConformerTorsions(smiles='C', device='cpu', level='full', force_field='mmff',
                             dtype=torch.float64)
    assert me.parity_coverage(free, x)['n_missed'] == 1, 'unlocked, CH4 holds both parities'
    h2co = ConformerTorsions(smiles='C=O', device='cpu', level='full', force_field='mmff',
                             dtype=torch.float64)
    planar = me.parity_coverage(h2co, torch.zeros(4, h2co.ndim, dtype=torch.float64))
    assert planar['n_centres'] == 0


# ------------------------------------------------------------------ fast: lock and clash

@pytest.fixture(scope='module')
def methane_locked():
    from energies.conformer_torsions import ConformerTorsions
    return ConformerTorsions(smiles='C', device='cpu', level='full', force_field='mmff',
                             dtype=torch.float64, stereo_coeff=300.0)


@pytest.mark.fast
def test_a_lock_violating_sample_is_counted(methane_locked):
    """The reference and its mirror image (every dihedral negated): the mirror puts the
    locked centre on the wrong side, so it is both lock-active and inverted."""
    en = methane_locked
    assert en.stereo.n == 1
    x0 = torch.zeros(1, en.ndim, dtype=torch.float64)
    r, th, ph = en.dof_from_state(x0)
    mirror = en.state_from_dof(r, th, -ph)
    s = me.stereo_lock_stats(en, torch.cat([x0, x0, x0, mirror]))
    assert s['n_active'] == 1 and s['n_inverted'] == 1
    assert s['active_frac'] == pytest.approx(0.25) and s['lock_max'] > 100.0
    assert me.judge({'lock': s}, me.Bars())['lock']['status'] == me.FAIL
    clean = me.stereo_lock_stats(en, x0.repeat(4, 1))
    assert clean['n_active'] == 0
    assert me.judge({'lock': clean}, me.Bars())['lock']['status'] == me.PASS


@pytest.mark.fast
def test_no_lock_is_na_not_a_pass():
    from energies.conformer_torsions import ConformerTorsions
    nh3 = ConformerTorsions(smiles='N', device='cpu', level='full', force_field='mmff',
                            dtype=torch.float64, stereo_coeff=300.0)
    s = me.stereo_lock_stats(nh3, torch.zeros(3, nh3.ndim, dtype=torch.float64))
    assert s['na'] == 'no locked element'
    assert me.judge({'lock': s}, me.Bars())['lock']['status'] == me.N_A


@pytest.mark.fast
def test_the_clash_count_reads_the_overlap_against_its_threshold(ethanol):
    en, _ = ethanol
    g = torch.Generator().manual_seed(5)
    x = (torch.rand(256, en.ndim, generator=g, dtype=torch.float64) * 2 - 1)
    from energies.prior_smoke import _worst_overlap
    _, ff = en._batch(256)
    worst = _worst_overlap(en.build_positions(x), ff, 256).numpy()
    for bar in (0.0, float(np.median(worst)), 0.5):
        c = me.clash_stats(en, x, bar)
        assert c['n_clash'] == int((worst > bar).sum()) and c['n_pairs'] == 15
    assert me.clash_stats(en, torch.zeros(1, en.ndim, dtype=torch.float64), 0.5)['n_clash'] == 0


# ------------------------------------------------------------------ fast: pairs

@pytest.mark.fast
def test_conditions_of_one_constitution_are_classified():
    from rdkit import Chem
    a = 'C[C@H](O)CC'
    same = Chem.MolToSmiles(Chem.MolFromSmiles(a))
    assert same != a
    pairs = me.stereo_pairs({'a': a, 'b': 'C[C@@H](O)CC', 'c': same,
                             'meso': 'C[C@H](O)[C@@H](C)O', 'rr': 'C[C@H](O)[C@H](C)O',
                             'x': 'CCO'})
    kinds = {(p['a'], p['b']): p['kind'] for p in pairs}
    assert kinds == {('a', 'b'): 'mirror', ('a', 'c'): 'same isomer', ('b', 'c'): 'mirror',
                     ('meso', 'rr'): 'diastereomer'}


# ------------------------------------------------------------------ slow, the real path

def _prior_path():
    p = Path(os.environ.get('GFN_CONFORMER_PRIOR', HERE / 'conformer_prior_v2.pt'))
    if not p.exists():
        pytest.fail(f'fitted InternalPrior not found at {p}. Set GFN_CONFORMER_PRIOR.')
    return p.as_posix()


def _write_conditions(path, smis, stereo_coeff=0.0):
    """Carrier-padded condition graphs as build_conformer_conditions.py --carrier writes them,
    random frozen embeddings in place of the encoder's (test_logz_check's construction)."""
    from energies.conformer_carrier import carrier_pad_condition
    from energies.conformer_data import (collate_conditions, condition_from_energy,
                                         save_condition_file)
    from energies.dof_features import free_dof_atom_index
    from energies.multi_conformer import MultiConformerTorsions

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        en = MultiConformerTorsions(smis, identifiers=smis, device='cpu', level='full',
                                    force_field='mmff', stereo_coeff=stereo_coeff)
        g = torch.Generator().manual_seed(0)
        rows = []
        for ident, mem in en._members.items():
            c = condition_from_energy(mem, identifier=ident)
            a, msk = free_dof_atom_index(mem)
            cc = carrier_pad_condition(c, en.carrier, ident, mem, atoms=a, mask=msk,
                                       R=int(a.shape[1]))
            cc.atom_embedding = torch.randn(int(cc.num_nodes), ENC, generator=g)
            cc.embedding = torch.randn(1, MOL_DIM, generator=g)
            rows.append(cc)
        save_condition_file(collate_conditions(rows), str(path))
    finally:
        torch.set_default_dtype(old)


def _write_config(tmp, name, cond, **energy):
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding='utf-8'))
    cfg.update(run_name='me', checkpoints_dir=tmp.as_posix(), molecules_path=cond.as_posix(),
               test_molecules_path=None, embedding_conditioning_dim=MOL_DIM, batch_size=64,
               max_batch_size=64, grow_batch_size=False, archive_period=0, eval_T=10)
    cfg['integrator']['T'] = 10
    for k in ('s_emb_dim', 't_hidden_dim', 's_hidden_dim', 'policy_hidden_dim',
              'flow_hidden_dim', 'cond_hidden_dim', 't_dim', 'harmonics_dim'):
        cfg['model'][k] = 32
    for k in ('s_layers', 'policy_layers', 'flow_layers', 'cond_layers'):
        cfg['model'][k] = 2
    cfg['model'].update(zero_init=True, t_scale=0.1, policy_kind='set', set_policy_hidden=32,
                        set_policy_layers=2, set_policy_corr_dim=8, dplr_rank=0)
    cfg['energy_config'].update(internal_prior_path=_prior_path(), prior_sample_size=60,
                                **energy)
    path = tmp / f'{name}.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    return path


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    """(dir, config, untrained modeller, checkpoint path, cuda-stamped checkpoint path)."""
    old_dtype, old_wandb = torch.get_default_dtype(), os.environ.get('WANDB_MODE')
    torch.set_default_dtype(torch.float32)
    os.environ['WANDB_MODE'] = 'disabled'
    try:
        tmp = tmp_path_factory.mktemp('model_eval')
        _write_conditions(tmp / 'cond.pt', SMIS)
        cfg = _write_config(tmp, 'me', tmp / 'cond.pt')
        m = me.init_eval_modeller(me.build_eval_modeller(str(cfg), None))
        out = m.ema_model.forward_policy.rho.output_layer
        assert out.bias is None and bool((out.weight == 0).all()), \
            'model.zero_init did not reach the set head: the untrained proposal is random'
        # a checkpoint: tracker rows and a step, EMA weights moved so a load is visible
        m.init_condition_log_z()
        reg = m.identifier_registry
        m.condition_log_z.ema_logw[reg['N']] = -12.5
        m.condition_log_z.count[reg['N']] = 40
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for p in m.ema_model.parameters():
                p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
        m.step_ind = 77
        m.args.checkpoint_read_only = False
        m.checkpointer.save('probe')
        m.args.checkpoint_read_only = True
        ck = Path(m.checkpointer.path_for('probe'))
        blob = torch.load(ck, map_location='cpu', weights_only=False)
        blob['gfn_config']['device'] = 'cuda'
        ck_cuda = tmp / 'probe_cuda.pt'
        torch.save(blob, ck_cuda)
        # the untrained modeller again, EMA as built, for the IS test
        fresh = me.init_eval_modeller(me.build_eval_modeller(str(cfg), None))
        yield tmp, cfg, fresh, ck, ck_cuda
    finally:
        torch.set_default_dtype(old_dtype)
        if old_wandb is None:
            os.environ.pop('WANDB_MODE', None)
        else:
            os.environ['WANDB_MODE'] = old_wandb


@pytest.mark.slow
def test_the_eval_rollouts_are_the_logz_checks(tiny):
    """Same seed, steps and batch: the same draws, so the IS here is conformer_logz_check's."""
    _, _, m, _, _ = tiny
    rows = lzc._condition_rows(m)
    ds, row = rows['C=O']
    ours = me.rollout_condition(m, 'C=O', ds, row, n=300, batch=150, seed=2,
                                steps=m.args.integrator.T)
    theirs = lzc.policy_rollouts(m, 'C=O', ds, row, n=300, batch=150, seed=2)
    # the same draws; lzc sums its log w in float32, this eval in float64 from float32 parts
    assert torch.allclose(ours['log_w'].double(), theirs['log_w'].double(), rtol=0, atol=1e-4)
    assert ours['condition_id'] == theirs['condition_id']
    assert ours['x'].shape == (300, 9) and bool(torch.isfinite(ours['energy']).all())


@pytest.mark.slow
def test_untrained_is_log_z_agrees_with_quadrature_on_nh3_and_h2co(tiny):
    """150000 draws at seed 0 in batches of 10000: conformer_logz_check's slow test's own draws
    (ESS 380 to 515 on NH3, 1600 to 1653 on H2CO over seeds 0 to 3 there), judged here with the
    bootstrap SE and the 0.1-nat floor of the bar."""
    tmp, _, m, _, _ = tiny
    res = me.evaluate(m, n=150_000, batch=10_000, seed=0, conditions=['N', 'C=O'],
                      steps=m.args.integrator.T, cache_dir=tmp / 'cache_is', n_boot=50)
    for r in res['conditions']:
        assert r['k'] == 6 and r['exact']['converged'], r['exact']
        assert r['tb']['ess'] >= 100, r['tb']
        assert r['exact_check']['verdict'] == 'PASS', (r['condition'], r['tb'], r['exact'])
        assert r['verdicts']['exact_logz']['status'] == me.PASS
        assert abs(r['tb']['log_z'] - r['exact']['log_z']) <= r['exact_check']['bar']
        assert r['verdicts']['coverage']['status'] == me.UNAVAILABLE, 'no refs were given'


def _refused(cfg, ck, match):
    with pytest.raises(me.Refused, match=match):
        me.init_eval_modeller(me.build_eval_modeller(str(cfg), str(ck)))


@pytest.mark.slow
def test_the_loader_reads_the_checkpoint_and_builds_a_cuda_stamped_one_on_the_cpu(tiny):
    tmp, cfg, _, ck, ck_cuda = tiny
    stored = torch.load(ck, map_location='cpu', weights_only=False)
    m = me.init_eval_modeller(me.build_eval_modeller(str(cfg), str(ck_cuda)))
    assert m.step_ind == 77 and m._eval_device_swap == ('cuda', 'cpu')
    assert all(p.device.type == 'cpu' for p in m.ema_model.parameters())
    for k, v in stored['model_eval'].items():
        assert torch.equal(m.ema_model.state_dict()[k].cpu(), v), k
    reg = m.identifier_registry
    assert float(m.condition_log_z.ema_logw[reg['N']]) == -12.5
    # the prior dataset was drawn at its minimum, 2 rows per member, and the config value
    # restored; the registry, the energy's library and the tracker rows are the trainer's
    assert m._eval_prior_rows == (60, 1) and m.args.energy_config.prior_sample_size == 60
    assert int(m.prior_dataset.batch.num_graphs) == 2 * len(SMIS)
    full = lzc.init_modeller(lzc.build_modeller(str(cfg), str(ck)))
    assert int(full.prior_dataset.batch.num_graphs) >= 60
    assert full.identifier_registry == reg
    assert int(full.energy_function.condition_library_size) == \
        int(m.energy_function.condition_library_size)
    assert me.identity_exempt(m) == {'bounding_coeff': 10.0, 'lj_coeff': 1.0,
                                     'energy_clip': m.energy_function.energy_clip}
    if not torch.cuda.is_available():
        # the reason for the swap: the trainer's own loader cannot build it without a card
        with pytest.raises(RuntimeError):
            lzc.init_modeller(lzc.build_modeller(str(cfg), str(ck_cuda)))


@pytest.mark.slow
def test_the_loader_refuses_a_mismatched_conditions_file_or_config(tiny):
    tmp, _, _, ck, _ = tiny
    # the conditions file was built under another stereo-lock coefficient than the run's
    _write_conditions(tmp / 'cond_locked.pt', SMIS, stereo_coeff=300.0)
    _refused(_write_config(tmp, 'locked', tmp / 'cond_locked.pt'), ck,
             'conditions file does not match.*stereo_coeff')
    # another molecule set: the carrier widens from K = 9 to 12
    _write_conditions(tmp / 'cond_wide.pt', ['N', 'C=O', 'CO'])
    _refused(_write_config(tmp, 'wide', tmp / 'cond_wide.pt'), ck,
             'does not match.*(width|layout)')
    # one more molecule inside the same widths: equal layout, a tracker of the wrong size
    _write_conditions(tmp / 'cond_four.pt', SMIS + ['OO'])
    _refused(_write_config(tmp, 'four', tmp / 'cond_four.pt'), ck, 'tracker holds 3')
    # the conditions are the checkpoint's, the energy's lock is not (energy_clip would not
    # do: utils._NON_IDENTITY_ENERGY_CONFIG_KEYS exempts it from the problem identity)
    _refused(_write_config(tmp, 'locked_cfg', tmp / 'cond.pt', stereo_coeff=300.0), ck,
             'different problem')


@pytest.mark.slow
def test_the_cli_writes_captioned_tables_and_a_json(tiny, capsys, monkeypatch):
    tmp, cfg, _, ck, _ = tiny
    out = tmp / 'cli'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    threads = torch.get_num_threads()                   # main sets --threads process-wide
    code = me.main(['--config', str(cfg), '--checkpoint', str(ck), '--n-per-condition', '256',
                    '--batch', '256', '--seed', '1', '--quadrature-max-d', '0', '--out',
                    str(out), '--threads', '2'])
    torch.set_num_threads(threads)
    text = capsys.readouterr().out
    for needle in ('Table 1a.', 'Table 1b.', 'Table 2.', 'Table 3a.', 'Table 3b.', 'Table 3c.',
                   'Table 4.', 'Table 5.', 'Table 6.', 'Table 7.', 'at step 77', 'N = 256', 'kT',
                   'Pareto k', 'WORKING ASSUMPTIONS', 'IS - ema_logw (nats)', 'EVAL_SEARCH',
                   'IS status', 'ess_floor', 'ema_decay null', 'bounding_coeff 10.0'):
        assert needle in text, needle
    res = json.loads((out / 'eval.json').read_text(encoding='utf-8'))
    assert (out / 'report.txt').exists()
    rows = {r['condition']: r for r in res['conditions']}
    assert set(rows) == set(SMIS) and res['meta']['step'] == 77
    assert rows['N']['tracker']['ema_logw'] == -12.5
    assert rows['N']['verdicts']['exact_logz']['status'] == me.N_A      # max d 0
    assert rows['C']['coverage']['n_accessible'] >= 1, 'the references were built and read'
    assert rows['C']['coverage']['kind'] == 'joint'
    assert res['meta']['refs']['source'] == 'eval_search'
    for r in rows.values():
        assert r['is_status'] == me.is_status(r['tb'], me.Bars())
        assert (r['verdicts']['ess_floor']['status'] == me.PASS) == \
            (lzc._is_problem(r['tb'], me.Bars().floors()) is None)
    assert code == (me.EXIT_PASS if all(r['overall'] == me.PASS for r in rows.values())
                    else me.EXIT_FAIL)
    assert res['exit_status'] == code


@pytest.mark.slow
def test_a_subset_eval_builds_references_for_that_subset_only(tiny):
    tmp, cfg, _, ck, _ = tiny
    m = me.init_eval_modeller(me.build_eval_modeller(str(cfg), str(ck)))
    cache = tmp / 'cache_subset'
    table, info = me.load_or_build_references(m, None, cache, identifiers=['C=O'])
    assert table.identifiers == ['C=O'] and info['missing'] == []
    # a second subset resumes nothing it does not need and adds what it does
    table, _ = me.load_or_build_references(m, None, cache, identifiers=['C=O', 'N'])
    assert sorted(table.identifiers) == ['C=O', 'N']
    table, _ = me.load_or_build_references(m, None, cache, identifiers=['N'])
    assert sorted(table.identifiers) == ['C=O', 'N'], 'a cache holding the subset is reused'
