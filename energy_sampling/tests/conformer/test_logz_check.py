"""eval/conformer_logz_check.py: the per-condition log Z check.

FAST, on synthetic numbers, one NH3 chart and a stand-in modeller: the estimator's statistics
against a closed-form constant, the quadrature rule against `brute_force_log_z` itself and, off
the isotropic grid, against a direct sum over every node, the two-grid refusal and its
unconverged flag, the integrand refusal, the verdict precedence (a row under the ESS floor is
never PASS, however close it lands), the floors on rows WITHOUT an exact value and the exit
statuses, the tracker read against the real tracker's own trust mask, the partition identity
on an exact partition, the `--partition-identity` refusal, the compute guard on a CPU check,
and the prior proposal's coverage label: which centres the prior holds at one parity, which a
stereo lock's table lifts, that a prior row without an exact value is KNOWN_BIASED rather than
ESTIMATE_ONLY and a pair on it never passes, and the refusal below level `full`.

SLOW, through the REAL modeller path (config loader, ConformerModeller, init_gfn, the eval's
own rollout function) on a width-9 CARRIER set {NH3, H2CO, CH4} at level `full`:
  * an UNTRAINED policy's IS agrees with the two-grid quadrature within the bar on NH3 and
    H2CO (d = 6), whose exact values also match the WP09 calibration's independent anchors;
    NH3 and H2CO are padded rows of the carrier, so the masked log-probs are in the test;
  * CH4 (d = 9) gets an IS estimate, judged by the floors alone;
  * too few rollouts give BELOW_ESS_FLOOR and a failing exit status, not a pass;
  * the fitted prior as proposal: NH3 FAILs against the quadrature, CH4 is KNOWN_BIASED;
  * a saved checkpoint is read through the CLI: step, tracker rows and the EMA weights.

THE UNTRAINED PROPOSAL. `zero_init` zeroes the set head's bias-free output layer
(ConformerModeller._install_set_policy), so the untrained forward policy emits exactly 0: a
zero-drift diffusion at `t_scale` 0.1 whose terminal spread covers both NH3 pyramids, whatever
the head's random init. The `untrained` fixture asserts that before any rollout. IS is
consistent for any proposal that covers the target; this one only makes the ESS affordable at
test size. Measured 2026-09-27 at N = 150000: ESS 380 to 515 on NH3 and 1600 to 1653 on H2CO
over seeds 0 to 3, and 169 on CH4 at seed 0 (ESS/N 1.1e-3).
  Until 2026-09-27 zero_init did not reach the set head, so the proposal was a RANDOM head,
re-drawn whenever its input width changed. Adding two per-coordinate features
(dof_features.row_ez_parity, is_held_bond; zero on these three molecules) moved seed 0 from
ESS 365 / 742 (NH3 / H2CO) to 38 / 3.5 with the chart, energy, prior draws and quadrature
bitwise unchanged; tests/conformer/test_set_policy_zero_init.py pins the fix.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/conformer/test_logz_check.py
"""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import eval.conformer_logz_check as lzc
from buffer import ConditionLogZTracker

HERE = Path(__file__).resolve().parents[2]            # energy_sampling/
BASE_CONFIG = HERE / 'configs' / 'conformer_mk_multi.yaml'
SMIS = ['N', 'C=O', 'C']                              # full/mmff: carrier K = 9 (4|3|2)
MOL_DIM, ENC = 16, 8

#: converged log Z at level full, T = 1, from artifacts/conformer_prod_stage0_2026-09-25/
#: wp09_final/out/logz.json: the (16, 12, 96) left-endpoint grid (-10.48921, -11.11261) and an
#: independent Laplace-mixture IS over 2e6 draws (-10.48958 +- 0.0006, -11.11314 +- 0.0005)
ANCHOR = {'N': -10.4893, 'C=O': -11.1128}


# ------------------------------------------------------------------ fast

@pytest.fixture(scope='module')
def nh3():
    from energies.conformer_torsions import ConformerTorsions
    return ConformerTorsions(smiles='N', device='cpu', level='full', force_field='mmff',
                             dtype=torch.float64)


@pytest.mark.fast
def test_equal_weights_give_the_weight_zero_se_and_full_ess():
    s = lzc.is_summary(torch.full((1000,), 2.5))
    assert s['log_z'] == pytest.approx(2.5, abs=1e-12)
    assert s['ess'] == pytest.approx(1000.0) and s['ess_frac'] == pytest.approx(1.0)
    assert s['se'] == pytest.approx(0.0, abs=1e-6) and s['mean_log_w'] == pytest.approx(2.5)


@pytest.mark.fast
def test_is_summary_recovers_a_closed_form_constant_and_its_se_is_the_delta_method():
    """Z = int exp(-x^2/2) dx = sqrt(2 pi) under a N(0, 1.5^2) proposal."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(20000, generator=g, dtype=torch.float64) * 1.5
    log_q = -0.5 * (x / 1.5) ** 2 - math.log(1.5 * math.sqrt(2 * math.pi))
    s = lzc.is_summary(-0.5 * x ** 2 - log_q)
    assert abs(s['log_z'] - 0.5 * math.log(2 * math.pi)) <= 4 * s['se']
    w = torch.exp(-0.5 * x ** 2 - log_q)
    direct = float(w.std(unbiased=False) / w.mean() / math.sqrt(len(w)))
    assert s['se'] == pytest.approx(direct, rel=1e-6)


@pytest.mark.fast
def test_nonfinite_rows_are_counted_not_hidden():
    lw = torch.tensor([0.0, 1.0, float('nan'), 2.0])
    s = lzc.is_summary(lw)
    assert s['n'] == 4 and s['n_nonfinite'] == 1
    assert s['log_z'] == pytest.approx(math.log((1 + math.e + math.e ** 2) / 3))
    ok = dict(s, ess=1e6, ess_frac=1.0)
    assert lzc.exact_verdict(ok, dict(log_z=s['log_z'], diff=0.0, converged=True),
                             lzc.Floors())[0] == 'NONFINITE_ROWS'


@pytest.mark.fast
def test_grid_spec_parses_counts_and_shift():
    g = lzc.GridSpec.parse('16,8,32@0.5')
    assert g.counts == (16, 8, 32) and g.shift == 0.5 and str(g) == '16,8,32@0.5'
    assert lzc.GridSpec.parse('4,4,4').shift == 0.0
    for bad in ('16,8', '16,8,0', '16,8,32@1.0'):
        with pytest.raises(ValueError):
            lzc.GridSpec.parse(bad)


@pytest.mark.fast
def test_the_per_block_rule_is_brute_force_log_z(nh3):
    """Isotropic, unshifted, box-only: the member's own brute_force_log_z at the same grid."""
    got, pts = lzc.grid_log_z(nh3, lzc.GridSpec((5, 5, 5)))
    assert pts == 5 ** 6
    assert got == pytest.approx(nh3.brute_force_log_z(grid=5), abs=1e-9)
    shifted, _ = lzc.grid_log_z(nh3, lzc.GridSpec((5, 5, 5), 0.5))
    assert abs(shifted - got) > 1e-3, 'the shift must move the nodes'
    wide, pts_wide = lzc.grid_log_z(nh3, lzc.GridSpec((5, 5, 5)), lin_half=1.5)
    assert pts_wide > pts and wide >= got - 1e-12, 'a superset of nodes adds mass'


@pytest.mark.fast
def test_each_column_takes_its_own_blocks_count_and_the_shift(nh3):
    """A non-isotropic, shifted grid against a direct sum over every node. The isotropic test
    above cannot see a column given another block's axis: there every axis is the same."""
    spec = lzc.GridSpec((5, 3, 4), 0.25)
    blocks = np.asarray(nh3._free_block).reshape(-1)
    assert len(set(blocks.tolist())) == 3, 'the test needs every block present'
    axes = [-1.0 + (2.0 / spec.counts[b]) * (torch.arange(spec.counts[b], dtype=torch.float64)
                                             + spec.shift) for b in blocks]
    nodes = torch.cartesian_prod(*axes)
    direct = (float(torch.logsumexp(-nh3.energy(nodes).double(), 0))
              + sum(math.log(2.0 / spec.counts[b]) for b in blocks))
    got, pts = lzc.grid_log_z(nh3, spec, chunk=1000)       # several chunks, a ragged last one
    assert pts == nodes.shape[0] == 5 ** 3 * 3 ** 2 * 4
    assert got == pytest.approx(direct, abs=1e-9)


@pytest.mark.fast
def test_a_pair_that_leaves_a_block_unrefined_is_refused(nh3):
    """H2CO at (12,8,48) and (12,10,64) agree to 6e-4 while both are 0.022 off: the r block
    kept its nodes, so its error was common to both values."""
    with pytest.raises(lzc.Refused, match=r"\['r'\]"):
        lzc.two_grid_log_z(nh3, lzc.GridSpec((12, 8, 48)), lzc.GridSpec((12, 10, 64)), 0.02)
    with pytest.raises(lzc.Refused, match='budget'):
        lzc.grid_log_z(nh3, lzc.GridSpec((16, 8, 32)), max_points=1e3)


class _RunEnergy:
    """The NH3 member standing as its own run energy, its log_reward at a temperature `scale`
    times the member's: what a run scored at the wrong T would hand the IS."""

    def __init__(self, member, scale):
        self.member, self.scale, self.log_temperature = member, scale, 0.0
        self.data_ndim, self.dtype, self.device = member.data_ndim, member.dtype, member.device

    def energy(self, x):
        return self.member.energy(x)

    def log_reward(self, x, mb, log_t):
        return -self.member.energy(x) / self.scale


class _OneRow:
    """A condition dataset of one row: the batch integrand_identity draws and orients."""

    def sample_graphs_at(self, rows, repeats):
        return self

    def to(self, device):
        return self

    def orient_molecule(self, mode):
        pass


@pytest.mark.fast
def test_the_integrand_check_refuses_a_reward_that_is_not_the_quadratures(nh3):
    assert lzc.check_integrand(_RunEnergy(nh3, 1.0), 'N', _OneRow(), 0, tol=1e-4) < 1e-12
    with pytest.raises(lzc.Refused, match='not one target'):
        lzc.check_integrand(_RunEnergy(nh3, 1.01), 'N', _OneRow(), 0, tol=1e-4)


@pytest.mark.fast
def test_a_coarse_pair_is_flagged_unconverged(nh3):
    r = lzc.two_grid_log_z(nh3, lzc.GridSpec((6, 4, 16)), lzc.GridSpec((6, 4, 16), 0.5), 0.02)
    assert r['converged'] is False and r['diff'] > 0.02
    est = dict(log_z=r['log_z'], se=0.01, ess=1e4, ess_frac=0.5, n_nonfinite=0)
    assert lzc.exact_verdict(est, r, lzc.Floors())[0] == 'UNCONVERGED'


def _est(log_z=-10.0, se=0.05, ess=500.0, ess_frac=0.05, **kw):
    return dict(dict(log_z=log_z, se=se, ess=ess, ess_frac=ess_frac, n_nonfinite=0), **kw)


QUAD = dict(log_z=-10.0, diff=0.004, converged=True)


@pytest.mark.fast
def test_exact_verdicts_and_their_precedence():
    fl = lzc.Floors()
    assert lzc.exact_verdict(None, QUAD, fl)[0] == 'NOT_RUN'
    assert lzc.exact_verdict(_est(clip_frac=0.01), QUAD, fl)[0] == 'INVALID_PROPOSAL'
    # the property the ESS floor exists for: on the exact value, still not a pass
    assert lzc.exact_verdict(_est(ess=50.0), QUAD, fl)[0] == 'BELOW_ESS_FLOOR'
    assert lzc.exact_verdict(_est(ess_frac=1e-4), QUAD, fl)[0] == 'BELOW_ESS_FLOOR'
    assert lzc.exact_verdict(_est(), dict(QUAD, converged=False), fl)[0] == 'UNCONVERGED'
    assert lzc.exact_verdict(_est(), None, fl)[0] == 'UNCONVERGED'
    v, bar = lzc.exact_verdict(_est(log_z=-10.15), QUAD, fl)
    assert v == 'PASS' and bar == pytest.approx(3 * 0.05 + 0.004)
    assert lzc.exact_verdict(_est(log_z=-10.16), QUAD, fl)[0] == 'FAIL'
    assert lzc.exact_verdict(_est(log_z=-10.16), QUAD, lzc.Floors(abs_tol=0.2))[0] == 'PASS'


@pytest.mark.fast
def test_pair_verdicts():
    fl = lzc.Floors()
    row = lambda name, est: dict(condition=name, **{'is': est}, exact=None,
                                 tracker=dict(ema_logw=float('nan')))
    a = row('R', _est(log_z=-9.0, se=0.03))
    p = lzc.pair_verdict(a, row('S', _est(log_z=-9.1, se=0.04)), fl)
    assert p['verdict'] == 'PASS' and p['se'] == pytest.approx(0.05)
    assert p['z'] == pytest.approx(2.0)
    assert lzc.pair_verdict(a, row('S', _est(log_z=-9.2, se=0.04)), fl)['verdict'] == 'FAIL'
    assert lzc.pair_verdict(a, row('S', _est(ess=10.0)), fl)['verdict'] == 'BELOW_ESS_FLOOR'
    # the floor holds on EITHER side of the pair
    low_a = row('R', _est(log_z=-9.0, se=0.03, ess=10.0))
    assert lzc.pair_verdict(low_a, row('S', _est(log_z=-9.0)), fl)['verdict'] == 'BELOW_ESS_FLOOR'


@pytest.mark.fast
def test_tracker_rows_are_read_by_condition_id_with_the_trackers_own_trust_mask():
    tr = ConditionLogZTracker(library_size=3, min_visits=20)
    tr.ema_logw[1], tr.ema_log_z_emp[1], tr.count[1] = -7.5, -7.1, 25
    tr.ema_logw[2], tr.ema_log_z_emp[2], tr.count[2] = -3.0, -2.9, 5
    state = tr.state_dict()
    one = lzc.tracker_reading(state, 1)
    assert (one['ema_logw'], one['ema_log_z_emp'], one['visits']) == (-7.5, pytest.approx(-7.1), 25)
    _, mask = tr.lookup([0, 1, 2])
    assert [lzc.tracker_reading(state, c)['trusted'] for c in range(3)] == mask.tolist()
    assert math.isnan(lzc.tracker_reading(state, 0)['ema_logw'])
    assert lzc.tracker_reading(None, 1)['trusted'] is None
    with pytest.raises(IndexError):
        lzc.tracker_reading(state, 3)


class _Toy:
    """Two columns on the box, r and phi, with HARD sign locks v_j(x) = x_j + off_j.

    No grid node sits on v_j = 0, so the locks partition the nodes exactly and the identity
    must hold to roundoff. `overlap` makes both signs admit v >= 0: not a partition.
    """
    data_ndim, dtype, device = 2, torch.float64, 'cpu'
    _free_block = np.array([0, 2])
    OFF = (0.137, -0.291)

    def __init__(self, signs=None, overlap=False):
        self.signs, self.overlap = signs, overlap

    def energy(self, x):
        e = 3.0 * x[:, 0] ** 2 - torch.cos(math.pi * x[:, 1])
        for j, s in enumerate(self.signs or ()):
            v = x[:, j] + self.OFF[j]
            keep = v >= 0 if self.overlap else s * v >= 0
            e = torch.where(keep, e, torch.full_like(e, float('inf')))
        return e


@pytest.mark.fast
def test_partition_identity_holds_on_an_exact_partition_and_not_otherwise():
    spec = lzc.GridSpec((40, 4, 40))
    for n in (1, 2):
        r = lzc.partition_identity(lambda s: _Toy(s), n, spec)
        assert abs(r['residual']) < 1e-10 and len(r['per_assignment']) == 2 ** n
        assert r['points'] == 40 * 40 * (1 + 2 ** n)
    bad = lzc.partition_identity(lambda s: _Toy(s, overlap=True), 1, spec)
    assert bad['residual'] > 0.1


@pytest.mark.fast
def test_the_partition_flag_refuses_in_both_states_before_reading_anything(capsys, monkeypatch):
    """No lock: refused as inert. A lock whose sign override is not wired here: refused as
    unwired. Either way exit 2 and nothing run, so this holds whichever state the stereo
    stream leaves ConformerTorsions in."""
    assert set(lzc.stereo_seam_missing()) <= set(lzc.STEREO_SEAM)
    argv = ['--config', 'no/such/config.yaml', '--partition-identity']
    monkeypatch.setattr(lzc, 'stereo_seam_missing', lambda: ['stereo_coeff'])
    assert lzc.main(argv) == lzc.EXIT_REFUSED
    assert 'inert' in capsys.readouterr().err
    monkeypatch.setattr(lzc, 'stereo_seam_missing', lambda: [])
    assert lzc.main(argv) == lzc.EXIT_REFUSED
    assert 'not wired' in capsys.readouterr().err


# ------------------------------------------------------------------ rows without an exact value

class _Energy:
    """What run_check reads off a run energy for a row with no quadrature (k > 6)."""
    level, data_ndim, temperature, condition_library_size = 'full', 9, 1.0, 3


def _stand_in(idents):
    ds = SimpleNamespace(batch=SimpleNamespace(identifier=list(idents)))
    return SimpleNamespace(energy_function=_Energy(), mol_dataset=ds, device='cpu', step_ind=0,
                           stage=None, args=SimpleNamespace(checkpoint_name=None, eval_T=10,
                                                            integrator=SimpleNamespace(T=10)))


def _draws():
    g = torch.Generator().manual_seed(1)
    good = torch.randn(1000, generator=g, dtype=torch.float64) * 0.1 - 7.0
    heavy = torch.zeros(1000, dtype=torch.float64)
    heavy[0] = 25.0                                  # one weight carries it: ESS ~ 1
    nan = good.clone()
    nan[3] = float('nan')
    return dict(good=good, heavy=heavy, nan=nan)


@pytest.fixture
def stand_in(monkeypatch):
    draws = _draws()
    ids = {k: i for i, k in enumerate(draws)}
    monkeypatch.setattr(lzc, 'policy_rollouts', lambda m, ident, *a: dict(
        log_w=draws[ident], head=-5.0, head_spread=0.0, condition_id=ids[ident]))
    return _stand_in(draws)


def _table_row(text, table, ident):
    """The whitespace-split cells of `ident`'s row in the named table of a report."""
    block = text.split(table, 1)[1].split('\n\n', 1)[0]
    (line,) = [ln for ln in block.splitlines() if ln.split() and ln.split()[0] == ident]
    return line.split()


@pytest.mark.fast
def test_the_floors_judge_rows_without_an_exact_value(stand_in):
    res = lzc.run_check(stand_in)
    rows = {r['condition']: r for r in res['conditions']}
    assert all(r['exact'] is None for r in rows.values())
    assert (rows['good']['verdict'], rows['good']['is_status']) == (lzc.ESTIMATE_ONLY, 'ok')
    assert rows['heavy']['verdict'] == rows['heavy']['is_status'] == 'BELOW_ESS_FLOOR'
    assert rows['nan']['verdict'] == 'NONFINITE_ROWS' and rows['nan']['is']['n_nonfinite'] == 1
    assert lzc.exit_status(res) == lzc.EXIT_FAIL
    text = lzc.format_report(res)
    assert 'non-finite log w (rows)' in text and 'IS status' in text
    # Table 1 counts the dropped row; Table 2 blanks head - IS on the refused rows only
    assert _table_row(text, 'Table 1.', 'nan')[7] == '1'
    assert _table_row(text, 'Table 2.', 'good')[-1] != '-'
    for ident in ('heavy', 'nan'):
        cells = _table_row(text, 'Table 2.', ident)
        assert cells[2] == rows[ident]['is_status'] and cells[-1] == '-'


@pytest.mark.fast
def test_estimates_alone_exit_unchecked_and_say_so(stand_in):
    res = lzc.run_check(stand_in, ['good'])
    assert lzc.exit_status(res) == lzc.EXIT_UNCHECKED
    assert 'NOTHING WAS CHECKED' in lzc.format_report(res)


@pytest.mark.fast
def test_exit_statuses_over_verdicts():
    res = lambda *v: dict(conditions=[dict(verdict=x) for x in v], pairs=[])
    assert lzc.exit_status(res('PASS', lzc.ESTIMATE_ONLY)) == lzc.EXIT_PASS
    assert lzc.exit_status(res(lzc.ESTIMATE_ONLY)) == lzc.EXIT_UNCHECKED
    assert lzc.exit_status(res()) == lzc.EXIT_UNCHECKED
    for bad in lzc.VERDICTS[2:]:
        assert lzc.exit_status(res('PASS', bad)) == lzc.EXIT_FAIL, bad
    paired = dict(conditions=[dict(verdict=lzc.ESTIMATE_ONLY)] * 2, pairs=[dict(verdict='PASS')])
    assert lzc.exit_status(paired) == lzc.EXIT_PASS


@pytest.mark.fast
def test_requests_the_check_cannot_answer_are_refused(stand_in):
    with pytest.raises(lzc.Refused, match='no condition row'):
        lzc.run_check(stand_in, ['absent'])
    with pytest.raises(lzc.Refused, match='temperature_conditioning'):
        lzc.init_modeller(SimpleNamespace(args=SimpleNamespace(temperature_conditioning=True)))


# ------------------------------------------------------------------ the prior's coverage

@pytest.fixture(scope='module')
def small():
    """CH4 (one tetrahedral centre), H2CO (planar) and CH3NH2 (a tetrahedral C and an amine
    N) at level full: the members the coverage label is judged on."""
    from energies.conformer_torsions import ConformerTorsions
    return {s: ConformerTorsions(smiles=s, device='cpu', level='full', force_field='mmff',
                                 dtype=torch.float64) for s in ('C', 'C=O', 'CN')}


def _four_coordinate(member):
    """Placement slots with four bonded neighbours: what the stereo lock's design locks."""
    deg = np.bincount(np.asarray(member.bond_index_slot).reshape(-1),
                      minlength=len(member.spec.z))
    return [int(i) for i in np.flatnonzero(deg == 4)]


class _Locked:
    """A real member seen through a stereo lock: `stereo_coeff` and a lock table (`stereo`,
    keys in placement slots) over it, everything else the member's own."""

    def __init__(self, member, keys, coeff=300.0):
        self._member, self.stereo_coeff = member, coeff
        self.stereo = None if keys is None else SimpleNamespace(key=np.asarray(keys))

    def __getattr__(self, name):
        return getattr(self._member, name)


@pytest.mark.fast
def test_the_prior_holds_one_parity_at_every_non_planar_centre(nh3, small):
    """The held rows' mirrors: at an sp3 centre and an amine N far outside the prior's width,
    at a planar centre on the reference itself."""
    z = lambda m: [int(np.asarray(m.spec.z)[c]) for c in lzc.prior_held_parity_centres(m)]
    assert z(nh3) == [7] and z(small['C']) == [6]
    assert z(small['C=O']) == [] and lzc.prior_coverage_bias(small['C=O']) is None
    assert sorted(z(small['CN'])) == [6, 7]


@pytest.mark.fast
def test_a_lock_lifts_the_label_only_on_the_centres_its_table_names(nh3, small):
    """The lock's design locks four-coordinate centres and leaves an amine N free, so a locked
    CH3NH2 is still biased at its N; an unreadable table lifts nothing."""
    cn = small['CN']
    four = _four_coordinate(cn)
    (n_slot,) = [c for c in lzc.prior_held_parity_centres(cn) if c not in four]
    assert 'unlocked target holds both' in lzc.prior_coverage_bias(cn)
    locked = lzc.prior_coverage_bias(_Locked(cn, four))
    assert f'N{n_slot} ' in locked and 'C' not in locked.split('parity at ')[1].split(' (')[0]
    assert lzc.prior_coverage_bias(_Locked(cn, four + [n_slot])) is None
    assert lzc.prior_coverage_bias(_Locked(small['C'], _four_coordinate(small['C']))) is None
    assert 'unreadable' in lzc.prior_coverage_bias(_Locked(small['C'], None))
    assert lzc.prior_coverage_bias(_Locked(nh3, [])) is not None, 'NH3 has no locked centre'
    # a table without the lock on locks nothing
    assert 'unlocked' in lzc.prior_coverage_bias(_Locked(small['C'], [0], coeff=0.0))


class _Graphs:
    """One condition's drawn batch, carrying its row as the condition id."""

    def __init__(self, row):
        self.row = row

    def to(self, device):
        return self

    def orient_molecule(self, mode):
        pass


class _MemberSet:
    """A run energy holding real members, for the prior path's reads."""

    def __init__(self, members):
        self._members, self.temperature = members, 1.0
        self.condition_library_size = len(members)

    def condition_samples(self, mb):
        return None, None, None, torch.full((2,), mb.row)


@pytest.mark.fast
def test_prior_rows_without_an_exact_value_are_known_biased_not_estimate_only(small,
                                                                              monkeypatch):
    """The round-2 defect: CH4 under --proposal prior read ESTIMATE_ONLY (a clean ESS of 6014)
    while its value runs about ln 2 low. The estimates here are clean by construction, so only
    the coverage label can refuse them."""
    members = {k: small[k] for k in ('C', 'C=O')}
    ds = SimpleNamespace(batch=SimpleNamespace(identifier=list(members)),
                         sample_graphs_at=lambda rows, repeats: _Graphs(rows[0]))
    m = SimpleNamespace(energy_function=_MemberSet(members), mol_dataset=ds, device='cpu',
                        step_ind=0, stage=None,
                        args=SimpleNamespace(checkpoint_name=None, eval_T=10,
                                             integrator=SimpleNamespace(T=10)))
    clean = {'C': -14.7, 'C=O': -11.1}
    monkeypatch.setattr(lzc, 'prior_estimate', lambda m, ident, n, seed: dict(
        _est(log_z=clean[ident], se=0.01, ess=6000.0, ess_frac=0.3), n=n, clip_frac=0.0))
    res = lzc.run_check(m, proposal='prior', max_quad_dim=0, pairs=[('C', 'C=O')])
    rows = {r['condition']: r for r in res['conditions']}
    assert rows['C']['verdict'] == rows['C']['is_status'] == lzc.KNOWN_BIASED
    assert any('parity at C0' in n for n in rows['C']['notes'])
    assert (rows['C=O']['verdict'], rows['C=O']['bias']) == (lzc.ESTIMATE_ONLY, None)
    (pair,) = res['pairs']
    assert pair['verdict'] == lzc.KNOWN_BIASED and math.isfinite(pair['delta'])
    assert lzc.exit_status(res) == lzc.EXIT_FAIL, 'a known-biased row never exits clean'
    text = lzc.format_report(res)
    assert _table_row(text, 'Table 1.', 'C')[-1] == lzc.KNOWN_BIASED
    assert 'KNOWN_BIASED: the fitted prior' in text.split('Notes:')[1]
    # the floors still come first, and the label belongs to the prior proposal alone
    monkeypatch.setattr(lzc, 'prior_estimate', lambda m, ident, n, seed: _est(ess=10.0, n=n))
    low = lzc.run_check(m, ['C'], proposal='prior', max_quad_dim=0)['conditions'][0]
    assert low['verdict'] == 'BELOW_ESS_FLOOR' and low['bias']
    monkeypatch.setattr(lzc, 'policy_rollouts', lambda m, ident, *a: dict(
        log_w=_draws()['good'], head=-5.0, head_spread=0.0, condition_id=0))
    pol = lzc.run_check(m, ['C'], proposal='policy', max_quad_dim=0)['conditions'][0]
    assert (pol['verdict'], pol['bias']) == (lzc.ESTIMATE_ONLY, None)


@pytest.mark.fast
def test_the_prior_proposal_is_refused_below_level_full():
    """is_log_z weighs every coordinate the prior draws; a lower level's state discards the
    frozen ones (measured at dihedral: H2CO 4.83 nats low against the grid)."""
    m = SimpleNamespace(internal_prior=object(),
                        energy_function=SimpleNamespace(temperature=1.0, level='dihedral'))
    with pytest.raises(lzc.Refused, match='needs level full'):
        lzc.prior_estimate(m, 'C=O', n=10, seed=0)


# ------------------------------------------------------------------ the card

@pytest.fixture
def guard(monkeypatch):
    """gpu_guard.require_free_gpu replaced by a recorder of the skip reason it would see; the
    CUDA runtime's first query modelled as reading CUDA_VISIBLE_DEVICES. No GPU is touched."""
    import gpu_guard
    seen = []
    monkeypatch.setattr(gpu_guard, 'require_free_gpu',
                        lambda **kw: seen.append(gpu_guard._skip_reason()))
    monkeypatch.setattr(torch.cuda, 'is_available',
                        lambda: os.environ.get('CUDA_VISIBLE_DEVICES') != '-1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    return seen


@pytest.mark.fast
def test_a_cpu_check_hides_the_card_and_the_guard_still_runs(guard):
    lzc.gpu_preflight('cpu', 'x.yaml')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    assert len(guard) == 1 and 'hides all GPUs' in guard[0]


@pytest.mark.fast
def test_cuda_initialised_before_the_hide_puts_the_card_back_under_the_guard(guard, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    lzc.gpu_preflight('cpu', 'x.yaml')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    assert len(guard) == 1 and (guard[0] is None or 'hides' not in guard[0])


@pytest.mark.fast
def test_a_busy_card_is_a_refusal(guard, monkeypatch, capsys):
    import gpu_guard

    def busy(**kw):
        raise gpu_guard.GPUBusy('another run holds the card')
    monkeypatch.setattr(gpu_guard, 'require_free_gpu', busy)
    assert lzc.main(['--config', 'no/such/config.yaml', '--device', 'cuda']) == lzc.EXIT_REFUSED
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0', 'a CUDA device is not hidden'
    assert 'another run holds the card' in capsys.readouterr().err


# ------------------------------------------------------------------ slow, the real path

def _prior_path():
    p = Path(os.environ.get('GFN_CONFORMER_PRIOR', HERE / 'conformer_prior_v2.pt'))
    if not p.exists():
        pytest.fail(f'fitted InternalPrior not found at {p}: this test cannot run without it. '
                    f'Set GFN_CONFORMER_PRIOR to the local conformer_prior_v2.pt.')
    return p.as_posix()


def _write_conditions(path):
    """Carrier-padded condition graphs as build_conformer_conditions.py --carrier writes
    them, with random frozen embeddings in place of the encoder's."""
    from energies.conformer_carrier import carrier_pad_condition
    from energies.conformer_data import (collate_conditions, condition_from_energy,
                                         save_condition_file)
    from energies.dof_features import free_dof_atom_index
    from energies.multi_conformer import MultiConformerTorsions

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        en = MultiConformerTorsions(SMIS, identifiers=SMIS, device='cpu', level='full',
                                    force_field='mmff')
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


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    """(config path, dir) for the carrier set, the multi-molecule config shrunk to CPU size."""
    old_dtype, old_wandb = torch.get_default_dtype(), os.environ.get('WANDB_MODE')
    torch.set_default_dtype(torch.float32)           # the route's dtype
    os.environ['WANDB_MODE'] = 'disabled'
    try:
        tmp = tmp_path_factory.mktemp('logz')
        _write_conditions(tmp / 'cond.pt')
        cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding='utf-8'))
        cfg.update(run_name='lz', checkpoints_dir=tmp.as_posix(),
                   molecules_path=(tmp / 'cond.pt').as_posix(), test_molecules_path=None,
                   embedding_conditioning_dim=MOL_DIM, batch_size=64, max_batch_size=64,
                   grow_batch_size=False, archive_period=0, eval_T=10)
        cfg['integrator']['T'] = 10
        for k in ('s_emb_dim', 't_hidden_dim', 's_hidden_dim', 'policy_hidden_dim',
                  'flow_hidden_dim', 'cond_hidden_dim', 't_dim', 'harmonics_dim'):
            cfg['model'][k] = 32
        for k in ('s_layers', 'policy_layers', 'flow_layers', 'cond_layers'):
            cfg['model'][k] = 2
        cfg['model'].update(zero_init=True, t_scale=0.1, policy_kind='set',
                            set_policy_hidden=32, set_policy_layers=2, set_policy_corr_dim=8,
                            dplr_rank=0)
        cfg['energy_config'].update(internal_prior_path=_prior_path(), prior_sample_size=60)
        path = tmp / 'lz.yaml'
        path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
        yield path, tmp
    finally:
        torch.set_default_dtype(old_dtype)
        if old_wandb is None:
            os.environ.pop('WANDB_MODE', None)
        else:
            os.environ['WANDB_MODE'] = old_wandb


@pytest.fixture(scope='module')
def untrained(tiny):
    m = lzc.init_modeller(lzc.build_modeller(str(tiny[0])))
    # THE PROPOSAL IS CHECKED, NOT ASSUMED (module docstring, THE UNTRAINED PROPOSAL): a
    # zero, bias-free output layer on the set head the rollouts use, so the untrained policy
    # emits exactly 0 and does not depend on the head's random init
    out = m.ema_model.forward_policy.rho.output_layer
    assert out.bias is None and bool((out.weight == 0).all()), \
        'model.zero_init did not reach the set head: the untrained proposal is a random head'
    return m, lzc.run_check(m, SMIS, n=150_000, batch=10_000, seed=0)


@pytest.mark.slow
def test_untrained_is_agrees_with_converged_quadrature_on_nh3_and_h2co(untrained):
    m, res = untrained
    assert m.energy_function.is_carrier and m.energy_function.data_ndim == 9
    rows = {r['condition']: r for r in res['conditions']}
    for ident in ('N', 'C=O'):
        r = rows[ident]
        assert r['level'] == 'full' and r['k'] == 6
        assert r['identity'] is not None and r['identity'] < 1e-4, 'integrand check ran'
        assert r['exact']['converged'] and r['exact']['points'] > 0
        assert abs(r['exact']['log_z'] - ANCHOR[ident]) < 0.01, (ident, r['exact'])
        assert r['is']['ess'] >= 100 and r['is']['n'] == 150_000
        assert r['verdict'] == 'PASS', (ident, r['is'], r['exact'], r['bar'])
        assert abs(r['is']['log_z'] - r['exact']['log_z']) <= r['bar']
    c = rows['C']
    assert c['k'] == 9 and c['exact'] is None
    assert math.isfinite(c['is']['log_z']) and any('no quadrature' in n for n in c['notes'])
    # no exact value, so the floors alone judge it. Measured here under the zero-output
    # proposal: ESS 169, ESS/N 1.1e-3, just clearing both floors (ESTIMATE_ONLY); under the
    # random untrained head this fixture used before 2026-09-27 it read ESS 6.7, floored
    floored = c['is']['ess'] < 100 or c['is']['ess_frac'] < 1e-3
    assert c['verdict'] == ('BELOW_ESS_FLOOR' if floored else lzc.ESTIMATE_ONLY)
    assert len({r['condition_id'] for r in res['conditions']}) == 3
    assert lzc.exit_status(res) == (lzc.EXIT_FAIL if floored else lzc.EXIT_PASS)


@pytest.mark.slow
def test_the_report_is_a_set_of_labelled_captioned_tables(untrained):
    _, res = untrained
    text = lzc.format_report(res)
    for needle in ('Table 1.', 'Table 2.', 'IS log Z (nats)', 'exact log Z (nats)',
                   'ema_logw (nats)', 'head (nats)', 'N = 150000', 'seed 0', 'level full',
                   'UNTRAINED policy', 'verdicts: PASS 2', '10 rollout steps'):
        assert needle in text, needle
    assert 'Table 3.' not in text, 'no pairs were requested'


@pytest.mark.slow
def test_too_few_rollouts_are_below_the_floor_not_a_pass(untrained):
    m, _ = untrained
    res = lzc.run_check(m, ['N'], n=50, seed=0, grids=('12,6,32', '12,6,32@0.5'))
    (r,) = res['conditions']
    assert r['exact']['converged'] and r['verdict'] == 'BELOW_ESS_FLOOR'
    assert lzc.exit_status(res) == 1
    assert 'verdicts: BELOW_ESS_FLOOR 1' in lzc.format_report(res)


@pytest.mark.slow
def test_the_prior_proposal_is_known_biased_where_it_holds_a_parity(untrained):
    """The real prior on the real path: NH3's missing pyramid FAILs against the quadrature and
    carries the label; CH4, with no exact value, is KNOWN_BIASED (it read ESTIMATE_ONLY at ESS
    6014 before); planar H2CO carries none and passes."""
    m, _ = untrained
    res = lzc.run_check(m, SMIS, proposal='prior', n=20_000, seed=0)
    rows = {r['condition']: r for r in res['conditions']}
    n, c, h2co = rows['N'], rows['C'], rows['C=O']
    assert n['verdict'] == 'FAIL' and n['is_status'] == lzc.KNOWN_BIASED and n['bias']
    assert n['is']['log_z'] - n['exact']['log_z'] < -0.5, 'the missing pyramid, ~ln 2'
    assert c['exact'] is None and c['verdict'] == lzc.KNOWN_BIASED and 'C0' in c['bias']
    assert c['is']['ess'] >= 100, 'the floors clear it: only the label refuses it'
    assert h2co['verdict'] == 'PASS' and h2co['bias'] is None
    assert lzc.exit_status(res) == lzc.EXIT_FAIL


@pytest.mark.slow
def test_a_checkpoint_is_read_through_the_cli(tiny, untrained, capsys, monkeypatch):
    """Step, tracker rows and EMA weights come off the file, not off a fresh build.

    The EMA weights are moved before the save: GFN construction reseeds torch at 0, so an
    unperturbed save would equal any fresh build and prove nothing about the load.
    """
    cfg_path, tmp = tiny
    fresh_heads = {r['condition']: r['head'] for r in untrained[1]['conditions']}
    a = lzc.init_modeller(lzc.build_modeller(str(cfg_path)))
    a.init_condition_log_z()
    reg = a.identifier_registry
    tr = a.condition_log_z
    tr.ema_logw[reg['N']], tr.ema_log_z_emp[reg['N']], tr.count[reg['N']] = -12.5, -11.9, 25
    tr.ema_logw[reg['C=O']], tr.count[reg['C=O']] = -13.25, 5
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for p in a.ema_model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    a.step_ind = 123
    a.args.checkpoint_read_only = False
    a.checkpointer.save('probe')
    ck = a.checkpointer.path_for('probe')
    direct = lzc.run_check(a, ['N', 'C=O'], n=2000, batch=1000, seed=4, max_quad_dim=0)

    out = tmp / 'probe.json'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')     # main hides the card; put it back after
    code = lzc.main(['--config', str(cfg_path), '--checkpoint', ck, '--conditions', 'N', 'C=O',
                     '--n', '2000', '--batch', '1000', '--seed', '4', '--max-quad-dim', '0',
                     '--pair', 'N', 'C=O', '--json', str(out)])
    text = capsys.readouterr().out
    res = json.loads(out.read_text(encoding='utf-8'))
    assert res['meta']['step'] == 123 and res['meta']['tracker'] is True
    rows = {r['condition']: r for r in res['conditions']}
    assert rows['N']['tracker'] == dict(ema_logw=-12.5, ema_log_z_emp=pytest.approx(-11.9),
                                        visits=25, trusted=True)
    assert rows['C=O']['tracker']['visits'] == 5 and rows['C=O']['tracker']['trusted'] is False
    for d in direct['conditions']:
        r = rows[d['condition']]
        assert r['head'] == pytest.approx(d['head'], abs=1e-6)
        assert r['is']['log_z'] == pytest.approx(d['is']['log_z'], abs=1e-5)
        assert abs(r['head'] - fresh_heads[d['condition']]) > 1e-3, 'the EMA was not loaded'
    for needle in ('Table 2.', 'Table 3.', '-12.5000', 'at step 123', 'N | C=O'):
        assert needle in text, needle
    # a pair was requested, so a verdict was formed; at N = 2000 on two different molecules
    # it cannot be PASS (under the floor, or two constants), and the exit status says so
    assert res['pairs'][0]['verdict'] in lzc.VERDICTS[2:] and code == lzc.EXIT_FAIL
