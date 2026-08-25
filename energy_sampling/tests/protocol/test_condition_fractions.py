"""
CPU tests for the per-condition breakdown of the two pooled batch fractions
('Reasonable Sample Fraction', 'Nonthermal Fraction') -- utils.per_condition_fraction
and its two wiring points in train.py.

WHAT THIS SUITE IS FOR. The claim being made is that a per-condition reading sees
a failure geometry the pooled fraction cannot: half the library at zero and every
condition at 50% are the same pooled number. So the load-bearing test is not "the
function returns something", it is "the function SEPARATES two batches the pooled
metric cannot separate" -- asserted here alongside the pooled value, which must
stay equal across the pair. Three mutation checks re-introduce the obvious bugs
(pool instead of group, forget to subset the condition ids, flip the direction)
and require a FAILURE, so a test that has gone blind cannot pass quietly.

The wiring half drives the REAL Modeller methods (bound onto a stub) rather than
re-implementing them, because the failure mode that costs a whole run is not a
wrong number -- it is a diagnostic that silently never fires. 'did it RUN' (keys
present) and 'is it right' (values) are asserted separately and reported
separately.

    python test_condition_fractions.py
"""
import os
import sys
from types import SimpleNamespace

import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import per_condition_fraction  # noqa: E402
from train import Modeller  # noqa: E402

WQ = 0.25  # conditional_worst_quantile, as mk_dev sets it


def _pooled(indicator):
    """The metric this replaces -- the mutation target for MUT-1."""
    return torch.as_tensor(indicator).float().mean().item()


# --------------------------------------------------------------------------
# the pure function
# --------------------------------------------------------------------------

def test_separates_concentrated_from_spread():
    """THE test. Two batches, identical pooled fraction, opposite geometry."""
    cid = torch.arange(8).repeat_interleave(4)  # 8 conditions x 4 samples

    # spread: every condition exactly half good
    spread = torch.tensor([1, 1, 0, 0] * 8, dtype=torch.bool)
    # concentrated: four conditions fully good, four fully dead
    concentrated = torch.tensor([1] * 16 + [0] * 16, dtype=torch.bool)

    assert _pooled(spread) == _pooled(concentrated) == 0.5, \
        "test setup broken: the two batches must have the same POOLED fraction"

    s = per_condition_fraction(spread, cid, bar=0.5, worst_quantile=WQ)
    c = per_condition_fraction(concentrated, cid, bar=0.5, worst_quantile=WQ)

    # bar 0.5, higher-is-better: 'failing' is p_c < 0.5, so an exactly-0.5
    # condition passes and a dead one fails
    assert s['failing_frac'] == 0.0, f"spread: {s['failing_frac']}"
    assert c['failing_frac'] == 0.5, f"concentrated: {c['failing_frac']}"
    assert s['worst'] == 0.5 and c['worst'] == 0.0, (s['worst'], c['worst'])
    assert s['n_conditions'] == c['n_conditions'] == 8

    # MUT-1: pool instead of group -- the discrimination must DISAPPEAR
    assert _pooled(spread) == _pooled(concentrated), \
        "MUT-1 is inert: the pooled metric already separates these, so the " \
        "per-condition assertions above prove nothing"
    print("PASS separates concentrated from spread (pooled 0.5 both ways; "
          f"failing_frac {s['failing_frac']} vs {c['failing_frac']})")


def test_orientation_and_worst_tail():
    """higher_is_worse flips which side of the bar fails AND which tail 'worst'
    reads. Both must move together or 'worst' names the wrong end."""
    cid = torch.arange(4).repeat_interleave(4)
    ind = torch.tensor([1, 1, 1, 1,        # p_c = 1.00
                        1, 1, 1, 0,        # p_c = 0.75
                        1, 0, 0, 0,        # p_c = 0.25
                        0, 0, 0, 0],       # p_c = 0.00
                       dtype=torch.bool)
    p_c = torch.tensor([1.0, 0.75, 0.25, 0.0])

    # bar 0.8, deliberately NOT 0.5: p_c here is symmetric about 0.5, so a
    # centred bar splits 2/2 either way and would pass with the direction
    # ignored entirely
    lo = per_condition_fraction(ind, cid, bar=0.8, worst_quantile=WQ,
                                higher_is_worse=False)
    hi = per_condition_fraction(ind, cid, bar=0.8, worst_quantile=WQ,
                                higher_is_worse=True)

    assert lo['failing_frac'] == 0.75, lo['failing_frac']  # 0.75, 0.25, 0.00 below
    assert hi['failing_frac'] == 0.25, hi['failing_frac']  # 1.00 above
    # MUT-3: had higher_is_worse been ignored these two would be equal
    assert lo['failing_frac'] != hi['failing_frac'], \
        "MUT-3 is inert: higher_is_worse changed nothing"

    # the worst_quantile convention, stated the way quick_tb_stats states it:
    # worst_quantile = the fraction of conditions allowed to sit BEYOND the bar
    assert abs(lo['worst'] - torch.quantile(p_c, WQ).item()) < 1e-6
    assert abs(hi['worst'] - torch.quantile(p_c, 1.0 - WQ).item()) < 1e-6
    assert lo['worst'] < p_c.mean().item() < hi['worst'], \
        "'worst' must sit on the bad side of the mean for both orientations"
    print(f"PASS orientation + worst tail (lo {lo['worst']:.3f} / hi {hi['worst']:.3f})")


def test_spread_recovers_geometry_and_failing_frac_does_not_travel():
    """The cross-stream claim. Same underlying model, two sampling budgets:
    'Spread' must agree and 'Failing Frac' must NOT, or the warning against
    comparing failing fractions across eval_fwd/eval_test is unfounded."""
    torch.manual_seed(0)
    k, true_p = 400, 0.6      # every condition identical: true spread is ZERO

    def draw(n_c):
        cid = torch.arange(k).repeat_interleave(n_c)
        ind = (torch.rand(k * n_c) < true_p)
        return per_condition_fraction(ind, cid, bar=0.5, worst_quantile=WQ)

    lo, hi = draw(5), draw(50)   # 10x apart, as two streams' n_c can be

    # 'Spread' is the n_c-invariant reading: both must recover ~0
    assert lo['spread'] < 0.06, f"noise leaked into spread at n_c=5: {lo['spread']}"
    assert hi['spread'] < 0.06, f"n_c=50: {hi['spread']}"

    # 'Failing Frac' travels with n_c on the SAME model -- the trap being warned
    # about. True answer is 0 (every condition is above the bar).
    assert lo['failing_frac'] > 0.2, lo['failing_frac']
    assert hi['failing_frac'] < 0.1, hi['failing_frac']
    assert lo['failing_frac'] > 3 * hi['failing_frac'], \
        (lo['failing_frac'], hi['failing_frac'])
    print(f"PASS spread is n_c-invariant ({lo['spread']:.3f} vs {hi['spread']:.3f}) "
          f"where failing_frac travels ({lo['failing_frac']:.2f} vs {hi['failing_frac']:.2f})")


def test_spread_measures_real_concentration():
    """...and it must still SEE a real split, at any n_c, or it is just a
    complicated zero."""
    for n_c in (2, 4, 40):
        cid = torch.arange(100).repeat_interleave(n_c)
        # half the library dead, half perfect: true sd of p_c is exactly 0.5
        ind = torch.cat([torch.ones(50 * n_c), torch.zeros(50 * n_c)]).bool()
        r = per_condition_fraction(ind, cid, bar=0.5, worst_quantile=WQ)
        assert abs(r['spread'] - 0.5) < 0.01, (n_c, r['spread'])

    # uniform-p_c control at the same n_c: must read ~0, not 0.5
    cid = torch.arange(100).repeat_interleave(4)
    ind = torch.tensor([1, 1, 0, 0] * 100, dtype=torch.bool)
    assert per_condition_fraction(ind, cid, bar=0.5, worst_quantile=WQ)['spread'] < 1e-6

    # singletons carry no information about Var(p): omitted, not 0
    single = per_condition_fraction(torch.tensor([1, 0, 1, 0], dtype=torch.bool),
                                    torch.arange(4), bar=0.5, worst_quantile=WQ)
    assert single['spread'] is None, single['spread']
    print("PASS spread recovers 0.5 on a half-dead library at n_c 2/4/40, 0 when uniform")


def test_conditions_are_not_count_weighted():
    """The only reason this differs from the pooled fraction: each condition
    contributes once, whatever its sample count."""
    cid = torch.tensor([0] * 90 + [1] * 10)
    ind = torch.tensor([1] * 90 + [0] * 10, dtype=torch.bool)  # pooled 0.90

    r = per_condition_fraction(ind, cid, bar=0.5, worst_quantile=WQ)
    assert abs(_pooled(ind) - 0.9) < 1e-6
    assert r['failing_frac'] == 0.5, r['failing_frac']  # one of two conditions dead
    # 'worst' INTERPOLATES (torch.quantile's default) -- over p_c = [0, 1] the
    # 0.25 quantile is 0.25, not either condition's own value. Only matters at
    # a handful of conditions; it is the same convention tb_err_worst uses
    assert abs(r['worst'] - 0.25) < 1e-6, r['worst']
    print("PASS conditions unweighted by count (pooled 0.90, failing_frac 0.50)")


def test_singleton_resolution_degeneracy():
    """DOCUMENTED BLIND SPOT, asserted so it cannot be forgotten: at one sample
    per condition p_c is 0 or 1 and failing_frac IS the pooled bad fraction.
    'Cond * N' is what tells the reader they are in this regime."""
    cid = torch.arange(10)
    ind = torch.tensor([1] * 7 + [0] * 3, dtype=torch.bool)
    r = per_condition_fraction(ind, cid, bar=0.5, worst_quantile=WQ)
    assert abs(r['failing_frac'] - (1.0 - _pooled(ind))) < 1e-6
    assert abs(r['failing_frac'] - 0.3) < 1e-6, r['failing_frac']
    assert r['n_conditions'] == 10
    print("PASS singleton degeneracy is exactly the pooled fraction (0.30)")


def test_degenerate_inputs_are_omitted_not_faked():
    """None means 'no key logged'. A 0 or a nan here would read as a measurement."""
    cid = torch.arange(4).repeat_interleave(2)
    ind = torch.ones(8, dtype=torch.bool)
    assert per_condition_fraction(ind, None, bar=0.5) is None, "no condition axis"
    assert per_condition_fraction(None, cid, bar=0.5) is None, "no indicator"
    assert per_condition_fraction(ind, cid, bar=None) is None, "bar unset = family off"
    assert per_condition_fraction(torch.ones(6, dtype=torch.bool),
                                  torch.zeros(6, dtype=torch.long), bar=0.5) is None, \
        "one condition: failing_frac would be a 0/1 step function"
    assert per_condition_fraction(torch.zeros(0, dtype=torch.bool),
                                  torch.zeros(0, dtype=torch.long), bar=0.5) is None, \
        "empty batch"
    print("PASS degenerate inputs omit the family (None, not 0/nan)")


def test_length_mismatch_raises():
    """MUT-2's backstop: indicator and condition ids come off the same pooled
    batch, so a mismatch is an alignment bug upstream, not a small-batch case."""
    try:
        per_condition_fraction(torch.ones(8, dtype=torch.bool),
                               torch.arange(6), bar=0.5)
    except ValueError as e:
        assert '8' in str(e) and '6' in str(e), str(e)
        print("PASS length mismatch raises")
        return
    raise AssertionError("length mismatch did NOT raise -- a silently mis-grouped "
                         "per-condition metric is worse than no metric")


# --------------------------------------------------------------------------
# the wiring: real Modeller methods, stub state
# --------------------------------------------------------------------------

class _StubModeller:
    """Only the attributes the real methods below actually touch."""
    log_thermo_properties = Modeller.log_thermo_properties
    # log_thermo_properties delegates the physical block to this, which was
    # extracted from it (2026-08-20). Bound as the REAL method, like every other
    # name here: the point of this stub is that the wiring under test is the
    # shipping wiring, so a stubbed stand-in would test a copy of the thing it is
    # meant to protect.
    log_physical_properties = Modeller.log_physical_properties
    log_nonthermal_tail = Modeller.log_nonthermal_tail
    log_test_metrics = Modeller.log_test_metrics
    log_condition_fraction = Modeller.log_condition_fraction
    _reasonable_sample_mask = Modeller._reasonable_sample_mask
    _log_setting = Modeller._log_setting
    _merge_metrics = Modeller._merge_metrics

    def __init__(self, floor=None, data_ndim=2):
        self._floor = floor
        self._settings_log_cache = {}
        self.gfn_model = None  # -> n_dof falls through to data_ndim
        self.energy_function = SimpleNamespace(data_ndim=data_ndim)
        self.args = SimpleNamespace(conditional_worst_quantile=WQ,
                                    nonthermal_entropy_per_dim=4.0,
                                    nonthermal_cond_bar=0.1,
                                    reasonable_cond_bar=0.5)

    def _condition_energy_floor(self, condition_id):
        return self._floor


class _StubBatch:
    """Enough of a crystal sample batch for log_thermo_properties: attribute
    access, dict access and keys(), which is all it uses."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def keys(self):
        return list(self.__dict__)

    def __getitem__(self, k):
        return self.__dict__[k]


def _arr(t):
    return t.cpu().detach().numpy()


def _val(t):
    return t.cpu().detach().item()


def test_log_condition_fraction_emits_the_family():
    m = _StubModeller()
    metrics = {}
    cid = torch.arange(4).repeat_interleave(4)
    ind = torch.tensor([1] * 8 + [0] * 8, dtype=torch.bool)

    m.log_condition_fraction(metrics, _arr, 'Reasonable', ind, cid,
                             bar=0.5, higher_is_worse=False)

    # DID IT RUN
    expected = {'Cond Reasonable Failing Frac', 'Cond Reasonable Worst',
                'Cond Reasonable Spread', 'Cond Reasonable Frac',
                'Cond Reasonable N', 'Cond Reasonable Bar'}
    missing = expected - set(metrics)
    assert not missing, f"family did not fire: missing {sorted(missing)}"

    # IS IT RIGHT
    assert metrics['Cond Reasonable Failing Frac'] == 0.5
    assert metrics['Cond Reasonable Worst'] == 0.0
    assert metrics['Cond Reasonable N'] == 4
    assert metrics['Cond Reasonable Bar'] == 0.5
    assert metrics['Cond Reasonable Frac'].shape == (4,), metrics['Cond Reasonable Frac'].shape

    # the bar is a SETTING: emitted once, then suppressed until it moves
    second = {}
    m.log_condition_fraction(second, _arr, 'Reasonable', ind, cid,
                             bar=0.5, higher_is_worse=False)
    assert 'Cond Reasonable Bar' not in second, "constant bar re-logged every eval"
    assert 'Cond Reasonable Failing Frac' in second, "the series stopped with the setting"
    third = {}
    m.log_condition_fraction(third, _arr, 'Reasonable', ind, cid,
                             bar=0.75, higher_is_worse=False)
    assert third['Cond Reasonable Bar'] == 0.75, "a MOVED bar was suppressed"

    # unconditional run: the whole family stays off the surface
    uncond = {}
    m.log_condition_fraction(uncond, _arr, 'Reasonable', ind, None,
                             bar=0.5, higher_is_worse=False)
    assert uncond == {}, f"unconditional run grew keys: {sorted(uncond)}"
    print("PASS Cond * family fires, values right, bar logged once, absent unconditional")


def test_prefix_namespaces_series_but_not_the_bar():
    """The held-out stream reads against the SAME bar, so the bar must be one
    series, not one per prefix -- otherwise 'the bar moved' shows up twice and
    the two copies can disagree in a panel."""
    m = _StubModeller()
    cid = torch.arange(4).repeat_interleave(4)
    ind = torch.tensor([1] * 8 + [0] * 8, dtype=torch.bool)

    train, test = {}, {}
    m.log_condition_fraction(train, _arr, 'Reasonable', ind, cid, bar=0.5,
                             higher_is_worse=False)
    m.log_condition_fraction(test, _arr, 'Reasonable', ind, cid, bar=0.5,
                             higher_is_worse=False, prefix='eval_test/')

    assert 'Cond Reasonable Failing Frac' in train
    assert 'eval_test/Cond Reasonable Failing Frac' in test, "prefix did not apply"
    assert 'eval_test/Cond Reasonable N' in test
    assert not any(k.startswith('eval_test/') for k in train), sorted(train)
    assert train['Cond Reasonable Bar'] == 0.5
    assert 'eval_test/Cond Reasonable Bar' not in test, "bar duplicated per prefix"
    assert 'Cond Reasonable Bar' not in test, "bar re-emitted for the second stream"
    print("PASS prefix namespaces the series, the bar stays a single channel")


def test_reasonable_mask_is_the_documented_window():
    """Both streams read this one indicator, so a drift here moves the pooled
    fraction and the per-condition family together and silently."""
    m = _StubModeller()
    m.energy_function.energy_function = 'gfn_energy'
    pc = torch.tensor([0.70, 0.70, 0.70, 0.50, 0.99])
    en = torch.tensor([-1., 1., -1., -1., -1.])
    good = m._reasonable_sample_mask(_StubBatch(mol_energy=en, packing_coeff=pc))
    assert good.tolist() == [True, False, True, False, False], good.tolist()

    # mol_energy WINS over the bare energy_function attribute (rescaled vs not);
    # falls back to it only when absent
    disagree = _StubBatch(mol_energy=torch.tensor([-1.]), gfn_energy=torch.tensor([1.]),
                          packing_coeff=torch.tensor([0.7]))
    assert m._reasonable_sample_mask(disagree).tolist() == [True], "bare attr won"
    fallback = _StubBatch(gfn_energy=torch.tensor([1.]), packing_coeff=torch.tensor([0.7]))
    assert m._reasonable_sample_mask(fallback).tolist() == [False], "fallback broken"
    print("PASS reasonable mask window + mol_energy precedence")


def test_log_test_metrics_publishes_the_heldout_family():
    """The held-out site through the REAL method, with sampling stubbed."""
    cid = torch.tensor([100, 100, 101, 101, 102, 102])  # disjoint from train ids
    mol_energy = torch.tensor([-1., -1., -1., 1., 1., 1.])
    batch = _StubBatch(mol_energy=mol_energy, packing_coeff=torch.full((6,), 0.7))

    m = _StubModeller()
    m.energy_function.energy_function = 'mol_energy'
    m.args.eval_num_samples = 6
    m.args.test_eval_num_samples = 6
    m.ema_model = None
    m.test_mol_dataset = object()
    seen_kwargs = {}

    def _fake_sampling(model, discretizer, override_num_samples=None, dataset=None,
                       side_effects=True):
        seen_kwargs.update(dataset=dataset, side_effects=side_effects,
                           n=override_num_samples)
        return {'condition_id': cid}, batch

    m.fwd_eval_sampling = _fake_sampling
    m.args.fwd_loss_coeffs = SimpleNamespace(beta=None)
    seen_streams = []

    def _fake_stats(stats, coeffs):
        seen_streams.append(stats)
        return {'tb_err_worst': 1.0, 'cond_tb_err': 0.5}

    m._eval_conditional_stats = _fake_stats

    metrics = m.log_test_metrics(None, {'condition_id': torch.zeros(6, dtype=torch.long)})

    # the held-out pass must still be measurement-only
    assert seen_kwargs['side_effects'] is False, "held-out pass gained side effects"
    assert seen_kwargs['dataset'] is m.test_mol_dataset and seen_kwargs['n'] == 6

    # NAMESPACE SEPARATION. This method scores the HELD-OUT stream and publishes
    # only 'eval_test/'. It used to also recompute the train-condition stats at a
    # different worst_quantile and republish four 'eval_fwd/' keys, which
    # log_metrics had already written -- the collision _merge_metrics now
    # forbids. Two assertions, because they fail for different reasons: the
    # duplicate PASS over the train batch is gone, and no key leaks.
    assert not [k for k in metrics if k.startswith('eval_fwd/')], \
        [k for k in metrics if k.startswith('eval_fwd/')]
    # the ONE unprefixed key is the shared bar, and it is shared on purpose:
    # _log_setting's cache makes the second stream to reach it a no-op, so it
    # is one series rather than a duplicate channel (log_condition_fraction).
    leaked = [k for k in metrics
              if not k.startswith('eval_test/') and not k.endswith(' Bar')]
    assert not leaked, f"log_test_metrics wrote outside 'eval_test/': {leaked}"
    assert len(seen_streams) == 1, \
        f"train-condition stats recomputed here ({len(seen_streams)} passes, want 1)"

    # DID IT RUN
    assert metrics['eval_test/tb_err_worst'] == 1.0
    for k in ('eval_test/Reasonable Sample Fraction', 'eval_test/Cond Reasonable Failing Frac',
              'eval_test/Cond Reasonable Worst', 'eval_test/Cond Reasonable Spread',
              'eval_test/Cond Reasonable Frac', 'eval_test/Cond Reasonable N'):
        assert k in metrics, f"missing {k}"
    # the non-thermal family has no Emin(c) on held-out conditions and must not
    # be faked into existence
    assert not any('Nonthermal' in k for k in metrics), \
        [k for k in metrics if 'Nonthermal' in k]

    # IS IT RIGHT -- p_c = [1.0, 0.5, 0.0] against a bar of 0.5
    assert metrics['eval_test/Reasonable Sample Fraction'] == 0.5
    assert abs(metrics['eval_test/Cond Reasonable Failing Frac'] - 1 / 3) < 1e-6
    assert metrics['eval_test/Cond Reasonable N'] == 3
    print("PASS held-out family published from log_test_metrics "
          f"(pooled 0.5, conditions failing {metrics['eval_test/Cond Reasonable Failing Frac']:.3f})")


def test_merge_metrics_refuses_a_silent_overwrite():
    """MUT-4: re-introduce the bug this guard exists for and require a FAILURE.

    The defect was two writers publishing eval_fwd/tb_err_worst at different
    worst_quantile values, resolved by dict-update order. Nothing about it was
    detectable from the logs -- both numbers were plausible. So the test is not
    'the helper merges dicts', it is 'the helper REFUSES the merge that shipped'.
    """
    m = _StubModeller()
    metrics = {}

    # the ordinary case still merges, and returns the accumulator
    out = m._merge_metrics(metrics, {'eval_fwd/tb_err_worst': 3.0}, 'log_metrics')
    assert out is metrics and metrics['eval_fwd/tb_err_worst'] == 3.0

    # disjoint namespaces are exactly what the fix produces, and must pass
    m._merge_metrics(metrics, {'eval_test/tb_err_worst': 2.5}, 'log_test_metrics')
    assert metrics['eval_test/tb_err_worst'] == 2.5

    # THE MUTATION: log_test_metrics republishing the train-condition key at its
    # own quantile, exactly as it did before the fix.
    try:
        m._merge_metrics(metrics, {'eval_fwd/tb_err_worst': 2.9}, 'log_test_metrics')
    except AssertionError as e:
        msg = str(e)
    else:
        raise AssertionError("MUT-4 FAILED TO FIRE: the silent overwrite was accepted")

    # the message has to name the key AND both values -- 'which one won' is the
    # question a reader hitting this will be asking
    for want in ('eval_fwd/tb_err_worst', 'log_test_metrics', '3.0', '2.9'):
        assert want in msg, f"collision message omits {want!r}: {msg}"
    # and it must not have half-applied the update before refusing
    assert metrics['eval_fwd/tb_err_worst'] == 3.0, "collision mutated the dict anyway"
    print("PASS MUT-4 duplicate-key merge refused, dict left intact")


def test_reasonable_wiring_through_log_thermo_properties():
    """The reasonable-fraction site driven through the REAL method, so 'the
    indicator the metric groups is the one the pooled fraction was built from'
    is asserted rather than assumed. Nonthermal is off here (no floor), which
    also shows the two families fire independently."""
    cid = torch.tensor([7, 7, 8, 8, 9, 9])
    # packing coeff inside the 0.55-0.95 window for every row, so 'reasonable'
    # turns purely on the energy sign: c7 both good, c8 split, c9 both bad
    mol_energy = torch.tensor([-1., -1., -1., 1., 1., 1.])
    batch = _StubBatch(mol_energy=mol_energy,
                       gfn_energy=mol_energy.clone(),
                       packing_coeff=torch.full((6,), 0.7),
                       reduction_en=torch.full((6,), 1e-3))

    m = _StubModeller(floor=None)  # pre-bootstrap: nonthermal family off
    m.energy_function.energy_function = 'mol_energy'
    metrics = {}
    m.log_thermo_properties(_arr, {'condition_id': cid}, torch.zeros(6),
                            torch.zeros(6), -mol_energy, metrics, batch, _val)

    # DID IT RUN, and did the pooled metric survive
    assert 'Reasonable Sample Fraction' in metrics
    for k in ('Cond Reasonable Failing Frac', 'Cond Reasonable Worst',
              'Cond Reasonable Frac', 'Cond Reasonable N', 'Cond Reasonable Bar'):
        assert k in metrics, f"missing {k}"
    assert not any(k.startswith('Cond Nonthermal') for k in metrics), \
        "nonthermal family fired with no energy floor"

    # IS IT RIGHT -- p_c = [1.0, 0.5, 0.0] against a bar of 0.5
    assert metrics['Reasonable Sample Fraction'] == 0.5
    assert abs(metrics['Cond Reasonable Failing Frac'] - 1 / 3) < 1e-6, \
        metrics['Cond Reasonable Failing Frac']
    assert abs(metrics['Cond Reasonable Worst'] - 0.25) < 1e-6, metrics['Cond Reasonable Worst']
    assert metrics['Cond Reasonable N'] == 3
    assert sorted(metrics['Cond Reasonable Frac'].tolist()) == [0.0, 0.5, 1.0]
    print("PASS reasonable wiring through log_thermo_properties "
          f"(pooled 0.5, conditions failing {metrics['Cond Reasonable Failing Frac']:.3f})")


def test_nonthermal_wiring_subsets_by_referenced_rows():
    """log_nonthermal_tail drops rows with no Emin(c) record. The condition ids
    have to be dropped with them -- through the REAL method, at T = 1 so
    u = E - Emin(c) exactly, with u* = 4.0 * 2 = 8 nats."""
    # rows: c10 x1 (nonthermal), c11 x3 (1 nonthermal), c12 x2 (no record)
    cid = torch.tensor([10, 11, 11, 11, 12, 12])
    floor = torch.tensor([0., 0., 0., 0., float('inf'), float('inf')])
    energy = torch.tensor([20., 20., 1., 1., 20., 20.])  # u = 20 or 1 vs u* = 8
    fwd_stats = {'condition_id': cid}
    log_T = torch.zeros(6)          # T = 1
    log_r = -energy                 # E = -log_r * T

    m = _StubModeller(floor=floor)
    metrics = {}
    m.log_nonthermal_tail(_arr, fwd_stats, log_T, log_r, metrics)

    # DID IT RUN -- and did the pre-existing family survive the insertion
    for k in ('Nonthermal Fraction', 'Nonthermal Threshold', 'Excess Energy Nats P99',
              'Cond Nonthermal Failing Frac', 'Cond Nonthermal Worst',
              'Cond Nonthermal Frac', 'Cond Nonthermal N', 'Cond Nonthermal Bar'):
        assert k in metrics, f"missing {k}"

    # IS IT RIGHT
    assert metrics['Nonthermal Threshold'] == 8.0
    assert abs(metrics['Excess Energy Referenced Fraction'] - 4 / 6) < 1e-6
    assert metrics['Nonthermal Fraction'] == 0.5, metrics['Nonthermal Fraction']
    assert metrics['Cond Nonthermal N'] == 2, "unreferenced condition 12 was scored"
    # c10 p = 1.0, c11 p = 1/3 -- both above the 0.1 bar, so every SCORED
    # condition is contaminated while the pooled fraction reads a benign 0.5
    assert metrics['Cond Nonthermal Failing Frac'] == 1.0
    want_worst = torch.quantile(torch.tensor([1.0, 1 / 3]), 1.0 - WQ).item()
    assert abs(metrics['Cond Nonthermal Worst'] - want_worst) < 1e-6
    assert metrics['Cond Nonthermal Failing Frac'] != metrics['Nonthermal Fraction'], \
        "per-condition and pooled agree here by accident -- the test discriminates nothing"

    # MUT-2: forget the `seen` subset and the ids no longer line up with u
    u = torch.tensor([20., 20., 1., 1.])
    try:
        per_condition_fraction(u > 8.0, cid, bar=0.1, worst_quantile=WQ,
                               higher_is_worse=True)
    except ValueError:
        pass
    else:
        raise AssertionError("MUT-2 is inert: un-subsetted ids were accepted")
    print("PASS nonthermal wiring subsets by referenced rows "
          f"(pooled {metrics['Nonthermal Fraction']}, "
          f"conditions failing {metrics['Cond Nonthermal Failing Frac']})")


def test_nonthermal_family_absent_without_records():
    """Pre-bootstrap: no floor at all, and no floor on any row. Neither may
    produce a zero that reads as 'no contamination'."""
    cid = torch.tensor([10, 10, 11, 11])
    log_T, log_r = torch.zeros(4), -torch.tensor([20., 20., 20., 20.])

    no_tracker = {}
    _StubModeller(floor=None).log_nonthermal_tail(
        _arr, {'condition_id': cid}, log_T, log_r, no_tracker)
    assert no_tracker == {}, f"pre-bootstrap emitted {sorted(no_tracker)}"

    none_seen = {}
    _StubModeller(floor=torch.full((4,), float('inf'))).log_nonthermal_tail(
        _arr, {'condition_id': cid}, log_T, log_r, none_seen)
    assert none_seen == {'Excess Energy Referenced Fraction': 0.0}, sorted(none_seen)
    print("PASS nonthermal family absent (not 0) with no per-condition records")


if __name__ == '__main__':
    test_separates_concentrated_from_spread()
    test_orientation_and_worst_tail()
    test_spread_recovers_geometry_and_failing_frac_does_not_travel()
    test_spread_measures_real_concentration()
    test_conditions_are_not_count_weighted()
    test_singleton_resolution_degeneracy()
    test_degenerate_inputs_are_omitted_not_faked()
    test_length_mismatch_raises()
    test_log_condition_fraction_emits_the_family()
    test_prefix_namespaces_series_but_not_the_bar()
    test_reasonable_mask_is_the_documented_window()
    test_log_test_metrics_publishes_the_heldout_family()
    test_merge_metrics_refuses_a_silent_overwrite()
    test_reasonable_wiring_through_log_thermo_properties()
    test_nonthermal_wiring_subsets_by_referenced_rows()
    test_nonthermal_family_absent_without_records()
    print("\nALL CONDITION FRACTION TESTS PASSED")
