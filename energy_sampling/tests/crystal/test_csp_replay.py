"""The replay half of the mock structure-prediction evaluator (eval/cond_panel/csp.py), on
constructed pools whose answers are known: which ends are one minimum, what a stopping rule
spends and whether it is credited, what a selection rule relaxes and charges, and the
recovered share and energy-call count of a whole replay; and the density modes of
constructed draws.

Nothing here draws from a model or relaxes a crystal: `collect` is covered by its own run."""
import numpy as np
import pytest

from eval.cond_panel import csp

pytestmark = pytest.mark.fast   # torch imported by the module, nothing built

T = 2.0


def descent(start, end, reach, steps=120):
    """A trajectory falling linearly from `start` to `end` over `reach` steps, then flat."""
    t = np.arange(steps)
    return np.where(t < reach, start + (end - start) * t / reach, end).astype(np.float32)


def pool(sampler, ends, packing, reaches, features=None, start=100.0):
    traj = np.stack([descent(start + i, e, r) for i, (e, r) in enumerate(zip(ends, reaches))], axis=1)
    n = len(ends)
    return {'sampler': sampler, 'split': 'train', 'traj': traj, 'e_end': np.array(ends, dtype=np.float64),
            'cp_end': np.array(packing, dtype=np.float64), 'e_ref': min(ends),
            'features': np.zeros((n, 12), dtype=np.float32) if features is None else features}


def test_ends_are_one_minimum_only_when_energy_and_packing_agree():
    e = np.array([-10.00, -10.05, -9.00, -10.02, -10.30])
    cp = np.array([0.700, 0.701, 0.700, 0.750, 0.700])
    group, rep = csp.assign_minima(e, cp)
    assert group[0] == group[1], 'same energy within E_TOL and same packing'
    assert group[3] != group[0], 'same energy, another packing coefficient: another minimum'
    assert len({group[0], group[2], group[3], group[4]}) == 4
    assert np.all(np.diff(rep) >= 0) and rep[0] == -10.30


def test_full_relaxation_spends_every_step_and_is_credited():
    traj = np.stack([descent(100, 0, 20), descent(100, 0, 60)], axis=1)
    calls, credited, at_stop = csp.stop_steps(traj, np.zeros(2), T, 'full')
    assert calls.tolist() == [120, 120] and credited.all() and np.allclose(at_stop, 0.0)


def test_plateau_stops_one_window_after_the_descent_ends():
    traj = descent(100, 0, 20)[:, None]
    calls, credited, _ = csp.stop_steps(traj, np.zeros(1), T, 'plateau')
    # best energy is flat from step index 20; the rule needs PLATEAU_WINDOW flat steps to see it
    assert calls[0] == 20 + csp.PLATEAU_WINDOW + 1 and credited[0]


def test_plateau_that_stops_on_a_false_floor_is_charged_and_not_credited():
    t = np.arange(120)
    stall = csp.PLATEAU_WINDOW + 20                                       # flat at 50 for longer than the window, then the real end
    stalled = np.where(t < stall, 50.0, 0.0).astype(np.float32)[:, None]
    calls, credited, at_stop = csp.stop_steps(stalled, np.zeros(1), T, 'plateau')
    assert calls[0] == csp.PLATEAU_WINDOW + 1 and not credited[0] and at_stop[0] == 50.0


def test_oracle_stops_at_the_first_step_within_the_credit():
    traj = descent(100, 0, 20)[:, None]
    calls, credited, _ = csp.stop_steps(traj, np.zeros(1), T, 'oracle')
    # 100 - 5 t <= CREDIT * T = 0.2 first at t = 20
    assert calls[0] == 21 and credited[0]


def test_selection_rules_relax_and_charge_what_they_say():
    p = pool('a', [0.0] * 8, [0.7] * 8, [10] * 8)
    subset = np.arange(8)
    chosen, overhead = csp.select(p, subset, 'all', 1.0)
    assert chosen.tolist() == list(range(8)) and overhead == 0
    chosen, overhead = csp.select(p, subset, 'screen', 0.25)
    assert sorted(chosen.tolist()) == [0, 1] and overhead == 6, 'the two lowest starting energies; their own first call is not charged twice'
    with pytest.raises(ValueError):
        csp.select(p, subset, 'cluster', 0.25)           # clustering a few hundred collected starts is not a strategy


def test_periodic_latents_are_close_across_the_wrap():
    x = np.zeros((2, 12))
    x[0, 7], x[1, 7] = -0.49, 0.49                       # v has period 1: 0.02 apart, not 0.98
    z = csp.embed(x)
    assert np.linalg.norm(z[0] - z[1]) == pytest.approx(0.02, rel=0.02)
    y = np.zeros((2, 12))
    y[1, 0] = 0.5                                        # a linear latent of range 2
    assert np.linalg.norm(np.diff(csp.embed(y), axis=0)) == pytest.approx(0.25)


def test_replay_counts_recovered_minima_and_energy_calls():
    # three minima inside a 3 kT window (0, 2, 4 raw units at T = 2) and one outside it (30)
    wide = pool('wide', [0.0, 2.0, 4.0, 30.0], [0.70, 0.71, 0.72, 0.73], [10, 10, 10, 10])
    narrow = pool('narrow', [0.0, 0.0, 0.0, 30.0], [0.70, 0.70, 0.70, 0.73], [10, 10, 10, 10])
    res, target, splits, samplers = csp.replay({"mol": {"wide": wide, "narrow": narrow}}, T, window=3.0, repeats=1)
    assert target.tolist() == [3] and samplers == ['narrow', 'wide']
    full = 'every start, every step'
    assert res[('wide', full)]['recall'][0, 0] == 1.0
    assert res[('narrow', full)]['recall'][0, 0] == pytest.approx(1 / 3)
    assert res[('wide', full)]['calls'][0, 0] == 4 * 120
    plateau = 'every start, stop on a plateau'
    assert res[('wide', plateau)]['calls'][0, 0] == 4 * (10 + csp.PLATEAU_WINDOW + 1)
    assert res[('wide', plateau)]['recall'][0, 0] == 1.0
    # the lowest energy in hand is the floor itself once a start has reached it
    assert res[('wide', full)]['best'][0, 0] == 0.0 and res[('wide', full)]['floor'][0, 0] == 1.0


def test_replay_refuses_a_molecule_one_sampler_lacks():
    a = pool('a', [0.0], [0.7], [10])
    b = pool('b', [0.0], [0.7], [10])
    with pytest.raises(AssertionError):
        csp.replay({"m1": {"a": a, "b": b}, "m2": {"a": a}}, T, window=3.0, repeats=1)


def test_density_modes_are_the_clusters_of_the_draws_largest_first():
    rng = np.random.default_rng(0)
    feats = np.zeros((301, 12))
    feats[:200, 0] = -0.5 + 0.005 * rng.standard_normal(200)     # a cluster of 200 and a cluster of 100, half a range apart ...
    feats[200:300, 0] = 0.5 + 0.005 * rng.standard_normal(100)
    feats[:300, 3] = 0.005 * rng.standard_normal(300)
    feats[300, 0], feats[300, 1] = 0.0, 0.9                       # ... and one draw with no neighbour inside the cutoff
    modes, mass, dens = csp.density_modes(feats, device='cpu')
    assert mass.tolist() == [200, 100, 1]
    assert modes[0] < 200 <= modes[1] < 300 and modes[2] == 300
    assert dens[modes[0]] == dens[:200].max() and dens[modes[1]] == dens[200:300].max(), 'a mode is the densest draw of its cluster'
    assert dens[300] == pytest.approx(1.0, abs=1e-4), 'a draw alone has only its own kernel weight'


def test_density_modes_keep_two_clusters_apart_only_beyond_the_cutoff():
    feats = np.zeros((200, 12))
    gap = 0.5 * csp.D_CUT * 2.0                                   # half the cutoff, in latent units of a range-2 linear latent
    feats[100:, 0] = gap
    assert len(csp.density_modes(feats, device='cpu')[0]) == 1, 'two groups inside one cutoff are one cluster'
    feats[100:, 0] = 4 * gap
    assert csp.density_modes(feats, device='cpu')[1].tolist() == [100, 100]


def test_a_cluster_first_pool_is_charged_its_draws_and_read_in_its_stored_order():
    # 4 relaxed modes of 1,000 draws; the first two (largest clusters) hold the two target minima
    first = pool('model+modes1000', [0.0, 2.0, 30.0, 30.0], [0.70, 0.71, 0.73, 0.73], [10, 10, 10, 10])
    first['from_draws'] = 1000
    plain = pool('random', [0.0, 30.0, 30.0, 30.0], [0.70, 0.73, 0.73, 0.73], [10, 10, 10, 10])
    res, target, _, _ = csp.replay({'mol': {'model+modes1000': first, 'random': plain}}, T, window=3.0, repeats=1)
    assert target.tolist() == [2]
    full = 'every start, every step'
    assert res[('model+modes1000', full)]['draws'].tolist() == [1000]
    assert res[('model+modes1000', full)]['recall'][0, -1] == 1.0
    labels = {lab for s, lab in res if s == 'model+modes1000'}
    assert all('screen' not in lab for lab in labels), 'no second selection on a pre-selected pool'
    assert res[('random', full)]['recall'][0, -1] == 0.5


def test_the_cascade_retires_a_relaxation_far_above_the_reference_and_charges_it_to_that_step():
    # 8 starts: the first two (a quarter) run without the cascade and set the reference at their lowest end, 0
    ends = [0.0, 0.0, 0.0, 0.0, 60.0, 60.0, 60.0, 60.0]
    p = pool('a', ends, [0.70] * 4 + [0.75] * 4, [10] * 8, start=60.0)
    for i in range(4, 8):
        p['traj'][:, i] = 60.0                                            # never comes down: 30 kT above the reference at T = 2
    stops = csp.stop_steps(p['traj'], p['e_end'], T, 'plateau')
    spent, got, held = csp.spend(p, np.arange(8), 'cascade', 1.0, stops, T)
    plateau = 10 + csp.PLATEAU_WINDOW + 1
    first_step = csp.CASCADE[0][0]
    # starts 0-3 reach their end and stop on the plateau; 4-7 are retired at the first cascade step
    assert spent == 4 * plateau + 4 * first_step
    assert sorted(got.tolist()) == [0, 1, 2, 3] and held == 0.0
    # without the cascade the four stuck starts would have run to their own plateau
    spent_all, _, _ = csp.spend(p, np.arange(8), 'all', 1.0, stops, T)
    assert spent_all > spent


def test_floor_in_hand_is_a_chance_over_subsets_not_a_property_of_their_mean():
    # one start of sixteen reaches the floor; the rest end 20 kT up. Eight draws hold it half the time.
    ends = [0.0] + [40.0] * 15
    p = pool('a', ends, [0.70] + [0.75] * 15, [10] * 16)
    res, _, _, _ = csp.replay({'mol': {'a': p}}, T, window=3.0, repeats=400, seed=1)
    full = 'every start, every step'
    at8 = res[('a', full)]['draws'].tolist().index(8)
    assert res[('a', full)]['floor'][0, at8] == pytest.approx(0.5, abs=0.08)
    assert res[('a', full)]['floor'][0, -1] == 1.0


def test_selection_bounds_take_the_cheapest_credited_start_of_each_minimum():
    # four minima: A (starts 0, 1) and B (2, 3) inside the 3 kT window, C (4) and D (5) 20 kT up and apart in packing
    feats = np.zeros((6, 12), dtype=np.float32)
    feats[:, 0] = [0.00, 0.01, 0.50, 0.51, 1.00, 1.01]                    # each start's nearest other start is its pair
    p = pool('a', [0.0, 0.0, 4.0, 4.0, 40.0, 40.0], [0.70, 0.70, 0.72, 0.72, 0.75, 0.76], [10, 30, 20, 20, 10, 10], features=feats)
    t = np.arange(p['traj'].shape[0])
    p['traj'][:, 1] = np.where(t < csp.PLATEAU_WINDOW + 20, 50.0, 0.0)     # start 1 stalls: stopped early, cheap, not credited
    groups, floor, target = csp.known_minima({'a': p}, ['a'], T, 3.0)
    assert floor == 0.0 and len(target) == 2 and groups['a'][0] == groups['a'][1] != groups['a'][2]
    first = pool('a+modes1000', [0.0], [0.70], [10])
    first['from_draws'] = 1000
    plain = pool('a', [0.0], [0.70], [10])
    assert set(csp.selection_bounds({'mol': {'a': plain, 'a+modes1000': first}}, T, 3.0)) == {'a'}, 'a pre-selected pool has no bound'
    b = csp.selection_bounds({'mol': {'a': p}}, T, 3.0)['a']
    w = csp.PLATEAU_WINDOW + 1
    assert b['starts'] == 6 and b['unsettled'] == 0 and b['distinct'] == 4 and b['in_target'] == 2 and b['low'] == pytest.approx(4 / 6)
    assert b['every'] == (10 + w) + w + 2 * (20 + w) + 2 * (10 + w)
    assert b['per_minimum'] == (10 + w) + (20 + w) + 2 * (10 + w), 'minimum A through start 0: start 1 is cheaper but not credited'
    assert b['per_target'] == (10 + w) + (20 + w)
    # pairs (0, 1) and (2, 3) share a minimum, (4, 5) are neighbours in different minima
    assert b['neighbour'] == pytest.approx(4 / 6) and b['any_two'] == pytest.approx(4 / 30)


def test_a_relaxation_still_descending_at_the_last_step_is_no_minimum():
    # two starts settle at 0; the third is still falling when the record ends, 1 kT up and at another packing
    p = pool('a', [0.0, 0.0, 2.0], [0.70, 0.70, 0.72], [10, 10, 10])
    steps = p['traj'].shape[0]
    p['traj'][:, 2] = np.linspace(100.0, 2.0, steps)                       # falls 0.8 a step to the end
    assert csp.settled(p['traj']).tolist() == [True, True, False]
    groups, floor, target = csp.known_minima({'a': p}, ['a'], T, 3.0)
    assert groups['a'].tolist() == [0, 0, -1] and floor == 0.0 and len(target) == 1, 'its end inside the window is not a target minimum'
    res, size, _, _ = csp.replay({'mol': {'a': p}}, T, window=3.0, repeats=1)
    full = 'every start, every step'
    assert size.tolist() == [1] and res[('a', full)]['recall'][0, -1] == 1.0
    assert res[('a', full)]['calls'][0, -1] == 3 * steps, 'its energy calls are spent all the same'
    b = csp.selection_bounds({'mol': {'a': p}}, T, 3.0)['a']
    assert b['unsettled'] == pytest.approx(1 / 3) and b['distinct'] == 1 and b['any_two'] == pytest.approx(2 / 6)
    # the floor is the lowest energy of any end: an unsettled end below every minimum moves it
    p['e_end'][2] = -5.0
    p['traj'][:, 2] = np.linspace(100.0, -5.0, steps)
    groups, floor, target = csp.known_minima({'a': p}, ['a'], T, 3.0)
    assert floor == -5.0 and groups['a'].tolist() == [0, 0, -1] and len(target) == 1    # the minimum at 0 is 2.5 kT above it


def test_replay_refuses_a_molecule_with_no_minimum_inside_the_window():
    # the stored reference energy is 50 kT below the only end: nothing is within 3 kT of the floor
    p = pool('a', [0.0], [0.70], [10])
    p['e_ref'] = -100.0
    with pytest.raises(AssertionError, match='no settled minimum'):
        csp.replay({'mol': {'a': p}}, T, window=3.0, repeats=1)
