"""cfg:condition_log_z.best_energy_from_anchors_on_switch -- a run that starts from ANOTHER run's checkpoint can take
its per-condition best energy from the restored anchor buffer instead of the restored record.

The record is a running minimum over every energy a run has scored, and no row is kept with it, so one absurd
evaluation poisons it for good. ppl_acr_n5 (2026-10-12) inherited -1.33e6 from its MLE seed against an anchor minimum
of -45.3; with the prior buffer's gates written as "excess above the record", no noised anchor was admitted and every
stored row expired as stale: 2.9M rows fell to 43k in 37k steps.
"""
import types

import pytest

torch = pytest.importorskip("torch")

from energy_sampling.checkpointing import best_energy_from_anchors_on_switch  # noqa: E402


class Anchors:
    def __init__(self, condition_id, energy):
        self.condition_id = torch.tensor(condition_id)
        self.energy = torch.tensor(energy, dtype=torch.float32)

    def __len__(self):
        return len(self.energy)


def _modeller(best, anchors, key=True, alias=True, run_name='ft_arm'):
    best = torch.tensor(best, dtype=torch.float32)
    tracker = types.SimpleNamespace(best_energy=best.clone(), best_energy_phys=best.clone(), phys_is_alias=alias)
    args = types.SimpleNamespace(condition_log_z=types.SimpleNamespace(best_energy_from_anchors_on_switch=key))
    return types.SimpleNamespace(args=args, run_name=run_name, condition_log_z=tracker, anchor_buffer=anchors)


SEED = {'run_name': 'seed_arm'}
INF = float('inf')


def test_a_poisoned_record_becomes_the_anchor_minimum():
    m = _modeller([-1.32755e6], Anchors([0, 0, 0], [-29.1, -45.26, -9.3]))
    assert best_energy_from_anchors_on_switch(m, SEED, 'seed.pt') is True
    assert m.condition_log_z.best_energy.tolist() == pytest.approx([-45.26])
    assert m.condition_log_z.best_energy_phys.tolist() == pytest.approx([-45.26])
    # what the prior buffer's gates then see, at ramp_floor 100: a noised anchor at -30 is admitted, a stored row at
    # -9.3 is not stale
    floor = float(m.condition_log_z.best_energy[0])
    assert -30.0 < floor + 100 and not (-9.3 - floor >= 100)


def test_each_condition_takes_its_own_minimum_and_one_without_anchors_is_unvisited():
    m = _modeller([-5.0, -1e9, -7.0, INF], Anchors([0, 1, 1, 0], [-3.0, -20.0, -22.0, -4.0]))
    best_energy_from_anchors_on_switch(m, SEED, 'seed.pt')
    assert m.condition_log_z.best_energy.tolist() == pytest.approx([-4.0, -22.0, INF, INF])


def test_the_runs_own_checkpoint_and_an_unset_key_change_nothing():
    for m, ck in ((_modeller([-1e6], Anchors([0], [-45.0]), run_name='seed_arm'), SEED),
                  (_modeller([-1e6], Anchors([0], [-45.0]), key=False), SEED)):
        assert best_energy_from_anchors_on_switch(m, ck, 'x.pt') is False
        assert m.condition_log_z.best_energy.tolist() == [-1e6] == m.condition_log_z.best_energy_phys.tolist()


def test_it_refuses_where_it_could_not_do_what_the_key_says():
    with pytest.raises(ValueError, match='no anchor buffer'):
        best_energy_from_anchors_on_switch(_modeller([-1e6], None), SEED, 'x.pt')
    with pytest.raises(ValueError, match='no anchor buffer'):
        best_energy_from_anchors_on_switch(_modeller([-1e6], Anchors([], [])), SEED, 'x.pt')
    with pytest.raises(ValueError, match='separate from its mixture minimum'):
        best_energy_from_anchors_on_switch(_modeller([-1e6], Anchors([0], [-45.0]), alias=False), SEED, 'x.pt')
