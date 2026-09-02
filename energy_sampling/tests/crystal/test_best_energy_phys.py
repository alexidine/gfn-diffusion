"""
The per-condition PHYSICAL minimum, and the two guards that stop it lying.

`best_energy_phys` exists because Emin(c) feeds IRREVERSIBLE operations
(AnchorBuffer.thin evicts one-way; _expire_stale_prior_rows purges) whose safety
argument is monotonicity -- and only the lambda=1 leg is monotone once the energy
definition itself starts moving under a lambda anneal.

While nothing supplies it the tensor is a bit-for-bit ALIAS of best_energy. That
is correct on a lambda-free run and a LIE on a mixing one, and the flag alone
cannot tell those apart: its literal meaning is "no caller passed energy_phys".
So two guards close the gap, and this file pins both.

    pytest tests/crystal/test_best_energy_phys.py
"""
import os
import sys

import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling.buffer import ConditionLogZTracker  # noqa: E402

N = 8


def tracker(**kw):
    return ConditionLogZTracker(library_size=N, **kw)


def test_alias_is_exact_while_nothing_supplies_the_physical_leg():
    """The lambda-free contract: the two tensors must be indistinguishable, so a
    lambda-free run is bit-identical to one that never had this feature."""
    t = tracker()
    cid = torch.tensor([0, 1, 1, 2, 2, 2])
    e = torch.tensor([5.0, -3.0, 2.0, 0.5, -1.5, 9.0])
    t.update_best_energy(cid, e)
    assert torch.equal(t.best_energy, t.best_energy_phys)
    assert t.phys_is_alias


def test_supplying_the_leg_separates_the_two_minima():
    t = tracker()
    cid = torch.tensor([0, 0, 1])
    mixed = torch.tensor([1.0, 2.0, 3.0])
    phys = torch.tensor([40.0, 10.0, 20.0])
    t.update_best_energy(cid, mixed, energy_phys=phys)
    assert not t.phys_is_alias
    assert float(t.best_energy[0]) == 1.0 and float(t.best_energy_phys[0]) == 10.0
    # the argmin genuinely differs between currencies -- that is the whole reason
    # a single tensor cannot serve both, and it is why a mixed minimum cannot be
    # converted after the fact
    assert float(t.best_energy_phys[1]) == 20.0


def test_a_mixing_run_cannot_silently_alias():
    """⚠ THE GUARD THAT MATTERS. A run whose energy function carries a prior_flow
    has two DIFFERENT quantities. Left unwired it would write mixed energies into
    best_energy_phys and then stamp phys_is_alias=True -- asserting exactly the
    claim the flag exists to make false. The trainer sets requires_phys_energy
    from the presence of a prior_flow; this pins that it then refuses."""
    t = tracker()
    t.requires_phys_energy = True
    with pytest.raises(ValueError) as e:
        t.update_best_energy(torch.tensor([0]), torch.tensor([1.0]))
    assert 'physical_energy' in str(e.value), 'the error must say what to pass'
    # ... and accepts once the leg is supplied
    t.update_best_energy(torch.tensor([0]), torch.tensor([1.0]),
                         energy_phys=torch.tensor([7.0]))
    assert not t.phys_is_alias and float(t.best_energy_phys[0]) == 7.0


def test_lookup_refuses_to_serve_the_mixture_under_the_physical_name():
    t = tracker()
    t.update_best_energy(torch.tensor([0]), torch.tensor([1.0]))
    with pytest.raises(ValueError) as e:
        t.lookup_best_energy(torch.tensor([0]), physical=True)
    assert 'ALIAS' in str(e.value)
    t.lookup_best_energy(torch.tensor([0]))          # the mixture is still fine
    t2 = tracker()
    t2.update_best_energy(torch.tensor([0]), torch.tensor([1.0]),
                          energy_phys=torch.tensor([3.0]))
    best, mask = t2.lookup_best_energy(torch.tensor([0]), physical=True)
    assert bool(mask[0]) and float(best[0]) == 3.0


def test_non_finite_in_either_leg_drops_the_row():
    """amin propagates NaN, so one bad row would pin a condition's minimum for the
    rest of the run. The `&` also keeps the two visited sets identical."""
    t = tracker()
    t.requires_phys_energy = True
    cid = torch.tensor([0, 1, 2])
    mixed = torch.tensor([1.0, float('nan'), 3.0])
    phys = torch.tensor([float('inf'), 5.0, 6.0])
    t.update_best_energy(cid, mixed, energy_phys=phys)
    # row 0 dropped (inf phys), row 1 dropped (nan mixed), row 2 kept in BOTH
    assert torch.isinf(t.best_energy[0]) and torch.isinf(t.best_energy_phys[0])
    assert torch.isinf(t.best_energy[1]) and torch.isinf(t.best_energy_phys[1])
    assert float(t.best_energy[2]) == 3.0 and float(t.best_energy_phys[2]) == 6.0
    visited_mixed = torch.isfinite(t.best_energy)
    visited_phys = torch.isfinite(t.best_energy_phys)
    assert torch.equal(visited_mixed, visited_phys), \
        'the two visited sets diverged; lookup masks and telemetry assume they cannot'


def test_state_round_trip_keeps_both_streams_and_the_flag():
    t = tracker()
    t.update_best_energy(torch.tensor([0, 1]), torch.tensor([1.0, 2.0]),
                         energy_phys=torch.tensor([9.0, 8.0]))
    back = ConditionLogZTracker.from_state_dict(t.state_dict(), current_step=5)
    assert torch.equal(back.best_energy, t.best_energy)
    assert torch.equal(back.best_energy_phys, t.best_energy_phys)
    assert back.phys_is_alias is False
    # requires_phys_energy is a fact about the LIVE run, not stored state -- the
    # trainer re-asserts it after loading, so it must come back False here
    assert back.requires_phys_energy is False


def test_a_pre_change_checkpoint_loads_as_an_alias():
    """Legacy state has neither key. Cloning best_energy is exact ONLY for a
    lambda-free provenance, which the loader cannot verify -- so it must come
    back flagged as an alias, leaving the requires_phys_energy guard to catch a
    mixing run on its first update."""
    t = tracker()
    t.update_best_energy(torch.tensor([0]), torch.tensor([4.0]))
    state = t.state_dict()
    del state['best_energy_phys'], state['phys_is_alias']
    back = ConditionLogZTracker.from_state_dict(state, current_step=1)
    assert torch.equal(back.best_energy_phys, back.best_energy)
    assert back.phys_is_alias is True
    # and the clone must be INDEPENDENT, or later updates would couple the streams
    back.update_best_energy(torch.tensor([0]), torch.tensor([1.0]),
                            energy_phys=torch.tensor([99.0]))
    assert float(back.best_energy[0]) == 1.0
    assert float(back.best_energy_phys[0]) == 4.0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
