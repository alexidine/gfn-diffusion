"""churn_rate is configured against the ENTRY batch; the draw grows past it.

The occupancy bar reads len(buffer) / LIVE batch, so a frozen churn made a
2.0-batch bar unreachable: steady state is O/B_live = (churn/B_live)*(tau/N_eff),
which at churn 1000 against a grown 4000 needs tau/N_eff = 8 while the arms ship
3. occupancy_min_batches then fires forever and drags the cadence to
N_eff = churn*tau/(2*B_live) = 0.375*N -- measured 7.5/18.3/33.1/72.9 against a
configured 20/50/100/200 across the whole rr_sep08 ELJ ladder.

What is pinned here is the RATIO, not equality: store-all configs (churn ==
batch, which the rr07 contract asserts) track the live batch exactly, while a
config that deliberately admits a fraction -- the conformer route ships churn 80
against batches of 16-1000 -- keeps that fraction instead of being silently
promoted to store-all.
"""
import pytest


def _admit(churn_rate, entry_batch, live_batch, eligible):
    """The arithmetic of train.py's admission cap, in isolation."""
    entry_b = max(1, int(entry_batch or 1))
    churn_live = int(round(float(churn_rate) * live_batch / entry_b))
    return min(eligible, max(1, churn_live))


def test_store_all_follows_the_batch_up():
    """churn == entry batch means store-all; it must stay store-all at 4x."""
    assert _admit(1000, 1000, 4000, eligible=4000) == 4000


def test_store_all_follows_the_batch_down():
    assert _admit(1000, 1000, 500, eligible=500) == 500


def test_the_frozen_behaviour_is_what_it_replaces():
    """The defect, stated as a number: a quarter of each grown batch admitted."""
    frozen = min(4000, 1000)
    assert frozen == 1000 and _admit(1000, 1000, 4000, eligible=4000) == 4000


def test_a_deliberate_fraction_is_preserved_not_promoted():
    """conformer: churn 80 against batch 256. At 512 it admits 160, not 512."""
    assert _admit(80, 256, 512, eligible=512) == 160


def test_it_never_admits_more_than_is_eligible():
    assert _admit(1000, 1000, 4000, eligible=37) == 37


def test_it_never_admits_zero():
    """A tiny ratio must still make progress or the buffer never fills."""
    assert _admit(1, 4000, 8, eligible=8) == 1


@pytest.mark.parametrize('tau_over_n,n', [(3, 20), (3, 50), (3, 100), (3, 200)])
def test_the_occupancy_bar_is_now_reachable_at_the_shipped_ratio(tau_over_n, n):
    """O/B_live = (churn_live/B_live) * (tau/N). With churn tracking the batch the
    first factor is 1, so the bar is met at tau/N = 3 > 2.0 and the trigger never
    fires -- which is the whole point. Frozen, the factor was 0.25 and 3*0.25 =
    0.75 < 2.0, unreachable at any cadence the config asked for."""
    live_b = 4000
    churn_live = _admit(1000, 1000, live_b, eligible=live_b)
    assert (churn_live / live_b) * tau_over_n >= 2.0
    assert (1000 / live_b) * tau_over_n < 2.0        # the defect, for contrast
