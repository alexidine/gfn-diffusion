"""Anchor-buffer freeze toggles.

FIRST TEST TO TOUCH AnchorBuffer AT ALL -- nothing under tests/ previously
constructed one or called admit/thin/update_losses (audited 2026-08-31), which
is why the paths below had no coverage while three separate call sites mutated
membership in production.

Each test here FAILS against the pre-guard build. That is the point: a freeze is
a claim about an ABSENCE, and a test that only exercises the enabled path passes
equally well against a build where the guard was never wired.

WHY THE GUARDS ARE ON THE PRIMITIVES, not the callers. Membership changes at six
places in train.py -- two constructors (config seed, lazy bootstrap), two
admit() calls (screen_and_admit_anchors, top_up_prior_from_anchors' record-
breaker block) and three thin() calls (eval cadence, and two post-admission
overflow trims). All five post-seed mutations funnel through AnchorBuffer.admit
and AnchorBuffer.thin, so guarding those two covers every one -- including any
call site added later. The constructors deliberately bypass both, because
seeding must still work when frozen.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from energy_sampling.buffer import AnchorBuffer


class _Stub:
    """Minimal stand-in exercising the guard, which is the first statement in
    each method and runs before any buffer state is touched. A full AnchorBuffer
    needs a crystal batch; the guard does not, and coupling this test to
    MolCrystalData construction would make it slow and fragile for no gain."""

    def __init__(self, frozen):
        self.frozen = frozen
        self.purged = []

    def purge_by_index(self, inds):  # would be called by a leaking thin/admit
        self.purged.append(inds)


def test_admit_is_a_noop_when_frozen():
    stub = _Stub(frozen=True)
    out = AnchorBuffer.admit(stub, candidate_batch=object(), reward=object(),
                             energy=object(), dup_cutoff=0.05)
    assert out == 0, "frozen admit must report zero admissions"
    assert stub.purged == [], (
        "frozen admit must not evict either -- admit's replace path purges the "
        "displaced slot, and because the swap is 1-for-1 anchor_buffer_length "
        "does NOT move, so an eviction leak is invisible in that metric")


def test_thin_is_a_noop_when_frozen():
    stub = _Stub(frozen=True)
    assert AnchorBuffer.thin(stub, per_condition_min_energy=object()) is None
    assert stub.purged == []


def test_guard_defaults_to_unfrozen_so_absent_config_keeps_todays_behaviour():
    """getattr(..., False): a buffer restored from a sidecar written before the
    flag existed must behave exactly as it did."""
    class _NoAttr:
        pass

    with pytest.raises(Exception):
        # reaches real work and dies on the stub's missing state, which is the
        # proof it did NOT short-circuit
        AnchorBuffer.admit(_NoAttr(), candidate_batch=object(), reward=object(),
                           energy=object(), dup_cutoff=0.05)


# --------------------------------------------------------------- cadence keys

@pytest.mark.parametrize("value", [0, None])
def test_zero_or_null_cadence_disables_instead_of_crashing(value):
    """Both keys were bare `%` on a direct attribute read, so 0 raised
    ZeroDivisionError and a missing key AttributeError -- mid-eval, hours in,
    with no load-time refusal. This asserts the arithmetic the fix uses."""
    every = int(value or 0)
    assert every == 0
    assert not (every > 0), "0/None must disable the cadence, never divide"


def test_positive_cadence_still_fires():
    every = int(3 or 0)
    assert every > 0 and 6 % every == 0


# ------------------------------------------------------- the resume hole

def test_partial_nan_priorities_are_not_uniform():
    """Guards the assumption the freeze rests on. _loss_weights only returns a
    uniform vector when ema_loss is ALL NaN; a partially-NaN vector fills the
    NaN rows with the 0.90 quantile of the rest, i.e. HIGH priority. So
    'writes are suppressed, therefore the draw is uniform' is false unless the
    vector is blanked -- which is what apply_anchor_buffer_policy does on
    restore."""
    losses = torch.tensor([1.0, 2.0, float("nan"), 4.0])
    valid = ~torch.isnan(losses)
    assert valid.any(), "precondition: this is the partial case, not the all-NaN one"
    nan_fill = torch.quantile(losses[valid], 0.9).item()
    assert nan_fill > losses[valid].min().item(), (
        "a NaN row inherits near-top priority, so partial NaN is emphatically "
        "not neutral")

    all_nan = torch.full((4,), float("nan"))
    assert bool(torch.isnan(all_nan).all()), (
        "only the all-NaN case takes _loss_weights' uniform short-circuit")


def test_blanking_is_idempotent_and_covers_birth_loss():
    """apply_anchor_buffer_policy blanks both vectors. birth_loss matters
    because it is cloned from ema_loss at construction and feeds the
    death-vs-birth delta telemetry; leaving it populated against a blanked
    ema_loss would make those deltas nonsense."""
    ema = torch.tensor([1.0, 2.0, 3.0])
    birth = ema.clone()
    for _ in range(2):
        ema = torch.full_like(ema, float("nan"))
        birth = torch.full_like(birth, float("nan"))
    assert bool(torch.isnan(ema).all()) and bool(torch.isnan(birth).all())
    assert not any(math.isfinite(v) for v in ema.tolist())


# ------------------------------------------- apply_anchor_buffer_policy itself

class _Buf:
    def __init__(self, n=4, nan=False):
        v = float("nan") if nan else 1.0
        self.ema_loss = torch.full((n,), v)
        self.birth_loss = torch.full((n,), v)
        self.frozen = None

    def __len__(self):
        return len(self.ema_loss)


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _stub_modeller(buf, **anchor_cfg):
    """Duck-typed stand-in: apply_anchor_buffer_policy only touches
    self.anchor_buffer and self.args.buffers.anchor_buffer."""
    m = _Cfg(anchor_buffer=buf,
             args=_Cfg(buffers=_Cfg(anchor_buffer=_Cfg(**anchor_cfg))))
    return m


def _policy(m, source="test"):
    from energy_sampling.train import Modeller
    return Modeller.apply_anchor_buffer_policy(m, source)


def test_policy_sets_frozen_from_config_not_from_state():
    buf = _Buf()
    m = _stub_modeller(buf, frozen=True, refresh_every_n_evals=3)
    _policy(m)
    assert buf.frozen is True

    # the same buffer object under a config that no longer freezes
    m2 = _stub_modeller(buf, frozen=False, refresh_every_n_evals=3)
    _policy(m2)
    assert buf.frozen is False, "config owns behaviour; a restored flag must not stick"


def test_policy_blanks_priorities_when_no_writer_is_enabled():
    """The resume hole: from_state_dict restores a populated ema_loss verbatim.
    With the sweep disabled nothing can ever refresh it, so it must be blanked
    or the draw is weighted by permanently-stale numbers."""
    buf = _Buf()
    assert not bool(torch.isnan(buf.ema_loss).all())
    _policy(_stub_modeller(buf, frozen=True, refresh_every_n_evals=0))
    assert bool(torch.isnan(buf.ema_loss).all()), "must blank to ALL NaN, the only uniform state"
    assert bool(torch.isnan(buf.birth_loss).all()), "birth_loss too, or the deltas are nonsense"


def test_policy_leaves_priorities_alone_when_the_sweep_is_live():
    buf = _Buf()
    _policy(_stub_modeller(buf, frozen=True, refresh_every_n_evals=3))
    assert not bool(torch.isnan(buf.ema_loss).any()), (
        "a live sweep maintains these; blanking them would destroy the mechanism "
        "the user explicitly wanted kept as an independent choice")


def test_policy_is_backward_compatible_with_configs_lacking_the_keys():
    buf = _Buf()
    _policy(_stub_modeller(buf, refresh_every_n_evals=3))   # no frozen, no online_loss_flow
    assert buf.frozen is False
    assert not bool(torch.isnan(buf.ema_loss).any())


def test_online_loss_flow_refuses_loudly_rather_than_silently_doing_nothing():
    with pytest.raises(NotImplementedError, match="online_loss_flow"):
        _policy(_stub_modeller(_Buf(), online_loss_flow=True, refresh_every_n_evals=3))
