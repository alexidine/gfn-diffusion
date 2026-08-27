"""
How a branch's loss TERMS combine into its loss.

Both reduction sites (forward and backward) route through
gflownet_losses.combine_branch_terms. Before 2026-08-26 they inlined a MEAN over
the ACTIVE term list, which made every coefficient secretly coeff / n_active:
switching any term off amplified every surviving term in that branch. These
tests pin the SUM semantics, and in particular pin MENU INDEPENDENCE -- the
property that actually distinguishes the two, and the one the old code failed.
"""
import torch

from gflownet_losses import combine_branch_terms


def test_terms_are_summed_not_averaged():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])
    out = combine_branch_terms([a, b], -1)
    assert torch.allclose(out, a + b)
    assert not torch.allclose(out, (a + b) / 2), 'terms are being averaged, not summed'


def test_a_terms_contribution_does_not_depend_on_how_many_others_are_active():
    """MENU INDEPENDENCE -- the invariant, and the one a mean cannot satisfy.

    A term's contribution must be the same whether it is alone on the branch or
    sharing it. Under the old mean, adding a SECOND term halved the first one's
    weight, so a zero-gradient sidecar changed the policy's effective step size
    just by being present in the list."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])
    both, only_b, only_a = (combine_branch_terms(x, -1) for x in ([a, b], [b], [a]))
    assert torch.allclose(both - only_b, only_a), (
        "a term's contribution changed when another term joined the branch")


def test_a_disabled_term_does_not_amplify_its_neighbours():
    """The operational form of the same property: dropping a term from the list
    must leave the survivors untouched. This is the regression that motivated
    the change -- emp_z: 0 doubled the forward VarGrad gradient."""
    vg = torch.tensor([5.0, 6.0, 7.0])
    sidecar = torch.tensor([100.0, 200.0, 300.0])
    with_sidecar = combine_branch_terms([vg, sidecar], -1)
    without = combine_branch_terms([vg], -1)
    assert torch.allclose(without, vg)
    assert torch.allclose(with_sidecar - sidecar, without)


def test_soft_clip_bounds_each_term_individually_not_the_combined_row():
    """soft_clip runs elementwise on the [n_terms, B] stack, i.e. BEFORE the
    reduction, so it bounds each TERM; the combined row is bounded by
    n_active x that. Pins the documented consequence of summing under a clip."""
    cutoff = 10.0
    big = torch.tensor([1000.0, 1000.0])
    one = combine_branch_terms([big], cutoff)
    two = combine_branch_terms([big, big], cutoff)
    assert torch.all(one < cutoff * 3), 'a single clipped term should sit near the cutoff'
    assert torch.allclose(two, 2 * one), 'each term is clipped, then summed'
