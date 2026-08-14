"""
Metamorphic test: a crystal's computed energy must not depend on WHAT IT WAS BATCHED WITH,
or on its position in the batch.

WHY THIS EXISTS AS ITS OWN FILE. `mol2cluster` collapsed a per-graph `T_fc` stack to a
single matrix (`...repeat_interleave(max_z_prime, dim=0)[1]`), so ONE crystal's cell metric
set the Z'>1 supercell padding (`zp_buffer`) for the whole batch -- always graph 0's, since
the interleaved stack is [g0, g0, g1, g1, ...].

The energy cost is rare and concentrated, not a broad bias. Over 4 random 100-crystal
batches of the real sg 9 Z'=2 prior the median |d elj| was 1e-4 kJ/mol (nil), with 3 of 400
crystals above 1 kJ/mol and a worst case of 126 kJ/mol against a median |elj| of ~970. One
batch contained no affected crystal at all. That distribution is exactly why a random pool
is the wrong instrument here and why the control below is built adversarially: the first
version of this test drew 12 crystals at random, measured a 0.038 kJ/mol effect, and would
have passed on most seeds while the bug was live.

Grepping for `stack[0]` / `stack[1]` finds candidate sites but cannot distinguish an
intentional single-element pick (the autoencoder equivariance checks applying one rotation
to everything; `crystal_reduction` deliberately comparing graph 0 against graph 1) from a
bug. This property can:

    a crystal alone == the same crystal in a batch == the same crystal in a SHUFFLED batch

Any collapsed-stack bug violates it by construction, wherever it lives, because the answer
then depends on whichever crystal happened to land at the indexed position. It also pins
the consequence that made the original defect worse than a plain inaccuracy:
NON-REPRODUCIBILITY. The same structure scored differently depending on its batch-mates, so
a rerun over a reshuffled prior produced different energies.

CPU-only.

    CUDA_VISIBLE_DEVICES="" python test_batch_invariance.py
"""
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_here), os.path.join(os.path.dirname(_here), '..', 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

PRIORS = r'D:\crystal_datasets\conditional\priors'
_R = []


def check(name, ok, detail=''):
    _R.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")


def _pool(path, n):
    """
    One batch of n crystals off a prior file.

    Every route below slices this ONE object with subsample_new_batch, which already returns
    a batch -- deliberately never re-collating a slice. collate_data_list on an
    already-subsampled batch double-wraps the list-valued `symmetry_operators`, and
    aunit2ucell then dies with "number of subscripts (3) does not match dimensions (4)".
    Using a single construction path for all three routes also keeps the comparison about
    batch COMPOSITION rather than about how the object was assembled.
    """
    data = torch.load(path, weights_only=False)
    batch = data['equalized_prior'] if isinstance(data, dict) else data
    return batch.subsample_new_batch(torch.arange(min(n, batch.num_graphs)))


def _elj(batch, cutoff=6, supercell=5):
    out = batch.analyze(['elj'], cutoff=cutoff, supercell_size=supercell,
                        std_orientation=False, predictor=None)
    return out['elj'].detach().clone()


def _tol(alone):
    """
    Absolute 1e-2 kJ/mol, floored well above the noise and well below the signal.

    Measured, not guessed. Correct code gives EXACTLY 0 across composition and order.
    Two independent analyze() calls on identical input differ by ~1e-4 kJ/mol (the elj
    path's run-to-run floor). The smallest per-crystal error the bug produced on a crystal
    it actually touched was 0.048. So 1e-2 sits 100x above the floor and 5x below the
    faintest real signal. A scale-relative 1e-3 tolerance -- the obvious choice for an
    O(1000) quantity -- is ~1.0 kJ/mol and is BLIND to all but the extreme tail.
    """
    return max(1e-2, 1e-6 * float(alone.abs().median().clamp_min(1.0)))


def _true_buffer(batch):
    """Correct per-crystal zp_buffer: the largest intra-crystal aunit centroid distance."""
    from mxtaltools.common.geometry_utils import fractional_transform
    n, zp = batch.num_graphs, int(batch.max_z_prime)
    cart = fractional_transform(
        batch.aunit_centroid.reshape(n * zp, 3),
        batch.T_fc.repeat_interleave(zp, dim=0)).reshape(n, zp, 3)
    return (cart[:, :, None, :] - cart[:, None, :, :]).norm(dim=-1).amax(dim=(1, 2))


def _adversarial_pool(path, k=12):
    """
    The worst case for a collapsed stack, chosen deterministically rather than by luck.

    `repeat_interleave(Zp, 0)[1]` always lands on GRAPH 0 (the interleaved stack is
    [g0, g0, g1, g1, ...]), so the bug applies graph 0's cell metric to everyone. Exposure
    is therefore maximal when graph 0 needs the SMALLEST buffer and its batch-mates need the
    largest. A random draw is a coin flip: of four random 100-crystal batches measured on
    this prior, one contained no affected crystal at all while another contained a 126 kJ/mol
    error. A control that only fires on some seeds is not a control.
    """
    data = torch.load(path, weights_only=False)
    batch = data['equalized_prior'] if isinstance(data, dict) else data
    buf = _true_buffer(batch)
    idx = torch.cat([buf.argmin().reshape(1), torch.topk(buf, k - 1).indices])
    return batch.subsample_new_batch(idx), buf[idx]


def _three_routes(pool):
    """(alone, batched, batched-then-unpermuted) elj for every crystal in `pool`."""
    m = pool.num_graphs
    alone = torch.stack([
        _elj(pool.subsample_new_batch(torch.tensor([i])).clone()).reshape(())
        for i in range(m)])
    batched = _elj(pool.subsample_new_batch(torch.arange(m)).clone())
    perm = torch.randperm(m, generator=torch.Generator().manual_seed(0))
    shuffled = _elj(pool.subsample_new_batch(perm).clone())
    unshuffled = torch.empty_like(shuffled)
    unshuffled[perm] = shuffled
    return alone, batched, unshuffled


def test_energy_is_invariant_to_batch_composition(fname, label, n=12):
    path = os.path.join(PRIORS, fname)
    if not os.path.exists(path):
        print(f"  SKIP {label} (prior not on this machine)")
        return
    pool = _pool(path, n)
    print(f"\n  {label}  (Z'={int(pool.max_z_prime)}, n={pool.num_graphs})")
    alone, batched, unshuffled = _three_routes(pool)
    tol = _tol(alone)
    d_batch, d_shuf = (batched - alone).abs(), (unshuffled - alone).abs()

    check(f"{label}: alone == batched", bool((d_batch < tol).all()),
          f"max |d| {d_batch.max():.4g} < tol {tol:.4g} "
          f"(median |elj| {alone.abs().median():.1f})")
    check(f"{label}: alone == batched in a DIFFERENT ORDER", bool((d_shuf < tol).all()),
          f"max |d| {d_shuf.max():.4g} < tol {tol:.4g}")


def test_invariance_on_the_worst_case_pool(fname, label):
    """
    The same property on the pool built to break it: minimum-buffer crystal at position 0,
    the widest-spread crystals behind it. If invariance survives here it survives anywhere.
    """
    path = os.path.join(PRIORS, fname)
    if not os.path.exists(path):
        print(f"  SKIP {label} worst-case (prior not on this machine)")
        return
    pool, buf = _adversarial_pool(path)
    print(f"\n  {label} WORST CASE  (buffer at position 0: {float(buf[0]):.2f} A, "
          f"batch-mates up to {float(buf.max()):.2f} A -> "
          f"{float(buf.max() - buf[0]):.2f} A of shortfall on tap)")
    alone, batched, unshuffled = _three_routes(pool)
    tol = _tol(alone)
    d = torch.maximum((batched - alone).abs(), (unshuffled - alone).abs())
    check(f"{label}: invariant even on the worst-case pool", bool((d < tol).all()),
          f"max |d| {d.max():.4g} < tol {tol:.4g}")


def test_the_test_would_have_caught_the_bug():
    """
    A regression test that cannot fail is worth nothing. Re-introduce the collapsed stack
    and require the invariance check to BREAK -- otherwise this file is reassurance-shaped
    silence, the failure mode that recurred all through this work.

    Patches mol2cluster itself rather than the shared `fractional_transform`: the shared
    helper is also used by aunit2ucell and get_aunit_positions, so corrupting it would fail
    the check for reasons unrelated to zp_buffer and prove nothing about THIS bug.
    """
    print("\n  negative control: re-introduce the bug, require a FAILURE")
    path = os.path.join(PRIORS, 'deadrow10k_sg9_zp2_elj.pt')
    if not os.path.exists(path):
        print("  SKIP negative control (no Z'=2 prior)")
        return
    from mxtaltools.common.geometry_utils import fractional_transform
    from mxtaltools.dataset_utils.data_class_methods.crystal_building import (
        MolCrystalBuilding)

    real = MolCrystalBuilding.mol2cluster

    def buggy(self, cutoff=6, supercell_size=10, std_orientation=True):
        if self.max_z_prime > 1:
            frac_centroids = self.aunit_centroid.reshape(
                self.num_graphs * self.max_z_prime, 3)
            cart_centroids = fractional_transform(
                frac_centroids,
                self.T_fc.repeat_interleave(self.max_z_prime, dim=0)[1]  # <-- the bug
            ).reshape(self.num_graphs, self.max_z_prime, 3)
            dists = (cart_centroids[:, :, None, :]
                     - cart_centroids[:, None, :, :]).norm(dim=-1)
            zp_buffer = dists.amax(dim=(1, 2)).repeat_interleave(self.z_prime, dim=0)
            zp1_batch = self.split_to_zp1_batch()
            zp1_batch.pose_aunit(std_orientation=std_orientation)
            zp1_batch.build_unit_cell()
            zp1_cluster = zp1_batch.build_cluster(
                cutoff=cutoff, supercell_size=supercell_size, zp_buffer=zp_buffer)
            return self.join_zp1_cluster_batch(zp1_cluster)
        return real(self, cutoff, supercell_size, std_orientation)

    pool, _ = _adversarial_pool(path)
    MolCrystalBuilding.mol2cluster = buggy
    try:
        alone, batched, unshuffled = _three_routes(pool)
        tol = _tol(alone)
        worst = max(float((batched - alone).abs().max()),
                    float((unshuffled - alone).abs().max()))
        broke = worst > tol
        check("the bug, re-introduced, IS detected by this test", broke,
              f"max |d| {worst:.4g} vs tol {tol:.4g}"
              + ('' if broke else '   <-- test is BLIND, it proves nothing'))
    finally:
        MolCrystalBuilding.mol2cluster = real


def main():
    print("batch-composition invariance (CPU)")
    # Z'=1 is the control: only the max_z_prime > 1 branch was affected, so Z'=1 must have
    # been correct before the fix and must stay correct after it.
    test_energy_is_invariant_to_batch_composition(
        'nehzor_sg14_zp1_elj_prior_dataset.pt', "sg14 Z'=1 (control)")
    test_energy_is_invariant_to_batch_composition(
        'deadrow10k_sg9_zp2_elj.pt', "sg9 Z'=2")
    test_energy_is_invariant_to_batch_composition(
        'deadrow10k_sg14_zp2_elj.pt', "sg14 Z'=2")
    test_invariance_on_the_worst_case_pool('deadrow10k_sg9_zp2_elj.pt', "sg9 Z'=2")
    test_invariance_on_the_worst_case_pool('deadrow10k_sg14_zp2_elj.pt', "sg14 Z'=2")
    test_the_test_would_have_caught_the_bug()

    bad = [r for r in _R if not r[1]]
    print("\n" + "=" * 74)
    print(f"{len(_R) - len(bad)}/{len(_R)} checks passed")
    for n, _, d in bad:
        print(f"  FAIL {n}  {d}")
    print("PASS" if (_R and not bad) else ("FAIL" if bad else "NO CHECKS RAN"))
    return 0 if (_R and not bad) else 1


if __name__ == '__main__':
    sys.exit(main())
