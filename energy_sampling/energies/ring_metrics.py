"""ONE definition of ring closure, pucker identity and ring-block classification.

WHY A SHARED MODULE. Three consumers need the same ring measurements -- the sampler's own
closure monitor (``ConformerTorsions.sample_prior_states``), the bank builder
(``build_ring_banks.basin_key``) and the reference table (``energies/prior_baselines.py``)
-- and a second, subtly different definition of "the ring is closed" or "this is the same
pucker" would let two of them disagree while both looked right. So the quantisation rule
lives here and ``build_ring_banks`` imports it rather than restating it.

WHAT IS AND IS NOT AVAILABLE HERE.

* CLOSURE is measured on the closure BOND, which is not a tree DoF -- it is whatever the
  ring's internals imply -- so it has to be measured on the draw. Reported both absolutely
  (angstrom) and in units of a bond's own thermal width, since that is the scale at which
  it stops mattering.
* PUCKER IDENTITY is the sign pattern of the ring torsions, with a flat band so a planar
  ring reads as one state rather than as a random sign pattern. Coarse and stable by
  design; it is a basin LABEL, not a coordinate.
* AROMATIC PLANARITY is the median |ring torsion| over an aromatic ring. An aromatic ring
  is held planar near the reference BY DESIGN (``ring_blocks`` refuses to bank it), so this
  column tests an intended contract rather than sampling quality.
* THERE IS NO DENSITY HERE, and none of these need one. Ring ESS, D_avoidable and an
  importance-sampled log Z remain unavailable: a bank/subspace draw is a mixture that is
  singular in the directions the subspace does not span, and ``prior_log_prob`` refuses on
  exactly that ground. Substituting the acyclic density, or an independent marginal, would
  produce a number for a DIFFERENT distribution.

RING-BLOCK CLASSES ARE FOUR, NOT ONE. Collapsing them into "handled" is what made the ring
column uninformative:

    banked_modes        a fitted RingModes pucker subspace -- pucker is SAMPLED
    banked_rows         a discrete RingBank above the row threshold -- pucker is SAMPLED
    held_aromatic       aromatic, deliberately never banked -- planar BY DESIGN
    held_unsupported    saturated, but no bank resolved for its (signature, n_dof) key --
                        closure preserved, pucker NOT sampled. A gap, not a design choice.

plus the orthogonal ``stale_prior`` flag: a prior predating the ring-signature fix has keys
that cannot resolve, so every ring reads ``held_unsupported`` for a reason that has nothing
to do with the molecule.
"""
from __future__ import annotations

import numpy as np
import torch

# A ring torsion inside this band reads FLAT rather than as a sign. Without it a planar
# ring's near-zero torsions take random signs and every draw looks like a different basin.
PUCKER_FLAT_DEG = 10.0


def basin_label(torsions_rad) -> tuple:
    """Pucker basin label for ONE ring: the quantised sign pattern of its torsions.

    ``build_ring_banks.basin_key`` is this function; it deduplicates minimised conformers
    with the same rule the benchmark later uses to count occupancy, which is the point.
    """
    return tuple(0 if abs(np.degrees(float(x))) < PUCKER_FLAT_DEG else int(np.sign(float(x)))
                 for x in torsions_rad)


def basin_labels(torsions_rad: np.ndarray) -> np.ndarray:
    """``[n, k]`` ring torsions in radians -> ``[n]`` integer basin ids and the label list.

    Returns ``(ids, labels)`` where ``labels[i]`` is the tuple for id ``i``.
    """
    t = np.degrees(np.asarray(torsions_rad, dtype=np.float64))
    q = np.where(np.abs(t) < PUCKER_FLAT_DEG, 0, np.sign(t)).astype(np.int64)
    labels, ids = np.unique(q, axis=0, return_inverse=True)
    return ids, [tuple(int(v) for v in row) for row in labels]


def ring_cycles(en) -> list:
    """Heavy-atom ring cycles in cyclic order, in PLACEMENT-SLOT numbering.

    ``cycle_basis`` over the full graph (not the spanning tree -- a tree has no cycles),
    restricted to heavy atoms because a ring's hydrogens are substituents, not members.
    """
    import networkx as nx
    z = np.asarray(en.spec.z)
    g = nx.Graph()
    heavy = [i for i in range(len(z)) if z[i] > 1 and bool(en.atom_in_ring[i])]
    g.add_nodes_from(heavy)
    hs = set(heavy)
    for a, b in np.asarray(en.spec.graph_bond_index):
        a, b = int(a), int(b)
        if a in hs and b in hs:
            g.add_edge(a, b)
    return [c for c in nx.cycle_basis(g) if len(c) >= 3]


def ring_torsions(en, x: torch.Tensor, cycles=None) -> list:
    """Per ring cycle, ``[n, ring_size]`` ring torsions in RADIANS.

    The i-th torsion of a cycle is the dihedral over atoms ``(i, i+1, i+2, i+3)`` mod the
    ring size -- the same walk ``build_ring_banks.basin_key`` uses.
    """
    from mxtaltools.conformers.geometry import dihedral
    if cycles is None:
        cycles = ring_cycles(en)
    if not cycles:
        return []
    n = int(x.shape[0])
    pos = en.build_positions(x).reshape(n, -1, 3)
    out = []
    for cyc in cycles:
        k = len(cyc)
        out.append(np.stack(
            [dihedral(pos[:, cyc[i]], pos[:, cyc[(i + 1) % k]],
                      pos[:, cyc[(i + 2) % k]], pos[:, cyc[(i + 3) % k]])
             .detach().cpu().numpy() for i in range(k)], axis=1))
    return out


def closure_error(en, x: torch.Tensor):
    """``(median |dr| in angstrom, the same in bond-thermal-sigma, n_closure_bonds)``.

    IDENTICAL to the sampler's own monitor: ``closure_length`` against ``ff.closure_r0``,
    worst closure bond per draw, median over draws, divided by the mean thermal bond width.
    Returns ``(nan, nan, 0)`` when the molecule has no closure bond, which is the acyclic
    case and is not a failure.
    """
    from mxtaltools.conformers.builder import closure_length
    n = int(x.shape[0])
    tree, ff = en._batch(n)
    if not ff.closure_index.numel():
        return float('nan'), float('nan'), 0
    cl = closure_length(tree, en.build_positions(x))
    err = (cl - ff.closure_r0).abs().reshape(n, -1).max(1).values
    s_r, _ = en.thermal_rtheta_sigma(float(en.temperature))
    med = float(err.median())
    return med, med / max(float(np.mean(s_r)), 1e-12), int(ff.closure_index.numel() // 2)


def classify_ring_blocks(en, prior) -> list:
    """One record per ring DoF block, from ``ring_blocks`` -- the ONLY block definition.

    ``ring_blocks`` records its per-block reasoning on ``en.ring_block_info`` as a side
    effect; this reads that rather than re-deriving aromaticity or the bank lookup, so the
    classification cannot drift from the sampler's own branch.
    """
    en.ring_blocks(prior)
    info = getattr(en, 'ring_block_info', None)
    if info is None:
        raise RuntimeError(
            'ring_blocks did not record ring_block_info; the classifier would have to '
            're-derive aromaticity and the bank lookup, which is exactly the second '
            'definition this module exists to prevent')
    return info


def ring_class_counts(records) -> dict:
    out = {c: 0 for c in ('banked_modes', 'banked_rows', 'held_aromatic', 'held_unsupported')}
    for r in records:
        out[r['ring_class']] = out.get(r['ring_class'], 0) + 1
    return out


def pucker_occupancy(en, x, prior=None, cycles=None, records=None):
    """Pucker-basin occupancy per ring cycle, and aromatic planarity per aromatic cycle.

    Returns ``(saturated, aromatic)``. Each entry names the cycle, so a molecule with both
    kinds of ring reports each against ITS OWN contract rather than one pooled number.

    Occupancy is reported ONLY for a cycle whose atoms are all non-aromatic: a planar
    aromatic ring has one basin by construction and reporting evenness there would be a
    column that passes because nothing can fail. Whether that saturated ring's pucker is
    actually SAMPLED is the block class's job to say, not this function's.
    """
    import math
    if cycles is None:
        cycles = ring_cycles(en)
    tors = ring_torsions(en, x, cycles)
    sat, aro = [], []
    for cyc, t in zip(cycles, tors):
        is_aro = bool(en.atom_is_aromatic[list(cyc)].all())
        if is_aro:
            aro.append({'size': len(cyc), 'atoms': [int(a) for a in cyc],
                        'median_abs_torsion_deg': float(np.median(np.abs(np.degrees(t)))),
                        'p90_abs_torsion_deg': float(np.percentile(np.abs(np.degrees(t)), 90))})
            continue
        ids, labels = basin_labels(t)
        frac = np.bincount(ids, minlength=len(labels)) / len(ids)
        nz = frac[frac > 0]
        even = (float(-(nz * np.log(nz)).sum() / math.log(len(labels)))
                if len(labels) > 1 else float('nan'))
        sat.append({'size': len(cyc), 'atoms': [int(a) for a in cyc],
                    'n_basins': int(len(labels)), 'evenness': even,
                    'top_frac': float(frac.max()),
                    'median_abs_torsion_deg': float(np.median(np.abs(np.degrees(t))))})
    return sat, aro


def ring_measurements(en, x, prior, stats=None) -> dict:
    """Every ring number the reference table reports, for ONE draw of one arm.

    Population guard: a ring statistic computed over zero ring systems is not a pass, it is
    an absence, and the caller must be able to tell those apart. ``n_ring_systems`` is that
    discriminator and is always present.
    """
    records = classify_ring_blocks(en, prior)
    cycles = ring_cycles(en)
    err_a, err_sig, n_closure = closure_error(en, x)
    sat, aro = pucker_occupancy(en, x, cycles=cycles)
    counts = ring_class_counts(records)
    out = {
        'n_ring_systems': len(records),
        'n_ring_cycles': len(cycles),
        'n_closure_bonds': n_closure,
        'closure_err_a': err_a,
        'closure_err_sigma': err_sig,
        'stale_prior': bool(getattr(en, 'ring_sig_stale', False)),
        'saturated': sat, 'aromatic': aro,
    }
    out.update(counts)
    out['n_ring_using_bank'] = counts['banked_modes'] + counts['banked_rows']
    out['n_ring_held_fallback'] = counts['held_aromatic'] + counts['held_unsupported']
    # ring DoF the sampler drew INDEPENDENTLY rather than as part of a joint block. With
    # joint rings off this is every ring DoF, which is the whole of the negative control.
    out['n_ring_dof_independent'] = (int(stats.get('n_ring_marginal', 0))
                                     if stats is not None else None)
    out['n_ring_block_dof'] = int(sum(r['n_block_dof'] for r in records))
    out['n_ring_extra_dof'] = int(sum(r['n_extra_dof'] for r in records))
    # DENSITY-DEPENDENT RING NUMBERS ARE UNAVAILABLE BY DERIVATION, not by omission, and
    # the label travels WITH the measurement so a consumer cannot pick up the numbers
    # without it. There is deliberately no 'ess', 'D_avoidable' or 'log_z' key here: the
    # only way to produce one would be to score a ring draw with the acyclic density or an
    # independent marginal, which is the density of a different distribution.
    out['ring_density'] = (
        'available (acyclic)' if out['n_ring_systems'] == 0 else
        'UNAVAILABLE: a ring block is a mixture over fitted rows / a subspace singular in '
        'the held directions, so ESS, D_avoidable and IS log Z have no matched q(x). '
        'prior_log_prob raises rather than returning the acyclic density.')
    return out
