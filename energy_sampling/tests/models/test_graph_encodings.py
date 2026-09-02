"""Gates for models/graph_encodings.py -- the encoder's structural inputs AND the battery's
labels.

Every check here is against a CLOSED-FORM answer on a graph whose symmetry group is known by
hand (a path, an even cycle, an odd cycle, benzene), not against a second implementation and
not against a stored blob. That matters more than usual: these functions produce the labels
the self-supervision battery is scored on, so an error here does not fail a test downstream,
it silently redefines what the encoder is being asked to learn.

The two properties the battery leans on hardest, each pinned by a test that fails without it:

  * RWSE encodes ring SIZE (the odd-cycle spike), which is the 1-WL bound being broken.
  * LapPE is NOT a function of the graph, which is why it is disqualified by the determinism
    gate rather than merely disfavoured.

    python -m pytest -q tests/models/test_graph_encodings.py
"""
import os
import sys

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _root in (os.path.dirname(_here),
              os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import numpy as np
import pytest

from energy_sampling.models.graph_encodings import (
    UNREACHABLE, cycle_rank, degree_histogram, diameter, eccentricity,
    graph_from_smiles, lap_pe, n_automorphisms, orbit_sizes, ring_membership,
    rwse, shortest_paths, smallest_ring_size, spectral_moments, to_dense_adjacency,
    wiener_index,
)


def _path(n):
    """P_n: 0-1-2-...-(n-1)."""
    return np.array([[i for i in range(n - 1)], [i + 1 for i in range(n - 1)]])


def _cycle(n):
    return np.array([[i for i in range(n)], [(i + 1) % n for i in range(n)]])


# ------------------------------------------------------------------ adjacency contract

def test_edge_convention_does_not_matter():
    """Half-edges and doubled edges must give the same adjacency, or every downstream
    number depends on which call site built the list."""
    single = _cycle(5)
    doubled = np.concatenate([single, single[::-1]], axis=1)
    assert np.array_equal(to_dense_adjacency(single, 5), to_dense_adjacency(doubled, 5))


def test_out_of_range_atom_is_refused():
    with pytest.raises(ValueError):
        to_dense_adjacency(np.array([[0], [7]]), 5)


# ------------------------------------------------------------------------- distances

def test_shortest_paths_on_a_path_is_the_index_difference():
    n = 6
    spd = shortest_paths(_path(n), n)
    want = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    assert np.array_equal(spd, want)


def test_disconnection_is_marked_not_zeroed():
    """Two isolated edges. A zero would read as 'adjacent', which is the dangerous filler."""
    spd = shortest_paths(np.array([[0, 2], [1, 3]]), 4)
    assert spd[0, 1] == 1 and spd[0, 0] == 0
    assert spd[0, 2] == UNREACHABLE and spd[0, 3] == UNREACHABLE


def test_eccentricity_diameter_and_wiener_closed_form():
    # path P_4: Wiener = n(n^2-1)/6 = 10, diameter 3, eccentricities [3,2,2,3]
    spd = shortest_paths(_path(4), 4)
    assert np.array_equal(eccentricity(spd), np.array([3, 2, 2, 3]))
    assert diameter(spd) == 3
    assert wiener_index(spd) == 10

    # cycle C_6: every eccentricity 3, Wiener = 6*(1+1+2+2+3)/2 = 27
    spd6 = shortest_paths(_cycle(6), 6)
    assert np.array_equal(eccentricity(spd6), np.full(6, 3))
    assert diameter(spd6) == 3
    assert wiener_index(spd6) == 27


# ------------------------------------------------------------------------ ring signal

def test_cycle_rank_counts_independent_cycles():
    assert cycle_rank(_path(6), 6) == 0                      # tree
    assert cycle_rank(_cycle(6), 6) == 1
    # two disjoint triangles: 6 edges, 6 nodes, 2 components -> rank 2
    two = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]])
    assert cycle_rank(two, 6) == 2


def test_smallest_ring_size_is_girth_through_the_node():
    assert np.array_equal(smallest_ring_size(_cycle(6), 6), np.full(6, 6))
    assert np.array_equal(smallest_ring_size(_path(6), 6), np.zeros(6, dtype=np.int64))
    assert not ring_membership(_path(6), 6).any()

    # a 5-ring with a 2-atom tail: ring atoms read 5, tail atoms read 0
    e = np.array([[0, 1, 2, 3, 4, 0, 5], [1, 2, 3, 4, 0, 5, 6]])
    got = smallest_ring_size(e, 7)
    assert np.array_equal(got[:5], np.full(5, 5))
    assert np.array_equal(got[5:], np.zeros(2, dtype=np.int64))


# ------------------------------------------------------------------------------ RWSE

def test_rwse_is_zero_on_odd_steps_of_a_bipartite_graph():
    """C_6 is bipartite, so a walk cannot return in an odd number of steps. Exact zeros."""
    r = rwse(_cycle(6), 6, k=7)
    assert np.allclose(r[:, 0::2], 0.0)          # steps 1, 3, 5, 7
    assert np.allclose(r[:, 1], 0.5)             # step 2: out and back, 2 * (1/2)(1/2)


def test_rwse_spikes_at_the_ring_size_and_that_is_the_1wl_break():
    """The property the encoder is being given RWSE FOR.

    C_5 is not bipartite, so the first odd return is at step 5 -- the ring size. On the
    matched path graph that entry is exactly zero. A plain MPNN cannot tell these two
    nodes apart at any depth; this feature can, in one number.
    """
    ring = rwse(_cycle(5), 5, k=6)
    tail = rwse(_path(5), 5, k=6)
    assert (ring[:, 4] > 0.05).all(), 'no return at step 5 on a 5-ring'
    assert np.allclose(tail[:, 4], 0.0), 'a path graph returned on an odd step'


def test_rwse_is_a_function_of_the_graph_only():
    """Determinism, the gate LapPE fails. Same graph relabelled -> permuted rows, nothing
    else."""
    perm = np.array([3, 0, 4, 1, 2])
    e = _cycle(5)
    relabelled = perm[e]
    a = rwse(e, 5, k=8)
    b = rwse(relabelled, 5, k=8)
    assert np.allclose(a, b[perm], atol=1e-12)


def test_isolated_atom_gets_zeros_not_nan():
    r = rwse(np.array([[0], [1]]), 3, k=4)       # atom 2 is isolated
    assert np.isfinite(r).all()
    assert np.allclose(r[2], 0.0)


# ----------------------------------------------------------------------------- LapPE

def test_lap_pe_is_not_a_function_of_the_graph():
    """The disqualifying property, asserted rather than described.

    Benzene's Laplacian is degenerate by symmetry, so the eigenvectors inside a degenerate
    eigenspace are fixed only up to rotation. Relabelling the atoms therefore does NOT
    simply permute the rows -- which is exactly what makes a cached {f_j} depend on the
    solver rather than on the molecule.
    """
    perm = np.array([2, 3, 4, 5, 0, 1])
    e = _cycle(6)
    a = lap_pe(e, 6, k=4)
    b = lap_pe(perm[e], 6, k=4)
    assert not np.allclose(a, b[perm], atol=1e-8), \
        'LapPE came out permutation-covariant here; if the degeneracy has been handled ' \
        'the determinism gate in the battery needs restating, not deleting'


def test_lap_pe_shape_is_padded_on_small_graphs():
    assert lap_pe(_path(3), 3, k=8).shape == (3, 8)


# ------------------------------------------------------------------------- moments etc

def test_normalized_laplacian_trace_is_n():
    """tr(L) = n exactly for a graph with no isolated vertex -- a closed form, so this
    catches a wrong Laplacian convention rather than merely a wrong number."""
    for e, n in [(_path(6), 6), (_cycle(5), 5)]:
        assert np.isclose(spectral_moments(e, n, ks=(1,))[0], n, atol=1e-9)


def test_degree_histogram_bins_and_saturates():
    h = degree_histogram(_path(4), 4, max_degree=3)
    assert h.tolist() == [0, 2, 2, 0]            # two ends (deg 1), two middles (deg 2)
    star = np.array([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]])
    assert degree_histogram(star, 6, max_degree=2).tolist() == [0, 5, 1]   # hub saturates


# -------------------------------------------------------------------------- symmetry

def test_orbits_and_automorphism_counts_on_known_groups():
    # C_6 has the dihedral group D_6, order 12, and a single orbit
    assert n_automorphisms(_cycle(6), 6) == 12
    assert np.array_equal(orbit_sizes(_cycle(6), 6), np.full(6, 6))
    # P_4 has only the reversal: order 2, orbits {0,3} and {1,2}
    assert n_automorphisms(_path(4), 4) == 2
    assert np.array_equal(orbit_sizes(_path(4), 4), np.full(4, 2))


def test_labels_shrink_the_automorphism_group():
    """The control the battery needs: what did the labels buy? Breaking one node's label
    must destroy the symmetry, or the labelled path is not actually being used."""
    lab = [0] * 6
    assert n_automorphisms(_cycle(6), 6, labels=lab) == 12
    lab[0] = 1
    assert n_automorphisms(_cycle(6), 6, labels=lab) == 2      # reflection through node 0


def test_orbit_cap_refuses_rather_than_truncating():
    """A truncated automorphism set gives WRONG orbits, not incomplete ones."""
    with pytest.raises(RuntimeError, match='refusing'):
        orbit_sizes(_cycle(6), 6, cap=3)


# ---------------------------------------------------------------------------- RDKit

def test_benzene_from_smiles_reads_as_benzene():
    z, e, parity = graph_from_smiles('c1ccccc1')
    assert z.tolist().count(6) == 6 and z.tolist().count(1) == 6
    n = len(z)
    ring = smallest_ring_size(e, n)
    assert np.array_equal(ring[z == 6], np.full(6, 6)), 'carbons are not on a 6-ring'
    assert np.array_equal(ring[z == 1], np.zeros(6, dtype=np.int64)), 'a hydrogen is in a ring'
    assert cycle_rank(e, n) == 1
    # with atomic-number labels: carbons one orbit of 6, hydrogens one orbit of 6
    assert np.array_equal(orbit_sizes(e, n, labels=z.tolist()), np.full(n, 6))
    assert (parity == 0).all()


def test_stereocentre_is_labelled_and_breaks_symmetry():
    """Parity has to reach the labels, or the enantiomer-blindness claim is untested."""
    z, e, parity = graph_from_smiles('C[C@H](N)C(=O)O')
    assert (parity != 0).sum() == 1, 'no stereocentre was labelled'
    bare = n_automorphisms(e, len(z), labels=z.tolist())
    with_p = n_automorphisms(e, len(z), labels=list(zip(z.tolist(), parity.tolist())))
    assert with_p <= bare
