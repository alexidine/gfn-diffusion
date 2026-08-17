"""
Dataset layer for the conformational GFlowNet: conditions and priors as graphs.

train.py's data layer is two files with two different jobs (see its ``init_mol_dataset``
/ ``init_prior_dataset``), and both wrap their contents in a buffer:

  - **condition file** (``molecules_path``), read through ``_load_condition_file``, so
    the on-disk form is ``{'prior': batch}``. Its job is to supply *identity*: one graph
    per condition, carrying whatever the conditioner needs to encode "which target is
    this". No state, no energy.
  - **prior file** (``prior_path``), read as ``prior_data['equalized_prior']``. Backward
    training draws terminal states straight out of it and scores them through
    ``prebuilt_sample_to_reward``, so unlike the condition file these rows need a real
    state *and* a pre-baked energy.

For crystals a condition is a molecule plus a space group and the graph is a
``MolCrystalData``. For a conformer the condition is the molecule *and its internal
parameterisation* -- so the object is a ``MolData`` carrying the canonical
internal-coordinate tree plus, and this is the part with nowhere else to live, **which
torsions the sampler actually drives**: the tree enumerates all N-3 dihedrals while the
GFN state is one angle per rotatable bridge bond, and the map between them is what makes
a stored state interpretable at all.

THE STORAGE RULE (this is the load-bearing one)
-----------------------------------------------
``MXtalBase``'s batch operations -- ``subsample_new_batch`` (every buffer draw and every
purge) and ``append_batch`` (every admission) -- classify a tensor **purely by
``size(0)``**: equal to num_nodes means node-level and gets indexed, equal to num_graphs
means graph-level and gets indexed, anything else is treated as shared metadata and
**passed through untouched**. Neither op consults ``_slice_dict``, and neither remaps
index *values*.

Two consequences, both found the hard way:

1. A **per-DoF** array (per bond, per angle, per torsion, per ring closure) has none of
   those leading dimensions, so it survives collation and then silently goes stale on the
   first draw. ``MolConformerMethods`` stores the tree exactly that way -- ``[k, n_dof]``
   index arrays -- which is right for a dataset that is collated once and iterated, and
   unusable for one that is drawn from row-wise.
2. An **absolute atom index** is re-offset by collation but *not* by subsample/append, so
   any stored atom reference is wrong the moment a batch is re-cut.

So every field here is either **per-atom with graph-relative content** or **per-graph**.
Atom references are stored as *deltas* from the owning atom, which are invariant under
every re-offsetting; per-DoF quantities are scattered onto their owning atom, which the
parameterisation makes unambiguous -- the atom at placement slot k owns exactly
``min(k, 3)`` DoF, which is the whole reason 3N-6 comes out exact. Under that rule all
four batch operations are correct with no special-casing, and ``check_state_convention``
re-verifies it *through* a subsample so a regression fails loudly.

Fields are prefixed ``ctree_`` rather than ``tree_`` deliberately: they are a different
encoding of the same tree, and reusing mxtaltools' names would let a stray
``build_conformer_tree()`` half-overwrite this one with differently-shaped arrays.

Two more conventions
--------------------
**Atoms are stored in placement order.** ``TreeSpec`` relabels atoms into the order the
builder places them and ``ConformerTorsions`` works in that numbering throughout. Storing
``pos``/``z`` already permuted makes the permutation the identity, so a state tensor
written against the energy is readable against the graph with nothing to get wrong. The
stored atom order is then not the SMILES order; it is canonical (Weisfeiler-Lehman) and
reproducible, which is the property that matters.

**``pos`` is the REFERENCE conformer, not the sample's.** The sampled quantity is
``torsion_state`` -- per-graph ``[k]`` deltas on ``[-1, 1]`` where 1 == pi -- and its
Cartesian realisation is built on demand from the frozen reference internals. A prior of
20k states therefore costs 20k * k floats rather than 20k * N * 3 of geometry, and every
row still presents the molecule the conditioner is meant to see. Nothing downstream wants
a per-sample ``pos``: the energy builds positions from the state, and
``prebuilt_sample_to_reward`` reads the baked ``conformer_energy``.

Field list
----------
Per-atom, ``[N]``::

    ctree_round                     BFS placement round. Round 0/1/2 are the frame seeds,
                                    and round == min(slot, 3) over 0..3, so this doubles
                                    as the DoF-ownership rank (see dof_rank()).
    ctree_ref_a, _ref_b, _ref_c     reference atoms as DELTAS from the owning atom; 0
                                    where that DoF does not exist.
    ctree_r0, _theta0, _phi0        the reference internal coordinate owned by this atom,
                                    0 where it does not own one.
    ctree_angle_is_linear           theta ~ pi: a genuine singularity of any atom tree.
    ctree_torsion_is_proper         False on near-root impropers.
    ctree_torsion_frame_is_linear   phi ill-conditioned.
    ctree_state_col                 which state dimension drives this atom's torsion,
                                    -1 if frozen. The sparse form of
                                    ``ConformerTorsions.mask``; sparse because that mask
                                    is a 0/1 selection (asserted at build time).
    ctree_closure, ctree_closure_r0 ring-closure partner delta (0 = none) and its
                                    as-built length. Closures are absent from the tree,
                                    so one is assigned to each free endpoint; a molecule
                                    needing two on one atom raises rather than dropping
                                    one silently.

Per-graph::

    n_torsions        [1] long, = k.
    torsion_state     [1, k] float, prior/replay rows only.
    conformer_energy  [1] float, prior/replay rows only. RAW energy in the force field's
                      own units (kcal/mol), i.e. baked at T = 1 --
                      ``prebuilt_sample_to_reward`` divides by the sampling temperature
                      itself, so baking E/T here would apply it twice.

Variable k
----------
The GFN's state dimension is fixed at construction, so one file is one k. That is the
model's constraint, not this layer's, and ``collate_conditions`` says so by name.
Per-graph ``n_torsions`` is stored anyway so the variable-dimension path, when it exists,
has the information rather than a padded tensor to reverse-engineer.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

# per-atom fields: node-level, graph-relative. Indexed correctly by every batch op.
CTREE_ATOM_FIELDS = (
    'ctree_round', 'ctree_ref_a', 'ctree_ref_b', 'ctree_ref_c',
    'ctree_r0', 'ctree_theta0', 'ctree_phi0',
    'ctree_angle_is_linear', 'ctree_torsion_is_proper', 'ctree_torsion_frame_is_linear',
    'ctree_state_col', 'ctree_closure', 'ctree_closure_r0',
)
# per-graph fields present on every conformer graph
CTREE_GRAPH_FIELDS = ('n_torsions',)
CONFORMER_FIELDS = CTREE_ATOM_FIELDS + CTREE_GRAPH_FIELDS
# additionally required of a prior / replay row, but not of a condition
STATE_FIELDS = ('torsion_state', 'conformer_energy')


@dataclass
class RingModes:
    """A ring signature's pucker manifold as a low-dimensional subspace.

    Replaces RingBank's discrete rows. The rows were eight isolated islands with a tiny
    jitter, so the proposal had near-zero mass BETWEEN basins -- and the boat saddles of
    cyclohexane sit exactly there, on the pseudorotation path between twists. A subspace
    plus a broadened density covers the path.

    Three further things fall out. Fused rings work, because PCA does not need an analytic
    puckering form. The weighting accident goes away: uniform-over-rows gave cyclohexane
    25% chair / 75% twist purely because those are the symmetry multiplicities (2 chairs,
    6 twist-boats), which is an arbitrary bias rather than a chosen one. And the components
    are meaningful -- whitened by the force field's own thermal widths, the covariance is
    kT * H^-1 at harmonic level, so the leading directions are the SOFT ones rather than
    merely the high-variance ones. That is the harmonic-limit version of a slow-mode
    analysis; Cremer-Pople is itself the Fourier/PCA decomposition of ring puckering, so
    for a monocycle this recovers CP empirically.

    Deliberately NOT a fitted density. Populations from a multi-start minimisation are not
    Boltzmann weights and are not claimed to be; the job is complete SUPPORT. `coords` is
    the projected sample cloud and sampling mixes a jittered draw from it with a uniform
    draw over its bounding box, in the same spirit as InternalPrior's `fatten`.
    """

    order: list              # [(kind, j)] block ordering this was fitted against
    ref: np.ndarray          # [d] reference the deviations are measured from
    periodic: np.ndarray     # [d] bool, which entries wrap
    scale: np.ndarray        # [d] whitening widths, from the FF's own constants
    components: np.ndarray   # [k, d] PCA directions in whitened deviation space
    coords: np.ndarray       # [n, k] the sample cloud, projected
    energies: np.ndarray = None   # [n] relative basin energy, kcal/mol. None = uniform
    bandwidth: float = 1.0   # jitter in THERMAL units: whitening makes 1 unit = 1 sigma
    comp_std: np.ndarray = None   # [k] sampled spread along each component; caps the jitter
    var_explained: float = 0.0
    n_samples: int = 0
    max_fold_deg: float = 0.0     # worst |deviation| after the branch cut; >150 is a warning

    def weights(self, temperature: float, temper: float = 1.0):
        """Boltzmann weights over the sample cloud, or uniform if no energies were stored.

        The populations are NOT cosmetic. Measured at kT=1: cyclohexane's chair holds
        99.7% and everything else 0.3%, while pyrrolidine -- proline's ring -- is 71/29
        with both basins genuinely occupied. Drawing uniformly over basins therefore
        over-samples cyclohexane's twist by ~200x while getting pyrrolidine roughly right,
        which is an arbitrary bias that varies per ring.

        `temper` > 1 flattens toward uniform, in the same spirit as InternalPrior's
        `fatten`: a proposal wants the target's support, and a rare-but-real basin is
        worth over-weighting slightly rather than losing.
        """
        if self.energies is None:
            return np.full(len(self.coords), 1.0 / max(len(self.coords), 1))
        w = np.exp(-(self.energies - self.energies.min())
                   / max(temperature * max(temper, 1e-6), 1e-6))
        t = w.sum()
        return w / t if t > 0 else np.full(len(w), 1.0 / len(w))

    def sample(self, n: int, rng, fill: float = 0.0, temperature: float = 1.0,
               temper: float = 1.0):
        """``[n, d]`` deviations in the ORIGINAL DoF units, ready to add to `ref`."""
        lo, hi = self.coords.min(0), self.coords.max(0)
        pick = self.coords[rng.choice(len(self.coords), size=n,
                                      p=self.weights(temperature, temper))]
        # PER-COMPONENT jitter, capped by the spread the samples actually show along that
        # direction. Thermal motion is 1 sigma along a direction TANGENT to the closed-ring
        # manifold, but zero across it -- closure is a constraint, not a soft coordinate --
        # and the sampled spread is what distinguishes the two. Jittering uniformly at 1
        # sigma is why adding components past the manifold's own dimension made closure
        # sharply worse (cyclohexane 2.7 -> 6.5 bond-sigma from k=5 to k=7): the extra
        # directions carry no manifold structure, so all that motion goes off-surface.
        sd = (np.minimum(self.bandwidth, self.comp_std) if self.comp_std is not None
              else np.full(pick.shape[1], self.bandwidth))
        c = pick + rng.normal(0.0, 1.0, size=pick.shape) * sd
        if fill > 0:
            # Uniform over the cloud's bounding box. DEFAULT OFF, and measured: the
            # bandwidth jitter along the soft subspace directions already covers 12/12
            # sectors of the pseudorotation circle on its own, so fill buys no coverage
            # and only costs population fidelity (cyclohexane 89% chair -> 69% at
            # fill=0.25) and closure (4.6 -> 5.5 bond-sigma). Kept as a knob for a ring
            # whose sampled basins turn out too sparse to bridge.
            m = rng.random(n) < fill
            if m.any():
                c[m] = rng.uniform(lo, hi, size=(int(m.sum()), self.coords.shape[1]))
        return (c @ self.components) * self.scale


def wrap_state(x):
    """Wrap to the state space (-1, 1]. Period 2, NOT 2*pi -- see conformer_torsions."""
    return (x + 1.0) % 2.0 - 1.0


# ---------------------------------------------------------------- construction


def condition_from_energy(energy, identifier: Optional[str] = None,
                          partial_charges: bool = True) -> MolData:
    """A conformer condition graph, built from a live ``ConformerTorsions``.

    Taking the energy as the source (rather than re-deriving from SMILES) is what
    guarantees the graph and the energy agree on all three things a stored state depends
    on: the bond graph, the reference conformer, and which bonds are rotatable. Rebuilding
    any of them independently gives a file whose states silently mean something else --
    the reference conformer especially, since RDKit embedding is seed-dependent and
    ``phi0`` is what the delta is a delta *from*.

    ``identifier`` defaults to the SMILES. train.py resolves condition identity purely
    through this string (``init_identifiers``), so distinct conditions need distinct
    identifiers -- pass one when the same molecule appears under several conditions.
    """
    spec = energy.spec
    n = spec.n_atoms
    dtype = energy.dtype
    slots = np.arange(n, dtype=np.int64)

    as_long = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long)
    as_bool = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.bool)
    as_real = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)

    mol = MolData(
        z=as_long(spec.z),
        # already in placement order: energy.ref_pos is ref_pos[spec.perm]
        pos=energy.ref_pos.detach().cpu().to(dtype),
        smiles=energy.smiles,
        identifier=identifier if identifier is not None else energy.smiles,
    )

    if partial_charges:
        q = _gasteiger_charges(energy.mol)
        if q is not None:
            mol.x = as_real(q[spec.perm]).reshape(-1, 1)

    # rank == min(slot, 3): the atom at slot k owns a bond (k>=1), an angle (k>=2) and a
    # dihedral (k>=3). round_id agrees with it over 0..3 -- rounds 1 and 2 contain exactly
    # slots 1 and 2, since a slot-2 atom needs two distinct placed references and so
    # cannot sit at depth 1 -- which is what lets `build`'s round-keyed seed rules and
    # these rank-keyed masks describe the same partition.
    mol.ctree_round = as_long(spec.round_id)
    rank = np.minimum(spec.round_id, 3)

    # references as DELTAS from the owning atom: invariant under every re-offsetting a
    # buffer draw performs, where an absolute index would not be (see the module
    # docstring's storage rule). 0 where the DoF does not exist.
    mol.ctree_ref_a = as_long(np.where(rank >= 3, spec.ref_a - slots, 0))
    mol.ctree_ref_b = as_long(np.where(rank >= 2, spec.ref_b - slots, 0))
    mol.ctree_ref_c = as_long(np.where(rank >= 1, spec.ref_c - slots, 0))

    # reference internals, scattered onto the owning atom. energy.r0/th0/ph0 were
    # measure()'d off the same ref_pos, so these are exactly the values build_positions
    # freezes r and theta at.
    mol.ctree_r0 = _scatter(energy.r0, rank >= 1, n, dtype)
    mol.ctree_theta0 = _scatter(energy.th0, rank >= 2, n, dtype)
    mol.ctree_phi0 = _scatter(energy.ph0, rank >= 3, n, dtype)

    mol.ctree_angle_is_linear = _scatter(as_bool(spec.angle_is_linear), rank >= 2, n,
                                         torch.bool)
    mol.ctree_torsion_is_proper = _scatter(as_bool(spec.torsion_is_proper), rank >= 3, n,
                                           torch.bool)
    mol.ctree_torsion_frame_is_linear = _scatter(
        as_bool(spec.torsion_frame_is_linear), rank >= 3, n, torch.bool)

    # the sampled-torsion selection, sparsified from energy.mask onto owning atoms
    mol.ctree_state_col = _scatter(_state_columns(energy), rank >= 3, n, torch.long,
                                   fill=-1)

    closure, closure_r0 = _closure_fields(spec, mol.pos, n)
    mol.ctree_closure = closure
    mol.ctree_closure_r0 = closure_r0.to(dtype)

    mol.n_torsions = torch.tensor([energy.data_ndim], dtype=torch.long)
    return mol


def condition_from_smiles(smiles: str, identifier: Optional[str] = None,
                          **energy_kwargs) -> MolData:
    """``condition_from_energy`` for callers that only have a SMILES.

    Constructs a ``ConformerTorsions`` and serialises it. Building the energy is the
    point, not overhead: it is the single definition of the reference conformer and the
    rotatable-bond set, and going around it is how the graph and the energy end up
    disagreeing. ``energy_kwargs`` must match what the run will use --
    ``include_trivial_rotations`` and ``seed`` change the stored parameterisation itself,
    the force-field constants do not.
    """
    from energies.conformer_torsions import ConformerTorsions

    energy = ConformerTorsions(smiles=smiles, device='cpu', **energy_kwargs)
    return condition_from_energy(energy, identifier=identifier)


def _scatter(values, mask, n: int, dtype, fill=0):
    """Place a compact per-DoF vector onto its owning atoms, ``[n]``.

    The per-atom counterpart of ``builder._scatter_to_atoms``, with an explicit fill so
    an absent DoF reads as a chosen value rather than a coincidental zero.
    """
    out = torch.full((n,), fill, dtype=dtype)
    out[torch.as_tensor(mask)] = torch.as_tensor(np.asarray(values) if not
                                                 torch.is_tensor(values) else values
                                                 ).detach().cpu().to(dtype)
    return out


def _closure_fields(spec, pos, n: int):
    """Ring-closure partner deltas and as-built lengths, one per atom.

    Closure bonds are absent from the spanning tree, so their lengths are *determined* by
    the other DoF rather than settable -- recording the reference is what makes closure
    drift measurable. Each closure is assigned to one free endpoint. An atom that would
    need two (possible in fused polycyclics) raises: the per-atom encoding is what makes
    the tree survive a buffer draw at all, and dropping a closure to preserve it would
    silently weaken a diagnostic.
    """
    delta = torch.zeros(n, dtype=torch.long)
    r0 = torch.zeros(n, dtype=pos.dtype)
    taken = np.zeros(n, dtype=bool)

    for i, j in np.asarray(spec.broken_bond_index, dtype=np.int64).reshape(-1, 2):
        home = int(i) if not taken[i] else (int(j) if not taken[j] else -1)
        if home < 0:
            raise ValueError(
                f"ring-closure bond ({i}, {j}) has no free endpoint: both atoms already "
                f"carry a closure. The per-atom encoding allows one each (see "
                f"_closure_fields); this molecule needs a per-bond channel")
        other = int(j) if home == int(i) else int(i)
        taken[home] = True
        delta[home] = other - home
        r0[home] = torch.linalg.norm(pos[home] - pos[other])
    return delta, r0


def _gasteiger_charges(rd_mol):
    """Gasteiger charges in RDKit atom order, or None if RDKit declines.

    Stored rather than left to the training run because computing them needs RDKit and a
    mol object, neither of which the train loop has -- and ``x`` is where every mxtaltools
    graph model looks for per-atom features.
    """
    try:
        from mxtaltools.dataset_utils.mol_building import get_partial_charges
        q = get_partial_charges(rd_mol)
    except Exception as exc:  # noqa: BLE001 -- charge assignment fails on exotic valences
        print(f"conformer_data: Gasteiger charges unavailable ({exc}); leaving x unset")
        return None
    if not np.isfinite(q).all():
        print("conformer_data: Gasteiger charges non-finite; leaving x unset")
        return None
    return q


def _state_columns(energy) -> torch.Tensor:
    """``ConformerTorsions.mask`` as an ``[N-3]`` column index, -1 where frozen.

    The dense mask is a 0/1 selection matrix: column j is
    ``(torsion_index[:,1] == u) & (torsion_index[:,2] == v)`` for axis j, and distinct
    axes give disjoint row sets. Asserting that rather than trusting it means a future
    change to the rotatable-bond rule that broke the property fails at dataset-build
    time, not as a wrong energy months later.
    """
    mask = energy.mask.detach().cpu()
    n_rows = mask.shape[0]
    if mask.shape[1] == 0:
        return torch.full((n_rows,), -1, dtype=torch.long)

    hits = (mask != 0)
    per_row = hits.sum(1)
    if int(per_row.max()) > 1:
        bad = torch.nonzero(per_row > 1).flatten().tolist()
        raise ValueError(
            f"torsion rows {bad} are driven by more than one rotatable axis; the sparse "
            f"ctree_state_col form assumes a 0/1 selection mask (see _state_columns)")

    col = torch.full((n_rows,), -1, dtype=torch.long)
    rows = torch.nonzero(per_row == 1).flatten()
    col[rows] = hits[rows].to(torch.uint8).argmax(1).long()
    return col


# -------------------------------------------------------------------- collation


def collate_conditions(mols: Sequence[MolData], require_state: bool = False):
    """Collate conformer graphs into the batch a buffer holds, k-uniformity enforced.

    A mixed-k file cannot be trained by the current GFN (its state dimension is fixed at
    construction), and without this check the failure surfaces as a shape error deep in
    the policy with nothing pointing back at the dataset.
    """
    if len(mols) == 0:
        raise ValueError("no conformer conditions to collate")

    ks = [int(m.n_torsions) for m in mols]
    if len(set(ks)) > 1:
        counts = {k: ks.count(k) for k in sorted(set(ks))}
        raise ValueError(
            f"conformer conditions have mixed state dimensions {counts}; one file is one "
            f"k while the GFN's state dimension is fixed at construction. Split the set "
            f"by k, or wait for the variable-dimension path")

    batch = collate_data_list(list(mols))
    require_conformer_fields(batch, require_state=require_state)
    return batch


def require_conformer_fields(batch, require_state: bool = False):
    """Assert a batch carries the conformer parameterisation (and optionally a state).

    Cheap, and it turns "something downstream got None" into one message naming the
    missing keys and the builder that should have written them.
    """
    keys = set(batch._store.keys())
    missing = [f for f in CONFORMER_FIELDS if f not in keys]
    if require_state:
        missing += [f for f in STATE_FIELDS if f not in keys]
    if missing:
        raise AttributeError(
            f"batch is missing conformer fields {missing}; build it with "
            f"energies.conformer_data (condition_from_energy / attach_states)")
    return batch


def state_dim(batch) -> int:
    """The single state dimension k of a conformer batch."""
    k = batch.n_torsions.reshape(-1)
    if int(k.min()) != int(k.max()):
        raise ValueError(f"batch has mixed state dimensions "
                         f"({int(k.min())}..{int(k.max())})")
    return int(k[0])


def batch_states(batch) -> torch.Tensor:
    """``torsion_state`` as ``[n_graphs, k]``."""
    return batch.torsion_state.reshape(batch.num_graphs, state_dim(batch))


def dof_rank(batch) -> torch.Tensor:
    """Per-atom DoF ownership rank, ``min(slot, 3)`` -- see ``ctree_round``."""
    return batch.ctree_round.clamp(max=3)


def rotatable_axes(batch):
    """The (u, v) axis atoms per state column, for the first graph in the batch.

    Derived rather than stored: the driven torsion rows about axis (u, v) are exactly
    those whose reference atoms are ``ref_b == u, ref_c == v``, so the axis is recoverable
    from any one atom carrying that state column. Readable check that a file selected the
    bonds you meant.
    """
    col = batch.ctree_state_col
    atoms = torch.arange(col.shape[0], device=col.device)
    graph = batch.batch if batch.is_batch else torch.zeros_like(col)
    out = []
    for j in range(state_dim(batch)):
        rows = torch.nonzero((col == j) & (graph == 0), as_tuple=True)[0]
        if rows.numel() == 0:
            out.append(None)
            continue
        a = rows[0]
        out.append((int(atoms[a] + batch.ctree_ref_b[a]),
                    int(atoms[a] + batch.ctree_ref_c[a])))
    return out


# ------------------------------------------------------------------- tree readout


def batch_tree(batch):
    """Reconstruct a ``BatchedTree`` from the per-atom encoding.

    The counterpart of ``MolConformerMethods.as_conformer_tree``, reading this module's
    node-level fields instead of mxtaltools' per-DoF ones -- so it is correct on a batch
    that has been drawn, purged and appended, which that one is not (see the module
    docstring's storage rule).

    Row order within each DoF block is atom order, which within a molecule is placement
    order -- the same order ``builder.collate`` produces, so ``build``/``measure`` and the
    reference vectors here stay aligned.
    """
    from mxtaltools.conformers.builder import BatchedTree

    n_atoms = int(batch.z.shape[0])
    dev = batch.z.device
    atoms = torch.arange(n_atoms, dtype=torch.long, device=dev)
    graph = (batch.batch if batch.is_batch
             else torch.zeros(n_atoms, dtype=torch.long, device=dev))
    n_mols = int(batch.num_graphs) if batch.is_batch else 1
    ptr = (batch.ptr if batch.is_batch
           else torch.tensor([0, n_atoms], dtype=torch.long, device=dev))

    rank = dof_rank(batch)
    m1, m2, m3 = rank >= 1, rank >= 2, rank >= 3

    # deltas -> absolute, then -1 where the DoF does not exist (the sentinel `build`
    # relies on for the seed rounds)
    ref_c = torch.where(m1, atoms + batch.ctree_ref_c, torch.full_like(atoms, -1))
    ref_b = torch.where(m2, atoms + batch.ctree_ref_b, torch.full_like(atoms, -1))
    ref_a = torch.where(m3, atoms + batch.ctree_ref_a, torch.full_like(atoms, -1))

    bond_index = torch.stack([ref_c[m1], atoms[m1]], dim=-1)
    angle_index = torch.stack([ref_b[m2], ref_c[m2], atoms[m2]], dim=-1)
    torsion_index = torch.stack([ref_a[m3], ref_b[m3], ref_c[m3], atoms[m3]], dim=-1)

    n_rounds = int(batch.ctree_round.max()) + 1
    rounds = [torch.nonzero(batch.ctree_round == t, as_tuple=True)[0]
              for t in range(n_rounds)]

    has_closure = batch.ctree_closure != 0
    broken = torch.stack([atoms[has_closure],
                          (atoms + batch.ctree_closure)[has_closure]], dim=-1)
    # graph bonds = tree bonds + ring closures, which is the whole bond set by
    # construction: a spanning tree plus the edges it could not represent
    graph_bond_index = torch.cat([bond_index, broken], dim=0)

    return BatchedTree(
        n_mols=n_mols, n_atoms=n_atoms, z=batch.z.long(), batch=graph,
        ref_a=ref_a, ref_b=ref_b, ref_c=ref_c, rounds=rounds,
        bond_index=bond_index, angle_index=angle_index, torsion_index=torsion_index,
        torsion_is_proper=batch.ctree_torsion_is_proper[m3],
        angle_is_linear=batch.ctree_angle_is_linear[m2],
        torsion_frame_is_linear=batch.ctree_torsion_frame_is_linear[m3],
        bond_batch=graph[m1], angle_batch=graph[m2], torsion_batch=graph[m3],
        broken_bond_index=broken, broken_bond_batch=graph[has_closure],
        graph_bond_index=graph_bond_index,
        graph_bond_batch=torch.cat([graph[m1], graph[has_closure]]),
        ptr=ptr, perm=atoms,
    )


def reference_internals(batch):
    """The frozen reference ``(r, theta, phi)`` vectors, aligned with ``batch_tree``."""
    rank = dof_rank(batch)
    return (batch.ctree_r0[rank >= 1],
            batch.ctree_theta0[rank >= 2],
            batch.ctree_phi0[rank >= 3])


# ------------------------------------------------------------ state <-> geometry


def state_to_phi(batch, state: torch.Tensor) -> torch.Tensor:
    """Torsion state ``[B, k]`` on [-1, 1] -> the batch's dihedral vector.

    The graph-native form of ``ConformerTorsions.build_positions``'s
    ``phi = phi0 + mask @ (pi * x)``: a pure translation on the torus, so the Jacobian
    stays constant. Driven atoms take their column's delta; frozen ones keep ``phi0``.
    """
    rank = dof_rank(batch)
    col = batch.ctree_state_col
    phi0 = batch.ctree_phi0
    graph = (batch.batch if batch.is_batch
             else torch.zeros_like(col))

    state = torch.as_tensor(state).to(phi0.dtype).reshape(-1, state_dim(batch))
    driven = (col >= 0)
    delta = torch.zeros_like(phi0)
    delta[driven] = np.pi * state[graph[driven], col[driven]]
    return (phi0 + delta)[rank >= 3]


def states_to_positions(batch, state: torch.Tensor) -> torch.Tensor:
    """Torsion state ``[B, k]`` -> Cartesian positions ``[A, 3]`` for the whole batch.

    Uses the stored reference r/theta, so local geometry is frozen exactly as the energy
    freezes it. The graph-driven equivalent of ``ConformerTorsions._batch``'s cached tree,
    and correct for a heterogeneous batch where that cache (one tree per batch *size*) is
    not.
    """
    from mxtaltools.conformers.builder import build

    r, theta, _ = reference_internals(batch)
    return build(batch_tree(batch), r, theta, state_to_phi(batch, state))


# ------------------------------------------------- state-bearing (prior) rows


def attach_states(condition: MolData, states, energies, identifier: Optional[str] = None):
    """Replicate one condition graph per state, into the batch a prior file holds.

    One graph per row is what the buffer *is*: it holds a resident batch and indexes it
    row-wise, so N states of one molecule are N copies of that molecule's graph. The
    replication is genuinely redundant, and for a large prior over a large molecule it
    dominates the file -- but it is the buffer's data model, not a choice made here.

    ``energies`` must be RAW energy (kcal/mol, T = 1): ``prebuilt_sample_to_reward``
    divides by the sampling temperature itself, so a pre-divided energy is applied twice.

    Refuses to emit a single row. A one-graph batch's per-graph tensors have
    ``size(0) == 1``, which every ``MXtalBase`` batch op reads as shared metadata and
    passes through unindexed -- the same wart ``generate_toy_prior.replicate`` clamps for.
    """
    dtype = condition.pos.dtype
    states = torch.as_tensor(states, dtype=dtype)
    states = wrap_state(states.reshape(states.shape[0], -1))
    energies = torch.as_tensor(energies, dtype=dtype).flatten()
    k = int(condition.n_torsions)

    if states.shape[1] != k:
        raise ValueError(f"states are {states.shape[1]}-dimensional but the condition has "
                         f"{k} rotatable torsions")
    if len(energies) != len(states):
        raise ValueError(f"{len(states)} states against {len(energies)} energies")
    if len(states) < 2:
        raise ValueError("need at least 2 states: per-graph tensors on a one-graph batch "
                         "are indistinguishable from shared metadata (see docstring)")

    rows = []
    for i in range(len(states)):
        row = condition.__copy__()
        row.torsion_state = states[i:i + 1]     # [1, k] -> graph-level on collation
        row.conformer_energy = energies[i:i + 1]
        if identifier is not None:
            row.identifier = identifier
        rows.append(row)
    return collate_conditions(rows, require_state=True)


def bake_energies(energy, states, chunk: int = 4096) -> torch.Tensor:
    """Raw POTENTIAL for a state block, in the force field's own units.

    Two things this deliberately excludes, for the same reason: the stored number is
    divided by the sampling temperature when ``prebuilt_sample_to_reward`` reads it back.

    - The temperature. Hence ``log_temperature = 0`` rather than the energy's configured
      one, so a T does not leak in and get divided out twice.
    - **The change of measure.** ``energy()`` carries ``-T log J``; dividing that by T
      gives ``-log J`` only at T = 1, so baking it would make the measure term scale as
      1/T -- and a change of measure is by definition temperature-independent. It is
      therefore added back on the read side, in log-reward units, after the division.

    So this calls ``potential_energy`` explicitly rather than ``energy``. The distinction
    did not exist before ladder step 2 and the two were the same function.
    """
    one = torch.tensor(1.0, dtype=energy.dtype, device=energy.device)  # T = 10**0
    states = torch.as_tensor(states, dtype=energy.dtype, device=energy.device)
    return torch.cat([energy.potential_energy(states[i:i + chunk], one)
                      for i in range(0, len(states), chunk)])


# ----------------------------------------------------------------------- files


def save_condition_file(batch, path):
    """Write the ``molecules_path`` form: ``{'prior': batch}``.

    The key is 'prior' only because ``_load_condition_file`` looks for it (a condition set
    is not a prior); matching the crystal convention means the conformer file loads
    through that same unmodified reader.
    """
    torch.save({'prior': batch}, path)
    return path


def save_prior_file(batch, path, equalized=None, **extra):
    """Write the ``prior_path`` form: ``{'prior': ..., 'equalized_prior': ...}``.

    ``init_prior_dataset`` reads ``equalized_prior`` and nothing else, so that key is the
    one that matters; 'prior' rides along for symmetry with the crystal/toy files. When
    ``equalized`` is None the same batch serves both roles -- for a mode-covering
    conformer prior, the distinction the crystal files draw (Boltzmann-weighted vs
    per-condition-equalised) has no content yet.

    ``thermal_scaling_factor`` is deliberately NOT written: in ``init_prior_dataset`` that
    key silently replaces the config's ``lj_coeff`` for the whole run. The conformer force
    field is already in kcal/mol, so there is no unit conversion to apply and writing the
    key would mean a scale factor nobody chose.
    """
    torch.save({'prior': batch,
                'equalized_prior': batch if equalized is None else equalized,
                **extra}, path)
    return path


# ------------------------------------------------------------------ validation


def check_state_convention(condition: MolData, energy, n: int = 64,
                           tol: float = 1e-9, seed: int = 0) -> float:
    """Assert the stored parameterisation reproduces the energy's own geometry.

    The whole format rests on one claim -- that a state tensor means the same thing read
    through the graph as through ``ConformerTorsions`` -- and that claim is directly
    checkable: build positions both ways for random states and compare. Worth running
    whenever a condition file is written, because every failure mode here (permuted atoms,
    a reordered state column, a different reference conformer) produces plausible energies
    rather than an error.

    The comparison is made **through a subsample**, not on the freshly collated batch:
    ``subsample_new_batch`` is where a per-DoF or absolute-index field goes stale, so a
    check that skipped it would pass on exactly the encoding this module exists to avoid.

    Returns the max absolute position discrepancy.
    """
    g = torch.Generator().manual_seed(seed)
    k = int(condition.n_torsions)
    states = torch.rand((n, k), generator=g, dtype=energy.dtype) * 2 - 1

    full = collate_data_list([condition] * (2 * n))
    batch = full.subsample_new_batch(np.arange(n))
    pos_graph = states_to_positions(batch, states)
    pos_energy = energy.build_positions(states)

    err = (pos_graph - pos_energy).abs().max().item()
    if not err < tol:
        raise AssertionError(
            f"stored conformer parameterisation disagrees with ConformerTorsions by "
            f"{err:.3g} A (tol {tol:g}); the state convention is not shared -- check "
            f"atom ordering (placement order?), ctree_state_col, and ctree_phi0")
    return err
