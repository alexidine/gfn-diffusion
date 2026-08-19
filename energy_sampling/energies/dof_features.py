"""Per-degree-of-freedom feature vectors -- ``f_j`` in the policy design.

WHAT THIS IS FOR. Today's conformer policy is a flat MLP over the whole state vector, so
its width is tied to one molecule and nothing tells it that column 7 is a ring torsion and
column 8 is a methyl spin. Every candidate replacement -- a set model, a variable-dimension
policy, a conditioned one -- needs the same thing first: a description of each coordinate
that travels WITH the coordinate instead of being implied by its column index. That is this
module, and it commits to none of those designs.

See docs/design/conformer_policy_architecture.md section 4A.

THE IDENTITY IS IN THE FEATURES, NOT THE INDEX. Storage order comes from the spanning-tree
traversal and is physically arbitrary: permuting it means nothing. A policy keyed on column
index cannot know that; a policy keyed on ``f_j`` is equivariant to it for free. So the one
property this module owes its consumers is that the features are a pure function of the
coordinate's CHEMISTRY, never of where it happens to sit in the table --
``test_dof_features.py`` asserts exactly that by re-deriving the same molecule from a
differently-ordered SMILES and requiring the same rows back.

THE FORCE FIELD ALREADY KNOWS THE STIFFNESS, so it is handed over rather than learned. Each
row carries ``log sqrt(kT/2k)`` -- its own thermal width, from the force field's own
constants. A learned embedding that has to rediscover ``k_angle`` from reward signal is
spending capacity on something available for free, which is the same argument that made the
thermal r/theta path beat fitted histograms in the prior.

NUMBERING. Everything here is in SPEC row numbering: ``r`` rows index ``spec.bond_index``,
``theta`` rows ``spec.angle_index``, ``phi`` rows ``spec.torsion_index``, concatenated in
that order to match the DoF vector the sampler and the state map use. Force-field constants
are NOT read from ``ff.k_bond`` / ``ff.k_angle`` directly: those are indexed by the GRAPH
bond/angle lists, which are a different and longer set (ethylcyclohexane: 24 graph bonds
against 23 tree bonds). ``thermal_rtheta_sigma`` does that lookup by atom identity and is
the only safe route.
"""
from __future__ import annotations

import numpy as np

# Elements the conformer track actually sees, plus a catch-all. Ordered, so a stored
# feature matrix keeps its meaning.
ELEMENTS = (1, 6, 7, 8, 9, 15, 16, 17)
N_ELEM = len(ELEMENTS) + 1          # + "other"
MAX_FRAME = 4                       # a torsion spans 4 atoms; r spans 2, theta 3


def _elem_onehot(z: int) -> np.ndarray:
    v = np.zeros(N_ELEM)
    v[ELEMENTS.index(z) if z in ELEMENTS else len(ELEMENTS)] = 1.0
    return v


def feature_names() -> list:
    """One name per column. Kept beside the builder so a feature cannot drift unnamed."""
    names = ['kind_r', 'kind_theta', 'kind_phi']
    for s in range(MAX_FRAME):
        names += ['a{}_z_{}'.format(s, e) for e in ELEMENTS] + ['a{}_z_other'.format(s)]
        names += ['a{}_degree'.format(s), 'a{}_in_ring'.format(s),
                  'a{}_aromatic'.format(s), 'a{}_parity'.format(s),
                  'a{}_present'.format(s)]
    names += ['row_in_ring', 'row_aromatic', 'is_improper',
              'is_group_member', 'is_rotatable', 'is_free_at_tier']
    names += ['log_thermal_sigma', 'ref_r', 'ref_theta']
    return names


# Columns that are EXACTLY reproducible from the graph, and columns that carry embedding
# noise. The split is not cosmetic: the reference conformer comes from an RDKit embedding,
# so anything measured off it inherits a seed dependence. Consumers that need a stable key
# (caching, cross-molecule matching, tests) must use the categorical block alone.
CONTINUOUS = ('log_thermal_sigma', 'ref_r', 'ref_theta')


def categorical_columns() -> list:
    n = feature_names()
    return [i for i, c in enumerate(n) if c not in CONTINUOUS]


def dof_features(en, prior=None) -> np.ndarray:
    """``[n_dof, F]`` features, one row per internal coordinate, in SPEC numbering.

    ``prior`` is optional and only adds the ring-block class; the rest is a property of the
    molecule and the force field alone. Passing it is what lets a policy distinguish "this
    ring's pucker is sampled from a bank" from "this ring is held", which are different
    dynamics for the same chemistry.
    """
    spec = en.spec
    bi = np.asarray(spec.bond_index)
    ai = np.asarray(spec.angle_index)
    ti = np.asarray(spec.torsion_index)
    z = np.asarray(spec.z)
    keys = en.atom_keys                       # [n_atoms, 2] = (z, graph degree)
    in_ring = en.atom_in_ring
    arom = en.atom_is_aromatic
    # the chirality pseudoscalar. Without it the features are ENANTIOMER-BLIND: a 2D graph
    # plus atom types is identical for a mirror pair. See atom_parity.
    parity = atom_parity(en)

    T = float(en.temperature)
    s_r, s_th = en.thermal_rtheta_sigma(T)    # SPEC-ordered, looked up by atom identity
    groups = en.torsion_groups()
    g_sigma = en.sibling_jitter_sigma(groups, T)
    s_imp = en.improper_phi_sigma(T)
    improper = set(en.improper_phi_rows())
    member = {j for g in groups for j in g}
    # a phi row is "rotatable" when its central bond is one of the rotatable axes
    rot_bonds = {tuple(sorted(uv)) for uv in en.rotatable}

    # ph0 is deliberately ABSENT. Two things measured off the reference conformer behave
    # very differently: r0 and theta0 sit near the force field's own equilibria and move by
    # ~4e-3 A / 3e-2 rad between embeddings, which is chemistry plus noise. ph0 is the
    # arbitrary rotational zero of the embedding and moves by up to 2.1 RAD between two
    # SMILES orderings of the SAME molecule -- it carries no chemistry at all, and the
    # state is already expressed as a displacement from it, so feeding it in would hand the
    # policy the embedding seed and nothing else.
    r0 = en.r0.detach().cpu().numpy()
    th0 = en.th0.detach().cpu().numpy()
    free = np.asarray(en.free_mask)

    rows = []
    for kind, table, n in (('r', bi, en.n_r), ('theta', ai, en.n_th), ('phi', ti, en.n_ph)):
        for j in range(n):
            atoms = [int(a) for a in table[j]]
            f = [1.0 if kind == k else 0.0 for k in ('r', 'theta', 'phi')]
            for s in range(MAX_FRAME):
                if s < len(atoms):
                    a = atoms[s]
                    f += list(_elem_onehot(int(z[a])))
                    f += [float(keys[a, 1]), float(in_ring[a]), float(arom[a]),
                          float(parity[a]), 1.0]
                else:
                    f += [0.0] * N_ELEM + [0.0, 0.0, 0.0, 0.0, 0.0]
            f.append(float(all(in_ring[a] for a in atoms)))
            f.append(float(all(arom[a] for a in atoms)))
            if kind == 'phi':
                gi = next((i for i, g in enumerate(groups) if j in g), None)
                sig = (s_imp if j in improper
                       else g_sigma[gi] if gi is not None else s_imp)
                central = tuple(sorted((int(ti[j, 1]), int(ti[j, 2]))))
                f += [float(j in improper), float(j in member),
                      float(central in rot_bonds)]
                row_global = en.n_r + en.n_th + j
                f += [float(np.log(max(sig, 1e-12))), 0.0, 0.0]
            else:
                sig = s_r[j] if kind == 'r' else s_th[j]
                row_global = j if kind == 'r' else en.n_r + j
                f += [0.0, 0.0, 0.0]
                f += [float(np.log(max(sig, 1e-12))),
                      float(r0[j]) if kind == 'r' else 0.0,
                      float(th0[j]) if kind == 'theta' else 0.0]
            # the tier flag goes in last so the block above stays tier-invariant
            f.insert(len(f) - 3, float(free[row_global]))
            rows.append(f)
    out = np.asarray(rows, dtype=np.float64)
    assert out.shape == (en.spec.n_dof, len(feature_names())), \
        (out.shape, len(feature_names()))
    return out


def state_features(en, prior=None):
    """``[data_ndim, F + 1]`` features per STATE COLUMN, not per DoF row.

    THE POLICY ACTS ON THE STATE, AND THE STATE IS NOT THE DoF VECTOR. At a selection tier
    each state column drives exactly one internal coordinate, so this is a row lookup. At
    ``torsion`` a column is COLLECTIVE -- rotating one bond shifts every dihedral about it,
    generally several -- and there is no single row to describe it. Those columns get the
    mean of their driven rows' features, plus a trailing count of how many rows they drive,
    which is the one thing that distinguishes a collective column from a selection one and
    would otherwise be invisible.

    Averaging is a real approximation and is why the count is carried: a policy that needs
    to tell "one dihedral" from "four dihedrals moving together" can read it directly rather
    than inferring it from a smeared feature vector.
    """
    f = dof_features(en, prior)
    m = en._M.detach().cpu().numpy()                 # [n_driven, n_cols]
    driven = en._driven_idx.detach().cpu().numpy()   # DoF rows, in _M's row order
    out = np.zeros((m.shape[1], f.shape[1] + 1))
    for j in range(m.shape[1]):
        rows = driven[np.flatnonzero(m[:, j])]
        out[j, :-1] = f[rows].mean(0)
        out[j, -1] = len(rows)
    return out


def state_feature_names() -> list:
    return feature_names() + ['n_driven_rows']


def atom_parity(en) -> np.ndarray:
    """``[n_atoms]`` in {-1, 0, +1} -- the chirality pseudoscalar, in placement numbering.

    WHY THIS EXISTS. A 2D graph plus atom types is IDENTICAL for two enantiomers, so any
    encoder over it -- message passing or attention -- is enantiomer-blind unless parity
    enters as an explicit atom feature. conformer_conditional_stack.md section 6 records
    that nothing in the suite currently fails if it is absent.

    ONLY AT REAL STEREOCENTRES, and that restriction is what makes it reproducible. The
    signed triple product of three neighbours is defined at ANY atom of degree >= 3, but at
    a centre carrying two identical substituents its sign flips when those two are
    exchanged -- so it would encode the placement order's tie-breaking rather than
    chemistry, and would differ between two SMILES orderings of one molecule. RDKit's
    perceived stereocentres are the set where the sign is an invariant of the molecule.

    The sign is read off the reference conformer, which is legitimate here in a way it was
    NOT for the reference dihedral: the embedding respects the SMILES stereo tags, so parity
    is discrete and stereochemically determined, where ph0 was a continuous quantity fixed by
    the embedding's arbitrary rotational zero.
    """
    from rdkit import Chem
    mol = en.mol
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centres = {int(i) for i, _ in
               Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)}
    n = en.spec.n_atoms
    out = np.zeros(n)
    if not centres:
        return out
    # original atom index -> placement slot
    slot = np.empty(n, dtype=np.int64)
    slot[np.asarray(en.spec.perm)] = np.arange(n)
    pos = np.asarray(en.ref_pos.detach().cpu().numpy())      # already in placement order
    nbr = {}
    for u, v in np.asarray(en.spec.graph_bond_index):
        nbr.setdefault(int(u), []).append(int(v))
        nbr.setdefault(int(v), []).append(int(u))
    for orig in centres:
        a = int(slot[orig])
        ns = sorted(nbr.get(a, []))
        if len(ns) < 3:
            continue
        u, v, w = (pos[ns[k]] - pos[a] for k in range(3))
        out[a] = float(np.sign(np.dot(u, np.cross(v, w))))
    return out
