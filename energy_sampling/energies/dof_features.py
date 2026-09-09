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
    """One name per column. Kept beside the builder so a feature cannot drift unnamed.

    THE KIND BLOCK NAMES THE QUANTITY, NOT THE SLOT. A linear bend is carried as the
    transverse pair (u, v) in the theta and phi SLOTS of the atom it places, but neither
    component is an angle and neither is a dihedral: they are the two Cartesian components of
    one 2-D displacement on the open disc rho < pi, non-periodic, with equilibrium exactly at
    the origin. Labelling them kind_theta / kind_phi inverted every inductive bias this block
    exists to supply -- the policy was told a coordinate centred at 0 was a bond angle on
    (0, pi) whose reference was pi.

    TWO flags rather than one, because u and v are not interchangeable: u lies along m2 and v
    along n, built from different cross products (geometry.place_nerf_transverse). With a
    single flag the only thing separating them would be an incidental padding bit, which is
    exactly the implicit encoding this module exists to remove.

    Widening this block costs nothing in stored artefacts. An earlier note here claimed it
    "invalidates stored conditions files and the policy input width"; that was wrong on the
    first count and moot on the second. `condition_from_energy` stores no feature matrix, and
    a set-policy resume is already refused outright (conformer_modeller.py), so no loadable
    checkpoint carries this width. What DOES depend on a real artefact is the atom frame --
    see :func:`free_dof_atom_index`.
    """
    names = ['kind_r', 'kind_theta', 'kind_phi', 'kind_bend_u', 'kind_bend_v']
    for s in range(MAX_FRAME):
        names += ['a{}_z_{}'.format(s, e) for e in ELEMENTS] + ['a{}_z_other'.format(s)]
        names += ['a{}_degree'.format(s), 'a{}_in_ring'.format(s),
                  'a{}_aromatic'.format(s), 'a{}_parity'.format(s),
                  'a{}_present'.format(s)]
    names += ['row_in_ring', 'row_aromatic', 'is_improper',
              'is_group_member', 'is_rotatable', 'is_free_at_tier']
    names += ['log_thermal_sigma', 'ref_r', 'ref_theta', 'ref_bend']
    return names


# Columns that are EXACTLY reproducible from the graph, and columns that carry embedding
# noise. The split is not cosmetic: the reference conformer comes from an RDKit embedding,
# so anything measured off it inherits a seed dependence. Consumers that need a stable key
# (caching, cross-molecule matching, tests) must use the categorical block alone.
CONTINUOUS = ('log_thermal_sigma', 'ref_r', 'ref_theta', 'ref_bend')


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

    # THE TRANSVERSE PAIR. `transverse_angles` is per ANGLE row and `transverse_partner`
    # gives the torsion row holding v, so a phi row is a bend component exactly when it is
    # some flagged angle row's partner -- hence the inverse lookup. `_ref_dof` rather than
    # th0/ph0 because it is the vector `dof_from_state` actually starts from; th0/ph0 stay
    # POLAR for the prior histograms, and reading them here is what put pi into a slot whose
    # true value is ~0.
    tv = np.asarray(en.transverse_angles, dtype=bool)
    tv_partner = np.asarray(en.transverse_partner, dtype=np.int64)
    angle_of_v = {int(tv_partner[j]): int(j) for j in np.flatnonzero(tv)}
    ref_dof = en._ref_dof.detach().cpu().numpy()
    KINDS = ('r', 'theta', 'phi', 'bend_u', 'bend_v')

    rows = []
    for kind, table, n in (('r', bi, en.n_r), ('theta', ai, en.n_th), ('phi', ti, en.n_ph)):
        for j in range(n):
            # which row of a transverse pair is this, if either
            if kind == 'theta' and bool(tv[j]):
                bend, angle_row = 'bend_u', j
            elif kind == 'phi' and j in angle_of_v:
                bend, angle_row = 'bend_v', angle_of_v[j]
            else:
                bend, angle_row = None, None

            # THE FOUR-ATOM PLACEMENT FRAME ON BOTH ROWS OF A PAIR, not the 3-atom angle
            # frame. This is correctness, not convenience: place_nerf_transverse builds n
            # from (pb - pa) and m2 from n, so the direction u points in is fixed by atom a,
            # which the angle frame omits entirely. It also makes the two rows of a pair
            # identical in every atom column and different only in the kind bit and the
            # reference -- which is precisely the truth about them.
            atoms = [int(a) for a in (ti[tv_partner[angle_row]] if bend else table[j])]
            f = [1.0 if (bend or kind) == k else 0.0 for k in KINDS]
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

            row_global = (j if kind == 'r' else en.n_r + j if kind == 'theta'
                          else en.n_r + en.n_th + j)
            if bend:
                # BOTH components share the bend force constant of the angle they replace --
                # they are measured in radians of the same rho. Taking the phi branch's
                # sigma on the v row handed it a torsion jitter (log 0.1) for a bend.
                sig = s_th[angle_row]
                # is_improper / is_group_member / is_rotatable are ROTATION-ABOUT-A-BOND
                # semantics. An out-of-plane bend component is none of the three, and the v
                # row was picking up is_group_member from torsion_groups().
                flags = [0.0, 0.0, 0.0]
                refs = [0.0, 0.0, float(ref_dof[row_global])]
            elif kind == 'phi':
                gi = next((i for i, g in enumerate(groups) if j in g), None)
                sig = (s_imp if j in improper
                       else g_sigma[gi] if gi is not None else s_imp)
                central = tuple(sorted((int(ti[j, 1]), int(ti[j, 2]))))
                flags = [float(j in improper), float(j in member),
                         float(central in rot_bonds)]
                refs = [0.0, 0.0, 0.0]
            else:
                sig = s_r[j] if kind == 'r' else s_th[j]
                flags = [0.0, 0.0, 0.0]
                refs = [float(r0[j]) if kind == 'r' else 0.0,
                        float(th0[j]) if kind == 'theta' else 0.0, 0.0]
            # BUILT EXPLICITLY, in feature_names() order. This used to end with
            # `f.insert(len(f) - 3, tier_flag)`, which positioned the tier flag by counting
            # back from the end of the trailing reference block -- so appending `ref_bend`
            # would have shifted it one slot silently, with every shape still correct and
            # every value still finite. Naming the tail removes that trap.
            f += flags
            f += [float(free[row_global])]
            f += [float(np.log(max(sig, 1e-12)))] + refs
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


def free_dof_atom_index(en):
    """``(atoms [k, R, MAX_FRAME], mask [k, R])`` -- which atoms each STATE COLUMN moves.

    The learned counterpart to :func:`state_features`. That function averages handcrafted
    per-row features over a column's driven rows; this returns the ATOM INDICES instead, so a
    correlator can learn the same reduction over per-atom encoder embeddings --
    ``f_j = F_tau(g_i1, ..., g_in)`` from conformer_conditional_stack.md section 5.

    THE COLUMN ORDER IS ``_M``'s, WHICH IS THE STATE'S. It must be, or coordinate j's features
    describe coordinate k's atoms -- silent, since every shape still matches. The ordering is
    taken from the same ``en._M`` / ``en._driven_idx`` pair `state_features` reads, rather
    than re-deriving it, so the two cannot drift apart.

    A column at ``level='torsion'`` is COLLECTIVE: rotating one bond shifts every dihedral
    about it. So a column owns SEVERAL rows, ``R`` is the widest such set in this molecule,
    and short columns are zero-padded with ``mask`` false. Atom indices are in SPEC (tree)
    numbering, which is the numbering `models.encoder_cache` stores embeddings in.
    """
    spec = en.spec
    tables = {'r': np.asarray(spec.bond_index), 'theta': np.asarray(spec.angle_index),
              'phi': np.asarray(spec.torsion_index)}
    m = en._M.detach().cpu().numpy()
    driven = en._driven_idx.detach().cpu().numpy()
    per_col = [driven[np.flatnonzero(m[:, j])] for j in range(m.shape[1])]
    R = max((len(c) for c in per_col), default=1) or 1

    atoms = np.zeros((m.shape[1], R, MAX_FRAME), dtype=np.int64)
    mask = np.zeros((m.shape[1], R), dtype=bool)
    for j, rows in enumerate(per_col):
        for s_, row in enumerate(rows):
            row = int(row)
            if row < en.n_r:
                kind, local = 'r', row
            elif row < en.n_r + en.n_th:
                kind, local = 'theta', row - en.n_r
            else:
                kind, local = 'phi', row - en.n_r - en.n_th
            # A TRANSVERSE ANGLE ROW TAKES ITS PARTNER'S FOUR-ATOM FRAME, for the reason
            # dof_features does: both transverse directions are built from (pb - pa), so the
            # 3-atom angle frame omits the atom that fixes which way u points, and padding
            # would repeat an atom where a real one belongs. Changes the VALUES of dof_atoms,
            # not its shape -- see the note in build_conformer_conditions.
            if kind == 'theta' and bool(np.asarray(en.transverse_angles, dtype=bool)[local]):
                frame = [int(a) for a in tables['phi'][int(en.transverse_partner[local])]]
            else:
                frame = [int(a) for a in tables[kind][local]]
            # r spans 2 atoms and theta 3, so short frames REPEAT their last atom rather
            # than padding with index 0 -- index 0 is a real atom, and a correlator cannot
            # tell a padded slot from a genuine reference to it.
            frame = frame + [frame[-1]] * (MAX_FRAME - len(frame))
            atoms[j, s_] = frame[:MAX_FRAME]
            mask[j, s_] = True
    return atoms, mask


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
