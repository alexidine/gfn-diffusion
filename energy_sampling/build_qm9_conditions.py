"""Pin QM9 crystal molecules to the frame the trainer will actually use, bake a frozen
Mo3ENet embedding onto each in that frame, and write the ``molecules_path``/``prior_path``
pair for a conditional molecular run.

WHY THE FRAME IS THE WHOLE POINT
--------------------------------
The condition is the molecule; the target is its crystal. A crystal's
``aunit_orientation`` is expressed relative to the molecule's own frame, so rotating the
molecule does NOT rotate the packing solution -- it changes the orientation values
INVERSELY. Packing is anti-covariant with the molecule frame, not equivariant with it. A
molecule embedded in one frame and built in another therefore decorrelates the condition
from its target, silently and with no error anywhere.

``train.py`` (732, 2692, 4279, 4393, 5190) and ``buffer.py`` (203) each call
``orient_molecule(mode='std')`` before rollouts and at buffer admission. So the frame the
GFN builds in is whatever that function produces -- there is no way to impose a different
one from a data file. This script therefore does not invent a frame: it applies THAT SAME
function once, here, so the saved file is a FIXED POINT of it. The trainer's own calls
then move nothing, and the frame the embedding was taken in is provably the frame the
crystal is built in.

WHAT THIS COSTS, DELIBERATELY
-----------------------------
``align_mol_batch_to_standard_axes`` maps the molecule's principal-axis basis onto a fixed
right-handed target, but ``batch_molecule_principal_axes_torch`` returns a basis whose
handedness is arbitrary -- measured on QM9, ``det(Ip) == -1`` for 38 of 100 molecules.
Mapping a left-handed basis onto a right-handed target is a REFLECTION, not a rotation.
Two consequences, both accepted here rather than worked around:

  * Those molecules are stored as their MIRROR IMAGE. Self-consistent (the embedding and
    the build agree, which is what matters), but for chiral molecules it is the
    enantiomer that gets trained on.
  * Their stored ``aunit_orientation`` CANNOT be repaired -- a rotation vector cannot
    encode a reflection -- so those structures no longer index the crystal they came from.

The rotated majority IS repaired: wherever the recovered transform is a proper rotation,
``aunit_orientation`` is re-expressed as ``R_aunit @ R^-1`` and the crystal is preserved
exactly (verified per run, to ~1e-6 A). Every structure carries ``crystal_valid``, 1 or 0,
and the file records ``n_crystals_valid``/``n_crystals_total``. **A stage with
``bwd_sampling_mode: dataset`` trains on stored cell parameters and must therefore be
restricted to the crystal_valid subset** -- or the prior built by sampling instead.

Note the ``handedness`` argument does NOT control this: it only selects the TARGET frame
(``eye[:, 0, 0]``), and this dataset's ``aunit_handedness`` is already all +1 -- passing it
explicitly and passing None give byte-identical results (62 proper / 38 improper either
way). The reflection comes from the SOURCE basis, which has no knob.

A second, smaller consequence: the function is not idempotent for molecules whose inertia
tensor is near-degenerate (symmetric tops), where the axis assignment tie-breaks on
numerical noise -- about 2 in 100. Those molecules would land in a different frame
depending on how many times they had been oriented, which is exactly the mismatch this
script exists to prevent. They are fixed by JITTER: perturbing their atoms by ~1e-3 A
separates the degenerate moments, and baking that perturbation in here turns a random
tie-break into a permanent one, so the molecule is kept rather than discarded. Anything
still oscillating after ``--jitter-attempts`` is dropped and named.

ROTATIONAL AUGMENTATION is deliberately not done -- there is no equivariance target to
augment toward. If it is added later it must rotate the molecule and its embedding as one
object: ``mol_methods.rotate_embedding(R)`` and
``orient_molecule(mode='random', override_random_rotations=R, correct_orientation=True)``
are the matched pair, and ``crystal_building.get_aunit_positions`` carries a commented-out
sketch of the same idea at its line 778. Note that ``correct_orientation`` has no caller in
the codebase today and is unsound for the improper cases above.

    # the 20-molecule eval set
    python build_qm9_conditions.py --source D:\crystal_datasets\eval_qm9_sg2_dataset.pt \
        --out-dir D:\crystal_datasets\conditional\priors --tag qm9_20

    # the 1-molecule rung of the ladder
    python build_qm9_conditions.py --source D:\crystal_datasets\eval_qm9_sg2_dataset.pt \
        --out-dir D:\crystal_datasets\conditional\priors --tag qm9_1 --n-molecules 1

Mo3ENet's atom-type vocabulary is [1, 6, 7, 8, 9], exactly QM9's element set -- that is why
the molecule set has to be QM9 for this conditioner, and it is asserted, not assumed.
"""

import argparse
import collections
from pathlib import Path

import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from mxtaltools.common.geometry_utils import fractional_transform, rotvec2rotmat
from mxtaltools.common.training_utils import load_molecule_autoencoder
from mxtaltools.dataset_utils.utils import collate_data_list

# 64-dim bottleneck, atom types [1,6,7,8,9], trained on centered molecules
DEFAULT_ENCODER = r'D:\crystal_datasets\model_checkpoints\_best_autoencoder_experiments_dev_26-09-13-48-15'

MO3ENET_ATOM_TYPES = {1, 6, 7, 8, 9}


def normalize_schema(samples):
    """Fill in the fields the QM9 crystal datasets leave unset that the batch collater and
    the crystal builder both require.

    QM9 SG2 entries carry ``z_prime=None`` and a 1-D ``aunit_handedness``; every Z'=1
    consumer here expects an integer z_prime and handedness shaped ``[n_graphs, z_prime]``
    (compare the mipcas prior). Fixed here so the published file is self-contained.
    """
    for s in samples:
        if getattr(s, 'z_prime', None) is None:
            s.z_prime = torch.ones(1, dtype=torch.long)
        if s.aunit_handedness.ndim == 1:
            s.aunit_handedness = s.aunit_handedness.reshape(1, -1)
    return samples


def select_molecules(samples, n_molecules, smiles, min_replicas, seed):
    """Carve the ladder rung: keep whole molecules, never partial replica sets.

    Selection is by identifier (the SMILES string), which is what train.py resolves
    condition identity through -- so a rung is exactly a set of condition_ids.
    """
    by_id = collections.defaultdict(list)
    for s in samples:
        by_id[s.identifier].append(s)

    if min_replicas > 1:
        by_id = {k: v for k, v in by_id.items() if len(v) >= min_replicas}
        if not by_id:
            raise ValueError(f"no molecule in the source has >= {min_replicas} replicas")

    if smiles:
        missing = [m for m in smiles if m not in by_id]
        if missing:
            raise ValueError(
                f"requested molecules absent from the source (or below --min-replicas): "
                f"{missing}. Available example identifiers: {sorted(by_id)[:5]}")
        keep = list(smiles)
    elif n_molecules is not None:
        if n_molecules > len(by_id):
            raise ValueError(f"asked for {n_molecules} molecules, source has {len(by_id)}")
        # deterministic, replica-richest first: a 1- or 2-molecule rung wants the
        # molecules with the most structures, not an arbitrary pair
        keep = sorted(by_id, key=lambda k: (-len(by_id[k]), k))[:n_molecules]
    else:
        keep = sorted(by_id)

    out = []
    for k in keep:
        out.extend(by_id[k])
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(out), generator=g).tolist()
    return [out[i] for i in perm], keep


def drop_pathological(samples, lj_max):
    """Drop structures whose baked LJ potential is absurd.

    The eval QM9 set has a thin tail reaching +24k while the bulk sits near -320; those are
    broken cells, and leaving them in means the buffer's admission floor spends its whole
    budget rejecting them.
    """
    if lj_max is None:
        return samples, 0
    kept = [s for s in samples if float(s.lj_pot.flatten()[0]) <= lj_max]
    return kept, len(samples) - len(kept)


def reconstruct_parameterization(batch):
    """Derive a VALID (aunit_centroid, aunit_orientation, handedness) from the real crystal
    coordinates, and replace ``pos`` with the canonical centered molecule.

    The raw QM9 crystal datasets are CIF-derived: ``pos`` holds the asymmetric unit at its
    real coordinates (centroid ~7 A off origin) and the stored aunit parameters do NOT
    describe it -- posing them reproduces ``pos`` to a median of 2.8 A, i.e. not at all. A
    prepared prior (compare the mipcas file) is the other way round: ``pos`` is the
    canonical centered molecule and the placement lives entirely in the aunit parameters.
    Training reads the aunit parameters, so a raw file has to be converted before it means
    anything -- ``bwd_sampling_mode: dataset`` would otherwise fit MLE to noise.

    Two steps, the first only there to enable the second:
      1. The real molecule already sits at its true position, so its fractional centroid IS
         the true ``aunit_centroid``; with that set, ``build_unit_cell`` applies the stored
         symmetry operators and recovers the true unit cell.
      2. ``reparameterize_unit_cell`` reads that unit cell back out as a proper
         parameterization -- centroid, orientation, handedness, well-definedness -- plus the
         canonical aunit positions.

    The heavy-atom centroid is not a free choice: ``aunit2ucell`` centers on heavy atoms
    internally (crystal_building/utils.py:704), so any other convention shifts every
    molecule in the rebuilt cell.

    Validated against ``qm9_like_csd_crystals.pt``, the one dataset shipping a stored
    ``unit_cell_pos``: reconstructing its cell from ``pos`` + symmetry operators reproduces
    the stored one to a median of 0.0039 A (max 0.0078, 100% within 0.01).
    """
    heavy = batch.z != 1
    idx = batch.batch[heavy]
    sums = torch.zeros(batch.num_graphs, 3, dtype=batch.pos.dtype).index_add_(
        0, idx, batch.pos[heavy])
    counts = torch.zeros(batch.num_graphs, dtype=batch.pos.dtype).index_add_(
        0, idx, torch.ones(idx.shape[0], dtype=batch.pos.dtype))
    if int((counts == 0).sum()):
        raise ValueError("a molecule has no heavy atoms; the centroid convention is undefined")
    centroid_f = fractional_transform(sums / counts[:, None], batch.T_cf)

    batch.aunit_centroid = centroid_f - torch.floor(centroid_f)
    batch.build_unit_cell()

    centroid, orientation, handedness, well_defined, pos = batch.reparameterize_unit_cell()
    well_defined = torch.as_tensor(well_defined, dtype=torch.bool)  # returned as a list at Z'=1
    batch.aunit_centroid = centroid
    batch.aunit_orientation = orientation
    batch.aunit_handedness = handedness
    batch.is_well_defined = well_defined
    batch.pos = pos
    batch.unit_cell_pos = None  # rebuilt on demand from the new parameters

    offset = torch.stack([batch.pos[batch.batch == i].mean(0).norm()
                          for i in range(batch.num_graphs)]).median()
    print(f"  reparameterized: {int(well_defined.sum())}/{batch.num_graphs} well-defined, "
          f"median |molecule centroid| now {offset:.3f} A (canonical form)")
    return batch


def verify_against_baked_energy(batch, baked, tol_frac):
    """Check the rebuilt crystals reproduce the dataset's OWN baked energies.

    This is the check whose absence let an invalid parameterization through. Comparing a
    structure before and after a transform only proves the transform is self-consistent; it
    cannot notice that the thing being transformed was wrong to begin with. This one is
    absolute -- it scores the rebuilt crystal against the number the source file shipped.
    """
    probe = batch.clone()
    probe.analyze(['lj'], assign_outputs=True, cutoff=10, supercell_size=10,
                  std_orientation=False)
    fresh = probe.lj.flatten().cpu().float()
    baked = baked.cpu().float()
    scale = baked.abs().median().clamp(min=1.0)
    rel = (fresh - baked).abs() / scale
    median_rel = float(rel.median())
    print(f"  rebuilt vs baked lj_pot: median relative error {median_rel:.2%} "
          f"(fresh median {fresh.median():.1f}, baked median {baked.median():.1f})")
    if median_rel > tol_frac:
        raise RuntimeError(
            f"rebuilt crystals do not reproduce the source file's own energies (median "
            f"relative error {median_rel:.1%} > {tol_frac:.0%}). The parameterization does "
            f"not describe the source structures, so every downstream check would be "
            f"verifying a crystal that was never in the dataset. Refusing to write.")
    return median_rel


def posed_unit_cell(batch):
    """Atom positions of the built unit cell, on a CLONE.

    ``pose_aunit`` overwrites ``batch.pos``, so this must never touch the batch it is
    measuring. ``std_orientation=False`` matches the GFN's own analyze path
    (energies/molecular_crystal.py:205), which is the convention whose displacement we
    want to report.
    """
    probe = batch.clone()
    probe.pose_aunit(std_orientation=False)
    probe.build_unit_cell()
    return probe.unit_cell_pos, probe.unit_cell_batch


def recover_transforms(pre_pos, post_pos, batch_index, n_graphs):
    """Per-graph orthogonal transform taking ``pre_pos`` to ``post_pos``, by Procrustes.

    Recovered from the coordinates themselves rather than read out of the library, so it is
    true by construction whatever ``orient_molecule`` did internally, and its determinant
    is the honest test of whether that molecule was rotated or mirrored.
    """
    out = torch.zeros(n_graphs, 3, 3, dtype=pre_pos.dtype)
    means_pre = torch.zeros(n_graphs, 3, dtype=pre_pos.dtype).index_add_(
        0, batch_index, pre_pos) / torch.bincount(batch_index, minlength=n_graphs)[:, None]
    means_post = torch.zeros(n_graphs, 3, dtype=post_pos.dtype).index_add_(
        0, batch_index, post_pos) / torch.bincount(batch_index, minlength=n_graphs)[:, None]
    a = pre_pos - means_pre[batch_index]
    b = post_pos - means_post[batch_index]
    out.index_add_(0, batch_index, torch.einsum('ni,nj->nij', b, a))
    u, _, vt = torch.linalg.svd(out.double())
    return (u @ vt).to(pre_pos.dtype)


def correct_proper_orientations(batch, transforms, rtol):
    """Re-express ``aunit_orientation`` as ``R_aunit @ R^-1`` wherever R is a proper
    rotation, so those structures still index the crystal they came from.

    Improper (mirrored) molecules are skipped and marked invalid: a rotation vector cannot
    encode a reflection, so there is no orientation that restores their crystal. Doing this
    for the proper majority is the difference between a file with no usable stored crystals
    and one where most are exact.

    scipy is used for the matrix->rotvec step rather than
    ``geometry_utils.rotmat2rotvec``, which does not reliably invert ``rotvec2rotmat``: over
    the full [0, 2pi) range it fails to round-trip on a fraction of rotations, and stored
    aunit_orientation here reaches norm 6.13, well outside the canonical [0, pi] range it
    returns. The forward direction is sound, so it is used again to verify what is written.
    """
    proper = torch.linalg.det(transforms) > 0
    if not bool(proper.any()):
        return proper, 0.0

    idx = proper.nonzero().flatten()
    m = (rotvec2rotmat(batch.aunit_orientation[idx, :3].cpu())
         @ torch.linalg.inv(transforms[idx].cpu()))
    new = torch.as_tensor(Rotation.from_matrix(m.double().numpy()).as_rotvec(),
                          dtype=batch.aunit_orientation.dtype)
    residual = (rotvec2rotmat(new) - m).abs().max().item()
    if residual > rtol:
        raise RuntimeError(
            f"rotation vector round-trip residual {residual:.3e} exceeds {rtol:.1e}: the "
            f"corrected aunit_orientation does not reproduce its own rotation matrix")

    orientation = batch.aunit_orientation.clone()
    orientation[idx, :3] = new.to(orientation.device)
    batch.aunit_orientation = orientation
    return proper, residual


def per_graph_shift(batch, reference_pos):
    """Max atom displacement per graph, against a reference copy of ``pos``."""
    delta = (batch.pos - reference_pos).abs().amax(dim=1)
    out = torch.zeros(batch.num_graphs, device=delta.device)
    return out.scatter_reduce(0, batch.batch, delta, reduce='amax')


def _pin_once(batch, tol):
    """One standardization pass, in place. Returns per-graph (proper, unstable) plus the
    displacement statistics used for reporting."""
    before_cell, _ = posed_unit_cell(batch)
    pre_pos = batch.pos.clone()

    batch.orient_molecule(mode='std')          # EXACTLY train.py:2692 / buffer.py:203

    transforms = recover_transforms(pre_pos, batch.pos, batch.batch, batch.num_graphs)
    proper, residual = correct_proper_orientations(batch, transforms, tol)
    batch.add_graph_attr(proper.to(batch.pos.dtype), 'crystal_valid')

    settled = batch.pos.clone()
    probe = batch.clone()
    probe.orient_molecule(mode='std')          # second application: must move nothing
    unstable = per_graph_shift(probe, settled) > tol

    after_cell, cell_index = posed_unit_cell(batch)
    per_graph = torch.zeros(batch.num_graphs, dtype=after_cell.dtype).scatter_reduce(
        0, cell_index, (after_cell - before_cell).abs().amax(dim=1), reduce='amax')
    kept_dev = per_graph[proper].max().item() if bool(proper.any()) else 0.0
    lost_dev = per_graph[~proper].max().item() if bool((~proper).any()) else 0.0
    return proper, unstable, residual, kept_dev, lost_dev


def pin_to_trainer_frame(batch, tol, jitter, max_attempts, seed):
    """Make the batch a FIXED POINT of ``orient_molecule(mode='std')``, jittering the
    stragglers until they hold still.

    Returns (batch, dropped_identifiers, jittered_identifiers).

    A molecule whose inertia tensor is near-degenerate (a symmetric top) has no unique
    principal-axis assignment, so the standardization tie-breaks on numerical noise and can
    land in a different frame every time it is applied -- the exact mismatch this script
    exists to prevent. Perturbing its atoms by a fraction of a picometre separates the
    degenerate moments, and because the perturbation is BAKED IN here rather than applied
    per call, a random tie-break becomes a permanent one. The retry restarts from the
    pristine geometry each time so the jitter never compounds.

    1e-3 A is ~0.07% of a C-C bond and far below the resolution of anything downstream --
    the encoder, the LJ potential, or the cell parameters -- but it is a real change to the
    stored molecule, so it is reported and recorded in the file.
    """
    pristine = batch.clone()
    generator = torch.Generator().manual_seed(seed)
    jittered = set()

    for attempt in range(max_attempts + 1):
        work = pristine.clone()
        proper, unstable, residual, kept_dev, lost_dev = _pin_once(work, tol)
        if not bool(unstable.any()):
            break

        bad_ids = {work.identifier[i] for i in unstable.nonzero().flatten().tolist()}
        if attempt == max_attempts:
            break

        # degeneracy is a property of the MOLECULE, so jitter every replica of it
        per_graph_mask = torch.tensor([i in bad_ids for i in pristine.identifier])
        atom_mask = per_graph_mask[pristine.batch]
        noise = torch.randn(pristine.pos.shape, generator=generator, dtype=pristine.pos.dtype)
        pristine.pos = pristine.pos + atom_mask[:, None] * noise * jitter
        jittered |= bad_ids
        print(f"  attempt {attempt + 1}: {len(bad_ids)} frame-unstable molecule(s); "
              f"jittering their atoms by {jitter:g} A to break the degeneracy")

    batch = work
    n_valid, n_improper = int(proper.sum()), int((~proper).sum())
    print(f"  {n_valid}/{batch.num_graphs} rotated (proper): aunit_orientation corrected, "
          f"crystals preserved to {kept_dev:.2e} A  [rotvec residual {residual:.1e}]")
    print(f"  {n_improper}/{batch.num_graphs} MIRRORED (improper): crystals moved up to "
          f"{lost_dev:.3f} A, unrecoverable -- marked crystal_valid=0")
    if kept_dev > 1e-3:
        raise RuntimeError(
            f"structures marked crystal_valid=1 moved by {kept_dev:.3e} A; the orientation "
            f"correction did not hold, so the 'valid' marking would be a lie")
    if jittered:
        print(f"  jittered {len(jittered)} molecule(s) to a stable frame: "
              f"{sorted(jittered)[:4]}{' ...' if len(jittered) > 4 else ''}")

    dropped = set()
    if bool(unstable.any()):
        # still oscillating after max_attempts jitters -- something other than a numerical
        # tie is going on, so drop rather than keep perturbing
        dropped = {batch.identifier[i] for i in unstable.nonzero().flatten().tolist()}
        keep = [s for s in batch.batch_to_list() if s.identifier not in dropped]
        if not keep:
            raise RuntimeError("every molecule was frame-unstable; nothing left to write")
        print(f"  DROPPED {len(dropped)} molecule(s) still unstable after {max_attempts} "
              f"jitters: {sorted(dropped)[:4]}{' ...' if len(dropped) > 4 else ''}")
        batch = collate_data_list(keep)
        batch.box_analysis()
    else:
        print("  no frame-unstable molecules remain")

    # the guarantee, verified on what is actually being written
    settled = batch.pos.clone()
    probe = batch.clone()
    probe.orient_molecule(mode='std')
    drift = (probe.pos - settled).abs().max().item()
    if drift > tol:
        raise RuntimeError(
            f"file is NOT a fixed point of orient_molecule(mode='std'): a second "
            f"application still moves atoms by {drift:.3e} A (tol {tol:.1e}). The "
            f"embedding frame would not match the build frame. Refusing to write.")
    print(f"  fixed point confirmed: trainer's std-orient moves atoms by {drift:.2e} A")
    return batch, dropped, jittered


@torch.no_grad()
def embed(batch, encoder, device, batch_size):
    """Frozen Mo3ENet encoding of every molecule, in the pinned frame.

    Returns ``[n_graphs, 3, bottleneck]``: the EQUIVARIANT latent, kept in that form rather
    than scalarized so a later augmentation can rotate it alongside the molecule. It is
    well defined as a flat condition vector precisely because the frame is pinned.

    ``Mo3ENet.encode`` divides ``pos`` in place by its radial normalization and asserts the
    batch is centered, so it only ever sees a clone, and that clone is re-centered on ALL
    atoms first -- the autoencoder was trained on global centroids while ``orient_molecule``
    centers on heavy atoms (see utils.embed_dataset:664).
    """
    # chunked off a plain list rather than by slicing the batch: indexing a re-collated
    # batch trips PyG's `separate` on the scalar attrs collate_data_list attaches
    items = batch.batch_to_list()
    out = []
    for start in tqdm(range(0, len(items), batch_size), desc='  embedding'):
        chunk = collate_data_list(items[start:start + batch_size]).to(device)
        chunk.recenter_molecules(center_on_heavy_atoms=False)
        out.append(encoder.encode(chunk).clone().cpu())
    return torch.cat(out, dim=0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, type=Path,
                   help='a QM9 SG2 crystal dataset (list of MolCrystalData)')
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--tag', required=True,
                   help='output files are {tag}_conditions.pt / {tag}_prior.pt')
    p.add_argument('--encoder', type=Path, default=Path(DEFAULT_ENCODER))
    p.add_argument('--n-molecules', type=int, default=None,
                   help='keep this many molecules, replica-richest first (the ladder rung)')
    p.add_argument('--molecules', nargs='*', default=None,
                   help='keep these exact identifiers (SMILES) instead of --n-molecules')
    p.add_argument('--min-replicas', type=int, default=1,
                   help='drop molecules with fewer structures than this')
    p.add_argument('--lj-max', type=float, default=0.0,
                   help='drop structures with baked lj_pot above this')
    p.add_argument('--valid-only', action='store_true',
                   help='after pinning, keep only molecules whose crystals survived the '
                        'frame change (crystal_valid == 1). Mirroring is a per-MOLECULE '
                        'property -- every replica of a molecule shares its fate -- so this '
                        'never splits a molecule. Use it for rungs that need '
                        'bwd_sampling_mode: dataset phase 1 on true crystals.')
    p.add_argument('--energy-tol', type=float, default=0.05,
                   help='max tolerated median relative error between the rebuilt crystals '
                        'and the source file\'s baked lj_pot -- the ABSOLUTE fidelity check')
    p.add_argument('--jitter', type=float, default=1e-3,
                   help='atom perturbation (A) applied to frame-unstable molecules to break '
                        'a near-degenerate principal-axis tie. Baked in, so the tie-break is '
                        'permanent. 0 disables (unstable molecules are then dropped)')
    p.add_argument('--jitter-attempts', type=int, default=3,
                   help='how many times to jitter-and-retry before dropping a molecule')
    p.add_argument('--tol', type=float, default=1e-4,
                   help='max atom movement (A) tolerated under a repeat standardization')
    p.add_argument('--device', default='cuda')
    p.add_argument('--embed-batch-size', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    print(f"loading {args.source}")
    samples = torch.load(args.source, map_location='cpu', weights_only=False)
    print(f"  {len(samples)} structures")

    samples = normalize_schema(samples)

    samples, n_dropped = drop_pathological(samples, args.lj_max)
    if n_dropped:
        print(f"  dropped {n_dropped} structures with lj_pot > {args.lj_max}")

    # with --valid-only the molecule count has to be applied AFTER the frame change:
    # whether a molecule survives is only known once it has been standardized, so cutting
    # to N first could select N molecules and then discard all of them
    defer_count = args.valid_only and args.n_molecules is not None
    samples, kept_ids = select_molecules(samples, None if defer_count else args.n_molecules,
                                         args.molecules, args.min_replicas, args.seed)
    counts = collections.Counter(s.identifier for s in samples)
    print(f"  keeping {len(kept_ids)} molecules / {len(samples)} structures "
          f"(replicas per molecule: min {min(counts.values())}, max {max(counts.values())})")

    types = set()
    for s in samples:
        types.update(s.z.unique().tolist())
    unsupported = types - MO3ENET_ATOM_TYPES
    if unsupported:
        raise ValueError(
            f"atom types {sorted(unsupported)} are outside Mo3ENet's vocabulary "
            f"{sorted(MO3ENET_ATOM_TYPES)}. This encoder cannot embed this molecule set.")
    print(f"  atom types {sorted(types)} -- within the encoder's vocabulary")

    batch = collate_data_list(samples)
    batch.box_analysis()   # QM9 sets store cell_lengths/angles but leave T_fc unbuilt
    # read off the samples, not the batch: lj_pot does not survive collation
    baked_lj = torch.stack([s.lj_pot.flatten()[0] for s in samples])

    print("reconstructing the crystal parameterization from real coordinates")
    reconstruct_parameterization(batch)
    verify_against_baked_energy(batch, baked_lj, args.energy_tol)

    print("pinning molecules to the trainer's standardization frame")
    batch, dropped, jittered = pin_to_trainer_frame(batch, args.tol, args.jitter,
                                                    args.jitter_attempts, args.seed)

    if args.valid_only:
        items = batch.batch_to_list()
        keep = [s for s in items if float(s.crystal_valid) > 0]
        if not keep:
            raise ValueError(
                "--valid-only left nothing: every molecule in this selection was mirrored. "
                "Widen --n-molecules, or drop --valid-only and build the prior by sampling.")
        lost = sorted({s.identifier for s in items} - {s.identifier for s in keep})
        print(f"  --valid-only: {len(set(s.identifier for s in keep))} molecules survived "
              f"the frame change, {len(lost)} were mirrored and dropped")

        if defer_count:
            keep, chosen = select_molecules(keep, args.n_molecules, None,
                                            args.min_replicas, args.seed)
            print(f"  cut to the {len(chosen)} replica-richest survivors")

        batch = collate_data_list(keep)
        batch.box_analysis()

    n_valid = int(batch.crystal_valid.sum())

    print(f"loading frozen encoder {args.encoder}")
    # loaded on CPU then moved: load_molecule_autoencoder maps the checkpoint to `device`
    # and then calls np.array() on the atom-type table, which throws on a CUDA tensor
    encoder = load_molecule_autoencoder(str(args.encoder), 'cpu').to(args.device)
    embeddings = embed(batch, encoder, args.device, args.embed_batch_size)
    print(f"  embeddings {tuple(embeddings.shape)} "
          f"(equivariant [n, 3, bottleneck]; flat condition dim = {3 * embeddings.shape[-1]})")

    batch.embedding = embeddings

    args.out_dir.mkdir(parents=True, exist_ok=True)
    conditions_path = args.out_dir / f'{args.tag}_conditions.pt'
    prior_path = args.out_dir / f'{args.tag}_prior.pt'

    # molecules_path unwraps 'prior' (train.py:_load_condition_file); prior_path reads
    # 'equalized_prior' (train.py:init_prior_dataset). One structure set serves as both,
    # matching how mk_dev points both keys at the same molecule file.
    # In-band provenance. crystal_valid rides on the batch per structure; these totals are
    # the summary. Structures with crystal_valid == 0 were MIRRORED, so their cell
    # parameters no longer describe the crystal they came from.
    meta = {'n_crystals_valid': n_valid,
            'n_crystals_total': int(batch.num_graphs),
            'frame': 'orient_molecule(mode=std) fixed point',
            'embedding_dim': int(3 * embeddings.shape[-1]),
            'n_molecules': len(set(batch.identifier)),
            'dropped_unstable': sorted(dropped),
            'jittered_unstable': sorted(jittered),
            'jitter_angstrom': args.jitter if jittered else 0.0}
    torch.save({'prior': batch, **meta}, conditions_path)
    torch.save({'prior': batch, 'equalized_prior': batch, **meta}, prior_path)
    print(f"wrote {conditions_path}")
    print(f"wrote {prior_path}")
    print()
    print(f"REMINDER: {n_valid}/{batch.num_graphs} structures carry crystal_valid=1 and "
          f"index their true crystal; the rest were mirrored and do not. A stage with "
          f"bwd_sampling_mode: dataset trains on stored cell parameters, so it must be "
          f"restricted to the crystal_valid subset -- or the prior built by sampling.")
    print("NOT set: thermal_scaling_factor. The mipcas prior carries one (0.3636) and "
          "train.py:init_prior_dataset lets it REPLACE lj_coeff for the whole run, so "
          "without it this set trains at the config's lj_coeff and the effective kT is "
          "energy_config.temperature as written. Calibrate with "
          "data_processing/utils.calibrate_energy_function_vs_uma before comparing "
          "energies against a mipcas-style run.")


if __name__ == '__main__':
    main()
