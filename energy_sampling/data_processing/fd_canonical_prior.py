"""Rewrite a space-group-2, Z' = 1 prior file as canonical forms, one row per crystal (crystal_fd; owner D6, D7).

From ``energy_sampling/``, CPU only:

    python -m data_processing.fd_canonical_prior SOURCE.pt [--tol 1e-6] [--check 2000] [--energy-check 1000]
        [--probe 256] [--config configs/mk_dev.yaml] [--companion OTHER_SPLIT_SOURCE.pt] [--replace]

writes beside SOURCE:

    <stem>_fd1.pt               the source dict with every batch rewritten (below) and every other entry kept
    <stem>_fd1_provenance.pt    per batch key and source row: output row or code, group, moved-image flag, the
                                canonical pose, and the latents behind every D7 refusal
    <stem>_fd1_manifest.json    counts, tolerance, crystal_fd table and code digests, and the checks (``run_checks``)

All three are first written as ``<name>.partial``. They are renamed into place only after the GFN loader checks
pass on the data file as written. The checks run in a child process (``checks_in_child``) when SOURCE is larger
than ``CHILD_CHECKS_ABOVE`` bytes, so that their loads do not sit on top of the rewrite; otherwise they run in
this process. An existing output is never overwritten, except with ``--replace``, which accepts only an
existing set that this script wrote from the same SOURCE: the old files are moved, not deleted, to
``_superseded_fd1/`` beside them, and are named in the manifest (``replaced``).

NOT A DROP-IN PRIOR FOR TODAY'S TRAINER. These rows are training data for a fundamental-domain model: one
description per crystal, ``t`` in the sg 2 fundamental domain [0, 1/2)^3. The trainer's latent box for sg 2 holds
four descriptions of each crystal (its y and z half shifts), and these rows fill one quarter of it, by design
(``CENTROID_DOMAIN_NOTE``, which the file carries under ``fd_canonical`` and the manifest repeats). Every row
still reads correctly through the trainer's loaders. But a model that fits a density in the trainer's box to
these rows as they stand puts all its mass in that quarter. A consumer in that box must first spread each row
over its four images (``domains.images(desc, within='latent_v1_box')``).

What a source row becomes
-------------------------
1. Import. A row in space group 2 with Z' = 1, finite parameters, a real cell and stored handedness +1 becomes a
   ``CrystalDescription``: float64 values of the stored float32 cell, centroid and rotation vector. Any other row
   is refused, with the code for its reason (``CODES``).
2. Canonicalise. ``ops.reduce_cell``, then ``domains.fold``, then ``domains.to_sheet(+1)``; the sheet step is a
   no-op, because both the det +1 basis change and Gamma+ keep h. reduce_cell's triclinic step is
   ``triclinic_basis`` on the cell, then ``change_basis(desc, N, 0)``. Here it is computed once per distinct stored
   cell and applied to each row by ``change_basis``. ``shortcut_check`` in the manifest compares this, bit for
   bit, against ``ops.reduce_cell`` on a seeded subsample.
3. One row per crystal. Two rows are one crystal when their canonical forms agree to within ``tol`` (Chebyshev)
   in the 18-D chart vector ``canon_vector``: CellChart y, PolytopeARChart u with its period-1 seams wrapped, and
   the nine rotation-matrix entries. Rows of different molecules (``identifier``) are never one crystal.
   - Image-written file (top-level ``source_row`` and ``image_id`` aligned with the rows of its only batch). The
     source grouping decides: the images of one source are one crystal, and the image with the lowest image_id is
     its representative.
   - A source whose images do not collapse to within ``tol``. Its writer moved an image at the box edge:
     ``pool_anchors.Chart.describe`` clamps a centre coordinate to [0, 0.9999], after snapping one in
     (1 - 1e-5, 1) to 0, and ``Chart.images`` writes images 4-7 of an x-face centre exactly onto the opposite face.
     The output takes the canonical form of the images that carry none of these signatures (``resolve_group``),
     provided those images agree with each other and only the centroid block of the moved images differs. Every
     such case is recorded. A source this does not resolve is refused (``images_do_not_collapse``).
   - A source whose images all carry an x the writer clamps to (x = float32(0.4999), or float32(0.9999) - 1/2)
     collapses: the source itself was written there, so no unmoved image exists and the output row is the stored
     crystal. These are counted (``stored_x_at_writer_edge``), not refused. Rows stored exactly on the x = 0
     wall are counted there too (``x_eq_0``): a general-position minimum has no reason to sit on it, so these are
     probably states a search or writer clamped at the wall (inference); they are carried as stored.
   - All sources (and every row of a file without a source grouping) are then deduplicated by greedy leaders in
     row order: a row within ``tol`` of an earlier kept row is mapped onto it.
4. D7 refusals, judged on the canonical form as it will be stored (float32), for a whole group at once:
   - the trainer's latent (``charts.LatentV1Chart.latents``: the live ``latent_transform`` without the clip)
     outside the live clamp of ``MolCrystalOps.latent_to_cell_params`` (``LatentV1Chart.clamp_bounds``, read off
     the live code). ``latent_params`` clips to [-1, 1] before that clamp, and the clamp is inside [-1, 1], so
     this is exactly the set of rows the trainer would build as a different crystal;
   - a canonical centroid coordinate in the builder clip band (1 - 1e-4, 1).
5. Write. The output row is its representative source row, taken by ``subsample_new_batch``, so every per-row
   field is carried, the stored energy included. Its pose fields are then replaced:
   - ``cell_lengths``, ``cell_angles``, ``aunit_centroid``;
   - ``aunit_orientation``: crystal_fd's +z-hemisphere rotation vector, the trainer's convention;
   - ``aunit_handedness``: +1;
   - then ``box_analysis``, which recomputes ``T_fc``, ``T_cf``, ``cell_volume``, ``packing_coeff`` and
     ``density``.
   ``pos`` is not touched. The trainer builds a row from its stored molecule (``pose_aunit`` with
   std_orientation=False, after ``CrystalBuffer``'s ``orient_molecule('std')``), and the description places that
   same molecule.

   Top-level tensors aligned with the rows of a single batch (``source_row``, ``image_id`` and ``is_anchor`` in a
   ``pool_assemble`` file) follow the representative rows, with two exceptions:
   - ``source_row`` becomes the representative's row index in SOURCE; its old value is kept in the provenance;
   - ``image_id`` becomes 0, since every output row is its own image 0; the old value is also in the provenance.
   A top-level ``n_*`` integer equal to a row or molecule count of this file becomes the output's count
   (``update_counts``). With ``--companion``, the same applies to the counts of the other file of a split (the
   qm9full train and held-out files each carry both), taken from the companion's own ``_fd1`` manifest, which must
   already exist. Other entries are kept as they are. The entry ``fd_canonical`` is added. It holds the build
   note, the centroid-domain note, the stored-energy note and the count record.

The manifest also records, per batch, the pairs of output rows close to the one-crystal tolerance
(``near_pair_census``), the source rows whose stored centroid lies in the builder clip band, and the checks.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mxtaltools.crystal_fd import charts, checks, domains, ops, tables
from mxtaltools.crystal_fd.description import CrystalDescription, MoleculeRef

F64 = torch.float64
SG = 2
TOL = 1e-6                 #: one crystal: canonical chart vectors within this (Chebyshev, seams wrapped)
RED_TOL = 1e-10            #: ops.reduce_cell's default search tolerance
CLIP_HI = 1.0 - 1e-4       #: the live builder clips centroids to [0, 1 - 1e-4]
EDGE32 = float(np.float32(0.9999))  #: pool_anchors.EDGE, as a float32 centroid stores it
X_EDGE_A = float(np.float32(0.4999))                    #: an x stored at float32(0.4999)
X_EDGE_B = float(np.float32(float(np.float32(0.9999)) - 0.5))  #: pool_anchors.into_box of an x clamped to EDGE
SUFFIX = '_fd1'
VERSION = 'fd_canonical_prior v2 (2026-10-17)'
SUPERSEDED_DIR = '_superseded_fd1'  #: where --replace moves this script's earlier output (beside the source)
CHILD_CHECKS_ABOVE = 2 ** 30  #: source files larger than this (bytes) are checked in a child process
CENSUS_RADII = (1e-5, 1e-4, 1e-3)  #: near-pair census of the output rows (chart units), above the one-crystal tol
K_BOX = 4                  #: descriptions of one sg 2 crystal in the trainer's latent box (y, z half shifts)
CENTROID_DOMAIN_NOTE = (
    'NOT A DROP-IN PRIOR FOR THE CURRENT TRAINER: these rows are training data for a fundamental-domain model, one '
    'description per crystal, by design. Every row reads correctly through the trainer\'s loaders, but as a '
    'training distribution in the trainer\'s latent box the file is not its source\'s. '
    't in [0, 1/2)^3: the crystal_fd fundamental domain of sg 2 on the +1 sheet (domains.fold). The trainer\'s '
    'latent box for sg 2 (ASYM_UNITS[2] = [0.5, 1, 1], y and z wrapped) holds k_box = 4 descriptions of each '
    'crystal, its y and z half shifts, and every row here is the one with y < 1/2 and z < 1/2. A density fitted '
    'in the trainer\'s latent box to these rows as they stand puts all its mass in that quarter. A consumer in '
    'that box must spread each row over its 4 images (mxtaltools.crystal_fd.domains.images(desc, '
    'within="latent_v1_box")); one on the domain chart uses the rows as they are, with log Z_trainer = log Z_FD '
    '+ log 4 (docs/design/fd_tooling.md, section 3.11). The source carried every image (an image-written file) '
    'or one image per crystal, spread over the four quarters.')
STORED_ENERGY_NOTE = (
    'Stored energies are the source rows\', carried bit for bit and not recomputed: a stored energy that did not '
    'match its source row\'s crystal still does not. Modeller.init_prior_dataset rescores every prior row at load; '
    'an offline consumer of the stored energy should rescore too. Where a source row\'s stored centroid lies in the '
    'builder clip band (1 - 1e-4, 1), and for the pool writer\'s moved-image sources, the stored energy is that of '
    'the crystal the builder scored with the centroid clipped to 1 - 1e-4, while the output row describes the '
    'unclipped crystal. The manifest checks report the differences (energy.*_outliers, source_clip_band, '
    'moved_images).')
ANALYZE_KW = dict(cutoff=10, supercell_size=10, std_orientation=False)  #: MolecularCrystal.analyze_crystal_batch
LATENT_NAMES = ('a', 'b', 'c', 'alpha', 'beta', 'gamma', 't_x', 't_y', 't_z', 'theta', 'phi', 'r')
POSE_FIELDS = ('cell_lengths', 'cell_angles', 'aunit_centroid', 'aunit_orientation', 'aunit_handedness',
               'T_fc', 'T_cf', 'cell_volume', 'packing_coeff', 'density')
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'mk_dev.yaml'

#: per source row, in the provenance. Codes below 10 map the row to an output row; 10 and above refuse it.
KEPT, SAME_CRYSTAL, MOVED_IMAGE = 0, 1, 2
CODES = {
    KEPT: 'kept',                              # the representative of its output row
    SAME_CRYSTAL: 'same_crystal',              # an image or duplicate of its output row's representative
    MOVED_IMAGE: 'moved_image',                # an image its writer moved at the box edge; mapped to its source
    10: 'not_sg2',
    11: 'not_zp1',
    12: 'nonfinite',
    13: 'no_real_cell',
    14: 'handedness_not_plus1',                # the trainer would build it at +1: a different crystal
    15: 'reduction_failed',                    # reduce_cell found no basis within RED_TOL
    16: 'images_do_not_collapse',              # image group not explained by the writer's edge rules
    20: 'latent_clamp_cell_length',            # D7: a canonical cell length outside the trainer's latent box
    21: 'latent_clamp_theta',                  # D7: hemisphere theta under the latent floor
    22: 'latent_clamp_other',                  # D7: any other latent entry outside the live clamp
    23: 'centroid_in_clip_band',               # D7: a canonical centroid coordinate in (1 - 1e-4, 1)
}
CODE = {v: k for k, v in CODES.items()}


def _mem():
    try:
        import psutil
        m = psutil.Process().memory_info()
        return f'[rss {m.rss / 2 ** 30:.1f} GB, private {getattr(m, "private", m.vms) / 2 ** 30:.1f} GB]'
    except Exception:
        return ''


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, _mem(), flush=True)


def _q(x, ps=(0.5, 0.99)):
    x = torch.as_tensor(x, dtype=F64).flatten()
    x = x[torch.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    out = {f'p{p * 100:g}': float(torch.quantile(x, p)) for p in ps}
    out.update(max=float(x.max()), n=int(len(x)))
    return out


# ---------------------------------------------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------------------------------------------

_CHARTS = {}


def _charts():
    if not _CHARTS:
        box = (torch.ones(3, dtype=F64), torch.full((3,), 1e3, dtype=F64))
        _CHARTS['cell'] = charts.CellChart('triclinic', box)
        _CHARTS['t'] = charts.PolytopeARChart(SG)
        per = torch.zeros(18, dtype=torch.bool)
        for i, d in enumerate(_CHARTS['t'].dims):
            if d.kind == 'circle':
                if d.period != 1.0:
                    raise RuntimeError(f'centroid chart dim {d.name} has period {d.period}, expected 1')
                per[6 + i] = True
        _CHARTS['per'] = per
    return _CHARTS


def canon_vector(desc):
    """[n, 18] float64: CellChart y (6), PolytopeARChart u reduced to [0, 1) (3), rotation-matrix entries (9)."""
    c = _charts()
    y, _ = c['cell'].unmap(desc.cell)
    u, _ = c['t'].unmap(desc.t[:, 0])
    u = u - torch.floor(u)
    return torch.cat([y, u, desc.R[:, 0].reshape(-1, 9)], 1)


def chart_dev(a, b):
    """[n] Chebyshev distance between chart vectors, the period-1 columns wrapped."""
    per = _charts()['per']
    d = (a - b).abs()
    d[:, per] = torch.minimum(d[:, per], 1 - d[:, per])
    return d.amax(1)


# ---------------------------------------------------------------------------------------------------------------
# import and canonicalisation
# ---------------------------------------------------------------------------------------------------------------

def stored_pose(batch):
    """The stored pose fields as float32 tensors [n, 3] (handedness [n, k])."""
    n = int(batch.num_graphs)
    p = dict(L=batch.cell_lengths.reshape(n, -1), A=batch.cell_angles.reshape(n, -1),
             c=batch.aunit_centroid.reshape(n, -1), rv=batch.aunit_orientation.reshape(n, -1),
             h=batch.aunit_handedness.reshape(n, -1))
    if p['c'].shape[1] != 3 or p['rv'].shape[1] != 3:
        raise SystemExit(f"the batch stores {p['c'].shape[1] // 3} asymmetric units per row: not a Z' = 1 layout")
    return p


def import_codes(batch):
    """[n] int8 refusal code per row (0: importable), and the stored pose (``stored_pose``)."""
    n = int(batch.num_graphs)
    p = stored_pose(batch)
    sg = torch.as_tensor(batch.sg_ind).reshape(-1).long()
    zp = getattr(batch, 'z_prime', None)
    zp = torch.ones(n, dtype=torch.long) if zp is None else torch.as_tensor(zp).reshape(-1).long()
    code = torch.zeros(n, dtype=torch.int8)

    def mark(mask, name):
        code[(code == 0) & mask] = CODE[name]

    mark(sg != SG, 'not_sg2')
    mark(zp != 1, 'not_zp1')
    fin = torch.isfinite(p['L']).all(1) & torch.isfinite(p['A']).all(1) & torch.isfinite(p['c']).all(1) & \
        torch.isfinite(p['rv']).all(1) & torch.isfinite(p['h'][:, :1]).all(1)
    mark(~fin, 'nonfinite')
    real = torch.zeros(n, dtype=torch.bool)
    real[fin] = ops.real_cells(torch.cat([p['L'], p['A']], 1)[fin].to(F64))
    mark(~real, 'no_real_cell')
    mark(p['h'][:, 0] != 1, 'handedness_not_plus1')
    return code, p


def reduce_distinct_cells(cell, chunk=20000):
    """``ops.reduce_cell`` once per distinct float64 cell row. Per row: N [n, 3, 3], n_tied, violation, n_auto,
    failed [n] bool (no basis within RED_TOL); and the number of distinct cells."""
    keys = np.ascontiguousarray(cell.numpy().astype(np.float64)).view(np.dtype((np.void, 48))).ravel()
    uk, inv = np.unique(keys, return_inverse=True)
    cu = torch.from_numpy(np.frombuffer(uk.tobytes(), dtype=np.float64).reshape(-1, 6).copy())
    m = cu.shape[0]
    N = torch.zeros(m, 3, 3, dtype=torch.long)
    tied = torch.zeros(m, dtype=torch.long)
    viol = torch.zeros(m, dtype=F64)
    auto = torch.zeros(m, dtype=torch.long)
    failed = torch.zeros(m, dtype=torch.bool)
    eye = torch.eye(3, dtype=F64)
    t0 = time.time()
    for lo in range(0, m, chunk):
        r = torch.arange(lo, min(lo + chunk, m))
        d = CrystalDescription.from_parts(SG, cu[r], torch.zeros(len(r), 3, dtype=F64), eye.expand(len(r), 3, 3))
        try:
            red = ops.reduce_cell(d, tol=RED_TOL)
            ok = torch.ones(len(r), dtype=torch.bool)
        except RuntimeError:
            _, _, _, v = ops.triclinic_basis(cu[r], tol=RED_TOL)
            ok = v <= RED_TOL
            red = ops.reduce_cell(d[torch.nonzero(ok)[:, 0]], tol=RED_TOL)
        k = r[ok]
        N[k], tied[k], viol[k], auto[k] = red.N, red.n_tied, red.violation, red.n_auto
        failed[r[~ok]] = True
        if (lo // chunk) % 10 == 9:
            log(f'    reduce_cell {min(lo + chunk, m):,}/{m:,} distinct cells ({time.time() - t0:.0f} s)')
    inv = torch.from_numpy(np.asarray(inv).reshape(-1).astype(np.int64))
    return dict(N=N[inv], n_tied=tied[inv], violation=viol[inv], n_auto=auto[inv], failed=failed[inv], distinct=m)


def canonicalise(batch, chunk=50000, shortcut_rows=3000, seed=0):
    """Canonical forms of every row of a stored sg 2 batch.

    Returns a dict of per-row arrays; rows refused at import or by the reduction carry NaN geometry:
    - code [n] int8;
    - cell [n, 6], t [n, 3], R [n, 3, 3] (float64);
    - X [n, 18], the chart vectors;
    - N [n, 3, 3];
    - n_auto, n_tied, tie, element, basis_changed [n];
    - 'stats', a dict of counts."""
    n = int(batch.num_graphs)
    t_start = time.time()
    code, p = import_codes(batch)
    ok = code == 0
    idx = torch.nonzero(ok)[:, 0]
    out = dict(cell=torch.full((n, 6), float('nan'), dtype=F64), t=torch.full((n, 3), float('nan'), dtype=F64),
               R=torch.full((n, 3, 3), float('nan'), dtype=F64), X=torch.full((n, 18), float('nan'), dtype=F64),
               N=torch.zeros(n, 3, 3, dtype=torch.long), n_auto=torch.zeros(n, dtype=torch.long),
               n_tied=torch.zeros(n, dtype=torch.long), tie=torch.full((n,), -1, dtype=torch.int8),
               element=torch.full((n,), -1, dtype=torch.long), basis_changed=torch.zeros(n, dtype=torch.bool))
    red = reduce_distinct_cells(torch.cat([p['L'], p['A']], 1)[idx].to(F64))
    bad = red['failed']
    code[idx[bad]] = CODE['reduction_failed']
    rows = idx[~bad]
    N_rows = red['N'][~bad]
    out['N'][rows] = N_rows
    out['n_auto'][rows] = red['n_auto'][~bad]
    out['n_tied'][rows] = red['n_tied'][~bad]
    out['basis_changed'][rows] = ~(N_rows == torch.eye(3, dtype=torch.long)).all(-1).all(-1)
    log(f'  reduced {red["distinct"]:,} distinct cells of {len(idx):,} imported rows ({time.time() - t_start:.0f} s)')
    t0 = time.time()
    for lo in range(0, len(rows), chunk):
        r = rows[lo:lo + chunk]
        desc = CrystalDescription.from_crystal_params(SG, p['L'][r], p['A'][r], p['c'][r], p['rv'][r],
                                                      torch.round(p['h'][r, :1]).long())
        fr = domains.fold(ops.change_basis(desc, N_rows[lo:lo + chunk], torch.zeros(len(r), 3, dtype=F64)))
        cd = fr.desc
        if bool((cd.h[:, 0] != 1).any()):
            cd, sheet_ok = domains.to_sheet(cd, 1)
            if not bool(sheet_ok.all()):
                raise RuntimeError('to_sheet failed on an sg 2 row (sg 2 holds the inversion)')
        out['cell'][r], out['t'][r], out['R'][r] = cd.cell, cd.t[:, 0], cd.R[:, 0]
        out['X'][r] = canon_vector(cd)
        out['tie'][r], out['element'][r] = fr.tie, fr.element
        if (lo // chunk) % 4 == 3:
            log(f'    folded {min(lo + chunk, len(rows)):,}/{len(rows):,} rows ({time.time() - t0:.0f} s)')
    # the per-distinct-cell shortcut against ops.reduce_cell itself, on a seeded subsample
    g = torch.Generator().manual_seed(seed)
    sub = rows[torch.randperm(len(rows), generator=g)[:shortcut_rows]]
    dsub = CrystalDescription.from_crystal_params(SG, p['L'][sub], p['A'][sub], p['c'][sub], p['rv'][sub],
                                                  torch.round(p['h'][sub, :1]).long())
    rr = ops.reduce_cell(dsub)
    via = ops.change_basis(dsub, out['N'][sub], torch.zeros(len(sub), 3, dtype=F64))
    mx = (lambda a, b: float((a - b).abs().max()) if len(sub) else 0.0)
    out['code'] = code
    out['stats'] = dict(
        rows=n, imported=int(ok.sum()), distinct_stored_cells=int(red['distinct']),
        reduction_failed=int(bad.sum()), violation_max=float(red['violation'][~bad].max()) if len(rows) else 0.0,
        n_tied_gt1=int((out['n_tied'] > 1).sum()), n_auto_gt1=int((out['n_auto'] > 1).sum()),
        basis_changed=int(out['basis_changed'].sum()),
        tie_counts={str(k): int((out['tie'][rows] == k).sum()) for k in range(4)},
        shortcut_check=dict(rows=len(sub), N_equal=bool(torch.equal(rr.N, out['N'][sub])),
                            cell_max_abs=mx(rr.desc.cell, via.cell), t_max_abs=mx(rr.desc.t, via.t),
                            R_max_abs=mx(rr.desc.R, via.R),
                            n_auto_equal=bool(torch.equal(rr.n_auto, out['n_auto'][sub])),
                            n_tied_equal=bool(torch.equal(rr.n_tied, out['n_tied'][sub]))),
        seconds=time.time() - t_start)
    return out


def canonical_description(cn, rows):
    """CrystalDescription of the canonical forms of ``rows`` (float64, +1 sheet)."""
    return CrystalDescription.from_parts(SG, cn['cell'][rows], cn['t'][rows], cn['R'][rows])


def stored_form(desc):
    """The description as the output file stores it: ``crystal_params`` rounded to float32 (``build_output`` writes
    exactly these values), read back in float64. D7 is judged on this, so a row within float32 rounding of a clamp
    bound is judged on the value the trainer will read."""
    L, A, t, rv, h = desc.crystal_params()
    f = lambda v: v.float().double()
    h = torch.round(h).long() if h.is_floating_point() else h.long()
    return CrystalDescription.from_crystal_params(SG, f(L), f(A), f(t), f(rv), h)


def stored_description(batch, rows=None):
    """CrystalDescription of stored rows, float64 of the float32 values."""
    p = stored_pose(batch)
    r = slice(None) if rows is None else rows
    return CrystalDescription.from_crystal_params(SG, p['L'][r], p['A'][r], p['c'][r], p['rv'][r],
                                                  torch.round(p['h'][r, :1]).long())


# ---------------------------------------------------------------------------------------------------------------
# D7: what the trainer's latent chart would clamp
# ---------------------------------------------------------------------------------------------------------------

def latent_clamp(cn, rows, radius, chunk=100000):
    """For the canonical forms of ``rows`` as stored (float32): (bitmask [n] int32, bit j set when the live clamp
    moves latent entry j (LATENT_NAMES); clip band [n] bool; and, for the rows with a bit set, a dict: ``index``
    into ``rows``, their ``latents`` [k, 12], and the clamp bounds ``lo``, ``hi`` [12] they were judged against
    (radius-normalised, one pair for every row), so a refusal can be re-checked from the provenance alone)."""
    mask = torch.zeros(len(rows), dtype=torch.int32)
    band = torch.zeros(len(rows), dtype=torch.bool)
    weights = 2 ** torch.arange(12, dtype=torch.int32)
    flagged = dict(index=[], latents=[])
    bounds = None
    for lo in range(0, len(rows), chunk):
        r = rows[lo:lo + chunk]
        d = stored_form(canonical_description(cn, r))
        ch = charts.LatentV1Chart(SG, radius[r].float())
        lat = ch.latents(d)
        clo, chi = ch.clamp_bounds()
        moved = (lat < clo) | (lat > chi) | ~torch.isfinite(lat)
        mask[lo:lo + len(r)] = (moved.int() * weights).sum(1).int()
        t32 = d.t[:, 0].float()
        band[lo:lo + len(r)] = ((t32 > CLIP_HI) & (t32 < 1)).any(1)
        k = torch.nonzero(moved.any(1))[:, 0]
        flagged['index'].append(k + lo)
        flagged['latents'].append(lat[k].float())
        if clo.dim() != 1:
            raise RuntimeError('LatentV1Chart.clamp_bounds is no longer one bound pair for every row')
        bounds = (clo.float().clone(), chi.float().clone())
    flagged = dict(index=torch.cat(flagged['index']) if len(rows) else torch.zeros(0, dtype=torch.long),
                   latents=torch.cat(flagged['latents']) if len(rows) else torch.zeros(0, 12),
                   lo=None if bounds is None else bounds[0], hi=None if bounds is None else bounds[1])
    return mask, band, flagged


def d7_code(mask, band):
    """[n] int8 D7 refusal code (0: none), by priority: cell length, theta, any other entry, clip band."""
    code = torch.zeros(mask.shape[0], dtype=torch.int8)
    code[band] = CODE['centroid_in_clip_band']
    code[mask != 0] = CODE['latent_clamp_other']
    code[(mask & (1 << 9)) != 0] = CODE['latent_clamp_theta']
    code[(mask & 0b111) != 0] = CODE['latent_clamp_cell_length']
    return code


# ---------------------------------------------------------------------------------------------------------------
# one row per crystal
# ---------------------------------------------------------------------------------------------------------------

def molecule_ids(batch):
    """[n] long: one id per distinct identifier string (rows of different molecules are never one crystal)."""
    n = int(batch.num_graphs)
    ident = getattr(batch, 'identifier', None)
    if ident is None:
        return torch.zeros(n, dtype=torch.long)
    uniq = {}
    return torch.tensor([uniq.setdefault(s, len(uniq)) for s in list(ident)], dtype=torch.long)


def _embed(X, rows, mol):
    """Points and periodic box for a cKDTree over the chart vectors of ``rows``: the period-1 columns wrap, the
    others sit in a box too large to wrap, and a molecule coordinate 100 apart per molecule keeps rows of different
    molecules beyond any radius used here."""
    per = _charts()['per']
    x = X[rows].numpy().copy()
    box = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        if bool(per[j]):
            x[:, j] = np.mod(x[:, j], 1.0)
            x[:, j][x[:, j] >= 1.0] = 0.0
            box[j] = 1.0
        else:
            x[:, j] -= x[:, j].min()
            box[j] = 2 * x[:, j].max() + 10.0
    g = mol[rows].numpy().astype(np.float64)[:, None] * 100.0
    return np.concatenate([x, g], 1), np.concatenate([box, [2 * g.max() + 1000.0]])


def greedy_leaders(X, rows, mol, tol):
    """Greedy leaders in row order among ``rows`` (ascending) at Chebyshev ``tol``, periodic columns wrapped, and
    rows of different molecules never paired. Returns [len(rows)] long: the row each row maps onto (itself if
    kept)."""
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree
    m = len(rows)
    if m < 2:
        return rows.clone()
    x, box = _embed(X, rows, mol)
    pairs = cKDTree(x, boxsize=box).query_pairs(tol, p=np.inf, output_type='ndarray')
    state = np.arange(m)
    if len(pairs):
        adj = coo_matrix((np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
        adj = (adj + adj.T).tocsr()
        ip, ind = adj.indptr, adj.indices
        state = np.full(m, -1)
        for k in range(m):
            if state[k] >= 0:
                continue
            state[k] = k
            for j in ind[ip[k]:ip[k + 1]]:
                if state[j] < 0 and j > k:
                    state[j] = k
    return rows[torch.from_numpy(state.astype(np.int64))]


def near_pair_census(X, rows, mol, energy=None, radii=CENSUS_RADII, top=10):
    """Output rows of one molecule that the one-crystal tolerance keeps apart although their chart vectors are close.

    ``rows``: the source rows of the output rows, in output order. Counts the pairs within each radius (Chebyshev
    chart distance, seams wrapped) and the rows in such a pair, and lists the ``top`` closest pairs with their
    stored energies. Separately relaxed copies of one minimum land here: the tolerance identifies descriptions of
    one stored crystal, not basins."""
    from scipy.spatial import cKDTree
    m = len(rows)
    res = dict(radii=[float(r) for r in radii], rows=m)
    if m >= 2:
        x, box = _embed(X, rows, mol)
        pairs = cKDTree(x, boxsize=box).query_pairs(max(radii), p=np.inf, output_type='ndarray')
    else:
        pairs = np.zeros((0, 2), dtype=np.int64)
    d = chart_dev(X[rows[pairs[:, 0]]], X[rows[pairs[:, 1]]]).numpy() if len(pairs) else np.zeros(0)
    res['pairs_within'] = {f'{r:g}': int((d <= r).sum()) for r in radii}
    res['rows_in_a_pair_within'] = {f'{r:g}': int(len(np.unique(pairs[d <= r]))) for r in radii}
    res['closest_chart_dev'] = float(d.min()) if len(d) else None
    res['closest'] = [dict(out_rows=[int(pairs[k, 0]), int(pairs[k, 1])], chart_dev=float(d[k]),
                           energy=None if energy is None else [float(energy[rows[pairs[k, 0]]]),
                                                               float(energy[rows[pairs[k, 1]]])])
                      for k in np.argsort(d)[:top].tolist()]
    return res


def resolve_group(members, X, centroid32, image_id, tol):
    """The unmoved image(s) of a non-collapsing image group.

    ``members``: [k] row indices sorted by image_id. A writer move (pool_anchors.Chart) leaves one of these
    signatures:
    - the clamp: a y or z centroid stored exactly at float32(0.9999);
    - an x-face copy: image_id >= 4;
    - the snap (second level only): a y or z centroid stored exactly at 0.

    Returns None, or a dict:
    - rep: the representative row, the first image without a signature;
    - moved: the rows more than tol from rep;
    - level: the signature set used;
    - signatures: the signatures of each moved row;
    - max_dev: the largest chart-unit deviation of a moved row;
    - max_dev_cell_R: the largest deviation of any member outside the centroid block."""
    c = centroid32[members]
    clamp = (c[:, 1:] == EDGE32).any(1)
    face = image_id[members] >= 4
    snap = (c[:, 1:] == 0).any(1)
    for level, flagged in (('clamp+face_copy', clamp | face), ('clamp+face_copy+snap', clamp | face | snap)):
        free = members[~flagged]
        if len(free) == 0:
            continue
        rep = free[0]
        if float(chart_dev(X[free], X[rep].expand(len(free), -1)).max()) > tol:
            continue
        dev = chart_dev(X[members], X[rep].expand(len(members), -1))
        off = dev > tol
        if not bool(flagged[off].all()):
            continue
        d = (X[members] - X[rep]).abs()
        max_dev_cell_R = float(torch.cat([d[:, :6], d[:, 9:]], 1).max())
        if max_dev_cell_R > tol:
            continue
        sig = ['+'.join(nm for nm, f in (('clamp', clamp), ('face_copy', face), ('snap', snap)) if bool(f[j]))
               for j in torch.nonzero(off)[:, 0].tolist()]
        return dict(rep=int(rep), moved=members[off].tolist(), level=level, signatures=sig,
                    max_dev=float(dev[off].max()) if bool(off.any()) else 0.0, max_dev_cell_R=max_dev_cell_R)
    return None


def group_images(cn, source_row, image_id, centroid32, tol):
    """Representatives of an image-written batch.

    Returns:
    - rep [n] long: the representative row of each row's source, -1 if the source is refused;
    - code [n] int8;
    - moved [n] bool;
    - the resolved cases (a list);
    - the unresolved groups (a list);
    - stats (a dict)."""
    n = cn['X'].shape[0]
    rows = torch.nonzero(cn['code'] == 0)[:, 0]
    order = np.lexsort((image_id[rows].numpy(), source_row[rows].numpy()))
    rows = rows[torch.from_numpy(order)]
    src = source_row[rows]
    first = torch.ones(len(rows), dtype=torch.bool)
    first[1:] = src[1:] != src[:-1]
    gid = torch.cumsum(first.long(), 0) - 1
    ref = rows[first][gid]
    dev = chart_dev(cn['X'][rows], cn['X'][ref])
    ng = int(first.sum())
    gmax = torch.zeros(ng, dtype=F64).scatter_reduce(0, gid, dev, 'amax', include_self=True)
    gsize = torch.bincount(gid, minlength=ng)
    collapsed = gmax <= tol
    rep = torch.full((n,), -1, dtype=torch.long)
    code = cn['code'].clone()
    moved = torch.zeros(n, dtype=torch.bool)
    rep[rows] = torch.where(collapsed[gid], ref, torch.full_like(ref, -1))
    code[rows] = torch.where(rows == ref, torch.full_like(code[rows], KEPT), torch.full_like(code[rows], SAME_CRYSTAL))
    cases, unresolved = [], []
    starts = torch.nonzero(first)[:, 0].tolist() + [len(rows)]
    for g in torch.nonzero(~collapsed)[:, 0].tolist():
        members = rows[starts[g]:starts[g + 1]]
        res = resolve_group(members, cn['X'], centroid32, image_id, tol)
        case = dict(source_row_original=int(source_row[members[0]]), rows=members.tolist(),
                    image_ids=image_id[members].tolist(), group_max_dev=float(gmax[g]))
        if res is None:
            code[members] = CODE['images_do_not_collapse']
            case['centroids'] = [[float(v) for v in centroid32[r]] for r in members.tolist()]
            unresolved.append(case)
            continue
        mv = torch.as_tensor(res['moved'], dtype=torch.long)
        rep[members] = res['rep']
        code[members] = SAME_CRYSTAL
        code[res['rep']] = KEPT
        code[mv] = MOVED_IMAGE
        moved[mv] = True
        case.update(rep_row=res['rep'], rep_image_id=int(image_id[res['rep']]), moved_rows=res['moved'],
                    moved_image_ids=image_id[mv].tolist(), signatures=res['signatures'], rule=res['level'],
                    moved_max_dev=res['max_dev'], max_dev_cell_R=res['max_dev_cell_R'],
                    moved_centroids=[[float(v) for v in centroid32[r]] for r in res['moved']],
                    rep_centroid=[float(v) for v in centroid32[res['rep']]])
        cases.append(case)
    cd = dev[collapsed[gid]]
    stats = dict(sources=ng, images_per_source={str(k): v for k, v in zip(*[t.tolist() for t in torch.unique(
                     gsize, return_counts=True)])},
                 sources_collapsed=int(collapsed.sum()), sources_resolved=len(cases),
                 sources_unresolved=len(unresolved),
                 sig_counts={s: sum(1 for c in cases for x in c['signatures'] if x == s)
                             for s in sorted({x for c in cases for x in c['signatures']})},
                 collapsed_within_group_dev=_q(cd, ps=(0.5, 0.99, 0.999)))
    return rep, code, moved, cases, unresolved, stats


# ---------------------------------------------------------------------------------------------------------------
# one batch
# ---------------------------------------------------------------------------------------------------------------

def _pos_dev(batch, l, rs):
    """Largest |pos difference| (Angstrom, mean-centred) between row l's stored molecule and rows rs'."""
    ptr = batch.ptr
    ref = batch.pos[int(ptr[l]):int(ptr[l + 1])].double()
    worst = 0.0
    for r in rs:
        p = batch.pos[int(ptr[r]):int(ptr[r + 1])].double()
        if p.shape != ref.shape:
            return float('inf')
        worst = max(worst, float(((p - p.mean(0)) - (ref - ref.mean(0))).abs().max()))
    return worst


def build_output(batch, cn, leaders):
    """The representative rows with their canonical pose (module docstring, step 5)."""
    out = batch.subsample_new_batch(leaders)
    L, A, t, rv, _ = canonical_description(cn, leaders).crystal_params()
    for name, v in (('cell_lengths', L), ('cell_angles', A), ('aunit_centroid', t), ('aunit_orientation', rv)):
        old = getattr(out, name)
        setattr(out, name, v.reshape(old.shape).to(old.dtype))
    out.aunit_handedness = torch.ones_like(out.aunit_handedness)
    out.box_analysis()
    return out


def rewrite_batch(batch, tol=TOL, source_row=None, image_id=None, energy_key='elj'):
    """Canonical one-row-per-crystal rows of one stored batch.

    Returns a dict:
    - out: the output batch;
    - leaders: [m] long, the source row of each output row;
    - prov: the provenance dict;
    - stats: counts;
    - cn: the canonicalise() arrays."""
    n = int(batch.num_graphs)
    t_start = time.time()
    cn = canonicalise(batch)
    stats = dict(canonicalise=cn['stats'])
    log(f'  canonicalised: {cn["stats"]}')
    centroid32 = batch.aunit_centroid.reshape(n, -1).float()
    mol = molecule_ids(batch)
    energy = getattr(batch, energy_key, None)
    energy = None if energy is None else torch.as_tensor(energy).reshape(-1).double()
    # ---- groups: the source grouping of an image-written file, otherwise one row each
    if source_row is not None:
        rep, code, moved, cases, unresolved, gstats = group_images(cn, source_row, image_id, centroid32, tol)
        stats['image_groups'] = gstats
        log(f'  image groups: {gstats}')
    else:
        code = cn['code'].clone()
        rep = torch.where(code == 0, torch.arange(n), torch.full((n,), -1, dtype=torch.long))
        moved = torch.zeros(n, dtype=torch.bool)
        cases, unresolved = [], []
    mapped = rep >= 0
    reps = torch.unique(rep[mapped])
    # ---- D7, on the representatives' canonical forms as stored
    mask_r, band_r, flagged = latent_clamp(cn, reps, torch.as_tensor(batch.radius).reshape(-1))
    d7_r = d7_code(mask_r, band_r)
    d7_audit = dict(source_rows=reps[flagged['index']], code=d7_r[flagged['index']], latents=flagged['latents'],
                    clamp_lo=flagged['lo'], clamp_hi=flagged['hi'], latent_names=LATENT_NAMES,
                    note=('per representative whose canonical latent the live clamp moves: its source row, the D7 '
                          'code, the 12 latents (charts.LatentV1Chart.latents of the canonical form as stored, '
                          'float32) and the clamp bounds they were judged against'))
    at = torch.full((n,), -1, dtype=torch.long)
    at[reps] = torch.arange(len(reps))
    clamp_mask = torch.zeros(n, dtype=torch.int32)
    clamp_mask[mapped] = mask_r[at[rep[mapped]]]
    d7_rows = torch.zeros(n, dtype=torch.int8)
    d7_rows[mapped] = d7_r[at[rep[mapped]]]
    refused_d7 = d7_rows != 0
    code[refused_d7] = d7_rows[refused_d7]
    stats['d7'] = dict(representatives=len(reps), refused_representatives=int((d7_r != 0).sum()),
                       by_code={CODES[k]: int((d7_r == k).sum()) for k in (20, 21, 22, 23)},
                       by_latent_entry={nm: int(((mask_r >> j) & 1).sum()) for j, nm in enumerate(LATENT_NAMES)
                                        if int(((mask_r >> j) & 1).sum())},
                       refused_rows=int(refused_d7.sum()))
    log(f'  D7: {stats["d7"]}')
    # ---- one row per crystal across the surviving representatives
    live = reps[d7_r == 0]
    lead = greedy_leaders(cn['X'], live, mol, tol)
    leader_of = torch.full((n,), -1, dtype=torch.long)
    leader_of[live] = lead
    dup, dup_lead = live[lead != live], lead[lead != live]
    code[dup] = SAME_CRYSTAL
    final = torch.where(mapped & ~refused_d7, rep, torch.full_like(rep, -1))
    ok_rows = final >= 0
    final[ok_rows] = leader_of[final[ok_rows]]
    leaders = torch.unique(lead)
    out_index = torch.full((n,), -1, dtype=torch.long)
    out_index[leaders] = torch.arange(len(leaders))
    out_row = torch.full((n,), -1, dtype=torch.long)
    out_row[ok_rows] = out_index[final[ok_rows]]
    if not bool((out_row[ok_rows] >= 0).all()):
        raise RuntimeError('a mapped row has no output row')
    group_size = torch.zeros(n, dtype=torch.long)
    group_size[ok_rows] = torch.bincount(out_row[ok_rows], minlength=len(leaders))[out_row[ok_rows]]
    gkey = torch.where(mapped, rep, torch.arange(n))
    group_size[~ok_rows] = torch.bincount(gkey, minlength=n)[gkey[~ok_rows]]
    dup_groups = {}
    for r, l in zip(dup.tolist(), dup_lead.tolist()):
        dup_groups.setdefault(l, []).append(r)
    dup_report = [dict(leader=l, merged=rs, energy=None if energy is None else [float(v) for v in energy[[l] + rs]],
                       chart_dev=float(chart_dev(cn['X'][rs], cn['X'][l].expand(len(rs), -1)).max()),
                       pos_max_dev=_pos_dev(batch, l, rs)) for l, rs in dup_groups.items()]
    stats['dedupe'] = dict(candidates=len(live), leaders=len(leaders), merged_duplicates=len(dup),
                           duplicate_groups=len(dup_groups))
    log(f'  dedupe: {stats["dedupe"]}')
    if energy is not None and bool(ok_rows.any()):
        o = out_row[ok_rows]
        lo_e = torch.full((len(leaders),), float('inf'), dtype=F64).scatter_reduce(0, o, energy[ok_rows], 'amin')
        hi_e = torch.full((len(leaders),), float('-inf'), dtype=F64).scatter_reduce(0, o, energy[ok_rows], 'amax')
        spread = hi_e - lo_e
        stats['stored_energy_spread_within_output_rows'] = dict(max=float(spread.max()),
                                                                rows_with_spread=int((spread > 0).sum()))
    for case in cases:
        case['out_row'] = int(out_row[case['rep_row']])
    # ---- source rows whose stored centroid the live builder clips: the stored energy is the clipped crystal's
    src_band = ((centroid32 > CLIP_HI) & (centroid32 < 1)).any(1)
    stats['source_clip_band'] = dict(
        source_rows=int(src_band.sum()), output_rows_whose_representative_is_one=int(src_band[leaders].sum()),
        note='stored centroid coordinate in (1 - 1e-4, 1): the builder scores it clipped to 1 - 1e-4')
    # ---- stored x at a value an upstream writer clamps to (pool_anchors: x in (0.9999, 1) clamped, then moved by
    # -1/2 into the sg 2 box), or exactly on the x = 0 wall of the sg 2 box (a general-position minimum has no
    # reason to sit there; a search that clamps at the wall leaves it). When every image of a source carries it, the
    # source itself was written there: the images collapse, no unmoved image exists, and the output row is that
    # stored crystal. Recorded, not refused.
    x_sig = {}
    for name, val in (('x_eq_f32_0.4999', X_EDGE_A), ('x_eq_f32_0.9999_minus_half', X_EDGE_B), ('x_eq_0', 0.0)):
        hit = centroid32[:, 0] == val
        rec = dict(source_rows=int(hit.sum()), output_rows=int(hit[leaders].sum()))
        if source_row is not None and bool(hit.any()):
            _, inv = torch.unique(source_row, return_inverse=True)
            tot = torch.bincount(inv)
            h = torch.bincount(inv, weights=hit.double(), minlength=len(tot)).long()
            rec.update(sources_every_image=int(((h == tot) & (h > 0)).sum()),
                       sources_some_images=int(((h > 0) & (h < tot)).sum()),
                       output_rows_whose_source_has_it_on_every_image=int(((h == tot) & (h > 0))[inv[leaders]].sum()))
        x_sig[name] = rec
    stats['stored_x_at_writer_edge'] = x_sig
    if 'reduction_en' in batch.keys():
        red = torch.as_tensor(batch.reduction_en).reshape(-1)[leaders]
        stats['stored_reduction_en'] = dict(
            output_rows_nonzero=int((red != 0).sum()),
            nonzero_and_basis_changed=int(((red != 0) & cn['basis_changed'][leaders]).sum()),
            note=('carried from the source row as stored; the canonical cell is reduced, so the rewritten row\'s '
                  'own reduction penalty is 0 (checks: energy.reduction_en_canonical_max)'))
    # ---- the census of output rows close to, but beyond, the one-crystal tolerance
    stats['near_pair_census'] = near_pair_census(cn['X'], leaders, mol, energy)
    log(f'  near-pair census: { {k: v for k, v in stats["near_pair_census"].items() if k != "closest"} }')
    mol_ident = getattr(batch, 'identifier', None)
    if mol_ident is not None:
        stats['molecules_in'] = int(len(torch.unique(mol)))
        stats['molecules_out'] = int(len(torch.unique(mol[leaders])))
    rotvec = torch.full((n, 3), float('nan'), dtype=torch.float32)
    ok_c = torch.nonzero(torch.isfinite(cn['cell']).all(1))[:, 0]
    for lo in range(0, len(ok_c), 200000):
        r = ok_c[lo:lo + 200000]
        rotvec[r] = canonical_description(cn, r).crystal_params()[3].reshape(len(r), 3).float()
    out = build_output(batch, cn, leaders)
    log(f'  built {len(leaders):,} output rows')
    prov = dict(out_row=out_row, code=code, group_size=group_size, moved_image=moved, representative=code == KEPT,
                source_row_of_output=leaders.clone(), special_metric=cn['n_auto'] > 1,
                basis_changed=cn['basis_changed'], latent_clamp_mask=clamp_mask, tie=cn['tie'],
                canonical_cell=cn['cell'].float(), canonical_t=cn['t'].float(), canonical_rotvec=rotvec,
                d7=d7_audit, source_centroid_in_clip_band=src_band,
                out_rows_with_moved_images=torch.tensor(sorted({c['out_row'] for c in cases}), dtype=torch.long),
                moved_image_cases=cases, unresolved_image_groups=unresolved, duplicate_groups=dup_report,
                refused_rows={CODES[k]: torch.nonzero(code == k)[:, 0] for k in CODES
                              if k >= 10 and bool((code == k).any())})
    if source_row is not None:
        prov['source_row_original'] = source_row.clone()
        prov['image_id_original'] = image_id.clone()
    stats.update(rows_in=n, rows_out=len(leaders),
                 rows_by_code={CODES[k]: int((code == k).sum()) for k in CODES if bool((code == k).any())},
                 rows_refused=int((code >= 10).sum()),
                 rows_mapped_onto_another_row=int(((code == SAME_CRYSTAL) | (code == MOVED_IMAGE)).sum()),
                 moved_image_cases=len(cases), moved_image_rows=int(moved.sum()),
                 special_metric_rows_out=int((cn['n_auto'][leaders] > 1).sum()),
                 basis_changed_rows_out=int(cn['basis_changed'][leaders].sum()),
                 seconds=time.time() - t_start)
    return dict(out=out, leaders=leaders, prov=prov, stats=stats, cn=dict(X=cn['X']))


# ---------------------------------------------------------------------------------------------------------------
# checks on the file as written (the GFN functions are called, never edited)
# ---------------------------------------------------------------------------------------------------------------

def _gfn():
    import energy_sampling.train as train
    from energy_sampling import buffer
    return train, buffer


_CONDITION_CHILD = r'''
import json, sys
from types import SimpleNamespace
import energy_sampling.train as train
path = sys.argv[1]
cond = train.Modeller._load_condition_file(path)
stub = SimpleNamespace(energy_function=SimpleNamespace(is_crystal=True, latent_energy=False))
train.Modeller._refuse_mirrored_rows(stub, cond, path, 'molecules_path')
zp1, zpn = train.mirrored_rows(cond)
print('FDCHILD' + json.dumps(dict(rows=int(cond.num_graphs), zp1=len(zp1), zp_gt1=len(zpn))))
'''


def condition_file_check(path, in_child):
    """``Modeller._load_condition_file`` (molecules_path: a full ``torch.load``, no mmap, then the 'prior' batch) and
    ``Modeller._refuse_mirrored_rows`` on the written file. ``in_child``: in a child process, for a large file,
    whose full load would otherwise sit beside the source and output this process already holds."""
    if not in_child:
        train, _ = _gfn()
        try:
            cond = train.Modeller._load_condition_file(str(path))
            stub = SimpleNamespace(energy_function=SimpleNamespace(is_crystal=True, latent_energy=False))
            train.Modeller._refuse_mirrored_rows(stub, cond, str(path), 'molecules_path')
            zp1, zpn = train.mirrored_rows(cond)
        except Exception as e:  # reported as a failed check, which stops the rename
            return dict(ok=False, error=f'{type(e).__name__}: {e}')
        res = dict(rows=int(cond.num_graphs), zp1=len(zp1), zp_gt1=len(zpn))
        del cond
    else:
        r = subprocess.run([sys.executable, '-c', _CONDITION_CHILD, str(path)], capture_output=True, text=True,
                           env=os.environ.copy(), cwd=str(Path(__file__).resolve().parents[1]))
        line = [x for x in r.stdout.splitlines() if x.startswith('FDCHILD')]
        if r.returncode != 0 or not line:
            return dict(ok=False, returncode=r.returncode, stderr_tail=r.stderr[-3000:])
        res = json.loads(line[-1][len('FDCHILD'):])
    res['ok'] = res['zp1'] == 0
    return res


def check_loaders(path, d, key, energy_key, buffer_rows=20000, seed=0):
    """The trainer's prior-file loader functions on the written file (``d``: that file, loaded once).

    - ``train.mirrored_rows`` and ``Modeller._refuse_mirrored_rows``, on the 'equalized_prior' batch that
      ``Modeller.init_prior_dataset`` reads (``condition_file_check`` runs the 'prior' batch that
      ``Modeller._load_condition_file`` returns for molecules_path);
    then, on the batch under ``key`` itself:
    - ``train.mirrored_rows`` again;
    - ``buffer.strip_lazy_sg_caches`` and ``buffer._uniform_lj_coeff``;
    - ``latent_params(gauge_fix_free_axes=True)``, the trainer's ``_batch_latents``, on every row;
    - D7 again, on every row as written: the unclipped latent inside the live clamp, no centroid in the clip band
      (``d7_on_written_rows``; both counts must be 0);
    - ``CrystalBuffer`` on a seeded subsample, including its ``orient_molecule('std')``; the pos it leaves is
      compared with the stored pos."""
    train, buffer = _gfn()
    res = dict(ok=False)
    prior = d['equalized_prior'] if 'equalized_prior' in d else d[key]
    stub = SimpleNamespace(energy_function=SimpleNamespace(is_crystal=True, latent_energy=False))
    zp1, zpn = train.mirrored_rows(prior)
    res['mirrored_rows'] = dict(zp1=len(zp1), zp_gt1=len(zpn))
    train.Modeller._refuse_mirrored_rows(stub, prior, str(path), 'prior_path')
    del prior
    own = d[key]                       # the rest runs on this key's own batch
    zp1, zpn = train.mirrored_rows(own)
    res['mirrored_rows_own_batch'] = dict(key=key, zp1=len(zp1), zp_gt1=len(zpn))
    buffer.strip_lazy_sg_caches(own)
    res['lj_coeff_stamp'] = buffer._uniform_lj_coeff(own, 'fd_canonical_prior check')
    n = int(own.num_graphs)
    fin, at_bound, outside, band = True, 0, 0, 0
    for lo in range(0, n, 100000):
        sub = own.subsample_new_batch(torch.arange(lo, min(lo + 100000, n)))
        # D7 on the rows as written: the unclipped latent (LatentV1Chart.latents) inside the live clamp, and no
        # centroid coordinate in the builder clip band
        ch = charts.LatentV1Chart(SG, torch.as_tensor(sub.radius).reshape(-1).float())
        raw = ch.latents(stored_description(sub))
        clo, chi = ch.clamp_bounds()
        outside += int(((raw < clo) | (raw > chi) | ~torch.isfinite(raw)).any(1).sum())
        c32 = sub.aunit_centroid.reshape(int(sub.num_graphs), -1).float()
        band += int(((c32 > CLIP_HI) & (c32 < 1)).any(1).sum())
        lat = sub.latent_params(gauge_fix_free_axes=True)
        fin &= bool(torch.isfinite(lat).all())
        at_bound += int((lat.abs() >= 1).any(1).sum())
    res['latent_params'] = dict(rows=n, finite=fin, rows_with_an_entry_at_plus_minus_1=at_bound,
                                note='an entry at +-1 is a box face (a centroid on a face, say), not a clamp')
    res['d7_on_written_rows'] = dict(rows=n, latent_outside_live_clamp=outside, centroid_in_clip_band=band)
    g = torch.Generator().manual_seed(seed)
    pick = torch.sort(torch.randperm(n, generator=g)[:buffer_rows]).values
    sub = own.subsample_new_batch(pick)
    pos0 = sub.pos.clone().double()
    buf = train.CrystalBuffer(sub, device='cpu', max_z_prime=1, x_fn=None,
                              y_fn=energy_key if energy_key in sub.keys() else None,
                              exclude_keys=train.BULKY_ATTR_EXCLUDE_KEYS)
    bb = buf.batch
    c0 = torch.zeros_like(pos0)
    c1 = torch.zeros_like(pos0)
    for b_, p_, c_ in ((sub, pos0, c0), (bb, bb.pos.double(), c1)):
        heavy = (b_.z > 1).double()
        cen = torch.zeros(int(b_.num_graphs), 3, dtype=F64).index_add_(0, b_.batch, p_ * heavy[:, None])
        cnt = torch.zeros(int(b_.num_graphs), dtype=F64).index_add_(0, b_.batch, heavy)
        c_[:] = p_ - (cen / cnt[:, None])[b_.batch]
    res['crystal_buffer'] = dict(rows=len(pick), x_finite=bool(torch.isfinite(buf.x).all()),
                                 orient_std_max_pos_change=float((c1 - c0).abs().max()))
    res['top_level_keys'] = sorted(map(str, d.keys()))
    res['ok'] = (res['mirrored_rows']['zp1'] == 0 and res['mirrored_rows_own_batch']['zp1'] == 0 and fin
                 and res['crystal_buffer']['x_finite'] and outside == 0 and band == 0)
    return res


def _energy_function(config):
    """MolecularCrystal as tests/crystal/test_fd_measure_contract.py builds it from the canonical config, on CPU."""
    import utils
    from energies.molecular_crystal import MolecularCrystal
    args = utils.get_train_args(['--config', str(config)])
    ec = {'device': 'cpu', 'energy_function': args.energy_function, 'mlip_path': args.mlip_path,
          'space_groups': args.space_groups, 'z_primes': args.z_primes, 'sg_conditioning': args.sg_conditioning,
          'temperature_conditioning': args.temperature_conditioning, 'zp_conditioning': args.zp_conditioning,
          'vector_conditioning': getattr(args, 'vector_conditioning', False),
          'vector_conditioning_dim': getattr(args, 'vector_conditioning_dim', None),
          'embedding_conditioning': getattr(args, 'embedding_conditioning', False),
          'embedding_conditioning_dim': getattr(args, 'embedding_conditioning_dim', None)}
    ec.update(vars(args.energy_config))
    return MolecularCrystal(**ec), args


def init_prior_dataset_probe(d, n_probe=256, seed=0, config=DEFAULT_CONFIG):
    """``Modeller.init_prior_dataset`` end to end on a probe file: ``n_probe`` seeded output rows, written with
    every non-batch entry of the output file to a temporary directory. Uses the real MolecularCrystal of the
    canonical config (CPU) under ``allow_unreferenced``, as ``Modeller.train`` calls it, and a stub Modeller for
    the attributes the loader reads. ``d``: the written file, loaded.

    It reports:
    - the trainer-read eLJ against the stored eLJ, in the stored (raw) currency;
    - the cell parameters the trainer built (through its latent round trip) against those written."""
    train, _ = _gfn()
    b = d['equalized_prior']
    n = int(b.num_graphs)
    g = torch.Generator().manual_seed(seed)
    pick = torch.sort(torch.randperm(n, generator=g)[:n_probe]).values
    sub = b.subsample_new_batch(pick)
    blob = {k: v for k, v in d.items() if not hasattr(v, 'num_graphs') and not (
        torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n)}
    blob['prior'] = sub
    blob['equalized_prior'] = sub
    ef, args = _energy_function(config)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'probe_prior.pt')
        torch.save(blob, p)
        m = train.Modeller.__new__(train.Modeller)
        m.energy_function = ef
        m.device = 'cpu'
        m.args = SimpleNamespace(prior_path=p, energy_function=args.energy_function,
                                 energy_config=args.energy_config, model=SimpleNamespace(hold_dead_latent_rows=True),
                                 space_groups=[SG], z_primes=[1], prior_model_name=None, checkpoints_dir=tmp,
                                 buffer_device='cpu')
        with torch.no_grad(), ef.allow_unreferenced():
            m.init_prior_dataset()
    pb = m.prior_dataset.batch
    coeff = float(ef.lj_coeff)
    res = dict(rows=len(pick), lj_coeff=coeff, energy_function=args.energy_function)
    if args.energy_function == 'elj' and 'elj' in sub.keys():
        read = pb.elj.double().flatten() / coeff
        stored = sub.elj.double().flatten()
        res['trainer_read_minus_stored_elj_raw'] = _q((read - stored).abs())
    built = torch.cat([pb.cell_lengths, pb.cell_angles, pb.aunit_centroid, pb.aunit_orientation], 1).double()
    want = torch.cat([sub.cell_lengths, sub.cell_angles, sub.aunit_centroid, sub.aunit_orientation], 1).double()
    res['trainer_built_minus_written_params'] = _q((built - want).abs().amax(1))
    res['prior_dataset_rows'] = int(len(m.prior_dataset))
    return res


def same_crystal_min_gamma(a, b, mol):
    """Largest atom deviation (Angstrom) of a against b, minimised over the 8 Gamma+ translations of sg 2 applied to
    a; also the second smallest (the control: a wrong translation)."""
    w = domains.translation_section(SG).w
    eye = torch.eye(3, dtype=F64)
    D = torch.cat([checks.same_crystal(ops.apply(a, eye, w[k]), b, mol, cell_rtol=1e-5) for k in range(len(w))])
    s = torch.sort(D).values
    return float(s[0]), float(s[1])


def check_same_crystal(src_batch, out_batch, leaders, sample, pairs=None):
    """Atoms: each sampled output row against its source row (reduced by ops.reduce_cell), with the output row's
    molecule. ``pairs``: extra (source row, output row) pairs (moved images)."""
    jobs = [(int(leaders[j]), int(j)) for j in sample.tolist()] + list(pairs or [])
    best, ctrl = [], []
    for s, j in jobs:
        a = ops.reduce_cell(stored_description(src_batch, torch.tensor([s]))).desc
        b = stored_description(out_batch, torch.tensor([j]))
        p0, p1 = int(out_batch.ptr[j]), int(out_batch.ptr[j + 1])
        mol = MoleculeRef.from_coords(out_batch.pos[p0:p1], out_batch.z[p0:p1])
        x, y = same_crystal_min_gamma(a, b, mol)
        best.append(x)
        ctrl.append(y)
    return torch.tensor(best, dtype=F64), torch.tensor(ctrl, dtype=F64)


def check_trainer_build(out_batch, sample):
    """The trainer's build of sampled output rows. The latent round trip (``latent_params``, then
    ``latent_to_cell_params``) is compared with the written parameters. The posed asymmetric unit (``pose_aunit``,
    std_orientation=False) is compared with T t + R X of the written description, X centred as MoleculeRef
    centres it. Control: R transposed, which coincides with R for half-turn rotations; its median and the share
    of rows above 0.1 Angstrom are reported."""
    b = out_batch.subsample_new_batch(sample)
    want = b.full_cell_parameters().double()
    rt = b.clone()
    rt.latent_to_cell_params(rt.clone().latent_params(gauge_fix_free_axes=True))
    got = rt.full_cell_parameters().double()
    posed = rt.clone()
    posed.pose_aunit(std_orientation=False)
    desc = stored_description(b)
    T = desc.cell_matrix()
    dev, ctrl = [], []
    for i in range(int(b.num_graphs)):
        p0, p1 = int(b.ptr[i]), int(b.ptr[i + 1])
        X = MoleculeRef.from_coords(b.pos[p0:p1], b.z[p0:p1]).pos
        c = T[i] @ desc.t[i, 0]
        lit = c + X @ desc.R[i, 0].T
        dev.append(float((posed.pos[p0:p1].double() - lit).norm(dim=1).max()))
        ctrl.append(float((posed.pos[p0:p1].double() - (c + X @ desc.R[i, 0])).norm(dim=1).max()))
    ctrl = torch.tensor(ctrl, dtype=F64)
    return dict(rows=len(sample), latent_round_trip_param_dev=_q((got - want).abs().amax(1)),
                posed_atoms_vs_description=_q(dev),
                control_R_transposed=dict(p50=float(ctrl.median()) if len(ctrl) else None,
                                          frac_gt_0p1=float((ctrl > 0.1).double().mean()) if len(ctrl) else None))


def score(batch, rows, chunk=100):
    """Raw eLJ and reduction penalty of stored rows through ``MolCrystalData.analyze`` (CPU, no grad)."""
    E, Rd = [], []
    rows = torch.as_tensor(rows, dtype=torch.long)
    for lo in range(0, len(rows), chunk):
        c = batch.subsample_new_batch(rows[lo:lo + chunk])
        c.box_analysis()
        with torch.no_grad():
            o = c.analyze(['reduction_en', 'elj'], assign_outputs=False, **ANALYZE_KW)
        E.append(o['elj'].double().flatten())
        Rd.append(o['reduction_en'].double().flatten())
    return torch.cat(E), torch.cat(Rd)


def check_carried(src_batch, out_batch, leaders, energy_key):
    """Every per-row and per-atom field other than the pose fields equals the representative source row's."""
    res, mism = {}, []
    sp = src_batch.ptr
    counts = (sp[leaders + 1] - sp[leaders])
    node_idx = torch.repeat_interleave(sp[leaders], counts) + (
        torch.arange(int(counts.sum())) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts))
    n_src, nn_src = int(src_batch.num_graphs), int(sp[-1])
    for k in sorted(out_batch.keys()):
        if k in POSE_FIELDS or k in ('batch', 'ptr'):
            continue
        v_out, v_src = out_batch[k], src_batch[k]
        if torch.is_tensor(v_src) and v_src.dim() >= 1 and v_src.shape[0] == n_src and n_src > 1:
            same = torch.equal(v_out, v_src[leaders]) or (v_out.is_floating_point() and torch.equal(
                torch.isnan(v_out), torch.isnan(v_src[leaders])) and torch.equal(
                torch.nan_to_num(v_out), torch.nan_to_num(v_src[leaders])))
            kind = 'row'
        elif torch.is_tensor(v_src) and v_src.dim() >= 1 and v_src.shape[0] == nn_src:
            same = torch.equal(v_out, v_src[node_idx])
            kind = 'atom'
        elif isinstance(v_src, (list, tuple)) and len(v_src) == n_src:
            li = leaders.tolist()
            if len(v_out) != len(li):
                same = False
            elif all(isinstance(v_src[i], np.ndarray) for i in li[:10]):
                same = all(np.array_equal(v_out[j], v_src[i]) for j, i in enumerate(li))
            else:
                same = all(v_out[j] == v_src[i] for j, i in enumerate(li))
            kind = 'list'
        else:
            same = (v_out is v_src) or (torch.is_tensor(v_out) and torch.equal(v_out, v_src)) or v_out == v_src
            kind = 'shared'
        res[k] = kind
        if not same:
            mism.append(k)
    return dict(fields=res, mismatched=mism, energy_key=energy_key,
                energy_carried=energy_key in res and energy_key not in mism)


def check_recanonical(ob, X_float64, sb, leaders, chunk=200000):
    """Every written row re-canonicalised (reduce_cell, fold) must be its own canonical form. Chunked over rows,
    so the check holds a fraction of the batch's working arrays at once."""
    m = int(ob.num_graphs)
    dev = torch.zeros(m, dtype=F64)
    refused = basis = 0
    for lo in range(0, m, chunk):
        r = torch.arange(lo, min(lo + chunk, m))
        part = ob.subsample_new_batch(r)
        cn2 = canonicalise(part)
        dev[r] = chart_dev(cn2['X'], canon_vector(stored_description(part)))
        refused += int((cn2['code'] != 0).sum())
        basis += int(cn2['basis_changed'].sum())
        del cn2, part
    Xw = canon_vector(stored_description(ob))
    # the box fields written beside the new cell are that cell's (box_analysis ran on the rewritten rows)
    from mxtaltools.common.geometry_utils import batch_compute_fractional_transform
    T_fc, T_cf, vol = batch_compute_fractional_transform(ob.cell_lengths, ob.cell_angles)
    box = dict(T_fc_max_abs=float((T_fc - ob.T_fc).abs().max()), T_cf_max_abs=float((T_cf - ob.T_cf).abs().max()),
               cell_volume_max_rel=float(((vol.flatten() - ob.cell_volume.flatten()).abs()
                                          / vol.flatten().abs()).max())) if m else {}
    return dict(rows=m, refused=refused, basis_changed=basis, box_fields_vs_written_cell=box,
                dev_written_vs_refolded=_q(dev, ps=(0.5, 0.99, 0.999)), rows_dev_gt_tol=int((dev > TOL).sum()),
                dev_written_vs_float64_canonical=_q(chart_dev(Xw, X_float64[leaders]), ps=(0.5, 0.99)),
                in_domain_rows=int(domains.in_domain(stored_description(ob))[0].sum()),
                stored_bits_equal=dict(
                    cell=int((torch.cat([ob.cell_lengths, ob.cell_angles], 1) == torch.cat(
                        [sb.cell_lengths[leaders], sb.cell_angles[leaders]], 1)).all(1).sum()),
                    orientation=int((ob.aunit_orientation == sb.aunit_orientation[leaders]).all(1).sum())))


def _quarter_share(b):
    """Share of rows in each (y, z) half of the trainer's sg 2 centroid box: (y<1/2, z<1/2), (y<1/2, z>=1/2),
    (y>=1/2, z<1/2), (y>=1/2, z>=1/2)."""
    c = b.aunit_centroid.reshape(int(b.num_graphs), -1)[:, :3].double()
    q = torch.bincount((c[:, 1] >= 0.5).long() * 2 + (c[:, 2] >= 0.5).long(), minlength=4).double()
    return [round(float(v), 6) for v in q / max(1.0, float(q.sum()))]


def check_trainer_box(src_batch, out_batch, prov, sample):
    """Where the rows sit in the trainer's sg 2 latent box (ASYM_UNITS [0.5, 1, 1]), and whether the source's rows
    are recovered by spreading each output row over its trainer-box images.

    - the share of rows in each (y, z) quarter of the box, source and output (every row);
    - the number of trainer-box images (``domains.images(within='latent_v1_box')``) of each sampled output row;
    - every source row mapped to a sampled output row (basis unchanged) against the nearest of those images:
      the largest |difference| of centroid (fractional; y and z wrapped with period 1, x with period 1/2, so a row
      on an x face matches its face partner) and rotation-matrix entries. Moved images sit at up to about 1e-4."""
    res = dict(quarters='(y<1/2,z<1/2), (y<1/2,z>=1/2), (y>=1/2,z<1/2), (y>=1/2,z>=1/2)',
               source_share=_quarter_share(src_batch), output_share=_quarter_share(out_batch), k_box=K_BOX)
    if len(sample) == 0:
        return res
    imgs, srcidx, _ = domains.images(stored_description(out_batch, sample), within='latent_v1_box')
    per_row = torch.bincount(srcidx, minlength=len(sample))
    res['images_per_sampled_output_row'] = {str(k): int(v) for k, v in zip(
        *[t.tolist() for t in torch.unique(per_row, return_counts=True)])}
    kmax = int(per_row.max())
    table = torch.full((len(sample), kmax), -1, dtype=torch.long)
    start = torch.cumsum(per_row, 0) - per_row
    order = torch.argsort(srcidx, stable=True)
    slot = torch.arange(len(srcidx)) - start[srcidx[order]]
    table[srcidx[order], slot] = order
    pos = torch.full((int(out_batch.num_graphs),), -1, dtype=torch.long)
    pos[sample] = torch.arange(len(sample))
    out_row = prov['out_row']
    s_rows = torch.nonzero((out_row >= 0) & (pos[out_row.clamp(min=0)] >= 0))[:, 0]
    basis = prov['basis_changed'][s_rows]
    s_rows = s_rows[~basis]
    i_of = pos[out_row[s_rows]]
    ds = stored_description(src_batch, s_rows)
    cand = table[i_of]                                                     # [S, kmax]
    valid = cand >= 0
    t_img = imgs.t[cand.clamp(min=0), 0]                                   # [S, k, 3]
    R_img = imgs.R[cand.clamp(min=0), 0]
    dt = (t_img - ds.t[:, None, 0]).abs()
    per = torch.tensor([0.5, 1.0, 1.0], dtype=F64)
    dt = torch.minimum(dt % per, per - dt % per).amax(-1)
    dR = (R_img - ds.R[:, None, 0]).abs().flatten(2).amax(-1)
    dev = torch.where(valid, torch.maximum(dt, dR), torch.full_like(dt, float('inf'))).amin(1)
    cell_dev = (ds.cell - stored_description(out_batch, out_row[s_rows]).cell).abs().amax(1)
    moved = prov['moved_image'][s_rows]
    res['source_rows_vs_nearest_image'] = dict(
        source_rows=len(s_rows), skipped_basis_changed=int(basis.sum()), cell_max_abs=float(cell_dev.max()),
        dev_unmoved=_q(dev[~moved], ps=(0.5, 0.99)), unmoved_rows_dev_gt_1em6=int((dev[~moved] > 1e-6).sum()),
        moved_rows=int(moved.sum()), dev_moved=_q(dev[moved], ps=(0.5,)))
    return res


def _energy_outliers(e, stored, rows_out, rows_src, top=10, thresholds=(0.01, 0.1, 1.0)):
    d = (e - stored).abs()
    o = torch.argsort(d, descending=True)[:top]
    return dict(rows=len(d), rows_above={f'{t:g}': int((d > t).sum()) for t in thresholds},
                largest=[dict(out_row=int(rows_out[k]), source_row=int(rows_src[k]), stored=float(stored[k]),
                              rescored=float(e[k])) for k in o.tolist()])


def run_checks(path, src, results, out_keys, check=2000, energy_check=1000, probe=256, seed=0,
               config=DEFAULT_CONFIG, condition_in_child=True):
    """Every check on the file as written. Returns {batch key: results, '_top_level': entries kept/rewritten,
    '_condition_file': ``condition_file_check``}."""
    d = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    out = {}
    top = {}
    for k, v in src.items():
        if k in out_keys and not hasattr(v, 'num_graphs'):
            w = d[k]
            if torch.is_tensor(v):
                top[k] = 'equal' if (torch.is_tensor(w) and w.shape == v.shape and torch.equal(w, v)) else (
                    f'rewritten ({tuple(v.shape)} -> {tuple(w.shape)})')
            else:
                top[k] = 'equal' if (w == v) else 'DIFFERENT'
    out['_top_level'] = dict(entries=top, missing=sorted(set(map(str, src)) - set(map(str, d))),
                             added=sorted(set(map(str, d)) - set(map(str, src))))
    out['_condition_file'] = condition_file_check(path, condition_in_child)
    log(f'  condition file{" (child process)" if condition_in_child else ""}: {out["_condition_file"]}')
    for key, r in results.items():
        t0 = time.time()
        log(f'[{key}] checks')
        sb, ob = src[key], d[key]
        leaders, cn = r['leaders'], r['cn']
        c = dict()
        c['loader'] = check_loaders(path, d, key, r['energy_key'], seed=seed)
        log(f'  loader: {c["loader"]}')
        m = int(ob.num_graphs)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(m, generator=g)
        sample = perm[:check]
        es = perm[:energy_check]       # the first rows of the same permutation: the atom sample's rows come first
        c['recanonical'] = check_recanonical(ob, cn['X'], sb, leaders)
        log(f'  recanonical: {c["recanonical"]}')
        c['carried'] = check_carried(sb, ob, leaders, r['energy_key'])
        log(f'  carried: mismatched {c["carried"]["mismatched"]}')
        c['trainer_box'] = check_trainer_box(sb, ob, r['prov'], sample)
        log(f'  trainer box: {c["trainer_box"]}')
        best, ctrl = check_same_crystal(sb, ob, leaders, sample)
        c['same_crystal'] = dict(rows=len(sample), min_over_gamma_plus_A=_q(best), control_wrong_translation_min_A=(
            float(ctrl.min()) if len(ctrl) else None))
        log(f'  same crystal: {c["same_crystal"]}')
        c['trainer_build'] = check_trainer_build(ob, sample)
        log(f'  trainer build: {c["trainer_build"]}')
        e_out, red_out = score(ob, es)
        e_src, _ = score(sb, leaders[es])
        res_e = dict(rows=len(es), canonical_minus_source_rescored=_q((e_out - e_src).abs()),
                     reduction_en_canonical_max=float(red_out.abs().max()) if len(es) else None)
        has_e = r['energy_key'] is not None and r['energy_key'] in ob.keys()
        st_all = torch.as_tensor(ob[r['energy_key']]).double().flatten() if has_e else None
        if has_e:
            st = st_all[es]
            res_e['canonical_rescored_minus_stored'] = _q((e_out - st).abs())
            res_e['source_rescored_minus_stored'] = _q((e_src - st).abs())
            res_e['canonical_rescored_minus_stored_outliers'] = _energy_outliers(e_out, st, es, leaders[es])
            res_e['source_rescored_minus_stored_outliers'] = _energy_outliers(e_src, st, es, leaders[es])
        c['energy'] = res_e
        log(f'  energy: {res_e}')
        # output rows whose source centroid the builder clips: the stored energy is the clipped crystal's
        band_out = torch.nonzero(r['prov']['source_centroid_in_clip_band'][leaders])[:, 0]
        if len(band_out):
            e_bo, _ = score(ob, band_out)
            e_bs, _ = score(sb, leaders[band_out])
            c['source_clip_band'] = dict(output_rows=len(band_out),
                                         output_rescored_minus_source_rescored=_q((e_bo - e_bs).abs()))
            if has_e:
                c['source_clip_band'].update(output_rescored_minus_stored=_q((e_bo - st_all[band_out]).abs()),
                                             source_rescored_minus_stored=_q((e_bs - st_all[band_out]).abs()))
            log(f'  source clip band: {c["source_clip_band"]}')
        cases = r['prov']['moved_image_cases']
        if cases:
            outj = [int(r['prov']['out_row'][cs['rep_row']]) for cs in cases]
            pairs_rep = [(cs['rep_row'], j) for cs, j in zip(cases, outj)]
            pairs_mv = [(mv, j) for cs, j in zip(cases, outj) for mv in cs['moved_rows']]
            b_rep, _ = check_same_crystal(sb, ob, leaders, torch.zeros(0, dtype=torch.long), pairs_rep)
            b_mv, _ = check_same_crystal(sb, ob, leaders, torch.zeros(0, dtype=torch.long), pairs_mv)
            e_o, _ = score(ob, outj)
            e_rep, _ = score(sb, [p[0] for p in pairs_rep])
            e_mv, _ = score(sb, [p[0] for p in pairs_mv])
            case_of_mv = torch.tensor([k for k, cs in enumerate(cases) for _ in cs['moved_rows']], dtype=torch.long)
            e_mv_out = e_o[case_of_mv]
            c['moved_images'] = dict(cases=len(cases), moved_rows=len(pairs_mv),
                                     output_vs_unmoved_image_A=_q(b_rep), output_vs_moved_image_A=_q(b_mv),
                                     elj_output_minus_unmoved_image=_q((e_o - e_rep).abs()),
                                     elj_output_minus_moved_image=_q((e_mv_out - e_mv).abs()))
            if has_e:
                st_o = st_all[torch.tensor(outj)]
                best_mv = torch.full((len(cases),), float('inf'), dtype=F64).scatter_reduce(
                    0, case_of_mv, (e_mv - st_o[case_of_mv]).abs(), 'amin')
                c['moved_images'].update(
                    stored_minus_rescored_output=_q((e_o - st_o).abs()),
                    stored_minus_rescored_nearest_moved_image=_q(best_mv),
                    cases_where_a_moved_image_is_nearer_the_stored_energy=int((best_mv < (e_o - st_o).abs()).sum()),
                    note=('the stored energy is the source state as the builder scored it, its centroid clipped to '
                          '1 - 1e-4 where it lay in the clip band; the output row is the unmoved image'))
            log(f'  moved images: {c["moved_images"]}')
        if probe:
            c['init_prior_dataset'] = init_prior_dataset_probe(d, probe, seed, config)
            log(f'  init_prior_dataset probe: {c["init_prior_dataset"]}')
        c['seconds'] = time.time() - t0
        out[key] = c
    return out


# ---------------------------------------------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------------------------------------------

def _is_batch(v):
    return hasattr(v, 'num_graphs') and hasattr(v, 'cell_lengths')


def output_paths(source):
    stem = str(Path(source).with_suffix(''))
    return (Path(stem + SUFFIX + '.pt'), Path(stem + SUFFIX + '_provenance.pt'),
            Path(stem + SUFFIX + '_manifest.json'))


def _git_head(path):
    try:
        return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              timeout=30).stdout.strip() or None
    except Exception:
        return None


def code_digest():
    """crystal_fd identity: the table digest (manifest) and schema; the sha256 of its .py sources, since the package
    is uncommitted; the MXtalTools and GFN HEAD commits; and this script's sha256."""
    import mxtaltools.crystal_fd as fd
    pkg = Path(fd.__file__).parent
    src = hashlib.sha256()
    for f in sorted(pkg.glob('*.py')):
        src.update(f.name.encode())
        src.update(f.read_bytes().replace(b'\r\n', b'\n'))
    return dict(fd_tables_v1_sha256=tables.TABLE_HASH, fd_table_schema=tables.SCHEMA_VERSION,
                crystal_fd_sources_sha256=src.hexdigest(), crystal_fd_path=str(pkg),
                mxtaltools_head=_git_head(pkg), gfn_head=_git_head(Path(__file__).parent),
                script=str(Path(__file__).resolve()),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                torch=torch.__version__, version=VERSION)


def _json_default(o):
    if torch.is_tensor(o):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def companion_counts(companion):
    """The row and molecule counts (in, out) of the other file of a split, from its own ``_fd1`` manifest, which
    this script wrote when it rewrote that file. Returns ({(what, label): (in, out)}, a record of the source)."""
    man_path = output_paths(companion)[2]
    if not man_path.exists():
        raise SystemExit(f'--companion {companion}: its manifest {man_path} does not exist; rewrite it first')
    man = json.loads(man_path.read_text())
    counts = {}
    for key, s in man['keys'].items():
        if isinstance(s, dict):
            label = f'{Path(companion).stem}[{key}]'
            counts[('rows', label)] = (int(s['rows_in']), int(s['rows_out']))
            if 'molecules_in' in s:
                counts[('molecules', label)] = (int(s['molecules_in']), int(s['molecules_out']))
    rec = dict(source=str(companion), manifest=str(man_path), manifest_version=man.get('digest', {}).get('version'),
               counts={f'{w} of {lab}': list(v) for (w, lab), v in counts.items()})
    return counts, rec


def update_counts(src, distinct, key_stats, companion=None):
    """Top-level integer entries named ``n_*`` that count this file's own rows or molecules, or those of the
    other file of a split (``companion``: ``companion_counts``).

    An entry equal to one such source count (rows, or distinct identifiers) and to no other count becomes that
    count's output value; every other ``n_*`` entry is kept, since it counts something else (a writer's inputs,
    or a companion not named). Returns {'updated': {key: {source, output, counts}}, 'kept': {key: value},
    'companion', 'note'}."""
    counts = {}
    for keys in distinct.values():
        key = 'equalized_prior' if 'equalized_prior' in keys else keys[0]
        s = key_stats[key]
        counts[('rows', key)] = (s['rows_in'], s['rows_out'])
        if 'molecules_in' in s:
            counts[('molecules', key)] = (s['molecules_in'], s['molecules_out'])
    comp_rec = None
    if companion is not None:
        comp, comp_rec = companion
        counts.update(comp)
    updated, kept = {}, {}
    for k, v in src.items():
        if not (isinstance(k, str) and k.startswith('n_') and isinstance(v, int) and not isinstance(v, bool)):
            continue
        hits = {(what, key): out for (what, key), (vin, out) in counts.items() if vin == v}
        if len(set(hits.values())) == 1:
            (what, key), out = next(iter(hits.items()))
            updated[k] = dict(source=v, output=int(out), counts=f'{what} of the {key!r} batch' + (
                f' (also {len(hits) - 1} other counts with the same values)' if len(hits) > 1 else ''))
        else:
            kept[k] = v
    return dict(updated=updated, kept=kept, companion=comp_rec,
                note=('an n_* entry equal to a row or molecule count of this file (or of the companion file of its '
                      'split, from that file\'s own manifest) now holds the output count, its source value under '
                      'updated; the kept entries count something else (the writer\'s inputs, say) and are the '
                      'source\'s values, not recomputed'))


def _load_source(source):
    try:
        return torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError:
        return torch.load(source, map_location='cpu', weights_only=False)


def checks_in_child(source, partial, results, out_keys, kw):
    """``run_checks`` in a child process (``--checks-child``), so its loads of the source and of the written file
    do not sit on top of this process's rewrite. Returns the checks dict (as JSON)."""
    state = dict(source=str(source), partial=str(partial), out_keys=out_keys, kw=kw, threads=torch.get_num_threads(),
                 results={k: dict(leaders=r['leaders'], cn=r['cn'], prov=r['prov'], energy_key=r['energy_key'])
                          for k, r in results.items()})
    with tempfile.TemporaryDirectory() as tmp:
        sp, jp = Path(tmp) / 'state.pt', Path(tmp) / 'checks.json'
        torch.save(state, sp)
        del state
        log(f'checks in a child process')
        sys.stdout.flush()
        r = subprocess.run([sys.executable, '-m', 'data_processing.fd_canonical_prior', '--checks-child', str(sp),
                            str(jp)], env=os.environ.copy(), cwd=str(Path(__file__).resolve().parents[1]))
        if r.returncode != 0 or not jp.exists():
            raise SystemExit(f'the checks child failed (exit {r.returncode}); {partial} left in place')
        return json.loads(jp.read_text())


def _checks_child(state_path, out_json):
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    torch.set_num_threads(int(state['threads']))
    src = _load_source(state['source'])
    log(f'checks child: loaded {state["source"]}')
    chk = run_checks(state['partial'], src, state['results'], state['out_keys'], condition_in_child=True,
                     **state['kw'])
    with open(out_json, 'x') as f:
        json.dump(chk, f, default=_json_default)
    log('checks child: done')


def checks_here(source, partial, results, out_keys, kw):
    """``run_checks`` in this process (a small source), returned through JSON as ``checks_in_child`` returns it."""
    src = _load_source(source)
    chk = run_checks(partial, src, results, out_keys, condition_in_child=False, **kw)
    del src
    return json.loads(json.dumps(chk, default=_json_default))


def replaceable(source, finals):
    """``--replace``: the existing output set, accepted only if all of it exists and its manifest says this script
    wrote it from ``source``. Returns {final path: the path in SUPERSEDED_DIR it will be moved to}, and the old
    manifest's version."""
    missing = [str(p) for p in finals if not p.exists()]
    if missing:
        raise SystemExit(f'--replace: {missing} do not exist, so there is no earlier output of this script to replace')
    old = json.loads(finals[2].read_text())
    script = Path(str(old.get('digest', {}).get('script', ''))).name
    if Path(old.get('source', '')) != Path(source) or script != Path(__file__).name:
        raise SystemExit(f'--replace: {finals[2]} does not record this script ({Path(__file__).name}) writing it from '
                         f'{source} (it records {old.get("source")!r}, {script!r}); refusing to replace')
    sup = Path(source).parent / SUPERSEDED_DIR
    moves = {p: sup / p.name for p in finals}
    taken = [str(d) for d in moves.values() if d.exists()]
    if taken:
        raise SystemExit(f'--replace: {taken} already exist; refusing to overwrite an earlier superseded set')
    return moves, old.get('digest', {}).get('version')


def rewrite_file(source, tol=TOL, check=2000, energy_check=1000, probe=256, seed=0, config=DEFAULT_CONFIG,
                 companion=None, replace=False, checks_child=None):
    """Rewrite one prior file (module docstring). Returns the manifest dict.

    ``companion``: the source path of the other file of a split, whose ``_fd1`` manifest gives its counts.
    ``replace``: replace this script's earlier output from the same source (``replaceable``).
    ``checks_child``: run the checks in a child process (None: when the source exceeds CHILD_CHECKS_ABOVE bytes)."""
    source = Path(source)
    finals = output_paths(source)
    data_path, prov_path, man_path = finals
    partials = [Path(str(p) + '.partial') for p in finals]
    partial = partials[0]
    failed = Path(str(man_path) + '.FAILED.json')
    for p in partials + [failed] + ([] if replace else list(finals)):
        if p.exists():
            raise SystemExit(f'refusing to overwrite {p}')
    moves, old_version = replaceable(source, finals) if replace else ({}, None)
    comp = companion_counts(companion) if companion is not None else None
    if checks_child is None:
        checks_child = source.stat().st_size > CHILD_CHECKS_ABOVE
    t0 = time.time()
    src = _load_source(source)
    log(f'loaded {source} ({time.time() - t0:.0f} s)')
    if not isinstance(src, dict):
        raise SystemExit(f'{source} is not a dict of batches')
    if 'fd_canonical' in src:
        raise SystemExit(f'{source} already carries an fd_canonical entry')
    batch_keys = [k for k, v in src.items() if _is_batch(v)]
    if not batch_keys:
        raise SystemExit(f'{source} holds no crystal batch')
    distinct = {}
    for k in batch_keys:
        distinct.setdefault(id(src[k]), []).append(k)
    aligned = {}
    if len(distinct) == 1:
        n = int(src[batch_keys[0]].num_graphs)
        aligned = {k: v for k, v in src.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n}
    else:
        lens = {int(src[ks[0]].num_graphs) for ks in distinct.values()}
        amb = [k for k, v in src.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] in lens]
        if amb:
            raise SystemExit(f'top-level tensors {amb} align with one of several batches: ambiguous')
    image_written = 'source_row' in aligned and 'image_id' in aligned
    results, provenance, key_stats = {}, {}, {}
    new_batch = {}
    for bid, keys in distinct.items():
        key = 'equalized_prior' if 'equalized_prior' in keys else keys[0]
        b = src[key]
        ekey = 'elj' if 'elj' in b.keys() else None
        log(f'[{key}] {int(b.num_graphs):,} rows' + (f' (the same object as {[k for k in keys if k != key]})'
                                                     if len(keys) > 1 else ''))
        r = rewrite_batch(b, tol, aligned['source_row'].long() if image_written else None,
                          aligned['image_id'].long() if image_written else None, energy_key=ekey or 'elj')
        r['energy_key'] = ekey
        results[key] = r
        new_batch[bid] = r['out']
        for k in keys:
            provenance[k] = r['prov'] if k == key else f'the same object as {key}'
            key_stats[k] = r['stats'] if k == key else f'the same object as {key}'
    out_dict = {}
    leaders_single = next(iter(results.values()))['leaders'] if len(distinct) == 1 else None
    for k, v in src.items():
        if _is_batch(v):
            out_dict[k] = new_batch[id(v)]
        elif k in aligned:
            if k == 'source_row':
                out_dict[k] = leaders_single.clone().to(v.dtype)
            elif k == 'image_id':
                out_dict[k] = torch.zeros(len(leaders_single), dtype=v.dtype)
            else:
                out_dict[k] = v[leaders_single].clone()
        else:
            out_dict[k] = v
    counts = update_counts(src, distinct, key_stats, comp)
    for k, rec in counts['updated'].items():
        out_dict[k] = rec['output']
    digest = code_digest()
    out_dict['fd_canonical'] = dict(
        version=VERSION, source=str(source), tolerance=tol, centroid_domain=CENTROID_DOMAIN_NOTE, k_box=K_BOX,
        stored_energy=STORED_ENERGY_NOTE, counts=counts,
        rule=('every row canonicalised by crystal_fd (ops.reduce_cell, domains.fold, +1 sheet); one row per crystal '
              '(chart vectors within the tolerance, seams wrapped; an image-written file grouped by source_row); '
              'D7 rows (trainer latent clamp, centroid clip band) refused; the pose fields replaced and every other '
              'per-row field carried from the representative source row'),
        source_row=('the row index in the source file of the representative row' if image_written else None),
        image_id=('set to 0: every output row is its own image 0' if image_written else None),
        provenance=str(prov_path), manifest=str(man_path), fd_tables_v1_sha256=digest['fd_tables_v1_sha256'],
        crystal_fd_sources_sha256=digest['crystal_fd_sources_sha256'],
        rows={k: (dict(rows_in=s['rows_in'], rows_out=s['rows_out'], rows_refused=s['rows_refused'])
                  if isinstance(s, dict) else s) for k, s in key_stats.items()})
    t0 = time.time()
    torch.save(out_dict, partial)
    log(f'wrote {partial} ({partial.stat().st_size / 2 ** 20:.0f} MiB, {time.time() - t0:.0f} s)')
    out_keys = list(out_dict.keys())
    src_keys = sorted(map(str, src.keys()))
    for r in results.values():  # the checks read the file as written, not these objects
        r['out'] = None
    new_batch.clear()
    del out_dict, src, b
    import gc
    gc.collect()
    kw = dict(check=check, energy_check=energy_check, probe=probe, seed=seed, config=str(config))
    chk = (checks_in_child if checks_child else checks_here)(source, partial, results, out_keys, kw)
    loader_ok = chk['_condition_file']['ok'] and all(c['loader']['ok'] for k, c in chk.items()
                                                     if not k.startswith('_'))
    man = dict(source=str(source), output=str(data_path), provenance=str(prov_path), tolerance=tol,
               tolerance_note=('Chebyshev in chart units (CellChart y, PolytopeARChart u with seams wrapped, R '
                               'entries); float32 storage noise is about 6e-8 per coordinate; the spread of the '
                               'unmoved images of one source is under keys.<key>.image_groups; the output rows '
                               'just beyond it are counted under keys.<key>.near_pair_census'),
               codes={str(k): v for k, v in CODES.items()}, digest=digest,
               centroid_domain=CENTROID_DOMAIN_NOTE, k_box=K_BOX, stored_energy=STORED_ENERGY_NOTE, counts=counts,
               source_top_level_keys=src_keys, output_top_level_keys=sorted(map(str, out_keys)), keys=key_stats,
               moved_image_cases={k: p['moved_image_cases'] for k, p in provenance.items() if isinstance(p, dict)},
               unresolved_image_groups={k: p['unresolved_image_groups'] for k, p in provenance.items()
                                        if isinstance(p, dict)},
               duplicate_groups={k: p['duplicate_groups'] for k, p in provenance.items() if isinstance(p, dict)},
               checks=chk, checks_in_child_process=bool(checks_child), loader_checks_passed=loader_ok,
               replaced=([dict(old=str(p), moved_to=str(d)) for p, d in moves.items()] if replace else None),
               replaced_version=old_version)
    if not loader_ok:
        with open(failed, 'x') as f:
            json.dump(man, f, indent=1, default=_json_default)
        raise SystemExit(f'loader checks failed; {partial} left in place, see {failed}')
    torch.save(dict(source=str(source), output=str(data_path), codes=CODES, tolerance=tol, keys=provenance,
                    latent_names=LATENT_NAMES,
                    note=('per batch key, per source row (an index into that batch): out_row (-1: refused), code '
                          '(CODES), group_size (source rows mapped to the same output row, or to the same refused '
                          'group), moved_image, representative, special_metric (n_auto > 1 at ops.FACE_TOL), '
                          'basis_changed, latent_clamp_mask (bit j: LATENT_NAMES[j] moved by the live clamp), tie, '
                          'canonical_cell / canonical_t / canonical_rotvec (float32 as stored; the rotation vector '
                          'on the trainer\'s +z hemisphere), source_centroid_in_clip_band; d7: the latents and clamp '
                          'bounds of every representative the live clamp moves; source_row_of_output: the source '
                          'row of each output row; out_rows_with_moved_images; refused_rows: code name -> source '
                          'rows; source_row_original / image_id_original for an image-written file')), partials[1])
    with open(partials[2], 'x') as f:
        json.dump(man, f, indent=1, default=_json_default)
    if replace:  # the earlier set is moved aside, never deleted
        next(iter(moves.values())).parent.mkdir(exist_ok=True)
        for p, d in moves.items():
            os.rename(p, d)
            log(f'moved {p} -> {d}')
    for p, f in zip(partials, finals):
        os.rename(p, f)  # raises if f appeared meanwhile: never overwrites
    log(f'done: {data_path}, {prov_path}, {man_path}')
    return man


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['--checks-child']:
        return _checks_child(argv[1], argv[2])
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('source')
    ap.add_argument('--tol', type=float, default=TOL)
    ap.add_argument('--check', type=int, default=2000, help='rows for the atom and trainer-build checks')
    ap.add_argument('--energy-check', type=int, default=1000,
                    help='rows rescored through analyze (eLJ): the first rows of the same seeded permutation')
    ap.add_argument('--probe', type=int, default=256, help='rows through Modeller.init_prior_dataset (0: skip)')
    ap.add_argument('--config', default=str(DEFAULT_CONFIG), help='the config the probe builds MolecularCrystal from')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--companion', default=None,
                    help='the source of the other file of a split; its _fd1 manifest must exist (update_counts)')
    ap.add_argument('--replace', action='store_true',
                    help="replace this script's earlier output from SOURCE; the old files move to " + SUPERSEDED_DIR)
    ap.add_argument('--checks-in-child', choices=('auto', 'yes', 'no'), default='auto')
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    rewrite_file(a.source, a.tol, a.check, a.energy_check, a.probe, a.seed, a.config, companion=a.companion,
                 replace=a.replace, checks_child=dict(auto=None, yes=True, no=False)[a.checks_in_child])


if __name__ == '__main__':
    main()
