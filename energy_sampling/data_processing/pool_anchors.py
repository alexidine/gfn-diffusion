"""Anchors of a crystal-search registry, re-described the way the trainer reads a prior row, for a pooled prior build.

    python -m data_processing.pool_anchors starts REG_DIR OUT.pt --mol MOL.pt --sg SG --key elj|uma [--window 10]
    python -m data_processing.pool_anchors export REG_DIR OUT.pt --mol MOL.pt --sg SG --key elj|uma --tsf F [--window 12]

REG_DIR holds a coordinator registry (mxtaltools.crystal_search.coordinator: registry.pt, coord.yaml). Both modes take
one row per basin within --window kT of the registry's lowest basin: the basin's lowest-energy state.

THE TRAINER'S READING OF A ROW. The trainer re-poses the stored molecule into its standard frame and builds the crystal
at handedness +1 from the row's 12 latent coordinates, ignoring the stored handedness
(energies/molecular_crystal.py::MolecularCrystal.init_blank_crystal_batch; analyze with std_orientation=False). A search
row carries handedness +1 or -1 and is scored with the molecule aligned to its standard axes under that handedness, so
a -1 row holds the mirror image of the molecule in its asymmetric unit. `Chart.to_trainer` re-describes each row with
the standard-frame molecule at +1:
  sg 2   the +1 molecule is the inversion mate at -centre, in the same cell: exact.
  sg 14  the +1 molecules of a -1 row are its inversion and glide mates, which no +1 description with an obtuse beta
         holds. Owner decision 2026-10-02 (NEHZOR, whose heavy atoms are planar to 0.06 A): treat the molecule as flat
         and EMBED -- fit the +1 molecule by a proper rotation onto the row's own (mirrored) molecule, in the same cell.
         Measured on 229 NEHZOR eLJ -1 basins: fit RMSD 0.52 A (two methylene hydrogens), energy change median
         +0.24 kT (p90 +0.55, max +0.87), atomwise RDF distance to the original median 0.09.
  Then the centre is brought into the asymmetric-unit box (sg 2: x in [0, 1/2] by the origin shift a/2; sg 14: y in
  [0, 1/4], through the screw mate for y in [1/2, 3/4]). A sg 14 row whose +1 centre has y mod 1/2 in (1/4, 1/2) has no
  description in the box and is left out (counted).

`starts` writes the rows as one collated MolCrystalData batch for run_search's init_sample_method 'data' (a polish
run): every row at +1 in the trainer's description, so the search's own scoring (std_orientation=True at +1) and the
trainer's agree. Embedded rows are not minima; the polish relaxes them.

`export` writes a prior file in train.py's layout: {'prior': batch, 'equalized_prior': the same batch,
'thermal_scaling_factor': --tsf, 'uma_energy_state': 2 for uma}, every row's energy attribute (--key) its energy READ
THE TRAINER'S WAY (raw eLJ, or the UMA lattice energy), and every row checked: trainer-read energy against the
registry's stored energy (embedded rows are re-scored, so they are reported apart), and a zero reduction penalty. Rows
failing a check are left out and counted; the export is refused when they exceed 2% of the rows.
"""
import argparse
import os
import time

import numpy as np
import torch

from mxtaltools.common.geometry_utils import rotmat2rotvec
from mxtaltools.crystal_building.utils import align_mol_batch_to_standard_axes, canonicalize_rotvec
from mxtaltools.crystal_search import coordinator as co
from mxtaltools.dataset_utils.utils import collate_data_list

EDGE = 0.9999  # mxtaltools crystal_opt_utils.CELL_EDGE: the builders clip a centre to [0, EDGE]
FACE_EPS = 1e-5  # fractional: a recomputed centre this close below a cell face is on the face
ANALYZE_KW = dict(cutoff=10, supercell_size=10, std_orientation=False)  # MolecularCrystal.analyze_crystal_batch
RED_TOL = 1e-6  # a row with a larger reduction penalty is outside the reduced-cell domain the trainer penalises leaving


def vec(*v):
    return torch.tensor(v, dtype=torch.double)


def proper_fit(M, X):
    """Per row, the rotation R (det +1) minimising |R m - x| and the RMSD it leaves; M, X [n, atoms, 3], centred."""
    H = torch.einsum('nai,naj->nij', M, X)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    return R, (torch.einsum('nij,naj->nai', R, M) - X).pow(2).sum(-1).mean(-1).sqrt()


class Chart:
    """Re-description of Z'=1 rows of space group 2 or 14 in the trainer's chart, and their normaliser images."""

    def __init__(self, mol_path, sg, key, device='cpu', predictor=None):
        from types import SimpleNamespace
        if sg not in (2, 14):
            raise NotImplementedError(f'space group {sg}: only 2 and 14 are implemented')
        self.cfg = SimpleNamespace(mol_path=mol_path, sg=sg, z_prime=1)
        self.sg, self.key, self.device, self.predictor = sg, key, device, predictor

    def setup(self, b):
        n, na = b.num_graphs, int(b.num_atoms.flatten()[0])
        heavy = (b.z.reshape(n, na)[:1] > 1).double()[..., None]
        self.cen = lambda X: X - (X * heavy).sum(1, keepdim=True) / heavy.sum(1)
        self.centre = lambda X: (X * heavy).sum(1) / heavy.sum(1)
        ref = align_mol_batch_to_standard_axes(b.clone(), handedness=torch.ones_like(b.aunit_handedness)[:, :1])
        self.M = self.cen(ref.pos.double().reshape(n, na, 3))[0]
        self.na = na

    def read(self, b, chunk=None):
        """[n] energy and [n] reduction penalty of each row, scored the way the trainer scores a row."""
        chunk = chunk or (100 if self.key == 'uma' else 256)
        kw = dict(predictor=self.predictor) if self.predictor is not None else {}
        e, r = [], []
        for lo in range(0, b.num_graphs, chunk):
            c = b.subsample_new_batch(torch.arange(lo, min(lo + chunk, b.num_graphs))).to(self.device)
            with torch.no_grad():
                o = c.analyze(['reduction_en', self.key], assign_outputs=False, **ANALYZE_KW, **kw)
            e.append(o[self.key].double().flatten().cpu())
            r.append(o['reduction_en'].double().flatten().cpu())
        return torch.cat(e), torch.cat(r)

    def describe(self, like, frac_atoms):
        """A +1 batch with the cell of `like` whose asymmetric unit is the molecule at the given fractional atom
        positions [n, atoms, 3]: centre wrapped into the cell, orientation by a proper fit of the standard-frame
        molecule, stored molecule = the standard-frame molecule. Returns (batch, fit RMSD)."""
        n = like.num_graphs
        out = like.clone()
        out.aunit_handedness = torch.ones_like(like.aunit_handedness)
        out.box_analysis()
        X = torch.einsum('nij,naj->nai', out.T_fc.double(), frac_atoms)
        c = torch.einsum('nij,nj->ni', out.T_cf.double(), self.centre(X))
        # a centre on a cell face recomputed from the atoms lands a float below it (-1e-7); the plain floor wrapped it
        # to the far side (y = 0 -> 0.9999, which sg 14 reads as outside its box: 1,398 of 24,461 NEHZOR eLJ rows)
        c = (c - torch.floor(c + FACE_EPS)).clamp(0.0, EDGE)
        R, rms = proper_fit(self.M[None].expand(n, -1, -1), self.cen(X))
        out.aunit_centroid = c.to(like.aunit_centroid.dtype)
        # scipy's conversion: rotmat2rotvec loses the axis for rotation angles near pi (2 of 3000 MIPCAS images read
        # back hundreds of eLJ units off, 2026-10-03)
        from scipy.spatial.transform import Rotation
        rv = torch.from_numpy(Rotation.from_matrix(R.numpy()).as_rotvec()).float()
        out.aunit_orientation = canonicalize_rotvec(rv).to(like.aunit_orientation.dtype)
        out.pos = self.M[None].expand(n, -1, -1).reshape(-1, 3).to(like.pos.dtype)
        # the stored pose must rebuild the atoms it was fitted to
        back = self.frac_atoms(out)
        dev = (torch.einsum('nij,naj->nai', out.T_fc.double(), back) - X - torch.einsum(
            'nij,nj->ni', out.T_fc.double(), c - torch.einsum('nij,nj->ni', out.T_cf.double(), self.centre(X)))[:, None]
               ).norm(dim=-1).amax(1)
        exact = rms < 1e-3  # an embedded row's fit leaves a real residual; its pose is checked by re-scoring
        if bool(exact.any()) and float(dev[exact].max()) > 5e-3:
            raise RuntimeError(f'describe: a stored pose is {float(dev[exact].max()):.3g} A off the atoms it was '
                               f'fitted to')
        return out, rms

    def frac_atoms(self, b):
        """Fractional atom positions [n, atoms, 3] of the asymmetric unit as the trainer builds it."""
        p = b.clone()
        p.pose_aunit(std_orientation=False)
        return torch.einsum('nij,naj->nai', b.T_cf.double(), p.pos.double().reshape(b.num_graphs, self.na, 3))

    def to_trainer(self, P, H):
        """Search-convention rows (cell parameters [n, 12], handedness [n, 1]) -> (+1 batch, embedded [n] bool,
        in_box [n] bool, fit RMSD [n]). Rows not in_box are still returned; the caller drops them."""
        b = collate_data_list(co.rebuild_crystals(self.cfg, P, H))
        for k in ('rdf', 'fingerprint', 'rdf_bins'):  # carried over from the molecule file: ~100 kB per row
            if k in b.keys():
                delattr(b, k)
        self.setup(b)
        posed = b.clone()
        posed.pose_aunit(std_orientation=True)
        f = torch.einsum('nij,naj->nai', b.T_cf.double(), posed.pos.double().reshape(b.num_graphs, self.na, 3))
        neg = H.flatten() < 0
        embedded = torch.zeros_like(neg)
        if self.sg == 2:
            f = torch.where(neg[:, None, None], -f, f)    # the inversion mate holds the +1 molecule
        else:
            embedded = neg                                 # the +1 molecule fitted onto the row's own molecule
        out, rms = self.describe(b, f)
        out, in_box = self.into_box(out)
        return out, embedded, in_box, rms

    def into_box(self, b):
        """Every centre into the asymmetric-unit box by a +1 operation (the same crystal). Returns (batch, in_box)."""
        c = b.aunit_centroid[:, :3].double().clone()
        if self.sg == 2:
            m = c[:, 0] > 0.5
            c[m, 0] -= 0.5
            b.aunit_centroid = c.to(b.aunit_centroid.dtype)
            return b, torch.ones(b.num_graphs, dtype=torch.bool)
        y = c[:, 1] % 0.5
        in_box = ~((y > 0.25 + 1e-6) & (y < 0.5 - 1e-6))
        hi = in_box & (c[:, 1] > 0.25 + 1e-6)              # y in [1/2, 3/4]: the screw mate is in the box
        if bool(hi.any()):
            f = self.frac_atoms(b)
            g = f * vec(-1.0, 1.0, -1.0) + vec(0.0, 0.5, 0.5)
            b, _ = self.describe(b, torch.where(hi[:, None, None], g, f))
        return b, in_box

    def images(self, b):
        """Euclidean-normaliser images of +1 rows that the +1 chart holds: (batch, source [m], image id [m]).
        sg 2: the y/z half shifts (4), plus the opposite x face for a centre on an x face (4 more).
        sg 14: the x/z half shifts (4) times {identity, the y half shift described through the screw mate} (8)."""
        f = self.frac_atoms(b)
        outs, src, iid = [], [], []
        n = b.num_graphs
        if self.sg == 2:
            for k, (dy, dz) in enumerate(((0, 0), (0, .5), (.5, 0), (.5, .5))):
                o, _ = self.describe(b, f + vec(0.0, dy, dz))
                o, _ = self.into_box(o)
                outs.append(o); src.append(torch.arange(n)); iid.append(torch.full((n,), k))
            x = b.aunit_centroid[:, 0]
            face = (x < 1e-6) | ((x - 0.5).abs() < 1e-6)
            if bool(face.any()):
                idx = torch.nonzero(face).flatten()
                for k in range(4):
                    o = outs[k].subsample_new_batch(idx)
                    c = o.aunit_centroid.clone()
                    c[:, 0] = torch.where(c[:, 0] < 0.25, torch.full_like(c[:, 0], 0.5), torch.zeros_like(c[:, 0]))
                    o.aunit_centroid = c
                    outs.append(o); src.append(idx); iid.append(torch.full((len(idx),), 4 + k))
        else:
            k = 0
            for via_screw in (False, True):
                g = f * vec(-1.0, 1.0, -1.0) + vec(0.0, 0.0, 0.5) if via_screw else f
                for dx, dz in ((0, 0), (0, .5), (.5, 0), (.5, .5)):
                    o, _ = self.describe(b, g + vec(dx, 0.0, dz))
                    outs.append(o); src.append(torch.arange(n)); iid.append(torch.full((n,), k)); k += 1
        allb = outs[0]
        for o in outs[1:]:
            allb = allb.append_batch(o)
        return allb, torch.cat(src), torch.cat(iid)


def basin_rows(reg_dir, window):
    """(registry, config, basin indices ascending in energy, params [n, 12], handedness [n, 1], energy [n] double)."""
    reg = torch.load(os.path.join(reg_dir, 'registry.pt'), weights_only=False)
    cfg = co.CampaignConfig.load(os.path.join(reg_dir, 'coord.yaml'))
    bE = torch.tensor(reg['basin_E']).double()
    keep = torch.nonzero(bE <= bE.min() + window * cfg.kT).flatten()
    keep = keep[torch.argsort(bE[keep])]
    P = torch.stack([reg['basin_params'][i] for i in keep]).float()
    H = torch.stack([reg['basin_hand'][i] for i in keep]).float().reshape(-1, 1)
    return reg, cfg, keep, P, H, bE[keep]


def _chart(a):
    dev, pred = 'cpu', None
    if a.key == 'uma' and a.mode == 'export':  # `starts` scores nothing (it loaded the model and failed without a path)
        from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
        dev = 'cuda'
        pred = init_uma_crystal_predictor(a.mlip_path, device=dev)
    return Chart(a.mol, a.sg, a.key, dev, pred)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['starts', 'export'])
    ap.add_argument('reg_dir')
    ap.add_argument('out')
    ap.add_argument('--mol', required=True, help='the molecule file the registry rows are built on')
    ap.add_argument('--sg', type=int, required=True)
    ap.add_argument('--key', required=True, choices=['elj', 'uma'])
    ap.add_argument('--window', type=float, default=10.0)
    ap.add_argument('--tsf', type=float, default=None, help='thermal_scaling_factor written into an export')
    ap.add_argument('--mlip_path', default=None)
    a = ap.parse_args(argv)
    t = time.time()
    reg, cfg, keep, P, H, E = basin_rows(a.reg_dir, a.window)
    ch = _chart(a)
    tb, emb, in_box, rms = ch.to_trainer(P, H)
    print(f'{a.reg_dir}: {len(keep)} basins within {a.window} kT of {float(E.min()):.4f} (kT {cfg.kT:.4g}); handedness '
          f'-1 rows {int((H < 0).sum())}, embedded {int(emb.sum())}, outside the box (left out) {int((~in_box).sum())};'
          f' fit RMSD max, exact rows {float(rms[~emb].max()) if bool((~emb).any()) else 0:.2e} A', flush=True)
    sel = torch.nonzero(in_box).flatten()
    tb, emb, E, keep = tb.subsample_new_batch(sel), emb[sel], E[sel], keep[sel]
    if a.mode == 'starts':
        torch.save(tb, a.out)  # a collated batch: run_search (init_sample_method data) splits it with batch_to_list
        print(f'wrote {tb.num_graphs} starts to {a.out} ({time.time() - t:.0f} s)')
        return
    e, red = ch.read(tb)
    d = (e - E).abs()
    tol = (0.02 if a.key == 'elj' else 0.15) * cfg.kT
    ex = ~emb
    print(f'trainer-read energy vs stored, exact rows: median |d| {float(d[ex].median()):.4f}, max {float(d[ex].max()):.4f} '
          f'({int((d[ex] > tol).sum())} over {tol:.3f}); embedded rows: median change {float((e - E)[emb].median()) / cfg.kT if bool(emb.any()) else 0:+.3f} kT; '
          f'reduction penalty max {float(red.max()):.3g}', flush=True)
    bad = (ex & (d > tol)) | (red > RED_TOL) | ~torch.isfinite(e)
    if bool(bad.any()):  # a few such rows are left out; many mean the conversion is wrong
        print(f'left out: {int((ex & (d > tol)).sum())} rows that do not read back as stored, {int((red > RED_TOL).sum())} '
              f'with a reduction penalty over {RED_TOL}, {int((~torch.isfinite(e)).sum())} non-finite', flush=True)
        if float(bad.double().mean()) > 0.02:
            raise SystemExit(f'export refused: {int(bad.sum())} of {len(bad)} rows fail the read-back checks')
        good = torch.nonzero(~bad).flatten()
        tb, emb, e, keep = tb.subsample_new_batch(good), emb[good], e[good], keep[good]
    setattr(tb, a.key, e.float())
    blob = {'prior': tb, 'equalized_prior': tb, 'thermal_scaling_factor': a.tsf if a.tsf is not None else 1,
            'basin': keep, 'embedded': emb, 'registry': os.path.abspath(a.reg_dir)}
    if a.key == 'uma':
        blob['uma_energy_state'] = 2
    torch.save(blob, a.out)
    print(f'wrote {tb.num_graphs} rows to {a.out} ({time.time() - t:.0f} s)')


if __name__ == '__main__':
    main()
