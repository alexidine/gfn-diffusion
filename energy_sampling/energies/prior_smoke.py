"""Committed smoke harness for the NON-LEARNED conformer pipeline.

    SMILES -> 2D graph -> TreeSpec -> prior draw -> DoF -> state x -> NeRF build
          -> Cartesians -> MMFF94 -> log reward

Every learned number downstream of that chain is meaningless if any link is wrong, and the
chain has been checked so far by throwaway scripts in a temp directory -- six result tables
in one session, none of them reproducible. This is the committed driver: one fixed molecule
set, one fixed set of thresholds, a non-zero exit on violation.

TWO RULES IT IS BUILT AROUND
----------------------------
**1. It must be able to FAIL.** A harness that prints numbers and exits 0 manufactures
reassurance. Every check asserts a threshold and names the molecule it fired on, and
``--inject`` re-introduces five real bug classes so the checks can be SHOWN to fire rather
than assumed to. Run ``--inject all`` after touching this file: a harness whose injections
stop firing has gone blind, and is worse than no harness.

**2. It must not be SELF-REFERENTIAL.** Scoring our own draws with our own energy cannot
detect a wrong energy. Three legs here are independent of the force field entirely:

* ``roundtrip_*`` -- draw DoF, build Cartesians, RE-MEASURE the internal coordinates from
  those Cartesians, compare to what was drawn. Pure geometry. Catches NeRF bugs, index
  scrambles, permutation errors.
* ``rigid_*`` -- shifting every dihedral about one bridge bond by a common delta must leave
  EVERY graph bond length and EVERY graph angle invariant, because it is a rigid motion of
  a fragment. This is the claim the torsion parameterisation rests on, and it is what
  separates a joint-draw group keyed on the CENTRAL BOND from one keyed on the parent atom.
* ``chirality`` -- RDKit re-perceives the stereocentre from the built Cartesians and must
  recover the SMILES' own CIP code. Every energy term is a function of interatomic
  distances or of even functions of the dihedral, so a mirrored molecule scores identically
  -- this is the only check in the file that can see a reflection.

and two are independent of OUR force field though not of RDKit:

* ``external_*`` -- ``rdkit.Chem.rdMolTransforms`` re-measures r/theta/phi off the built
  conformer through the ORIGINAL atom numbering. Our own ``measure`` shares an index
  convention with ``build``, so it cannot see an error that is consistent between them.
* ``mmff_*`` -- RDKit's own MMFF94, term by term, on the geometries WE built.

KNOWN BUG CLASSES THIS EXISTS TO CATCH
--------------------------------------
- a tree dihedral that is really an ANGLE drawn from a rotamer histogram (ethanol's
  O4-C0-H1 landed at 14.5 deg against theta0 = 108.6, and that one angle carried 251 of the
  252 kcal/mol of angle strain in the molecule)
- joint-draw groups keyed on the wrong frame (parent atom instead of central bond)
- vdW reading GetMMFFVdWParams' first two returns instead of the donor-acceptor rescaled
  third and fourth -- silent on every molecule that does not hydrogen bond
- a whole force-field term missing (electrostatics was absent, and it dominates on amides)
- spec numbering mixed up with mxtaltools' ``tree_*`` numbering
- ring closure silently violated
- mass piling on the box wall

WHAT IT IS NOT
--------------
It does not measure whether the fitted prior is any GOOD -- that is what
``energies/prior_diagnostics.py`` is for, and this calls INTO it (the ``prior_*`` and
``coverage_*`` checks) rather than reimplementing it. Where that module raises at the
requested level, the check is reported SKIPPED with the exception text verbatim. That the
only level which ships is the one those functions cannot measure is a finding, not a
nuisance to route around.

    python energies/prior_smoke.py                      # level torsion, mmff
    python energies/prior_smoke.py --level full
    python energies/prior_smoke.py --json out.json
    python energies/prior_smoke.py --inject all         # prove the checks can fire
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------------------
# THE MOLECULE SET. Fixed and committed: a moving molecule set makes a moving baseline.
# Each entry names the hazard it is here for.
# ---------------------------------------------------------------------------------------
MOLECULES = [
    ('propanol', 'CCCO',
     'acyclic chain; the molecule configs/conformer_propanol.yaml actually runs'),
    ('butanol', 'CCCCO',
     'size ladder 2/4; two rotatable bonds; the build_prior_states.py default'),
    ('hexanol', 'CCCCCCO',
     'size ladder 3/4; four rotatable bonds, chain long enough to self-contact'),
    ('nma', 'CC(=O)NC',
     'polar AMIDE: electrostatics dominates every other term, sp2 out-of-plane, and the '
     'H-bond donor/acceptor vdW rescaling that GetMMFFVdWParams hides in returns 3 and 4'),
    ('glycerol', 'OCC(O)CO',
     'our known coverage pathology; strong intramolecular H-bond; three hydroxyls'),
    ('butyronitrile', 'CCCC#N',
     "LINEAR centre: exercises the held-singular-DoF chart-dimension path (data_ndim is no "
     "longer 3N-6) and MMFF's separate 143.9325 ka (1 + cos theta) angle form"),
    ('ethylcyclohexane', 'CCC1CCCCC1',
     'saturated RING: one closure bond the tree cannot represent, pucker held, and '
     'substituent hydrogens that must follow the ring instead of rotating against it'),
    ('ethylnaphthalene', 'CCc1ccc2ccccc2c1',
     'FUSED aromatic rings: two closure bonds, aromatic rigidity, out-of-plane terms'),
    ('butan-2-ol-R', 'C[C@H](O)CC',
     'chiral pair 1/2: the NeRF / standard-orientation mirror hazard'),
    ('butan-2-ol-S', 'C[C@@H](O)CC',
     'chiral pair 2/2: identical machinery must produce the OPPOSITE CIP code'),
    ('ala-dipeptide', 'CC(=O)NC(C)C(=O)NC',
     'size ladder 4/4: two amides, four rotatable bonds; the peptide case ff_from_graph '
     'raises on outright'),
]

# ---------------------------------------------------------------------------------------
# THRESHOLDS. Each carries the argument for its value; a threshold without one is a number
# somebody liked. The "measured" figures are the observed worst case on the clean tree --
# they are not the bar, they are the DISTANCE from the bar.
# ---------------------------------------------------------------------------------------
TOL = {
    # --- energy-free geometry ---------------------------------------------------------
    # build and measure are exact inverses in exact arithmetic, so the only floor is
    # float64 round-off. r and theta land under 1e-15. phi's worst case is the near-linear
    # nitrile frame, where the dihedral is ill-conditioned by construction, at ~2e-13. A
    # bar of 1e-7 is six orders above that: nothing but an index convention, a permutation
    # or a units error reaches it, and each of those lands at 1e-2 or worse.
    'roundtrip_r': 1e-9,
    'roundtrip_theta': 1e-9,
    'roundtrip_phi': 1e-7,
    # identical numbers for the external re-measure: rdMolTransforms does the same
    # arithmetic, just addressed through the ORIGINAL atom numbering instead of slots.
    'external_r': 1e-9,
    'external_theta': 1e-9,
    'external_phi': 1e-7,
    # A common delta on every dihedral about one bridge bond is a rigid motion of the
    # moving fragment, so bond lengths and graph angles are invariant EXACTLY; round-off
    # only, measured under 1e-14. If a torsion column drove only the first of its bond's
    # dihedrals the fragment would shear, and graph angles at the axis would move by tens
    # of degrees -- 1e-1 rad, seven orders above this bar (--inject shear-torsion).
    'rigid_bond': 1e-8,
    'rigid_angle': 1e-8,

    # --- external energy cross-check ---------------------------------------------------
    # Angle / StretchBend / Oop carry RDKit ACCESSOR ROUNDING, not a functional-form error:
    # theta0 comes back at three decimals and koop at two or three, and the residual scales
    # with |theta - theta0|. test_mmff_matches_rdkit uses 3e-4 at a mildly perturbed
    # minimum; these geometries are drawn further out, so 5e-3 kcal/mol. The other four
    # terms carry no such rounding and land at ~1e-12, hence 1e-6 for them.
    #   CONSEQUENCE, and it is a limitation rather than a defect: because the residual
    # grows with the deviation, these three bars are only meaningful on geometries in the
    # thermal range. On a draw whose local geometry has been destroyed they fire for the
    # RIGHT reason -- RDKit's rounded theta0 against ours, amplified -- not because the
    # conversion is wrong. Observed: 2.4e-4 clean at `full`, 1.0e-2 under
    # --inject sibling-independent.
    'mmff_angle': 5e-3,
    'mmff_stretch_bend': 5e-3,
    'mmff_oop': 5e-3,
    'mmff_bond': 1e-6,
    'mmff_torsion': 1e-6,
    'mmff_lj': 1e-6,
    'mmff_electrostatic': 1e-6,
    'mmff_total': 2e-2,          # the three rounded terms, added

    # --- physical scale ----------------------------------------------------------------
    # Ring closure: the closure bond is not a tree DoF, it is whatever the ring internals
    # imply, so it must be MEASURED rather than assumed. The scale at which a deviation
    # stops being distinguishable from ordinary bond vibration is the bond's own thermal
    # width; 3 of those is where the ring is visibly open (the same bar
    # sample_prior_states already prints against). Unit-free, so it does not depend on the
    # molecule's bond stiffnesses.
    'closure_sigma': 3.0,
    # Box wall. A clipped row sits exactly ON the wall, where the wall is zero, so mass
    # piling there is invisible in the energy. The thermal width in state units is
    # sigma_r/delta_r_max ~ 0.13 and sigma_theta/delta_theta_max ~ 0.12, so the box edge is
    # ~8 sigma out and the true rate is ~1e-15. 1% is therefore not a tolerance on a real
    # rate; it is a tripwire on the box or the scale being wrong.
    'clip_frac': 0.01,
    # Per-term energy NORMALISED PER TERM, in kT. For a harmonic term driven at its exact
    # thermal width, equipartition fixes the mean at 0.5 kT per DoF. The force field scores
    # the REDUNDANT graph set, which is larger than the DoF set, so per graph term it sits
    # near or below that -- measured 0.63 kT/angle at `full`, 0.07 at `torsion`. 2.0 kT is
    # 4x equipartition: the smallest bar that cannot fire on a correct thermal draw. For
    # scale, the ethanol improper bug put 251 kcal/mol into 19 angles = 13 kT/angle, 6x
    # over the bar. The non-harmonic terms have NO equipartition prediction, so their bars
    # are magnitude bounds from the measured spread with margin, and are correspondingly
    # weaker evidence; electrostatics genuinely reaches -0.76 kT/pair on the amide.
    'kt_per_bond': 2.0,
    'kt_per_angle': 2.0,
    'kt_per_stretch_bend': 1.0,
    'kt_per_oop': 1.0,
    'kt_per_lj': 2.0,
    'kt_per_torsion': 2.0,
    'kt_per_electrostatic': 5.0,
    # T_eff/T from the bond term over FREE, NON-RING bonds: 2 <E_bond> / (n kT). For a
    # harmonic term drawn at sqrt(kT/2k) this is exactly 1 by equipartition -- the one
    # place in the file where theory predicts a number rather than bounding it. Two things
    # push it off: MMFF's quartic stretch correction (about +0.6%) and the offset between
    # the embedded reference conformer's bond lengths and MMFF's TYPED r0, since the
    # reference is not the MMFF minimum. Measured 0.97-1.06 across the set. +/-30% admits
    # those with room and still catches a factor-2 error in the width or a lost factor of T.
    'T_eff_lo': 0.70,
    'T_eff_hi': 1.40,
    # Worst nonbonded overlap as a FRACTION of the pair's own sigma, in EXCESS of the
    # reference conformer's own value, at the 10th percentile of draws.
    #   Three choices, each load-bearing. (a) As a fraction of sigma, so it is comparable
    # across molecules. (b) In EXCESS of the reference: 1-4 pairs are intrinsically inside
    # sigma and the MMFF-optimised reference conformer already sits at 0.13-0.25, so an
    # absolute bar measures the molecule rather than the draw. The baseline is a relaxed
    # GEOMETRY, not our energy, so this stays outside the self-referential trap. (c) The
    # 10th percentile, not the median: a legitimate torsional excursion raises the TAIL --
    # a floppy chain folded onto itself is a real conformer -- while destroyed local
    # geometry raises the FLOOR, because siblings overlap whatever the global torsion does.
    #   Measured across the set: clean 0.00-0.11 (worst is hexanol), improper-scramble
    # reaches 0.28 and sibling-scramble 0.36. 0.20 sits at 1.8x the worst clean value.
    'clash_excess_p10': 0.20,
    # One batch of n against two of n/2. _batch is a getter that MUTATES (it fills the
    # tree/ff caches as a side effect) and a stale tree is a silent wrong answer, not a
    # crash. Relative, because energies here span 1e0 to 1e5.
    'batch_invariance_rel': 1e-9,

    # --- prior_diagnostics ---------------------------------------------------------------
    # An ESS FRACTION is bounded above by 1 by construction: it is sum(w)^2 / (n sum(w^2)),
    # which is Cauchy-Schwarz. Above 1 means the object being averaged is not a set of
    # importance weights. This is the only HARD bound the prior leg supplies -- see the
    # note in _prior_leg for why eta and n_missed are reported rather than asserted.
    'prior_ess_max': 1.0,
}

# Known bug classes, re-introduced on demand. They act on the DoF VECTOR (or on the built
# Cartesians), never on the state draw, for two reasons: a DoF-space injection reproduces
# what the bug actually did rather than an imitation of it, and it is level-independent --
# at `torsion` the state has one column per rotatable bond and cannot express an improper
# or a sibling offset at all, so a draw-level injection there would be a silent no-op.
# The right-hand column names the check that is supposed to catch each one.
INJECTIONS = {
    'improper-uniform': (
        'kt_per_angle',
        'overwrite the IMPROPER phi rows with a uniform rotamer draw. An improper tree '
        'dihedral IS an angle at the parent -- ethanol row 1 is precisely O4-C0-H1 -- and '
        'a histogram draw put it at a median 14.5 deg against theta0 = 108.6, where that '
        'one angle carried 251 of the molecule\'s 252 kcal/mol of angle strain'),
    'sibling-independent': (
        'kt_per_angle',
        'draw every non-leader member of a torsion group independently instead of at the '
        'leader\'s displacement. Even from perfect marginals this puts two substituents in '
        'the same place for about a third of sibling pairs'),
    'ring-float': (
        'closure_sigma',
        'let ring-locked DoF float at their FULL thermal width. Closure is nonlinear, so '
        'independent per-DoF perturbations accumulate around the loop with a lever arm'),
    'shear-torsion': (
        'rigid_bond / rigid_angle',
        'rotate only the FIRST dihedral of a bond instead of all of them. This is the bug '
        'the collective state -> DoF map was written to fix: it shears the moving fragment '
        'rather than rotating it'),
    'perm-scramble': (
        'external_r / external_theta / external_phi',
        'address the built Cartesians through the IDENTITY instead of spec.perm when '
        'handing them to RDKit -- the spec-numbering vs mxtaltools tree_* numbering bug '
        'class, which permutes rather than raising'),
    'mirror': (
        'chirality',
        'negate EVERY dihedral in the DoF vector, which builds the exact mirror image. '
        'This is the invisible form of the bug -- a sign convention flipped consistently '
        'in build AND measure -- so the round trip agrees with itself, RDKit re-measures '
        'the same negated values, and every energy term is even under it (the torsion '
        'phases are 0 and pi, so cos is even; oop is chi squared; everything else is a '
        'function of distance). Only the CIP check can see it'),
}


# =======================================================================================
# ledger
# =======================================================================================

class Result:
    __slots__ = ('name', 'mol', 'level', 'status', 'value', 'tol', 'cmp', 'reason', 'units')

    def __init__(self, name, mol, level, status, value=None, tol=None, cmp='<=',
                 reason='', units=''):
        self.name, self.mol, self.level = name, mol, level
        self.status, self.value, self.tol, self.cmp = status, value, tol, cmp
        self.reason, self.units = reason, units

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


class Ledger:
    """Every check that ran, skipped or fired. Nothing here is allowed to be silent."""

    def __init__(self):
        self.rows = []

    def _add(self, r):
        self.rows.append(r)
        return r

    def check(self, name, mol, level, value, tol, cmp='<=', units=''):
        """Assert ``value cmp tol``. A non-finite value is a FAILURE, never a pass."""
        v = float(value)
        if not math.isfinite(v):
            ok = False
        elif cmp == '<=':
            ok = v <= tol
        elif cmp == '>=':
            ok = v >= tol
        elif cmp == '==':
            ok = v == tol
        else:
            raise ValueError(cmp)
        return self._add(Result(name, mol, level, 'pass' if ok else 'FAIL', v, tol, cmp,
                                units=units))

    def band(self, name, mol, level, value, lo, hi, units=''):
        v = float(value)
        ok = math.isfinite(v) and lo <= v <= hi
        return self._add(Result(name, mol, level, 'pass' if ok else 'FAIL', v, [lo, hi],
                                'in', units=units))

    def skip(self, name, mol, level, reason):
        """A check that did not run. Loud by construction: skips print in their own block."""
        return self._add(Result(name, mol, level, 'skip', reason=reason))

    def note(self, name, mol, level, value, units=''):
        """Reported, not asserted. Counted apart so it cannot be mistaken for a pass."""
        return self._add(Result(name, mol, level, 'note', value=value, units=units))

    @property
    def failures(self):
        return [r for r in self.rows if r.status == 'FAIL']

    @property
    def skips(self):
        return [r for r in self.rows if r.status == 'skip']


# =======================================================================================
# geometry helpers
# =======================================================================================

def _np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def ring_locked(en):
    """DoF rows whose value is fixed by ring closure, in SPEC numbering.

    Closure is a hard constraint the tree cannot express, so a product of independent draws
    violates it by construction. The rule is structural, not fitted:

    * an r row is locked when BOTH its atoms are in a ring
    * a theta row is locked when all three are
    * a phi row is locked when its CENTRAL BOND (b, c) is a ring bond -- which also catches
      the substituent hydrogens on a ring carbon, whose group must follow the ring rather
      than rotate against it

    A bond from a ring atom to a substituent (ethylcyclohexane's exocyclic C-C) is
    deliberately NOT locked: rotating about it moves the ring rigidly and closure is
    untouched. Locking it would leave ethylcyclohexane with zero sampled dimensions at
    `torsion`, which is its only dimension.
    """
    bi = np.asarray(en.spec.bond_index)
    ai = np.asarray(en.spec.angle_index)
    ti = np.asarray(en.spec.torsion_index)
    inr = en.atom_in_ring
    rows = set()
    for j in range(en.n_r):
        if inr[bi[j, 0]] and inr[bi[j, 1]]:
            rows.add(j)
    for j in range(en.n_th):
        if inr[ai[j, 0]] and inr[ai[j, 1]] and inr[ai[j, 2]]:
            rows.add(en.n_r + j)
    for j in range(en.n_ph):
        if inr[ti[j, 1]] and inr[ti[j, 2]]:
            rows.add(en.n_r + en.n_th + j)
    return rows


def column_rows(en):
    """Per STATE COLUMN, the DoF rows it drives, in SPEC numbering. Valid at every level.

    At a selection level this is one row per column. At `torsion` a column is COLLECTIVE:
    it drives every dihedral whose central bond is that rotatable bond, which is exactly
    what makes the resulting motion a rotation rather than a shear.
    """
    M = _np(en._M)
    driven = _np(en._driven_idx)
    return [driven[M[:, c] != 0].astype(int).tolist() for c in range(en.data_ndim)]


def draw_states(en, n, rng, ring_jitter=0.1):
    """``[n, d]`` states on [-1, 1] drawn PHYSICALLY, plus a stats dict.

    This is not a fitted prior. It is the thermal / rotameric draw the pipeline's own
    primitives already define, so the harness has a sample source at every level and for
    every molecule without needing a fit to exist:

    * r, theta      N(reference, sqrt(kT / 2k)) -- the EXACT Boltzmann marginal of a
                    harmonic term, with k taken from the force field's own constant
    * improper phi  N(reference, improper_phi_sigma). An improper dihedral IS an angle at
                    the parent, so it rattles; it must NOT take a rotamer draw
    * proper phi    one shared U(-pi, pi) per torsion GROUP (all dihedrals about one central
                    bond), each member jittered by sibling_jitter_sigma. In STATE units a
                    group is simply a shared value, because the state is already a delta
                    from the reference -- which is the same statement as the collective
                    column at `torsion`
    * ring-locked   all of the above at `ring_jitter` of their width

    phi columns WRAP (period 2). The non-periodic columns are CLIPPED to the box and the
    rate is returned, because a high rate means the box is too narrow for the physical
    distribution, and that is information rather than noise.
    """
    T = float(en.temperature)
    s_r, s_th = en.thermal_rtheta_sigma(T)
    s_imp = en.improper_phi_sigma(T)
    groups = en.torsion_groups()
    g_sigma = en.sibling_jitter_sigma(groups, T)
    n0 = en.n_r + en.n_th
    locked = ring_locked(en)
    grp_of = {j: gi for gi, rows in enumerate(groups) for j in rows}
    cols = column_rows(en)
    blk = np.asarray(en._free_block)
    scale = _np(en._free_scale)

    grp_locked = [all((n0 + j) in locked for j in rows) for rows in groups]
    shared = [np.zeros(n) if grp_locked[gi] else rng.uniform(-np.pi, np.pi, n)
              for gi in range(len(groups))]

    x = np.zeros((n, en.data_ndim))
    for c in range(en.data_ndim):
        rows = cols[c]
        row = rows[0]
        held = all(k in locked for k in rows)
        f = ring_jitter if held else 1.0
        if blk[c] == 0:                                        # r
            x[:, c] = rng.normal(0.0, f * s_r[row] / scale[c], n)
        elif blk[c] == 1:                                      # theta
            x[:, c] = rng.normal(0.0, f * s_th[row - en.n_r] / scale[c], n)
        else:                                                  # phi
            gids = {grp_of[k - n0] for k in rows if (k - n0) in grp_of}
            if not gids:                                       # improper / ungrouped
                x[:, c] = rng.normal(0.0, f * s_imp / scale[c], n)
            else:
                gi = sorted(gids)[0]
                jit = rng.normal(0.0, f * g_sigma[gi], n)
                x[:, c] = (shared[gi] + jit) / scale[c]

    per = blk == 2
    x[:, per] = (x[:, per] + 1.0) % 2.0 - 1.0                  # phi wraps, it does not clip
    lin = ~per
    clip = float((np.abs(x[:, lin]) > 1.0).mean()) if lin.any() else None
    x[:, lin] = np.clip(x[:, lin], -1.0, 1.0)
    return torch.as_tensor(x, dtype=en.dtype, device=en.device), {'clip_frac': clip}


def inject_dof(en, r, th, ph, rng, inject):
    """Re-introduce a known bug in the DoF VECTOR, after ``dof_from_state`` and before
    ``build``. Returns the modified ``(r, theta, phi)``.

    Injecting here rather than in the state draw is deliberate. At ``level='torsion'`` the
    state carries one column per rotatable bond, and an improper dihedral or a sibling
    offset is not addressable from it at all -- a draw-level injection would be a silent
    no-op at exactly the level that ships, which is the failure mode this whole file
    exists to avoid. The DoF vector is where the historical bugs actually lived: they were
    bugs in ``sample_prior_states``, which emits DoF.
    """
    if not inject:
        return r, th, ph
    n, n0 = r.shape[0], en.n_r + en.n_th
    r, th, ph = r.clone(), th.clone(), ph.clone()
    U = lambda: torch.as_tensor(rng.uniform(-np.pi, np.pi, n), dtype=ph.dtype,
                                device=ph.device)
    if 'improper-uniform' in inject:
        for j in en.improper_phi_rows():
            ph[:, j] = U()
    if 'sibling-independent' in inject:
        for rows in en.torsion_groups():
            for j in rows[1:]:                       # leader keeps its value; the rest drift
                ph[:, j] = U()
    if 'mirror' in inject:
        ph = -ph                                     # a self-consistent global reflection
    if 'ring-float' in inject:
        T = float(en.temperature)
        s_r, s_th = en.thermal_rtheta_sigma(T)
        s_imp = en.improper_phi_sigma(T)
        g = lambda s: torch.as_tensor(rng.normal(0.0, s, n), dtype=r.dtype, device=r.device)
        for k in ring_locked(en):
            if k < en.n_r:
                r[:, k] = r[:, k] + g(s_r[k])
            elif k < n0:
                th[:, k - en.n_r] = th[:, k - en.n_r] + g(s_th[k - en.n_r])
            else:
                ph[:, k - n0] = ph[:, k - n0] + g(s_imp)
    return r, th, ph


def to_rdkit_mol(en, pos_slot, inject=()):
    """A copy of the molecule whose conformer holds ``pos_slot``, in ORIGINAL numbering.

    ``pos_slot[i]`` is placement slot i and ``spec.perm[i]`` is that slot's RDKit atom
    index. Getting this backwards is the "spec numbering vs mxtaltools tree_* numbering"
    bug class, so every external check is addressed through it deliberately.
    """
    from rdkit import Chem
    perm = np.asarray(en.spec.perm)
    if 'perm-scramble' in inject:
        perm = np.arange(len(perm))                # the identity: slot i <- atom i
    mol = Chem.Mol(en.mol)
    conf = mol.GetConformer()
    pos = np.asarray(pos_slot, dtype=np.float64)
    for slot in range(en.spec.n_atoms):
        conf.SetAtomPosition(int(perm[slot]), pos[slot].tolist())
    return mol


def _worst_overlap(pos, ff, n):
    """Per molecule, the deepest nonbonded overlap as a FRACTION of that pair's sigma."""
    d = torch.linalg.norm(pos[ff.pair_index[:, 0]] - pos[ff.pair_index[:, 1]], dim=-1)
    frac = torch.relu(ff.sigma - d) / ff.sigma
    return torch.zeros(n, dtype=pos.dtype, device=pos.device).scatter_reduce(
        0, ff.pair_batch, frac, reduce='amax', include_self=True)


def _batch1_dof(en):
    """``(tree, r, theta, phi)`` for the REFERENCE conformer, as ``build`` wants them."""
    tree, _ = en._batch(1)
    r, th, ph = en.dof_from_state(torch.zeros(1, en.data_ndim, dtype=en.dtype,
                                              device=en.device))
    return tree, r.reshape(-1), th.reshape(-1), ph.reshape(-1)


def graph_geometry(pos, ff):
    """Every graph bond length and every graph angle, from Cartesians alone."""
    b = torch.linalg.norm(pos[ff.bond_index[:, 0]] - pos[ff.bond_index[:, 1]], dim=-1)
    u = pos[ff.angle_index[:, 0]] - pos[ff.angle_index[:, 1]]
    v = pos[ff.angle_index[:, 2]] - pos[ff.angle_index[:, 1]]
    a = torch.atan2(torch.linalg.cross(u, v).norm(dim=-1), (u * v).sum(-1))
    return b, a


# =======================================================================================
# per-molecule checks
# =======================================================================================

RDKIT_TERMS = {'bond': 'Bond', 'angle': 'Angle', 'stretch_bend': 'StretchBend',
               'oop': 'Oop', 'torsion': 'Torsion', 'lj': 'VdW', 'electrostatic': 'Ele'}
ALL_RDKIT = ('Bond', 'Angle', 'StretchBend', 'Oop', 'Torsion', 'VdW', 'Ele')


def run_molecule(name, smiles, level, ff_choice, n, seed, n_external, led, prior,
                 prior_n, inject=()):
    """Every check for one molecule at one level. Returns a per-molecule summary dict."""
    from mxtaltools.conformers.builder import measure, closure_length
    from mxtaltools.conformers.energy import intramolecular_energy
    from energies.conformer_torsions import ConformerTorsions

    t0 = time.time()
    try:
        en = ConformerTorsions(smiles=smiles, level=level, force_field=ff_choice)
    except ValueError as e:
        msg = str(e)
        # A LEVEL may legitimately have nothing to sample: `torsion` refuses a molecule with
        # no rotatable bond, and any level refuses one with no free DoF. That is a design
        # refusal stated in ConformerTorsions.__init__, not a pipeline fault, so it is a
        # loud skip. Any OTHER ValueError propagates and fails the run.
        if 'no rotatable bonds' in msg or 'no free degrees of freedom' in msg:
            led.skip('MOLECULE', name, level, f'construction refused: {msg}')
            return {'name': name, 'smiles': smiles, 'skipped': msg}
        raise

    from mxtaltools.conformers.builder import build

    T = float(en.temperature)
    rng = np.random.default_rng(seed)
    x, dstats = draw_states(en, n, rng)

    # state -> DoF -> Cartesians, with the injection sitting exactly where the historical
    # bugs sat: in the DoF vector that sample_prior_states emits.
    r, th, ph = en.dof_from_state(x)
    r, th, ph = inject_dof(en, r, th, ph, np.random.default_rng(seed + 11), inject)
    tree, ff = en._batch(n)
    pos = build(tree, r.reshape(-1), th.reshape(-1), ph.reshape(-1))
    e_state = en.energy(x)                       # the pure state path: no injection reaches it

    led.check('finite_positions', name, level, int((~torch.isfinite(pos)).sum()), 0, '==')
    led.check('finite_energy', name, level, int((~torch.isfinite(e_state)).sum()), 0, '==')
    led.check('finite_e_ref', name, level, 0 if math.isfinite(en.e_ref) else 1, 0, '==')

    # ------------------------------------------ LEG 1: energy-free structural round trip
    rm, thm, phm = measure(tree, pos)
    dphi = (phm - ph.reshape(-1) + np.pi) % (2 * np.pi) - np.pi
    led.check('roundtrip_r', name, level, (rm - r.reshape(-1)).abs().max().item(),
              TOL['roundtrip_r'], units='A')
    led.check('roundtrip_theta', name, level, (thm - th.reshape(-1)).abs().max().item(),
              TOL['roundtrip_theta'], units='rad')
    led.check('roundtrip_phi', name, level, dphi.abs().max().item(),
              TOL['roundtrip_phi'], units='rad')

    _rigid(en, name, level, led, seed, inject)

    # ------------------------------------------------- LEG 2: external cross-checks
    _external_geometry(en, name, level, led, r, th, ph, pos, n_external, inject)
    _chirality(en, name, level, led, pos, n_external, inject)
    if ff_choice == 'mmff':
        _external_mmff(en, name, level, led, pos, n_external, inject)
    else:
        led.skip('mmff_* (7 terms + total)', name, level,
                 f"force_field={ff_choice!r} is not MMFF94, so RDKit's MMFF is not a "
                 f"reference for it and the whole external ENERGY leg is unavailable. The "
                 f"external GEOMETRY checks above still ran, and they are the ones that "
                 f"do not depend on the force field at all")

    # --------------------------------------------------------- LEG 3: physical scale
    total, comp = intramolecular_energy(tree, pos, ff, components=True)
    counts = {'bond': len(ff.bond_index) // n, 'angle': len(ff.angle_index) // n,
              'lj': len(ff.pair_index) // n, 'electrostatic': len(ff.pair_index) // n,
              'torsion': (len(ff.torsion_index) // n) if ff.torsion_index is not None else 0,
              'stretch_bend': (len(ff.sb_index) // n) if ff.sb_index is not None else 0,
              'oop': (len(ff.oop_index) // n) if ff.oop_index is not None else 0}
    if ff.ele_qq is None or not ff.ele_qq.numel():
        counts['electrostatic'] = 0
    per_term = {}
    for k in ('bond', 'angle', 'stretch_bend', 'oop', 'lj', 'torsion', 'electrostatic'):
        if counts[k] == 0:
            led.skip(f'kt_per_{k}', name, level,
                     f"the {ff_choice!r} force field carries no {k} term for this molecule "
                     f"(term count 0), so there is nothing to normalise. Under 'reference' "
                     f"that is true of torsion, stretch_bend, oop and electrostatic on "
                     f"EVERY molecule")
            continue
        val = float(comp[k].median()) / counts[k] / T
        per_term[k] = val
        led.check(f'kt_per_{k}', name, level, abs(val), TOL[f'kt_per_{k}'], units='kT/term')

    s_r, _ = en.thermal_rtheta_sigma(T)
    if ff.closure_index.numel():
        cl = closure_length(tree, pos)
        err = (cl - ff.closure_r0).abs().reshape(n, -1).max(1).values
        led.check('closure_sigma', name, level,
                  float(err.median()) / float(np.mean(s_r)), TOL['closure_sigma'],
                  units='bond-sigma')
        led.note('closure_err_A', name, level, float(err.median()), units='A')
    else:
        led.skip('closure_sigma', name, level,
                 'acyclic: the spanning tree covers every bond, so there is no closure '
                 'bond that could be violated')

    if dstats['clip_frac'] is None:
        led.skip('clip_frac', name, level,
                 f"level {level!r} frees no r or theta column, so no state block is "
                 f"non-periodic: phi wraps and there is no box to pile on. "
                 f"ConformerTorsions skips the bounding-energy term for the same reason, "
                 f"which is what keeps this level bitwise identical to the pre-ladder code")
    else:
        led.check('clip_frac', name, level, dstats['clip_frac'], TOL['clip_frac'])

    _t_eff(en, name, level, led, ff, pos, n, T, ff_choice)

    # worst nonbonded overlap, in EXCESS of the reference conformer's own
    worst = _worst_overlap(pos, ff, n)
    _, ff1 = en._batch(1)
    ref_pos = build(*(_batch1_dof(en)))
    ref_worst = float(_worst_overlap(ref_pos, ff1, 1)[0])
    led.check('clash_excess_p10', name, level,
              float(torch.quantile(worst, 0.10)) - ref_worst, TOL['clash_excess_p10'],
              units='fraction of sigma above the reference conformer')
    led.note('clash_p10', name, level, float(torch.quantile(worst, 0.10)))
    led.note('clash_median', name, level, float(worst.median()))
    led.note('clash_reference', name, level, ref_worst)

    half = n // 2
    e_split = torch.cat([en.energy(x[:half]), en.energy(x[half:])])
    rel = ((e_split - e_state).abs() / e_state.abs().clamp_min(1.0)).max().item()
    led.check('batch_invariance', name, level, rel, TOL['batch_invariance_rel'],
              units='relative')

    # ------------------------------------------------- LEG 4: prior_diagnostics.py
    _prior_leg(en, name, level, led, prior, prior_n, seed)

    return {'name': name, 'smiles': smiles, 'd': int(en.data_ndim),
            'n_atoms': int(en.spec.n_atoms), 'n_rotatable': len(en.rotatable),
            'n_linear_angle': int(en.angle_is_linear.sum()),
            'n_linear_frame': int(en.torsion_frame_is_linear.sum()),
            'collective': bool(en.collective), 'e_ref': float(en.e_ref),
            # the POTENTIAL over T of the built geometry (no change of measure), so it
            # reflects any injection; e_state carries log J and does not
            'u_over_T_median': float(total.median()) / T,
            'energy_median': float(e_state.median()), 'per_term_kT': per_term,
            'seconds': time.time() - t0}


def _rigid(en, name, level, led, seed, inject):
    """Rotating one bridge bond must leave every graph bond and graph angle invariant.

    Energy-free, and it is the check that distinguishes a torsion group keyed on the
    CENTRAL BOND from one keyed on the parent atom: a common displacement applied to
    dihedrals measured about DIFFERENT axes is not a rotation of anything.
    """
    from mxtaltools.conformers.builder import build
    if not en.rotatable:
        for k in ('rigid_bond', 'rigid_angle'):
            led.skip(k, name, level,
                     'no rotatable (bridge, heavy-fragment) bond on this molecule, so '
                     'there is no rigid rotation to test')
        return
    nb = 8
    rng = np.random.default_rng(seed + 7)
    x0, _ = draw_states(en, nb, rng)
    cols = column_rows(en)
    mask = _np(en.mask)
    n0 = en.n_r + en.n_th
    scale = _np(en._free_scale)
    tree, ff = en._batch(nb)
    b0, a0 = graph_geometry(en.build_positions(x0), ff)

    worst_b = worst_a = 0.0
    tested = 0
    for jb in range(len(en.rotatable)):
        rows = {n0 + int(k) for k in np.flatnonzero(mask[:, jb] != 0)}
        drive = [c for c in range(en.data_ndim) if rows & set(cols[c])]
        driven = set().union(*[set(cols[c]) for c in drive]) if drive else set()
        if not rows.issubset(driven):
            continue                       # this bond is not fully driven at this level
        delta = float(rng.uniform(0.4, 2.0))
        if 'shear-torsion' in inject:
            # drive only the FIRST dihedral of the bond, in DoF space so it works at every
            # level: the fragment shears instead of rotating
            r_, th_, ph_ = en.dof_from_state(x0)
            ph_ = ph_.clone()
            ph_[:, sorted(rows)[0] - n0] += delta
            p1 = build(tree, r_.reshape(-1), th_.reshape(-1), ph_.reshape(-1))
        else:
            x1 = x0.clone()
            for c in drive:
                v = _np(x1[:, c]) + delta / scale[c]
                x1[:, c] = torch.as_tensor((v + 1.0) % 2.0 - 1.0, dtype=x1.dtype)
            p1 = en.build_positions(x1)
        b1, a1 = graph_geometry(p1, ff)
        worst_b = max(worst_b, (b1 - b0).abs().max().item())
        worst_a = max(worst_a, (a1 - a0).abs().max().item())
        tested += 1
    if not tested:
        for k in ('rigid_bond', 'rigid_angle'):
            led.skip(k, name, level,
                     'no rotatable bond had ALL of its dihedrals driven by state columns '
                     'at this level, so no shift here is a whole rotation')
        return
    led.note('rigid_bonds_tested', name, level, tested)
    led.check('rigid_bond', name, level, worst_b, TOL['rigid_bond'], units='A')
    led.check('rigid_angle', name, level, worst_a, TOL['rigid_angle'], units='rad')


def _external_geometry(en, name, level, led, r, th, ph, pos, n_sub, inject):
    """rdMolTransforms re-measures r/theta/phi through the ORIGINAL atom numbering."""
    from rdkit.Chem import rdMolTransforms as rdmt
    perm = np.asarray(en.spec.perm)
    bi = np.asarray(en.spec.bond_index)
    ai = np.asarray(en.spec.angle_index)
    ti = np.asarray(en.spec.torsion_index)
    N = en.spec.n_atoms
    pos_np = _np(pos).reshape(-1, N, 3)
    rn, thn, phn = _np(r), _np(th), _np(ph)
    wr = wt = wp = 0.0
    for s in range(min(n_sub, pos_np.shape[0])):
        conf = to_rdkit_mol(en, pos_np[s], inject).GetConformer()
        got_r = np.array([rdmt.GetBondLength(conf, int(perm[a]), int(perm[b]))
                          for a, b in bi])
        got_t = np.array([rdmt.GetAngleRad(conf, int(perm[a]), int(perm[b]), int(perm[c]))
                          for a, b, c in ai])
        got_p = np.array([rdmt.GetDihedralRad(conf, *[int(perm[q]) for q in row])
                          for row in ti])
        wr = max(wr, np.abs(got_r - rn[s]).max())
        wt = max(wt, np.abs(got_t - thn[s]).max())
        wp = max(wp, np.abs((got_p - phn[s] + np.pi) % (2 * np.pi) - np.pi).max())
    led.check('external_r', name, level, wr, TOL['external_r'], units='A')
    led.check('external_theta', name, level, wt, TOL['external_theta'], units='rad')
    led.check('external_phi', name, level, wp, TOL['external_phi'], units='rad')


def _chirality(en, name, level, led, pos, n_sub, inject):
    """RDKit re-perceives the stereocentres from the BUILT Cartesians.

    Every energy term is a function of interatomic distances, or of the dihedral through
    terms that are even under a global reflection, so a mirrored molecule scores
    identically. This is the only check in the file that can see a reflection.
    """
    from rdkit import Chem
    ref = Chem.AddHs(Chem.MolFromSmiles(en.smiles))
    Chem.AssignStereochemistry(ref, cleanIt=True, force=True)
    want = {a.GetIdx(): a.GetProp('_CIPCode') for a in ref.GetAtoms()
            if a.HasProp('_CIPCode')}
    if not want:
        led.skip('chirality', name, level,
                 'no specified stereocentre in the SMILES, so there is no CIP code to '
                 'recover and a reflection is not observable on this molecule')
        return
    N = en.spec.n_atoms
    pos_np = _np(pos).reshape(-1, N, 3)
    bad, got_any = 0, None
    for s in range(min(n_sub, pos_np.shape[0])):
        m = to_rdkit_mol(en, pos_np[s], inject)
        Chem.AssignStereochemistryFrom3D(m)
        got = {a.GetIdx(): a.GetProp('_CIPCode') for a in m.GetAtoms()
               if a.HasProp('_CIPCode')}
        got_any = got
        bad += int(got != want)
    led.check('chirality', name, level, bad, 0, '==', units='mismatched conformers')
    led.note('cip_built', name, level,
             ','.join(f'{k}:{v}' for k, v in sorted((got_any or {}).items())))


def _external_mmff(en, name, level, led, pos, n_sub, inject):
    """RDKit's own MMFF94, TERM BY TERM, on the geometries we built.

    Per term rather than on the total, because a total hides two errors this code has
    actually had: an un-rescaled vdW R*/eps (silent unless the molecule hydrogen bonds) and
    a missing electrostatic term (which a total would have absorbed into vdW).
    """
    from rdkit.Chem import AllChem
    from mxtaltools.conformers.energy import intramolecular_energy
    N = en.spec.n_atoms
    pos_np = _np(pos).reshape(-1, N, 3)
    n_sub = min(n_sub, pos_np.shape[0])
    tree1, ff1 = en._batch(1)
    worst = defaultdict(float)
    worst_total = 0.0
    for s in range(n_sub):
        p1 = torch.as_tensor(pos_np[s], dtype=en.dtype, device=en.device)
        tot, comp = intramolecular_energy(tree1, p1, ff1, components=True)
        mol = to_rdkit_mol(en, pos_np[s], inject)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            led.skip('mmff_* (7 terms + total)', name, level,
                     'RDKit could not MMFF-type this molecule')
            return
        for key, rd in RDKIT_TERMS.items():
            for u in ALL_RDKIT:
                getattr(props, f'SetMMFF{u}Term')(u == rd)
            ref = AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()
            worst[key] = max(worst[key], abs(float(comp[key].sum()) - ref))
        for u in ALL_RDKIT:
            getattr(props, f'SetMMFF{u}Term')(True)
        ref_tot = AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()
        worst_total = max(worst_total, abs(float(tot.sum()) - ref_tot))
    for key in RDKIT_TERMS:
        led.check(f'mmff_{key}', name, level, worst[key], TOL[f'mmff_{key}'],
                  units='kcal/mol')
    led.check('mmff_total', name, level, worst_total, TOL['mmff_total'], units='kcal/mol')


def _t_eff(en, name, level, led, ff, pos, n, T, ff_choice):
    """2 <E_bond> / (n_bond kT) over FREE, NON-RING bonds. Exactly 1 by equipartition.

    The bond term is isolated by ZEROING k_bond on every other bond and calling the shipped
    ``intramolecular_energy`` again, rather than by reimplementing the functional form here
    -- MMFF's stretch is quartic, and a hand-rolled copy would be one more thing that can
    drift away from the code it is supposed to be checking.
    """
    from mxtaltools.conformers.energy import intramolecular_energy
    free_r = np.asarray(en.free_mask)[:en.n_r]
    if not free_r.any():
        led.skip('T_eff', name, level,
                 f"level {level!r} freezes every r column at the reference conformer, so "
                 f"the bond term is a CONSTANT in x. 2E/(n kT) there is the reference "
                 f"conformer's own strain against MMFF's typed r0, not a temperature, and "
                 f"reporting it as one would be a fabricated measurement")
        return
    locked = ring_locked(en)
    bi = np.asarray(en.spec.bond_index)
    keep = [j for j in range(en.n_r) if free_r[j] and j not in locked]
    if not keep:
        led.skip('T_eff', name, level,
                 'every free bond is ring-locked and therefore drawn at a fraction of its '
                 'thermal width, so equipartition does not apply to any of them')
        return
    want = {frozenset((int(bi[j, 0]), int(bi[j, 1]))) for j in keep}
    n_at = en.spec.n_atoms
    ffbi = _np(ff.bond_index) % n_at
    sel = np.array([frozenset((int(a), int(b))) in want for a, b in ffbi])
    if not sel.any():
        led.skip('T_eff', name, level,
                 'no force-field bond term maps onto a free non-ring tree bond')
        return
    kb = ff.k_bond.clone()
    kb[torch.as_tensor(~sel, device=kb.device)] = 0.0
    masked = dataclasses.replace(ff, k_bond=kb)
    _, comp = intramolecular_energy(None, pos, masked, components=True)
    n_sel = int(sel.sum()) // n
    val = 2.0 * float(comp['bond'].mean()) / (n_sel * T)
    led.note('T_eff_n_bonds', name, level, n_sel)
    led.band('T_eff', name, level, val, TOL['T_eff_lo'], TOL['T_eff_hi'], units='T_eff/T')
    if ff_choice == 'reference':
        led.skip('T_eff_detects_per_bond_k', name, level,
                 "force_field='reference' assigns a CONSTANT k_bond=300 and k_angle=50 to "
                 "every term, so every thermal sigma in the molecule is identical. T_eff "
                 "still detects a global scale error or a lost factor of T, but it is BLIND "
                 "to a force constant assigned to the wrong bond -- there is no wrong bond "
                 "to assign it to")


def _prior_leg(en, name, level, led, prior, prior_n, seed):
    """Call into ``energies/prior_diagnostics.py`` -- the module with zero callers repo-wide.

    ``prior_report`` and ``coverage_report`` RAISE at ``level='torsion'``: the state -> DoF
    map is collective there, so ``state_from_dof`` has no row-wise inverse. ``prior_report``
    raises again on any molecule with a ring, because a ring block's density is a mixture
    that is singular in the directions its subspace does not span. Both are caught and
    reported SKIPPED with the exception text VERBATIM. That the only level which ships is
    the one these functions cannot measure is the finding, not a nuisance to route around.

    WHY THIS LEG MOSTLY REPORTS RATHER THAN ASSERTS. Two of its headline numbers cannot
    carry a threshold that a correct pipeline would pass, and inventing one would be worse
    than having none:

    * ``eta = ESS_fitted / ESS_oracle`` is NOT bounded by 1. The oracle is built by scanning
      each group leader's 1-D slice of the true energy with the other coordinates held at
      the reference, and prior_diagnostics' own docstring calls it "A LOWER BOUND ON THE
      PRODUCT-FORM CEILING, not the ceiling". A fitted histogram can and does beat a
      one-dimensional slice -- measured eta = 2.41 on propanol at `full`, with D_avoidable
      going negative. So eta above 1 is legal and there is no bar.
    * ``n_missed`` counts accessible basins with zero draws, where "accessible" is defined
      at 10 kT. A basin 10 kT up carries Boltzmann weight e^-10 ~ 5e-5, so a CORRECT prior
      is expected to miss some of them, and on top of that there is a Monte-Carlo empty-bin
      floor of order n_accessible * exp(-n / n_accessible). Asserting 0 fails on correct
      code (measured: 2 missed of 729 modes on hexanol); asserting some tolerated count
      would be a number with no argument behind it.

    What IS assertable: the ESS FRACTION lies in (0, 1] -- that is Cauchy-Schwarz, not a
    modelling choice -- and the box-clip rate, which is a property of the pipeline rather
    than of the fit.
    """
    import energies.prior_diagnostics as pdg
    if prior is None:
        for k in ('prior_report', 'coverage_report'):
            led.skip(k, name, level,
                     'no fitted InternalPrior available (--no-prior, or the cache file is '
                     'missing). These are the only checks in the file that need a fit')
        return
    try:
        rep = pdg.prior_report(en, prior, n=prior_n, seed=seed, n_boot=60)
    except Exception as ex:
        led.skip('prior_report', name, level, f'{type(ex).__name__}: {ex}')
    else:
        led.check('prior_ess_positive', name, level, rep['ess_fitted'], 1e-12, '>=')
        led.check('prior_ess_le_one', name, level, rep['ess_fitted'], TOL['prior_ess_max'])
        led.check('prior_clip_frac', name, level, rep['clip_frac'], TOL['clip_frac'])
        led.note('prior_ess_pct', name, level, 100 * rep['ess_fitted'], units='%')
        led.note('prior_eta', name, level, rep['eta'], units='NOT bounded by 1, see docstring')
        led.note('prior_D_avoidable', name, level, rep['D_avoidable'], units='nats')
    try:
        cov = pdg.coverage_report(en, prior, n=prior_n, seed=seed)
    except Exception as ex:
        led.skip('coverage_report', name, level, f'{type(ex).__name__}: {ex}')
        return
    if 'skipped' in cov:
        led.skip('coverage_report', name, level, cov['skipped'])
        return
    led.skip('coverage_missed', name, level,
             'coverage_report RAN and its numbers are in the REPORTED block, but n_missed '
             'carries NO threshold: "accessible" is defined at 10 kT, whose Boltzmann '
             'weight is e^-10 ~ 5e-5, so a CORRECT prior is expected to miss some of those '
             'basins; and there is an empty-bin floor of order n_accessible * '
             'exp(-n / n_accessible) on top. Asserting 0 fails on correct code and any '
             'tolerated count would be a number with no argument behind it')
    led.note('coverage_n_modes', name, level, cov['n_modes'])
    led.note('coverage_n_accessible', name, level, cov['n_accessible'])
    led.note('coverage_n_missed', name, level, cov['n_missed'])
    led.note('coverage_empty_bin_floor', name, level,
             cov['n_accessible'] * math.exp(-prior_n / max(cov['n_accessible'], 1)),
             units='basins expected empty by chance')
    led.note('coverage_worst_frac_pct', name, level, 100 * cov['worst_frac'], units='%')
    led.note('coverage_excess_median_kt', name, level, cov['excess_median_kt'], units='kT')


def chiral_pair_check(led, level, summaries, results_cip):
    """The two enantiomers must build to OPPOSITE CIP codes from identical machinery.

    A single molecule's CIP check catches a reflection applied to everything. This catches
    the narrower failure where the builder ignores the input stereochemistry entirely and
    emits the same hand for both SMILES -- which would leave each molecule's own check
    passing on one of the two.
    """
    a, b = results_cip.get('butan-2-ol-R'), results_cip.get('butan-2-ol-S')
    if not a or not b:
        led.skip('chiral_pair_opposite', 'butan-2-ol', level,
                 'one or both members of the chiral pair did not run at this level')
        return
    led.check('chiral_pair_opposite', 'butan-2-ol', level, int(a == b), 0, '==',
              units='identical CIP assignments (want 0)')


# =======================================================================================
# reporting
# =======================================================================================

def git_rev():
    try:
        r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO,
                           capture_output=True, text=True, timeout=10)
        d = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO,
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() + ('  (working tree DIRTY)' if d.stdout.strip() else
                                   '  (clean)')
    except Exception:
        return 'unknown'


def fmt(v):
    if v is None:
        return ''
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(fmt(u) for u in v) + ']'
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)) or float(v) == int(float(v)):
        if abs(float(v)) < 1e6:
            return f'{int(float(v))}'
    a = abs(float(v))
    return f'{float(v):.4g}' if 1e-3 <= a < 1e5 else f'{float(v):.3e}'


def print_report(cfg, led, summaries, out=sys.stdout):
    p = lambda s='': print(s, file=out)
    p('=' * 96)
    p('prior_smoke -- committed smoke harness for the NON-LEARNED conformer pipeline')
    p('=' * 96)
    shipped = ' <-- THE SHIPPED LEVEL' if cfg['level'] == 'torsion' else \
        ' <-- NOT the shipped level; nothing in the repo instantiates this one'
    p(f"  level             {cfg['level']!r}{shipped}")
    p(f"  force field       {cfg['force_field']!r}"
      + ("   (reference: k_bond=300 / k_angle=50 CONSTANT, no torsion, oop, stretch-bend "
         "or electrostatic term -- several checks go blind, see SKIPPED)"
         if cfg['force_field'] == 'reference' else ''))
    p(f"  states / molecule {cfg['n']}     seed {cfg['seed']}     "
      f"external subset {cfg['n_external']} conformer(s)/molecule")
    p(f"  prior             {cfg['prior']}")
    p(f"  dtype / device    {cfg['dtype']} / {cfg['device']}")
    p(f"  git               {cfg['git']}")
    if cfg['inject']:
        p(f"  INJECTED BUGS     {', '.join(cfg['inject'])}   "
          f"(this run is EXPECTED to fail)")
    p()

    # ---- per molecule -------------------------------------------------------------
    p('MOLECULES')
    p(f"  {'name':<18}{'SMILES':<22}{'N':>4}{'d':>5}{'rot':>5}{'lin':>5}"
      f"{'E/T median':>13}{'pass':>7}{'skip':>6}{'FAIL':>6}   hazard")
    for s in summaries:
        by = defaultdict(int)
        for r_ in led.rows:
            if r_.mol == s['name']:
                by[r_.status] += 1
        if 'skipped' in s:
            p(f"  {s['name']:<18}{s['smiles']:<22}{'-':>4}{'-':>5}{'-':>5}{'-':>5}"
              f"{'SKIPPED':>13}{by['pass']:>7}{by['skip']:>6}{by['FAIL']:>6}   "
              f"{s['skipped']}")
            continue
        p(f"  {s['name']:<18}{s['smiles']:<22}{s['n_atoms']:>4}{s['d']:>5}"
          f"{s['n_rotatable']:>5}{s['n_linear_angle']:>5}"
          f"{s['energy_median']:>13.2f}{by['pass']:>7}{by['skip']:>6}{by['FAIL']:>6}   "
          f"{s['hazard'][:44]}")
    p()

    # ---- per check ----------------------------------------------------------------
    p('CHECKS  (worst molecule per check; "worst" = closest to the bar)')
    p(f"  {'check':<26}{'pass':>5}{'skip':>5}{'FAIL':>5}   {'worst on':<20}"
      f"{'value':>12}{'bar':>14}   units")
    seen = []
    for r_ in led.rows:
        if r_.name not in seen and r_.status in ('pass', 'FAIL', 'skip'):
            seen.append(r_.name)
    for nm in seen:
        rows = [r_ for r_ in led.rows if r_.name == nm]
        npass = sum(r_.status == 'pass' for r_ in rows)
        nskip = sum(r_.status == 'skip' for r_ in rows)
        nfail = sum(r_.status == 'FAIL' for r_ in rows)
        scored = [r_ for r_ in rows if r_.value is not None]
        if scored:
            if any(r_.cmp == 'in' for r_ in scored):
                w = max(scored, key=lambda q: abs(float(q.value) - 1.0))
            elif scored[0].cmp == '>=':
                w = min(scored, key=lambda q: float(q.value))
            else:
                w = max(scored, key=lambda q: float(q.value))
            p(f"  {nm:<26}{npass:>5}{nskip:>5}{nfail:>5}   {w.mol:<20}"
              f"{fmt(w.value):>12}{fmt(w.tol):>14}   {w.units}")
        else:
            p(f"  {nm:<26}{npass:>5}{nskip:>5}{nfail:>5}   {'-':<20}{'-':>12}{'-':>14}")
    p()

    # ---- skips, grouped by reason --------------------------------------------------
    p('SKIPPED  (a check that did not run is not a check that passed)')
    groups = defaultdict(list)
    for r_ in led.skips:
        groups[(r_.name, r_.reason)].append(r_.mol)
    if not groups:
        p('  none')
    for (nm, reason), mols in sorted(groups.items()):
        p(f"  {nm}  [{len(mols)}: {', '.join(mols)}]")
        for line in _wrap(reason, 88):
            p(f"      {line}")
    p()

    # ---- notes ---------------------------------------------------------------------
    notes = [r_ for r_ in led.rows if r_.status == 'note']
    if notes:
        p('REPORTED, NOT ASSERTED  (no bar; here so the number is on the record)')
        byname = defaultdict(list)
        for r_ in notes:
            byname[r_.name].append(r_)
        for nm, rows in byname.items():
            vals = '  '.join(f'{r_.mol}={fmt(r_.value)}' for r_ in rows)
            for i, line in enumerate(_wrap(vals, 82)):
                p(f"  {nm if i == 0 else '':<27}{line}")
        p()

    # ---- verdict --------------------------------------------------------------------
    if led.failures:
        p('VIOLATIONS')
        for r_ in led.failures:
            p(f"  {r_.name}  on  {r_.mol}  ({r_.level}): "
              f"{fmt(r_.value)} {r_.cmp} {fmt(r_.tol)} is FALSE"
              + (f'  [{r_.units}]' if r_.units else ''))
        p()
    n_pass = sum(r_.status == 'pass' for r_ in led.rows)
    p('-' * 96)
    p(f"  {n_pass} passed   {len(led.skips)} skipped   {len(led.failures)} FAILED   "
      f"({cfg['seconds']:.1f}s)")
    p(f"  VERDICT: {'FAIL' if led.failures else 'PASS'}")
    p('-' * 96)


def _wrap(s, width):
    words, line, out = s.split(), '', []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f'{line} {w}'.strip()
    if line:
        out.append(line)
    return out or ['']


# =======================================================================================
# driver
# =======================================================================================

def run(level='torsion', force_field='mmff', n=2000, seed=0, n_external=4,
        prior_path=None, prior_n=1500, only=None, inject=()):
    from energies.conformer_torsions import ConformerTorsions
    if level not in ConformerTorsions.LEVELS:
        raise SystemExit(f'--level must be one of {ConformerTorsions.LEVELS}, got {level!r}')
    if force_field not in ('mmff', 'reference'):
        raise SystemExit(f"--force-field must be 'mmff' or 'reference', got {force_field!r}")

    prior, prior_desc = None, 'none (--no-prior)'
    if prior_path is not None:
        pp = Path(prior_path)
        if pp.exists():
            prior = torch.load(pp, weights_only=False)
            # vars(), NOT getattr: InternalPrior is a dataclass, so a field with a default
            # is also a CLASS attribute and getattr on a prior pickled before the field
            # existed reports the current default. Only __dict__ distinguishes the two.
            sig = vars(prior).get('ring_sig_version', '<absent: predates the field>')
            prior_desc = (f'{pp.name}  (ring_sig_version={sig}, fatten={prior.fatten}, '
                          f'{prior.n_fitted} molecules fitted)')
        else:
            prior_desc = f'{pp} NOT FOUND -- the prior_diagnostics leg cannot run'

    led = Ledger()
    summaries, cips = [], {}
    t0 = time.time()
    todo = [m for m in MOLECULES if only is None or m[0] in only]
    for name, smiles, hazard in todo:
        s = run_molecule(name, smiles, level, force_field, n, seed, n_external, led,
                         prior, prior_n, inject)
        s['hazard'] = hazard
        summaries.append(s)
        cip = [r_ for r_ in led.rows if r_.mol == name and r_.name == 'cip_built']
        if cip:
            cips[name] = cip[-1].value
    chiral_pair_check(led, level, summaries, cips)

    cfg = {'level': level, 'force_field': force_field, 'n': n, 'seed': seed,
           'n_external': n_external, 'prior': prior_desc, 'prior_n': prior_n,
           'dtype': str(torch.float64), 'device': 'cpu', 'git': git_rev(),
           'inject': list(inject), 'seconds': time.time() - t0}
    return cfg, led, summaries


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level', default='torsion',
                    choices=['torsion', 'dihedral', 'flex', 'full'],
                    help="free-DoF level. DEFAULTS TO 'torsion', which is the only level "
                         "anything in the repo instantiates")
    ap.add_argument('--force-field', default='mmff', choices=['mmff', 'reference'])
    ap.add_argument('--n', type=int, default=2000, help='states per molecule')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-external', type=int, default=4,
                    help='conformers per molecule sent through the RDKit legs')
    ap.add_argument('--prior-path', default=str(REPO / 'conformer_prior_v2.pt'))
    ap.add_argument('--no-prior', action='store_true',
                    help='skip the prior_diagnostics leg entirely')
    ap.add_argument('--prior-n', type=int, default=1500,
                    help='draws inside prior_report / coverage_report')
    ap.add_argument('--only', nargs='*', default=None, help='restrict to these molecules')
    ap.add_argument('--inject', nargs='*', default=[],
                    choices=sorted(INJECTIONS) + ['all'],
                    help='re-introduce a known bug and require the harness to FAIL')
    ap.add_argument('--json', default=None, help='write the full record here')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(argv)

    torch.set_num_threads(a.threads)
    inject = tuple(sorted(INJECTIONS)) if 'all' in a.inject else tuple(a.inject)
    cfg, led, summaries = run(level=a.level, force_field=a.force_field, n=a.n,
                              seed=a.seed, n_external=a.n_external,
                              prior_path=None if a.no_prior else a.prior_path,
                              prior_n=a.prior_n, only=a.only, inject=inject)
    print_report(cfg, led, summaries)

    if a.json:
        Path(a.json).write_text(json.dumps(
            {'config': cfg, 'tolerances': TOL,
             'molecules': [{k: v for k, v in s.items()} for s in summaries],
             'checks': [r.as_dict() for r in led.rows],
             'n_pass': sum(r.status == 'pass' for r in led.rows),
             'n_skip': len(led.skips), 'n_fail': len(led.failures),
             'verdict': 'FAIL' if led.failures else 'PASS'},
            indent=2, default=str), encoding='utf-8')
        print(f'wrote {a.json}')

    if inject:
        # An injection run INVERTS the exit code: the point is that the harness fires, and
        # that it fires on the check that was SUPPOSED to catch that bug. "Something
        # failed" is not evidence -- a check can fail for an unrelated reason and leave the
        # intended detector blind, which is precisely how a test battery goes quietly dead.
        fired = {r.name for r in led.failures}
        print('\n  INJECTION RUN -- the named detector must be among the failures')
        ok = True
        for b in inject:
            targets = [t.strip() for t in INJECTIONS[b][0].split('/')]
            hit = [t for t in targets if t in fired]
            ok &= bool(hit)
            mols = sorted({r.mol for r in led.failures if r.name in hit})
            print(f"    {b:<22} target {INJECTIONS[b][0]:<38} "
                  f"{'CAUGHT by ' + ', '.join(hit) if hit else 'NOT CAUGHT'}"
                  + (f'  on {len(mols)}/{len(summaries)} molecules' if hit else ''))
        others = sorted(fired - {t.strip() for b in inject
                                 for t in INJECTIONS[b][0].split('/')})
        if others:
            print(f"    (also fired, incidentally: {', '.join(others)})")
        print(f"  VERDICT: {'PASS -- the harness can fail' if ok else 'FAIL -- BLIND'}")
        return 0 if ok else 1
    return 1 if led.failures else 0


if __name__ == '__main__':
    sys.exit(main())
