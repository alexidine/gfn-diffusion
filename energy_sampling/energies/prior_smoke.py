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
``--inject`` re-introduces eleven real bug classes so the checks can be SHOWN to fire rather
than assumed to. Run ``--inject all`` after touching this file -- it makes ONE PASS PER BUG,
so the question "did THIS bug's named detector fire" stays answerable. A harness whose
injections stop firing has gone blind, and is worse than no harness.

**3. A MEASUREMENT CANNOT SEE A FROZEN ROW.** This one was learned by the harness failing an
audit rather than by design. Every check below leg 0 measures a SAMPLE, and ``torsion`` --
the only level anything in the repo instantiates -- freezes every improper phi row and every
r/theta row at the reference. A grouping, typing or draw-width defect in those rows is then
invisible to a sample at exactly the level that ships, and four separately injected defects
duly passed the default invocation while failing only at ``--level full``, which nothing
runs. Leg 0 is the answer: integer counts of contract violations, no draw, no energy, no
fitted prior, no tolerance. Freezing a row does not blind a count of it.

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

* ``group_rigid_*`` -- the same rigid-motion claim, but with the rows taken from
  ``torsion_groups()`` rather than from ``_find_rotatable``'s mask. The two row sources are
  computed independently, which is why ``rigid_angle`` sits at 2e-15 on a grouping defect
  that this one catches at 1.0 rad.
* ``prior_key_external`` -- ``prior_dof_types``' per-row keys against the mxtaltools
  ``tree_*`` derivation, which is the route ``InternalPrior.fit`` actually used to build the
  tables. Two implementations of the same claim, not one implementation checked against
  itself.

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

KNOWN BLIND SPOT, STATED RATHER THAN PAPERED OVER
-------------------------------------------------
Keying ``torsion_groups`` on the PARENT ATOM instead of the central bond -- the failure its
own docstring warns about, and which the ethanol damage is attributed to -- changes NOTHING
on this tree spec. In a spanning tree every atom has exactly one parent, so for a PROPER row
the reference b is a function of the parent c and the two keys induce the identical
partition; measured, 0 multi-reference parents among proper rows on all 11 molecules at both
levels. Swapping the key and re-running gives byte-identical output and exit 0. No
behavioural check can see it because there is nothing to see: on this tree the central-bond
key is a redundant restatement of an invariant the tree already guarantees, and the ethanol
damage is attributable to the IMPROPER rows being in the group, which is check (1) of leg 0.
``tree_parent_unique`` asserts the precondition instead, so the day the tree stops
guaranteeing it -- and the parent key starts being silently wrong -- is a visible event.

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

    # --- LEG 0: structural invariants of the tree and the grouping -----------------------
    # These carry NO tolerance and need no draw, no energy and no fitted prior. That is the
    # point of them: every other check in this file is a measurement on a sample, and a
    # parameterisation that FREEZES the suspect rows -- which `torsion`, the only level that
    # ships, does to every improper row -- defeats a measurement by construction. An integer
    # count of contract violations is not defeated by it. Each bar is 0 because each is a
    # definitional identity, not a rate; a tolerated count here would be a number with no
    # argument behind it.
    'improper_rows_ungrouped': 0,
    'group_frame_bond': 0,
    'tree_parent_unique': 0,
    'prior_key_external': 0,
    # A common displacement over the members of ONE group is a rigid motion of the placed
    # fragment, so every graph bond and graph angle is invariant in exact arithmetic --
    # identical argument to rigid_bond/rigid_angle, hence identical bars, except that the
    # rows come from torsion_groups() rather than from _find_rotatable's mask. The two row
    # sources are computed independently, which is why rigid_angle sits at 2e-15 on a
    # grouping defect that this one catches.
    'group_rigid_bond': 1e-8,
    'group_rigid_angle': 1e-8,
    # Realised sample width of sample_prior_states' OWN draw, as a fold deviation from the
    # force field's thermal sigma, worst over free non-ring rows. For a harmonic term drawn
    # at its own thermal width the ratio is exactly 1 BY CONSTRUCTION -- the same
    # equipartition statement kt_per_bond rests on, measured on the DRAW instead of on the
    # energy, which is why it survives at `torsion` where the bond term is constant in x.
    # Two-sided (max of q and 1/q), so a lost width or a lost factor of T fires too.
    # Measured clean 1.04 across the set at both levels; the pooled-histogram regression
    # gives 3.1-14.2. 1.5 sits 1.45x above clean and 2.1x below the failure.
    'prior_rtheta_width': 1.5,
    # Same quantity for the IMPROPER phi rows, against improper_phi_sigma. Looser (2.0)
    # for a stated reason: that sigma is a STAND-IN -- the median tree-angle width, with the
    # dihedral-to-angle Jacobian taken as order one -- so the prediction is order-one rather
    # than exact, where thermal_rtheta_sigma IS the exact marginal. Measured clean 1.02;
    # a rotamer-histogram draw on an improper spreads over the circle at circstd ~1.8 rad
    # against s_imp ~0.1, i.e. ~18x.
    'prior_improper_sigma': 2.0,

    # --- FLOORS AND POSITIVE CONTROLS -----------------------------------------------------
    # Everything above this line except T_eff is ONE-SIDED, and 31 of them are ceilings on
    # a RESIDUAL. A residual is smallest when nothing happened, so zero is simultaneously
    # the correct answer and the signature of a dead pipeline, and no floor ON THE RESIDUAL
    # can separate those two. The separator is a positive control on the PERTURBATION and a
    # floor on the POPULATION -- never on the residual itself. That is what this block is.
    #
    # (a) The perturbation actually moved the molecule. rigid_* and group_rigid_* assert
    # that a common dihedral displacement leaves every graph bond and graph angle invariant.
    # A builder that ignores its dihedral argument satisfies that claim perfectly. The set
    # the invariance claim does NOT cover is the NONBONDED pair distances, so that is what
    # this measures: the largest change in any nonbonded pair distance under the same
    # displacement the invariance checks apply. A whole-fragment rotation moves those by
    # angstroms; float round-off on the quantity is ~1e-14, so the bar has room to be low
    # and still be unreachable by noise. The clean margin is printed in the CHECKS table.
    'perturbation_moved': 0.10,
    # (b) The DRAW actually moved the molecule. Max over atoms of the positional standard
    # deviation across the batch. Positional rather than energetic on purpose: U/T std
    # spans six orders across this molecule set, so an energy-based bar would be either
    # loose everywhere or molecule-specific, while a thermal torsional draw moves atoms by
    # of order an angstrom on every molecule at every level. This is the one check that
    # generalises over the whole residual-ceiling class at once.
    'draw_response': 0.05,
    # (c) POPULATION floors. Bar 1, and it needs no calibration because it is not a
    # tolerance: an empty population makes every ceiling above pass at exactly 0 and every
    # violation-count pass at exactly 0. `--n-external 0` alone turns the entire external
    # geometry and MMFF legs into vacuous passes. There is no defensible "0 is fine" case
    # for any of these, so the bar is the smallest integer that is not vacuous.
    'population_min': 1,
    # (d) T_eff/T from the ANGLE term, 2 <E_ang> / (n kT) over free non-ring angles.
    # Equipartition fixes this at exactly 1, so it is a theory number and not a calibrated
    # one, the same argument as T_eff. The band is where the FLOOR on the angle term
    # belongs -- NOT on kt_per_angle, which at `torsion` is a constant in x and under
    # `force_field='reference'` is machine zero for a correct pipeline, so a floor there
    # would false-fire on correct code. T_eff_angle instead SKIPS wherever the theta block
    # is frozen, which is exactly where a floor would have lied. The upper bar is looser
    # than T_eff's because the angle ratio runs systematically higher on long chains: MMFF's
    # cubic bend correction is one-sided and the reference conformer's typed theta0 offset
    # adds in the same direction.
    'T_eff_angle_lo': 0.70,
    'T_eff_angle_hi': 1.50,
    # (e) The electrostatic FLOOR, and it is the weakest bar in the file for a stated
    # reason. On most of this molecule set MMFF puts charge only on the heteroatom, its
    # hydrogen and the attached carbon, every charged pair is then 1-2 or 1-3, and the
    # nonbonded electrostatic energy is EXACTLY zero by construction -- RDKit's own Ele
    # term agrees at exactly zero. A floor there would false-fire on correct code, so the
    # check is CONDITIONED on the charge parameters being non-degenerate and skips
    # otherwise. On the molecules that do carry charged nonbonded pairs the bar is a
    # magnitude bound calibrated to THIS committed molecule set, which is legitimate only
    # because the set is fixed and committed. It is not the check that closes the
    # electrostatics hole -- ele_pairs_charged and term_energy_live are, because they are
    # count and liveness identities rather than thresholds, and they are meaningful on
    # every molecule including the ones where every energy-based instrument is zero.
    'kt_per_electrostatic_lo': 0.003,
}

# Skip classification. The count that must not be rounded to "0 failed" is UNREACHABLE.
K_MOL = 'inapplicable/molecule'      # the molecule has no such feature. A fact about chemistry.
K_LEVEL = 'inapplicable/level'       # this LEVEL freezes or removes the rows the check reads.
K_CONFIG = 'inapplicable/config'     # this INVOCATION turned it off (--no-prior, reference FF).
K_UNREACHABLE = 'UNREACHABLE'        # a code path RAISED. Nothing measured the property.
K_UNASSERTED = 'unasserted'          # it RAN; there is deliberately no defensible bar.
SKIP_KINDS = (K_MOL, K_LEVEL, K_CONFIG, K_UNREACHABLE, K_UNASSERTED)

# Every PER-MOLECULE assertion name the harness can emit. Not decoration: a skip is at
# least visible, but a check that emits NO ROW AT ALL is invisible, and the checks that
# emit no row are exactly the ones downstream of a raise. Diffing this list against what
# was actually emitted turns that silence into a printed number. The diff runs both ways --
# a name emitted that is not listed here is also reported, so the list cannot rot quietly.
ASSERTIONS = (
    # leg 0: structural counts and populations, no draw, no energy, no fitted prior
    'improper_rows_ungrouped', 'group_frame_bond', 'tree_parent_unique',
    'n_groups', 'n_improper_rows',
    'group_rigid_bond', 'group_rigid_angle', 'group_perturbation_moved',
    'prior_key_external', 'n_key_rows',
    # the draw and its response
    'finite_positions', 'finite_energy', 'finite_e_ref', 'draw_response',
    # leg 1: energy-free round trip and rigidity
    'roundtrip_r', 'roundtrip_theta', 'roundtrip_phi',
    'rigid_bond', 'rigid_angle', 'rigid_perturbation_moved',
    # leg 2: external cross-checks
    'external_r', 'external_theta', 'external_phi', 'n_external_geom',
    'chirality', 'n_chirality_conformers',
    'mmff_bond', 'mmff_angle', 'mmff_stretch_bend', 'mmff_oop', 'mmff_torsion',
    'mmff_lj', 'mmff_electrostatic', 'mmff_total', 'n_external_mmff',
    'ele_pairs_charged',
    # leg 3: physical scale
    'kt_per_bond', 'kt_per_angle', 'kt_per_stretch_bend', 'kt_per_oop', 'kt_per_lj',
    'kt_per_torsion', 'kt_per_electrostatic', 'term_energy_live',
    'closure_sigma', 'clip_frac', 'T_eff', 'T_eff_angle',
    'clash_excess_p10', 'n_clash_pairs', 'batch_invariance',
    # leg 4: sample_prior_states' own draw, then prior_diagnostics
    'prior_rtheta_width', 'prior_improper_sigma',
    'prior_report', 'prior_ess_positive', 'prior_ess_le_one', 'prior_clip_frac',
    'coverage_report', 'coverage_missed',
)

# Rows that are deliberately not per-molecule assertions and are excluded from the darkness
# diff: a construction refusal, a cross-molecule check, and a stated blind spot.
NON_ASSERTION_ROWS = ('MOLECULE', 'chiral_pair_opposite', 'T_eff_detects_per_bond_k')

# The tier this harness exists to supply and the one the user's memory flags as the
# replacement for ESS. Printed as its own line because it is the tier that is entirely dark
# at the shipped level, and a headline pass count hides that completely.
PRIOR_QUALITY_TIER = ('T_eff', 'T_eff_angle', 'prior_report', 'prior_ess_positive',
                      'prior_ess_le_one', 'prior_clip_frac', 'coverage_report',
                      'coverage_missed')

# The entry point and everything downstream of it. When the entry point raises, EVERY name
# here is skipped by name -- otherwise the downstream ones emit no row at all and the hole
# is invisible to any census.
PRIOR_REPORT_CHECKS = ('prior_report', 'prior_ess_positive', 'prior_ess_le_one',
                       'prior_clip_frac')
COVERAGE_CHECKS = ('coverage_report', 'coverage_missed')

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
    # ---- STRUCTURAL injections. These patch the OBJECT rather than the DoF vector,
    # because the defects they reproduce are defects in what the object returns, and at
    # `torsion` -- the level that ships -- the rows they damage are frozen, so no draw-level
    # injection can express them. That is exactly why the sample-based checks were blind to
    # this family and the leg-0 counts are not.
    'improper-in-group': (
        'improper_rows_ungrouped / prior_improper_sigma',
        "drop the improper-row exclusion from torsion_groups, so an improper dihedral is "
        "bucketed with the real rotations about its central bond and then takes the "
        "leader's rotamer displacement. This is the ethanol O4-C0-H1 bug reintroduced at "
        "its actual source. At `torsion` every improper row is FROZEN, so nothing that "
        "measures a draw can see it -- the structural count can"),
    'group-coarse-key': (
        'group_frame_bond',
        'key torsion_groups on the REFERENCE atom b alone instead of the central bond '
        '(b, c), merging dihedrals about DIFFERENT axes into one group. Caught by the '
        'structural count only, and that is the finding: a MERGE is behaviourally benign '
        'here, because the sibling sets it unions are each still displaced together and '
        'no graph angle spans two of them. Splitting is what shears -- see group-split-key'),
    'group-split-key': (
        'group_rigid_angle',
        'key torsion_groups on the PLACED atom, i.e. one row per group. Every sibling then '
        'takes an INDEPENDENT displacement, which is the failure the joint draw exists to '
        'prevent: the H-C-H angle between two substituents of one parent is a DIFFERENCE '
        'of their dihedrals, is scored by the force field, and is not a tree coordinate at '
        'all, so nothing holds it'),
    'prior-key-scramble': (
        'prior_key_external',
        "transpose columns 0 and 1 of spec.torsion_index inside prior_dof_types, so every "
        "phi row is typed on the pair (a, c) instead of the central bond (b, c) and draws "
        "from a different bond type's rotamer histogram. This is the spec-numbering vs "
        "mxtaltools tree_* numbering bug class: it PERMUTES rather than raising, and it "
        "keeps sampler and density self-consistent, so ESS stays a valid fraction"),
    'prior-pooled-rtheta': (
        'prior_rtheta_width',
        'take r and theta from the fitted POOLED histograms instead of the thermal path. '
        'The pooled marginals are 2.5x the thermal width in r and 1.8x in theta, because '
        'they are pooled over chemical environments; the draw is then not the Boltzmann '
        'marginal of the term it is supposed to be'),
    # ---- VACUITY injections. These do not make a number WRONG, they make the population
    # the number is computed over EMPTY, or they stop the pipeline responding to its input.
    # Every one of them left the pre-floor harness at "0 FAILED", which is the finding the
    # floors above exist to fix: a ceiling on an error cannot tell "small error" from
    # "nothing happened", so the detector has to be a positive control or a population
    # count, never a tighter bar.
    'dead-map': (
        'draw_response / rigid_perturbation_moved',
        'shadow dof_from_state so it IGNORES x and returns the reference conformer DoF for '
        'every row -- the pipeline stops responding to its input entirely. Every residual '
        'ceiling in the file is then satisfied EXACTLY, because a residual is smallest when '
        'nothing happened: roundtrip, external, rigid, group_rigid, all eight mmff terms, '
        'batch_invariance and every kt_per_* pass'),
    'null-perturbation': (
        'rigid_perturbation_moved / group_perturbation_moved',
        "set the harness's OWN dihedral displacement to zero in _rigid and _group_rigid. "
        'The invariance claim is then trivially true and four checks report exact zeros '
        'against their bars. This is a defect in the MEASUREMENT rather than in the '
        'pipeline, which is the point: a positive control is the only thing that can see it'),
    'empty-groups': (
        'n_groups',
        'torsion_groups returns an empty list -- the shape a keying or filtering regression '
        'takes when its exclusion becomes total. improper_rows_ungrouped and '
        'group_frame_bond then both count zero violations over zero groups and pass'),
    'no-improper-rows': (
        'n_improper_rows',
        'improper_phi_rows returns an empty list. This is the ethanol bug at its detection '
        'end rather than its damage end: improper_rows_ungrouped -- the leg-0 count the '
        'whole file leans on -- passes vacuously, torsion_groups stops excluding anything, '
        'and prior_improper_sigma skips for want of a row'),
    'empty-prior-keys': (
        'n_key_rows',
        'prior_dof_types returns an empty mapping, so prior_key_external compares zero rows '
        'against the mxtaltools tree_* authority'),
    'no-pairs': (
        'n_clash_pairs',
        'empty the force field NONBONDED pair list. clash_excess_p10 is then measured over '
        'no pairs at all, and it passes -- it goes NEGATIVE, because the reference '
        "conformer's own overlap is measured over the same empty list"),
    'drop-electrostatics': (
        'kt_per_electrostatic',
        'zero the electrostatic ENERGY PATHWAY (ele_scale) while leaving the partial '
        'charges in place. This is the "a whole force-field term went missing" class named '
        'in the module docstring, and against a CEILING it is invisible by construction: a '
        'vanishing term makes abs(E)/count/T go to exactly 0, which every ceiling passes'),
    'drop-charges': (
        'ele_pairs_charged',
        'zero the partial-charge PARAMETERS (ele_qq) instead of the energy pathway. The '
        'complement of drop-electrostatics and deliberately a separate injection: the '
        'kt_per_electrostatic floor cannot fire here, because its guard correctly turns '
        'into a skip when the charges are degenerate. Only a COUNT against RDKit sees it'),
    'cold-theta': (
        'T_eff_angle',
        'halve-and-halve-again the thermal ANGLE width: thermal_rtheta_sigma returns a '
        'theta sigma a tenth of the correct one. Self-referential by construction -- '
        'prior_rtheta_width scores the realised draw against the SAME damaged function and '
        'sees a ratio of 1 -- and kt_per_angle only goes DOWN, which a ceiling passes. Only '
        'an equipartition band on the angle term is independent of the damaged quantity. '
        'Requires a level with a free theta block'),
}

# Injections that cannot express themselves at every level, with the levels at which they
# can. Reporting these as BLIND at a level where the rows they damage carry no free state
# column would be a false alarm; reporting them as CAUGHT would be worse. `--inject all`
# reports them as N/A instead, and names the level to run them at.
INJECTION_REQUIRES_LEVEL = {
    'cold-theta': ('flex', 'full'),
}

# The injections that patch the ENERGY OBJECT rather than the DoF vector. Applied in
# run_molecule immediately after construction, by shadowing the bound method on the
# instance, so every downstream consumer -- draw_states here AND sample_prior_states inside
# conformer_torsions -- sees the damaged version, which is what the real regression does.
STRUCTURAL_INJECTIONS = ('improper-in-group', 'group-coarse-key', 'group-split-key',
                         'prior-key-scramble', 'prior-pooled-rtheta',
                         'dead-map', 'empty-groups', 'no-improper-rows',
                         'empty-prior-keys', 'no-pairs', 'drop-electrostatics',
                         'drop-charges', 'cold-theta')

# Injections that damage the FORCE FIELD OBJECT, applied by wrapping _batch so every cached
# batch size gets the same damage and every consumer -- energy(), the external MMFF leg,
# _worst_overlap -- sees it.
FF_INJECTIONS = ('drop-electrostatics', 'drop-charges', 'no-pairs')

# Three of them rewrite the SAME method (torsion_groups) with three incompatible keying
# rules, so combining them is not "both bugs at once" -- it is whichever one was applied
# last, silently. Refusing is the point: a combined run that quietly drops two injections is
# how a battery goes dead while still printing PASS. `--inject all` runs every injection in
# its OWN pass for the same reason.
EXCLUSIVE_FAMILY = ('improper-in-group', 'group-coarse-key', 'group-split-key',
                    'empty-groups')


# =======================================================================================
# ledger
# =======================================================================================

class Result:
    __slots__ = ('name', 'mol', 'level', 'status', 'value', 'tol', 'cmp', 'reason', 'units',
                 'kind')

    def __init__(self, name, mol, level, status, value=None, tol=None, cmp='<=',
                 reason='', units='', kind=''):
        self.name, self.mol, self.level = name, mol, level
        self.status, self.value, self.tol, self.cmp = status, value, tol, cmp
        self.reason, self.units, self.kind = reason, units, kind

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

    def skip(self, name, mol, level, reason, kind=K_MOL):
        """A check that did not run, CLASSIFIED.

        ``kind`` is the whole point of this signature. "N passed, M skipped, 0 FAILED" gets
        quoted as verification, and it cannot be read at all unless the M separates a
        property that does not exist on this molecule from a code path that RAISED. The
        first is a fact about chemistry; the second is a hole in the pipeline wearing the
        same word.
        """
        if kind not in SKIP_KINDS:
            raise ValueError(f'unknown skip kind {kind!r}; must be one of {SKIP_KINDS}')
        return self._add(Result(name, mol, level, 'skip', reason=reason, kind=kind))

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
# structural injection: shadow a bound method on the INSTANCE
# =======================================================================================

def apply_structural_injections(en, inject):
    """Damage the energy object itself, in place, before anything reads it.

    Each of these reproduces a defect in what a METHOD returns. That matters because at
    ``level='torsion'`` -- the only level anything in the repo instantiates -- the rows these
    defects damage carry no free state column at all (every improper row is frozen; the
    r/theta blocks are frozen), so a draw-level injection is a silent no-op there and the
    whole sample-based battery is blind by construction. Shadowing the bound method is also
    what the real regression does to every consumer at once: ``draw_states`` in this file
    and ``sample_prior_states`` inside conformer_torsions both go through it.
    """
    clash = [b for b in EXCLUSIVE_FAMILY if b in inject]
    if len(clash) > 1:
        raise SystemExit(
            f"--inject {' '.join(clash)}: these rewrite torsion_groups with incompatible "
            f"keying rules, so only the last would survive and the others would be silently "
            f"dropped. Run them one at a time, or use --inject all, which does.")
    if 'improper-in-group' in inject:
        def groups_with_impropers(_en=en):
            ti = np.asarray(_en.spec.torsion_index)
            g = defaultdict(list)
            for j in range(_en.n_ph):                     # the exclusion, simply dropped
                g[(int(ti[j, 1]), int(ti[j, 2]))].append(j)
            return [sorted(rows) for rows in g.values()]
        en.torsion_groups = groups_with_impropers

    if 'group-coarse-key' in inject:
        def groups_by_reference_atom(_en=en):
            ti = np.asarray(_en.spec.torsion_index)
            imp = set(_en.improper_phi_rows())
            g = defaultdict(list)
            for j in range(_en.n_ph):
                if j in imp:
                    continue
                g[int(ti[j, 1])].append(j)                # the REFERENCE atom, not the bond
            return [sorted(rows) for rows in g.values()]
        en.torsion_groups = groups_by_reference_atom

    if 'group-split-key' in inject:
        def groups_by_placed_atom(_en=en):
            imp = set(_en.improper_phi_rows())
            return [[j] for j in range(_en.n_ph) if j not in imp]   # every row alone
        en.torsion_groups = groups_by_placed_atom

    if 'prior-key-scramble' in inject:
        real = en.prior_dof_types

        def scrambled(prior, _en=en, _real=real):
            keep = np.asarray(_en.spec.torsion_index)
            try:
                _en.spec.torsion_index = keep[:, [1, 0, 2, 3]]
                return _real(prior)
            finally:
                _en.spec.torsion_index = keep
        en.prior_dof_types = scrambled

    # ---- VACUITY family: make a population empty, or stop the map responding to x -------
    if 'empty-groups' in inject:
        en.torsion_groups = lambda: []

    if 'no-improper-rows' in inject:
        en.improper_phi_rows = lambda: []

    if 'empty-prior-keys' in inject:
        en.prior_dof_types = lambda prior: {}

    if 'dead-map' in inject:
        real_map = en.dof_from_state

        def dead_map(x, _r=real_map, _en=en):
            # the reference DoF for every row, whatever x is: build, measure, rigid,
            # external and every energy term then agree with themselves perfectly
            z = torch.zeros(x.shape[0], _en.data_ndim, dtype=_en.dtype, device=_en.device)
            return _r(z)
        en.dof_from_state = dead_map

    if 'cold-theta' in inject:
        real_sig = en.thermal_rtheta_sigma

        def cold(temperature, _r=real_sig):
            s_r, s_th = _r(temperature)
            return s_r, 0.1 * s_th
        en.thermal_rtheta_sigma = cold

    ffbugs = [b for b in FF_INJECTIONS if b in inject]
    if ffbugs:
        real_batch = en._batch

        def damaged_batch(batch_size, _r=real_batch, _bugs=tuple(ffbugs)):
            tree, ff = _r(batch_size)
            if 'drop-electrostatics' in _bugs and ff.ele_scale is not None:
                ff.ele_scale = torch.zeros_like(ff.ele_scale)      # energy pathway only
            if 'drop-charges' in _bugs and ff.ele_qq is not None:
                ff.ele_qq = torch.zeros_like(ff.ele_qq)            # parameters only
            if 'no-pairs' in _bugs:
                empty_i = ff.pair_index[:0]
                empty_f = ff.sigma[:0]
                ff.pair_index, ff.pair_batch = empty_i, ff.pair_batch[:0]
                ff.sigma, ff.epsilon = empty_f, ff.epsilon[:0]
                for fld in ('ele_qq', 'ele_scale', 'vdw_rstar'):
                    if getattr(ff, fld, None) is not None:
                        setattr(ff, fld, getattr(ff, fld)[:0])
            return tree, ff
        en._batch = damaged_batch
    return en


# =======================================================================================
# LEG 0 -- structural invariants. No draw, no energy, no fitted prior, no tolerance.
# =======================================================================================

def _global_row(en, kind, j):
    return {'r': j, 'theta': en.n_r + j, 'phi': en.n_r + en.n_th + j}[kind]


def _structure_leg(en, name, level, led):
    """Integer contract violations in the tree spec and in the torsion grouping.

    WHY THIS LEG EXISTS AT ALL. Every other check in the file measures a SAMPLE, and a
    parameterisation that freezes the suspect rows defeats a measurement without fixing
    anything: at `torsion` the state has one column per rotatable bond, every improper row
    is held at its reference, and every r/theta row is held too. A grouping or typing defect
    is then perfectly invisible to the draw at exactly the level that ships. These counts
    are not measurements, so freezing does not blind them, and they cost microseconds.
    """
    ti = np.asarray(en.spec.torsion_index)
    imp = set(en.improper_phi_rows())
    groups = en.torsion_groups()
    grouped = set().union(*[set(g) for g in groups]) if groups else set()

    # (1) improper rows must not appear in a torsion group. torsion_groups' own docstring
    # states this as a contract: an improper dihedral IS an angle at the parent, so giving
    # it the group leader's rotamer displacement destroys that angle outright -- ethanol's
    # O4-C0-H1 at 14.5 deg against theta0 = 108.6, carrying 251 of 252 kcal/mol.
    led.check('improper_rows_ungrouped', name, level, len(imp & grouped),
              TOL['improper_rows_ungrouped'], '==', units='improper rows inside a group')

    # (2) every member of a group must share ONE central bond (b, c). The group's mechanism
    # is that each member takes the leader's angular displacement, and that is a rotation
    # only when they share the reference axis. A coarser key gathers dihedrals measured
    # about DIFFERENT axes and shears the fragment instead.
    bad = sum(1 for g in groups if len({(int(ti[j, 1]), int(ti[j, 2])) for j in g}) != 1)
    led.check('group_frame_bond', name, level, bad, TOL['group_frame_bond'], '==',
              units='groups spanning more than one central bond')

    # (3) THE PRECONDITION TRIPWIRE, and it is worth being explicit about what it is for.
    # On this tree spec, keying the groups on the PARENT ATOM c gives the IDENTICAL
    # partition to keying on the central bond (b, c), because every atom has exactly one
    # parent in a spanning tree, so for a proper row b is a function of c. Measured: 0
    # multi-reference parents among proper rows on all 11 molecules at both levels.
    #   The consequence is uncomfortable and is stated here rather than hidden: swapping the
    # central-bond key for the parent key today changes NOTHING -- not the partition, not a
    # single downstream number -- so no behavioural check can see it, and none in this file
    # does. What CAN be asserted is the property that makes the two keys agree. The day the
    # NeRF reference-selection policy changes and a parent starts carrying two references,
    # the parent key becomes silently wrong; this fires on that day instead of leaving it a
    # surprise. (Improper rows are excluded because they are the root-frame rows, where the
    # multiple references are expected and legitimate -- and where they are already handled
    # by the exclusion checked in (1).)
    refs = defaultdict(set)
    for j in range(en.n_ph):
        if j not in imp:
            refs[int(ti[j, 2])].add(int(ti[j, 1]))
    multi = {c: sorted(s) for c, s in refs.items() if len(s) > 1}
    led.check('tree_parent_unique', name, level, len(multi), TOL['tree_parent_unique'], '==',
              units='parent atoms carrying more than one reference')

    # (4) THE POPULATIONS the three counts above are computed over. Every one of those is a
    # violation count with a bar of 0, and an empty population has zero violations: with no
    # groups, (1) and (2) both pass at 0 over nothing; with no improper rows, (1) passes at
    # 0 while the exact defect it was written to catch is present and unrecorded. The floor
    # goes here, on the population, and never on the count -- a tolerated violation count
    # would be a number with no argument behind it, whereas "at least one row to count" is
    # not a tolerance at all.
    led.check('n_groups', name, level, len(groups), TOL['population_min'], '>=',
              units='torsion groups the grouping checks are computed over')
    led.check('n_improper_rows', name, level, len(imp), TOL['population_min'], '>=',
              units='improper phi rows the exclusion check is computed over')


def _nonbonded_d(pos, ff):
    """Every nonbonded pair distance. The set a rigid-motion claim does NOT cover.

    ``rigid_*`` and ``group_rigid_*`` assert that a common dihedral displacement leaves
    every graph BOND and every graph ANGLE invariant. That is true of a rotation and it is
    equally true of doing nothing at all, so those checks cannot distinguish the two. The
    nonbonded distances are precisely the coordinates a real fragment rotation DOES change,
    which is what makes them the positive control for the same perturbation.
    """
    if not ff.pair_index.numel():
        return None
    return torch.linalg.norm(pos[ff.pair_index[:, 0]] - pos[ff.pair_index[:, 1]], dim=-1)


def _moved(d0, d1):
    if d0 is None or d1 is None or d0.shape != d1.shape or not d0.numel():
        return 0.0
    return (d1 - d0).abs().max().item()


def _group_rigid(en, name, level, led, seed, inject=()):
    """A common delta on ALL members of one torsion group must be a RIGID motion.

    This is the behavioural half of the grouping contract, and it is a genuinely different
    check from ``rigid_bond``/``rigid_angle`` above even though the bars are identical: that
    one takes its rows from ``_find_rotatable``'s mask and only ever exercises bonds the
    LEVEL frees, this one takes them from ``torsion_groups()`` -- the object the joint draw
    actually consults -- and exercises every group, including the ones no state column
    drives. A grouping keyed too coarsely leaves rigid_angle at 2e-15 and fires here.

    Ring groups are excluded: rotating a dihedral whose central bond is in a ring breaks the
    closure bond, and closure bonds and ring angles ARE in the force field's graph lists, so
    the motion is legitimately not rigid there. That is what closure_sigma measures instead.
    """
    from mxtaltools.conformers.builder import build
    groups = en.torsion_groups()
    ti = np.asarray(en.spec.torsion_index)
    inr = en.atom_in_ring
    nb = 8
    rng = np.random.default_rng(seed + 23)
    x0, _ = draw_states(en, nb, rng)
    tree, ff = en._batch(nb)
    r0, th0, ph0 = en.dof_from_state(x0)
    p0 = build(tree, r0.reshape(-1), th0.reshape(-1), ph0.reshape(-1))
    b0, a0 = graph_geometry(p0, ff)
    d0 = _nonbonded_d(p0, ff)

    worst_b = worst_a = 0.0
    moved = 0.0
    tested = 0
    for g in groups:
        if any(inr[int(ti[j, 1])] and inr[int(ti[j, 2])] for j in g):
            continue                      # a ring bond: rotating it opens the closure
        ph1 = ph0.clone()
        delta = 0.0 if 'null-perturbation' in inject else float(rng.uniform(0.4, 2.0))
        for j in g:
            ph1[:, j] += delta            # ONE shared displacement over the whole group
        p1 = build(tree, r0.reshape(-1), th0.reshape(-1), ph1.reshape(-1))
        b1, a1 = graph_geometry(p1, ff)
        worst_b = max(worst_b, (b1 - b0).abs().max().item())
        worst_a = max(worst_a, (a1 - a0).abs().max().item())
        moved = max(moved, _moved(d0, _nonbonded_d(p1, ff)))
        tested += 1
    if not tested:
        for k in ('group_rigid_bond', 'group_rigid_angle', 'group_perturbation_moved'):
            led.skip(k, name, level,
                     'every torsion group on this molecule is keyed on a RING bond, where a '
                     'rotation legitimately opens the closure and rigidity does not hold',
                     K_MOL)
        return
    led.note('group_rigid_tested', name, level, tested)
    led.check('group_rigid_bond', name, level, worst_b, TOL['group_rigid_bond'], units='A')
    led.check('group_rigid_angle', name, level, worst_a, TOL['group_rigid_angle'], units='rad')
    # the positive control on the SAME displacement the two lines above call invariant
    led.check('group_perturbation_moved', name, level, moved, TOL['perturbation_moved'],
              '>=', units='A, largest nonbonded pair-distance change under the group shift')


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
            led.skip('MOLECULE', name, level, f'construction refused: {msg}', K_MOL)
            return {'name': name, 'smiles': smiles, 'skipped': msg}
        raise

    from mxtaltools.conformers.builder import build

    # ---------------------------------------- LEG 0: structural invariants, before any draw
    apply_structural_injections(en, inject)
    _structure_leg(en, name, level, led)
    _group_rigid(en, name, level, led, seed, inject)
    _prior_key_external(en, name, level, led)

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

    # THE POSITIVE CONTROL ON THE DRAW, and it is the one check here that generalises over
    # the whole residual-ceiling class. Every roundtrip_*, external_*, rigid_*, mmff_* and
    # batch_invariance bar is a ceiling on an ERROR, and an error is smallest when nothing
    # happened -- a dof_from_state that ignores x satisfies all of them EXACTLY. Nothing
    # downstream can distinguish "the pipeline is correct" from "the pipeline is dead",
    # because in both cases the residual is zero. This asserts that the batch is not a
    # single conformer repeated n times.
    led.check('draw_response', name, level,
              float(pos.reshape(n, en.spec.n_atoms, 3).std(dim=0).pow(2).sum(-1)
                    .sqrt().max()),
              TOL['draw_response'], '>=',
              units='A, max over atoms of the positional std across the batch')

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
        _ele_pairs_external(en, name, level, led, n, inject)
    else:
        _skip_mmff(led, name, level,
                   f"force_field={ff_choice!r} is not MMFF94, so RDKit's MMFF is not a "
                   f"reference for it and the whole external ENERGY leg is unavailable. The "
                   f"external GEOMETRY checks above still ran, and they are the ones that "
                   f"do not depend on the force field at all", K_CONFIG)
        led.skip('ele_pairs_charged', name, level,
                 f"force_field={ff_choice!r} carries no partial charges at all, so there is "
                 f"no charge assignment to compare against RDKit's", K_CONFIG)

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
                     f"EVERY molecule", K_MOL)
            continue
        val = float(comp[k].median()) / counts[k] / T
        per_term[k] = val
        if k == 'electrostatic':
            # THE ONE PLACE A CEILING BECOMES A BAND. A vanishing term is invisible to a
            # ceiling: drop the electrostatic pathway and abs(val) goes to exactly 0, which
            # abs(val) <= bar passes. The floor is CONDITIONAL because on most of this set
            # MMFF charges only the heteroatom, its hydrogen and the attached carbon, every
            # charged pair is then 1-2 or 1-3 and excluded from the nonbonded list, and the
            # nonbonded electrostatic energy is exactly zero FOR CORRECT CODE -- RDKit's own
            # Ele term returns exactly zero on the same geometries. An unconditional floor
            # would false-fire there, which is worse than no floor.
            if float(ff.ele_qq.abs().max()) <= 0.0:
                led.skip('kt_per_electrostatic', name, level,
                         'MMFF assigns partial charge only to atoms whose every charged '
                         'nonbonded partner is 1-2 or 1-3 on this molecule, so every '
                         'charged pair is excluded from the nonbonded list and the '
                         "electrostatic energy is exactly zero by construction. RDKit's "
                         'own Ele term agrees at exactly zero. There is nothing here for a '
                         'floor to bound, and a floor calibrated to pass this molecule '
                         'would not be a floor. ele_pairs_charged and term_energy_live '
                         'still cover the charge assignment on this molecule', K_MOL)
                continue
            led.band('kt_per_electrostatic', name, level, abs(val),
                     TOL['kt_per_electrostatic_lo'], TOL['kt_per_electrostatic'],
                     units='kT/term')
            continue
        led.check(f'kt_per_{k}', name, level, abs(val), TOL[f'kt_per_{k}'], units='kT/term')

    _term_liveness(en, name, level, led, ff, comp, counts)

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
                 'bond that could be violated', K_MOL)

    if dstats['clip_frac'] is None:
        led.skip('clip_frac', name, level,
                 f"level {level!r} frees no r or theta column, so no state block is "
                 f"non-periodic: phi wraps and there is no box to pile on. "
                 f"ConformerTorsions skips the bounding-energy term for the same reason, "
                 f"which is what keeps this level bitwise identical to the pre-ladder code",
                 K_LEVEL)
    else:
        led.check('clip_frac', name, level, dstats['clip_frac'], TOL['clip_frac'])

    _t_eff(en, name, level, led, ff, pos, n, T, ff_choice)
    _t_eff_angle(en, name, level, led, ff, pos, n, T)

    # worst nonbonded overlap, in EXCESS of the reference conformer's own
    worst = _worst_overlap(pos, ff, n)
    _, ff1 = en._batch(1)
    ref_pos = build(*(_batch1_dof(en)))
    ref_worst = float(_worst_overlap(ref_pos, ff1, 1)[0])
    # over an empty pair list clash_excess_p10 is 0 - 0, and a molecule with no nonbonded
    # pairs at all is not a molecule this pipeline can be trusted on
    led.check('n_clash_pairs', name, level, len(ff.pair_index) // n, TOL['population_min'],
              '>=', units='nonbonded pairs the overlap statistic is computed over')
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

    # --------------------------- LEG 4: sample_prior_states' own draw, then prior_diagnostics
    _prior_draw_leg(en, name, level, led, prior, prior_n, seed, inject)
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
        for k in ('rigid_bond', 'rigid_angle', 'rigid_perturbation_moved'):
            led.skip(k, name, level,
                     'no rotatable (bridge, heavy-fragment) bond on this molecule, so '
                     'there is no rigid rotation to test', K_MOL)
        return
    nb = 8
    rng = np.random.default_rng(seed + 7)
    x0, _ = draw_states(en, nb, rng)
    cols = column_rows(en)
    mask = _np(en.mask)
    n0 = en.n_r + en.n_th
    scale = _np(en._free_scale)
    tree, ff = en._batch(nb)
    p0 = en.build_positions(x0)
    b0, a0 = graph_geometry(p0, ff)
    d0 = _nonbonded_d(p0, ff)

    worst_b = worst_a = 0.0
    moved = 0.0
    tested = 0
    for jb in range(len(en.rotatable)):
        rows = {n0 + int(k) for k in np.flatnonzero(mask[:, jb] != 0)}
        drive = [c for c in range(en.data_ndim) if rows & set(cols[c])]
        driven = set().union(*[set(cols[c]) for c in drive]) if drive else set()
        if not rows.issubset(driven):
            continue                       # this bond is not fully driven at this level
        delta = 0.0 if 'null-perturbation' in inject else float(rng.uniform(0.4, 2.0))
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
        moved = max(moved, _moved(d0, _nonbonded_d(p1, ff)))
        tested += 1
    if not tested:
        for k in ('rigid_bond', 'rigid_angle', 'rigid_perturbation_moved'):
            led.skip(k, name, level,
                     'no rotatable bond had ALL of its dihedrals driven by state columns '
                     'at this level, so no shift here is a whole rotation', K_LEVEL)
        return
    led.note('rigid_bonds_tested', name, level, tested)
    led.check('rigid_bond', name, level, worst_b, TOL['rigid_bond'], units='A')
    led.check('rigid_angle', name, level, worst_a, TOL['rigid_angle'], units='rad')
    # the positive control on the SAME rotation the two lines above call invariant. Without
    # it a builder that ignores its dihedral argument satisfies both of them exactly.
    led.check('rigid_perturbation_moved', name, level, moved, TOL['perturbation_moved'],
              '>=', units='A, largest nonbonded pair-distance change under the bond rotation')


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
    compared = 0
    for s in range(min(n_sub, pos_np.shape[0])):
        compared += 1
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
    # The three residuals above start at 0.0 and are updated inside a loop over a SUBSET.
    # With an empty subset they report exactly 0.0 and pass -- three vacuous passes per
    # molecule that look identical to three perfect ones. The population floor is what
    # separates them; there is no bar on a residual that can.
    led.check('n_external_geom', name, level, compared, TOL['population_min'], '>=',
              units='conformers actually re-measured by rdMolTransforms')
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
        for k in ('chirality', 'n_chirality_conformers'):
            led.skip(k, name, level,
                     'no specified stereocentre in the SMILES, so there is no CIP code to '
                     'recover and a reflection is not observable on this molecule', K_MOL)
        return
    N = en.spec.n_atoms
    pos_np = _np(pos).reshape(-1, N, 3)
    bad, got_any, checked = 0, None, 0
    for s in range(min(n_sub, pos_np.shape[0])):
        m = to_rdkit_mol(en, pos_np[s], inject)
        Chem.AssignStereochemistryFrom3D(m)
        got = {a.GetIdx(): a.GetProp('_CIPCode') for a in m.GetAtoms()
               if a.HasProp('_CIPCode')}
        got_any = got
        bad += int(got != want)
        checked += 1
    # zero conformers re-perceived is zero mismatches, and the only check in the file that
    # can see a reflection then passes without looking at anything
    led.check('n_chirality_conformers', name, level, checked, TOL['population_min'], '>=',
              units='conformers RDKit re-perceived the stereocentre from')
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
    compared = 0
    for s in range(n_sub):
        p1 = torch.as_tensor(pos_np[s], dtype=en.dtype, device=en.device)
        tot, comp = intramolecular_energy(tree1, p1, ff1, components=True)
        mol = to_rdkit_mol(en, pos_np[s], inject)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            _skip_mmff(led, name, level, 'RDKit could not MMFF-type this molecule',
                       K_UNREACHABLE)
            return
        compared += 1
        for key, rd in RDKIT_TERMS.items():
            for u in ALL_RDKIT:
                getattr(props, f'SetMMFF{u}Term')(u == rd)
            ref = AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()
            worst[key] = max(worst[key], abs(float(comp[key].sum()) - ref))
        for u in ALL_RDKIT:
            getattr(props, f'SetMMFF{u}Term')(True)
        ref_tot = AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()
        worst_total = max(worst_total, abs(float(tot.sum()) - ref_tot))
    # eight residual ceilings, all initialised to 0.0 and all updated only inside the loop:
    # with an empty subset every one of them reports 0.0 and passes
    led.check('n_external_mmff', name, level, compared, TOL['population_min'], '>=',
              units="conformers actually scored by RDKit's MMFF94")
    for key in RDKIT_TERMS:
        led.check(f'mmff_{key}', name, level, worst[key], TOL[f'mmff_{key}'],
                  units='kcal/mol')
    led.check('mmff_total', name, level, worst_total, TOL['mmff_total'], units='kcal/mol')


MMFF_CHECKS = tuple(f'mmff_{k}' for k in RDKIT_TERMS) + ('mmff_total', 'n_external_mmff')


def _skip_mmff(led, name, level, reason, kind):
    """Skip the external MMFF leg BY NAME, not as one summary row.

    A single ``mmff_* (7 terms + total)`` row is one skip standing in for nine, and the
    eight names it stands in for then emit no row at all -- invisible to a skip census,
    which is the failure this file is currently being audited for.
    """
    for k in MMFF_CHECKS:
        led.skip(k, name, level, reason, kind)


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
                 f"reporting it as one would be a fabricated measurement", K_LEVEL)
        return
    locked = ring_locked(en)
    bi = np.asarray(en.spec.bond_index)
    keep = [j for j in range(en.n_r) if free_r[j] and j not in locked]
    if not keep:
        led.skip('T_eff', name, level,
                 'every free bond is ring-locked and therefore drawn at a fraction of its '
                 'thermal width, so equipartition does not apply to any of them', K_MOL)
        return
    want = {frozenset((int(bi[j, 0]), int(bi[j, 1]))) for j in keep}
    n_at = en.spec.n_atoms
    ffbi = _np(ff.bond_index) % n_at
    sel = np.array([frozenset((int(a), int(b))) in want for a, b in ffbi])
    if not sel.any():
        led.skip('T_eff', name, level,
                 'no force-field bond term maps onto a free non-ring tree bond', K_MOL)
        return
    kb = ff.k_bond.clone()
    kb[torch.as_tensor(~sel, device=kb.device)] = 0.0
    masked = dataclasses.replace(ff, k_bond=kb)
    tree, _ = en._batch(n)
    _, comp = intramolecular_energy(tree, pos, masked, components=True)
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
                 "to assign it to", K_CONFIG)


def _t_eff_angle(en, name, level, led, ff, pos, n, T):
    """2 <E_ang> / (n_angle kT) over FREE, NON-RING angles. Exactly 1 by equipartition.

    THIS IS WHERE THE FLOOR ON THE ANGLE TERM BELONGS, and the reason it does not go on
    ``kt_per_angle`` is worth stating. ``kt_per_angle`` is a ceiling on a magnitude, so a
    defect that drives the angle term toward zero -- a lost force constant, a lost factor
    of T, a theta width collapsed by a factor -- passes it comfortably. But a FLOOR on
    ``kt_per_angle`` false-fires on correct code: at ``level='torsion'`` every theta row is
    frozen, so the quantity is a constant in x rather than a measurement of the draw, and
    under ``force_field='reference'`` at that level it is machine zero because the reference
    force field's theta0 IS the reference conformer's own angle.

    An equipartition band does not have that problem, because it is only defined where the
    theta block is free, and it SKIPS everywhere a floor would have lied. It is also
    independent of the width the draw was generated from, which a check that scores the
    realised draw against ``thermal_rtheta_sigma`` is not: that comparison uses the same
    function on both sides and cannot see a factor lost inside it.

    Isolation is by masking ``k_angle`` and calling the shipped ``intramolecular_energy``
    again, exactly as ``_t_eff`` masks ``k_bond`` -- not by reimplementing the functional
    form, which would be one more copy that can drift from the code it checks. Angles at a
    LINEAR centre are already excluded upstream: ``ConformerTorsions`` holds those rows as
    parameterisation singularities, so they are never in ``free_mask``.
    """
    from mxtaltools.conformers.energy import intramolecular_energy
    free_th = np.asarray(en.free_mask)[en.n_r:en.n_r + en.n_th]
    if not free_th.any():
        led.skip('T_eff_angle', name, level,
                 f"level {level!r} freezes every theta column at the reference conformer, "
                 f"so the angle term is a CONSTANT in x. 2E/(n kT) there is the reference "
                 f"conformer's own strain against the typed theta0, not a temperature. "
                 f"This is the level that ships, so the angle term's only equipartition "
                 f"instrument is dark on it", K_LEVEL)
        return
    locked = ring_locked(en)
    ai = np.asarray(en.spec.angle_index)
    keep = [j for j in range(en.n_th) if free_th[j] and (en.n_r + j) not in locked]
    if not keep:
        led.skip('T_eff_angle', name, level,
                 'every free angle is ring-locked and therefore drawn at a fraction of its '
                 'thermal width, so equipartition does not apply to any of them', K_MOL)
        return
    want = {(int(ai[j, 1]), frozenset((int(ai[j, 0]), int(ai[j, 2])))) for j in keep}
    n_at = en.spec.n_atoms
    ffai = _np(ff.angle_index) % n_at
    sel = np.array([(int(b), frozenset((int(a), int(c)))) in want for a, b, c in ffai])
    if not sel.any():
        led.skip('T_eff_angle', name, level,
                 'no force-field angle term maps onto a free non-ring tree angle', K_MOL)
        return
    ka = ff.k_angle.clone()
    ka[torch.as_tensor(~sel, device=ka.device)] = 0.0
    masked = dataclasses.replace(ff, k_angle=ka)
    tree, _ = en._batch(n)
    _, comp = intramolecular_energy(tree, pos, masked, components=True)
    n_sel = int(sel.sum()) // n
    led.note('T_eff_angle_n', name, level, n_sel)
    led.band('T_eff_angle', name, level, 2.0 * float(comp['angle'].mean()) / (n_sel * T),
             TOL['T_eff_angle_lo'], TOL['T_eff_angle_hi'], units='T_eff/T')


# The parameter tensor whose being non-degenerate is what makes each force-field term LIVE.
# Used by term_energy_live, which is an IDENTITY rather than a threshold: a term with live
# parameters must produce a non-zero energy somewhere in the batch, and a term with no live
# parameters must not.
TERM_PARAMS = {'bond': ('k_bond',), 'angle': ('k_angle',),
               'stretch_bend': ('sb_k_ijk', 'sb_k_kji'), 'oop': ('oop_k',),
               'torsion': ('tors_v',), 'lj': ('epsilon',), 'electrostatic': ('ele_qq',)}


def _term_liveness(en, name, level, led, ff, comp, counts):
    """Every force-field term with live PARAMETERS must produce a live ENERGY, and vice versa.

    This is the general form of the electrostatics finding, and it is the check that
    actually closes it. ``kt_per_electrostatic`` is a magnitude bound, so it needs a
    calibrated floor, it needs a guard against the molecules where zero is correct, and the
    guard turns it into a skip on exactly the defect that damages the parameters. This
    needs none of that: it compares two booleans, so there is no bar to calibrate, it is
    meaningful on every molecule at every level, and it holds under
    ``force_field='reference'`` where the whole external MMFF leg is unavailable and nothing
    else covers the question at all.

    It catches the "a whole force-field term went missing" class named in the module
    docstring from either end -- a dropped energy pathway with the parameters intact, or
    dropped parameters -- and it does so without a draw-dependent threshold.
    """
    bad = []
    for term, fields in TERM_PARAMS.items():
        params = [getattr(ff, f, None) for f in fields]
        live_p = any(t is not None and t.numel() and float(t.abs().max()) > 0.0
                     for t in params)
        e = comp.get(term)
        live_e = e is not None and e.numel() > 0 and float(e.abs().max()) > 0.0
        if live_p != live_e:
            bad.append(f'{term}(params={"live" if live_p else "dead"},'
                       f'energy={"live" if live_e else "dead"})')
    led.check('term_energy_live', name, level, len(bad), 0, '==',
              units='terms whose parameters and energy disagree about being live'
                    + (': ' + ' '.join(bad) if bad else ''))


def _ele_pairs_external(en, name, level, led, n, inject):
    """Charged nonbonded pairs, OURS against RDKit's own MMFF partial charges.

    A COUNT, not a magnitude, and that is the whole point. On most of this molecule set the
    nonbonded electrostatic ENERGY is exactly zero for correct code, so every energy-based
    instrument -- ours and RDKit's alike -- is identically zero and can bound nothing. The
    charge assignment is still there to be checked, and it is checked here: the same
    nonbonded pair list, scored for charge from RDKit's ``GetMMFFPartialCharge`` instead of
    from our typing. No draw, no calibration, no threshold.

    Complementary to the ``kt_per_electrostatic`` floor rather than redundant with it: that
    one catches a dropped energy PATHWAY with the parameters intact, this one catches
    dropped PARAMETERS, where the floor's own guard correctly turns it into a skip.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _, ff = en._batch(n)
    if ff.ele_qq is None or ff.pair_index is None or not ff.pair_index.numel():
        led.skip('ele_pairs_charged', name, level,
                 'the force field carries no nonbonded pair list on this molecule, so '
                 'there are no charged pairs to count', K_MOL)
        return
    n_at = en.spec.n_atoms
    npair = ff.pair_index.shape[0] // n
    pi = _np(ff.pair_index[:npair]) % n_at
    qq = ff.ele_qq[:npair]
    ours = int((qq.abs() > 0).sum())

    mol = Chem.Mol(en.mol)
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        led.skip('ele_pairs_charged', name, level,
                 'RDKit could not MMFF-type this molecule, so there is no independent '
                 'charge assignment to compare against', K_UNREACHABLE)
        return
    perm = np.asarray(en.spec.perm)
    q = np.array([props.GetMMFFPartialCharge(int(perm[s])) for s in range(n_at)])
    theirs = int(sum(1 for a, b in pi if abs(q[int(a)] * q[int(b)]) > 0))
    led.note('ele_pairs_ours', name, level, ours)
    led.note('ele_pairs_rdkit', name, level, theirs)
    led.check('ele_pairs_charged', name, level, abs(ours - theirs), 0, '==',
              units='disagreement on how many nonbonded pairs carry charge')


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
        for k in PRIOR_REPORT_CHECKS + COVERAGE_CHECKS:
            led.skip(k, name, level,
                     'no fitted InternalPrior available (--no-prior, or the cache file is '
                     'missing). These are the only checks in the file that need a fit',
                     K_CONFIG)
        return
    try:
        rep = pdg.prior_report(en, prior, n=prior_n, seed=seed, n_boot=60)
    except Exception as ex:
        # SKIP THE DOWNSTREAM NAMES TOO. A raise here used to emit ONE skip row and leave
        # the three assertions it feeds emitting NO ROW AT ALL -- neither a pass, nor a
        # skip, nor a failure. That is the shape a census cannot see and a headline count
        # rounds to "verified". They are UNREACHABLE, not inapplicable: nothing about this
        # molecule or this level says the property does not hold, only that no code got to
        # measure it.
        for k in PRIOR_REPORT_CHECKS:
            led.skip(k, name, level, f'{type(ex).__name__}: {ex}', K_UNREACHABLE)
    else:
        # THE MIRROR OF THE CEILING BELOW, and definitional for the same reason. An ESS
        # FRACTION is sum(w)^2 / (n sum(w^2)); Cauchy-Schwarz bounds it above by 1 and the
        # same inequality applied the other way bounds it below by 1/n for any non-negative
        # weights. So this is not a calibrated quality bar -- it cannot false-fire on a
        # correct draw however bad the fit is. What it catches is the object not being a
        # set of importance weights at all. The old 1e-12 caught only exact zero and NaN.
        led.check('prior_report', name, level, int(prior_n), TOL['population_min'], '>=',
                  units='draws the fitted-prior report was computed over')
        led.check('prior_ess_positive', name, level, rep['ess_fitted'],
                  1.0 / max(int(prior_n), 1), '>=',
                  units='ESS fraction; 1/n is the hard floor for non-negative weights')
        led.check('prior_ess_le_one', name, level, rep['ess_fitted'], TOL['prior_ess_max'])
        led.check('prior_clip_frac', name, level, rep['clip_frac'], TOL['clip_frac'])
        led.note('prior_ess_pct', name, level, 100 * rep['ess_fitted'], units='%')
        led.note('prior_eta', name, level, rep['eta'], units='NOT bounded by 1, see docstring')
        led.note('prior_D_avoidable', name, level, rep['D_avoidable'], units='nats')
    try:
        cov = pdg.coverage_report(en, prior, n=prior_n, seed=seed)
    except Exception as ex:
        for k in COVERAGE_CHECKS:
            led.skip(k, name, level, f'{type(ex).__name__}: {ex}', K_UNREACHABLE)
        return
    if 'skipped' in cov:
        for k in COVERAGE_CHECKS:
            led.skip(k, name, level, cov['skipped'], K_MOL)
        return
    led.skip('coverage_missed', name, level,
             'coverage_report RAN and its numbers are in the REPORTED block, but n_missed '
             'carries NO threshold: "accessible" is defined at 10 kT, whose Boltzmann '
             'weight is e^-10 ~ 5e-5, so a CORRECT prior is expected to miss some of those '
             'basins; and there is an empty-bin floor of order n_accessible * '
             'exp(-n / n_accessible) on top. Asserting 0 fails on correct code and any '
             'tolerated count would be a number with no argument behind it', K_UNASSERTED)
    # coverage over zero enumerated modes is vacuous: worst_frac and excess_median would be
    # computed over an empty basin set and every number below would still print
    led.check('coverage_report', name, level, cov['n_modes'], TOL['population_min'], '>=',
              units='rotamer basins the coverage statistics are computed over')
    led.note('coverage_n_modes', name, level, cov['n_modes'])
    led.note('coverage_n_accessible', name, level, cov['n_accessible'])
    led.note('coverage_n_missed', name, level, cov['n_missed'])
    led.note('coverage_empty_bin_floor', name, level,
             cov['n_accessible'] * math.exp(-prior_n / max(cov['n_accessible'], 1)),
             units='basins expected empty by chance')
    led.note('coverage_worst_frac_pct', name, level, 100 * cov['worst_frac'], units='%')
    led.note('coverage_excess_median_kt', name, level, cov['excess_median_kt'], units='kT')


def _prior_key_external(en, name, level, led):
    """``prior_dof_types``' per-row keys against the mxtaltools ``tree_*`` derivation.

    NOT a re-derivation of the same expression. ``InternalPrior.fit`` built the fitted
    tables by walking ``mol.tree_bond_index / tree_angle_index / tree_torsion_index`` after
    ``build_conformer_tree()`` (prior.py fit / _layout), so that route is the AUTHORITY for
    what a key means; ``prior_dof_types`` deliberately reaches the same keys through this
    class's ``spec`` instead, and its own docstring names mixing the two numberings as the
    hazard. Comparing the two closes that loop with an independent second implementation.

    Runs off a BARE, unfitted InternalPrior: the key functions are static and the histogram
    lookups are irrelevant here, so this check does not need conformer_prior_v2.pt and is
    therefore available on every invocation, including --no-prior.

    LIMITATION, stated because it bounds what a pass means: keys are (element, degree)
    types, not atom indices, so a scramble that lands on a DIFFERENT atom pair of the SAME
    type is invisible to it. Measured under --inject prior-key-scramble: 1 of 9 phi rows on
    propanol, 15 of 19 on ala-dipeptide, but 11 of 11 MOLECULES fire, which is what decides
    the exit code. A per-index comparison would be strictly stronger, and prior_dof_types
    does not return indices.
    """
    from mxtaltools.conformers.prior import InternalPrior
    from energies.conformer_data import condition_from_energy
    bare = InternalPrior()
    try:
        got = en.prior_dof_types(bare)
        m = condition_from_energy(en, partial_charges=False)
        m.build_conformer_tree()
        keys = InternalPrior._atom_keys(m)
        bi = m.tree_bond_index.detach().cpu().numpy()
        ai = m.tree_angle_index.detach().cpu().numpy()
        ti = m.tree_torsion_index.detach().cpu().numpy()
    except Exception as ex:
        for k in ('prior_key_external', 'n_key_rows'):
            led.skip(k, name, level,
                     f'the mxtaltools tree_* route could not be built for this molecule, so '
                     f'there is no independent authority to compare against: '
                     f'{type(ex).__name__}: {ex}', K_UNREACHABLE)
        return
    # `bad` below is a count of disagreeing rows, so with no rows there is nothing to
    # disagree and the check passes without comparing anything
    led.check('n_key_rows', name, level, len(got), TOL['population_min'], '>=',
              units='DoF rows whose prior key was compared against the tree_* derivation')
    want = {}
    for j in range(bi.shape[1]):
        want[j] = InternalPrior.bond_key(*keys[bi[:, j]])
    for j in range(ai.shape[1]):
        want[en.n_r + j] = InternalPrior.angle_key(*keys[ai[:, j]])
    for j in range(ti.shape[1]):
        # ti[1:3] is the CENTRAL BOND -- the same two columns prior_dof_types must use
        want[en.n_r + en.n_th + j] = InternalPrior.torsion_key(*keys[ti[1:3, j]])
    if len(want) != len(got):
        led.check('prior_key_external', name, level, abs(len(want) - len(got)),
                  TOL['prior_key_external'], '==', units='DoF-count disagreement')
        return
    bad = sum(1 for row in range(len(got)) if got[row][2] != want[row])
    led.check('prior_key_external', name, level, bad, TOL['prior_key_external'], '==',
              units='rows whose key disagrees with the tree_* derivation')


def _capture_prior_dof(en, prior, n, rng, **kw):
    """``(r, theta, phi)`` as ``sample_prior_states`` actually emitted them, at EVERY level.

    ``sample_prior_states`` ends by calling ``state_from_dof``, which RAISES at `torsion`
    because the state -> DoF map is collective there. That single line is why the whole
    prior-draw leg has been unavailable at the only level that ships, and why a regression
    inside the draw itself has had no detector at all there. The DoF vector is fully formed
    before that call, so shadowing ``state_from_dof`` on the instance intercepts it: the spy
    records the DoF, forwards to the real method where it works, and hands back a zero state
    where it does not. Nothing is reimplemented -- the code under test is the shipped
    function, improper draw, group draw, ring handling and all.
    """
    box = {}
    real = en.state_from_dof

    def spy(r, th, ph):
        box['dof'] = (_np(r).copy(), _np(th).copy(), _np(ph).copy())
        try:
            return real(r, th, ph)
        except NotImplementedError:
            return torch.zeros(r.shape[0], en.data_ndim, dtype=en.dtype, device=en.device)

    en.state_from_dof = spy
    try:
        en.sample_prior_states(prior, n, rng, report=False, **kw)
    finally:
        del en.state_from_dof                       # restore the class method
    return box['dof']


def _prior_draw_leg(en, name, level, led, prior, n, seed, inject=()):
    """Width checks on ``sample_prior_states``' own draw. Available at EVERY level.

    The rest of the file samples ``draw_states`` in this module, which is a thermal /
    rotameric draw built from the pipeline's primitives -- deliberately, so the harness has
    a sample source without needing a fit. The consequence is that a regression INSIDE
    ``sample_prior_states`` (the function that feeds train_conformer.py) is seen by nothing
    except leg 4, and leg 4 is unavailable at `torsion`. This leg closes that: it drives the
    shipped function and asserts the one thing about it that theory fixes rather than bounds.
    """
    if prior is None:
        for k in ('prior_rtheta_width', 'prior_improper_sigma'):
            led.skip(k, name, level,
                     'no fitted InternalPrior available (--no-prior, or the cache file is '
                     'missing), and sample_prior_states cannot be driven without one',
                     K_CONFIG)
        return
    kw = {'thermal_rtheta': False} if 'prior-pooled-rtheta' in inject else {}
    try:
        r, th, ph = _capture_prior_dof(en, prior, n, np.random.default_rng(seed + 31), **kw)
    except Exception as ex:
        for k in ('prior_rtheta_width', 'prior_improper_sigma'):
            led.skip(k, name, level, f'sample_prior_states raised: {type(ex).__name__}: {ex}',
                     K_UNREACHABLE)
        return

    T = float(en.temperature)
    s_r, s_th = en.thermal_rtheta_sigma(T)
    s_imp = en.improper_phi_sigma(T)
    n0 = en.n_r + en.n_th

    # Rows a ring system OWNS are rattled at ring_jitter_scale (0.1) of thermal width on
    # purpose -- closure is nonlinear and full-width independent draws open the ring -- so
    # equipartition does not apply to them and they are excluded. The set is taken from
    # ring_blocks, the same source sample_prior_states uses, rather than guessed.
    held = set()
    try:
        for order, _bank, extra in en.ring_blocks(prior):
            for kind, j in list(order) + list(extra):
                held.add(_global_row(en, kind, j))
    except Exception:
        held = set(ring_locked(en))

    fold = lambda q: max(q, 1.0 / q) if q > 0 else float('inf')
    q = []
    for j in range(en.n_r):
        if j not in held:
            q.append(fold(float(r[:, j].std()) / s_r[j]))
    for j in range(en.n_th):
        if en.n_r + j not in held:
            q.append(fold(float(th[:, j].std()) / s_th[j]))
    if not q:
        led.skip('prior_rtheta_width', name, level,
                 'every r and theta row on this molecule is owned by a ring system and is '
                 'therefore held at a fraction of thermal width by design', K_MOL)
    else:
        led.check('prior_rtheta_width', name, level, max(q), TOL['prior_rtheta_width'],
                  units='fold deviation from the FF thermal sigma')

    ph0 = _np(en.ph0)
    imp = [j for j in en.improper_phi_rows() if n0 + j not in held]
    if not imp:
        led.skip('prior_improper_sigma', name, level,
                 'no improper phi row outside a ring block on this molecule (an improper '
                 'row inside one is held by the ring, not drawn)', K_MOL)
        return
    qi = []
    for j in imp:
        d = ph[:, j] - ph0[j]
        # circular std about the reference: sqrt(-2 ln |mean exp(i d)|). A rotamer draw
        # spreads over the whole circle and lands near 1.8 rad however it is measured; a
        # linear std would wrap and understate it.
        R = float(np.abs(np.mean(np.exp(1j * d))))
        circ = math.sqrt(max(-2.0 * math.log(max(R, 1e-300)), 0.0))
        qi.append(fold(circ / s_imp) if s_imp > 0 else float('inf'))
    led.check('prior_improper_sigma', name, level, max(qi), TOL['prior_improper_sigma'],
              units='fold deviation from improper_phi_sigma')


def chiral_pair_check(led, level, results_cip):
    """The two enantiomers must build to OPPOSITE CIP codes from identical machinery.

    A single molecule's CIP check catches a reflection applied to everything. This catches
    the narrower failure where the builder ignores the input stereochemistry entirely and
    emits the same hand for both SMILES -- which would leave each molecule's own check
    passing on one of the two.
    """
    a, b = results_cip.get('butan-2-ol-R'), results_cip.get('butan-2-ol-S')
    if not a or not b:
        led.skip('chiral_pair_opposite', 'butan-2-ol', level,
                 'one or both members of the chiral pair did not run at this level', K_MOL)
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
    # A non-finite VALUE is a legitimate outcome -- Ledger.check scores it FAIL -- so it has
    # to survive formatting. It used to raise here and take the whole report with it, which
    # turned "the harness caught something" into "the harness crashed before printing".
    if not isinstance(v, (int, np.integer)) and not math.isfinite(float(v)):
        return f'{float(v)}'
    if isinstance(v, (int, np.integer)) or float(v) == int(float(v)):
        if abs(float(v)) < 1e6:
            return f'{int(float(v))}'
    a = abs(float(v))
    return f'{float(v):.4g}' if 1e-3 <= a < 1e5 else f'{float(v):.3e}'


def skip_census(led, summaries):
    """Which of the suite ran, which did not, and WHY -- as numbers rather than as prose.

    "N passed, M skipped, 0 FAILED" is unreadable without this. M mixes three unrelated
    things: a property this molecule does not have, a property this LEVEL froze out of
    existence, and a code path that RAISED. Only the third is a hole, and it is the one a
    headline count silently rounds into the second.

    The fourth number is the one nothing else in the file could see: a check name that
    emitted NO ROW AT ALL, which is what happens downstream of a raise unless the raise
    handler skips the downstream names explicitly. It is neither a pass nor a skip, so it
    is invisible to both counts. Diffing the emitted names against ``ASSERTIONS`` is what
    makes it a printed integer, and the diff runs the other way too so the list cannot rot.
    """
    ran_mols = [s['name'] for s in summaries if 'skipped' not in s]
    emitted = {(r.name, r.mol) for r in led.rows if r.status in ('pass', 'FAIL', 'skip')}
    dark = [(nm, m) for m in ran_mols for nm in ASSERTIONS if (nm, m) not in emitted]
    known = set(ASSERTIONS) | set(NON_ASSERTION_ROWS)
    unregistered = {r.name for r in led.rows
                    if r.status in ('pass', 'FAIL', 'skip') and r.name not in known}
    by_kind = defaultdict(int)
    for r in led.skips:
        by_kind[r.kind] += 1
    n_skip = len(led.skips)
    n_ran = sum(r.status in ('pass', 'FAIL') for r in led.rows)
    tier = [r for r in led.rows
            if r.name in PRIOR_QUALITY_TIER and r.status in ('pass', 'FAIL', 'skip')]
    return {
        'n_slots': n_ran + n_skip + len(dark),
        'n_ran': n_ran,
        'n_skip': n_skip,
        'by_kind': dict(by_kind),
        'n_inapplicable': sum(by_kind.get(k, 0) for k in (K_MOL, K_LEVEL, K_CONFIG)),
        'n_unreachable': by_kind.get(K_UNREACHABLE, 0),
        'dark': dark,
        'unregistered': sorted(unregistered),
        'tier_ran': sum(r.status in ('pass', 'FAIL') for r in tier),
        'tier_slots': len(tier) + sum(1 for nm, _m in dark if nm in PRIOR_QUALITY_TIER),
    }


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
      f"{'E/T median':>13}{'pass':>7}{'inapp':>7}{'UNRE':>6}{'FAIL':>6}   hazard")
    for s in summaries:
        by = defaultdict(int)
        for r_ in led.rows:
            if r_.mol == s['name']:
                by[r_.status] += 1
                if r_.status == 'skip':
                    by['UNRE' if r_.kind == K_UNREACHABLE else 'inapp'] += 1
        if 'skipped' in s:
            p(f"  {s['name']:<18}{s['smiles']:<22}{'-':>4}{'-':>5}{'-':>5}{'-':>5}"
              f"{'SKIPPED':>13}{by['pass']:>7}{by['inapp']:>7}{by['UNRE']:>6}"
              f"{by['FAIL']:>6}   {s['skipped']}")
            continue
        p(f"  {s['name']:<18}{s['smiles']:<22}{s['n_atoms']:>4}{s['d']:>5}"
          f"{s['n_rotatable']:>5}{s['n_linear_angle']:>5}"
          f"{s['energy_median']:>13.2f}{by['pass']:>7}{by['inapp']:>7}{by['UNRE']:>6}"
          f"{by['FAIL']:>6}   {s['hazard'][:40]}")
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

    # ---- SKIP CENSUS ---------------------------------------------------------------
    cen = skip_census(led, summaries)
    p('SKIP CENSUS  (a check that did not run is not a check that passed)')
    p(f"  {cen['n_ran']} of {cen['n_slots']} check x molecule slots RAN; "
      f"{cen['n_skip']} did not "
      f"({100.0 * cen['n_skip'] / max(cen['n_slots'], 1):.0f}% of the suite is dark at "
      f"level {cfg['level']!r}).")
    p(f"    {cen['n_inapplicable']:>4}  legitimately INAPPLICABLE -- the property does not "
      f"exist to be checked")
    for k in (K_MOL, K_LEVEL, K_CONFIG):
        if cen['by_kind'].get(k):
            p(f"    {cen['by_kind'][k]:>4}    {k}")
    p(f"    {cen['by_kind'].get(K_UNREACHABLE, 0):>4}  UNREACHABLE -- a code path RAISED. "
      f"This is the number that must NOT be read as 0 FAILED.")
    p(f"    {cen['by_kind'].get(K_UNASSERTED, 0):>4}  ran, deliberately UNASSERTED "
      f"(no defensible bar; see the reason text)")
    if cen['dark']:
        # a name that emitted NO row of any kind on a molecule that ran: invisible to both
        # the pass count and the skip count. The whole census is worthless if this grows.
        p(f"  !! {len(cen['dark'])} check x molecule slots emitted NO ROW AT ALL -- not a "
          f"pass, not a skip, not a failure:")
        byname = defaultdict(list)
        for nm, m in cen['dark']:
            byname[nm].append(m)
        for nm, mols in sorted(byname.items()):
            p(f"       {nm}  [{len(mols)}: {', '.join(mols)}]")
    else:
        p(f"     {0:>3}  emitted NO ROW at all (every listed assertion produced a pass, a "
          f"skip or a failure)")
    if cen['unregistered']:
        p(f"  !! emitted but absent from ASSERTIONS, so the census cannot account for it: "
          f"{', '.join(sorted(cen['unregistered']))}")
    tier_ran, tier_slots = cen['tier_ran'], cen['tier_slots']
    p(f"  PRIOR-QUALITY TIER ({', '.join(PRIOR_QUALITY_TIER)}):")
    p(f"     {tier_ran} of {tier_slots} slots ran"
      + ('   <-- ZERO. The tier this harness exists to supply contributes nothing to the '
         'pass count at this level.' if tier_ran == 0 else ''))
    p()

    p('SKIPPED, BY REASON')
    groups = defaultdict(list)
    for r_ in led.skips:
        groups[(r_.kind, r_.name, r_.reason)].append(r_.mol)
    if not groups:
        p('  none')
    for (kind, nm, reason), mols in sorted(groups.items()):
        p(f"  [{kind}]  {nm}  [{len(mols)}: {', '.join(mols)}]")
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
    chiral_pair_check(led, level, cips)

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
                    help='re-introduce a known bug and require the harness to FAIL. "all" '
                         'makes ONE PASS PER BUG rather than one combined pass')
    ap.add_argument('--json', default=None, help='write the full record here')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(argv)

    torch.set_num_threads(a.threads)
    go = lambda inj: run(level=a.level, force_field=a.force_field, n=a.n, seed=a.seed,
                         n_external=a.n_external,
                         prior_path=None if a.no_prior else a.prior_path,
                         prior_n=a.prior_n, only=a.only, inject=inj)

    if 'all' in a.inject:
        # ONE PASS PER INJECTION, not one pass with all of them. Combining them is worse
        # than useless: three of them rewrite torsion_groups and only the last would apply,
        # and the ones that do compose bury each other's evidence -- a combined run leaves a
        # dozen checks failing "incidentally", which is exactly the state in which a dead
        # detector is invisible. Per pass, the question is answerable: did THIS bug's named
        # detector fire.
        print('=' * 96)
        print(f'prior_smoke --inject all  --  {len(INJECTIONS)} passes, one per bug class, '
              f'level {a.level!r}, n={a.n}')
        print('=' * 96)
        ok_all = True
        for b in sorted(INJECTIONS):
            cfg, led, summaries = go((b,))
            fired = {r.name for r in led.failures}
            targets = [t.strip() for t in INJECTIONS[b][0].split('/')]
            hit = [t for t in targets if t in fired]
            ok_all &= bool(hit)
            mols = sorted({r.mol for r in led.failures if r.name in hit})
            extra = sorted(fired - set(targets))
            print(f"  {b:<22} target {INJECTIONS[b][0]:<46} "
                  f"{'CAUGHT on %d/%d molecules' % (len(mols), len(summaries)) if hit else 'NOT CAUGHT -- BLIND'}"
                  f"  ({cfg['seconds']:.0f}s)")
            if hit:
                print(f"    {'':<22} by {', '.join(hit)}")
            if extra:
                print(f"    {'':<22} also fired, incidentally: {', '.join(extra)}")
        print('-' * 96)
        print(f"  VERDICT: {'PASS -- every named detector fired' if ok_all else 'FAIL -- BLIND'}")
        print('-' * 96)
        return 0 if ok_all else 1

    inject = tuple(a.inject)
    cfg, led, summaries = go(inject)
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
