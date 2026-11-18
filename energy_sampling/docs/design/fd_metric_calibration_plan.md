# Broadening the FD metric's calibration: plan

- **Status:** RAN 2026-11-18 (owner go): record `docs/results/fd-metric-day-calibration-2026-11-18.md`. Simplified the same day at the owner's request. It has
  no authority over code, config or tests.
- **Question:** does the glued FD distance (MXtalTools `crystal_fd/metric.py::GluedFDMetric`, MXT `073b821d`) track
  COMPACK beyond the two molecules it was calibrated on, in every crystal family?
- **Today:** it reads 1.17× / 1.30× COMPACK RMSD, with Spearman 0.997 / 0.981, and every pair below 0.3 Å was a match.
  That rests on MIPCAS (P-1) and NEHZOR (P2₁/c) only (`docs/design/fd_si_draft.md` §3.2).

## The experiment

1. **Molecules.** 20 molecules from Day's FIT+DMA landscapes, drawn at random (seeded) among those with no proper
   point symmetry (C₁, or planar C_s, which the twin already handles).
2. **Groups.** Each molecule in up to 3 of its landscape groups, chosen so that the panel covers triclinic,
   monoclinic, orthorhombic and higher families. That gives about 50 (molecule, group) sets.
3. **Pairs.** 12 per set, about 600 in all:
   - 6 **perturbed**: one low-energy minimum against copies of itself moved rigidly, in centroid, orientation and
     cell, at 6 amplitudes chosen so the FD distance spans about 0.01–1 Å;
   - 6 **random**: random pairs of distinct minima within 10 kJ/mol.
4. **Score.** COMPACK on every pair, using the repaired 20-molecule route already in `eval/fd_analysis/common.py`.
   Plus the FD distance with the twin.
5. **Read** one table: per crystal family, P(match | d), the scale (FD / COMPACK RMSD on matches), Spearman, and
   whether every pair below 0.05 Å matched.

**Cost:** about 600 × 1.5 s ≈ 15 min of COMPACK on one worker. There is one new script: a Day pair builder that reuses
attack A's proven CIF parser and pose fit.

**Pass:** in every family, all pairs below 0.05 Å match, and the scale's median lies within the 1.17–1.30 already
measured, ± 0.3.

## Later, only if this passes and the paper needs it

- molecules with a proper rotation axis, which first needs the point-symmetry quotient in the metric;
- AMD/PDD and RDF on the same pairs, for positioning;
- our own search minima (`pool8sg`) as a second population.
