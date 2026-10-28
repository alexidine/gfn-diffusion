# acr_foundation_oct26: acridine with foundation energy models

Owner, 2026-10-26: redo acridine with one or more foundation models in place of the fine-tuned acridine MACE checkpoints.
Order agreed: (1) check candidate models on the known polymorphs; (2) per model, gas-phase conformer and the 1 kT kick /
identity-cut calibration; (3) re-relax the pooled existing states (old priors, the September campaigns, the October walk)
under the model, on the cluster; (4) a short seeded exploration with a stopping rule; (5) walk, assembly, then the
autoregressive flow job.

## 1. Model check (`submit_model_check.sbatch`)

`acr_model_check.py` relaxes, per model, the isolated molecule and each of the ten H-bearing acridine CIFs (atoms in the
experimental cell, then atoms and cell), and scans a rigid-molecule expansion of each relaxed crystal; `summarize.py`
prints every model's form energies relative to Form II (ACRDIN04, ACRDIN11). Models: MACE-OFF24 medium, UMA 1.1 small and
medium (task `omc`), UMA 1.2 small when `/scratch/mk8347/models/uma/uma-s-1p2p1.pt` exists, and the two fine-tuned
acridine MACE checkpoints for reference. MACE models run in float32 with energies read as node-energy sums: MACE-OFF's
float32 totals (about -1.46e6 kJ/mol per molecule, DFT atomic reference energies included) jitter by about 1 kJ/mol per
molecule under a rigid translation of the crystal, the node sums by 0.003.

Dev-box results so far (2026-10-26, RTX 5080 laptop, not the job's): UMA 1.1 small best lattice energy -101.8 kJ/mol
(ACRDIN12), all forms within 1.3 kJ/mol, volumes 0.968-1.002 of experiment; fine-tuned old / new -62.7 / -32.4 (ACRDIN04),
ACRDIN08 +18.2 / +2.0 above Form II; experiment about -96 to -99. MACE-OFF24 still to be redone with node energies; UMA
1.1 medium needs more host memory than the dev box allows.

```
sbatch /scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/acr_foundation_oct26/submit_model_check.sbatch
```

Outputs: `/scratch/mk8347/data/crystal_datasets/acridine/model_check_oct26/` (`<label>.json`, `<label>.log`,
`summary.txt`); the job log in `joblogs/`, which also prints the container's fairchem and mace versions.
