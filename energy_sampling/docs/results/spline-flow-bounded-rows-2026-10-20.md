# Spline flow: bounded rows, knot derivatives, height floor and replay (MIPCAS eLJ)

*Battery record. Closed 2026-10-20. Code at commit `06950ee0`. Not a finding.*

## Design

The question: what bounded latent rows (`--bounded_rows linear`, a closed-interval spline with a uniform base) cost the
exact-density autoregressive flow of `inn_modeller.py` against unbounded rows, and how that cost moves with the knot-derivative
option (free or `--tied_derivatives`), the bin-height floor (`--min_bin_height` 1e-5 or 1e-7) and error-filtered replay.

Common to every arm: MIPCAS, eLJ energy at T 2.5, `energy_config.bounding_coeff` and `reduction_coeff` 1000, prior
`mipcas_elj_pooled_oct05_prior.pt`, one autoregressive block with the causal-transformer conditioner (width 256, 4 layers,
8 heads), 32 bins, block order `8.11.10.9.7.6.5.4.3.2.1.0`, held out by crystal (`--val_group_ignore_rows 6,7,8`).
Stage 1: 10,000 likelihood steps on noised stored rows (`--mle_noise_log_range -1.6126 -1.6126`), batch 8192, lr 1e-3,
floor 1e-5. Stage 2: 1500 updates of 2000 fresh draws (3M energy calls) plus stored rows, Huber TB (knee 10, stored-row
knee 80), lr 3e-4 cosine to 1e-5, grad clip 10, tied level, fp32 transformer. Replay arms add `--bank_frac 0.1
--bank_additive --bank_admit 2 --bank_admit_top 1.0 --bank_max_excess 1e9 --bank_batch 256 --bank_alpha 1.0`. One seed
per arm (seed 12345) except the seed-1 arm (`--set seed=1 --split_seed 12346`, same held-out split). Local GPU,
sequential. MXtalTools was a pinned copy of the v2c builder.

Arms (run names, prefixed `inn_` on disk and in wandb):

| arm | rows | derivatives | floor | replay | stage 1 | stage 2 |
|---|---|---|---|---|---|---|
| unbounded, free, 1e-5 | unbounded | free | 1e-5 | no | `mip_floor_s1_oct18c` | `mip_ctrl3_oct20` |
| bounded, free, 1e-5 | bounded | free | 1e-5 | no | `mip_bnd_s1_oct20` | `mip_bnd_s2_oct20` |
| unbounded, tied, 1e-5 | unbounded | tied (F 2) | 1e-5 | no | `mip_tctrl_s1_oct20` | `mip_tctrl_s2_oct20` |
| bounded, tied, 1e-5 | bounded | tied (F 2) | 1e-5 | no | `mip_tbnd_s1_oct20` | `mip_tbnd_s2_oct20` |
| unbounded, tied, 1e-7 | unbounded | tied (F 2) | 1e-7 (stage 2) | no | `mip_tctrl_s1_oct20` | `mip_tctrl7_s2_oct20` |
| bounded, tied, 1e-7 | bounded | tied (F 2) | 1e-7 (stage 2) | no | `mip_tbnd_s1_oct20` | `mip_tbnd7_s2_oct20` |
| unbounded, tied, 1e-5, replay | unbounded | tied (F 2) | 1e-5 | yes | `mip_tctrl_s1_oct20` | `mip_trep_s2_oct20c` |
| bounded, tied, 1e-7, replay | bounded | tied (F 2) | 1e-7 (stage 2) | yes | `mip_tbnd_s1_oct20` | `mip_tbnd7rep_s2_oct20` |
| bounded, tied, 1e-7, replay, seed 1 | bounded | tied (F 2) | 1e-7 (stage 2) | yes | `mip_tbnd_s1_seed1_oct20c` | `mip_tbnd7rep_s2_seed1_oct20c` |
| bounded, pure HM, 1e-7, replay | bounded | tied (F 1) | 1e-7 (stage 2) | yes | `mip_tbnd_s1_pure_oct20c` | `mip_tbnd7rep_s2_pure_oct20c` |
| bounded, free, 1e-5, 6M | bounded | free | 1e-5 | no | `mip_bnd_s1_oct20` | `mip_bnd6m_oct20` (3000 updates) |

Two 6M unbounded free arms from the night before (`mip_s2ctrl6m_oct19`, `mip_ctrl6m_s1b_oct19`, from the
`mip_floor_s1_oct18c` stage 1) are included for the 6M comparison.

Arms that ran before 18:35 on 20 Oct ran on the working tree that became `06950ee0`; the default (factor 2) tied path of
that commit is bit-identical to the one they ran.

## Data

All under `energy_sampling/artifacts/inn_proto/` on the local machine (git-ignored): one directory per run name with
`metrics.jsonl`, `best.pt` / `final.pt` and per-eval `excess_log10_step*.npy`; 40k-draw dumps of each final model in
`inn_dump_<run>/dump_draws.npz` (draws, log R, log q, per-row log q terms); queue scripts `bounded_oct20.sh`,
`bounded6m_oct20.sh`, `tied_oct20.sh`, `floor7_oct20.sh`, `hour_oct20b.sh`, `gpu_oct20c.sh` and their logs; gradient-split
logs `grad_mip_*.log` (`diag_oct20.sh`). Eval-only dumps with the floor changed on trained weights:
`inn_dump_mip_tbnd_s2_f7eval_oct20`, `inn_dump_mip_tctrl_s2_f7eval_oct20`. The dump of `mip_bnd_s2_oct20` failed (a
per-row split error in the dump code, fixed before `06950ee0`), so that arm has 40k eval draws only. The pinned
MXtalTools copy lived in a session scratch directory and is not kept.

## Measurements

Final models after stage 2. ELBO is the mean of the final in-run eval and the dump (40k draws each). Excess is kT above
the best stored row; clash counts and quantiles pool the final eval and the dump (80k draws, 40k for `bounded, free,
1e-5`). Held-out NLL (nats, lower is better) is on held-out stored rows. Gradient median is the pre-clip norm over the
last 375 updates. Reduction energy is the run's last logged mean. One seed per arm except where marked. Reruns of the
same arm differed by about 0.04 nats in ELBO; a clash count n carries Poisson noise of about sqrt(n).

| arm | ELBO by eval at 0 / 0.75 / 1.5 / 2.25 / 3M calls (nats) | final ELBO (nats) | draws >30 kT | draws >50 kT | P99 excess (kT) | P99.9 excess (kT) | held-out NLL (nats) | grad median | reduction energy |
|---|---|---|---|---|---|---|---|---|---|
| unbounded, free, 1e-5 | 29.25 / 34.13 / 34.85 / 35.22 / 35.34 | 35.33 | 189 | 72 | 19.5 | 44.4 | -17.95 | 44 | 0.00033 |
| bounded, free, 1e-5 | 29.19 / 31.00 / 32.07 / 32.80 / 33.11 | 33.11 | 247 (40k) | 150 (40k) | 23.7 | 267.5 | -19.37 | 136 | 0.077 |
| unbounded, tied, 1e-5 | 29.25 / 34.35 / 35.02 / 35.32 / 35.43 | 35.44 | 122 | 41 | 19.0 | 36.1 | -17.91 | 47 | 0.0003 |
| bounded, tied, 1e-5 | 29.06 / 33.61 / 34.55 / 34.86 / 35.02 | 35.00 | 254 | 143 | 18.9 | 248.7 | -18.07 | 59 | 0.031 |
| unbounded, tied, 1e-7 | 29.26 / 34.42 / 35.04 / 35.34 / 35.42 | 35.43 | 152 | 52 | 19.1 | 39.2 | -17.91 | 41 | 0.0003 |
| bounded, tied, 1e-7 | 29.37 / 33.98 / 34.84 / 35.19 / 35.31 | 35.31 | 116 | 35 | 18.1 | 35.3 | -18.03 | 49 | 0.0017 |
| unbounded, tied, 1e-5, replay | 29.25 / 34.26 / 34.91 / 35.24 / 35.36 | 35.36 | 49 | 13 | 16.8 | 25.4 | -17.93 | 54 | 0.00019 |
| bounded, tied, 1e-7, replay | 29.37 / 33.80 / 34.72 / 35.14 / 35.29 | 35.29 | 30 | 6 | 16.5 | 23.5 | -18.06 | 66 | 0.00021 |
| bounded, tied, 1e-7, replay, seed 1 | 29.16 / 33.73 / 34.67 / 35.11 / 35.26 | 35.26 | 23 | 4 | 16.4 | 23.1 | -18.12 | 74 | 0.00019 |
| bounded, pure HM, 1e-7, replay | 29.65 / 33.64 / 34.60 / 35.07 / 35.23 | 35.22 | 35 | 4 | 16.6 | 24.7 | -18.08 | 77 | 0.00021 |

6M calls (evals at 0 / 1.5 / 3 / 4.5 / 6M), same columns:

| arm | ELBO by eval (nats) | final ELBO (nats) | draws >30 kT | draws >50 kT | P99 (kT) | P99.9 (kT) | held-out NLL (nats) | grad median | reduction energy |
|---|---|---|---|---|---|---|---|---|---|
| bounded, free, 1e-5, 6M | 29.19 / 31.46 / 32.85 / 33.84 / 34.19 | 34.18 | 397 | 183 | 22.2 | 266.8 | -18.67 | 140 | 0.16 |
| unbounded, free, 1e-5, 6M | 29.11 / 34.77 / 35.35 / 35.64 / 35.75 | 35.74 | 109 | 32 | 18.2 | 34.0 | -17.69 | 39 | 0.00028 |
| unbounded, free, 1e-5, 6M, seed 1 | 29.15 / 34.82 / 35.38 / 35.64 / 35.73 | 35.73 | 115 | 42 | 18.2 | 35.8 | -17.67 | 35 | 0.00026 |

Stage-1 held-out NLL at step 10,000 (nats): unbounded free -23.10, bounded free -22.79, unbounded tied -23.22, bounded
tied -23.15, bounded tied seed 1 -23.23, bounded pure HM -22.75.

Floor changed on trained weights, no retraining (40k dump each): bounded tied from floor 1e-5 to 1e-7, ELBO 34.98 to
35.31, draws >30 kT 127 to 56, >50 kT 75 to 13, P99.9 266.1 to 33.2 kT; unbounded tied from 1e-5 to 1e-7, ELBO 35.45 to
35.44, >30 kT 56 to 69, >50 kT 17 to 23, P99.9 34.5 to 37.8 kT.

Diagnostics on final models (CPU unless noted):

- Gradient split (GPU, 8 batches of 2000 draws plus 8192 stored rows, `grad_*.log`): pre-clip parameter-gradient median
  of the fresh-draw term 185 (bounded free 6M) against 36 (unbounded free 6M), and 210 against 44 at 3M. Share of the
  squared gradient at the spline outputs on cell length a's bin widths: 93% and 84% (bounded free 6M, 3M) against 0.9%
  (unbounded free 6M). In one batch of 4000 dump draws, one draw carried 99.7% of that share; it sat in a bin of mass 1e-5
  with mean slope 4.4e-4 and a left knot derivative of 1.6, at 267 kT excess.
- Draws whose bin on some row has a knot derivative over 100 times the bin's mean slope (40k dumps, 6M finals): 0.08% of
  bounded free draws (75% of them above 30 kT, 12% of that arm's clash draws) against 0.00% of unbounded free draws.
- Empty-interior knot derivatives on bounded free 6M (4000 contexts): median 0.00 to 0.01 on every bounded row, P90 0.61
  on cell length a and 0.01 to 0.06 on the other bounded rows.
- Draws in a bin of mass below 3e-5 on some bounded row (bounded tied 1e-5, 40k dump): 0.19%, of which 88% were above 30
  kT; they made up 68 of the arm's 127 clash draws, and the other draws were above 30 kT at 0.15%.
- Centroid-x marginal (20 bins over [-1, 1]): both stage-1 models put 17.6 to 19.4% in each edge bin and 0.6 to 0.7% in
  the shoulder bins; after stage 2, unbounded free 9.4 to 10.2% and 2.7 to 2.8%, bounded free 14.8 to 15.2% and 0.9% at 3M
  and 12.0 to 12.3% and 1.6 to 1.9% at 6M; a target estimate from importance-reweighted unbounded draws 9.9 to 10.3% and
  2.4 to 3.1%.
- Box contacts (dump draws with a linear row outside [-1, 1]): 0 on every bounded arm, 4.1 to 4.2% on the unbounded free
  arms.

## Trends

- Across the four 3M arms without replay at floor 1e-5, the gap between bounded and unbounded rows was 2.22 nats with free
  derivatives and 0.44 nats with tied derivatives. The bounded free gradient median was 3 times the unbounded one, and the
  tied bounded median was within 1.3 times of it.
- The bounded free gap at matched calls of the 6M schedule went 3.31, 2.50, 1.80 and 1.56 nats at 1.5, 3, 4.5 and 6M.
- Lowering the floor from 1e-5 to 1e-7 moved the bounded tied arm by +0.31 nats, with clashes above 50 kT falling from 143
  to 35 (80k), and left the unbounded tied arm within 0.01 nats.
- Replay at weight 0.1 changed ELBO by -0.02 nats on the bounded tied 1e-7 arm and -0.08 on the unbounded tied arm, and
  cut clashes above 30 kT by factors of 3.9 and 2.5.
- With replay on both, bounded tied 1e-7 against unbounded tied: -0.07 and -0.10 nats ELBO (two seeds), 30 and 23 against
  49 clashes above 30 kT, 6 and 4 against 13 above 50 kT.
- The pure harmonic mean (factor 1) against factor 2: stage-1 held-out NLL 0.40 to 0.48 nats higher, final ELBO 0.04 to
  0.07 nats lower, clashes above 30 kT 35 against 30 and 23, above 50 kT 4 against 6 and 4.

## Not covered

One molecule (MIPCAS) and one energy (eLJ). One block; with a stack of blocks a bin's height fraction is its mass only in
the last block. No conditional runs, which refuse every option here. The floor 1e-7 was used only in stage 2, and stage 1
always at 1e-5. Unbounded rows were not run with replay and floor 1e-7 together, nor bounded rows with free derivatives at
floor 1e-7. Replay weight, the derivative factor range other than 1 and 2, bin count and block order were not varied.
Seeds: two for the bounded tied 1e-7 replay arm, one for every other 3M arm. Runs longer than 3M calls only for bounded
free and unbounded free.
