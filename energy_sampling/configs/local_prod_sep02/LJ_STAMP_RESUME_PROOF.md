# Resume proof for the lj_coeff buffer-currency fix (2026-09-07)

Evidence for `buffer.BUFFER_FORMAT_VERSION` / `CrystalBuffer._refuse_unknown_currency`
/ `migrate_buffer_sidecar.py`. All runs on this machine (RTX 5080 laptop, venv
`csd_mxt_gfn`), wandb project `GFN Energy`, `GFN_GPU_GUARD=0`. Forged inputs are
copies of a real run's `_running.pt` paired with its sidecar rewritten into the exact
pre-relocation shape (no `lj_coeff`, raw `.elj` on the elj route) -- the shape of the
cluster's pt100 phase-1 exit sidecars.

| run (config) | code | sidecar restored | outcome | wandb |
|---|---|---|---|---|
| `lp02L_resume` (ELJ mipcas, resume 2000->2100) | pre-fix `conditional` | forged legacy | AttributeError at first bwd draw, same frames as cluster (train.py:5387 -> molecular_crystal.py:1072 -> :662 -> :534), exit 1 | 1uygr2qx |
| `lp02L_resume` | fix | forged legacy | `BufferCurrencyError` at restore, before the prior re-analysis, naming the migration, exit 1 | e5hs57z3 |
| `migrate_buffer_sidecar.py` on that sidecar | fix | -- | coefficient read from prior (0.3635836825); prior/replay/anchor means after migration equal lp02's original sidecar to the printed digit (-106.59228 / -6.53028 / -127.24032) | -- |
| `lp02L_resume` | fix | migrated | trains 2000->2100, Bwd Frac 0.475, prior buffer churns 67,500 -> 72,332 rows, sidecars written as version 2, exit 0 | zu9bgv9b |
| `lp02_compat` (ELJ, lp02's REAL post-relocation, pre-version sidecar) | fix | stamped, no version key | loads, trains 2000->2100, re-saved as version 2, exit 0 | dwa3ovra |
| `mipu_smoke` (UMA mipcas, fresh 0->40) | fix | -- | writes a version-2 sidecar, exit 0 | wiryzf0n |
| `mipu_smokeL_resume` (UMA, resume 40->60) | fix | forged legacy | refused at restore, exit 1 | 9r9pviqb |
| `mipu_smokeL_resume` | fix | migrated (relabel at 1.0, nothing rescaled) | trains 40->60, Bwd Frac 0.475, version-2 sidecars, exit 0 | ijjrcvf6 |

Caption: each row is one `train.py` launch; "outcome" is the process exit code and the
first diagnostic line, read from the launch log. The comparison is pre-fix vs fix on the
same forged input (rows 1-2), and refused-then-migrated on both energy routes (rows 2-4,
7-8). The elj row is the one where a wrong stamp would be numerically visible.

Rig files: `lp02L_resume.yaml`, `lp02_compat.yaml`, `mipu_smoke.yaml`,
`mipu_smokeL_resume.yaml`. The forged inputs live under
`D:/crystal_datasets/gfn_checkpoints/localprod_lp02L_*` and `localprod_mipu_smokeL_*`
(migrated in place; the pre-migration copies are `*.pre_lj_migration.bak`).
