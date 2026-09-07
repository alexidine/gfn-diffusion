# Scope: rarer forward rollouts — v0 for pre-production tonight, v1 deferred

Written 2026-09-07. Target: cluster pre-production this evening, production
overnight. That timeline forces a split. **v0 is the cadence mechanism alone, on
top of the shipped prod_sep02 shape, with fixed fractions and the existing Z
machinery re-pointed.** Everything that needs a new sensor or a new controller
is v1. v0 is small enough to build, test locally, and smoke on the cluster in one
evening; v1 is not.

Every design rule below is a consequence of the verified analysis in memory
(`project_logz_lag_sign_and_cadence`, `project_replay_memorisation_monotone_in_lr`,
`project_under_coverage_gate_calibration`); the ones that matter most are restated
here so the spec stands alone.

---

## Why, in three sentences

The MLIP is evaluated on the FULL forward batch every step (`fracs` are loss
weights, not sample fractions), and only the forward branch calls it — bwd and
replay use pre-scored rows. Skipping forward rollouts on k−1 of every k steps
therefore removes ~all training-time energy calls; at the shipped fracs the fwd
branch is only 5% of the gradient, so what is lost is the on-policy Z reading
and the on-policy 5%. The design is safe exactly when Z is **snapped** to the
fresh batch at every rollout and the unpinned interval is bounded.

## The invariants (do not trade these away)

1. **Z is pinned at every rollout by a closed-form fill, never by an EMA.** The
   winsorized root's se is ~rms_clipped/(√B·frac_unclipped) ≈ 0.3 nat at B≈1000,
   so a full-batch fill is more precise than any tracker. Between rollouts Z is
   frozen. Under rare rollouts the dangerous lag sign is LOW (the policy absorbs
   the level onto stored/anchor rows and the next root walks down to meet the
   stale Z — the identifiability loop); a snap is the only brake.
2. **Nothing else may call the MLIP or move Z between rollouts.** That means
   `z_calibration` OFF (its `rollout` mode does its own forward rollout + energy
   call per Z step, off a sensor that is frozen between rollouts and would fire
   every step), and the controller force-refresh must not run fwd.
3. **The replay buffer's residence clock is in STEPS, not manage calls.** Today
   hazard is "per manage call, matching churn_rate" and the call happens on fwd
   steps; the backstop compares an age in steps to `tau·backstop_mult`. At
   cadence N the two clocks disagree by N×. Fix the clock; do not retune around it.
4. **Store the whole forward batch** (`churn_rate := batch_size`). By Little's law
   draws/row = N and occupancy = B·τ_steps/N; the buffer side is nearly free and
   the pin frequency is the change. Fill event size = N/τ_steps; keep it ≤ 0.2.
5. **Single-leg jobs only until the lj_coeff resume crash is fixed** (Monday
   routine). A killed arm cannot resume today.

---

## v0 — TONIGHT

### Config (one new stage key, five existing keys re-pointed)

```yaml
protocols.<name>.stages[equilibration]:
  fwd_rollout_every: N            # NEW. 0/absent = today's behaviour (fwd every step)
  z_calibration: false            # invariant 2
  fracs: {fwd: 0.05, bwd: 0.475, replay: 0.475}   # unchanged; renormalised over
                                                   # active branches per step
z_calibration:                    # z_level_fill lives under this block; keep it
  fill_threshold: 0.5             # >0 keeps the fill ON; small = snap every rollout
  fill_se: 3.0                    # a batch must resolve the gap it claims
  fill_cooldown_steps: 0          # once per rollout; the stash is single-use anyway
buffers.replay_buffer:
  churn_rate: <batch_size>        # store-all
  mean_residence_steps: 5*N       # in STEPS after the clock fix; event size 0.2
  backstop_mult: 5                # unchanged
controller:
  refresh_every: 10               # unchanged, but fwd is EXCLUDED from force-refresh
                                  # when fwd_rollout_every is set (code, below)
```

Anchors stay frozen (prod_sep02 shape). Prior buffer untouched — its churn and
MLIP re-scoring already run inside `evaluation()` at `eval_period`, not per step.
`hot_lr_sensor.action: report`, `fire_cut_factor: 1.0`, `mode: fixed` with
`burn_in_scale == fixed_scale`: all unchanged.

### Code (four touch points; each is a few lines)

1. **`fused_train_step` (train.py ~3745):** read
   `N = stage.fwd_rollout_every`. If `N > 0`:
   `fwd_ran = (self.step_ind % N == 0); fwd_active = False`, and the force-refresh
   clause must NOT set `fwd_ran`. `fwd_ran` gates the rollout, the `z_fill` stash
   and replay admission; `fwd_active` gates the loss — so a rollout step is
   rollout → stash → admit → a pure bwd+replay fused step → `z_fill`, and the
   forward branch never contributes a gradient (owner review 2026-09-07: `fwd_frac`
   was a loss weight, and the fill must be the ONLY thing that moves Z). EVERY fused
   step is therefore bwd/replay only, renormalised over the two.
2. **`z_level_fill` (train.py ~5073):** no change needed — it early-returns when
   nothing is stashed. Verify the fill fires on every rollout step: `z_fill/gap`
   must be logged exactly once per N steps.
3. **`manage_replay_buffer` (train.py ~8130):** convert the hazard from per-call to
   per-step. Track `_last_replay_manage_step`; per call, evict a fraction
   `min(1, (step − last)/tau)` of survivors instead of `1/tau`. Leave the backstop
   as is (it already compares an age in steps). Document `mean_residence_steps`
   as steps. This also fixes the same latent mismatch on today's eval-site call.
4. **Stage schema:** register `fwd_rollout_every` wherever stage keys are
   validated so an unknown-key check does not refuse it, and add one invariant:
   `fwd_rollout_every > 0` requires `z_calibration: false` on that stage and
   `fill_threshold > 0` (invariant 2 must be unforgeable by config).

Not touched in v0: the LR controller (its `fused` channel still sees a loss every
step; the fwd term is ≤5% of it, far under the 10× excursion bar), the hot-LR
sensor (report), the grad guard, the anchor buffer, the prior buffer, eval.

### What to watch in v0 (existing metrics, no new instrumentation)

| metric | expectation | what a violation means |
|---|---|---|
| `z_fill/gap`, `z_fill/se` | logged once per N steps; `|gap|` ≤ 3·se after each fill | fill not firing, or batch does not resolve Z |
| `lr_ctrl/scale` | flat | a fire moved the rate |
| `replay_buffer` occupancy | ≈ B·τ_steps/N (e.g. 1000·100/20 = 5000) | clock fix wrong |
| `replay/resid_vs_intake` | > 0.368; compare to the same-LR p02 arm | reuse = N is memorising |
| `bwd/under_coverage` | MA(150) jump < 1 nat (read by eye or offline) | forgetting |
| step time, non-rollout steps | ≈ (1 − fwd share) of today's; rollout steps ≈ today's | MLIP still being called between rollouts |
| `nvidia-smi` sidecar | no periodic dips into the 38–49% band | the cluster's utilization killer |

### Built the same evening, after owner review (was "v1")

- **Loss-weight controller `balance.kind: gated_ramp`** (protocol.py): one sensor,
  two motions, hard rails. `ramp: replay`, `guard: bwd`,
  `metric: bwd/under_coverage_rise150`, `bar: 1.0`, `up: 0.0017` (0.50 → 0.75 replay
  share over ~1500 steps), `down: 0.043` (0.75 → 0.10 over ~150 steps when the guard
  fires), `pinned: {fwd: 0.0}`, `bounds` = bwd [0.5, 0.9] / replay [0.1, 0.5] on
  MLIP, bwd [0.25, 0.9] / replay [0.1, 0.75] on ELJ. Ticks every 10 steps. Chosen
  over a `constraint`-kind mapping because the owner's rule IS a ramp with a gate and
  "drift unconditionally to a cap" has no honest second metric; memorisation is
  owned by the buffer/rollout side, not this controller. `active_modes` /
  `read_modes` enumerate kinds — the new kind is registered there (missing it is a
  KeyError on the first fused step, as `ratio` once found).
- **The sensor** (train.py, beside the bwd stats): `under_coverage_rise150` = mean of
  the last 150 steps of `bwd/under_coverage` minus the mean of the previous 150;
  written only once the 300-step window is full (the tracker EMAs what it is
  given, so NaN would poison it). Calibrated: ~150-step oscillation cancelled, 0–5%
  false positives on four healthy arms, 120–190-step latency on a +3-nat/300-step
  injected deterioration.
- **Anchor-only prior** (`buffers.prior_buffer.source: anchors`, opt-in): the churn
  budget is met by `top_up_prior_from_anchors` (noise + re-score frozen anchors)
  instead of prior-model draws, which were otherwise SKIPPED with a warning when no
  model existed and left `rebuild_prior_by_churn` admitting nothing. This is the
  paper's S2 "noised buffer" with refresh. `snapshot_prior` is dropped from the
  stages; `prior_model_name` resolves to null and nothing needs it, so the leg-2
  prior-model resume gap does not exist under this source. The default keeps the
  running prod_sep02 arms untouched.
- **Local acceptance uses N = 7**, coprime with the 10-step metric cadence: at N = 10
  every logged row was a rollout step and the 0.5/0.5 renormalisation was
  unobservable.

### Acceptance — local (ELJ rig, ~45 min)

On `configs/local_prod_sep02/lp02.yaml` shape, N = 7 (`configs/local_rr_sep07`), 2000 steps:
(a) `z_fill/gap` appears at steps 10, 20, … and nowhere else; (b) `lr_ctrl/scale`
flat; (c) `anchor_buffer_length` constant; (d) replay occupancy ≈ B·τ/N;
(e) `Bwd Frac` reads 0.5 on non-rollout steps and 0.475 on rollout steps;
(f) zero divergences; (g) energies and log Z track the N = 1 baseline (lp02) within
noise at matched step — this is the one that says the design works, not merely
runs. N = 1 must reproduce today's behaviour bit-for-bit in the fused step.

### Acceptance — cluster smoke (20-min wall, one arm per system)

Same shape as `configs/smoke_sep02`: mip, mipu, nehu, acr at their p02 centre
rates, N = 10. Pass = all four reach wandb and log `z_fill/gap` on the cadence.
Then read step time: UMA/MACE non-rollout steps should be a small fraction of
today's 15–27 s/it.

### Overnight production proposal (single-leg, 2-day walls)

4 systems × N ∈ {5, 20} at the p02 centre rates = 8 arms. N = 5 is already a 5×
cut in energy calls with FEWER draws/row than today (5 vs 12–20); N = 20 is 20×
with slightly more. `mean_residence_steps` = 5N. Compare against the running p02
arms at matched step count. Keep the 7-day p02 paper arms running untouched.

---

## v1 — BUILT 2026-09-07 evening (all six items; short version in `handoff_rr_v1.md`)

Status: A (birth log p_F → `replay/policy_drift_std`, trigger shipped off), B (held-out
replay split → `replay/val_gap`), C (eval rollouts feed the fill, `fill_from_eval`),
D (level-blind forward policy step, `tb_z_source: batch_root`, own arm `rr_n7_fwd`),
E (absorber: 1-D Kalman on log Z, `fill_mode: absorb`), F (the forgetting sensor now
reads `bwd/relative_under` — the Z-anchored under-coverage was measuring the fill; a
warm-up would have hidden that). Commits 1d7eee7, a9b4560, 671da83 on top of v0/v1a.
The section below is the original plan, kept for the reasoning.

## v1 — the plan as written before the build

- **`birth_log_pf`** (one float per row, stored at admission from
  `fwd_stats['log_pf']`, currently dropped at train.py ~8071) →
  `replay/policy_drift_ess_frac` = (Σw)²/(n·Σw²) with w = exp(log_pf_now −
  birth_log_pf). This is the NON-circular rollout trigger (roll out when ESS < ~0.3,
  hard cap at N_max); "Z-fit quality" is the circular one. Same weights give an
  IS estimate of the on-policy root, trustworthy while ESS is high.
- **Forgetting gate:** MA(150) difference on `bwd/under_coverage` > 1 nat as the
  fast gate (120–190-step latency, 0–5% FP on healthy arms), slow EMA(hl 1500)
  rise > 1 as the low-FP backstop. Actuator: bwd weight → 0.9 over ~100 steps.
- **Memorisation gate:** replay validation split (tag 5–10% of admitted rows,
  exclude from the draw, score at the metric cadence); train/val gap caps the
  replay weight. Keep `birth` log_w SIGNED and score `resid_vs_intake` against the
  CURRENT Z so a fill does not read as absorption.
- **Eval rollouts → z_fill** (on-policy at training T; a different batch size
  changes the root's se, not its location). Not into the buffer.
- **fwd policy step on rollout steps** (`freeze_policy: 0` on the fwd branch):
  the only level-blind policy gradient available; nearly free once the rollout is
  paid for. Off in v0 to match today's branch roles exactly.
- **Anchor-only bwd** (drop the prior model): the paper's "noised buffer" IS the
  anchor buffer; the prior model is unclaimed machinery and the source of both
  open resume failures. Needs a bwd memorisation sensor
  (`prior_buffer.absorption_stats()` is one call away) and a coverage statement.
- **Pin/absorber β**: today's fwd β = 10 (pin) vs bwd/replay β = 80 (absorbers)
  is backwards for this design; revisit once v0 is measured.

## Risks stated plainly

- The synthetic +3-nat/300-step deterioration used to calibrate the gate is not a
  measured collapse; real collapse shape is unknown. v0 has no automatic gate —
  it relies on the LR controller's hard bars and on eyes.
- Periodic utilization: rollout steps and buffer steps look different to the
  cluster's sampler. The sidecar is in place; the usage investigation should
  look at v0's profile before it is trusted overnight.
- Reuse per row = N. At N = 20 that exceeds today's mipu (≈20) only marginally,
  but `resid_vs_intake` is the number to read, and there is no gate on it in v0.
- A killed v0 arm cannot resume until the lj_coeff stamp fix lands.

## Local acceptance and cadence measurements — 2026-09-07 evening

Rig: `configs/local_rr_sep07` (ELJ mipcas, batch 400, 2000 steps, from `lp02`'s
phase-1 exit). Runs `rr07_rr_n{1,7,20,50}`. Read with an UNFILTERED
`scan_history` — `keys=[...]` keeps only rows carrying every key, and no row carries a
fill, the controller sensor and an eval-only counter at once. `z_fill/gap` is a
persisted metric; count fills from `z_fill/fired`.

**rr_n7 — acceptance.** Anchors static (158,998). Prior buffer 63,000 at entry, then
500/eval from noised anchors, 0 from any prior model. Replay occupancy 2,095 vs
B·τ/N = 2,000 (per-step hazard clock). fracs 0.500/0.500 on and off rollout rows (fwd
pinned 0; the old 0.475 no longer applies). 0 divergences, LR flat. Memorisation
`resid_vs_intake` 0.97–1.00 at reuse 7. Energies 15.9 → 4.9 over the run.

**Defect found by this run, fixed in 34dba58.** `bwd/under_coverage_rise150` never
reached the tracker: the sensor lived in `_update_rolling`, which runs every 10th
trained bwd step, so a 300-*sample* window was 3,000 *steps* — 10× the calibration
(done on 10-step wandb rows) and longer than the run. `gr_share` held 0.5 throughout
with `gr_sensor` unwritten. Now `Modeller._forgetting_sensor`: step-stamped samples,
windows in steps, cadence-independent; `tests/protocol/test_forgetting_sensor.py`.
Any sensor computed in `_update_rolling` sees one sample per 10 bwd steps.

**The Z pin mostly refuses early in training, and that is correct.** At batch 400 the
fill's se is ~1.6 nat against |gap| ~1.9, so the 3-se gate blocked ~88% of pins
(`fired` on 22 report rows, `blocked_by_se` on 174): Z sat in a ±5 nat dead band, within
the estimator's own noise. On the live production arms the same se is 0.07 nat (ELJ,
batch 1000, step 69k) and 0.53 (UMA, batch 1600, step 31k) with |gap|/se ≈ 0.9 — the
band collapses with convergence because se ∝ std(log w)/√B. Ship `fill_se: 3`; a
precision-weighted absorber (Z += (root−Z)·se₀²/(se₀²+se²)) stays a v1 option.

**Cost model, clean (no GPU co-tenant).** t(N) = 0.26 + 0.51/N s per step at batch 400:
N=1 1.3 it/s (lp02), N=7 3.0 (steps 100–500), N=20 predicted 3.5 / measured 3.57. The
rollout is 66% of an every-step ELJ step here; the speedup saturates at ~2.7× by N≈20
on ELJ. On MLIP systems the rollout is a larger fraction, so gains continue to larger N.

**Cadence sweep (Z drift and memorisation vs N), settled window steps > 600:**

| N | \|gap\| med | se | gap/se | memo (min) | bwd under_cov @2000 | E_mean @2000 |
|---|---|---|---|---|---|---|
| 7 | 1.73 | 1.55 | 1.10 | 0.985 (0.966) | 39.9 | 4.95 |
| 20 | 2.16 | 1.73 | 1.23 | 0.899 (0.860) | 44.9 | 3.65 |
| 50 | 1.60 | 2.16 | 0.76 | 0.911 (0.732) | 42.1 | **20.24** |
| 1 | 1.80 | 2.17 | 0.84 | 0.997 (0.98) | 40.6 | 10.53 |

**Reading.** Z drift is never the binding constraint here: gap/se ≤ 1.2 at every N up
to 50 — the fill's own batch noise dominates what the policy does to Z between
rollouts. What binds is **on-policy quality per training step**: fwd scatter 52.0 /
54.4 / 58.1 and mean sample energy 4.95 / 3.65 / 20.2 at N = 7 / 20 / 50 at step 2000,
with N = 50's scatter *rising* over the run (56.4 → 58.5) while N = 7's falls (56.6 →
52.0). N = 20 is indistinguishable from N = 7 at matched step; N = 50 is not viable on
this system — 40 rollouts in 2000 steps, each stored row reused 50×. Since the ELJ
speedup is already 2.7× at N = 20 (asymptote 2.96×), there is no reason to go past
~20 on ELJ; the MLIP arms are where N = 20's larger gain matters, and the cluster
battery's {5, 20} brackets the useful range. Caveats: one seed, batch 400, 2000 steps
from a phase-1 exit (early training, where the policy moves fastest).

**The controller worked live, in rr_n50** (the only sweep arm that loaded 34dba58):
`bwd/under_coverage_rise150` written from step 300; bwd under-coverage rose 45.2 → 48.0
as replay-heavy training at reuse 50 forgot the prior; the sensor crossed +1 nat (1.12 @
510, 2.57 @ 720); replay share cut 0.50 → 0.36 → 0.10 (the rail); under-coverage fell to
41.7, the sensor went to −3.8, and the ramp resumed (0.26 by step 1760). Memorisation
dipped to 0.77 at the replay-heavy peak and recovered to 0.95–0.99 after the cut. rr_n20
ran the pre-fix code (sensor never written, share held 0.5); rr_n1 was the second live run
and repeated it (sensor 1.45/2.14 at 510/720, share cut to 0.10 by 760, back to 0.24 by 1760).

**Open — the guard fires on the post-transition transient.** In both live runs the rise
came at steps 300–750, when under-coverage climbs 4–5 nat as the policy first leaves
the prior; both then ran ~1000 steps replay-starved and ended with worse energy (10.5,
20.2) than the inert 0.5/0.5 runs (4.95, 3.65). The bar was calibrated on mature arms.
Needs a warm-up (v1 item F). Also: N = 1 under the new gate runs 2.1 it/s, not 1.3 —
the rollout's backward pass is gone even at N = 1, so the cost model's N = 1 point is
the old design's.
