# Project change history

Generated from `config_state.CHANGES` -- do not edit by hand.

Selected config-schema transitions and their closely coupled rationale.
This is not a general development log. A change marked **STATE N** altered
how persisted state is interpreted and is the only kind that moves
`project_state_version` or carries a migration.

## STATE 1 — Baseline.

Baseline. Introduces `project_state_version` and this module. State 1 is the config schema as it stands at the start of the infrastructure stabilization pass, with configs/mk_dev.yaml as the canonical master. The key changes recorded here are the retirements that had accumulated in utils._RETIRED_KEYS without version stamps: previously a config carrying any of them was rejected at load with no route forward, and the mechanical subset can now be repaired instead.

**Components:** `config_state.py`, `utils.py`, `configs/mk_dev.yaml`

**Added:**

- `project_state_version` = `1`

**Renamed:**

- `adaptive_lr.reset_loss_abs` -> `adaptive_lr.divergence_loss_abs`
- `adaptive_lr.reset_grad_abs` -> `adaptive_lr.divergence_grad_abs`
- `adaptive_lr.cut_ratio` -> `adaptive_lr.divergence_cut`
- `buffers.anchor_buffer.health_gate_r2` -> `buffers.anchor_buffer.health_gate_floor`

**Removed:**

- `gpu_util_floor` -- see utils._RETIRED_KEYS
- `batch_growth_min_gain` -- see utils._RETIRED_KEYS
- `step_probe` -- see utils._RETIRED_KEYS
- `adaptive_lr.servo.ceiling_halflife_steps` -- see utils._RETIRED_KEYS
- `adaptive_lr.servo.trigger` -- see utils._RETIRED_KEYS
- `adaptive_lr.servo.discovery` -- see utils._RETIRED_KEYS
- `adaptive_lr.trigger` -- see utils._RETIRED_KEYS
- `adaptive_lr.boost` -- see utils._RETIRED_KEYS
- `adaptive_lr.discovery` -- see utils._RETIRED_KEYS
- `adaptive_lr.damage` -- see utils._RETIRED_KEYS
- `adaptive_lr.enabled` -- see utils._RETIRED_KEYS
- `adaptive_lr.hold_steps` -- see utils._RETIRED_KEYS
- `adaptive_lr.decay_halflife_steps` -- see utils._RETIRED_KEYS
- `adaptive_lr.decay_floor_scale` -- see utils._RETIRED_KEYS
- `adaptive_lr.cut_loss_abs` -- see utils._RETIRED_KEYS
- `adaptive_lr.cut_grad_abs` -- see utils._RETIRED_KEYS
- `adaptive_lr.fire_cooldown_steps` -- see utils._RETIRED_KEYS
- `adaptive_lr.recovery_target_frac` -- see utils._RETIRED_KEYS
- `adaptive_lr.recovery_wait_steps` -- see utils._RETIRED_KEYS
- `adaptive_lr.recovery_ramp_steps` -- see utils._RETIRED_KEYS
- `reuse_prior` -- see utils._RETIRED_KEYS
- `terminal_logw_std` -- see utils._RETIRED_KEYS
- `terminal_box_violation` -- see utils._RETIRED_KEYS
- `terminal_frozen_steps` -- see utils._RETIRED_KEYS
- `integrator.min_traj_length` -- see utils._RETIRED_KEYS
- `integrator.max_traj_length` -- see utils._RETIRED_KEYS
- `integrator.traj_length_strategy` -- see utils._RETIRED_KEYS
- `integrator.discretizer` -- see utils._RETIRED_KEYS
- `integrator.discretizer_max_ratio` -- see utils._RETIRED_KEYS
- `z_calibration.unclipped` -- see utils._RETIRED_KEYS
- `buffers.anchor_buffer.mcmc` -- see utils._RETIRED_KEYS
- `buffers.replay_buffer.admit_cap_max` -- see utils._RETIRED_KEYS
- `buffers.replay_buffer.admit_cap_min` -- see utils._RETIRED_KEYS
- `buffers.replay_buffer.admit_cap_health_h0` -- see utils._RETIRED_KEYS
- `buffers.replay_buffer.admit_temperature` -- see utils._RETIRED_KEYS

**Requires judgment:**

- `max_reloads` -- renamed -> max_reloads_per_1k_steps AND changed from a COUNT to a RATE. The old integer is not a valid value for the new key: a budget of 5 reloads is not 5 reloads per 1000 steps. Choose the rate the run should carry (canonical: 0.2).
- `buffers.anchor_buffer.health_gate_zerr` -- renamed -> health_gate_ceiling, but the RULER changed with it. The bar now applies to tb_resid_clipped (signed, beta-bounded) rather than tb_err_worst (unbounded RMS, ~18-21 when healthy). A bar does not survive a ruler swap; carrying the old number across would gate on a threshold that means something else. Canonical: 0.5.
- `adaptive_lr.servo` -- the block was split, not renamed: seed_lr -> adaptive_lr.seed_lr and bounds -> adaptive_lr.bounds carry across unchanged, while target/clip/period/min_readings/max_bad_rate belonged to the online median servo and have no successor. Migrate the two survivors by hand and drop the rest.
- `batch_growth_max_step_regression` -- replaced by batch_growth_min_throughput_gain, which is a DIFFERENT criterion: the old key bounded step-time regression, the new one sets a throughput-saturation floor. The values are not interchangeable. Canonical: 0.05, and it must stay below batch_growth_factor - 1 or every jump is rejected and the batch freezes.

**Invariants:**

- A config at CURRENT_STATE_VERSION passes migrate() unchanged.
- Migration never alters the problem identity (energy_function, paths, conditioning flags, space_groups, temperature) -- those are the checkpoint schema_version's axis, not this one.
- A key appears in at most one of added/renamed/removed/manual.

**Validation:**

- test_config_state.py: round-trip at current version is a no-op; each mechanical class applies; manual keys are reported and never rewritten; the canonical config migrates clean.

## at state 1 — configs/mk_dev.

configs/mk_dev.yaml: both stages now declare an LR sensor -- train_prior `{kind: hyper, beta: 0.1}`, equilibration `{kind: ray}`. Neither declared one before, and because the sensor is opt-in per stage and omission means 'none' SILENTLY, all four `auto` lr_* keys sat at adaptive_lr.seed_lr for the whole run while the config read as adaptive. The kinds are not interchangeable: `ray` draws from replay and scores replay_loss_coeffs, so it is coherent only in a fused stage training replay TB (equilibration), while `hyper` reads no loss and is therefore the sensor for stages that train neither (train_prior, which is bwd/MLE). NO STATE TRANSITION: this adds keys with new meanings rather than reinterpreting existing ones, so a config written before it still means exactly what it meant.

**Components:** `configs/mk_dev.yaml`

**Invariants:**

- `auto` on an lr_* key requires an adaptive sensor to own it; otherwise the rate is fixed at the seed while the config claims otherwise (config_invariants.auto_lr_requires_an_adaptive_sensor).
- A stage declaring lr_sensor kind 'ray' requires ray_calibration.enabled -- train.py::_check_ray_wiring raises otherwise.
- `ray` is only coherent in a fused stage that trains replay TB.
- An explicit float lr_* takes the warmup envelope and divergence handling but NOT peak_scale (controller.py::_apply_lrs).

**Validation:**

- config_snapshot reports exactly two changed values (stages[0] and stages[1] lr_sensor), confirming nothing else moved.
- test_mode_safety.py drives the full load path: auto without a sensor raises, explicit floats load without one, and a hyper-only config loads with ray_calibration disabled.
- beta 0.1 is the best WORST-CASE value from the 12-cell bench sweep, which covered MLE surfaces -- the closest measured evidence for this stage. Production conditional configs run 0.05, but on a var_conditioning stage and from a battery with no committed verdict.

## at state 1 — configs/mode_presets.

configs/mode_presets.yaml is RETIRED and replaced by configs/problems.yaml. The old file declared itself "Reference only -- never loaded by train.py" and drifted accordingly: it prescribed SEVEN keys that are now in utils._RETIRED_KEYS and hard-fail at load (cut_grad_abs, reset_grad_abs, cut_loss_abs, reset_loss_abs, hold_steps, decay_halflife_steps, fire_cooldown_steps), taught the learning-rate rule `anchor x 25/T` that utils.py records as deleted, and labelled its worked example the 'current mk_dev state' at W256/T40 when the file had long been W512/T10. A checklist that recommends keys the schema rejects is worse than no checklist. problems.yaml carries PROBLEM-INTRINSIC settings only -- energy function, paths, conditioning flags, space groups, temperature, and the model/buffer values that follow from the domain -- and is covered by tests, including one that rejects any retired key and one that rejects a tuning knob leaking in. The durable half of the retired file, the intensive/extensive derivation explaining why gradient-space bars scale with (W, T) and loss-space bars do not, moved to docs/design/width_and_length_scaling.md. NO STATE TRANSITION: no config key changed meaning; a reference file was deleted and a new registry added, neither of which alters how an existing config reads.

**Components:** `configs/problems.yaml`, `configs/mode_presets.yaml`, `docs/design/width_and_length_scaling.md`, `configs/mk_dev.yaml`

**Invariants:**

- problems.yaml carries no key that a run would TUNE -- a per-problem difference in a tuning knob is a mode-safety defect in mk_dev.yaml, not a property of the problem.
- periodic_centroids follows the domain: true for crystals, false for toys.
- A conditional problem declares a held-out test set (R17: held-out is read before train metrics).

**Validation:**

- test_problems.py: 18 tests, including agreement between mk_dev.yaml and its mipcas_elj entry, and mutation tests proving the retired-key check fires on a real key and ignores prose.

## STATE 2 — The `ray` LR sensor loses its `enabled` flag and its block moves under `adaptive_lr`.

The `ray` LR sensor loses its `enabled` flag and its block moves under `adaptive_lr`. A stage declaring `lr_sensor: {kind: ray}` IS the switch; a separate flag was a second mechanism for the same decision and the two could disagree -- a stage asking for ray while the flag said false trained at its seed LR with the config claiming a sensor. `enabled` is now DERIVED in train.py from which stages ask, so that disagreement is unrepresentable and the check that caught it is gone with it. The asymmetry that gave the flag away: `hyper` has no block at all and declares itself inline at the stage. The block itself survives -- seven shared parameters are worth storing once -- but under `adaptive_lr`, since it parameterises one of the LR sensors rather than standing alone.

**Components:** `configs/mk_dev.yaml`, `train.py`, `utils.py`

**Renamed:**

- `ray_calibration` -> `adaptive_lr.ray_calibration`

**Removed:**

- `adaptive_lr.ray_calibration.enabled` -- derived from the stages that declare lr_sensor kind ray
- `ray_calibration.enabled` -- derived from the stages that declare lr_sensor kind ray

**Invariants:**

- The ray probe arms iff some stage declares lr_sensor kind 'ray'.
- A config carrying the old top-level `ray_calibration` block fails at load rather than silently falling through to the code defaults -- which would have disabled the probe, since the old constructor defaulted enabled to False.

**Validation:**

- Negative controls for all four dotted-path validators were shown to fail before the move and to still fail after it.
- config_snapshot vs the committed baseline: CHANGED empty.

## STATE 3 — Three replay-buffer keys move from a first-use guard to the load-time retired-key gate: max_residence_steps, toxic_min_draws and toxic_delta_threshold.

Three replay-buffer keys move from a first-use guard to the load-time retired-key gate: max_residence_steps, toxic_min_draws and toxic_delta_threshold. All three were already dead -- manage_replay_buffer raised on them -- but that function first runs at a stage transition, so the rejection arrived hours into a run instead of at load. This is the failure mode _RETIRED_KEYS was created to prevent, and the reason is written directly above that dict: the aug02 battery lost all 16 arms' entire phase 1 (1.1-7.8 h each) to a retired-key guard that lived inside this very function. THE GATE IS ALSO STRICTER THAN THE GUARD IT REPLACES. The guard tested truthiness (toxic_min_draws) or non-None (max_residence_steps), so a config carrying `toxic_min_draws: 0` or `max_residence_steps: null` passed it and ran with the key silently ignored. The gate fires on PRESENCE, because a key in a config is a value its author believes is doing something. Those configs now fail at load, which is the intended widening rather than a side effect. The runtime guards are deleted rather than kept as defence in depth: every load path runs preflight_config, and duplicating the reason text in two places is how the two come to disagree.

**Components:** `utils.py`, `config_state.py`, `train.py`, `configs/mk_dev.yaml`

**Removed:**

- `buffers.replay_buffer.toxic_min_draws` -- see utils._RETIRED_KEYS
- `buffers.replay_buffer.toxic_delta_threshold` -- see utils._RETIRED_KEYS

**Requires judgment:**

- `buffers.replay_buffer.max_residence_steps` -- replaced by mean_residence_steps, and the value does NOT carry across: the old key was a hard age cap (every row evicted at exactly N, a ~uniform age profile, CV ~0.58), the new one is the MEAN of a memoryless hazard (exponential residence, CV ~1). Sizing is Little's law, occupancy = churn_rate * mean_residence_steps, so the new value follows from the occupancy you want and not from the old cap. Canonical: mean_residence_steps: 50. Leaving it unset is NOT 'off' -- it reads as tau = 0, which disarms the hazard and the backstop together.

**Invariants:**

- A retired key is rejected at LOAD, never at first use -- the cost of a first-use guard is the whole run up to that point.
- Replay eviction is residual-INDEPENDENT (hazard + age backstop). That is what makes birth_loss an unbiased intake baseline for Buffer.absorption_stats; a residual-dependent purge cause silently invalidates that sensor.
- Every reason for a retirement has one home, utils._RETIRED_KEYS.

**Validation:**

- Counted over git-TRACKED yaml only (the working tree also holds git-ignored wandb artifacts, which inflate a naive grep): 106 configs carry max_residence_steps, 103 carry toxic_min_draws, none carry toxic_delta_threshold. None had a mechanical route forward before this record, since the keys were rejected by code no migration knew about.
- The max_residence_steps / mean_residence_steps overlap is ZERO -- no config carrying the old key also carries its replacement. That measurement is why it is `manual` rather than `removed`: dropping it would leave tau unset, and tau = 0 disarms both eviction causes (train.py, `if tau > 0`).
- configs/mk_dev.yaml carries none of the three, so the canonical config is unaffected beyond its version stamp.

## STATE 4 — The protocol becomes SELECTABLE.

The protocol becomes SELECTABLE. `protocol:` now names a protocol; `protocols:` holds every one of them, keyed by name. Switching route is that one word plus the problem keys, instead of rewriting the stage list -- and both routes sit in the same file where they can be compared rather than reconstructed from git. Protocols live ONLY here: configs/problems.yaml selects a problem and says nothing about stages, so the two files cannot disagree about which stages a route runs.

**Components:** `configs/mk_dev.yaml`, `configs/problems.yaml`, `protocol.py`, `config_invariants.py`, `config_snapshot.py`

**Invariants:**

- `protocol` names a key of `protocols`, and that protocol has stages -- enforced by config_invariants.protocol_selector_resolves, which must be checked FIRST because every stage-scoped rule reads the ACTIVE stage list and a bad selector silently empties it.
- The trainer and the validators resolve stages through ONE function (config_invariants.active_stages), so a check can never pass on a stage list the run does not execute.

**Validation:**

- The four pre-existing negative controls were shown to fail before the move and to still fail after it.
- A FIFTH failure mode was found by the gate and closed: a selector naming a missing protocol resolved to zero stages, and the auto-LR gate went quiet on a config it had just rejected. That is now its own rule with its own mutation tests.
- config_snapshot vs the migrated baseline: CHANGED empty.

## STATE 5 — The MLE descent gate becomes a stage BLOCK.

The MLE descent gate becomes a stage BLOCK. `flags: {mle_gate: true}` plus top-level `mle_slope_t` / `mle_min_rate` / `mle_slope_window` becomes `mle_gate: {slope_t, min_rate, window}` on the stage that declares it. The switch and its settings were in different places, and the settings read as global config though only the one MLE stage ever consulted them -- so a reader had no way to tell they were stage-scoped, and a second MLE stage could not have differed. Presence of the block is now the switch; `{}` means gate on with defaults, which still declares intent where a missing block does not.

**Components:** `protocol.py`, `train.py`, `configs/mk_dev.yaml`, `utils.py`

**Moved:**

- `mle_slope_t` -> the declaring stage's mle_gate.slope_t
- `mle_min_rate` -> the declaring stage's mle_gate.min_rate
- `mle_slope_window` -> the declaring stage's mle_gate.window

**Invariants:**

- A stage publishing gates/mle_flat carries the parameters that shape it, so a stage's exit condition and the metric it reads are declared together.
- mle_gate.window >= 40: the gate samples every 10 steps and the slope regression needs at least 4 points, below which it fits noise.

**Validation:**

- protocol.Stage rejected the block before the change and accepts it after; the old flag spelling is now rejected, so a config cannot half-migrate.
- config_snapshot vs the migrated baseline: CHANGED empty.

## at state 5 — The eval metric streams stop disagreeing about what `conditional_worst_quantile` means.

The eval metric streams stop disagreeing about what `conditional_worst_quantile` means. NO CONFIG KEY CHANGES -- the knob already existed and its value is untouched; what changes is that the eval path now READS it. Every train-step site already passed it, but the two eval calls in log_metrics omitted it and took quick_tb_stats' 0.5 default, while log_test_metrics recomputed the train-condition stats at the config value and overwrote four 'eval_fwd/' keys on its way out. Dict-update order in evaluation() decided which definition reached wandb. Because that overwrite only ran when a held-out set was configured, `eval_fwd/tb_err_worst` meant the MEDIAN condition on a run without `test_molecules_path` and the upper-tail one on a run with it -- the meaning of a published series set by an unrelated key. All three eval streams now go through one call site. READING OLD RUNS: `eval_fwd/{tb_err_worst, z_grad_worst}` on a conditional run with no held-out set, and `eval_bwd/*` on every conditional run, were logged at 0.5 and are NOT comparable with post-change runs; the same keys on a run with a held-out set already carried the config value and are unaffected. Unconditional runs are unaffected throughout -- with one condition group the quantile is inert. No gate, controller or tracker reads these keys (the sole eval-stream consumer is protocol.py's bootstrap_z handoff, which reads `eval_fwd/jensen_z`; that is a mean and carries no quantile), so no training behavior changes -- only what is logged.

**Components:** `train.py`, `test_condition_fractions.py`, `docs/module_metrics.md`

**Invariants:**

- Exactly one eval-time quick_tb_stats call site (Modeller._eval_conditional_stats), so an argument that changes what a shared metric NAME means cannot be set on one stream and not another.
- log_test_metrics writes only under 'eval_test/'. The lone shared key is the unprefixed `Cond * Bar`, which reaches the merge at most once because _log_setting's cache makes the second writer a no-op.
- Metric dicts are merged with Modeller._merge_metrics, which refuses to overwrite an existing key rather than letting call order pick a winner.

**Validation:**

- Read the numbers rather than the code: on a skewed synthetic condition set, tb_err_worst and z_grad_worst move between the two quantiles while cond_tb_err and logw_std_within do not -- confirming the blast radius was the *_worst family only.
- MUT-4 (test_condition_fractions.test_merge_metrics_refuses_a_silent_overwrite) re-introduces the exact overwrite that shipped and requires an AssertionError, so the guard cannot go blind.
- The held-out test asserts no 'eval_fwd/' key leaks and that the train-condition stats are computed once per eval, not twice.
- Fast tier: 680 passed.

## at state 5 — Stage-exit `patience` counts METRIC WRITES instead of trigger checks, and two config-load rules reject exit conditions that cannot fire.

Stage-exit `patience` counts METRIC WRITES instead of trigger checks, and two config-load rules reject exit conditions that cannot fire. NO CONFIG KEY CHANGES. Every value an exit term can read persists its last value -- metric_tracker is an EMA dict and `gates/*` a plain one -- while `_exit_tick` ran every 10 steps and advanced the streak on whatever `_resolve` returned. A term therefore counted ONE sample as N: measured on the prod0810 phase-1 block, a single `bwd/tbc` write at step 100 carried the streak to 20 over 20 quiet ticks, and one `gates/mle_flat` publish cleared its `patience: 5` three ticks later. `patience` on an `eval/*` term had the mirror defect -- accepted by _parse_exit, then silently discarded, so it fired on the first clean eval. THIS CORRECTS docs/design/next_battery.md 1.3, which recorded the opposite mechanism (a streak resetting on quiet ticks and never accumulating). The pinned `protocol/exit_streak_eval_wass_debiased` = 0 that prompted that reading was a LOGGING ARTEFACT: eval terms were skipped by the tick loop, so the series meant 'never judged here', not 'never passes'. READING OLD RUNS: any pre-change `protocol/exit_streak_*` on a metric slower than 10 steps over-counts, and eval-term streaks are structurally 0; neither is comparable with post-change runs. Stage transitions that already fired are unaffected in ORDER but a stage whose patience was cleared by stale reads may now dwell longer, which is the intended correction.

**Components:** `protocol.py`, `utils.py`, `config_invariants.py`, `test_protocol_exit_streaks.py`, `test_config_invariants.py`, `docs/design/next_battery.md`

**Invariants:**

- A streak advances only on a FRESH write: fresh pass advances, fresh fail resets, no fresh write HOLDS. Holding is load-bearing -- resetting on a quiet tick would make patience > 1 unreachable for every metric slower than the 10-step tick, which is the bug 1.3 described rather than the one that existed.
- MetricTracker.written_at is within-process and NOT checkpointed, so a restored stamp can never claim freshness nobody observed.
- `patience` means the same thing on every term kind; eval/* terms keep a real streak advanced once per eval.
- config_invariants abstains rather than guesses: exit_bar_is_within_measured_range only judges metrics in MEASURED_METRIC_RANGES, and a missing entry is not a pass.
- That rule is BASELINE-ONLY and pinned so by its own test. It reasons from a MEASUREMENT, which is not a property of the config being checked -- the 17.1 floor was read off a battery with five railed controls, so the configs written to unrail them are the ones that should be free to aim under it. Only exit_patience_is_reachable can ERROR, and only on the arithmetic case (patience x cadence > epochs), which is provable from the file alone.

**Validation:**

- Mechanism established by RUNNING the real prod0810 exit block through the engine, not by reading it -- which is what showed the reported cause was inverted.
- Three mutations, each required to FAIL the suite: M1 removes freshness gating (6 failures), M2 restores the discarded eval-term patience (3), M3 implements the wrong fix -- reset on a quiet tick -- and is rejected by 4.
- The positive path is asserted separately: the real three-term phase-1 block driven at real cadences must still transition, so a fix that tightened the stage into never advancing cannot pass.
- exit_bar_is_within_measured_range reports on configs/mk_dev.yaml the moment `protocol:` is switched to conditional_vargrad, which still declares `fwd/logw_std_within < 6.0` -- and `errors()` stays EMPTY there, so the route still loads and runs.
- Full suite minus bench: 848 passed, 2 pre-existing failures in test_replay_gating.py (its fake modeller lacks `lr_controller`, unrelated to this change).

## STATE 6 — `condition_block_m` moves from the two buffer blocks to the two loss-coefficient blocks: buffers.

`condition_block_m` moves from the two buffer blocks to the two loss-coefficient blocks: buffers.prior_buffer -> bwd_loss_coeffs, buffers.replay_buffer -> replay_loss_coeffs. It never configured a store -- the buffer holds the same rows either way -- it shapes the DRAW for one loss, and both read sites were already gated on that loss's own coefficients (bwd on vg_lb; replay on vg_by_condition plus vg_lb/vg_lme). So the value only ever meant anything in conjunction with coefficients living in a different block, and a reader of either block alone could not tell whether it was active. THE MOVE ALSO CHANGES ITS SCOPE, which is the point rather than a side effect: loss coefficients are per-stage overridable (protocol.coeffs merges stage `loss_coeffs` over the base block), so a protocol can now run blocked draws in the VarGrad stage and independent ones elsewhere by declaring it. The vg_lb gate was emulating exactly that with the one stage-scoped signal it had. Value and semantics are unchanged: >= 2 blocks the draw, 0/1 does not, and the activation gates are untouched.

**Components:** `train.py`, `utils.py`, `config_invariants.py`, `configs/mk_dev.yaml`

**Renamed:**

- `buffers.prior_buffer.condition_block_m` -> `bwd_loss_coeffs.condition_block_m`
- `buffers.replay_buffer.condition_block_m` -> `replay_loss_coeffs.condition_block_m`

**Invariants:**

- A knob that is inert unless a loss coefficient is nonzero lives with that coefficient, so presence and activation are readable in one place.
- The value is read as an INT at the draw site: _sample_condition_blocked_indices passes it to np.random.choice's `size`, which rejects a float. Loss-coefficient blocks are written in floats throughout, so `2.0` is now a spelling a config can reach and the cast is load-bearing rather than defensive.
- vargrad_needs_groups reads it through config_invariants._coeff, the same resolver as `repeats`, so the rule sees a stage override instead of only the base block.

**Validation:**

- test_config_state.py: the retired-key gate and the transitions describe the same set (both old paths retired, both renamed), and the canonical config migrates clean at the new version.
- test_config_invariants.py: the bwd disjunction mutation test now drives the coefficient, and its aug13/aug14 spellings still resolve the same way.

## STATE 7 — The MODE-KEY MIGRATION gets its load gate, three states late.

The MODE-KEY MIGRATION gets its load gate, three states late. States 4-6 moved the keys a mode switch used to hand-edit so that selecting a protocol carries them: {fwd,bwd,replay}_tb_z_source into the per-branch loss_coeffs a stage can override, z_calibration into a stage flag, and lr_flow into an on_enter action. The moves shipped; THE RETIREMENTS DID NOT. So the old spellings kept loading clean and being read by nothing -- measured on the real load path with a known-retired key as a positive control -- which is the precise failure _RETIRED_KEYS exists to prevent, and it had already cost something: every arm of configs/a100_stab_aug16 wrote tb_z_source into the dead home, where it resolved to the code fallback `learned` on a CONDITIONAL route -- the F-042 detonation value -- while the file read `persistent` in three places and stamped state 6. A version stamp asserting `migrated` over unmigrated contents is worse than no stamp, because it answers the question a reader would otherwise ask. adaptive_lr.envelope_freeze_drop rides along: it became unnecessary rather than wrong, when the ramp-exit detector stopped being the actuator and left no noise in peak_scale to threshold. One re-snapshot pays for all three.

**Components:** `utils.py`, `config_state.py`, `controller.py`, `configs/mk_dev.yaml`

**Added:**

- `adaptive_lr.envelope_freeze` = `True`

**Renamed:**

- `condition_log_z.fwd_tb_z_source` -> `fwd_loss_coeffs.tb_z_source`
- `condition_log_z.bwd_tb_z_source` -> `bwd_loss_coeffs.tb_z_source`
- `condition_log_z.replay_tb_z_source` -> `replay_loss_coeffs.tb_z_source`

**Removed:**

- `adaptive_lr.envelope_freeze_drop` -- replaced by adaptive_lr.envelope_freeze; only its on/off sense was ever reachable, and no tracked config set it.

**Requires judgment:**

- `z_calibration.enabled` -- became a per-stage flag, and the value does NOT carry. Measured over git-tracked yaml: 205 configs hold the key, and of the 10 that are state 4 or later, 8 set it TRUE while no stage declares the flag -- so they already ran with the sidecar OFF. Dropping the key preserves what those runs did; adding `flags: {z_calibration: true}` to a stage changes them. Which is right depends on what the config was for, so migrate reports it and refuses to guess.

**Invariants:**

- A retired key is rejected at LOAD, never read-and-ignored. The mode-key migration violated this for three states; the gate is what makes "the schema moved" and "your config moved" the same event.
- project_state_version means the contents match the state named. A config may only stamp 7 once the old spellings are gone, because they now hard-fail.
- envelope_freeze is a BOOLEAN. The freeze either arms or it does not; there is no threshold, because nothing noisy reaches it.

**Validation:**

- The gap was MEASURED before it was fixed: on the real load path (dict2namespace -> preflight_config) all four old spellings loaded clean, with mle_slope_t as a positive control that rejected -- so the harness could see a retired key and these were genuinely invisible.
- Counted over git-tracked yaml, not a naive grep: 440 configs carry each *_tb_z_source key, 205 carry z_calibration.enabled (166 true, 39 false), and ZERO carry envelope_freeze_drop -- which is what makes that one a pure drop rather than a judgment call.
- Only 10 configs are state 4 or later, i.e. mechanically migratable at all; the rest predate the protocol library and need a stage-list rewrite regardless. Those 10 were migrated and re-verified.

## STATE 8 — THE BATCH SIZER IS REPLACED (phase 6: replace, do not patch further).

THE BATCH SIZER IS REPLACED (phase 6: replace, do not patch further). The throughput-saturation walk -- grow until samples/sec flattens, pin, periodically recheck downward -- is deleted from train.py, because its objective was decided against it (user, 2026-08-16): optimizer steps/sec at a threshold effective batch A = fused_grad_accum_min_samples, under which the throughput optimum is the constant B = A and there is no knee to find. Growth above the base batch is now occupancy's business alone: the new select_batch_size walks a finite ladder once per stage, measures step time and raw occupancy per rung over real train steps, holds the smallest rung clearing the new `batch_util_target` (absent/0 = off, the shipping default, under which the batch simply holds the base), declares INFEASIBLE loudly when no rung clears, and audits a kept growth against a full policy window, standing down if the growth did not deliver (docs/design/phase6_batch_sizer.md S1/S2/S3). Four keys of the walk retire with it; checkpoints swap the knee-pin fields (batch_size_saturated_stage/batch_size_pinned_at/batch_size_ever_oomed) for the sizer's conclusion dict, with no checkpoint back-compat by standing policy.

**Components:** `train.py`, `protocol.py`, `checkpointing.py`, `utils.py`, `config_invariants.py`, `bench/`, `benchmarks/registry.yaml`, `configs/mk_dev.yaml`

**Removed:**

- `auto_batch_throughput_opt` -- the walk it switched no longer exists; priority 2 is the constant B = fused_grad_accum_min_samples.
- `batch_growth_min_throughput_gain` -- the walk's saturation bar; with no walk there is nothing to bar. Its config_invariants rule (growth_gain_below_growth_factor) is deleted with it.
- `batch_knee_recheck_steps` -- paced the pin's drop-and-reclimb recheck. The sizer's conclusion is re-opened only by a stage transition, an OOM, or the OOM ceiling's expiry (batch_oom_ceiling_retest_steps, which survives).
- `batch_growth_slow_interval` -- the AIMD post-OOM regrow spacing; nothing regrows on an interval. batch_growth_interval survives as the per-rung calibration dwell.

**Invariants:**

- With batch_util_target unset -- every existing config -- the sizer holds the configured batch_size and only the safety bounds (OOM ceiling, max_step_seconds, cooldown) ever move it. On the canonical 1000/1000 pair this is bit-identical to what the old walk could reach, since its domain had one rung.
- Occupancy evidence may only veto candidate sizes under a fixed selection rule (S1); no occupancy reading orders the batch on its own. gpu_util_floor stays retired.
- A set batch_util_target must be actuable: config_invariants.util_target_actuable hard-errors a target paired with grow_batch_size: false or max_batch_size <= batch_size, so the flag cannot load as inert reassurance.
- Every safety mechanism is a domain bound, never a selector: an OOM ceiling shrinks the ladder, it cannot pin a selection.

**Validation:**

- bench/test_batch_traps.py: trap (a) still convicted when injected and not convicted on the replacement; the structural horizon-invariance (B1) now holds at n_distinct=1 under FLAT with no util target; an injected floorless descent walk still breaks it.
- test_config_state.py: the retired-key gate and this transition describe the same four keys, and the canonical config migrates clean at state 8.

## STATE 9 — `batch_util_target` IS REINTERPRETED FROM PERCENT TO A FRACTION of the card (0.

`batch_util_target` IS REINTERPRETED FROM PERCENT TO A FRACTION of the card (0.6 = 60% busy), and the canonical config ships the occupancy ladder ARMED for the first time: batch_util_target 0.6, grow_batch_size true, max_batch_size 20000 -- the last of which puts F-045's missing ELJ 15-25k rung inside the ladder's domain. The unit change is not cosmetic: the two spellings are NOT distinguishable by inspection at the value that matters. Under the percent reading 0.6 is a legal 0.6% target, which every rung clears, so the ladder holds the first rung it measures and the constraint reports itself as served while serving nothing -- precisely the inert-flag failure util_target_actuable exists to catch, and one its old (0, 100] range could not see. The sensor still reports percent; the single conversion lives at the one read site in train.select_batch_size.

**Components:** `train.py`, `config_invariants.py`, `configs/mk_dev.yaml`, `bench/fake_modeller.py`

**Invariants:**

- batch_util_target is a fraction in (0, 1]; a value above 1 is the pre-state-9 percent spelling and hard-errors at load (config_invariants.util_target_actuable).
- The actuability clauses are unchanged: a set target still requires grow_batch_size true and max_batch_size > batch_size, so the armed canonical config satisfies its own rule.
- Nothing about the control law changes -- S1/S2/S3 and the capped-geometric ladder are as shipped at state 8. Only the unit in which the target is written, and where it is converted.

**Validation:**

- Local shakeout 2026-08-19 (configs/synth_aug19/): the ladder walks and advances 1000 -> 1600 on the shipped capped-geometric step, with the per-rung reading taken from raw occupancy samples rather than the trailing windowed mean.
- MEASURED consequence of arming the ladder, and the reason max_batch_size is a budgeting decision rather than a free bound: at max_batch_size 20000 the ladder is 21 rungs, and a rung needs both a batch_growth_interval dwell and 3 occupancy samples at gpu_util_sample_period_s -- >= 63 min of calibration per stage on a fast route, and ~2.5 h per rung at prod0810-scale MLIP step times, where the walk cannot finish inside a job at all.
