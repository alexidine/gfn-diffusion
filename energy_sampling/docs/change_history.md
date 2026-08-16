# Project change history

Generated from `config_state.CHANGES` -- do not edit by hand.

One entry per material functional change. A change marked **STATE N** is
one that altered how persisted state is interpreted, and is the only kind
that moves `project_state_version` or carries a migration; the rest record
what changed and why. The line-level diff is in git.

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
