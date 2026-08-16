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
