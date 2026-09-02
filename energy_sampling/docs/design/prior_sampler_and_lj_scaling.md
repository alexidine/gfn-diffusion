# Scope: prior-sampler restoration + lj_coeff as carried data

Two independent changes, written 2026-09-02. Job 1 is small and operational;
job 2 is a currency change across two repos.

---

## Job 1 — the prior sampler does not survive a resume

### What is broken

`_has_prior_sampler()` is `hasattr(self, 'prior_model')` (`train.py:534`). That
object is produced ONLY by `snapshot_prior`, the `train_prior` stub's `on_exit`
action (`protocol.py:1733`). It is not in the checkpoint and nothing rebuilds it.

- seed from `phase1_exit` -> arm enters the stub, exits, `snapshot_prior` fires,
  churn works.
- seed from `_running.pt` -> arm resumes INSIDE `equilibration`; `advance()`
  never runs, so neither the stub's `on_exit` nor equilibration's `on_enter:
  rebuild_prior_by_churn` fires. No prior model for the rest of the run.

`_prior_churn_cycle` then returns early at its guard (`train.py:7213`) and the
prior buffer freezes: `added` 0, `evicted` 0, `turnover` 0.

### Measured, 2026-09-02

| run | seeded from | admit_rate | added | turnover |
|---|---|---|---|---|
| `mb31_*` (x7) | phase-1 exit | 0.78-0.81 | 2000-3500 | 0.028-0.049 |
| `acrb31_*` (x3) | phase-1 exit | 0.87 | 3500 | 0.049 |
| `pt100p2_mip2_lr1p0` | `_running.pt` | **NaN** | **0** | **0** |
| `pt100p2_neh2_lr1p0` | `_running.pt` | **NaN** | **0** | **0** |

Splits perfectly on how the arm was seeded. The first-launch arms are not
immune -- they lose it at their first 24 h requeue.

### Why it matters

`bwd_sampling_mode: 'prior'` draws the backward branch from that buffer. Frozen
means bwd trains on a fixed, ageing dataset. Candidate cause for `bwd/r2` and
`under_coverage` degrading while fwd and replay improve.

### The fix, and the constraint on it

`train.py:2648` records that auto-discovery of a prior was DELIBERATELY DELETED:

> the old `reuse_prior` auto-discovery (this run identity's own `*_prior.pt`,
> then `find_shared_prior` over any matching `problem_def`) is deleted: it made
> "which prior did this arm train against" a function of what happened to be on
> disk [...] A stage that needs a prior now either gets a path or trains one.

So the code must not silently re-acquire a prior. Two parts:

1. **Operational** -- the sbatch sets `prior_model_name` on the REQUEUE branch
   only, where the file provably exists. This is "gets a path", the contract the
   deletion left in place, rather than re-adding discovery.
2. **Observability** -- emit `prior_buffer_has_sampler` as a metric. Today the
   only tell is `prior_buffer_prior_admit_rate` going NaN (0/0), plus one stdout
   line once per stage. A silent capability loss must be visible in wandb.

NOT done, needs an owner decision: loading this run's own `path_for('prior')`
automatically on resume. It is narrower than what was deleted (same run_name,
same problem_slug, written by this very run) but it IS re-adding auto-discovery
and reverses a documented decision.

---

## Job 2 — lj_coeff rides with the data

### The principle (owner, 2026-09-02)

`.y` / `.elj` is a property of the bare crystal: `lj_coeff * mol_energy` and
nothing else. Jacobian, pressure, density, bounding and reduction belong to
REWARD calculations only. Bare `.elj` is meaningless in scale, so the
coefficient should travel with the data and apply automatically, rather than
being passed at every call site (forgettable) or applied at every use (already
demonstrably messy).

### Why this is tractable

`lj_coeff` is applied in exactly ONE place on the crystal route --
`molecular_crystal.py:645` -- and written in one, `train.py:2595`, where
`thermal_scaling_factor` from the prior file overrides the config for the whole
run. Everything else matching `lj_coeff` in the tree is documentation about that
override.

### The change

1. **mxtaltools**: `compute_eLJ_energy` multiplies by a per-graph `lj_coeff`
   attribute, DEFAULTING TO 1 when absent (the library stays usable by other
   consumers).
2. **gfn**: drop the `self.lj_coeff *` factor at `molecular_crystal.py:645`.
   The composite total is then arithmetically unchanged -- same number, computed
   one line earlier.
3. **gfn**: stamp `lj_coeff` onto batches at the entry points -- prior load,
   buffer restore, sample construction.
4. **gfn**: at those same boundaries assert PRESENCE and EQUALITY (within a
   float tolerance) against the run's value. Permissive library, strict
   application. Equality is what catches two sources that are both stamped but
   disagree -- a prior at 0.3636 mixed with an anchor set at 1.0.
5. **migration**: absent attr -> stamp the run value and multiply the stored
   `.elj` column by it. The stamp DOUBLES AS THE MIGRATION MARKER, so the
   migration is idempotent for free and cannot double-scale.

### What does NOT need repair

Moving the multiplication leaves every composite numerically identical, so
`condition_log_z.best_energy`, stored rewards, replay losses and model weights
are untouched. Only the stored `.elj` column changes meaning.

### Why the migration is mandatory, not cosmetic

`generator_energy` READS the stored attribute -- `mol_energy =
getattr(crystal_batch, self.energy_function)` -- it does not recompute it. Rows
re-scored through `prebuilt_sample_to_reward` go straight through that path. So
an un-migrated prior or buffer sidecar does not produce slightly-off
diagnostics; it produces wrong training energies on every backward and replay
draw, short by the coefficient (~2.75x on mipcas).

### Decisions taken

- `/z_prime` stays where it is (owner: already managed).
- Library defaults to 1; the GFN side is strict.
- Old data is migrated gracefully rather than rejected.
- Run-level coefficient sourced from the prior file, as today.

### Assumption the migration rests on

"Absent" must mean "raw". That is true of every file on disk today, because the
coefficient has only ever been applied at score time (`:645`), never baked into
a stored column. It is a one-time property and would not survive anyone writing
a pre-scaled `.elj`. Asserted in the migration itself.

---

## Deferred

`ramp_floor` (100), `screen_energy_window` and `thin_energy_window` are absolute
constants tuned against the composite currency. Under job 2 the prior-gate
comparisons move to the bare crystal energy and all three need re-deriving. Not
attempted here -- they need a run to validate, with the owner present.

Also deferred: splitting `condition_log_z.best_energy` into a structural minimum
(for the six prior/anchor gates) and a reward-currency minimum (for the Excess
Energy metric at `train.py:6328`, which subtracts it from `-log_r*T`).

---

## MEASURED, 2026-09-02: the currency gap on our own config

256 real mipcas rows scored through the CURRENT code (`golden_pre.pt`), with
`lj_coeff` = 0.363584 taken from the prior file's `thermal_scaling_factor` exactly
as `train.py:2595` does. `Emin(c)` stands in as the min composite total over the
batch, -152.310.

The expiry / admission test is `stale = (row_energy - Emin(c)) >= ramp_floor`,
`ramp_floor` = 100. Only the left-hand side differs between the options:

| left-hand side | mean | min excess | max excess | fires? |
|---|---|---|---|---|
| TODAY: `y` = raw elj | -357.671 | -235.186 | -194.380 | **NEVER** |
| FIX A: `_prior_row_energy` = `physical_energy` | -140.720 | 0.000 | 20.243 | not yet |
| FIX B: owner's `lj_coeff * mol_energy` | -130.043 | 11.423 | 26.259 | not yet |

**TODAY is unreachable by ~294 units**, not merely quiet -- confirmed by
measurement on this config, not inferred from the qm9c numbers in
[[project_prior_buffer_scores_raw_elj]]. The currency offset here is
`total - elj` = **216.95** mean (208.8 to 235.2).

**Neither fix fires immediately**, which vindicates the correction to "expiry
will start firing for the first time": both become REACHABLE, and then fire only
once Emin(c) falls a further 79.8 (FIX A) or 73.7 (FIX B). Expiry is a function
of frontier movement, not of turning the channel on.

**FIX A is exactly commensurate on this data**: min excess is 0.000, because the
minimising row IS the argmin and `bounding_energy` is exactly 0.0 on all 256
rows. That is the empirical form of the earlier argument that bounding vanishes
at the argmin.

**FIX B carries a systematic +10.7 bias** against a COMPOSITE Emin --
`total - lj*mol` is -10.677 mean (-12.3 to -4.9), dominated by the Jacobian. So
mixing FIX B's row side with today's Emin fires ~11% early against a bar of 100.
That bias disappears entirely if Emin also becomes structural, which is the
owner's stated intent. The two coherent pairings are therefore:

  * FIX A rows + composite Emin -- commensurate, cheap, keeps one tracked minimum
  * FIX B rows + STRUCTURAL Emin -- commensurate, and the quantity the owner
    actually wants; needs the second per-condition minimum (see Deferred)

Mixing across the pairings is what produces the 10.7 bias. Both are self
consistent; only the mixture is not.

### Arithmetic confirmed end to end

    physical = lj_coeff*mol + density_coeff*density + pressure
               + reduction_coeff*reduction + jacobian
    0.363584 * (-357.671) + 0 + 0.008506 + 0.000506 + (-10.686) = -140.720  [matches]
    total = physical + bounding_total, and bounding_total = 0 on every row here

### Local reproduction

`mk_dev` hashes to `e01bd1`, IDENTICAL to
`dev_elj_p2_cruise_..._phase1_exit.pt` in `D:/crystal_datasets/gfn_checkpoints`
(stage `train_prior`, step 2910, train_T 10), which has its buffer sidecar. So
the equilibration medium-run can be done entirely locally.

---

## MEASURED: what actually needs migrating (narrower than feared)

`init_prior_dataset` RE-ANALYSES the whole prior dataset at every load --
`batched_analyze_crystal_batch(..., return_batch=True)` at `train.py:2607`,
confirmed running locally ("Re-analyzing prior energies"). And `train.py:2595`
sets `lj_coeff` from `thermal_scaling_factor` BEFORE that pass, deliberately, so
"every generator_energy call this run, init included, uses one coefficient".

Consequences for Change A:

* **The prior dataset is SELF-HEALING.** Its `.elj` is recomputed through
  whatever the current code does, never read from the pickle. Under Change A it
  comes back SCALED with no migration at all.
* **`pd.y` therefore rescales together with `sample_batch[y_fn]`**, which
  settles the open question about the progress gate's energy marginal: both legs
  of `energy_marginal_overlap` move by the same factor, so `E/emarg_overlap` and
  `E/emarg_overlap_rel` are INVARIANT and only the absolute `E/emarg_w1_kT`
  changes. No re-derivation of gate bars is needed -- and `w1r/*`, which the
  gate actually keys on, is computed from `_batch_latents`, not energies, so it
  never sees this at all.
* **The buffers are NOT self-healing.** `restore_buffers` rebuilds
  prior/replay/anchor stores via `from_state_dict` with no re-scoring, so their
  stored `.elj` is whatever was written. THOSE are what the migration is for.

So the migration's scope is: buffer sidecars and any stored batch that is not
re-analysed on load. Not the prior `.pt` files.
