# Replay buffer

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

The replay buffer is a store of forward trajectories. A row is a terminal state, the full trajectory that produced it, and the log reward computed at admission; the replay branch draws rows and re-scores the stored path under the current $P_F$ and $P_B$, calling no energy function. Rollout cadence and its cost are on [rollout-cadence-and-dose](rollout-cadence-and-dose.md), what the replay loss does with the rows is on [loss-composition](loss-composition.md), and the branch weights are on [balance-controllers](balance-controllers.md). The buffer is built lazily: `train.py::Modeller.replay_in_play` derives from the stage, not from a config switch, whether the stage has a use for it, and `train.py::Modeller.manage_replay_buffer` returns immediately when it is false.

## Admission

Every admission runs through `manage_replay_buffer`, called once per forward rollout and, under `cfg:buffers.replay_buffer.admit_from_eval`, once per evaluation. It computes the per-row residual $\text{resid} = (\log P_F - \log P_B) - (\log R - \log Z)$ on the rollout's rows and builds an eligible pool by two hard exclusions: non-finite $\log R$ or residual, and $\log R$ below `cfg:buffers.replay_buffer.admit_reward_min`.

Selection within that pool is uniform without replacement, via `train.py::_uniform_draw`. The budget is `cfg:buffers.replay_buffer.churn_rate` rows per call, scaled by the live batch against the configured one so a grown batch keeps the fraction the config expressed; at `churn_rate` 0 the budget is the live batch size, which is store-all at forward `repeats` 1 and a uniform $1/R$ of the rollout at `repeats` $R$. Admission writes the columns that never change again: `birth_step`, `birth_log_pf` (NaN when the call is not on-policy, the eval-site case, so those rows leave the drift statistic and show as `replay/policy_drift_covered_frac` below 1), `birth_loss`, `is_val`, `origin`, and the stored terminal-force legs where the replay loss reads them. `ema_loss` is seeded from $|\text{resid}|$; `ema_logw` is born NaN and is written exactly on a row's first draw.

Residual-scored admission is retired: `buffers.replay_buffer.admit_cap_max`, `admit_cap_min`, `admit_cap_health_h0` and `admit_temperature` are refused at load by `utils.py::_RETIRED_KEYS`, whose message states that admission and the displacement purge both draw uniformly.

## Residence

Eviction has two residual-independent causes, both keyed on `cfg:buffers.replay_buffer.mean_residence_steps`, written $\tau$. The **hazard** is memoryless: a fraction $\min(1, \Delta/\tau)$ of the surviving rows is evicted uniformly at random, with $\Delta$ the steps since the last managed call, so the budget is per *step* and $\tau$ means steps at any cadence; an eval-site call landing on the same step as a fused one has $\Delta = 0$ and spends nothing. The **backstop** is a hard age ceiling at `cfg:buffers.replay_buffer.backstop_mult` times $\tau$, evaluated before the hazard; at 0 there is no ceiling. `replay_buffer_backstop_frac` and `replay_buffer_hazard_frac` split resolved exits between the two, and $\tau = 0$ disables both arms.

`cfg:buffers.replay_buffer.max_size` is a third exit. Headroom is computed after those two causes, and whatever admission needs beyond it is freed by a further uniform-random draw over the live incumbents. Where the cap binds rather than giving headroom over the hazard's equilibrium, eviction is displacement: rows leave by overflow before the hazard would take them. The canonical config's comment on the key states that this breaks `birth_loss` as an unbiased intake baseline, and carries a `todo` stating that the cap currently binds.

`buffers.replay_buffer.max_residence_steps`, the hard age cap that preceded the hazard, and `toxic_min_draws`, which fed a residual-dependent stalled-row purge, are both refused at load.

## The draw

`train.py::Modeller.draw_replay_sample` is the replay branch's only source of rows, called from `train.py::Modeller.replay_train_step`, and returns a full batch. With the `cfg:buffers.replay_buffer.prioritise` block absent or `enabled` false, `train.py::Modeller.replay_priority_config` returns None and the draw is uniform over the trainable rows.

Under `prioritise`, `buffer.py::CrystalBuffer.prioritised_weights` builds the measure from the signed residual reconstructed per row as $\delta_i = \log Z - \texttt{ema\_logw}_i$; `ema_loss` stores $|\text{resid}|$ and carries no sign. The score is $\delta^+ = \max(\delta, 0)$ by default and $|\delta|$ under `cfg:buffers.replay_buffer.prioritise.symmetric`; a never-drawn row, whose `ema_logw` is NaN, takes the 0.90 quantile of the observed scores rather than zero. Rows scoring exactly zero are ineligible and drawn at $p = 0$ rather than floored in. Under the one-sided score that exclusion is one-way: `ema_logw` is written only at draw time, by `buffer.py::CrystalBuffer.update_logw_stats` on the drawn indices, so a row that falls to $\delta \le 0$ holds a frozen estimate and can re-enter only if $\log Z$ drifts back past it. Under `symmetric`, $|\delta| > 0$ fails only at exact float equality, so every row stays eligible. The canonical config sets `symmetric: true`; the code default is false.

Eligible scores are floored at `cfg:buffers.replay_buffer.prioritise.floor_frac` times their median, which bounds the weight range by $(\text{median}/\text{floor})^\kappa$, then raised to $\kappa$ = `cfg:buffers.replay_buffer.prioritise.kappa` and normalised. The per-row weight returned beside the measure is $w_i = (1/n_{\text{elig}})/p_i$, so $\mathbb{E}_p[w f] = \mathbb{E}_{\text{uniform over eligible}}[f]$ and the unnormalised estimator is unbiased at every $\kappa$; the loss self-normalises over the drawn batch, which carries an $O(1/n)$ finite-batch bias. `replay/is_ess_frac` and `replay/is_w_max_ratio` report the batch weight tail.

`beta` inside `buffer.py::CrystalBuffer._sample_indices` is the *fraction of the batch drawn uniformly*, not a temperature: the batch splits into $\lfloor B\beta \rfloor$ uniform rows and the rest from $p$. `draw_replay_sample` passes 0 whenever it computed a $p$ and 1 otherwise, so a prioritised draw comes entirely from $p$. Two other draw shapes exclude prioritisation: the condition-blocked draw (`cfg:replay_loss_coeffs.condition_block_m` at 2 or more, read as 0 unless condition-grouped VarGrad is running on the replay branch) returns before $p$ is consulted, and the aligned per-condition draw fixes both the conditions and the rows per condition. Both raise rather than combine.

## The held-out split

`cfg:buffers.replay_buffer.val_frac` flags that fraction of each admission batch as held out, Bernoulli per row, in `train.py::_val_flags`. Held-out rows are never drawn for training on any path: `_sample_indices` reads `is_val` itself, and `prioritised_weights` takes the same mask as `exclude` before the eligible count, the median and the weights are computed; the draw asserts the two agree. The split is refused beside `condition_block_m` at 2 or more, which selects whole conditions before any row mask is read. At `val_frac` 0 no row is flagged and no mask is built.

`buffer.py::CrystalBuffer.sample_val_graphs` draws the probe uniformly without replacement, and not through `loader`, so held-out rows never register a draw count. The probe size is $\min(\texttt{val\_cap}, B_{\text{live}}, n_{\text{val}})$ and is 0 below `cfg:buffers.replay_buffer.val_min` (`train.py::Modeller._replay_val_size`), so a `val_cap` above the batch size is inert.

`train.py::Modeller._replay_val_stats` scores the probe at the same parameters and the same $\log Z$ as that step's training draw, through the stored trajectories. It reports `val_gap`, the mean Huber loss val minus train, in nats squared; `val_gap_nats`, the median $|\text{resid}|$ val minus the importance-weighted median train (`train.py::Modeller._weighted_median`), in nats; and `val_gap_nats_se`, $1.858\,\text{MAD}/\sqrt{n}$ from the val side alone. Because the differenced statistic is a median of $|\delta|$ and $|\cdot|$ is nonlinear, a shared shift in $\delta$ does not cancel algebraically between the sides. The standard error omits the training median's variance and the covariance of the two, so it bounds the gap's true standard error in neither direction.

## Two row properties

A resident row carries two independent histories: its **age**, the steps since `birth_step`, which moves whether or not the row is drawn and which the eviction arms act on, and its **draw count**, `select_counts`, which only the draw moves and which the loss acts on. Two instruments split the same way: `train.py::Modeller._policy_drift_stats` compares current $\log P_F$ against `birth_log_pf` on the same trajectory, an age quantity that never reads $\log Z$, while `replay/val_gap_nats` contrasts trained rows against never-trained ones, a draws quantity.

The population relations (docs/design/replay_population_structure.md) are approximations, not identities. With $B$ the batch, $N$ the effective steps between rollouts and $v$ the held-out fraction:

$$
\text{admissions per step} \approx \frac{B}{N}, \qquad
O \approx \frac{B\tau}{N}, \qquad
\text{draws per trainable row} \approx \frac{N}{1 - v} .
$$

The design note states what breaks the exactness: the hazard is a rounded per-call fractional budget rather than continuous exponential death, the backstop truncates the tail, overflow at `max_size` evicts irrespective of age, admissions arrive as bursts of $B$ every $N$ steps rather than as a stationary stream, and a batch change moves $B$ under a population that needs about $\tau$ steps to follow. Memoryless lifetimes give mean resident age $\tau$ only under stationary arrivals. `mean_residence_steps`, `max_size` and `churn_rate` all enter mean row age.

## The intake baseline

`birth_loss` is a frozen copy of `ema_loss` at admission. `ema_loss` is an EMA updated by `buffer.py::CrystalBuffer.update_losses` on drawn rows only, so an undrawn row has `ema_loss` equal to `birth_loss` exactly, and a held-out row does by construction. `buffer.py::CrystalBuffer.absorption_stats` publishes the ratio over rows not held out, with both values finite and `birth_loss` positive, and returns nothing below eight such rows:

$$
\texttt{replay/resid\_vs\_intake} = \frac{\overline{\texttt{ema\_loss}}}{\overline{\texttt{birth\_loss}}} \in (0, 1],
$$

with `replay/absorbed_frac` its complement, `replay/lambda_tau` its negative log, and `replay/absorption_n` the row count. `birth_loss` exists only for rows still resident, so the denominator is the intake distribution of survivors; the docstring's survivorship argument is that a residual-independent hazard makes that an unbiased sample of admissions. Three effects enter the ratio together: a fall in the residual from learning, a fall from fitting the stored rows themselves, and a shift in $\log Z$ that re-signs every residual at once.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:buffers.replay_buffer.{churn_rate, mean_residence_steps, max_size, admit_reward_min, backstop_mult, admit_from_eval, val_frac, val_min, val_cap}`, `cfg:buffers.replay_buffer.prioritise.{enabled, kappa, floor_frac, symmetric}`, `cfg:replay_loss_coeffs.condition_block_m`, `cfg:batch_size`.

Code: `train.py::Modeller.manage_replay_buffer`, `.draw_replay_sample`, `.replay_train_step`, `._replay_val_stats`, `._replay_val_size`, `._replay_val_frac`, `._policy_drift_stats`, `._weighted_median`, `.replay_priority_config`, `.replay_priority_symmetric`, `.replay_priority_floor`, `.replay_in_play`, `._buffer_core_stats`; `train.py::_uniform_draw`, `::_val_flags`; `buffer.py::CrystalBuffer.prioritised_weights`, `.absorption_stats`, `.sample_val_graphs`, `.update_losses`, `.update_logw_stats`, `.add`, `.purge_by_index`, `._sample_indices`, `.loader`; `utils.py::_RETIRED_KEYS`.

## Could be tooling

`replay_buffer_length`, `replay_buffer_mean_age`, `replay_buffer_age_cv`, the admitted and evicted counts and the two eviction-cause fractions are published per logging window, and the three approximations above are functions of `churn_rate`, `mean_residence_steps`, the live batch and the effective cadence. A script reading a run's config beside its logged series would report predicted against observed occupancy and reuse, and name which exactness break is live there: whether `replay_buffer_length` equals `max_size`, whether `replay_buffer_backstop_frac` is materially above zero, and whether the batch moved during the window.

The draw's mutual exclusions live in the raises inside `draw_replay_sample` and `_sample_indices` and in the canonical config's comments; a pass collecting the raise sites keyed on config attributes would give the refusal set as data to diff those comments against.

## Sources

Repo: docs/design/replay_population_structure.md (birth, use and eviction; the corrected relations and the instruments), docs/design/replay_occupancy_and_cadence.md, and the code above read at the stamped commit, with the `buffers.replay_buffer` and `prioritise` blocks of the canonical config. Memory files were used to locate the code and not as evidence: project_replay_buffer_band_redesign, project_replay_inversion_is_loss_weight, project_val_gap_bar_calibration.
