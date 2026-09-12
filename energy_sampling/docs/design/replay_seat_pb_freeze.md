# Why P_B is frozen on the replay seat

Status 2026-09-12. Measured on the qm9c conditional route at lambda 0 (T=20, ELJ, pooled
VarGrad with `pooled_source: replay`, fwd as the Z(c) sidecar on 1 step in 20). Configs
`configs/qm9c_null_*.yaml`; wandb project "GFN Energy", runs `dev_qm9c_null_*`.

## The mechanism

The replay seat re-scores STORED trajectories under the current parameters
(`GFN.get_traj_replay`). On a fixed path the log weight is

    u = log R + log P_B(tau | x) - log P_F(tau)

and the pooled term is its within-condition variance. Widening P_F's per-step variance by
`s` shifts `u` on each stored path by `-s (q_F - mean q_F)`, where `q_F` is the path's
standardised quadratic form under P_F (spread ~sqrt(2 T d) across paths): a large penalty.
Widening P_B by the same `s` shifts it by `+s (q_B - mean q_B)`. Both kernels see the same
increments against nearly the same bridge drift, so `q_F ~= q_B` path by path and the JOINT
widening is a flat direction of the loss on any bag of stored rows. This holds whichever
loss moves P_B: it is a property of the values on the stored rows, not of the gradient
route (run C below).

The only force opposing the joint move is the reward on paths the widened policy actually
generates. On the fwd seat that arrives the next step. On the replay seat it arrives after
admission, with a lag set by residence and cadence, and it is a small fraction of the
per-step force (run 2 below). A frozen P_B removes the compensating half, so P_F's widening
is penalised immediately on every bag, fresh or stale.

## The runs

All from the same phase-1 exit and seed, batch 1000, LR scale 0.05 unless stated.

| run | seat | P_B | lag (residence / cadence) | outcome |
|---|---|---|---|---|
| `qm9c_null_fwdseat` | fwd | learned | 0 | stable, best of the series |
| `qm9c_null_replay_mirror` | replay | learned | ~1 (tau 2, churn 2000) | stable after entry transient |
| `qm9c_null_pbfree_fresh1` | replay | learned | ~72 (tau 72, cadence 1, reuse 1) | violent: pooled 2.96 at +400, LR fire +500 |
| `qm9c_null_replay_pbfree` | replay | learned | ~1200 (cadence 20, reuse ~17) | slow runaway: step_var 6x by +1450 |
| `qm9c_null_replay` | replay | FROZEN | ~1200 | stable 3000 steps |
| A `qm9c_null_fresh1_pbfrozen` | replay | FROZEN | ~72 | stable, monotone, tracks the frozen tau-1200 run |
| B `qm9c_null_fresh1_lr4` | replay | learned, LR/4 | ~72 | stable, damped step_var bump |
| C `qm9c_null_fresh1_pbdetach` | replay | learned, log P_B detached on replay rows | ~72 | no rescue: copy of fresh1 |
| 1 `qm9c_null_replay_pbfree_lr4` | replay | learned, LR/4 | ~1200 | slower drift, not stable |
| 2 `qm9c_null_replay_pbfree_fwdgrad` | replay | learned, fwd policy grad on rollout steps (0.4 weight) | ~1200 | no rescue |
| 3 `..._fwdgrad_pbonpolicy` | replay | learned ONLY from the fwd rollouts | ~1200 | no rescue |

Reuse per row, pair coherence (`churn_rate`) and row vintage are NOT the variable: fresh1
sees each row about once and was the worst run; the frozen tau-1200 run reuses each row ~17
times and was fine. Freshness enters only as a feedback delay, and any moving P_B destabilises
the seat regardless of which loss moves it.

Direct evidence of the compensation: with P_B learned the replay loss was BETTER than with it
frozen (1.055 vs 1.145) while the forward fit degraded, and `replay/policy_drift_std` read
~9 nats at row age 68 and at 266 (frozen: 1.3): P_F's density on its own recent paths moves
while `u` stays flat on the stored rows.

## Why a condition-blind frozen P_B is not a compromise

For a fixed backward kernel, trajectory balance has a unique forward solution and the
target path law is `P_B(tau | x) p_lambda(x | c)`. The condition enters only through the
terminal marginal, so the backward bridge given the endpoint never needs `c`. The lambda
path `p_lambda ~ p_0 exp(-lambda Delta)` is a re-weighting of the lambda-0 law, and the
phase-1 P_B is that law's posterior. P_F learns the conditioning onto a fixed unconditional
bridge. The costs are variance-side only: representability inside the Gaussian family
(already paid at lambda 0, r2 0.988) and a mismatch floor as P_F sharpens. If `1 - r2`
ratchets along the ladder, the remedy is a refit of P_B at the rung as the posterior of the
current P_F on fresh rollouts with P_F detached (still condition-blind), never an unfreeze
on the replay seat.

## What ships

- `freeze_pb` in `var_conditioning.on_enter` (mk_dev; full snapshot of trunk + head).
- Anneal gate on `<branch>/r2_unexplained` (= 1 - r2, `utils.quick_tb_stats`), relative to
  its running best, margin 1.1, drift 0.001/tick; cooldown 1000 steps; `lambda_mix` rate
  0.8 (x1.25 per event) from 0.001. Arm: `configs/qm9c_anneal.yaml` (`qm9c_anneal_r2gate`),
  generator `configs/qm9c_anneal/make.py`.
- `replay_loss_coeffs.detach_pb` / `bwd_loss_coeffs.detach_pb` (default 0): kept as
  instrumentation; not a fix for this route.
