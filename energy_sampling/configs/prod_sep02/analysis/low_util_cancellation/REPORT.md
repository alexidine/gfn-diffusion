# prod_sep02: why the cluster cancels our runs for low GPU utilization, and what predicts it

*2026-09-07. Data: wandb project `mkilgour/GFN Energy`, `config.tag == p02`, 17 runs
(16 with history; the 17th is the 169 s lj_coeff resume failure). Both the training
history (10-step cadence) and wandb's system stream (`system.gpu.0.*`, `system.memory_percent`,
`system.network.*`, ~7.5 s cadence) were pulled in full. The `*_smi.csv` sidecars and
`joblogs/` were NOT available on this machine; see §7 for what they would add.*

Figures in this directory. `run_<arm>.png` is the per-run seven-panel read (step time and
its components, device util with trailing means, GPU memory/power, CPU, host memory, disk
and network, replay/batch). `A_trailing2h_util_by_family.png` is the kill boundary.
`B_util_vs_inverse_steptime.png` is the mechanism. `C_onset_alignment.png` is the onset
read on the eight runs with a sustained step-time rise. `regimes.csv` and `predictor.csv`
carry the numbers quoted below; the four `*.py` files regenerate everything.

---

## 0. Answer in five lines

1. **The kill boundary is a trailing ~2 h mean of device utilization at 54-55%**, not 60%. Every
   cancelled run ended with its 2 h mean at 48-53%; no survivor's 2 h mean ever went below 53.5%.
2. **The GPU work per step never changed.** Utilization times step time is constant within every
   run across every fast and slow regime (5% scatter). The added step time is pure GPU idle, it sits
   entirely in the non-MLIP part of the step, and the MLIP forward itself does not slow. So the cause
   is a host-side stall of a dispatch-bound loop, not more work, not throttling, not a GPU co-tenant.
3. **The stall is imposed by the node, not by the process.** Identical configs agree to 2-3% in their
   fast regimes across nodes and differ by 20-100% in their slow ones; slow regimes start and end
   abruptly, last 1-30 h, reverse, and one run carries an external ~11 h periodicity. Nothing in
   our process (eval, checkpoint, buffer growth, z-cal) lines up with any onset.
4. **The predictor is the trailing 1 h device-util mean crossing 56% (warning) and 54% (act),** or
   equivalently the 1 h median step time exceeding `B / 0.54` where `B` is the config's GPU-busy
   seconds per step. It led the five long-run kills by 1.6-16 h with zero false positives on nine
   survivors.
5. **mipu is killed and nehu is not because of margin, not mechanism.** mipcas at batch 1600 runs at
   63-67% device util in its fast regime; nehzor at 1600 runs at 75-85%. The same +30-50% host
   spell drops mipu under the line and leaves nehu at 60-70%.

---

## 1. Cases and controls

| arm | route | batch (actual) | wall | end state | 2 h util at end | min 2 h util | node |
|---|---|---:|---:|---|---:|---:|---|
| neh_lr0p5 | ELJ | 4000 | 24.3 h | **killed** | 48.7 | 48.4 | ga023 |
| mipu_lr0p0625 (7-day paper arm) | UMA | 1600 | 41.8 h | **killed** | 52.7 | 51.2 | ga016 |
| mipu_lr0p125 | UMA | 1600 | 30.4 h | **killed** | 49.6 | 47.4 | ga036 |
| mipu_lr0p03125 | UMA | 1600 | 21.1 h | **killed** | 48.3 | 48.1 | ga042 |
| mipu_lr0p25 | UMA | 1600 then **1000** (OOM cut at 1.5 h) | 15.5 h | **killed** | 53.0 | 53.0 | ga015 |
| mipu_lr0p015625 (= 16795765_0) | UMA | 1600 | 2.3 h | **killed** | 52.6 | 52.4 | ga037 |
| acr_lr0p025 | MACE | 1000 | 3.8 h | **killed** | 50.5 | 50.3 | ga024 |
| mip_lr1 | ELJ | 4000 | 116 h, running | survived | 68.4 | 54.5 | ga043 |
| nehu_lr0p125 | UMA | 1000 (early OOM cut) | 116 h, running | survived | 72.7 | 55.7 | ga026 |
| nehu_lr0p0625 | UMA | 1000 (early OOM cut) | 48 h wall | survived | 61.4 | 58.7 | ga012 |
| nehu_lr0p015625 / 0p03125 / 0p25 | UMA | 1600 | 48 h wall | survived | 74.9 | 67.4-72.1 | ga031/ga018/ga034 |
| acr_lr0p0125 / 0p05 / 0p1 | MACE | 1000 | 48 h wall | survived | 59.8-63.0 | 53.5-61.4 | ga021/ga033/ga030 |

wandb marks every SLURM-killed run `crashed` (SIGTERM), so state alone does not identify a
cancellation; wall time against the arm's `--time` does. The seven "killed" rows all ended well
short of their allocation with no traceback in the history. `sacct` end timestamps would confirm
each kill time to the minute; they were not available here.

## 2. The kill boundary (fig A)

Trailing means of `system.gpu.0.gpu` (the same NVML counter `nvidia-smi` reports), evaluated at
the end of each cancelled run and as the minimum over each survivor:

| window | max at kill, cancelled | min over run, survivors | separates? |
|---|---:|---:|---|
| 30 min | 54.5 | 47.1 | no |
| 60 min | 53.2 | 49.7 | no |
| 90 min | 52.7 | 52.9 | yes, by 0.2 |
| **120 min** | **53.0** | **53.5** | **yes, by 0.5** |
| 180 min | 57.5 | 54.9 | no |
| 240 min | 57.0 | 55.4 | no |

Continuous time below a level on the 2 h window: survivors spent **0.0 h below 54%** (up to 2.1 h
below 56%, mip_lr1). Cancelled runs had been below 54% for 0.1-4.7 h at the moment of the kill.
So the operative rule is consistent with "a roughly 2 h mean under ~54%, checked periodically,
acted on within a few hours". The site's own sampler will differ slightly from wandb's, so treat
54-55% as the line and 56% as the last safe reading. The owner's "roughly 60%" is not what the
data shows; the runs sat at 58-62% for tens of hours without being touched.

Two of the seven kills involved **no step-time rise at all**: mipu_lr0p25 after its OOM cut sat at
55% from hour 3 and was killed when the mean drifted to 53%; mipu_lr0p015625 on ga037 ran 13.4 s/step
from its first step (its siblings run 9.6-9.7 s at the same config) and was killed as soon as a 2 h
window existed. The other five are the "spell" pattern the owner described.

## 3. Mechanism: the GPU work is unchanged, the added time is idle (figs B, C, `regimes.csv`)

Segmenting every run into step-time regimes (rolling median, >10% shift held for 50 steps) and
tabulating per regime:

| run | regime | step s | energy s | non-energy s | device util | **util x t (GPU-busy s/step)** | SM clock |
|---|---|---:|---:|---:|---:|---:|---:|
| neh_lr0p5 | 1-19 h | 8.04 | 0.61 | 7.44 | 68.6 | **5.52** | 1410 |
|  | 19-21.6 h | 10.05 | 0.64 | 9.40 | 54.8 | **5.51** | 1410 |
|  | 21.6-23.3 h | 10.70 | 0.66 | 10.03 | 51.2 | **5.48** | 1410 |
|  | 23.3-24.2 h (kill) | 12.33 | 0.68 | 11.65 | 44.7 | **5.51** | 1410 |
| mipu_lr0p0625 | 1.8-32.6 h | 9.73 | 3.91 | 5.81 | 66.3 | **6.45** | 1410 |
|  | 32.6-41.6 h (kill) | 12.46 | 4.09 | 8.37 | 53.5 | **6.67** | 1397 |
| mipu_lr0p125 | 16-27.8 h | 9.69 | 3.89 | 5.80 | 66.3 | **6.43** | 1410 |
|  | 28.6-30.3 h (kill) | 14.59 | 4.42 | 10.17 | 46.0 | **6.72** | 1339 |
| mipu_lr0p03125 | 3.5-10.3 h | 12.28 | 4.12 | 8.17 | 53.4 | **6.56** | 1403 |
|  | 10.3-19 h (sped UP) | 9.67 | 3.92 | 5.74 | 66.6 | **6.43** | 1410 |
|  | 19-21 h (kill) | 14.97 | 4.46 | 10.51 | 49.0 | **7.33** | 1349 |
| nehu_lr0p125 (survivor) | fast regimes | 14.2 | 8.2 | 6.0 | 74 | **10.5-10.7** | 1410 |
|  | slow regimes | 17.4-21.0 | 8.4-8.9 | 9.0-12.1 | 58-64 | **10.8-12.3** | 1360-1370 |
| mip_lr1 (survivor) | fast/slow alternating | 7.99 / 9.77 | 0.53 / 0.57 | 7.46 / 9.20 | 69.8 / 57.3 | **5.58 / 5.60** | 1410 |

Three things follow, and each excludes a candidate:

- **util x t is constant (CV 5-9% per run).** Utilization falls exactly as 1/step-time (fig B: every
  run's 30-minute bins lie on a line through the origin). The GPU does the same work per step in a
  slow regime as in a fast one; the extra seconds are seconds the card sits idle waiting for the host.
  That excludes "more work per step" (bigger buffers, more z-cal rollouts, worse samples) and it
  excludes a GPU co-tenant, which would raise device util, not lower it.
- **The MLIP portion does not slow.** `energy/seconds_in_step` moves +2-13% while the non-energy
  portion moves +45-100%. A 13 s UMA forward is a few large kernels and does not care about host
  contention; the policy rollout (T=100, hundreds of small launches per step, dispatch-bound per
  `project_policy_rollout_is_dispatch_bound`) and the Python bookkeeping around it are exactly what a
  slowed host turns into GPU idle.
- **No throttling.** SM clock 1410 MHz in every fast regime, 1340-1400 in slow ones (the normal
  clock-down of a partly idle card), power falling with utilization, temperature 31-47 C. wandb's
  `system.gpu.0.smClock` and `powerWatts` say the same thing the sidecar's `clocks_throttle_reasons`
  said on the 2026-09-01 case.

## 4. The stall is the node's, not the process's

- **Identical configs agree in their fast regime and disagree in their slow ones.** mipu at batch
  1600: 9.61, 9.63, 9.65 s on ga016/ga036/ga042 versus 13.37 s on ga037 from the first step. acr at
  1000: 10.27, 10.28, 10.31 s on ga030/ga033/ga021 versus 19.2-28.5 s on ga024. nehu at 1600: 19.06,
  19.07, 19.10 s on three nodes; at 1000: 14.09, 14.15 s on two. mip/neh ELJ at 4000: 7.95, 8.01 s.
  A process-intrinsic speed reproducible to 2% across nodes is the baseline; everything above it is
  environment.
- **Slow regimes are reversible plateaus.** mipu_lr0p03125 sped UP from 12.3 to 9.7 s at hour 10 and
  stayed there 9 h. mip_lr1 alternated 8.0 / 9.0 / 9.8 / 8.1 / 9.3 / 8.1 s over five days.
  nehu_lr0p125 alternated 14.2 and 17.4 s, with 20-minute fast windows recurring every ~11 h (steps
  14670, 16930, 19240, 21590: spacing 2260-2350 steps, unrelated to our 1000-step eval grid). In-process
  growth (a prior buffer filling from 65k toward 250k rows, an anchor buffer of 43k-200k rows) cannot
  reverse in a step, and no run shows any step-time trend tracking buffer size.
- **Nothing periodic of ours is implicated.** Onsets fell at steps 22450, 16780, 14180 and 10880, none
  on the eval/figure grid (every 1000 steps; 500 on ELJ). Evals cost 14-30 s each, invisible on a 2 h
  mean. The 50-step checkpoint (`running.pt`, ~30 MB, hardlinked archives) leaves no 50-step ripple in
  the 10-step step-time trace. The buffer sidecar is written at eval cadence only. z-cal ran at
  `z_cal/seconds` 0 on every equilibration arm.
- **The node's population changes near onsets, but the sign is not consistent.** Host-wide
  `system.memory_percent` (all processes on the node; our own RSS is a flat 2.7-2.9 GB) moved at or
  within an hour of several onsets: neh 6 -> 9 -> 22%; mipu_lr0p03125 5 -> 13%; mipu_lr0p0625 3 -> 9%;
  nehu_lr0p125 5 -> 14%. But nehu_lr0p015625's fastest 21 h ran under 27-36% host memory and
  acr_lr0p1's fastest 9 h under 17%. So other jobs arriving and leaving is visible, but "a job arrived"
  is not itself the slowdown; which job, using which shared resource, is what the wandb stream cannot
  say (§7).
- **The init MLIP pass is not the culprit.** Device util during the first 15-30 minutes was 80-100% on
  every arm: the whole-prior scoring is GPU-bound. acr_lr0p025 died at 3.8 h because its steady state
  was 2x slower than its three siblings on their nodes, not because of anything at init.

**Ranked reading of the host-side cause, since the data cannot name the shared resource:**

1. **Node co-tenancy contending for a host resource the rollout depends on** (CPU time on the launching
   core, memory bandwidth/L3, PCIe): supported by reversibility, node dependence, host-memory shifts,
   the external 11 h periodicity, and the earlier sidecar case. Most likely.
2. **Scratch/filesystem stalls:** not supported. A stalled write would make individual steps spiky; the
   observed rise is a uniform shift of every 10-step mean, and disk/network rates show nothing at onsets.
3. **GPU clock or power throttling:** excluded (§3).
4. **A GPU co-tenant:** excluded (§3).
5. **In-process growth or a code path that got slower:** excluded (§4, reversibility).

## 5. The predictor (`predictor.csv`)

Because util x t is a per-config constant `B`, device utilization is predictable from step time
alone: `util = B / t`. The kill line in step-time units is `t_crit = B / 0.54`:

| config | B, GPU-busy s/step | fast-regime step | fast-regime util | **t_crit** | margin before the line |
|---|---:|---:|---:|---:|---|
| mipu UMA, batch 1600 | 6.45-6.55 | 9.6-9.7 s | 66% | **12.0 s** | +24% |
| mipu UMA, batch 1000 (after OOM cut) | 4.8 | 8.4 s | 55-57% | **8.8 s** | +5% |
| nehu UMA, batch 1600 | 16.2-16.6 | 19.1 s | 85% | **30 s** | +57% |
| nehu UMA, batch 1000 (after OOM cut) | 10.7-10.9 | 14.1-14.3 s | 75% | **20 s** | +40% |
| acr MACE, batch 1000 | 7.9-8.2 | 10.3 s | 76% | **14.9 s** | +45% (its slow regimes reach 14.6 s) |
| mip / neh ELJ, batch 4000 | 5.5-5.6 | 8.0 s | 69% | **10.3 s** | +29% |

Lead time from the first crossing to the kill, long cancelled runs:

| run | 1 h device util < 54 | 1 h util lead | 15-min step-time predictor lead |
|---|---:|---:|---:|
| mipu_lr0p03125 | 4.7 h | **16.4 h** | 17.3 h |
| mipu_lr0p0625 | 33.7 h | **8.1 h** | 8.9 h |
| mipu_lr0p125 | 28.8 h | **1.6 h** | (false-fired in startup wobble at 3.8 h) |
| mipu_lr0p25 | 6.2 h | **9.3 h** | 1.0 h (chronic case; step time barely moved) |
| neh_lr0p5 | 20.0 h | **4.3 h** | 3.8 h |

Survivors' longest continuous time below 54% on the 1 h window was 0.5 h, so **"1 h mean under 54%
for more than 30 minutes" is a zero-false-positive rule on nine survivors and fired 1.6-16 h ahead on
all five long kills.** 56% is the warning level (survivors did spend up to 1.3 h there). The step-time
form needs no sensor and explains the mechanism, but it fails the chronic case (mipu_lr0p25: the run
was at the line from hour 3 with no rise to detect) and it fires on startup unless gated past hour 3.
The device-util form catches both. Use the device-util reading as the alarm and the step-time ratio
as the diagnosis of why.

## 6. What `gpu/util_policy` should become

Per-run median offset of `gpu/util_policy` against wandb's 2 h mean over the same window:

| route / batch | offset |
|---|---:|
| ELJ mip / neh at 4000 | **+9 / +13** |
| UMA mipu at 1600 | **-11 to -12** (-25 on ga037) |
| UMA mipu at 1000 | -9 |
| UMA nehu at 1000-1600 | **-29 to -34** |
| MACE acr at 1000 | **-26 to -28** (acr_lr0p025: +3) |

It over-reads on ELJ and under-reads by up to 34 points on UMA, extending the documented
batch-dependent sign flip to a route-dependent one. It cannot be fixed by switching the source: it
already reads `torch.cuda.utilization()` (NVML), the same counter `nvidia-smi` and wandb report. The
defect is the **sampling**: one reading every 60 s, taken from the training loop at the step boundary
(train.py, the `_sample_gpu_util` call after `step_dt`), i.e. always at the same phase of a periodic
signal, and never during eval. NVML's utilization is "fraction of the last sample period a kernel
ran"; sampled at the boundary it sees the loop's host-side tail on UMA (reads low) and whatever the
ELJ loop's tail happens to be (reads high). A time-average the scheduler judges can only be
reproduced by a time-uniform sampler.

Recommendation: **retire `gpu/util_policy` and `gpu/util_recent` as occupancy estimates.** Replace
with a daemon-thread sampler (NVML if present, else `nvidia-smi --query-gpu=utilization.gpu`, every
5-10 s, independent of the step) publishing `gpu/util_1h`, `gpu/util_2h`, and
`gpu/kill_margin = util_2h - 54`. The wandb system stream already provides the ground truth for
post-hoc reads; the in-process copy is only needed if the run is to act on it (§7, item 4). If nothing
in-process is going to act on it, delete it and point readers at the sidecar and the system stream.

## 7. Mitigations, ranked, with the trade-off stated

1. **Make every arm resumable and chained, today.** mipu_lr0p0625 was a 7-day single-leg paper arm and
   died at 41.8 h with no leg 2 and, until the lj_coeff sidecar stamp is serialized
   (`project_lj_stamp_lost_on_buffer_restore`), no possibility of resume. A kill is a node event we do
   not control; a resume costs at most 50 steps and lands on a fresh node, which in every observed case
   restores the fast regime. Trade-off: queue wait per leg, and the pending-job sweep of 2026-09-02
   showed chains can be purged; single-leg arms should carry `--requeue` and the resume path so a kill
   is a restart rather than a loss. (sbatch + the buffer fix; no config change.)
2. **Buy margin on mipu with batch, and stop OOM cuts from parking a run below the line.** mipu at
   1600 has +24% margin; the observed spells are +25% to +55%. At batch 3200 (GPU memory allocated
   was 22% of 80 GB at 1600) `B` roughly doubles on the MLIP half and the rollout half, while the
   host-side portion barely grows; predicted fast-regime util is ~80% and a +100% host spell still
   clears 54%. Trade-off: about 30-40% fewer optimizer updates per hour at fixed
   `fused_grad_accum_min_samples`, twice the samples per update, and a batch the LR bracket was not
   measured at, so this is a recipe change for the paper models. Separately, mipu_lr0p25's OOM cut to
   1000 at 1.5 h was never restored by `batch_oom_ceiling_retest_steps` over 13 h and cost it the
   run; nehu_lr0p125 and nehu_lr0p0625 have run the whole battery at 1000 for the same reason. A cut
   that lowers `B` permanently is a self-inflicted margin loss and the retest should be made to fire.
3. **Remove the host dependence at the source (code, medium term).** The stall only becomes GPU idle
   because the rollout is dispatch-bound. CUDA graphs on the per-step policy forward (torch.compile
   `reduce-overhead`, rejected earlier for a real cudagraph-output-overwrite crash that needs
   `cudagraph_mark_step_begin` and clone discipline around the loop) collapse hundreds of launches into
   a few replays. That both raises the base utilization and makes the run indifferent to host
   contention. Trade-off: engineering time and a compile-mode fragility that was already hit once.
4. **Act in-process before the scheduler does.** With the async sampler of §6, when `gpu/util_1h` sits
   under 54% for 30 min (or `t_1h > 0.9 t_crit`), checkpoint and `scontrol requeue $SLURM_JOB_ID`, with
   `--requeue` in the sbatch. This converts an involuntary kill into a controlled migration to another
   node. Trade-off: you re-enter the queue (which under the current GPU quota can mean hours), and the
   rule must be gated past hour 3 to avoid firing on startup and on the init scoring pass.
5. **Request a node shape less exposed to neighbours.** `--cpus-per-task` above 8, more `--mem`, or
   `--exclusive`. Whether this helps depends on which shared resource the neighbour is contending
   for, and the wandb stream cannot tell CPU from memory bandwidth from PCIe. Do not spend the quota on
   this until the joblogs answer that question. Trade-off: `--exclusive` idles three GPUs per node and
   would not pass the same policy it is trying to satisfy.

Not recommended: a GPU busy-loop to hold utilization up during a stall. It would pass the check while
defeating the policy's purpose, and the node's other tenants would pay for it.

**What the cluster-side records would settle** (the owner has them; this machine does not):
`joblogs/<arm>_<jobid>.info` names the node; `sacct -N <node> -S <onset-1h> -E <onset+1h> -a
--format=JobID,User,Start,End,NCPUS,ReqMem,State` lists what else started or ended on that node at
each onset in `regimes.csv`. If neighbour arrivals line up with onsets, item 5 gets its answer
(what they requested tells you which resource). The `*_smi.csv` sidecars on the 2026-09-03 arms
confirm `clocks_throttle_reasons.active == 0` and put the site's own counter beside wandb's. And
`sacct -j 16795759,16795765,16819123` gives the kill times to the minute for §2.

## 8. Reproduce

    python pull_p02.py        # wandb API -> data/<run_id>.pkl (history + system stream)
    python plot_runs.py       # figs/run_<arm>.png
    python regimes.py         # regimes.csv (step-time regime table)
    python predictor.py       # predictor.csv, figs/A_*, figs/B_*
    python onset_composite.py # figs/C_onset_alignment.png


## 9. Addendum 2026-09-07: the neighbour, observed

The owner ran `sacct -N ga036` for the window around mipu_lr0p125's onset. Job `H2O_prod`
(user em5339, partition `all`, **48 CPUs, 150 GB, no GPU**) started at 2026-09-03
22:30:40 ET. Our first slow 10-step block ran 22:30:03 to 22:32:12 ET. The neighbour ran
to 09:30 the next morning; we were killed at 01:00. Candidate 1 in §4 is therefore
observed, not inferred: the `all` partition schedules large CPU-only jobs onto GPU nodes,
and a 48-core tenant slows our 8-core host share. Since cgroups already pin our cores,
the contended resource is memory bandwidth, cache or SMT siblings, not CPU time, so a
larger `--cpus-per-task` would not help; only isolation from CPU-only tenants or a less
host-bound loop would. The second jump 47 minutes later matches no job start and is most
likely the neighbour's own phase change.

Also corrected: the three arms that ran at batch 1000 were NOT OOM cuts. All three show
zero OOM events and `batch/sizer_reason` 7 (`wallclock_cut`): the `max_step_seconds`
runaway guard fired on the stage-transition transient in the first 15-80 minutes and the
cut is permanent by construction (restore runs only when `batch_sizer` is None; the
ceiling-expiry re-derivation excludes `wallclock_cut`). Item 5 of §7 should read
"make a wallclock cut expire", not "make the OOM retest fire".

`configs/prof_sep07/` carries the profiling arms (mipu and mip, production shape, trace
window in steady state) that attribute the 3 s host chunk of §3.
