"""
Experiments -- the part of the bench that produces ANSWERS rather than green ticks.

Each function prints a table and returns its rows. Run one:

    python -m bench.experiments probe_blindness

or all of them:

    python -m bench.experiments

WHAT THESE CAN AND CANNOT SETTLE. They are statements about CONTROL LOGIC on a
surface whose ground truth is known. Where a result is a derivation checked
against a measurement it generalises (the required-alpha_target formula is of
that kind). Where it is a number measured on one synthetic surface -- a specific
ramp rate, a specific jitter tolerance -- it is a fact about that surface, and
its value is that it bounds what the controller CAN do, not what it will do on a
crystal. No parameter measured here should be copied into a production config.
"""

import math
import sys

import numpy as np

from bench.old.clock import SyntheticGPU
from bench.old.harness import BenchRun
from bench.surfaces import EquilibrationGame


def _hr(title):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


# ---------------------------------------------------------------------------

def probe_blindness():
    """
    WHY A RAY PROBE RAMPS TO EXPLOSION ON AN ILL-CONDITIONED SURFACE.

    The probe rates the step it just took, so it measures curvature along d and
    nothing else: lambda_eff = d'Hd / d'd. SGD stability, meanwhile, is set by
    lambda_max over ALL directions.

    On an anisotropic surface the stiff modes converge first, so d migrates into
    the soft subspace and lambda_eff falls monotonically while lambda_max does
    not move. The probe therefore reports "far too cold" -- pinned at the top of
    the alpha grid -- for the entire climb, including the part of it that is past
    2/lambda_max. The sensor is not noisy here; it is answering a question about
    a subspace that no longer contains the constraint.
    """
    _hr('probe_blindness: curvature the probe rides vs curvature that limits LR')
    run = BenchRun(game='mle', need_batch_sizer=False,
                   game_kwargs=dict(dim=32, cond=300.0, noise=0.01, lr=5e-5,
                                    init_scale=3.0, seed=3),
                   args_overrides={'adaptive_lr.warmup_steps': 50,
                                   'adaptive_lr.ray_calibration.period': 50,
                                   'lr_policy': 5e-5, 'min_lr': 1e-10},
                   probe_batch=2048)
    g = run.game
    lam_max = float(g.H.max())
    print(f'lambda_max = {lam_max:.0f};  SGD stability limit lr = 2/lambda_max = '
          f'{2 / lam_max:.3g}\n')
    print(f'{"step":>6} {"lr":>10} {"lr*lam_max":>11} {"lam_eff(d)":>11} '
          f'{"alpha*":>8} {"status":>12}')
    rows = []
    for i in range(700):
        before = g.policy_params[0].detach().clone()
        run.step()
        d = g.policy_params[0].detach() - before
        if i % 50 or float(d.norm()) == 0:
            continue
        h = g.hessian_diag(before)
        lam_eff = float((d * h * d).sum() / (d * d).sum())
        cal = run.calibrations[-1] if run.calibrations else None
        lr = run.history[-1]['lr']
        alpha = '{:.1f}'.format(cal['alpha_star']) if cal else '-'
        status = cal['status'] if cal else '-'
        rows.append(dict(step=i, lr=lr, ratio=lr * lam_max, lam_eff=lam_eff,
                         status=status))
        print(f'{i:>6} {lr:>10.3g} {lr * lam_max:>11.3f} {lam_eff:>11.2f} '
              f'{alpha:>8} {status:>12}')
    crossed = [r for r in rows if r['ratio'] > 2.0]
    print(f'\ncrossed the stability limit at step {crossed[0]["step"] if crossed else "never"}; '
          f'divergences: {run.divergences}')
    print('the sensor read `above_range` throughout -- it never saw the constraint')
    return rows


# ---------------------------------------------------------------------------

def saturation_policy():
    """
    A/B: MAY A SATURATED READING RAISE THE RATE?

    When the truth is above the grid every reading returns `above_range` pinned
    at 32, and the controller applies a constant (32/4)^0.25 = 1.68x per period
    carrying no information about how far above the grid the truth is. That is
    open loop, and on the surface above it runs to the divergence bar.

    Arm B suppresses the raise on a saturated reading: `above_range` may lower
    (it never does -- it is a lower bound) but may not raise. The rate then only
    moves on readings that actually landed inside the grid.

    Nothing in controller.py is edited; the arm is a reading_filter, so arm A is
    the shipping code exactly.
    """
    _hr('saturation_policy: may an out-of-grid reading raise the rate?')
    WIDE = tuple(float(2 ** k) for k in range(12))       # 1..2048, tests alpha* to 1024

    class BoundedOpenLoop:
        """
        CANDIDATE FIX. A saturated reading carries no magnitude, so the raise it
        licenses is open loop -- but forbidding it entirely strands a cold run at
        its seed (arm B). Bound the CUMULATIVE open-loop excursion instead: allow
        saturated raises until they have moved the rate `cap` times since the last
        reading that actually landed inside the grid, then hold until one does.

        A bracketed or below_range reading is real feedback and resets the budget.
        """

        def __init__(self, cap=8.0, per_raise=(32.0 / 4.0) ** 0.25):
            self.cap, self.per_raise, self.spent = cap, per_raise, 1.0

        def __call__(self, reading):
            if reading['status'] != 'above_range':
                self.spent = 1.0             # real feedback: budget resets
                return reading
            if self.spent * self.per_raise > self.cap:
                return None                  # budget exhausted -- hold
            self.spent *= self.per_raise
            return reading

    arms = {
        'A shipping grid, sat. raises': (None, None),
        'B saturated may not raise': (lambda r: None if r['status'] == 'above_range' else r, None),
        'C wider alpha grid (to 2048)': (None, (0.0,) + WIDE),
        'D bounded open loop (cap 8x)': (BoundedOpenLoop(8.0), None),
        'E bounded open loop (cap 64x)': (BoundedOpenLoop(64.0), None),
    }
    print(f'{"arm":>32} {"final lr":>10} {"peak lr":>10} {"lr*lam_max":>11} '
          f'{"diverge":>8} {"final dist":>11}')
    rows = []
    for name, (filt, alphas) in arms.items():
        overrides = {'adaptive_lr.warmup_steps': 50, 'adaptive_lr.ray_calibration.period': 50,
                     'lr_policy': 5e-5, 'min_lr': 1e-10}
        if alphas is not None:
            overrides['adaptive_lr.ray_calibration.alphas'] = alphas
        run = BenchRun(game='mle', need_batch_sizer=False, reading_filter=filt,
                       game_kwargs=dict(dim=32, cond=300.0, noise=0.01, lr=5e-5,
                                        init_scale=3.0, seed=3),
                       args_overrides=overrides,
                       probe_batch=2048).run(3000, stop_on_divergence=False)
        lam_max = float(run.game.H.max())
        peak_lr = max(h['lr'] for h in run.history)
        s = run.summary()
        rows.append(dict(arm=name, final_lr=s['final_lr'], peak_lr=peak_lr,
                         ratio=peak_lr * lam_max, div=run.divergences,
                         dist=s['final_dist']))
        print(f'{name:>32} {s["final_lr"]:>10.3g} {peak_lr:>10.3g} '
              f'{peak_lr * lam_max:>11.2f} {run.divergences:>8} {s["final_dist"]:>11.4g}')
    print('\nlr*lambda_max > 2 is the SGD stability limit; `diverge` counts bar trips.')
    return rows


# ---------------------------------------------------------------------------

def alpha_target_sweep():
    """
    WHAT alpha_target HAS TO BE, ON A SURFACE WHERE THE ANSWER IS DERIVABLE.

    On the equilibration game the LR the multi-player loop survives is
    2c/(c^2 + a*b*w_rep), while a frozen-target ray probe reads alpha* = 1 at
    lr = 1/c. The smallest safe target is their ratio, (1 + loop_gain)/2 at
    c = 1, and a servo asked to hold a SMALLER alpha* is being asked to sit
    outside the stability boundary.
    """
    _hr('alpha_target_sweep: servo outcome vs the derived minimum')
    # The Z head is PINNED at lr_flow and exempt from the servo, so what matters
    # is the RATIO of level rate to policy rate. mk_dev runs lr_flow 0.1 against a
    # policy rate near 1.25e-4 -- the level moves ~800x faster. Measured on this
    # game, the alpha_target constraint binds only in that fast-level regime:
    #
    #     lr_flow 0.01..0.30 -> stability 2.01..2.36, min alpha_target 0.42..0.50
    #     lr_flow 0.50       -> stability 0.565,      min alpha_target 1.77
    #     lr_flow 1.00       -> stability 0.359,      min alpha_target 2.79
    #
    # A SLOW level is not the dangerous case: under the anti-phase coupling a
    # level that responds quickly amplifies the oscillation, while a sluggish one
    # averages it out. lr_flow 1.0 puts this in the regime the real config is in.
    LR_FLOW = 1.0
    # noise high enough that the run sits at a noise floor instead of reaching the
    # optimum -- at dist ~1e-4 the gradient is pure noise, alpha* becomes a noise
    # ratio, and the servo ratchets on it (see findings: converged runs)
    game_kw = dict(dim=4, a=4.0, b=1.0, w_rep=0.7, w_bwd=0.3, kappa=0.05,
                   noise=0.3, init_scale=1.0)
    ref = EquilibrationGame(**game_kw)
    # stability of the POLICY rate with the level rate held at lr_flow, which is
    # the real configuration -- not the symmetric case the closed form covers
    stability = ref.stability_lr(lr_level=LR_FLOW)
    required = ref.one_step_lr() / stability
    print(f'loop gain a*b*w_rep = {4.0 * 1.0 * 0.7:.2f}   lr_flow(pinned) = {LR_FLOW}')
    print(f'stability_lr(policy) = {stability:.4f}   one_step_lr = {ref.one_step_lr():.4f}')
    print(f'derived minimum alpha_target = {required:.2f}\n')
    # REPORT THE MAX EXCURSION, NOT THE ENDPOINT. Once the run reaches its noise
    # floor the gradient is pure noise, alpha* becomes a noise ratio and the servo
    # random-walks -- so the FINAL lr of a converged run says nothing about the
    # target that produced it (measured: targets 1.0 and 8.0 both ended within a
    # factor of 2 of each other). What the target controls is how far the rate
    # climbs while there is still signal, which is what can cross the boundary.
    print(f'{"alpha_target":>13} {"max lr":>10} {"max lr/stab":>12} '
          f'{"final dist":>12} {"diverged":>9}')
    rows = []
    for target in [1.0, 1.5, 2.0, 3.0, 4.0, 8.0]:
        run = BenchRun(game='equilibration', need_batch_sizer=False,
                       game_kwargs=dict(**game_kw, lr=0.05, seed=0),
                       args_overrides={'adaptive_lr.warmup_steps': 50,
                                       'adaptive_lr.ray_calibration.period': 50,
                                       'adaptive_lr.calibration.alpha_target': target,
                                       'lr_fused': 0.05, 'lr_flow': LR_FLOW,
                                       'min_lr': 1e-9},
                       probe_batch=2048).run(4000, stop_on_divergence=False)
        s = run.summary()
        lr = max(h['lr'] for h in run.history)
        rows.append(dict(target=target, lr=lr, ratio=lr / stability,
                         dist=s['final_dist'], diverged=s['diverged']))
        print(f'{target:>13.1f} {lr:>10.4f} {lr / stability:>12.2f} '
              f'{s["final_dist"]:>12.4g} {str(s["diverged"]):>9}')
    print('\nmax lr/stab > 1 means the servo took the run outside the boundary.')
    return rows


# ---------------------------------------------------------------------------

def loop_gain_vs_replay_weight():
    """
    THE PREDICTION WORTH TAKING BACK TO THE REAL SYSTEM.

    loop_gain = a*b*w_rep, and w_rep is the REPLAY BRANCH WEIGHT. So the minimum
    safe alpha_target rises with the replay weight -- which the balance
    controller moves during a run, while alpha_target is a fixed config value.

    A stage that shifts weight onto replay tightens the LR ceiling without any
    sensor noticing.
    """
    _hr('loop_gain_vs_replay_weight: the LR ceiling moves when the branch mix does')
    print(f'{"w_rep":>7} {"loop gain":>10} {"stability_lr":>13} '
          f'{"min alpha_target":>17} {"vs w_rep=0.1":>13}')
    rows, base = [], None
    for w_rep in [0.1, 0.3, 0.5, 0.7, 0.9]:
        g = EquilibrationGame(a=4.0, b=1.0, w_rep=w_rep, w_bwd=1.0 - w_rep, kappa=0.05)
        s = g.stability_lr()
        required = g.one_step_lr() / s
        base = base or required
        rows.append(dict(w_rep=w_rep, gain=4.0 * w_rep, stability=s, required=required))
        print(f'{w_rep:>7.1f} {4.0 * w_rep:>10.2f} {s:>13.4f} {required:>17.2f} '
              f'{required / base:>13.2f}x')
    print('\nA fixed alpha_target that is safe at one branch mix is not safe at another.')
    return rows


# ---------------------------------------------------------------------------

def sensor_mismatch():
    """
    WHAT THE PROBE'S OBJECTIVE MISMATCH COSTS.

    The real probe draws from replay and scores with replay_loss_coeffs, while
    the step it is rating trained the full fused mixture. Arm A reproduces that;
    arm B scores the same objective that trained.
    """
    _hr('sensor_mismatch: probe scores replay only vs the objective that trained')
    LR_FLOW = 0.3
    print(f'{"probe scores":>16} {"final lr":>10} {"lr/stability":>13} '
          f'{"final dist":>12} {"resolved":>9}')
    rows = []
    for scores in ['replay', 'total']:
        kw = dict(dim=4, a=4.0, b=1.0, w_rep=0.7, w_bwd=0.3, kappa=0.05,
                  noise=0.02, init_scale=1.0, probe_scores=scores)
        ref = EquilibrationGame(**kw)
        stability = ref.stability_lr(lr_level=LR_FLOW)
        run = BenchRun(game='equilibration', need_batch_sizer=False,
                       game_kwargs=dict(**kw, lr=0.05, seed=0),
                       args_overrides={'adaptive_lr.warmup_steps': 50,
                                       'adaptive_lr.ray_calibration.period': 50,
                                       'lr_fused': 0.05, 'lr_flow': LR_FLOW,
                                       'min_lr': 1e-9},
                       probe_batch=2048).run(4000, stop_on_divergence=False)
        s = run.summary()
        rows.append(dict(scores=scores, lr=s['final_lr'],
                         ratio=s['final_lr'] / stability,
                         dist=s['final_dist'], resolved=s['n_resolved']))
        print(f'{scores:>16} {s["final_lr"]:>10.4f} '
              f'{s["final_lr"] / stability:>13.2f} '
              f'{s["final_dist"]:>12.4g} {s["n_resolved"]:>9}')
    return rows


# ---------------------------------------------------------------------------

def knee_jitter():
    """
    HOW NOISY MAY STEP TIMES BE BEFORE THE KNEE TEST MISFIRES?

    The ladder is coarse (1.65x) next to the tolerance (1.25x), so at the
    decisive rung the true step-time ratio is 1.2766 against a 1.25 threshold --
    a 2.1% margin, decided from 20 timings.
    """
    from bench.old.test_batch_sizer import batch_run
    _hr('knee_jitter: pin distribution vs step-time noise')
    gpu = SyntheticGPU(t_fixed=2.0, sps_max=5000.0)
    print(f'analytic knee bound {gpu.knee_bound(1.65, 0.25):.0f}; '
          f'noise-free pin {gpu.expected_pin(1000, 1.65, 0.25)}')
    print(f'decisive rung 7410->12226 true ratio '
          f'{gpu.true_step_time(12226) / gpu.true_step_time(7410):.4f} vs threshold 1.25\n')
    rows = []
    for j in [0.0, 0.05, 0.10, 0.20, 0.40]:
        pins = [batch_run(steps=3000,
                          gpu=dict(t_fixed=2.0, sps_max=5000.0, jitter=j, seed=100 + s),
                          seed=s).m.batch_size for s in range(10)]
        counts = {k: pins.count(k) for k in sorted(set(pins))}
        rows.append(dict(jitter=j, pins=counts))
        print(f'jitter {j:4.2f}  pins: {counts}')
    return rows


def knee_realism():
    """
    DOES THE KNEE TEST SURVIVE A COST CURVE WITH STEPS IN IT?

    Real step time is not smooth in batch. Three effects, separately and
    together: wave quantisation (`tile`), torch.compile's per-shape recompile
    (`recompile_s`), and cuBLAS/cuDNN kernel switches (`regimes`).

    Two different questions, and they have different answers:

      * does the controller TRACK the cost model it is given? Yes -- it lands on
        expected_pin (a walk against the actual model) under mild discreteness.
      * is the CRITERION right? No, not on a non-monotone curve. The gate is a
        local two-point comparison, so a one-off step between two rungs is
        indistinguishable from saturation, and it pins there permanently.
    """
    from bench.old.test_batch_sizer import batch_run
    _hr('knee_realism: the gate under discrete cost models')
    BASE = dict(t_fixed=2.0, sps_max=5000.0)
    cases = {
        'smooth (reference)': dict(BASE),
        'wave quantisation, tile 256': dict(BASE, tile=256),
        '+ recompile stall 40 s': dict(BASE, tile=256, recompile_s=40.0),
        '+ kernel switch @4096 x0.8': dict(BASE, tile=256, regimes=[(4096, 0.8)]),
        '+ jitter 0.10': dict(BASE, tile=256, recompile_s=40.0,
                              regimes=[(4096, 0.8)], jitter=0.1),
        'HARD kernel switch @2722 x0.5': dict(BASE, regimes=[(2722, 0.5)]),
    }
    print(f'{"cost model":>32} {"predicted":>10}  pins over 6 seeds')
    rows = []
    for name, kw in cases.items():
        predicted = SyntheticGPU(**kw).expected_pin(1000, 1.65, 0.25)
        pins = [batch_run(steps=4000, gpu=dict(kw, seed=100 + s), seed=s).m.batch_size
                for s in range(6)]
        counts = {k: pins.count(k) for k in sorted(set(pins))}
        rows.append(dict(model=name, predicted=predicted, pins=counts))
        print(f'{name:>32} {str(predicted):>10}  {counts}')

    print('\nPATH DEPENDENCE on the non-monotone curve (kernel switch @2722 x0.5).')
    print('Same cost model, same config -- only the starting batch differs:')
    kw = dict(BASE, regimes=[(2722, 0.5)])
    gpu = SyntheticGPU(**kw)
    best = max([1650, 2722, 4491, 7410, 12226], key=gpu.throughput)
    print(f'{"start":>10} {"pin":>10} {"sps":>10} {"% of best":>11}')
    for start in (1000, 1650, 2722, 4491, 7410):
        run = BenchRun(
            game='mle', game_kwargs=dict(dim=4, cond=2.0, noise=0.0, lr=1e-3),
            gpu_kwargs=dict(kw, seed=0),
            args_overrides={'grow_batch_size': True,
                            'max_batch_size': 200000, 'max_step_seconds': 0,
                            'batch_size': start},
            probe_enabled=False,
        ).run(6000, stop_on_divergence=False)
        p = run.m.batch_size
        print(f'{start:>10} {p:>10} {gpu.throughput(p):>10.0f} '
              f'{100 * gpu.throughput(p) / gpu.throughput(best):>10.0f}%')
    print('\nOn a non-monotone curve `batch_size` does not merely floor the walk '
          '-- it selects the answer.')
    return rows


EXPERIMENTS = {f.__name__: f for f in [
    probe_blindness, saturation_policy, alpha_target_sweep,
    loop_gain_vs_replay_weight, sensor_mismatch, knee_jitter, knee_realism]}


if __name__ == '__main__':
    names = sys.argv[1:] or list(EXPERIMENTS)
    for n in names:
        if n not in EXPERIMENTS:
            raise SystemExit(f'unknown experiment {n!r}; have {list(EXPERIMENTS)}')
        EXPERIMENTS[n]()
