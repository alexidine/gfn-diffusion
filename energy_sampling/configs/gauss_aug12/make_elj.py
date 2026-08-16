"""
The physical second-order check for D33: elj on nehzor, SG 14, Z'=1, T=10.

WHAT IT CAN AND CANNOT SHOW. Physical energies have never converged perfectly on this
codebase, so this arm cannot CERTIFY a log Z the way the analytic battery does -- there
is no closed form to land on. It is a smoke test: does a real energy, at a real space
group with two genuinely dead rows (alpha, gamma), train sanely with those rows held out
of the SDE. A clean result proves less than it looks like it does; a messy one does not
implicate D33 without further work.

WHY sg 14 IS THE RIGHT CHOICE. Monoclinic, so `enforce_crystal_system` clobbers rows 3
and 5 -- exactly the case F-009 found the defect in, and the case with a physical prior on
disk. No free aunit axes, so this exercises the angle half of D33 only.

TWO THINGS TO KNOW ABOUT THIS PRIOR
  1. thermal_scaling_factor 0.1556 rides in the .pt and REPLACES energy_config.lj_coeff
     for the whole run (train.py's init_prior_dataset is loud about it). So the effective
     sampling temperature is temperature / 0.1556: at 2.5 that reads as ~16.1 in raw elj
     units. Not comparable to the mipcas runs, whose factor is 0.3636 (2.5 -> ~6.9).
  2. Use the FULL 207k-row prior, NOT deadrow10k_sg14_zp1_elj.pt. The 10k subsample was
     written without `thermal_scaling_factor`, so a run on it silently falls back to the
     config's lj_coeff and trains at a different energy scale.

NO EARLY STOP. epochs is set very large deliberately; this run is cancelled by hand when
it looks satisfactory. The stage `exit` blocks are stage TRANSITIONS, not run termination.

LR SERVO LEFT ON, unlike the analytic battery. There the probe saturated (F-025) because
a converged toy's loss reaches ~0.003 and cannot be bracketed; a physical energy will not
do that, and the servo is what production runs. Read `lr_ctrl/peak_scale` and
`raycal/status` anyway -- status 3 pinned means saturated, not "too hot".

    python configs/gauss_aug12/make_elj.py
"""
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIGS = HERE.parent
MK_DEV = CONFIGS / 'mk_dev.yaml'
def out_path(cap, resume=False):
    tag = '_cap' if cap else ('_r2' if resume else '')
    return CONFIGS / f'elj_nehzor_sg14_t10{tag}.yaml'

PRIOR = r'D:\crystal_datasets\conditional\priors\nehzor_sg14_zp1_elj_prior_dataset.pt'
BUDGET = 10000          # not a convergence estimate -- cancel by hand
TRAJ_T = 10

# --resume: pick run 1 back up mid-MLE instead of starting over.
#
# Run 1 hit its 10000-step ceiling still in stage 1, with bwd/tbc (0.865 vs 2.0) and
# wass_debiased (0.0118 vs 0.015) both satisfied and only gates/mle_flat blocking
# (mle_gate_rate_hi 0.145 vs a 0.05 bar). The MLE gate is KEPT here rather than dropped,
# because run 1's own deceleration says it should clear on its own with more room:
# d(bwd/mle) per 2000 steps ran -7.2, -2.9, -1.8, -1.4, -0.9, decaying ~0.7x per 2000,
# so rate_hi should reach 0.05 near step ~16000. Giving it that room is the point of the
# resume; pre-empting the gate would answer a different question.
#
# load_full restores modeller_state, so the stage AND the MLE gate's 300-step window
# carry over -- this continues stage 1, it does not restart it.
RESUME_CKPT = ('d33elj_elj_nehzor_sg14_t10_elj-nehzor_sg14_zp1_elj_prior_dataset'
               '-T2.5-990198_final.pt')
RESUME_BUDGET = 60000   # a CEILING with headroom past the expected ~16000 transition.
                        # Must exceed the resume step: a resumed loop is
                        # trange(init_step, epochs+1), so a smaller epochs silently runs
                        # ZERO steps and verifies nothing.


def main():
    with MK_DEV.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cap = '--cap-stage1' in sys.argv
    resume = '--resume' in sys.argv
    if cap and resume:
        raise SystemExit('--cap-stage1 and --resume answer different questions; pick one')
    suffix = '_cap' if cap else ('_r2' if resume else '')
    cfg['run_name'] = 'elj_nehzor_sg14_t10' + suffix
    cfg['tag'] = 'd33elj'
    cfg['epochs'] = RESUME_BUDGET if resume else BUDGET

    # The resume point is PINNED by name. continue_from_checkpoint stays False either
    # way: the mk_dev defaults resolve null + True to '{tag}_{run_name}_..._running.pt',
    # which for a fresh run_name finds nothing and silently starts over -- a different
    # wrong answer, and invisible. checkpoint_name is the only load path.
    # A NEW run_name on the resume keeps run 1's _final/_best artifacts from being
    # clobbered by this run's own save('final'); cross-run loads are legitimate and
    # hash-guarded by assert_problem_match.
    cfg['checkpoint_name'] = RESUME_CKPT if resume else None
    cfg['prior_model_name'] = None
    cfg['continue_from_checkpoint'] = False
    cfg['load_weights_only'] = False    # need optimizers, buffers, step AND stage_ctrl

    # the problem
    cfg['energy_function'] = 'elj'
    cfg['space_groups'] = [14]
    cfg['z_primes'] = [1]
    cfg['prior_path'] = PRIOR
    cfg['molecules_path'] = PRIOR
    cfg['test_molecules_path'] = None
    cfg['integrator']['T'] = TRAJ_T
    cfg['eval_T'] = TRAJ_T          # must track train T; a mismatch dominates the metrics

    # D33 under test, stated rather than inherited
    cfg['model']['hold_dead_latent_rows'] = True

    if cap:
        # MEASURED, not guessed: the uncapped run spent all 10000 steps in stage 1 with
        # bwd/tbc (0.87 vs 2.0) and wass_debiased (0.012 vs 0.015) BOTH satisfied from
        # ~step 8000, blocked solely by gates/mle_flat. mle_gate_rate_hi bottomed at
        # 0.089 and rose back to 0.145 -- it oscillates 0.09-0.19 and never approaches
        # the 0.05 bar, because bwd/mle is still descending linearly at step 10000.
        # MLE on 207k real structures does not flatten on this budget, so the term
        # cannot gate anything except forever. Drop it and let tbc + wass carry the
        # exit; the mle_gate FLAG stays on so gates/mle_flat is still published and we
        # can keep watching it fail to flatten.
        for stage in cfg['protocol']['stages']:
            if not stage.get('exit'):
                continue
            kept = [t for t in stage['exit'] if t.get('metric') != 'gates/mle_flat']
            if not kept:
                raise ValueError(f"stage {stage.get('name')}: dropping gates/mle_flat "
                                 f"would leave no exit condition, making it terminal")
            stage['exit'] = kept

    if not os.path.exists(PRIOR):
        raise FileNotFoundError(PRIOR)

    OUT = out_path(cap, resume)
    with OUT.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    print(f'wrote {OUT.relative_to(CONFIGS.parent)}')
    print(f'  elj / nehzor / SG 14 / Z\'=1 / T={TRAJ_T} / epochs {BUDGET} (cancel by hand)')
    print(f'  batch_size {cfg["batch_size"]}  temperature {cfg["energy_config"]["temperature"]}'
          f'  hold_dead_latent_rows {cfg["model"]["hold_dead_latent_rows"]}')
    if resume:
        print(f'  RESUMING from {RESUME_CKPT}')
        print(f'  epochs {RESUME_BUDGET} (must exceed the resume step 10000, or the '
              f'loop runs ZERO steps)')
        print("  MLE gate KEPT: expect stage 2 near step ~16000 on run 1's deceleration")
    print(f'  expect: dead rows (3, 5) held -> flowing 10 of 12 dims')
    print(f'  expect: "Re-analyzing prior energies" then thermal_scaling_factor 0.1556 '
          f'replacing lj_coeff')
    return 0


if __name__ == '__main__':
    sys.exit(main())
