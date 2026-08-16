"""
umaperf0812 -- LOCAL uma A/B: does the MLIP work actually raise GPU occupancy?

    python configs/umaperf0812/make.py

WHAT THIS MEASURES, and why it needs a real training run rather than a benchmark.
The AtomicData vectorisation and the hosted gas reference are proven CORRECT (15 CPU
+ 14 GPU tests) and measured at 1.38x on the energy call in isolation. What is
completely unmeasured is the thing that actually matters: what they do to
`gpu/util_recent` and `energy/frac_of_step` inside the training loop, where the
rollout, the buffers and z_calibration all compete for the same GPU.

Two arms differing in NOTHING but the two switches:

    a_baseline   MXT_VECTORISED_ATOMICDATA=0 + host_gas_phase_reference: false
    b_optimised  MXT_VECTORISED_ATOMICDATA=1 + host_gas_phase_reference: true

Same seed, same config, same source. The DELTA is the measurement; the absolute
numbers do not transfer to the A100 anyway (different card, different host CPU,
16 GB vs 80 GB).

=============================================================================
WHY PHASE 1 IS DELIBERATELY TRIVIAL
=============================================================================
`train_prior` is bwd/dataset MLE -- it NEVER CALLS THE ENERGY. A from-scratch run
would spend most of its budget in the one stage that cannot exercise the MLIP. So
phase 1 here carries a single exit term that passes immediately (`bwd/tbc` below
1e9) and exits at the first eval, ~EVAL_PERIOD steps in.

MLE convergence quality is NOT load-bearing for this measurement (user's call,
2026-08-12): both arms get the same undertrained policy, and what is being compared
is the cost of scoring its samples, not their quality. The transition itself is kept
rather than skipped, because that is where the batch controller and z_calibration
do their work and where prod0810's uma arm died.

=============================================================================
SAFETY FOR AN UNATTENDED OVERNIGHT RUN
=============================================================================
  * checkpoint_read_only: true -- NO checkpoint writes at all. train.py's
    save('final') fires outside the training loop and save('running') every 50
    steps, so any run reusing a run_name overwrites that run's checkpoints. These
    arms write nothing, so they cannot clobber anything of the user's.
  * cuda_memory_fraction 0.7, not 0.9. This is a 16 GB laptop card also driving the
    desktop; 0.9 is the setting that put two runs into a collision earlier today.
  * max_batch_size is bounded well below what the card could take, so the batch
    controller has room to move without approaching the OOM edge unattended.
  * gpu_guard runs from train.py's __main__ and will refuse to start if anything
    else holds the card.

NB train.py IGNORES wandb_mode -- these appear as real runs in "GFN Energy" under
tag `umaperf0812`.
"""
import argparse
import os
from copy import deepcopy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIGS = HERE.parent
BASE = CONFIGS / 'mk_dev.yaml'

TAG = 'umaperf0812'
PRIOR = r'D:\crystal_datasets\conditional\priors\mipcas_sg2_zp1_uma_prior_dataset.pt'
MLIP = r'D:\crystal_datasets\esen_s.pt'

# Budgets. Phase 1 exits at the first eval, so almost all of this is phase 2 --
# which is the only part that calls the MLIP.
EVAL_PERIOD = 50
EPOCHS = 400
FIGURE_PERIOD = 100000        # effectively off: figures cost GPU and answer nothing here

BATCH = 100                   # 16 GB card, UMA. Small on purpose.
MAX_BATCH = 800               # room for the controller to move, far from the OOM edge

# EVAL IS WHAT OOM'd, not training. mk_dev's 10000 is sized for an 80 GB card; here
# it asked for a 29.68 GiB allocation on a 16 GB one and took the run down after the
# phase transition. Training itself was fine at batch 100 -- the OOM handler even
# recovered it to 50. Eval quality is irrelevant to this measurement (we are timing
# steps, not judging a model), so it is cut to something that fits.
EVAL_SAMPLES = 500
TEST_EVAL_SAMPLES = 200

# T=10, NOT prod0810's 60. Tried 60 first for fidelity to the cluster arm; it ran
# ~12 minutes on this card without completing a single training step, which is the
# user's warning borne out ("T=60 will struggle locally quite a bit").
#
# THE CAVEAT THIS BUYS, stated so the result is not over-read. The rollout scales
# with T; the energy call does not. So at T=10 the energy is a LARGER share of the
# step than it is on the cluster at T=60 -- `energy/frac_of_step` measured here is
# an UPPER BOUND on the cluster's value, not an estimate of it.
#
# The A/B is unaffected: both arms run the same T, so the DELTA in step time and
# utilization is still the quantity of interest, and a bigger energy share makes
# that delta easier to see, not harder.
TRAJECTORY_T = 10


def load_base():
    with BASE.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build(name, host_gas, grow_batch=False):
    """
    grow_batch defaults FALSE for the A/B arms, and that is a design decision rather
    than caution. The batch controller responds to measured throughput -- which is
    exactly what the two arms differ in -- so leaving it live would let the arms
    drift to DIFFERENT batch sizes and then the utilization comparison would be
    confounded by the very effect it is trying to measure. Pin the batch, vary one
    thing. The controller gets its own arm (c_controller) where it is the subject
    rather than a nuisance variable.
    """
    cfg = deepcopy(load_base())
    cfg.update(
        run_name=name,
        tag=TAG,
        seed=12345,                       # identical across arms: same policy, same draws
        device='cuda',
        # 0.7 capped PyTorch at 11.14 GiB and the transition OOM'd with 4 GiB of the
        # card still physically free but out of reach. 0.85 = 13.5 GiB, leaving ~2.4
        # GiB for the desktop. Still well under the 0.9 that caused the collision.
        cuda_memory_fraction=0.85,
        checkpoint_read_only=True,
        checkpoint_name=None,
        continue_from_checkpoint=False,
        prior_model_name=None,
        load_weights_only=False,
        prior_path=PRIOR,
        molecules_path=PRIOR,
        mlip_path=MLIP,
        energy_function='uma',
        space_groups=[2],
        z_primes=[1],
        epochs=EPOCHS,
        eval_num_samples=EVAL_SAMPLES,
        test_eval_num_samples=TEST_EVAL_SAMPLES,
        eval_period=EVAL_PERIOD,
        figure_period=FIGURE_PERIOD,
        figs_period=FIGURE_PERIOD,
        batch_size=BATCH,
        max_batch_size=MAX_BATCH if grow_batch else BATCH,
        grow_batch_size=grow_batch,
        archive_period=0,                 # no archives from a read-only smoke
    )
    # BOTH, or preflight_config refuses the run: evaluating at a different dt than
    # the policy was trained at integrates a different SDE, so the eval numbers
    # become a dt artifact. (Caught by loading the config, not by generating it.)
    cfg['integrator']['T'] = TRAJECTORY_T
    cfg['eval_T'] = TRAJECTORY_T
    cfg['energy_config']['host_gas_phase_reference'] = host_gas
    # TRUE here, FALSE on the cluster, and this costs the measurement NOTHING.
    #
    # The transition's bulk scoring calls (top_up_prior_from_anchors draws
    # reach_topup_size=1000 crystals and scores them in one shot) are sized for an
    # 80 GB card. With recovery FALSE there is no sub-batching to fall back on, so
    # on 16 GB they ask for ~5 GiB in a single allocation and die -- and because
    # they run from protocol.advance(), outside the training loop's try/except,
    # the OOM is FATAL rather than recoverable. Two smokes died exactly there.
    #
    # Why it does not contaminate the A/B: the chunker only engages when its own
    # size is BELOW the batch it is handed (molecular_crystal.py's
    # `self.batch_size < n_samples` growth gate), and it starts at 1000 for uma.
    # Training steps here run at batch 100, so they take a single call either way
    # -- identical to the cluster's behaviour. Only the 1000-crystal bulk calls
    # chunk, and those are transition scaffolding, not what is being timed.
    cfg['energy_config']['internal_oom_recovery'] = True

    # THE TRANSITION IS THE EXPENSIVE PART, and its cost is NOT protected. The
    # `equilibration` stage's on_enter runs rebuild_prior_by_churn from inside
    # protocol.advance(), which is called from evaluation() -- OUTSIDE the training
    # loop's try/except. So an OOM there is fatal: no batch cut, no recovery, run
    # over. That is what killed the first two smokes here, and it is the same
    # transition where prod0810's uma arm got into trouble.
    #
    # mk_dev's prior buffer targets max_size * init_fraction = 62500 rows, which on
    # the cluster took 7 churn cycles of UMA scoring. Scaled down here so the
    # transition fits a 16 GB card. Does NOT affect the A/B: both arms churn
    # identically, and what is being compared is the per-step energy cost afterwards.
    prior = cfg['buffers']['prior_buffer']
    prior['max_size'] = 20000            # -> target 5000 rows
    prior['min_size'] = 2000
    prior['churn_batch_ref'] = 200       # smaller draws per churn cycle
    cfg['buffers']['replay_buffer']['max_size'] = 4000
    cfg['buffers']['anchor_buffer']['max_size'] = 20000

    # PHASE 1 EXITS IMMEDIATELY -- see module docstring. Exit terms are AND-ed and
    # non-eval terms use a pass-streak, so one loose term with the default patience
    # of 1 fires at the first eval.
    stages = cfg['protocol']['stages']
    stages[0]['exit'] = [{'metric': 'bwd/tbc', 'below': 1.0e+9}]
    return cfg


#: (name, host_gas_phase_reference, grow_batch_size)
#:
#:   a/b  the controlled A/B -- batch PINNED so the only difference is the MLIP path
#:   c    the batch controller as the SUBJECT, on the optimised path. Watches whether
#:        it does anything alarming in a real loop; its timings are not comparable to
#:        a/b because its batch moves.
#:
#: HISTORICAL NOTE -- c_controller originally also carried gpu_util_floor: 70, and
#: that arm is what killed the floor. Every one of its four growths was floor-driven
#: (the throughput gate never fired): batch 100->741 took utilization 52->42% and
#: samples/sec 57.7->24.3, and the arm timed out at 240/400 steps while b_optimised
#: finished 400 in half the wall clock. `gpu_util_floor` is now retired and setting it
#: hard-fails preflight, so re-running this arm exercises the throughput gate alone.
#: `smoke` is GENERATED here rather than sed'd out of another arm at run time. The
#: sed version produced a 0-byte config the moment anything disturbed it, and the
#: resulting failure ("'NoneType' has no attribute 'items'") points at the config
#: loader rather than at the generator that actually broke.
ARMS = [
    ('smoke', True, False, 60),
    ('a_baseline', False, False, EPOCHS),
    ('b_optimised', True, False, EPOCHS),
    ('c_controller', True, True, EPOCHS),
]


def write_all():
    for name, host_gas, grow, epochs in ARMS:
        cfg = build(name, host_gas, grow_batch=grow)
        cfg['epochs'] = epochs
        assert cfg['checkpoint_read_only'] is True, f'{name}: MUST NOT write checkpoints'
        assert cfg['checkpoint_name'] is None and not cfg['continue_from_checkpoint'], \
            f'{name}: must start cold, not resume something of the user\'s'
        assert cfg['epochs'] > 0
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        print(f'  {name:14s} host_gas={str(host_gas):5s} grow={str(grow):5s}')
    print(f'\nwrote {len(ARMS)} arms in {HERE}')
    print('run a_baseline with MXT_VECTORISED_ATOMICDATA=0, b_optimised with =1')


def preflight():
    bad = 0
    for path in (PRIOR, MLIP):
        if not os.path.isfile(path):
            print(f'  MISSING  {path}'); bad += 1
    if bad:
        print(f'{bad} missing file(s) -- these arms cannot run'); return 1
    print('preflight OK -- prior dataset and MLIP checkpoint both present')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    a = ap.parse_args()
    raise SystemExit(preflight() if a.preflight else (write_all() or 0))
