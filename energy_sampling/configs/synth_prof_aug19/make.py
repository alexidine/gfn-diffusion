"""
synth_prof_aug19 -- turn the profiling machinery ON and see what it says.

WHY A SEPARATE ARM RATHER THAN A FLAG ON AN EXISTING ONE. Both profiling layers
ship `enabled: false`, so no run in this session's earlier batteries carries a
single `perf/*` or `energy/seconds_gpu` value -- the machinery is built and
tested (`test_profiling.py`, 22 tests) and has never been read on a real run.

THE LADDER IS DELIBERATELY OFF HERE, and that is the one non-obvious choice.
Everywhere else this session the armed ladder is the point; for a profiling run
it is a confound. A batch that changes mid-run changes the work per step, the
compile shape, and the memory profile, so per-step timings before and after a
rung are not comparable -- and the torch.profiler window would straddle the
change. Fixed batch, fixed shape, then measure.

WHAT THE TWO LAYERS ANSWER, and they are not the same question:

  1. REGION TIMER (`profiling.enabled`) -> `energy/seconds_gpu` and
     `energy/gpu_over_wall`. CUDA events bracketing the energy call, compared
     against the wall clock already measured around it. This is the validation
     the profiling module's own docstring asks for FIRST: `energy/seconds_in_step`
     is a wall-clock SUBDIVISION of the step taken without synchronising, and
     wall clock around async CUDA work measures when the launches returned, not
     when the device finished. `gpu_over_wall` is the number that says whether
     that subdivision is honest.

     NB only ONE region is instrumented in the tree today (`energy`, in
     energies/base_set.py). So this layer validates the energy split and CANNOT
     decompose the step -- there are no rollout/backward/optimizer regions to
     report. That is a real limit of the current instrumentation, not of the run.

  2. TRACE WINDOW (`profiling.trace.enabled`) -> a chrome trace + an op table
     written to `profiling_results/`, never to wandb. This is where a step
     DECOMPOSITION actually comes from today, and it is the layer that can speak
     to the standing dispatch-bound result (~937 nn.Module calls per training
     step; widths 64/256/512 costing the same). Bounded to a few steps because
     left on, torch.profiler dominates what it measures.

PREDICTIONS, written before the run (the arm is worthless if it cannot come back
negative):

  * `energy/gpu_over_wall` < 1 on this route -- ELJ is a cheap energy on a
    dispatch-bound box, so device-busy time should be a FRACTION of the wall
    window around the call. If it comes back at or above 1, the wall clock is
    UNDER-attributing the energy call, and `energy/frac_of_step` -- which every
    MLIP optimisation decision leans on -- is reading low.
  * `perf/*` and `energy/seconds_gpu` present and non-zero, i.e. the machinery
    actually emits on a real run rather than only in its tests.
  * the trace window writes files and then latches shut, so the run's step time
    returns to baseline after `active_steps`.

    python configs/synth_prof_aug19/make.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'configs'))

import generate                                    # noqa: E402

TAG = 'synth_prof_aug19'
ARCHIVE = ('dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92'
           '_phase1_exit.pt')
ARCHIVE_STEP = 430
STEPS = 500

#: The trace window opens well after the resume so it profiles TRAINING rather
#: than the resume itself -- buffer restore, the prior re-analysis and the first
#: post-transition steps are startup, and canonical's own start_step is late for
#: the same reason.
TRACE_START = ARCHIVE_STEP + 200


def arms():
    return {
        f'{TAG}_prof': generate.arm(
            f'{TAG}_prof', problem='mipcas_elj', tag=TAG,
            checkpoint_name=ARCHIVE, continue_from_checkpoint=False,
            load_weights_only=False, prior_model_name=None,
            checkpoint_read_only=True,
            epochs=ARCHIVE_STEP + STEPS,
            eval_period=100, figs_period=200, archive_period=100000,
            eval_num_samples=2000,
            # FIXED BATCH -- see the module docstring
            batch_util_target=0, grow_batch_size=False, max_batch_size=1000,
            **{'profiling.enabled': True,
               'profiling.regions': None,
               'profiling.trace.enabled': True,
               'profiling.trace.start_step': TRACE_START,
               'profiling.trace.active_steps': 8,
               'profiling.trace.outdir': 'profiling_results'}),
    }


def check(cfgs):
    for name, cfg in cfgs.items():
        assert cfg['epochs'] > ARCHIVE_STEP + 100, name
        assert cfg['profiling']['enabled'] is True, name
        assert cfg['profiling']['trace']['enabled'] is True, name
        # the trace window must OPEN inside the run, or it silently never fires
        assert ARCHIVE_STEP < cfg['profiling']['trace']['start_step'] < cfg['epochs'], (
            f"{name}: trace start_step {cfg['profiling']['trace']['start_step']} "
            f"outside the run's ({ARCHIVE_STEP}, {cfg['epochs']}) range")
        # a moving batch would confound every per-step number
        assert cfg['grow_batch_size'] is False, name
        assert float(cfg['batch_util_target']) == 0, name
    print(f'  checks passed on {len(cfgs)} arms')


if __name__ == '__main__':
    cfgs = arms()
    check(cfgs)
    generate.emit(cfgs, outdir=HERE)
