"""
Eight short local arms, whose only job is to test the off-thread occupancy sampler
against wandb's out-of-process `system.gpu.0.gpu`.

WHY A SWEEP AND NOT A RUN. The defect being tested for was not an offset; it was a
bias whose SIGN FOLLOWED THE BATCH -- handoff §2 measured -6 pts at batch 1000 and
+40 pts at 7410 against the same out-of-process window. A single arm agreeing
proves only that one duty cycle was sampled fairly, which is exactly what the old
sampler also managed at some batch sizes. So the axes swept are the ones that move
the SHAPE of a step -- how much of it is GPU-saturated and how much is host, and
how many launches it is chopped into:

    batch      125 / 500 / 2000     more work per launch
    T          10 / 100             more launches per step (ship length is T=100)
    width      128 / 512 / 1024     more work per launch, no change in count
    energy     elj / latent_gaussian / uma
                                    latent_gaussian is nearly pure host; uma is
                                    the saturated extreme and the route the
                                    original disagreement was measured on

One arm at a time off a shared baseline (b500), so a delta that moves is
attributable to the axis that moved.

WHAT THIS CANNOT SHOW. An RTX 5080 laptop is not an A100, eval here is seconds
rather than minutes, and no arm runs long enough to enter the regimes §3 studies.
A clean result here says the sampler is unbiased ACROSS THESE DUTY CYCLES; it does
not license restating any number in the handoff, and the cluster comparison still
has to be made on a real arm.

DELIBERATELY INERT CONTROLS. `grow_batch_size: false` and `batch_util_target: 0`,
so the batch sizer cannot move the batch: a moving batch is the one thing that
would make the two series legitimately disagree, and this measures the sensor, not
the sizer. NB canonical mk_dev SETS `batch_util_target: 0.65`, so switching it off
is a departure from every other arm, not the default -- which is also why the
sensor's calibration matters: the shipped configuration grows batches on it.

Arms are wall-clock-bounded by `run_sweep.py`, not by `epochs` -- see there.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs import generate  # noqa: E402

#: shared by every arm.
#:
#: THE SAMPLING RATE IS THE MEASUREMENT'S RESOLUTION, and 10 s was not enough --
#: measured here, not assumed. A 180 s nvidia-smi trace at 1 Hz taken while an arm
#: of this battery was training:
#:
#:     mean 34.1, sd 19.2, range 2..83     -- occupancy is BIMODAL, not steady
#:     autocorrelation ~0.4-0.5 out to 20 s
#:     true 120 s window mean            25.7
#:     what a 10 s POINT sampler reports  21.6 .. 30.2 depending only on which
#:                                        phase it starts at (sd 3.4 across phases)
#:
#: So at a 10 s period a 120 s window holds 12 point samples of a signal that
#: swings 2->83, and the window mean carries +-3.4 pts of pure sampling phase.
#: The first two arms measured -6.5 and +4.2 -- barely outside that, i.e. the
#: comparison was noise-limited by our own sampler and could not have answered
#: the question either way. At 2 s the window is covered densely enough that its
#: mean estimates the window rather than a lottery of instants. The cost is a
#: 46 ms nvidia-smi fork every 2 s (~2% of one core), off the training thread.
#:
#: NB this is a property of the SHIPPING sensor too: canonical mk_dev samples at
#: 60 s into a 900 s window, i.e. 15 point samples -- see the report, this wants
#: changing and mk_dev is not mine to edit.
COMMON = dict(
    tag='utilphase_sep09',
    epochs=1000000,          # never the binding limit; run_sweep.py stops on time
    grow_batch_size=False,
    batch_util_target=0,
    gpu_util_sample_period_s=2,
    gpu_util_window_s=120,
    gpu_util_policy_window_s=300,
    eval_period=250,
    checkpoint_read_only=True,
    continue_from_checkpoint=False,
    # WEIGHTS ONLY. The canonical warm start also restores the local dev buffer
    # sidecar, which is still format v1 and is refused by the 2026-09-07 currency
    # check. Migrating that file is a real repair of a real artifact and has
    # nothing to do with this test, so these arms take the weights and start
    # their buffers fresh.
    load_weights_only=True,
)


def widths(n):
    """Every per-sub-model hidden dim at once -- a 'model size' that moves one of
    them is a different experiment (and mostly a null one)."""
    return {f'model.{k}_hidden_dim': n
            for k in ('t', 's', 'policy', 'flow', 'cond')}


def elj(name, **over):
    return generate.arm(f'up_{name}', problem='mipcas_elj',
                        **{**COMMON, **over, 'max_batch_size': over.get('batch_size', 500)})


arms = {
    # ---- baseline, and the batch axis around it
    'b125': elj('b125', batch_size=125),
    'b500': elj('b500', batch_size=500),
    # THE NOISE FLOOR, measured rather than modelled: b500 run twice. Whatever
    # these two differ by is the resolution of every other comparison in the
    # table, and an axis effect smaller than it is not an effect.
    'b500r': elj('b500r', batch_size=500),
    'b2000': elj('b2000', batch_size=2000),
    # ---- launches per step
    # eval_T tracks integrator.T by rule -- the policy's heads are learned at one
    # dt, so evaluating at another integrates a different SDE (generation refuses
    # the mismatch outright).
    'T100': elj('T100', batch_size=500, eval_T=100, **{'integrator.T': 100}),
    # ---- model size
    'w128': elj('w128', batch_size=500, **widths(128)),
    'w1024': elj('w1024', batch_size=500, **widths(1024)),
    # ---- energy function: the host-bound end...
    # COLD. The canonical `checkpoint_name` is an eLJ phase-1 exit; its weights do
    # not describe this problem and its problem hash would refuse the load anyway.
    'lgauss': generate.arm('up_lgauss', problem='latent_gaussian',
                           **{**COMMON, 'batch_size': 500, 'max_batch_size': 500,
                              'checkpoint_name': None}),
    # ...and the saturated end. Batch 100 because eSEN at 500 does not fit on a
    # 16 GB laptop card; the point of this arm is the duty cycle, not the size.
    # also cold: the energy function is part of the problem identity, so the eLJ
    # exit is not a legitimate warm start for it.
    #
    # AND A DIFFERENT PRIOR. The mipcas eLJ prior carries a thermal_scaling_factor
    # of 0.3636, which is applied inside compute_eLJ_energy and reaches no other
    # route -- train.py refuses the pairing outright rather than train at a
    # silently wrong effective temperature. The SMOKE prior is the same problem
    # (mipcas, sg2, Z'=1) built for the uma route, so the energy function is the
    # only thing this arm moves.
    'uma': elj('uma', batch_size=100, energy_function='uma', checkpoint_name=None,
               mlip_path=r'D:\crystal_datasets\esen_s.pt',
               prior_path=r'D:\crystal_datasets\conditional\priors\SMOKE_mipcas_sg2_zp1_uma_prior_dataset.pt',
               molecules_path=r'D:\crystal_datasets\conditional\priors\SMOKE_mipcas_sg2_zp1_uma_prior_dataset.pt'),
}

if __name__ == '__main__':
    generate.emit(arms, outdir=Path(__file__).parent)
