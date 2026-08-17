"""
ONE BATCH RUN: a device, an arm, a seed. The trace is the only output.

`bench/runner.py` deliberately does NOT exercise the batch sizer -- it says so at
`bench/runner.py:185` ("The batch sizer is not exercised here"), because an LR arm and
a batch arm need different clocks. This is the batch half, built to the same four rules
(`bench/README.md`): no oracle, no threshold/budget/censoring, metrics are pure
functions of the trace, and the MODELLER is faked while the CONTROLLER is real.

THE STEP-BODY ORDER IS LOAD-BEARING AND IS COPIED, NOT INVENTED. `train.py` appends
the timing and the work to the deques and THEN calls `increment_batch_size`, so the
controller always scores the step it just timed, at the batch that actually ran it.
`bench/old/harness.py` inverted parts of this and that is how a rung baseline could be
compared against a step the next rung had already paid for. The order here is:

    attempt captured  ->  step timed (or OOM)  ->  deques appended  ->  clock advanced
    (incl. eval)  ->  _sample_gpu_util()  ->  handle_train_epoch_error | increment_batch_size

TWO LOSSES, TWO CLOCKS, DELIBERATELY -- the same split `bench/runner.py:203-208` makes
between `loss` and `eloss`:

  * the CONTROLLER acts on `dt_observed`  -- jittered, recompile-charged: what a real
    run's `_recent_step_times` would hold;
  * the SCORING reads `true_t`            -- noise-free: the quality of the decision.

Scoring on the jittered series ranks arms partly on their draw. Scoring on the
noise-free one while the controller acts on the noisy one is the whole point.
"""

from bench.fake_modeller import (FakeModeller, FakeStage, attach_real_batch_sizer,
                                 make_args)

#: Fraction of a step's time an OOM'd step still costs. An OOM is not free -- the
#: allocation attempt, the exception, and `handle_train_epoch_error`'s gc +
#: empty_cache all take wall clock. Charged so a thrashing arm pays for its thrash.
OOM_STEP_FRAC = 0.5


class BatchRun:
    """
    One arm on one device, stepped to completion, recording a trace.

    `arm` supplies the config overrides and (optionally) a `patch(cls)` hook that
    installs an injected defect. Nothing else -- the control law under test is the
    REAL `train.Modeller.increment_batch_size`, bound on by `attach_real_batch_sizer`.
    """

    def __init__(self, device, arm, seed=0, steps=20000, stage='equilibration',
                 train_mode='fused', zcal=None, oom_step_frac=OOM_STEP_FRAC):
        attach_real_batch_sizer()
        self.device = device
        self.arm = arm
        self.seed = int(seed)
        self.steps = int(steps)
        self.stage = stage
        self.oom_step_frac = float(oom_step_frac)
        #: z_calibration rollouts attached to a step. `train.py` charges the rung
        #: `batch * (1 + n_zcal_rollouts)`; getting that wrong (seconds in the
        #: denominator, samples not in the numerator) is one of the three accounting
        #: bugs that walked prod0810 to batch 12226, so it is modelled, not ignored.
        self.zcal = zcal or (lambda step_ind: 0)

        args = make_args(**arm.args_overrides())
        self.m = FakeModeller(args, optimizers={},
                              stage=FakeStage(name=stage, train_mode=train_mode))
        #: THE ONE LINE THAT MAKES THE OCCUPANCY CHANNEL LIVE. `_read_gpu_util` returns
        #: None when `_bench_gpu` is unset, and `train.py` then latches `_gpu_util_off`
        #: PERMANENTLY, prints once, and does not raise. A sandbox that forgets this
        #: gets a dead sensor that looks like a clean run -- the swallowed-diagnostic
        #: shape. The only other writer of `_bench_gpu` in the repo is the retired
        #: `bench/old/harness.py`.
        self.m._bench_gpu = device
        self.m._bench_attempted = self.m.batch_size

        self.trace = []
        self.n_oom = 0
        arm.reset(self)

    # ------------------------------------------------------------------- step

    def step(self):
        m, dev = self.m, self.device
        m.step_ind += 1
        attempted = int(m.batch_size)
        z = int(self.zcal(m.step_ind))
        true_work = attempted * (1 + z)

        oom = False
        try:
            dt = dev.step_time(true_work)
        except RuntimeError:
            # a failed allocation still costs wall clock, and the recovery costs more
            dt = dev.true_step_time(true_work) * self.oom_step_frac
            oom = True
            self.n_oom += 1

        m._recent_step_times.append(dt)
        m._z_cal_rollouts = z
        m._recent_step_work.append(true_work)
        # VIRTUAL CLOCK, advanced by the step AND by any eval block. The eval block is
        # what makes the in-process sensor's blindness expressible: it moves the clock
        # forward without producing a sample, so the trailing-window mean is computed
        # over a span it did not observe.
        eval_s = dev.eval_block(m.step_ind)
        m.sim_time += dt + eval_s
        m._bench_gpu, m._bench_attempted = dev, attempted
        m._sample_gpu_util()                       # the REAL sampler and its 60 s gate

        if oom:
            m.handle_train_epoch_error(
                RuntimeError('CUDA out of memory. Tried to allocate 2.00 GiB '
                             '(synthetic bench OOM)'), self.stage)
        else:
            m.increment_batch_size()               # the REAL control law

        outside = getattr(dev, 'outside_range', lambda b: False)
        self.trace.append({
            'step': m.step_ind,
            'batch': attempted,
            'stage': self.stage,
            'dt_observed': dt,                     # what the controller acted on
            'true_t': dev.true_step_time(true_work),   # what the SCORING reads
            'true_work': float(true_work),
            'true_sps': dev.throughput(true_work),
            'true_util': dev.true_utilization(attempted),   # ground truth
            'util_reading': m._gpu_util_mean(
                float(m.args.gpu_util_policy_window_s)),    # what the run would log
            'eval_s': eval_s,
            'oom': oom,
            'ceiling': m.batch_size_oom_ceiling,
            # --- GRID-EDGE BOOKKEEPING. Every row carries whether the batch was
            #     pressed against a boundary, so a metric can restrict to interior
            #     readings instead of a human remembering to. Not doing this by hand
            #     is what the retracted ray result cost.
            'outside_range': bool(outside(attempted)),
            'at_max_batch': attempted >= int(m.args.max_batch_size),
            'at_floor': attempted <= int(m._batch_floor()),
        })

    def run(self):
        for _ in range(self.steps):
            self.step()
        return self
