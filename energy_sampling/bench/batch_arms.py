"""
The arms. Fixed batches are ARMS, never an oracle.

`bench/README.md`'s first rule: the retired stack selected a "best fixed rate" and
divided by it; that selection went wrong three separate ways and each time silently
rescaled results rather than failing. The same applies one axis over. A fixed-batch
ladder needs no reference -- "the controller must not be beaten by every constant
batch" is read by comparing rows.

EVERY ARM MUST BE PROVABLY DISTINGUISHABLE FROM `Null`. Enforced in
`bench/test_batch_traps.py`. An arm that silently no-ops does not error -- it posts a
plausible row. That has now fired twice in this repo (`ray+ray` armed after the
optimizer step; and, measured during this design, an injected occupancy rule that was
BIT-IDENTICAL to null on a device whose throughput happened to rise).
"""


class Arm:
    """Config overrides plus an optional live-code patch. Nothing else."""

    name = 'arm'

    def args_overrides(self):
        return {}

    def reset(self, run):
        pass

    #: Injected-defect arms override this to mutate the REAL class. Returning the
    #: originals lets a test restore them, so an injection cannot leak between cases.
    def patch(self, cls):
        return {}


class Null(Arm):
    """
    The do-nothing control: growth off, batch never moves.

    Load-bearing twice over. It is the reference every dominance verdict is taken
    against, and it is the tell for a silent no-op -- two rows agreeing to many
    significant figures means an arm did nothing, and the only way to know is to have
    something that genuinely does nothing to compare against.
    """

    name = 'null'

    def __init__(self, batch=1000):
        self.batch = int(batch)

    def args_overrides(self):
        return dict(batch_size=self.batch, max_batch_size=self.batch,
                    grow_batch_size=False, batch_util_target=0)


class Fixed(Arm):
    """A constant batch, growth machinery ON but pinned by `max_batch_size == batch`."""

    def __init__(self, batch):
        self.batch = int(batch)
        self.name = f'fixed@{self.batch}'

    def args_overrides(self):
        return dict(batch_size=self.batch, max_batch_size=self.batch,
                    grow_batch_size=True, batch_util_target=0)


class Ship(Arm):
    """
    The shipping controller at mk_dev settings, with the ladder bounds opened.

    Under the state-8 replacement (train.select_batch_size) and WITHOUT a
    `batch_util_target`, this HOLDS the base batch (S3: no constraint, no walk) --
    so its batch trace is identical to `Null`'s BY DESIGN, and a
    distinguishability check must use `Sizer` below, not this arm. `max_batch_size`
    is still opened so the safety bounds and any injected defect have room to act;
    mk_dev ships `1000/1000`, where the domain has ONE rung regardless.
    """

    name = 'ship'

    def __init__(self, batch=1000, max_batch=50000, **overrides):
        self.batch = int(batch)
        self.max_batch = int(max_batch)
        self.overrides = overrides

    def args_overrides(self):
        # target 0 EXPLICITLY: this arm's definition is the no-target regime
        # (see the docstring), and the canonical default it would otherwise
        # inherit became 60 on 2026-08-19. Sizer re-overrides it.
        return dict(batch_size=self.batch, max_batch_size=self.max_batch,
                    grow_batch_size=True, batch_util_target=0, **self.overrides)


class Sizer(Ship):
    """
    The occupancy ladder ARMED: `batch_util_target` set, so the controller
    calibrates rung by rung over real steps and holds the smallest rung whose
    measured occupancy clears the target -- or the argmax-occupancy rung, said
    INFEASIBLE, when none does. This is the arm that must be distinguishable
    from `Null`: it moves during calibration even when it concludes by
    returning to the base.
    """

    def __init__(self, util_target=0.6, **kw):
        super().__init__(**kw)
        self.util_target = float(util_target)
        self.name = f'sizer@{self.util_target:g}'

    def args_overrides(self):
        return dict(super().args_overrides(), batch_util_target=self.util_target)


class DescentWalk(Ship):
    """
    TRAP (b) INJECTED: a periodic, floorless downward walk -- the retired knee
    recheck's shape, reintroduced.

    The shipping controller no longer contains ANY walk, so trap (b) is prevented
    by construction and `_batch_floor` is no longer what stops a descent. That is
    precisely why this injection exists: the design says the detection cases stay
    because the walk could be reintroduced, and a detector needs a reintroduction
    to redden on. The injected rule drops one rung every `period` steps with no
    floor, which is the old recheck minus the part that saved it.

    The defect's real name, measured on the old controller: not "descends
    forever" (that needed t_fixed exactly 0) but "descends to a level the
    configuration never chose" -- prod0810 ran a whole stage at 0.825x its
    configured batch that way.
    """

    name = 'ship+descentwalk'

    def __init__(self, period=2000, **kw):
        super().__init__(**kw)
        self.period = int(period)

    def patch(self, cls):
        orig = cls.select_batch_size
        period = self.period

        def injected(m):
            orig(m)
            if m.step_ind - getattr(m, '_dw_last_drop', 0) >= period:
                m._dw_last_drop = m.step_ind
                f = float(getattr(m.args, 'batch_growth_factor', 1.65))
                m.batch_size = max(1, int(round(m.batch_size / f)))

        cls.select_batch_size = injected
        return {'select_batch_size': orig}


class OccupancyFloor(Ship):
    """
    TRAP (a) INJECTED: an occupancy ACTUATOR restored above the controller.

    Reproduces the deleted `gpu_util_floor` in the position it occupied -- at the
    TOP, with its own early return, so it outranks everything below it. That
    textual position IS the trap: the rule was priority 1, so nothing could
    refuse the growths it drove. It is also the S1 violation in its purest form:
    an occupancy reading SELECTING a batch (grow!) rather than vetoing candidates
    under a fixed rule.

    `gpu_util_floor` is a retired key whose mere PRESENCE is a load-time hard
    error, so this arm carries the threshold itself rather than putting it in
    `args` -- which is also a check that the retirement machinery still bites.
    """

    name = 'ship+occfloor'

    def __init__(self, util_floor=60.0, **kw):
        super().__init__(**kw)
        self.util_floor = float(util_floor)

    def patch(self, cls):
        orig = cls.select_batch_size
        floor, factor_key = self.util_floor, 'batch_growth_factor'

        def injected(m):
            u = m._gpu_util_mean(float(m.args.gpu_util_policy_window_s))
            if (u is not None and u < floor
                    and m.batch_size < m.args.max_batch_size
                    and m.step_ind - getattr(m, 'batch_size_last_grow', 0)
                    >= int(m.args.batch_growth_interval)):
                f = float(getattr(m.args, factor_key, 1.65))
                m.batch_size = min(int(m.args.max_batch_size),
                                   max(m.batch_size + 1, int(round(m.batch_size * f))))
                m.batch_size_last_grow = m.step_ind
                m.batch_sizer = None    # the rule outranks whatever was concluded
                return                  # <-- the early return IS the priority
            return orig(m)

        cls.select_batch_size = injected
        return {'select_batch_size': orig}
