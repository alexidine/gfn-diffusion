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
                    grow_batch_size=False)


class Fixed(Arm):
    """A constant batch, growth machinery ON but pinned by `max_batch_size == batch`."""

    def __init__(self, batch):
        self.batch = int(batch)
        self.name = f'fixed@{self.batch}'

    def args_overrides(self):
        return dict(batch_size=self.batch, max_batch_size=self.batch,
                    grow_batch_size=True)


class Ship(Arm):
    """
    The shipping controller at mk_dev settings, with the ladder bounds opened.

    `max_batch_size` must be raised above `batch_size` or the walk cannot move at all:
    mk_dev ships `1000/1000`, so on the canonical config the domain has ONE rung and
    the entire growth mechanism is inert. That is worth knowing and is not a cell.
    """

    name = 'ship'

    def __init__(self, batch=1000, max_batch=50000, **overrides):
        self.batch = int(batch)
        self.max_batch = int(max_batch)
        self.overrides = overrides

    def args_overrides(self):
        return dict(batch_size=self.batch, max_batch_size=self.max_batch,
                    grow_batch_size=True, auto_batch_throughput_opt=True,
                    **self.overrides)


class NoFloor(Ship):
    """
    TRAP (b) INJECTED: `_batch_floor` returns 1.

    `train.py`'s own docstring calls the floor load-bearing: with throughput flat in
    batch every jump fails the gate and the periodic recheck walks the batch down a
    rung at a time, because each recheck re-tests one rung LOWER than the last pin and
    never re-tests the level above it.

    MEASURED CORRECTION TO THE HEADLINE, and it matters for how the case is written:
    the descent does NOT run to 1 in general. It runs to the closed-form knee and
    stops -- `t_fixed=0.001 -> 50`, `0.01 -> 366`, `0.1 -> no descent at all`. Reaching
    batch 1 requires `t_fixed` EXACTLY zero, a measure-zero and physically impossible
    point. So "descends forever" is true only at a point no hardware occupies; the real
    defect is "descends to a level the configuration never chose", which is what
    actually cost prod0810 (it ran a whole stage at 0.825x its configured batch).
    """

    name = 'ship+nofloor'

    def patch(self, cls):
        orig = cls._batch_floor
        cls._batch_floor = lambda self: 1
        return {'_batch_floor': orig}


class OccupancyFloor(Ship):
    """
    TRAP (a) INJECTED: an occupancy rule restored above the throughput gate.

    Reproduces the deleted `gpu_util_floor` in the position it occupied -- at the TOP
    of the ladder, with its own early return, so it outranks everything below it
    including the throughput gate. That textual position IS the trap: the rule was
    priority 1, so it overrode a gate that would have refused all four of the growths
    it drove.

    `gpu_util_floor` is a retired key whose mere PRESENCE is a load-time hard error, so
    this arm carries the threshold itself rather than putting it in `args` -- which is
    also a check that the retirement machinery still bites.
    """

    name = 'ship+occfloor'

    def __init__(self, util_floor=60.0, **kw):
        super().__init__(**kw)
        self.util_floor = float(util_floor)

    def patch(self, cls):
        orig = cls.increment_batch_size
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
                m._rung_throughput = None
                m.batch_size_saturated_stage = None
                return                      # <-- the early return IS the priority
            return orig(m)

        cls.increment_batch_size = injected
        return {'increment_batch_size': orig}
