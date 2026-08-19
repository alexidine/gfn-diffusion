"""
`bench/old/` is not uniformly old, and `norecursedirs` could not say so.

The directory was created when `bench/` was rebuilt (2026-08-13) after an
adversarial review retired the previous generation. `pytest.ini` then excluded the
whole directory -- correct for the retired BATTERY MACHINERY, and wrong for the
three test files that drive SHIPPING code and had simply been sitting next to it.

MEASURED 2026-08-16, whole directory, project venv:

    bench/old/                          111 passed, 3 skipped     65 s

Nothing here is broken. Everything passes against the current trainer. The
question was never "do these still run", it was "do they protect anything the
collected suite does not", and that was settled by mutation:

  * `train.Modeller._batch_floor -> 1` (re-introduces trap (b), the knee walk with
    no floor that descends forever under flat throughput):
        bench/old/test_batch_sizer.py + test_batch_adversarial.py   3 RED
            test_flat_throughput_walks_down_only_to_the_floor
            test_gain_at_or_above_factor_minus_one_rejects_every_jump
            test_A4_pin_does_not_rebuild_the_oom_sawtooth
        bench/test_oom_ceiling_expiry.py (COLLECTED)                3 RED
            -- 2 genuine (batch stuck at 303; descent 1000 -> 606), 1 a
               scaffolding assert that only noticed its own precondition moved.
    So the floor itself is NOT unprotected. The batch files overlap the collected
    suite more than the exclusion's reputation suggested.

  * `LRController.on_calibration -> no-op` (neuters the ray sensor's actuator path):
        bench/old/test_lr_controller.py                            12 RED
        the ENTIRE collected bench/                                 1 RED
            -- and that one is test_arms.py's generic
               `test_ray_is_distinguishable_from_null`, which says an arm went
               inert, not WHICH behaviour broke.
    Twelve named v8 behaviours -- warmup hold, asymmetric update, peak bounds, the
    permanent divergence ceiling, the servo cut from a hot start, saturated-sensor
    open loop, unresolved/inconsistent producing no move, the unmanaged-key control
    arm -- were protected here and nowhere else, on the controller that ships.

Hence this file instead of a directory-wide exclusion: collect what drives live
symbols, ignore what tests the machinery the review condemned.

WHAT STAYS IGNORED, and why each is genuinely retired rather than merely old --
all three test `bench/old`'s OWN apparatus, and that apparatus is the thing the
adversarial review found defective (see bench/README.md, "What killed the previous
generation"):

    test_scenarios.py            bench.old.oracle / bench.old.scenarios -- the
                                 selected-reference-rate machinery. A reference
                                 used as four things at once, 187x off on one
                                 surface family.
    test_off_target.py           bench.old.scenarios.ON_TARGET_BAND -- the band
                                 that was exactly the reciprocal of the
                                 controller's divergence_cut, so one cut landed
                                 bit-exactly on the boundary.
    test_crucible_feasibility.py bench.old.crucible._cold_start_feasible -- the
                                 cold-start budget that was wrong at both ends.

None of the three imports a shipping symbol; they are self-tests of retired code,
and re-collecting them would re-assert the constants the rebuild exists to be rid
of.

ADDED 2026-08-19 (state 8): the two BATCH files joined the ignore list when the
throughput knee walk they test was deleted from `train.Modeller` (phase 6:
"replace, do not patch further" -- see docs/design/phase6_batch_sizer.md and the
state-8 record in config_state.py). Everything the mutation audit above credited
them with protecting was a property OF THE WALK -- the floor stopping its
descent, the gain gate freezing it, the pin/sawtooth interaction -- and the
replacement contains no walk: those behaviours are now protected, in their new
form, by the collected `bench/test_batch_traps.py` (injection-detected) and
`bench/test_oom_ceiling_expiry.py` (rewritten for the restore rule):

    test_batch_sizer.py          drives auto_batch_throughput_opt / the knee gate,
                                 both retired keys that now hard-fail at load
    test_batch_adversarial.py    adversarial cases against the same walk
"""

collect_ignore = [
    'test_scenarios.py',
    'test_off_target.py',
    'test_crucible_feasibility.py',
    'test_batch_sizer.py',
    'test_batch_adversarial.py',
]
