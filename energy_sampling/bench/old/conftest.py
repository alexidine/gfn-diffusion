"""
`bench/old/` is not uniformly old, and `norecursedirs` could not say so.

STATUS 2026-09-09: it is uniformly old NOW. Every test file here is on the ignore
list at the bottom, so the directory collects nothing. Worth keeping the account
below rather than collapsing it, because the LAST step to that state was not taken
here: the ignore list reached five entries on 2026-08-19, leaving exactly one file
collected, and that file was then deleted somewhere else entirely (see the
2026-09-09 note under the second mutation bullet).

The directory was created when `bench/` was rebuilt (2026-08-13) after an
adversarial review retired the previous generation. `pytest.ini` then excluded the
whole directory -- correct for the retired BATTERY MACHINERY, and wrong for the
three test files that drive SHIPPING code and had simply been sitting next to it.

MEASURED 2026-08-16, whole directory, project venv:

    bench/old/                          111 passed, 3 skipped     65 s

RE-MEASURED 2026-09-09, files named explicitly so the ignore list below does not
hide them:

    test_crucible_feasibility.py        DOES NOT IMPORT -- `MK_DEV_ADAPTIVE` has
                                        left bench.fake_modeller
    the other four                      20 passed, 48 failed, 3 skipped

So "nothing here is broken" has stopped being true, and so has the 111. Note
also what the ignore list below now adds up to: it names ALL FIVE test files in
the directory, so `bench/old/` contributes ZERO collected tests, and pytest.ini's
decision to recurse into it is inert either way.

The question was never "do these still run", it was "do they protect anything
the collected suite does not", and in 2026-08 that was settled by mutation:

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

  * `LRController.on_calibration -> no-op` (neutered the ray sensor's actuator
    path, back when it had one):
        bench/old/test_lr_controller.py                            12 RED
        the ENTIRE collected bench/                                 1 RED
            -- and that one is test_arms.py's generic
               `test_ray_is_distinguishable_from_null`, which says an arm went
               inert, not WHICH behaviour broke.
    Twelve named v8 behaviours -- warmup hold, asymmetric update, peak bounds, the
    permanent divergence ceiling, the servo cut from a hot start, saturated-sensor
    open loop, unresolved/inconsistent producing no move, the unmanaged-key control
    arm -- were protected here and nowhere else, on the controller that then shipped.

    NONE OF THAT HOLDS NOW (checked 2026-09-09), and the file this paragraph
    names IS GONE: `bench/old/test_lr_controller.py` was deleted 2026-08-24
    (0dfeef2, 681 lines, collateral in a commit about z-calibration timing).
    LRController v8 went with it -- the controller was rebuilt on the brute-force
    bracket, and controller.py's "WHAT IS GONE" accounts for seven of the eight
    families above as DELETED MECHANISM: the warmup envelope and its freeze rules,
    the divergence LR cut and its ceiling, the alpha target, the warm restart, and
    `peak_scale`, a v8 state key the loader now discards rather than reinterprets.
    The retired servo law itself was not lost, it was MOVED: `bench/arms.py::RayRay`
    transcribes ETA_UP / ETA_DOWN / ALPHA_TARGET out of the controller, kept
    runnable so the bracket can be compared against the old rule instead of merely
    declared better than it.

    The eighth family, the unmanaged-key control arm, SURVIVES -- the actuator and
    its managed-key rule were never in question -- and it IS pinned, twice over:
    tests/lr/test_lr_absolute_cap.py drives a `managed=()` controller (and pins the
    min_lr floor, the flow-group pin and the max_lr rail besides), and
    tests/lr/test_lr_bracket_driver.py::test_no_managed_rate_means_no_trials_are_run.
    The other surviving v8 piece, the hard tripwire, is pinned by
    test_lr_bracket_driver.py and test_lr_bracket_trial_guards.py via
    `lr_bracket_probe.HardFailureBars`.

    RE-RUN 2026-09-09, the same mutation against the current tree: 0 RED.
    (`tests/lr` + `bench`, 384 passed with and without it; bench/test_fidelity.py's
    3 failures are pre-existing and identical unmutated.) Read that carefully --
    it is NOT an unguarded controller. `on_calibration` now records and moves
    nothing, so the mutation has no actuator behaviour left to break. What the
    zero does expose is small, and is a real gap rather than a tidiness item:
    nothing anywhere asserts on the `_calibrations` counter or the
    `lr_ctrl/calibrations` metric it feeds, and `_last_ray` is written at
    controller.py:1101 and read nowhere in the repo.

    What pins the LIVE controller is tests/lr/ (238 passed) and the collected
    bench/ (155 tests). The ray sensor's remaining contract -- that it is a
    diagnostic and reaches no rate -- is pinned by bench/test_ray_calibration.py
    and tests/lr/test_ray_dual_score.py, including
    `test_the_diagnostic_is_off_by_default`.

Hence this file instead of a directory-wide exclusion: collect what drives live
symbols, ignore what tests the machinery the review condemned. That was the rule;
it simply has nothing left to select, since the files that drove live symbols have
all since been deleted or retired (see the dated notes above and below).

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
