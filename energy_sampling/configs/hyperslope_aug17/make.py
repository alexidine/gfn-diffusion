"""
hyperslope_aug17 -- ONE measurement: does `cos` respond to the learning rate on
the QM9 conditional route?

    python configs/hyperslope_aug17/make.py

WHY THIS EXISTS. `LRController.on_hypergradient` is a pure integrator in log
space, `peak_scale *= exp(beta*(cos - cos_target))`, with no leak and no
restoring term. Whether that is FIXABLE or merely CONTAINABLE turns on one
quantity nobody has measured on this route:

    d(cos) / d(log lr)

If it is reliably negative, the loop has a real fixed point, the observed
excursions are an aiming error, and a setpoint plus a leak is a genuine fix. If
it is indistinguishable from zero, `cos` is not an error signal at all, the loop
is open, and no value of `beta` helps -- the rate wants a schedule, not a servo.

The local evidence that motivates it (7 runs, 2026-08-17, all `var_conditioning`
off WARM_qm9_mle3k):

  * `peak_scale` never ran away UPWARD -- max 5.14 across every run, against a
    ceiling of 2000. The failure is not the integrator hitting its rail.
  * every non-finite onset sat at lr_fwd 8.3e-5 - 1.5e-4 (4 of 7 runs), and
    `adaptive_lr.seed_lr` is 1.25e-4 -- inside that band.
  * the one arm carrying `cos_target: 0.1` (`qm9cond_costgt`) walked
    monotonically to the peak_scale floor and finished at `min_lr`. Its live cos
    averaged +0.0117, so the error was about -0.088 every firing. Predicted rail
    at step 2088 from ln(0.01)/(0.05*-0.0883)/0.5-firings-per-step; observed
    2320. The setpoint sits near the p90 of the achievable distribution.
  * live cos across arms: mean -0.043 .. +0.052, sd 0.062 .. 0.168.

THE ARMS DIFFER IN ONE KEY. Every arm is `configs/shakeout_aug16/qm9_cond.yaml`
with the four policy LRs pinned to one float and NOTHING else changed but the
run name. Same seed (12345) on every arm, so the LR ladder is the only axis.

WHY THE RATES ARE EXPLICIT FLOATS RATHER THAN `auto`. That makes
`lr_servo_managed` empty (utils.py::resolve_derived_config only records keys
written `auto`), which is the control arm `LRController._managed_keys` documents:
the sensor still fires, still logs, and still moves `peak_scale`, but
`_apply_lrs` never applies `peak_scale` to a rate. So the LR is a fixed
INDEPENDENT variable and `cos` is the response -- which is the only way to get a
slope. Letting the servo act would make the LR a function of cos and the
regression would be circular.

`config_invariants.auto_lr_requires_an_adaptive_sensor` returns [] when no key is
`auto` ("every rate explicitly pinned: nothing to own"), so this is legal.

WHY lr_warmup_ratio: 1. The envelope is
`(1/lr_warmup_ratio)**(1 - elapsed/warmup_steps)`, so ratio 1 makes it
identically 1.0 and the live rate equals the pinned float for the whole run. At
the shipped ratio of 10 the rate would sweep 10x DURING each arm and there would
be no fixed LR to regress against. It also removes the confound that the ramp
raises the rate at ln(10)/1000 = 2.3e-3 per step while the sensor's own
authority here is beta*|cos|*0.5 firings ~ 7.5e-4 per step -- the ramp out-guns
the sensor about 3x, which is why `_maybe_freeze_envelope` is load-bearing.

WHY cos_target: 0.0 ON EVERY ARM. The reported `lr_ctrl/hyper_cos` is the raw
cosine, not the error, so a nonzero target makes the published statistic and the
actuator describe different quantities. With target 0, `log(peak_scale)` is
exactly `beta * sum(cos)` and the actuator becomes a second, independent readout
of the same statistic -- a consistency check on the sensor channel for free.

WHY checkpoint_read_only. train.py saves 'running' every 50 steps and 'final'
outside the loop, keyed on run_name, and `checkpoints/` is not in git. Every arm
loads WARM_qm9_mle3k.pt; a writing arm could overwrite the very warm start the
next arm needs. Read-only leaves loading active and suppresses every write
(checkpointing.py:124). This is not optional.

THE LADDER is 5e-6 .. 5e-4, factor ~2.5, six arms: a 100x span (4.6 nats) so the
regression has lever arm, bracketing the 8e-5 - 1.5e-4 failure band with two arms
clearly below it and two clearly above. The top arms are EXPECTED to go
non-finite around step 900-1050; that is the threshold measurement, and their
truncation is why the ladder extends two rungs below the interesting region
rather than centring on it.

Budget: 0.49 s/step wall (measured, phorfy0f/gl6vigfe, includes eval), 2000
steps -> ~16 min/arm, ~100 min for the ladder.
"""

import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "shakeout_aug16" / "qm9_cond.yaml"

STEPS = 2000
SEED = 12345

# label -> pinned policy LR. Factor ~2.5 per rung.
ARMS = {
    "lr5e6":   5.0e-6,
    "lr12e6":  1.25e-5,
    "lr3e5":   3.0e-5,
    "lr8e5":   8.0e-5,
    "lr2e4":   2.0e-4,
    "lr5e4":   5.0e-4,
}

# ---------------------------------------------------------------- second axis
# THE LR LADDER MAY BE MEASURING THE WRONG THING, and this arm is how that gets
# found out rather than assumed.
#
# Per-submodel grad norms over the 80 steps before each shakeout detonation
# (gradnorm/*, 2026-08-17) say the blowup does not START in the policy:
#
#   run          conditions_emb  flow_model   forward_policy  backward_policy
#   liveservo           5513x       6011x            59x            100x
#   nolvlgap             508x       2917x             7x             34x
#   ctrl2               1002x       2482x             6x             38x
#   beta002              114x        207x          1506x          26851x
#
# Three of four are led by the CONDITIONER and the Z head by 60-400x; only
# beta002 -- the one with the LOWEST live rate, envelope frozen at 0.2512 -- is
# policy-led. And `config_invariants.conditional_z_settings_are_conditional`
# already flags this base: `condition_log_z.half_life_visits` is 7.0, the
# UNCONDITIONAL default, where the conditional route wants 28.0. Every
# conditional battery that completed carried 28; the documented detonation shape
# for 7 is a conditioner/Z blowup.
#
# So the "LR threshold at ~1e-4" read off the shakeout runs is CONFOUNDED: with
# lr_warmup_ratio 10 and warmup_steps 1000 the envelope is a deterministic
# function of the step, so "the LR at onset" and "the step at onset" are the same
# variable, and all three Z-led onsets sit at 902-1052 -- i.e. right where the
# ramp completes. The ladder above breaks that confound by pinning the rate
# (ratio 1, so the envelope is 1.0 throughout): if the detonation step moves with
# the rate, it is LR-driven; if every arm dies at the same step regardless, it is
# not, and no seed_lr change would have helped.
#
# `hl28` is the direct test of the alternative. It differs from `lr8e5` in ONE
# key -- 8.0e-5 is where three of four shakeout runs died, so the comparison is
# made where the effect exists.
EXTRA = {
    "hl28": {"lr": 8.0e-5, "half_life_visits": 28.0},
    # hl28 WAS PLACED AT THE WRONG RATE, and hl28b is the correction rather than a
    # second question. 8.0e-5 was chosen because three of four SHAKEOUT runs died
    # near it -- but those carried the warmup ramp, and at a CONSTANT 8.0e-5 the
    # ladder's own lr8e5 survives all 2000 steps with zero non-finite. A pair in
    # which neither member detonates cannot discriminate.
    #
    # 2.0e-4 is where the twin does die (lr2e4, non-finite from step 1558, and
    # lr5e4 dies ~50 steps after the conditioner and flow head switch on at the
    # var_conditioning transition, growing 187x and 70x while forward_policy goes
    # 1.1x). So the test is: does the conditional half_life move that threshold?
    "hl28b": {"lr": 2.0e-4, "half_life_visits": 28.0},
}

_LR_KEYS = ("lr_policy", "lr_back", "lr_replay", "lr_fused")


def _iter_stages(node):
    """Every protocol stage dict in the config, wherever it lives."""
    if isinstance(node, dict):
        if "name" in node and "train_mode" in node:
            yield node
        for value in node.values():
            yield from _iter_stages(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_stages(value)


def build(label, lr, base):
    cfg = yaml.safe_load(yaml.safe_dump(base))      # deep copy

    for key in _LR_KEYS:
        cfg[key] = float(lr)

    cfg["lr_warmup_ratio"] = 1          # envelope == 1.0 for the whole run
    cfg["checkpoint_read_only"] = True  # NOT optional -- see the module docstring
    cfg["epochs"] = STEPS
    cfg["seed"] = SEED
    cfg["tag"] = "hyslope"
    cfg["run_name"] = f"hyslope_{label}"

    # Neutralise the setpoint everywhere it appears, so hyper_cos and
    # log(peak_scale) measure the same thing on every arm.
    sensors = 0
    for stage in _iter_stages(cfg):
        sensor = stage.get("lr_sensor")
        if isinstance(sensor, dict) and sensor.get("kind") == "hyper":
            sensor["cos_target"] = 0.0
            sensors += 1
    if not sensors:
        raise SystemExit("no hyper sensor found in the base config -- the base "
                         "moved and this battery would measure nothing")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="re-parse what was written and assert the arms differ "
                         "ONLY in the LR and the run name")
    args = ap.parse_args()

    if not BASE.exists():
        raise SystemExit(f"base config missing: {BASE}")
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))

    rows = []
    for label, lr in ARMS.items():
        cfg = build(label, lr, base)
        out = HERE / f"{label}.yaml"
        with open(out, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
        rows.append((label, lr, 7.0, out.name))
        print(f"wrote {out.relative_to(HERE.parent.parent)}   lr={lr:g}")

    for label, spec in EXTRA.items():
        cfg = build(label, spec["lr"], base)
        cfg.setdefault("condition_log_z", {})["half_life_visits"] = spec["half_life_visits"]
        out = HERE / f"{label}.yaml"
        with open(out, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
        rows.append((label, spec["lr"], spec["half_life_visits"], out.name))
        print(f"wrote {out.relative_to(HERE.parent.parent)}   lr={spec['lr']:g} "
              f"half_life={spec['half_life_visits']:g}")

    index = HERE / "INDEX.tsv"
    with open(index, "w", encoding="utf-8") as fh:
        fh.write("arm\tlr\thalf_life_visits\tsteps\tseed\tconfig\n")
        for label, lr, hl, name in rows:
            fh.write(f"{label}\t{lr:g}\t{hl:g}\t{STEPS}\t{SEED}\t{name}\n")
    print(f"wrote {index.relative_to(HERE.parent.parent)}")

    if args.check:
        # THE ARMS MUST DIFFER IN ONE AXIS. A generator that silently varied a
        # second key would produce a slope that is not a slope in the LR, and
        # nothing downstream could tell.
        loaded = {label: yaml.safe_load((HERE / name).read_text(encoding="utf-8"))
                  for label, _, _, name in rows}
        ref_label = next(iter(ARMS))
        ref = loaded[ref_label]
        allowed = set(_LR_KEYS) | {"run_name"}
        for label in ARMS:
            if label == ref_label:
                continue
            cfg = loaded[label]
            differing = {k for k in set(ref) | set(cfg) if ref.get(k) != cfg.get(k)}
            extra = differing - allowed
            if extra:
                raise SystemExit(f"{label} differs from {ref_label} in {sorted(extra)} "
                                 f"-- the ladder varies more than the LR")
        # THE SECOND AXIS MUST ALSO BE ONE KEY. `hl28` is compared against the
        # ladder rung at its own rate, not against the ladder's first rung, so
        # the only difference that may survive is condition_log_z.
        for label in EXTRA:
            cfg = loaded[label]
            twin = next(l for l, lr in ARMS.items() if lr == EXTRA[label]["lr"])
            differing = {k for k in set(loaded[twin]) | set(cfg)
                         if loaded[twin].get(k) != cfg.get(k)}
            extra = differing - {"run_name", "condition_log_z"}
            if extra:
                raise SystemExit(f"{label} differs from {twin} in {sorted(extra)} "
                                 f"-- the half_life axis varies more than half_life")
            a = dict(loaded[twin].get("condition_log_z") or {})
            b = dict(cfg.get("condition_log_z") or {})
            moved = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
            if moved != {"half_life_visits"}:
                raise SystemExit(f"{label}: condition_log_z differs in {sorted(moved)}, "
                                 f"expected only half_life_visits")
            print(f"check OK: {label} vs {twin} differ ONLY in half_life_visits "
                  f"({a.get('half_life_visits')} -> {b.get('half_life_visits')})")
        for label, cfg in loaded.items():
            assert len({cfg[k] for k in _LR_KEYS}) == 1, f"{label}: LR keys disagree"
            assert cfg["checkpoint_read_only"] is True, f"{label}: writes not suppressed"
            assert cfg["lr_warmup_ratio"] == 1, f"{label}: envelope would ramp"
            for stage in _iter_stages(cfg):
                s = stage.get("lr_sensor")
                if isinstance(s, dict) and s.get("kind") == "hyper":
                    assert s.get("cos_target") == 0.0, f"{label}: setpoint survived"
        print(f"check OK: {len(loaded)} arms differ only in {sorted(allowed)}")


if __name__ == "__main__":
    main()
