"""
Read the hyperslope_aug17 ladder and answer the two questions it was run for.

    python configs/hyperslope_aug17/report.py

Q1  DOES cos RESPOND TO THE LEARNING RATE?  Every arm pins its rate (envelope
    identically 1.0, lr_servo_managed empty), so the rate is an independent
    variable and mean cos is the response. The regression across arms of
    mean(cos) on log(lr) is therefore a real slope, not the circular one you get
    within a servo-driven run where the rate is a function of cos.

    Read: `slope` and whether the per-arm means separate by more than their
    standard errors. A slope indistinguishable from zero means the statistic
    carries no rate information on this route and no beta, setpoint or leak can
    make it a controller.

Q2  IS THE DETONATION DRIVEN BY THE RATE, OR BY THE STEP?  In the shakeout runs
    the envelope was a deterministic function of the step, so "LR at onset" and
    "step at onset" were the same variable and could not be separated. Here the
    rate is constant per arm. If onset step falls as the rate rises, it is
    rate-driven. If every arm dies near the same step regardless of rate, it is
    not -- and `hl28` vs `lr8e5` (same rate, half_life_visits 28 vs 7) says
    whether the conditional Z setting is the actual driver.

    Also read `lead`: the ratio of conditioner+flow grad-norm growth to policy
    grad-norm growth over the 80 steps before onset. Above 1 means the blowup
    starts on the Z side, which is where three of four shakeout failures started.

No verdicts are printed. The report ends where judgement begins.
"""

import glob
import json
import math
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
Z_SUB = ("conditions_embedding_model", "flow_model")
P_SUB = ("forward_policy", "backward_policy", "s_model", "t_model")


def history(path):
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    out = []
    while True:
        try:
            blob = ds.scan_data()
        except Exception:
            break
        if blob is None:
            break
        rec = pb.Record()
        rec.ParseFromString(blob)
        if rec.WhichOneof("record_type") == "history":
            row = {}
            for item in rec.history.item:
                key = item.key or ".".join(item.nested_key)
                try:
                    row[key] = json.loads(item.value_json)
                except Exception:
                    pass
            out.append(row)
    return out


def run_name(path):
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    while True:
        blob = ds.scan_data()
        if blob is None:
            return None
        rec = pb.Record()
        rec.ParseFromString(blob)
        if rec.WhichOneof("record_type") == "run":
            return rec.run.display_name


def find_runs():
    """arm label -> newest wandb dir whose run name is hyslope_<arm>."""
    want = {}
    for line in (HERE / "INDEX.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            want[f"hyslope_{line.split(chr(9))[0]}"] = line.split(chr(9))[0]
    found = {}
    for d in sorted(glob.glob(str(ROOT / "wandb" / "run-*")), key=os.path.getmtime):
        f = glob.glob(os.path.join(d, "*.wandb"))
        if not f:
            continue
        try:
            nm = run_name(f[0])
        except Exception:
            continue
        for full, arm in want.items():
            if nm and nm.endswith(full):
                found[arm] = f[0]
    return found


def col(rows, key):
    return np.array([r.get(key, np.nan) if isinstance(r.get(key), (int, float))
                     else np.nan for r in rows], float)


def analyse(path):
    rows = history(path)
    step = col(rows, "_step")
    cos = col(rows, "lr_ctrl/hyper_cos")
    n = col(rows, "lr_ctrl/hyper_n")
    hg = col(rows, "lr_ctrl/hypergrads")
    lr = col(rows, "lr_fused")
    if np.all(np.isnan(lr)):
        lr = col(rows, "lr_fwd")

    # LIVE readings only. hyper_cos is drained per reporting period, so an absent
    # value means the sensor did not fire -- which is the point of the drain.
    live = ~np.isnan(cos)
    if not np.all(np.isnan(n)):
        live &= np.nan_to_num(n) > 0
    c = cos[live]

    # onset of the non-finite regime, from the counter rather than the log text
    nf = col(rows, "gradnorm/nonfinite_steps")
    onset = None
    if not np.all(np.isnan(nf)):
        hit = np.where(np.nan_to_num(nf) > 0)[0]
        if hit.size:
            onset = float(step[hit[0]])

    lead = float("nan")
    if onset is not None:
        i1 = int(np.nanargmin(np.abs(step - (onset - 20))))
        i0 = int(np.nanargmin(np.abs(step - (onset - 100))))
        def growth(names):
            g = []
            for s in names:
                v = col(rows, "gradnorm/" + s)
                a, b = v[i0], v[i1]
                if a and np.isfinite(a) and np.isfinite(b) and a > 0:
                    g.append(b / a)
            return float(np.median(g)) if g else float("nan")
        gz, gp = growth(Z_SUB), growth(P_SUB)
        if gp and np.isfinite(gz) and np.isfinite(gp) and gp > 0:
            lead = gz / gp

    return {
        "lr": float(np.nanmedian(lr)) if not np.all(np.isnan(lr)) else float("nan"),
        "steps": float(np.nanmax(step)) if step.size else 0.0,
        "n_cos": int(c.size),
        "cos_mean": float(c.mean()) if c.size else float("nan"),
        "cos_sd": float(c.std(ddof=1)) if c.size > 1 else float("nan"),
        "cos_se": float(c.std(ddof=1) / math.sqrt(c.size)) if c.size > 1 else float("nan"),
        "firings": float(np.nanmax(hg)) if not np.all(np.isnan(hg)) else float("nan"),
        "onset": onset,
        "lead": lead,
    }


def main():
    found = find_runs()
    if not found:
        raise SystemExit("no hyslope_* runs found under wandb/ -- has the ladder run?")

    order = [l.split(chr(9))[0] for l in
             (HERE / "INDEX.tsv").read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
    res = {a: analyse(found[a]) for a in order if a in found}

    print(f"\n{'arm':<9}{'lr':>10}{'steps':>7}{'firings':>9}{'n_cos':>7}"
          f"{'cos mean':>11}{'+-se':>8}{'cos sd':>8}{'nf onset':>10}{'Z lead':>8}")
    for arm, r in res.items():
        onset = f"{r['onset']:.0f}" if r["onset"] is not None else "-"
        lead = f"{r['lead']:.1f}x" if np.isfinite(r["lead"]) else "-"
        print(f"{arm:<9}{r['lr']:>10.3g}{r['steps']:>7.0f}{r['firings']:>9.0f}{r['n_cos']:>7}"
              f"{r['cos_mean']:>+11.4f}{r['cos_se']:>8.4f}{r['cos_sd']:>8.4f}"
              f"{onset:>10}{lead:>8}")

    # Q1 -- the slope, over the LR ladder only (the hl* arms share a rate with a
    # ladder rung), and ONLY over arms that stayed healthy.
    #
    # A DETONATED ARM MUST NOT ENTER THIS FIT. Its cos is measured inside the
    # blowup, not at its nominal rate: lr5e4 died at step 560 and contributed 6
    # readings averaging +0.1596 (sd 0.213) against 0.03-0.07 for every healthy
    # arm. Including it alone flips the fitted slope from -0.006 to +0.020 and
    # inverts the conclusion. The healthy range on this route is only 5e-6 to
    # 8e-5, so that is the span over which this statistic can be measured at all
    # -- which is itself a finding, and a limit on what the ladder can say.
    ladder = [(r["lr"], r["cos_mean"], r["cos_se"]) for a, r in res.items()
              if not a.startswith("hl") and r["n_cos"] > 5
              and np.isfinite(r["cos_mean"]) and r["onset"] is None]
    dropped = [a for a, r in res.items()
               if not a.startswith("hl") and r["onset"] is not None]
    if dropped:
        print(f"\n    excluded from the slope (detonated, cos measured inside the "
              f"blowup): {', '.join(dropped)}")
    if len(ladder) >= 3:
        x = np.log(np.array([p[0] for p in ladder]))
        y = np.array([p[1] for p in ladder])
        se = np.array([p[2] for p in ladder])
        slope, icpt = np.polyfit(x, y, 1)
        resid = y - (slope * x + icpt)
        dof = max(len(x) - 2, 1)
        s_err = math.sqrt(float((resid ** 2).sum()) / dof / float(((x - x.mean()) ** 2).sum()))
        print(f"\nQ1  d(cos)/d(log lr) = {slope:+.4f} +- {s_err:.4f} per e-fold "
              f"(n={len(x)} arms, span {x.max()-x.min():.2f} nats)")
        print(f"    t = {slope/s_err if s_err else float('nan'):+.2f}    "
              f"per-arm SEs {se.min():.4f}-{se.max():.4f}")
        print(f"    cos would cross zero at lr = {math.exp(-icpt/slope):.3g}"
              if slope else "    slope is zero: no crossing")
    else:
        print("\nQ1  too few arms with live cos readings to fit a slope")

    # Q2 -- onset vs rate, and the half_life pair
    print("\nQ2  onset step by rate:")
    for arm, r in res.items():
        if arm == "hl28":
            continue
        print(f"      lr {r['lr']:>9.3g}  onset {r['onset'] if r['onset'] is not None else 'none':>8}")
    if "hl28" in res and "lr8e5" in res:
        print("\n    the half_life pair (same rate 8e-5, one key apart):")
        for arm, label in (("lr8e5", "half_life 7 "), ("hl28", "half_life 28")):
            r = res[arm]
            onset = f"{r['onset']:.0f}" if r["onset"] is not None else "none"
            lead = f"{r['lead']:.1f}x" if np.isfinite(r["lead"]) else "n/a"
            print(f"      {label} ({arm:<6}) onset {onset:>6}   ran {r['steps']:.0f} steps"
                  f"   Z lead {lead}   cos {r['cos_mean']:+.4f}")


if __name__ == "__main__":
    main()
