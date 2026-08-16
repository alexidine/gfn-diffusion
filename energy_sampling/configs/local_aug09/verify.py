"""
Verify that the new subsystems actually ENGAGED in a run -- not that they were
configured. Every check below is a code-path tell, chosen because the failure
mode it catches is SILENT in the ordinary metrics.

    python configs/local_aug09/verify.py [run_name_substring]

Reads the local wandb datastore directly (no network).

Why each tell, and what it catches:

  PROBE          lrprobe/* exist and second_diff_rel sits far above
                 step_probe.py's 1e-6 SECOND_DIFF_REL_FLOOR. At the floor the
                 probe is resolving float32 rounding, not curvature, and alpha*
                 is meaningless while still LOOKING like a number.
                 aborted_rate > 0 means the probe raised mid-ray (OOM in its
                 extra forward); params are force-restored now, but the sensor
                 stopped reading.

  RATIO CTRL     protocol/rt_* are emitted ONLY by kind: ratio. Their absence
                 with a balance block present means some other kind is running.

  PRIORITISED    replay/is_* exist AND is_ess_frac is not exactly 1.000.
  DRAW           This is the one that matters. prioritised_weights() has FOUR
                 silent fallbacks (empty buffer, all-NaN ema_logw, no eligible
                 rows, non-finite weight sum) and current_log_z() returns None
                 on any conditional model -- every one of them returns a uniform
                 draw with w == 1 and is otherwise indistinguishable from a
                 working kappa=0. ESS exactly 1.000 is the only tell.

  UNIFORM        replay_buffer_absorbed_frac and stalled_frac must be ~0: under
  INTAKE         uniform_intake the floor and stalled eviction causes are forced
                 off (train.py:5067-5072). Non-zero means the OLD path ran.

  DISPLACEMENT   The residual-conditioned purge at train.py:5151-5157 is NOT
  PURGE          gated on uniform_intake and lands in NO cohort bucket, so it is
                 invisible per-metric. Proxy: it can only fire when admits
                 exceed headroom, so max_size >= 2 x churn x tau keeps it
                 dormant. Reported as a config check plus the eviction
                 accounting gap.

  OLD LR         lr_fused bitwise flat, lr_ctrl/cut_factor == 1, scale == 1.
                 Any movement means the envelope or a tripwire actuated.

  SERVO          protocol/bs_* present, and bs_log_boost logged. A servo reading
                 fine with no authority looks identical to one correctly
                 holding, which is why the actuator is checked separately from
                 the sensor.
"""
import glob
import json
import math
import os
import sys

import numpy as np
import yaml
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

FLOOR = 1e-6


def load(d):
    wf = glob.glob(os.path.join(d, 'run-*.wandb'))
    if not wf:
        return None, {}
    out, step, name = {}, None, None
    try:
        ds = datastore.DataStore()
        ds.open_for_scan(wf[0])
    except Exception:
        return None, {}
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            break
        t = rec.WhichOneof('record_type')
        if t == 'run' and rec.run.display_name:
            name = rec.run.display_name
        if t != 'history':
            continue
        for it in rec.history.item:
            k = it.key or '.'.join(it.nested_key)
            try:
                v = json.loads(it.value_json)
            except Exception:
                continue
            if k == '_step':
                step = v
            if isinstance(v, (int, float)) and math.isfinite(v):
                out.setdefault(k, []).append(v)
    return name, out


def med(o, k):
    v = o.get(k)
    return float(np.median(v)) if v else float('nan')


def check(label, ok, detail):
    # numpy bools are not `is True` / `is False`, and silently reading as
    # indeterminate is exactly the failure this harness exists to prevent
    if ok is not None:
        ok = bool(ok)
    mark = 'PASS' if ok is True else ('FAIL' if ok is False else '????')
    print(f'  [{mark}] {label:<34} {detail}')
    return ok


def verify(name, o, cfg):
    print(f'\n=== {name} ===')
    results = []
    rb = (cfg.get('buffers') or {}).get('replay_buffer') or {}

    # --- probe -------------------------------------------------------------
    has = [k for k in o if k.startswith('lrprobe/')]
    sdr = med(o, 'lrprobe/second_diff_rel')
    ab = med(o, 'lrprobe/aborted_rate')
    okr = med(o, 'lrprobe/fit_ok_rate')
    results.append(check('probe emitting', bool(has), f'{len(has)} lrprobe/* keys'))
    results.append(check('probe resolving curvature', sdr > 100 * FLOOR,
                         f'second_diff_rel med={sdr:.3g} vs floor {FLOOR:g}'))
    results.append(check('probe not aborting', (ab == 0 or math.isnan(ab)),
                         f'aborted_rate={ab:.3g}, fit_ok_rate={okr:.3g}'))

    # --- ratio controller --------------------------------------------------
    rt = [k for k in o if k.startswith('protocol/rt_')]
    results.append(check('ratio controller live', bool(rt),
                         f'{sorted(k.split("/")[-1] for k in rt)}'))

    # --- prioritised draw --------------------------------------------------
    ess = med(o, 'replay/is_ess_frac')
    elig = med(o, 'replay/is_elig_frac')
    wmax = med(o, 'replay/is_w_max_ratio')
    if math.isnan(ess):
        results.append(check('prioritised draw engaged', False,
                             'replay/is_ess_frac ABSENT -- draw never ran prioritised'))
    else:
        # exactly 1.000 == every silent fallback in prioritised_weights()
        degenerate = abs(ess - 1.0) < 1e-9
        results.append(check('prioritised draw engaged', not degenerate,
                             f'is_ess_frac={ess:.4f} (1.0000 == silent uniform fallback), '
                             f'is_elig_frac={elig:.3f}, w_max_ratio={wmax:.3g}'))

    # --- uniform intake ----------------------------------------------------
    ab_f = med(o, 'replay_buffer_absorbed_frac')
    st_f = med(o, 'replay_buffer_stalled_frac')
    if math.isnan(ab_f) and math.isnan(st_f):
        results.append(check('uniform intake', None, 'no eviction-cause metrics logged yet'))
    else:
        quiet = (np.nan_to_num(ab_f) < 1e-9) and (np.nan_to_num(st_f) < 1e-9)
        results.append(check('uniform intake (floor/stalled off)', quiet,
                             f'absorbed_frac={ab_f:.3g}, stalled_frac={st_f:.3g} '
                             f'(both forced to 0 under uniform_intake)'))

    # --- displacement purge ------------------------------------------------
    churn, tau = rb.get('churn_rate'), rb.get('mean_residence_steps')
    ms = rb.get('max_size')
    if churn and tau and ms:
        equil = churn * tau
        results.append(check('displacement purge dormant', ms >= 2 * equil,
                             f'max_size={ms} vs 2 x churn x tau = {2*equil} '
                             f'(equilibrium occupancy {equil})'))
    else:
        results.append(check('displacement purge dormant', None, 'sizing keys missing'))

    # --- old LR ------------------------------------------------------------
    lrf = o.get('lr_fused') or []
    cutf = o.get('lr_ctrl/cut_factor') or []
    scale = o.get('lr_ctrl/scale') or []
    flat = (len(set(lrf)) <= 1) if lrf else None
    results.append(check('lr_fused flat (no actuation)', flat,
                         f'{len(set(lrf))} distinct value(s), med={med(o,"lr_fused"):.4g}'))
    results.append(check('no LR cut fired', (all(abs(x - 1.0) < 1e-12 for x in cutf)
                                             if cutf else None),
                         f'cut_factor distinct={sorted(set(cutf))[:4]}, '
                         f'scale distinct={sorted(set(scale))[:4]}'))

    # --- memorisation sensor + servo --------------------------------------
    lam = med(o, 'replay/lambda_tau')
    results.append(check('B7d sensor reading', not math.isnan(lam),
                         f'lambda_tau={lam:.4g} (bar 0.368), '
                         f'ema_loss_mean={med(o,"replay/ema_loss_mean"):.4g}, '
                         f'birth_loss_mean={med(o,"replay/birth_loss_mean"):.4g}'))
    bs = [k for k in o if k.startswith('protocol/bs_')]
    declared = 'buffer_servo' in (cfg.get('protocol') or {}).get('stages', [{}, {}])[1]
    if declared:
        boost = med(o, 'protocol/bs_log_boost')
        results.append(check('servo live + has authority', bool(bs),
                             f'{len(bs)} bs_* keys, bs_log_boost={boost:.4g} '
                             f'(0 = holding OR no authority -- read with the sensor)'))
    else:
        results.append(check('servo not declared (v0 by design)', not bs,
                             'buffer_servo absent from the stage, as intended'))

    n_pass = sum(1 for r in results if r is True)
    n_fail = sum(1 for r in results if r is False)
    n_unk = sum(1 for r in results if r is None)
    print(f'  ---- {n_pass} pass, {n_fail} FAIL, {n_unk} indeterminate')
    return n_fail


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else 'batt0809'
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'wandb')
    fails = 0
    seen = 0
    for d in sorted(glob.glob(os.path.join(base, 'run-*'))):
        name, o = load(d)
        if not name or want not in name or not o:
            continue
        cfgp = os.path.join(os.path.dirname(__file__),
                            f"{name.replace('batt0809_', '')}.yaml")
        cfg = yaml.safe_load(open(cfgp)) if os.path.exists(cfgp) else {}
        seen += 1
        fails += verify(name, o, cfg)
    if not seen:
        print(f'no runs matching {want!r} found')
        return 1
    print(f'\n{"ALL TELLS PASS" if fails == 0 else f"{fails} FAILING TELL(S)"}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
