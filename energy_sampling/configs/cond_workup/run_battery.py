"""
cond_workup battery runner -- bake + train + live-monitor + cancel-at-phase-5,
one pair at a time, sequentially (single local GPU).

For each experiment i:
  1. python data_processing/generate_toy_prior.py configs/cond_workup/{i}_toy.yaml
     (bakes {tag}_prior.pt / {tag}_conditions.pt)
  2. python train.py --config configs/cond_workup/{i}_train.yaml   (background)
  3. poll THIS run's live .wandb datastore for phase / _step / bwd/relative_under
  4. cancel when the stop policy fires, then move to i+1.

Stop policy (per run), all read live from the datastore -- no train.py edits:
  * terminal = (phase == 5)  [stages: 1 train_prior .. 5 terminal]
  * once in terminal, stop on   bwd/relative_under < 2   OR   3000 terminal steps
  * absolute backstop: _step >= 8000  (stalled/slow) -> cancel
  * process death before a stop -> recorded as died_*

Nothing self-terminates cleanly: terminal has no exit gate, so the runner is
what ends every run. Outcomes + wandb run ids are written to run_logs/status.json
after every run so progress is inspectable mid-battery.

Usage (from energy_sampling/, csd_mxt_gfn venv -- though it hard-codes the venv
python for the child procs regardless of what launches it):
    python configs/cond_workup/run_battery.py            # all runs in experiment_log.yaml
    python configs/cond_workup/run_battery.py 0 3 6      # only these indices
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

BATTERY_DIR = Path(__file__).resolve().parent
ENERGY_DIR = BATTERY_DIR.parents[1]            # .../energy_sampling
WANDB_DIR = ENERGY_DIR / "wandb"
LOG_DIR = BATTERY_DIR / "run_logs"

PY = r"C:\Users\mikem\venvs\csd_mxt_gfn\Scripts\python.exe"
PYTHONPATH = (r"C:\Users\mikem\Projects\mxt_gfn\mxtaltools;"
              r"C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion")

# --- stop policy knobs ---
TERMINAL_PHASE = 5
REL_UNDER_STOP = 2.0
TERMINAL_STEP_BUDGET = 3000
TOTAL_STEP_CAP = 12000   # stall backstop only; the real stops are rel_under<2 / 3k terminal steps.
                         # raised from 8000 so a late-entering hard run (rich multi-modal field)
                         # isn't truncated mid-terminal-budget
# --- runner mechanics ---
POLL_SECONDS = 30
RUNDIR_WAIT_S = 240        # how long to wait for wandb to create this run's dir
GPU_COOLDOWN_S = 15        # let VRAM free between runs

from wandb.proto import wandb_internal_pb2  # noqa: E402
from wandb.sdk.internal import datastore    # noqa: E402


def child_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    return env


def scan_datastore(wandb_path):
    """Return (latest_step, latest_phase, latest_rel_under, terminal_entry_step).
    Any may be None if not yet logged. Robust to the partial final record on a
    live run (see local-wandb-reading recipe)."""
    ds = datastore.DataStore()
    try:
        ds.open_for_scan(str(wandb_path))
    except Exception:
        return None, None, None, None
    latest_step = latest_phase = latest_rel = entry = None
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        try:
            rec = wandb_internal_pb2.Record()
            rec.ParseFromString(data)
        except Exception:
            break
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in rec.history.item:
            key = item.key or ".".join(item.nested_key)
            try:
                row[key] = json.loads(item.value_json)
            except Exception:
                row[key] = item.value_json
        step = row.get("_step")
        if step is None:
            continue
        latest_step = step
        if "phase" in row:
            latest_phase = row["phase"]
            if row["phase"] == TERMINAL_PHASE and entry is None:
                entry = step
        if "bwd/relative_under" in row:
            latest_rel = row["bwd/relative_under"]
    return latest_step, latest_phase, latest_rel, entry


def newest_new_rundir(before):
    """The run-* dir that appeared since `before` (a set of names)."""
    now = {p.name: p for p in WANDB_DIR.glob("run-*") if p.is_dir()}
    fresh = [now[n] for n in now if n not in before]
    if not fresh:
        return None
    return max(fresh, key=lambda p: p.stat().st_mtime)


def wandb_file(rundir):
    hits = list(rundir.glob("*.wandb"))
    return hits[0] if hits else None


def kill_tree(proc):
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                   capture_output=True)
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def decide_stop(step, phase, rel, entry):
    """Return a stop-reason string, or None to keep running."""
    in_terminal = phase == TERMINAL_PHASE and entry is not None
    if in_terminal:
        if rel is not None and rel < REL_UNDER_STOP:
            return "terminal_converged"
        if step is not None and (step - entry) >= TERMINAL_STEP_BUDGET:
            return "terminal_budget"
    if step is not None and step >= TOTAL_STEP_CAP:
        return "cap_in_terminal" if in_terminal else "cap_pre_terminal"
    return None


def run_one(ind, status, statuslog):
    tag = f"cw{ind:02d}"
    toy_cfg = f"configs/cond_workup/{ind}_toy.yaml"
    train_cfg = f"configs/cond_workup/{ind}_train.yaml"
    rec = {"index": ind, "tag": tag, "outcome": "started"}
    status.append(rec)
    write_status(status, statuslog)

    # 1. bake artifacts (blocking)
    with (LOG_DIR / f"{tag}_bake.log").open("w") as blog:
        bake = subprocess.run([PY, "data_processing/generate_toy_prior.py", toy_cfg],
                              cwd=ENERGY_DIR, env=child_env(),
                              stdout=blog, stderr=subprocess.STDOUT)
    if bake.returncode != 0:
        rec.update(outcome="bake_failed")
        write_status(status, statuslog)
        return

    # 2. launch training (background)
    before = {p.name for p in WANDB_DIR.glob("run-*") if p.is_dir()}
    trainlog = (LOG_DIR / f"{tag}_train.log").open("w")
    proc = subprocess.Popen([PY, "train.py", "--config", train_cfg],
                            cwd=ENERGY_DIR, env=child_env(),
                            stdout=trainlog, stderr=subprocess.STDOUT)

    # 3. locate this run's wandb dir
    rundir = None
    t0 = time.time()
    while time.time() - t0 < RUNDIR_WAIT_S:
        rundir = newest_new_rundir(before)
        if rundir is not None and wandb_file(rundir):
            break
        if proc.poll() is not None:
            rec.update(outcome="died_at_launch")
            trainlog.close()
            write_status(status, statuslog)
            return
        time.sleep(3)
    if rundir is None:
        kill_tree(proc)
        rec.update(outcome="no_wandb_dir")
        trainlog.close()
        write_status(status, statuslog)
        return
    rec["wandb_run"] = rundir.name
    write_status(status, statuslog)
    wf = wandb_file(rundir)

    # 4. poll -> stop
    while True:
        if proc.poll() is not None:  # exited on its own (crash or ran out)
            step, phase, rel, entry = scan_datastore(wf)
            reached = phase == TERMINAL_PHASE or entry is not None
            rec.update(outcome=("died_in_terminal" if reached else "died_pre_terminal"),
                       final_step=step, phase=phase, rel_under=rel,
                       terminal_entry=entry)
            trainlog.close()
            write_status(status, statuslog)
            return
        step, phase, rel, entry = scan_datastore(wf)
        reason = decide_stop(step, phase, rel, entry)
        rec.update(final_step=step, phase=phase, rel_under=rel, terminal_entry=entry,
                   terminal_steps=(step - entry) if (step is not None and entry is not None) else None)
        write_status(status, statuslog)
        if reason is not None:
            kill_tree(proc)
            rec.update(outcome=reason)
            trainlog.close()
            write_status(status, statuslog)
            return
        time.sleep(POLL_SECONDS)


def write_status(status, statuslog):
    statuslog.write_text(json.dumps(status, indent=2))


def main():
    LOG_DIR.mkdir(exist_ok=True)
    statuslog = LOG_DIR / "status.json"
    log = yaml.safe_load((BATTERY_DIR / "experiment_log.yaml").read_text())
    all_idx = [e["index"] for e in log]
    want = [int(a) for a in sys.argv[1:]] or all_idx

    status = []
    for ind in want:
        print(f"=== cw{ind:02d}: bake + train + monitor ===", flush=True)
        try:
            run_one(ind, status, statuslog)
        except Exception as e:
            status[-1].update(outcome=f"runner_error: {e}") if status else None
            write_status(status, statuslog)
        r = status[-1]
        print(f"=== cw{ind:02d} -> {r['outcome']} "
              f"(step {r.get('final_step')}, phase {r.get('phase')}, "
              f"rel_under {r.get('rel_under')}) ===", flush=True)
        time.sleep(GPU_COOLDOWN_S)

    print("battery complete. summary:", flush=True)
    for r in status:
        print(f"  {r['tag']}: {r['outcome']:20s} "
              f"step={r.get('final_step')} termsteps={r.get('terminal_steps')} "
              f"rel_under={r.get('rel_under')}", flush=True)


if __name__ == "__main__":
    main()
