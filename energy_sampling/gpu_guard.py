"""
Pre-flight for GPU work: will this job FIT alongside whatever is already there?

WHY THIS EXISTS. Three BSODs in 24 hours (2026-08-11/12), every one from a second
training run starting while another held the card. The driver does not politely OOM,
it takes the machine down, so there is nothing to catch afterwards -- a launch has to
refuse BEFORE it initialises CUDA.

THE POLICY, WHICH IS DELIBERATELY PESSIMISTIC
    If anything is on the GPU, assume this job must NOT be added.
Sharing a card is something you do ON PURPOSE, with configs calibrated for it
(smaller batch, smaller model, a `cuda_memory_fraction` that actually divides the
card). It is never something to discover by launching and hoping. So co-tenancy is
opt-in and must be declared, and the declaration is checked for coherence rather
than taken at face value.

HOW THE PROJECTION IS MADE, AND WHAT IT REFUSES TO GUESS
Per-process GPU memory is unavailable on this box -- Windows WDDM makes
`nvidia-smi --query-compute-apps` report `[N/A]` for used_memory -- so a job's
footprint cannot be attributed after the fact. Two honest sources remain, in
preference order:

  1. A MEASURED peak for this config's EXACT signature, recorded by a previous run of
     it (`record_peak`, called from train.py). Used RAW for "is there room now" -- the
     measurement is the evidence -- and with a fragmentation margin for co-tenancy
     budgeting. Deliberately NOT extrapolated to a different batch size: a peak is
     largely batch-independent (parameters, cuda-resident buffers), so scaling it by a
     batch ratio over-estimates wildly and once refused 42 configs on an idle card.
  2. Failing that, the config's OWN DECLARED CEILING: `cuda_memory_fraction` x total.
     That is what torch is permitted to grow into, and the allocator does grow into
     it and cache. At the 0.9 every config here carries, that is 14.7 GB of a 16.3 GB
     card -- which is precisely why two such runs cannot coexist.

There is deliberately NO parametric model of activation memory from batch/T/width.
Inventing coefficients would produce a confident number with nothing behind it, and
an under-estimate here costs the machine. Unknown resolves to the declared ceiling,
which is conservative, or to a refusal.

USE
    from gpu_guard import require_free_gpu
    require_free_gpu()                       # config read from --config in argv
    require_free_gpu(cotenants=3)            # I have CALIBRATED for 3-way sharing
    require_free_gpu(wait_s=7200)            # block until it fits

    python gpu_guard.py --config configs/x.yaml
    python gpu_guard.py --wait 7200
    python gpu_guard.py --cotenants 2 --config configs/x.yaml

OVERRIDE. GFN_ALLOW_GPU_SHARING=1 downgrades every block to a warning. An env var,
not a config key: this is a property of the machine at launch time, not of the
experiment, and a YAML key would enter every arm's problem definition and go through
preflight_config's schema.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

TRAIN_ENTRYPOINTS = ('train.py', 'train_conformer.py')
OVERRIDE_ENV = 'GFN_ALLOW_GPU_SHARING'

# Declared co-tenancy for a run that was LAUNCHED as one of N deliberate siblings.
# train.py calls require_free_gpu() with no arguments, so without this a co-tenant
# would refuse itself the moment a sibling came up. NOT a bypass: the coherence check
# still runs, so declaring 6 while each config claims 90% of the card is still
# refused. An env var for the same reason as the override -- it describes this
# launch, not the experiment, and a YAML key would enter every arm's problem
# definition and preflight_config's schema.
COTENANTS_ENV = 'GFN_COTENANTS'


def declared_cotenants(explicit=None):
    if explicit is not None:
        return max(1, int(explicit))
    try:
        return max(1, int(os.environ.get(COTENANTS_ENV, '1')))
    except (TypeError, ValueError):
        return 1

_HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(_HERE, '.vram_registry.json')

# Fragmentation headroom on top of any projection. `expandable_segments` is NOT
# supported on this platform (torch warns at startup), so the allocator cannot give
# pages back and a run's high-water mark is what it keeps.
MARGIN_FRAC = 0.15
MARGIN_FLOOR_MB = 512

# Desktop compositing baseline (Chrome/Slack/Teams/explorer). Below this, "in use"
# is not evidence of a compute tenant. Measured ~1.7 GB idle on this box.
DESKTOP_BASELINE_MB = 2500


class GPUBusy(RuntimeError):
    pass


# ------------------------------------------------------------------ nvidia-smi
def _smi(args):
    try:
        out = subprocess.run(['nvidia-smi'] + args, capture_output=True, text=True,
                             timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _visible_index():
    """
    Which physical GPU this process can actually see, per CUDA_VISIBLE_DEVICES.

    Reading nvidia-smi's FIRST line unconditionally was a real bug on any multi-GPU
    box: a job pinned to GPU 3 would have been judged against GPU 0's occupancy, so a
    busy GPU 0 could block a launch onto an idle GPU 3 (and vice versa -- worse -- an
    idle GPU 0 could wave through a launch onto a full one).
    """
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cvd is None or cvd.strip() == '' or no_gpu_visible():
        return 0
    first = cvd.split(',')[0].strip()
    try:
        # CLAMPED. A bare negative index would silently read nvidia-smi's LAST row --
        # the wrong card, in whichever direction happens to be worse. -1 is handled
        # above by no_gpu_visible(); this covers any other negative.
        return max(0, int(first))
    except ValueError:
        return 0        # a UUID form; fall back to the first reported device


def no_gpu_visible():
    """
    True when this process can see NO GPU: CUDA_VISIBLE_DEVICES empty, or the
    conventional hide-all -1.

    This project's standing local-verification recipe is CUDA_VISIBLE_DEVICES=""
    (reference_local_run_recipe), and treating that as "judge GPU 0" made the guard refuse
    CPU-only work whenever a GPU run was active -- blocking exactly the runs that cannot
    possibly collide with anything.
    """
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cvd is None:
        return False
    toks = [t.strip() for t in cvd.split(',') if t.strip() != '']
    return len(toks) == 0 or all(t == '-1' for t in toks)


def gpu_memory():
    """(used_mb, free_mb, total_mb, util_pct) for THIS PROCESS'S visible GPU, or None."""
    raw = _smi(['--query-gpu=memory.used,memory.free,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'])
    if not raw:
        return None
    lines = raw.splitlines()
    idx = _visible_index()
    line = lines[idx] if idx < len(lines) else lines[0]
    try:
        used, free, total, util = (int(v.strip()) for v in line.split(','))
    except ValueError:
        return None
    return used, free, total, util


def opaque_compute_apps():
    """Compute processes whose identity nvidia-smi will not reveal. Reported, not blocked on."""
    raw = _smi(['--query-compute-apps=pid,process_name', '--format=csv,noheader'])
    if not raw:
        return []
    return [l.strip() for l in raw.splitlines()
            if 'Insufficient Permissions' in l or 'Not Found' in l]


# ------------------------------------------------------- other training processes
def _is_training_cmdline(cmdline):
    """
    True only if a command-line TOKEN's basename is a training entrypoint.

    Not a substring test on the joined command line. That matches far too much and
    yields false BUSY, which is worse than useless -- it refuses every launch:
      - any path merely CONTAINING the name: `_probe_train.py`, `retrain.py`
      - `python -c "...train.py..."`, where it appears in source text
      - a wrapper or logger that names the script it is about to run
    """
    for tok in (cmdline or []):
        tok = tok.strip().strip('"').strip("'")
        if not tok:
            continue
        if os.path.basename(tok.replace('\\', '/')).lower() in TRAIN_ENTRYPOINTS:
            return True
    return False


def _own_lineage():
    """
    This process plus its ancestors. The venv's `Scripts/python.exe` is a launcher
    stub that spawns the base interpreter with the same script, so one run really is
    two processes; `os.getpid()` alone would let a run detect ITSELF as somebody
    else's job and refuse to start on an idle GPU.
    """
    pids = {os.getpid()}
    try:
        import psutil
        for parent in psutil.Process().parents():
            pids.add(parent.pid)
    except Exception:
        pass
    return pids


def _collapse_launcher_pairs(found):
    """
    One RUN, not one process. The venv's `Scripts/python.exe` is a launcher stub that
    spawns the base interpreter with the same argv, so a single training run shows up
    twice. Left uncollapsed, one run reads as two tenants -- which does not matter for
    the default (any tenant blocks) but silently breaks every co-tenancy count, and
    would refuse a legitimate `cotenants=2` because one run looked like two.

    Drops any process whose PARENT is also in the set. What remains is the topmost
    process of each run.
    """
    if len(found) < 2:
        return found
    pids = {p for p, _ in found}
    try:
        import psutil
    except ImportError:
        return found
    keep = []
    for pid, cmd in found:
        try:
            if psutil.Process(pid).ppid() in pids:
                continue        # child of another matched process: same run
        except Exception:
            pass
        keep.append((pid, cmd))
    return keep or found


def training_processes(exclude_pids=None):
    """OTHER training RUNS on this machine: [(pid, cmdline)], one entry per run."""
    exclude = set(exclude_pids) if exclude_pids else _own_lineage()
    try:
        import psutil
    except ImportError:
        return _collapse_launcher_pairs(_training_processes_powershell(exclude))
    found = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            info = proc.info
            if info['pid'] in exclude:
                continue
            if 'python' not in (info['name'] or '').lower():
                continue
            if _is_training_cmdline(info['cmdline']):
                found.append((info['pid'], ' '.join(info['cmdline'])))
        except Exception:
            continue
    return _collapse_launcher_pairs(found)


def _training_processes_powershell(exclude):
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', ps],
                             capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    found = []
    for line in (out.stdout or '').splitlines():
        pid, _, cmd = line.partition('\t')
        try:
            pid = int(pid.strip())
        except ValueError:
            continue
        if pid not in exclude and _is_training_cmdline(cmd.split()):
            found.append((pid, cmd.strip()))
    return found


# ------------------------------------------------------------------- the config
def config_from_argv(argv=None):
    """
    The --config path, read the same way get_train_args does (it takes remaining[1]).
    Lets train.py call the guard before Modeller() exists, so nothing has touched
    CUDA yet.
    """
    argv = sys.argv if argv is None else argv
    for i, a in enumerate(argv):
        if a == '--config' and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith('--config='):
            return a.split('=', 1)[1]
    for a in argv[1:]:
        if a.endswith(('.yaml', '.yml')):
            return a
    return None


def read_config(path):
    if not path or not os.path.exists(path):
        return None
    try:
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _get(cfg, key, default=None):
    """
    Read a key from either a dict (the YAML, read by the CLI) or a Namespace
    (train.py's live self.args). Both callers exist, so accepting both here beats a
    converter at one of the call sites that the other would then need too.
    """
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        v = cfg.get(key, default)
    else:
        v = getattr(cfg, key, default)
    return default if v is None else v


def config_signature(cfg):
    """
    What a VRAM measurement is a measurement OF. Keep it to the things that actually
    move the footprint; anything else would fragment the registry into single-use rows.
    """
    if cfg is None:
        return None
    model = _get(cfg, 'model', {})
    integ = _get(cfg, 'integrator', {})
    try:
        return '|'.join(str(v) for v in (
            _get(cfg, 'energy_function'),
            # The OPERATING batch, not the growth CAP. Keying on max_batch_size treated a
            # ceiling as the working size: configs/aug02/0.yaml runs at batch 1000 with a
            # max of 50000, so it signed as ...|50000|... and (via the since-removed
            # cross-batch scaling) projected 698400 MiB, refused on an idle card. Only
            # consult max_batch_size when growth is actually enabled, since then the run
            # really can climb to it.
            int((_get(cfg, 'max_batch_size', 0) if _get(cfg, 'grow_batch_size', False)
                 else 0) or _get(cfg, 'batch_size', 0)
                or _get(cfg, 'max_batch_size', 0) or 0),
            int(_get(integ, 'T', 0)),
            int(_get(model, 's_emb_dim', 0)),
            int(_get(model, 'dplr_rank', 0)),
            int(max(_get(cfg, 'z_primes', [1]) or [1])),
            # traj_checkpoint trades ~33x trajectory activation memory for time (mk_dev's
            # own comment; 33.6x measured at T=100). Omitting it let a peak measured with
            # it ON be reused with it OFF -- an UNDER-estimate, the direction that
            # crashes the box.
            int(bool(_get(cfg, 'traj_checkpoint', False))),
            # record_peak fires right after the eval block, so eval memory is INSIDE the
            # measurement; and buffer_device: cuda puts the datasets and buffers on the
            # card. Both move the peak, so both belong in the key.
            int(_get(cfg, 'eval_num_samples', 0) or 0),
            str(_get(cfg, 'buffer_device', 'cuda')),
        ))
    except (TypeError, ValueError):
        return None


def declared_ceiling_mb(cfg, total_mb):
    """
    cuda_memory_fraction x total: what torch is PERMITTED to grow into, which the
    caching allocator does and then keeps. Not a prediction of the peak -- a bound on
    it, and the right conservative stand-in when no measurement exists.
    """
    if cfg is None or not total_mb:
        return None
    frac = _get(cfg, 'cuda_memory_fraction')
    if frac is None:
        return None
    try:
        return int(float(frac) * total_mb)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------- the projection
def load_registry():
    try:
        with open(REGISTRY, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def record_peak(cfg, peak_mb, peak_allocated_mb=None):
    """
    Store this config's observed high-water mark so future launches can project from
    a measurement instead of the declared ceiling. Called from train.py at eval time
    rather than at exit: a BSOD or OOM kill never reaches an atexit hook, and those
    are exactly the runs whose footprint we most want on record.

    Keeps the MAX per signature. A run that peaked higher is the one that matters.

    BOTH numbers are recorded because their RATIO is the actionable diagnostic:
      reserved ~ allocated  -> the memory is live tensors. To share the card you must
                               genuinely shrink something (batch, model, buffers, or
                               move buffer_device off cuda).
      reserved >> allocated -> it is caching and fragmentation under the
                               cuda_memory_fraction ceiling. Lowering that fraction
                               then costs little or nothing, and co-tenancy is
                               available WITHOUT degrading the experiment.
    Guessing between those two leads to shrinking the wrong thing.
    """
    sig = config_signature(cfg)
    if not sig or not peak_mb:
        return
    reg = load_registry()
    row = reg.get(sig) or {}
    if peak_allocated_mb:
        row['peak_allocated_mb'] = max(int(peak_allocated_mb),
                                       int(row.get('peak_allocated_mb', 0)))
    if peak_mb <= row.get('peak_reserved_mb', 0):
        reg[sig] = row
        _write_registry(reg)
        return
    row['peak_reserved_mb'] = int(peak_mb)
    row['observations'] = int(row.get('observations', 0)) + 1
    row['updated'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    reg[sig] = row
    _write_registry(reg)


def _write_registry(reg):
    # Unique temp name: with several arms running in parallel a shared '.tmp' would
    # have them clobbering each other's partial writes.
    tmp = f'{REGISTRY}.{os.getpid()}.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(reg, f, indent=2, sort_keys=True)
        os.replace(tmp, REGISTRY)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def project_need_mb(cfg, total_mb):
    """
    (need_mb, basis, raw_mb) -- the margined figure for BUDGETING, the grounds, and the
    RAW figure for asking "is there room now".

    Both are returned because they answer different questions. `raw_mb` is the measured
    high-water mark, which already contains fragmentation and is therefore direct
    evidence the job fits in that much. `need_mb` adds slack on top, which is right when
    reserving space alongside a co-tenant and wrong when re-asking whether a run that
    already succeeded can run again. `basis` is returned so a number's provenance
    travels with it.
    """
    if cfg is None:
        return None, 'no config: cannot project', None

    sig = config_signature(cfg)
    reg = load_registry()

    exact = reg.get(sig)
    if exact and exact.get('peak_reserved_mb'):
        need = exact['peak_reserved_mb']
        return (_with_margin(need),
                f"measured peak {need} MiB for this exact config", need)

    # NO CROSS-BATCH SCALING -- REMOVED, it was live breakage.
    #
    # This used to scale a measured peak by batch/ref_batch and floor it at the
    # measurement. Wrong in both directions, and only the scale-DOWN direction was
    # guarded. Much of a measured peak is batch-INDEPENDENT: parameters, and (with
    # buffer_device: cuda) buffers whose caps are 250k prior / 200k anchor. Multiplying
    # the whole figure by a batch ratio therefore over-estimates badly. With mk_dev's
    # 13968 MiB at batch 1000 on record, configs/aug02/0.yaml projected 698400 MiB and was
    # REFUSED on an idle card; a sweep found 42 configs in the tree refused this way.
    #
    # Separating base from slope would need two measurements at different batch sizes for
    # the same signature, which we do not have. So a measurement now counts for its OWN
    # signature and nothing else; everything else falls through to the declared ceiling,
    # which is conservative and honest about being a bound rather than a prediction.

    ceiling = declared_ceiling_mb(cfg, total_mb)
    if ceiling:
        # raw is None on purpose: a ceiling is not a measurement, so there is no
        # evidence-backed figure to run the room check against
        return (ceiling,
                f"cuda_memory_fraction {_get(cfg, 'cuda_memory_fraction')} x "
                f"{total_mb} MiB -- the config's DECLARED CEILING, no measurement "
                f"on record", None)
    return None, 'no measurement and no cuda_memory_fraction: cannot project', None


def _with_margin(mb):
    return int(mb + max(MARGIN_FLOOR_MB, MARGIN_FRAC * mb))


# ------------------------------------------------------------------- the check
def check(cotenants=None, config_path=None, cfg=None):
    """
    (ok, reasons, detail). ok False means do not launch.

    cotenants == 1 (default) is "this run expects the card to itself": ANY other
    training run blocks. cotenants == N > 1 asserts the caller has calibrated for
    N-way sharing, which is then checked for coherence rather than believed.
    """
    cotenants = declared_cotenants(cotenants)
    reasons, detail = [], {}
    if cfg is None:
        config_path = config_path or config_from_argv()
        cfg = read_config(config_path)
    detail['config_path'] = config_path
    detail['signature'] = config_signature(cfg)

    mem = gpu_memory()
    detail['memory'] = mem
    total_mb = mem[2] if mem else None
    free_mb = mem[1] if mem else None

    others = training_processes()
    detail['training_processes'] = others
    allowed_others = max(0, int(cotenants) - 1)
    if len(others) > allowed_others:
        if allowed_others == 0:
            reasons.append(
                f"{len(others)} training run(s) already on the GPU. Default policy is "
                f"one run per card -- pass cotenants=N only if these configs were "
                f"CALIBRATED to share (smaller batch/model, cuda_memory_fraction ~1/N)")
        else:
            reasons.append(f"{len(others)} training run(s) running, but only "
                           f"{allowed_others} co-tenant(s) declared")

    if mem is None:
        reasons.append("nvidia-smi did not answer -- GPU state UNKNOWN, not known-free")
    else:
        need, basis, raw = project_need_mb(cfg, total_mb)
        detail['need_mb'], detail['basis'], detail['raw_need_mb'] = need, basis, raw
        measured = basis and basis.startswith(('measured', 'scaled'))

        # THE ROOM CHECK, and what it must NOT do.
        #
        # `cuda_memory_fraction` is a CAP, not a demand: it says "this run may grow to
        # 90% of the card", never "it needs 90% now". Treating the ceiling as a
        # requirement made the guard refuse EVERY mk_dev-derived run on a completely
        # idle card -- 0.9 x 16303 = 14672 always exceeds free once the desktop has its
        # ~2.5 GB. That is a total false positive: it blocked the exact launches it has
        # no business having an opinion about.
        #
        # So the room check applies only where memory is genuinely contested or where
        # the need is a MEASUREMENT rather than a cap:
        #   - other tenants present, or a co-tenancy declared -> contested, check it
        #   - sole tenant -> ALLOW, measured or not. The run's own cap is its own
        #     business; if it does not fit it will OOM alone, which is not the failure
        #     mode this module exists to prevent. A measured peak marginally above free
        #     VRAM is the desktop having grown since the measurement, and it refuses a
        #     launch whose only risk is a recoverable solo OOM that the batch controller
        #     already handles.
        contested = bool(others) or cotenants > 1
        # THE ROOM FIGURE IS NOT THE BUDGETING FIGURE.
        #
        # `peak_reserved` ALREADY contains whatever fragmentation the run suffered -- it
        # is a high-water mark, not a steady state. Adding the margin on top of it and
        # then asking "does that fit?" invents a requirement larger than the thing we
        # measured actually needed: a job measured at 13968 MiB (which RAN, on this card)
        # projected to 16063 and was refused on an EMPTY card. Any config peaking above
        # ~86% of usable could never relaunch, despite direct proof that it fits.
        #
        # So a measured peak is used RAW for "is there room right now" -- the measurement
        # is the evidence -- and the margin is kept only for co-tenancy budgeting, where
        # slack is being reserved for somebody else's growth.
        room_need = detail.get('raw_need_mb') if measured else need
        if contested:
            if room_need is None:
                if mem[0] > DESKTOP_BASELINE_MB:
                    reasons.append(
                        f"cannot project this job's VRAM ({basis}) and {mem[0]} MiB is "
                        f"already in use (desktop baseline ~{DESKTOP_BASELINE_MB})")
            elif room_need > free_mb:
                reasons.append(f"projected need {room_need} MiB exceeds {free_mb} MiB "
                               f"free [{basis}]")
        else:
            detail['room_check'] = ('skipped: sole tenant and the projection is a '
                                    'declared CAP, not a measurement')

        # A co-tenancy claim has to be arithmetically possible. This is the check that
        # catches the misconfiguration that would crash the box even WITH the flag set:
        # declaring 3-way sharing while each config still claims 90% of the card.
        if int(cotenants) > 1 and need:
            if need * int(cotenants) > total_mb:
                reasons.append(
                    f"cotenants={cotenants} is incoherent: {cotenants} x {need} MiB = "
                    f"{need * int(cotenants)} MiB against a {total_mb} MiB card. Lower "
                    f"cuda_memory_fraction to about {1.0 / int(cotenants):.2f} and cut "
                    f"batch_size, or run them one at a time")

    detail['opaque'] = opaque_compute_apps()
    return (not reasons), reasons, detail


def describe(cotenants=None, config_path=None, cfg=None):
    ok, reasons, detail = check(cotenants, config_path, cfg)
    lines = []
    mem = detail.get('memory')
    if mem:
        used, free, total, util = mem
        lines.append(f"GPU: {used} MiB used / {total} MiB total, {free} MiB free, {util}% util")
    else:
        lines.append("GPU: state unreadable (nvidia-smi unavailable)")
    if detail.get('config_path'):
        lines.append(f"job: {detail['config_path']}  [{detail.get('signature')}]")
    n = declared_cotenants(cotenants)
    if n > 1:
        lines.append(f"co-tenancy: {n} declared -- this launch expects {n - 1} sibling(s)")
    if detail.get('need_mb'):
        lines.append(f"projected need: {detail['need_mb']} MiB -- {detail.get('basis')}")
    elif detail.get('basis'):
        lines.append(f"projected need: UNKNOWN -- {detail.get('basis')}")
    if detail.get('room_check'):
        lines.append(f"room check: {detail['room_check']}")
    for pid, cmd in detail.get('training_processes', []):
        lines.append(f"  tenant  pid {pid}: {cmd if len(cmd) < 140 else cmd[:137] + '...'}")
    for line in detail.get('opaque', []):
        lines.append(f"  note    compute process not readable by this user: {line}")
    for r in reasons:
        lines.append(f"  BLOCK   {r}")
    lines.append("GPU preflight: CLEAR TO LAUNCH" if ok else "GPU preflight: DO NOT LAUNCH")
    return ok, "\n".join(lines)


# Off switch, and the environments where this guard has no business having an opinion.
DISABLE_ENV = 'GFN_GPU_GUARD'          # set to 0 to skip the check entirely
# Batch schedulers ALREADY own the allocation decision: the job was given a GPU (often a
# cgroup-restricted slice of a shared node), so scanning the node for other people's
# train.py and refusing on their account is both wrong and rude. Skip, loudly.
SCHEDULER_ENV = ('SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID', 'SGE_TASK_ID',
                 'FLUX_JOB_ID')


def _env_true(name):
    """
    Tolerant, case-insensitive truthiness. The override was a strict `== '1'` while the
    disable switch took a lowercase list, so `GFN_ALLOW_GPU_SHARING=true` silently did
    nothing -- even though the refusal message tells the user to set it, and it is the
    ONLY escape hatch from a false refusal.
    """
    v = os.environ.get(name)
    return v is not None and v.strip().lower() in ('1', 'true', 'on', 'yes', 'y')


def _env_false(name):
    v = os.environ.get(name)
    return v is not None and v.strip().lower() in ('0', 'false', 'off', 'no', 'n')


def _skip_reason():
    # CPU-only work cannot collide with anything on the card, so the guard has no
    # business having an opinion about it.
    if no_gpu_visible():
        return 'CUDA_VISIBLE_DEVICES hides all GPUs -- CPU-only run'
    if _env_false(DISABLE_ENV):
        return f'{DISABLE_ENV}=0'
    for var in SCHEDULER_ENV:
        if os.environ.get(var):
            return f'{var} is set -- the scheduler owns GPU allocation here'
    return None


def require_free_gpu(cotenants=None, config_path=None, cfg=None, wait_s=0, poll_s=20,
                     quiet=False):
    """
    Raise GPUBusy unless this job should be launched now. wait_s > 0 blocks and
    re-checks until it fits or the deadline passes.

    Returns immediately under a batch scheduler or with GFN_GPU_GUARD=0.
    """
    skip = _skip_reason()
    if skip:
        if not quiet:
            print(f'GPU preflight: SKIPPED ({skip})')
        return True
    deadline = time.time() + max(0, wait_s)
    announced = False
    while True:
        ok, text = describe(cotenants, config_path, cfg)
        if ok:
            if not quiet:
                print(text)
            return True
        if _env_true(OVERRIDE_ENV):
            print(text)
            print(f"WARNING: {OVERRIDE_ENV}=1 -- launching onto a GPU this check says "
                  f"is not clear. Two training runs on one card is what BSOD'd this "
                  f"machine three times on 2026-08-11/12.")
            return True
        if time.time() >= deadline:
            raise GPUBusy(text + "\n\nRefusing to launch. Wait for the other run, "
                                 "calibrate both configs to share and pass "
                                 f"cotenants=N, or set {OVERRIDE_ENV}=1 to override.")
        if not announced:
            print(text)
            print(f"waiting up to {wait_s}s for the GPU to clear...")
            announced = True
        time.sleep(poll_s)


def main(argv=None):
    ap = argparse.ArgumentParser(description='GPU pre-flight: will this job fit?')
    ap.add_argument('--config', default=None, help='the job about to be launched')
    ap.add_argument('--cotenants', type=int, default=None,
                    help='declare deliberate N-way sharing (default 1 = card to itself)')
    ap.add_argument('--wait', type=int, default=0, metavar='SECONDS')
    ap.add_argument('--registry', action='store_true', help='dump measured peaks and exit')
    args = ap.parse_args(argv)

    if args.registry:
        reg = load_registry()
        if not reg:
            print(f"no measurements yet ({REGISTRY})")
            print("projections will fall back to each config's declared "
                  "cuda_memory_fraction ceiling until a run records one")
            return 0
        print(f"{REGISTRY}")
        for k, v in sorted(reg.items()):
            print(f"  {k:<48} {v.get('peak_reserved_mb')} MiB  "
                  f"(n={v.get('observations')}, {v.get('updated')})")
        return 0

    try:
        require_free_gpu(cotenants=args.cotenants, config_path=args.config,
                         wait_s=args.wait)
    except GPUBusy as e:
        print(str(e))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
