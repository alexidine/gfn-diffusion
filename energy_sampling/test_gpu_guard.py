"""
Tests for gpu_guard.py.

THE FAILURE THAT MATTERS IS A FALSE BUSY. A guard that misses a real collision costs
a BSOD; a guard that reports BUSY on an idle GPU refuses every launch, and would be
removed within the hour -- so the false-positive cases get the most coverage here.
Both were live defects in the first version: a substring match on the whole command
line, and self-detection when the caller's own command line names the entrypoint.

    python test_gpu_guard.py
"""
import os
import subprocess
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import gpu_guard as G

VENV = r'C:\Users\mikem\venvs\csd_mxt_gfn\Scripts\python.exe'
_R = []


def check(name, ok, detail=''):
    _R.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")


def test_cmdline_matching():
    """Token-basename matching, not substring. Every case here was a false BUSY."""
    print("\n1. command-line classification")
    yes = [
        ['python', 'train.py'],
        ['python.exe', 'train.py', '--config', 'a.yaml'],
        [r'C:\venv\Scripts\python.exe', r'C:\proj\energy_sampling\train.py'],
        ['python', 'train_conformer.py'],
        ['python', '"C:/proj/train.py"'],
    ]
    no = [
        # the whole point: these merely MENTION the name
        ['python', '-c', 'import time; print("train.py")'],
        ['python', '_probe_train.py'],
        ['python', 'retrain.py'],
        ['python', 'pretrain.py'],
        ['python', 'test_train.py'],
        ['python', 'gpu_guard.py'],
        ['python', 'read_results.py', '--note', 'after train.py finishes'],
        [],
        None,
    ]
    for c in yes:
        check(f"IS training: {c}", G._is_training_cmdline(c))
    for c in no:
        check(f"NOT training: {c}", not G._is_training_cmdline(c))


def test_self_not_detected():
    """
    A process whose own command line names the entrypoint must not detect ITSELF.
    This was a real defect: it made the guard report BUSY on a completely idle GPU,
    which would have blocked every launch.
    """
    print("\n2. self and ancestors are excluded")
    script = os.path.join(_here, 'train.py')
    if not os.path.exists(script):
        check("train.py present to impersonate", False, 'missing')
        return
    # run the REAL entrypoint name, but only import the guard and report -- never
    # reaching Modeller(), so nothing touches CUDA
    code = (
        "import os,sys;"
        f"sys.path.insert(0, r'{_here}');"
        "import gpu_guard as G;"
        "print('PID', os.getpid());"
        "print('OTHERS', [p for p,_ in G.training_processes()])"
    )
    # argv[0] is the -c string, so to genuinely test self-exclusion the process must
    # have train.py as a token: use a copy named train.py in a temp dir
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        fake = os.path.join(tmp, 'train.py')
        with open(fake, 'w', encoding='utf-8') as f:
            f.write(code.replace(';', '\n'))
        out = subprocess.run([VENV, fake], capture_output=True, text=True, timeout=180)
        txt = (out.stdout or '') + (out.stderr or '')
        pid = None
        others = None
        for line in txt.splitlines():
            if line.startswith('PID '):
                pid = int(line.split()[1])
            if line.startswith('OTHERS '):
                others = line[len('OTHERS '):].strip()
        check("self-impersonating process ran", pid is not None, txt[-300:] if pid is None else '')
        # assert on its OWN pid's absence, not on an empty list: a real battery arm may
        # legitimately be running, and this test is about self-detection only
        check("it does NOT report itself as another training process",
              pid is not None and str(pid) not in (others or ''),
              f"own pid {pid}, others={others}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_detects_a_real_other_process():
    """The guard must still catch the thing it exists for."""
    print("\n3. a genuine second training process IS detected")
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    proc = None
    # PIN CUDA_VISIBLE_DEVICES for the duration. The suite is often invoked with
    # CUDA_VISIBLE_DEVICES="" (this project's CPU-only recipe), which now correctly means
    # "CPU run, do not judge the card" -- and would silently SKIP this test, the one that
    # checks the guard still catches the collision it exists for. Inheriting the variable
    # made the outcome depend on how the suite was invoked.
    _saved_cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    try:
        fake = os.path.join(tmp, 'train.py')
        with open(fake, 'w', encoding='utf-8') as f:
            f.write('import time\ntime.sleep(90)\n')
        proc = subprocess.Popen([VENV, fake],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 30
        found = []
        while time.time() < deadline:
            found = [p for p, _ in G.training_processes()]
            if proc.pid in found:
                break
            time.sleep(1)
        check("detects the spawned train.py", proc.pid in found,
              f"pid {proc.pid}, found {found}")
        ok, reasons, _ = G.check()
        # match on the tenant COUNT being reported, not exact wording -- the previous
        # form of this assertion went stale silently when the message was reworded
        check("check() reports DO NOT LAUNCH",
              (not ok) and any('training run' in r for r in reasons), f"reasons={reasons}")
        try:
            G.require_free_gpu()
            check("require_free_gpu raises GPUBusy", False, 'it returned instead')
        except G.GPUBusy:
            check("require_free_gpu raises GPUBusy", True)
        os.environ[G.OVERRIDE_ENV] = '1'
        try:
            check("override lets it through", G.require_free_gpu() is True)
        finally:
            os.environ.pop(G.OVERRIDE_ENV, None)
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=30)
        shutil.rmtree(tmp, ignore_errors=True)
        if _saved_cvd is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = _saved_cvd
    # the SPAWNED pid must disappear; other genuine runs may still be present, so do
    # not assert on an empty list -- that made this test fail whenever a real arm ran
    gone_pid = proc.pid
    deadline = time.time() + 30
    while time.time() < deadline:
        if gone_pid not in [p for p, _ in G.training_processes()]:
            break
        time.sleep(1)
    still = [p for p, _ in G.training_processes()]
    check("the spawned run is gone once it exits", gone_pid not in still, f"still {still}")


def test_cpu_only_run_is_not_judged():
    """
    CUDA_VISIBLE_DEVICES empty (or -1) means NO GPU is visible -- a CPU-only run, which
    cannot collide with anything on the card. Judging GPU 0 anyway made the guard refuse
    CPU work whenever a GPU run was active, and that is this project's standing local
    verification recipe, so it blocked exactly the launches that are always safe.
    """
    print("\n4. a CPU-only run is skipped, not judged")
    _saved = os.environ.get('CUDA_VISIBLE_DEVICES')
    real_proc, real_mem = G.training_processes, G.gpu_memory
    try:
        # a tenant present AND no free memory: a hard block for a real GPU run
        G.training_processes = lambda *a, **k: [(999, 'python train.py --config x.yaml')]
        G.gpu_memory = lambda: (15900, 100, 16303, 99)
        for hidden in ('', '-1', '-1,-1'):
            os.environ['CUDA_VISIBLE_DEVICES'] = hidden
            check(f"CVD={hidden!r} is hidden", G.no_gpu_visible())
            check(f"CVD={hidden!r} returns instead of raising",
                  G.require_free_gpu(quiet=True) is True)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        check("CVD='0' is still judged", not G.no_gpu_visible())
        try:
            G.require_free_gpu(quiet=True)
            check("CVD='0' raises GPUBusy", False, 'returned instead')
        except G.GPUBusy:
            check("CVD='0' raises GPUBusy", True)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-2'
        check("a negative index clamps to 0, never nvidia-smi's last row",
              G._visible_index() == 0, str(G._visible_index()))
    finally:
        G.training_processes, G.gpu_memory = real_proc, real_mem
        if _saved is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = _saved


def test_unknown_gpu_is_not_free():
    """
    No nvidia-smi must NOT read as 'free'. The whole point is preventing a
    machine-killing collision, so unknown is a block, not a pass.
    """
    print("\n4. unreadable GPU is treated as unknown, not free")
    real = G._smi
    G._smi = lambda args: None
    try:
        ok, reasons, detail = G.check()
        check("blocks when nvidia-smi is silent", (not ok) and any('UNKNOWN' in r for r in reasons),
              f"reasons={reasons}")
    finally:
        G._smi = real


def test_room_check_only_where_it_belongs():
    """
    A CAP IS NOT A DEMAND. `cuda_memory_fraction: 0.9` means "may grow to 90%", not
    "needs 90%". Treating the ceiling as a requirement made the guard refuse EVERY
    mk_dev-derived run on a completely idle card (0.9 x 16303 = 14672 > free), i.e. a
    total false positive against exactly the launches it should not judge. So the room
    check applies only where memory is contested, or where the need is MEASURED.

    A false BUSY is the failure mode that gets a guard deleted, so all three branches
    are pinned here.
    """
    print("\n5. the room check fires only when contested or measured")
    cfg = {'energy_function': 'zzz_test', 'batch_size': 100, 'max_batch_size': 100,
           'cuda_memory_fraction': 0.5, 'z_primes': [1],
           'model': {'s_emb_dim': 8, 'dplr_rank': 0}, 'integrator': {'T': 4}}
    sig = G.config_signature(cfg)
    real_mem, real_proc, real_reg = G.gpu_memory, G.training_processes, G.load_registry
    G.gpu_memory = lambda: (15000, 500, 16000, 99)      # only 500 MiB free
    try:
        # (a) sole tenant + CEILING projection -> must ALLOW. This is the regression.
        G.training_processes = lambda *a, **k: []
        G.load_registry = lambda: {}
        ok, reasons, detail = G.check(cfg=cfg)
        check("sole tenant + declared CAP -> allowed even at 500 MiB free", ok,
              f"reasons={reasons}")

        # (b) sole tenant + MEASURED need -> must BLOCK. A measurement is a real number.
        G.load_registry = lambda: {sig: {'peak_reserved_mb': 8000}}
        ok, reasons, _ = G.check(cfg=cfg)
        check("sole tenant + MEASURED need -> blocked",
              (not ok) and any('exceeds' in r for r in reasons), f"reasons={reasons}")

        # (c) contested (a tenant present) + ceiling -> must BLOCK
        G.load_registry = lambda: {}
        G.training_processes = lambda *a, **k: [(999, 'python train.py --config x.yaml')]
        ok, reasons, _ = G.check(cfg=cfg)
        check("tenant present + declared CAP -> blocked",
              (not ok) and any('exceeds' in r for r in reasons), f"reasons={reasons}")
    finally:
        G.gpu_memory, G.training_processes, G.load_registry = real_mem, real_proc, real_reg


def test_signature_agrees_across_dict_and_namespace():
    """
    train.py records a peak from a Namespace; the CLI looks it up from the YAML dict.
    If the two disagree, every measurement is written to a key nothing ever reads and
    the registry silently stays empty forever.
    """
    print("\n6. registry key is the same from the REAL args object and from the YAML")
    import glob
    import yaml
    # Through the REAL loader, not a hand-built Namespace. Hand-copying the fields is
    # what an earlier version did, and it asserted agreement against the copy rather than
    # against the object train.py actually passes to record_peak -- so a signature field
    # the copy forgot would read as agreement. Every field must come from the same path
    # the run uses.
    from energy_sampling.utils import load_yaml, dict2namespace, preflight_config, \
        resolve_derived_config
    cfgs = sorted(glob.glob(os.path.join(_here, 'configs', 'gauss_aug12', '*_sg*_o*.yaml')))
    if not cfgs:
        check("an arm config to compare against", False, 'none found')
        return
    checked = 0
    for path in cfgs[:3]:
        d = yaml.safe_load(open(path, encoding='utf-8'))
        args = resolve_derived_config(preflight_config(dict2namespace(load_yaml(path))))
        a, b = G.config_signature(args), G.config_signature(d)
        check(f"{os.path.basename(path)}: real args and YAML agree",
              a == b and a is not None, f"{a} vs {b}")
        checked += 1
    args = resolve_derived_config(preflight_config(dict2namespace(load_yaml(cfgs[0]))))
    check("declared ceiling reads from the real args object",
          G.declared_ceiling_mb(args, 16303) == int(0.9 * 16303),
          str(G.declared_ceiling_mb(args, 16303)))
    # every signature field must actually be reachable on the real object, or it silently
    # contributes a default on one side of the comparison
    for field in ('energy_function', 'batch_size', 'max_batch_size', 'grow_batch_size',
                  'z_primes', 'traj_checkpoint', 'eval_num_samples', 'buffer_device'):
        check(f"real args exposes {field}", hasattr(args, field))


def test_projection_prefers_measurement_and_is_labelled():
    """
    The projection must say WHAT it is based on. A bare number gets trusted more than
    it deserves, and the fallback (a declared ceiling) is a bound, not a prediction.
    """
    print("\n7. projection basis, and measurement beats the declared ceiling")
    cfg = {'energy_function': 'zzz_test', 'batch_size': 100, 'max_batch_size': 100,
           'cuda_memory_fraction': 0.5, 'z_primes': [1],
           'model': {'s_emb_dim': 8, 'dplr_rank': 0}, 'integrator': {'T': 4}}
    real = G.load_registry
    try:
        G.load_registry = lambda: {}
        need, basis, _raw = G.project_need_mb(cfg, 16000)
        check("no measurement -> declared ceiling", need == 8000 and 'DECLARED CEILING' in basis,
              f"{need} MiB, {basis}")

        sig = G.config_signature(cfg)
        G.load_registry = lambda: {sig: {'peak_reserved_mb': 3000}}
        need, basis, _raw = G.project_need_mb(cfg, 16000)
        check("exact measurement is used, with margin", need > 3000 and 'measured peak' in basis,
              f"{need} MiB, {basis}")

        # A DIFFERENT batch must NOT be extrapolated from. Cross-batch scaling was
        # removed: a peak is largely batch-independent (parameters, cuda-resident
        # buffers), so scaling it by a batch ratio over-estimated wildly -- mk_dev's
        # 13968 MiB at batch 1000 projected 698400 MiB for configs/aug02/0.yaml, refusing
        # it on an idle card, and 42 configs in the tree were refused that way. A
        # measurement is evidence for its OWN signature only; anything else falls to the
        # ceiling.
        for other_batch in ('|50|', '|400|'):
            other = sig.replace('|100|', other_batch, 1)
            G.load_registry = lambda o=other: {o: {'peak_reserved_mb': 8000}}
            need, basis, raw = G.project_need_mb(cfg, 16000)
            check(f"batch {other_batch.strip('|')} measurement is NOT extrapolated",
                  'DECLARED CEILING' in basis and raw is None,
                  f"{need} MiB, {basis}")

        # no ceiling and no measurement -> refuse to invent a number
        G.load_registry = lambda: {}
        bare = dict(cfg)
        bare.pop('cuda_memory_fraction')
        need, basis, _raw = G.project_need_mb(bare, 16000)
        check("refuses to guess with no basis at all", need is None, f"{need}, {basis}")
    finally:
        G.load_registry = real


def test_measured_config_relaunches_on_an_empty_card():
    """
    THE INVARIANT THAT GENERALISES ALL THREE FALSE POSITIVES SO FAR:
    if a config has demonstrably RUN on this card, and the card is now empty, the guard
    must not refuse it.

    The concrete failure: run 1 of the elj arm peaked at 13968 MiB and completed. That
    measurement then blocked run 2 on an idle card, because the 15% margin was added on
    top of a `peak_reserved` figure that ALREADY contains fragmentation -- projecting
    16063 MiB, more than the physical card. Any config peaking above ~86% of usable
    could never relaunch, despite direct proof it fits.
    """
    print("\n8. a config that already ran must relaunch on an empty card")
    cfg = {'energy_function': 'zzz_big', 'batch_size': 1000, 'max_batch_size': 1000,
           'cuda_memory_fraction': 0.9, 'z_primes': [1],
           'model': {'s_emb_dim': 512, 'dplr_rank': 6}, 'integrator': {'T': 10}}
    sig = G.config_signature(cfg)
    real_mem, real_proc, real_reg = G.gpu_memory, G.training_processes, G.load_registry
    try:
        # exactly the elj numbers: peaked at 13968 on a 16303 card, now idle
        G.gpu_memory = lambda: (1563, 14441, 16303, 10)
        G.training_processes = lambda *a, **k: []
        G.load_registry = lambda: {sig: {'peak_reserved_mb': 13968}}
        ok, reasons, detail = G.check(cfg=cfg)
        check("measured 13968 relaunches with 14441 free", ok, f"reasons={reasons}")
        check("the RAW measurement is what the room check used",
              detail.get('raw_need_mb') == 13968, f"raw={detail.get('raw_need_mb')}")
        check("the margined figure is still reported for budgeting",
              detail.get('need_mb', 0) > 13968, f"need={detail.get('need_mb')}")

        # and it must still refuse when the measurement genuinely does not fit
        G.gpu_memory = lambda: (8000, 4000, 16303, 40)
        ok, reasons, _ = G.check(cfg=cfg)
        check("but refuses when the measurement exceeds free",
              (not ok) and any('exceeds' in r for r in reasons), f"reasons={reasons}")
    finally:
        G.gpu_memory, G.training_processes, G.load_registry = real_mem, real_proc, real_reg


def test_cotenancy_coherence():
    """
    The check that catches the misconfiguration which would crash the box even WITH
    the flag set: declaring N-way sharing while each config still claims most of the
    card. A co-tenancy claim is validated, not believed.
    """
    print("\n8. co-tenancy claims are checked for arithmetic possibility")
    cfg = {'energy_function': 'zzz_test', 'batch_size': 100, 'max_batch_size': 100,
           'cuda_memory_fraction': 0.9, 'z_primes': [1],
           'model': {'s_emb_dim': 8, 'dplr_rank': 0}, 'integrator': {'T': 4}}
    real_mem, real_reg, real_proc = G.gpu_memory, G.load_registry, G.training_processes
    try:
        G.gpu_memory = lambda: (2000, 14000, 16000, 5)
        G.load_registry = lambda: {}
        G.training_processes = lambda *a, **k: []
        ok, reasons, _ = G.check(cotenants=3, cfg=cfg)
        check("3-way sharing at 0.9 each is refused",
              (not ok) and any('incoherent' in r for r in reasons), f"{reasons}")

        share = dict(cfg, cuda_memory_fraction=0.3)
        ok, reasons, _ = G.check(cotenants=3, cfg=share)
        check("3-way sharing at 0.3 each is allowed", ok, f"{reasons}")

        # and a declared co-tenant budget still cannot exceed what is FREE right now
        G.gpu_memory = lambda: (13000, 3000, 16000, 90)
        ok, reasons, _ = G.check(cotenants=3, cfg=share)
        check("but not when the free memory is not there",
              (not ok) and any('exceeds' in r for r in reasons), f"{reasons}")
    finally:
        G.gpu_memory, G.load_registry, G.training_processes = real_mem, real_reg, real_proc


def test_free_gpu_passes():
    """The real machine, right now. If this fails with nothing running, it is unusable."""
    print("\n9. the actual GPU, as it stands")
    ok, text = G.describe()
    print('    ' + text.replace('\n', '\n    '))
    others = G.training_processes()
    if others:
        check("a run IS active, so DO NOT LAUNCH is correct", not ok,
              f"tenants: {[p for p, _ in others]}")
    else:
        check("idle GPU reads CLEAR", ok,
              'if this fails with no run active, the guard would block every launch')


def main():
    for fn in (test_cmdline_matching, test_self_not_detected,
               test_detects_a_real_other_process,
               test_cpu_only_run_is_not_judged, test_unknown_gpu_is_not_free,
               test_room_check_only_where_it_belongs,
               test_signature_agrees_across_dict_and_namespace,
               test_projection_prefers_measurement_and_is_labelled,
               test_measured_config_relaunches_on_an_empty_card,
               test_cotenancy_coherence, test_free_gpu_passes):
        try:
            fn()
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
            _R.append((fn.__name__, False, f'{type(e).__name__}: {e}'))
    bad = [r for r in _R if not r[1]]
    print("\n" + "=" * 72)
    print(f"{len(_R) - len(bad)}/{len(_R)} checks passed")
    for n, _, d in bad:
        print(f"  FAIL {n}  {d}")
    print("PASS" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
