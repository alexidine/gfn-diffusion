"""Cache of the trainer's init-time re-scoring of the prior dataset (cfg:prior_scan_cache).

`train.py::Modeller.init_prior_dataset` scores every row of `prior_path` through the run's energy function at start-up
and keeps the batch that pass returns (it carries the fields every later batch must also carry). On an MLIP route with
a few hundred thousand rows that is minutes to an hour, on every launch and every requeue.

With the cache on, the first launch does the pass as before and writes its result beside the prior file as
`<prior_path>.scan-<identity hash>.pt`. A later launch loads that file when its IDENTITY matches, then re-scores
CHECK_ROWS random rows of the prior and compares them with the cached energies; if more than MAX_BAD_FRACTION of them
differ by more than ENERGY_TOL the cache is treated as stale, the whole prior is scored again and the file rewritten.

IDENTITY is everything the config can say about the scoring: the prior file (name, size, a hash of its first and last
MiB), the row count, the energy function, the MLIP checkpoint (name and size), the space groups and Z' values, the
coefficient the run applies to eLJ, the temperature, and every key of energy_config. It cannot see a CODE change in
the scoring; the row check is what catches that.
"""
import hashlib
import json
import os

import torch

VERSION = 1
#: rows of the prior re-scored against a loaded cache
CHECK_ROWS = 512
#: energy units; a checked row disagrees when its two energies differ by more than this. Two scorings of one UMA
#: crystal in different batches differ by up to ~0.3 kJ/mol (median 0.02); eLJ by ~0.002 raw units.
ENERGY_TOL = 0.5
#: a cache is stale when more than this share of the checked rows disagree
MAX_BAD_FRACTION = 0.01


def file_identity(path):
    """(base name, size in bytes, sha1 of the first MiB and, past 2 MiB, the last MiB) of a file."""
    size = os.path.getsize(path)
    h = hashlib.sha1()
    with open(path, 'rb') as fh:
        h.update(fh.read(1 << 20))
        if size > 2 << 20:
            fh.seek(-(1 << 20), os.SEEK_END)
            h.update(fh.read(1 << 20))
    return [os.path.basename(str(path)), int(size), h.hexdigest()[:16]]


def _plain(obj):
    """A JSON-serialisable copy of a config value (namespaces and dicts become sorted dicts)."""
    if hasattr(obj, '__dict__') and not isinstance(obj, type):
        obj = vars(obj)
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def scan_identity(args, n_rows, lj_coeff):
    """The identity dict of a scoring pass of args.prior_path under args (see the module docstring)."""
    mlip = getattr(args, 'mlip_path', None)
    return {
        'version': VERSION,
        'prior': file_identity(args.prior_path),
        'n_rows': int(n_rows),
        'energy_function': str(args.energy_function),
        'mlip': [os.path.basename(str(mlip)), int(os.path.getsize(mlip))] if mlip else None,
        'space_groups': _plain(getattr(args, 'space_groups', None)),
        'z_primes': _plain(getattr(args, 'z_primes', None)),
        'lj_coeff': None if lj_coeff is None else float(lj_coeff),
        'energy_config': _plain(args.energy_config),
    }


def identity_hash(identity):
    return hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]


def cache_path(prior_path, identity):
    return f'{prior_path}.scan-{identity_hash(identity)}.pt'


def save(path, identity, energy, batch):
    """Write the cache atomically (a temporary file, then os.replace). Returns True, or False with a printed reason
    when it cannot be written: a missing cache only costs the next launch a full pass."""
    tmp = f'{path}.tmp.{os.getpid()}'
    try:
        torch.save({'version': VERSION, 'identity': identity, 'energy': energy.detach().cpu(),
                    'batch': batch.clone().detach().cpu()}, tmp)
        os.replace(tmp, path)
        return True
    except (OSError, RuntimeError) as err:  # torch.save reports a missing directory as RuntimeError
        print(f'prior scan cache: could not write {path} ({type(err).__name__}: {err})')
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def load(path, identity):
    """(energy, batch) from a cache file whose stored identity equals `identity`, else None with a printed reason."""
    if not os.path.exists(path):
        print(f'prior scan cache: none at {path}')
        return None
    try:
        blob = torch.load(path, weights_only=False)
    except Exception as err:  # noqa: BLE001 -- a truncated or half-written file: score again and rewrite it
        print(f'prior scan cache: {path} is unreadable ({type(err).__name__}); scoring the prior again')
        return None
    if blob.get('version') != VERSION or blob.get('identity') != identity:
        print(f'prior scan cache: {path} was written for another identity; scoring the prior again')
        return None
    return blob['energy'], blob['batch']


def check_rows(n_rows, seed=0):
    """The row indices a loaded cache is checked on: min(CHECK_ROWS, n_rows) distinct rows, fixed by `seed`."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(int(n_rows), generator=g)[:CHECK_ROWS]


def agrees(cached, rescored, tol=ENERGY_TOL, max_bad=MAX_BAD_FRACTION):
    """(ok, bad fraction, median |difference|, max |difference|) of cached against freshly scored energies of the
    same rows. A non-finite value on either side counts as a disagreement unless both are non-finite alike."""
    a, b = cached.detach().double().flatten().cpu(), rescored.detach().double().flatten().cpu()
    if a.shape != b.shape:
        return False, 1.0, float('nan'), float('nan')
    both_bad = ~torch.isfinite(a) & ~torch.isfinite(b)
    d = (a - b).abs()
    d[both_bad] = 0.0
    bad = ~(d <= tol)
    frac = float(bad.float().mean()) if d.numel() else 0.0
    fin = d[torch.isfinite(d)]
    med = float(fin.median()) if fin.numel() else float('nan')
    mx = float(fin.max()) if fin.numel() else float('nan')
    return frac <= max_bad, frac, med, mx
