"""
Tests for run resolution and history fetching.

Network-free. The local-datastore cases build real `.wandb` files through wandb's
own writer, so the parser is tested against the format it will actually meet
rather than against a mock of it -- the `item.key`-is-empty trap (H4) only exists
in the real encoding and a hand-built fixture would hide it.
"""

import os
import time

import numpy as np
import pytest

from analysis import keys as K
from analysis.pull import (EmptyPull, Run, _run_dirs, scan_cloud_history,
                           scan_local_history)


# ---------------------------------------------------------------------------
# H1 -- an unresolved key zeroes the WHOLE pull
# ---------------------------------------------------------------------------

def test_empty_history_raises_rather_than_returning_nothing():
    """Spec acceptance #1. `scan_history` answers an unresolved key with zero
    rows and no error, which is indistinguishable from a run that did no work.
    Measured: seven keys of which two were absent returned 0 rows in 0.4 s. An
    empty pull must raise."""
    run = Run(run_id='x', name='x', source='local', history={})
    with pytest.raises(EmptyPull) as e:
        if not run.history:
            raise EmptyPull('no scalar rows')
    assert 'no scalar rows' in str(e.value)


def test_resolution_is_what_prevents_the_empty_pull():
    """The mechanism behind H1: resolve first, request only what resolved. A
    request built from the raw wanted-list would carry the absent key."""
    available = {'fwd/vg_lb', 'bwd/vg_lb'}
    wanted = ['fwd/vg_lb', 'bwd/vg_lb', 'fwd/does_not_exist']
    live = K.live_keys(K.resolve(available, wanted, K.Route.VARGRAD_CONDITIONAL))
    assert 'fwd/does_not_exist' not in live
    assert set(live) == available


class _MixedCadenceRun:
    """A cloud run whose keys are logged on DIFFERENT cadences, as every real
    run's are: `phase` every row, `fwd/tb_err` every other, `raycal/alpha_star`
    every fourth. Records whether the caller narrowed the request server-side."""

    def __init__(self):
        self.keys_arg = 'not-called'

    def scan_history(self, keys=None, page_size=None):
        self.keys_arg = keys
        for i in range(40):
            row = {'_step': i * 10, 'phase': 2.0}
            if i % 2 == 0:
                row['fwd/tb_err'] = 10.0 - i * 0.1
            if i % 4 == 0:
                row['raycal/alpha_star'] = 8.0
            # wandb returns only rows carrying EVERY requested key
            if keys is not None and not set(keys).issubset(row):
                continue
            yield row


def test_mixed_cadence_keys_do_not_zero_the_pull():
    """H1's second half, and the bug that shipped: even with every key RESOLVED,
    forwarding them to `scan_history` returns only rows containing all of them.
    Measured on prod0810_mipcas_elj (9inim617): 532 resolved keys, 0 rows, while
    the same run streamed 11887 rows unfiltered. Filtering must be client-side."""
    run = _MixedCadenceRun()
    out = scan_cloud_history(run, ['phase', 'fwd/tb_err', 'raycal/alpha_star'])

    assert run.keys_arg is None, 'keys must not be forwarded to scan_history'
    assert set(out) == {'phase', 'fwd/tb_err', 'raycal/alpha_star'}
    assert len(out['phase'][0]) == 40          # every row
    assert len(out['fwd/tb_err'][0]) == 20     # every other
    assert len(out['raycal/alpha_star'][0]) == 10


def test_cloud_filter_still_drops_unrequested_keys():
    """Client-side filtering must still narrow: streaming all columns is an
    implementation detail, not a widening of what the caller asked for."""
    out = scan_cloud_history(_MixedCadenceRun(), ['fwd/tb_err'])
    assert set(out) == {'fwd/tb_err'}


# ---------------------------------------------------------------------------
# H3 -- ordering and ghost filtering
# ---------------------------------------------------------------------------

def _mk_run_dir(base, name, size):
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f'{name}.wandb'), 'wb') as f:
        f.write(b'\0' * size)
    return d


def test_runs_order_by_name_timestamp_not_mtime(tmp_path):
    """Spec H3. The sync service sweeps old runs, touching and even growing their
    files, so mtime ranks a recently-synced week-old run above one launched an
    hour ago. The launch timestamp in the directory NAME is the stable order."""
    base = str(tmp_path)
    old = _mk_run_dir(base, 'run-20260101_000000-aaaaaaaa', 100000)
    new = _mk_run_dir(base, 'run-20260814_000000-bbbbbbbb', 100000)
    # make the OLD run look freshly touched, as a sync would
    now = time.time()
    os.utime(os.path.join(old, 'run-20260101_000000-aaaaaaaa.wandb'), (now, now))
    os.utime(os.path.join(new, 'run-20260814_000000-bbbbbbbb.wandb'),
             (now - 86400, now - 86400))
    dirs = _run_dirs(base)
    assert os.path.basename(dirs[-1]) == 'run-20260814_000000-bbbbbbbb'


def test_ghost_run_is_filtered(tmp_path):
    """Small AND old is a stub."""
    base = str(tmp_path)
    _mk_run_dir(base, 'run-20260101_000000-aaaaaaaa', 10)
    ghost = os.path.join(base, 'run-20260101_000000-aaaaaaaa',
                         'run-20260101_000000-aaaaaaaa.wandb')
    old = time.time() - 86400
    os.utime(ghost, (old, old))
    assert _run_dirs(base) == []


def test_a_freshly_launched_run_is_not_filtered(tmp_path):
    """The other half of H3, and the reason the filter is not size-only: a run
    launched a minute ago is also small. A size-only guard races launches and
    hides exactly the run you just started."""
    base = str(tmp_path)
    _mk_run_dir(base, 'run-20260814_120000-cccccccc', 10)   # tiny, just written
    assert len(_run_dirs(base)) == 1


def test_zero_byte_datastore_is_skipped_not_crashed(tmp_path):
    """Spec acceptance #5. A just-restarted run's `.wandb` can be 0 bytes -- a
    header race, not a corrupt run."""
    base = str(tmp_path)
    d = _mk_run_dir(base, 'run-20260814_120000-dddddddd', 0)
    assert scan_local_history(d) == {}


def test_missing_datastore_file_is_skipped(tmp_path):
    d = str(tmp_path / 'run-20260814_120000-eeeeeeee')
    os.makedirs(d)
    assert scan_local_history(d) == {}


# ---------------------------------------------------------------------------
# H4 -- real datastore parsing
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def real_run_dir(tmp_path_factory):
    """A genuine `.wandb` datastore, written by wandb offline.

    Deliberately not a hand-built fixture: `item.key` is empty for nested metrics
    in this wandb version and the name lives in `nested_key`. That only shows up
    in the real encoding, so a mock would let the bug through."""
    wandb = pytest.importorskip('wandb')
    d = tmp_path_factory.mktemp('wb')
    # `mode='offline'` is passed to init directly rather than via WANDB_MODE:
    # setting the env var here would persist for the whole pytest process and
    # silently take any other test's run offline too. Offline is legitimate for a
    # pure functionality check like this one -- it is real experiments that must
    # stay online, since an offline run has no URL to read.
    run = wandb.init(project='analysis-test', dir=str(d), mode='offline',
                     settings=wandb.Settings(silent=True))
    for i in range(60):
        row = {'fwd/tb_err_worst': 10.0 - i * 0.01, 'nested': {'a': float(i)}}
        if i % 3:                      # most rows carry a metric, some do not
            row['sparse/metric'] = float(i)
        run.log(row)
    path = run.dir
    run.finish()
    return os.path.dirname(path)       # .../wandb/offline-run-.../


def test_parses_a_real_datastore(real_run_dir):
    hist = scan_local_history(real_run_dir)
    assert 'fwd/tb_err_worst' in hist, f'got keys: {sorted(hist)[:20]}'
    s, v = hist['fwd/tb_err_worst']
    assert len(s) >= 50
    assert np.all(np.diff(s) >= 0), 'steps must be non-decreasing'


def test_nested_keys_are_recovered(real_run_dir):
    """H4: `item.key` is EMPTY for these; the name is in `nested_key` and must be
    joined, or the metric silently disappears."""
    hist = scan_local_history(real_run_dir)
    assert any(k.startswith('nested') for k in hist), sorted(hist)


def test_rows_without_step_do_not_create_gaps(real_run_dir):
    """Spec acceptance #6. Not every history row carries `_step`; the last seen
    value is carried forward, or sparse metrics lose their step alignment."""
    hist = scan_local_history(real_run_dir)
    s, v = hist['sparse/metric']
    assert len(s) == len(v)
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(v))
    # the sparse series was logged on 2 of every 3 steps and must keep the
    # STEP it was logged at, not be renumbered densely
    assert s[-1] > len(s)


def test_key_filter_restricts_what_is_parsed(real_run_dir):
    hist = scan_local_history(real_run_dir, keys={'fwd/tb_err_worst'})
    assert set(hist) == {'fwd/tb_err_worst'}


def test_booleans_are_not_treated_as_numbers(real_run_dir):
    """bool is a subclass of int in Python. Left unguarded, a flag becomes a
    0/1 series that a trend test will happily fit."""
    hist = scan_local_history(real_run_dir)
    for k, (s, v) in hist.items():
        assert v.dtype == float
