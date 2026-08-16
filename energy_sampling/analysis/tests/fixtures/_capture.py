"""Capture real local runs as network-free test fixtures.

Run from the repo root with the project venv, with `wandb/` present:

    python -m analysis.tests.fixtures._capture

The fixtures are REAL RUN DATA, not mocks. The whole package exists because
assumptions about this data were wrong in ways that produced silence rather than
an error, and a hand-built fixture agrees with whatever the author assumed.

Each run is captured as three files:

  <name>.config.json   the flattened config, values UNWRAPPED (the local
                       `config.yaml` wraps each as {'value': x}; the cloud API
                       returns it bare, and `keys._value` handles both -- the
                       fixture keeps the wrapped form so the reader is exercised)
  <name>.summary.json  scalar summary entries only. wandb histogram blobs are
                       dicts of hundreds of bin edges and carry no scalar; they
                       are dropped, and dropping them is what keeps a fixture
                       kilobytes instead of megabytes.
  <name>.history.npz   curated series as float32 (steps, values) pairs.

The curation list is deliberate: everything a check reads, plus the toplines.
Capturing all ~400 series would be ~10x the size for series no check touches.
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# The runs, chosen for what each one PROVES. A fixture set that is all healthy
# TB runs cannot mutation-test anything.
RUNS = {
    # TB/unconditional, terminal stage. Ray calibration FIRED (26 events), z-cal
    # fired, ratio balance controller live, Fwd Frac pinned BY DECLARATION.
    'tb_ramp':      'wandb/run-20260811_003606-5t7ny5lw',
    # Conditional VarGrad. ray_calibration.enabled is true and the probe NEVER
    # calibrated -- `lr_ctrl/calibrations` is pinned at 0 and no `raycal/*`
    # series exists at all. A real inert mechanism, which is what R2 is for.
    'vg_normal':    'wandb/run-20260811_180230-x4rbzv88',
    # Sibling arm of the above: same battery, different config.
    'vg_blowup':    'wandb/run-20260811_173524-iik0dq38',
    # Died in phase 1 -> the MLE/prior route, not the terminal stage's.
    'mle_only':     'wandb/run-20260813_100030-fhqkvc0e',
    # Five stages, and the only fixture carrying `protocol/thr_*` -- the live
    # annealed exit thresholds R13 is about.
    'buildout':     'wandb/run-20260730_081443-bgdu90w4',
    # Resumed from an explicit checkpoint (continue_from_checkpoint: True,
    # checkpoint_name set) -- the §4 chaining confound, on a real run.
    'tb_resumed':   'wandb/run-20260730_110233-44gt5whr',
    # A two-arm battery whose members differ by more than one knob.
    'ring_probe':   'wandb/run-20260802_120301-vlqklgmy',
    'ring_cal':     'wandb/run-20260802_125615-gegz7as8',
}

# Series any check reads, as regexes. Kept here rather than imported from
# keys.py: the capture must not silently narrow when the taxonomy changes --
# a fixture missing the series a new check needs should fail that check's test
# loudly, not hide it.
#
# The mode-namespaced families are restricted BY TAIL rather than kept whole.
# Keeping every `fwd|bwd|replay/*` series made the fixture set 5.2 MB of data no
# check reads; the tails below are the ones a check or a topline names.
_TAILS = (
    'tb_err_worst|tb_err|scatter_err|tb_resid_clipped|tb_resid|over_coverage|'
    'under_coverage|under_coverage_wcen|relative_under|relative_under_wcen|'
    'logw_std_within|logw_std|vg_lb|step_var|terminal_var|z_gap|jensen_z|'
    'log_Z_learned|loss|r2|mle|tbc|birth_loss_mean|ema_loss_mean|'
    'is_elig_frac|is_ess_frac|absorbed_frac|condition_log_z_visited_frac|'
    'cond_tb_err'
)
KEEP = [
    rf'^(fwd|bwd|replay|eval_test)/({_TAILS})$',
    r'^(protocol|lr_ctrl|raycal|z_cal|zmatch|tracker|lr_slope)/',
    r'^loss_coeffs/',
    r'^(Fwd|Bwd|Replay) Frac$',
    r'^(phase|Batch Size|_step|lr_[a-z_]+|log Z learned|train_step_time|samples_per_sec)$',
    r'^(replay|prior)_buffer_(length|turnover|mean_loss|mean_age|mean_energy)$',
    r'^wass',
    r'^grad_norm_pre_clip$',
]
_KEEP = re.compile('|'.join(KEEP))

# Runs kept for their CONFIG. The §4 confound checks are config-level -- code
# version, checkpoint chaining, T vs eval_T, arms differing by omission -- and
# a battery needs several arms, so carrying full history for each of them would
# treble the fixture set to serve checks that never look at a series. They keep
# a token history so `load()` stays uniform and no test reads an empty history
# as 'this run did no work'.
CONFIG_ONLY = {'ring_probe', 'ring_cal', 'tb_resumed'}
_KEEP_TOKEN = re.compile(r'^(phase|_step)$|^(fwd|replay)/scatter_err$')


def _scalar(v):
    """True for entries that are a NUMBER. wandb summary also holds histogram
    dicts and strings; only the numbers are series-like."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def capture(name: str, run_dir: str) -> None:
    from analysis.pull import _local_config, _local_summary, scan_local_history

    cfg = _local_config(run_dir)
    summ = _local_summary(run_dir)
    hist = scan_local_history(run_dir)
    if not cfg or not summ:
        raise SystemExit(f'{name}: {run_dir} has no config/summary')

    # The `_wandb` blob is ~4 KB of host details wrapping the one thing §4 needs:
    # the git commit the run actually executed. Keep that and the argv; drop the
    # disk/cpu/gpu inventory.
    cfg = dict(cfg)
    blob = cfg.get('_wandb')
    if isinstance(blob, dict):
        val = blob.get('value', blob)
        slim = {}
        for k, entry in (val.get('e') or {}).items():
            if isinstance(entry, dict):
                slim[k] = {q: entry[q] for q in ('git', 'args', 'codePath')
                           if q in entry}
        cfg['_wandb'] = {'value': {'cli_version': val.get('cli_version'),
                                   'e': slim}}

    keep_summary = {k: v for k, v in summ.items()
                    if _scalar(v) or isinstance(v, str)}
    pat = _KEEP_TOKEN if name in CONFIG_ONLY else _KEEP
    keep_hist = {k: (s, v) for k, (s, v) in hist.items() if pat.search(k)}

    with open(os.path.join(HERE, f'{name}.config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=0, sort_keys=True, default=str)
    with open(os.path.join(HERE, f'{name}.summary.json'), 'w', encoding='utf-8') as f:
        json.dump(keep_summary, f, indent=0, sort_keys=True, default=str)

    flat = {}
    for k, (s, v) in keep_hist.items():
        # '#' and not '\0': npz names become entries in a zip, and the writer
        # SILENTLY STRIPS a NUL, collapsing the steps and values of one key onto
        # the same name. '#' does not occur in a metric name here.
        flat[f'{k}#s'] = np.asarray(s, np.float32)
        flat[f'{k}#v'] = np.asarray(v, np.float32)
    np.savez_compressed(os.path.join(HERE, f'{name}.history.npz'), **flat)

    size = sum(os.path.getsize(os.path.join(HERE, f'{name}.{p}'))
               for p in ('config.json', 'summary.json', 'history.npz'))
    print(f'{name:12s} {os.path.basename(run_dir):34s} '
          f'cfg {len(cfg):4d}  summary {len(keep_summary):4d}  '
          f'series {len(keep_hist):4d}  {size / 1024:7.1f} KB')


def main():
    if not os.path.isdir('wandb'):
        raise SystemExit('run from the repo root, with wandb/ present')
    for name, d in RUNS.items():
        capture(name, d)


if __name__ == '__main__':
    sys.exit(main())
