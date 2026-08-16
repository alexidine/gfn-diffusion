"""
Entry point: `python -m analysis <run-spec>`.

Prints the resolved-key table first, then the feature report in reading_runs.md
§1 read order. The key table comes first deliberately -- a report whose holes are
invisible reads as a complete picture of a run, and on this project a metric that
is absent, and one that is present but not meaningful on the route, are different
problems requiring different responses.

No verdicts. The report ends where judgment begins.
"""

from __future__ import annotations

import argparse
import sys

from . import features as F
from . import keys as K
from .pull import DEFAULT_PROJECT, CONFORMER_PROJECT, EmptyPull, list_local, pull


def _print_key_table(resolutions, route):
    live = [r for r in resolutions if r.state is K.KeyState.LIVE]
    absent = [r for r in resolutions if r.state is K.KeyState.ABSENT]
    na = [r for r in resolutions if r.state is K.KeyState.NA_ROUTE]
    renamed = [r for r in live if r.resolved_to]

    print(f'\nKEYS  route={route.value}  '
          f'live {len(live)} | absent {len(absent)} | na-on-route {len(na)}')
    for r in renamed:
        print(f'  RENAMED   {r.wanted}  ->  {r.resolved_to}')
    for r in na:
        print(f'  NA_ROUTE  {r.wanted}  ({r.note})')
    for r in absent:
        print(f'  ABSENT    {r.wanted}  ({r.note})')
    if na:
        print('  NB na-on-route keys ARE populated. They are withheld because they '
              'do not track on this route, not because they are missing.')


def _report(run, route, window):
    order = list(K.READ_ORDER)
    topline = K.TOPLINE[route]
    order.insert(0, ('TOPLINE', topline))

    shown = set()
    for group, wanted in order:
        res = K.resolve(run.available_keys(), wanted, route)
        rows, withheld = [], []
        for r in res:
            # An NA_ROUTE key must be NAMED in the group where it would have
            # appeared. Filtering it out silently is the same failure as
            # rendering it as zero: the reader sees a section with a metric
            # missing and no reason given, and fills the gap with an assumption.
            if r.state is K.KeyState.NA_ROUTE:
                withheld.append(r.wanted)
                continue
            key = r.key
            if key is None or key not in run.history or key in shown:
                continue
            s, v = run.history[key]
            feat = F.extract(
                key, s, v, window,
                is_ema=K.is_ema(key),
                low_trust=key in K.LOW_TRUST,
                watch_escape=key in K.ESCAPE_KEYS)
            if feat is not None:
                rows.append(feat)
                shown.add(key)
        if rows or withheld:
            print(f'\n{group}')
            for f in rows:
                print(F.format_feature(f))
            for w in withheld:
                print(f'  {w:30s}  NA on {route.value} -- logged and populated, '
                      f'but does not track here')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='analysis')
    ap.add_argument('spec', nargs='?', default='newest',
                    help="'newest', a local run dir, a run id, name, or tag")
    ap.add_argument('--window', type=float, default=6000,
                    help='trailing window in steps')
    ap.add_argument('--project', default=DEFAULT_PROJECT)
    ap.add_argument('--conformers', action='store_true',
                    help=f'use {CONFORMER_PROJECT}')
    ap.add_argument('--list', action='store_true', help='list local runs and exit')
    ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args(argv)

    if a.list:
        for d in list_local():
            print(d)
        return 0

    project = CONFORMER_PROJECT if a.conformers else a.project
    try:
        run = pull(a.spec, project=project, use_cache=not a.no_cache)
    except EmptyPull as e:
        print(f'EMPTY PULL: {e}', file=sys.stderr)
        return 2
    except LookupError as e:
        print(f'{e}', file=sys.stderr)
        return 2

    # Classify against the stage the run is ACTUALLY in, not the last declared
    # one: a run that died in train_prior is on the MLE route, and reading it
    # with the terminal stage's topline describes a stage it never reached.
    stages = K.stage_names(run.config)
    stage_idx = K.current_stage_index(run.summary, run.config)
    route = K.detect_route(run.config, stage_idx)
    stage = K.current_stage(run.summary, run.config)

    print(f'{run.name}  [{run.source}]  id={run.run_id}')
    print(f'  last step {run.last_step:.0f}  |  window {a.window:.0f}  |  '
          f'{len(run.history)} scalar series')
    print(f'  stages {stages}  current={stage or "unknown"}')

    _print_key_table(K.resolve(run.available_keys(), K.TOPLINE[route], route), route)
    _report(run, route, a.window)
    print('\n  legend: * significant trend  ~ EMA (significance suppressed)  '
          '! low-trust')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
