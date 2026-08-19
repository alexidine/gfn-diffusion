"""
Tier 3 -- the figures a run already produced, indexed so a headless reader can
ask for one by name and step.

WHAT THIS IS NOT. It does not plot anything. wandb already rendered these; the
only problem is that they are hundreds of files whose names encode what they are,
in a layout nobody can navigate from a terminal. A run with 30 media files has
about four distinct figures at several steps each, and what a reader wants is
"the newest parity plot", not a directory listing.

WHY IT IS LOWEST PRIORITY, AND WHO IT IS FOR. The user reads figures in the wandb
UI, where this problem does not exist. This exists for an agent reading a run it
cannot click through -- which is the position anyone is in when a cluster battery
lands and the question is what the arms did.

THE FILENAME IS THE INDEX:

    files/media/images/<name>_<step>_<hash>.png
    files/media/plotly/<name>_<step>_<hash>.plotly.json

The step in that name is a wandb media step, NOT necessarily the trainer's
`_step`: wandb stamps it from the logging call's own counter. Treat it as an
ordering key within one figure name, and read the axis off the figure itself
before quoting a number against training progress.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from .pull import CACHE_DIR, Run

#: `<name>_<step>_<hash>` with the hash a hex blob wandb appends for uniqueness.
#: Anchored on the tail so a name containing underscores or digits survives --
#: 'Backward TB Parity Plot' and 'Forward Lattice Latents Trajectories' both do.
_MEDIA = re.compile(r'^(?P<name>.+)_(?P<step>\d+)_(?P<hash>[0-9a-f]{8,})$')

_KINDS = {'images': '.png', 'plotly': '.plotly.json'}


@dataclass(frozen=True)
class Figure:
    name: str
    step: int
    kind: str                 # 'images' | 'plotly'
    path: Optional[str]       # local path, or None until fetched
    remote: Optional[str] = None      # wandb file name, for a cloud run

    @property
    def fetched(self) -> bool:
        return bool(self.path) and os.path.exists(self.path)


def _parse(fname: str, kind: str) -> Optional[tuple[str, int]]:
    """(name, step) from a media filename, or None if it does not match.

    Returns None rather than guessing. An unrecognised name means wandb changed
    its convention, and inventing a name for it would put a figure in the index
    under something no caller can ask for."""
    stem = fname[: -len(_KINDS[kind])] if fname.endswith(_KINDS[kind]) else None
    if stem is None:
        return None
    m = _MEDIA.match(stem)
    if not m:
        return None
    return m.group('name'), int(m.group('step'))


def index(run: Run) -> dict[str, list[Figure]]:
    """{figure name: [Figure, ...]} ordered by step, oldest first.

    Local runs index off disk. Cloud runs index off the file MANIFEST without
    downloading: listing is cheap, fetching is not, and a reader almost always
    wants one figure out of hundreds."""
    out: dict[str, list[Figure]] = {}

    def add(name, step, kind, path=None, remote=None):
        out.setdefault(name, []).append(Figure(name, step, kind, path, remote))

    if run.source == 'local' and run.path:
        for kind in _KINDS:
            d = os.path.join(run.path, 'files', 'media', kind)
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                parsed = _parse(fname, kind)
                if parsed:
                    add(*parsed, kind, path=os.path.join(d, fname))
    else:
        for f in _cloud_files(run):
            parts = f.name.split('/')
            if len(parts) < 3 or parts[0] != 'media' or parts[1] not in _KINDS:
                continue
            parsed = _parse(parts[-1], parts[1])
            if parsed:
                add(*parsed, parts[1], path=_cached_path(run, f.name), remote=f.name)

    for figs in out.values():
        figs.sort(key=lambda f: f.step)
    return out


def _cloud_files(run: Run):
    """The run's file manifest. Isolated so `index` stays testable offline."""
    import wandb
    api = wandb.Api()
    return api.run(f"{run.config.get('_project', '')}/{run.run_id}".lstrip('/')).files()


def _cached_path(run: Run, remote_name: str) -> str:
    return os.path.join(CACHE_DIR, 'figures', run.run_id, remote_name.replace('/', os.sep))


def latest(run: Run, name: str) -> Optional[Figure]:
    """The newest version of one figure, or None if the run never logged it.

    None is a real answer and is distinct from 'the figure exists but is empty'.
    A run that never reached the stage which logs a parity plot has no parity
    plot, and reporting that as a missing file would read as a fetch failure."""
    figs = index(run).get(name)
    return figs[-1] if figs else None


def fetch(fig: Figure, force: bool = False) -> Optional[str]:
    """Ensure `fig` is on disk; return its path.

    A local run is already on disk and is returned untouched -- never copied
    into the cache, because two paths for one file is how a reader ends up
    looking at the stale one."""
    if fig.fetched and not force:
        return fig.path
    if not fig.remote:
        return fig.path if fig.fetched else None
    raise NotImplementedError(
        'cloud figure download is not wired: `index` lists a cloud run from its '
        'manifest, which is cheap, but pulling bytes needs a wandb file handle '
        'this module does not hold. Wire it when a cloud run is the thing being '
        'read -- and until then this raises rather than returning a path to a '
        'file that is not there.')


def render(run: Run, limit: int = 12) -> str:
    """A terminal index: one line per figure name, with the steps available."""
    figs = index(run)
    if not figs:
        return 'figures: none logged'
    lines = [f'figures: {len(figs)} distinct, '
             f'{sum(len(v) for v in figs.values())} files']
    for name in sorted(figs):
        steps = [f.step for f in figs[name]]
        shown = steps if len(steps) <= 6 else [steps[0], steps[1], '...', steps[-1]]
        lines.append(f"  {name:<44} {figs[name][0].kind:<7} "
                     f"steps {', '.join(str(s) for s in shown)}")
        if len(lines) > limit:
            lines.append(f'  ... {len(figs) - limit + 1} more')
            break
    return '\n'.join(lines)
