"""Where generated artifacts live, resolved from this file rather than the CWD.

THE PROBLEM THIS EXISTS TO STOP. A writer that defaults to a bare filename --
`torch.save(x, 'conformer_buffer_CCCCO.pt')` -- resolves it against the CWD, and the
CWD is always `energy_sampling/` because that is where every run is launched from. So
the package root fills with loose artifacts, and moving them is useless: the next build
run puts them straight back. Settled 2026-08-16, re-broken by seven writers since.

ANCHORED TO THIS FILE, NOT THE CALLER'S. `ARTIFACTS` is derived from `paths.py`'s own
location, so it keeps naming the same directory when a producer moves into a
subpackage. A caller that computed `Path(__file__).parent / 'artifacts'` for itself
would silently start writing one level down the day it moved -- which is the same
fixed-depth `dirname()` bug that already sits in three of the build scripts.

WHAT IS NOT REDIRECTED. An explicit path -- absolute, or carrying any directory
component -- passes through untouched, so a `--out` flag and a config value still mean
exactly what they say. Only a bare name is claimed. Consumers that must READ an
artifact resolve it the same way, which is the repo's rule that resolution belongs in
the consumer and not in the user-owned yaml.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'artifacts'


def artifact(name) -> Path:
    """Resolve a default output name against ARTIFACTS, passing explicit paths through.

    Creates the directory, so a writer never fails on a missing parent -- the same
    shape as `bench/calibrate_noise.py`'s `_resolve`.
    """
    p = Path(name)
    if p.is_absolute() or len(p.parts) > 1:
        return p
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS / p.name
