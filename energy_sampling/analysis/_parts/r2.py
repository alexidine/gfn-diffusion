"""R2 -- confirm the thing ever fired. STUB: see the build brief."""

from __future__ import annotations

from typing import Optional

from .. import keys as K
from .base import (CheckResult, Context, Finding, State, context, count_active,
                   declared, series, stage_value, trailing)


def check_r2(run, *, ctx: Optional[Context] = None,
             window: Optional[float] = None) -> CheckResult:
    raise NotImplementedError('r2')
