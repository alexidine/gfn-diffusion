"""
LR-divergence rewinds target the ROLLING checkpoint (owner decision 2026-08-24).

'best' is a QUALITY record, not a recovery point: its selector can legitimately
lag far behind the present (the old r2 form froze at ~step 300 on every warm
start, and rewinding into that near-init state re-tripped the excursion bar
every 2 steps until the budget aborted the run -- toy_wk_aug24). 'running' is
<= 50 steps old and saved before the bar fired, so it is the most recent state
known to predate the incident. Cross-stage guards unchanged: never reverse the
phase (stab_july21c 512x6_T60).

`pytest tests/protocol/test_rewind_target.py -q`
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.fast


def modeller(tmp_path, step_ind, tags):
    """A minimal fake carrying only what _rewind_checkpoint_path reads, plus
    real checkpoint files for `tags` = {tag: (stage_name, step)}."""
    from train import Modeller

    for tag, (stage, step) in tags.items():
        torch.save({'modeller_state': {'stage': stage, 'step_ind': step}},
                   tmp_path / f'{tag}.pt')
    m = SimpleNamespace(
        protocol=SimpleNamespace(
            stages=[SimpleNamespace(name='s0', index=0),
                    SimpleNamespace(name='s1', index=1)],
            stage=SimpleNamespace(name='s1')),
        checkpointer=SimpleNamespace(path_for=lambda tag: str(tmp_path / f'{tag}.pt')),
        step_ind=step_ind,
        args=SimpleNamespace(eval_period=250),
    )
    return lambda: Modeller._rewind_checkpoint_path(m)


def test_rolling_is_preferred_even_over_a_fresh_best(tmp_path):
    """THE DOCTRINE: recency wins for LR incidents; quality records are for
    quality, not recovery."""
    pick = modeller(tmp_path, step_ind=5000,
                    tags={'best': ('s1', 4999),
                          'running': ('s1', 4950)})
    assert pick().endswith('running.pt')


def test_prior_stage_running_defers_to_stage_start(tmp_path):
    """The cross-stage guard: a rolling checkpoint from an EARLIER stage must
    not reverse the phase; this stage's start is the recovery point."""
    pick = modeller(tmp_path, step_ind=5000,
                    tags={'running': ('s0', 4950),
                          'stage_start': ('s1', 4000)})
    assert pick().endswith('stage_start.pt')


def test_best_is_the_last_resort(tmp_path):
    pick = modeller(tmp_path, step_ind=5000, tags={'best': ('s1', 300)})
    assert pick().endswith('best.pt')


def test_prior_stage_best_never_reverses_the_phase_silently(tmp_path):
    """Nothing same-stage exists at all: fall through to bare best (never worse
    than having no rule), even from a prior stage -- the legacy fallback."""
    pick = modeller(tmp_path, step_ind=5000, tags={'best': ('s0', 4999)})
    assert pick().endswith('best.pt')


def test_nothing_on_disk_returns_none(tmp_path):
    pick = modeller(tmp_path, step_ind=5000, tags={})
    assert pick() is None
