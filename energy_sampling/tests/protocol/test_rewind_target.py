"""
LR-divergence rewinds target a FRESH SAME-STAGE 'best' first (owner decision
2026-08-25, reversing 2026-08-24's rolling-first).

'running' is <= 50 steps old but written UNCONDITIONALLY, so during a slow
excursion it carries the excursion's onset -- the qm9c fire cascade rewound
through three rolling checkpoints and still landed ~70 nats up. The
phase-dependent 'best' does not advance while its stage's health metric
worsens, so a fresh best predates the incident by construction. The old
objection (a frozen near-init best re-tripping the bar every 2 steps,
toy_wk_aug24) is handled by the freshness bound: stale past 10 x eval_period
and 'best' demotes below 'running' again. Cross-stage guards unchanged: never
reverse the phase (stab_july21c 512x6_T60).

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


def test_a_fresh_same_stage_best_beats_the_rolling_checkpoint(tmp_path):
    """THE DOCTRINE: a fresh phase-correct quality record predates the incident
    by construction; the rolling checkpoint may carry the excursion's onset."""
    pick = modeller(tmp_path, step_ind=5000,
                    tags={'best': ('s1', 4999),
                          'running': ('s1', 4950)})
    assert pick().endswith('best.pt')


def test_a_stale_best_demotes_below_the_rolling_checkpoint(tmp_path):
    """Past 10 x eval_period (2500 here) the old objection applies again: a
    lagging best is a quality record, not a recovery point."""
    pick = modeller(tmp_path, step_ind=5000,
                    tags={'best': ('s1', 2000),
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

def test_the_best_selector_is_phase_dependent():
    """Owner 2026-08-25: each stage's 'best' is judged by what that stage
    optimizes -- the global tb_err combo sat flat through a 20x vg_lb
    excursion on var_conditioning, so 'best' advanced into poisoned state."""
    from train import Modeller

    def channels(train_mode, name):
        m = SimpleNamespace(protocol=SimpleNamespace(
            stage=SimpleNamespace(train_mode=train_mode, name=name)))
        return Modeller._best_metric_channels(m)

    assert channels('bwd', 'train_prior') == (('bwd', 'mle'),)
    assert channels('fused', 'var_conditioning') == (('fwd', 'logw_std_within'),)
    assert channels('fused', 'equilibration') == (
        ('fwd', 'tb_err_worst'), ('bwd', 'tb_err_worst'))
