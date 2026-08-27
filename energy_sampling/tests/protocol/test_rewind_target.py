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
            stage=SimpleNamespace(train_mode=train_mode, name=name, balance=None)))
        return Modeller._best_metric_channels(m)

    assert channels('bwd', 'train_prior') == (('bwd', 'mle', 1.0),)
    assert channels('fused', 'var_conditioning') == (('fwd', 'logw_std_within', 1.0),)
    assert channels('fused', 'equilibration') == (
        ('fwd', 'tb_err_worst', 1.0), ('bwd', 'tb_err_worst', 1.0))

def test_a_metric_change_clears_the_best_record():
    """qm9c selC, 2026-08-25: a record restored from a checkpoint written under
    the OLD global combo (~60, tb_err scale) made every new-metric sample
    (~26, logw_std scale) a fresh record -- 'best' re-linked to 'running'
    every 50 steps and degenerated into it, so three fire rewinds all restored
    the excursion onset. Values under different metrics are not comparable;
    the record resets when the signature changes, and the signature is
    checkpointed (old checkpoints restore None and self-heal)."""
    from train import Modeller

    m = SimpleNamespace(
        _nonfinite_pending=False,
        metric_tracker=SimpleNamespace(get=lambda mode, key: 26.0),
        protocol=SimpleNamespace(stage=SimpleNamespace(
            train_mode='fused', name='var_conditioning', balance=None)),
        combo_loss_record=[60.0, 58.0],   # old-metric values off a checkpoint
        combo_loss_metric=None,           # what an old checkpoint restores
    )
    m._best_metric_channels = lambda: Modeller._best_metric_channels(m)
    Modeller.monitor_losses(m, 1.0, 'fused')
    assert m.combo_loss_metric == '1*fwd/logw_std_within'
    assert m.combo_loss_record == [26.0], (
        'the old-metric record survived the switch; every new sample beats its '
        "min and 'best' degenerates to 'running'")
    # same metric next tick: the record accumulates normally
    Modeller.monitor_losses(m, 1.0, 'fused')
    assert m.combo_loss_record == [26.0, 26.0]


def test_a_balance_block_supplies_the_score_weights():
    """Owner 2026-08-27: where the controller has already declared an exchange
    rate between branches -- targets {fwd: 5, bwd: 1} says "fwd at 5 is as good
    as bwd at 1" -- that IS the normalisation that makes two branch metrics
    commensurable, so 'best' (and with it 'last_ok') scores by sum(metric/target).

    The single-channel var_conditioning selector it replaces was blind to bwd,
    and fwd/logw_std_within is exactly the metric MODE COLLAPSE improves: a
    policy narrowing into safe modes lowered fwd spread and advanced 'best'
    into the collapsed state."""
    from train import Modeller

    stage = SimpleNamespace(train_mode='fused', name='var_conditioning', balance={
        'kind': 'proportional',
        'metrics': {'fwd': 'fwd/logw_std_within', 'bwd': 'bwd/logw_std_within'},
        'targets': {'fwd': 5.0, 'bwd': 1.0}})
    m = SimpleNamespace(protocol=SimpleNamespace(stage=stage))
    assert Modeller._best_metric_channels(m) == (
        ('bwd', 'logw_std_within', 1.0), ('fwd', 'logw_std_within', 0.2))

    # the incident's real numbers: healthy 17/1.9 vs the peak 117/2
    score = lambda f, b: f * 0.2 + b * 1.0
    assert abs(score(17.0, 1.9) - 5.3) < 0.01
    assert abs(score(117.0, 2.0) - 25.4) < 0.01

    # a ratio-style balance declares a setpoint, not targets -> falls back
    stage.balance = {'kind': 'ratio', 'metrics': {'replay': 'fwd/over_coverage'},
                     'setpoint': 5.0}
    assert Modeller._best_metric_channels(m) == (('fwd', 'logw_std_within', 1.0),)


def test_changing_a_balance_target_invalidates_the_record():
    """A target change rescales the score, so old samples are not comparable --
    the same hazard as a stage transition (qm9c selC: a record carried across a
    metric change made every new sample a 'record', so 'best' degenerated into
    'running'). The weights are therefore part of the signature."""
    from train import Modeller

    def sig_for(t_fwd):
        stage = SimpleNamespace(train_mode='fused', name='var_conditioning', balance={
            'kind': 'proportional',
            'metrics': {'fwd': 'fwd/logw_std_within', 'bwd': 'bwd/logw_std_within'},
            'targets': {'fwd': t_fwd, 'bwd': 1.0}})
        m = SimpleNamespace(
            _nonfinite_pending=False,
            metric_tracker=SimpleNamespace(get=lambda mode, key: 20.0),
            protocol=SimpleNamespace(stage=stage),
            combo_loss_record=[], combo_loss_metric=None)
        m._best_metric_channels = lambda: Modeller._best_metric_channels(m)
        Modeller.monitor_losses(m, 1.0, 'fused')
        return m.combo_loss_metric

    assert sig_for(1.0) != sig_for(5.0), (
        'the signature must move with the targets, or a 1:1 -> 5:1 switch '
        'compares values on two different scales')


def test_a_ratio_balance_supplies_weights_too():
    """Owner 2026-08-27: equilibration's metrics are balanced in a similar way.
    A ratio balance states the same exchange rate a proportional target does,
    just differently -- numerator `replay` with setpoint 5 means
    over_coverage / relative_under_wcen -> 5, i.e. over-coverage at 5 is as
    good as under-coverage at 1. Scoring on it makes 'best' a two-sided
    COVERAGE score (the two directions of distributional mismatch) rather than
    tb_err_worst, a calibration error that sat FLAT through a 20x vg_lb
    excursion on the neighbouring stage.

    ⚠ These read against the learned head, so they mean what they say only once
    Z has converged (owner). That does not favour the previous selector:
    tb_err_worst is Z-dependent too; only logw_std_within is Z-free."""
    from train import Modeller

    stage = SimpleNamespace(train_mode='fused', name='equilibration', balance={
        'kind': 'ratio', 'pinned': {'fwd': 0.05},
        'metrics': {'replay': 'fwd/over_coverage', 'bwd': 'bwd/relative_under_wcen'},
        'numerator': 'replay', 'setpoint': 5.0})
    m = SimpleNamespace(protocol=SimpleNamespace(stage=stage))
    assert Modeller._best_metric_channels(m) == (
        ('bwd', 'relative_under_wcen', 1.0), ('fwd', 'over_coverage', 0.2))

    # a setpoint of 0/None is malformed -> fall back rather than divide by it
    stage.balance = dict(stage.balance, setpoint=0.0)
    assert Modeller._best_metric_channels(m) == (
        ('fwd', 'tb_err_worst', 1.0), ('bwd', 'tb_err_worst', 1.0))

    # a numerator naming a branch that has no metric -> fall back
    stage.balance = {'kind': 'ratio',
                     'metrics': {'replay': 'fwd/over_coverage', 'bwd': 'bwd/x'},
                     'numerator': 'nope', 'setpoint': 5.0}
    assert Modeller._best_metric_channels(m) == (
        ('fwd', 'tb_err_worst', 1.0), ('bwd', 'tb_err_worst', 1.0))
