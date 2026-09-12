"""The lambda anneal's two rule metrics RESOLVE on var_conditioning.

The anneal fires only off an all-rules-clean streak and both rules are
`if_missing: violated`, so a rule whose metric never reaches the metric tracker
holds lambda at its start value for the whole run. That is what the rules these
replaced did: pooled/* and zmatch/* are logged, never tracked.

Driven on the arm's own stage (configs/qm9c_anneal.yaml, which copies mk_dev's
conditional_vargrad whole):

  get_gfn_forward_loss / get_gfn_backward_loss at var_conditioning's RESOLVED
  coefficients (base block, then the stage's override)
    -> Modeller.fused_train_step (which branches ran, which count as trained)
    -> Modeller.record_fused_substep_losses -> _update_rolling -> quick_tb_stats
    -> MetricTracker
    -> StageProtocol._rule_value / _balance_tick over the parsed stage.

FAKED: the model (trajectories only), the energy, the rollout gates, the replay
buffer and admission, and -- in all but the last two tests -- the three branch
draws, which hand each loss two synthetic rows per condition and leave the
stage's condition_draw out. The last two drive the bwd seam for real:
_choose_draw_conditions -> bwd_train_step -> draw_bwd_sample over a prior buffer
that hands back exactly the rows it is asked for.

Pinned: fwd/logw_std_within is written on a sidecar rollout step (fwd is not a
trained branch there, so _update_rolling runs on every rollout) and on no
off-cadence step; bwd/logw_std_within is written on every 10th bwd step, and only
because the aligned draw hands bwd its groups; both rules then read finite values
and run clean once the entry cooldown is over.
"""
import copy
import math
import pathlib
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import yaml

from energy_sampling.gflownet_losses import get_gfn_backward_loss, get_gfn_forward_loss
from energy_sampling.protocol import StageProtocol, fresh_stage_ctrl
from energy_sampling.train import Modeller
from energy_sampling.utils import MetricTracker, dict2namespace

ARM = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'qm9c_anneal.yaml'
T, D = 20, 12
N_COND = 8
RULE_METRICS = ['fwd/logw_std_within', 'bwd/logw_std_within']


def _cfg():
    return yaml.safe_load(ARM.read_text(encoding='utf-8'))


def _vc(cfg):
    return cfg['protocols']['conditional_vargrad']['stages'][1]


def _resolved(cfg, mode):
    """A branch's effective coefficients: the base block, then the stage's override."""
    block = dict(cfg[f'{mode}_loss_coeffs'])
    block.update(_vc(cfg)['loss_coeffs'].get(mode, {}))
    return SimpleNamespace(**block)


class _Gfn:
    """Trajectories only. Random log-probabilities spread log w within a
    condition the way a real batch does; the stash stays unarmed, so the pooled
    term reads INERT and adds nothing."""
    device = 'cpu'
    _stash_live_branches = False

    @staticmethod
    def _traj(n):
        return (torch.randn(n, T + 1, D), torch.randn(n, T), torch.randn(n, T),
                torch.randn(n, T + 1))

    def get_traj_fwd(self, initial_state, *a, **kw):
        return self._traj(initial_state.shape[0])

    def get_traj_bwd(self, samples, *a, **kw):
        return self._traj(samples.shape[0])


def _pairs():
    """Two rows per condition: what fwd repeats 2 and condition_draw's
    prior_rows/replay_rows 2 hand each branch."""
    return torch.arange(N_COND).repeat_interleave(2)


def _fwd_loss(coeffs):
    cids = _pairs()
    n = cids.numel()
    return get_gfn_forward_loss(
        coeffs, torch.zeros(n, D), _Gfn(), lambda x, *a, **kw: -x.pow(2).sum(-1), None, None,
        torch.zeros(n), condition=torch.zeros(n, 1), repeats=int(coeffs.repeats),
        report_losses=True, condition_id=cids, tb_z_source=coeffs.tb_z_source)


def _bwd_loss(coeffs):
    cids = _pairs()
    n = cids.numel()
    return get_gfn_backward_loss(
        coeffs, torch.zeros(n, D), _Gfn(), -10.0 * torch.rand(n), None, None,
        condition=torch.zeros(n, 1), repeats=int(coeffs.repeats), report_losses=True,
        condition_id=cids, tb_z_source=coeffs.tb_z_source, live_stash=None)


def _modeller(cfg):
    fwd_c, bwd_c, rep_c = (_resolved(cfg, mode) for mode in ('fwd', 'bwd', 'replay'))
    vc = _vc(cfg)
    stage = SimpleNamespace(name=vc['name'], deactivate_threshold=vc['deactivate_threshold'],
                            fwd_z_sidecar=vc['fwd_z_sidecar'])
    m = SimpleNamespace(
        args=SimpleNamespace(
            controller=SimpleNamespace(deactivate_threshold=cfg['controller']['deactivate_threshold'],
                                       refresh_every=cfg['controller']['refresh_every']),
            fwd_loss_coeffs=fwd_c, bwd_loss_coeffs=bwd_c, replay_loss_coeffs=rep_c,
            conditional_worst_quantile=cfg['conditional_worst_quantile']),
        protocol=SimpleNamespace(stage=stage, mode_boostable=lambda mode: True,
                                 mode_dormant=lambda mode: False),
        gfn_model=SimpleNamespace(live_idx=None), replay_buffer=[0] * 4,
        fwd_frac=0.0, bwd_frac=0.5, replay_frac=0.5, fused_step_count=0,
        fwd_step_count=0, bwd_step_count=0, replay_step_count=0,
        metric_tracker=MetricTracker(period=100), step_ind=0, _last_stats={},
        _UC_WINDOW_STEPS=Modeller._UC_WINDOW_STEPS,
        _FORGETTING_CHANNELS=Modeller._FORGETTING_CHANNELS)

    def fwd_step(discretizer, return_exp, repeats, report_losses):
        loss, d = _fwd_loss(fwd_c)
        return loss, 'crystal_batch', d

    m.fwd_train_step = fwd_step
    m.bwd_train_step = lambda discretizer, repeats, report_losses, target_cids: _bwd_loss(bwd_c)
    m.replay_train_step = lambda discretizer, repeats, report_losses: _bwd_loss(rep_c)
    m.mode_repeats = lambda mode: 1
    m._stash_z_fill_logw = lambda d: None
    m._fused_grad_diag_armed = lambda: False
    m.manage_replay_buffer = lambda d, b: None
    for name in ('fused_train_step', 'record_fused_substep_losses', '_update_rolling',
                 '_per_step_probe', '_reward_ramp_kwargs', '_forgetting_sensor'):
        setattr(m, name, MethodType(getattr(Modeller, name), m))
    return m


def _step(m, step, rollout):
    """One fused step as train_step runs it: the step, then the rolling stats."""
    m.step_ind = step
    m._fwd_gates = lambda deact, force_refresh: (rollout, False, False)
    _, subs = m.fused_train_step(None)
    m.record_fused_substep_losses(subs)
    return subs


def _protocol(cfg, tracker):
    pm = SimpleNamespace(args=dict2namespace(copy.deepcopy(cfg)), stage='var_conditioning',
                         stage_ctrl=fresh_stage_ctrl(), metric_tracker=tracker, step_ind=0,
                         fwd_frac=0.0, bwd_frac=0.5, replay_frac=0.5)
    return StageProtocol(pm), pm


def _ten_steps(cfg):
    """A rollout at step 0 (fwd repeats 2 under the sidecar), then nine ordinary
    steps: bwd's tenth trained step lands on step 9."""
    m = _modeller(cfg)
    subs = [_step(m, step, rollout=(step == 0)) for step in range(10)]
    return m, subs


def test_a_sidecar_rollout_writes_fwd_logw_std_within():
    m = _modeller(_cfg())
    subs = _step(m, 0, rollout=True)
    assert subs['fwd'][2] is False, 'the sidecar is not a trained branch of the mix'
    v = m.metric_tracker.get('fwd', 'logw_std_within')
    assert v is not None and math.isfinite(v)
    assert m.metric_tracker.written_step('fwd', 'logw_std_within') == 0


def test_ordinary_steps_write_bwd_on_every_tenth_bwd_step_and_never_fwd():
    m, subs = _ten_steps(_cfg())
    for step, s in enumerate(subs[1:], start=1):
        assert 'fwd' not in s and s['bwd'][2] is True, step
    assert m.bwd_step_count == 10
    assert m.metric_tracker.written_step('bwd', 'logw_std_within') == 9
    assert m.metric_tracker.written_step('fwd', 'logw_std_within') == 0, 'rollout steps only'


def test_both_rules_resolve_and_run_clean_past_the_entry_cooldown():
    cfg = _cfg()
    m, _ = _ten_steps(cfg)
    p, pm = _protocol(cfg, m.metric_tracker)
    rules = p.stage.balance['rules']
    assert [r['metric'] for r in rules] == RULE_METRICS
    for i, rule in enumerate(rules):
        v = p._rule_value(rule, p._rule_state(i))
        assert v is not None and math.isfinite(v), rule['metric']
    cooldown = p.stage.balance['anneal_cooldown_steps']
    assert cooldown > 0
    for step in (0, cooldown):          # the entry stamp, then the first tick past it
        pm.step_ind = step
        p._balance_tick()
    assert p.ctrl['anneal_streak'] == 1, 'both rules read, both clean'
    assert all(p.ctrl['rules'][i].get('best') is not None for i in range(len(rules)))


def test_the_replaced_rule_metrics_never_reach_the_tracker():
    """Why the rules moved: pooled/* and zmatch/* are logged, never tracked, so
    under if_missing: violated they held the anneal forever."""
    cfg = _cfg()
    m, _ = _ten_steps(cfg)
    p, _ = _protocol(cfg, m.metric_tracker)
    for name in ('pooled/pooled_vg', 'zmatch/delta_mean'):
        assert p._resolve(name) is None, name
    for name in RULE_METRICS:
        assert p._resolve(name) is not None, name


# ------------------------------------------------------------ the bwd seam

class _MolBatch:
    """The rows a draw hands back: their condition ids, nothing else."""

    def __init__(self, cid):
        self.cid = cid

    def to(self, device):
        return self


class _Buffer:
    """condition_row_counts at 2 rows per condition, and a loader that hands back
    exactly the rows it is asked for: rows_per_condition of each draw_cid under
    the aligned draw, batch_size singletons under any other draw."""

    def __init__(self):
        self.calls = []

    def __len__(self):
        return 2 * N_COND

    def condition_row_counts(self, minlength=0):
        return np.full(N_COND, 2, dtype=np.int64)

    def loader(self, **kw):
        self.calls.append(kw)
        if kw.get('draw_cids') is not None:
            cid = torch.as_tensor(np.repeat(kw['draw_cids'], kw['rows_per_condition']))
        else:
            cid = torch.arange(kw['batch_size'])
        yield _MolBatch(cid), np.arange(cid.numel())

    def update_losses(self, priority, inds):
        pass


def _seam_modeller(cfg, draw):
    """_modeller, with the stage's condition_draw on and the REAL
    _choose_draw_conditions -> bwd_train_step -> draw_bwd_sample chain."""
    m = _modeller(cfg)
    bwd_c = m.args.bwd_loss_coeffs
    m.protocol.stage.condition_draw = draw
    m.protocol.flag = lambda name: False
    m.prior_buffer, m.replay_buffer = _Buffer(), _Buffer()
    m.gfn_model = _Gfn()
    m.gfn_model.live_idx = None
    m.batch_size, m.device, m.bwd_sampling_mode = 2 * N_COND, 'cpu', 'prior'
    m.condition_log_z = None
    m.tb_z_source = lambda mode: bwd_c.tb_z_source
    m._batch_latents = lambda mb: torch.zeros(mb.cid.numel(), D)
    m._bwd_retention_priority = lambda d: d['resid'].abs()
    m._larder_harvest = lambda *a, **kw: None
    m.energy_function = SimpleNamespace(
        n_sg=1, n_zp=1, condition_library_size=N_COND,
        condition_samples=lambda mb, repeats: (mb, torch.zeros(mb.cid.numel()),
                                               torch.zeros(mb.cid.numel(), 1), mb.cid),
        prebuilt_sample_to_reward=lambda mb, temperature: -10.0 * torch.rand(mb.cid.numel()))
    m.replay_train_step = \
        lambda discretizer, repeats, report_losses, **kw: _bwd_loss(m.args.replay_loss_coeffs)
    for name in ('_choose_draw_conditions', 'bwd_train_step', 'draw_bwd_sample'):
        setattr(m, name, MethodType(getattr(Modeller, name), m))
    return m


def test_bwd_logw_std_within_is_fed_by_the_aligned_draw():
    """THE SEAM. bwd has repeats 1 and condition_block_m 0 here, so its groups
    exist only because bwd_train_step hands the fused step's decision on to
    draw_bwd_sample. Drop it and bwd draws singletons, quick_tb_stats omits the
    key, and under if_missing: violated the bwd rule holds lambda at its start."""
    cfg = _cfg()
    m = _seam_modeller(cfg, _vc(cfg)['condition_draw'])
    for step in range(10):
        _step(m, step, rollout=(step == 0))
    assert len(m.prior_buffer.calls) == 10
    for kw in m.prior_buffer.calls:
        assert kw['rows_per_condition'] == 2
        assert sorted(kw['draw_cids'].tolist()) == list(range(N_COND))
    assert m.metric_tracker.written_step('bwd', 'logw_std_within') == 9


def test_singleton_bwd_draws_leave_the_bwd_rule_metric_unwritten():
    """What the seam test guards: the same ten steps at one prior row per
    condition train bwd every step and never write the key."""
    cfg = _cfg()
    m = _seam_modeller(cfg, dict(_vc(cfg)['condition_draw'], prior_rows=1))
    for step in range(10):
        _step(m, step, rollout=(step == 0))
    assert m.bwd_step_count == 10
    assert m.metric_tracker.written_step('bwd', 'logw_std_within') is None
