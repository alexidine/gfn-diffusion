"""The conditional arm, configs/qm9c_anneal.yaml (written by
configs/qm9c_anneal/make.py), and mk_dev's replay-seat var_conditioning, which
the arm copies whole.

What is pinned here:
  - the arm is CURRENT: its conditional_vargrad equals mk_dev's, so an edit to
    mk_dev's stage without re-running make.py fails here;
  - it LOADS: no invariant ERROR, the real loader (preflight_config ->
    resolve_derived_config), and the trainer's step-0 refusal
    (Modeller.set_loss_coeffs over every stage of the live protocol);
  - it is a FRESH start that reaches var_conditioning's on_enter: a weights-only
    load, prior_model_name set, and train_prior first with skip_if prior_loaded,
    so protocol.begin() enters var_conditioning through advance();
  - its route globals are make.GLOBALS, and every "CONDITIONAL ARM:" note in
    mk_dev names exactly those keys at those values;
  - fwd has no frac on the stage: min_fracs.fwd 0 holds the lexicographic nudge
    at exactly 0, where the controller.min_mode_frac fallback would lift it;
  - on mk_dev's own globals the stage is REFUSED at step 0 (the prioritised
    replay draw), and the one global problems.yaml's qm9_conditional carries
    clears it;
  - the batch is pinned and the replay cap is ~2x the equilibrium occupancy,
    whose premise -- churn_rate 0 admits the live batch_size rows per manage
    call, not batch_size x repeats -- is checked on the real manage_replay_buffer.
"""
import ast
import copy
import pathlib
import re
from types import MethodType, SimpleNamespace

import pytest
import yaml

from energy_sampling import config_invariants as ci

REPO = pathlib.Path(__file__).resolve().parents[2]
MK_DEV = REPO / 'configs' / 'mk_dev.yaml'
ARM = REPO / 'configs' / 'qm9c_anneal.yaml'
MAKE = REPO / 'configs' / 'qm9c_anneal' / 'make.py'
PROBLEMS = REPO / 'configs' / 'problems.yaml'

#: a key line carrying a route note: `  key: value  # ... CONDITIONAL ARM: <value> ...`
_NOTE = re.compile(r'^\s*([A-Za-z_]+):.*CONDITIONAL ARM:\s*(\S+)')


def _yaml(path):
    return yaml.safe_load(pathlib.Path(path).read_text(encoding='utf-8'))


def _make_globals():
    """make.GLOBALS, read by AST: importing make.py would put the repo root at
    the front of sys.path for every later import in the session."""
    tree = ast.parse(MAKE.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'GLOBALS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError('make.py defines no GLOBALS')


def _mk_dev_notes():
    notes = {}
    for line in MK_DEV.read_text(encoding='utf-8').splitlines():
        hit = _NOTE.match(line)
        if hit:
            key, token = hit.groups()
            assert key not in notes, f'two CONDITIONAL ARM notes on keys named {key}'
            notes[key] = yaml.safe_load(token.rstrip('.,;'))
    return notes


def _get(cfg, dotted):
    for k in dotted.split('.'):
        cfg = cfg[k]
    return cfg


def _modeller(cfg):
    from energy_sampling.protocol import StageProtocol
    from energy_sampling.train import Modeller
    from energy_sampling.utils import dict2namespace
    m = SimpleNamespace(args=dict2namespace(copy.deepcopy(cfg)), gfn_model=SimpleNamespace(),
                        stage=None, step_ind=0, _warn_if_z_untrained=lambda: None)
    m.protocol = StageProtocol(m)
    m.set_loss_coeffs = MethodType(Modeller.set_loss_coeffs, m)
    return m


# ------------------------------------------------------------------ the arm

def test_the_arm_carries_mk_devs_protocol_unedited():
    """make.py copies conditional_vargrad whole, so the two differ only when
    mk_dev moved and nobody re-ran the generator."""
    assert _yaml(ARM)['protocols']['conditional_vargrad'] == \
        _yaml(MK_DEV)['protocols']['conditional_vargrad'], 're-run configs/qm9c_anneal/make.py'


def test_the_arm_loads_through_the_invariants_and_the_real_loader():
    from energy_sampling import utils
    cfg = _yaml(ARM)
    assert ci.errors(cfg) == [], [str(v) for v in ci.errors(cfg)]
    args = utils.resolve_derived_config(utils.preflight_config(
        utils.dict2namespace(utils.load_yaml(str(ARM)))))
    assert args.protocol == 'conditional_vargrad'


def test_the_trainer_refusal_is_silent_on_the_arm():
    _modeller(_yaml(ARM)).set_loss_coeffs()


def test_the_arm_is_a_fresh_start_that_enters_var_conditioning_through_advance():
    """A weights-only start is at step 0, where protocol.begin() walks the skip
    chain: train_prior is skipped only while a prior model is loaded, and the
    skip calls advance() -- the path that runs var_conditioning's on_enter. With
    no prior model the run would sit in train_prior."""
    cfg = _yaml(ARM)
    assert cfg['load_weights_only'] is True and cfg['continue_from_checkpoint'] is False
    exit_, prior = cfg['checkpoint_name'], cfg['prior_model_name']
    assert exit_.endswith('_phase1_exit.pt') and prior.endswith('_prior.pt')
    assert exit_[:-len('_phase1_exit.pt')] == prior[:-len('_prior.pt')], \
        'the policy and the prior model must come from the same phase-1 run'
    stages = cfg['protocols']['conditional_vargrad']['stages']
    assert [s['name'] for s in stages] == ['train_prior', 'var_conditioning']
    assert stages[0]['skip_if'] == 'prior_loaded'
    on_enter = stages[1]['on_enter']
    # P_B is FROZEN (full snapshot) at entry on the replay seat: every learned-P_B
    # variant diverged there (2026-09-12)
    assert 'freeze_pb' in on_enter
    assert 'rebuild_prior_by_churn' in on_enter
    assert not any(a.startswith('bootstrap_z') for a in on_enter)

    for loaded in (True, False):
        m = _modeller(cfg)
        if loaded:
            m.prior_model = object()
        calls = []

        def advance(eval_metrics, run_exit_actions=True, m=m, calls=calls):
            calls.append((m.protocol.stage.name, run_exit_actions))
            m.stage = m.protocol.stages[m.protocol.stage.index + 1].name

        m.protocol.advance = advance
        m.protocol.begin()
        if loaded:
            assert calls == [('train_prior', False)] and m.stage == 'var_conditioning'
        else:
            assert calls == [] and m.stage == 'train_prior'


def test_the_route_globals_are_make_globals_and_match_mk_devs_notes():
    """make.py sets the route's globals and mk_dev notes each beside its key.
    Both directions: every GLOBALS key sits in the arm at its value and is noted
    in mk_dev at that value, and every note in mk_dev is a GLOBALS key."""
    glob, cfg, notes = _make_globals(), _yaml(ARM), _mk_dev_notes()
    assert glob, 'make.GLOBALS is empty'
    for dotted, value in glob.items():
        assert _get(cfg, dotted) == value, dotted
    assert notes == {d.rsplit('.', 1)[-1]: v for d, v in glob.items()}


# ------------------------------------------------------------ the stage itself

def test_min_fracs_holds_fwd_at_exactly_zero_under_the_nudge():
    """fwd has no frac on the stage. The lexicographic nudge floors every mode
    at the stage's min_fracs, else controller.min_mode_frac: without the stage's
    min_fracs.fwd 0 the first tick lifts fwd to the fallback."""
    cfg = _yaml(ARM)
    bare = copy.deepcopy(cfg)
    del bare['protocols']['conditional_vargrad']['stages'][1]['min_fracs']
    for c, want in ((cfg, 0.0), (bare, cfg['controller']['min_mode_frac'])):
        m = _modeller(c)
        m.stage = 'var_conditioning'
        st = m.protocol.stage
        assert not st.balance_can_raise('fwd', st.deactivate_threshold)
        m.fwd_frac, m.bwd_frac, m.replay_frac = 0.0, 0.5, 0.5
        for _ in range(5):
            m.protocol._nudge_mode_fracs(st.balance['default_boost'])
        assert m.fwd_frac == pytest.approx(want, abs=1e-15)
        assert m.bwd_frac == pytest.approx(m.replay_frac)
    assert want > 0.0, 'the fallback floor must be nonzero, or the pair above proves nothing'


def test_mk_devs_own_globals_refuse_the_stage_and_the_registry_global_clears_it():
    """mk_dev keeps the prioritised replay draw for the unconditional route, so
    selecting conditional_vargrad on mk_dev's globals fails at step 0; turning
    that one draw off -- what problems.yaml's qm9_conditional carries -- is
    enough to start."""
    cfg = _yaml(MK_DEV)
    cfg['protocol'] = 'conditional_vargrad'
    with pytest.raises(ValueError, match='prioritise'):
        _modeller(cfg).set_loss_coeffs()
    cfg['buffers']['replay_buffer']['prioritise']['enabled'] = False
    _modeller(cfg).set_loss_coeffs()
    reg = _yaml(PROBLEMS)['problems']['qm9_conditional']
    assert reg['protocol'] == 'conditional_vargrad'
    assert reg['buffers']['replay_buffer']['prioritise']['enabled'] is False


# ------------------------------------------------------------ batch and replay sizing

def test_the_batch_is_pinned_and_the_replay_cap_is_headroom_over_its_equilibrium():
    """make.py's GLOBALS note: at churn_rate 0 each manage call admits batch_size
    rows, one call per rollout and one per eval, so occupancy is
    batch_size x (1/fwd_rollout_every + 1/eval_period) x mean_residence_steps.
    The cap sits at ~2x that, so the hazard sets occupancy, and the warm-up
    releases well below it. With growth off, the batch this arithmetic assumes
    is the batch that runs."""
    cfg = _yaml(ARM)
    st = cfg['protocols']['conditional_vargrad']['stages'][1]
    rb = cfg['buffers']['replay_buffer']
    assert cfg['grow_batch_size'] is False and cfg['batch_util_target'] == 0
    assert rb['churn_rate'] == 0
    occupancy = (cfg['batch_size'] * (1 / st['fwd_rollout_every'] + 1 / cfg['eval_period'])
                 * rb['mean_residence_steps'])
    assert occupancy == pytest.approx(72000)
    assert 2 * occupancy <= rb['max_size'] <= 2.5 * occupancy
    assert st['replay_warmup_rows'] <= occupancy / 2


def test_churn_zero_admits_the_live_batch_not_the_tiled_rollout():
    """The premise of the arithmetic above, on the real manage_replay_buffer: a
    rollout at fwd repeats 2 carries 2 x batch_size eligible rows, and churn_rate
    0 admits batch_size of them (uniformly) per call."""
    from collections import defaultdict

    import torch

    from energy_sampling.train import Modeller

    batch, repeats = 100, 2

    class _Buf:
        def __init__(self, n):
            self.n, self.admitted = n, []
            self.ema_loss, self.birth_loss = torch.zeros(n), torch.zeros(n)
            self.birth_step = torch.zeros(n, dtype=torch.long)
            self.select_counts = torch.zeros(n, dtype=torch.long)

        def __len__(self):
            return self.n

        def purge_by_index(self, idx):
            self.n -= int(idx.numel())

        def add(self, rows, traj, **kw):
            self.admitted.append(int(traj.shape[0]))
            self.n += int(traj.shape[0])

    rb_cfg = SimpleNamespace(churn_rate=0, mean_residence_steps=1200, max_size=150000,
                             admit_reward_min=None)
    m = SimpleNamespace(args=SimpleNamespace(buffers=SimpleNamespace(replay_buffer=rb_cfg),
                                             batch_size=batch),
                        batch_size=batch, step_ind=20, buffer_device='cpu',
                        replay_buffer=_Buf(5000), replay_churn=defaultdict(int),
                        replay_cohort=defaultdict(int), _replay_managed=True,
                        _last_replay_manage_step=0, replay_in_play=lambda: True,
                        _replay_val_frac=lambda: 0.0)
    n = batch * repeats
    stats = {'log_r': -torch.rand(n), 'log_pf': torch.zeros(n), 'log_pb': torch.zeros(n),
             'log_Z': torch.zeros(n), 'flow_states': torch.zeros(n, 21, 12)}
    MethodType(Modeller.manage_replay_buffer, m)(
        stats, SimpleNamespace(subsample_new_batch=lambda inds: inds))
    assert m.replay_buffer.admitted == [batch]
