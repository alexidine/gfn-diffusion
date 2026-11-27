"""The Boltzmann-slope readout (train.py Modeller._within_condition_slope, _boltzmann_slope_panel,
_boltzmann_slope_readouts): under temperature_conditioning, log q(x | c, T) of a fixed panel of stored crystals regressed
on their energy within each condition, at T = 1, 2 and 4 x the run's temperature. What each test would catch:

  * a slope that reads per-condition offsets, or rows with a non-finite value;
  * a readout that is not 1 for a Boltzmann density at each T, or not 0 for one blind to the energy;
  * the condition built at another temperature than the one read, or the chart's log reference left in the regression;
  * the readout running off its route, or a panel that moves between evaluations.

CPU, a store-stub Modeller and a stub policy; no energy computed.

    python -m pytest -q tests/crystal/test_boltzmann_slope.py
"""
import math
from types import SimpleNamespace

import pytest
import torch

from energy_sampling.tests.crystal import store_stub as S
from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast


def test_the_slope_is_pooled_within_conditions():
    g = torch.Generator().manual_seed(0)
    cid = torch.arange(6).repeat_interleave(50)
    u = torch.rand(300, generator=g, dtype=torch.float64) * 20
    y = -u / 2.5 + 100.0 * cid + 0.01 * torch.randn(300, generator=g, dtype=torch.float64)
    y[3] = float('nan')
    b, r2 = Modeller._within_condition_slope(y, u, cid)
    assert abs(b + 1 / 2.5) < 1e-3 and r2 > 0.999
    assert all(math.isnan(v) for v in Modeller._within_condition_slope(y, torch.ones(300), cid))


def test_the_log_log_slope_is_the_power_of_t():
    T = [2.5, 5.0, 10.0]
    assert abs(Modeller._log_log_slope(T, [0.3 * t ** 0.5 for t in T]) - 0.5) < 1e-12
    assert abs(Modeller._log_log_slope(T, [0.7, 0.7, 0.7])) < 1e-12
    for bad in ([0.5, 0.0, 1.0], [0.5, -1.0, 1.0], [0.5, float('nan'), 1.0]):
        assert math.isnan(Modeller._log_log_slope(T, bad))
    assert math.isnan(Modeller._log_log_slope([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]))


def _modeller(blind=False, ref=0.0, t_blind=False):
    m = S.build()
    m.energy_function.temperature_conditioning = True
    m.energy_function.crystal_coordinates = 'fd'
    m.energy_function.fd_rows_to_params = lambda x, mol: (None, None, torch.full((x.shape[0],), float(ref)))
    m.BOLTZMANN_PANEL_MIN_ROWS = 2
    orig = m.energy_function.condition_samples

    def condition_samples(mol, temperature=None, **kw):
        mol, log_T, cond, cid = orig(mol, temperature=temperature, **kw)
        return mol, log_T, torch.cat([cond, temperature.reshape(-1, 1).float()], 1), cid
    m.energy_function.condition_samples = condition_samples
    T_run = float(m.args.energy_config.temperature)

    def log_density(x, condition, mol):
        """Boltzmann at the condition's T in the stub's energy u = T_run (|x|^2 + cid / 10) plus the reference; blind to
        the energy; or (t_blind) Boltzmann at T_run whatever the condition's T."""
        if blind:
            return torch.zeros(x.shape[0])
        u = T_run * ((x.double() ** 2).sum(-1) + mol.condition_id.double() / 10)
        T = torch.full_like(u, T_run) if t_blind else condition[:, -1].double()
        return (-u / T + ref).float()
    m.gfn_model = SimpleNamespace(log_density=log_density)
    return m


def test_a_boltzmann_density_reads_one_at_every_temperature(capsys):
    m = _modeller(ref=0.7)
    out = m._boltzmann_slope_readouts(chunk=5)
    T = float(m.args.energy_config.temperature)
    for f in Modeller.BOLTZMANN_T_FACTORS:
        assert abs(out[f'eval_T/boltzmann_slope@{f * T:g}'] - 1.0) < 1e-4
        assert out[f'eval_T/boltzmann_r2@{f * T:g}'] > 0.9999
    assert abs(out['eval_T/boltzmann_t_exponent']) < 1e-4
    assert 'T exponent' in capsys.readouterr().out


def test_a_policy_blind_to_its_temperature_input_reads_exponent_one():
    m = _modeller(t_blind=True)
    out = m._boltzmann_slope_readouts(chunk=64)
    T = float(m.args.energy_config.temperature)
    for f in Modeller.BOLTZMANN_T_FACTORS:
        assert abs(out[f'eval_T/boltzmann_slope@{f * T:g}'] - f) < 1e-4
    assert abs(out['eval_T/boltzmann_t_exponent'] - 1.0) < 1e-4


def test_a_density_blind_to_the_energy_reads_zero():
    m = _modeller(blind=True)
    out = m._boltzmann_slope_readouts(chunk=64)
    assert all(abs(v) < 1e-9 for k, v in out.items() if 'slope' in k)
    assert math.isnan(out['eval_T/boltzmann_t_exponent'])


def test_the_panel_is_fixed_between_evaluations():
    m = _modeller()
    m._boltzmann_slope_readouts(chunk=64)
    first = {k: v.clone() for k, v in m._boltzmann_panel.items()}
    m.row_store.log_r[:] = 0.0                                   # the store moves; the panel does not
    m._boltzmann_slope_readouts(chunk=64)
    assert all(torch.equal(first[k], m._boltzmann_panel[k]) for k in first)


@pytest.mark.parametrize('what', ['tcond', 'chart', 'density', 'store'])
def test_off_its_route_the_readout_is_empty(what):
    m = _modeller()
    if what == 'tcond':
        m.energy_function.temperature_conditioning = False
    elif what == 'chart':
        m.energy_function.crystal_coordinates = 'latent'
    elif what == 'density':
        m.gfn_model = SimpleNamespace()
    else:
        m.row_store = None
    assert m._boltzmann_slope_readouts(chunk=64) == {}


def test_a_store_too_sparse_for_a_panel_is_retried_at_the_next_evaluation(capsys):
    m = _modeller()
    m.BOLTZMANN_PANEL_MIN_ROWS = 1000                           # no condition qualifies yet
    assert m._boltzmann_slope_readouts(chunk=64) == {} and m._boltzmann_slope_readouts(chunk=64) == {}
    assert capsys.readouterr().out.count('off until a condition holds') == 1     # said once, not every eval
    assert getattr(m, '_boltzmann_panel', None) is None
    m.BOLTZMANN_PANEL_MIN_ROWS = 2                              # growth brought conditions up to the floor
    assert any(k.startswith('eval_T/boltzmann_slope@') for k in m._boltzmann_slope_readouts(chunk=64))
    assert isinstance(m._boltzmann_panel, dict)


def test_the_panel_gathers_at_most_spare_x_panel_rows_per_condition(monkeypatch):
    m = _modeller()
    m.BOLTZMANN_PANEL_ROWS, m.BOLTZMANN_PANEL_SPARE = 1, 2
    st, seen = m.row_store, []
    orig = st.mol_batch

    def mol_batch(cids, repeats=1):
        seen.append(torch.as_tensor(cids).clone())
        return orig(cids, repeats)
    monkeypatch.setattr(st, 'mol_batch', mol_batch)
    panel = m._boltzmann_slope_panel()
    assert int(torch.bincount(seen[0]).max()) <= 2              # the template batch before the clip filter
    assert int(torch.bincount(panel['cid']).max()) <= 1


def test_the_panel_rides_the_buffer_sidecar_to_its_own_runs_resume():
    from energy_sampling.checkpointing import Checkpointer
    m = _modeller()
    m._boltzmann_slope_readouts(chunk=64)
    panel = m._boltzmann_panel
    mgr = Checkpointer.__new__(Checkpointer)
    mgr.modeller = SimpleNamespace(_boltzmann_panel=panel, problem_def={'p': 1}, problem_hash='h', run_name='arm',
                                   step_ind=7)
    mgr._store_state = lambda name: None
    mgr._row_store_entry = lambda: None
    state = Checkpointer.buffer_state(mgr)
    assert state['boltzmann_panel'] is panel
    for run_name, kept in (('arm', True), ('other_arm', False)):
        mgr.modeller = SimpleNamespace(problem_def={'p': 1}, problem_hash='h', run_name=run_name, args=None)
        mgr.restore_buffers(state, 'sidecar')
        assert (getattr(mgr.modeller, '_boltzmann_panel', None) is panel) is kept, run_name
