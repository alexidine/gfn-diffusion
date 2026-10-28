"""Eval-relative cadences: figures (`figs_every_n_evals`) and the cross-condition matrix
(`xcond_eval.every_n_evals`) fire on evaluations whose step is a multiple of
N * eval_period (train.py::eval_cadence_fires). Null or 0 is off. The step-unit keys they
replaced, `figs_period` and `xcond_eval.period`, are refused off the conformer route at
Modeller.init_gfn; on `conformer_torsions`, whose configs set no `figs_every_n_evals`,
Modeller.figs_due still reads `figs_period`.

    python -m pytest -q tests/crystal/test_eval_cadence.py
"""
from types import SimpleNamespace

import pytest

from energy_sampling.config_invariants import retired_step_cadence_keys
from energy_sampling.train import Modeller, eval_cadence_fires
from energy_sampling.utils import dict2namespace

pytestmark = pytest.mark.fast


def test_fires_on_the_step_grid_of_n_eval_periods():
    fired = [s for s in range(0, 2001, 50) if eval_cadence_fires(s, 2, 250)]
    assert fired == [0, 500, 1000, 1500, 2000]
    # step 50 (the warm-up eval) and an off-grid request_eval step are not on the grid
    assert not eval_cadence_fires(50, 1, 250) and not eval_cadence_fires(730, 1, 250)
    # the decision is a function of the step alone, so a resumed run keeps the phase
    assert eval_cadence_fires(2500, 10, 250) and not eval_cadence_fires(2750, 10, 250)


@pytest.mark.parametrize('n, period', [(None, 250), (0, 250), (2, None), (2, 0)])
def test_null_or_zero_is_off(n, period):
    assert not any(eval_cadence_fires(s, n, period) for s in range(0, 5001, 250))


def _figs(step, **args):
    return Modeller.figs_due(SimpleNamespace(args=SimpleNamespace(**args), step_ind=step))


def test_figs_due_reads_figs_every_n_evals():
    assert _figs(500, figs_every_n_evals=2, eval_period=250, energy_function='elj')
    assert not _figs(750, figs_every_n_evals=2, eval_period=250, energy_function='elj')
    assert not _figs(500, figs_every_n_evals=0, eval_period=250, energy_function='elj')
    # off the conformer route figs_period is not read (init_gfn refuses it)
    assert not _figs(500, figs_period=500, eval_period=250, energy_function='elj')


def test_figs_due_reads_figs_period_on_the_conformer_route():
    conf = dict(eval_period=500, energy_function='conformer_torsions')
    assert _figs(1000, figs_period=1000, **conf)
    assert not _figs(500, figs_period=1000, **conf)
    assert not _figs(1000, figs_period=0, **conf)
    # figs_every_n_evals wins when a conformer config sets it
    assert _figs(500, figs_every_n_evals=1, figs_period=1000, **conf)


def _init_gfn(**cfg):
    Modeller.init_gfn(SimpleNamespace(args=dict2namespace(cfg)))


@pytest.mark.parametrize('cfg, key', [
    ({'figs_period': 500}, 'figs_period'),
    ({'xcond_eval': {'enabled': True, 'period': 2500}}, 'xcond_eval.period'),
])
def test_init_gfn_refuses_the_step_unit_keys_off_the_conformer_route(cfg, key):
    with pytest.raises(ValueError, match=key.replace('.', r'\.')):
        _init_gfn(energy_function='elj', **cfg)


def test_the_retired_keys_are_found_on_a_namespace():
    """init_gfn reads the resolved Namespace, the invariant the raw dict: one resolver."""
    cfg = {'energy_function': 'elj', 'figs_period': 500, 'xcond_eval': {'period': 2500}}
    assert retired_step_cadence_keys(dict2namespace(cfg)) == ['figs_period', 'xcond_eval.period']
    assert retired_step_cadence_keys(cfg) == ['figs_period', 'xcond_eval.period']
    assert retired_step_cadence_keys(dict2namespace(
        {**cfg, 'energy_function': 'conformer_torsions'})) == []
    assert retired_step_cadence_keys(dict2namespace(
        {'energy_function': 'elj', 'figs_every_n_evals': 2, 'xcond_eval': {'every_n_evals': 10}})) == []
