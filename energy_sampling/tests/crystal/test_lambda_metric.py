"""energy/lambda_mix is the lambda the energy function is actually using: written on
every run that carries a prior flow, absent on every run that does not."""
import ast
import inspect
import textwrap
from types import SimpleNamespace

from train import Modeller


def _metrics(energy_function):
    return Modeller._lambda_metrics(SimpleNamespace(energy_function=energy_function))


def test_a_flow_run_logs_the_live_lambda():
    ef = SimpleNamespace(prior_flow=object(), lambda_mix=0.0)
    assert _metrics(ef) == {'energy/lambda_mix': 0.0}
    # what set_energy_coeffs writes on an anneal event: the readback follows it
    ef.lambda_mix = 0.0107
    assert _metrics(ef) == {'energy/lambda_mix': 0.0107}


def test_a_lambda_free_run_logs_nothing():
    assert _metrics(SimpleNamespace(prior_flow=None, lambda_mix=1.0)) == {}
    # conformer energies carry neither attribute
    assert _metrics(SimpleNamespace()) == {}


def test_ten_step_reporting_merges_it():
    tree = ast.parse(textwrap.dedent(inspect.getsource(Modeller.ten_step_reporting)))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == '_lambda_metrics']
    assert calls, 'ten_step_reporting never merges _lambda_metrics()'
