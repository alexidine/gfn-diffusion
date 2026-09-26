"""cfg:buffers.fresh_on_switch (checkpointing.fresh_buffers_on_switch): loading ANOTHER run's checkpoint leaves the
buffers unrestored so the init path seeds every store under the live energy function (a model switch); the run's
own checkpoint (a requeue) restores as usual; off by default. The gate is what Checkpointer.load_full consults
before the sidecar lookup.
"""
import inspect
from types import SimpleNamespace

from energy_sampling.checkpointing import Checkpointer, fresh_buffers_on_switch


def _m(flag, run_name='am2_am2_acr_ft'):
    return SimpleNamespace(args=SimpleNamespace(buffers=SimpleNamespace(fresh_on_switch=flag)), run_name=run_name)


def test_off_by_default_restores_any_checkpoint(capsys):
    absent = SimpleNamespace(args=SimpleNamespace(buffers=SimpleNamespace()), run_name='a')
    assert fresh_buffers_on_switch(absent, {'run_name': 'b'}, 'x.pt') is False
    assert fresh_buffers_on_switch(_m(False), {'run_name': 'b'}, 'x.pt') is False
    assert capsys.readouterr().out == ''


def test_another_runs_checkpoint_starts_fresh_and_says_so(capsys):
    assert fresh_buffers_on_switch(_m(True), {'run_name': 'p23_p23_acr_ft'}, 'x.pt') is True
    out = capsys.readouterr().out
    assert 'NOT restored' in out and "'p23_p23_acr_ft'" in out and "'am2_am2_acr_ft'" in out


def test_the_runs_own_checkpoint_restores(capsys):
    assert fresh_buffers_on_switch(_m(True), {'run_name': 'am2_am2_acr_ft'}, 'x.pt') is False
    assert capsys.readouterr().out == ''


def test_a_checkpoint_without_a_run_name_counts_as_foreign():
    assert fresh_buffers_on_switch(_m(True), {}, 'x.pt') is True


def test_load_full_consults_the_gate_before_the_sidecar_lookup():
    src = inspect.getsource(Checkpointer.load_full)
    assert src.index('fresh_buffers_on_switch(') < src.index('self.load_buffers_for(path)')
