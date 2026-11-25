"""train.py::Modeller.handle_train_epoch_error frees the failed step's graph while the exception is still alive, so the
retry after an OOM does not allocate on top of it.

    python -m pytest -q tests/infra/test_oom_recovery_frees_graph.py
"""
import weakref
from types import SimpleNamespace

import pytest

from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast


class Graph:
    def __init__(self):
        self.me = self   # a reference cycle, as an autograd graph holds


def failed_step(refs):
    graph = Graph()
    refs.append(weakref.ref(graph))
    raise RuntimeError('CUDA out of memory. Tried to allocate 20.00 MiB')


def modeller(step):
    m = Modeller.__new__(Modeller)
    m.step_ind = step
    m.optimizers = {}
    m.batch_size = 1000
    m.fused_accum_count = 0
    m.args = SimpleNamespace(oom_batch_shrink_factor=0.625, oom_cooldown_steps=10)
    return m


@pytest.mark.parametrize('step', [0, 5])
def test_the_failed_steps_graph_is_gone_before_the_handler_returns(step):
    m, refs = modeller(step), []
    try:
        failed_step(refs)
    except RuntimeError as e:
        m.handle_train_epoch_error(e, 'bwd')
        assert refs[0]() is None
    assert m.batch_size == (1000 if step == 0 else 625)


def test_an_error_that_is_not_an_oom_is_raised_with_its_frames_intact():
    m = modeller(5)
    with pytest.raises(ValueError, match='not memory') as info:
        try:
            raise ValueError('not memory')
        except ValueError as e:
            m.handle_train_epoch_error(e, 'bwd')
    assert info.value.__traceback__.tb_frame.f_locals is not None
    assert m.batch_size == 1000
