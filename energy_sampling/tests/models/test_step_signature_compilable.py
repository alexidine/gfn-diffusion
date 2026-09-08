"""The per-timestep step bodies must stay free of Python int arguments.

WHY THIS TEST EXISTS. `_fwd_step` and `_replay_step` are the units
`compile_policy: step` hands to torch.compile. Dynamo SPECIALISES on a Python int, so
a single `i: int` parameter turns one graph into one per timestep -- 100 against a
`cache_size_limit` of 24, which blows the limit and falls back to eager. With
`suppress_errors` on that fallback is SILENT: no error, no warning, and the only
symptom is that a compile-mode run performs exactly like an eager one, which is
indistinguishable from "compiled and didn't help" without a profiler.

So compilability here is a property that regresses invisibly, and re-adding `i` is the
natural thing a future edit does. Hence the pin.

READ FROM SOURCE, NOT BY IMPORT. `models.gfn` pulls in `energy_sampling.utils`, which
pulls in `mxtaltools`, so importing it needs two extra entries on sys.path that a bare
pytest run does not have -- and a guard that cannot run in the environment it is meant
to guard is not a guard. The signature is a syntactic property, so `ast` answers it
with no imports and no GPU.

Bools are fine and deliberate: `is_first` and `detach_traj` specialise into at most a
handful of graphs, well inside the limit.
"""
import ast
import os

import pytest

GFN_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'models', 'gfn.py')
STEP_FNS = ('_fwd_step', '_replay_step')


def _signature(name):
    """(arg names, {arg: annotation-id}) for a method of class GFN, read from source."""
    tree = ast.parse(open(GFN_SRC, encoding='utf-8').read())
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'GFN'):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name):
            args = [a.arg for a in fn.args.args if a.arg != 'self']
            ann = {a.arg: getattr(a.annotation, 'id', None) for a in fn.args.args
                   if a.annotation is not None}
            return args, ann
    pytest.fail(f'GFN.{name} not found in {GFN_SRC}')


@pytest.mark.parametrize('name', STEP_FNS)
def test_step_body_takes_no_int_argument(name):
    args, ann = _signature(name)
    ints = [a for a, t in ann.items() if t == 'int']
    assert not ints, (
        f'GFN.{name} takes {ints} annotated `int`. Dynamo specialises on Python ints, so '
        f'this makes the step one graph PER TIMESTEP, blows the dynamo cache limit, and '
        f'silently falls back to eager under suppress_errors. Hoist the indexing to the '
        f'trajectory loop and pass tensors, or pass a bool.')
    assert 'i' not in args, (
        f'GFN.{name} takes `i` again -- hoist `ts[:, i]` / `ts[:, i + 1]` into the caller '
        f'and pass the two time slices, as the loop in get_traj_fwd does.')


@pytest.mark.parametrize('name', STEP_FNS)
def test_step_body_receives_time_as_tensors(name):
    """The hoist is only real if the step actually takes the two time slices."""
    args, _ = _signature(name)
    assert 't_cur' in args and 't_next' in args, (
        f'GFN.{name} lost t_cur/t_next; the step can no longer be traced as one graph.')
    assert 'ts' not in args, (
        f'GFN.{name} takes the whole `ts` grid again, which only makes sense alongside an '
        f'index -- the indexing belongs in the loop.')


@pytest.mark.parametrize('name', STEP_FNS)
def test_first_step_branch_is_a_bool(name):
    """`if i > 0` was the one genuine control-flow use of the index."""
    args, ann = _signature(name)
    assert 'is_first' in args, f'GFN.{name} lost the is_first flag'
    assert ann.get('is_first') == 'bool'


def test_eval_pb_logprob_is_index_free():
    """The shared P_B scorer is inside both step bodies, so it carries the same rule."""
    args, ann = _signature('_eval_pb_logprob')
    assert 'i' not in args and 'ts' not in args, (
        '_eval_pb_logprob takes an index again; it is called from inside both compiled '
        'step bodies, so an int here specialises the whole step.')
    assert 'is_first' in args and ann.get('is_first') == 'bool'
