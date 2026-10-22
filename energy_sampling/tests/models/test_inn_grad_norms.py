"""
CPU tests for the exact-density flow's gradnorm/* family (inn_modeller.grad_parts, part_grad_norms).

WHAT THIS FILE IS FOR. The trainer logs one gradient norm per part of the flow on reporting steps, before the clip. Two
failures would run without an error: a parameter no part holds (its gradient never shows on the dashboard), and a part
whose norm is not its parameters' (a wrong grouping still prints plausible numbers). So every parameter must sit in
exactly one part, and the parts' norms must recombine to the whole norm clip_grad_norm_ returns. A part the loss does
not reach must read 0.0, not go missing: the conditional flow's level head is the NEGATIVE CONTROL, since log q does
not read it.

    python test_inn_grad_norms.py
"""
import math
import os
import sys

import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
for p in (_here, os.path.dirname(_here)):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

import inn_modeller as im  # noqa: E402

WRAP = [False] * 7 + [True, True, False, True, True]


def _flows():
    torch.manual_seed(0)
    tf = im.LatentFlow(12, WRAP, [], 1, 8, (64,), 5.0, 0, transformer=(32, 3, 4))
    tf.set_bf16(False)
    made = im.LatentFlow(12, WRAP, [], 3, 8, (64, 64), 5.0, 0)
    cond = im.ConditionalLatentFlow(12, WRAP, [], 4, 16, 32, 2, 8, (64,), 5.0, 0, transformer=(32, 2, 4))
    for blk in cond.flow.blocks:
        blk.made.bf16 = False
    with torch.no_grad():   # off the zero-head identity start, through which no gradient reaches the layers
        for f in (tf, made, cond):
            for prm in f.parameters():
                if prm.dim() > 1 and float(prm.abs().max()) == 0.0:
                    prm.normal_(0, 0.05)
    return {'transformer': (tf, None), 'made': (made, None), 'conditional': (cond, torch.randn(64, 4))}


def _backward(f, c):
    x = 0.5 * torch.randn(64, 12)
    lq = f.log_prob(x) if c is None else f.log_prob(x, c)
    (-lq.mean()).backward()


def test_parts_hold_every_parameter_once():
    for kind, (f, _) in _flows().items():
        parts = im.grad_parts(f)
        ids = [id(p) for ps in parts.values() for p in ps]
        assert len(ids) == len(set(ids)), kind
        assert set(ids) == {id(p) for p in f.parameters()}, kind
        assert 'other' not in parts, (kind, sorted(parts))


def test_part_names():
    flows = _flows()
    assert sorted(im.grad_parts(flows['transformer'][0])) == [
        'block0/embed', 'block0/head', 'block0/layer0', 'block0/layer1', 'block0/layer2']
    assert sorted(im.grad_parts(flows['made'][0])) == ['block0', 'block1', 'block2']
    assert {'context', 'level', 'block1/layer1', 'block1/embed'} <= set(im.grad_parts(flows['conditional'][0]))


def test_norms_recombine_to_the_clip_norm():
    for kind, (f, c) in _flows().items():
        _backward(f, c)
        norms = im.part_grad_norms(f)
        whole = float(torch.nn.utils.clip_grad_norm_(f.parameters(), math.inf))
        assert whole > 0.0, kind
        assert math.isclose(math.sqrt(sum(v * v for v in norms.values())), whole, rel_tol=1e-5), kind
        assert all(k.startswith('gradnorm/') for k in norms), kind


def test_unreached_part_reads_zero_negative_control():
    f, c = _flows()['conditional']
    _backward(f, c)
    norms = im.part_grad_norms(f)
    assert norms['gradnorm/level'] == 0.0           # log q does not read the level head: present, and zero
    assert norms['gradnorm/context'] > 0.0          # the condition embedding it does read
    assert all(norms[f'gradnorm/block{i}/layer0'] > 0.0 for i in range(2))


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
