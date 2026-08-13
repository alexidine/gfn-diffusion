"""
Can the noise calibration write a checkpoint?

It could, for as long as the docstring promised it could not. `m.args.
save_checkpoints = False` assigned an attribute that NOTHING in the codebase
reads, so every run of the diagnostic saved under the config's own run name;
on 2026-08-13 that overwrote `_running.pt`, `_best.pt`, `_buffers.pt`,
`_prior.pt`, `_stage_start.pt` and `phase1_exit.pt` for an active experiment.

Two tests, because the bug had two halves and either alone would have let it
through:

  * `test_the_suppressed_names_are_read_by_the_code` -- would have caught it
    BEFORE any run. A flag that no module reads cannot suppress anything, and
    that is checkable with a grep and no GPU.
  * `test_every_write_path_is_blocked` -- checks the replacement mechanism on
    the write methods themselves, including the `protocol.py` transition path
    that `train.py` does not own.

Neither needs torch, a GPU, or a checkpoint on disk.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = os.path.join(ROOT, 'bench', 'calibrate_noise.py')
#: every module that could honour a suppression flag
READERS = ('train.py', 'checkpointing.py', 'protocol.py')


def _source(name):
    with open(os.path.join(ROOT, name), encoding='utf-8', errors='ignore') as f:
        return f.read()


def _calib_source():
    with open(CALIB, encoding='utf-8', errors='ignore') as f:
        return f.read()


def test_the_suppressed_names_are_read_by_the_code():
    """
    EVERY `m.args.X = ...` in the calibration must name a field the training
    code actually reads. This is the test that was missing: `save_checkpoints`
    and `use_wandb` were both assigned, neither existed, and both read as
    working safety measures for as long as they were there.
    """
    body = _calib_source()
    # assignments only, not the comments describing the historical bug
    assigned = set(re.findall(r'^\s*m\.args\.([a-zA-Z_][\w]*)\s*=',
                              body, re.M))
    assert assigned, 'no m.args assignments found -- has the file moved?'
    code = '\n'.join(_source(n) for n in READERS)
    dead = sorted(a for a in assigned if a not in code)
    assert not dead, (
        f'calibrate_noise sets {dead} on args, and no module in {READERS} reads '
        f'those names. An assignment that nothing reads is not a setting, and '
        f'if it is standing in for a safety promise the promise is fiction.')


def test_save_checkpoints_is_still_not_a_real_field():
    """
    Pins the diagnosis. If a `save_checkpoints` flag is ever introduced for
    real, this fails and someone re-reads the suppression rather than assuming
    the old comment still describes the code.
    """
    code = '\n'.join(_source(n) for n in READERS)
    assert 'save_checkpoints' not in code


def test_every_write_path_is_blocked():
    """
    Both writers, replaced on the class, with the tags recorded. Uses a stand-in
    Checkpointer because the real one needs a Modeller; the mechanism under test
    is the patch, and the patch is name-agnostic by construction.
    """
    class Checkpointer:
        def __init__(self):
            self.wrote = []

        def save(self, tag, with_buffers=False):
            self.wrote.append(tag)

        def save_buffers(self, tag=None):
            self.wrote.append(f'buffers:{tag}')

    ck = Checkpointer()
    cls = type(ck)
    real_save, real_buf = cls.save, cls.save_buffers
    blocked = []

    def _no_save(self, tag='?', *a, **kw):
        blocked.append(str(tag))

    cls.save, cls.save_buffers = _no_save, _no_save
    try:
        ck.save('running')                       # train.py:2030
        ck.save('final', with_buffers=True)      # train.py:2049
        ck.save_buffers()                        # train.py:1997
        ck.save('stage_start')                   # protocol.py:1207
        ck.save('phase1_exit', with_buffers=True)  # protocol.py:1242
        ck.save('prior')                         # protocol.py:1253
    finally:
        cls.save, cls.save_buffers = real_save, real_buf

    assert ck.wrote == [], f'a write got through: {ck.wrote}'
    assert len(blocked) == 6
    assert 'phase1_exit' in blocked, (
        'the transition path must be blocked too -- a resume from '
        'phase1_exit.pt transitions on its first step and rewrites that exact '
        'file, which is how the regime clobbered its own definition')
    # and the real methods are back
    ck.save('running')
    assert ck.wrote == ['running']


@pytest.mark.parametrize('name', ['save', 'save_buffers'])
def test_the_real_checkpointer_still_has_these_methods(name):
    """If the write API is renamed, the patch above silently stops covering it."""
    src = _source('checkpointing.py')
    assert re.search(rf'^    def {name}\(', src, re.M), (
        f'Checkpointer.{name} is gone; calibrate_noise patches a method that no '
        f'longer exists and writes are live again')
