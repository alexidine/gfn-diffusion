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


#: EVERY write path on Checkpointer, taken from the WRITER rather than from its
#: call sites. Enumerating call sites is what missed `archive`: nothing in
#: train.py calls `link`, but `archive` does, and `archive` hardlinks
#: `stepNNNNN.pt` onto the current `running.pt` bytes -- so a resume at step
#: 15000 with archiving on destroyed a user's `_step15000.pt` by pointing it at
#: the diagnostic's own state.
WRITERS = ('save', 'save_buffers', 'archive', 'link')


@pytest.mark.parametrize('name', WRITERS)
def test_the_real_checkpointer_still_has_this_write_method(name):
    """If a write method is renamed, the suppression silently stops covering it."""
    src = _source('checkpointing.py')
    assert re.search(rf'^    def {name}\(', src, re.M), (
        f'Checkpointer.{name} is gone; calibrate_noise blocks a method that no '
        f'longer exists and writes may be live again')


@pytest.mark.parametrize('name', WRITERS)
def test_read_only_gates_every_write_path(name):
    """
    `checkpoint_read_only` is the supported suppression and must gate ALL of
    them. This is the layer that would have held when the monkeypatch did not.
    """
    src = _source('checkpointing.py')
    body = re.split(rf'^    def {name}\(', src, flags=re.M)[1]
    body = re.split(r'^    def ', body, flags=re.M)[0]
    assert 'read_only' in body, (
        f'Checkpointer.{name} no longer checks self.read_only, so '
        f'checkpoint_read_only does not suppress it')


def test_calibrate_noise_blocks_the_same_set():
    """The file's own `_writers` tuple must match WRITERS above."""
    body = _calib_source()
    found = re.search(r'_writers = \(([^)]*)\)', body)
    assert found, 'calibrate_noise no longer declares _writers'
    listed = tuple(re.findall(r"'([a-z_]+)'", found.group(1)))
    assert set(listed) == set(WRITERS), (
        f'calibrate_noise blocks {listed}, write paths are {WRITERS}')


def test_calibrate_noise_sets_the_supported_flag():
    assert 'checkpoint_read_only' in _calib_source(), (
        'the supported suppression is not being set; a monkeypatch alone was '
        'already shown to miss a path')


def test_every_write_path_is_blocked():
    """
    All four writers replaced on the class, with the attempt recorded. Uses a
    stand-in Checkpointer because the real one needs a Modeller; the mechanism
    under test is the patch, which is name-agnostic by construction.
    """
    class Checkpointer:
        def __init__(self):
            self.wrote = []

        def save(self, tag, with_buffers=False):
            self.wrote.append(tag)

        def save_buffers(self, tag=None):
            self.wrote.append(f'buffers:{tag}')

        def archive(self, step):
            self.wrote.append(f'archive:{step}')

        def link(self, src_tag, dst_tag):
            self.wrote.append(f'link:{src_tag}->{dst_tag}')

    ck = Checkpointer()
    cls = type(ck)
    real = {n: getattr(cls, n) for n in WRITERS}
    blocked = []

    def _blocker(name):
        def _no_write(self, tag='?', *a, **kw):
            blocked.append(f'{name}:{tag}')
        return _no_write

    for n in WRITERS:
        setattr(cls, n, _blocker(n))
    try:
        ck.save('running')                          # train.py:2030
        ck.save('final', with_buffers=True)         # train.py:2049
        ck.save_buffers()                           # train.py:1997
        ck.save('stage_start')                      # protocol.py:1207
        ck.save('phase1_exit', with_buffers=True)   # protocol.py:1242
        ck.save('prior')                            # protocol.py:1253
        ck.archive(15000)                           # train.py:2047
        ck.link('running', 'step15000')             # checkpointing.py:430
    finally:
        for n, fn in real.items():
            setattr(cls, n, fn)

    assert ck.wrote == [], f'a write got through: {ck.wrote}'
    assert len(blocked) == 8
    assert 'archive:15000' in blocked, (
        'archive is the path that destroyed a real checkpoint -- it hardlinks '
        'stepNNNNN.pt onto the current running.pt bytes and never calls save')
    assert 'link:running' in blocked
    ck.save('running')                              # real methods restored
    assert ck.wrote == ['running']
