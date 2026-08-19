"""
Tier 3 -- the figure index.

WHAT HAS TO BE TRUE, and why each is mutation-tested rather than asserted:

  * a filename that does not match the convention is DROPPED, not guessed at.
    Inventing a name for an unrecognised file puts a figure in the index under
    something no caller can ask for, and the caller then reads "no such figure"
    as "the run never logged it";
  * a figure name containing digits, spaces and underscores survives the parse.
    Every real name in this project has spaces, and the step/hash suffix is the
    only fixed part -- a greedy split on `_` would truncate
    'Bwd Traj Mean Step Sizes' at the wrong place;
  * a missing figure returns None, distinct from an empty or unfetched one. A
    run that never reached the stage which logs a parity plot HAS no parity
    plot, and reporting that as a fetch failure sends the reader after a file
    that was never supposed to exist.

Network-free: builds a media tree on tmp_path rather than touching wandb.
"""

import os

import pytest

from analysis import figures as F
from analysis.pull import Run

pytestmark = pytest.mark.fast


#: Real names from a local run -- spaces, digits, and a trailing word that a
#: naive rsplit would eat.
_NAMES = ('Backward TB Parity Plot', 'Bwd Traj Mean Step Sizes',
          'Forward Lattice Latents Trajectories')


def _make_run(tmp_path, files=None):
    """A local run with a wandb-shaped media tree."""
    media = tmp_path / 'files' / 'media'
    (media / 'images').mkdir(parents=True)
    (media / 'plotly').mkdir(parents=True)
    files = files if files is not None else [
        ('plotly', f'{_NAMES[0]}_91_f0118b7f5d62893e75cc.plotly.json'),
        ('plotly', f'{_NAMES[0]}_191_f69a0c93d83298f893b9.plotly.json'),
        ('plotly', f'{_NAMES[1]}_191_aaaaaaaabbbbbbbbcccc.plotly.json'),
        ('images', f'{_NAMES[2]}_91_e97b9b976d77200d5858.png'),
    ]
    for kind, name in files:
        (media / kind / name).write_text('x')
    return Run(run_id='testrun', name='t', source='local', path=str(tmp_path))


def test_index_groups_by_name_and_orders_by_step(tmp_path):
    idx = F.index(_make_run(tmp_path))
    assert set(idx) == set(_NAMES)
    assert [f.step for f in idx[_NAMES[0]]] == [91, 191], 'must be oldest-first'
    assert idx[_NAMES[2]][0].kind == 'images'


def test_a_name_containing_an_underscore_survives_the_parse(tmp_path):
    """The suffix `_<step>_<hash>` is the only fixed part, so the name match has
    to be greedy and anchored on the tail.

    THE FIXTURE NEEDS AN UNDERSCORE IN THE NAME or this proves nothing -- every
    figure this project logs today uses spaces, so a name pattern of `[^_]+`
    passes against real data while being wrong. Found by mutating the regex and
    watching this test stay green."""
    run = _make_run(tmp_path, files=[
        ('plotly', 'fwd_tb_parity_191_f69a0c93d83298f893b9.plotly.json')])
    assert list(F.index(run)) == ['fwd_tb_parity']


def test_names_with_spaces_survive_the_parse(tmp_path):
    idx = F.index(_make_run(tmp_path))
    assert 'Bwd Traj Mean Step Sizes' in idx
    assert 'Backward TB Parity Plot' in idx


def test_latest_returns_the_highest_step(tmp_path):
    fig = F.latest(_make_run(tmp_path), _NAMES[0])
    assert fig.step == 191 and fig.fetched


def test_a_figure_the_run_never_logged_is_None(tmp_path):
    """Distinct from 'exists but unfetched'. A run that never reached the stage
    logging this has no such figure, and that is an answer."""
    assert F.latest(_make_run(tmp_path), 'Never Logged') is None


@pytest.mark.parametrize('bad', [
    'no_step_or_hash.png',
    'Missing Hash_191.png',
    'Bad Step_abc_f0118b7f5d62893e75cc.png',
    'shorthash_191_dead.png',
])
def test_unrecognised_filenames_are_dropped_not_guessed(tmp_path, bad):
    """MUTATION TARGET. If `_parse` ever falls back to a heuristic instead of
    returning None, these land in the index under invented names."""
    run = _make_run(tmp_path, files=[('images', bad)])
    assert F.index(run) == {}, f'{bad!r} was parsed into a figure'


def test_the_parser_can_see_a_GOOD_name(tmp_path):
    """MUTATION IN THE PASSING DIRECTION. Without this, `_parse` returning None
    unconditionally would satisfy every rejection test above."""
    run = _make_run(tmp_path, files=[
        ('images', 'Good Name_5_0123456789abcdef.png')])
    assert list(F.index(run)) == ['Good Name']


def test_render_says_none_rather_than_printing_an_empty_table(tmp_path):
    (tmp_path / 'files' / 'media').mkdir(parents=True)
    run = Run(run_id='r', name='t', source='local', path=str(tmp_path))
    assert F.render(run) == 'figures: none logged'


def test_fetch_returns_a_local_path_untouched(tmp_path):
    """A local run is already on disk. Copying it into the cache would give one
    file two paths, which is how a reader ends up looking at the stale one."""
    fig = F.latest(_make_run(tmp_path), _NAMES[0])
    assert F.fetch(fig) == fig.path
    assert os.path.dirname(fig.path).endswith(os.path.join('media', 'plotly'))


def test_an_unfetched_cloud_figure_raises_rather_than_lying(tmp_path):
    """It must not return a path to a file that is not there. A path that does
    not resolve is worse than a refusal: the caller opens it and blames the
    figure rather than the fetch."""
    fig = F.Figure('X', 1, 'images', path=str(tmp_path / 'nope.png'),
                   remote='media/images/X_1_dead.png')
    with pytest.raises(NotImplementedError, match='not wired'):
        F.fetch(fig)
