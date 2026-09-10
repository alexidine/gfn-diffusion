"""Every generated sbatch must put the PROJECT ROOT on PYTHONPATH.

train.py does `from energy_sampling.eval...`, so the PARENT of energy_sampling/
must be importable. An sbatch that cd's into energy_sampling/ and runs
`python train.py` leaves only energy_sampling/ on sys.path, and every arm dies at
import with ModuleNotFoundError -- after the queue slot is spent, with no wandb
run to show for it. mle_sep09 lost a whole 8-arm battery to exactly this.

The container binds are the same shape of failure one step later: the seed glob
runs OUTSIDE the container and passes regardless, so a missing --bind fails at
data load and looks unrelated to the launch line.
"""
import pathlib

import pytest

CONFIGS = pathlib.Path(__file__).resolve().parents[2] / 'configs'
SBATCH = sorted(CONFIGS.glob('*/*.sbatch'))


def _launch(text):
    """the singularity invocation, which is where all of this lives"""
    i = text.find('singularity exec')
    return text[i:i + 1200] if i >= 0 else ''


@pytest.mark.parametrize('path', SBATCH, ids=lambda p: f'{p.parent.name}/{p.name}')
def test_the_project_root_reaches_pythonpath(path):
    launch = _launch(path.read_text(encoding='utf-8'))
    if not launch:
        pytest.skip('no singularity launch in this sbatch')
    assert 'PYTHONPATH' in launch, (
        f'{path.parent.name}/{path.name} never sets PYTHONPATH: train.py imports '
        f'`energy_sampling.*`, so the parent directory must be on sys.path or '
        f'every arm dies at import')
    assert 'gfn-diffusion' in launch, (
        f'{path.parent.name}/{path.name} sets PYTHONPATH without the project root')


@pytest.mark.parametrize('path', SBATCH, ids=lambda p: f'{p.parent.name}/{p.name}')
def test_the_container_can_see_the_project_and_the_data(path):
    launch = _launch(path.read_text(encoding='utf-8'))
    if not launch:
        pytest.skip('no singularity launch in this sbatch')
    assert '--bind' in launch, (
        f'{path.parent.name}/{path.name} binds nothing into the container -- the '
        f'project tree and /scratch data are invisible inside it')
