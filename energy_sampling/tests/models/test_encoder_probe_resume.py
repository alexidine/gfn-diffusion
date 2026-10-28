"""`models/encoder_probe.py`: a run cut off and continued from its state file is the run that
was never cut off, a state written by another run is refused, and `--source` reads a
(dataset_index, smiles) table in place of the local QM9 file.
"""
import gzip

import pytest
import torch

from models import encoder_probe as ep

pytestmark = pytest.mark.fast

SMIS = ['CCO', 'CCCO', 'CC(C)O', 'CCOC', 'C1CCC1', 'C1CCOC1', 'CC=O', 'CC#N', 'CCN', 'CC(=O)O',
        'C1CC1C', 'CCCC', 'CC(C)C', 'OCCO', 'NCCO', 'C=CC', 'C#CC', 'CC(F)F', 'c1ccoc1', 'C1CN1']
KW = dict(steps=30, hidden=16, layers=2, k=6, batch_mols=8, lr=3e-4, seed=0, device='cpu',
          lr_schedule='cosine', lr_final=1e-5, warmup=4)


@pytest.fixture(scope='module')
def samples():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    built = [ep.build_sample(s, 'rwse', 6) for s in SMIS]
    yield built[:16], built[16:]
    torch.set_default_dtype(old)


def _weights(path):
    return torch.load(path, weights_only=False)['state_dict']


def test_a_cut_run_continues_as_the_uncut_one(samples, tmp_path, monkeypatch):
    train, test = samples
    whole = str(tmp_path / 'whole.pt')
    ref = ep.run('mp+attn+spd', train, test, save_to=whole, **KW)

    cut = str(tmp_path / 'cut.pt')
    state = str(tmp_path / 'cut_state.pt')
    real, calls = ep.collate, {'n': 0}

    def dying(batch, device, want_spd):
        # the held-out batches are collated once up front (2 calls); die inside step 17
        calls['n'] += 1
        if calls['n'] == 2 + 17:
            raise KeyboardInterrupt
        return real(batch, device, want_spd)

    monkeypatch.setattr(ep, 'collate', dying)
    with pytest.raises(KeyboardInterrupt):
        ep.run('mp+attn+spd', train, test, save_to=cut, state_path=state, save_every=5, **KW)
    monkeypatch.setattr(ep, 'collate', real)
    assert torch.load(state, weights_only=False)['step'] == 15
    got = ep.run('mp+attn+spd', train, test, save_to=cut, state_path=state, save_every=5, **KW)

    a, b = _weights(whole), _weights(cut)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert got['best_step'] == ref['best_step'] and got['best_heldout'] == ref['best_heldout']
    assert [c['step'] for c in got['curve']] == [c['step'] for c in ref['curve']]
    # a finished run's state is at the last step: running it again trains nothing
    assert torch.load(state, weights_only=False)['step'] == KW['steps']

    with pytest.raises(SystemExit, match='different run'):
        ep.run('mp+attn+spd', train, test, save_to=cut, state_path=state, save_every=5,
               **{**KW, 'hidden': 24})


def test_a_run_that_got_worse_after_its_best_point_says_so():
    point = lambda step, test, train: {'step': step, 'test_total': test, 'total': train}
    # enc03_w256l6 as it ran: best at the second curve point, far worse at the end
    blown = {'steps': 400000, 'best_step': 6666, 'best_heldout': 0.0506,
             'curve': [point(0, 240.0, 250.0), point(6666, 0.0506, 0.05), point(399999, 18.25, 1.65)]}
    msg = ep.degraded(blown)
    assert msg and 'step 6666 of 400000' in msg and '18.2' in msg
    # a run whose best is late, or whose end is near its best, says nothing
    late = dict(blown, best_step=390000)
    flat = dict(blown, curve=blown['curve'][:2] + [point(399999, 0.09, 0.03)])
    assert ep.degraded(late) is None and ep.degraded(flat) is None


def test_source_table_replaces_the_local_dataset(tmp_path):
    path = tmp_path / 'index.tsv.gz'
    with gzip.open(path, 'wt') as f:
        f.write('dataset_index\tsmiles\n0\tCCO\n1\tCCC\n2\tCCO\n3\tCCN\n')
    assert ep.load_qm9(3, source=str(path)) == ['CCO', 'CCC', 'CCN']
    with pytest.raises(RuntimeError, match='holds 3'):
        ep.load_qm9(4, source=str(path))
