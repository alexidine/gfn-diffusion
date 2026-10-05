"""cfg:prior_scan_cache -- the cache of the init-time prior scoring pass (prior_scan_cache.py and its two trainer
methods).

The claims pinned here: the identity moves with everything the config says about the scoring and with the prior file's
content; a cache is used only when its identity matches AND a re-score of a sample of rows agrees; any failure to use
it returns None, which makes the caller score every row (the behaviour without the key).
"""
import types

import pytest

torch = pytest.importorskip("torch")

from energy_sampling import prior_scan_cache as psc  # noqa: E402

pytestmark = pytest.mark.slow


class FakeBatch:
    """The three things the cache asks of a batch: clone/detach/cpu/to, num_graphs, subsample_new_batch."""

    def __init__(self, x):
        self.x = x

    num_graphs = property(lambda self: len(self.x))

    def clone(self):
        return FakeBatch(self.x.clone())

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, device):
        return self

    def subsample_new_batch(self, idx):
        return FakeBatch(self.x[idx])


def _args(tmp_path, content=b'prior-bytes', **energy):
    prior = tmp_path / 'prior.pt'
    prior.write_bytes(content)
    cfg = dict(temperature=2.5, density_coeff=10.0, reward_range=250)
    cfg.update(energy)
    return types.SimpleNamespace(prior_path=str(prior), energy_function='elj', mlip_path=None, space_groups=[2],
                                 z_primes=[1], energy_config=types.SimpleNamespace(**cfg))


def test_identity_moves_with_the_file_and_the_scoring_config(tmp_path):
    base = psc.scan_identity(_args(tmp_path), 10, 0.36)
    assert base == psc.scan_identity(_args(tmp_path), 10, 0.36), 'the same inputs give the same identity'
    moved = [
        psc.scan_identity(_args(tmp_path, content=b'other-bytes'), 10, 0.36),
        psc.scan_identity(_args(tmp_path), 11, 0.36),
        psc.scan_identity(_args(tmp_path), 10, 0.37),
        psc.scan_identity(_args(tmp_path, temperature=3.0), 10, 0.36),
        psc.scan_identity(_args(tmp_path, density_coeff=1.0), 10, 0.36),
    ]
    a = _args(tmp_path)
    a.energy_function = 'uma'
    moved.append(psc.scan_identity(a, 10, 0.36))
    a = _args(tmp_path)
    a.space_groups = [14]
    moved.append(psc.scan_identity(a, 10, 0.36))
    hashes = {psc.identity_hash(m) for m in moved}
    assert len(hashes) == len(moved) and psc.identity_hash(base) not in hashes
    assert psc.cache_path('p.pt', base) == f'p.pt.scan-{psc.identity_hash(base)}.pt'


def test_the_mlip_file_is_part_of_the_identity(tmp_path):
    a = _args(tmp_path)
    m = tmp_path / 'model.pt'
    m.write_bytes(b'x' * 100)
    a.mlip_path = str(m)
    one = psc.scan_identity(a, 10, None)
    m.write_bytes(b'x' * 101)
    assert one != psc.scan_identity(a, 10, None), 'a replaced checkpoint of another size is another identity'


def test_save_and_load_round_trip_and_every_refusal_returns_none(tmp_path):
    a = _args(tmp_path)
    ident = psc.scan_identity(a, 4, 0.36)
    path = psc.cache_path(a.prior_path, ident)
    assert psc.load(path, ident) is None, 'no file'
    energy = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert psc.save(path, ident, energy, FakeBatch(torch.arange(4.0)))
    got = psc.load(path, ident)
    assert got is not None and torch.equal(got[0], energy) and torch.equal(got[1].x, torch.arange(4.0))
    assert psc.load(path, psc.scan_identity(a, 5, 0.36)) is None, 'another identity under the same file name'
    with open(path, 'wb') as fh:
        fh.write(b'truncated')
    assert psc.load(path, ident) is None, 'an unreadable file'
    assert not psc.save(str(tmp_path / 'no_such_dir' / 'c.pt'), ident, energy, FakeBatch(torch.arange(4.0))), \
        'a cache that cannot be written is reported, not raised'


def test_agreement_rule():
    e = torch.linspace(-10, 10, 400)
    assert psc.agrees(e, e + 0.1)[0], 'every row within the tolerance'
    few = e.clone()
    few[:3] += 5.0                                    # 0.75% of rows off: under the 1% bar
    assert psc.agrees(e, few)[0]
    many = e.clone()
    many[:8] += 5.0                                   # 2% of rows off
    ok, frac, _med, mx = psc.agrees(e, many)
    assert not ok and frac == pytest.approx(0.02) and mx == pytest.approx(5.0)
    assert not psc.agrees(e, e[:399])[0], 'a different row count never agrees'
    nan = e.clone()
    nan[:40] = float('nan')
    assert not psc.agrees(e, nan)[0], 'a fresh NaN where the cache holds a number is a disagreement'
    assert psc.agrees(nan, nan)[0], 'the same rows non-finite on both sides are not'


def test_check_rows_are_fixed_distinct_and_bounded():
    a, b = psc.check_rows(10_000), psc.check_rows(10_000)
    assert torch.equal(a, b) and len(a) == psc.CHECK_ROWS and len(set(a.tolist())) == len(a)
    assert sorted(psc.check_rows(7).tolist()) == list(range(7)), 'a prior smaller than the sample is checked whole'


def _modeller(tmp_path, rescored):
    """The two trainer methods bound to a stand-in: `rescored` maps the checked rows' x to the fresh energies."""
    train = pytest.importorskip("energy_sampling.train")
    fake = types.SimpleNamespace(args=_args(tmp_path), device='cpu',
                                 energy_function=types.SimpleNamespace(lj_coeff=0.36))
    fake._score_prior_rows = lambda batch, announce=True: (rescored(batch.x), batch)
    fake._prior_scan_from_cache = types.MethodType(train.Modeller._prior_scan_from_cache, fake)
    return fake


def test_the_trainer_uses_a_cache_only_when_the_rows_agree(tmp_path):
    n = 2000
    prior = FakeBatch(torch.arange(float(n)))
    energy = -prior.x                                 # the "true" energy of row x is -x
    good = _modeller(tmp_path, lambda x: -x)
    assert good._prior_scan_from_cache(prior) is None, 'no cache yet: the caller scores every row'
    ident = psc.scan_identity(good.args, n, 0.36)
    path = psc.cache_path(good.args.prior_path, ident)
    psc.save(path, ident, energy, prior)
    got = good._prior_scan_from_cache(prior)
    assert got is not None and torch.equal(got[0], energy) and got[1].num_graphs == n
    stale = _modeller(tmp_path, lambda x: -x + 3.0)   # the scoring has changed under the cache
    assert stale._prior_scan_from_cache(prior) is None
    assert good._prior_scan_from_cache(FakeBatch(torch.arange(float(n + 1)))) is None, 'another row count'
    good.energy_function.lj_coeff = 0.5               # the run applies another eLJ coefficient
    assert good._prior_scan_from_cache(prior) is None
