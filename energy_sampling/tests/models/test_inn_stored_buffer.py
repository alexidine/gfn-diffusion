"""inn_modeller.py --stored_source buffer: the stored rows come from train.py's anchor -> prior buffers. Pinned here:
the option refusals (InnModeller.check_buffer_options), the per-condition anchor draw over the eligible pool
(pbuf_refresh_pool + _stratified_anchor_draw), the per-condition top-up quotas and their starvation exit
(pbuf_top_up), the gauss tile's kick and its bounded calibration (_gauss_kick, calibrate_gauss_sigma), and that the
INN route never reaches train.py's whole-prior seeders (which would admit held-out rows)."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import inn_modeller as im  # noqa: E402

pytestmark = pytest.mark.fast


# -- check_buffer_options ---------------------------------------------------------------

def _opts_host(**inn_over):
    inn = dict(stored_source='buffer', gauss_excess_kT=None, data_noise_log_range=None, grow_stored=False,
               rows_per_molecule=1, w_mle=0.0, pbuf_rows_per_condition=2, mle_noise_log_range=None)
    inn.update(inn_over)
    host = object.__new__(im.InnModeller)
    host.inn = SimpleNamespace(**inn)
    host.args = SimpleNamespace(
        buffers=SimpleNamespace(anchor_buffer=SimpleNamespace(tile='iso', seed_source='prior_dataset', frozen=True),
                                prior_buffer=SimpleNamespace(ramp_floor=100.0)),
        energy_config=SimpleNamespace(temperature=6.9), molecules_path='conditions.pt', prior_path='prior.pt')
    host.mol_conditional, host.t_conditional = True, False
    return host


def test_buffer_options_accept_the_plain_route_and_set_the_switch():
    host = _opts_host()
    host.check_buffer_options()
    assert host.pbuf_on
    host = _opts_host(stored_source='prior')
    host.check_buffer_options()
    assert not host.pbuf_on


def test_a_non_iso_tile_needs_the_buffer_route_and_gauss_goes_with_its_excess():
    host = _opts_host(stored_source='prior')
    host.args.buffers.anchor_buffer.tile = 'shaped'
    with pytest.raises(ValueError, match='read only by --stored_source buffer'):
        host.check_buffer_options()
    host = _opts_host()
    host.args.buffers.anchor_buffer.tile = 'gauss'
    with pytest.raises(ValueError, match='--gauss_excess_kT goes with'):
        host.check_buffer_options()
    host = _opts_host(gauss_excess_kT=6.0)
    with pytest.raises(ValueError, match='--gauss_excess_kT goes with'):
        host.check_buffer_options()            # an excess without the gauss tile


@pytest.mark.parametrize('excess', [0.0, 100.0 / 6.9, 20.0])
def test_the_gauss_excess_lies_under_the_admission_gate(excess):
    host = _opts_host(gauss_excess_kT=excess)
    host.args.buffers.anchor_buffer.tile = 'gauss'
    with pytest.raises(ValueError, match='must lie in'):
        host.check_buffer_options()
    host = _opts_host(gauss_excess_kT=6.0)
    host.args.buffers.anchor_buffer.tile = 'gauss'
    host.check_buffer_options()


@pytest.mark.parametrize('what, set_bad', [
    ('molecule conditions', lambda h: setattr(h, 'mol_conditional', False)),
    ('temperature conditioning', lambda h: setattr(h, 't_conditional', True)),
    ('--data_noise_log_range', lambda h: setattr(h.inn, 'data_noise_log_range', (-3.0, -1.5))),
    ('--grow_stored', lambda h: setattr(h.inn, 'grow_stored', True)),
    ('--rows_per_molecule', lambda h: setattr(h.inn, 'rows_per_molecule', 2)),
    ('--w_mle', lambda h: setattr(h.inn, 'w_mle', 0.1)),
    ('--pbuf_rows_per_condition', lambda h: setattr(h.inn, 'pbuf_rows_per_condition', 0)),
    ('--mle_noise_log_range', lambda h: setattr(h.inn, 'mle_noise_log_range', (-2.5, -1.5))),
    ('molecules_path == prior_path', lambda h: setattr(h.args, 'molecules_path', 'prior.pt')),
    ('seed_source', lambda h: setattr(h.args.buffers.anchor_buffer, 'seed_source', 'generated')),
    ('frozen false', lambda h: setattr(h.args.buffers.anchor_buffer, 'frozen', False)),
])
def test_buffer_route_refusals(what, set_bad):
    host = _opts_host()
    set_bad(host)
    with pytest.raises(NotImplementedError, match=what.replace('(', r'\(')):
        host.check_buffer_options()


# -- the eligible anchor pool and the per-condition draw --------------------------------------------

class _Anchors:
    def __init__(self, cid, energy):
        self.condition_id, self.energy = torch.tensor(cid), torch.tensor(energy)
        self.bumped = []
        self.batch = SimpleNamespace(subsample_new_batch=lambda inds: list(inds))

    def _bump_counts(self, inds):
        self.bumped.extend(int(i) for i in inds)

    @staticmethod
    def _drop_keys(graphs, keys):
        return graphs


def _pool_host():
    # condition 0: anchors 0, 1 under the gate, 2 over; condition 1: anchors 3, 4 under; condition 2: none under
    host = object.__new__(im.InnModeller)
    host.pbuf_on = True
    host.anchor_buffer = _Anchors([0, 0, 0, 1, 1, 2, 2], [1.0, 5.0, 50.0, 2.0, 3.0, 100.0, 200.0])
    host.energy_function = SimpleNamespace(condition_library_size=3)
    host._condition_energy_floor = lambda cid: torch.zeros(cid.shape)
    host._ramp_params = lambda: (10.0, 5.0)
    host.pbuf_train_conds = torch.tensor([0, 1, 2])
    host.pbuf_refresh_pool()
    return host


def test_the_pool_keeps_the_anchors_under_their_conditions_gate():
    host = _pool_host()
    assert host.pbuf_conds.tolist() == [0, 1]
    assert host._anc_count.tolist() == [2, 2, 0]
    assert sorted(host._anc_order.tolist()) == [0, 1, 3, 4]


def test_the_anchor_draw_gives_each_condition_its_exact_quota_from_its_own_pool(capsys):
    host = _pool_host()
    torch.manual_seed(0)
    graphs, inds = host._stratified_anchor_draw(torch.tensor([30, 20, 4]))
    inds = torch.as_tensor(inds)
    assert graphs == inds.tolist() and host.anchor_buffer.bumped == inds.tolist()
    assert int(torch.isin(inds, torch.tensor([0, 1])).sum()) == 30           # condition 0's quota, its own anchors
    assert int(torch.isin(inds, torch.tensor([3, 4])).sum()) == 20
    assert inds.numel() == 50                                                # condition 2's quota dropped, not moved
    assert set(inds.tolist()) == {0, 1, 3, 4}                                # with replacement, both anchors reached
    assert 'quota dropped' in capsys.readouterr().out
    assert host._stratified_anchor_draw(torch.tensor([0, 0, 3])) == (None, None)


# -- per-condition top-up quotas -------------------------------------------------------------------

class _Prior:
    def __init__(self, counts):
        self.counts = list(counts)

    def condition_row_counts(self, n):
        return self.counts[:n]


def _quota_host(counts, admits, k=2, conds=(0, 1, 2)):
    host = object.__new__(im.InnModeller)
    n = len(counts)
    host.energy_function = SimpleNamespace(condition_library_size=n)
    host.inn = SimpleNamespace(pbuf_rows_per_condition=k)
    host.pbuf_conds = torch.tensor(conds)
    host._pbuf_off = torch.zeros(n, dtype=torch.bool)
    host._pbuf_miss = torch.zeros(n, dtype=torch.long)
    host.prior_buffer = _Prior(counts)
    host.prior_churn = dict(from_anchors=0)
    host._prior_buffer_len = lambda: sum(host.prior_buffer.counts)
    host.asked = []

    def top_up(m, per_condition):
        q = per_condition.tolist()
        assert sum(q) == m
        host.asked.append(q)
        for c, n_c in enumerate(q):
            if c in admits:
                host.prior_buffer.counts[c] += n_c
                host.prior_churn['from_anchors'] += n_c
    host.top_up_prior_from_anchors = top_up
    return host


def test_top_up_asks_each_eligible_condition_for_its_deficit():
    host = _quota_host([0, 1, 2, 0], admits={0, 1, 2, 3})
    assert host.pbuf_top_up(10 ** 9) == 3
    assert host.asked == [[2, 1, 0, 0]]              # condition 2 full; condition 3 has no eligible anchor
    assert host.prior_buffer.counts == [2, 2, 2, 0]
    assert host.pbuf_top_up(10 ** 9) == 0 and len(host.asked) == 1   # nothing missing: no call


def test_top_up_spends_at_most_its_budget_on_missing_slots():
    host = _quota_host([0, 0, 0, 0], admits={0, 1, 2})
    assert host.pbuf_top_up(4) == 4
    q = host.asked[0]
    assert sum(q) == 4 and max(q) <= 2 and q[3] == 0


def test_a_condition_that_never_admits_leaves_the_quota_after_the_starvation_bound(capsys):
    host = _quota_host([0, 2, 2, 0], admits={1, 2})
    calls = 0
    while not bool(host._pbuf_off[0]):
        host.pbuf_top_up(10 ** 9)
        calls += 1
        assert calls <= im.PBUF_STARVE_MISSES       # bounded: 2 misses per call
    assert int(host._pbuf_miss[0]) >= im.PBUF_STARVE_MISSES
    assert 'left the quota' in capsys.readouterr().out
    n = len(host.asked)
    assert host.pbuf_top_up(10 ** 9) == 0 and len(host.asked) == n   # excluded: never asked again


def _churn_host(counts, admit_every):
    """A buffer of per-condition counts whose top-up admits every admit_every-th requested child."""
    host = _quota_host(counts, admits=set())
    seen = [0]

    def top_up(m, per_condition):
        host.asked.append(per_condition.tolist())
        host.pbuf_scored += m
        for c, n_c in enumerate(per_condition.tolist()):
            for _ in range(n_c):
                seen[0] += 1
                if seen[0] % admit_every == 0:
                    host.prior_buffer.counts[c] += 1
                    host.prior_churn['from_anchors'] += 1

    def purge_by_index(drop):
        rows = [c for c, n in enumerate(host.prior_buffer.counts) for _ in range(n)]
        for i in drop:
            host.prior_buffer.counts[rows[int(i)]] -= 1
    host.top_up_prior_from_anchors = top_up
    host.prior_buffer.purge_by_index = purge_by_index
    host.prior_churn.update(evicted=0)
    host.args = SimpleNamespace(buffers=SimpleNamespace(prior_buffer=SimpleNamespace(churn_batch_ref=1000,
                                                                                     mean_lifetime=100)))
    host._expire_stale_prior_rows = lambda: None
    host.pbuf_refresh_pool = lambda: None
    host.pbuf_scored, host.energy_samples_scored, host._pbuf_mark = 0, 0, 0
    host.pbuf_target = 2 * 3
    host.pbuf_window = dict(drawn=0, retired=0, churn_scored=0)
    return host


def test_churn_retires_at_the_configured_rate_and_refills_in_bounded_passes():
    host = _churn_host([2] * 3 + [0], admit_every=2)   # 6 rows, half of all children admitted
    host.pbuf_updates, host.pbuf_updates_churned = 1, 0
    host.inn.pbuf_rows_per_condition = 2
    host.pbuf_conds = torch.tensor([0, 1, 2])
    torch.manual_seed(0)
    host.pbuf_churn()
    assert host.prior_churn['evicted'] == 6               # 1 update x 1000 / 100 = 10, capped at the 6 rows
    assert len(host.asked) <= 4 + im.PBUF_REFILL_ROUNDS   # bounded: 4 per fill cycle of deficit + the rounds
    assert len(host.asked) >= 2                           # a pass that left a deficit was followed by another
    assert sum(host.prior_buffer.counts[:3]) >= 5         # one pass alone would leave half the slots empty
    n_asked = len(host.asked)
    host.pbuf_churn()                                     # no update since: nothing retired
    assert host.prior_churn['evicted'] == 6
    assert host.energy_samples_scored == host.pbuf_scored == host.pbuf_window['churn_scored']   # each counted once


def test_churn_is_safe_to_retry_and_clears_the_starvation_exclusion():
    host = _churn_host([2] * 3 + [0], admit_every=1)
    host.pbuf_updates, host.pbuf_updates_churned = 1, 0
    host.inn.pbuf_rows_per_condition = 2
    host.pbuf_conds = torch.tensor([0, 1, 2])
    host._pbuf_off[1] = True                              # excluded by an earlier refill
    real, calls = host.top_up_prior_from_anchors, []

    def flaky(m, per_condition):                          # the first refill pass dies after scoring (an OOM)
        real(m, per_condition)
        calls.append(m)
        if len(calls) == 1:
            raise RuntimeError('CUDA out of memory')
    host.top_up_prior_from_anchors = flaky
    with pytest.raises(RuntimeError):
        host.pbuf_churn()
    assert host.prior_churn['evicted'] == 6               # retired once
    host.pbuf_churn()                                     # the retry: no second retirement, the refill runs
    assert host.prior_churn['evicted'] == 6
    assert host.prior_buffer.counts[:3] == [2, 2, 2]      # condition 1 asked again despite its old exclusion
    assert not bool(host._pbuf_off.any())
    assert host.energy_samples_scored == host.pbuf_scored   # the failed call's children counted, once


def test_a_buffer_update_is_committed_only_once_it_steps():
    host = object.__new__(im.InnModeller)
    got = []
    host.prior_buffer = SimpleNamespace(update_losses=lambda losses, inds: got.append((losses, inds)))
    host.pbuf_updates = 0
    host.commit_pbuf_step()                               # no draw: nothing counted, nothing written
    assert got == [] and host.pbuf_updates == 0
    host._pbuf_drawn, host._pbuf_pending = True, None     # a stage-1 update: counted, no ema_loss
    host.commit_pbuf_step()
    assert got == [] and host.pbuf_updates == 1
    host._pbuf_drawn, host._pbuf_pending = True, (torch.tensor([1.0, 2.0]), [3, 4])   # an energy-stage update
    host.commit_pbuf_step()
    assert len(got) == 1 and host.pbuf_updates == 2
    host.commit_pbuf_step()                               # committed once
    assert len(got) == 1 and host.pbuf_updates == 2


# -- the gauss tile ---------------------------------------------------------------------------

class _Cells:
    def __init__(self, lat):
        self.lat, self.decoded, self.cleaned = lat, None, None

    def latent_params(self, gauge_fix_free_axes):
        assert gauge_fix_free_axes is False
        return self.lat.clone()

    def latent_to_cell_params(self, x):
        self.decoded = x

    def clean_cell_parameters(self, mode):
        self.cleaned = mode


def test_the_gauss_kick_adds_sigma_times_the_noise_clips_decodes_and_cleans():
    lat = torch.tensor([[0.0, 0.5, -0.99], [0.2, 0.999, -0.2]])
    eps = torch.tensor([[1.0, -1.0, -1.0], [0.5, 1.0, 2.0]])
    b = _Cells(lat)
    im.InnModeller._gauss_kick(b, 0.1, eps=eps)
    want = (lat + 0.1 * eps).clip(min=-1 + 1e-6, max=1 - 1e-6)
    assert torch.allclose(b.decoded, want) and b.cleaned == 'hard'
    assert float(b.decoded.abs().max()) < 1.0


def test_gauss_calibration_reaches_a_quadratic_target_in_few_rounds():
    calls = []

    def score(s):
        calls.append(s)
        return 600.0 * s * s, 0.05             # excess quadratic in sigma: sigma* = 0.1 for 6 kT
    sigma, table = im.calibrate_gauss_sigma(score, 6.0, 0.01)
    assert abs(sigma - 0.1) < 0.01 * 0.1 and len(table) == len(calls) <= im.GAUSS_PILOT_ROUNDS
    assert abs(table[-1][1] - 6.0) <= im.GAUSS_TOL * 6.0


def test_gauss_calibration_is_bounded_and_raises_when_no_scale_reaches_the_target():
    calls = []

    def flat(s):
        calls.append(s)
        return 0.5, 0.01                        # an excess that no scale moves
    with pytest.raises(RuntimeError, match='no scale reached'):
        im.calibrate_gauss_sigma(flat, 6.0, 0.01)
    assert len(calls) == im.GAUSS_PILOT_ROUNDS


def test_gauss_calibration_raises_on_a_nonfinite_pilot():
    with pytest.raises(RuntimeError, match='non-finite'):
        im.calibrate_gauss_sigma(lambda s: (float('nan'), 0.1), 6.0, 0.01)


def test_gauss_calibration_widens_from_a_nonpositive_excess():
    sigma, table = im.calibrate_gauss_sigma(lambda s: (600.0 * s * s if s > 0.05 else -0.1, 0.05), 6.0, 0.01)
    assert table[1][0] == pytest.approx(0.04)   # x 4 while the excess is <= 0
    assert abs(sigma - 0.1) < 0.01


# -- the conditions file's carriers and Emin(c) ----------------------------------------------------------

class _Sub:
    def __init__(self, n):
        self.n = n

    def to(self, dev):
        return self

    def orient_molecule(self, mode):
        pass


def _fold_host():
    host = object.__new__(im.InnModeller)
    host.args = SimpleNamespace(energy_config=SimpleNamespace(temperature=6.9), seed=0)
    host.inn = SimpleNamespace(atom_tokens=False)
    host.device = torch.device('cpu')
    host.floors = []
    host.condition_log_z = SimpleNamespace(update_best_energy=lambda cid, e: host.floors.append((cid, e)))
    host.test_floor = torch.full((4,), float('inf'))
    host.energy_function = SimpleNamespace(
        condition_samples=lambda sub: (sub, torch.zeros(sub.n), None, torch.arange(sub.n)),
        log_reward=lambda x, sub, log_T, grad, internal_oom_recovery: -torch.ones(sub.n))
    host._batch_latents = lambda sub: torch.zeros(sub.n, 3)
    host.note_conditions = lambda cid, cond: None
    batch = SimpleNamespace(num_graphs=4, device='cpu', subsample_new_batch=lambda idx: _Sub(int(idx.numel())))
    return host, SimpleNamespace(batch=batch)


def test_the_training_carriers_set_emin_except_on_the_buffer_route():
    host, ds = _fold_host()
    host.fold_condition_crystals(ds, held_out=False)
    assert len(host.floors) == 1                          # the prior route folds the carriers
    host, ds = _fold_host()
    host.fold_condition_crystals(ds, held_out=False, fold_floor=False)
    assert host.floors == [] and host.mol_crystals_train[0].shape[0] == 4   # floors left alone, the rest kept


# -- no whole-prior seeding on the INN route ---------------------------------------------------------

def test_the_inn_route_reaches_no_whole_prior_seeder():
    """train.py's prior-buffer seeders read the whole prior dataset, held-out rows included; the INN route seeds
    anchors only through init_anchor_buffer_seed under init_inn_buffers' training-row view."""
    src = Path(im.__file__).read_text(encoding='utf-8')
    tree = ast.parse(src)
    forbidden = {'init_prior_buffer_seed', 'seed_prior_from_condition_minima', 'rebuild_prior_by_churn',
                 'manage_prior_buffer', 'init_buffers'}
    called = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    called.setdefault(node.func.attr, set()).add(fn.name)
    assert not forbidden & set(called), {k: called[k] for k in forbidden & set(called)}
    assert called.get('init_anchor_buffer_seed') == {'init_inn_buffers'}
