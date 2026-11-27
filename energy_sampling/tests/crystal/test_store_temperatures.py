"""Multi-temperature TB on the crystal store (train.py Modeller._stage_temperatures, ._draw_temperatures,
._at_run_temperature, ._store_draw, ._store_admit, ._check_store_terms, ._eval_pool_add, ._row_tracker_rows,
.fwd_train_step's store branch, .bootstrap_z_offset, ._store_check_currency, ._temperature_readouts), on a Modeller
built without __init__ around a real CrystalStore with term columns and a stub energy whose log R is an exact function
of its terms. CPU.

    python -m pytest -q tests/crystal/test_store_temperatures.py
"""
import math
from types import SimpleNamespace

import pytest
import torch

from energy_sampling import crystal_store as CS
from energy_sampling.buffer import ORIGIN_BOOTSTRAP, ORIGIN_EVAL, ORIGIN_ROLLOUT
from energy_sampling.tests.crystal import store_stub as S
from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast
T_RUN = S.T


class TermEnergy(S.Energy):
    """log R(T) = -(U + T pen - T logref) / T with U = mol_energy + 10 dens + pv, from per-row terms that depend on the
    state and condition only: the shape MolecularCrystal.log_r_from_terms rebuilds."""
    temperature_conditioning = True
    with_terms = True

    def __init__(self):
        super().__init__()
        self.untagged_temperature = T_RUN

    @staticmethod
    def seed_terms(x, cid):
        sq = (x ** 2).sum(-1).float()
        c = torch.as_tensor(cid).float()
        return torch.stack([0.1 * sq + 0.01, 0.01 * c, 0.2 * sq, -0.05 * c - 0.1], 1)

    @staticmethod
    def mol_energy(x, cid):
        return (x ** 2).sum(-1).float() + torch.as_tensor(cid).float()

    def log_r_from_terms(self, energy, terms, log_T):
        energy = torch.as_tensor(energy).float().flatten()
        T = 10 ** torch.as_tensor(log_T, dtype=torch.float32).reshape(-1)
        dens, pv, pen, logref = torch.as_tensor(terms).float().unbind(1)
        u = energy + 10.0 * dens + pv
        return -(u + T * pen - T * logref) / T

    def seed_rows(self, x, cid):
        e = self.mol_energy(x, cid)
        return e, self.log_r_from_terms(e, self.seed_terms(x, cid), math.log10(T_RUN))

    @staticmethod
    def store_terms_from(src):
        return src.terms

    def condition_samples(self, mol, temperature=None, sg_inds=None, z_primes=None, repeats=1):
        cid = mol.mol_id.reshape(-1).long() * self.n_sg + self.sg_to_local[sg_inds.reshape(-1).long()]
        if temperature is None:
            temperature = torch.full((mol.num_graphs,), float(self.untagged_temperature))
        log_T = torch.log10(torch.as_tensor(temperature, dtype=torch.float32)).reshape(-1)
        cond = torch.stack([log_T, cid.float() * 0.1], 1)
        mol.condition_id, mol.conditions = cid, cond
        return mol, log_T, cond, cid

    def scored(self, x, cid):
        terms = self.seed_terms(x, cid)
        return SimpleNamespace(mol_energy=self.mol_energy(x, cid), density_energy=terms[:, 0],
                               pressure_energy=terms[:, 1], packing_coeff=torch.full((len(x),), float(self.packing)),
                               terms=terms)

    def log_reward(self, x, mol_batch=None, log_temperature=None, return_exp=False, internal_oom_recovery=None):
        cid = mol_batch.condition_id
        self.calls.append(x.detach().clone())
        sc = self.scored(x, cid)
        log_r = self.log_r_from_terms(sc.mol_energy, sc.terms, log_temperature)
        return (log_r, sc) if return_exp else log_r


def build(temps=None, **kw):
    m = S.build(ef=TermEnergy(), **kw)
    m.protocol = SimpleNamespace(stage=SimpleNamespace(name='tb', bwd_sampling_mode='prior', temperatures=temps),
                                 flag=lambda name: name == 'buffers_active')
    m._store_cfg = dict(mle_draw='anchor', replay_weight='visits')
    return m


def rollout(m, cid, temps):
    """(fwd_stats, sample_batch) of a scored rollout on conditions `cid`, row i at kT temps[i]."""
    ef = m.energy_function
    cid = torch.as_tensor(cid)
    n = cid.numel()
    x = torch.linspace(-0.5, 0.5, n * S.DIM).reshape(n, S.DIM)
    log_T = torch.log10(torch.as_tensor(temps, dtype=torch.float32))
    sc = ef.scored(x, cid)
    log_r = ef.log_r_from_terms(sc.mol_energy, sc.terms, log_T)
    fwd = {'condition_id': cid, 'log_r': log_r, 'log_pf': torch.full((n,), -2.0), 'log_pb': torch.zeros(n),
           'log_Z_learned': torch.full((n,), 3.0), 'log_Z': torch.full((n,), 3.0), 'log_T_tensor': log_T,
           'flow_states': torch.stack([torch.zeros_like(x), x], 1)}
    sc.condition_id = cid
    return fwd, sc


# ---- the stratified draw --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('n', [1, 2, 3, 7, 16, 33])
@pytest.mark.parametrize('temps', [(25.0, 2.0), (25.0, 10.0, 5.0, 2.0)])
def test_the_draw_takes_each_entry_within_one_row_of_every_other(n, temps):
    for seed in range(20):
        torch.manual_seed(seed)
        got = Modeller._draw_temperatures(temps, n)
        assert got.shape == (n,) and got.dtype == torch.float32
        counts = [int((got == v).sum()) for v in temps]
        assert sum(counts) == n and max(counts) - min(counts) <= 1
        if n <= len(temps):                                 # fewer rows than entries: no entry twice
            assert len(set(got.tolist())) == n


def test_a_repeated_entry_counts_as_a_weight():
    torch.manual_seed(0)
    got = Modeller._draw_temperatures((25.0, 25.0, 2.0), 30)
    assert int((got == 25.0).sum()) == 20 and int((got == 2.0).sum()) == 10


def test_the_draw_reads_the_global_generator():
    torch.manual_seed(5)
    a = Modeller._draw_temperatures((25.0, 10.0, 2.0), 50)
    torch.manual_seed(5)
    assert torch.equal(a, Modeller._draw_temperatures((25.0, 10.0, 2.0), 50))


def test_run_temperature_rows_are_recognised():
    m = build()
    lt = torch.log10(torch.tensor([T_RUN, 25.0, T_RUN], dtype=torch.float32))
    assert m._at_run_temperature(lt).tolist() == [True, False, True]
    from energy_sampling import config_invariants as ci
    assert ci.RUN_TEMPERATURE_ATOL == Modeller.RUN_TEMPERATURE_ATOL      # the config rules judge the same match


# ---- bwd and replay draws ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('kind', ['bwd', 'replay'])
def test_each_drawn_row_takes_a_listed_temperature_and_its_rebuilt_log_r(kind):
    m = build(temps=(25.0, T_RUN), batch=9)
    st, ef = m.row_store, m.energy_function
    if kind == 'replay':
        fwd, batch = rollout(m, [0, 1, 2, 3] * 3, [T_RUN] * 6 + [25.0] * 6)
        m._store_admit(fwd, batch, ORIGIN_ROLLOUT)
    R = 2
    torch.manual_seed(0)
    condition, cid, inds, latents, log_reward, mol, traj = m._store_draw(kind, R)
    idx = inds[0]
    log_T = mol.row_log_T
    assert torch.equal(condition[:, 0], log_T)
    want = ef.log_r_from_terms(st.energy[idx], st.terms(idx), log_T)
    assert torch.equal(log_reward, want)
    tiles = log_T.reshape(-1, R)
    assert bool((tiles == tiles[:, :1]).all())                     # the rows of one repeats tile share T
    rows = tiles[:, 0]
    n_hot, n_run = int((rows == math.log10(25.0)).sum()), int(m._at_run_temperature(rows).sum())
    assert n_hot + n_run == rows.numel() and abs(n_hot - n_run) <= 1
    assert m._row_tracker_rows(mol).tolist() == m._at_run_temperature(log_T).tolist()


def test_without_temperatures_the_draw_reads_the_stored_log_r_bit_for_bit():
    m = build(temps=None, batch=9)
    st = m.row_store
    condition, cid, inds, latents, log_reward, mol, traj = m._store_draw('bwd', 2)
    assert torch.equal(log_reward, st.log_r[inds[0]])
    assert torch.equal(condition[:, 0], torch.full_like(condition[:, 0], math.log10(T_RUN)))
    assert getattr(mol, 'row_log_T', None) is None and m._row_tracker_rows(mol) is None


def test_a_stage_with_temperatures_on_a_store_without_terms_raises():
    m = S.build()
    m.protocol = SimpleNamespace(stage=SimpleNamespace(name='tb', temperatures=(25.0,)))
    with pytest.raises(RuntimeError, match='without its term columns'):
        m._stage_temperatures()


# ---- admission ------------------------------------------------------------------------------------------------------

def test_admission_stores_log_r_at_the_run_temperature():
    m = build(temps=(25.0, T_RUN))
    st, ef = m.row_store, m.energy_function
    cid = [0, 1, 2, 3]
    fwd, batch = rollout(m, cid, [25.0, 25.0, T_RUN, T_RUN])
    slots = m._store_admit(fwd, batch, ORIGIN_ROLLOUT)
    assert slots.numel() == 4
    x = fwd['flow_states'][:, -1]
    rescored = ef.log_r_from_terms(batch.mol_energy, batch.terms, math.log10(T_RUN))
    torch.testing.assert_close(st.log_r[slots[:2]], rescored[:2], rtol=0, atol=1e-5)   # the hot rows, re-scored
    assert torch.equal(st.log_r[slots[2:]], fwd['log_r'][2:])                         # the run's rows, as scored
    assert torch.equal(st.terms(slots), batch.terms)
    assert torch.equal(st.log_q[slots], fwd['log_pf'])                                # log q stays at the birth T
    st.check()
    assert x.shape == (4, S.DIM)


def test_admission_on_a_store_without_terms_writes_the_live_reward_and_no_terms():
    m = S.build()
    st = m.row_store
    fwd, batch = S.rollout([0, 1, 2], log_r=[-1.5, -2.5, -3.5], energy=[4.0, 5.0, 6.0])
    slots = m._store_admit(fwd, batch, ORIGIN_ROLLOUT)
    assert torch.equal(st.log_r[slots], fwd['log_r']) and not st.with_terms
    assert not any(hasattr(st, name) for name in CS.TERM_COLUMNS)


def test_admission_raises_when_the_terms_do_not_rebuild_the_live_reward():
    m = build(temps=(25.0, T_RUN))
    fwd, batch = rollout(m, [0, 1], [25.0, T_RUN])
    batch.terms = batch.terms.clone()
    batch.terms[0, 2] += 1.0                                     # one row's penalty corrupted
    before = m.row_store.n_rows
    with pytest.raises(RuntimeError, match=r'store terms \(admission\): the rebuilt log R of 1 of 2 rows'):
        m._store_admit(fwd, batch, ORIGIN_ROLLOUT)
    assert m.row_store.n_rows == before


def test_admission_raises_on_a_finite_row_with_a_non_finite_term_and_drops_a_non_finite_one():
    m = build(temps=(25.0, T_RUN))
    fwd, batch = rollout(m, [0, 1], [25.0, T_RUN])
    batch.terms = batch.terms.clone()
    batch.terms[1, 0] = float('nan')
    with pytest.raises(RuntimeError, match=r'1 of 2 rows with a finite log R and energy have a non-finite term'):
        m._store_admit(fwd, batch, ORIGIN_ROLLOUT)
    fwd['log_r'] = fwd['log_r'].clone()
    fwd['log_r'][1] = float('nan')                               # the store drops this row, as without terms
    slots = m._store_admit(fwd, batch, ORIGIN_ROLLOUT)
    assert slots.numel() == 1 and m.row_store.nonfinite_rows == 1


def test_the_eval_pool_keeps_only_run_temperature_rows():
    m = build(temps=(25.0, T_RUN))
    m._eval_pool = CS.RolloutPool(16)
    seen = []
    fwd, batch = rollout(m, [0, 1, 2, 3], [25.0, T_RUN, 25.0, T_RUN])
    batch.ptr = torch.zeros(5, dtype=torch.long)
    batch.subsample_new_batch = lambda at: seen.append(at.tolist()) or SimpleNamespace(num_graphs=len(at))
    m._eval_pool_add(fwd, batch)
    assert m._eval_pool.rows == 2 and seen == [[1, 3]]
    rows, _, _ = m._eval_pool.read(lambda bs: SimpleNamespace(clone=lambda: None))
    assert bool(m._at_run_temperature(rows['log_T_tensor']).all())
    fwd, batch = rollout(m, [0, 1], [25.0, 25.0])
    m._eval_pool_add(fwd, batch)
    assert m._eval_pool.rows == 2                               # no run-temperature row: nothing added


def test_growth_writes_terms_scored_at_the_run_temperature(monkeypatch):
    m = build(temps=(25.0, T_RUN))
    S.shift_kick(monkeypatch)
    m._growth_owed = 3
    m._anchor_energy = lambda scored, energy: (energy, None)
    m.condition_log_z = SimpleNamespace(update_best_energy=lambda *a, **k: None)
    before = m.row_store.high_water
    m._store_grow()
    st = m.row_store
    new = torch.arange(before, st.high_water)
    assert new.numel() == 3
    want = m.energy_function.log_r_from_terms(st.energy[new], st.terms(new), math.log10(T_RUN))
    torch.testing.assert_close(st.log_r[new], want, rtol=0, atol=1e-5)
    st.check()


# ---- the forward branch ---------------------------------------------------------------------------------------------

def _fwd_stub(m, monkeypatch):
    import energy_sampling.train as train_mod
    got = {}

    def loss(*args, **kw):
        got.update(kw, log_T=args[6], mol=args[5])
        return torch.tensor(0.0), None
    monkeypatch.setattr(train_mod, 'get_gfn_forward_loss', loss)
    m._rollout_mol_batch = lambda r: m.row_store.templates.subsample_new_batch(torch.tensor([0, 2, 0, 2]))
    m._rollout_group_mode = lambda: 'uniform'
    m._fwd_exploration_std = lambda s: None
    m.tb_z_source = lambda b: 'learned'
    m._stash_z_cal_cache = lambda c: None
    m._anchor_energy_phys = None
    m.condition_log_z = None
    m.gfn_model = SimpleNamespace()
    m.args.fwd_loss_coeffs = SimpleNamespace()
    m.energy_function.data_ndim = S.DIM
    return got


def test_a_forward_rollout_draws_one_temperature_per_molecule_and_masks_the_tracker(monkeypatch):
    m = build(temps=(25.0, T_RUN))
    got = _fwd_stub(m, monkeypatch)
    torch.manual_seed(1)
    m.fwd_train_step(None, repeats=2)
    log_T = got['log_T']
    tiles = log_T.reshape(-1, 2)
    assert bool((tiles == tiles[:, :1]).all())
    hot = int((tiles[:, 0] == math.log10(25.0)).sum())
    assert abs(hot - (tiles.shape[0] - hot)) <= 1
    assert got['group_by_temperature'] is True
    assert got['tracker_rows'].tolist() == m._at_run_temperature(log_T).tolist()


def test_a_forward_rollout_without_temperatures_passes_neither_key(monkeypatch):
    m = build(temps=None)
    got = _fwd_stub(m, monkeypatch)
    state = torch.get_rng_state()
    m.fwd_train_step(None, repeats=2)
    assert 'tracker_rows' not in got and 'group_by_temperature' not in got
    assert bool((got['log_T'] == math.log10(T_RUN)).all())
    assert torch.equal(torch.get_rng_state(), state)             # no temperature draw reads the generator


# ---- bootstrap ------------------------------------------------------------------------------------------------------

class Policy:
    def __init__(self):
        self.shifts = []

    def shift_level(self, d):
        self.shifts.append(float(d))


def _boot(m, monkeypatch, temps_rows, logw_rows):
    """bootstrap_z_offset over stats whose row i is at kT temps_rows[i] with log w logw_rows[i] and head 0."""
    import energy_sampling.train as train_mod
    pol = Policy()
    m._policies = lambda: [pol]
    m.gfn_model = SimpleNamespace()
    m.args.fwd_loss_coeffs = SimpleNamespace(beta=10.0)
    m.args.eval_num_samples = len(logw_rows)
    monkeypatch.setattr(train_mod, 'get_discretizer', lambda *a: None)
    monkeypatch.setattr(train_mod, 'trajectory_T', lambda a: 1)
    asked = {}

    def sampling(model, disc, override_num_samples=None, side_effects=True, temperatures=None):
        asked['temperatures'] = temperatures
        n = len(logw_rows)
        stats = {'log_r': torch.tensor(logw_rows, dtype=torch.float32), 'log_pbs': torch.zeros(n, 1),
                 'log_pfs': torch.zeros(n, 1), 'log_flow': torch.zeros(n, 2),
                 'log_T_tensor': torch.log10(torch.tensor(temps_rows, dtype=torch.float32))}
        return stats, None
    m.fwd_eval_sampling = sampling
    m._store_admit = lambda *a, **k: None
    m.bootstrap_z_offset()
    return pol, asked


def test_bootstrap_takes_one_root_per_listed_temperature_and_shifts_by_the_smallest(monkeypatch):
    m = build(temps=(25.0, T_RUN))
    pol, asked = _boot(m, monkeypatch, [25.0] * 4 + [T_RUN] * 4, [1.0, 2.0, 3.0, 2.0, -5.0, -6.0, -4.0, -5.0])
    assert asked['temperatures'] == (25.0, T_RUN)
    assert pol.shifts == [pytest.approx(-5.0)]
    roots = m._temperature_readouts()
    assert roots['boot/root@25'] == pytest.approx(2.0) and roots['boot/root@2'] == pytest.approx(-5.0)


def test_bootstrap_with_one_listed_temperature_applies_the_single_root(monkeypatch):
    m = build(temps=(25.0,))
    pol, asked = _boot(m, monkeypatch, [25.0] * 4, [1.0, 2.0, 3.0, 2.0])
    assert asked['temperatures'] == (25.0,) and pol.shifts == [pytest.approx(2.0)]


def test_bootstrap_without_temperatures_draws_at_the_run_temperature(monkeypatch):
    m = build(temps=None)
    pol, asked = _boot(m, monkeypatch, [T_RUN] * 4, [1.0, 2.0, 3.0, 2.0])
    assert asked['temperatures'] is None and pol.shifts == [pytest.approx(2.0)]


# ---- readouts and the smoke currency check --------------------------------------------------------------------------

def test_per_temperature_readouts_count_rows_and_residuals():
    m = build(temps=(25.0, T_RUN))
    lt = torch.log10(torch.tensor([25.0, 25.0, T_RUN], dtype=torch.float32))
    m._tally_temperature_rows('bwd', lt, torch.tensor([1.0, 3.0, -2.0]))
    m._tally_temperature_rows('fwd', lt, torch.tensor([0.0, 2.0, 4.0]), torch.tensor([0.6, 0.8, 0.7]))
    out = m._temperature_readouts()
    assert out['tbT/bwd/rows@25'] == 2 and out['tbT/bwd/resid_mean@25'] == 2.0
    assert out['tbT/bwd/resid_rms@25'] == pytest.approx(math.sqrt(5.0)) and out['tbT/bwd/rows@2'] == 1
    assert out['tbT/fwd/packing_p50@25'] == pytest.approx(0.7) and out['tbT/fwd/packing_p50@2'] == pytest.approx(0.7)
    assert m._temperature_readouts() == {}
    assert build(temps=None)._temperature_readouts() == {}


@pytest.mark.parametrize('keep', [True, False])
def test_a_training_rollout_tallies_its_rows_per_temperature_whether_or_not_it_is_written(keep):
    """buffers.store.keep_rollouts false writes no rollout row; the forward readouts still count every scored row, at
    the residual of its live log R."""
    m = build(temps=(25.0, T_RUN))
    m._store_cfg = dict(m._store_cfg, keep_rollouts=keep)
    fwd, batch = rollout(m, [0, 1, 2, 3], [25.0, 25.0, T_RUN, T_RUN])
    slots = m._store_admit(fwd, batch, ORIGIN_ROLLOUT)
    assert slots.numel() == (4 if keep else 0)
    out = m._temperature_readouts()
    assert out['tbT/fwd/rows@25'] == 2 and out['tbT/fwd/rows@2'] == 2
    resid = (fwd['log_pf'] - fwd['log_pb']) - (fwd['log_r'] - fwd['log_Z_learned'])
    assert out['tbT/fwd/resid_mean@25'] == pytest.approx(float(resid[:2].double().mean()))
    assert out['tbT/fwd/packing_p50@2'] == pytest.approx(float(m.energy_function.packing))
    m._store_admit(fwd, batch, ORIGIN_EVAL, eval_row=True, drawable=False)
    assert m._temperature_readouts() == {}                       # an eval rollout is not a training rollout


def test_the_currency_check_also_checks_the_terms_at_a_listed_temperature():
    m = build(temps=(25.0, T_RUN))
    fwd, batch = rollout(m, [0, 1, 2, 3], [25.0, T_RUN, 25.0, T_RUN])
    m._store_admit(fwd, batch, ORIGIN_ROLLOUT)
    m._store_check_currency()
    m.row_store.pen[0] += 1.0                                    # a seed row's penalty: its rebuild no longer holds
    with pytest.raises(RuntimeError, match='term rebuild at T 2'):
        m._store_check_currency()


def test_a_store_without_terms_stamps_none_and_compares_equal_to_a_stamp_without_the_key():
    m = S.build()
    assert not m._store_terms_on() and build(temps=(25.0,))._store_terms_on()
    old = {'store_format': 3, 'temperature': 2.0}
    assert CS.stamp_diff(old, {**old, 'store_terms': None}) == []
    assert CS.stamp_diff(old, {**old, 'store_terms': True}) == ['store_terms']
