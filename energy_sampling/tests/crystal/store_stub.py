"""A Modeller built without __init__ around a real CrystalStore with real template batches, a stub energy function and a
kick stub. Shared by test_row_store_trainer.py, test_store_tb.py and test_store_growth.py."""
import math
from types import SimpleNamespace

import torch

from energy_sampling import crystal_store as CS
from energy_sampling import physical_kick
from energy_sampling.train import Modeller
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

DIM, T = 3, 2.0
GROUPS = (2, 14)


def molecule(i):
    scale = 1.0 + 0.1 * i
    pos = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1]], dtype=torch.float32) * scale
    return MolData(z=torch.tensor([6, 7, 8], dtype=torch.long), pos=pos, x=torch.zeros((3, 1), dtype=torch.float32),
                   identifier=f'm{i}', mol_volume=torch.tensor(55.0 * scale ** 3), mass=torch.tensor(42.0),
                   radius=torch.tensor(1.2 * scale), z_prime=torch.tensor(1, dtype=torch.long))


def templates(n_mol=2):
    """One graph per (molecule, group), in condition id order: condition = molecule * 2 + local group."""
    b = collate_data_list([molecule(m) for m in range(n_mol) for _ in GROUPS])
    b.add_graph_attr(torch.tensor([g for _ in range(n_mol) for g in GROUPS], dtype=torch.long), 'sg_ind')
    b.add_graph_attr(torch.arange(n_mol).repeat_interleave(len(GROUPS)), 'mol_id')
    del b['identifier']
    return b


class Energy:
    """log R = -|x|^2 - cid / 10; mol_energy = |x|^2 + cid; seed energy = mol_energy + 1000 * extra."""
    temperature = T
    temperature_conditioning = False
    is_crystal = True
    space_groups = list(GROUPS)
    z_primes = [1]
    n_sg, n_zp = len(GROUPS), 1

    def __init__(self):
        self.sg_to_local = torch.full((231,), -1, dtype=torch.long)
        self.sg_to_local[torch.tensor(GROUPS)] = torch.arange(len(GROUPS))
        self.zp_to_local = torch.tensor([-1, 0], dtype=torch.long)
        self.calls = []
        self.packing = 0.7
        self.extra = 0.0

    def condition_ids_of(self, batch):
        return batch.mol_id.reshape(-1).long() * self.n_sg + self.sg_to_local[batch.sg_ind.reshape(-1).long()]

    def condition_samples(self, mol, temperature=None, sg_inds=None, z_primes=None, repeats=1):
        cid = mol.mol_id.reshape(-1).long() * self.n_sg + self.sg_to_local[sg_inds.reshape(-1).long()]
        mol.condition_id = cid
        cond = cid.float()[:, None] * 0.1
        mol.conditions = cond
        return mol, torch.full((mol.num_graphs,), math.log10(T)), cond, cid

    def log_reward(self, x, mol_batch=None, log_temperature=None, return_exp=False, internal_oom_recovery=None):
        cid = mol_batch.condition_id.float()
        sq = (x ** 2).sum(-1).float()
        self.calls.append(x.detach().clone())
        scored = SimpleNamespace(mol_energy=sq + cid, density_energy=torch.zeros_like(sq),
                                 pressure_energy=torch.full_like(sq, float(self.extra)),
                                 packing_coeff=torch.full_like(sq, float(self.packing)))
        log_r = -sq - cid / 10
        return (log_r, scored) if return_exp else log_r

    @staticmethod
    def seed_energy_from(ens):
        return ens['mol_energy'] + ens['density_energy'] + ens['pressure_energy']


def build(spec=((2, 4), (1, 3), (1, 2), (1, 2)), max_rows=2000, H=64, batch=8, store_seed=0, dim=DIM):
    """A Modeller around a seeded store. spec: (anchors, children) per condition id."""
    cid, kind, parent = [], [], []
    for c, (na, nk) in enumerate(spec):
        first = len(cid)
        cid += [c] * (na + nk)
        kind += [CS.ANCHOR] * na + [CS.CHILD] * nk
        parent += [-1] * na + [first + j % na for j in range(nk)]
    cid, kind, parent = torch.tensor(cid), torch.tensor(kind), torch.tensor(parent)
    n = cid.numel()
    x = torch.linspace(-0.9, 0.9, n * dim).reshape(n, dim)
    energy = (x ** 2).sum(-1) + cid
    log_r = -(x ** 2).sum(-1) - cid / 10
    t = templates()
    st = CS.CrystalStore(dim, t.num_graphs, max_rows, H, seed=store_seed)
    st.seed(x, cid, kind, log_r, energy, parent, torch.zeros(n, dtype=torch.bool), t, torch.arange(t.num_graphs),
            torch.full((t.num_graphs,), 100.0, dtype=torch.float64))
    m = Modeller.__new__(Modeller)
    m.row_store = st
    m._store_seed_best = st.best_energy.clone()
    m._store_draw_tally = None
    m.energy_function = Energy()
    m.device = torch.device('cpu')
    m.batch_size = batch
    m.step_ind = 5
    m.protocol = SimpleNamespace(stage=SimpleNamespace(name='stub', bwd_sampling_mode='dataset'))
    m.args = SimpleNamespace(seed=3, energy_config=SimpleNamespace(temperature=T), eval_num_samples=16,
                             bwd_loss_coeffs=SimpleNamespace(mle=1.0, tb=0.0), smoke=True)
    m.gfn_model = SimpleNamespace()
    m._store_cfg = dict(mle_draw='anchor', replay_weight='uniform')
    m._noised_prior_settings = dict(live_redraws=4, window_kT=10.0, packing_window=(0.5, 0.9))
    m._physical = dict(kits={}, lo=-2.0, hi=-1.0, no_parent=0, rows=0, full_route=0, redrawn=0, kept=0,
                       mle_kept_dropped=0, gen=torch.Generator().manual_seed(1))
    m._growth_owed = 0
    m._store_growth = dict(kicks=0, unchanged=0, scored=0, in_window=0, out_of_box=0)
    m._bwd_draw_is_reward_free = lambda: True
    return m


def shift_kick(monkeypatch, step=0.01, keep=None):
    """physical_kick.kick -> centres + step (uniform noise from the generator it is handed); `keep(centres)` marks rows
    returned unchanged. Returns the list of (rows) calls."""
    calls = []

    def kick(kits, centres, sg, radius, rg, lo, hi, tries, generator=None):
        noise = torch.rand(centres.shape, generator=generator) * step + step
        x = (centres + noise).float()
        kept = torch.zeros(len(x), dtype=torch.bool) if keep is None else keep(centres)
        x[kept] = centres[kept].float()
        calls.append(dict(n=len(x), generator=generator))
        return x, dict(rows=len(x), full_route=0, redrawn=0, kept=int(kept.sum()))
    monkeypatch.setattr(physical_kick, 'kick', kick)
    return calls


def rollout(cid, x=None, log_r=None, log_pf=None, energy=None):
    """(fwd_stats, sample_batch) of a scored rollout on condition ids `cid`."""
    cid = torch.as_tensor(cid)
    n = cid.numel()
    x = torch.full((n, DIM), 0.25) if x is None else x
    log_r = torch.full((n,), -1.0) if log_r is None else torch.as_tensor(log_r, dtype=torch.float32)
    log_pf = torch.full((n,), -2.0) if log_pf is None else torch.as_tensor(log_pf, dtype=torch.float32)
    energy = torch.full((n,), 5.0) if energy is None else torch.as_tensor(energy, dtype=torch.float32)
    fwd = {'condition_id': cid, 'log_r': log_r, 'log_pf': log_pf, 'log_pb': torch.zeros(n),
           'log_Z_learned': torch.full((n,), 3.0), 'flow_states': torch.stack([torch.zeros_like(x), x], 1)}
    return fwd, SimpleNamespace(mol_energy=energy[:, None], condition_id=cid)
