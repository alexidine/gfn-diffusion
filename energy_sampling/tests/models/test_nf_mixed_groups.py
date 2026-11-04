"""The NF policy on a mixed-group FD state (energies/fd_coordinates.py::FDSlotLayout; NFPolicy fd_space_groups;
models/fd_slot_describer.py).

Contracts:
- a row's pad slots (those its space group leaves empty) are drawn as exactly 0, and its log q is the density of its
  live slots alone: checked against base log-density + log |det| of the autograd Jacobian of the map from the live
  slots to the base, row by row, so a pad that leaked into a live row's conditioner or added to the log-det would
  show;
- draws round-trip through the cached decode;
- the slot describer gives each row its own group's features in that group's slots (bit for bit), the row's type
  one-hot and a pad flag, reads slots < i only, and its stepper matches the full pass;
- a mixed-group state is refused off the conditional molecule route."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from energies.fd_coordinates import FDSlotLayout  # noqa: E402
from models.fd_row_describer import FD_TYPES, FDRowDescriber  # noqa: E402
from models.fd_slot_describer import SlotRowDescriber  # noqa: E402
from models.nf_policy import NFPolicy  # noqa: E402
from test_nf_policy import SMALL, gfn_config  # noqa: E402

pytestmark = pytest.mark.fast
GROUPS = (2, 14)
SGS = torch.tensor([2, 14, 14, 2])
TF = dict(conditioner='transformer', tf_width=32, tf_layers=2, tf_heads=2, tf_fp32=True, atom_node_dim=16,
          atom_convs=2, atom_tokens='split', sym_ops='pooled')


def _states(layout, sgs, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([layout.pack(int(sg), torch.rand(1, len(layout.slots[int(sg)]), generator=g,
                                                         dtype=torch.float64) * 1.8 - 0.9)[0] for sg in sgs])


def _mols(sgs, seed=0):
    n = len(sgs)
    g = torch.Generator().manual_seed(seed)
    counts = torch.tensor([3, 5, 4, 6][:n])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return SimpleNamespace(z=torch.randint(1, 9, (int(counts.sum()),), generator=g),
                           pos=torch.randn(int(counts.sum()), 3, generator=g) * 1.5,
                           batch=torch.repeat_interleave(torch.arange(n), counts), ptr=ptr, num_graphs=n,
                           radius=torch.rand(n, generator=g) * 2 + 2.5, mol_volume=torch.rand(n, generator=g) * 100 + 80,
                           sg_ind=torch.as_tensor(sgs))


def _policy(scale=0.05, **flow):
    layout = FDSlotLayout(GROUPS)
    cfg = dict(gfn_config(True, cond_dim=4), dim=layout.dim, angular_mask=list(layout.wrap_mask),
               periodic_centroid_axes=None, dead_latent_rows=None, conditions_type='molecule')
    torch.manual_seed(0)
    pol = NFPolicy(cfg, {**SMALL, **TF, **flow}, seed=0, fd_space_groups=list(GROUPS)).double()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():   # random, non-zero weights everywhere (zero-initialised heads would hide a leak)
        for p in pol.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return pol.eval(), layout


def _context(pol, sgs, seed=0):
    mb = _mols(sgs, seed)
    cond = torch.randn(len(sgs), 4, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return pol._context(cond, mb, len(sgs)).double()


@pytest.mark.parametrize('rows', ['plain', 'typed', 'prefix'])
def test_pads_are_drawn_as_zero_and_log_q_is_the_live_slots_density(rows):
    pol, layout = _policy(row_tokens=rows)
    f = pol.sampler.flow
    c = _context(pol, SGS)
    live = layout.live_mask(SGS)
    assert not bool(live.all())   # the sg 14 rows have pads
    with torch.no_grad():
        x, z0 = pol.sampler.sample(len(SGS), c, return_base=True)
        assert bool((x[~live] == 0).all()) and bool((z0[~live] == 0).all())
        z, _ = pol.sampler.to_base(x, c)
        assert torch.allclose(z, z0, atol=1e-8), (z - z0).abs().max()
        lq = pol.sampler.log_prob(x, c)
    assert torch.isfinite(lq).all()
    for r in range(len(SGS)):   # log q = base + log|det| of d z_live / d x_live, one row at a time
        idx = torch.nonzero(live[r]).flatten()

        def to_base_live(v, r=r, idx=idx):
            xr = x.clone()
            xr[r, idx] = v
            return pol.sampler.to_base(xr, c)[0][r, idx]

        J = torch.autograd.functional.jacobian(to_base_live, x[r, idx].clone())
        zr = to_base_live(x[r, idx])
        unit = f.unit_base[idx]
        base = -math.log(2.0) * int(unit.sum()) - 0.5 * (zr[~unit] ** 2).sum() \
            - 0.5 * math.log(2 * math.pi) * int((~unit).sum())
        want = base + torch.linalg.slogdet(J)[1]
        assert torch.allclose(lq[r], want, atol=1e-7), (rows, r, float(lq[r]), float(want))


def test_a_pads_value_does_not_reach_log_q():
    pol, layout = _policy()
    c = _context(pol, SGS)
    x = _states(layout, SGS, 4)
    with torch.no_grad():
        a = pol.sampler.log_prob(x, c)
        x2 = x.clone()
        x2[~layout.live_mask(SGS)] = 0.7
        assert torch.equal(pol.sampler.log_prob(x2, c), a)


@pytest.mark.parametrize('prefix', (False, True))
def test_slot_describer_is_each_groups_own_in_its_slots(prefix):
    layout = FDSlotLayout(GROUPS)
    desc = SlotRowDescriber(layout, prefix=prefix)
    sgs = torch.tensor([14, 2, 14, 2, 2])
    desc.set_row_groups(sgs)
    s = _states(layout, sgs, 1)
    r = torch.rand(5, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) * 3 + 2
    v = torch.rand(5, dtype=torch.float64, generator=torch.Generator().manual_seed(3)) * 200 + 100
    v = torch.stack([v, torch.full_like(v, 1.9), torch.full_like(v, 5.0)], -1)   # fd_row_describer.MOL_INPUTS
    out = desc(s, r, v if prefix else None)
    assert out.shape == (5, layout.dim, desc.k) and desc.k == desc.k_group + len(FD_TYPES) + 1
    for g in GROUPS:
        rows = torch.nonzero(sgs == g).flatten()
        own = FDRowDescriber(g, prefix=prefix)(layout.unpack(g, s[rows]), r[rows], v[rows] if prefix else None)
        slots = layout.slots[g]
        assert torch.equal(out[rows][:, slots, :desc.k_group], own)
        names = layout.group_names[g]
        for i in range(layout.dim):
            tag = out[rows][:, i, desc.k_group:]
            if names[i] == 'pad':
                assert bool((out[rows][:, i, :-1] == 0).all()) and bool((tag[:, -1] == 1).all())
            else:
                assert bool((tag[:, FD_TYPES.index(names[i])] == 1).all()) and float(tag.sum()) == len(rows)
    for i in range(layout.dim):   # slot i reads slots < i only, and the stepper gives the full pass one slot at a time
        s2 = s.clone()
        s2[:, i:] = _states(layout, sgs, 40 + i)[:, i:]
        assert torch.equal(desc(s2, r, v if prefix else None)[:, i], out[:, i]), i
    st = desc.stepper(r, v if prefix else None, 5)
    for i in range(layout.dim):
        assert torch.equal(st.row(s, i), out[:, i]), i
    assert len(set(desc.types)) == layout.dim   # every slot its own embedding


def test_a_mixed_group_policy_that_cannot_read_the_group_is_refused():
    """A shared slot means different coordinates in different groups: without e_g (sym_ops pooled) or a
    space-group condition column the flow would learn one blend over the groups. Typed rows do not carry it."""
    layout = FDSlotLayout(GROUPS)
    cfg = dict(gfn_config(True, cond_dim=4), dim=layout.dim, angular_mask=list(layout.wrap_mask),
               periodic_centroid_axes=None, dead_latent_rows=None, conditions_type='molecule')
    for rows in ('plain', 'prefix'):
        with pytest.raises(ValueError, match="read each row's"):
            NFPolicy(cfg, {**SMALL, **TF, 'sym_ops': 'none', 'row_tokens': rows}, seed=0,
                     fd_space_groups=list(GROUPS))
    NFPolicy(cfg, {**SMALL, **TF, 'sym_ops': 'none'}, seed=0, fd_space_groups=list(GROUPS),
             fd_group_in_condition=True)


def test_mixed_groups_are_refused_off_the_conditional_molecule_route():
    layout = FDSlotLayout(GROUPS)
    cfg = dict(gfn_config(False), dim=layout.dim, angular_mask=list(layout.wrap_mask),
               periodic_centroid_axes=None, dead_latent_rows=None)
    with pytest.raises(NotImplementedError, match='molecule route'):
        NFPolicy(cfg, dict(SMALL), seed=0, fd_space_groups=list(GROUPS))
