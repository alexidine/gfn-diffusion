"""Rebuild entries of physical_kick.py::BASIS_TABLES: every distinct basis change canon.canonicalize chooses on metric
kicks of a pre-noised prior's anchors, identity first, then most frequent first. Half the kicks are at the radius law
and half at 1.4 times it. A triclinic or monoclinic group gets a table; an orthorhombic group reads axis_order.axis_orders
and needs none. Prints the entries as a Python literal and writes nothing. A table is accepted by
tests/crystal/test_physical_kernel.py.

    python -m data_processing.mine_basis_tables <prior.pt> --groups 2 14
"""
import argparse
import math
from collections import Counter

import torch

from energies.fd_coordinates import FDCoordinates, FDSlotLayout, OK_STATUS
from energy_sampling import noised_prior as NP, physical_kick as PK
from mxtaltools.crystal_fd import canon
from mxtaltools.crystal_fd.description import CrystalDescription

F64 = torch.float64
IDENTITY = (1, 0, 0, 0, 1, 0, 0, 0, 1)
RADIUS_MULTIPLES = (1.0, 1.4)


def mine(k, s, radius, rg, kicks, log_lo, log_hi, seed=0, chunk=2048):
    """[(row-major basis change, count)], identity first, of `kicks` kicks of the anchors s [n, d] in [-1, 1] of the group
    kit k. Raises when a chosen basis carries a lattice shift, which the table's groups do not hold."""
    gen = torch.Generator().manual_seed(seed)
    cell, t, R = PK.decode(k, ((s + 1) / 2).clamp(0, 1), radius)
    T = PK.t_fc(cell)
    codes = torch.tensor([canon._CODE[x] for x in OK_STATUS])
    count = Counter()
    for i, mult in enumerate(RADIUS_MULTIPLES):
        left = kicks // len(RADIUS_MULTIPLES) + (kicks % len(RADIUS_MULTIPLES) if i == 0 else 0)
        shift = math.log10(mult)
        while left > 0:
            b = torch.randint(len(s), (min(chunk, left),), generator=gen)
            st = PK.draw_step(k, len(b), log_lo + shift, log_hi + shift, gen)
            c2, t2, R2 = PK.apply_step(st, cell[b], t[b], R[b], rg[b], T[b])
            sane = torch.isfinite(c2).all(1) & (c2[:, :3] > 0).all(1) & (PK.gram_factor(c2) > 0)
            c = canon.canonicalize(CrystalDescription.from_parts(
                torch.full((int(sane.sum()),), k.g), c2[sane], t2[sane], R2[sane],
                torch.ones(int(sane.sum()), dtype=torch.long)))
            ok = torch.isin(c.status, codes.to(c.status.dtype))
            if bool((c.s[ok].abs() > 1e-9).any()):
                raise ValueError(f'space group {k.g}: a chosen basis carries a lattice shift')
            uniq, cnt = torch.unique(c.N[ok].reshape(-1, 9).long(), dim=0, return_counts=True)
            count.update({tuple(u): int(v) for u, v in zip(uniq.tolist(), cnt.tolist())})
            left -= len(b)
    return [(IDENTITY, count.pop(IDENTITY, 0))] + count.most_common()


def anchors(prior, g, fit_frac, max_anchors, seed=0, chunk=2048):
    """(chart rows s [m, d] float64, radii [m], r_g [m]) of a random `fit_frac` of the anchors of space group g in `prior`,
    at most `max_anchors`, the rows with a chart point."""
    kinds, sg = NP.tag(prior, 'noised_kind'), prior.sg_ind.reshape(-1).long().cpu()
    rows = torch.nonzero((sg == g) & (kinds == NP.KIND_ANCHOR)).flatten()
    rows = rows[torch.randperm(len(rows), generator=torch.Generator().manual_seed(1000 + g + seed))]
    rows = rows[:min(int(fit_frac * len(rows)), max_anchors)]
    s, radius, rg = [], [], []
    for lo in range(0, len(rows), chunk):
        sub = prior.subsample_new_batch(rows[lo:lo + chunk].to(prior.ptr.device))
        r = torch.round(sub.radius.reshape(-1).detach().cpu().double() * 1e6) / 1e6
        got = FDCoordinates(g, r).from_params(sub.full_cell_parameters().detach().cpu(),
                                              sub.aunit_handedness.reshape(-1).detach().cpu())
        s.append(got.s[got.ok].to(F64))
        radius.append(r[got.ok])
        rg.append(NP.radius_of_gyration(sub).cpu()[got.ok])
    return torch.cat(s), torch.cat(radius), torch.cat(rg)


def literal(g, entries):
    """The BASIS_TABLES entry of group g, three bases to a line."""
    rows = ['(' + ', '.join(str(v) for v in n) + ')' for n, _ in entries]
    lines = [', '.join(rows[i:i + 3]) + ',' for i in range(0, len(rows), 3)]
    return f'    {g}: (\n' + '\n'.join('        ' + ln for ln in lines) + '\n    ),'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('prior')
    ap.add_argument('--groups', type=int, nargs='+', required=True)
    ap.add_argument('--kicks', type=int, default=300000, help='kicks per group')
    ap.add_argument('--log_range', type=float, nargs=2, default=(-2.0, -0.375), help='log10 Angstrom')
    ap.add_argument('--fit_frac', type=float, default=0.6, help='share of each group\'s anchors the kicks start from')
    ap.add_argument('--max_anchors', type=int, default=20000)
    a = ap.parse_args()
    prior = torch.load(a.prior, weights_only=False, map_location='cpu')['equalized_prior']
    layout = FDSlotLayout(tuple(a.groups))
    for g in a.groups:
        k = PK.group_kit(g, layout.slots[g], table=(IDENTITY,))
        if k.family == 'orthorhombic':
            print(f'    {g}: None,')
            continue
        s, radius, rg = anchors(prior, g, a.fit_frac, a.max_anchors)
        entries = mine(k, s, radius, rg, a.kicks, a.log_range[0], a.log_range[1])
        total = sum(c for _, c in entries)
        print(f'# space group {g}: {len(entries)} distinct basis changes over {total} kicks of {len(s)} anchors')
        print(literal(g, entries), flush=True)


if __name__ == '__main__':
    main()
