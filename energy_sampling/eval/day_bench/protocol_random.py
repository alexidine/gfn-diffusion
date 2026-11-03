"""Random-search raw samples for the benchmark protocol: ``random_search.starts`` built with the constant conformer and
written unrelaxed, one labelled extended-XYZ file per start (``cifs/sample_sg<sg>_<k>.extxyz``, ``bb_indices`` per atom),
so ``protocol_refine`` reads whole molecules even when a random cell is smaller than a molecule and the molecules
overlap their own images.

PLAN is either explicit, ``SG:N[,SG:N...]``, or ``csd:N``: N starts over Day's 26 space groups in proportion to their
frequency among Z' = 1 CSD structures (``csd_statistics.json``, population ``--population``), apportioned by largest
remainder (``allocate``). No molecule's own crystal enters the allocation.

usage (licensed CSD Python API):
    python -m eval.day_bench.protocol_random OUT_DIR REFCODE CONFORMER PLAN [SEED] [--population CHNOF_zp1]
"""
import argparse
import json
import os

import numpy as np
from ase.io import write

from eval.day_bench.p1_relax import to_atoms
from eval.day_bench.protocol_refine import labelled
from eval.day_bench.random_search import starts

DAY_GROUPS = (1, 2, 4, 5, 7, 9, 13, 14, 15, 18, 19, 20, 29, 33, 43, 56, 60, 61, 76, 86, 88, 96, 145, 148, 154, 169)
CSD_STATS = os.path.join(os.path.dirname(__file__), 'csd_statistics.json')


def allocate(freq, n):
    """{group: starts} summing to ``n``, proportional to ``freq`` ({group: weight}) by largest remainder; ties go to the
    lower group number."""
    groups = sorted(freq)
    w = np.array([float(freq[g]) for g in groups])
    exact = n * w / w.sum()
    base = np.floor(exact).astype(int)
    order = sorted(range(len(groups)), key=lambda i: (-(exact[i] - base[i]), groups[i]))
    for i in order[:n - base.sum()]:
        base[i] += 1
    return {g: int(k) for g, k in zip(groups, base) if k > 0}


def csd_frequencies(population, path=CSD_STATS, count='raw'):
    """{group: frequency} over Day's 26 groups for ``population`` of ``csd_statistics.json``, renormalised over the 26;
    ``count`` 'raw' (every structure) or 'dedup' (one per refcode family)."""
    sg = json.load(open(path))['populations'][population][count]['day26']['freq_renormalised_over_26']
    freq = {int(g): float(v) for g, v in sg.items() if int(g) in DAY_GROUPS}
    assert len(freq) == len(DAY_GROUPS), f'{population}: frequencies for {len(freq)} of the 26 groups'
    return freq


def parse_plan(plan, population):
    if plan.startswith('csd:'):
        return sorted(allocate(csd_frequencies(population), int(plan[4:])).items())
    return [tuple(int(v) for v in p.split(':')) for p in plan.split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('refcode')
    ap.add_argument('conformer')
    ap.add_argument('plan')
    ap.add_argument('seed', nargs='?', type=int, default=0)
    ap.add_argument('--population', default='CHNOF_zp1')
    a = ap.parse_args()
    from ccdc import io as ccdc_io
    from eval.day_bench.day_landscape import load_reference
    ref = load_reference(a.refcode, a.conformer, ccdc_io.EntryReader('CSD'))
    raw = os.path.join(a.out_dir, 'cifs')
    os.makedirs(raw, exist_ok=True)
    plan = parse_plan(a.plan, a.population)
    print(f'plan ({a.plan}): ' + ', '.join(f'{sg}:{n}' for sg, n in plan), flush=True)
    for sg, n in plan:
        k = 0
        for batch in starts(ref, sg, n, a.seed):
            for c in batch.batch_to_list():
                write(os.path.join(raw, f'sample_sg{sg}_{k:05d}.extxyz'), labelled(to_atoms(c), ref), format='extxyz')
                k += 1
        print(f'space group {sg}: {k} starts written to {raw}', flush=True)


if __name__ == '__main__':
    main()
