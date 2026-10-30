"""Random-search raw samples for the benchmark protocol: ``random_search.starts`` built with the constant conformer and
written unrelaxed, one labelled extended-XYZ file per start (``cifs/sample_sg<sg>_<k>.extxyz``, ``bb_indices`` per atom),
so ``protocol_refine`` reads whole molecules even when a random cell is smaller than a molecule and the molecules
overlap their own images.

usage (licensed CSD Python API):
    python -m eval.day_bench.protocol_random OUT_DIR REFCODE CONFORMER SG:N[,SG:N...] [SEED]
"""
import os
import sys

from ase.io import write

from eval.day_bench.p1_relax import to_atoms
from eval.day_bench.protocol_refine import labelled
from eval.day_bench.random_search import starts


def main(out_dir, refcode, conformer, plan, seed=0):
    from ccdc import io as ccdc_io
    from eval.day_bench.day_landscape import load_reference
    ref = load_reference(refcode, conformer, ccdc_io.EntryReader('CSD'))
    raw = os.path.join(out_dir, 'cifs')
    os.makedirs(raw, exist_ok=True)
    for sg, n in plan:
        k = 0
        for batch in starts(ref, sg, n, seed):
            for c in batch.batch_to_list():
                write(os.path.join(raw, f'sample_sg{sg}_{k:05d}.extxyz'), labelled(to_atoms(c), ref), format='extxyz')
                k += 1
        print(f'space group {sg}: {k} starts written to {raw}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], [tuple(int(v) for v in p.split(':')) for p in sys.argv[4].split(',')],
         int(sys.argv[5]) if len(sys.argv) > 5 else 0)
