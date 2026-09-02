"""
COMPACK his structures DIRECTLY against the known polymorphs.

Both sides are rigid-body with our reference conformer, so this is like-for-like,
and it answers "is this his structure the experimental form?" without routing the
question through our landscape.

The __main__ guard is REQUIRED: batch_compack spawns mp.Pool, and on Windows each
child re-imports this module. Without the guard the module-level work re-runs in
every child and recursively spawns more pools.
"""
import os
import torch
from mxtaltools.dataset_utils.utils import collate_data_list

OUT = r"D:\crystal_datasets\acridine\nikos_comparison"


def main():
    work = os.path.join(OUT, '_direct_compack')
    os.makedirs(work, exist_ok=True)
    levels = torch.load(f"{OUT}/nikos_levels.pt", weights_only=False,
                        map_location='cpu')
    his = levels['l1']
    ident = list(his.identifier)
    poly = torch.load(r"D:\crystal_datasets\acridine\std_acridine_polymorphs.pt",
                      weights_only=False, map_location='cpu').cpu()
    pl, pids = poly.batch_to_list(), list(poly.identifier)
    hl = his.batch_to_list()

    cwd = os.getcwd()
    rows = []
    try:
        os.chdir(work)
        refs = []
        for i, p in enumerate(pl):
            b = collate_data_list([p])
            b.mol2ucell()
            b.write_cif(torch.arange(1), f'ref_{pids[i]}', 'unit cell')
            refs.append((pids[i], os.path.abspath(f'ref_{pids[i]}_0.cif')))

        #: his Cc set -- where the landscape analysis found a robust correspondence
        targets = [i for i in range(len(ident)) if int(his.sg_ind[i]) == 9]
        for i in targets:
            best = []
            for pid, rp in refs:
                one = collate_data_list([hl[i]])
                m, r = one.batch_compack(rp, torch.arange(1), n_cpus=1)
                best.append((pid, int(m[0]), float(r[0])))
            best.sort(key=lambda t: (-t[1], t[2]))
            rows.append((ident[i], best))
            print(f"  {ident[i]} done", flush=True)
    finally:
        os.chdir(cwd)

    print(f"\n{'his structure':11s} {'best polymorph':14s} {'matched':>9s} "
          f"{'RMSD':>7s}   runners-up")
    for key, best in rows:
        pid, nm, rm = best[0]
        ru = ", ".join(f"{p}:{n}/20 {r:.2f}A" for p, n, r in best[1:3])
        print(f"{key:11s} {pid:14s} {nm:>6d}/20 {rm:7.3f}   {ru}")
    torch.save(rows, os.path.join(OUT, 'direct_compack.pt'))


if __name__ == '__main__':
    main()
