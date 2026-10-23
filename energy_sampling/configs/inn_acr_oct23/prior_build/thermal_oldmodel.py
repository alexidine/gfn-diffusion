"""The 1 kT latent kick of acridine sg14 Z'=1 basins under the OLD MACE checkpoint (acr_112025_mh1_stagetwo.model),
with the OLD conformer the old prior carries (acridine_conformer.pt, N at index 13): MXtalTools'
configs/crystal_searches/acr_zp1_sep30/seed_pool_build/thermal_newmodel.py unchanged, with its model, molecule and output
directory swapped and its pool widened from the 16 to the 64 lowest anchors of acridine_sg14_zp1_mace_prior_dataset_w3.pt
(the 16 lowest hold only 3 basins more than 0.15 apart in atomwise RDF; the 64 lowest hold 5). Its log labels the
energies 'new-model'; here they are the old checkpoint's. Dev-box paths, as in the original.

    THERMAL_DEVICE=cuda PYTHONPATH=<MXtalTools> python thermal_oldmodel.py
"""
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'C:/Users/mikem/Projects/mxt_gfn/mxtaltools/configs/crystal_searches/acr_zp1_sep30/seed_pool_build')
import thermal_newmodel as t  # noqa: E402

t.MODEL = 'D:/crystal_datasets/acr_112025_mh1_stagetwo.model'
t.MOL = 'D:/crystal_datasets/acridine/acridine_conformer.pt'
t.CFG = SimpleNamespace(mol_path=t.MOL, sg=14, z_prime=1)
t.OUT = os.path.join(HERE, 'oldmodel')
t.N_POOL = 64

if __name__ == '__main__':
    t.main()
