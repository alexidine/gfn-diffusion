"""The reference campaign (read only) and the output root of the FD-start search experiment.

The campaign is chosen by the environment variable ``FD_SEARCH_CAMPAIGN`` (``CAMPAIGNS``; default ``nehzor``):
``D:/crystal_datasets/nehzor_elj_sep28`` (NEHZOR, sg 14) or ``D:/crystal_datasets/mipcas_elj_sep28`` (MIPCAS, sg 2),
both Z' = 1 eLJ campaigns written by MXtalTools 6e9c2ba9. Their configs name cluster paths; ``LOCAL_MOL`` is the local
copy of the conformer file the campaign names (``mol_path``). Nothing here writes into the campaign; every output goes
to a new directory under ``OUT_ROOT``.
"""
import os
from types import SimpleNamespace

import torch
import yaml

#: per campaign: directory, local and cluster conformer paths, space group, and the energy reference (the registry
#: minimum, stats.md) that --early-stop-ref campaign uses
CAMPAIGNS = {
    'nehzor': dict(dir='D:/crystal_datasets/nehzor_elj_sep28', local_mol='D:/crystal_datasets/nehzor/NEHZOR0_std_conf.pt',
                   cluster_mol='/scratch/mk8347/data/crystal_datasets/nehzor/NEHZOR0_std_conf.pt', sg=14,
                   ref=-670.5643),
    'mipcas': dict(dir='D:/crystal_datasets/mipcas_elj_sep28', local_mol='D:/crystal_datasets/mipcas/MIPCAS_standardized.pt',
                   cluster_mol='/scratch/mk8347/data/crystal_datasets/mipcas/MIPCAS_standardized.pt', sg=2,
                   ref=-387.9489),
}
CAMPAIGN = os.environ.get('FD_SEARCH_CAMPAIGN', 'nehzor')
if CAMPAIGN not in CAMPAIGNS:
    raise ValueError(f'FD_SEARCH_CAMPAIGN={CAMPAIGN!r}: the campaigns are {sorted(CAMPAIGNS)}')
CAMPAIGN_DIR = CAMPAIGNS[CAMPAIGN]['dir']
LOCAL_MOL = CAMPAIGNS[CAMPAIGN]['local_mol']
CLUSTER_MOL = CAMPAIGNS[CAMPAIGN]['cluster_mol']
SG = CAMPAIGNS[CAMPAIGN]['sg']
CAMPAIGN_REF = CAMPAIGNS[CAMPAIGN]['ref']
OUT_ROOT = 'D:/crystal_datasets/fd_search_oct26'
CAMPAIGN_COMMIT = '6e9c2ba92664e8addec6031440e9017c9366af09'


def coord_config():
    """The campaign's coord.yaml as a dict (kT, identity_cut, rdf_mode, window_kT, bands_kT, sg, z_prime, ...)."""
    with open(os.path.join(CAMPAIGN_DIR, 'coord.yaml')) as fh:
        return yaml.safe_load(fh)


def stream_config(stream='random', device='cpu'):
    """The campaign stream's search config (streams/<stream>.yaml) as the namespace run_search builds, with the
    conformer path mapped to ``LOCAL_MOL`` and the device replaced. Refuses a config naming another conformer."""
    with open(os.path.join(CAMPAIGN_DIR, 'streams', f'{stream}.yaml')) as fh:
        cfg = yaml.safe_load(fh)
    if cfg['mol_path'] != CLUSTER_MOL:
        raise ValueError(f"campaign stream config names {cfg['mol_path']!r}, expected {CLUSTER_MOL!r}")
    cfg['mol_path'] = LOCAL_MOL
    cfg['device'] = device
    return SimpleNamespace(**cfg)


def load_molecule():
    """The campaign conformer (a MolCrystalData holding the NEHZOR molecule) from ``LOCAL_MOL``."""
    mol = torch.load(LOCAL_MOL, weights_only=False)
    return mol[0] if isinstance(mol, list) else mol


def new_run_dir(run_name):
    """Create ``OUT_ROOT/<run_name>`` and return it; refuses an existing directory (never writes into one)."""
    if not run_name or any(c in run_name for c in '/\\:') or run_name in ('.', '..'):
        raise ValueError(f'bad run name {run_name!r}')
    path = os.path.join(OUT_ROOT, run_name)
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(path, exist_ok=False)  # FileExistsError on an existing run directory
    return path


def run_dir(run_name):
    """``OUT_ROOT/<run_name>``, which must exist."""
    path = os.path.join(OUT_ROOT, run_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    return path


def check_not_campaign(path):
    """Raise if ``path`` lies inside the campaign directory (it is read only)."""
    a = os.path.normcase(os.path.abspath(path))
    b = os.path.normcase(os.path.abspath(CAMPAIGN_DIR))
    if a == b or a.startswith(b + os.sep):
        raise PermissionError(f'{path} is inside the read-only campaign directory {CAMPAIGN_DIR}')
