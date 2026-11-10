"""Search configs for the exp-6 recall test (docs/design/exp6_recall_test.md), one per molecule x energy x mode.

Every config is a MXtalTools streaming search with converged minima (as
mxtaltools configs/crystal_searches/minima_stream/mipcas_elj.yaml): stage 1 Rprop and stage 2 per-row BFGS under one
energy, then the polish to minima with Hessians, merged into a library. Modes:
    search  random starts, ended by the Good-Turing stop rule (stop_p over minima within stop_band_kT) or at
            num_samples starts, whichever comes first
    seeds   num_samples 0: only the rows of a seeds file are relaxed (they start in the last stage) and polished;
            the MLIP leg of the exp-6 and eLJ arms, and the anchors

Energies: 'elj' (its search kT, as MIPCAS), 'exp6' (FIT + MMFF charges, kJ/mol per molecule), 'uma' (UMA-s 1.1,
uma-s-1p1.pt, through the uma route, lattice energy in kJ/mol per molecule; the owner's reference, 2026-11-10, after
eSEN-S was found to compress aromatic crystals). kT for the kJ/mol energies is RT at 300 K.

usage:
    python -m eval.exp6_recall.make_configs PANEL_DIR OUT_DIR MODE ENERGY[,ENERGY] [--seeds PATTERN] [--n N]
        [--site local|cluster] [--molecules a,b] [--panel_at DIR] [--out_at DIR] [--tag TAG]
PATTERN may contain {name} and {energy}; e.g. '{name}_anchor.pt' relative to PANEL_DIR.
"""
import argparse
import json
import os

import yaml

KT_KJ = 2.494339  # RT at 300 K, kJ/mol
KT = {'elj': 6.859493755713105, 'exp6': KT_KJ, 'uma': KT_KJ}
SITES = {
    'local': dict(uma='D:/crystal_datasets/uma-s-1p1.pt'),
    'cluster': dict(uma='/scratch/mk8347/models/uma/uma-s-1p1.pt'),
}
SLOTS = {'elj': 250, 'exp6': 250, 'uma': 96}  # starts in flight; the MLIP holds far fewer rows per call
# the polish's tolerance (kT); None: MXtalTools' default (1e-4, MLIP_TOL on an MLIP). exp-6's float32 energy repeats to
# 5.5e-5 kT (imidazole, 2026-11-10), so at 1e-4 its polish stalled short of the test
POLISH_TOL = {'elj': None, 'exp6': 1e-3, 'uma': None}
# the 65 Sohncke space groups (no improper element): a crystal and its mirror image are different crystals
SOHNCKE = {1, 3, 4, 5, *range(16, 25), *range(75, 81), *range(89, 99), *range(143, 147), *range(149, 156),
           *range(168, 174), *range(177, 183), *range(195, 200), *range(207, 215)}


def stages(energy):
    kT = KT[energy]
    s1 = dict(optim_target=energy, enforce_reduced=False, compression_factor=0.0, cutoff=10, init_lr=0.05,
              convergence_eps=0.01, optimizer_func='rprop', anneal_lr=False, grad_norm_clip=0.1, show_tqdm=False,
              max_num_steps=350, target_packing_coeff=None, centroid_boundary='wrap')
    s2 = dict(optim_target=energy, enforce_reduced=False, compression_factor=0.0, cutoff=10, convergence_eps=1e-4,
              optimizer_func='bfgs', show_tqdm=False, max_num_steps=200, target_packing_coeff=None,
              centroid_boundary='wrap', early_stop=[[25, 20.0], [50, 15.0], [100, 12.0]], early_stop_kT=kT)
    return [s1, s2]


def config(name, sg, energy, mode, panel_dir, out_dir, site, n=2000, seeds=None, stop_band_kT=4.0, stop_p=0.01,
           tag=''):
    run = f'{name}_{energy}_{mode}' + (f'_{tag}' if tag else '')
    c = dict(device='cuda', mol_path=f'{panel_dir}/{name}_mol.pt', dataset_path=None, target_path=None,
             out_dir=out_dir, run_name=run, target_identifier=None, save_trajs=False, init_sample_method='random',
             init_reduced=True, force_restart_run=False, mol_seed=0, opt_seed=1000000000, sampling_mode='all',
             mols_to_sample=1, num_samples=0 if mode == 'seeds' else int(n), sgs_to_search=[int(sg)],
             zp_to_search=[1], batch_size=SLOTS[energy], grow_batch_size=False, init_target_cp='std',
             oom_ceiling=False,
             minima=dict(kT=KT[energy], capture_d2=9.0, **({'tol': POLISH_TOL[energy]} if POLISH_TOL[energy] else {})),
             stream=dict(slots=SLOTS[energy], check_every=5, seeds=seeds,
                         stop_p=None if mode == 'seeds' else stop_p,
                         stop_band_kT=stop_band_kT),  # also the band the conditions file reports (None raises)
             opt=stages(energy))
    if energy == 'uma':
        c['uma_predictor_path'] = SITES[site]['uma']
        c['uma_activation_checkpointing'] = False
        # UMA-s is not reflection-equivariant (owner, 2026-11-10): a mirror-mean gas reference everywhere, and the
        # periodic energy averaged over the crystal and its mirror image in chiral (Sohncke) groups only, where the
        # two are different crystals; in a group with an improper element the crystal is its own mirror image
        c['mlip_gas_reference'] = 'mirror_mean'
        c['mlip_mirror_average'] = int(sg) in SOHNCKE
    return run, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('panel_dir')
    p.add_argument('out_dir')
    p.add_argument('mode', choices=['search', 'seeds'])
    p.add_argument('energies')
    p.add_argument('--seeds', default=None)
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--site', default='local', choices=list(SITES))
    p.add_argument('--molecules', default=None)
    p.add_argument('--panel_at', default=None, help='the panel folder as the runs see it (default: PANEL_DIR)')
    p.add_argument('--out_at', default=None, help='the output folder as the runs see it (default: OUT_DIR)')
    p.add_argument('--tag', default='', help='appended to every run name (two seeded runs of one molecule)')
    a = p.parse_args()
    panel = json.load(open(os.path.join(a.panel_dir, 'panel.json')))
    names = a.molecules.split(',') if a.molecules else list(panel)
    run_panel = a.panel_at or a.panel_dir
    run_out = a.out_at or a.out_dir
    os.makedirs(a.out_dir, exist_ok=True)
    for name in names:
        rec = panel[name]
        for energy in a.energies.split(','):
            seeds = None
            if a.mode == 'seeds':
                rel = a.seeds.format(name=name, energy=energy)
                if not os.path.exists(os.path.join(a.panel_dir, rel)) and not os.path.isabs(rel):
                    print(f'{name} {energy}: no seeds file {rel}; skipped')
                    continue
                seeds = rel if os.path.isabs(rel) else f'{run_panel}/{rel}'
            run, c = config(name, rec['sg_searched'], energy, a.mode, run_panel, run_out, a.site, a.n, seeds,
                            tag=a.tag)
            with open(os.path.join(a.out_dir, f'{run}.yaml'), 'w') as fh:
                fh.write(f'# exp-6 recall test, {name} (sg {rec["sg_searched"]}), {energy}, {a.mode}. '
                         f'Generated by eval/exp6_recall/make_configs.py.\n')
                yaml.safe_dump(c, fh, sort_keys=False)
            print(os.path.join(a.out_dir, f'{run}.yaml'))


if __name__ == '__main__':
    main()
