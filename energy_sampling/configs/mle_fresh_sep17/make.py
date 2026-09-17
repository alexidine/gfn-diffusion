"""mle_fresh_sep17 -- phase 1 (bwd MLE) from FRESH weights with DPLR off (model.dplr_rank 0), on the
W3 P2_1/c priors (neh, nehu, acr) and the Niggli P-1 priors (mip, mipu).

    python configs/mle_fresh_sep17/make.py

Arm shape = mle_w3_sep16 (rate, burn-in, batch policy, eval, archives, terminal MLE stage); prior files
and byte sizes from mle_w3_sep16 / mle_nig_sep17; sbatch from mle_nig_sep17 (fresh start, requeue,
prior size guard, Niggli penalty guard) plus the monoclinic class-wall guard.
"""
import copy
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / rel / 'make.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nig = _load('mle_nig_sep17', 'nigmake')
w3 = nig.w3
#: fresh weights with DPLR off detonated at the 0.05 -> 4.0 promotion (mlefr_*_lr4, 17916557)
w3.SCALE = 2.0

TAG = 'mlefr'
BATTERY = 'mle_fresh_sep17'

#: family -> (seed arm yaml for the problem identity, prior filename transform, prior bytes)
FAM = {
    'neh':  (w3.FAM['neh'], w3.w3_path),
    'nehu': (w3.FAM['nehu'], w3.w3_path),
    'acr':  (w3.FAM['acr'], w3.w3_path),
    'mip':  (nig.FAM['mip'], nig.nig_path),
    'mipu': (nig.FAM['mipu'], nig.nig_path),
}


def build_arm(base, fam):
    spec, to_prior = FAM[fam]
    spec = dict(spec)
    seed = w3.load(spec['seed'])
    name = f'{TAG}_{fam}_lr{w3.SCALE:g}'
    w3.FAM[fam] = spec
    cfg, dropped = w3.deltas(copy.deepcopy(base), fam, name)
    cfg['tag'] = TAG
    prior = to_prior(seed['prior_path'])
    cfg['prior_path'] = prior
    cfg['molecules_path'] = prior
    cfg['checkpoint_name'] = None
    cfg['load_weights_only'] = False
    cfg['warm_start_ignore_problem_keys'] = None
    cfg['model']['dplr_rank'] = 0
    return name, cfg, dropped


def check(cfg, name, fam):
    spec, to_prior = FAM[fam]
    seed = w3.load(spec['seed'])
    mine, theirs = w3.problem_def(cfg), w3.problem_def(seed)
    moved = sorted(k for k in set(mine) | set(theirs) if k != 'prior_path' and mine.get(k) != theirs.get(k))
    assert not moved, f'{name}: problem identity moved from the seed problem on {moved}'
    assert cfg['prior_path'] == cfg['molecules_path'] == to_prior(seed['prior_path']), name
    assert cfg['checkpoint_name'] is None and cfg['continue_from_checkpoint'] == w3.CONT_PLACEHOLDER, name
    assert cfg['model']['dplr_rank'] == 0, name
    assert cfg['lr_control']['fixed_scale'] == w3.SCALE == 2.0, name
    st = cfg['protocols'][w3.PROTOCOL]['stages']
    assert len(st) == 1 and 'exit' not in st[0] and st[0]['train_mode'] == 'bwd', name
    assert cfg['integrator']['T'] == w3.SHIP_T == cfg['eval_T'], name
    w3._scan_local_paths(cfg, name)


WALL_GUARD = """
if ! grep -q 'MONO_CLASS' ${SYM}; then
    echo "FATAL: ${SYM} lacks MONO_CLASS (the monoclinic class walls) -- git pull MXtalTools" >&2; exit 1
fi
"""


def main(argv):
    dirty = w3.dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted:\n  ' + '\n  '.join(dirty))
    base = yaml.safe_load(w3.MK_DEV.read_text(encoding='utf-8'))
    arms = {}
    for fam in FAM:
        name, cfg, dropped = build_arm(base, fam)
        check(cfg, name, fam)
        w3.load_check(cfg, name)
        arms[name] = (cfg, fam, dropped)

    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n',
                                               encoding='utf-8')
    for name, (cfg, *_r) in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tstart\twarm_src\tprior\tprior_bytes\n')
        for name, (cfg, fam, _d) in arms.items():
            f.write(f"{name}\t{fam}\tfresh\t-\t{cfg['prior_path'].rsplit('/', 1)[1]}\t{FAM[fam][0]['prior_bytes']}\n")
    anchor = "SYM=${PROJECT_ROOT}/MXtalTools/mxtaltools/common/sym_utils.py\n"
    assert anchor in nig.SBATCH
    sb = (nig.SBATCH.replace(anchor, anchor + WALL_GUARD)
          .replace('phase-1 MLE on the Niggli P-1 priors, warm (mle09 best, weights-only) and fresh.',
                   'phase-1 MLE from fresh weights, DPLR off, on the W3 P2_1/c and Niggli P-1 priors.')
          .replace('__LAST__', str(len(arms) - 1)).replace('__TAG__', TAG).replace('__BATTERY__', BATTERY)
          .replace('__CKPTS__', w3.CLUSTER_CKPTS).replace('__DATA__', w3.CLUSTER_DATA))
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(sb)
    for i, (name, (cfg, fam, dropped)) in enumerate(arms.items()):
        print(f"[{i}] {name:<16} dplr_rank={cfg['model']['dplr_rank']} prior={cfg['prior_path'].rsplit('/', 1)[1]}")


if __name__ == '__main__':
    main(sys.argv[1:])
