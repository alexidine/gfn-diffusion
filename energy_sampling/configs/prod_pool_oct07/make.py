"""prod_pool_oct07 -- PRODUCTION phase 2 on the pooled priors, from the mle_pool_oct05 best-MLE checkpoints.

    python configs/prod_pool_oct07/make.py       # base = the COMMITTED mk_dev (git show HEAD:), warns on a dirty tree

THE ARMS are prod_sep20's production arm (its build_arm and check, unchanged: a rollout every 5th step, P_B frozen
at the phase-2 entry, replay 0.3 / bwd 0.7 pinned, tau 600, no forces, batch 1600 pinned (acr 1000, and no per-branch
trajectory checkpointing there, both prod_sep20's), rate 0.5) for mip / neh / mipu / nehu / acr, with three things
taken from the seed run instead of mk_dev:

  seed  *mlepl_<fam>_lr2_*_best.pt (mle_pool_oct05), resolved by the job at launch; identity, model and
        energy_config verbatim from that arm's yaml, so prior_path = molecules_path = <system>_pooled_oct05_prior.pt.
  buffers.prior_buffer.max_size = buffers.anchor_buffer.max_size = the prior file's row count, as in the seed
        (mk_dev's caps are below every file; a smaller cap subsamples at seeding and breaks up the image sets).
  prior_scan_cache = true: the seed run already wrote <prior>.scan-<hash>.pt for this energy_config, so start-up
        re-scores 512 rows instead of every row (the MLIP arms turn internal_oom_recovery on, which the cache
        identity does not count: prior_scan_cache.CHUNKING_KEYS).

THREE EXPLORATION ARMS ON MIP (owner 2026-10-07: "one with the terminal force, one with Pb unfrozen, and one with
both"), each the mip production arm with only the named thing moved (asserted):
  ppl_mip_force_n5       the stored terminal force on replay rows (replay_loss_coeffs.stored_force_k 1, prod_sep20's
                         _f arm)
  ppl_mip_unpb_n5        P_B left trainable through phase 2: equilibration's on_enter without freeze_pb:full
  ppl_mip_unpb_force_n5  both
The variant sits BEFORE _n5 in the name because the job finds an arm's own checkpoint by the glob *<arm>_*_running.pt:
under prod_sep20's naming (p20_mip_n5_f) the production arm's glob also matched its variant's file, and a requeued
production arm would have resumed whichever of them was written last. main() asserts no arm's glob matches another.
The force is the replay-row one because this recipe gives the forward branch no loss weight (fracs fwd 0): rollouts
only feed the replay buffer, so a forward-seat force would have nothing to act on.

ACRIDINE (owner 2026-10-07: "do acridine for me too in the same battery") seeds from mle_pool_acr_oct05's arm, so its
prior is acridine_mace_pooled_oct05_prior.pt and its MACE checkpoint is that arm's mlip_path (acr_newmodel.model, the
model the prior was searched and scored under; the family default is an older one, and mlip_path is not part of the
trainer's problem identity, so the generator asserts it).

EARLY PENALTY RAMP, one mip arm (owner 2026-10-09: "so much probability bunches up on the edges it makes me wonder if
stiffer boxes earlier on would be better"): ppl_mip_ramp_n5 is the mip production arm plus prod_sep23_ft's
coeff_schedule on the equilibration stage, both penalty coefficients 10 -> 1000 over 20,000 steps, anchored at the
stage's entry, so the ramp runs during the climb from MLE and not after convergence. Why: the box penalty is
coefficient x (overshoot)^2 nats, so at 10 a density flat up to a wall keeps a shelf 0.28 latent units wide beyond it
(0.028 at 1000). ppl_mip_n5 at 80,000 phase-2 steps had 26% of its draws beyond the box, all of it in u (21.7%, both
walls) and theta (5.8%, upper wall), up from 2% at 5,000 steps: the model is still finding the shelf the soft wall
grants, and log Z rises with it.

Not here: prod_sep20's every-10th arms (settled there). ROW ORDER IS THE ARRAY INDEX: mip / neh / mipu / nehu
production are 0-3, the mip exploration arms 4-6, acr production 7, the mip early-ramp arm 8.
"""
import copy
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
_spec = importlib.util.spec_from_file_location('p20make', ROOT / 'prod_sep20' / 'make.py')
p20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p20)
fin = p20.fin
sys.path.insert(0, str(ROOT.parent))
from prior_scan_cache import CHUNKING_KEYS  # noqa: E402

TAG = 'ppl'
BATTERY = 'prod_pool_oct07'
#: family -> the battery whose mlepl_<fam>_lr2 arm is the seed
SEED_BATTERY = {'mip': 'mle_pool_oct05', 'neh': 'mle_pool_oct05', 'mipu': 'mle_pool_oct05', 'nehu': 'mle_pool_oct05',
                'acr': 'mle_pool_acr_oct05'}
FAMS = ['mip', 'neh', 'mipu', 'nehu']
ACR_MLIP = '/scratch/mk8347/data/acr_newmodel.model'
N = 5
FREEZE = 'freeze_pb:full'
#: prod_sep23_ft's ramp of both penalty coefficients (geometric, anchored at the stage's entry)
SCHEDULE = {'bounding_coeff': {'target': 1000, 'steps': 20000}, 'reduction_coeff': {'target': 1000, 'steps': 20000}}
#: (family, stored force on replay rows, P_B frozen at entry, penalty ramp from entry). ROW ORDER IS THE ARRAY INDEX:
#: append only.
ARMS = ([(fam, False, True, False) for fam in FAMS]
        + [('mip', True, True, False), ('mip', False, False, False), ('mip', True, False, False)]
        + [('acr', False, True, False)] + [('mip', False, True, True)])
p20.TAG = TAG
fin.TAG = TAG
fin.SEED_ARM.update({fam: f'{bat}/mlepl_{fam}_lr2.yaml' for fam, bat in SEED_BATTERY.items()})
fin.SRC.update({fam: f'mlepl_{fam}_lr2' for fam in SEED_BATTERY})


def build_arm(base, fam, force=False, frozen=True, ramp=False):
    name, cfg = p20.build_arm(base, fam, N, force)
    if force or ramp or not frozen:
        name = (f'{TAG}_{fam}' + ('' if frozen else '_unpb') + ('_force' if force else '') + ('_ramp' if ramp else '')
                + f'_n{N}')
        cfg['run_name'] = name
    if ramp:
        fin._stage(cfg, 'equilibration')['coeff_schedule'] = copy.deepcopy(SCHEDULE)
    if not frozen:
        eq = fin._stage(cfg, 'equilibration')
        assert eq['on_enter'][-1] == FREEZE and eq['on_enter'].count(FREEZE) == 1, name
        eq['on_enter'] = eq['on_enter'][:-1]
    seed = fin.load(fin.SEED_ARM[fam])
    for buf in ('prior_buffer', 'anchor_buffer'):
        cfg['buffers'][buf]['max_size'] = seed['buffers'][buf]['max_size']
    cfg['prior_scan_cache'] = True
    return name, cfg


def _leaves(node, path=''):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, f'{path}.{k}' if path else str(k))
    else:
        yield path, node


def moved(a, b):
    """Leaf paths on which two arms differ (a list is one leaf)."""
    la, lb = dict(_leaves(a)), dict(_leaves(b))
    return sorted(k for k in set(la) | set(lb) if la.get(k) != lb.get(k))


def check(cfg, name, fam, force=False, frozen=True, ramp=False, production=None):
    probe = copy.deepcopy(cfg)
    eq = fin._stage(cfg, 'equilibration')
    assert (eq.get('coeff_schedule') or {}) == (SCHEDULE if ramp else {}), name
    assert cfg['energy_config']['bounding_coeff'] == 10.0 == cfg['energy_config']['reduction_coeff'], name
    if not frozen:   # prod_sep20's check is the frozen arm's: judge this arm as that one plus the freeze
        assert not any(str(a).startswith('freeze_pb') for a in eq['on_enter']), name
        assert cfg.get('freeze_backward_policy') in (False, None), name
        fin._stage(probe, 'equilibration')['on_enter'] = list(eq['on_enter']) + [FREEZE]
    p20.check(probe, name, fam, N, force)
    if production is not None:   # an exploration arm is its production arm with only the named thing moved
        want = {'run_name'}
        if force:
            want.add('replay_loss_coeffs.stored_force_k')
        if ramp or not frozen:
            want.add('protocols.unconditional_tb.stages')
        assert set(moved(cfg, production)) == want, (name, moved(cfg, production))
    seed = fin.load(fin.SEED_ARM[fam])
    rows = seed['buffers']['anchor_buffer']['max_size']
    assert cfg['buffers']['prior_buffer']['max_size'] == cfg['buffers']['anchor_buffer']['max_size'] == rows, name
    assert seed['buffers']['prior_buffer']['max_size'] == rows and rows > 500_000, name
    assert cfg['prior_path'].endswith('_pooled_oct05_prior.pt') and cfg['prior_scan_cache'] is True, name
    assert cfg['buffers']['anchor_buffer']['seed_source'] == 'prior_dataset', name
    assert cfg.get('mlip_path') == seed.get('mlip_path'), name + ': the MLIP checkpoint moved from the seed arm'
    if fam == 'acr':
        assert cfg['mlip_path'] == ACR_MLIP and cfg['prior_path'].endswith('/acridine_mace_pooled_oct05_prior.pt'), name
    same = lambda ec: {k: v for k, v in ec.items() if k not in CHUNKING_KEYS}
    assert same(cfg['energy_config']) == same(seed['energy_config']), name + ': a moved energy_config would miss the scan cache'


def main(argv):
    dirty = fin.w3.dirty_files()
    if dirty:
        print('WARNING: the working tree is dirty (base read from git HEAD; the cluster runs HEAD):\n  ' + '\n  '.join(dirty))
    base = fin.committed_mk_dev()
    arms = {}
    for fam, force, frozen, ramp in ARMS:
        name, cfg = build_arm(base, fam, force, frozen, ramp)
        plain = force is False and frozen is True and ramp is False
        check(cfg, name, fam, force, frozen, ramp, production=None if plain else arms[f'{TAG}_{fam}_n{N}'][0])
        fin.load_check(cfg, name, ['train_prior', 'equilibration'])
        arms[name] = (cfg, fam)
    for a in arms:   # the job's glob for an arm's own checkpoints is *<arm>_*: it must match no other arm's files
        clash = [b for b in arms if b != a and f'{a}_' in f'{TAG}_{b}_']
        assert not clash, f'the checkpoint glob of {a} also matches {clash}'
    prior_index = {l.split('\t')[1]: l.split('\t') for bat in sorted(set(SEED_BATTERY.values()))
                   for l in (ROOT / bat / 'INDEX.tsv').read_text(encoding='utf-8').splitlines()[1:]}
    for stale in HERE.glob(f'{TAG}_*.yaml'):
        stale.unlink()
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n',
                                               encoding='utf-8')
    rows = []
    for name, (cfg, fam) in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        row = prior_index[fam]
        assert cfg['prior_path'].rsplit('/', 1)[1] == row[4], name
        rows.append((name, fam, 'seed', fin.SRC[fam], row[4], row[5]))
    fin._write_index(HERE / 'INDEX_a.tsv', rows)
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(fin.SBATCH.format(
            wall=fin.WALL, last=len(arms) - 1, tag=TAG, battery=BATTERY, leg='a', ckpts=fin.w3.CLUSTER_CKPTS,
            data=fin.w3.CLUSTER_DATA, seed_block=fin.SEED_A,
            what='PRODUCTION phase 2 on the pooled priors from the mlepl best-MLE checkpoints: a rollout every 5th '
                 'step, P_B frozen at entry, replay pinned 0.3 (mip, neh, mipu, nehu, acr); plus mip exploration arms with the stored replay '
                 'force, with P_B left trainable, with both, and with the box and reduction penalties ramped 10 -> 1000 '
                 'over the first 20k steps.'))
    for i, (name, (cfg, fam)) in enumerate(arms.items()):
        lc = cfg['lr_control']
        print(f"[{i}] {name:<12} N={N} rate={lc['fixed_scale']:g} (lr {lc['fixed_scale'] * lc['seed_lr']:g}) "
              f"batch={cfg['batch_size']} tau={p20.TAU} {'MLIP' if fin.MLIP[fam] else 'ELJ '} "
              f"P_B {'frozen at entry' if FREEZE in fin._stage(cfg, 'equilibration')['on_enter'] else 'TRAINABLE'} "
              f"{'STORED FORCE k=1 ' if cfg['replay_loss_coeffs']['stored_force_k'] else ''}"
              f"{'PENALTY RAMP 10->1000/20k ' if fin._stage(cfg, 'equilibration').get('coeff_schedule') else ''}"
              f"anchors={cfg['buffers']['anchor_buffer']['max_size']:,} noise={cfg['buffers']['anchor_buffer']['noise_log_range']} "
              f"seed=*{fin.SRC[fam]}_*_best.pt prior={cfg['prior_path'].rsplit('/', 1)[1]}")


if __name__ == '__main__':
    main(sys.argv[1:])
