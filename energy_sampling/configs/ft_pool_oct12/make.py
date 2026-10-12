"""ft_pool_oct12 -- penalty fine-tuning legs on the prod_pool_oct07 production samplers (owner 2026-10-12: "queue up
fine tuning runs for all of them excluding the different mip runs").

    python configs/ft_pool_oct12/make.py         # base = the COMMITTED mk_dev (git show HEAD:), warns on a dirty tree

THE ARMS: pft_<fam>_ft for mip / neh / mipu / nehu / acr -- the family's prod_pool_oct07 production config (its
build_arm and check, unchanged: every 5th step, P_B frozen, replay pinned 0.3 / bwd 0.7, tau 600, rate 0.5, batch
1600; acr MACE, batch 1000) plus prod_sep23_ft's ramp on the equilibration stage,
    coeff_schedule: {bounding_coeff: {target: 1000, steps: 20000}, reduction_coeff: {target: 1000, steps: 20000}}
a geometric ramp of both penalty coefficients from the config's 10 to 1000 over 20,000 train steps, anchored at the
leg's first tick (stage_ctrl.coeff_sched_entry rides the checkpoint, so a requeue continues it), then held. Each
arm is a FULL resume of its ppl_<fam>_n5 arm: by default that arm's newest step archive (_stepN.pt with its frozen
_stepN_buffers.pt); with SRC_RUNNING=1 in the job's environment, its _running.pt and rolling sidecar, for a source
arm that has ended (nothing is lost to the 5,000-step archive cadence). The coefficients are not part of the problem
identity, so the load is the ordinary continuation: stage equilibration, P_B snapshot, LR scale and buffers restored.

ACRIDINE differs in three ways (owner 2026-10-12: "The acr run started making super diffuse crystals ... go back to a
checkpoint close to when it started really exploding, around 100k, and add density energy ramping").

  seed   ppl_acr_n5's step-100000 archive, NAMED in the index (seed_step) and resolved by the job in preference to
         SRC_RUNNING. From that run's evals (medians over 4,000 steps): mean packing coefficient 0.608 at 100,000,
         0.596 at 104,000, 0.508 at 108,000, 0.376 at 124,000; reasonable fraction 0.91 / 0.81 / 0.35 / 0.07;
         learned log Z 0.29 / 0.73 / 2.57 / 4.58. 100,000 is the last archive before the slide.
  ramp   density_coeff 10 -> 1000 as well, over DENSITY_STEPS = 2,000 steps, not 20,000. The density penalty is
         coefficient x f(u) nats with u = ln(0.55 / packing coefficient) and f = u^2 up to u = 1: at 10 a draw at
         packing 0.38 pays 1.4 nats while the diffuse region carried about 4 nats more log Z, so the soft wall loses;
         at 1000 it pays 140, and 1 nat is reached at packing 0.533. At the seed step no draw touches the penalty
         (mean density energy 5e-5), so a fast ramp shocks nothing, and the slide took 4,000 to 8,000 steps: a
         20,000-step geometric ramp would be at 25 to 63 over that window. The box and reduction ramps stay at 20,000.
  floor  condition_log_z.best_energy_from_anchors_on_switch = true. The seed's tracker holds a best energy of
         -1,327,550 for acridine (inherited from the MLE run; the anchor minimum is -45.3). The prior buffer's
         admission gate and staleness rule are both measured from that record, so ppl_acr_n5 admitted no noised anchor
         and expired its rows: 2.88M at step 86,660, 766k at 100,000, 43k at 124,000. The key resets the record to
         the anchor minimum on this load, so top-ups and normal turnover resume from the 766k rows the archive holds.

READ: energy_func//bounding_coeff, //reduction_coeff, //density_coeff (the live ramps), protocol/coeff_sched_p_*,
fwd/box_contact_frac, Mean bounding_energy / reduction_energy / density_energy, Mean Packing Coeff, Reduced Valid
Fraction, fwd/log_Z_learned (falls as mass leaves the walls: the stiff target has its own Z), eval_fwd/tb_err. On
acr also condition_best_energy_median (about -45, not -1.3e6), anchor_topup_last_n (above 0) and
prior_buffer_length (no longer falling).

ROW ORDER IS THE ARRAY INDEX: mip 0, neh 1, mipu 2, nehu 3, acr 4.
"""
import copy
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
_spec = importlib.util.spec_from_file_location('pplmake', ROOT / 'prod_pool_oct07' / 'make.py')
ppl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ppl)
p20, fin = ppl.p20, ppl.fin

TAG = 'pft'
BATTERY = 'ft_pool_oct12'
SRC_TAG = ppl.TAG                       # the source arms are ppl_<fam>_n5
FAMS = ['mip', 'neh', 'mipu', 'nehu', 'acr']
N = ppl.N
TARGET = 1000.0
STEPS = 20000
DENSITY_STEPS = 2000
SCHEDULE = {'bounding_coeff': {'target': TARGET, 'steps': STEPS, 'kind': 'geometric'},
            'reduction_coeff': {'target': TARGET, 'steps': STEPS, 'kind': 'geometric'}}
ACR_SCHEDULE = {**SCHEDULE, 'density_coeff': {'target': TARGET, 'steps': DENSITY_STEPS, 'kind': 'geometric'}}
#: family -> the source arm's step archive to seed from; absent = its newest archive (or _running.pt under SRC_RUNNING)
SEED_STEP = {'acr': 100000}
FLOOR_KEY = 'best_energy_from_anchors_on_switch'

# fin.SEED_B with one more way in, tried first: a step NAMED in the index's seventh column
_HEAD = '    if [ -n "${SRC_RUNNING:-}" ]; then\n'
assert fin.SEED_B.count(_HEAD) == 1
SEED_BLOCK = fin.SEED_B.replace(_HEAD, '''    # A STEP NAMED IN THE INDEX (column 7, "-" = none) is that arm's seed whatever SRC_RUNNING says: the source arm's
    # _step<N>.pt archive and its frozen _step<N>_buffers.pt.
    SEED_STEP=$(awk -F'\\t' -v n=${ROW} 'NR==n {print $7}' ${INDEX})
    if [ -n "${SEED_STEP}" ] && [ "${SEED_STEP}" != "-" ]; then
        NS=$(ls ${CKPTS}/*${SRC}_*_step${SEED_STEP}.pt 2>/dev/null | wc -l)
        if [ "${NS}" -ne 1 ]; then
            echo "FATAL: ${NS} matches for *${SRC}_*_step${SEED_STEP}.pt in ${CKPTS} (need exactly 1)" >&2; exit 1
        fi
        CK=$(ls ${CKPTS}/*${SRC}_*_step${SEED_STEP}.pt)
        if [ ! -f "${CK%.pt}_buffers.pt" ]; then
            echo "FATAL: ${CK} has no frozen buffers sidecar beside it" >&2; exit 1
        fi
        echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  SWITCH from the NAMED archive: $(basename ${CK}) (mtime $(stat -c %y ${CK}))"
    elif [ -n "${SRC_RUNNING:-}" ]; then
''')


def build_arm(base, fam):
    p20.TAG = fin.TAG = TAG
    _, production = ppl.build_arm(base, fam)
    name = f'{TAG}_{fam}_ft'
    production['run_name'] = name
    ppl.check(production, name, fam)            # the production arm under this tag, before anything is added
    cfg = copy.deepcopy(production)
    fin._stage(cfg, 'equilibration')['coeff_schedule'] = copy.deepcopy(ACR_SCHEDULE if fam == 'acr' else SCHEDULE)
    if fam == 'acr':
        cfg['condition_log_z'][FLOOR_KEY] = True
    return name, cfg, production


def check(cfg, name, fam, production):
    want = {'protocols.unconditional_tb.stages'} | ({f'condition_log_z.{FLOOR_KEY}'} if fam == 'acr' else set())
    assert set(ppl.moved(cfg, production)) == want, (name, ppl.moved(cfg, production))
    eq, eq0 = fin._stage(cfg, 'equilibration'), fin._stage(production, 'equilibration')
    assert {k: v for k, v in eq.items() if k != 'coeff_schedule'} == \
           {k: v for k, v in eq0.items() if k != 'coeff_schedule'}, name + ': the stage moved beyond its schedule'
    sched = eq['coeff_schedule']
    assert sched == (ACR_SCHEDULE if fam == 'acr' else SCHEDULE), name
    ec = cfg['energy_config']
    for coef, spec in sched.items():           # a geometric ramp starts from the config's value
        assert ec[coef] == 10.0 and spec['target'] == TARGET and spec['kind'] == 'geometric', (name, coef)
    assert ('density_coeff' in sched) == (fam == 'acr') == bool(cfg['condition_log_z'].get(FLOOR_KEY)), name
    assert cfg['buffers'].get('fresh_on_switch') in (False, None), name + ': the floor reset reads the restored anchors'
    assert cfg['run_name'] == name and cfg['tag'] == TAG, name


def main(argv):
    dirty = fin.w3.dirty_files()
    if dirty:
        print('WARNING: the working tree is dirty (base read from git HEAD; the cluster runs HEAD):\n  ' + '\n  '.join(dirty))
    base = fin.committed_mk_dev()
    assert base['condition_log_z'].get(FLOOR_KEY) is False, f'mk_dev at HEAD does not carry condition_log_z.{FLOOR_KEY}'
    arms = {}
    for fam in FAMS:
        name, cfg, production = build_arm(base, fam)
        check(cfg, name, fam, production)
        fin.load_check(cfg, name, ['train_prior', 'equilibration'])
        arms[name] = (cfg, fam)
    src = {name: f'{SRC_TAG}_{fam}_n{N}' for name, (cfg, fam) in arms.items()}
    for a in arms:      # own-checkpoint glob *<arm>_* and seed glob *<src>_* must each match one arm's files only
        clash = [b for b in arms if b != a and f'{a}_' in f'{TAG}_{b}_']
        assert not clash and not any(f'{s}_' in f'{TAG}_{a}_' for s in src.values()), (a, clash)
    assert len(set(src.values())) == len(src)
    prior_index = {l.split('\t')[0]: l.split('\t')
                   for l in (ROOT / 'prod_pool_oct07' / 'INDEX_a.tsv').read_text(encoding='utf-8').splitlines()[1:]}
    for stale in HERE.glob(f'{TAG}_*.yaml'):
        stale.unlink()
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n',
                                               encoding='utf-8')
    with (HERE / 'INDEX_a.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tstart\twarm_src\tprior\tprior_bytes\tseed_step\n')
        for name, (cfg, fam) in arms.items():
            with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as g:
                yaml.safe_dump(cfg, g, sort_keys=False, default_flow_style=False)
            row = prior_index[src[name]]
            assert row[1] == fam and cfg['prior_path'].rsplit('/', 1)[1] == row[4], name
            f.write('\t'.join([name, fam, 'switch', src[name], row[4], row[5].strip(),
                               str(SEED_STEP.get(fam, '-'))]) + '\n')
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(fin.SBATCH.format(
            wall=fin.WALL, last=len(arms) - 1, tag=TAG, battery=BATTERY, leg='a', ckpts=fin.w3.CLUSTER_CKPTS,
            data=fin.w3.CLUSTER_DATA, seed_block=SEED_BLOCK,
            what='fine-tuning legs on the prod_pool_oct07 production samplers: box and reduction penalties ramped '
                 '10 -> 1000 over 20k steps (coeff_schedule), each seeded from its ppl arm (newest step archive, or '
                 '_running.pt under SRC_RUNNING=1); acr from its step-100000 archive with the density penalty ramped '
                 'too and the tracker floor reset from the anchors.'))
    for i, (name, (cfg, fam)) in enumerate(arms.items()):
        sched = fin._stage(cfg, 'equilibration')['coeff_schedule']
        print(f"[{i}] {name:<12} rate={cfg['lr_control']['fixed_scale']:g} batch={cfg['batch_size']} ramps "
              + ', '.join(f"{k.replace('_coeff', '')} 10->{v['target']:g}/{v['steps']}" for k, v in sched.items())
              + f"  seed=*{src[name]}_*_" + (f"step{SEED_STEP[fam]}.pt" if fam in SEED_STEP else 'step<newest>.pt | _running.pt')
              + ('  FLOOR RESET' if cfg['condition_log_z'].get(FLOOR_KEY) else ''))


if __name__ == '__main__':
    main(sys.argv[1:])
