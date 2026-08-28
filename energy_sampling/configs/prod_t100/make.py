"""prod_t100: fresh phase-1 (MLE) rate fans at the SHIP length, T=100.

    python configs/prod_t100/make.py     # configs + INDEX + sbatch

WHY THIS EXISTS. Every prod_aug26/28 arm ran at `integrator.T: 10`, inherited
from mk_dev, when the production length is 100 (owner, 2026-08-28). T is not in
`problem_def`, so nothing failed loudly -- 37 arms trained at the wrong length
and reported healthily. This battery restarts phase 1 at the ship length.

WHAT WAS ACTUALLY LOST, so the grid is not built on it. The T=10 fans returned
one usable number and four non-answers:

  acridine mace  the ONLY real measurement: monotone in rate, best at the TOP
                 rung (0.1 -> -7.09, 0.2 -> -7.37, 0.4 -> -8.12, 0.8 -> -10.00),
                 no fires. A LOWER BOUND (optimum >= 0.8), not a point.
  mipcas elj     ARTIFACT. The warm source was already inside the gate bars, so
  nehzor elj     the arms measured which rate disturbs a converged checkpoint
                 least -- an answer that is always "the smallest", and both
                 families duly produced exactly that, monotonically. Not an
                 optimum.
  mipcas uma     1 of 3 arms uncut; the other two were rate-halved at promotion.
  nehzor uma     Same. A single point each, not an ordering.

COLD, and that is the point. Warm-starting from the T=10 exits would carry a
policy whose per-step drift and variance are calibrated to steps 10x too large,
AND would re-create the exact confound that ruined the ELJ fans -- a fan whose
arms all start converged measures perturbation, not rate. Cold is the only clean
rate measurement available, and phase 1 is the cheap phase to spend it on.

THE GRID IS WIDE ON PURPOSE -- 2x spacing over 16x, not sqrt(2) over 4x. The
optimum's location at T=100 is genuinely unknown, and the two measured points on
record disagree with the retired scaling rule:

    1.25e-4 @ T=10 (local_aug08)     <- this is exactly lr_control.seed_lr,
    4.0e-4  @ T=25 (aug02)              so scale 1.0 IS the T=10 optimum

i.e. the optimum ROSE 3.2x over a 2.5x rise in T, where the deleted `x25/T` rule
predicted a FALL. Mechanism for the rise: Adam is invariant to a constant
gradient rescale, so the 17x rise in grad-norm median from T=10 to T=100 is
absorbed -- what does not cancel is noise, and more integrator steps average
more, so a higher rate stays stable. Log-linear extrapolation of those two
points to T=100 lands near scale 18, which utils.py explicitly warns against
promoting to a law (one battery, one energy, one T, one W). So the grid brackets
BOTH hypotheses -- scale 1.0 (T-invariant) through 8.0 (most of the way to the
extrapolation) -- rather than betting on either.

An arm that detonates at the top of this grid is a RESULT, not a failure.

TWO DELIBERATE SETTINGS:

  traj_checkpoint: false   Owner rule 2026-08-28: grad checkpointing rides the
                           MLIPs in phase > 1 ONLY. Phase 1 is one run per arm,
                           so the rule is just this flag here; the phase-2
                           generator sets it true on the mace/uma families.
                           ^ SEE THE FOOTER WARNING: at T=100 this caps the
                           batch hard, and batch is what buys occupancy.
  loss_excursion_k: 40     prod_aug26 fired 5 of 20 arms at step 502-505, within
                           5 steps of the burn-in promotion, each fire halving
                           the arm's rate (mip_lr1p6 -> scale 0.025). The bar is
                           fitted at burn_in_scale 0.05 where the band is narrow,
                           then held live across a jump of up to 32x. In a fan
                           the paired rate cut rewrites the independent variable,
                           and it lands ONLY on the high arms. 40 passes the four
                           promotion transients on record and still trips the one
                           real detonation (90 band widths).

FOOTER WARNING, unresolved and worth a decision before the phase-2 wave.
Occupancy is set by GPU work per unit time against a ~0.35 s fixed per-step
cost. At T=10 the survivable regime was batch ~20k. At T=100 the activation
memory per sample is 10x, so WITHOUT trajectory checkpointing the batch ceiling
falls roughly 10x -- which lands the same total work per step but split into
100 small kernel launches instead of 10 large ones, and the rollout is already
dispatch-bound. T=100 buys occupancy only if the batch can STAY large, which is
what traj_checkpoint is for (33.6x less activation VRAM for 1.7x step time,
measured at T=100/B=512). These phase-1 arms are short and gate out, so they
may finish inside the scheduler's 7200 s policy window regardless; the phase-2
arms will not, and that is where this has to be right.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIGS = HERE.parent
MK_DEV = CONFIGS / 'mk_dev.yaml'

CLUSTER_DATA = '/scratch/mk8347/data/crystal_datasets/conditional/priors'
CLUSTER_CKPTS = '/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints/'

SHIP_T = 100
EXCURSION_K = 40.0
#: 2x spacing over 16x. scale 1.0 == the measured T=10 optimum (seed_lr 1.25e-4).
SCALES = [0.5, 1.0, 2.0, 4.0, 8.0]

FAMILIES = {
    'mip': {
        'prior_path': f'{CLUSTER_DATA}/mipcas_sg2_zp1_elj_prior_dataset.pt',
        'space_groups': [2],
        'energy_function': 'elj',
        'mlip_path': None,
        'max_batch_size': 20000,
    },
    'neh': {
        'prior_path': f'{CLUSTER_DATA}/nehzor_sg14_zp1_elj_prior_dataset.pt',
        'space_groups': [14],
        'energy_function': 'elj',
        'mlip_path': None,
        'max_batch_size': 20000,
    },
    # acridine keeps its own identity fields (mace + mlip_path). Its T=10 fan is
    # the one that measured anything, and it said "higher" -- the same direction
    # the T-rise argument points, so it takes the shared grid rather than a
    # shifted one.
    'acr': {
        'prior_path': f'{CLUSTER_DATA}/acridine_sg14_zp1_mace_prior_dataset.pt',
        'space_groups': [14],
        'energy_function': 'mace',
        'mlip_path': '/scratch/mk8347/data/acr_112025_mh1_stagetwo.model',
        'max_batch_size': 50000,
    },
}

# UMA (mipcas + nehzor, the *_uma_f047_* priors) is deliberately NOT here. Cold
# phase 1 is its slow case -- mipu did not gate in 30k steps at T=10 -- and at
# T=100 a five-rung cold UMA fan is the most expensive thing in the programme.
# It waits for these three to localise the range, then runs a NARROW grid.


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def scale_tag(s):
    return str(s).replace('.', 'p')


def build():
    arms = {}
    for fam, spec in FAMILIES.items():
        for s in SCALES:
            cfg = base()
            name = f'pt100_{fam}_lr{scale_tag(s)}'

            # -- identity + location ------------------------------------------
            cfg['run_name'] = name
            cfg['tag'] = 'pt100'
            cfg['checkpoints_dir'] = CLUSTER_CKPTS
            cfg['prior_path'] = spec['prior_path']
            # one molecule, unconditional: the condition set IS the prior file.
            # mk_dev ships a LOCAL D:\ path here that exists on the dev box, so
            # leaving it passes every local preflight and kills the arm on the
            # cluster at init_mol_dataset (prod_aug26, 2026-08-27).
            cfg['molecules_path'] = spec['prior_path']
            cfg['test_molecules_path'] = None
            cfg['space_groups'] = spec['space_groups']
            cfg['energy_function'] = spec['energy_function']
            cfg['mlip_path'] = spec['mlip_path']
            cfg['max_batch_size'] = spec['max_batch_size']

            # -- COLD ---------------------------------------------------------
            cfg['checkpoint_name'] = None
            cfg['load_weights_only'] = False
            cfg['continue_from_checkpoint'] = False
            cfg['prior_model_name'] = None

            # -- THE SHIP LENGTH ----------------------------------------------
            # both, together: utils enforces eval_T == integrator.T at load, and
            # an elj battery once ran eval_T = 2T and floored its metrics on a
            # trajectory the policy never trained.
            cfg['integrator']['T'] = SHIP_T
            cfg['eval_T'] = SHIP_T
            # owner rule: grad checkpointing rides the MLIPs in phase > 1 only.
            cfg['traj_checkpoint'] = False

            # -- run shape ----------------------------------------------------
            cfg['epochs'] = 30000
            cfg['eval_period'] = 500
            cfg['figs_period'] = 1000

            # -- the fixed rate under test ------------------------------------
            lc = cfg['lr_control']
            lc['mode'] = 'fixed'
            lc['fixed_scale'] = float(s)
            lc['burn_in_steps'] = 500
            lc['burn_in_scale'] = 0.05
            lc['repeat_every'] = 0
            lc['hard_failure']['loss_excursion_k'] = EXCURSION_K

            # -- MLE only: single stage, level-gated, STOPS -------------------
            cfg['protocol'] = 'prod_mle'
            cfg['protocols']['prod_mle'] = {'stages': [{
                'name': 'train_prior',
                'train_mode': 'bwd',
                'bwd_sampling_mode': 'dataset',
                'flags': {'update_log_z': True, 'scramble_conditions': True},
                'hot_lr_sensor': {'channel': 'bwd/mle', 'form': 'absolute',
                                  'rows': 31, 'above': 5.0, 'action': 'fire'},
                'loss_coeffs': {'bwd': {'mle': 1.0, 'tbc': 0.0, 'repeats': 1.0,
                                        'tb_z_source': 'persistent'}},
                'exit': [{'metric': 'gates/progress_done', 'above': 0.5,
                          'patience': 1}],
                'on_exit': ['snapshot:phase1_exit', 'snapshot_prior', 'stop'],
            }]}

            check(cfg, fam, s)
            arms[name] = cfg
    return arms


def check(cfg, fam, s):
    """Re-assert every battery property on the FINISHED dict."""
    spec = FAMILIES[fam]
    assert cfg['integrator']['T'] == SHIP_T, 'the whole point of this battery'
    assert cfg['eval_T'] == cfg['integrator']['T'], \
        'utils hard-fails on eval_T != integrator.T at load'
    assert cfg['traj_checkpoint'] is False, \
        'owner rule: grad checkpointing is phase > 1 only'
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == float(s)
    assert lc['repeat_every'] == 0
    assert lc['burn_in_steps'] == 500 and lc['burn_in_scale'] == 0.05
    assert lc['hard_failure']['loss_excursion_k'] == EXCURSION_K
    # the guards that must SURVIVE loosening the relative bar
    assert lc['hard_failure']['loss_abs'] >= 1e6
    assert lc['hard_failure']['grad_abs'] >= 1e6

    assert cfg['protocol'] == 'prod_mle'
    stages = cfg['protocols']['prod_mle']['stages']
    assert len(stages) == 1
    st = stages[0]
    assert st['name'] == 'train_prior'
    assert st['on_exit'] == ['snapshot:phase1_exit', 'snapshot_prior', 'stop'], \
        'arms must snapshot the exit AND yield the GPU'
    assert len(st['exit']) == 1 and st['exit'][0] == {
        'metric': 'gates/progress_done', 'above': 0.5, 'patience': 1}
    assert st['hot_lr_sensor']['action'] == 'fire', \
        'the hot sensor is what still guards a detonation once k is loosened'

    # COLD -- no warm source may leak in, or the fan re-creates the prod_aug26
    # confound (arms that start converged measure perturbation, not rate)
    assert cfg['checkpoint_name'] is None
    assert cfg['load_weights_only'] is False
    assert cfg['continue_from_checkpoint'] is False
    assert cfg['prior_model_name'] is None

    assert cfg['progress_gate']['mode'] == 'level'
    bars = {m['key']: m['bar'] for m in cfg['progress_gate']['metrics']}
    assert bars == {'w1r/median': 5.0, 'w1r/worst': 10.0}
    assert cfg['grow_batch_size'] is True
    assert cfg['grad_clip_guard']['enabled'] is True
    assert cfg['prior_path'] == spec['prior_path']
    assert cfg['molecules_path'] == spec['prior_path']
    assert cfg['test_molecules_path'] is None
    assert cfg['energy_function'] == spec['energy_function']
    assert cfg['mlip_path'] == spec['mlip_path']
    assert_no_local_paths(cfg)


def assert_no_local_paths(node, trail='cfg'):
    """No Windows drive path may survive into a cluster arm. mk_dev is a LOCAL
    dev config: a path key the generator forgets to override ships a value that
    exists on the dev box -- so local preflight passes -- and 404s on the
    cluster. That shape killed all 20 prod_aug26 arms in ~10 s on 2026-08-27."""
    if isinstance(node, dict):
        for k, v in node.items():
            assert_no_local_paths(v, f'{trail}.{k}')
    elif isinstance(node, list):
        for i, v in enumerate(node):
            assert_no_local_paths(v, f'{trail}[{i}]')
    elif isinstance(node, str):
        assert not (len(node) > 2 and node[1] == ':' and node[2] in '\\/'), \
            f'local drive path leaked into a cluster arm at {trail}: {node!r}'


SBATCH = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array={first}-{last}
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pt100{wave}
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prod_t100/joblogs/%x_%A_%a.out

# prod_t100 phase-1 fan at the ship length. Arm = row of INDEX.tsv (line 1 is
# the header). DO NOT EDIT --array BY HAND: make.py rewrites it to match.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/prod_t100
LOGS=${{ARMS}}/joblogs
mkdir -p ${{LOGS}}

ARM=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $1}}' ${{ARMS}}/INDEX.tsv)
if [ -z "${{ARM}}" ]; then echo "no arm at row ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; fi
CONFIG=${{ARMS}}/${{ARM}}.yaml
if [ ! -f "${{CONFIG}}" ]; then echo "missing config ${{CONFIG}}" >&2; exit 1; fi

# COLD BATTERY -- nothing to resolve. The config is used verbatim; a per-job
# copy is still taken so the joblog directory carries the exact config the arm
# ran, the way the warm batteries do.
J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}
RESOLVED=${{J}}.yaml
cp ${{CONFIG}} ${{RESOLVED}}
echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  (cold, T=100)"

{{ nvidia-smi -L
  nvidia-smi --query-gpu=mig.mode.current,uuid,name,memory.total,driver_version --format=csv
  scontrol show job ${{SLURM_JOB_ID}}
  echo "nodelist: ${{SLURM_NODELIST}}  host: $(hostname)"
}} > ${{J}}.info 2>&1

stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \\
    --format=csv,nounits -l 10 > ${{J}}_smi.csv &
SMI_PID=$!

epilogue() {{
    kill ${{SMI_PID}} 2>/dev/null
    sacct -j ${{SLURM_JOB_ID}} --format=JobID,State,ExitCode,Elapsed,NodeList,Reason,Comment%64 \\
        > ${{J}}_sacct.txt 2>&1
}}
trap epilogue EXIT TERM

srun singularity exec --nv \\
    --overlay ${{OVERLAY}}:ro \\
    --bind ${{PROJECT_ROOT}}:${{PROJECT_ROOT}} \\
    --bind /scratch/mk8347/data:/scratch/mk8347/data \\
    --pwd ${{WORKDIR}} \\
    ${{IMAGE}} \\
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${{PROJECT_ROOT}}/MXtalTools:${{PROJECT_ROOT}}/gfn-diffusion:\\$PYTHONPATH
        python -u train.py --config ${{RESOLVED}}
    "
"""

#: one sbatch per family, so a task can be submitted or dropped on its own.
#: 24 h walls: T=100 step timing is UNMEASURED on this cluster, cold phase 1 is
#: the slow case (mipu never gated in 30k at T=10), and the level gate + `stop`
#: yields the GPU the moment an arm converges -- so a generous wall costs
#: nothing while a short one loses the arm.
WAVES = {'mip': '1-00:00:00', 'neh': '1-00:00:00', 'acr': '1-00:00:00'}


def main():
    arms = build()
    (HERE / 'joblogs').mkdir(exist_ok=True)
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    names = list(arms)
    fam_of = lambda n: n.split('_')[1]
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tscale\n')
        for name in names:
            f.write(f"{name}\t{fam_of(name)}"
                    f"\t{arms[name]['lr_control']['fixed_scale']}\n")

    for fam, wall in WAVES.items():
        rows = [i for i, n in enumerate(names) if fam_of(n) == fam]
        assert rows == list(range(rows[0], rows[-1] + 1)), \
            f'family {fam} rows are not contiguous: {rows}'
        fname = f'submit_prod_t100_{fam}.sbatch'
        with (HERE / fname).open('w', encoding='utf-8', newline='\n') as f:
            f.write(SBATCH.format(first=rows[0], last=rows[-1], time=wall,
                                  wave=fam))
        print(f'{fname}: rows {rows[0]}-{rows[-1]}  wall {wall}')

    print(f'\n{len(arms)} arms written to {HERE}')
    for name in names:
        print(f"  {name:<22} scale {arms[name]['lr_control']['fixed_scale']}")


if __name__ == '__main__':
    main()
