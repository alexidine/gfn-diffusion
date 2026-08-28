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
        # ---- WAVE 2, 2026-08-28: RESUMED with the occupancy fixes -----------
        # All four surviving arms were CANCELLED together at ~2.47 h with
        # external GPU 46-50%. Measured cause, and it is NOT the energy
        # function reaching the loss (MLE never reads it -- train_step_time is
        # 1.50/1.66 s here against 1.41/1.51 s on the ELJ families, i.e. the
        # training step is molecule-independent, exactly as expected):
        #
        #   wall-clock OUTSIDE the training step   acr 47-48%   elj 6-9%
        #   fraction of util samples under 40%     acr 36%      elj 3-5%
        #
        # That is eval, and only eval: 10000 samples rolled out at T=100 and
        # scored with MACE every 500 steps, at near-zero occupancy. It drags
        # the MEDIAN to 45% (mipcas 55%) while barely moving the mean, which
        # is what put acridine in the cancellation band and left ELJ out of it.
        # eval_period/figs_period stay 500/1000 -- those are the wandb storage
        # billing knobs, and the cost here is COMPUTE per eval, not artifacts.
        'eval_num_samples': 2500,
        # The batch never grew: "64.0% at 1000 clears the 60% target; holding
        # the base batch", on every arm of every family. NOT a VRAM wall --
        # zero OOM anywhere, 73 GB cap untouched, and the ladder only ever
        # tried rungs 1000/1600. So `traj_checkpoint` is NOT needed in phase 1
        # and the owner's phase > 1 rule stands; the target is what stalls it.
        'batch_util_target': 0.95,
        # mip_lr1p0 false-fired at step 503 on an excursion of 41.9 band widths
        # against a bar set at exactly 40. The T=100 promotion transient is
        # bigger than the T=10 one this was calibrated on (10x the integrator
        # steps compounding into it). The two scale-8 fires were genuine at 463
        # and 1257 widths, so 60 still separates them by an order of magnitude.
        'excursion_k': 60.0,
        # RESUME the four survivors from their own _running.pt rather than
        # re-paying ~3000 steps. Same run_name + continue_from_checkpoint is a
        # FULL resume (load_weights_only is read only on the checkpoint_name
        # branch; the auto-resume path always calls load_full), so optimizers,
        # step count and buffers all carry.
        'resume': True,
        # 8.0 is DROPPED, not resumed: it detonated unrecoverably at step 670
        # (FrozenTrainingState, 4 rewinds) and the ceiling it establishes is
        # already recorded by nehzor's identical death. There is no useful
        # checkpoint to continue from.
        'scales': [0.5, 1.0, 2.0, 4.0],
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
        for s in spec.get('scales', SCALES):
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
            # MLIP eval cost is per-SAMPLE, so this is the knob -- NOT
            # eval_period/figs_period, which stay 500/1000 because they are the
            # wandb storage billing knobs and the cost here is compute.
            if 'eval_num_samples' in spec:
                cfg['eval_num_samples'] = spec['eval_num_samples']
            if 'batch_util_target' in spec:
                cfg['batch_util_target'] = spec['batch_util_target']

            # -- COLD, or RESUMED ---------------------------------------------
            # A resumed arm keeps its run_name, so `continue_from_checkpoint`
            # picks up its OWN {tag}_{run_name}_{problem}_running.pt. That path
            # always calls load_full -- load_weights_only is read only on the
            # checkpoint_name branch -- so optimizers, step and buffers carry.
            cfg['checkpoint_name'] = None
            cfg['load_weights_only'] = False
            cfg['continue_from_checkpoint'] = bool(spec.get('resume', False))
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
            lc['hard_failure']['loss_excursion_k'] = float(
                spec.get('excursion_k', EXCURSION_K))

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
    assert lc['hard_failure']['loss_excursion_k'] == float(
        spec.get('excursion_k', EXCURSION_K))
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

    # No WARM SOURCE may ever leak in, or the fan re-creates the prod_aug26
    # confound (arms that start converged measure perturbation, not rate).
    # A RESUMED arm is a different thing: it continues its OWN history at the
    # same rate, so it adds steps to a measurement rather than contaminating it.
    assert cfg['checkpoint_name'] is None, 'no cross-run warm start in this fan'
    assert cfg['load_weights_only'] is False
    assert cfg['prior_model_name'] is None
    assert cfg['continue_from_checkpoint'] is bool(spec.get('resume', False))

    # eval cost is the MLIP failure mode: acridine spent 47-48% of wall clock
    # outside the training step (ELJ: 6-10%) purely on scoring eval draws
    if spec.get('eval_num_samples'):
        assert cfg['eval_num_samples'] == spec['eval_num_samples']
        assert cfg['energy_function'] != 'elj', \
            'the eval-cost cut is for MLIP routes; ELJ eval is already 6-10%'
    # the billing knobs are NOT the lever and must not drift
    assert cfg['eval_period'] == 500 and cfg['figs_period'] == 1000
    assert cfg['figs_period'] % cfg['eval_period'] == 0, \
        'figs_period must be a multiple of eval_period or figures never fire'

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
CKPTS={ckpts}
mkdir -p ${{LOGS}}

ARM=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $1}}' ${{ARMS}}/INDEX.tsv)
RESUME=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $4}}' ${{ARMS}}/INDEX.tsv)
if [ -z "${{ARM}}" ]; then echo "no arm at row ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; fi
CONFIG=${{ARMS}}/${{ARM}}.yaml
if [ ! -f "${{CONFIG}}" ]; then echo "missing config ${{CONFIG}}" >&2; exit 1; fi

# The config is used verbatim -- there is no cross-run warm start to resolve.
# A per-job copy is still taken so the joblog carries the exact config the arm
# ran. A RESUMED arm continues its OWN _running.pt via run_name, so nothing is
# substituted; but the file MUST exist, because `continue_from_checkpoint` with
# nothing to continue silently COLD-STARTS -- which would look like a resumed
# arm and quietly throw away the steps this wave exists to keep.
J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}
RESOLVED=${{J}}.yaml
cp ${{CONFIG}} ${{RESOLVED}}
if [ "${{RESUME}}" = "resume" ]; then
    N=$(ls ${{CKPTS}}/*${{ARM}}*_running.pt 2>/dev/null | wc -l)
    if [ "${{N}}" -eq 0 ]; then
        echo "FATAL: arm ${{ARM}} is marked resume but no *${{ARM}}*_running.pt exists in ${{CKPTS}}" >&2
        exit 1
    fi
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  RESUME <- $(ls -t ${{CKPTS}}/*${{ARM}}*_running.pt | head -1 | xargs basename)"
else
    echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  (cold, T=100)"
fi

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
        # column 4 is APPENDED: the mip/neh sbatch already on the cluster reads
        # $1 only, so widening the file cannot disturb a live array.
        f.write('arm\tfamily\tscale\tmode\n')
        for name in names:
            mode = 'resume' if FAMILIES[fam_of(name)].get('resume') else 'cold'
            f.write(f"{name}\t{fam_of(name)}"
                    f"\t{arms[name]['lr_control']['fixed_scale']}\t{mode}\n")

    for fam, wall in WAVES.items():
        rows = [i for i, n in enumerate(names) if fam_of(n) == fam]
        assert rows == list(range(rows[0], rows[-1] + 1)), \
            f'family {fam} rows are not contiguous: {rows}'
        fname = f'submit_prod_t100_{fam}.sbatch'
        with (HERE / fname).open('w', encoding='utf-8', newline='\n') as f:
            f.write(SBATCH.format(first=rows[0], last=rows[-1], time=wall,
                                  wave=fam, ckpts=CLUSTER_CKPTS.rstrip('/')))
        print(f'{fname}: rows {rows[0]}-{rows[-1]}  wall {wall}')
    # THE LIVE ROWS ARE FROZEN. mip (0-4) and neh (5-9) are running on the
    # cluster right now and their sbatch maps array index -> INDEX row, so any
    # renumbering would repoint a live array at different arms.
    assert names[:10] == [f'pt100_{f}_lr{scale_tag(s)}'
                          for f in ('mip', 'neh') for s in SCALES], \
        'rows 0-9 moved -- mip/neh arrays are LIVE, do not regenerate'

    print(f'\n{len(arms)} arms written to {HERE}')
    for name in names:
        print(f"  {name:<22} scale {arms[name]['lr_control']['fixed_scale']}")


if __name__ == '__main__':
    main()
