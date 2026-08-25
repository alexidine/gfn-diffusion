"""
baseline_aug24 -- baseline setting for mipcas elj (primary) + qm9 conditional
(secondary) under the NEW stack (LR bracket, progress gate, hot sensor, race
telemetry). 2026-08-24, design reviewed by owner (decisions 1-5 settled in-chat;
decision 1 resolved as "canonical k=10 stands, a false trip is survivable and is
itself calibration data").

    python configs/baseline_aug24/make.py     # configs + INDEX + sbatch

WHAT THIS IS. Reference numbers for what HEALTHY looks like on the real routes:
phase-1 + early-equilibration traces, per-stage boundary/selection (race
tables), gate exit step + progress/reason, sensor levels. Plus the bracket's
pre-registered acceptance test and the hot-sensor 2x/4x calibration arms.

WHAT THIS IS NOT. Not convergence: epochs 30000 / 12 h caps -- efficient
training, working out the features (owner directive).

THE ACCEPTANCE BAR (pre-registered): U3 (bracket) within 15% of the BEST fixed
arm at EQUAL TOTAL COMPUTE, discarded trial steps charged against the bracket.
Compare at equal compute or matched loss depth -- NEVER slopes at matched step
count on a decaying curve.

ARMS (each differs from canonical mk_dev by the smallest possible key set;
values written by omission would inherit mk_dev silently, so everything battery-
relevant is written EXPLICITLY and re-asserted in check()):

  U0 fixed 0.05 (6.25e-6)   conservative floor control
  U1 fixed 0.4  (5e-5)      the "operator chose well" baseline
  U2 fixed 1.2  (1.5e-4)    the "operator chose hot" arm; a detonation is data
                            and is survivable (rolling rewind + budget abort)
  U3 bracket, canonical     THE feature arm (k=10 as shipped -- decision 1)
  U4 bracket, repeat_every 0  isolates the repeat cost (single-key vs U3)
  H1 fixed 0.8 (1e-4, 2x U1), excursion bar PARKED   sensor calibration
  H2 fixed 1.6 (2e-4, 4x U1 = the measured non-finite edge), bar PARKED
  C0 conditional, fixed 0.4   conditional baseline (no clean fixed-rate record
                            exists from qm9anchor_aug14 -- those arms ran the
                            adaptive era at seed 1.25e-4; 0.4 = the cross-route
                            stable point, stated honestly as a choice)
  C1 conditional, bracket   var_conditioning under the bracket: the one thing
                            the toy workout could not pre-validate
  C2 conditional, fixed 0.8 (2x C0), bar PARKED   conditional hot point;
                            fwd/vg_lb is the sensor channel to read

Toy-workout findings this battery watches for on the real routes:
  D1  a late false trip of the k=10 excursion bar on a converged stage
      (lr_ctrl DIVERGENCE + rolling rewind; budget abort would be a finding)
  D8  lr_bracket/refusal_step_* stamps (accumulation starving Adam's counter --
      should NOT occur at batch 20000)
  D9  boundary conservatism: compare U3's selected rate against U1/U2 survival
"""

import copy
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIGS = HERE.parent
sys.path.insert(0, str(CONFIGS.parent))      # energy_sampling/ for config_snapshot

MK_DEV = CONFIGS / 'mk_dev.yaml'

CLUSTER_DATA = '/scratch/mk8347/data/crystal_datasets/conditional/priors'
CLUSTER_CKPTS = ('/scratch/mk8347/projects/gfn_cond/gfn-diffusion/'
                 'energy_sampling/checkpoints/')

EPOCHS = 30000
TIME = '12:00:00'            # likely need, never the 48 h max
EVAL_PERIOD = 500            # cluster cadence
FIGS_PERIOD = 1000
GRAD_ABS = 5.0e7             # decision 2: the backstop must clear stage-entry
                             # transients (toy equilibration entry ran 2.6e6
                             # pre-clip); still catches numerical death only
K_PARKED = 1.0e6             # H/C2 arms: excursion bar out of the way so the
                             # overheated section PERSISTS for the sensor


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def deep_set(cfg, dotted, value):
    node = cfg
    keys = dotted.split('.')
    for k in keys[:-1]:
        node = node[k]                       # KeyError = the base moved; loud
    node[keys[-1]] = value


def common(cfg, name):
    cfg['run_name'] = name
    cfg['tag'] = 'base24'
    cfg['seed'] = 12345
    cfg['device'] = 'cuda'
    cfg['cuda_memory_fraction'] = 0.9        # aug19 battery convention
    cfg['checkpoints_dir'] = CLUSTER_CKPTS
    cfg['epochs'] = EPOCHS
    cfg['eval_period'] = EVAL_PERIOD
    cfg['figs_period'] = FIGS_PERIOD
    cfg['archive_period'] = 5000
    cfg['archive_buffers'] = True
    deep_set(cfg, 'lr_control.hard_failure.grad_abs', GRAD_ABS)
    # The directive-batch knobs, EXPLICIT because they are mechanism-under-test
    # (a value by omission would be a code default two repos away):
    cfg['lr_control']['trial_settle_steps'] = 10      # the switch-splash window (D9)
    cfg['lr_control']['logz_detour_nats'] = 2.0       # the guiding-star guard, production bar per owner
    cfg['lr_control']['fire_cut_factor'] = 0.5        # both fire tiers cut by this
    cfg['lr_control']['fire_cooldown_steps'] = 100    # one incident, one moderate cut
    fresh(cfg)
    return cfg


def fresh(cfg):
    """From scratch, EXPLICITLY -- mk_dev currently warm-starts from a pinned
    phase-1 exit, and an arm that silently inherits that measures the wrong
    thing and says so nowhere. Phase 1 is part of this battery's observation."""
    cfg['continue_from_checkpoint'] = False
    cfg['checkpoint_name'] = None
    cfg['prior_model_name'] = None
    cfg['load_weights_only'] = False


def uncond(cfg):
    """mipcas elj on cluster paths. Everything else is mk_dev canonical."""
    p = f'{CLUSTER_DATA}/mipcas_sg2_zp1_elj_prior_dataset.pt'
    cfg['prior_path'] = p
    cfg['molecules_path'] = p
    cfg['test_molecules_path'] = None
    cfg['mlip_path'] = None
    return cfg


def cond(cfg):
    """qm9 conditional elj: the registry stanza's keys, written out here
    because the battery must not depend on registry resolution order. The Z
    regime (tb_z_source persistent, z_calibration off, lr_flow 1e-4) rides the
    conditional_vargrad protocol + the one lr_flow key -- the F-042 trio."""
    cfg['protocol'] = 'conditional_vargrad'
    cfg['prior_path'] = f'{CLUSTER_DATA}/qm9split_prior.pt'
    cfg['molecules_path'] = f'{CLUSTER_DATA}/qm9split_conditions.pt'
    cfg['test_molecules_path'] = f'{CLUSTER_DATA}/qm9split_test_conditions.pt'
    cfg['mlip_path'] = None
    cfg['embedding_conditioning'] = True
    cfg['embedding_conditioning_dim'] = 192
    cfg['vector_conditioning'] = False
    cfg['molecule_conditioning'] = False
    cfg['space_groups'] = [2]
    cfg['z_primes'] = [1]
    cfg['lr_flow'] = 1.0e-4                  # conditional Z(c) is a NETWORK; 0.1 detonates it
    deep_set(cfg, 'energy_config.temperature', 6.9)   # the qm9 route's kT (registry)
    deep_set(cfg, 'model.periodic_centroids', True)
    # NO ENERGY TERMS in the conditional phase-1 gate (owner, 2026-08-25): the
    # prior trains UNCONDITIONALLY, samples converge to the pooled marginal
    # {x}, whose energy under any particular c is structurally garbage vs the
    # dataset's matched {x,c} pairs -- E/sample vs E/ref can never close
    # (pooled or per-condition) and CONVERGED would be unreachable, hanging
    # the arm to the 20k SATURATED window. w1r + the MLE slope are the valid
    # phase-1 instruments on this route.
    cfg['progress_gate']['metrics'] = [
        {'key': 'w1r/median', 'target_key': 'w1r/perfect_median', 'bar': 1.5},
    ]
    return cfg


def fixed(cfg, scale, park_bar=False):
    deep_set(cfg, 'lr_control.mode', 'fixed')
    deep_set(cfg, 'lr_control.fixed_scale', scale)
    if park_bar:
        deep_set(cfg, 'lr_control.hard_failure.loss_excursion_k', K_PARKED)
    return cfg


def build():
    arms = {}

    arms['base24_u0_fixed005'] = fixed(uncond(common(base(), 'base24_u0_fixed005')), 0.05)
    arms['base24_u1_fixed04'] = fixed(uncond(common(base(), 'base24_u1_fixed04')), 0.4)
    arms['base24_u2_fixed12'] = fixed(uncond(common(base(), 'base24_u2_fixed12')), 1.2)
    arms['base24_u3_bracket'] = uncond(common(base(), 'base24_u3_bracket'))
    u4 = uncond(common(base(), 'base24_u4_bracket_norepeat'))
    deep_set(u4, 'lr_control.repeat_every', 0)
    arms['base24_u4_bracket_norepeat'] = u4
    arms['base24_h1_hot2x'] = fixed(uncond(common(base(), 'base24_h1_hot2x')), 0.8, park_bar=True)
    arms['base24_h2_hot4x'] = fixed(uncond(common(base(), 'base24_h2_hot4x')), 1.6, park_bar=True)

    arms['base24_c0_fixed04'] = fixed(cond(common(base(), 'base24_c0_fixed04')), 0.4)
    arms['base24_c1_bracket'] = cond(common(base(), 'base24_c1_bracket'))
    arms['base24_c2_hot2x'] = fixed(cond(common(base(), 'base24_c2_hot2x')), 0.8, park_bar=True)

    return arms


def check(arms):
    """Assertions over INTENT, then a real load of every emitted file."""
    for name, cfg in arms.items():
        assert cfg['continue_from_checkpoint'] is False, name
        assert cfg['checkpoint_name'] is None, name
        assert cfg['prior_model_name'] is None, name
        assert cfg['epochs'] == EPOCHS, name
        assert cfg['lr_control']['hard_failure']['grad_abs'] == GRAD_ABS, name

    # the single-key claim U3 vs U4
    u3, u4 = arms['base24_u3_bracket'], arms['base24_u4_bracket_norepeat']
    diff = {}
    for k in set(u3['lr_control']) | set(u4['lr_control']):
        if u3['lr_control'].get(k) != u4['lr_control'].get(k):
            diff[k] = (u3['lr_control'].get(k), u4['lr_control'].get(k))
    assert set(diff) == {'repeat_every'}, f'U3 vs U4 differ by {sorted(diff)}'

    for name in ('base24_h1_hot2x', 'base24_h2_hot4x', 'base24_c2_hot2x'):
        assert arms[name]['lr_control']['hard_failure']['loss_excursion_k'] == K_PARKED, name
        assert arms[name]['lr_control']['mode'] == 'fixed', name
    for name in ('base24_u3_bracket', 'base24_c1_bracket'):
        assert arms[name]['lr_control']['mode'] == 'bracket', name
        # decision 1: canonical k stands on the bracket arms
        assert arms[name]['lr_control']['hard_failure']['loss_excursion_k'] == 10.0, name
    for name, cfg in arms.items():
        want = 'conditional_vargrad' if '_c' in name else 'unconditional_tb'
        assert cfg['protocol'] == want, (name, cfg['protocol'])

    import config_snapshot
    for name in arms:
        snap = config_snapshot.snapshot(str(HERE / f'{name}.yaml'))
        assert not snap.get('load_error'), (name, snap.get('load_error'))


SBATCH = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-{last}
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=base24
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/baseline_aug24/joblogs/%x_%A_%a.out

# baseline_aug24. Arm = row of INDEX.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/baseline_aug24
LOGS=${{ARMS}}/joblogs
mkdir -p ${{LOGS}}

ARM=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $1}}' ${{ARMS}}/INDEX.tsv)
if [ -z "${{ARM}}" ]; then echo "no arm at row ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; fi
CONFIG=${{ARMS}}/${{ARM}}.yaml
if [ ! -f "${{CONFIG}}" ]; then echo "missing config ${{CONFIG}}" >&2; exit 1; fi
echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}"
J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}

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
        python -u train.py --config ${{CONFIG}}
    "
"""


def main():
    arms = build()
    HERE.mkdir(exist_ok=True)
    (HERE / 'joblogs').mkdir(exist_ok=True)
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tquestion\n')
        notes = {
            'base24_u0_fixed005': 'conservative floor control',
            'base24_u1_fixed04': 'operator-chose-well baseline',
            'base24_u2_fixed12': 'operator-chose-hot control',
            'base24_u3_bracket': 'THE feature arm; acceptance bar vs best fixed',
            'base24_u4_bracket_norepeat': 'repeat cost isolation (single key vs U3)',
            'base24_h1_hot2x': 'sensor calibration 2x, bar parked',
            'base24_h2_hot4x': 'sensor calibration 4x = measured edge, bar parked',
            'base24_c0_fixed04': 'conditional baseline',
            'base24_c1_bracket': 'var_conditioning under the bracket',
            'base24_c2_hot2x': 'conditional hot point, fwd/vg_lb channel',
        }
        for name in arms:
            f.write(f'{name}\t{notes[name]}\n')
    with (HERE / 'submit_baseline_aug24.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(time=TIME, last=len(arms) - 1))
    check(arms)
    print(f'baseline_aug24: {len(arms)} arms generated + INDEX + sbatch; checks pass')
    print('submit line: cd /scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling '
          '&& sbatch configs/baseline_aug24/submit_baseline_aug24.sbatch')


if __name__ == '__main__':
    main()
