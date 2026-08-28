"""prod_aug28: the two phase-1 fans prod_aug26 left unfinished.

    python configs/prod_aug28/make.py     # configs + INDEX + sbatch

WHY A NEW DIRECTORY. prod_aug26's INDEX row order is FROZEN -- its make.py
asserts (warm, cold) == (14, 6) because three sbatch arrays already on the
cluster map array index -> INDEX row, so appending a phase-1 family there
would renumber a live array onto different arms. These waves are phase-1
fans, so they cannot ride the phase-2 tail either. New battery, new index.

WHAT prod_aug26 PHASE 1 ACTUALLY SETTLED (measured 2026-08-27, 20/20 arms):

  mipcas elj  NO SIGNAL. The warm source was already under both gate bars at
              the first eval (w1r 4.30/6.92 vs 5.0/10.0), so all 5 arms exited
              at step 2510 having only waited out min_history. -> phase 2.
  nehzor elj  same shape, all 5 at 2510. -> phase 2.
  acridine    THE ONE FAN THAT MEASURED SOMETHING, and it says the grid is too
  mace        COLD: bwd/mle is monotone in rate and BEST AT THE TOP RUNG --
              0.1 -> -7.09, 0.2 -> -7.37, 0.4 -> -8.12, 0.8 -> -10.00, with no
              excursion fires anywhere in the family, so those four labels are
              trustworthy. The optimum is at or above 0.8 and off the grid.
  mipcas uma  NEVER GATED in 30k steps, and only just: median 5.0-5.4 against
              a 5.0 bar, worst 9.0-13.1 against 10.0. Best mle at the highest
              EFFECTIVE rate (-26.07).
  nehzor uma  gated at 5510-9510; the higher the effective rate, the sooner.

So: WAVE A extends acridine upward, WAVE B resumes mipcas UMA. Both are the
same phase-1 machinery as prod_aug26 (fixed LR, level gate, stop on exit) with
ONE deliberate change, below.

THE ONE CHANGE -- loss_excursion_k 10 -> 40. In prod_aug26, 5 of 20 arms fired
`lr_ctrl FIRE` at step 502-505, within 5 steps of the burn-in promotion at 500,
and every fire HALVED the arm's rate. mip_lr1p6 was cut to scale 0.025, a 64x
cut, so the top rung of that grid spent the run at the bottom of it.

The mechanism is in controller.py and is DELIBERATE, not a bug: the bar is
`band_top + k * band_width` fitted during burn-in at scale 0.05, where the loss
is nearly static and the band is therefore narrow (mip_lr1p6: [-23.15, -22.09],
width 1.06 -> bar -11.5). The rate then jumps up to 32x and the ordinary
promotion transient clears it. The bars ARE refitted at the real rate, but only
after a 200-step settle plus a root_window -- step 899, ~400 steps after the
window the fires happen in. controller.py's own docstring chooses this: in fixed
mode "a bar that is too tight beats no bar at all", because a tight bar is
assumed to cost only a rewind.

IN A RATE FAN THAT ASSUMPTION INVERTS. The rewind is not the cost; the paired
rate cut is, because it silently rewrites the independent variable and voids the
arm. And the bias is not random -- it lands only on the HIGH-rate arms, which is
exactly where a rate search's answer lives. A fan can come back cleanly
"lower is better" purely because every high arm was cut.

k = 40 is calibrated on those 5 fires, measured as excursion above the band top
in units of band width:

    mip_lr1p6    10.4      nehu_lr0p4   26.5
    mipu_lr0p4   11.4      nehu_lr0p8   90.0
    mipu_lr0p8   24.0

40 passes the four that look like promotion transients and still trips on
nehu_lr0p8, which at 90 widths (a 62-nat jump) was a real detonation. This is a
judgement call on 5 points, not a derived constant -- it is the single knob
worth arguing with before these launch.

WHAT STILL GUARDS THE RUN. Two independent tripwires are untouched: the
absolute backstops (loss_abs / grad_abs, 1e6) and the stage's own
`hot_lr_sensor` (bwd/mle absolute, bar 5.0, action fire), which on both these
tasks sits far above the working range and catches a true detonation. The
relative excursion bar is the redundant one here, and it is the one that
misfires.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIGS = HERE.parent
MK_DEV = CONFIGS / 'mk_dev.yaml'

CLUSTER_DATA = '/scratch/mk8347/data/crystal_datasets/conditional/priors'
CLUSTER_CKPTS = '/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints/'
PLACEHOLDER = 'WARM_CHECKPOINT_PLACEHOLDER'

#: see the module docstring. The one deliberate departure from prod_aug26.
EXCURSION_K = 40.0

FAMILIES = {
    # ---- WAVE A: acridine MACE, extended UP ---------------------------------
    # Identity verbatim from prod_aug26's 'acr', which is itself verbatim from
    # prod0810_acridine_sg14_zp1_mace -- so this fan is directly poolable with
    # the completed 0.1/0.2/0.4/0.8 one rather than being a separate experiment.
    #
    # GRID. 0.8 is a BRIDGE arm, not a replicate: it re-runs the old fan's best
    # rung under the new excursion bar, and its job is to show the two waves are
    # comparable so the seven rungs can be read as one grid. If it does not
    # land near the old -10.00, the waves are NOT poolable and the new rungs
    # must be read on their own. Then 1.2/1.6/2.4 -- 3x above the old ceiling,
    # matching the ELJ grid's top (1.6) and one rung beyond, because the old
    # fan was still improving monotonically when it ran out of grid.
    'acr2': {
        'prior_path': f'{CLUSTER_DATA}/acridine_sg14_zp1_mace_prior_dataset.pt',
        'space_groups': [14],
        'warm_src': 'prod0810_acridine_sg14_zp1_mace',
        'energy_function': 'mace',
        'mlip_path': '/scratch/mk8347/data/acr_112025_mh1_stagetwo.model',
        'scales': [0.8, 1.2, 1.6, 2.4],
        'max_batch_size': 50000,
        'wave': 'acr2',
    },
    # ---- WAVE B: mipcas UMA, RESUMED ----------------------------------------
    # prod_aug26's mipu arms cold-started and ran the full 30k without gating,
    # ending just short (median 5.0-5.4 vs a 5.0 bar). Rather than pay for that
    # 30k again, this wave WARM-STARTS from mipu_lr0p8's own _running.pt: same
    # f047 prior, same space group, same energy_config, so assert_problem_match
    # passes and its 30k steps of progress carry over. It is the best mle on
    # that family (-26.07) and no other mipu arm is meaningfully ahead.
    #
    # NB mipu has NO phase1_exit snapshot -- the gate never fired, so on_exit
    # never ran. _running.pt is the only tagged file, which is what the phase-1
    # sbatch globs for anyway.
    #
    # Weights-only, per the fan doctrine: fresh optimizers so each arm's rate is
    # the only thing under test. GRID shifted up -- the family's own evidence is
    # that higher was better right up to the top of the old grid, and two of
    # those three arms were rate-cut by the very bar this battery loosens.
    'mipu2': {
        'prior_path': f'{CLUSTER_DATA}/mipcas_sg2_zp1_uma_f047_prior_dataset.pt',
        'space_groups': [2],
        'warm_src': 'prod26_mipu_lr0p8',
        'energy_function': 'uma',
        'mlip_path': '/scratch/mk8347/models/uma/esen_s.pt',
        'scales': [0.4, 0.8, 1.6],
        'max_batch_size': 50000,
        'wave': 'mipu2',
    },
}

#: wave -> (sbatch filename, wall clock). Each wave gets its OWN sbatch and its
#: own contiguous array range, so one can be submitted without the other.
WAVES = {
    'acr2':  ('submit_prod_aug28_acr2.sbatch', '08:00:00'),
    'mipu2': ('submit_prod_aug28_mipu2.sbatch', '12:00:00'),
}


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def scale_tag(s):
    return str(s).replace('.', 'p')


def build():
    arms = {}
    for fam, spec in FAMILIES.items():
        for s in spec['scales']:
            cfg = base()
            name = f'prod28_{fam}_lr{scale_tag(s)}'

            # -- identity + location ------------------------------------------
            cfg['run_name'] = name
            cfg['tag'] = 'prod28'
            cfg['checkpoints_dir'] = CLUSTER_CKPTS
            cfg['prior_path'] = spec['prior_path']
            # one molecule, unconditional: the condition set IS the prior file.
            # mk_dev's value is a LOCAL D:\ path that exists on the dev box, so
            # leaving it kills every arm on-cluster while passing local
            # preflight (prod_aug26, 2026-08-27).
            cfg['molecules_path'] = spec['prior_path']
            cfg['test_molecules_path'] = None
            cfg['space_groups'] = spec['space_groups']
            cfg['energy_function'] = spec['energy_function']
            cfg['mlip_path'] = spec['mlip_path']
            cfg['max_batch_size'] = spec['max_batch_size']
            cfg['checkpoint_name'] = PLACEHOLDER
            cfg['load_weights_only'] = True
            cfg['continue_from_checkpoint'] = False
            cfg['prior_model_name'] = None

            # -- run shape ----------------------------------------------------
            cfg['epochs'] = 30000            # cap; the level gate + stop ends it
            cfg['eval_period'] = 500
            cfg['figs_period'] = 1000

            # -- the fixed rate under test ------------------------------------
            lc = cfg['lr_control']
            lc['mode'] = 'fixed'
            lc['fixed_scale'] = float(s)
            lc['burn_in_steps'] = 500
            lc['burn_in_scale'] = 0.05
            lc['repeat_every'] = 0
            # THE ONE DEPARTURE FROM prod_aug26 -- see the module docstring.
            lc['hard_failure']['loss_excursion_k'] = EXCURSION_K

            # -- MLE only: single stage, level-gated, STOPS -------------------
            # Written out explicitly rather than filtered from mk_dev, so no
            # battery-relevant key arrives by omission.
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
    """Re-assert every battery property on the FINISHED dict -- a generator that
    silently drops an override writes a whole wave of wrong arms at once."""
    spec = FAMILIES[fam]
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == float(s)
    assert lc['repeat_every'] == 0
    assert lc['burn_in_steps'] == 500 and lc['burn_in_scale'] == 0.05
    # the one departure, asserted so it cannot be lost silently -- an arm that
    # quietly reverts to k=10 is exactly the arm whose label stops being true
    assert lc['hard_failure']['loss_excursion_k'] == EXCURSION_K
    # the two guards that must SURVIVE the loosening
    assert lc['hard_failure']['loss_abs'] >= 1e6
    assert lc['hard_failure']['grad_abs'] >= 1e6

    assert cfg['protocol'] == 'prod_mle'
    stages = cfg['protocols']['prod_mle']['stages']
    assert len(stages) == 1, 'phase 1 is a single stage'
    st = stages[0]
    assert st['name'] == 'train_prior'
    assert st['on_exit'] == ['snapshot:phase1_exit', 'snapshot_prior', 'stop'], \
        'arms must snapshot the exit AND yield the GPU'
    # exit must stay ONE term at index 0: streaks restore BY INDEX
    assert len(st['exit']) == 1 and st['exit'][0] == {
        'metric': 'gates/progress_done', 'above': 0.5, 'patience': 1}
    assert st['hot_lr_sensor']['action'] == 'fire', \
        'the hot sensor is what still guards a true detonation once k is loosened'
    assert st['hot_lr_sensor']['channel'] == 'bwd/mle'

    assert cfg['progress_gate']['mode'] == 'level', 'mk_dev must carry the level gate'
    bars = {m['key']: m['bar'] for m in cfg['progress_gate']['metrics']}
    assert bars == {'w1r/median': 5.0, 'w1r/worst': 10.0}, \
        'the gate bars must match prod_aug26 or the waves are not comparable'

    assert cfg['load_weights_only'] is True, 'phase-1 fans take weights only'
    assert cfg['checkpoint_name'] == PLACEHOLDER
    assert cfg['continue_from_checkpoint'] is False
    assert cfg['grow_batch_size'] is True, 'auto batch sizer is a battery property'
    assert cfg['grad_clip_guard']['enabled'] is True
    assert cfg['prior_path'] == spec['prior_path']
    assert cfg['molecules_path'] == spec['prior_path']
    assert cfg['test_molecules_path'] is None
    assert cfg['energy_function'] == spec['energy_function']
    assert cfg['mlip_path'] == spec['mlip_path']
    assert_no_local_paths(cfg)


def assert_no_local_paths(node, trail='cfg'):
    """No Windows drive path may survive into a cluster arm. mk_dev is a LOCAL
    dev config: any path key the generator forgets to override ships a value
    that exists on the dev box -- so local preflight passes -- and 404s on the
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
#SBATCH --job-name=prod28{wave}
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prod_aug28/joblogs/%x_%A_%a.out

# prod_aug28 phase-1 fan. Arm = row of INDEX.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/prod_aug28
LOGS=${{ARMS}}/joblogs
CKPTS={ckpts}
mkdir -p ${{LOGS}}

ARM=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $1}}' ${{ARMS}}/INDEX.tsv)
SRC=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $2}}' ${{ARMS}}/INDEX.tsv)
if [ -z "${{ARM}}" ]; then echo "no arm at row ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; fi
CONFIG=${{ARMS}}/${{ARM}}.yaml
if [ ! -f "${{CONFIG}}" ]; then echo "missing config ${{CONFIG}}" >&2; exit 1; fi

# WARM-START RESOLUTION. The checkpoint filename embeds a problem hash the
# generator does not recompute; resolve by run-name glob into a per-job COPY of
# the config (the repo config keeps its placeholder). LOUD on a miss: a fan arm
# silently cold-starting would corrupt the whole comparison.
#
# REFUSES AN AMBIGUOUS MATCH rather than taking the newest. Every arm in a fan
# must provably seed from the SAME file or the comparison is not a comparison,
# and `ls -t | head -1` cannot promise that when a rerun adds a second match.
J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}
RESOLVED=${{J}}.yaml
N=$(ls ${{CKPTS}}/*${{SRC}}*_running.pt 2>/dev/null | wc -l)
if [ "${{N}}" -eq 0 ]; then
    echo "FATAL: no warm checkpoint matches *${{SRC}}*_running.pt in ${{CKPTS}}" >&2
    exit 1
fi
if [ "${{N}}" -gt 1 ]; then
    echo "FATAL: ${{N}} checkpoints match *${{SRC}}*_running.pt -- ambiguous, refusing:" >&2
    ls ${{CKPTS}}/*${{SRC}}*_running.pt >&2
    exit 1
fi
CK=$(ls ${{CKPTS}}/*${{SRC}}*_running.pt)
echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  warm <- $(basename ${{CK}})"
sed "s|{placeholder}|$(basename ${{CK}})|" ${{CONFIG}} > ${{RESOLVED}}

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


def main():
    arms = build()
    HERE.mkdir(exist_ok=True)
    (HERE / 'joblogs').mkdir(exist_ok=True)
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    names = list(arms)
    wave_of = lambda n: FAMILIES[n.split('_')[1]]['wave']
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\twave\n')
        for name in names:
            spec = FAMILIES[name.split('_')[1]]
            f.write(f"{name}\t{spec['warm_src']}\t{spec['wave']}\n")

    # each wave must occupy ONE CONTIGUOUS run of rows, or its --array range
    # would silently include another wave's arms
    for wave, (fname, wall) in WAVES.items():
        rows = [i for i, n in enumerate(names) if wave_of(n) == wave]
        assert rows == list(range(rows[0], rows[-1] + 1)), \
            f'wave {wave} rows are not contiguous: {rows}'
        with (HERE / fname).open('w', encoding='utf-8', newline='\n') as f:
            f.write(SBATCH.format(first=rows[0], last=rows[-1], time=wall,
                                  wave=wave, ckpts=CLUSTER_CKPTS.rstrip('/'),
                                  placeholder=PLACEHOLDER))
        print(f'{fname}: rows {rows[0]}-{rows[-1]}  wall {wall}')

    print(f'\n{len(arms)} arms written to {HERE}')
    for name in names:
        fam = name.split('_')[1]
        print(f"  {name:<24} lr x{arms[name]['lr_control']['fixed_scale']:<6} "
              f"warm <- {FAMILIES[fam]['warm_src']}")


if __name__ == '__main__':
    main()
