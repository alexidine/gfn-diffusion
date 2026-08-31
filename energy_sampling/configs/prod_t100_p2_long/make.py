"""prod_t100_p2_long: run the two winning ELJ arms to convergence, 5-day jobs.

    python configs/prod_t100_p2_long/make.py

THE FAN IS ANSWERED: scale 1.0 in both ELJ families. Measured at the end of
prod_t100_p2 (mipcas ~29.4k steps, nehzor ~24.9k):

  arm            E        eval r2   tb_err    scale live
  mip2_lr0p25   -122.28    0.905     9.63      0.25
  mip2_lr0p5    -123.13    0.897    10.23      0.5
  mip2_lr1p0    -123.17    0.891    10.34      1        <- PICKED
  mip2_lr2p0    -119.47    0.822    12.12      0.025    <- detonated, 40x below label
  neh2_lr0p25    -86.64    0.748    11.99      0.25
  neh2_lr0p5     -87.74    0.801    10.62      0.5
  neh2_lr1p0     -87.90    0.797    10.82      1        <- PICKED
  neh2_lr2p0     -86.40    0.728    12.11      2

1.0 wins on energy in both families and never fired in either. It also lands
inside the 0.55-1.1 band prod_t100_p2's generator predicted from two
independent routes BEFORE the fan ran, so the prediction held.

Two honest caveats on the pick, recorded because n=1:
  * on mipcas, `lr0p25` has the BEST TB fit (r2 0.905 / tb_err 9.63) and the
    WORST energy of the three survivors. The rungs trade fit against energy and
    1.0 is chosen because energy is the axis with the most headroom left.
  * on nehzor, 0.5 and 1.0 are within noise on every axis (0.801/10.62 against
    0.797/10.82, energy -87.74 against -87.90). That pick is close to a coin
    flip, broken toward 1.0 for consistency and throughput.

WHY LONG. The best mipcas ELJ result on record, `prod0810_mipcas_elj`
(9inim617, T=60), reached E -126.31 / r2 0.971 / tb_err 5.31 at 200,500 steps.
Our best is -123.17 / 0.891 / 10.34 at 29,420. The baseline dominates on every
axis at ~7x the steps, and our energy is still descending 0.5-0.9 kJ/mol per
quarter-run with no flattening. These are not converged; they are early.
NB the baseline is T=60 and these are T=100, so the sampler differs -- the
energies are comparable as a sample-quality statistic, the step counts are not
directly comparable as an equal amount of work.

THE RUN NAMES AND TAG ARE UNCHANGED FROM prod_t100_p2, DELIBERATELY. The sbatch
resumes an arm by globbing `*${ARM}_*_running.pt`, and the checkpoint stem is
`{tag}_{run_name}_{problem}`. Renaming either would miss the glob and silently
restart from the phase-1 exit, throwing away 10-15k steps of equilibration.
Only `epochs` and the sbatch wall time differ from the parent battery -- these
configs are read FROM the parent so nothing else can drift.

EPOCHS is set well past what 5 days can reach, on purpose. Equilibration is
terminal by design, so `epochs` is only an ending; making the WALL the binding
constraint means a resubmit always extends rather than stopping short. At
~8 s/step, 5 days is ~54k steps, landing mipcas near 83k and nehzor near 79k.
90000 clears both with margin.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'prod_t100_p2'

CLUSTER_CKPTS = '/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints/'
PLACEHOLDER = 'WARM_CHECKPOINT_PLACEHOLDER'

#: absolute bound; see the docstring -- the 5-day wall should bind, not this
EPOCHS = 90000

#: the winners, read straight out of the parent battery
WINNERS = ['pt100mip2_lr1p0', 'pt100neh2_lr1p0']


def build():
    arms = {}
    for name in WINNERS:
        src = PARENT / f'{name}.yaml'
        assert src.exists(), f'missing parent config {src}'
        with src.open('r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        prev = cfg['epochs']
        cfg['epochs'] = EPOCHS
        check(cfg, name, prev)
        arms[name] = cfg
    return arms


def check(cfg, name, prev_epochs):
    # THE RESUME CONTRACT. Both must match the parent or the sbatch's
    # `*${ARM}_*_running.pt` glob misses and the arm restarts from phase 1.
    assert cfg['tag'] == 'pt100p2', \
        'tag is part of the checkpoint stem; changing it orphans the resume'
    assert cfg['run_name'] == name, \
        'run_name is part of the checkpoint stem; changing it orphans the resume'

    assert cfg['epochs'] == EPOCHS
    assert cfg['epochs'] > prev_epochs, (
        f'{name}: epochs {cfg["epochs"]} must exceed the parent cap '
        f'{prev_epochs}, or the continuation runs zero iterations -- `epochs` '
        f'is ABSOLUTE and these arms resume past 24000')

    # scale 1.0 is the whole point of this directory
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and float(lc['fixed_scale']) == 1.0, \
        f'{name}: this battery continues the scale-1.0 winners only'

    assert cfg['energy_function'] == 'elj', 'ELJ families only; MLIP is uma_stab'
    assert cfg['integrator']['T'] == 100 and cfg['eval_T'] == 100
    assert cfg['load_weights_only'] is False
    assert cfg['checkpoint_name'] == PLACEHOLDER
    eq = cfg['protocols']['prod_eq']['stages'][1]
    assert eq['name'] == 'equilibration' and eq.get('exit') is None, \
        'equilibration is terminal -- epochs is the only ending'
    assert_no_local_paths(cfg)


def assert_no_local_paths(node, trail='cfg'):
    """No Windows drive path may survive into a cluster arm -- it exists on the
    dev box, so local preflight passes, and 404s on the cluster."""
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
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array={first}-{last}
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pt100p2long
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prod_t100_p2_long/joblogs/%x_%A_%a.out

# prod_t100_p2_long: the two scale-1.0 ELJ winners, run long.
# Arm = row of INDEX.tsv (line 1 is the header).
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/prod_t100_p2_long
LOGS=${{ARMS}}/joblogs
CKPTS={ckpts}
mkdir -p ${{LOGS}}

ARM=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $1}}' ${{ARMS}}/INDEX.tsv)
SRC=$(awk -F'\\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {{print $2}}' ${{ARMS}}/INDEX.tsv)
if [ -z "${{ARM}}" ]; then echo "no arm at row ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 1; fi
CONFIG=${{ARMS}}/${{ARM}}.yaml
if [ ! -f "${{CONFIG}}" ]; then echo "missing config ${{CONFIG}}" >&2; exit 1; fi

J=${{LOGS}}/${{ARM}}_${{SLURM_JOB_ID}}
RESOLVED=${{J}}.yaml

# THIS IS A CONTINUATION: the arm's OWN _running.pt is the expected seed, and
# falling back to the phase-1 exit would silently discard 10-15k steps of
# equilibration. So unlike the parent battery, a missing _running.pt is FATAL
# here rather than a cue to start over.
OWN=$(ls -t ${{CKPTS}}/*${{ARM}}_*_running.pt 2>/dev/null | head -1)
if [ -z "${{OWN}}" ]; then
    echo "FATAL: no *${{ARM}}_*_running.pt in ${{CKPTS}} -- this battery CONTINUES" >&2
    echo "       prod_t100_p2 and must not restart from the phase-1 exit." >&2
    exit 1
fi
echo "array ${{SLURM_ARRAY_TASK_ID}} -> arm ${{ARM}}  CONTINUE from $(basename ${{OWN}})"
CK=${{OWN}}
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
    (HERE / 'joblogs').mkdir(exist_ok=True)
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    names = list(arms)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tcontinues\tscale\tepochs\n')
        for name in names:
            f.write(f'{name}\tprod_t100_p2/{name}\t'
                    f"{arms[name]['lr_control']['fixed_scale']}\t"
                    f"{arms[name]['epochs']}\n")

    with (HERE / 'submit_prod_t100_p2_long.sbatch').open(
            'w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(first=0, last=len(names) - 1,
                              ckpts=CLUSTER_CKPTS.rstrip('/'),
                              placeholder=PLACEHOLDER))

    print(f'{len(arms)} arms written to {HERE}\n')
    for i, name in enumerate(names):
        c = arms[name]
        print(f"  [{i}] {name:<20} scale {c['lr_control']['fixed_scale']} "
              f"epochs {c['epochs']}  ({c['energy_function']})")
    print(f'\n  5-day wall, array 0-{len(names) - 1}. '
          f'CONTINUES prod_t100_p2 -- refuses to start without a _running.pt.')


if __name__ == '__main__':
    main()
