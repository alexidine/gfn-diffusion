"""
prod0810 -- config generator for the MIPCAS/NEHZOR/acridine production runs.

    python configs/prod0810/make.py            # write configs + INDEX + sbatch
    python configs/prod0810/make.py --preflight  # verify every referenced file exists ON THE
                                                   # MACHINE THIS RUNS ON. Run on the cluster
                                                   # before submitting -- the prior/mlip paths
                                                   # below are cluster paths mirrored from this
                                                   # repo's local D:\crystal_datasets\conditional\
                                                   # priors\ tree, and that mirror has not been
                                                   # verified from here.

Reusable shape: base.yaml IN THIS DIRECTORY is the battery's whole shared config, and TARGETS
is a flat list of per-arm overrides deep-merged on top of it via the same overwrite_nested_dict
helper configs/generate_configs.py uses elsewhere in this repo. This file applies NO defaults of
its own -- every setting shared across arms is edited in base.yaml, and only the six keys in
build_config() below vary per arm. To generate a different battery, copy this directory, edit
base.yaml, and replace TARGETS.

base.yaml is a snapshot of configs/mk_dev.yaml taken 2026-08-11 with the cluster/no-warm-start
layer folded in. It is a SNAPSHOT, deliberately: mk_dev is the user's live dev config and
changes under this battery otherwise. Re-snapshot by hand to pick up mk_dev changes.

=============================================================================
WARM STARTS: THREE ARMS RESUME, FOUR STILL TRAIN PHASE 1 FROM SCRATCH  (v2)
=============================================================================
Each arm's warm start must be ITS OWN. get_problem_definition() keys the problem identity on
prior_path (among other fields), so a mipcas+elj checkpoint loaded under nehzor, mipcas+uma or
any acridine target hits assert_problem_match() and hard-raises (see configs/rb0808/make.py's
docstring for the same trap). base.yaml therefore still carries checkpoint_name: null and
write_all() asserts it -- the per-arm value comes from WARM_STARTS below, keyed by arm name.

v1 (submitted 2026-08-11) got three arms through train_prior, so those three resume from their
own phase1_exit snapshot; the other four never left phase 1 and have no exit file to resume
from. See WARM_STARTS for the per-arm evidence and for why a missing exit file must never be
papered over with _best.pt.

prior_model_name stays null on every arm: a phase1_exit resume re-enters train_prior and
replays its own exit, whose on_exit runs snapshot_prior -- so the prior is regenerated in-run
rather than side-loaded. Only a resume that starts INSIDE equilibration needs it.

=============================================================================
KNOWN GAP: acridine sg19/zp1 zp3 (Form IV) has no built prior dataset
=============================================================================
The old 4-polymorph sweep (configs/old/acridine/generate_configs.py) covered sg14/zp1, sg14/zp2,
sg9/zp2, sg19/zp3. Only the first three have an assembled
<mol>_sg<N>_zp<N>_mace_prior_dataset.pt under crystal_datasets/conditional/priors/ -- sg19/zp3
only has raw, unassembled prior_chunks/may_acridine_sg19_zp3_*.pt fragments. This generator
writes 7 arms, not 8, and skips that polymorph rather than pointing at data that doesn't exist.
Build its prior dataset first if the fourth polymorph is wanted here.

=============================================================================
UNVERIFIED: uma/mace step throughput and reward/temperature calibration
=============================================================================
Every other production/battery config in this repo (rb0808, aug02, tw_july31, ...) runs energy_
function: elj, and every measured steps/hour number in docs/decisions.md and this repo's memory
comes from elj runs. uma and mace route through an external MLIP call per energy evaluation
(mxtaltools MolecularCrystal.__init__, energies/molecular_crystal.py) that is far more expensive
than the analytic elj potential -- confirmed by batched_analyze_crystal_batch's own internal
MLIP sub-batch chunk size (1000 for uma/mace vs 10000 otherwise), independent of this config's
batch_size. There is no measured sph (steps/hour) figure for uma or mace to size an epochs
budget against, so `epochs` itself is NOT re-tuned per arm -- it stays mk_dev's own absolute
step ceiling, inherited unmodified. --time IS set explicitly per TIME_CLASSES below (user call,
2026-08-10): 7 days for MIPCAS/NEHZOR (elj+uma), 48h for acridine (mace) -- these are wall-clock
budgets, not a throughput measurement, so a run hitting its wall before `epochs` just means the
SLURM ceiling bound it first. Watch the first uma/mace arm's throughput and revisit epochs/
TIME_CLASSES once it's measured -- do not assume it matches the elj sph numbers in T_BLOCK
(configs/rb0808/make.py).

reward_range, energy_config.temperature, and lj_coeff (which despite the name is the generic
potential-scale coefficient applied to lj/qlj/elj/silu/uma/mace alike -- energies/
molecular_crystal.py) all sit in base.yaml at mk_dev's values, which were tuned against elj.
Nothing here re-tunes them for uma/mace; watch reward/energy histograms on the first run of each.
"""
import argparse
import os
from copy import deepcopy
from pathlib import Path

import yaml

HERE = Path(__file__).parent
BASE_CONFIG = HERE / 'base.yaml'   # this battery's own base -- NOT configs/mk_dev.yaml
TAG = 'prod0810'

# --- cluster paths -----------------------------------------------------------
# checkpoints_dir is NOT here: it lives in base.yaml with everything else shared.
CLUSTER_PRIOR_DIR = '/scratch/mk8347/data/crystal_datasets/conditional/priors/'
CLUSTER_UMA_MLIP = '/scratch/mk8347/models/uma/esen_s.pt'
CLUSTER_MACE_MLIP = '/scratch/mk8347/data/acr_112025_mh1_stagetwo.model'


def load_yaml(path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def overwrite_nested_dict(d1, d2):
    """Deep-merge d2 onto d1. Shared with configs/generate_configs.py's helper of the same name."""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            d1[k] = overwrite_nested_dict(d1[k], v)
        else:
            d1[k] = v
    return d1


def prior(mol, sg, zp, efunc):
    return f'{CLUSTER_PRIOR_DIR}{mol}_sg{sg}_zp{zp}_{efunc}_prior_dataset.pt'


# --- phase-1 warm starts (v2) -------------------------------------------------
# The v1 submission got three arms through train_prior; those three resume from their
# own phase1_exit snapshot instead of re-running ~8700 steps of MLE. The other four
# never left phase 1 and MUST stay null: `_best.pt` without a sibling `_phase1_exit.pt`
# is proof the exit conditions were never met, and warm-starting off one is fatal at
# step 0 with a non-finite gradient (rb0808 arm 0). Verified against wandb `phase`:
#   0 mipcas_elj  phase2@8730 | 1 mipcas_uma phase2@8740 | 2 nehzor_elj phase2@5580
#   3 nehzor_uma  max phase 1 | 4 acridine_sg14_zp1_mace max phase 1 | 5,6 no history
#
# The hashes below are the PRE-internal_oom_recovery-exemption slugs, i.e. the names
# actually on disk. Adding that key to _NON_IDENTITY_ENERGY_CONFIG_KEYS re-hashes every
# slug this config would generate today, but the compatibility check normalizes BOTH
# sides, so the old file still loads under the new config -- only the FILENAME is
# frozen at the old hash. Reconstruction validated against mipcas_uma's a29ef0, which
# appears in that run's own output.log.
#
# load_weights_only stays false: these are pre-transition snapshots carrying optimizer
# state, buffers (save(tag, with_buffers=True)) and stage_ctrl with request_eval stamped
# on, so the resumed run pulls an eval to its first step and re-fires the train_prior
# exit gate through the normal path. Weights-only would restart at step 0 in train_prior
# and throw away the thing we are trying to skip.
#
# Only triclinic arms can warm-start off a v1 snapshot. A v1 phase1_exit.pt carries no
# `dead_latent_rows`, so it matches the () that sg-1/sg-2 resolve to but NOT the (3, 5)
# of a monoclinic cell -- dead rows fix expanded_dim, so checkpointing refuses the load
# (decisions.md D33). nehzor_elj (sg 14) therefore re-runs train_prior from scratch.
WARM_STARTS = {
    'mipcas_elj': 'prod0810_mipcas_elj_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-2df5a5_phase1_exit.pt',
    'mipcas_uma': 'prod0810_mipcas_uma_uma-mipcas_sg2_zp1_uma_prior_dataset-T2.5-a29ef0_phase1_exit.pt',
}


# name -> (molecule, sg, zp, energy_function, mlip_path)
TARGETS = [
    ('mipcas_elj', 'mipcas', 2, 1, 'elj', None),
    ('mipcas_uma', 'mipcas', 2, 1, 'uma', CLUSTER_UMA_MLIP),
    ('nehzor_elj', 'nehzor', 14, 1, 'elj', None),
    ('nehzor_uma', 'nehzor', 14, 1, 'uma', CLUSTER_UMA_MLIP),
    ('acridine_sg14_zp1_mace', 'acridine', 14, 1, 'mace', CLUSTER_MACE_MLIP),
    ('acridine_sg14_zp2_mace', 'acridine', 14, 2, 'mace', CLUSTER_MACE_MLIP),
    ('acridine_sg9_zp2_mace', 'acridine', 9, 2, 'mace', CLUSTER_MACE_MLIP),
    # acridine sg19/zp3 (Form IV) intentionally excluded -- see module docstring.
]


def build_config(base, name, mol, sg, zp, efunc, mlip_path):
    """The ONLY keys that vary per arm. Everything else comes from base.yaml untouched.
    Deep-merged, so a nested override (e.g. energy_config: {temperature: ...}) can be added
    to this dict without flattening it."""
    cfg = deepcopy(base)
    p = prior(mol, sg, zp, efunc)
    overwrite_nested_dict(cfg, dict(
        prior_path=p,
        molecules_path=p,
        mlip_path=mlip_path,
        energy_function=efunc,
        space_groups=[sg],
        z_primes=[zp],
        # bare, human-readable run_name -- tag carries the battery id separately.
        # Matches configs/rb0808/make.py's add() (run_name='base_T25', tag='rb0808'),
        # not the tag-prefixed form: every current battery generator keeps these separate.
        run_name=name,
        tag=TAG,
        # null for the four arms with no phase1_exit on disk -- see WARM_STARTS
        checkpoint_name=WARM_STARTS.get(name),
        load_weights_only=False,
        max_step_seconds=MAX_STEP_SECONDS[efunc],
        traj_checkpoint=TRAJ_CHECKPOINT[efunc],
    ))
    return cfg


# --- rollout gradient checkpointing, MLIP arms only -------------------------------
# This is what makes internal_oom_recovery: false survivable. That flag means the
# whole batch goes through the MLIP in ONE call (the point: energy batch == rollout
# batch), so the MLIP's memory ceiling now sets the training batch instead of being
# hidden by the energy function's own sub-batching. On v2's uma arm at T=60 that
# ceiling was under 1000 crystals: the run OOM'd at 1000, halved to 500, OOM'd again,
# halved to 250 (oom_batch_shrink_factor 0.5), and a batch that small left the GPU at
# 48% utilization until the scheduler cancelled it for low usage.
#
# traj_checkpoint makes rollout activation memory ~O(1) in T (models/gfn.py:144) for
# one extra policy forward in backward -- ~1.7x step time, ~33x VRAM at T=100. Trading
# step time for the headroom to run a BIG batch is the right way round here: a big
# batch with slower steps keeps the GPU busy, which is what the usage policy checks.
# elj does not need it and should not pay the 1.7x -- its energy is analytic and it
# already runs at batch 1650-4491 without OOM.
TRAJ_CHECKPOINT = {'elj': False, 'uma': True, 'mace': True}


# --- per-energy RUNAWAY GUARD (not a tuning knob) ---------------------------------
# max_step_seconds does not serve the objective -- it cuts the batch, and it measures
# loop-iterations/hour rather than the samples/sec that opt-step throughput is
# proportional to. It survives only to catch the v1 pathology (181-262 s steps at 15
# samples/s), so it is set FAR above the operating point on purpose: elj runs 2-4 s
# steps, uma ran 3.7-67 s, both with wide margin. It also stands itself down for a
# stage if a cut fails to move step time, since a fixed per-step cost is not something
# the batch can fix.
MAX_STEP_SECONDS = {'elj': 60, 'uma': 300, 'mace': 300}

# --- GPU occupancy: MEASURED, NOT CONTROLLED --------------------------------------
# The cluster cancels a job whose GPU utilization averages under 60% for ~2 h, and
# prod0810's uma arm (4r351oqm) died that way at 5.2 h. These arms used to carry
# gpu_util_floor: 70, which grew the batch whenever the trailing mean fell under it.
# That key is RETIRED (utils._RETIRED_KEYS) and setting it now hard-fails preflight.
#
# It was removed because its premise is false here. umaperf0812/c_controller, whose
# every growth was floor-driven: batch 100->741 took utilization 52->42% and
# samples/sec 57.7->24.3. Occupancy does not rise with batch once the MLIP dominates
# the step, so the floor overrode a throughput gate that would have refused all four
# jumps, and cost 58% of throughput for nothing. It was inert on these very arms
# besides -- a 900 s window cannot fill from a per-10-step sample at 200 s/step.
#
# Occupancy is still logged (gpu/util_recent, gpu/util_policy, now sampled on a
# wall-clock cadence so slow arms report too). If an arm is cancelled for low usage
# the levers are work per kernel launch and unpaired host stalls, NOT batch size.


# --- SLURM time classes ------------------------------------------------------
# MIPCAS/NEHZOR (elj+uma) run longer; acridine (mace) is capped at 48h. Sizes are
# the user's explicit call, not a throughput measurement -- see docstring above.
TIME_CLASSES = [
    ('mipcas_nehzor', '7-00:00:00', 8, lambda mol: mol in ('mipcas', 'nehzor')),
    ('acridine',      '2-00:00:00', 8, lambda mol: mol == 'acridine'),
]

# --- cpus-per-task 8: kept, but it is NOT the fix ---------------------------------
# Raised from 1 on the theory that the low-util/low-power signature was host-bound
# and single-threaded. Host-bound is right; single-threaded-because-of-CPU-count is
# NOT. The hot paths are serial Python `for` loops over graphs -- one AtomicData
# object built per crystal in mlip_interfaces/uma_utils.py, then collated one at a
# time, ~52k scalar extractions and ~29k CUDA syncs per 1000-crystal call, done TWICE
# (crystal leg + gas-phase leg). A second core cannot enter a serial loop, so extra
# CPUs buy nothing here.
#
# Left at 8 anyway: CPUs are near-free beside an A100 and it costs only queue time.
# The real lever is vectorising that construction (measured 720 ms -> 3.1 ms at
# B=1000, bit-identical output) and caching the gas-phase leg, which is recomputed
# every call for a quantity that depends only on molecule identity.


def _slurm_ranges(idxs):
    """Compact a sorted index list into slurm array syntax: 0-3,7,11-13."""
    out, i = [], 0
    while i < len(idxs):
        j = i
        while j + 1 < len(idxs) and idxs[j + 1] == idxs[j] + 1:
            j += 1
        out.append(str(idxs[i]) if j == i else f'{idxs[i]}-{idxs[j]}')
        i = j + 1
    return ','.join(out)


def write_all():
    assert BASE_CONFIG.is_file(), (
        f'{BASE_CONFIG} not found. This battery builds from its OWN base, not configs/mk_dev.yaml '
        f'-- re-snapshot it with:  cp configs/mk_dev.yaml {BASE_CONFIG}  then re-apply the '
        f'cluster/no-warm-start edits described in that file\'s header.')
    base = load_yaml(BASE_CONFIG)
    # base.yaml owns everything shared, so a stale local path there would silently ship to
    # all 7 arms. Catch the ones a mk_dev re-snapshot would reintroduce.
    assert base['checkpoint_name'] is None and base['prior_model_name'] is None, \
        'base.yaml carries a warm start -- see WHY EVERY ARM HERE TRAINS PHASE 1 FROM SCRATCH'
    assert str(base['checkpoints_dir']).startswith('/'), \
        f"base.yaml checkpoints_dir is not a cluster path: {base['checkpoints_dir']}"
    assert base['eval_period'] == 500 and base['figs_period'] == 1000, \
        'base.yaml must carry the cluster eval/figs cadence (500/1000), never 250/250'
    HERE.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_names = set()
    for idx, (name, mol, sg, zp, efunc, mlip_path) in enumerate(TARGETS):
        assert name not in seen_names, f'duplicate target name {name}'
        seen_names.add(name)
        cfg = build_config(base, name, mol, sg, zp, efunc, mlip_path)

        assert cfg['checkpoint_name'] == WARM_STARTS.get(name), \
            f'{name}: checkpoint_name must come from WARM_STARTS (or be null)'
        if cfg['checkpoint_name'] is not None:
            # a warm start that silently falls back to training from scratch wastes a
            # week of cluster time before anyone notices; one that silently loads the
            # WRONG arm's weights is worse. Both are visible right here.
            assert cfg['checkpoint_name'].startswith(f'{TAG}_{name}_'), \
                (f'{name}: warm start {cfg["checkpoint_name"]} belongs to another arm -- '
                 f'the problem identity would reject it at load, but not before burning '
                 f'the queue slot')
            assert cfg['load_weights_only'] is False, \
                f'{name}: a phase1_exit resume needs full state, not weights-only'
        assert (cfg['mlip_path'] is None) == (efunc == 'elj'), \
            f'{name}: mlip_path must be null for elj and set for uma/mace'

        with (HERE / f'{idx}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        rows.append((idx, name, mol, f'sg{sg}_zp{zp}', efunc))
        print(f'  {idx}  {name:<26} {efunc:<5} sg{sg} zp{zp}')

    (HERE / 'INDEX.tsv').write_text(
        'index\tname\tmolecule\tpolymorph\tenergy_function\n' +
        '\n'.join('\t'.join(str(x) for x in r) for r in rows) + '\n', encoding='utf-8')

    unclassed = [r[0] for r in rows if not any(pred(r[2]) for _, _, _, pred in TIME_CLASSES)]
    assert not unclassed, f'arms with no TIME_CLASSES match: {unclassed}'
    made = []
    for label, tlim, cpus, pred in TIME_CLASSES:
        idxs = [r[0] for r in rows if pred(r[2])]
        if not idxs:
            continue
        (HERE / f'submit_{label}.sbatch').write_text(SBATCH_TEMPLATE.format(
            time=tlim, array=_slurm_ranges(idxs), label=label, cpus=cpus), encoding='utf-8')
        made.append((label, tlim, _slurm_ranges(idxs), len(idxs), cpus))

    print(f'\nwrote {len(rows)} configs + INDEX.tsv in {HERE}  (base: {BASE_CONFIG.name})')
    print('acridine sg19/zp3 (Form IV) excluded -- no assembled prior dataset (see docstring)')
    print('uma/mace epochs are an UNMEASURED guess inherited from base.yaml -- see docstring')
    print('submit files:')
    for label, tlim, arr, n, cpus in made:
        print(f'  submit_{label}.sbatch   --time={tlim:<11} --array={arr}   '
              f'({n} arms, {cpus} cpus)')
    print('\nNEXT: run  python configs/prod0810/make.py --preflight  ON THE CLUSTER')


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task={cpus}
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END
#SBATCH --array={array}
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=prod0810_{label}

# prod0810 :: {label}
# index -> arm mapping is in configs/prod0810/INDEX.tsv
module purge

# ---- paths ----
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling

# ---- config selection ----
CONFIG=${{WORKDIR}}/configs/prod0810/${{SLURM_ARRAY_TASK_ID}}.yaml

# ---- run ----
srun singularity exec --nv \\
    --overlay ${{OVERLAY}}:ro \\
    --bind ${{PROJECT_ROOT}}:${{PROJECT_ROOT}} \\
    --pwd ${{WORKDIR}} \\
    ${{IMAGE}} \\
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${{PROJECT_ROOT}}/MXtalTools:${{PROJECT_ROOT}}/gfn-diffusion:\\$PYTHONPATH
        python train.py --config ${{CONFIG}}
    "
"""


def arm_configs():
    """The generated per-arm yamls -- base.yaml is not one of them."""
    return sorted(HERE.glob('[0-9]*.yaml'), key=lambda p: int(p.stem))


def preflight():
    """Verify every checkpoints_dir/prior_path/molecules_path/mlip_path this battery references
    actually exists ON THIS MACHINE. Run on the cluster: the paths assume the local
    D:\\crystal_datasets\\conditional\\priors\\ tree was mirrored under CLUSTER_PRIOR_DIR,
    which has not been checked from the machine that generated these configs."""
    paths = arm_configs()
    if not paths:
        print(f'no arm configs in {HERE} -- run make.py first'); return 1
    ckpt_dir = yaml.safe_load(paths[0].read_text(encoding='utf-8'))['checkpoints_dir']
    print(f'checkpoints_dir: {ckpt_dir}')
    if not os.path.isdir(ckpt_dir):
        print('  !! directory does not exist -- edit checkpoints_dir in base.yaml, re-run make.py')
        return 1
    bad = 0
    for path in paths:
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        for key in ('prior_path', 'molecules_path', 'mlip_path'):
            val = cfg.get(key)
            if val and not os.path.isfile(val):
                print(f'  MISSING  {path.name:<12} {key} -> {val}')
                bad += 1
        # warm starts are resolved relative to checkpoints_dir by train.py's init_gfn.
        # HARD FAIL rather than warn: rb0808 lost a battery to a preflight that let a
        # missing exit file fall back to _best.pt, and the fallback was itself the
        # warning nobody read. A missing sidecar is equally fatal -- the resumed run
        # would rebuild empty buffers and re-derive the stage from nothing.
        ckpt = cfg.get('checkpoint_name')
        if ckpt:
            full = os.path.join(ckpt_dir, ckpt)
            if not os.path.isfile(full):
                print(f'  MISSING  {path.name:<12} checkpoint_name -> {full}')
                bad += 1
            sidecar = full.replace('.pt', '_buffers.pt')
            if not os.path.isfile(sidecar):
                print(f'  MISSING  {path.name:<12} buffer sidecar   -> {sidecar}')
                bad += 1
    if bad:
        print(f'\n{bad} missing file(s). Fix CLUSTER_PRIOR_DIR/CLUSTER_UMA_MLIP/CLUSTER_MACE_MLIP '
              f'or WARM_STARTS above, or the cluster mirror, then re-run make.py before '
              f'submitting. Do NOT null out a warm start to make this pass without checking '
              f'why the file is absent -- an arm that never reached phase 2 has no exit '
              f'snapshot by construction.')
        return 1
    print('preflight OK -- every referenced file exists. Safe to submit.')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true',
                    help='verify referenced files exist ON THIS MACHINE (run on the cluster)')
    a = ap.parse_args()
    if a.preflight:
        raise SystemExit(preflight())
    write_all()
