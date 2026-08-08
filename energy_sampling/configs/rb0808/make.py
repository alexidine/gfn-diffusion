"""
rb0808 -- weekend cluster battery. 26 arms, ~632 GPU-h of 16 x 48 = 768.

Designed 2026-08-08 off docs/to_do_rebuild.md sections 0a-0d and docs/decisions.md.
Read those first; this file only encodes the decisions, it does not argue them.

    python configs/rb0808/make.py            # write configs + launch list
    python configs/rb0808/make.py --preflight  # RUN THIS ON THE CLUSTER FIRST

=============================================================================
THE TRAP THAT KILLS EVERY ARM IF YOU GET IT WRONG
=============================================================================
get_problem_definition() stores `prior_path` as a RAW STRING, so the same
problem hashes differently per filesystem:

    local    D:\\crystal_datasets\\...     ->  573c92
    cluster  /scratch/mk8347/data/...      ->  2df5a5

load_weights_only() calls assert_problem_match(), which HARD-RAISES. So a
config carrying a local-hash checkpoint filename dies in the first seconds on
the cluster. This generator therefore NEVER hardcodes a slug: it stores only
the run PREFIX, and --preflight globs the real filename on the machine it runs
on. Run the preflight on the cluster before submitting anything.

=============================================================================
WARM START -- the only correct pattern
=============================================================================
    checkpoint_name: <same-T *_phase1_exit.pt or *_best.pt>
    load_weights_only: true
    continue_from_checkpoint: false
    reuse_prior: false          # MUST stay false

NEVER reuse_prior:true or prior_model_name -- both trigger
skip_if: prior_loaded, which skips phase 1 AND LEAVES THE POLICY RANDOM
(the prior model is sampling-only). configs/aug02/generate_configs.py carries
the same warning.

integrator.T is NOT in the problem identity, so a T=10 checkpoint LOADS into a
T=60 run -- but it is not interchangeable (user, 2026-08-08): the policy learned
per-step transitions for a specific step count. Warm starts are same-T ONLY.

=============================================================================
OTHER TRAPS ENCODED HERE
=============================================================================
* `epochs` is an ABSOLUTE step ceiling. On a resumed run the loop is
  trange(init_step, epochs+1), so an arm resuming at 14000 with epochs: 20000
  runs 6000 steps, not 20000. Every warm-started arm below sets epochs as an
  absolute figure sized from the measured throughput table.
* batch MUST float on the cluster (grow_batch_size + auto_batch_throughput_opt).
  Do not pin it -- these are A100s and the knee is not where the laptop's was.
* eval_T must equal train T.
* eval_period 500 / figs_period 1000 on cluster. NEVER 250/250.
* lrprobe/fit_*_rate are LIFETIME-CUMULATIVE (step_probe.report divides by
  sum(counts), never reset), so a late spike is diluted. Read the ramps on
  lr_opt = lrprobe/alpha_median * live lr_fused, NOT on fit_downward_rate.
* prioritise {enabled: true, kappa: 0} is NOT a null -- replay_priority_config
  returns 0.0 (not None), so the whole package engages: uniform intake,
  hazard-only purge, AND the delta_plus>0 eligibility filter. kappa=0 draws
  uniformly over the POSITIVE HALF, not over the buffer.
"""
import argparse
import copy
import glob
import os
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MK_DEV = HERE.parent / 'mk_dev.yaml'
TAG = 'rb0808'

# --- cluster paths. EDIT THESE TWO IF THE CLUSTER LAYOUT MOVED ---------------
CLUSTER_PRIOR = ('/scratch/mk8347/data/crystal_datasets/conditional/priors/'
                 'mipcas_sg2_zp1_elj_prior_dataset.pt')
CLUSTER_CKPT_DIR = ('/scratch/mk8347/projects/gfn_cond/gfn-diffusion/'
                    'energy_sampling/checkpoints/')

# --- per-T blocks -----------------------------------------------------------
# warm: run PREFIX only (filesystem-independent). --preflight resolves the slug.
# lr:   each T at its OWN best-known LR, so the four baselines are NOT an
#       LR-controlled ladder and must not be read as one.
# sph:  conservative long-run steps/hour (NOT the median -- long runs are
#       slower as batch grows and eval accumulates).
T_BLOCK = {
    10:  dict(lr=1.25e-4, sph=3500, warm='tw_july31_tw_T10_lr64',
              note='weakest warm start of the four; no tb_err/r2 recorded, EffDim 9.37'),
    25:  dict(lr=4.0e-4,  sph=1700, warm='aug02_a2_T25_lr2_tight',
              note='BEST source: tb_err 3.89, r2 0.95, EffDim 5.85. The workhorse T'),
    60:  dict(lr=4.0e-4,  sph=1250, warm='aug02_a2_T60_lr2_tight',
              note='tb_err 4.82, r2 0.99. nys7cfrt regime on current HEAD'),
    100: dict(lr=2.0e-4,  sph=950,  warm='uncond_july28_prop_T100',
              note='ONLY high-T run with healthy phase1 exit AND endpoint (EffDim 6.04/5.98). '
                   'Collapse is the predicted outcome; read EffDim before tb_err'),
}

ARMS = []          # filled by main(); (name, T, hours, priority, cfg)


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def common(cfg, T):
    """Settings every arm shares. Applied before any per-arm delta."""
    b = T_BLOCK[T]
    cfg['prior_path'] = CLUSTER_PRIOR
    cfg['molecules_path'] = CLUSTER_PRIOR
    cfg['checkpoints_dir'] = CLUSTER_CKPT_DIR

    cfg['integrator']['T'] = T
    cfg['eval_T'] = T                      # eval_T MUST equal train T
    cfg['integrator']['min_traj_length'] = T
    cfg['integrator']['max_traj_length'] = T

    # batch floats -- A100 VRAM, and the knee is not the laptop's
    cfg['grow_batch_size'] = True
    cfg['auto_batch_throughput_opt'] = True
    cfg['max_batch_size'] = 50000
    cfg['cuda_memory_fraction'] = 0.9

    cfg['eval_period'] = 500               # cluster cadence; never 250/250
    cfg['figs_period'] = 1000
    cfg['archive_period'] = 5000
    cfg['archive_buffers'] = False

    # the LR sensor rides every arm: it is instrumentation, costs ~2%, and
    # lr_opt = alpha_median * live lr_fused is comparable across every cell
    cfg['step_probe'] = {'enabled': True, 'cadence': 20, 'window': 25, 'span': 2.0}

    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        cfg[k] = b['lr']
    return cfg


def warm(cfg, T):
    """Same-T warm start. Stores the PREFIX; preflight resolves the real file."""
    cfg['_warm_prefix'] = T_BLOCK[T]['warm']      # stripped at write time
    cfg['load_weights_only'] = True
    cfg['continue_from_checkpoint'] = False
    cfg['reuse_prior'] = False                     # MUST stay false
    cfg['prior_model_name'] = None
    return cfg


def scratch(cfg):
    """Train phase 1 in-run (~13-16k steps regardless of T). Only affordable
    inside a 46h baseline; never inside a short arm."""
    cfg['checkpoint_name'] = None
    cfg['load_weights_only'] = False
    cfg['continue_from_checkpoint'] = False
    cfg['reuse_prior'] = False
    cfg['prior_model_name'] = None
    return cfg


def naive(cfg):
    """The phase-2 stage dict, for deltas."""
    return cfg['protocol']['stages'][1]


def set_fracs(cfg, fracs):
    """Change the entry fracs AND keep balance.pinned in sync.

    Stage validation rejects `pinned.fwd` disagreeing with `fracs.fwd` -- they
    are the same quantity, and mk_dev's naive stage pins fwd at 0.2. An arm that
    edits fracs and forgets the pin dies at config parse. (This exact mistake
    killed a local arm on 2026-08-07, and the config regression is what caught
    it, not the run.)"""
    st = naive(cfg)
    st['fracs'] = dict(fracs)
    bal = st.get('balance')
    if isinstance(bal, dict) and isinstance(bal.get('pinned'), dict):
        for mode in list(bal['pinned']):
            if mode in fracs:
                bal['pinned'][mode] = fracs[mode]
    return cfg


def add(name, T, hours, priority, cfg, decides):
    cfg['run_name'] = name
    cfg['tag'] = TAG
    cfg['_decides'] = decides
    ARMS.append((name, T, hours, priority, cfg))


# ===========================================================================
def main():
    # ---------------- baselines: 4 arms, 184 h ----------------------------
    # Warm-started where a same-T source exists. These are reference traces,
    # NOT an LR-controlled T ladder -- each runs at its own best-known LR.
    for T, ep, pri in ((25, 80000, 1), (60, 60000, 1), (10, 150000, 1), (100, 45000, 2)):
        c = warm(common(base(), T), T)
        c['epochs'] = ep
        add(f'base_T{T}', T, 46, pri, c,
            f'reference trace at T={T}; closes D2(a). {T_BLOCK[T]["note"]}')

    # ---------------- D30: freeze_policy x LR, 4 arms, 96 h ---------------
    # The assumption every other area rests on. base_T25 is the unfrozen@4e-4
    # cell, so only three more are needed for the 2x2 plus the dose control.
    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    naive(c)['loss_coeffs']['fwd'] = {'tb': 1.0, 'freeze_policy': 1.0}
    add('d30_frz_lr4', 25, 24, 1, c, 'frozen at optimum LR; D30 replication at 41k steps not 800')

    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    naive(c)['loss_coeffs']['fwd'] = {'tb': 1.0, 'freeze_policy': 1.0}
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'): c[k] = 2.0e-4
    add('d30_frz_lr2', 25, 24, 1, c,
        'THE DISCRIMINATOR. frz@2e-4 ~ unf@4e-4 => unfreeze and halve-LR are ONE '
        'mechanism (D30 is an LR mis-setting). frozen worse at BOTH LRs => architectural, '
        'and synthesis.md section 1 is in trouble')

    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'): c[k] = 2.0e-4
    add('d30_unf_lr2', 25, 24, 1, c, 'unfrozen half of the LR leg; without it the 2x2 is 3 cells')

    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    set_fracs(c, {'fwd': 0.05, 'bwd': 0.75, 'replay': 0.20})
    add('d30_dose', 25, 24, 2, c,
        'weight-matched DOSE control. Adam is invariant to uniform loss rescale, so total '
        'weight cancels; what changes is ADDING a gradient direction at weight 0.2. '
        'Dose-dependent => gradient-weight confound real. All-or-nothing => architectural')

    # ---------------- noise floor: 1 arm, 30 h ----------------------------
    c = warm(common(base(), 25), 25); c['epochs'] = 55000; c['seed'] = 20260808
    add('base_T25_seedB', 25, 30, 1, c,
        'THE ARM THAT MUST NEVER BE CUT. aug02 measured ~35% seed spread; everything since '
        'is n=1. On the SHARED reference cell so one replicate gives a detection threshold '
        'to the LR ladder, D30, replay, beta and Z at once. Also the first ever measurement '
        'of alpha_median seed spread -- section A4 clip(median,0.9,1.1) is mis-sized if wide')

    # ---------------- LR controller: 5 arms, 46 h -------------------------
    # section A6, the crux, never run in four batteries. Open-loop ramp with the
    # feedback cut: does alpha* turn down BEFORE the cliff?
    for T, hrs, ep in ((25, 5, 11500), (10, 3, 11500), (60, 8, 11500)):
        c = warm(common(base(), T), T)
        for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'): c[k] = 2.0e-3
        c['lr_warmup_ratio'] = 100
        c['adaptive_lr'] = dict(c['adaptive_lr']); c['adaptive_lr']['warmup_steps'] = 8000
        c['epochs'] = ep
        add(f'ramp_T{T}', T, hrs, 1, c,
            'section A6 open-loop ramp. PASS BAR: servo needs 1 window (500) + 1 doubling '
            '(1204) ~ 1700 steps of warning, so lr_opt must turn down while live LR <= 0.38x '
            'of fatal. Read lr_opt = alpha_median * live lr_fused, NOT fit_downward_rate '
            '(cumulative). Three T rungs give the cliff-vs-T shape, never produced')

    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'): c[k] = 1.0e-4
    add('hold_T25_lr1', 25, 24, 2, c,
        'TWO JOBS. (a) 4x-under rung: aug02 had 14.09 vs 4e-4 5.90 -- does that ordering '
        'survive HEAD? (b) THE EXACT-ANSWER CELL, free: LR 4x lower, nothing else changed, '
        'so alpha_median MUST rise ~4x. This is the alpha* prop 1/lr null -- the analogue of '
        'kappa=0 => ESS 1.000. Materially off 4x => every LR reading in the battery is void')

    c = warm(common(base(), 25), 25); c['epochs'] = 30000
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'): c[k] = 8.0e-4
    add('hold_T25_lr8', 25, 6, 2, c, 'past aug02 cliff; expected to abort. Cheap, and the '
        'abort STEP is the datum the ramps are validated against')

    # ---------------- replay: 5 arms, 108 h -------------------------------
    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    set_fracs(c, {'fwd': 0.2, 'bwd': 0.8, 'replay': 0.0})
    add('rep_off', 25, 24, 1, c, 'P8 arm (i) at length: is a corrector necessary at all')

    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    c['buffers']['replay_buffer']['prioritise'] = {'enabled': True, 'kappa': 0.0}
    c['archive_buffers'] = True
    add('rep_b7b', 25, 24, 1, c,
        'the B7b package (uniform intake + hazard-only purge + eligibility filter) at kappa=0. '
        'NOT a null -- kappa=0 still draws over the POSITIVE HALF only. archive_buffers on so '
        'the residual histogram is recoverable for a post-hoc 1+CV^2')

    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    c['buffers']['replay_buffer']['prioritise'] = {'enabled': True, 'kappa': 1.0}
    add('rep_kappa1', 25, 24, 2, c, 'kappa ladder at 41k steps -- the 1500-step local ladder '
        'resolved nothing and 1+CV^2 ~ 1.7 bounds the gain')

    c = warm(common(base(), 25), 25); c['epochs'] = 35000
    c['buffers']['replay_buffer'].update({'prioritise': {'enabled': True, 'kappa': 0.0},
                                          'churn_rate': 20, 'mean_residence_steps': 200,
                                          'max_size': 12000})
    add('rep_starve', 25, 18, 2, c, 'induce memorisation naturally-ish: lambda_tau should '
        'cross the derived 1/e = 0.368 bar. The no-servo control for the arm below')

    c = copy.deepcopy(c); c['epochs'] = 35000
    naive(c)['buffer_servo'] = {'numerator': 'replay/ema_loss_mean',
                                'denominator': 'replay/birth_loss_mean',
                                'bar': 0.368, 'release': 0.60, 'scale': 0.15,
                                'gain': 0.01, 'relax': 0.5, 'max_step': 0.03,
                                'max_boost': 12.0}
    add('rep_starve_servo', 25, 18, 2, c,
        'THE SERVO TEST WITH REAL AUTHORITY. Locally gain 0.05 x ~8 ticks = 1.4x churn '
        'against a 27x starve -- unprovable by arithmetic. Here: 35k steps / eval 500 = 70 '
        'ticks x max_step 0.03 = log_boost 2.1 = 8.2x churn, inside max_boost 12. The loop '
        'CAN recover within the budget, so a null result now means something')

    # ---------------- beta: 2 arms, 48 h ----------------------------------
    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    naive(c)['loss_coeffs']['bwd'] = {'tb': 1.0, 'beta': 1.0e6}
    c['adaptive_lr'] = dict(c['adaptive_lr']); c['adaptive_lr']['cut_loss_abs'] = 2.0e4
    add('beta_bq', 25, 24, 2, c,
        'bwd quadratic. Local found beta 10 > 60 > 1e6 on BOTH slopes, against B5b -- but '
        'that ladder ran entirely in the degrading (frozen, lr-auto) regime and may be void. '
        'cut_loss_abs raised because an unwinsorised bwd TB loss is legitimately ~625')

    c = copy.deepcopy(c)
    naive(c)['loss_coeffs']['bwd'] = {'tb': 1.0, 'beta': 1.0e6, 'mean_before_clip': True}
    add('beta_bq_gm', 25, 24, 3, c, 'section L8c free move: average rollouts BEFORE the clip, '
        'recovering drive lost to a Jensen gap on high-variance terminals')

    # ---------------- Z: 3 arms, 72 h -------------------------------------
    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    c['condition_log_z'] = dict(c.get('condition_log_z') or {})
    c['condition_log_z'].update({'bwd_tb_z_source': 'persistent',
                                 'replay_tb_z_source': 'persistent'})
    naive(c)['flags'] = dict(naive(c).get('flags') or {}); naive(c)['flags']['update_log_z'] = False
    add('z_track', 25, 24, 2, c,
        'TRACK log Z instead of learning it. decisions.md section 2c ranks this the biggest '
        'available swing on joint convergence and it costs one config key. If tracking wins, '
        'the fwd branch job changes entirely and much Z machinery becomes dead weight')

    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    c['z_calibration'] = dict(c['z_calibration']); c['z_calibration']['enabled'] = False
    add('zcal_off', 25, 24, 2, c, 'z_calibration is enabled in mk_dev and its value is unproven. '
        'WATCH: D29 assumes |fwd/tb_resid_clipped| < 0.5 at all times; z_cal is what enforces '
        'it. If this arm breaches, every delta-plus reading in it is off-origin')

    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    c['condition_log_z'] = dict(c.get('condition_log_z') or {})
    c['condition_log_z'].update({'bwd_tb_z_source': 'persistent', 'replay_tb_z_source': 'persistent'})
    naive(c)['flags'] = dict(naive(c).get('flags') or {}); naive(c)['flags']['update_log_z'] = False
    c['z_calibration'] = dict(c['z_calibration']); c['z_calibration']['enabled'] = False
    add('z_track_zcal_off', 25, 24, 3, c, 'the 2x2 corner: if tracking works, z_cal is redundant '
        'by construction and this is the cheapest configuration on the route')

    # ---------------- opportunistic: 2 arms, 48 h -------------------------
    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    set_fracs(c, {'fwd': 0.05, 'bwd': 0.60, 'replay': 0.35})
    add('rep_hi', 25, 24, 3, c, 'P7: replay share up, fwd down. Pairs with d30_dose to separate '
        'fwd-frac dose from replay-frac dose')

    c = warm(common(base(), 25), 25); c['epochs'] = 45000
    add('rep_ctrl_ratio', 25, 24, 3, c,
        'RESTORE mk_dev balance + min_fracs (kind: ratio). D8 landed the ratio controller and '
        'it has NEVER run long. Every other arm here runs fixed fracs deliberately, so this is '
        'the only arm that exercises it')

    write_all()


# ===========================================================================
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END
#SBATCH --array={array}
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name={jobname}

# {TAG} :: {label}
# {desc}
# index -> arm mapping is in configs/{TAG}/INDEX.tsv
module purge

# ---- paths ----
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling

# ---- config selection ----
MOL_DIR="{TAG}"
CONFIG=${{WORKDIR}}/configs/${{MOL_DIR}}/${{SLURM_ARRAY_TASK_ID}}.yaml

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
    HERE.mkdir(parents=True, exist_ok=True)
    # INDEX ORDER IS PRIORITY ORDER. With --array=0-25%16 slurm starts the low
    # indices first, so if only half the queue ever runs, the decision-relevant
    # arms are the half that ran.
    ordered = sorted(ARMS, key=lambda a: (a[3], -a[2], a[0]))
    seen, rows, total = set(), [], 0.0
    for idx, (name, T, hours, pri, cfg) in enumerate(ordered):
        assert name not in seen, f'duplicate run_name {name}'
        seen.add(name)
        assert cfg['integrator']['T'] == cfg['eval_T'], f'{name}: eval_T != train T'
        assert cfg['grow_batch_size'] is True, f'{name}: batch must float on cluster'
        assert cfg['reuse_prior'] is False, f'{name}: reuse_prior would blank the policy'
        assert cfg['prior_model_name'] is None, f'{name}: prior_model_name skips phase 1'
        assert cfg['prior_path'] == CLUSTER_PRIOR, f'{name}: wrong prior_path -> wrong hash'
        decides = ' '.join((cfg.pop('_decides', '') or '').split())
        prefix = cfg.pop('_warm_prefix', None)
        if prefix:
            # placeholder: --preflight rewrites this to the resolved filename
            cfg['checkpoint_name'] = f'@@WARM:{prefix}@@'
        # run_name stays HUMAN readable -- it is the wandb name and the
        # checkpoint prefix. Only the FILENAME is the array index.
        with (HERE / f'{idx}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        total += hours
        rows.append((idx, name, T, hours, pri, prefix or 'SCRATCH', decides))
        print(f'  {idx:>2}  P{pri}  {name:<22} T={T:<4} {hours:>4.0f}h')

    (HERE / 'INDEX.tsv').write_text(
        'index\tname\tT\thours\tpriority\twarm_from\tdecides\n' +
        '\n'.join('\t'.join(str(x) for x in r) for r in rows) + '\n', encoding='utf-8')

    # Time classes: one submit file each, all pointing at the SAME numbered
    # configs via explicit slurm index lists. Splitting by --time is what lets
    # a 3h ramp backfill instead of queueing behind a 2-day request.
    classes = [
        ('long',  '2-0:00:00', lambda h: h > 30,
         'baselines: 46h reference traces, also the checkpoint sources for the NEXT battery'),
        ('mid',   '1-4:00:00', lambda h: 12 < h <= 30,
         'the 18-30h experiment arms: D30 2x2, replay, beta, Z'),
        ('short', '0-12:00:00', lambda h: h <= 12,
         'LR ramps and the cliff probe: hours, not days'),
    ]
    made = []
    for label, tlim, pred, desc in classes:
        idxs = [r[0] for r in rows if pred(r[3])]
        if not idxs:
            continue
        (HERE / f'submit_{label}.sbatch').write_text(SBATCH_TEMPLATE.format(
            time=tlim, array=_slurm_ranges(idxs) + '%16', jobname=f'{TAG}_{label}',
            TAG=TAG, label=label, desc=desc), encoding='utf-8')
        made.append((label, tlim, _slurm_ranges(idxs), len(idxs)))

    # and the simplest possible option: everything, one array, 2-day request
    (HERE / 'submit_all.sbatch').write_text(SBATCH_TEMPLATE.format(
        time='2-0:00:00', array=f'0-{len(rows)-1}%16', jobname=f'{TAG}_all', TAG=TAG,
        label='all', desc='every arm in one array; short arms just exit early'),
        encoding='utf-8')

    print(f'\n{len(ARMS)} arms, {total:.0f} GPU-h of 768 ({100*total/768:.0f}%)')
    print(f'\nwrote {len(rows)} numbered configs + INDEX.tsv in {HERE}')
    print('submit files:')
    for label, tlim, arr, n in made:
        print(f'  submit_{label}.sbatch   --time={tlim:<11} --array={arr}%16   ({n} arms)')
    print(f'  submit_all.sbatch     --time=2-0:00:00  --array=0-{len(rows)-1}%16  (all {len(rows)})')
    print('\nNEXT: run  python configs/rb0808/make.py --preflight  ON THE CLUSTER')


def preflight():
    """Resolve every @@WARM:prefix@@ against the real checkpoints_dir, and refuse
    to proceed if any source is missing. Run this ON THE CLUSTER."""
    print(f'checkpoints_dir: {CLUSTER_CKPT_DIR}')
    if not os.path.isdir(CLUSTER_CKPT_DIR):
        print('  !! directory does not exist -- edit CLUSTER_CKPT_DIR'); return 1
    cache, bad = {}, 0
    for path in sorted(HERE.glob('*.yaml')):
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        cn = cfg.get('checkpoint_name') or ''
        if not (isinstance(cn, str) and cn.startswith('@@WARM:')):
            continue
        prefix = cn[len('@@WARM:'):-2]
        if prefix not in cache:
            # phase1_exit preferred; fall back to best. NEVER hardcode the slug --
            # prior_path is in the problem identity so it differs per filesystem.
            hits = (sorted(glob.glob(os.path.join(CLUSTER_CKPT_DIR, f'{prefix}_*_phase1_exit.pt')))
                    or sorted(glob.glob(os.path.join(CLUSTER_CKPT_DIR, f'{prefix}_*_best.pt'))))
            cache[prefix] = os.path.basename(hits[0]) if hits else None
        resolved = cache[prefix]
        if resolved is None:
            print(f'  MISSING  {path.name:<26} <- no checkpoint for prefix {prefix}')
            bad += 1
            continue
        cfg['checkpoint_name'] = resolved
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
                        encoding='utf-8')
        print(f'  ok       {path.name:<26} <- {resolved}')
    print()
    for prefix, r in cache.items():
        if r and '-2df5a5' not in r:
            print(f'  NOTE {prefix} resolved to a slug that is not the expected cluster hash '
                  f'2df5a5: {r}. Verify prior_path matches the checkpoint you want.')
    if bad:
        print(f'{bad} arm(s) have no warm-start source. Fix T_BLOCK[..]["warm"] or '
              f'switch those arms to scratch() before submitting.')
        return 1
    print('preflight OK -- every warm start resolved. Safe to submit.')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true',
                    help='resolve warm-start filenames on THIS machine (run on cluster)')
    a = ap.parse_args()
    if a.preflight:
        raise SystemExit(preflight())
    main()
