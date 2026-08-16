"""
lr_aug08/verify.py -- the two paths the v7 rewrite added that NOTHING has run.

A battery measures a quantity. These two arms measure nothing; they ask whether
code executes at all. Both are short, and both cover a path where the failure
mode is silent or catastrophic rather than merely wrong:

  v0_diverge   the divergence response. fire_loss_spike was rewired (terminal
               param dropped, on_explosion -> on_divergence, the no-rewind-target
               branch changed) and no run has ever taken it. If it raises, a real
               detonation ABORTS instead of recovering -- and the only time you
               find out is on a run you cared about.

               Forcing it needs a REAL explosion, not a lowered bar:
               _check_bars refuses a divergence bar under 1e5 precisely so a
               config cannot reintroduce a graduated tier. So this arm sets
               lr_fused: 1.0e-1 (800x the reference) and lets the policy blow
               up on its own. That is docs/to_do_rebuild.md 0b Run 4's
               "deliberate explosion" done for the new code path.

               PASS = a 'lr_ctrl DIVERGENCE' line, lr_ctrl/divergences > 0,
               peak_scale cut, and the run still going afterwards.
               Note the LR is an explicit float here, so it is NOT servo-managed
               and the peak cut cannot lower it -- max_reloads is what stops the
               run. That is the intended behaviour for a pinned-LR config and
               this arm is also the check that it reports rather than hangs.

  v1_zcalrep   z_calibration.mode: replay (_z_replay_step). Never executed. It
               takes the same Huber TB Z gradient over stored trajectories, so
               its failure mode is quiet: a wrong coefficient copy or a missing
               requires_grad and it silently trains nothing while the sensor
               keeps reporting.

               Runs WITH prioritise.enabled, because _z_replay_step raises
               without it (scored admission skims the residual tail, so Z would
               calibrate to the tail rather than the on-policy mean).

               PASS = z_cal/replay_loss present, z_cal/steps > 0, and
               fwd/tb_resid_clipped still inside D29's +-0.5.

Run both after a pair finishes -- they are ~10 minutes together.
"""
import copy
import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MK_DEV = HERE.parent / 'mk_dev.yaml'
TAG = 'lrv0808'

P2_STEPS = 2650
P2_CKPT = 'batt0807_p1_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_running.pt'
CKPT_STAGE = 'naive'
ARMS = []


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def stage2(cfg):
    return cfg['protocol']['stages'][1]


def common(cfg, budget):
    cfg['eval_period'] = 100          # short arms need to reach an eval
    cfg['figs_period'] = 100
    cfg['archive_period'] = 0
    cfg['checkpoint_read_only'] = True
    stage2(cfg)['name'] = CKPT_STAGE
    cfg['adaptive_lr']['warmup_steps'] = 50
    cfg['batch_size'] = 1000
    cfg['max_batch_size'] = 1000
    cfg['grow_batch_size'] = False
    cfg['auto_batch_throughput_opt'] = False
    cfg['cuda_memory_fraction'] = 0.45
    cfg['checkpoint_name'] = P2_CKPT
    cfg['continue_from_checkpoint'] = False
    cfg['epochs'] = P2_STEPS + budget
    return cfg


def arm(name, cfg, asks):
    cfg['run_name'] = name
    cfg['tag'] = TAG
    ARMS.append((name, cfg, ' '.join(asks.split())))


def main():
    # --- v0: force a real divergence -------------------------------------
    c = common(base(), 400)
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        c[k] = 1.0e-1                 # 800x the reference: this WILL detonate
    c['adaptive_lr']['servo']['enabled'] = False
    c['max_reloads'] = 3              # abort quickly once the path is proven
    arm('v0_diverge', c,
        'does the divergence response execute? Wants a lr_ctrl DIVERGENCE line, '
        'divergences > 0, a rewind, and the run continuing -- then an orderly '
        'UNRECOVERABLE abort at max_reloads rather than a traceback.')

    # --- v1: z_calibration on the replay buffer ---------------------------
    c = common(base(), 400)
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        c[k] = 1.25e-4
    c['adaptive_lr']['servo']['enabled'] = False   # isolate the z path
    c['z_calibration']['mode'] = 'replay'
    c['z_calibration']['threshold'] = 0.2          # lowered so the loop actually FIRES
    assert c['buffers']['replay_buffer']['prioritise']['enabled'] is True, \
        '_z_replay_step raises without uniform intake -- that assertion is the point'
    arm('v1_zcalrep', c,
        'does _z_replay_step run and train Z? Wants z_cal/replay_loss present, '
        'z_cal/steps > 0, and fwd/tb_resid_clipped still inside +-0.5. threshold '
        'is lowered to 0.2 so the tick fires within 400 steps rather than sitting '
        'in its deadband and proving nothing.')


def emit():
    main()
    seen = {}
    for i, (name, cfg, asks) in enumerate(ARMS):
        (HERE / f'v{i}.yaml').write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
            encoding='utf-8')
        c = copy.deepcopy(cfg)
        c.pop('run_name'), c.pop('epochs')
        k = json.dumps(c, sort_keys=True, default=str)
        if k in seen:
            raise SystemExit(f'DUPLICATE: {seen[k]} and {name}')
        seen[k] = name
        print(f'v{i}.yaml  {name:12s} {asks[:70]}')


if __name__ == '__main__':
    emit()
