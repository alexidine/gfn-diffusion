r"""cont.py -- write a long continuation of one cond_lam_sep21 arm, LOCAL.

    python configs/cond_lam_sep21/cont.py lam0p05_f --epochs 30000

The cl21c lambda=0.05 pair was cut off at step 1950 (force) / 2050 (control) when the
session that launched them ended: no _final.pt, and a half-written _running.pt.tmp on the
control. This resumes one arm from its own rolling checkpoint into a NEW run prefix, so the
cut-short state stays on disk and the overnight leg is a separate W&B run.

WHY AN EXPLICIT checkpoint_name AND NOT continue_from_checkpoint. train.py::Modeller.init_gfn
takes the `checkpoint_name` branch FIRST and, with load_weights_only false, calls
Checkpointer.load_full -- optimizers, buffers, step_ind, stage and the `pb_frozen` snapshot.
`continue_from_checkpoint` instead auto-finds THIS run's own prefix, which a renamed leg does
not have. The rolling buffer sidecar is read from the loaded checkpoint's prefix, so the
source arm's <prefix>_buffers.pt is what comes back.

P_B STAYS FROZEN ACROSS THE RESUME, and not because on_enter re-fires -- it does not, since a
resume restores the stage rather than entering it. `pb_frozen` is saved beside the model state
(checkpointing.py) and restored by set_pb_freeze('full', source_state=...) on every load path,
so the resumed leg scores P_B on the SAME snapshot rather than re-snapshotting a drifted
trunk. Verified present in the source checkpoint before writing.

Archives are re-enabled here (the 3k probe had archive_period 0): a multi-hour leg wants
intermediate checkpoints it can be read from without stopping it.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
CKPT_DIR = 'D:\\crystal_datasets\\gfn_checkpoints'
SRC_TAG = 'cl21c'      # the tag the source arm ran under


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm', help="arm name as generated, e.g. 'lam0p05_f'")
    ap.add_argument('--epochs', type=int, required=True, help='absolute final step, not a delta')
    ap.add_argument('--from-tag', default=SRC_TAG)
    ap.add_argument('--archive-period', type=int, default=5000)
    ap.add_argument('--name', default=None, help='output arm name (default <arm>_cont)')
    ap.add_argument('--with-stack', action='store_true',
                    help='record python stacks in the trace window. MEASURED 2026-09-22: on '
                         'this box close() then grew the host working set ~2.2 GB/min with no '
                         'progress and had to be killed. Leave it off unless you need source '
                         'attribution and can spare the RAM.')
    ap.add_argument('--trace-steps', type=int, default=4)
    ap.add_argument('--ckpt', default=None,
                    help='basename of an EXPLICIT source checkpoint in CKPT_DIR, e.g. '
                         '<prefix>_step20000.pt. Prefer this over the default _running.pt '
                         'glob: a rolling checkpoint is rewritten by any run under the same '
                         'prefix, so a leg that names one cannot say later what it resumed.')
    ap.add_argument('--fwd-repeats', type=int, default=None,
                    help='fwd rollouts per drawn condition on the var_conditioning stage. '
                         'With a fixed batch this does not add energy calls -- it trades '
                         'conditions-per-step for rows-per-condition, which is the pooled '
                         'VarGrad group size.')
    ap.add_argument('--bwd-block-m', type=int, default=None,
                    help='distinct buffer rows per condition on the backward branch. Move '
                         'this WITH --fwd-repeats: pooled lambda_b is the buffer share of '
                         'the group, so changing one alone shifts the mixture as well as '
                         'the group size and the arm answers two questions at once.')
    ap.add_argument('--profile', action='store_true',
                    help='region timers ON plus a bounded torch.profiler window shortly '
                         'after the restored step. The only instrumented region is the '
                         'energy call (energies/base_set.py), so the region layer validates '
                         'energy/seconds_in_step -- which its own module says never '
                         'synchronises -- while the trace window is what subdivides the '
                         'REST of the step.')
    a = ap.parse_args()

    base = HERE / f'{a.arm}.yaml'
    assert base.exists(), f'no such arm config: {base}'
    cfg = yaml.safe_load(base.open('r', encoding='utf-8'))

    if a.ckpt:
        src = os.path.join(CKPT_DIR, a.ckpt)
        assert os.path.exists(src), f'no such checkpoint: {src}'
    else:
        pat = os.path.join(CKPT_DIR, f'{a.from_tag}_{a.arm}_*_running.pt')
        hits = sorted(glob.glob(pat))
        assert len(hits) == 1, f'expected one rolling checkpoint matching {pat}, found {hits}'
        src = hits[0]
    # the rolling sidecar hangs off the run PREFIX, so strip whatever suffix src carries
    _pfx = src
    for _suf in ('_running.pt', '_last_ok.pt', '_best.pt', '_final.pt'):
        if _pfx.endswith(_suf):
            _pfx = _pfx[:-len(_suf)]
            break
    else:
        _pfx = _pfx[:_pfx.rindex('_step')] if '_step' in _pfx else _pfx[:-3]
    side = _pfx + '_buffers.pt'
    assert os.path.exists(side), f'{src}: no rolling buffer sidecar {side} beside it'

    import torch
    ck = torch.load(src, map_location='cpu', weights_only=False)
    ms = ck.get('modeller_state') or {}
    step = int(ms.get('step_ind', -1))
    assert int(ck['train_T']) == int(cfg['integrator']['T']), \
        f"checkpoint train_T {ck['train_T']} != config T {cfg['integrator']['T']}"
    assert ck['problem_hash'] == cfg.get('_problem_hash', ck['problem_hash'])
    # THE ZERO-STEP TRAP: a resume runs trange(init_step, epochs + 1), so an epochs at or
    # below the restored step silently executes nothing and verifies nothing.
    assert a.epochs > step + 100, \
        f'epochs {a.epochs} leaves only {a.epochs - step} steps from the restored step {step}'
    # P_B: assert the snapshot is really there rather than trusting the load path.
    frozen_declared = 'freeze_pb' in (
        [s for s in cfg['protocols'][cfg['protocol']]['stages']
         if s['name'] == 'var_conditioning'][0].get('on_enter', []))
    has_snapshot = ck.get('pb_frozen') is not None
    assert frozen_declared == has_snapshot, (
        f'config declares freeze_pb={frozen_declared} but the checkpoint carries '
        f'pb_frozen={has_snapshot}; a resume would change which P_B is scored')

    cfg['run_name'] = a.name or f'{a.arm}_cont'
    cfg['checkpoint_name'] = os.path.basename(src)
    cfg['load_weights_only'] = False        # full resume: optimizers, buffers, step, stage
    cfg['continue_from_checkpoint'] = False  # checkpoint_name wins anyway; stated to be explicit
    cfg['epochs'] = int(a.epochs)
    cfg['archive_period'] = int(a.archive_period)
    cfg['archive_buffers'] = False           # the rolling sidecar is enough; archives are ~1 GB each
    cfg['cuda_memory_fraction'] = 0.8        # runs alone

    # POOLED GROUP SIZE. The pooled term groups by condition_id across both branches, so the
    # group is (fwd rows for c) + (bwd rows for c). fwd.repeats sets the first, bwd's
    # condition_block_m the second. Written on the var_conditioning STAGE, which overrides
    # the top-level block for the stage that actually trains the policy; the top-level copy
    # is kept in step so a reader of either sees one answer.
    stage = [st for st in cfg['protocols'][cfg['protocol']]['stages']
             if st['name'] == 'var_conditioning'][0]
    grp = {}
    if a.fwd_repeats is not None:
        grp['fwd'] = ('repeats', float(a.fwd_repeats))
    if a.bwd_block_m is not None:
        grp['bwd'] = ('condition_block_m', float(a.bwd_block_m))
    for branch, (key, val) in grp.items():
        assert branch in stage['loss_coeffs'], f'var_conditioning has no {branch} block'
        assert key in stage['loss_coeffs'][branch],             f'{key} absent from the stage {branch} block: an added key is a DEAD key here, '             f'because absence means the code default, not this value'
        stage['loss_coeffs'][branch][key] = val
        if branch in cfg.get('loss_coeffs', {}) and key in cfg['loss_coeffs'][branch]:
            cfg['loss_coeffs'][branch][key] = val
    if grp:
        # the pooled term is what consumes these; an arm that raises them with the term off
        # would look like a group-size test and be measuring nothing.
        assert float(stage['loss_coeffs']['fwd'].get('pooled_vg', 0)) > 0,             'pooled_vg is off on var_conditioning: group size would reach no live term'
        assert float(stage['fracs'].get('replay', 0)) == 0.0,             'replay carries weight; its condition_block_m would also need to move'

    if a.profile:
        cfg['profiling'] = {
            'enabled': True,
            'regions': None,
            'trace': {
                'enabled': True,
                # past the restore and any reload transient, but early enough to read today
                'start_step': step + 200,
                'active_steps': int(a.trace_steps),
                'outdir': 'profiling_results',
                'write_trace': True,
                'record_shapes': True,
                'with_stack': bool(a.with_stack),
            },
        }

    name = a.name or f'{a.arm}_cont'
    out = HERE / f'{name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    check = yaml.safe_load(out.open('r', encoding='utf-8'))
    cstage = [st for st in check['protocols'][check['protocol']]['stages']
              if st['name'] == 'var_conditioning'][0]
    got_r = float(cstage['loss_coeffs']['fwd']['repeats'])
    got_m = float(cstage['loss_coeffs']['bwd']['condition_block_m'])
    if a.fwd_repeats is not None:
        assert got_r == float(a.fwd_repeats), (got_r, a.fwd_repeats)
    if a.bwd_block_m is not None:
        assert got_m == float(a.bwd_block_m), (got_m, a.bwd_block_m)
    print(f'wrote {out.name}')
    print(f'  var_conditioning group: fwd.repeats {got_r:g} + bwd.condition_block_m {got_m:g} '
          f'-> pooled group {got_r + got_m:g} at batch {check["batch_size"]} '
          f'({int(check["batch_size"] / max(got_r, 1))} conditions/step)')
    print(f'  resumes {os.path.basename(src)} at step {step}, P_B snapshot '
          f'{"present" if has_snapshot else "absent"}, freeze_pb declared {frozen_declared}')
    print(f'  runs to epochs {a.epochs} -> {a.epochs - step} steps, archives every {a.archive_period}')


if __name__ == '__main__':
    sys.exit(main())
