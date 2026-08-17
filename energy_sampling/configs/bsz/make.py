"""
bsz -- measure the CRITICAL BATCH SIZE with the ray calibrator.

THE QUANTITY. For a step at batch B the rate maximising expected progress is

    eta*(B) = |g|^2 / (g'Hg + tr(H*Sigma_1)/B) = eta*_inf * B / (B + B_crit)

so B_crit = tr(H*Sigma_1)/(g'Hg) is where gradient noise stops dominating.
Below it, batch buys step size roughly linearly; above it, batch buys nothing.

Since alpha* = eta*/eta, measuring alpha* at two batch sizes at the SAME fixed
learning rate gives B_crit directly:

    r = alpha*(B/2) / alpha*(B)        B_crit = B * (1 - r) / (2r - 1)

    r ~ 1.00  ->  B_crit ~ 0     batch is free to cut
    r ~ 0.75  ->  B_crit ~ B/2   modest headroom
    r ~ 0.60  ->  B_crit ~ 2B    already noise-limited
    r -> 0.50 ->  noise-dominated, do not cut

DESIGN NOTES, each of which is load-bearing:

  - LRs are EXPLICIT FLOATS, identical in both arms. So nothing is servo-managed,
    the calibration runs and logs but actuates nothing, and alpha* is a clean
    measurement with no feedback loop closed around it.
  - n_sub is doubled in the half-batch arm (8 -> 16) so that BOTH arms probe on
    the same 8000 total samples. The probe's sub-batches are drawn at
    `batch_size`, so without this the small-batch arm would have a noisier
    instrument as well as a noisier gradient, and the two would be confounded.
  - warmup_steps is dropped to 10. The calibration is held through warmup (it
    cannot rate a deliberately-shrunken step), and the default 1000 would let
    the arms diverge for 1000 steps before the first reading.
  - grow_batch_size off and accum_min pinned to batch_size: the batch under test
    must be the batch actually used, with no growth and no accumulation.
  - Read the FIRST calibration most heavily -- by construction both arms start
    from one checkpoint, so the earliest reading is the closest to a matched
    theta. Later readings drift apart because the arms see data at different
    rates.

Resumed from _phase1_exit.pt, the FRESH lineage -- production trains phase 1
in-run, so that is the state these numbers should describe.
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', 'mk_dev.yaml')
CKPT = 'dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_phase1_exit.pt'
LR = 1.25e-4
PROBE_SAMPLES = 8000     # held equal across arms


def build():
    arms = []
    for batch in (1000, 500):
        with open(BASE) as f:
            cfg = yaml.safe_load(f)
        cfg['run_name'] = f'b{batch}'
        cfg['tag'] = 'bsz'
        cfg['checkpoint_name'] = CKPT
        cfg['continue_from_checkpoint'] = False
        cfg['load_weights_only'] = False
        cfg['checkpoint_read_only'] = True
        cfg['epochs'] = 1900               # phase1_exit sits at ~430 -> ~1470 steps
        cfg['eval_period'] = 500
        cfg['figs_period'] = 1000

        cfg['batch_size'] = batch
        cfg['max_batch_size'] = batch
        cfg['grow_batch_size'] = False
        cfg['fused_grad_accum_min_samples'] = batch   # no accumulation either arm

        for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
            cfg[k] = LR                                # explicit -> unmanaged
        cfg['adaptive_lr']['warmup_steps'] = 10
        cfg['adaptive_lr']['ray_calibration']['period'] = 200
        cfg['adaptive_lr']['ray_calibration']['n_sub'] = PROBE_SAMPLES // batch

        # THE PROBE IS THIS BATTERY'S INSTRUMENT, and it now arms only because a
        # stage asks for it (train.py::_ray_askers) -- the `ray_calibration.
        # enabled: True` that used to say so in the file is retired. Nothing in
        # this generator sets that declaration, so it is inherited from mk_dev
        # and can be dropped there by an edit that has nothing to do with bsz.
        # An inert probe logs no raycal/alpha_star at all and the battery would
        # simply have no reading, so it is asserted rather than assumed.
        askers = [st['name'] for st in cfg['protocols'][cfg['protocol']]['stages']
                  if (st.get('lr_sensor') or {}).get('kind') == 'ray']
        assert askers, (
            f"b{batch}: no stage in protocol {cfg['protocol']!r} declares "
            f"lr_sensor kind 'ray', so the calibrator is inert and alpha* -- the "
            f"only quantity this battery measures -- is never logged.")

        arms.append((f'b{batch}', cfg))

    for name, c in arms:
        p = os.path.join(HERE, f'{name}.yaml')
        with open(p, 'w') as f:
            yaml.safe_dump(c, f, sort_keys=False, default_flow_style=False)
        rc = c['adaptive_lr']['ray_calibration']
        print(f"wrote {p}")
        print(f"   batch {c['batch_size']:>5}  n_sub {rc['n_sub']:>3}  "
              f"probe samples {c['batch_size'] * rc['n_sub']}  "
              f"lr {c['lr_fused']:.3g} fixed  period {rc['period']}")
    print()
    print('read raycal/alpha_star, earliest calibration first;'
          '  B_crit = 1000 * (1 - r) / (2r - 1),  r = alpha*(500)/alpha*(1000)')
    return [n for n, _ in arms]


if __name__ == '__main__':
    build()
