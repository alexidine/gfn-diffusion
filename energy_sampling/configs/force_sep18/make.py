"""force_sep18 -- reshaped terminal-force ladder on the forward seat, LOCAL (RTX 5080),
P_B frozen, off the pg14_p1_cont MLE run's BEST checkpoint (step 80050, still in the
train_prior stage -- it never met the phase-1 exit gate -- T=10, batch 1000).

WHY. dose_sep16's two force arms lost after an early gain. Read against their controls
and against per-sample measurements on the frozen mip checkpoint (2026-09-18): the
truncated terminal-force path lowers the residual by RELOCATING samples -- over-dense
rows slide downhill (search, the early gain), under-dense rows slide uphill away from
states that should gain probability (the loss keeps improving while the backward
sensors on fixed prior rows degrade). 30% of the raw push was that flight and ~1% of
rows (clash forces 100x the median) set its direction; the shipped reward_grad_clip
clips d loss/d log R, which the Huber already bounds, so it was inert. Scored offline,
gating the term to residuals above ~1 nat and clipping |F| per row put ~75% of the push
on rows where descent is the right move (raw: ~25%).

ARMS (fwd_loss_coeffs; everything else identical):
  ctrl_pb   k 0, reward_grads 0          the density route alone
  raw_pb    k 1, reward_grads 1          the dose16_fs_k1 form (clip inert), P_B frozen
  gc_pb     k 1, gate 1.0, |F| clip 250  reshaped: descent only, tail bounded
  gcs_pb    gc + path_grad_scale 0       ... and the noise-scale channel detached
  g0c_pb    k 1, gate 0.0, |F| clip 250  the gate threshold's sensitivity
  g_pb      k 1, gate 1.0, no clip       added 14:10 after gc_pb: the clip cost half the search gain
                                          and all of the clash-tail cleanup; this isolates gate from clip
  bad_pb    NO force; prior buffer rebuilt from ANCHORS ONLY at the stage transition, anchor noise
            0.04-0.08 latent (mk_dev: 0.003-0.03, prior-model source), prior-buffer energy window
            ramp_floor 1000 / ramp_width 500 (mk_dev: 100 / 50). Owner's absorption hypothesis
            (2026-09-18): high-energy rows are the negative examples MLE lacks; does the still-
            absorbing sampler absorb faster with them in the backward branch? Read against ctrl_pb
            on the FORWARD oracles (Z, eval_fwd/tb_err, excess-energy P90/P99); every bwd/* metric
            averages over this arm's own harder rows and is not comparable.
  anch_pb   NO force; prior buffer from ANCHORS ONLY at the DEFAULT noise (0.003-0.03) and the
            default window (100 / 50). The control bad_pb lacked: differs from bad_pb in the noise
            and the window only, and from ctrl_pb in the source only (owner, 2026-09-18).
  bad2_pb   bad_pb with the noise raised again: 0.06-0.13 latent, window unchanged (1000 / 500), so it
            differs from bad_pb in the noise only. NB the ramp gates admission/expiry and weights the
            under-coverage METRIC; the backward LOSS weights every admitted row equally (owner asked
            2026-09-18). Rows above Emin + 1000 are refused at admission.
  therm_pb  anch_pb with the anchor jitter SHAPED: buffers.anchor_buffer.tile 'shaped', so each drawn
            anchor is redrawn from N(x_min, V) -- x_min the anchor's relaxed minimum and V built from
            the per-anchor Hessian eigenpairs -- instead of the isotropic noise_log_range ball around
            the stored anchor. tile_temperature 1.0, tile_width_cap at mk_dev's 0.15. Source 'anchors'
            and the default window (100 / 50), so it differs from anch_pb in the tile only.
            READS D:\\crystal_datasets\\gfn_checkpoints\\anchor_shapes_e01bd1.pt (format_version 1:
            x_min, evals, evecs per anchor, plus a sha1 of anchor_buffer.x the loader must match).
  therm05_pb therm_pb at tile_temperature 0.5, i.e. every width scaled by 1/sqrt(2); differs from
            therm_pb in that number only.
  therm2_pb  therm_pb on the v2 sidecar (anchor_shapes_e01bd1_v2.pt: 5 saddle-free Newton iterations at trust 0.04
            instead of 2 at 0.1, so the relaxed centres carry a smaller residual gradient) with tile_width_cap 0.08
            (v1: 0.15) on the flat and capped directions; tile_temperature 1.0. Differs from therm_pb in the sidecar
            and the cap only.
  therm2_05_pb  therm2_pb at tile_temperature 0.5.
  smoke_therm_pb  therm_pb at 60 fused steps -- the mechanics check, same shape as smoke_gc_pb.

SEAT. Rollout every step, fwd trains the policy at a pinned 0.3 share, bwd 0.5 /
replay 0.2 with the ramp frozen; constant LR (burn_in_scale == fixed_scale); P_B
frozen by the top-level key (snapshotted from the MLE weights at load). RESUME
MECHANICS: the seed is a train_prior-stage frame, so that stage's exit gate
(gates/progress_done) is replaced by an always-true term; at the first eval (250 steps
of MLE at the fixed rate) the run exits train_prior (snapshot:phase1_exit and
snapshot_prior fire), enters equilibration (rebuild_prior_by_churn, bootstrap_z) and the
forward seat takes over. The ELJ energy runs under MXT_LEGACY_TRICLINIC_WALLS=1 at
launch (the seed predates the Niggli walls).

READ. Z and eval_fwd/tb_err at matched steps (the gain); bwd/tb_err, bwd/tb_resid,
bwd/under_coverage and zmatch/delta_mean (the damage sensors -- forward improving while
these degrade is the relocation fingerprint); rewardgrad/gate_frac, force_clipped_frac,
force_norm_max, push_norm_mean; Excess Energy P50/P90/P99 at eval.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MK_DEV = HERE.parent / 'mk_dev.yaml'
TAG = 'fs18'
SEED_FILE = 'pg14_p1_cont_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-e01bd1_best.pt'
RESUME_STEP = 80050
STUB_STEPS = 250   # one eval period of MLE before the always-true exit fires
STEPS = 3000
FWD_SHARE = 0.3
FORCE_CLIP = 250.0
SHAPE_PATH = 'D:\\crystal_datasets\\gfn_checkpoints\\anchor_shapes_e01bd1.pt'
SHAPE_PATH_V2 = 'D:\\crystal_datasets\\gfn_checkpoints\\anchor_shapes_e01bd1_v2.pt'   # 5 Newton iterations, trust 0.04


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def fwd_seat(cfg):
    stages = cfg['protocols']['unconditional_tb']['stages']
    assert [s['name'] for s in stages] == ['train_prior', 'equilibration'], [s['name'] for s in stages]
    tp = stages[0]
    assert tp['exit'] == [{'metric': 'gates/progress_done', 'above': 0.5, 'patience': 1}], tp['exit']
    tp['exit'] = [{'metric': 'bwd/mle', 'above': -1.0e9, 'patience': 1}]
    eq = stages[1]
    eq['fwd_rollout_every'] = 0
    eq['z_pin_rollout_every'] = 0
    eq.pop('fwd_rollout_triggers', None)
    eq['fracs'] = {'fwd': FWD_SHARE, 'bwd': 0.5, 'replay': 0.2}
    eq['balance']['pinned'] = {'fwd': FWD_SHARE}
    eq['balance']['up'] = 1.0e-4
    eq['balance']['down'] = 1.0e-4
    eq['balance']['bounds'] = {'bwd': [0.25, 0.6], 'replay': [0.1, 0.45]}
    eq['loss_coeffs']['fwd'] = {'tb': 1.0, 'freeze_policy': 0.0}
    return cfg


def common(cfg, name, steps=STEPS):
    cfg['run_name'] = name
    cfg['tag'] = TAG
    cfg['cuda_memory_fraction'] = 0.75
    cfg['epochs'] = RESUME_STEP + STUB_STEPS + steps
    cfg['eval_period'] = 250
    cfg['eval_num_samples'] = 2500
    cfg['figs_period'] = 1000
    cfg['archive_period'] = 0
    cfg['freeze_backward_policy'] = 'full'
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed'
    lc['burn_in_scale'] = lc['fixed_scale']
    lc['repeat_every'] = 0
    lc['fire_cut_factor'] = 1.0
    # the seed's stored problem_def predates energy_config.physical_energy_clip; absent = the code default
    cfg['energy_config'].pop('physical_energy_clip', None)
    assert cfg['checkpoint_name'].startswith('dev_mk_dev_') and cfg['checkpoint_name'].endswith('_phase1_exit.pt')
    cfg['checkpoint_name'] = SEED_FILE   # a _best.pt: full resume, stage train_prior
    assert cfg['load_weights_only'] is False and cfg['continue_from_checkpoint'] is False
    assert int(cfg['integrator']['T']) == 10, cfg['integrator']['T']
    return cfg


def bad_buffer(cfg, lo=-1.4, hi=-1.1, floor=1000.0, width=500.0):
    # lo/hi: log10 of the isotropic anchor-noise radius range in latent units
    pb = cfg['buffers']['prior_buffer']
    pb['source'] = 'anchors'
    pb['ramp_floor'] = float(floor)
    pb['ramp_width'] = float(width)
    cfg['buffers']['anchor_buffer']['noise_log_range'] = [float(lo), float(hi)]
    assert cfg['buffers']['anchor_buffer']['frozen'] is True
    return cfg


def shaped_tile(cfg, temperature, path=None, cap=None):
    # the anchor jitter becomes a per-anchor Gaussian read from the sidecar, centred on
    # each anchor's stored relaxed minimum; noise_log_range is unread under this tile
    ab = cfg['buffers']['anchor_buffer']
    ab['tile'] = 'shaped'
    ab['shape_path'] = SHAPE_PATH if path is None else path
    ab['tile_temperature'] = float(temperature)
    assert ab['tile_width_cap'] == 0.15, ab['tile_width_cap']
    if cap is not None:
        ab['tile_width_cap'] = float(cap)
    return cfg


def arm(name, k, rg, gate=None, force_clip=0.0, scale=1, steps=STEPS, bad=False, anchors=False,
        noise=None, tile_t=None, tile_path=None, tile_cap=None):
    cfg = common(fwd_seat(base()), name, steps=steps)
    if bad:
        cfg = bad_buffer(cfg, *(noise or (-1.4, -1.1)))
    elif anchors:
        cfg['buffers']['prior_buffer']['source'] = 'anchors'
    else:
        # the six force arms and ctrl_pb ran BEFORE mk_dev's default moved to 'anchors' (2026-09-18):
        # keep them on the prior model so the set stays internally comparable
        cfg['buffers']['prior_buffer']['source'] = 'prior_model'
    if tile_t is not None:
        cfg = shaped_tile(cfg, tile_t, path=tile_path, cap=tile_cap)
    fc = cfg['fwd_loss_coeffs']
    fc['path_grad_last_k'] = int(k)
    fc['reward_grads'] = float(rg)
    fc['reward_grad_clip'] = 0.0
    fc['reward_grad_gate'] = None if gate is None else float(gate)
    fc['reward_grad_force_clip'] = float(force_clip)
    fc['path_grad_scale'] = int(scale)
    fc['traj_grads'] = 0.0
    out = HERE / f'{name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f'wrote {out.name:16s} k={k} reward_grads={rg} gate={gate} force_clip={force_clip} '
          f'path_grad_scale={scale} steps={steps} tile={cfg["buffers"]["anchor_buffer"]["tile"]}'
          f'{"" if tile_t is None else f"@{tile_t}"}')
    return out


if __name__ == '__main__':
    arm('ctrl_pb', 0, 0.0)
    arm('raw_pb', 1, 1.0)
    arm('gc_pb', 1, 1.0, gate=1.0, force_clip=FORCE_CLIP)
    arm('gcs_pb', 1, 1.0, gate=1.0, force_clip=FORCE_CLIP, scale=0)
    arm('g0c_pb', 1, 1.0, gate=0.0, force_clip=FORCE_CLIP)
    arm('g_pb', 1, 1.0, gate=1.0, force_clip=0.0)
    arm('bad_pb', 0, 0.0, bad=True)
    arm('anch_pb', 0, 0.0, anchors=True)
    arm('bad2_pb', 0, 0.0, bad=True, noise=(-1.2, -0.9))
    arm('therm_pb', 0, 0.0, anchors=True, tile_t=1.0)
    arm('therm05_pb', 0, 0.0, anchors=True, tile_t=0.5)
    arm('therm2_pb', 0, 0.0, anchors=True, tile_t=1.0, tile_path=SHAPE_PATH_V2, tile_cap=0.08)
    arm('therm2_05_pb', 0, 0.0, anchors=True, tile_t=0.5, tile_path=SHAPE_PATH_V2, tile_cap=0.08)
    arm('smoke_gc_pb', 1, 1.0, gate=1.0, force_clip=FORCE_CLIP, scale=0, steps=60)   # crosses the stub exit at +250, then 60 fused steps
    arm('smoke_therm_pb', 0, 0.0, anchors=True, tile_t=1.0, steps=60)
