"""
postfix_july30 -- post-periodic-fix reconnaissance: re-earn the crystal
benchmarks under corrected SDE scoring before any deep cuts.

SUBMIT ONLY AFTER THE PERIODIC-WRAP SCORING FIX IS MERGED (spec:
docs/periodic_scoring_fix.html -- P_B drift is representative-dependent, and
replay scores crossing steps off by a full period; poison ~ replay share x
crossing rate). On pre-fix code these arms are just expensive replicates of
uncond_july28.

Every arm is a FROM-SCRATCH clone of one uncond_july28 arm: byte-identical to
its parent except the from_scratch() keys, the apply_periodic_fix() keys,
tag/run_name, and (ladder arms only) the four lr_* values.

FIX-CONTRACT KEYS (apply_periodic_fix, every arm; values = mk_dev.yaml
defaults and the gfn.py constructor defaults):
  model.dplr_mask_angular: false -> true   parents predate the key's new
        contract; GFN construction ASSERTS true when DPLR runs with periodic
        dims (gfn.py), so the parents' false would refuse to build.
  model.pb_exact_reversal: true (ADDED)    exact reversal of the wrapped
        reference bridge (mixture over arrival lifts); absent in parents.
Consequence for the cross-era A/Bs: the pre/post delta is the fix PACKAGE --
periodic scoring correction + DPLR angular masking together; the assert makes
them inseparable on this problem.

RUN-HYGIENE KEYS (apply_hygiene, every arm; new keys, absent in parents):
  archive_period: 5000 + archive_buffers: true   periodic frozen checkpoint
        archives with buffer sidecars (_running.pt + _buffers.pt pairs are
        what mid-stage resumes consume; arm 0's snapshots are the next
        battery's warm-start base).
  buffers.replay_buffer.admit_reward_min: -600   hard pre-softmax exclusion
        of garbage-reward candidates (admission otherwise scores purely on
        |resid|). Matched to buffers.prior_buffer.reward_min on this problem;
        mk_dev's -200 is toy-scaled. Exclusions are logged per eval window as
        replay_buffer_reward_rejected -- expect ~0 on a healthy run, so this
        does not confound the dose-response arms unless it visibly binds. Nothing may load a pre-fix checkpoint or prior --
pre-fix weights carry the bug-shaped wall-adjacent drift -- so each arm
re-pays phase 1 (~13k steps / ~7.5 h at the pre-fix rate). The shared-
warm-start economy is deliberately given up for this one battery; later
batteries warm-start from THESE snapshots, with arm 0's phase1_exit as the
intended shared base.

All arms share the parents' seed (12345). Arms 0-5 are identical in every
phase-1-affecting key, so their phase-1 evolution matches up to GPU
nondeterminism and the phase 1 -> 2 handoff remains a controlled comparison.
Ladder arms change lr_* which phase 1 also rides, so their entry states are
their own.

  PRE-FIX RECORD NOTE (2026-07-29 evening): uncond_july28 arms 14-18 and every
  press_july29 arm were submitted but never fired before all runs were
  stopped, so the pre-fix record closes at uncond arms 0-13 plus 1xz7zd9n.
  Pre/post A/B anchors are therefore runs that actually ran: arm 0 vs
  2z5oo55f (prop_fwd20), arm 2 vs 1xz7zd9n and pre-fix arm 1 (fixed mix).
  The arm 15/18 beta questions never got pre-fix answers; arms 10-11 absorb
  them here.

  BENCHMARKS (0-2)
   0 : postfix_ref        <- 4.yaml  (prop_fwd20, decay on). New floors +
                            settled bands under the pre-fix winner config;
                            stamps the post-fix phase1_exit + prior; pre/post
                            A/B vs 2z5oo55f.
   1 : postfix_nodecay    <- 17.yaml (decay_halflife 0). Constant-LR floor
                            (pre-fix arm 17 never fired, so no pre-fix
                            counterpart exists); the matched control for a
                            rebased press_july29 Tier A and the decay A/B
                            against arm 0.
   2 : postfix_fixed_null <- 1.yaml  (lexicographic, rules []). Controller-
                            free band read (bwd/relative_under,
                            fwd/over_coverage); successor to 1xz7zd9n, the
                            calibration source of the proportional targets.
                            The targets 3.0/18.0 in the prop arms are PRE-FIX
                            numbers: if the fix moves the bands, those arms
                            degrade to their default_boost mix by design
                            while this arm reads the new bands.

  REPLAY DOSE-RESPONSE RE-ADJUDICATION (3-5)
   3 : postfix_r15  <- 3.yaml   idle replay 0.15 (held ~0.135), cap 0.25
   4 : postfix_r30  <- 11.yaml  idle replay 0.30 (held ~0.27),  cap 0.40
   5 : postfix_r50  <- 12.yaml  idle replay 0.50 (held ~0.45),  cap 0.55
  Pre-fix record: replay/tb_err pinned at 50-60 in every arm, handoff
  grad-norm spike linear in replay share (2389 / 6763 / 27878 / 47006 at
  shares 0.048 / 0.135 / 0.27 / 0.45, vs cut_grad_abs 13631), toxicity
  threshold in (0.135, 0.27]. The fix removes the fictitious crossing
  residuals; what remains is the true mixture-over-past-policies floor.
  Read: replay/tb_err level vs the fwd branch, gradnorm/s_model at the
  phase 1 -> 2 handoff, tripwire fire count.

  LR LADDER (6-9) <- 17.yaml + all four lr_* x{2,4,8,16}
   6 : postfix_lr2x    1.0e-4
   7 : postfix_lr4x    2.0e-4
   8 : postfix_lr8x    4.0e-4
   9 : postfix_lr16x   8.0e-4
  Constant LR (nodecay parent) so a detonation read is unambiguous. From
  scratch, the ladder LR applies to phase 1 too: a phase-1 detonation reads
  as the MLE ceiling, a phase-2 one as the TB ceiling (MLE's ceiling is a
  structural upper bound on TB's). Pre-fix survivals are lower bounds on the
  post-fix ceiling; pre-fix detonations condemn nothing.

  HUBER BETA (10-11) <- 15.yaml / 18.yaml -- the orphaned second-wave
  questions (never fired pre-fix)
  10 : postfix_beta20         beta 20 whole-run + the five guard rails
                              hand-scaled x2 (inherited from the parent);
                              shape-limiter test vs arm 0.
  11 : postfix_beta20_nodecay 10 + decay 0 compound, the fast-default
                              candidate; vs arm 1.
  Parent caveat carries over: tb_resid_clipped / z_grad / tracker-zerr read
  on a 2x-wider ruler vs every other arm; nat-space metrics (tb_err,
  coverage, EffDim) are unchanged. beta applies whole-run incl. phase 1, so
  like the ladder these arms own their phase-1 evolution (the matched-entry
  property holds across arms 0-5 only).

  FIX ABLATION + DOSE/REPLAY EXTENSIONS (12-15)
  12 : postfix_pb_single  <- 4.yaml + model.pb_exact_reversal: false --
       the single-image P_B term under the otherwise-fixed wrapped scoring;
       vs arm 0 this prices the mixture-over-arrival-lifts component alone
       (and its compute cost).
  13 : postfix_r06        <- 0.yaml (prop_ref): the dose-response ZERO POINT
       (fwd 0.1, idle replay 0.06) -- matched-fwd control for arms 3-5/14,
       and a bonus pre/post A/B (pre-fix arm 0 ran and has a record).
  14 : postfix_r70        <- 12.yaml + idle 0.30/0.70, cap 0.80 (entry fracs
       0.1/0.27/0.63): extends the dose curve into the band where every
       pre-fix ratcheted/gentle arm rang (replay 0.79-0.90).
  15 : postfix_fresh50    <- 4.yaml + churn_rate 200, max_residence_steps 60
       (residence ~50 steps, occupancy held at max_size): post-fix the
       replay residual should be the mixture-over-past-policies floor, and
       residence is the direct dial on mixture width. Predecessor:
       tsched_july24 ttl100/ttl1000 arms (read 2026-07-27) -- staleness
       toxic (ttl1000 worse on every axis, expired_delta stayed positive),
       but fresher-than-250 NOT established: cutting TTL below the
       occupancy cliff (max_size/churn = 200) silently halved the buffer,
       so fresh and small were perfectly confounded, and all TTL arms ran
       at bwd 0.001 (collapsing regime). This arm is the deconfounded
       version: churn raised in lockstep so occupancy holds. press_july29
       caveat carries: n_admit = min(elig.numel(), churn_rate), so verify
       replay_buffer_admitted reaches ~200 and replay_buffer_length holds
       10000; if not, cut max_size instead of raising churn further.

Order for a partial submission: 0, 1, 2, 4 first (benchmarks + the headline
dose arm), then 3, 5, 13, 14, then 12, 15, 10, 11, ladder 6-9 last. All 16
fit the 16-slot cluster maximum in one submission.

Reading guide: primary outcome is step-matched fwd/tb_err_worst AT MATCHED
EffDim, exactly as in uncond_july28. Guard rails stay 'auto' (resolver-owned)
everywhere; the hand LR base is the parents' 5e-5 at T=60.
"""

from pathlib import Path

import yaml

OUTDIR = Path(__file__).parent
PARENT_DIR = OUTDIR.parent / 'uncond_july28'
TAG = 'postfix_july30'

LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')
BASE_LR = 5.0e-05        # parents' hand LR at T=60


def load_parent(index):
    with (PARENT_DIR / f'{index}.yaml').open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def from_scratch(cfg):
    """Sever every pre-fix artifact path. checkpoint_name/prior_model_name
    load nothing; reuse_prior false so skip_if: prior_loaded cannot skip
    train_prior; continue_from_checkpoint true resumes only THIS run's own
    'running' checkpoint via find_matching (run_names are new, so nothing
    pre-fix can match) -- slurm requeue resilience for day-plus runs."""
    cfg['checkpoint_name'] = None
    cfg['continue_from_checkpoint'] = True
    cfg['reuse_prior'] = False
    cfg['prior_model_name'] = None
    return cfg


def set_lrs(cfg, lr):
    for key in LR_KEYS:
        cfg[key] = lr
    return cfg


def apply_periodic_fix(cfg):
    """Stamp the periodic-scoring-fix contract keys (mk_dev.yaml defaults).
    dplr_mask_angular true is asserted at GFN construction whenever DPLR runs
    with periodic dims; pb_exact_reversal is new with the fix and absent from
    the parents."""
    cfg['model']['dplr_mask_angular'] = True
    cfg['model']['pb_exact_reversal'] = True
    return cfg


def apply_hygiene(cfg):
    """Archive cadence + replay admission floor (new keys, absent in the
    parents; see the RUN-HYGIENE section of the module docstring)."""
    cfg['archive_period'] = 5000
    cfg['archive_buffers'] = True
    cfg['buffers']['replay_buffer']['admit_reward_min'] = -600
    return cfg


def pb_single_image(cfg):
    """Fix ablation: score/propagate P_B's single-image term instead of the
    exact arrival-lift mixture (dplr_mask_angular stays true -- asserted)."""
    cfg['model']['pb_exact_reversal'] = False
    return cfg


def set_dose(cfg, idle_replay, cap, fwd=0.1):
    """uncond_july28 prop_arm recipe: idle split sets the operating point;
    entry fracs put the bwd/replay pair's mass at the idle ratio."""
    idle_bwd = round(1.0 - idle_replay, 6)
    pair = round(1.0 - fwd, 6)
    naive = next(s for s in cfg['protocol']['stages'] if s['name'] == 'naive')
    naive['fracs'] = {'fwd': fwd,
                      'bwd': round(pair * idle_bwd, 4),
                      'replay': round(pair * idle_replay, 4)}
    naive['balance']['default_boost'] = {'bwd': idle_bwd, 'replay': idle_replay}
    naive['balance']['max_fracs'] = {'replay': cap}
    return cfg


def fresh_replay(cfg):
    """Replay residence 200 -> ~50 train steps with occupancy held:
    residence = min(max_size/churn_rate, max_residence_steps)."""
    cfg['buffers']['replay_buffer'].update(churn_rate=200, max_residence_steps=60)
    return cfg


EXTRA_BUILD = {
    12: pb_single_image,
    14: lambda c: set_dose(c, idle_replay=0.70, cap=0.80),
    15: fresh_replay,
}


# (index, run_name, parent arm, lr multiplier or None, log note)
ARMS = [
    (0, 'postfix_ref', 4, None,
     'arm 4 (prop_fwd20) from scratch: post-fix floors/bands; stamps the shared phase1_exit + prior'),
    (1, 'postfix_nodecay', 17, None,
     'arm 17 from scratch: constant-LR floor, pre/post A/B, press Tier-A control'),
    (2, 'postfix_fixed_null', 1, None,
     'arm 1 from scratch: fixed mix, rules []; controller-free band read, 1xz7zd9n successor'),
    (3, 'postfix_r15', 3, None,
     'dose-response: idle replay 0.15 (held ~0.135), cap 0.25'),
    (4, 'postfix_r30', 11, None,
     'dose-response: idle replay 0.30 (held ~0.27), cap 0.40'),
    (5, 'postfix_r50', 12, None,
     'dose-response: idle replay 0.50 (held ~0.45), cap 0.55'),
    (6, 'postfix_lr2x', 17, 2, 'LR ladder x2 = 1.0e-4, constant'),
    (7, 'postfix_lr4x', 17, 4, 'LR ladder x4 = 2.0e-4, constant'),
    (8, 'postfix_lr8x', 17, 8, 'LR ladder x8 = 4.0e-4, constant'),
    (9, 'postfix_lr16x', 17, 16, 'LR ladder x16 = 8.0e-4, constant'),
    (10, 'postfix_beta20', 15, None,
     'arm 15 from scratch: beta 20 whole-run, rails x2; 2x-ruler caveat on clipped-residual metrics'),
    (11, 'postfix_beta20_nodecay', 18, None,
     'arm 18 from scratch: beta 20 + decay 0 compound, fast-default candidate'),
    (12, 'postfix_pb_single', 4, None,
     'fix ablation: pb_exact_reversal false (single-image P_B term) vs arm 0'),
    (13, 'postfix_r06', 0, None,
     'prop_ref clone: dose-response zero point (fwd 0.1, idle replay 0.06); control for 3-5/14'),
    (14, 'postfix_r70', 12, None,
     'dose-response: idle replay 0.70 (fracs 0.1/0.27/0.63), cap 0.80'),
    (15, 'postfix_fresh50', 4, None,
     'replay residence ~50: churn_rate 200, max_residence_steps 60, occupancy held'),
]


def build_all():
    log = []
    for index, run_name, parent, mult, note in ARMS:
        cfg = apply_hygiene(apply_periodic_fix(from_scratch(load_parent(parent))))
        if mult is not None:
            set_lrs(cfg, BASE_LR * mult)
        if index in EXTRA_BUILD:
            EXTRA_BUILD[index](cfg)
        cfg['tag'] = TAG
        cfg['run_name'] = run_name

        with (OUTDIR / f'{index}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)

        naive = next(s for s in cfg['protocol']['stages'] if s['name'] == 'naive')
        log.append({'index': index, 'run_name': run_name,
                    'parent': f'uncond_july28/{parent}.yaml', 'note': note,
                    'lr_fused': cfg['lr_fused'],
                    'lr_decay_halflife': cfg['adaptive_lr']['decay_halflife_steps'],
                    'kind': naive['balance']['kind'],
                    'fracs': naive['fracs'],
                    'epochs': cfg['epochs'],
                    'from_scratch': True})

    with (OUTDIR / 'experiment_log.yaml').open('w', encoding='utf-8', newline='\n') as f:
        yaml.safe_dump(log, f, default_flow_style=False, sort_keys=False)
    return log


if __name__ == '__main__':
    for r in build_all():
        print(f"{r['index']:>2}  {r['run_name']:<20} <- {r['parent']:<24} "
              f"lr {r['lr_fused']:.1e}  decay {r['lr_decay_halflife']:<5d} "
              f"{r['kind']:<13} fwd {r['fracs']['fwd']:.2f}")
