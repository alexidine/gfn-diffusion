"""local_prod_sep02: a ~30 min LOCAL rehearsal of the production battery layout.

    python configs/local_prod_sep02/make.py

Everything the cluster arms will see, at local scale, so the wiring is exercised
before a real submission: frozen anchors, no anchor updates, no anchor-driven
prior top-up, fixed branch fractions, a fixed rate that does not move, and the
calibrated eLJ energy.

Seeded from dev_elj_p2_cruise's phase-1 exit (D:/crystal_datasets/gfn_checkpoints,
stage train_prior, step 2910, T=10), whose problem_def hashes to e01bd1 --
identical to today's mk_dev -- and which has its buffer sidecar. ~1.3 it/s at
batch 400 measured, so 2000 steps is ~26 min.

WHAT EACH REQUEST MAPS TO, and where it is NOT a config knob:

  frozen anchor set        buffers.anchor_buffer.frozen: true. Checked at the
                           PRIMITIVE (buffer.py:2963 admit, :3113 thin), not at
                           the five call sites, so nothing can route around it.
                           NB admit's replace path is 1-for-1, so a leak would
                           NOT move anchor_buffer_length -- judge the freeze on
                           content, never on length.

  no anchor updates        the same flag covers admit AND thin. refresh and
                           cadence-thin are additionally pinned off
                           (refresh_every_n_evals / thin_every_n_evals: 0) so an
                           inherited base cannot switch them on.

  no anchor->prior draw    TWO paths feed anchors into the prior buffer and both
                           are closed: the reach trigger
                           (anchor_buffer.reach_topup_size -> 0) and the floor
                           (prior_buffer.anchor_floor_frac -> 0). A THIRD, the
                           shortfall backfill at train.py:7295, fires only when
                           the prior-model draw under-delivers its budget and is
                           NOT disabled -- watch prior_buffer_from_anchors, which
                           must stay 0.

  fixed fractions          the equilibration stage declares `fracs` and NO
                           `balance` block. protocol.tick only calls
                           _balance_tick `if self.stage.balance is not None`, and
                           fracs are written once at the transition and never
                           touched again. This is the only way to get genuinely
                           fixed fractions -- a ratio controller with equal
                           bounds still integrates.

  fixed LR, no cut         mode: fixed + repeat_every 0, so nothing re-races the
                           rate. The hard-failure bars are left at their base
                           values ON PURPOSE -- see the note in build().
                           ⚠ fire_cut_factor would NOT hold the rate still even
                           if set to 1.0: the rewind restores lr_ctrl.scale from
                           the TARGET checkpoint before the factor applies, so it
                           can RAISE the rate. The trigger is the lever here, not
                           the response.

  hot_lr_sensor            action: report. At 'fire' it shares the rewind seat
                           with the divergence bars (train.py:6416), so a
                           drawdown would produce an intervention this run is
                           meant to be free of.

  corrected energy scaling nothing to set -- it is in the code. lj_coeff now
                           rides on the batch and is applied inside
                           compute_eLJ_energy, so `.elj` is calibrated at the
                           point it is written. Verified: composite bit-identical
                           through analyze_crystal_batch, `.elj` scaled by
                           exactly the coefficient.

  soft energy clip         ARMED, by a code change: `set_reward_clip` had ZERO
                           callers, so `energy_clip` was None on every run to
                           date and `reward_range` was an inert knob. It is now
                           called once at the end of the prior re-analysis, off
                           the unclipped prior distribution:
                               energy_clip = reward_range * T + min(E_prior)
                           `reward_range` stays at its base value -- the arm is
                           in the call site, not in a retuned number, so this rig
                           exercises the clip at exactly the setting the cluster
                           arms will inherit.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MK_DEV = HERE.parent / 'mk_dev.yaml'

CKPT_DIR = 'D:/crystal_datasets/gfn_checkpoints'
SEED_EXIT = ('dev_elj_p2_cruise_elj-mipcas_sg2_zp1_elj_prior_dataset'
             '-T2.5-e01bd1_phase1_exit.pt')
#: the SAME run's frozen prior. A *_prior.pt is a GFN -- weights, no energies --
#: so it is currency-independent and an existing one is safe to reuse under the
#: new eLJ scaling. Loading it is what makes train_prior skip.
SEED_PRIOR = ('dev_elj_p2_cruise_elj-mipcas_sg2_zp1_elj_prior_dataset'
              '-T2.5-e01bd1_prior.pt')
STEPS = 2000              # ~26 min at the measured 1.3 it/s; run STARTS at 0
BATCH = 400
EVAL_PERIOD = 200

#: production-shaped fixed mix. fwd stays small; bwd and replay split the rest.
FRACS = {'fwd': 0.05, 'bwd': 0.475, 'replay': 0.475}
LR_SCALE = 0.125          # 1.56e-5 against seed_lr 1.25e-4

# ---------------------------------------------------------------- arms 2 and 3
# Two deliberate probes of machinery the baseline arm cannot exercise, because a
# healthy run never touches either path.
#
#   lp02_hot     DIVERGENCE AND RECOVERY. Eight times the baseline rate, with
#                hot_lr_sensor.action flipped from 'report' to 'fire' so a
#                detection actually routes into the unified rewind-and-cut
#                response. The baseline deliberately reports and never moves,
#                which means it proves the sensor SEES a blow-up and proves
#                nothing about what happens next. Watch specifically whether the
#                rewind RE-INFLATES the rate: set_state_dict restores
#                lr_ctrl.scale from the TARGET checkpoint before fire_cut_factor
#                applies, so rewinding to an older, hotter checkpoint can RAISE
#                the rate rather than cut it.
#
#   lp02_resume  FULL RESTORE. The baseline runs weights-only precisely so it
#                never reads a pre-change buffer; this arm does the opposite and
#                reads the buffers the baseline just WROTE, which are the first
#                buffers in the project's history carrying a stamped lj_coeff and
#                a physical_energy computed under an armed clip. It is the only
#                test of three things at once: that assert_lj_coeff_stamped
#                passes on restored rows, that _prior_row_energy can compose a
#                restored physical_energy, and that the clip -- which is NOT
#                checkpointed and is re-derived from the prior at every startup
#                -- lands on the same value the stored rows were scored under.
#
# protocol.begin() returns early when step_ind != 0, so the resume arm does NOT
# re-run the skip chain and does NOT re-fire equilibration's on_enter. The
# restored buffer is therefore the buffer under test, not one rebuilt over it.
LR_SCALE_HOT = 1.0        # 1.25e-4, 8x the baseline
STEPS_HOT = 800

# MEASURED 2026-09-02: 8x DID NOT DIVERGE. 800 clean steps -- hot/fires_total 0,
# lr_ctrl/divergences 0, gradclip fire rate 0. It was visibly stressed but stable
# (bwd/tb 1037 vs the baseline's 544, mean sample energy +2.2 vs -6.0,
# fwd/scatter_err 55.2 vs 48.9), so the rate headroom on mipcas elj is well past
# 1.25e-4 and the recovery path went untested.
#
# 32x, and longer, because the point is an UNAMBIGUOUS divergence: a 150-step
# trial is known to overestimate the sustained rate by ~2x, so a short arm that
# merely survives proves nothing. 1500 steps also clears the 400-step window
# after promotion in which the burn-in bars are held live -- a fire inside that
# window is stale-bar noise, and this arm needs a fire that is not.
LR_SCALE_HOT2 = 4.0       # 5e-4, 32x the baseline
STEPS_HOT2 = 1500

# `epochs` is ABSOLUTE in this codebase, not a delta: the resume arm must exceed
# the baseline's 2000 or it starts already past its own budget and exits at once.
STEPS_RESUME = 2600
RESUME_FROM = ('localprod_lp02_elj-mipcas_sg2_zp1_elj_prior_dataset'
               '-T2.5-e01bd1_running.pt')

#   lp02_tight   DOES THE COMPRESSION BRANCH EVER RUN? At the shipped
#                reward_range the clip lands at 472.8 while on-policy energies
#                reach 113.7 (measured on the baseline: condition floor -152.2
#                plus a max excess of 265.9), so it fires on NOTHING and the
#                branch is never executed by any arm above. A clip that has
#                never run is not a tested clip.
#
#                reward_range 60 puts the cutoff at 60*2.5 - 152.2 = -2.2, which
#                sits between the measured P90 (-40.7) and P99 (+87.5) -- so
#                roughly the worst few percent of samples get compressed, often
#                enough to exercise the path every step and rarely enough to
#                leave the bulk of the distribution alone.
#
#                THIS IS A MECHANISM TEST, NOT A PRODUCTION SETTING. It
#                deliberately deforms the target, so its log Z and losses are not
#                comparable with the other arms. What it is asked to show is
#                narrow: the branch executes, stays finite, does not NaN, and the
#                run remains stable while a real fraction of samples ride it.
#                Read `Mean clip_active_frac` -- it is 0.0 on every other arm.
#   lp02_tighthot  DOES A LIVE CLIP ARREST THE REWIND-POISONING LOOP?
#                  hot2 (32x) fired at 303, 403, 504 -- exactly on the 100-step
#                  cooldown, every time rewinding to the SAME step-200 'last_ok',
#                  every time re-entering at scale 4 because fire_cut_factor is
#                  1.0. A fixed-point loop, except the excursions GREW: 8806 ->
#                  12530 -> 29640 against a fixed bar of 7662.
#
#                  They grow because fire_loss_spike restores WEIGHTS and
#                  deliberately keeps the LIVE BUFFERS. Clean weights, dirtier
#                  buffer, every cycle -- and loss-priority admission retains the
#                  worst rows preferentially. The clip is the only thing that
#                  bounds how bad a retained row can be, and at 472.8 it never
#                  fires, so it bounds nothing.
#
#                  This arm is hot2 with the tight clip. If the escalation
#                  flattens, the clip does the job the July notes claimed for it.
#                  If it still climbs, the poisoning is not energy-magnitude
#                  driven and the clip is the wrong lever.
#   lp02_noaccum  BATCH REGIME. batch_size 400 sits BELOW the inherited
#                 fused_grad_accum_min_samples 1000, so every fused step in the
#                 arms above was a MICRO-step and 2000 train steps were ~667
#                 optimizer updates. Dropping the threshold to the batch size
#                 makes one train step one update, which is the regime the
#                 cluster arms (larger batch) actually run in. Short: the check
#                 is that behaviour is unchanged in kind, not a new measurement.
ACCUM_NOACCUM = 400
STEPS_NOACCUM = 400

REWARD_RANGE_TIGHT = 60
# 2000, not 10000: the question is binary -- does the compression branch execute
# and stay finite -- and it is answerable in the first few hundred steps. A 10k
# arm buys distributional detail we do not need against a launch window we do.
STEPS_TIGHT = 2000


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def committed_energy_kwargs():
    """energy_config keys the COMMITTED MolecularCrystal.__init__ accepts.

    mk_dev is live and carries keys whose consumer may still be uncommitted
    (prior_flow_path, lambda_mix), and train.py does
    MolecularCrystal(**energy_config) -- so an ungated generate produces a config
    the cluster cannot construct. That killed mipu_bwd_aug31's first submit.
    """
    import ast
    import subprocess
    for path in ('energy_sampling/energies/molecular_crystal.py',
                 'energies/molecular_crystal.py'):
        try:
            src = subprocess.run(['git', 'show', f'HEAD:{path}'], capture_output=True,
                                 text=True, check=True,
                                 cwd=str(HERE.parent.parent)).stdout
        except Exception:
            continue
        if not src.strip():
            continue
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ClassDef) and node.name == 'MolecularCrystal':
                for fn in node.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == '__init__':
                        return {a.arg for a in fn.args.args} - {'self'}
    return None


def build():
    cfg = base()
    cfg['run_name'] = 'lp02'
    cfg['tag'] = 'localprod'
    cfg['checkpoints_dir'] = CKPT_DIR
    # WEIGHTS ONLY + A LOADED PRIOR. This is the combination protocol.py:1484
    # names explicitly: the loaded prior is the product of the warm-start, so
    # train_prior is skipped, but it is a sampling-only object and does NOT
    # warm-start the policy -- that comes from checkpoint_name under
    # load_weights_only, which restores the GFN weights and NOTHING else
    # (checkpointing.py:646: optimizers, schedulers, buffers, metrics,
    # condition_log_z and every MODELLER_STATE_DEFAULTS field start fresh).
    #
    # Buffers therefore begin EMPTY and seed from the prior dataset, which is
    # re-analysed at every load -- so every row is born in the calibrated eLJ
    # currency and NO migration of pre-change rows is needed anywhere.
    #
    # It also removes the frozen-prior-buffer failure: prior_model_name being in
    # the CONFIG is exactly what a requeue needs, so a resumed arm reloads its
    # sampler instead of arriving without one. That is "a stage that needs a
    # prior gets a path", not the auto-discovery train.py:2648 deleted.
    cfg['checkpoint_name'] = SEED_EXIT
    cfg['load_weights_only'] = True
    cfg['continue_from_checkpoint'] = False
    cfg['prior_model_name'] = SEED_PRIOR

    # weights-only starts at step 0, so this is the whole budget
    cfg['epochs'] = STEPS
    cfg['batch_size'] = BATCH
    cfg['max_batch_size'] = BATCH
    cfg['grow_batch_size'] = True
    cfg['batch_util_target'] = 0
    cfg['eval_period'] = EVAL_PERIOD
    cfg['eval_num_samples'] = 500
    cfg['figs_period'] = EVAL_PERIOD * 5
    cfg['traj_checkpoint'] = False           # T=10 locally; activations are cheap
    cfg['progress_gate']['level_window'] = 1000

    # ---- anchors: frozen, and no path from them into the prior buffer --------
    ab = cfg['buffers']['anchor_buffer']
    ab['frozen'] = True
    ab['refresh_every_n_evals'] = 0
    ab['thin_every_n_evals'] = 0
    ab['reach_topup_size'] = 0
    pb = cfg['buffers']['prior_buffer']
    pb['anchor_floor_frac'] = 0.0

    # ---- a rate that does not move ------------------------------------------
    lc = cfg['lr_control']
    lc['mode'] = 'fixed'
    lc['fixed_scale'] = LR_SCALE
    lc['repeat_every'] = 0
    lc['burn_in_steps'] = 100
    lc['burn_in_scale'] = 0.05
    # THE EXCURSION BARS ARE LEFT ALONE, deliberately (owner, 2026-09-02).
    # _check_bars validates only the absolute bars, so loss_excursion_k /
    # grad_excursion_x COULD be raised out of reach -- but they are the only
    # brake on a real detonation across six unattended days, and the spurious
    # cut we actually saw came from a tail-driven statistic tripping the
    # hot_lr_sensor, not from these. The fix belongs at the trigger, which is
    # why the stage sets hot_lr_sensor.action: report.
    #
    # NOTE fire_cut_factor is NOT set to 1.0 here, and setting it would not mean
    # what it looks like: fire_loss_spike calls set_state_dict, and lr_ctrl
    # (scale included) is in MODELLER_STATE_DEFAULTS, so a rewind first restores
    # whatever rate the TARGET checkpoint recorded and only then applies the
    # factor. Rewinding to an older checkpoint can therefore RAISE the rate --
    # the re-inflation that pinned acr_b's lr0p25 at 0.0125. A fire is an
    # intervention on the weights and the rate regardless of this knob.

    cfg['protocol'] = 'prod_eq'
    cfg['protocols']['prod_eq'] = {'stages': [
        {
            # SKIPPED, not shortened. With prior_model loaded this stage is
            # stepped over at step 0 without running its on_exit actions
            # (protocol.begin -> skip chain). The alternative -- loosening the
            # progress gate -- still costs max(min_history, 6*eval_period) =
            # 2000 steps at eval_period 250, and leaves a bar in the config that
            # reads like a quality standard while no longer being one.
            # The stage must still EXIST: the checkpoint carries its stage by
            # name and StageProtocol raises without it.
            'name': 'train_prior',
            'skip_if': 'prior_loaded',
            'train_mode': 'bwd',
            'bwd_sampling_mode': 'dataset',
            'flags': {'update_log_z': True, 'scramble_conditions': True},
            'loss_coeffs': {'bwd': {'mle': 1.0, 'tbc': 0.0, 'repeats': 1.0,
                                    'tb_z_source': 'persistent'}},
            'exit': [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}],
            'on_exit': ['snapshot_prior'],
        },
        {
            'name': 'equilibration',
            'train_mode': 'fused',
            'bwd_sampling_mode': 'prior',
            # report, never fire -- see the module docstring
            'hot_lr_sensor': {'channel': 'fwd/scatter_err', 'rows': 11,
                              'above': 2.0, 'action': 'report'},
            'flags': {'update_log_z': True, 'buffers_active': True,
                      'z_calibration': True},
            # ⚠ NO `bootstrap_z`. It CANNOT run on this route and killed the
            # first rehearsal at startup: `skip_if` steps over train_prior inside
            # protocol.begin(), so equilibration is entered at step 0 -- and
            # advance() runs the INCOMING stage's on_enter unconditionally. At
            # step 0 there is no eval stream and, because the seed is loaded
            # weights-only, no restored condition_log_z tracker either, so
            # _bootstrap_z has neither of its two sources and raises.
            #
            # THE TRADE, stated because it is not free. bootstrap_z overwrites
            # `flow_model.scalar` with an eval-derived log Z. Dropping it leaves
            # the scalar at the value RESTORED from the phase-1 exit -- which is
            # the MLE-phase Z, i.e. the anchor level, exactly the handoff
            # _bootstrap_z's own docstring calls an opening transient for phase 2.
            # Accepted here: phase-2 Z is a one-way slave to fwd and the z servo
            # pulls it in, and the alternative (transition at the first eval
            # instead of step 0) costs a stage of MLE training this rig exists to
            # avoid. Watch `log_Z_learned` from step 0: it opens at the phase-1
            # exit's value, and how far it travels IS the size of the transient.
            'on_enter': ['rebuild_prior_by_churn'],
            # NO `balance` KEY. protocol.tick gates _balance_tick on
            # `stage.balance is not None`, so omitting it is what makes these
            # fractions genuinely fixed rather than merely initialised.
            'fracs': dict(FRACS),
            'deactivate_threshold': 0.01,
            'loss_coeffs': {
                'fwd': {'tb': 1.0, 'freeze_policy': 1.0},
                'bwd': {'tb': 1.0, 'beta': 80},
                'replay': {'tb': 1.0, 'beta': 80},
            },
            'buffer_servo': {
                'numerator': 'replay/ema_loss_mean',
                'denominator': 'replay/birth_loss_mean',
                'bar': 0.368, 'release': 0.6, 'scale': 0.15,
                'gain': 0.05, 'relax': 0.5, 'max_step': 0.05,
                'max_boost': 8.0,
            },
        },
    ]}

    accepted = committed_energy_kwargs()
    if accepted is not None:
        for k in sorted(set(cfg['energy_config']) - accepted):
            del cfg['energy_config'][k]

    check(cfg)
    return cfg


def check(cfg, arm='base'):
    """Invariants every arm shares, plus the baseline-only ones.

    The three arms differ ONLY in what they are probing -- rate, restore path,
    horizon -- so everything else stays pinned across all of them. An assert that
    is skipped for an arm is named here rather than silently dropped.
    """
    eq = cfg['protocols']['prod_eq']['stages'][1]
    assert 'balance' not in eq, \
        'a balance block would make the fractions a controller output again'
    assert eq['fracs'] == FRACS
    assert abs(sum(FRACS.values()) - 1.0) < 1e-9
    if arm not in ('hot', 'hot2', 'tighthot'):
        # the hot arm flips this to 'fire' on purpose -- see check_hot
        assert eq['hot_lr_sensor']['action'] == 'report'

    ab = cfg['buffers']['anchor_buffer']
    assert ab['frozen'] is True
    assert ab['refresh_every_n_evals'] == 0 and ab['thin_every_n_evals'] == 0
    # replay_beta 1.0 makes the anchor draw 100% UNIFORM: _sample_indices
    # computes n_uniform = max(1, int(k * beta)) and n_weighted = k - n_uniform,
    # so beta 1.0 leaves no priority-weighted slice at all, on both the pooled
    # and the stratified path. Anchor top-up is allowed here; priority-weighted
    # anchor top-up is not, and this is the knob that decides which.
    assert ab['replay_beta'] == 1.0, (
        'anchor draw would be priority-weighted on ema_loss -- a replay-training '
        'notion used as a prior-buffer intake rule')
    assert ab['reach_topup_size'] == 0
    assert cfg['buffers']['prior_buffer']['anchor_floor_frac'] == 0.0

    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['repeat_every'] == 0
    b = base()['lr_control']['hard_failure']
    for k in ('loss_excursion_k', 'grad_excursion_x', 'loss_abs', 'grad_abs'):
        assert lc['hard_failure'][k] == b[k], (
            f'{k} was tuned away from the base: the excursion bars are the only '
            f'brake on a real detonation over an unattended run, and the false '
            f'positive we saw was a TRIGGER problem (hot_lr_sensor), not these')

    # reward_range is now LIVE (set_reward_clip is called at prior re-analysis).
    # Hold it at the base anyway: this rig is the first run in which the knob does
    # anything at all, and a retuned value would confound the arm with a setting
    # change. It must also be non-null, or the clip silently stays disarmed.
    _rr = cfg['energy_config'].get('reward_range')
    if arm not in ('tight', 'tighthot'):
        # the tight arm moves this ON PURPOSE -- see check_tight
        assert _rr == base()['energy_config']['reward_range']
    assert _rr is not None, 'reward_range: null leaves energy_clip None -- clip not armed'

    assert cfg['prior_model_name'] == SEED_PRIOR
    if arm == 'resume':
        # this arm deliberately inverts all three: it resumes off the BASELINE's
        # checkpoints, whose buffers are post-change and need no migration
        assert cfg['epochs'] == STEPS_RESUME
        assert cfg['checkpoint_name'] == RESUME_FROM
        assert cfg['load_weights_only'] is False
    else:
        if arm not in ('tighthot', 'noaccum'):
            assert cfg['epochs'] == {'hot': STEPS_HOT, 'hot2': STEPS_HOT2,
                                     'tight': STEPS_TIGHT}.get(arm, STEPS)
        assert cfg['checkpoint_name'] == SEED_EXIT
        assert cfg['load_weights_only'] is True, (
            'a full resume would restore PRE-change buffers and need a migration')
    st = cfg['protocols']['prod_eq']['stages'][0]
    assert st['skip_if'] == 'prior_loaded',         'without this the MLE stage trains instead of being skipped'

    # bootstrap_z is unreachable on a step-0 transition with a weights-only seed
    # (no eval stream, no restored tracker) -- it raises rather than degrading.
    _oe = cfg['protocols']['prod_eq']['stages'][1].get('on_enter', [])
    assert not any(str(a).startswith('bootstrap_z') for a in _oe), (
        'bootstrap_z in equilibration.on_enter dies at startup on this route: '
        'skip_if enters the stage at step 0, where it has neither eval metrics '
        'nor a visited condition_log_z tracker')
    assert 'rebuild_prior_by_churn' in _oe, (
        'the buffer rebuild is the whole point of entering this stage cleanly')
    assert cfg['batch_size'] == cfg['max_batch_size'] == BATCH
    if arm != 'noaccum':
        assert cfg['fused_grad_accum_min_samples'] == base()['fused_grad_accum_min_samples']


def make_hot(cfg):
    """Arm 2: deliberately too hot, with the fire response ARMED."""
    cfg['run_name'] = 'lp02_hot'
    cfg['epochs'] = STEPS_HOT
    cfg['lr_control']['fixed_scale'] = LR_SCALE_HOT
    # the ONE knob that separates this from the baseline's posture: a detection
    # must actuate, or the arm measures the sensor and not the recovery
    cfg['protocols']['prod_eq']['stages'][1]['hot_lr_sensor']['action'] = 'fire'
    return cfg


def make_hot2(cfg):
    """Arm 5: hot enough to actually break, after 8x did not."""
    cfg = make_hot(cfg)                      # same posture: fire armed, bars at base
    cfg['run_name'] = 'lp02_hot2'
    cfg['epochs'] = STEPS_HOT2
    cfg['lr_control']['fixed_scale'] = LR_SCALE_HOT2
    return cfg


def check_hot2(cfg):
    check_hot(cfg)                           # inherits every hot-arm invariant
    assert cfg['run_name'] == 'lp02_hot2'
    assert cfg['lr_control']['fixed_scale'] == LR_SCALE_HOT2 > LR_SCALE_HOT, (
        'hot2 exists BECAUSE the 8x arm stayed stable -- it must be hotter than that')
    assert cfg['epochs'] > STEPS_HOT, (
        'and longer: 8x survived 800 steps, so a shorter arm would prove nothing')


def make_tighthot(cfg):
    """Arm 6: the tight clip AT the rate that loops, to see if it breaks the loop."""
    cfg = make_hot2(cfg)                      # scale 4, fire armed
    cfg['run_name'] = 'lp02_tighthot'
    cfg['epochs'] = 700                       # 3 fires happened inside 504 steps
    cfg['energy_config']['reward_range'] = REWARD_RANGE_TIGHT
    return cfg


def check_tighthot(cfg):
    assert cfg['run_name'] == 'lp02_tighthot'
    assert cfg['lr_control']['fixed_scale'] == LR_SCALE_HOT2, (
        'must run at the SAME rate as hot2 or the comparison is confounded')
    assert cfg['energy_config']['reward_range'] == REWARD_RANGE_TIGHT
    assert cfg['protocols']['prod_eq']['stages'][1]['hot_lr_sensor']['action'] == 'fire'
    assert cfg['epochs'] > 504, 'hot2 needed 504 steps to show three fires'


def make_noaccum(cfg):
    """Arm 7: one train step = one optimizer update."""
    cfg['run_name'] = 'lp02_noaccum'
    cfg['epochs'] = STEPS_NOACCUM
    cfg['fused_grad_accum_min_samples'] = ACCUM_NOACCUM
    return cfg


def check_noaccum(cfg):
    assert cfg['run_name'] == 'lp02_noaccum'
    assert cfg['fused_grad_accum_min_samples'] <= cfg['batch_size'], (
        'the whole point is that the threshold no longer exceeds the batch, so '
        'a fused step stops being a micro-step')
    assert cfg['epochs'] == STEPS_NOACCUM


def make_resume(cfg):
    """Arm 3: full resume off the baseline's own checkpoints AND buffers."""
    cfg['run_name'] = 'lp02_resume'
    cfg['epochs'] = STEPS_RESUME
    cfg['continue_from_checkpoint'] = True
    cfg['load_weights_only'] = False          # the whole point: restore buffers too
    cfg['checkpoint_name'] = RESUME_FROM
    return cfg


def check_hot(cfg):
    assert cfg['run_name'].startswith('lp02_hot')
    assert cfg['lr_control']['fixed_scale'] > LR_SCALE, (
        'a hot arm must actually be hotter than the baseline')
    assert cfg['protocols']['prod_eq']['stages'][1]['hot_lr_sensor']['action'] == 'fire', (
        'with action: report the sensor observes and never actuates, so the arm '
        'would prove the detector works and nothing about the recovery')
    b = base()['lr_control']['hard_failure']
    for k in ('loss_excursion_k', 'grad_excursion_x', 'loss_abs', 'grad_abs'):
        assert cfg['lr_control']['hard_failure'][k] == b[k], (
            f'{k} moved: the excursion bars are part of what this arm is testing')


def check_resume(cfg):
    assert cfg['run_name'] == 'lp02_resume'
    assert cfg['continue_from_checkpoint'] is True
    assert cfg['load_weights_only'] is False, (
        'weights-only would skip the buffer restore, which is the entire point')
    assert cfg['checkpoint_name'] == RESUME_FROM
    assert cfg['epochs'] > STEPS, (
        'epochs is ABSOLUTE: a resume at or below the baseline budget exits '
        'immediately instead of running')
    # the clip must be derived the same way, or the arm confounds a restore
    # failure with a target change
    assert cfg['energy_config']['reward_range'] == base()['energy_config']['reward_range']


def make_tight(cfg):
    """Arm 4: a clip tight enough to actually fire."""
    cfg['run_name'] = 'lp02_tight'
    cfg['epochs'] = STEPS_TIGHT
    cfg['energy_config']['reward_range'] = REWARD_RANGE_TIGHT
    return cfg


def check_tight(cfg):
    assert cfg['run_name'] == 'lp02_tight'
    rr = cfg['energy_config']['reward_range']
    assert rr == REWARD_RANGE_TIGHT < base()['energy_config']['reward_range'], (
        'the tight arm must be TIGHTER than the base, or it fires as rarely as '
        'the baseline (i.e. never) and tests nothing')
    # the cutoff must land inside the measured on-policy range or it still never
    # fires; -152.2 is the measured condition floor, 113.7 the measured reach
    clip = rr * cfg['energy_config']['temperature'] - 152.223
    assert -152.223 < clip < 113.7, (
        f'clip would land at {clip:.1f}, outside the measured on-policy range '
        f'(-152.2 .. 113.7): it would fire on everything or on nothing')
    # the outlier bar scales with reward_range -- make sure tightening it does
    # not turn the measured 0.600 prior gap into a spurious startup raise
    assert 0.1 * rr > 0.600, (
        f'outlier bar {0.1 * rr:.2f} is below the measured prior gap 0.600 -- '
        f'this arm would raise at startup instead of training')


def main():
    cfg = build()
    check(cfg, arm='base')
    out = HERE / 'lp02.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f'{out.name}  epochs={cfg["epochs"]} batch={cfg["batch_size"]} '
          f'eval={cfg["eval_period"]} fracs={FRACS} lr_scale={LR_SCALE}')
    print(f'seed: {CKPT_DIR}/{SEED_EXIT}')

    hot = make_hot(build()); check(hot, arm='hot'); check_hot(hot)
    with (HERE / 'lp02_hot.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(hot, f, sort_keys=False, default_flow_style=False)
    print(f'lp02_hot.yaml  epochs={hot["epochs"]} lr_scale={LR_SCALE_HOT} '
          f'({LR_SCALE_HOT / LR_SCALE:.0f}x baseline) hot_lr_sensor.action=fire '
          f'-- expect divergence, then rewind+cut; WATCH for rate RE-INFLATION')

    tight = make_tight(build()); check(tight, arm='tight'); check_tight(tight)
    with (HERE / 'lp02_tight.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(tight, f, sort_keys=False, default_flow_style=False)
    _clip = REWARD_RANGE_TIGHT * tight['energy_config']['temperature'] - 152.223
    print(f'lp02_tight.yaml  epochs={tight["epochs"]} reward_range={REWARD_RANGE_TIGHT} '
          f'-> clip ~{_clip:.1f} (vs 472.8 baseline; on-policy P90 -40.7, P99 +87.5) '
          f'-- MECHANISM TEST, target deliberately deformed; read Mean clip_active_frac')

    hot2 = make_hot2(build()); check(hot2, arm='hot2'); check_hot2(hot2)
    with (HERE / 'lp02_hot2.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(hot2, f, sort_keys=False, default_flow_style=False)
    print(f'lp02_hot2.yaml  epochs={hot2["epochs"]} lr_scale={LR_SCALE_HOT2} '
          f'({LR_SCALE_HOT2 / LR_SCALE:.0f}x baseline, 4x the arm that survived) '
          f'-- 8x did NOT diverge; this one should')

    th = make_tighthot(build()); check(th, arm='tighthot'); check_tighthot(th)
    with (HERE / 'lp02_tighthot.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(th, f, sort_keys=False, default_flow_style=False)
    print(f'lp02_tighthot.yaml  epochs={th["epochs"]} lr_scale={LR_SCALE_HOT2} '
          f'reward_range={REWARD_RANGE_TIGHT} -- hot2 WITH a live clip; '
          f'compare excursion growth against hot2 (8806/12530/29640)')

    na = make_noaccum(build()); check(na, arm='noaccum'); check_noaccum(na)
    with (HERE / 'lp02_noaccum.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(na, f, sort_keys=False, default_flow_style=False)
    print(f'lp02_noaccum.yaml  epochs={na["epochs"]} batch={na["batch_size"]} '
          f'fused_grad_accum_min_samples={ACCUM_NOACCUM} (was 1000) '
          f'-- 1 train step = 1 optimizer update')

    res = make_resume(build()); check(res, arm='resume'); check_resume(res)
    with (HERE / 'lp02_resume.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(res, f, sort_keys=False, default_flow_style=False)
    print(f'lp02_resume.yaml  epochs={res["epochs"]} (absolute; baseline ends at '
          f'{STEPS}) full restore from {RESUME_FROM}')
    print(f'soft energy clip: ARMED at reward_range='
          f'{cfg["energy_config"]["reward_range"]}, T={cfg["energy_config"]["temperature"]} '
          f'-> energy_clip = reward_range*T + min(E_prior), computed at startup. '
          f'Confirm the run log prints "energy clip armed".')


if __name__ == '__main__':
    main()
