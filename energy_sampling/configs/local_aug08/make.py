"""
local_aug08 -- T=10 laptop battery, 2026-08-08. Two arms at a time, 1-2 h each.

WHY THIS EXISTS. On 2026-08-08 the rb0808 cluster battery was found to carry
`freeze_policy: 1.0` on ALL 26 arms: mk_dev's naive stage ships it, and the two
cells meant to be unfrozen were written by OMISSION. So the freeze axis was
never varied, and D30 -- the discriminator that `decisions.md` says every other
area assumes an answer to -- could not be answered by the battery that was
supposed to answer it. The generator is fixed (configs/rb0808/make.py, with an
assert_distinct guard), but those arms need a resubmission and a weekend.

The laptop can answer the same question TODAY at T=10. That is this battery.

READ PAIR A BEFORE LAUNCHING C OR D. Every arm after pair A is unfrozen, and
the resume checkpoint was trained FROZEN -- if unfreezing at T=10 is unstable
off this checkpoint, pairs C and D would measure the instability, not their own
variable. Pair A is the gate.

SIZING (measured 2026-08-07, see local_aug07/make.py::paired). Two arms share
the GPU at batch 1000 / cuda_memory_fraction 0.45. Paired throughput is ~1.0
step/s at T=10, so budget ~= seconds: 3600 steps ~ 1 h, 7200 ~ 2 h. Batch 2831
is a SOLO size -- one arm at 2831 reserves 14.7 of 16.3 GB.

Everything here writes batt0808_* prefixes and runs checkpoint_read_only, so it
is read-only with respect to every checkpoint on disk.
"""
import copy
import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MK_DEV = HERE.parent / 'mk_dev.yaml'
TAG = 'batt0808'

# Phase-2 resume, POST log-Z transient. Measured in local_aug07: log_Z_learned
# settles by ~25% of phase 2 and tb_resid_clipped is inside D29's +-0.5 after
# it, while alpha_median is still moving at 75%. Anything read before this step
# is reading the transient (and delta_plus is not shift-invariant, B8).
P2_STEPS = 2650
P2_CKPT = f'batt0807_p1_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_running.pt'

PAIR_BATCH = 1000
ARMS = []


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def naive(cfg):
    return cfg['protocol']['stages'][1]


def local(cfg):
    cfg['eval_period'] = 250
    cfg['figs_period'] = 1000        # must be a multiple of eval_period
    cfg['archive_period'] = 0        # throwaway arms
    cfg['checkpoint_read_only'] = True
    cfg['step_probe'] = {'enabled': True, 'cadence': 20, 'window': 25}
    return cfg


def paired(cfg):
    """Size so TWO arms share the GPU. See local_aug07/make.py::paired.

    max_batch_size is pinned EQUAL to batch_size: an OOM cut in one arm and not
    the other silently gives the two different lambda*tau (B7a derives it as
    ~B/rate), which would confound any paired comparison. Watch for 'OOM' in the
    logs -- if it fires, the pair is void, not merely noisy.
    """
    cfg['batch_size'] = PAIR_BATCH
    cfg['max_batch_size'] = PAIR_BATCH
    cfg['grow_batch_size'] = False
    cfg['auto_batch_throughput_opt'] = False
    cfg['cuda_memory_fraction'] = 0.45
    return cfg


def resume(cfg, budget):
    """Pin the resume point EXPLICITLY.

    mk_dev defaults to continue_from_checkpoint: true + checkpoint_name: null,
    which resolves to {tag}_{run_name}_{problem}_running.pt. run_name is unique
    per arm so that file never exists -- a generator that forgets this does not
    chain arms, it silently RETRAINS PHASE 1 in every one, which is invisible in
    the results and costs the whole day.

    `epochs` is an ABSOLUTE ceiling on a resumed run (the loop is
    trange(init_step, epochs+1)), not a budget -- so it must be P2_STEPS + N.
    """
    cfg['checkpoint_name'] = P2_CKPT
    cfg['continue_from_checkpoint'] = False   # checkpoint_name takes precedence
    cfg['reuse_prior'] = False                # or skip_if: prior_loaded blanks the policy
    cfg['epochs'] = P2_STEPS + budget
    assert cfg['checkpoint_name'] == P2_CKPT, 'resume point not pinned'
    assert cfg['continue_from_checkpoint'] is False, 'would resolve to _running.pt'
    assert cfg['reuse_prior'] is False, 'would auto-reload a stale prior'
    assert budget > 0, 'zero budget runs no steps and verifies nothing'
    return cfg


def unfreeze(cfg):
    """fwd trains the POLICY, not only Z.

    mk_dev ships `fwd: {tb: 1.0, freeze_policy: 1.0}`, so FROZEN is the
    inherited default and unfrozen requires explicit removal. This is the
    omission that flattened the rb0808 D30 block.
    gflownet_losses.py:157 reads `freeze_policy > 0.5`, so pop == set 0.
    """
    naive(cfg)['loss_coeffs']['fwd'].pop('freeze_policy', None)
    return cfg


def arm(name, pair, budget, cfg, asks):
    cfg['run_name'] = name
    cfg['tag'] = TAG
    ARMS.append((name, pair, budget, cfg, ' '.join(asks.split())))


# ===========================================================================
def main():
    # ---- PAIR A: D30 at T=10. The cell rb0808 cannot currently answer. ----
    # base_T25 (frozen) vs d30_unf_lr4 (unfrozen) is the same contrast at T=25,
    # but it needs a resubmission and ~24 h. This is it in an hour, at the T
    # where the local measurement of 2026-08-07 (800 steps, one seed) was taken
    # -- so it is also that measurement's first replication at 4.5x the length.
    c = paired(local(base()))
    arm('a_frz', 'A', 3600, resume(c, 3600),
        'D30 frozen cell at T=10. mk_dev as-shipped. The control.')

    c = unfreeze(paired(local(base())))
    arm('a_unf', 'A', 3600, resume(c, 3600),
        'D30 unfrozen cell at T=10. If fwd/tb_err falls here and rises in a_frz, the '
        '2026-08-07 result replicates at 4.5x length and synthesis.md section 1s thesis '
        '-- policy trained entirely off-policy, fwd trains only Z -- is in trouble.')

    # ---- PAIR B: the noise floor, ON THE CELL BEING COMPARED. -------------
    # aug02 measured ~35% seed spread and everything local since is n=1. Both
    # arms resume from the SAME checkpoint, so this measures forward stochastic
    # divergence only -- which is exactly the right null here, because every arm
    # in this battery shares that resume. Any A-gap smaller than the B-gap is
    # "not resolvable", not an effect.
    for nm, fn in (('b_frz_seedB', lambda x: x), ('b_unf_seedB', unfreeze)):
        c = fn(paired(local(base())))
        c['seed'] = 20260808
        arm(nm, 'B', 3600, resume(c, 3600),
            'seed replicate of the pair-A cell of the same name. Gives the detection '
            'threshold for A, C and D at once, and the first local read of alpha_median '
            'seed spread -- A4s clip(median, 0.9, 1.1) is mis-sized if that is wide.')

    # ---- PAIR C: can REPLAY WEIGHT do the fwd policy gradient's job? ------
    # REPOINTED (was z_track / zcal_off -- both are rb0808 indices 17/18 and
    # will be answered there; this question is live and they are not).
    #
    # The claim under test (user, 2026-08-08): "by our design, replay should
    # generate a superior forward policy grad to actual forward training, so if
    # fwd policy grads are net helpful they would be better replaced by higher
    # weight replay steps or higher LR." Pair D tests the LR half. This is the
    # replay half, and it is a DOSE experiment, not an on/off one.
    #
    # Two things forced the design:
    #
    # 1. THE CONTROLLER MUST BE OFF. mk_dev's balance is kind: ratio with fwd
    #    PINNED at 0.2 and bwd<->replay traded. Measured over pair A, the two
    #    arms therefore realised DIFFERENT mixes -- a_frz ended bwd 0.514 /
    #    replay 0.286, a_unf bwd 0.562 / replay 0.238. fwd was 0.200 in both, so
    #    the freeze contrast itself is clean, but a replay DOSE cannot be run
    #    against a controller that reallocates replay in response to the very
    #    metrics being read. balance: None pins the entry mix.
    #
    # 2. The dose must bracket what the controller actually chose. a_frz drifted
    #    to replay 0.286 on its own; 0.2 -> 0.4 brackets that and doubles the
    #    weight, which is the strongest form of the claim that is still a mix
    #    (fracs are LOSS WEIGHTS, so 0.4 IS double the force).
    #
    # PREDICTION -- stated on bwd/tb_err, and NOT on r2. r2 here is
    #   r2 = 1 - sum(resid^2) / sum((y - ybar)^2),  y = log_pf + log_Z
    # so its denominator is the BATCH'S OWN DIVERSITY (utils.py:1418). Derived
    # as sigma_y = tb_err / sqrt(1 - r2), the replay batch runs sigma_y ~20.2
    # against fwd's ~10.3 -- 4x the variance -- across all four A/B arms. So
    # replay's r2 = +0.39 vs fwd's -2.72 is mostly the denominator: standardise
    # fwd's residual on replay's sigma_y and it scores +0.04, not -2.72. Any
    # cross-branch r2 comparison on this route is a composition reading first
    # and a fit reading second. Use tb_err, which has no denominator.
    #
    # The denominator-free fact worth testing: unfreezing improved bwd/tb_err
    # from 15.81 to 14.89 (seed B: 15.80 -> 14.86), against a bwd seed spread of
    # 0.006-0.038. ~0.92 nats, ~25x the noise, on a branch the fwd gradient does
    # NOT train -- so it is not the tautology that fwd training improves fwd
    # metrics. That transfer is the thing replay weight has to reproduce.
    #   user right  -> c_frz_rep40 pulls bwd/tb_err toward a_unf's 14.89
    #   not a dose  -> bwd/tb_err stays near c_frz_rep20's, and doubling replay
    #                  buys only replay's own metrics
    for nm, fr in (('c_frz_rep20', {'fwd': 0.2, 'bwd': 0.6, 'replay': 0.2}),
                   ('c_frz_rep40', {'fwd': 0.2, 'bwd': 0.4, 'replay': 0.4})):
        c = paired(local(base()))
        naive(c)['balance'] = None       # protocol.py:714 -- no reallocation
        naive(c)['fracs'] = dict(fr)
        arm(nm, 'C', 3600, resume(c, 3600),
            f'FROZEN, fixed mix {fr}. rep20 is the pinned-mix control (isolates "controller '
            f'off" from "more replay"); rep40 doubles replay weight. Both read against a_unf, '
            f'which got LESS replay (0.238) than a_frz (0.286) and still won.')

    # ---- PAIR D: is "unfreeze" just "more LR"? Completes a 2x2 with A. ----
    # REPOINTED TWICE. Originally a beta ladder (justified by "frozen is the
    # degrading regime", which pair A dissolved), then a length extension. This
    # is the better question, and it is the user's: D30's own discriminator asks
    # whether unfreezing and raising LR are ONE mechanism.
    #
    # The direct reading of that hypothesis -- "unfreezing doubles the effective
    # step" -- does NOT survive pair A's probe data:
    #     lrprobe/step_norm    a_frz 0.06496   a_unf 0.06360   ratio 0.979
    # The step is the SAME SIZE. Under Adam it is ~lr per coordinate largely
    # independent of which terms feed the gradient (A2), so unfreezing changes
    # the step's DIRECTION, not its norm. And alpha* moves the wrong way for
    # overstepping: 1.77 frozen vs 2.25 unfrozen, i.e. unfrozen is undershooting
    # MORE, with lower curvature (3.69 -> 2.67) along the step.
    #
    # But alpha* > 1 in BOTH arms, so pair A compared two cells that are each
    # below their own LR optimum, and an off-optimum comparison cannot separate
    # "better direction" from "further from the ceiling". That is the surviving
    # form of the hypothesis and it needs the LR leg.
    #
    # 1.72x is a_frz's own alpha* -- the probe's estimate of what it should have
    # been running. Both arms take the SAME raised LR so the cell is a clean
    # 2x2 against pair A's two at 1.25e-4:
    #
    #           lr 1.25e-4      lr 2.15e-4
    #   frozen    a_frz          d_frz_lrup
    #   unfrozen  a_unf          d_unf_lrup
    #
    # If unfreeze == more LR, d_frz_lrup should land on a_unf and d_unf_lrup
    # should overshoot. If the freeze effect is real, the two columns differ by
    # the same amount at both LRs.
    #
    # RISK, stated: 2.15e-4 is above this T's best-known LR and the aug02 cliff
    # is real. An arm that aborts is not a wasted hour -- the abort step is the
    # datum ramp arms are validated against -- but read cut_factor FIRST on any
    # arm that looks merely slow.
    # ONLY lr_fused is set. The naive stage is train_mode: fused, where
    # lr_policy / lr_back / lr_replay are DEAD knobs -- setting them would be
    # inert noise today and a silent three-way confound if the stage ever ran
    # unfused. lr_flow stays at 0.1 (scalar-calibrated, a separate quantity).
    #
    # The baseline is `lr_fused: auto`, and auto is a STRING, not a number --
    # 1.25e-4 is what it RESOLVED to. Verified before relying on it: across all
    # 3600 steps of both pair-A arms, lr_fused was bitwise FLAT at 1.25e-4 with
    # lr_ctrl/cut_factor == 1 and scale == 1 throughout, i.e. no cut, no warmup,
    # no decay. So auto behaved as a constant here and the 2x2 is clean. If a
    # future arm's cut_factor is not 1, this comparison is void -- check it
    # FIRST, per the LR-tripwire-deadlock note.
    LR_UP = 2.15e-4          # 1.72x the resolved 1.25e-4 = a_frz's alpha_median
    for nm, fn in (('d_frz_lrup', lambda x: x), ('d_unf_lrup', unfreeze)):
        c = fn(paired(local(base())))
        c['lr_fused'] = LR_UP
        arm(nm, 'D', 3600, resume(c, 3600),
            f'lr_fused {LR_UP:g} = 1.72x the resolved base, the probes own estimate of '
            f'a_frzs optimum. With pair A this is the D30 2x2 at T=10 -- the cell rb0808 '
            f'could not run.')


IDENTITY_EXEMPT = ('run_name',)


def assert_distinct(name, cfg, seen):
    """No two arms may differ only in label. See rb0808/make.py::assert_distinct
    -- an arm written by omission is a duplicate whenever mk_dev already sets
    the key, which is how the D30 block came to have two identical cells.

    `seed` is deliberately NOT exempt: pair B varies seed and nothing else, and
    that is a real arm, not a duplicate.

    `epochs` gets the middle verdict. Two arms alike but for run length are not
    a mistake -- pair D is deliberately pair A at 2x -- but they are also not
    independent: the shorter is a strict prefix of the longer, so only the
    longer carries new information and the overlap is recompute. That is worth
    SAYING at generation time and not worth blocking, so it warns.
    """
    ident = {k: v for k, v in sorted(cfg.items()) if k not in IDENTITY_EXEMPT}
    key = json.dumps(ident, sort_keys=True, default=str)
    if key in seen:
        raise AssertionError(f'{name} is identical to {seen[key]} except for '
                             f'{IDENTITY_EXEMPT}. An arm that varies nothing measures nothing.')
    seen[key] = name

    no_ep = json.dumps({k: v for k, v in ident.items() if k != 'epochs'},
                       sort_keys=True, default=str)
    if no_ep in seen.setdefault('_noep', {}):
        other, other_ep = seen['_noep'][no_ep]
        shorter = min(cfg['epochs'], other_ep)
        print(f'    NOTE {name} is {other} at a different length -- deliberate prefix '
              f'extension. Same seed and config, so both must agree up to step {shorter}; '
              f'treat any divergence below it as a determinism bug, not a result.')
    else:
        seen['_noep'][no_ep] = (name, cfg['epochs'])


def write_all():
    HERE.mkdir(parents=True, exist_ok=True)
    seen, rows = {}, []
    for name, pair, budget, cfg, asks in ARMS:
        assert_distinct(name, cfg, seen)
        assert cfg['batch_size'] == cfg['max_batch_size'], f'{name}: batch may float'
        assert cfg['checkpoint_read_only'] is True, f'{name}: would write checkpoints'
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        rows.append((pair, name, budget, P2_STEPS + budget, asks))
        print(f'  {pair}  {name:<14} +{budget:<5} steps  -> epochs {P2_STEPS + budget}')

    (HERE / 'INDEX.tsv').write_text(
        'pair\tname\tbudget\tepochs\tasks\n' +
        '\n'.join('\t'.join(str(x) for x in r) for r in rows) + '\n', encoding='utf-8')

    pairs = sorted({r[0] for r in rows})
    lines = ['# Launch ONE pair at a time. Read pair A before launching C or D.', '']
    for p in pairs:
        names = [r[1] for r in rows if r[0] == p]
        lines.append(f'# --- pair {p} ---')
        for n in names:
            lines.append(f'$env:PYTHONPATH="C:\\Users\\mikem\\Projects\\mxt_gfn\\mxtaltools;'
                         f'C:\\Users\\mikem\\Projects\\mxt_gfn\\gfn_diffusion"; '
                         f'& "C:\\Users\\mikem\\venvs\\csd_mxt_gfn\\Scripts\\python.exe" '
                         f'train.py --config configs\\local_aug08\\{n}.yaml')
        lines.append('')
    (HERE / 'launch.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\n{len(rows)} arms in {len(pairs)} pairs -> {HERE}')


if __name__ == '__main__':
    main()
    write_all()
