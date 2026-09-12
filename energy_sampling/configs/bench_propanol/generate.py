"""Generate the paired propanol benchmark: conditional vs unconditional, same target.

SIX CONFIGS -- two arms x three seeds. The arms differ in EXACTLY the keys listed in `ARMS`
below and in nothing else; everything else is patched identically from one base, so a
difference in the result is attributable to the architecture rather than to a setting that
drifted between two hand-written files.

    UNCONDITIONAL   molecules_path: null, embedding_conditioning: false -> SetPolicy
    CONDITIONAL     32 duplicate propanol conditions with real embeddings
                    -> ConditionalSetPolicy (per-DoF correlator + pooled context + Z(c) head)

Both are propanol at `full` (d = 30 = 3N-6, no linear centre, so the transverse chart is
inert here and this measures conditioning alone). Both read prior files built by
build_propanol_benchmark.py, which holds the SAME 4096 states bitwise.

THE PROTOCOL IS DELIBERATELY THE SIMPLEST THING THAT TRAINS: an MLE warm start, then TB only
on a 50:50 fused fwd:bwd mixture. No VarGrad, no replay branch, no balance controller, no
annealing, no lambda path. The point of a paired benchmark is that the only interesting
variable is the arm, and every controller left switched on is one more thing that can differ
between them for reasons unrelated to the question.

THE STAGE SWITCH IS A STEP COUNT, NOT A GATE. `mle_gate` is the designed mechanism for
"stop MLE when it stops descending", and it is the wrong one here: it is adaptive, so the two
arms would switch at different steps and the comparison would confound the warm start's
LENGTH with the architecture. Stage exits are metric-based with a 10-step tick cadence, so a
term that is always satisfied plus `patience: N` is exactly a counter of 10N steps. It is
written as a step counter on purpose; the trivially-true threshold is the mechanism, not an
accident.
"""
import argparse
import io
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'conformer_propionitrile_full_tb.yaml'

#: the MLE warm start, in TRAIN STEPS. 10 steps per exit tick.
MLE_STEPS = 3000
SEEDS = (0, 1, 2)

ARMS = {
    'uncond': dict(molecules_path='null', embedding_conditioning='false'),
    'cond': dict(molecules_path="'{d}/propanol_cond_conditions.pt'",
                 embedding_conditioning='true'),
}

PROTOCOL = """protocols:
  # A SIMPLIFICATION OF THE CRYSTAL `unconditional_tb` PROTOCOL, not a fresh one. That
  # protocol is the reference for this problem shape, and a hand-rolled substitute was
  # measurably worse: a warm start of pure MLE followed by TB on both branches DETONATED at
  # step 61 -- four rewinds, FrozenTrainingState, unrecoverable -- because log Z was still at
  # its initial value when TB engaged and the policy and Z then fought in one gradient.
  #
  # WHAT IS KEPT from the reference, because removing it is what broke:
  #   * the warm start trains `mle + tbc`, not MLE alone. `tbc` is a TB-consistency residual
  #     over K same-terminal backward rollouts (hence repeats: 2), so log Z trains DURING the
  #     warm start. This is the mechanism; `bootstrap_z` is not an alternative here -- it was
  #     dead on every route until this branch fixed it, and even fixed it anchors on tracker
  #     evidence that a pure-MLE stage never accumulates (measured: "0 conditions").
  #   * the terminal stage SPLITS ROLES rather than pointing TB at everything. fwd trains
  #     Z only (freeze_policy); bwd trains the policy only (freeze_z, inherited). Each
  #     parameter group gets exactly one trainer, which is what keeps the two from fighting.
  #
  # WHAT IS DROPPED, because a paired benchmark wants the fewest moving parts:
  #   the replay branch, the ratio balance controller, the ray LR sensor, z_calibration,
  #   rebuild_prior_by_churn on entry, and the adaptive multi-term exit. Each is another way
  #   the two arms could diverge for a reason that is not the architecture.
  bench_propanol_tb:
    stages:
      # -- WARM START. Backward MLE from the prebuilt prior, with tbc keeping log Z honest.
      - name: warm_start
        train_mode: bwd
        bwd_sampling_mode: dataset
        flags:
          update_log_z: true          # accumulate ema_logw / best_energy evidence
          buffers_active: true
        # DECLARED FOR THE DIAGNOSTIC, NOT FOR THE EXIT. This publishes gates/mle_flat, so
        # after the fact we can see whether __MLE_STEPS__ steps was actually long enough --
        # without letting an adaptive gate decide it, which would let the two arms warm-start
        # for DIFFERENT lengths and confound the comparison with the architecture.
        mle_gate:
          slope_t: 2.0
          min_rate: 0.05
          window: 300
        loss_coeffs:
          # repeats > 1 is REQUIRED by tbc: the consistency residual is defined over K
          # same-terminal backward rollouts, and get_tbc_loss asserts it.
          bwd: { mle: 1.0, tbc: 1.0, repeats: 2.0, tb_z_source: learned }
        # A STEP COUNTER. `bwd/loss` is written every tick and is always above -1e9, so the
        # term passes on every tick and `patience` alone sets the length: 10 steps per tick
        # x __PATIENCE__ = __MLE_STEPS__ steps, IDENTICALLY in both arms.
        exit:
          - { metric: bwd/loss, above: -1.0e+9, patience: __PATIENCE__ }
        # NOT `snapshot_prior`. The crystal route freezes the MLE-trained policy as THE prior
        # model because its prior has to be learned; this route already has a fitted
        # InternalPrior drawing r/theta at the force field's own thermal widths.
        on_exit: [ 'snapshot:phase1_exit' ]

      # -- TB, 50:50 fwd:bwd, fused. Terminal: no exit block, runs to `epochs`.
      # `fracs` ARE LOSS WEIGHTS ON FULL BATCHES, not a sampling split: both branches run
      # every step and each contributes half the gradient. replay at 0 falls below the 0.01
      # deactivate_threshold and is skipped entirely.
      - name: tb
        train_mode: fused
        bwd_sampling_mode: dataset
        flags:
          update_log_z: true
          buffers_active: true
        fracs: { fwd: 0.5, bwd: 0.5, replay: 0.0 }
        loss_coeffs:
          # ONE TRAINER PER PARAMETER GROUP, the reference protocol's arrangement:
          #   fwd -> Z only, on-policy, policy detached at the source
          #   bwd -> policy only, from the prebuilt prior (freeze_z inherited from the base)
          # tb_z_source EXPLICIT AND IDENTICAL IN BOTH ARMS. `persistent` adds a second,
          # per-condition log Z bookkeeping system alongside the flow head; every past
          # CONDITIONAL battery used it and config_invariants says so. Deliberately not used:
          # this benchmark compares log Z BETWEEN the arms, so both must take it from the same
          # place -- the flow head, which on the conditional route already IS
          # Z_MLP(condition embedding). Using `persistent` on one arm only would make the log Z
          # comparison a comparison of Z mechanisms.
          fwd: { tb: 1.0, freeze_policy: 1.0, tb_z_source: learned }
          bwd: { tb: 1.0, mle: 0.0, tb_z_source: learned }
"""


def patch(text, key, value, *, top_level=True):
    """Replace a scalar assignment, keeping its trailing comment. Fails if absent."""
    pat = re.compile(r'^(%s%s:)([^\n#]*)' % ('' if top_level else r'\s+', re.escape(key)),
                     re.MULTILINE)
    new, n = pat.subn(lambda m: f'{m.group(1)} {value}  ', text, count=1)
    if n != 1:
        raise SystemExit(f'could not patch {key!r} ({n} matches)')
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True,
                    help='where build_propanol_benchmark.py wrote its .pt files')
    ap.add_argument('--epochs', type=int, default=15000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--eval-period', type=int, default=500)
    # must be a MULTIPLE of eval_period or the two schedules never coincide and no figure is
    # ever logged -- config_invariants.figs_period_fires calls that an ERROR, correctly: a
    # figure period that cannot fire reads as "figures are on" and produces none.
    ap.add_argument('--figs-period', type=int, default=2500)
    args = ap.parse_args()
    d = args.data_dir.replace('\\', '/').rstrip('/')

    base = io.open(BASE, encoding='utf-8').read()
    # strip the base's protocol block and append ours
    i = base.index('protocols:')
    j = base.index('\nbuffers:', i)
    proto = (PROTOCOL.replace('__PATIENCE__', str(MLE_STEPS // 10))
             .replace('__MLE_STEPS__', str(MLE_STEPS)))
    base = base[:i] + proto + base[j:]

    written = []
    for arm, keys in ARMS.items():
        for seed in SEEDS:
            t = base
            t = patch(t, 'run_name', f"'bench_propanol_{arm}_s{seed}'")
            t = patch(t, 'seed', str(seed))
            t = patch(t, 'device', "'cuda'")
            t = patch(t, 'buffer_device', 'cuda')
            t = patch(t, 'protocol', 'bench_propanol_tb')
            t = patch(t, 'epochs', str(args.epochs))
            t = patch(t, 'batch_size', str(args.batch_size))
            t = patch(t, 'eval_period', str(args.eval_period))
            t = patch(t, 'figs_period', str(args.figs_period))
            t = patch(t, 'prior_path',
                      f"'{d}/propanol_{'cond' if arm == 'cond' else 'uncond'}_prior.pt'")
            t = patch(t, 'smiles', "'CCCO'", top_level=False)
            for k, v in keys.items():
                t = patch(t, k, v.format(d=d))
            out = HERE / f'bench_propanol_{arm}_s{seed}.yaml'
            io.open(out, 'w', encoding='utf-8', newline='').write(t)
            written.append(out.name)
    print(f'wrote {len(written)} configs into {HERE}:')
    for w in written:
        print('  ', w)
    print(f'\nMLE warm start: {MLE_STEPS} steps (patience {MLE_STEPS // 10} x 10-step ticks), '
          f'then TB 50:50 fwd:bwd to epoch {args.epochs}')


if __name__ == '__main__':
    main()
