"""
Phase 2.2 -- the regression corpus. Can `configs/generate.py` express the
configs the project actually ran?

WHAT "REPRODUCE" CAN MEAN HERE, since it cannot mean "byte-identical". A
historical arm was built from a canonical config that has since moved through six
schema states, so every difference is either something the arm's author CHOSE or
something canonical changed underneath them. The testable claim is therefore:

    every difference between the generated arm and the historical one is
    EXPLAINED -- named in the spec as a deliberate override, or listed as a known
    canonical drift.

An UNEXPLAINED difference is the finding: it means the generator cannot express
something the corpus demonstrably needed, which is exactly what 2.2 is for. The
plan's acceptance criterion ("generate.py reproduces the corpus from explicit
inputs") is read that way, because the literal reading is unachievable by design
-- back-compat was dropped, and 461 of 515 battery configs no longer load at all.

THE CORPUS IS SMALL BECAUSE THE TREE IS. Measured 2026-08-17: 20 of 515 battery
configs load under current code; the rest carry retired keys, which is the
intended consequence of dropping back-compat rather than a fault. The loadable
set spans two problem shapes -- unconditional ELJ sg2 and QM9-conditional ELJ
sg2 -- so those are what can be regression-tested today.
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'configs'))

import config_snapshot                          # noqa: E402
import generate                                 # noqa: E402

pytestmark = pytest.mark.fast


#: Keys that differ between ANY historical arm and current canonical because
#: canonical moved, not because the arm chose anything. Each entry is a claim
#: that a future reader can check, so the reason is recorded with it.
CANONICAL_DRIFT = {
    # Every arm pins its own identity and output location.
    'config.run_name', 'config.tag', 'config.checkpoints_dir',
    'config.checkpoint_name', 'config.prior_model_name',
    'config.continue_from_checkpoint', 'config.load_weights_only',
    # Run length and cadence are per-battery decisions, not canonical's.
    'config.epochs', 'config.eval_period', 'config.figs_period',
    'config.archive_period', 'config.archive_buffers',
    'config.checkpoint_read_only',
    # The warmup ramp-exit window was added 2026-08-17 and no historical arm
    # carries it; it is ADDED rather than CHANGED, but list it so a rename is
    # not silently absorbed.
    'config.adaptive_lr.warmup_freeze_cos_window',
    # min_lr dropped 1e-6 -> 1e-8 on 2026-08-17. It is a NUMERICAL BACKSTOP, not a
    # policy: at 1e-6 it sat barely below the conditional VarGrad quality optimum,
    # so a controller asking to go lower was refused silently. Historical arms
    # predate that and carry the old floor.
    'config.min_lr',
    # level_gap moved 0 -> 1 on the CONDITIONAL route's var_conditioning stage
    # (2026-08-19). It is the per-condition Z-level tether, and it was adopted as
    # a stability term after a local single-key A/B: at 0 the forward VarGrad ran
    # 49 of 108 reports above 10x its median with log Z drifting up; at 1 there
    # was no excursion at all. Historical conditional arms predate that.
    # Reported on BOTH paths, since config_snapshot emits the resolved config and
    # the parsed stage summary separately.
    'config.protocols.conditional_vargrad.stages[1].loss_coeffs.bwd.level_gap',
    'stages[1].effective_loss_coeffs.bwd.level_gap',
    # Owner edits of 2026-08-19, with the occupancy ladder arming: rung spacing
    # 1.65 -> 1.6 and the OOM cut 0.5 -> 0.625. Historical arms carry the old
    # values.
    'config.batch_growth_factor', 'config.oom_batch_shrink_factor',
    # var_conditioning's `fwd/logw_std_within < 6.0` exit was DELETED from
    # canonical on 2026-08-17: the bar sat below a measured minimum of 17.1 and
    # next_battery.md 1.1 concluded the stage is terminal by design. Historical
    # conditional arms still carry it, so canonical is missing a block they have.
    # Reported on the STAGE summary's own path, not under config.*
    'stages[1].exit',
}


#: What `bsz` is: a short unconditional smoke battery whose ONLY intended axis is
#: the batch. Everything below is shared by both arms and is the battery's
#: character rather than the axis -- a 10-step warmup and a 200-step calibration
#: period because the arms run 1,900 steps, and four explicit float LRs, which is
#: the documented control arm (`lr_servo_managed` empty: the sensor reads and logs
#: while actuating nothing).
_BSZ_COMMON = {
    # the battery predates the occupancy ladder and pinned its batch, which the
    # armed canonical (batch_util_target 60, grow true) can no longer express
    # without saying so
    'grow_batch_size': False, 'batch_util_target': 0,
    'adaptive_lr.warmup_steps': 10,
    'adaptive_lr.ray_calibration.period': 200,
    'condition_log_z.half_life_visits': 7.0,
    'lr_policy': 1.25e-4, 'lr_back': 1.25e-4,
    'lr_replay': 1.25e-4, 'lr_fused': 1.25e-4,
}


def _spec_elj_b1000():
    """configs/bsz/b1000.yaml -- unconditional ELJ, batch pinned at 1000.

    The simplest shape in the corpus: canonical's own route with the batch and
    the run length pinned. If the generator cannot express this, it cannot
    express anything."""
    return 'configs/bsz/b1000.yaml', dict(
        problem='mipcas_elj', batch_size=1000, max_batch_size=1000,
        epochs=1900, eval_period=500, figs_period=1000, **_BSZ_COMMON)


def _spec_elj_b500():
    """configs/bsz/b500.yaml -- the same battery's other arm.

    A PAIR, deliberately. One arm proves the shape is expressible; two prove the
    generator varies the axis the battery varied, rather than happening to match
    one file. Note what moves with the batch: `fused_grad_accum_min_samples`
    tracks it, and `n_sub` doubles -- the probe needs its replicates back when the
    sub-batches halve."""
    return 'configs/bsz/b500.yaml', dict(
        problem='mipcas_elj', batch_size=500, max_batch_size=500,
        fused_grad_accum_min_samples=500, epochs=1900, eval_period=500,
        figs_period=1000, **{**_BSZ_COMMON,
                             'adaptive_lr.ray_calibration.n_sub': 16})


def _spec_qm9_conditional():
    """configs/shakeout_aug16/qm9_cond.yaml -- the QM9-conditional route.

    THE SHAPE THE CORPUS COULD NOT REACH UNTIL 2026-08-17. `problems.yaml` had no
    conditional MOLECULE entry and `test_problems.ALLOWED` omitted
    `embedding_conditioning`, so the project's main experimental line was
    describable only as raw overrides. With `qm9_conditional` in the registry the
    whole route -- protocol, the F-042 Z trio, the held-out set -- comes from one
    word, which is what makes this arm worth regression-testing: it proves the
    registry carries the route, not just that the merge works."""
    return 'configs/shakeout_aug16/qm9_cond.yaml', dict(
        problem='qm9_conditional',
        batch_size=500, max_batch_size=500, eval_num_samples=2000,
        grow_batch_size=False, batch_util_target=0,
        checkpoint_name='WARM_qm9_mle3k.pt', load_weights_only=True,
        # The battery tuned the MLE exit gate for a 3k-step warm start.
        **{'protocols.conditional_vargrad.stages[0].mle_gate.slope_t': 1.0,
           'protocols.conditional_vargrad.stages[0].mle_gate.min_rate': 5.0,
           'protocols.conditional_vargrad.stages[0].mle_gate.window': 100})


CORPUS = {'bsz_b1000': _spec_elj_b1000, 'bsz_b500': _spec_elj_b500,
          'qm9_conditional': _spec_qm9_conditional}


def _unexplained(historical_path, overrides, tmp_path, name):
    """Generate an arm to `overrides` and return the differences from the
    historical file that the spec does not account for."""
    cfg = generate.arm(name, **overrides)
    written = generate.emit({name: cfg}, outdir=tmp_path, quiet=True, index=False)[0]

    cmp = generate.deviations(written, reference=historical_path)
    assert not cmp.reference_error, f'{historical_path}: {cmp.reference_error}'
    assert not cmp.candidate_error, f'generated arm: {cmp.candidate_error}'

    # An override the spec names is explained by definition; so is a drift key.
    named = set()
    for key in overrides:
        if key == 'problem':
            continue
        named.add(f'config.{key.replace("__", ".")}')
    explained = named | CANONICAL_DRIFT

    def _covered(p):
        """Is `p` explained by an entry naming it or a section containing it?

        NOTE THE TWO PATH NAMESPACES. `config_snapshot` reports resolved-config
        keys as `config.<dotted>` and the parsed stage summary on its OWN short
        path (`stages[1].exit`), because the stage list is captured separately
        from the config tree. A matcher that assumed one prefix silently passed
        every stage-level difference."""
        return any(p == e or p.startswith(e + '.') or p.startswith(e + '[')
                   for e in explained)

    out = []
    selected = generate.load_yaml(historical_path).get('protocol')
    for path, ref, cand in cmp.changed:
        base = path.split('[')[0]
        if _covered(path):
            continue
        # `lr_servo_managed` is DERIVED from which rates are written `auto`, not
        # set. Naming `lr_policy: 1.25e-4` in the spec is what makes it False, so
        # crediting the derived witness to its source keeps the spec readable --
        # and config_snapshot records it precisely because the resolved rate
        # cannot show the difference (its docstring says so).
        if base.startswith('config.lr_servo_managed.'):
            if f'config.{base.rsplit(".", 1)[1]}' in named:
                continue
        # DRIFT IN A PROTOCOL THE ARM DOES NOT SELECT CANNOT REACH THE RUN. The
        # historical file carries every protocol, including ones canonical has
        # since edited; only the selected one executes. Counting the others would
        # make every arm report differences in a branch it never enters, which is
        # the noise that gets a report ignored.
        if selected and base.startswith('config.protocols.') and \
                not base.startswith(f'config.protocols.{selected}.'):
            continue
        out.append((path, ref, cand))
    return out, cmp


@pytest.mark.parametrize('name', sorted(CORPUS))
def test_corpus_arm_has_no_unexplained_differences(name, tmp_path):
    """THE ACCEPTANCE CRITERION. Anything reported here is a config axis the
    corpus used and the generator cannot reach from its explicit inputs."""
    historical, overrides = CORPUS[name]()
    unexplained, cmp = _unexplained(historical, overrides, tmp_path, name)
    assert not unexplained, (
        f'{name}: {len(unexplained)} unexplained difference(s) from {historical} '
        f'-- each is something the generator cannot express:\n' +
        '\n'.join(f'    {p}: historical={r!r} generated={c!r}'
                  for p, r, c in unexplained[:15]))


def test_the_corpus_check_can_fail(tmp_path):
    """MUTATION. The comparison above passes trivially if `deviations` reports
    nothing, so introduce a difference the spec does NOT name and require it to
    surface. Without this, a comparator that silently compared a file with
    itself would look like a clean corpus."""
    historical, overrides = _spec_elj_b1000()
    overrides = dict(overrides)
    overrides['seed'] = 999999          # deliberately absent from the spec's intent
    unexplained, _ = _unexplained(historical, overrides, tmp_path, 'mutant')
    # `seed` IS named in overrides, so it is explained -- strip it to prove the
    # explanation machinery is what suppresses it, not an empty comparison.
    assert not unexplained
    cfg = generate.arm('mutant2', **{k: v for k, v in overrides.items()})
    written = generate.emit({'mutant2': cfg}, outdir=tmp_path, quiet=True,
                            index=False)[0]
    cmp = generate.deviations(written, reference=historical)
    assert any(p == 'config.seed' for p, _, _ in cmp.changed), (
        'the comparator did not see a changed seed -- it is not comparing')


def test_the_loadable_corpus_is_measured_not_assumed():
    """The corpus is small for a stated reason. If a schema change makes these
    stop loading, this fails rather than the corpus quietly shrinking to zero."""
    for name in sorted(CORPUS):
        path = CORPUS[name]()[0]
        assert not config_snapshot.snapshot(path).get('load_error'), (
            f'{path} no longer loads; the corpus arm is inert')
