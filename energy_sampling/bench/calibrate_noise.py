"""
CALIBRATE THE BENCH'S NOISE AXIS AGAINST A REAL TB RUN.

Every noise result in `docs/lr_control_summary.md` is scored against a scale that
was invented. Median `cos(g_t, g_{t-1})` is 0.9997 at the quietest synthetic cell
and 0.29 at the noisiest; deep nets with modest batches often sit at 0.01-0.1,
which would put the bench's WORST case inside the real system's easy regime and
make every noise-driven ranking optimistic.

It also answers two things the synthetic surfaces structurally cannot:

  * is `cos` usable as a control signal on real TB gradients AT ALL? If it sits at
    the dimension null sqrt(2/(pi*d)) it carries no information and hypergradient
    is dead on arrival whatever the bench says. The null is printed beside it.
  * does MK's standing objection bite? `g_t` and `g_{t-1}` come from different
    replay draws AND a branch mixture the balance controller moves, so cos < 0
    can mean "overshot" or merely "different objective". `cos` is reported PER
    STEP_TYPE, so a fused-vs-fused comparison is like-for-like and the spread
    across types is the objective-realization component.

METHOD. Intercept `torch.nn.utils.clip_grad_norm_` and read the gradient THERE,
before the clip rescales it.

Reading `.grad` after `train_step` returns -- what this file used to do, and what
the paragraph here used to describe -- reads a vector the clip has ALREADY
rescaled in place (train.py:2854), so ||g|| comes back pinned at
`gradient_norm_clip`: measured EXACTLY 37.88 for 398 samples across two step
types and three runs, which is the clip constant, not a gradient. `cos` survives
that (a uniform positive rescale leaves a cosine unchanged) but ||g|| and the
descent check built on it do not.

`train_step` is still wrapped, for the CLOCK rather than for the gradient: it
pairs the captured vector with the previous one OF THE SAME `step_type`, counts
samples, and raises the stop sentinel. Nothing is reimplemented and nothing is
written back: this observes the real loop.

WHICH CHECKPOINT ACTUALLY LOADED IS RECORDED AT THE LOADER, NOT ASSERTED FROM THE
ARGS. `REGIMES['mle_fresh'] = None` used to mean "do not OVERRIDE the checkpoint",
which is not "do not LOAD one": with no override the config's own
`checkpoint_name` applies, and `elj_nehzor_sg14_t10_r2.yaml` ships
`..._final.pt` -- byte-identical to the `mle_converged` regime's path. So two of
the four regimes were the SAME converged run under different names, and the
`mle_fresh` column was never a fresh model. Preserved evidence:
`noise_calib_mle_fresh.json` covers steps 10001-10399, and a fresh run starts at
0.

`None` now FORCES a fresh model (`checkpoint_name=None` AND
`continue_from_checkpoint=False`, the two fields `train.py:1509-1526` actually
branches on), and `_assert_regime` checks the EFFECT from three independent
places: the paths the `Checkpointer` really loaded, the step index the run
resumed at, and -- reported, since it is what the regime names refer to -- the
protocol stage the samples came from.

WHY NOT `m.train()` + a loop: the first version of this file did exactly that,
and the instrumentation was never reached because `train()` does not return until
the run ends. It would have produced a clean-looking file of nothing -- the
documented "swallowed diagnostics fail as REASSURANCE" mode. The wrapper below
cannot fail that way: it asserts it captured samples and raises if it did not.

NO TRAINING IS KEPT -- AND THE FIRST TWO MECHANISMS FOR THAT WERE BOTH FICTION.

This file used to set `m.args.save_checkpoints = False` and claim on that basis
that it "cannot clobber a user checkpoint". **There is no `save_checkpoints`
anywhere in the codebase.** It was an attribute assigned to an args object that
nothing reads -- the same inert-flag shape as the `checkpoint_path` /
`model_path` / `load_checkpoint` trio described above, in the same file, one
paragraph apart. Every run of this diagnostic wrote checkpoints under the
CONFIG'S OWN run name for as long as the promise had been in the docstring.

Measured damage on 2026-08-13: runs of this file overwrote `_running.pt`,
`_best.pt`, `_buffers.pt`, `_prior.pt`, `_stage_start.pt` and -- via the
`train_prior -> equilibration` transition, which writes it -- `phase1_exit.pt`,
for `d33elj_elj_nehzor_sg14_t10_r2`.

The suppression is now the only kind that can be checked: `Checkpointer.save`
and `.save_buffers` are replaced with recording no-ops on the instance's own
class, so every call site in `train.py` AND `protocol.py` (stage_start,
phase1_exit, prior) hits the same wall regardless of which name imported it.
`run()` prints what WOULD have been written, so the suppression is visible when
it works rather than only when it fails.

The run is aborted after `steps` via a sentinel exception.

GPU CO-TENANCY. The card had 3.4 of 16.3 GB resident when this was written and
the box has a BSOD history from over-subscription, so `cuda_memory_fraction` is
capped far below the default 0.9. Raise only if the card is idle.
"""
import json
import math
import os
import statistics as st
import sys

CONFIG = 'configs/elj_nehzor_sg14_t10_r2.yaml'
# THE FALLBACK DEFAULT for a bare `run()`. `__main__` never uses it -- it drives
# the four REGIMES below, because one checkpoint answers only about its own
# operating point.
#
# MID-TB, NOT CONVERGED. The first attempt loaded a `_final.pt` and measured 400
# steps at a stationary point: cos ~ 0 with a coin-flip sign, which is the
# CORRECT reading at a stationary point and not a noise verdict, so that run
# could not answer the question. `_running.pt` from the r2 run is mid-early TB --
# the live regime, and the one that matters for overall acceleration.
#
# HALF OF THAT DIAGNOSIS WAS AN ARTIFACT, which is worth keeping rather than
# quietly dropping: the other evidence cited for "stationary" was "gradient norm
# dead flat at 37.9 for the whole window", and 37.9 is the POST-CLIP PIN (37.88,
# see the clip spy in `run`), which reads flat at every checkpoint whether or not
# the model is converged. Only the cos half of that argument survives.
CKPT = ('checkpoints/d33elj_elj_nehzor_sg14_t10_r2_elj-nehzor_sg14_zp1_'
        'elj_prior_dataset-T2.5-990198_running.pt')
STEPS = 400
MEM_FRACTION = 0.45
OUT = 'bench_noise_calibration.json'
#: The window is only informative if the model is ACTUALLY DESCENDING in it.
#: Checked and reported, not assumed -- a flat ||g|| means the measurement is
#: about convergence rather than about noise, whatever cos says.
MIN_GNORM_FALL = 1.02
#: A run that loaded nothing must START at zero. Generous because the first
#: sample lands a step or two in; the failure this catches resumed at 10001.
FRESH_MAX_STEP = 100

#: THE FOUR REGIMES. A single checkpoint answers only about its own operating
#: point -- the first attempt measured a CONVERGED model and cos ~ 0 there is
#: correct rather than damning.
#:
#: `None` MEANS NO CHECKPOINT AT ALL, and did not before -- see the module
#: docstring. Under the old reading `mle_fresh` resolved to the config's
#: `checkpoint_name`, which is the same file `mle_converged` names explicitly,
#: so the two regimes measured one checkpoint. `_assert_regime` now fails the
#: run rather than reporting that a second time.
CK = ('checkpoints/d33elj_elj_nehzor_sg14_t10_r2_elj-nehzor_sg14_zp1_'
      'elj_prior_dataset-T2.5-990198_')
CK_OLD = ('checkpoints/d33elj_elj_nehzor_sg14_t10_elj-nehzor_sg14_zp1_'
          'elj_prior_dataset-T2.5-990198_')
#: A REGIME MUST NAME AN IMMUTABLE CHECKPOINT. `eq_descent` used to point at
#: `_running.pt`, which every run of this config overwrites -- so the regime
#: silently meant "whatever ran last". Measured: on the morning of 2026-08-13 it
#: resolved to step 11001, and by midday to step 350, because runs in between
#: had rewritten it. Two measurements under one regime name, of two different
#: models, with nothing in the output to say so. `step15000.pt` is a periodic
#: checkpoint of the user's own r2 run, mid-equilibration and never rewritten.
#: A LADDER ALONG ONE TRAJECTORY, not four unrelated points. The first three are
#: the same MLE run at step 0 / 5000 / 10000 (its `final.pt` never left
#: `train_prior`, so "converged MLE" is literally what it is), which makes cos
#: comparable across them -- same problem, same width, same branch, only the
#: distance-from-convergence changes. The last two are the other stage.
REGIMES = {
    'mle_fresh':      None,                        # step 0, from scratch
    'mle_mid':        CK_OLD + 'step5000.pt',      # step 5000, train_prior
    'mle_converged':  CK_OLD + 'final.pt',         # step 10000, train_prior
    'eq_phase1exit':  CK + 'phase1_exit.pt',       # step 10640 -> equilibration
    'eq_descent':     CK + 'step15000.pt',         # equilibration, past the exit
}


class _Done(Exception):
    """Sentinel: stop the real train loop once enough steps are recorded."""


def _assert_regime(loaded, ckpt, first_step, stage):
    """
    Fail the run unless the model the loop is stepping IS the regime's model.

    Checks the effect from two independent mechanisms, because the mechanism that
    failed last time -- the args -- looked correct the whole way through. `loaded`
    comes from inside `Checkpointer`, `first_step` from the restored training
    state; a load that silently no-ops shows up in the second even if the first
    is somehow bypassed.

    Called on the FIRST wrapped step and deliberately NOT inside that function's
    `except Exception` (which prints and continues, by design, so a logging
    failure cannot kill a run). An assertion swallowed into a `skipped a sample`
    line would be the same reassurance-shaped failure this exists to catch.
    """
    where = f'stage={stage!r} first_step={first_step}'
    if ckpt is None:
        if loaded:
            raise RuntimeError(
                f'FRESH regime LOADED A CHECKPOINT: {loaded}. Do NOT read the '
                f'numbers from this run as a fresh model. ({where})')
        if first_step > FRESH_MAX_STEP:
            raise RuntimeError(
                f'FRESH regime resumed at step {first_step} (> {FRESH_MAX_STEP}) '
                f'with no recorded load -- training state came from somewhere '
                f'this spy does not see. ({where})')
    else:
        want = os.path.abspath(ckpt)
        got = [os.path.abspath(p) for p in loaded]
        if got != [want]:
            raise RuntimeError(
                f'regime asked for {want!r} and the loader opened {got!r}. '
                f'({where})')
        if first_step <= FRESH_MAX_STEP:
            raise RuntimeError(
                f'loaded {os.path.basename(ckpt)} but the run starts at step '
                f'{first_step}: training state did not restore, so this is a '
                f'FRESH model wearing the regime\'s name. Check '
                f'load_weights_only. ({where})')
    print(f'  [calib] VERIFIED loaded={loaded or "nothing"}  {where}', flush=True)


def run(steps=STEPS, config=CONFIG, out=OUT, ckpt=CKPT):
    import torch
    # BEFORE construction: setting m.args.use_wandb afterwards was too late and
    # the first attempt created a real wandb run. WANDB_MODE=disabled needs no
    # cooperation from the arg parser and creates no offline run to sync.
    os.environ['WANDB_MODE'] = 'disabled'
    sys.argv = ['train.py', '--config', config]
    import train

    m = train.Modeller()
    m.args.cuda_memory_fraction = MEM_FRACTION
    # `m.args.use_wandb = False` used to sit here too. There is no `use_wandb`
    # in the codebase either -- it read as a second layer of wandb suppression
    # and was a second inert assignment. `WANDB_MODE=disabled` above is the one
    # that works, and it needs no cooperation from the arg parser.

    # WHAT LOADED, READ AT THE LOADER. Both reload paths in `init_gfn` funnel
    # through these two methods, so this records the file the run actually
    # opened -- the only statement about the checkpoint that is not an inference
    # from the args we set.
    #
    # PATCH THE INSTANCE'S OWN CLASS, NEVER THE IMPORTED ONE. `import
    # checkpointing` and train.py:29's `from energy_sampling.checkpointing
    # import Checkpointer` produce TWO DISTINCT MODULE OBJECTS with two distinct
    # class objects -- PYTHONPATH carries both `gfn_diffusion` and
    # `gfn_diffusion/energy_sampling`, so every module here is importable under
    # two names. Verified: `checkpointing.Checkpointer is
    # energy_sampling.checkpointing.Checkpointer` -> False.
    #
    # The first version of this spy patched the imported one and recorded
    # NOTHING while the log printed `Loading model from checkpoint ...` two lines
    # later. It was caught only because the step-index check is independent and
    # disagreed. `type(m.checkpointer)` is whatever train.py actually built,
    # under any import name.
    loaded = []
    _ck = type(m.checkpointer)
    _real_full, _real_weights = _ck.load_full, _ck.load_weights_only

    # NOTHING MAY BE WRITTEN -- TWO LAYERS, THE FIRST ONE SUPPORTED.
    #
    # `checkpoint_read_only` is a REAL field. `Checkpointer.read_only`
    # (checkpointing.py:101) reads it and every write path checks it: `save`,
    # `save_buffers`, `archive` and `link`. This is the mechanism the codebase
    # already provides, and the earlier `save_checkpoints = False` was an
    # invented name standing where it should have been.
    #
    # The second layer exists because the first version of THIS block patched
    # only `save`/`save_buffers` -- reasoning from the call sites in `train.py`
    # and `protocol.py` -- and missed `archive`, which calls `link` directly and
    # never goes through `save`. `archive` hardlinks `stepNNNNN.pt` onto the
    # CURRENT `running.pt` bytes, so a resume at step 15000 with archiving on
    # pointed the user's `_step15000.pt` at this diagnostic's step-10700 state
    # and destroyed it. Enumerating call sites is not enumerating write paths;
    # the list below comes from the WRITER's own methods.
    m.args.checkpoint_read_only = True
    blocked = []
    _writers = ('save', 'save_buffers', 'archive', 'link')
    _real_writers = {n: getattr(_ck, n) for n in _writers}

    def _blocker(name):
        def _no_write(self, tag='?', *a, **kw):
            blocked.append(f'{name}:{tag}')
        return _no_write

    for _n in _writers:
        setattr(_ck, _n, _blocker(_n))
    # If a write path is ever ADDED, this is the line that notices.
    _unknown = [n for n in dir(_ck)
                if n.startswith('save') and n not in _writers]
    if _unknown:
        raise RuntimeError(
            f'Checkpointer grew write methods this file does not block: '
            f'{_unknown}. Add them to _writers before running.')

    def _spy_full(self, path, *a, **kw):
        loaded.append(str(path))
        return _real_full(self, path, *a, **kw)

    def _spy_weights(self, path, *a, **kw):
        loaded.append(str(path))
        return _real_weights(self, path, *a, **kw)

    _ck.load_full, _ck.load_weights_only = _spy_full, _spy_weights

    # train.py:1511 builds the reload path as
    #     f'{args.checkpoints_dir}/{args.checkpoint_name}'
    # and reads NOTHING called checkpoint_path/model_path/load_checkpoint. The
    # previous attempt set those three, printed a confident line, and silently
    # loaded the config's own _final.pt -- the exact "swallowed diagnostics fail
    # as REASSURANCE" mode, twice in one night. So set the two fields that are
    # actually read, and VERIFY the resolved path rather than announcing intent.
    if ckpt:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        m.args.checkpoints_dir = os.path.dirname(ckpt) or '.'
        m.args.checkpoint_name = os.path.basename(ckpt)
        resolved = f'{m.args.checkpoints_dir}/{m.args.checkpoint_name}'
        if os.path.abspath(resolved) != os.path.abspath(ckpt):
            raise RuntimeError(f'checkpoint override did not resolve: {resolved}')
        print(f'  [calib] checkpoints_dir={m.args.checkpoints_dir}', flush=True)
        print(f'  [calib] checkpoint_name={m.args.checkpoint_name}', flush=True)
    else:
        # FRESH MEANS FRESH. Leaving these alone is not "no checkpoint" -- it is
        # "the config's checkpoint", which for this config is a converged
        # `_final.pt`. `init_gfn` builds a new model only when BOTH of these are
        # falsy, so both are cleared; `continue_from_checkpoint` is the `elif`
        # branch and would otherwise reload the newest `_running.pt` it can find.
        m.args.checkpoint_name = None
        m.args.continue_from_checkpoint = False
        print('  [calib] FRESH -- checkpoint_name=None, '
              'continue_from_checkpoint=False', flush=True)

    rec, state = [], {'prev': {}, 'n': 0, 'pre': None, 'first_step': None}

    # READ THE GRADIENT BEFORE CLIPPING. train.py:2854 calls
    # clip_grad_norm_(model.parameters(), gradient_norm_clip), which rescales in
    # place, so a read after train_step returns sees a vector pinned at the clip
    # norm -- measured ||g|| was EXACTLY 37.88 for 398 samples across two step
    # types and three runs, which is the signature, not a gradient. cos survives
    # that (a uniform positive rescale leaves cosine unchanged) but ||g|| and any
    # descent check built on it do not. Wrapping the clip call is the earliest
    # point that needs no model surgery.
    _real_clip = torch.nn.utils.clip_grad_norm_

    def _clip_spy(parameters, *a, **kw):
        # MATERIALISE FIRST. `train.py` passes `model.parameters()`, a GENERATOR:
        # iterating it here would consume it and hand the real clip an empty
        # sequence, silently disabling gradient clipping for the whole run. Pass
        # the list on, never the original iterator.
        ps = ([parameters] if isinstance(parameters, torch.Tensor)
              else list(parameters))
        try:
            live = [p.grad.detach().reshape(-1) for p in ps if p.grad is not None]
            state['pre'] = torch.cat(live).float().clone() if live else None
        except Exception:
            state['pre'] = None
        return _real_clip(ps, *a, **kw)

    torch.nn.utils.clip_grad_norm_ = _clip_spy
    original = type(m).train_step

    def wrapped(self, step_type, *a, **kw):
        # OUTSIDE the try below, on purpose -- see _assert_regime. This is the
        # first moment the loop is provably stepping the regime's model, and 400
        # steps are not worth spending on the wrong one.
        stage = getattr(getattr(self, 'protocol', None), 'stage', None)
        stage_name = getattr(stage, 'name', None)
        if state['first_step'] is None:
            state['first_step'] = int(self.step_ind)
            _assert_regime(loaded, ckpt, state['first_step'], stage_name)
        outv = original(self, step_type, *a, **kw)
        try:
            g = state.pop('pre', None)
            state['pre'] = None
            if g is not None:
                if torch.isfinite(g).all():
                    n = float(g.norm())
                    p_ = state['prev'].get(step_type)
                    if p_ is not None and n > 0:
                        pn = float(p_.norm())
                        if pn > 0:
                            rec.append(dict(
                                step_type=step_type, step=int(self.step_ind),
                                stage=stage_name,
                                cos=float(torch.dot(g, p_)) / (n * pn),
                                gnorm=n, dim=int(g.numel())))
                    state['prev'][step_type] = g.clone()
                    state['n'] += 1
        except Exception as e:                       # never kill the run to log
            print(f'  [calib] skipped a sample: {type(e).__name__}: {e}', flush=True)
        if state['n'] >= steps:
            raise _Done()
        return outv

    type(m).train_step = wrapped
    print(f'config {config}\nsteps  {steps}   mem_fraction {MEM_FRACTION}',
          flush=True)
    try:
        m.train()
    except _Done:
        print('  [calib] captured enough steps, stopping the run', flush=True)
    finally:
        type(m).train_step = original
        torch.nn.utils.clip_grad_norm_ = _real_clip
        _ck.load_full, _ck.load_weights_only = _real_full, _real_weights
        for _n, _fn in _real_writers.items():
            setattr(_ck, _n, _fn)
        # PRINT THE SUPPRESSION WHEN IT WORKS. A guard that is only visible on
        # failure is indistinguishable from an absent one, which is exactly how
        # `save_checkpoints=False` survived as a docstring promise.
        if blocked:
            from collections import Counter
            print(f'  [calib] BLOCKED {len(blocked)} checkpoint writes: '
                  f'{dict(Counter(blocked))}', flush=True)
        else:
            print('  [calib] no checkpoint writes were attempted', flush=True)

    # A run that logged nothing is a FAILURE, not a quiet success.
    if not rec:
        raise RuntimeError(
            'no gradient samples captured -- the wrapper never saw a live .grad. '
            'Do NOT read this as "the noise is low".')

    # AGAIN AT THE END. The first-step check cannot see a load that happens
    # later, and a mid-run reload would mean these samples span two models.
    _assert_regime(loaded, ckpt, state['first_step'], rec[-1].get('stage'))
    report(rec, out)
    return rec


def report(rec, out=OUT):
    d = rec[0]['dim']
    null = math.sqrt(2.0 / (math.pi * max(d, 2)))
    print(f'\n{"=" * 72}\nREAL TB GRADIENT NOISE -- {len(rec)} samples, '
          f'{d:,} policy params\n{"=" * 72}')
    print(f'  null |cos| for independent vectors at this width: {null:.5f}')
    # SPLIT BY STAGE AS WELL AS BY STEP TYPE. The regimes are named after stages,
    # so a run that crosses a transition mid-window is pooling two regimes into
    # one median -- visible here, invisible in a step_type-only table.
    steps = [r['step'] for r in rec]
    print(f'  steps {min(steps)}-{max(steps)}')
    print(f'\n  {"stage":<16} {"step_type":<10} {"n":>5} {"median cos":>11} '
          f'{"p25":>8} {"p75":>8} {"cos/null":>9} {"median |g|":>11}')
    keys = sorted({(r.get('stage') or '?', r['step_type']) for r in rec})
    for sname, t in keys:
        sel = [r for r in rec
               if (r.get('stage') or '?') == sname and r['step_type'] == t]
        c = sorted(r['cos'] for r in sel)
        gn = [r['gnorm'] for r in sel]
        if len(c) < 2:
            continue
        med = st.median(c)
        q = lambda f: c[min(int(f * len(c)), len(c) - 1)]          # noqa: E731
        print(f'  {sname:<16} {t:<10} {len(c):>5} {med:>11.4f} {q(.25):>8.4f} '
              f'{q(.75):>8.4f} {med / null:>9.1f} {st.median(gn):>11.4g}')
    if len({k[0] for k in keys}) > 1:
        print('  >> THIS WINDOW CROSSES A STAGE TRANSITION. The OVERALL median '
              'below pools\n     two regimes; read the per-stage rows instead.')
    # IS THE MODEL ACTUALLY DESCENDING? If not, cos ~ 0 means "converged", not
    # "too noisy to control", and the whole measurement is about the wrong thing.
    g0 = st.median([r['gnorm'] for r in rec[:max(len(rec) // 5, 2)]])
    g1 = st.median([r['gnorm'] for r in rec[-max(len(rec) // 5, 2):]])
    fall = g0 / max(g1, 1e-12)
    print(f'\n  ||g||  {g0:.4g} -> {g1:.4g}   '
          f'({fall:.2f}x fall over the window)')
    # THREE CASES, NOT TWO. `fall < MIN_GNORM_FALL` is true both when ||g|| is
    # FLAT and when it RISES, and the two mean opposite things -- the first run
    # of the fixed fresh regime rose 5.9x and was told it looked stationary and
    # should "pick an earlier checkpoint", on a run that starts at step 0. A
    # warning that points the wrong way costs more than no warning.
    if fall > 1.0 / MIN_GNORM_FALL and fall < MIN_GNORM_FALL:
        print('  >> FLAT ||g||. If cos is also ~0 that is the CORRECT reading at '
              'a stationary\n     point and not a noise verdict; this window '
              'cannot say whether cos carries\n     signal during active '
              'convergence -- pick an earlier checkpoint.')
    elif fall <= 1.0 / MIN_GNORM_FALL:
        print(f'  >> ||g|| is RISING ({1 / fall:.2f}x), not flat -- so this is '
              'NOT a stationary point\n     and the cos reading below is from an '
              'ACTIVE regime. Ordinary on a fresh\n     model, and after a stage '
              'transition, which rebuilds the optimizers and\n     re-warms the '
              'LR. ||g|| is a weak descent proxy either way: it can rise\n     '
              'while the loss falls.')

    allc = sorted(r['cos'] for r in rec)
    med = st.median(allc)
    print(f'\n  OVERALL median cos {med:.4f}   = {med / null:.1f}x the noise null')
    print('  bench reference: 0.9997 at noise 0.01, 0.29 at noise 2 (the worst '
          'cell tested)')
    if med < 0.29:
        print('  >> REAL IS NOISIER THAN THE BENCH\'S WORST CELL. Every '
              'noise-driven\n     ranking in docs/lr_control_summary.md is '
              'optimistic.')
    with open(out, 'w') as f:
        json.dump(rec, f)
    print(f'\n  raw samples -> {out}')


if __name__ == '__main__':
    which = [a for a in sys.argv[1:] if not a.isdigit()]
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), STEPS)
    names = which or list(REGIMES)
    for name in names:
        if name not in REGIMES:
            raise SystemExit(f'unknown regime {name!r}; have {list(REGIMES)}')
        bar = '#' * 72
        print('\n' + bar + '\n# REGIME: ' + name + '\n' + bar, flush=True)
        try:
            run(n, ckpt=REGIMES[name], out=f'noise_calib_{name}.json')
        except Exception as e:
            # one regime failing must not hide the others
            print(f'  !! {name} FAILED: {type(e).__name__}: {e}', flush=True)
