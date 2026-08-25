"""Route-agnostic convergence measurement: per-column marginal fit, and the exit gate.

WHY THIS IS NOT IN conformer_modeller. Nothing here knows what a conformer is. The inputs
are a 2-D array of samples, a 2-D reference, a periodicity mask and a metric history -- which
the crystal route supplies just as readily (``latent_params()`` against the prior dataset's
own latents). Leaving it on the conformer subclass meant the crystal route emitted none of
it, and the exit criterion could only ever be validated on one problem at one timescale.

THE ONE THING THAT MAKES IT PORTABLE is that every target is MEASURED rather than written.
``w1r/perfect_median`` and ``w1r/perfect_worst`` come from scoring a genuine draw of the
reference through the identical code path, so they re-derive themselves on a 12-dimensional
crystal latent exactly as on an 87-column conformer state -- where the extreme-value
statistic in particular calibrates very differently. Nothing in this module carries a
constant tuned to one problem, with the single exception of ``window``, which buys
measurement precision rather than encoding an assumption.
"""
from __future__ import annotations

import numpy as np


def _column_w1(samples, reference, periodic, n_offsets: int = 24):
    """Per-column 1-D Wasserstein distance, CIRCULAR on the periodic columns.

    Quantile form (mean |sorted difference|) rather than scipy, so it costs nothing and
    tolerates unequal sample counts.

    A periodic column lives on a circle of period 2, so a distribution straddling the wrap
    would read as maximally far from an identical one centred at zero. The circular
    distance is therefore the minimum over a rigid rotation of the sampler, scanned on a
    coarse grid -- exact when the two differ by a shift, and an upper bound otherwise,
    which is the safe direction for a "worst columns" ranking.
    """
    q = np.linspace(0.0, 1.0, 512)
    out = np.zeros(samples.shape[1])
    for j in range(samples.shape[1]):
        b = np.quantile(reference[:, j], q)
        v = samples[:, j]
        d = float(np.abs(np.quantile(v, q) - b).mean())
        if periodic[j]:
            for off in np.linspace(-1.0, 1.0, n_offsets, endpoint=False):
                w = ((v + off + 1.0) % 2.0) - 1.0
                d = min(d, float(np.abs(np.quantile(w, q) - b).mean()))
        out[j] = d
    return out


def _column_w1_ratio(samples, reference, periodic, reps: int = 4, cache: dict = None):
    """Per-column W1 in units of ITS OWN SAMPLING-NOISE FLOOR. Dimensionless; 1.0 = perfect.

    A raw W1 of 0.26 is uninterpretable: nobody knows what a converged sampler scores,
    because a finite sample never matches its own parent exactly. Measure that instead --
    draw n rows FROM the reference and score them against the rest of it. That is what a
    sampler drawing exactly from the target would read at this n, and the observed W1
    divided by it is a number with an absolute meaning on any molecule, any dimension.

    A RATIO, NOT A DIFFERENCE, and that is the whole point. `wass_debiased` subtracts its
    null, which removes the offset but keeps the units -- so the value still has no scale,
    and eval/evaluations.py documents the consequence: dead latent rows scale BOTH terms by
    ~0.933 and the subtraction cannot cancel it, leaving a threshold that drifts with space
    group. Division cancels any multiplicative distortion exactly.

    WORST-OVER-COLUMNS, because sliced-Wasserstein pools over random projections and a
    sampler badly wrong on six columns out of 87 averages away to nothing. The worst column
    is the one that catches a collapsed or over-broad marginal.

    THE RATIO IS n-DEPENDENT, deliberately surfaced rather than hidden: the floor falls as
    ~n^-1/2 while a real discrepancy does not, so the same distribution reads a larger ratio
    at larger n. It behaves like a significance measure, not an effect size. Anything gating
    on it must pin the eval sample size, or the bar tightens silently as n moves.
    """
    n = int(samples.shape[0])
    n_ref = int(reference.shape[0])
    if n < 32 or n_ref < 4 * n:      # too few rows for a floor whose noise isn't the signal
        return {}
    if cache is None:
        cache = {}
    floor = cache.get(n)
    if floor is None:
        rng = np.random.default_rng(0)
        acc = []
        for _ in range(reps):
            idx = rng.permutation(n_ref)
            # DISJOINT: the n-row draw is scored against the reference rows it is not in,
            # so the floor is not deflated by comparing a sample against itself.
            acc.append(_column_w1(reference[idx[:n]], reference[idx[n:]], periodic))
        floor = np.median(np.stack(acc), axis=0)
        cache[n] = floor
    # SELF-CALIBRATION. The perfect-sampler value of the ratio is NOT 1.0 for every
    # statistic: the median is (~0.90 measured), but the MAX over 87 noisy columns is an
    # extreme value and sits near 3.6. Assuming 1.0 would make `worst` permanently
    # unreachable and any gate on it never fire. So score a genuine dataset draw through
    # the identical code path and keep what it gets -- these are the targets.
    cal = cache.get(('cal', n))
    if cal is None:
        rng = np.random.default_rng(1)
        idx = rng.permutation(n_ref)
        w_perf = _column_w1(reference[idx[:n]], reference, periodic)
        lv = floor > 1e-9
        rp = w_perf[lv] / floor[lv]
        cal = (float(np.median(rp)), float(rp.max()))
        cache[('cal', n)] = cal
    perf_med, perf_worst = cal

    obs = _column_w1(samples, reference, periodic)
    live = floor > 1e-9              # a column constant in the reference has no scale to
    if not bool(live.any()):         # divide by, and its ratio would be noise/0
        return {}
    ratio = obs[live] / floor[live]
    cols = np.flatnonzero(live)
    worst = int(cols[int(np.argmax(ratio))])

    # EFFECT SIZE: the same W1 over the reference column's own SPREAD rather than over the
    # sampling noise. n-independent, and it is what the eye compares -- 0.15 means "off by
    # 15% of the distribution's width". The ratio above answers "can I tell these apart",
    # which at large n is yes for almost anything; this answers "does it matter".
    #
    # GUARDED DENOMINATOR, and the guard is load-bearing: a column the reference holds
    # nearly fixed (IQR -> 0) would otherwise produce an enormous meaningless number and
    # take over the ranking -- the same failure as dividing by a vanishing floor, with the
    # sign flipped. Columns that hit the guard are counted so the guard cannot hide.
    iqr = np.subtract(*np.percentile(reference, [75, 25], axis=0))[live]
    guard = 0.02                      # 1% of the [-1, 1] state domain
    den = np.maximum(iqr, guard)
    eff = obs[live] / den
    eworst = int(cols[int(np.argmax(eff))])
    return {'w1r/worst': float(ratio.max()),
            'w1r/p90': float(np.percentile(ratio, 90)),
            'w1r/median': float(np.median(ratio)),
            'w1r/worst_col': float(worst),
            'w1r/n_above_2x': float((ratio > 2.0).sum()),
            'w1r/n_live': float(live.sum()),
            # the perfect-sampler values of the two headline stats: the TARGETS, measured
            # rather than assumed, so a gate can be written against them on any molecule
            'w1r/perfect_median': perf_med,
            'w1r/perfect_worst': perf_worst,
            # effect size
            'w1e/worst': float(eff.max()),
            'w1e/median': float(np.median(eff)),
            'w1e/p90': float(np.percentile(eff, 90)),
            'w1e/worst_col': float(eworst),
            'w1e/n_guarded': float((iqr < guard).sum()),
            # context, so a moving ratio can be attributed to the numerator or the floor
            'w1r/w1_worst': float(obs[worst]),
            'w1r/floor_at_worst': float(floor[worst]),
            'w1r/n_eval': float(n)}


#: Default progress-gate spec. Every target is READ FROM ANOTHER METRIC rather than written
#: here -- w1r/perfect_median is what a real dataset draw scores through the same code, and
#: E/ref_median is the dataset's own energy -- so these bars carry to another molecule
#: unchanged. bwd/mle appears only under veto_only because it has no usable target.
_PROGRESS_GATE = {
    'horizon': 10000,        # N: project the gain over this many steps. FREE -- it sets the
                             # question, not the estimator, so 1k or 100k are equally valid.
    'window': 20000,         # slope-fit span. Sets BOTH the noise on X and the latency after
                             # a real plateau (~= window). See projected() for the numbers.
    'level_window': 2500,    # short median-smoothing for the LEVEL test, so a molecule that
                             # converges in ~3k steps can exit at ~3k rather than waiting out
                             # the slope window it never needed
    'min_history': 2000,
    'rate_bar': 0.02,        # X: exit when the projected gain over N falls under this
    'veto_rate': 0.05,
    'metrics': [
        {'key': 'w1r/median', 'target_key': 'w1r/perfect_median', 'bar': 1.5},
        {'key': 'E/sample_median', 'target_key': 'E/ref_median', 'bar': 2.0},
        {'key': 'E/emarg_w1_kT', 'target': 0.0, 'bar': 0.5},
    ],
    # Veto entries carry targets like everything else. bwd/mle is deliberately absent: no
    # target exists for it, and the target-free normalisation it used to get was what put an
    # absolute 200k step scale into the gate.
    'veto_metrics': [
        {'key': 'w1r/worst', 'target_key': 'w1r/perfect_worst'},
    ],
}


def progress_gate(history, spec, step):
    """Should this stage exit? Judged on RATE OF IMPROVEMENT, not on a slope threshold.

    WHY NOT A SLOPE GATE. `gates/mle_flat` tests the slope of bwd/mle against a fixed bar,
    and that is unfixable in principle for two compounding reasons. bwd/mle carries an
    unknown additive constant H(p_data), so no value of it means anything absolutely; and a
    slowly-descending curve satisfies ANY slope bar eventually, so "slow" and "converged"
    come apart permanently. Measured here: the rate decayed 1.0 -> 0.04 nats/100 steps while
    the loss fell in a visibly straight line, and it fired at 0.04 -- correctly by its own
    definition, and far too early.

    WHAT REPLACES IT. Every metric below has a KNOWN TARGET, so the remaining gap
    R = value - target is computable, and the question becomes "what fraction of R is being
    removed per window". That number is scale-free, comparable across metrics, and directly
    decision-relevant: "5% of the remaining gap per 10k steps" answers whether more compute
    is worth spending. It also degrades correctly -- if a metric has no true asymptote the
    rate simply falls and the exit becomes economic rather than a claim about convergence.

    TARGETS ARE MEASURED, NOT WRITTEN. `target_key` reads the target from another metric in
    the same dict -- w1r/perfect_median is what a genuine dataset draw scores through the
    identical code path, E/ref_median is the dataset's own energy. So the bars transfer to
    any molecule without retuning, which no absolute threshold does.

    TWO WAYS OUT, REPORTED SEPARATELY, because they demand opposite responses:
      converged  every metric is inside its bar     -> the fit is good
      saturated  every rate has fallen below rate_bar -> the fit has stopped improving,
                 wherever it happens to be. On this system that may be well short.

    AND A VETO. If ANY tracked metric -- including ones with no usable target, like bwd/mle
    -- is still improving faster than veto_rate, do not exit regardless of the above. A veto
    only has to be conservative, which is the one role an arbitrary threshold can safely
    play; a trigger has to be right, which is why bwd/mle appears here and nowhere else.
    """
    # `progress/reason` is the ONLY verdict channel; a retired 'progress/verdict'
    # key used to ride the early returns alone, so the channel appeared before
    # min_history and vanished the moment the gate went live (toy_wk_aug24).
    if step < float(spec.get('min_history', 5000)):
        return {'gates/progress_done': 0.0, 'progress/reason': 0.0}

    horizon = float(spec.get('horizon', 10000))       # N: the projection horizon
    window = float(spec.get('window', 20000))          # slope-fit span: NOISE + LATENCY

    def _series(key):
        h = history.get(key) or []
        if len(h) < 6:
            return None, None
        return (np.array([a for a, _ in h], dtype=float),
                np.array([b for _, b in h], dtype=float))

    level_w = float(spec.get('level_window', 2500))

    def level(key, target):
        """Remaining gap NOW, median-smoothed over a short trailing window.

        THE LEVEL TEST NEEDS NO RATE, and keeping it independent is what lets an easy
        molecule exit early. A problem that genuinely converges at step 3k has every metric
        inside its bar at 3k; making that verdict wait for a 20k slope window would burn 17k
        steps proving something already true. Only the SATURATED branch -- "has it stopped
        improving, wherever it stopped" -- needs a rate, because that question is about a
        derivative.

        MEDIAN, not last value: a single noisy eval dipping under the bar should not end a
        stage, and a median over a handful of evals costs a few thousand steps of latency
        rather than a full window.
        """
        t, v = _series(key)
        if t is None:
            return None
        hi = float(t.max())
        m = t >= hi - level_w
        if m.sum() < 3:
            return None
        return float(np.median(v[m])) - float(target)

    def projected(key, target):
        """Fraction of the remaining gap expected to close over the NEXT `horizon` steps.

        THE PROJECTION IS WHAT MAKES N FREE. Fit the local decay of the remaining gap, then
        report 1 - exp(slope * N). Because the fit yields a rate per step, N is a property
        of the QUESTION and not of the estimator -- 1k, 10k and 100k are all answerable off
        one fit, and X can be set against any of them. The earlier design quoted the rate
        over its own averaging window, which forced the two to be the same number and made
        a 2% bar require a 20k window comparable to the decay constant itself.

        HARD WINDOW, NOT EXPONENTIAL WEIGHTING, and this was measured the other way first.
        Recency weighting has lower noise per unit of nominal scale (sd 2.0 at h=10k against
        2.5 for a 20k window), but its tail never fully forgets, so after a genuine plateau
        it takes ~46k steps to register against ~20k for the window -- 2.3x the latency for
        25% less noise. On a gate, latency is wasted compute, so the window wins.

        X AND `window` ARE LINKED, and there is no way around it: the projected-rate sd at
        N = 10k is ~5.0 points on a 10k window and ~2.5 on a 20k one. A 2% bar therefore
        needs ~20k of window to sit outside the noise; a 10% bar is comfortable at 10k. The
        third lever is the metric itself -- the scatter is inherited from eval-to-eval noise,
        so a larger eval sample tightens X without lengthening the window at all.

        Returns (remaining_now, projected_fraction). A metric getting WORSE yields a negative
        fraction rather than being clipped: not progress, and not the same as having stalled.
        """
        t, v = _series(key)
        if t is None or t.max() - t.min() < window:
            return None, None
        R = v - float(target)
        hi = float(t.max())
        m = (t >= hi - window) & (R > 1e-12)
        if m.sum() < 6:
            return None, None
        slope = float(np.polyfit(t[m], np.log(R[m]), 1)[0])
        return float(np.interp(hi, t, R)), float(1.0 - np.exp(slope * horizon))

    out, in_bar, stalled, vetoed, degrading = {}, [], [], False, False
    for m in spec.get('metrics', []):
        key = m['key']
        tk = m.get('target_key')
        tgt = None
        if tk:
            h = history.get(tk) or []
            if h:
                tgt = float(h[-1][1])
        if tgt is None:
            tgt = m.get('target')
        if tgt is None:
            continue
        # LEVEL -- available almost immediately, and sufficient on its own for CONVERGED
        r_now = level(key, tgt)
        if r_now is None:
            return {'gates/progress_done': 0.0, 'progress/reason': 0.0}
        out[f'progress/{key}/remaining'] = r_now
        in_bar.append(r_now <= float(m['bar']))

        # RATE -- needs the full slope window. Its ABSENCE is not evidence of stalling, so
        # a metric with no usable rate blocks only the SATURATED verdict, never the level one
        _, proj = projected(key, tgt)
        if proj is None:
            stalled.append(False)
        else:
            out[f'progress/{key}/rate'] = proj
            stalled.append(proj < float(spec.get('rate_bar', 0.02)))
            # A metric moving BACKWARDS is not the same event as one that has stopped, even
            # though both sit under the bar. Simulated sustained degradation exits with proj
            # still slightly POSITIVE -- the decay and the degradation cancel -- so this is
            # rarely reached; but when it is, "saturated" would read as finished rather than
            # as going backwards, and those want opposite responses.
            if proj < -float(spec.get('rate_bar', 0.02)):
                degrading = True
            if proj > float(spec.get('veto_rate', 0.05)):
                vetoed = True

    # VETO ENTRIES NEED A TARGET, exactly like the trigger metrics, and this replaced a
    # normalisation that silently broke the whole gate. A quantity with no target has no
    # scale-free rate, so the previous version divided recent progress by TOTAL progress so
    # far -- which for any steadily-moving series is horizon/t, releasing the veto at
    # t = horizon/veto_rate = 200,000 steps on every problem regardless of its timescale.
    # Simulated, a molecule that stalled at step 3,000 was held to 200,000: 197k steps of
    # pure waste, and an absolute step scale smuggled into the one place meant to be free
    # of them. bwd/mle is therefore NOT a veto any more -- it cannot have a target, and its
    # role is covered by metrics that do.
    for m in spec.get('veto_metrics', []):
        tk = m.get('target_key')
        tgt = None
        if tk:
            h = history.get(tk) or []
            if h:
                tgt = float(h[-1][1])
        if tgt is None:
            tgt = m.get('target')
        if tgt is None:
            continue
        _, proj = projected(m['key'], tgt)
        if proj is None:
            continue
        out[f'progress/{m["key"]}/rate'] = proj
        if proj > float(spec.get('veto_rate', 0.05)):
            vetoed = True

    if not in_bar:
        return {'gates/progress_done': 0.0, 'progress/reason': 0.0}
    converged = all(in_bar)
    saturated = all(stalled)
    # THE VETO GUARDS `saturated` ALONE. "Do not call it stalled while it is still moving"
    # is coherent; "do not call it good while it is still improving" is not -- a metric
    # inside its bar is done whether or not it would keep improving given more steps.
    done = converged or (saturated and not vetoed)
    out['gates/progress_done'] = float(done)
    out['progress/vetoed'] = float(vetoed)
    out['progress/degrading'] = float(degrading)
    # 1 = converged (good) | 2 = saturated (stopped, possibly short) | 3 = DEGRADING
    # (stopped because it is getting worse). Three different situations demanding three
    # different responses, so the gate names which one rather than just advancing.
    out['progress/reason'] = float(
        1 if (done and converged) else (3 if (done and degrading) else (2 if done else 0)))
    return out

