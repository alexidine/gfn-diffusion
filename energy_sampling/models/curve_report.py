"""Read the training curves, not just the final score.

WHY THIS EXISTS. A summary statistic cannot distinguish the three ways a run goes wrong, and
this project has twice shipped a conclusion built on one that could not. The 2026-09-01
collapse -- model learns for ~1,500 steps, then snaps back to knowing nothing and stays there
-- was INVISIBLE in a final score and unmistakable in a curve. "Message passing cannot do
distances, 46%" was that wreck being read out as an architecture result.

THE THREE SHAPES, and what separates them:

    UNDER-TRAINED   loss still falling at the last step. The run was cut short; nothing about
                    the model is established. Fix: more steps.
    MEMORISING      train falls by orders of magnitude while HELD-OUT IS FLAT. Neither of the
                    two flags above fires, so it reads as healthy-ish while the model stores
                    answers instead of learning the rule. Fix: MORE DATA, not more steps.
    OVER-FIT        train loss falling while HELD-OUT loss rises. Looks perfectly healthy from
                    the training side, which is why train-only curves miss it. Fix: more data,
                    or stop earlier.
    COLLAPSED       loss rises and STAYS risen, ending materially worse than its own best.
                    Not noise, not a plateau. Fix: lower the learning rate.

The held-out trace is what makes over-fitting visible at all, so `curve` entries without a
`test_total` key can only be judged for the other two -- and are reported as such rather than
silently passed.

    python -m models.curve_report models/results/arms_fixed_recipe.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def _smooth(v: np.ndarray, w: int = 5) -> np.ndarray:
    """Rolling median. The train trace is ONE minibatch per point, so it is genuinely noisy;
    judging a trend off raw points invents both collapses and recoveries."""
    if len(v) < w:
        return v
    return np.array([np.median(v[max(0, i - w + 1):i + 1]) for i in range(len(v))])


def diagnose(steps: np.ndarray, tr: np.ndarray, te: Optional[np.ndarray]) -> Dict:
    """Verdict for one loss trace. Thresholds are deliberately loose: this is a triage
    instrument meant to tell you WHICH plot to look at, not a hypothesis test."""
    s_tr = _smooth(tr)
    best, final = float(np.min(s_tr)), float(s_tr[-1])
    span = max(float(s_tr[0]) - best, 1e-12)
    regress = (final - best) / span

    tail = max(2, len(s_tr) // 5)                       # last 20% of the run
    head_of_tail = float(s_tr[-tail])
    still_falling = (head_of_tail - final) / max(abs(head_of_tail), 1e-12)

    out = {'train_best': best, 'train_final': final, 'regress_frac': regress,
           'tail_improvement': still_falling, 'flags': []}

    if regress > 0.25:
        out['flags'].append('COLLAPSED')
    if still_falling > 0.02 and regress <= 0.25:
        out['flags'].append('UNDER-TRAINED')

    if te is None:
        out['flags'].append('no-heldout-trace')
    else:
        s_te = _smooth(te)
        te_best, te_final = float(np.min(s_te)), float(s_te[-1])
        out.update({'test_best': te_best, 'test_final': te_final,
                    'test_regress': (te_final - te_best) / max(abs(te_best), 1e-12),
                    'gen_gap': te_final / max(final, 1e-12)})
        # over-fitting: held-out loss materially off its own minimum while train did NOT
        # collapse. Both rising together is a collapse, not over-fitting -- different fix.
        # OVER-FIT needs the rise to PERSIST, not to be one noisy point above the minimum.
        # The first cut compared the final value to the running minimum at a 10% threshold and
        # fired on converged, flat, perfectly healthy curves -- including ones whose held-out
        # loss was BELOW their train loss (gap 0.97). It reported 0/30 traces healthy, which is
        # no more useful than reporting 30/30: a detector that always fires carries no
        # information. Compare the TAIL MEAN against the minimum instead.
        tail_te = float(np.mean(s_te[-max(2, len(s_te) // 5):]))
        out['test_tail_rise'] = (tail_te - te_best) / max(abs(te_best), 1e-12)
        # ...and held-out must actually be WORSE than train. Without this the flag fired on
        # traces with gap 0.62 -- held-out loss BELOW train loss -- which cannot be
        # over-fitting under any reading. A tail that wanders 25% off its own minimum while
        # sitting under the training curve is noise in a converged run, not memorisation.
        if out['test_tail_rise'] > 0.25 and regress <= 0.25 and out['gen_gap'] > 1.2:
            out['flags'].append('OVER-FIT')

        # MEMORISING -- the third shape, and the one that fooled me. Train keeps falling by
        # orders of magnitude while held-out is FLAT. It is not over-fitting (held-out never
        # gets worse) and not under-training (more steps will not move held-out), so both of
        # those flags read as reassuring while the model stores answers instead of learning
        # the rule. THE FIX IS MORE DATA, NOT MORE STEPS -- the opposite of what
        # UNDER-TRAINED alone would tell you to do.
        h = len(s_tr) // 2
        tr_gain = (float(s_tr[h]) - final) / max(abs(float(s_tr[h])), 1e-12)
        te_gain = (float(s_te[h]) - te_final) / max(abs(float(s_te[h])), 1e-12)
        if tr_gain > 0.50 and te_gain < 0.05:
            out['flags'].append('MEMORISING')
        # ...and the ROBUST version of the same signal. The criterion above only sees
        # memorisation that happens in the second half; chiral_moment and formula memorised
        # EARLY, so their train curve was already flat by the midpoint and the flag missed
        # them entirely. The held-out/train RATIO does not care when it happened.
        if out['gen_gap'] > 10.0:
            out['flags'].append('HIGH-GAP x%.0f' % out['gen_gap'])
        out['train_gain_2nd_half'] = tr_gain
        out['test_gain_2nd_half'] = te_gain

    if not out['flags']:
        out['flags'].append('healthy')
    return out


def _traces(curve: List[dict], task: Optional[str]):
    steps = np.array([c['step'] for c in curve], dtype=float)
    if task is None:
        tr = np.array([c['total'] for c in curve], dtype=float)
        te = ([c['test_total'] for c in curve] if 'test_total' in curve[0] else None)
    else:
        tr = np.array([c['parts'][task] for c in curve], dtype=float)
        te = ([c['test_parts'][task] for c in curve] if 'test_parts' in curve[0] else None)
    return steps, tr, (np.array(te, dtype=float) if te is not None else None)


def plot(runs: List[dict], out_png: str, title: str = '') -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tasks = sorted(runs[0]['curve'][0]['parts'].keys())
    panels = ['TOTAL'] + tasks
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.2 * nrow), squeeze=False)
    colours = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, name in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        task = None if name == 'TOTAL' else name
        for j, r in enumerate(runs):
            steps, tr, te = _traces(r['curve'], task)
            lab = str(r.get('arm', '?')) + ' s' + str(r.get('seed', '?'))
            c = colours[j % 10]
            ax.plot(steps, np.maximum(tr, 1e-8), color=c, lw=1.1, alpha=0.85, label=lab)
            if te is not None:
                ax.plot(steps, np.maximum(te, 1e-8), color=c, lw=1.1, ls='--', alpha=0.85)
        ax.set_yscale('log')
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.25, which='both', lw=0.4)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=6, ncol=2)
            ax.set_xlabel('solid = train,  dashed = held out', fontsize=7)

    for k in range(len(panels), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle(title or os.path.basename(out_png), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def report(path: str, out_png: Optional[str] = None) -> List[dict]:
    with open(path) as f:
        blob = json.load(f)
    runs = blob['rows'] if 'rows' in blob else blob.get('runs', blob)
    runs = [r for r in runs if r.get('curve')]
    if not runs:
        raise SystemExit('no curves in ' + path + ' -- was it written by an older run?')

    tasks = sorted(runs[0]['curve'][0]['parts'].keys())
    print(str(len(runs)) + ' runs from ' + os.path.basename(path) + '\n')
    hdr = ('run'.ljust(22) + 'task'.ljust(16) + 'train'.rjust(10)
           + 'held out'.rjust(11) + 'gap'.rjust(7) + '  verdict')
    print(hdr)
    print('-' * len(hdr))
    rows = []
    for r in runs:
        tag = str(r.get('arm', '?')) + ' s' + str(r.get('seed', '?')) + ' n' + str(r.get('n_train', ''))
        for name in ['TOTAL'] + tasks:
            steps, tr, te = _traces(r['curve'], None if name == 'TOTAL' else name)
            d = diagnose(steps, tr, te)
            d.update(run=tag, task=name)
            rows.append(d)
            gap = ('%.2f' % d['gen_gap']) if 'gen_gap' in d else '-'
            te_s = ('%.4f' % d['test_final']) if 'test_final' in d else '-'
            print(tag.ljust(22) + name.ljust(16) + ('%.4f' % d['train_final']).rjust(10)
                  + te_s.rjust(11) + gap.rjust(7) + '  ' + ', '.join(d['flags']))
        print()

    # FILTER FLAGS, NOT ROWS. The first cut of this dropped any row carrying
    # 'no-heldout-trace', which silently swallowed the UNDER-TRAINED verdict sitting beside it
    # and printed "10/10 healthy" over eight flagged traces -- the exact failure this tool was
    # built to stop, committed by the tool itself.
    NON_FAULTS = {'healthy', 'no-heldout-trace'}
    bad = [r for r in rows if set(r['flags']) - NON_FAULTS]
    print(str(len(rows) - len(bad)) + '/' + str(len(rows)) + ' traces healthy; '
          + str(len(bad)) + ' flagged' + (':' if bad else ''))
    for r in bad:
        print('    ' + r['run'].ljust(22) + r['task'].ljust(16) + ', '.join(r['flags']))

    png = out_png or os.path.join(RESULTS, 'curves',
                                  os.path.basename(path).replace('.json', '.png'))
    print('\nwrote ' + plot(runs, png, os.path.basename(path)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('path')
    ap.add_argument('--png', default=None)
    a = ap.parse_args(argv)
    report(a.path, a.png)


if __name__ == '__main__':
    main()
