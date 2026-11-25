"""Figures for the exp-6 calibration (docs/design/exp6_calibration_plan.md).

parity   per validation molecule (the fit's held-out molecules) and pooled over the fit molecules: exp-6 energy against
         the MLIP energy of every usable frame, for FIT and for the tuned file. Both relative to the molecule's lowest
         MLIP frame; the exp-6 axis is shifted by the fit's weighted mean residual (a molecule's constant offset is not
         fitted). Frames as in fit.Frames (labelled, contact >= 1.2 A, within 80 kJ/mol); drawn to 40 kJ/mol.
ladder   per panel molecule, every pooled MLIP basin (compare.report's reference) up to 20 kJ/mol above the lowest,
         with the arms that found it and, from roundtrip.json, whether each exp-6 field keeps it.

usage: python -m eval.exp6_calibration.plots OUT TUNED_JSON {parity,ladder,all}
Writes OUT/fig_parity.png and OUT/fig_ladder.png. Tuned energies of the frames are cached in OUT/frames_tuned.pt.
"""
import argparse
import json
import os

import numpy as np
import torch

from eval.exp6_calibration import common as C
from eval.exp6_calibration import fit as F

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SOURCES = {'exp6_traj': ('#2a78d6', 'o', 'exp-6 trajectory'), 'exp6_min': ('#eb6834', 's', 'exp-6 minimum'),
           'mace_csp_traj': ('#1baf7a', '^', 'MACE-CSP trajectory'),
           'mace_csp_min': ('#eda100', 'D', 'MACE-CSP minimum')}
ARMS = {'R': ('#2a78d6', 'MACE-CSP search (R)'), 'U': ('#eb6834', 'FIT exp-6 + MACE-CSP (U)'),
        'T': ('#1baf7a', 'tuned exp-6 + MACE-CSP (T)')}
SHOW_KJ = 40.0


def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def tuned_energies(out, tuned, frames_path):
    """Tuned exp-6 energy (kJ/mol per molecule) at every frame, through StageEnergy with the parameter file."""
    path = os.path.join(out, 'frames_tuned.pt')
    if os.path.exists(path):
        return torch.load(path)
    from mxtaltools.crystal_search import minima as mn
    d = torch.load(frames_path, weights_only=False)
    fr, tm = d['frames'], d['templates']
    e = torch.full((len(fr['name']),), float('nan'), dtype=torch.float64)
    for name in dict.fromkeys(fr['name']):
        idx = [i for i, n in enumerate(fr['name']) if n == name]
        x = fr['params'][idx].float()
        rows = C.rows_at(tm[name], x, fr['hand'][idx])
        se = mn.StageEnergy(rows, 'exp6', 1.0, exp6_kwargs={'params': tuned, 'charges': 'mmff'}, device='cpu',
                            chunk=64)
        with torch.no_grad():
            for s in range(0, len(rows), 64):
                sl = list(range(s, min(s + 64, len(rows))))
                e[[idx[j] for j in sl]] = se.f(x[sl], sl).double()
    torch.save(e, path)
    return e


def parity(out, tuned):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec = json.load(open(tuned))
    frames_path = spec['provenance']['frames']
    fr = F.Frames(frames_path, 'cpu', spec['provenance']['args']['val_frac'])
    assert fr.val_mols == spec['provenance']['validation_molecules'], 'validation split differs from the fit'
    e_t = tuned_energies(out, tuned, frames_path)
    rows = [(m, [m]) for m in fr.val_mols] + [(f'{len(fr.fit_mols)} fit molecules, pooled', fr.fit_mols)]
    fig, axes = plt.subplots(len(rows), 2, figsize=(8.2, 3.3 * len(rows)), facecolor=SURF, squeeze=False)
    for r, (title, mols) in enumerate(rows):
        for c, (label, e6) in enumerate((('published FIT', fr.e_fit), ('tuned', e_t))):
            ax = axes[r, c]
            style(ax)
            lows = []
            for m in mols:
                idx = fr.of(m)
                em = fr.e_mlip[idx].numpy() - fr.emin[m]
                d = e6[idx].numpy() - fr.e_mlip[idx].numpy()
                w = np.maximum(np.exp(-em / F.T_W), F.W_FLOOR)
                y = em + d - (w * d).sum() / w.sum()
                low = em < 10.0
                if low.any():
                    dl = (y - em)[low]
                    lows.append(np.sqrt(((dl - dl.mean()) ** 2).mean()))
                src = fr.source[idx].astype(str)
                show = em <= SHOW_KJ
                for s, (col, mk, name) in SOURCES.items():
                    k = show & (src == s)
                    if k.any():
                        ax.scatter(em[k], y[k], s=14, c=col, marker=mk, linewidths=0.4, edgecolors=SURF, zorder=3)
            ax.plot([0, SHOW_KJ], [0, SHOW_KJ], color=INK2, linewidth=1, linestyle='--', zorder=2)
            ax.set_xlim(-1, SHOW_KJ)
            ax.set_ylim(-10, SHOW_KJ + 10)
            ax.set_title(f'{title}: {label}', fontsize=9, color=INK, loc='left')
            ax.text(0.98, 0.04, f'RMS below 10 kJ/mol: {np.median(lows):.1f} kJ/mol' +
                    (' (median)' if len(lows) > 1 else ''), transform=ax.transAxes, ha='right', fontsize=8,
                    color=INK2)
            if r == len(rows) - 1:
                ax.set_xlabel('MACE-CSP energy above lowest frame (kJ/mol)', fontsize=8, color=INK2)
            if c == 0:
                ax.set_ylabel('exp-6 energy, aligned (kJ/mol)', fontsize=8, color=INK2)
    from matplotlib.lines import Line2D
    hand = [Line2D([], [], marker=mk, linestyle='', color=col, label=name) for col, mk, name in SOURCES.values()]
    fig.legend(handles=hand, loc='upper center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('Exp-6 against MACE-CSP on fit frames (top rows: molecules held out of the fit)', fontsize=10,
                 color=INK, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p = os.path.join(out, 'fig_parity.png')
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=SURF)
    print('wrote', p)


def ladder(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rt = json.load(open(os.path.join(out, 'roundtrip.json')))
    names = list(rt)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 5.2), facecolor=SURF, sharey=True,
                             squeeze=False)
    for ax, name in zip(axes[0], names):
        style(ax)
        ax.grid(axis='x', visible=False)
        for b in rt[name]['basins']:
            if b['e_kJ'] > 20:
                continue
            y = b['e_kJ']
            ax.plot([0.2, 3.8], [y, y], color=GRID, linewidth=1, zorder=1)
            for k, a in enumerate(('R', 'U', 'T')):
                if b['found'][a]:
                    ax.scatter(k + 1, y, s=46, c=ARMS[a][0], marker='o', edgecolors=SURF, linewidths=1, zorder=3)
            for k, a in ((2, 'U'), (3, 'T')):  # round trip of an in-window basin: ring = kept, cross = lost
                t = b.get(f'rt_{a}')
                if t is None:
                    continue
                if not isinstance(t, dict):  # a relax returned no minimum: undetermined, not lost
                    ax.scatter(k + 0.32, y, s=46, marker='$?$', color=INK2, zorder=4)
                    continue
                kept = t['kept']
                ax.scatter(k + 0.32, y, s=40, marker='o' if kept else 'x', facecolors='none' if kept else INK,
                           edgecolors=INK if kept else None, linewidths=1.1, zorder=4)
        ax.axhline(10, color=INK2, linewidth=0.8, linestyle=':')
        ax.set_xticks([1, 2, 3], ['R', 'U', 'T'], fontsize=8, color=INK2)
        ax.set_xlim(0.3, 3.9)
        ax.set_title(name, fontsize=9, color=INK, loc='left')
    axes[0][0].set_ylabel('MACE-CSP basin energy above the lowest found (kJ/mol)', fontsize=8, color=INK2)
    hand = [Line2D([], [], marker='o', linestyle='', color=ARMS[a][0], label=f'found by {ARMS[a][1]}')
            for a in ARMS]
    hand += [Line2D([], [], marker='o', linestyle='', markerfacecolor='none', markeredgecolor=INK,
                    label='round trip keeps the basin'),
             Line2D([], [], marker='x', linestyle='', color=INK, label='round trip loses it'),
             Line2D([], [], marker='$?$', linestyle='', color=INK2, label='a relax returned no minimum')]
    fig.legend(handles=hand, loc='upper center', ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Pooled MACE-CSP basins per held-out molecule; round trips for those within 10 kJ/mol (dotted)',
                 fontsize=9.5, color=INK, y=1.08)
    fig.tight_layout()
    p = os.path.join(out, 'fig_ladder.png')
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=SURF)
    print('wrote', p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('tuned')
    ap.add_argument('step', choices=['parity', 'ladder', 'all'])
    a = ap.parse_args()
    if a.step in ('parity', 'all'):
        parity(a.out, os.path.abspath(a.tuned))
    if a.step in ('ladder', 'all'):
        ladder(a.out)


if __name__ == '__main__':
    main()
