"""
One look for the figures of the per-condition suite, and one place that writes them.

Colours are the validated three-slot categorical set (blue, orange, aqua: all pairs pass the
colour-vision checks on this surface), one blue ramp for magnitude and a blue-grey-red ramp
for signed quantities. Every figure is saved with its caption and the numbers it shows
(`<name>.png`, `<name>.csv`, and a row in `figures.json`), so a reading never rests on colour
or on a mark's position alone.
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

TRAIN, HELD, THIRD = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div_blue_red', ['#104281', '#3987e5', '#f0efec', '#e34948', '#8f1f1f'])

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'DejaVu Sans'], 'font.size': 9,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK, 'axes.titlesize': 10, 'axes.titlecolor': INK,
    'axes.linewidth': 0.8, 'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2,
    'ytick.labelcolor': INK2, 'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
    'legend.labelcolor': INK, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'lines.linewidth': 2.0, 'lines.markersize': 5,
})


def style(ax, grid='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, axis=grid)


def panels(n, ncols=3, width=4.0, height=3.2, **kw):
    """A grid of `n` styled axes; unused cells are removed."""
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False, **kw)
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.remove()
    return fig, list(axes[:n])


class Figures:
    """Writes each figure with its caption and table, and keeps the list a report reads."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.index_path = os.path.join(out_dir, 'figures.json')
        self.items = json.load(open(self.index_path)) if os.path.exists(self.index_path) else []

    def save(self, fig, name, title, caption, table=None, points=None):
        """`table` is (header, rows): the numbers the figure shows. `points` are short
        findings read off it, each a plain statement with its number."""
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, f'{name}.png'), dpi=130)
        plt.close(fig)
        if table is not None:
            with open(os.path.join(self.out_dir, f'{name}.csv'), 'w', newline='', encoding='utf-8') as fh:
                w = csv.writer(fh)
                w.writerow(table[0])
                w.writerows(table[1])
        item = {'name': name, 'title': title, 'caption': caption, 'points': points or [],
                'table': None if table is None else {'header': list(table[0]), 'rows': [list(r) for r in table[1]]}}
        self.items = [i for i in self.items if i['name'] != name] + [item]
        with open(self.index_path, 'w', encoding='utf-8') as fh:
            json.dump(self.items, fh, indent=1, default=float)
        print(f'[{name}] {title}', flush=True)
        for p in item['points']:
            print(f'    - {p}', flush=True)
