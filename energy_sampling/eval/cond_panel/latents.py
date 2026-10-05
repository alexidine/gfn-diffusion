"""
The 12 crystal latents of the Z' = 1 route as the per-condition analyses read them.

Layout (docs/wiki/latent-dimension-structure.md): rows 0:3 the normalised asymmetric-unit
lengths, 3:6 the cell angles, 6:9 the asymmetric-unit centroid, 9:12 the orientation
(theta, phi, r). On the P-1 route the model wraps v, w, phi and r with period 2.

ORIGIN IMAGES. In P-1 the asymmetric-unit box is (1/2, 1, 1) of the cell, so a shift of the
origin by half a cell along b or c is a latent shift of 1 in v or w and describes the same
crystal. Two conditions whose draws favour different images would differ in v and w for no
physical reason, so `fold` maps v and w onto period 1. The half-cell shift along a is not a
latent shift in this chart and is left alone.

Distances between distributions are taken on the folded latents, each coordinate in units of
its own range, so a periodic coordinate cannot outweigh a linear one by being wider.
"""
from __future__ import annotations

import numpy as np

NAMES = ('a', 'b', 'c', 'alpha', 'beta', 'gamma', 'u', 'v', 'w', 'theta', 'phi', 'r')
BLOCKS = {'cell lengths': (0, 1, 2), 'cell angles': (3, 4, 5), 'centroid': (6, 7, 8), 'orientation': (9, 10, 11)}
# period of each coordinate AFTER folding; 0 marks a linear coordinate on [-1, 1]
PERIOD = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2], dtype=np.float64)
RANGE = np.where(PERIOD > 0, PERIOD, 2.0)  # the width a coordinate can span


def fold(x):
    """v and w onto [-1/2, 1/2): one origin image of the P-1 crystal. numpy or torch."""
    out = x.copy() if isinstance(x, np.ndarray) else x.clone()
    for k in (7, 8):
        out[..., k] = (out[..., k] + 0.5) % 1.0 - 0.5
    return out


def delta(a, b):
    """Coordinate-wise a - b on folded latents, nearest image on periodic coordinates, in
    units of each coordinate's range (so every entry lies in [-1/2, 1/2] or [-1, 1])."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    per = PERIOD > 0
    d[..., per] = (d[..., per] + PERIOD[per] / 2) % PERIOD[per] - PERIOD[per] / 2
    return d / RANGE


def circ_mean_spread(x):
    """Per coordinate of folded draws [n, 12]: a centre and a spread, both in units of the
    coordinate's range. Linear rows: mean and standard deviation. Periodic rows: the circular
    mean and the circular standard deviation sqrt(-2 ln R), which equals the linear standard
    deviation for a narrow distribution and grows without bound as the draws go uniform."""
    x = np.asarray(x, dtype=np.float64)
    centre, spread = np.empty(12), np.empty(12)
    for k in range(12):
        if PERIOD[k] > 0:
            ang = 2 * np.pi * x[:, k] / PERIOD[k]
            c, s = np.cos(ang).mean(), np.sin(ang).mean()
            centre[k] = np.arctan2(s, c) / (2 * np.pi) * PERIOD[k] / RANGE[k]
            spread[k] = np.sqrt(max(-2.0 * np.log(max(np.hypot(c, s), 1e-12)), 0.0)) / (2 * np.pi)
        else:
            centre[k] = x[:, k].mean() / RANGE[k]
            spread[k] = x[:, k].std() / RANGE[k]
    return centre, spread


def _circular_w1(a, b, period):
    """Exact W1 between two empirical measures on a circle: the integral of |F - G - m| over
    one period, m the length-weighted median of the CDF difference F - G (the optimal cut)."""
    pts = np.concatenate([a % period, b % period])
    step = np.concatenate([np.full(len(a), 1.0 / len(a)), np.full(len(b), -1.0 / len(b))])
    order = np.argsort(pts)
    pts, diff = pts[order], np.cumsum(step[order])       # F - G just after each point
    seg = np.diff(np.concatenate([pts, [pts[0] + period]]))  # the last segment wraps
    by_value = np.argsort(diff)
    cum = np.cumsum(seg[by_value])
    median = diff[by_value][np.searchsorted(cum, cum[-1] / 2.0)]
    return float((seg * np.abs(diff - median)).sum())


def w1_per_dim(x, y):
    """1-D Wasserstein distance between two sets of folded draws, per coordinate, in units of
    the coordinate's range; the circular W1 on periodic rows."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    out = np.empty(12)
    for k in range(12):
        if PERIOD[k] > 0:
            out[k] = _circular_w1(x[:, k], y[:, k], PERIOD[k]) / RANGE[k]
        else:
            qs = np.linspace(0.0, 1.0, max(len(x), len(y)) + 1)[1:] - 0.5 / max(len(x), len(y))
            out[k] = np.abs(np.quantile(x[:, k], qs) - np.quantile(y[:, k], qs)).mean() / RANGE[k]
    return out


def energy_distance(x, y):
    """Energy distance between two sets of folded draws under the torus-aware metric
    ||delta(a, b)||_2: zero only for equal distributions, and sensitive to the joint
    structure the per-coordinate W1 cannot see. The within-set terms leave out each draw's
    distance to itself, so the estimate is unbiased: it averages zero for two samples of
    one distribution whatever their size, and can come out slightly negative."""
    def cross(p, q):
        d = delta(p[:, None, :], q[None, :, :])
        return np.sqrt((d ** 2).sum(-1))

    def within(p):
        return cross(p, p).sum() / (len(p) * (len(p) - 1))
    return 2 * cross(x, y).mean() - within(x) - within(y)
