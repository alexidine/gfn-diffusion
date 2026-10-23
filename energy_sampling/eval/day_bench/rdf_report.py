"""Read one molecule's RDF embedding against its COMPACK identities: calibrate, then ask where samples sit.

Inputs in SAMPLE_DIR: ``rdf_embed_<energy>.npz`` (``rdf_embed``), ``compack_<energy>.csv`` and
``compack_pairs_<energy>.csv`` (``compack_endpoints``).

1. Calibration. Every pair COMPACK actually compared (relaxed endpoint against an experimental form, endpoint
   against a minimum's representative) is labelled matched when all 30 molecules overlay (Day's rule). Two
   known matches are added: each experimental form as deposited against itself relaxed, and the P1 control's
   end against both (measured by the P1 control run). The identity cut is the largest compared distance at
   which the isotonic estimate of P(match | RDF distance) is >= 0.95, ``d_hi`` the largest at which it is >=
   0.05 (``eval/campaign_compack.py`` rule), with 95% pair-bootstrap intervals.
2. Raw and brief samples against their own relaxed endpoint: how far, and whether the nearest relaxed
   endpoint already belongs to the same COMPACK minimum.
3. Experimental forms against the sample cloud at each stage.
4. Spread: COMPACK minima among the relaxed endpoints against RDF clusters at the identity cut.

usage:
    python -m eval.day_bench.rdf_report SAMPLE_DIR [energy]
"""
import csv
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from eval.campaign_compack import auc, rule_summary

SHELL = 30


def main(sample_dir, energy='exp6', nboot=500):
    z = np.load(os.path.join(sample_dir, f'rdf_embed_{energy}.npz'))
    D, kind, name = z['dist'], z['kind'].astype(str), z['name'].astype(str)
    idx = {(k, n): i for i, (k, n) in enumerate(zip(kind, name))}
    ends = {r['file']: r for r in csv.DictReader(open(os.path.join(sample_dir, f'compack_{energy}.csv')))}
    pairs = list(csv.DictReader(open(os.path.join(sample_dir, f'compack_pairs_{energy}.csv'))))
    forms = list(dict.fromkeys(name[kind == 'exp_csd']))  # rdf_embed order: the reference refcode first
    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    def node(n):
        return idx.get(('exp_relaxed', n[4:])) if n.startswith('exp_') else idx.get(('relaxed', n))

    # 1. calibration
    d, y, src = [], [], []
    for p in pairs:
        if p['matched'] in ('', 'None'):
            continue
        a, b = node(p['a']), node(p['b'])
        if a is None or b is None:
            continue
        d.append(D[a, b]); y.append(int(p['matched']) == SHELL); src.append('compared')
    for rc in forms:
        for k1, k2 in (('exp_csd', 'exp_relaxed'),):
            if (k1, rc) in idx and (k2, rc) in idx:
                d.append(D[idx[(k1, rc)], idx[(k2, rc)]]); y.append(True); src.append('known: deposited vs relaxed')
    if ('control', 'end_relaxed') in idx:
        for k in ('exp_csd', 'exp_relaxed'):
            if (k, forms[0]) in idx:
                d.append(D[idx[('control', 'end_relaxed')], idx[(k, forms[0])]]); y.append(True)
                src.append('known: P1 control end')
    d, y = np.array(d), np.array(y, dtype=int)
    rs = rule_summary(d, y, nboot)
    cut = rs['cut'][0]
    out(f'## {os.path.basename(os.path.normpath(sample_dir))}: RDF embedding against COMPACK ({energy})')
    out()
    out('**Calibration.** Pairs with a COMPACK verdict under Day\'s rule (30 of 30 molecules within 30% / 30 deg); '
        'RDF distance = mean per-channel earth mover\'s distance, envwise channels, 10 A.')
    out()
    out('| pairs (count) | matched (count) | AUC | identity cut (RDF distance, 95% CI) | d_hi (RDF distance, 95% CI) |')
    out('|---|---|---|---|---|')
    out(f'| {len(d)} ({src.count("compared")} compared, {len(d) - src.count("compared")} known) | {y.sum()} | '
        f'{auc(d, y):.3f} | {rs["cut"][0]:.3f} [{rs["cut"][1]:.3f}, {rs["cut"][2]:.3f}] | '
        f'{rs["d_hi"][0]:.3f} [{rs["d_hi"][1]:.3f}, {rs["d_hi"][2]:.3f}] |')
    out()
    if y.any():
        out(f'Matched pairs, RDF distance: max {d[y == 1].max():.3f}, median {np.median(d[y == 1]):.3f}. '
            f'Unmatched: min {d[y == 0].min():.3f}, median {np.median(d[y == 0]):.3f}.')
        out()

    # 2. raw / brief against their own relaxed endpoint
    minimum = {f: int(r['minimum']) for f, r in ends.items()}
    rel_names = [n for n in name[kind == 'relaxed'] if n in minimum]
    rel_idx = np.array([idx[('relaxed', n)] for n in rel_names])
    out('**Samples against their own relaxed endpoint.** Same sample, same reference conformer; the nearest relaxed '
        'endpoint is searched over all relaxed endpoints.')
    out()
    out('| stage | samples (count) | RDF distance to own endpoint, median [IQR] | within identity cut (fraction) | '
        'nearest endpoint in own COMPACK minimum (fraction) |')
    out('|---|---|---|---|---|')
    for stage in ('raw', 'brief', 'relaxed'):
        dd, same = [], []
        for n in rel_names:
            if (stage, n) not in idx:
                continue
            i = idx[(stage, n)]
            dd.append(D[i, idx[('relaxed', n)]])
            others = rel_idx if stage != 'relaxed' else rel_idx[rel_idx != i]
            j = others[np.argmin(D[i, others])]
            same.append(minimum[name[j]] == minimum[n])
        dd = np.array(dd)
        if len(dd):
            q = np.percentile(dd, [25, 50, 75])
            out(f'| {stage} | {len(dd)} | {q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}] | {np.mean(dd <= cut):.2f} | '
                f'{np.mean(same):.2f} |')
    out()

    # 3. experimental forms against the cloud
    out('**Experimental forms against the samples.** Nearest RDF distance from each form to each stage\'s samples, '
        'and how many lie within the identity cut.')
    out()
    out('| form | nearest raw (RDF distance) | nearest brief | nearest relaxed | relaxed within cut (count) | '
        'deposited vs relaxed form (RDF distance) |')
    out('|---|---|---|---|---|---|')
    for rc in forms:
        if ('exp_relaxed', rc) not in idx:
            continue
        i = idx[('exp_relaxed', rc)]
        row = [rc]
        for stage in ('raw', 'brief', 'relaxed'):
            js = np.nonzero(kind == stage)[0]
            row.append(f'{D[i, js].min():.3f}' if len(js) else '-')
        row.append(str(int((D[i, rel_idx] <= cut).sum())))
        row.append(f'{D[i, idx[("exp_csd", rc)]]:.3f}')
        out('| ' + ' | '.join(row) + ' |')
    out()

    # 4. spread
    sizes = np.bincount([minimum[n] for n in rel_names])
    sub = D[np.ix_(rel_idx, rel_idx)]
    labels = fcluster(linkage(squareform(sub, checks=False), method='average'), t=cut, criterion='distance')
    cm = {}
    for n, l in zip(rel_names, labels):
        cm.setdefault(minimum[n], set()).add(l)
    split = sum(len(v) > 1 for v in cm.values())
    inv = {}
    for m, ls in cm.items():
        for l in ls:
            inv.setdefault(l, set()).add(m)
    merged = sum(len(v) > 1 for v in inv.values())
    out(f'**Spread.** {len(rel_names)} relaxed endpoints reach {len(sizes)} COMPACK minima '
        f'(endpoints per minimum: {sorted(sizes.tolist(), reverse=True)[:10]}{"..." if len(sizes) > 10 else ""}); '
        f'RDF average-linkage clusters at the identity cut: {labels.max()}; COMPACK minima split across RDF clusters: '
        f'{split}; RDF clusters merging COMPACK minima: {merged}.')
    with open(os.path.join(sample_dir, f'rdf_report_{energy}.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'exp6')
