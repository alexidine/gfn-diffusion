"""One table over every acr_model_check.py result in a directory: each form's relaxed lattice energy relative to Form II
(the mean of ACRDIN04 and ACRDIN11), the best form's lattice energy, and the relaxed / experimental volume range.

    python summarize.py OUT_DIR
"""
import glob
import json
import os
import sys

FORMS = [('II', ['ACRDIN04', 'ACRDIN11']), ('A', ['ACRDIN07', 'ACRDIN09', 'ACRDIN10', 'ACRDIN13']), ('ACRDIN05', ['ACRDIN05']),
         ('ACRDIN06', ['ACRDIN06']), ('ACRDIN08', ['ACRDIN08']), ('ACRDIN12', ['ACRDIN12'])]


def main(out_dir):
    res = {}
    for f in sorted(glob.glob(os.path.join(out_dir, '*.json'))):
        d = json.load(open(f))
        res[d['label']] = d
    if not res:
        raise SystemExit(f'no results in {out_dir}')
    labels = list(res)
    print('Relaxed lattice energy minus Form II (mean of ACRDIN04, ACRDIN11), kJ/mol per molecule; one relaxation per CIF, '
          'atoms and cell; a range where a form has several CIFs.')
    print('| form | ' + ' | '.join(labels) + ' |')
    print('|---|' + '---|' * len(labels))
    for form, cifs in FORMS:
        cells = []
        for lab in labels:
            r = res[lab]['rows']
            if not all(c in r for c in cifs + ['ACRDIN04', 'ACRDIN11']):
                cells.append('n/a')
                continue
            ref = (r['ACRDIN04']['L_rel'] + r['ACRDIN11']['L_rel']) / 2
            v = [r[c]['L_rel'] - ref for c in cifs]
            cells.append(f'{min(v):+.1f}' if max(v) - min(v) < 0.15 else f'{min(v):+.1f} to {max(v):+.1f}')
        print(f'| {form} | ' + ' | '.join(cells) + ' |')
    print('\n| model | file | best lattice energy (kJ/mol) | form | relaxed / exp volume | gas minima spread (kJ/mol) |')
    print('|---|---|---|---|---|---|')
    for lab in labels:
        d = res[lab]
        r = d['rows']
        best = min(r, key=lambda c: r[c]['L_rel'])
        vr = [r[c]['V_ratio'] for c in r]
        print(f"| {lab} | {d['model']} | {r[best]['L_rel']:.1f} | {best} | {min(vr):.3f}-{max(vr):.3f} | {d['gas_spread']:.2f} |")
    print('\nExperiment: lattice energy about -96 to -99 kJ/mol (sublimation enthalpy 90.8-94.5 kJ/mol minus 2RT).')


if __name__ == '__main__':
    main(sys.argv[1])
