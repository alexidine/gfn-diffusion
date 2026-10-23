"""Which relaxed endpoints are an experimental form, and how many distinct minima they reach, by COMPACK.

Reads the relaxed P1 cells ``p1_relax`` wrote (``<sample_dir>/relaxed_<energy>/``), their energies from
``p1_relax_<energy>.csv``, and the relaxed experimental forms (``exp_<refcode>.cif`` in the same folder). Uses the
CSD Python API's packing similarity with the duplicate rule of the Day landscapes (Taylor, Butler & Day 2025, ESI
S1.1): 30-molecule shells, 30% distance and 30 degree angle tolerances, a match being all 30 molecules overlaid.

- identity: each endpoint against each experimental form;
- distinct minima: endpoints in ascending energy; one joins the first earlier minimum within 1.0 kJ/mol and
  0.05 g/cm3 of it that COMPACK matches, else opens a new minimum (Day's gate for comparing a pair).

usage (CPU, licensed CSD Python API):
    python -m eval.day_bench.compack_endpoints SAMPLE_DIR [elj|exp6]
"""
import csv
import glob
import os
import sys

SHELL, DIST_TOL, ANGLE_TOL = 30, 0.3, 30.0
E_GATE, RHO_GATE = 1.0, 0.05  # kJ/mol, g/cm3


def engine():
    from ccdc.crystal import PackingSimilarity
    ps = PackingSimilarity()
    s = ps.settings
    s.packing_shell_size = SHELL
    s.distance_tolerance = DIST_TOL
    s.angle_tolerance = ANGLE_TOL
    s.ignore_bond_types = True  # bonds are perceived from geometry on both sides
    s.timeout_ms = 20000
    return ps


def read(path):
    from ccdc import io
    with io.CrystalReader(path) as r:
        return r[0]


TIMEOUTS = []  # (a, b) identifiers of comparisons the engine abandoned; reported, never read as a non-match
PAIRS = []  # (a, b, molecules matched or None, RMSD or None): every comparison made, for calibrating other metrics


def compare(ps, a, b, na=None, nb=None):
    """(molecules matched, RMSD), or (None, None) when the engine returns nothing or times out (recorded). ``na``, ``nb``
    name the two crystals in the records (CIFs written by ASE all carry the block name image0)."""
    na, nb = na or a.identifier, nb or b.identifier
    try:
        res = ps.compare(a, b)
    except RuntimeError as e:
        if 'Timeout' not in str(e):
            raise
        TIMEOUTS.append((na, nb))
        PAIRS.append((na, nb, None, None))
        return None, None
    if res is None:
        PAIRS.append((na, nb, None, None))
        return None, None
    PAIRS.append((na, nb, int(res.nmatched_molecules), float(res.rmsd)))
    return int(res.nmatched_molecules), float(res.rmsd)


def main(sample_dir, energy='exp6'):
    d = os.path.join(sample_dir, f'relaxed_{energy}')
    rows = {r['file']: r for r in csv.DictReader(open(os.path.join(sample_dir, f'p1_relax_{energy}.csv')))
            if r['built'] == 'True'}
    forms = {os.path.basename(p)[4:-4]: read(p) for p in sorted(glob.glob(os.path.join(d, 'exp_*.cif')))}
    ps = engine()

    ends = []
    for f, r in rows.items():
        c = read(os.path.join(d, f))
        hits = {}
        for name, fc in forms.items():
            n, rmsd = compare(ps, c, fc, f, f'exp_{name}')
            hits[name] = (n, rmsd)
        ends.append({'file': f, 'e': float(r['e_end']), 'rho': float(c.calculated_density), 'sg': r['sg'] or 'none',
                     'crystal': c, 'hits': hits})

    ends.sort(key=lambda x: x['e'])
    minima = []  # representatives
    n_compared = 0
    for x in ends:
        x['minimum'] = None
        for k, m in enumerate(minima):
            if abs(x['e'] - m['e']) > E_GATE or abs(x['rho'] - m['rho']) > RHO_GATE:
                continue
            n_compared += 1
            n, _ = compare(ps, x['crystal'], m['crystal'], x['file'], m['file'])
            if n == SHELL:
                x['minimum'] = k
                break
        if x['minimum'] is None:
            x['minimum'] = len(minima)
            minima.append(x)

    with open(os.path.join(sample_dir, f'compack_{energy}.csv'), 'w', newline='') as fh:
        keys = ['file', 'energy_kjmol', 'density_gcm3', 'space_group', 'minimum'] + \
               [f'{n}_{q}' for n in forms for q in ('matched', 'rmsd')]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for x in ends:
            row = {'file': x['file'], 'energy_kjmol': x['e'], 'density_gcm3': x['rho'], 'space_group': x['sg'],
                   'minimum': x['minimum']}
            for n, (m, rmsd) in x['hits'].items():
                row[f'{n}_matched'], row[f'{n}_rmsd'] = m, rmsd
            w.writerow(row)

    with open(os.path.join(sample_dir, f'compack_pairs_{energy}.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['a', 'b', 'matched', 'rmsd'])
        w.writerows(PAIRS)

    print(f'{len(ends)} relaxed endpoints ({energy}); COMPACK {SHELL}-molecule shells, {int(DIST_TOL * 100)}% / '
          f'{ANGLE_TOL:.0f} deg; a match is {SHELL} of {SHELL}; comparisons timed out (no answer): {len(TIMEOUTS)}')
    for name in forms:
        unanswered = [x['file'] for x in ends if x['hits'][name][0] is None]
        if unanswered:
            print(f'  against {name}, no answer for: {", ".join(unanswered)}')
    for name in forms:
        full = [x for x in ends if x['hits'][name][0] == SHELL]
        best = max((x['hits'][name][0] or 0) for x in ends)
        listed = ', '.join(f"{x['file']} (RMSD30 {x['hits'][name][1]:.2f} A, {x['e']:.1f} kJ/mol)" for x in full)
        print(f'  matches experimental form {name}: {len(full)} of {len(ends)} endpoints'
              f'{"; " + listed if full else ""}; most molecules matched by any endpoint: {best}')
    sizes = {}
    for x in ends:
        sizes[x['minimum']] = sizes.get(x['minimum'], 0) + 1
    print(f'  distinct minima among the endpoints: {len(minima)} (pairs compared under the {E_GATE} kJ/mol, '
          f'{RHO_GATE} g/cm3 gate: {n_compared}); endpoints per minimum: {sorted(sizes.values(), reverse=True)}')
    for k, m in enumerate(minima):
        members = [x['file'] for x in ends if x['minimum'] == k]
        tags = [n for n in forms if m['hits'][n][0] == SHELL]
        print(f'    minimum {k}: {m["e"]:.1f} kJ/mol, {m["rho"]:.3f} g/cm3, sg {m["sg"]}, {len(members)} endpoint(s)'
              f'{" = " + "/".join(tags) if tags else ""}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'exp6')
